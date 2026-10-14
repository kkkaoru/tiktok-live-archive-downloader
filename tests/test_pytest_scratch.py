"""Exercise real pytest startup and temporary paths in repository-local sandboxes."""

import json
import os
import shutil
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pytest
from coverage import Coverage

PROBE = """import json
import os
import subprocess
import sys
import tempfile
import time
from pathlib import Path

# This must never run when the base is rejected during configuration.
Path("collected").write_text("yes", encoding="utf-8")
with tempfile.TemporaryDirectory() as collection_temp:
    COLLECTION_TEMP = collection_temp

def test_probe(tmp_path, tmp_path_factory):
    marker = tmp_path / "alive"
    marker.write_text("yes", encoding="utf-8")
    with tempfile.TemporaryDirectory() as actual:
        child = subprocess.run(
            [sys.executable, "child_temp.py"], capture_output=True, text=True,
            check=True, timeout=10,
        )
        observations = {
            "pytest": str(tmp_path_factory.getbasetemp()),
            "collectionPython": COLLECTION_TEMP,
            "python": actual,
            "gettempdir": tempfile.gettempdir(),
            "environment": [os.environ[key] for key in
                            ("TMPDIR", "TMP", "TEMP", "PYTEST_DEBUG_TEMPROOT")],
            "child": json.loads(child.stdout),
        }
        label = os.environ.get("PROBE_LABEL")
        if label:
            Path(label + ".json").write_text(json.dumps(observations), encoding="utf-8")
            deadline = time.monotonic() + 10
            other = Path("b.json" if label == "a" else "a.json")
            while not other.exists() and time.monotonic() < deadline:
                time.sleep(0.01)
            assert other.exists()
            assert marker.read_text(encoding="utf-8") == "yes"
        else:
            Path("observations.json").write_text(json.dumps(observations), encoding="utf-8")
"""
CHILD = """import json
import tempfile

with tempfile.TemporaryDirectory() as actual:
    print(json.dumps({"python": actual, "gettempdir": tempfile.gettempdir()}))
"""


def _project(tmp_path: Path) -> Path:
    project = tmp_path / "project"
    (project / "tests").mkdir(parents=True)
    shutil.copyfile(Path(__file__).with_name("conftest.py"), project / "tests" / "conftest.py")
    (project / "pyproject.toml").write_text(
        '[tool.pytest.ini_options]\naddopts = ""\n', encoding="utf-8"
    )
    (project / "tests" / "test_probe.py").write_text(PROBE, encoding="utf-8")
    (project / "child_temp.py").write_text(CHILD, encoding="utf-8")
    (project / "temp_cache.py").write_text(
        'import tempfile\ntempfile.tempdir = "/unconfigured-cached-root"\n', encoding="utf-8"
    )
    (project / "capture_probe.py").write_text(
        "import json\nimport tempfile\nfrom pathlib import Path\n"
        "import _pytest.capture as capture\n"
        "original = capture.TemporaryFile\n"
        "def observed(*args, **kwargs):\n"
        '    Path("capture-root.json").write_text(\n'
        '        json.dumps(tempfile.gettempdir()), encoding="utf-8")\n'
        "    return original(*args, **kwargs)\n"
        "capture.TemporaryFile = observed\n",
        encoding="utf-8",
    )
    return project


def _run(
    project: Path, *arguments: str, label: str = "", measure: bool = True, bootstrap: bool = True
) -> subprocess.CompletedProcess[str]:
    environment = {
        **os.environ,
        "PYTEST_DISABLE_PLUGIN_AUTOLOAD": "1",
        "PYTHONDONTWRITEBYTECODE": "1",
        "PYTEST_ADDOPTS": "",
        "PYTEST_DEBUG_TEMPROOT": "/unconfigured-pytest-root",
        "TMPDIR": "/unconfigured-python-root",
        "TMP": "/unconfigured-python-root",
        "TEMP": "/unconfigured-python-root",
    }
    if label:
        environment["PROBE_LABEL"] = label
    command = [sys.executable, "-B", "-m", "pytest"]
    active = Coverage.current()
    if measure and active is not None:
        environment["COVERAGE_FILE"] = str(Path(active.config.data_file).absolute())
        command = [
            sys.executable,
            "-B",
            "-m",
            "coverage",
            "run",
            "--parallel-mode",
            "--source=replay.pytest_scratch",
            "-m",
            "pytest",
        ]
    if bootstrap:
        command.extend(["-p", "replay.pytest_scratch"])
    return subprocess.run(
        [*command, "-q", *arguments],
        cwd=project,
        env=environment,
        capture_output=True,
        text=True,
        timeout=30,
    )


def test_direct_pytest_uses_actual_repository_temp_paths(tmp_path: Path) -> None:
    project = _project(tmp_path)
    completed = _run(project, "-p", "temp_cache", "-p", "capture_probe", measure=False)
    assert completed.returncode == 0, completed.stdout + completed.stderr
    observed = json.loads((project / "observations.json").read_text(encoding="utf-8"))
    assert Path(observed["pytest"]).name == "pytest"
    assert Path(observed["pytest"]).parent.parent.name == "pytest-default"
    assert Path(observed["pytest"]).is_relative_to(project / "edits" / "work")
    assert Path(observed["python"]).is_relative_to(project / "edits" / "work")
    assert Path(observed["collectionPython"]).is_relative_to(project / "edits" / "work")
    assert Path(observed["child"]["python"]).is_relative_to(project / "edits" / "work")
    assert Path(observed["gettempdir"]).name == "python-tmp"
    assert observed["child"]["gettempdir"] == observed["gettempdir"]
    assert observed["environment"] == [observed["gettempdir"]] * 4
    assert Path(observed["gettempdir"]) != Path(observed["pytest"])
    capture_root = json.loads((project / "capture-root.json").read_text(encoding="utf-8"))
    assert capture_root == observed["gettempdir"]


def _ready(path: Path) -> None:
    deadline = time.monotonic() + 10
    while not path.exists() and time.monotonic() < deadline:
        time.sleep(0.01)
    if not path.exists():
        raise TimeoutError("Child pytest did not reach its active fixture")


def test_concurrent_explicit_base_is_refused_without_deleting_active_fixture(
    tmp_path: Path,
) -> None:
    project = _project(tmp_path)
    base = project / "edits" / "work" / "specified-temp"
    base.mkdir(parents=True)
    with ThreadPoolExecutor(max_workers=1) as executor:
        first = executor.submit(_run, project, f"--basetemp={base}", label="a")
        _ready(project / "a.json")
        second = _run(project, f"--basetemp={base}", label="b")
        assert second.returncode == 4
        assert "scratch is already owned by another test run" in second.stderr
        assert not (project / "b.json").exists()
        (project / "b.json").write_text("{}", encoding="utf-8")
        completed = first.result(timeout=35)
    assert completed.returncode == 0, completed.stdout + completed.stderr


def test_concurrent_default_runs_keep_independent_fixtures(tmp_path: Path) -> None:
    project = _project(tmp_path)
    with ThreadPoolExecutor(max_workers=2) as executor:
        first = executor.submit(_run, project, label="a")
        second = executor.submit(_run, project, label="b")
        completed_a = first.result(timeout=35)
        completed_b = second.result(timeout=35)
    assert completed_a.returncode == 0, completed_a.stdout + completed_a.stderr
    assert completed_b.returncode == 0, completed_b.stdout + completed_b.stderr
    a = json.loads((project / "a.json").read_text(encoding="utf-8"))
    b = json.loads((project / "b.json").read_text(encoding="utf-8"))
    assert a["pytest"] != b["pytest"]
    assert a["gettempdir"] != b["gettempdir"]
    assert Path(a["pytest"]).parent.parent.name == "pytest-default"
    assert Path(b["pytest"]).parent.parent.name == "pytest-default"


def test_valid_explicit_base_is_preserved(tmp_path: Path) -> None:
    project = _project(tmp_path)
    base = project / "edits" / "work" / "specified-temp"
    base.mkdir(parents=True)
    completed = _run(project, f"--basetemp={base}")
    assert completed.returncode == 0, completed.stdout + completed.stderr
    observed = json.loads((project / "observations.json").read_text(encoding="utf-8"))
    assert Path(observed["pytest"]).name == "specified-temp"
    assert Path(observed["pytest"]).parent.name == "work"
    assert Path(observed["gettempdir"]).name == "python-tmp"


@pytest.mark.parametrize(
    "mode", ["outside", "traversal", "symlink", "work", "default", "default-child", "file"]
)
def test_unsafe_base_is_rejected_before_collection_or_fixture_deletion(
    tmp_path: Path, mode: str
) -> None:
    project = _project(tmp_path)
    outside = tmp_path / "outside"
    outside.mkdir()
    sentinel = outside / "preserved"
    sentinel.write_text("untouched", encoding="utf-8")
    work = project / "edits" / "work"
    work.mkdir(parents=True)
    candidates = {
        "outside": outside,
        "traversal": work / ".." / ".." / ".." / "outside",
        "symlink": work / "linked",
        "work": work,
        "default": work / "pytest-default",
        "default-child": work / "pytest-default" / "active",
        "file": work / "file",
    }
    if mode == "symlink":
        (work / "linked").symlink_to(outside, target_is_directory=True)
    if mode == "file":
        (work / "file").write_text("preserved", encoding="utf-8")
    completed = _run(project, f"--basetemp={candidates[mode]}")
    assert completed.returncode == 4
    assert "pytest scratch must be an unlinked descendant" in completed.stderr or (
        mode == "default-child" and "cannot occupy the default scratch tree" in completed.stderr
    )
    assert sentinel.read_text(encoding="utf-8") == "untouched"
    assert not (project / "collected").exists()
    assert not (work / "pytest-default").exists()


@pytest.mark.parametrize("mode", ["work-link", "default-link", "default-file", "locks-link"])
def test_invalid_default_tree_is_refused_before_collection(tmp_path: Path, mode: str) -> None:
    project = _project(tmp_path)
    outside = tmp_path / "outside"
    outside.mkdir()
    sentinel = outside / "preserved"
    sentinel.write_text("untouched", encoding="utf-8")
    work = project / "edits" / "work"
    work.parent.mkdir()
    if mode == "work-link":
        work.symlink_to(outside, target_is_directory=True)
    else:
        work.mkdir()
        default = work / "pytest-default"
        if mode == "default-link":
            default.symlink_to(outside, target_is_directory=True)
        elif mode == "default-file":
            default.write_text("untouched", encoding="utf-8")
        else:
            default.mkdir()
            (default / "locks").symlink_to(outside, target_is_directory=True)
    completed = _run(project)
    assert completed.returncode == 4
    assert "must be a real repository work directory" in completed.stderr or (
        mode == "locks-link" and "must be an unlinked descendant" in completed.stderr
    )
    assert sentinel.read_text(encoding="utf-8") == "untouched"
    assert not (project / "collected").exists()


@pytest.mark.parametrize("mode", ["cache", "environment", "base"])
def test_modified_bootstrap_state_is_rejected_before_fixtures(tmp_path: Path, mode: str) -> None:
    project = _project(tmp_path)
    changes = {
        "cache": "tempfile.tempdir = str(config.rootpath)",
        "environment": 'os.environ["TMPDIR"] = str(config.rootpath)',
        "base": 'config.option.basetemp = str(config.rootpath / "edits" / "work" / "other")',
    }
    (project / "tests" / "conftest.py").write_text(
        "import os\nimport tempfile\nimport pytest\n"
        "from replay.pytest_scratch import verify_bootstrap\n"
        "@pytest.hookimpl(tryfirst=True)\n"
        "def pytest_configure(config):\n"
        f"    {changes[mode]}\n"
        "    verify_bootstrap(config)\n",
        encoding="utf-8",
    )
    completed = _run(project)
    assert completed.returncode == 4
    assert "changed after scratch bootstrap" in completed.stderr
    assert not (project / "collected").exists()


def test_missing_early_plugin_is_rejected_before_fixture_collection(tmp_path: Path) -> None:
    project = _project(tmp_path)
    # sys capture needs no tempfile before the conftest refuses this deliberate
    # removal. Normal entry points explicitly load the early plugin for fd capture.
    completed = _run(project, "--capture=sys", bootstrap=False)
    assert completed.returncode == 4
    assert "require the early -p replay.pytest_scratch plugin" in completed.stderr
    assert not (project / "collected").exists()
