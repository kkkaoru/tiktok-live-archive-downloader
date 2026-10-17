"""Runner isolation and persistent launch reservation, without external jobs."""

import os
import shlex
import shutil
import subprocess
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pytest


@pytest.fixture
def project(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    root = tmp_path / "repository with spaces"
    (root / "scripts").mkdir(parents=True)
    (root / "edits/work/jobs").mkdir(parents=True)
    shutil.copyfile(
        Path(__file__).resolve().parents[1] / "scripts/tmux-job.sh",
        root / "scripts/tmux-job.sh",
    )
    binary = root / "bin"
    binary.mkdir()
    fake = binary / "tmux"
    fake.write_text(
        "#!/bin/bash\nset -eu\n"
        'test "$1" = -f; test "$2" = /dev/null\n'
        'test "$3" = -S; test "$4" = tmux.sock\n'
        'if [[ "$5" == has-session ]]; then exit 1; fi\n'
        "pwd > launched-cwd\n"
        'while [[ "$1" != /bin/bash ]]; do shift; done\n'
        'exec "$@"\n',
        encoding="utf-8",
    )
    fake.chmod(0o700)
    monkeypatch.setenv("PATH", f"{binary}{os.pathsep}{os.environ['PATH']}")
    return root


def launch(project: Path, *command: str, name: str = "probe") -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        ["/bin/bash", "scripts/tmux-job.sh", "edits/work/jobs", name, *command],
        cwd=project,
        capture_output=True,
        text=True,
        timeout=10,
        check=False,
    )


def test_success_and_private_output(project: Path) -> None:
    result = launch(project, "/usr/bin/printf", "%s", "literal ; $HOME")
    work = project / "edits/work/jobs"
    assert result.returncode == 0
    assert (work / "probe.log").read_text(encoding="utf-8") == "literal ; $HOME"
    assert (work / "probe.status").read_text(encoding="utf-8") == "0\n"
    assert (work / "probe.log").stat().st_mode & 0o777 == 0o600
    assert (work / "probe.launch").is_dir()
    assert (work / "launched-cwd").read_text(encoding="utf-8").strip() == str(work)


def test_child_failure_keeps_status_and_refuses_retry(project: Path) -> None:
    assert launch(project, "/bin/bash", "-c", "exit 7").returncode == 7
    assert (project / "edits/work/jobs/probe.status").read_text(encoding="utf-8") == "7\n"
    assert launch(project, "/usr/bin/true").returncode == 3


def test_dispatch_failure_reservation_prevents_retry(project: Path) -> None:
    (project / "bin/tmux").write_text("#!/bin/bash\nexit 12\n", encoding="utf-8")
    assert launch(project, "/usr/bin/true").returncode == 12
    assert not (project / "edits/work/jobs/probe.status").exists()
    assert launch(project, "/usr/bin/true").returncode == 3


@pytest.mark.parametrize("name", ["tmp", "cache", "tmux.sock", "probe.log", "probe.status"])
def test_symlink_destinations_are_refused(project: Path, name: str) -> None:
    (project / "edits/work/jobs" / name).symlink_to(project / "missing-target")
    assert launch(project, "/usr/bin/true").returncode != 0
    assert not (project / "missing-target").exists()


@pytest.mark.parametrize("name", ["tmp", "cache", "tmux.sock"])
def test_existing_files_at_directory_or_socket_paths_are_preserved(
    project: Path, name: str
) -> None:
    path = project / "edits/work/jobs" / name
    path.write_text("preserve", encoding="utf-8")
    assert launch(project, "/usr/bin/true").returncode == 2
    assert path.read_text(encoding="utf-8") == "preserve"


def test_concurrent_dispatch_has_one_winner(project: Path) -> None:
    with ThreadPoolExecutor(max_workers=2) as pool:
        first = pool.submit(launch, project, "/usr/bin/true")
        second = pool.submit(launch, project, "/usr/bin/true")
        codes = sorted([first.result().returncode, second.result().returncode])
    assert codes == [0, 3]


@pytest.mark.parametrize(
    "report", ["edits/work/check/report.json", "edits/work/check reports/report.json"]
)
def test_make_check_can_keep_reports_out_of_shared_root(
    project: Path, report: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    makefile = Path(__file__).resolve().parents[1] / "Makefile"
    # A surrounding make check may intentionally select the owned gate controller.
    # This subprocess checks Makefile defaults, rather than inheriting that run.
    monkeypatch.setenv("PYTEST_COMMAND", "python -m replay.pytest_gate_lifecycle outer.json")
    monkeypatch.setenv("MAKEFLAGS", "PYTEST_COMMAND=outer-controller")
    environment = {
        name: value
        for name, value in os.environ.items()
        if name not in {"MAKEFLAGS", "MAKEOVERRIDES", "MFLAGS", "PYTEST_COMMAND", "COVERAGE_JSON"}
    }
    result = subprocess.run(
        ["make", "-n", "-f", str(makefile), "check", f"COVERAGE_JSON={report}"],
        cwd=project,
        env=environment,
        capture_output=True,
        text=True,
        timeout=10,
        check=True,
    )
    commands = [shlex.split(line, comments=True) for line in result.stdout.splitlines()]
    test_commands = [
        args for args in commands if args[:5] == ["umask", "022;", "uv", "run", "pytest"]
    ]
    assert len(test_commands) == 1
    arguments = test_commands[0][5:]
    assert "-p" in arguments
    assert arguments[arguments.index("-p") + 1] == "replay.pytest_scratch"
    assert f"--cov-report=json:{report}" in arguments
    coverage_checks = [args for args in commands if args[:4] == ["uv", "run", "python", "-c"]]
    assert len(coverage_checks) == 1
    assert "Path(sys.argv[1])" in coverage_checks[0][4]
    assert coverage_checks[0][5:] == [report]


def test_make_check_selects_one_explicit_owned_runner(project: Path) -> None:
    makefile = Path(__file__).resolve().parents[1] / "Makefile"
    environment = {
        name: value
        for name, value in os.environ.items()
        if name not in {"MAKEFLAGS", "MAKEOVERRIDES", "MFLAGS", "PYTEST_COMMAND", "COVERAGE_JSON"}
    }
    result = subprocess.run(
        [
            "make",
            "-n",
            "-f",
            str(makefile),
            "check",
            "PYTEST_COMMAND=python -B -m replay.pytest_gate_lifecycle owned-request.json",
            "COVERAGE_JSON=edits/work/check reports/coverage.json",
        ],
        cwd=project,
        env=environment,
        capture_output=True,
        text=True,
        timeout=10,
        check=True,
    )
    commands = [shlex.split(line, comments=True) for line in result.stdout.splitlines()]
    test_commands = [args for args in commands if args[:2] == ["umask", "022;"]]
    assert test_commands == [
        [
            "umask",
            "022;",
            "python",
            "-B",
            "-m",
            "replay.pytest_gate_lifecycle",
            "owned-request.json",
            "--cov-report=json:edits/work/check reports/coverage.json",
        ]
    ]
    coverage_checks = [args for args in commands if args[:4] == ["uv", "run", "python", "-c"]]
    assert len(coverage_checks) == 1
    assert coverage_checks[0][5:] == ["edits/work/check reports/coverage.json"]


def test_environment_and_repository_cwd(project: Path) -> None:
    result = launch(
        project,
        "/bin/bash",
        "-c",
        'test "$PWD/edits/work/jobs/tmp" = "$TMPDIR"; '
        'test "$TMP" = "$TEMP"; test "$TMPDIR" = "$TEMP"',
    )
    assert result.returncode == 0


def _fake_python(project: Path, exit_code: int) -> None:
    binary = project / ".venv/bin"
    binary.mkdir(parents=True)
    fake = binary / "python"
    fake.write_text(f"#!/bin/bash\necho disk-check-ran\nexit {exit_code}\n", encoding="utf-8")
    fake.chmod(0o700)


def test_launch_is_refused_below_the_free_space_floor(project: Path) -> None:
    _fake_python(project, 1)
    result = launch(project, "/usr/bin/true")
    assert result.returncode == 4
    assert "free-space floor not met" in result.stderr
    assert not (project / "edits/work/jobs/probe.launch").exists()


def test_launch_proceeds_when_the_floor_is_met(project: Path) -> None:
    _fake_python(project, 0)
    result = launch(project, "/usr/bin/true")
    assert result.returncode == 0
    assert (project / "edits/work/jobs/probe.launch").is_dir()


def test_a_job_directory_under_the_external_work_root_is_accepted(
    project: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    external = tmp_path / "external-work"
    (external / "jobs").mkdir(parents=True)
    monkeypatch.setenv("REPLAY_EXTERNAL_WORK_ROOT", str(external.resolve()))
    result = subprocess.run(
        ["/bin/bash", "scripts/tmux-job.sh", str(external / "jobs"), "probe", "/usr/bin/true"],
        cwd=project,
        capture_output=True,
        text=True,
        timeout=10,
        check=False,
    )
    assert result.returncode == 0
    assert (external / "jobs/probe.status").read_text(encoding="utf-8") == "0\n"


def test_a_job_directory_outside_both_roots_is_refused(project: Path, tmp_path: Path) -> None:
    other = tmp_path / "elsewhere"
    other.mkdir()
    result = subprocess.run(
        ["/bin/bash", "scripts/tmux-job.sh", str(other), "probe", "/usr/bin/true"],
        cwd=project,
        capture_output=True,
        text=True,
        timeout=10,
        check=False,
    )
    assert result.returncode == 2
