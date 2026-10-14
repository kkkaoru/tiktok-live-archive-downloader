"""Exercise the configured build backend only against tiny synthetic projects."""

from __future__ import annotations

import importlib.metadata
import os
import subprocess
import sys
import tarfile
import tomllib
import zipfile
from pathlib import Path
from typing import Literal

import pytest

ROOT = Path(__file__).resolve().parents[1]
BUILD_TIMEOUT_SECONDS = 15
SCAN_AUDIT = """\
import os
import sys
from pathlib import Path

root = Path(os.environ["PACKAGING_FIXTURE_ROOT"])
audit = Path(os.environ["PACKAGING_SCAN_AUDIT"])
denied = frozenset({"private", "downloads", "edits", "youtube-originals",
                    "captures", "screenshots", "browser-state", "backups"})

def scan_guard(event: str, arguments: tuple[object, ...]) -> None:
    if event != "os.scandir" or not arguments:
        return
    argument = arguments[0]
    if not isinstance(argument, (str, bytes)):
        return
    path = Path(os.fsdecode(argument))
    if not path.is_relative_to(root):
        return
    relative = path.relative_to(root)
    with audit.open("a", encoding="utf-8") as stream:
        stream.write(relative.as_posix() + "\\n")
    if os.environ["PACKAGING_DENY_EXCLUDED"] == "1" and relative.parts:
        if relative.parts[0] in denied:
            raise RuntimeError("Excluded synthetic directory was scanned")

sys.addaudithook(scan_guard)
"""


@pytest.fixture
def project(tmp_path: Path) -> Path:
    files = {
        "src/replay/__init__.py": "",
        "src/replay/cli.py": "def main() -> int:\n    return 0\n",
        "src/replay/py.typed": "",
        "src/replay/unapproved.txt": "synthetic excluded runtime asset",
        "src/replay/nested/unapproved.py": "synthetic excluded nested module",
        "README.md": "Synthetic README",
        "SECURITY.md": "Synthetic security policy",
        "docs/public.md": "Synthetic documentation",
        "docs/unapproved.bin": "synthetic excluded documentation asset",
        "tests/test_public.py": "def test_public() -> None:\n    assert True\n",
        "tests/unapproved.py": "synthetic excluded helper",
        "scripts/public.sh": "#!/bin/sh\nexit 0\n",
        "scripts/unapproved.py": "synthetic excluded script",
        "TranscriptViewer/Sources/TranscriptCore/Public.swift": "struct Public {}\n",
        "TranscriptViewer/Sources/TranscriptCore/unapproved.txt": "synthetic excluded asset",
        "private/marker.txt": "synthetic only",
        "downloads/marker.txt": "synthetic only",
        "edits/marker.txt": "synthetic only",
        "youtube-originals/marker.txt": "synthetic only",
        "captures/marker.txt": "synthetic only",
        "screenshots/marker.txt": "synthetic only",
        "browser-state/marker.txt": "synthetic only",
        "backups/marker.txt": "synthetic only",
    }
    for name, content in files.items():
        path = tmp_path / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content, encoding="utf-8")
    (tmp_path / ".git").mkdir()
    (tmp_path / ".gitignore").write_text(
        (ROOT / ".gitignore").read_text(encoding="utf-8"), encoding="utf-8"
    )
    (tmp_path / "pyproject.toml").write_text(
        (ROOT / "pyproject.toml").read_text(encoding="utf-8"), encoding="utf-8"
    )
    return tmp_path


def build(root: Path, *, target: Literal["sdist", "wheel"], deny_excluded: bool = True) -> Path:
    """Use the normal dev backend CLI, without importing its untyped implementation."""
    instrumentation = root / "instrumentation"
    instrumentation.mkdir(exist_ok=True)
    (instrumentation / "sitecustomize.py").write_text(SCAN_AUDIT, encoding="utf-8")
    scratch = root / "scratch"
    scratch.mkdir(exist_ok=True)
    output = root / "dist"
    output.mkdir(exist_ok=True)
    environment = os.environ.copy()
    python_paths = [str(instrumentation)]
    if inherited_path := environment.get("PYTHONPATH"):
        python_paths.append(inherited_path)
    environment.update(
        {
            "PYTHONPATH": os.pathsep.join(python_paths),
            "PYTHONDONTWRITEBYTECODE": "1",
            "PYTHONNOUSERSITE": "1",
            "TMPDIR": str(scratch),
            "TMP": str(scratch),
            "TEMP": str(scratch),
            "PACKAGING_FIXTURE_ROOT": str(root),
            "PACKAGING_SCAN_AUDIT": str(root / "scan-audit.txt"),
            "PACKAGING_DENY_EXCLUDED": "1" if deny_excluded else "0",
        }
    )
    result = subprocess.run(
        [sys.executable, "-B", "-m", "hatchling", "build", "-t", target, "-d", str(output)],
        cwd=root,
        env=environment,
        capture_output=True,
        text=True,
        encoding="utf-8",
        timeout=BUILD_TIMEOUT_SECONDS,
        check=False,
    )
    if result.returncode != 0:
        pytest.fail(f"Synthetic {target} build failed: {result.stdout}\n{result.stderr}")
    extension = "*.tar.gz" if target == "sdist" else "*.whl"
    artifacts = list(output.glob(extension))
    if len(artifacts) != 1:
        pytest.fail(f"Expected exactly one synthetic {target} archive")
    return artifacts[0]


def source_members(path: Path) -> list[str]:
    with tarfile.open(path, "r:gz") as archive:
        return sorted(
            member.name.split("/", 1)[1] for member in archive.getmembers() if member.isfile()
        )


def test_dev_and_build_backend_have_same_exact_version() -> None:
    configuration = tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))
    build_system: object = configuration.get("build-system")
    dependency_groups: object = configuration.get("dependency-groups")
    assert importlib.metadata.version("hatchling") == "1.32.4"
    assert isinstance(build_system, dict)
    assert build_system.get("requires") == ["hatchling==1.32.4"]
    assert isinstance(dependency_groups, dict)
    dev: object = dependency_groups.get("dev")
    assert isinstance(dev, list)
    assert "hatchling==1.32.4" in dev


def test_unpruned_ignored_alias_reproduces_missing_runtime_source(project: Path) -> None:
    configuration = (project / "pyproject.toml").read_text(encoding="utf-8")
    (project / "pyproject.toml").write_text(
        configuration.replace("skip-excluded-dirs = true", "skip-excluded-dirs = false"),
        encoding="utf-8",
    )
    (project / "edits" / "early-source-alias").symlink_to(project / "src/replay")
    members = source_members(build(project, target="sdist", deny_excluded=False))
    assert "src/replay/__init__.py" not in members
    assert "src/replay/cli.py" not in members
    assert "src/replay/py.typed" not in members
    assert "README.md" in members
    assert "edits/early-source-alias" in (
        (project / "scan-audit.txt").read_text(encoding="utf-8").splitlines()
    )


def test_excluded_roots_are_not_scanned_without_vcs_ignore(project: Path) -> None:
    (project / ".gitignore").unlink()
    (project / "edits" / "early-source-alias").symlink_to(project / "src/replay")
    members = source_members(build(project, target="sdist"))
    assert "src/replay/cli.py" in members
    assert "src/replay/unapproved.txt" not in members
    assert "src/replay/nested/unapproved.py" not in members
    assert "." in (project / "scan-audit.txt").read_text(encoding="utf-8").splitlines()


def test_sdist_preserves_exact_public_allowlist(project: Path) -> None:
    (project / "edits" / "early-source-alias").symlink_to(project / "src/replay")
    assert source_members(build(project, target="sdist")) == [
        ".gitignore",
        "PKG-INFO",
        "README.md",
        "SECURITY.md",
        "TranscriptViewer/Sources/TranscriptCore/Public.swift",
        "docs/public.md",
        "pyproject.toml",
        "scripts/public.sh",
        "src/replay/__init__.py",
        "src/replay/cli.py",
        "src/replay/py.typed",
        "tests/test_public.py",
    ]


def test_wheel_contains_only_selected_runtime_package(project: Path) -> None:
    (project / "edits" / "early-source-alias").symlink_to(project / "src/replay")
    with zipfile.ZipFile(build(project, target="wheel")) as archive:
        assert sorted(archive.namelist()) == [
            "replay/__init__.py",
            "replay/cli.py",
            "replay/py.typed",
            "tiktok_live_archive_downloader-0.1.0.dist-info/METADATA",
            "tiktok_live_archive_downloader-0.1.0.dist-info/RECORD",
            "tiktok_live_archive_downloader-0.1.0.dist-info/WHEEL",
            "tiktok_live_archive_downloader-0.1.0.dist-info/entry_points.txt",
        ]
        assert archive.read("replay/cli.py") == b"def main() -> int:\n    return 0\n"
        assert (
            archive.read("tiktok_live_archive_downloader-0.1.0.dist-info/entry_points.txt")
            == b"[console_scripts]\nreplay = replay.cli:main\n"
        )


def test_sdist_to_wheel_round_trip_retains_runtime_code(project: Path, tmp_path: Path) -> None:
    (project / "edits" / "early-source-alias").symlink_to(project / "src/replay")
    output = tmp_path / "round-trip"
    output.mkdir()
    with tarfile.open(build(project, target="sdist"), "r:gz") as archive:
        assert all(member.isfile() and ".." not in Path(member.name).parts for member in archive)
        archive.extractall(output, filter="data")
    with zipfile.ZipFile(
        build(output / "tiktok_live_archive_downloader-0.1.0", target="wheel")
    ) as archive:
        assert archive.read("replay/cli.py") == b"def main() -> int:\n    return 0\n"
        assert "replay/py.typed" in archive.namelist()
        assert "replay/unapproved.txt" not in archive.namelist()
        assert "replay/nested/unapproved.py" not in archive.namelist()
        assert "private/marker.txt" not in archive.namelist()
