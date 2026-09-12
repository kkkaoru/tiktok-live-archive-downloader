import shutil
import subprocess
from pathlib import Path

import pytest


@pytest.fixture
def bootstrap(tmp_path: Path) -> tuple[Path, dict[str, str]]:
    root = tmp_path / "project"
    (root / "scripts").mkdir(parents=True)
    source = Path(__file__).resolve().parents[1]
    shutil.copyfile(source / "scripts/bootstrap.sh", root / "scripts/bootstrap.sh")
    shutil.copyfile(source / ".python-version", root / ".python-version")
    binaries = tmp_path / "bin"
    binaries.mkdir()
    (binaries / "uname").write_text(
        '#!/bin/sh\nprintf "%s\\n" "${MOCK_OS:-Darwin}"\n', encoding="utf-8"
    )
    (binaries / "brew").write_text(
        '#!/bin/sh\nprintf "brew %s\\n" "$*" >> "$TRACE"\n', encoding="utf-8"
    )
    (binaries / "uv").write_text(
        '#!/bin/sh\nprintf "uv %s\\n" "$*" >> "$TRACE"\n', encoding="utf-8"
    )
    (binaries / "uname").chmod(0o755)
    (binaries / "brew").chmod(0o755)
    (binaries / "uv").chmod(0o755)
    return root, {
        "PATH": f"{binaries}:/usr/bin:/bin",
        "HOME": str(tmp_path),
        "TRACE": str(tmp_path / "trace"),
    }


def test_developer_setup_is_locked_and_does_not_login(
    bootstrap: tuple[Path, dict[str, str]],
) -> None:
    root, environ = bootstrap
    result = subprocess.run(
        ["bash", str(root / "scripts/bootstrap.sh"), "--dev", "--no-browser"],
        env=environ,
        check=True,
        capture_output=True,
        text=True,
        timeout=10,
    )
    assert Path(environ["TRACE"]).read_text(encoding="utf-8") == (
        "brew install uv ffmpeg\nbrew install gitleaks\n"
        "uv python install 3.12.13\nuv sync --locked --extra web --extra capture\n"
        "uv run --locked replay --help\n"
    )
    assert "No authentication, media transfer" in result.stdout
    assert (root / "private").stat().st_mode & 0o777 == 0o700
    assert (root / "downloads").stat().st_mode & 0o777 == 0o700


def test_runtime_setup_installs_browser(bootstrap: tuple[Path, dict[str, str]]) -> None:
    root, environ = bootstrap
    subprocess.run(
        ["bash", str(root / "scripts/bootstrap.sh")],
        env=environ,
        check=True,
        capture_output=True,
        timeout=10,
    )
    assert Path(environ["TRACE"]).read_text(encoding="utf-8") == (
        "brew install uv ffmpeg\nbrew install --cask google-chrome\n"
        "uv python install 3.12.13\nuv sync --locked --extra web\n"
        "uv run --locked replay --help\n"
    )


def test_existing_private_data_survives_setup(bootstrap: tuple[Path, dict[str, str]]) -> None:
    root, environ = bootstrap
    (root / "private/jobs").mkdir(parents=True)
    (root / "private/jobs/job.json").write_text("existing", encoding="utf-8")
    subprocess.run(
        ["bash", str(root / "scripts/bootstrap.sh"), "--no-browser"],
        env=environ,
        check=True,
        capture_output=True,
        timeout=10,
    )
    assert (root / "private/jobs/job.json").read_text(encoding="utf-8") == "existing"


def test_wrong_os_stops_before_installation(bootstrap: tuple[Path, dict[str, str]]) -> None:
    root, environ = bootstrap
    result = subprocess.run(
        ["bash", str(root / "scripts/bootstrap.sh")],
        env=environ | {"MOCK_OS": "Linux"},
        capture_output=True,
        timeout=10,
    )
    assert result.returncode == 1
    assert not Path(environ["TRACE"]).exists()


def test_missing_homebrew_is_actionable(bootstrap: tuple[Path, dict[str, str]]) -> None:
    root, environ = bootstrap
    Path(environ["HOME"], "bin/brew").unlink()
    result = subprocess.run(
        ["bash", str(root / "scripts/bootstrap.sh")],
        env=environ,
        capture_output=True,
        text=True,
        timeout=10,
    )
    assert result.returncode == 1
    assert "Install Homebrew" in result.stderr


@pytest.mark.parametrize("option", ["--help", "--invalid"])
def test_bootstrap_option_does_not_install(
    bootstrap: tuple[Path, dict[str, str]], option: str
) -> None:
    root, environ = bootstrap
    subprocess.run(
        ["bash", str(root / "scripts/bootstrap.sh"), option],
        env=environ,
        capture_output=True,
        timeout=10,
    )
    assert not Path(environ["TRACE"]).exists()


def test_dependency_failure_is_not_success(bootstrap: tuple[Path, dict[str, str]]) -> None:
    root, environ = bootstrap
    command = Path(environ["HOME"], "bin/uv")
    command.write_text("#!/bin/sh\nexit 9\n", encoding="utf-8")
    result = subprocess.run(
        ["bash", str(root / "scripts/bootstrap.sh"), "--no-browser"],
        env=environ,
        capture_output=True,
        timeout=10,
    )
    assert result.returncode == 9
    assert not (root / "private").exists()
