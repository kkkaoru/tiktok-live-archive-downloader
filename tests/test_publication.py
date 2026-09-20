import os
import shutil
import subprocess
from pathlib import Path

import pytest


@pytest.fixture
def publication(tmp_path: Path) -> tuple[Path, dict[str, str]]:
    root = tmp_path / "repository"
    (root / "scripts").mkdir(parents=True)
    source = Path(__file__).resolve().parents[1]
    shutil.copyfile(source / "scripts/check-public.sh", root / "scripts/check-public.sh")
    shutil.copyfile(source / ".gitleaks.toml", root / ".gitleaks.toml")
    (root / ".gitignore").write_text("/private/\n", encoding="utf-8")
    (root / "README.md").write_text("Public source\n", encoding="utf-8")
    environment = {
        "PATH": os.environ["PATH"],
        "HOME": str(tmp_path),
        "GIT_CONFIG_GLOBAL": os.devnull,
        "GIT_CONFIG_NOSYSTEM": "1",
        "GIT_AUTHOR_NAME": "test",
        "GIT_COMMITTER_NAME": "test",
        "GIT_AUTHOR_EMAIL": "test@users.noreply.github.com",
        "GIT_COMMITTER_EMAIL": "test@users.noreply.github.com",
    }
    subprocess.run(
        ["git", "init", "-b", "main"],
        cwd=root,
        env=environment,
        check=True,
        capture_output=True,
        timeout=10,
    )
    subprocess.run(
        ["git", "add", "."], cwd=root, env=environment, check=True, capture_output=True, timeout=10
    )
    subprocess.run(
        ["git", "commit", "-m", "test fixture"],
        cwd=root,
        env=environment,
        check=True,
        capture_output=True,
        timeout=10,
    )
    return root, environment


def test_public_scan_excludes_private_data(publication: tuple[Path, dict[str, str]]) -> None:
    root, environment = publication
    (root / "private").mkdir()
    (root / "private/token").write_text("sessionid=" + "0123456789abcdef" * 2, encoding="utf-8")
    result = subprocess.run(
        ["bash", "scripts/check-public.sh"],
        cwd=root,
        env=environment,
        capture_output=True,
        text=True,
        timeout=20,
    )
    assert result.returncode == 0
    assert "Index and complete referenced history passed" in result.stdout


def test_agent_guide_is_an_approved_public_path(
    publication: tuple[Path, dict[str, str]],
) -> None:
    root, environment = publication
    (root / "AGENTS.md").write_text("# Agent guide\nNever publish credentials.\n", encoding="utf-8")
    subprocess.run(
        ["git", "add", "AGENTS.md"],
        cwd=root,
        env=environment,
        check=True,
        capture_output=True,
        timeout=10,
    )
    result = subprocess.run(
        ["bash", "scripts/check-public.sh"],
        cwd=root,
        env=environment,
        capture_output=True,
        text=True,
        timeout=20,
    )
    assert result.returncode == 0
    assert "Index and complete referenced history passed" in result.stdout


def test_agent_skill_is_an_approved_public_path(
    publication: tuple[Path, dict[str, str]],
) -> None:
    root, environment = publication
    skill = root / ".agents/skills/example/SKILL.md"
    skill.parent.mkdir(parents=True)
    skill.write_text("# Example skill\n", encoding="utf-8")
    subprocess.run(
        ["git", "add", ".agents/skills/example/SKILL.md"],
        cwd=root,
        env=environment,
        check=True,
        capture_output=True,
        timeout=10,
    )
    result = subprocess.run(
        ["bash", "scripts/check-public.sh"],
        cwd=root,
        env=environment,
        capture_output=True,
        text=True,
        timeout=20,
    )
    assert result.returncode == 0
    assert "Index and complete referenced history passed" in result.stdout


def test_forced_private_path_is_rejected(publication: tuple[Path, dict[str, str]]) -> None:
    root, environment = publication
    (root / "private").mkdir()
    (root / "private/capture.json").write_text("{}", encoding="utf-8")
    subprocess.run(
        ["git", "add", "-f", "private/capture.json"],
        cwd=root,
        env=environment,
        check=True,
        capture_output=True,
        timeout=10,
    )
    result = subprocess.run(
        ["bash", "scripts/check-public.sh"],
        cwd=root,
        env=environment,
        capture_output=True,
        text=True,
        timeout=20,
    )
    assert result.returncode == 1
    assert "Unapproved public path: private/capture.json" in result.stderr


def test_index_secret_is_rejected_and_redacted(publication: tuple[Path, dict[str, str]]) -> None:
    root, environment = publication
    (root / "README.md").write_text("sessionid=" + "0123456789abcdef" * 2, encoding="utf-8")
    subprocess.run(
        ["git", "add", "README.md"],
        cwd=root,
        env=environment,
        check=True,
        capture_output=True,
        timeout=10,
    )
    result = subprocess.run(
        ["bash", "scripts/check-public.sh"],
        cwd=root,
        env=environment,
        capture_output=True,
        text=True,
        timeout=20,
    )
    assert result.returncode == 1
    assert "0123456789abcdef0123456789abcdef" not in result.stdout + result.stderr


def test_removed_secret_still_fails_history_scan(publication: tuple[Path, dict[str, str]]) -> None:
    root, environment = publication
    (root / "README.md").write_text("sessionid=" + "0123456789abcdef" * 2, encoding="utf-8")
    subprocess.run(
        ["git", "commit", "-am", "synthetic leak"],
        cwd=root,
        env=environment,
        check=True,
        capture_output=True,
        timeout=10,
    )
    (root / "README.md").write_text("Public source\n", encoding="utf-8")
    subprocess.run(
        ["git", "commit", "-am", "remove fixture"],
        cwd=root,
        env=environment,
        check=True,
        capture_output=True,
        timeout=10,
    )
    result = subprocess.run(
        ["bash", "scripts/check-public.sh"],
        cwd=root,
        env=environment,
        capture_output=True,
        text=True,
        timeout=20,
    )
    assert result.returncode == 1
    assert "0123456789abcdef0123456789abcdef" not in result.stdout + result.stderr


def test_public_symlink_is_rejected(publication: tuple[Path, dict[str, str]]) -> None:
    root, environment = publication
    (root / "src/replay").mkdir(parents=True)
    (root / "src/replay/link.py").symlink_to("missing")
    subprocess.run(
        ["git", "add", "src/replay/link.py"],
        cwd=root,
        env=environment,
        check=True,
        capture_output=True,
        timeout=10,
    )
    result = subprocess.run(
        ["bash", "scripts/check-public.sh"],
        cwd=root,
        env=environment,
        capture_output=True,
        text=True,
        timeout=20,
    )
    assert result.returncode == 1
    assert "Public symlinks and submodules require" in result.stderr
