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


@pytest.mark.parametrize("filename", ["stateful_guard_fixture.py", "unapproved_fixture.py"])
def test_static_guard_fixture_exception_is_narrow(
    publication: tuple[Path, dict[str, str]], filename: str
) -> None:
    root, environment = publication
    fixture = root / "tests" / filename
    fixture.parent.mkdir()
    shutil.copyfile(Path(__file__).with_name("stateful_guard_fixture.py"), fixture)
    subprocess.run(
        ["git", "add", str(fixture.relative_to(root))],
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
    assert (filename, result.returncode) in {
        ("stateful_guard_fixture.py", 0),
        ("unapproved_fixture.py", 1),
    }


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


@pytest.mark.parametrize(
    "path",
    [
        "native/apple-pro-apps/Package.swift",
        "native/apple-pro-apps/Package.resolved",
        "native/apple-pro-apps/Sources/ProAppsCore/Example.swift",
        "native/apple-pro-apps/Tests/ProAppsNativeTests/Example.swift",
    ],
)
def test_shared_native_source_is_approved(
    publication: tuple[Path, dict[str, str]], path: str
) -> None:
    root, environment = publication
    source = root / path
    source.parent.mkdir(parents=True)
    source.write_text("{}\n", encoding="utf-8")
    subprocess.run(
        ["git", "add", path], cwd=root, env=environment, check=True, capture_output=True, timeout=10
    )
    result = subprocess.run(
        ["bash", "scripts/check-public.sh"],
        cwd=root,
        env=environment,
        capture_output=True,
        timeout=20,
    )
    assert result.returncode == 0


@pytest.mark.parametrize("name", ["black.mp4", "quadrants-30.mp4"])
def test_only_verified_synthetic_native_media_is_approved(
    publication: tuple[Path, dict[str, str]], name: str
) -> None:
    root, environment = publication
    relative = Path("native/apple-pro-apps/Tests/ProAppsNativeTests/Fixtures") / name
    destination = root / relative
    destination.parent.mkdir(parents=True)
    shutil.copyfile(Path(__file__).resolve().parents[1] / relative, destination)
    subprocess.run(
        ["git", "add", str(relative)],
        cwd=root,
        env=environment,
        check=True,
        capture_output=True,
        timeout=10,
    )
    accepted = subprocess.run(
        ["bash", "scripts/check-public.sh"],
        cwd=root,
        env=environment,
        capture_output=True,
        timeout=20,
    )
    assert accepted.returncode == 0
    destination.write_bytes(b"different media must not inherit the fixture exception")
    subprocess.run(
        ["git", "add", str(relative)],
        cwd=root,
        env=environment,
        check=True,
        capture_output=True,
        timeout=10,
    )
    rejected = subprocess.run(
        ["bash", "scripts/check-public.sh"],
        cwd=root,
        env=environment,
        capture_output=True,
        text=True,
        timeout=20,
    )
    assert rejected.returncode == 1
    assert "Synthetic fixture hash differs:" in rejected.stderr


@pytest.mark.parametrize(
    "path",
    [
        "native/apple-pro-apps/.build/state.json",
        "native/apple-pro-apps/model/weights.npz",
        "native/apple-pro-apps/Tests/ProAppsNativeTests/Fixtures/recording.mp4",
    ],
)
def test_native_build_models_and_other_media_remain_private(
    publication: tuple[Path, dict[str, str]], path: str
) -> None:
    root, environment = publication
    artifact = root / path
    artifact.parent.mkdir(parents=True)
    artifact.write_text("local artifact\n", encoding="utf-8")
    subprocess.run(
        ["git", "add", path], cwd=root, env=environment, check=True, capture_output=True, timeout=10
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
    assert "Unapproved public path:" in result.stderr


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
