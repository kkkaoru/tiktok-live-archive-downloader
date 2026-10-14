import shutil
import subprocess
from pathlib import Path

import pytest


@pytest.fixture
def launcher(tmp_path: Path) -> tuple[Path, dict[str, str]]:
    root = tmp_path / "project"
    (root / "scripts").mkdir(parents=True)
    (root / "private").mkdir(mode=0o700)
    shutil.copyfile(
        Path(__file__).resolve().parents[1] / "scripts/replay-authorized.sh",
        root / "scripts/replay-authorized.sh",
    )
    token = root / "private/tiktok-sessionid.secret"
    token.write_text("a" * 32 + "\n", encoding="utf-8")
    token.chmod(0o600)
    anchor = root / "private/target-anchor-id"
    anchor.write_text("123\n", encoding="utf-8")
    anchor.chmod(0o600)
    binary = tmp_path / "bin"
    binary.mkdir()
    uv = binary / "uv"
    uv.write_text(
        '#!/bin/bash\n[[ "$TIKTOK_SESSIONID" == "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa" '
        '&& "$TIKTOK_TARGET_ANCHOR_ID" == "123" ]] || exit 5\n'
        'printf "%s\\n" "$*" > "$TRACE"\n',
        encoding="utf-8",
    )
    uv.chmod(0o755)
    return root / "scripts/replay-authorized.sh", {
        "PATH": f"{binary}:/usr/bin:/bin",
        "HOME": str(tmp_path),
        "TRACE": str(tmp_path / "trace"),
    }


def test_recordings_use_private_credentials_not_argv(
    launcher: tuple[Path, dict[str, str]], tmp_path: Path
) -> None:
    script, environ = launcher
    result = subprocess.run(
        ["bash", str(script), "recordings", "--refresh-session"],
        cwd=tmp_path,
        env=environ,
        capture_output=True,
        text=True,
        timeout=10,
    )
    assert result.returncode == 0
    assert result.stdout == ""
    assert result.stderr == ""
    assert Path(environ["TRACE"]).read_text(encoding="utf-8") == (
        "run --locked replay recordings --refresh-session\n"
    )


def test_download_uses_default_store_and_destination(
    launcher: tuple[Path, dict[str, str]], tmp_path: Path
) -> None:
    script, environ = launcher
    result = subprocess.run(
        ["bash", str(script), "download-all", "--workers", "3"],
        cwd=tmp_path,
        env=environ,
        capture_output=True,
        timeout=10,
    )
    assert result.returncode == 0
    assert Path(environ["TRACE"]).read_text(encoding="utf-8") == (
        "run --locked replay download-all downloads --workers 3\n"
    )


def test_download_configures_private_external_directory(
    launcher: tuple[Path, dict[str, str]], tmp_path: Path
) -> None:
    script, environ = launcher
    destination = tmp_path / "external"
    destination.mkdir(mode=0o700)
    result = subprocess.run(
        ["bash", str(script), "download-all"],
        cwd=tmp_path,
        env=environ | {"TIKTOK_DOWNLOAD_DIR": str(destination)},
        capture_output=True,
        timeout=10,
    )
    assert result.returncode == 0
    repeated = subprocess.run(
        ["bash", str(script), "download-all"],
        cwd=tmp_path,
        env=environ | {"TIKTOK_DOWNLOAD_DIR": str(destination)},
        capture_output=True,
        timeout=10,
    )
    assert repeated.returncode == 0
    link = script.parent.parent / "downloads"
    assert link.is_symlink()
    assert link.resolve() == destination
    assert Path(environ["TRACE"]).read_text(encoding="utf-8") == (
        "run --locked replay download-all downloads\n"
    )


@pytest.mark.parametrize("kind", ["nonempty", "different-link", "missing", "relative", "public"])
def test_download_refuses_unsafe_destination_without_changing_data(
    launcher: tuple[Path, dict[str, str]], tmp_path: Path, kind: str
) -> None:
    script, environ = launcher
    destination = tmp_path / "external"
    destination.mkdir(mode=0o700)
    link = script.parent.parent / "downloads"
    if kind == "nonempty":
        link.mkdir()
        (link / "keep.mp4").write_bytes(b"media")
    elif kind == "different-link":
        link.symlink_to(tmp_path)
    elif kind == "missing":
        destination = tmp_path / "missing"
    elif kind == "relative":
        environ["TIKTOK_DOWNLOAD_DIR"] = "relative"
    else:
        destination.chmod(0o755)
    result = subprocess.run(
        ["bash", str(script), "download-all"],
        cwd=tmp_path,
        env=environ | {"TIKTOK_DOWNLOAD_DIR": environ.get("TIKTOK_DOWNLOAD_DIR", str(destination))},
        capture_output=True,
        timeout=10,
    )
    assert result.returncode == 1
    assert not Path(environ["TRACE"]).exists()
    if kind == "nonempty":
        assert (link / "keep.mp4").read_bytes() == b"media"
    elif kind == "different-link":
        assert link.is_symlink() and link.resolve() == tmp_path
    else:
        assert not link.exists()


def test_download_refuses_unavailable_existing_link(
    launcher: tuple[Path, dict[str, str]], tmp_path: Path
) -> None:
    script, environ = launcher
    link = script.parent.parent / "downloads"
    link.symlink_to(tmp_path / "unmounted")
    result = subprocess.run(
        ["bash", str(script), "download-all"],
        cwd=tmp_path,
        env=environ,
        capture_output=True,
        timeout=10,
    )
    assert result.returncode == 1
    assert link.is_symlink()
    assert not Path(environ["TRACE"]).exists()


def test_download_replaces_only_empty_bootstrap_directory(
    launcher: tuple[Path, dict[str, str]], tmp_path: Path
) -> None:
    script, environ = launcher
    destination = tmp_path / "external"
    destination.mkdir(mode=0o700)
    (script.parent.parent / "downloads").mkdir(mode=0o700)
    result = subprocess.run(
        ["bash", str(script), "download-all"],
        cwd=tmp_path,
        env=environ | {"TIKTOK_DOWNLOAD_DIR": str(destination)},
        capture_output=True,
        timeout=10,
    )
    assert result.returncode == 0
    assert (script.parent.parent / "downloads").is_symlink()


def test_missing_selector_stops_before_uv(
    launcher: tuple[Path, dict[str, str]], tmp_path: Path
) -> None:
    script, environ = launcher
    (script.parent.parent / "private/target-anchor-id").unlink()
    result = subprocess.run(
        ["bash", str(script), "recordings"],
        cwd=tmp_path,
        env=environ,
        capture_output=True,
        timeout=10,
    )
    assert result.returncode == 1
    assert not Path(environ["TRACE"]).exists()


def test_insecure_session_stops_before_uv(
    launcher: tuple[Path, dict[str, str]], tmp_path: Path
) -> None:
    script, environ = launcher
    (script.parent.parent / "private/tiktok-sessionid.secret").chmod(0o644)
    result = subprocess.run(
        ["bash", str(script), "recordings"],
        cwd=tmp_path,
        env=environ,
        capture_output=True,
        timeout=10,
    )
    assert result.returncode == 1
    assert not Path(environ["TRACE"]).exists()


@pytest.mark.parametrize("option", ["--anchor-id", "--user", "--store", "--host"])
def test_conflicting_option_stops_before_uv(
    launcher: tuple[Path, dict[str, str]], tmp_path: Path, option: str
) -> None:
    script, environ = launcher
    result = subprocess.run(
        ["bash", str(script), "recordings", option, "456"],
        cwd=tmp_path,
        env=environ,
        capture_output=True,
        timeout=10,
    )
    assert result.returncode == 2
    assert b"456" not in result.stderr
    assert not Path(environ["TRACE"]).exists()


def test_inherited_handle_stops_before_uv(
    launcher: tuple[Path, dict[str, str]], tmp_path: Path
) -> None:
    script, environ = launcher
    result = subprocess.run(
        ["bash", str(script), "recordings"],
        cwd=tmp_path,
        env=environ | {"TIKTOK_TARGET_USER": "@other"},
        capture_output=True,
        timeout=10,
    )
    assert result.returncode == 2
    assert b"@other" not in result.stderr
    assert not Path(environ["TRACE"]).exists()


def test_malformed_selector_stops_before_uv(
    launcher: tuple[Path, dict[str, str]], tmp_path: Path
) -> None:
    script, environ = launcher
    (script.parent.parent / "private/target-anchor-id").write_text("123\n4", encoding="utf-8")
    result = subprocess.run(
        ["bash", str(script), "recordings"],
        cwd=tmp_path,
        env=environ,
        capture_output=True,
        timeout=10,
    )
    assert result.returncode == 1
    assert not Path(environ["TRACE"]).exists()
