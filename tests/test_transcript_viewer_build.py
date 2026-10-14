import shutil
import subprocess
from pathlib import Path

import pytest


@pytest.fixture
def viewer_build(tmp_path: Path) -> tuple[Path, dict[str, str]]:
    root = tmp_path / "checkout with spaces" / "TranscriptViewer"
    (root / "App").mkdir(parents=True)
    (root / "dist/TranscriptViewer.app/Contents/MacOS").mkdir(parents=True)
    (root / "fake-bin").mkdir()
    binaries = tmp_path / "bin"
    binaries.mkdir()
    source = Path(__file__).resolve().parents[1]
    shutil.copyfile(source / "TranscriptViewer/build-app.sh", root / "build-app.sh")
    (root / "App/Info.plist").write_text("fixture", encoding="utf-8")
    (root / "App/AppIcon.icns").write_bytes(b"icon")
    (root / "dist/TranscriptViewer.app/Contents/MacOS/TranscriptViewer").write_text(
        "old", encoding="utf-8"
    )
    (root / "fake-bin/TranscriptViewer").write_text("new", encoding="utf-8")
    (binaries / "swift").write_text(
        '#!/bin/sh\nprintf "%s\\n" "$@" >> "$ARGS_CAPTURE"\n'
        'printf "%s\\n" "$TMPDIR" "$TMP" "$TEMP" "$XDG_CACHE_HOME" '
        '"$CLANG_MODULE_CACHE_PATH" "$SWIFTPM_MODULECACHE_OVERRIDE" '
        '"$LLVM_PROFILE_FILE" > "$ENV_CAPTURE"\n'
        'for arg in "$@"; do if [ "$arg" = "--show-bin-path" ]; then '
        'printf "%s\\n" "$MOCK_BIN"; fi; done\n'
        'exit "${SWIFT_STATUS:-0}"\n',
        encoding="utf-8",
    )
    (binaries / "codesign").write_text(
        '#!/bin/sh\nprintf "codesign %s\\n" "$1" >> "$TRACE"\n'
        'if [ "$1" = "--sign" ] && [ "${FAIL_SIGN:-}" = 1 ]; then exit 9; fi\n',
        encoding="utf-8",
    )
    (binaries / "ps").write_text(
        "#!/bin/sh\n"
        'if [ -f "$RUNNING_STATE" ]; then\n'
        '  if [ "$(cat "$RUNNING_STATE")" = other ]; then exit 0; fi\n'
        '  if [ "$1" = "-p" ]; then printf "%s\\n" "$MOCK_APP/Contents/MacOS/TranscriptViewer"; '
        'else printf "456 %s\\n" "$MOCK_APP/Contents/MacOS/TranscriptViewer"; fi\n'
        "fi\n",
        encoding="utf-8",
    )
    (tmp_path / "bash_env").write_text(
        'kill() { printf "kill %s\\n" "$*" >> "$TRACE"; rm "$RUNNING_STATE"; }\n',
        encoding="utf-8",
    )
    (binaries / "open").write_text(
        '#!/bin/sh\nprintf "open %s\\n" "$*" >> "$TRACE"\nprintf new > "$RUNNING_STATE"\n',
        encoding="utf-8",
    )
    (binaries / "swift").chmod(0o755)
    (binaries / "codesign").chmod(0o755)
    (binaries / "ps").chmod(0o755)
    (binaries / "open").chmod(0o755)
    app = root / "dist/TranscriptViewer.app"
    state = tmp_path / "running"
    state.write_text("old", encoding="utf-8")
    return root, {
        "PATH": f"{binaries}:/usr/bin:/bin",
        "MOCK_BIN": str(root / "fake-bin"),
        "MOCK_APP": str(app),
        "RUNNING_STATE": str(state),
        "TRACE": str(tmp_path / "trace"),
        "HOME": str(tmp_path),
        "BASH_ENV": str(tmp_path / "bash_env"),
        "ARGS_CAPTURE": str(tmp_path / "swift-args"),
        "ENV_CAPTURE": str(tmp_path / "swift-environment"),
    }


def test_signed_upgrade_replaces_old_app_and_restarts_only_matching_process(
    viewer_build: tuple[Path, dict[str, str]],
) -> None:
    root, environ = viewer_build
    result = subprocess.run(
        ["bash", str(root / "build-app.sh")],
        env=environ,
        check=True,
        capture_output=True,
        text=True,
        timeout=10,
    )
    assert (root / "dist/TranscriptViewer.app/Contents/MacOS/TranscriptViewer").read_text(
        encoding="utf-8"
    ) == "new"
    assert not tuple((root / "dist").glob("TranscriptViewer-backup.*"))
    assert not tuple((root / "dist").glob(".TranscriptViewer.stage.*"))
    assert Path(environ["RUNNING_STATE"]).read_text(encoding="utf-8") == "new"
    assert Path(environ["TRACE"]).read_text(encoding="utf-8") == (
        "codesign --sign\ncodesign --verify\nkill -TERM 456\n"
        "codesign --verify\nopen -g -a " + str(root / "dist/TranscriptViewer.app") + "\n"
    )
    assert result.stdout.strip() == str(root / "dist/TranscriptViewer.app")


def test_build_does_not_launch_when_checkout_app_was_not_running(
    viewer_build: tuple[Path, dict[str, str]],
) -> None:
    root, environ = viewer_build
    Path(environ["RUNNING_STATE"]).write_text("other", encoding="utf-8")
    subprocess.run(
        ["bash", str(root / "build-app.sh")],
        env=environ,
        check=True,
        capture_output=True,
        timeout=10,
    )
    assert (root / "dist/TranscriptViewer.app/Contents/MacOS/TranscriptViewer").read_text(
        encoding="utf-8"
    ) == "new"
    assert Path(environ["RUNNING_STATE"]).read_text(encoding="utf-8") == "other"
    assert Path(environ["TRACE"]).read_text(encoding="utf-8") == (
        "codesign --sign\ncodesign --verify\ncodesign --verify\n"
    )


def test_failed_signing_preserves_original_and_running_process(
    viewer_build: tuple[Path, dict[str, str]],
) -> None:
    root, environ = viewer_build
    result = subprocess.run(
        ["bash", str(root / "build-app.sh")],
        env=environ | {"FAIL_SIGN": "1"},
        capture_output=True,
        timeout=10,
    )
    assert result.returncode == 9
    assert (root / "dist/TranscriptViewer.app/Contents/MacOS/TranscriptViewer").read_text(
        encoding="utf-8"
    ) == "old"
    assert Path(environ["RUNNING_STATE"]).read_text(encoding="utf-8") == "old"
    assert Path(environ["TRACE"]).read_text(encoding="utf-8") == "codesign --sign\n"
    assert not tuple((root / "dist").glob("TranscriptViewer-backup.*"))
    assert tuple((root / "dist").glob(".TranscriptViewer.stage.*"))


def test_installer_binds_both_swift_commands_and_environment_to_checkout(
    viewer_build: tuple[Path, dict[str, str]],
) -> None:
    root, environ = viewer_build
    subprocess.run(
        ["bash", str(root / "build-app.sh")],
        env=environ,
        check=True,
        capture_output=True,
        timeout=10,
    )
    work = root.parent / "edits/work/transcript-viewer-build"
    arguments = Path(environ["ARGS_CAPTURE"]).read_text(encoding="utf-8")
    # Normalize just the synthetic checkout prefix so expected contracts stay literal.
    assert arguments.replace(str(root.parent), "CHECKOUT").splitlines() == [
        "build",
        "--package-path",
        "CHECKOUT/TranscriptViewer",
        "--scratch-path",
        "CHECKOUT/TranscriptViewer/.build",
        "--cache-path",
        "CHECKOUT/edits/work/transcript-viewer-build/cache",
        "--config-path",
        "CHECKOUT/edits/work/transcript-viewer-build/config",
        "--security-path",
        "CHECKOUT/edits/work/transcript-viewer-build/security",
        "--manifest-cache",
        "local",
        "--disable-automatic-resolution",
        "-c",
        "release",
        "-Xswiftc",
        "-warnings-as-errors",
        "--product",
        "TranscriptViewer",
        "build",
        "--package-path",
        "CHECKOUT/TranscriptViewer",
        "--scratch-path",
        "CHECKOUT/TranscriptViewer/.build",
        "--cache-path",
        "CHECKOUT/edits/work/transcript-viewer-build/cache",
        "--config-path",
        "CHECKOUT/edits/work/transcript-viewer-build/config",
        "--security-path",
        "CHECKOUT/edits/work/transcript-viewer-build/security",
        "--manifest-cache",
        "local",
        "--disable-automatic-resolution",
        "-c",
        "release",
        "-Xswiftc",
        "-warnings-as-errors",
        "--show-bin-path",
    ]
    environment = Path(environ["ENV_CAPTURE"]).read_text(encoding="utf-8")
    assert environment.replace(str(work), "WORK").splitlines() == [
        "WORK/tmp",
        "WORK/tmp",
        "WORK/tmp",
        "WORK/cache",
        "WORK/module-cache",
        "WORK/module-cache",
        "WORK/profiles/%p-%m.profraw",
    ]


@pytest.mark.parametrize(
    "relative",
    [
        "TranscriptViewer/.build",
        "edits",
        "edits/work",
        "edits/work/transcript-viewer-build",
        "edits/work/transcript-viewer-build/tmp",
        "edits/work/transcript-viewer-build/cache",
        "edits/work/transcript-viewer-build/config",
        "edits/work/transcript-viewer-build/security",
        "edits/work/transcript-viewer-build/module-cache",
        "edits/work/transcript-viewer-build/profiles",
    ],
)
def test_linked_working_directory_fails_before_swift_or_app_operations(
    viewer_build: tuple[Path, dict[str, str]],
    relative: str,
) -> None:
    root, environ = viewer_build
    destination = root.parent / relative
    destination.parent.mkdir(parents=True, exist_ok=True)
    outside = root.parent / "outside"
    outside.mkdir()
    destination.symlink_to(outside)
    result = subprocess.run(
        ["bash", str(root / "build-app.sh")],
        env=environ,
        capture_output=True,
        text=True,
        timeout=10,
    )
    assert result.returncode == 1
    assert "real repository directory" in result.stderr
    assert not Path(environ["ARGS_CAPTURE"]).exists()
    assert not Path(environ["TRACE"]).exists()
    assert list(outside.iterdir()) == []


def test_failed_swift_build_preserves_app_and_releases_lock(
    viewer_build: tuple[Path, dict[str, str]],
) -> None:
    root, environ = viewer_build
    result = subprocess.run(
        ["bash", str(root / "build-app.sh")],
        env=environ | {"SWIFT_STATUS": "19"},
        capture_output=True,
        timeout=10,
    )
    assert result.returncode == 19
    assert not (root / "dist/.TranscriptViewer.install.lock").exists()
    assert not Path(environ["TRACE"]).exists()
    assert (root / "dist/TranscriptViewer.app/Contents/MacOS/TranscriptViewer").read_text(
        encoding="utf-8"
    ) == "old"


def test_linked_package_is_rejected_before_any_build(
    viewer_build: tuple[Path, dict[str, str]],
) -> None:
    root, environ = viewer_build
    retained = root.with_name("retained-package")
    root.rename(retained)
    root.symlink_to(retained)
    result = subprocess.run(
        ["bash", str(root / "build-app.sh")],
        env=environ,
        capture_output=True,
        text=True,
        timeout=10,
    )
    assert result.returncode == 1
    assert "real repository directory" in result.stderr
    assert not Path(environ["ARGS_CAPTURE"]).exists()
