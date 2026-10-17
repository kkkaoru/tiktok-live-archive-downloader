import shutil
import subprocess
from pathlib import Path

import pytest


@pytest.fixture
def native_swift(tmp_path: Path) -> tuple[Path, dict[str, str]]:
    root = tmp_path / "checkout with spaces"
    (root / "scripts").mkdir(parents=True)
    (root / "native/apple-pro-apps").mkdir(parents=True)
    binaries = tmp_path / "bin"
    binaries.mkdir()
    shutil.copyfile(
        Path(__file__).resolve().parents[1] / "scripts/native-swift.sh",
        root / "scripts/native-swift.sh",
    )
    command = binaries / "swift"
    command.write_text(
        '#!/bin/sh\nprintf "%s\\n" "$@" > "$ARG_CAPTURE"\n'
        'printf "%s\\n" "$TMPDIR" "$TMP" "$TEMP" "$XDG_CACHE_HOME" '
        '"$CLANG_MODULE_CACHE_PATH" "$SWIFTPM_MODULECACHE_OVERRIDE" '
        '"$LLVM_PROFILE_FILE" > "$ENV_CAPTURE"\n'
        'exit "${SWIFT_STATUS:-0}"\n',
        encoding="utf-8",
    )
    command.chmod(0o700)
    return root, {
        "PATH": f"{binaries}:/usr/bin:/bin",
        "HOME": str(tmp_path),
        "ARG_CAPTURE": str(tmp_path / "args"),
        "ENV_CAPTURE": str(tmp_path / "environment"),
    }


@pytest.mark.parametrize("action", ["build", "test"])
@pytest.mark.parametrize("work_name", [None, "native-text-recognition-v1"])
def test_native_runner_uses_shared_source_and_local_working_storage(
    native_swift: tuple[Path, dict[str, str]], action: str, work_name: str | None
) -> None:
    root, environment = native_swift
    subprocess.run(
        ["bash", str(root / "scripts/native-swift.sh"), action, "--jobs", "4"],
        cwd=root.parent,
        env={**environment, **({"REPLAY_NATIVE_WORK_NAME": work_name} if work_name else {})},
        check=True,
        capture_output=True,
        timeout=10,
    )
    work = root / "edits/work" / (work_name or "native-swift")
    assert Path(environment["ARG_CAPTURE"]).read_text(encoding="utf-8").splitlines() == [
        action,
        "--package-path",
        str(root / "native/apple-pro-apps"),
        "--scratch-path",
        str(work / "build"),
        "--cache-path",
        str(root / "edits/work/swiftpm-shared/cache"),
        "--config-path",
        str(work / "config"),
        "--security-path",
        str(work / "security"),
        "--force-resolved-versions",
        "-Xswiftc",
        "-warnings-as-errors",
        "--jobs",
        "4",
    ]
    assert Path(environment["ENV_CAPTURE"]).read_text(encoding="utf-8").splitlines() == [
        str(work / "tmp"),
        str(work / "tmp"),
        str(work / "tmp"),
        str(root / "edits/work/swiftpm-shared/cache"),
        str(root / "edits/work/swiftpm-shared/module-cache"),
        str(root / "edits/work/swiftpm-shared/module-cache"),
        str(work / "profiles/%p-%m.profraw"),
    ]


@pytest.mark.parametrize("name", ["..", ".", "../escape", "sub/dir", "/absolute", "with space"])
def test_native_runner_rejects_escaping_names(
    native_swift: tuple[Path, dict[str, str]], name: str
) -> None:
    root, environment = native_swift
    result = subprocess.run(
        ["bash", str(root / "scripts/native-swift.sh"), "build"],
        env={**environment, "REPLAY_NATIVE_WORK_NAME": name},
        capture_output=True,
        timeout=10,
    )
    assert result.returncode == 2
    assert not (root / "edits").exists()
    assert not Path(environment["ARG_CAPTURE"]).exists()


def test_native_runner_preserves_failure_status(native_swift: tuple[Path, dict[str, str]]) -> None:
    root, environment = native_swift
    result = subprocess.run(
        ["bash", str(root / "scripts/native-swift.sh"), "build"],
        env={**environment, "SWIFT_STATUS": "17"},
        capture_output=True,
        timeout=10,
    )
    assert result.returncode == 17


def test_native_runner_rejects_unknown_actions_before_work(
    native_swift: tuple[Path, dict[str, str]],
) -> None:
    root, environment = native_swift
    result = subprocess.run(
        ["bash", str(root / "scripts/native-swift.sh"), "clean"],
        env=environment,
        capture_output=True,
        timeout=10,
    )
    assert result.returncode == 2
    assert not (root / "edits").exists()
    assert not Path(environment["ARG_CAPTURE"]).exists()


def test_native_runner_rejects_linked_work_storage_before_writing(
    native_swift: tuple[Path, dict[str, str]],
) -> None:
    root, environment = native_swift
    outside = root.parent / "other-data"
    outside.mkdir()
    (root / "edits").symlink_to(outside, target_is_directory=True)
    result = subprocess.run(
        ["bash", str(root / "scripts/native-swift.sh"), "test"],
        env=environment,
        capture_output=True,
        timeout=10,
    )
    assert result.returncode == 1
    assert tuple(outside.iterdir()) == ()
    assert not Path(environment["ARG_CAPTURE"]).exists()
