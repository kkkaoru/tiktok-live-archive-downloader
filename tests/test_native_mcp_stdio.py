"""Generic MCP stdio isolation using a synthetic executable, never an app."""

import os
import shutil
import subprocess
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pytest


@pytest.fixture
def stdio_project(tmp_path: Path) -> Path:
    root = tmp_path / "repository with spaces"
    (root / "scripts").mkdir(parents=True)
    (root / "fixed").mkdir()
    shutil.copyfile(
        Path(__file__).resolve().parents[1] / "scripts/native-mcp-stdio.sh",
        root / "scripts/native-mcp-stdio.sh",
    )
    binary = root / "fixed/synthetic-mcp"
    binary.write_text(
        "#!/bin/bash\nset -euo pipefail\n"
        "printf 'child diagnostic\\n' >&2\n"
        'printf "%s\\n" "$PWD" "$TMPDIR" "$TMP" "$TEMP" "$XDG_CACHE_HOME" '
        '"$CLANG_MODULE_CACHE_PATH" "$SWIFTPM_MODULECACHE_OVERRIDE" '
        '"$LLVM_PROFILE_FILE" > "$TMPDIR/environment.txt"\n'
        'printf "%s\\n" "$@" > "$TMPDIR/arguments.txt"\n'
        'printf "%s\\n" "$$" > "$TMPDIR/pid.txt"\n'
        'printf "synthetic stdout\\n"\ncat\nexit "${NATIVE_FAKE_EXIT:-0}"\n',
        encoding="utf-8",
    )
    binary.chmod(0o700)
    return root


def test_stdio_preserved_and_all_child_storage_private(stdio_project: Path) -> None:
    result = subprocess.run(
        [
            "/bin/bash",
            str(stdio_project / "scripts/native-mcp-stdio.sh"),
            "mcp-tests",
            str(stdio_project / "fixed/synthetic-mcp"),
            "serve",
            "literal ; $HOME",
        ],
        cwd=stdio_project.parent,
        input=b'{"jsonrpc":"2.0"}\n',
        capture_output=True,
        timeout=10,
        check=False,
    )
    assert result.returncode == 0
    assert result.stdout == b'synthetic stdout\n{"jsonrpc":"2.0"}\n'
    assert result.stderr == b""
    history = stdio_project / "edits/work/mcp-tests/mcp-stdio"
    launches = tuple(history.iterdir())
    assert len(launches) == 1
    launch = launches[0]
    assert (launch / "stderr.log").read_text(encoding="utf-8") == "child diagnostic\n"
    assert (launch / "tmp/arguments.txt").read_text(encoding="utf-8") == "serve\nliteral ; $HOME\n"
    assert (launch / "tmp/environment.txt").read_text(encoding="utf-8").splitlines() == [
        str(stdio_project),
        str(launch / "tmp"),
        str(launch / "tmp"),
        str(launch / "tmp"),
        str(launch / "cache"),
        str(launch / "module-cache"),
        str(launch / "module-cache"),
        str(launch / "profiles/%p-%m.profraw"),
    ]
    assert history.stat().st_mode & 0o777 == 0o700
    assert launch.stat().st_mode & 0o777 == 0o700
    assert (launch / "tmp").stat().st_mode & 0o777 == 0o700
    assert (launch / "stderr.log").stat().st_mode & 0o777 == 0o600
    assert (launch / "launch.properties").stat().st_mode & 0o777 == 0o600
    assert (launch / "binary.sha256").stat().st_mode & 0o777 == 0o600
    assert (launch / "launch.properties").read_text(encoding="utf-8").splitlines()[0] == (
        "pid=" + (launch / "tmp/pid.txt").read_text(encoding="utf-8").strip()
    )


def test_repeated_launch_keeps_exclusive_history_and_native_exit(stdio_project: Path) -> None:
    command = [
        "/bin/bash",
        str(stdio_project / "scripts/native-mcp-stdio.sh"),
        "mcp-tests",
        str(stdio_project / "fixed/synthetic-mcp"),
        "serve",
    ]
    first = subprocess.run(command, input=b"first", capture_output=True, timeout=10, check=False)
    history = stdio_project / "edits/work/mcp-tests/mcp-stdio"
    original = next(history.iterdir())
    original_bytes = (original / "stderr.log").read_bytes()
    second = subprocess.run(
        command,
        input=b"second",
        env={**os.environ, "NATIVE_FAKE_EXIT": "17"},
        capture_output=True,
        timeout=10,
        check=False,
    )
    assert first.returncode == 0
    assert second.returncode == 17
    assert first.stdout == b"synthetic stdout\nfirst"
    assert second.stdout == b"synthetic stdout\nsecond"
    assert first.stderr == second.stderr == b""
    assert len(tuple(history.iterdir())) == 2
    assert (original / "stderr.log").read_bytes() == original_bytes


@pytest.mark.parametrize("name", ["..", ".", "../escape", "/absolute", "with space", "x" * 65])
def test_escaping_work_names_refused_before_storage(stdio_project: Path, name: str) -> None:
    result = subprocess.run(
        [
            "/bin/bash",
            str(stdio_project / "scripts/native-mcp-stdio.sh"),
            name,
            str(stdio_project / "fixed/synthetic-mcp"),
        ],
        capture_output=True,
        timeout=10,
        check=False,
    )
    assert result.returncode == 2
    assert result.stdout == result.stderr == b""
    assert not (stdio_project / "edits").exists()


@pytest.mark.parametrize(
    "component", ["edits", "edits/work", "edits/work/mcp-tests", "edits/work/mcp-tests/mcp-stdio"]
)
def test_symlinked_storage_never_written(stdio_project: Path, component: str) -> None:
    destination = stdio_project / component
    destination.parent.mkdir(parents=True, exist_ok=True)
    outside = stdio_project.parent / "other-owned-data"
    outside.mkdir()
    destination.symlink_to(outside, target_is_directory=True)
    result = subprocess.run(
        [
            "/bin/bash",
            str(stdio_project / "scripts/native-mcp-stdio.sh"),
            "mcp-tests",
            str(stdio_project / "fixed/synthetic-mcp"),
        ],
        capture_output=True,
        timeout=10,
        check=False,
    )
    assert result.returncode == 2
    assert result.stdout == result.stderr == b""
    assert tuple(outside.iterdir()) == ()


@pytest.mark.parametrize("mode", [0o755, 0o770, 0o777])
def test_shared_history_permissions_refused_without_chmod(stdio_project: Path, mode: int) -> None:
    history = stdio_project / "edits/work/mcp-tests/mcp-stdio"
    history.mkdir(parents=True)
    history.chmod(mode)
    result = subprocess.run(
        [
            "/bin/bash",
            str(stdio_project / "scripts/native-mcp-stdio.sh"),
            "mcp-tests",
            str(stdio_project / "fixed/synthetic-mcp"),
        ],
        capture_output=True,
        timeout=10,
        check=False,
    )
    assert result.returncode == 2
    assert history.stat().st_mode & 0o777 == mode
    assert tuple(history.iterdir()) == ()


@pytest.mark.parametrize(
    "kind", ["outside", "linked-file", "linked-parent", "non-executable", "writable"]
)
def test_unbound_binary_rejected_before_launch(stdio_project: Path, kind: str) -> None:
    binary = stdio_project / "fixed/synthetic-mcp"
    if kind == "outside":
        binary = stdio_project.parent / "outside-mcp"
        shutil.copyfile(stdio_project / "fixed/synthetic-mcp", binary)
        binary.chmod(0o700)
    elif kind == "linked-file":
        binary = stdio_project / "fixed/link-mcp"
        binary.symlink_to("synthetic-mcp")
    elif kind == "linked-parent":
        (stdio_project / "linked").symlink_to(stdio_project / "fixed", target_is_directory=True)
        binary = stdio_project / "linked/synthetic-mcp"
    elif kind == "non-executable":
        binary.chmod(0o600)
    else:
        binary.chmod(0o720)
    result = subprocess.run(
        [
            "/bin/bash",
            str(stdio_project / "scripts/native-mcp-stdio.sh"),
            "mcp-tests",
            str(binary),
        ],
        capture_output=True,
        timeout=10,
        check=False,
    )
    assert result.returncode == 2
    assert result.stdout == result.stderr == b""
    assert not (stdio_project / "edits").exists()


def test_inherited_stderr_file_is_untouched(stdio_project: Path) -> None:
    inherited = stdio_project.parent / "existing-daemon-error.log"
    inherited.write_bytes(b"preserve\n")
    with inherited.open("ab") as stream:
        result = subprocess.run(
            [
                "/bin/bash",
                str(stdio_project / "scripts/native-mcp-stdio.sh"),
                "mcp-tests",
                str(stdio_project / "fixed/synthetic-mcp"),
                "serve",
            ],
            input=b"request",
            stdout=subprocess.PIPE,
            stderr=stream,
            timeout=10,
            check=False,
        )
    assert result.returncode == 0
    assert result.stdout == b"synthetic stdout\nrequest"
    assert inherited.read_bytes() == b"preserve\n"


def test_missing_arguments_emit_no_transport_noise(stdio_project: Path) -> None:
    result = subprocess.run(
        ["/bin/bash", str(stdio_project / "scripts/native-mcp-stdio.sh")],
        capture_output=True,
        timeout=10,
        check=False,
    )
    assert result.returncode == 2
    assert result.stdout == result.stderr == b""


def test_concurrent_launches_reserve_distinct_private_histories(stdio_project: Path) -> None:
    command = [
        "/bin/bash",
        str(stdio_project / "scripts/native-mcp-stdio.sh"),
        "mcp-tests",
        str(stdio_project / "fixed/synthetic-mcp"),
        "serve",
    ]
    with ThreadPoolExecutor(max_workers=2) as pool:
        first = pool.submit(
            subprocess.run, command, input=b"first", capture_output=True, timeout=10, check=False
        )
        second = pool.submit(
            subprocess.run, command, input=b"second", capture_output=True, timeout=10, check=False
        )
        first_result = first.result()
        second_result = second.result()
    assert first_result.returncode == second_result.returncode == 0
    assert first_result.stdout == b"synthetic stdout\nfirst"
    assert second_result.stdout == b"synthetic stdout\nsecond"
    assert first_result.stderr == second_result.stderr == b""
    history = stdio_project / "edits/work/mcp-tests/mcp-stdio"
    assert len(tuple(history.iterdir())) == 2
