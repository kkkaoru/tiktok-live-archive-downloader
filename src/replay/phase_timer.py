"""Measure one command, with a private repository-local append-only timing log.

`python -m replay.phase_timer LOG PHASE -- COMMAND...` validates LOG before the
child starts. Epoch start/end are diagnostic clock readings; elapsed seconds use
the monotonic clock. No command arguments or environment values enter the log.
The child retains its stdout/stderr and the caller's configured scratch paths.
"""

from __future__ import annotations

import fcntl
import json
import os
import stat
import subprocess
import sys
import time
from collections.abc import Iterator, Sequence
from contextlib import contextmanager
from pathlib import Path

from replay.core import ReplayError

MAX_LOG_BYTES = 16 * 1024 * 1024
MAX_ROW_BYTES = 4096
MAX_PHASE_CHARS = 256
LOCK_SECONDS = 5.0
LOCK_BACKOFF = 0.01


def _log_path(log: Path) -> Path:
    repository = Path(__file__).resolve().parents[2]
    path = log.absolute()
    if (
        Path.cwd().resolve() != repository
        or path != path.resolve()
        or not path.is_relative_to(repository / "edits" / "work")
        or not path.parent.is_dir()
    ):
        raise ReplayError("Timer log must have an existing canonical repository work parent")
    return path


def _private_log(descriptor: int, path: Path) -> os.stat_result:
    opened = os.fstat(descriptor)
    named = path.lstat()
    if (
        not stat.S_ISREG(opened.st_mode)
        or opened.st_uid != os.getuid()
        or stat.S_IMODE(opened.st_mode) != 0o600
        or opened.st_nlink != 1
        or (opened.st_dev, opened.st_ino) != (named.st_dev, named.st_ino)
        or path != path.resolve()
        or opened.st_size > MAX_LOG_BYTES - MAX_ROW_BYTES
    ):
        raise ReplayError(
            "Timer log must remain an owned, private, single-link bounded regular file"
        )
    return opened


def _complete_tail(descriptor: int, opened: os.stat_result) -> None:
    if opened.st_size and os.pread(descriptor, 1, opened.st_size - 1) != b"\n":
        raise ReplayError("Timer log has a partial final record; it was retained without appending")


def _check_parent(descriptor: int, path: Path, expected: os.stat_result) -> None:
    current = os.fstat(descriptor)
    named = path.parent.lstat()
    identity = (expected.st_dev, expected.st_ino)
    if (
        current.st_uid != os.getuid()
        or (current.st_dev, current.st_ino) != identity
        or (named.st_dev, named.st_ino) != identity
        or _log_path(path) != path
    ):
        raise ReplayError("Timer parent changed during preflight")


def _open_descriptor(parent: int, path: Path, expected: os.stat_result) -> int:
    try:
        existing = path.lstat()
    except FileNotFoundError:
        existing = None
    flags = os.O_RDWR | os.O_APPEND | os.O_NOFOLLOW | os.O_NONBLOCK
    if existing is None:
        flags |= os.O_CREAT
    try:
        descriptor = os.open(path.name, flags, 0o600, dir_fd=parent)
    except FileNotFoundError:
        if existing is not None:
            raise
        # One first-creation ENOENT was observed with simultaneous writers. Its
        # cause is unknown; retry only the same open before any child is started,
        # after proving that both the retained parent FD and its name are stable.
        _check_parent(parent, path, expected)
        sys.stderr.write("PHASE_TIMER_OPEN_RETRY errno=2\n")
        return os.open(path.name, flags, 0o600, dir_fd=parent)
    try:
        opened = os.fstat(descriptor)
        if existing is not None and (existing.st_dev, existing.st_ino) != (
            opened.st_dev,
            opened.st_ino,
        ):
            raise ReplayError("Timer existing log changed during preflight")
    except (OSError, ReplayError):
        os.close(descriptor)
        raise
    return descriptor


@contextmanager
def _open_log(path: Path) -> Iterator[int]:
    parent = os.open(path.parent, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW)
    try:
        opened_parent = os.fstat(parent)
        _check_parent(parent, path, opened_parent)
        descriptor = _open_descriptor(parent, path, opened_parent)
        try:
            _check_parent(parent, path, opened_parent)
            with _append_lease(descriptor):
                _complete_tail(descriptor, _private_log(descriptor, path))
            yield descriptor
        finally:
            os.close(descriptor)
    finally:
        os.close(parent)


@contextmanager
def _append_lease(descriptor: int) -> Iterator[None]:
    deadline = time.monotonic() + LOCK_SECONDS
    while True:
        try:
            fcntl.flock(descriptor, fcntl.LOCK_EX | fcntl.LOCK_NB)
            break
        except BlockingIOError:
            if time.monotonic() >= deadline:
                raise ReplayError("Timer append lease exceeded its finite deadline") from None
            time.sleep(LOCK_BACKOFF)
    try:
        yield
    finally:
        fcntl.flock(descriptor, fcntl.LOCK_UN)


def _write_all(descriptor: int, data: bytes) -> None:
    offset = 0
    while offset < len(data):
        written = os.write(descriptor, data[offset:])
        if written <= 0:
            raise ReplayError("Timer append made no progress; the partial record is retained")
        offset += written


def run_timed(log: Path, phase: str, command: Sequence[str]) -> int:
    """Run one child after log preflight; failures never start a replacement child.

    A logging error after execution raises with the child exit code. The log may
    contain a partial final record after an I/O failure; it is never truncated.
    """
    if (
        not phase
        or len(phase) > MAX_PHASE_CHARS
        or any(character in phase for character in "\0\n\r")
        or not command
        or any(not value or "\0" in value for value in command)
    ):
        raise ReplayError("Timer requires a bounded phase and a valid command argument vector")
    path = _log_path(log)
    with _open_log(path) as descriptor:
        start = time.time()
        monotonic_start = time.monotonic()
        try:
            code = subprocess.run(list(command), stdin=subprocess.DEVNULL, check=False).returncode
        except OSError as error:
            sys.stderr.write(f"PHASE_TIMER_LAUNCH_ERROR errno={error.errno}\n")
            code = 127
        end = time.time()
        seconds = time.monotonic() - monotonic_start
        if code < 0:
            code = 128 - code
        row = {
            "phase": phase,
            "start": start,
            "end": end,
            "seconds": round(seconds, 3),
            "exit": code,
        }
        data = (json.dumps(row, ensure_ascii=False, allow_nan=False) + "\n").encode("utf-8")
        try:
            with _append_lease(descriptor):
                _complete_tail(descriptor, _private_log(descriptor, path))
                if len(data) > MAX_ROW_BYTES:
                    raise ReplayError("Timer record exceeds its byte budget")
                _write_all(descriptor, data)
                os.fsync(descriptor)
        except (OSError, ReplayError) as error:
            raise ReplayError(f"Timer logging failed after child exit {code}: {error}") from error
    return code


def main(argv: list[str] | None = None) -> int:
    arguments = sys.argv[1:] if argv is None else argv
    if len(arguments) < 4 or arguments[2] != "--":
        sys.stderr.write("usage: python -m replay.phase_timer LOG PHASE -- COMMAND...\n")
        return 2
    try:
        return run_timed(Path(arguments[0]), arguments[1], arguments[3:])
    except (ReplayError, OSError, ValueError) as error:
        sys.stderr.write(f"PHASE_TIMER_ERROR {error}\n")
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
