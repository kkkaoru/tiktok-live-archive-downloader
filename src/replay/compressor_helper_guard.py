"""Explicit, temporary reclamation of old selected-app RegisterProExtension helpers.

Age and process identity do not prove semantic idleness or launch causality.
Only opt-in callers authorize SIGTERM; there is no SIGKILL or application restart.
ps start identities have second resolution and recheck/signal cannot be atomic.
"""

from __future__ import annotations

import fcntl
import json
import os
import re
import signal
import stat
import subprocess
import sys
import time
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import TextIO

from replay.core import ReplayError

THRESHOLD = 300
MIN_AGE_SECONDS = 300
CLEANUP_SECONDS = 30.0
LOCK_SECONDS = 35.0
PS_SECONDS = 2.0
OBSERVATION_BACKOFF = (0.1, 0.2)
MAX_ELAPSED_TOKEN_CHARS = 64
MAX_LOG_BYTES = 16 * 1024 * 1024
FIELDS = "pid=,ppid=,uid=,etime=,lstart=,comm="
HELPER = "Helpers/RegisterProExtension.app/Contents/MacOS/RegisterProExtension"
ELAPSED = re.compile(
    r"(?:(?P<days>[0-9]+)-)?(?:(?P<hours>[0-9]+):)?(?P<minutes>[0-9]{2}):(?P<seconds>[0-9]{2})"
)


class ElapsedParseError(ReplayError):
    """Invalid elapsed observation, with only bounded, sanitized token diagnostics."""

    def __init__(self, token: str) -> None:
        safe_token = re.sub(r"[^0-9:-]", "?", token[:MAX_ELAPSED_TOKEN_CHARS])
        super().__init__(f"Invalid ps elapsed time: {safe_token!r}")


@dataclass(frozen=True)
class Process:
    pid: int
    ppid: int
    uid: int
    age: int
    start: str
    executable: str

    @property
    def identity(self) -> tuple[int, int, str, str]:
        return self.pid, self.uid, self.start, self.executable


def elapsed_seconds(text: str) -> int:
    match = ELAPSED.fullmatch(text)
    if match is None:
        raise ElapsedParseError(text)
    days, hours, minutes, seconds = (int(value or 0) for value in match.groups())
    if (
        minutes >= 60
        or seconds >= 60
        or hours >= 24
        or (match.group("days") is not None and match.group("hours") is None)
    ):
        raise ElapsedParseError(text)
    return days * 86400 + hours * 3600 + minutes * 60 + seconds


def parse_processes(text: str) -> list[Process]:
    rows: list[Process] = []
    seen: set[int] = set()
    for line in text.splitlines():
        if not line.strip():
            continue
        fields = line.split(maxsplit=9)
        if (
            len(fields) != 10
            or not all(re.fullmatch(r"[0-9]+", v) for v in fields[:2])
            or re.fullmatch(r"-?[0-9]+", fields[2]) is None
        ):
            raise ReplayError("Malformed ps metadata")
        pid, ppid, uid = map(int, fields[:3])
        start = " ".join(fields[4:9])
        datetime.strptime(start, "%a %b %d %H:%M:%S %Y")
        if pid <= 0 or pid in seen:
            raise ReplayError("Invalid or duplicate ps PID")
        seen.add(pid)
        rows.append(Process(pid, ppid, uid, elapsed_seconds(fields[3]), start, fields[9]))
    return rows


def selected_helper(executable: Path) -> str:
    selected = executable.resolve(strict=True)
    if selected.parts[-3:] != ("Contents", "MacOS", "Compressor") or not selected.is_file():
        raise ReplayError("Guard requires a selected Compressor app executable")
    app = selected.parents[2]
    helper = app / "Contents" / HELPER
    if app.suffix != ".app" or helper.resolve(strict=True) != helper or not helper.is_file():
        raise ReplayError("Guard requires the exact unlinked selected-app helper")
    return str(helper)


def guard_directory(directory: Path) -> Path:
    root = Path.cwd().resolve(strict=True)
    target = directory.resolve()
    repository = Path(__file__).resolve().parents[2]
    if not root.is_relative_to(repository) or target == root or not target.is_relative_to(root):
        raise ReplayError("Guard directory must be inside the repository cwd")
    target.mkdir(mode=0o700, parents=True, exist_ok=True)
    info = target.stat()
    if info.st_uid != os.getuid() or stat.S_IMODE(info.st_mode) != 0o700:
        raise ReplayError("Guard directory must be private and owned by this UID")
    return target


@contextmanager
def private_file(path: Path) -> Iterator[TextIO]:
    flags = os.O_RDWR | os.O_APPEND | os.O_NOFOLLOW | os.O_NONBLOCK
    try:
        descriptor = os.open(path, flags | os.O_CREAT | os.O_EXCL, 0o600)
    except FileExistsError:
        descriptor = os.open(path, flags)
    with os.fdopen(descriptor, "a+", encoding="utf-8") as stream:
        info = os.fstat(stream.fileno())
        if (
            not stat.S_ISREG(info.st_mode)
            or info.st_uid != os.getuid()
            or stat.S_IMODE(info.st_mode) != 0o600
            or info.st_nlink != 1
        ):
            raise ReplayError("Guard files must be private, single-link regular files")
        yield stream


def evidence(stream: TextIO, event: str, **values: object) -> None:
    line = json.dumps({"event": event, "time": time.time(), **values}) + "\n"
    if os.fstat(stream.fileno()).st_size + len(line.encode("utf-8")) > MAX_LOG_BYTES:
        raise ReplayError("Guard evidence limit reached; preserve evidence before resuming")
    stream.write(line)
    stream.flush()
    os.fsync(stream.fileno())


def processes(directory: Path, *, deadline: float, pid: int | None = None) -> list[Process]:
    original_error: ElapsedParseError | None = None
    attempt = 0
    while True:
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            if original_error is not None:
                raise original_error
            raise ReplayError("Helper guard deadline reached")
        selector = ["-ax"] if pid is None else ["-p", str(pid)]
        result = subprocess.run(
            ["/bin/ps", "-ww", *selector, "-o", FIELDS],
            stdin=subprocess.DEVNULL,
            capture_output=True,
            text=True,
            timeout=min(PS_SECONDS, remaining),
            check=False,
            env={
                **os.environ,
                "LC_ALL": "C",
                "TZ": "UTC",
                "TMPDIR": str(directory),
                "TMP": str(directory),
                "TEMP": str(directory),
            },
        )
        if (
            pid is not None
            and result.returncode == 1
            and not result.stdout.strip()
            and not result.stderr.strip()
        ):
            return []
        if result.returncode or result.stderr.strip():
            raise ReplayError(f"Helper guard ps failed (exit {result.returncode})")
        try:
            rows = parse_processes(result.stdout)
        except ElapsedParseError as exc:
            print(str(exc), file=sys.stderr, flush=True)
            if original_error is None:
                original_error = exc
            if attempt == len(OBSERVATION_BACKOFF):
                raise original_error from None
            delay = OBSERVATION_BACKOFF[attempt]
            if time.monotonic() + delay >= deadline:
                raise original_error from None
            time.sleep(delay)
            attempt += 1
            continue
        if pid is None and not rows:
            raise ReplayError("Helper guard ps returned an empty enumeration")
        if pid is not None and (len(rows) > 1 or any(row.pid != pid for row in rows)):
            raise ReplayError("Helper guard ps returned an unexpected PID")
        return rows


def eligible(process: Process, helper: str) -> bool:
    return process.executable == helper and process.uid == os.getuid() and process.ppid == 1


def reclaim(directory: Path, helper: str, stream: TextIO) -> None:
    deadline = time.monotonic() + CLEANUP_SECONDS
    rows = processes(directory, deadline=deadline)
    scoped = [row for row in rows if eligible(row, helper)]
    evidence(
        stream,
        "observation",
        count=len(scoped),
        helper=helper,
        uid=os.getuid(),
        threshold=THRESHOLD,
        minimum_age=MIN_AGE_SECONDS,
    )
    if len(scoped) < THRESHOLD:
        return
    signaled: set[tuple[int, int, str, str]] = set()
    for row in scoped:
        if row.age < MIN_AGE_SECONDS:
            continue
        evidence(stream, "recheck", process=asdict(row))
        current = processes(directory, deadline=deadline, pid=row.pid)
        if (
            not current
            or current[0].identity != row.identity
            or not eligible(current[0], helper)
            or current[0].age < MIN_AGE_SECONDS
        ):
            evidence(
                stream,
                "disappeared_or_changed",
                pid=row.pid,
                observed=[asdict(process) for process in current],
            )
            continue
        # No intervening I/O between the final identity/eligibility check and SIGTERM.
        try:
            os.kill(row.pid, signal.SIGTERM)
        except ProcessLookupError:
            evidence(stream, "disappeared_before_signal", pid=row.pid)
            continue
        except OSError as exc:
            evidence(stream, "signal_failed", pid=row.pid, error=type(exc).__name__)
            raise
        signaled.add(row.identity)
        evidence(stream, "sigterm_sent", process=asdict(current[0]))
    while True:
        rows = processes(directory, deadline=deadline)
        count = sum(eligible(row, helper) for row in rows)
        remaining = signaled.intersection(row.identity for row in rows)
        evidence(stream, "post_signal", count=count, pending_exits=len(remaining))
        if count < THRESHOLD and not remaining:
            evidence(stream, "recovered", count=count)
            return
        if time.monotonic() + 1 >= deadline:
            raise ReplayError("Helper count/exits could not be made safe within the guard deadline")
        time.sleep(1)


def check(directory: Path, executable: Path) -> None:
    """Serialize bounded observation/reclamation; fail before any new submission."""
    target = guard_directory(directory)
    helper = selected_helper(executable)
    with private_file(target / "guard.lock") as lock:
        deadline = time.monotonic() + LOCK_SECONDS
        while True:
            try:
                fcntl.flock(lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
                break
            except BlockingIOError:
                if time.monotonic() + 0.25 >= deadline:
                    raise ReplayError("Helper guard lock wait timed out") from None
                time.sleep(0.25)
        with private_file(target / "guard.jsonl") as stream:
            try:
                reclaim(target, helper, stream)
            except (ReplayError, OSError, ValueError, subprocess.SubprocessError) as exc:
                evidence(stream, "blocked", error=type(exc).__name__)
                raise
