"""Machine-wide GPU leases shared by every recording pipeline and cleanup chunk job.

Per-pipeline slots cannot see other pipelines, and cleanup chunk jobs (the largest GPU user)
had no GPU bound at all, so two recordings could run a dozen GPU jobs that only slowed each
other. Each lease here is an exclusive ``flock`` on one slot file. ``flock`` (unlike POSIX
record locks) also excludes threads of one process that open the file separately, and the
kernel drops it when its holder exits, so a killed job never leaks a lease.

Recordings are served one at a time: a waiter yields while a recording ranked ahead of it is
waiting too, so the leading recording finishes first and the others only use leases it leaves
idle (its serial stages, its memory waits). Rank is the position in ``priority`` (replay IDs in
``settings.json``, read at every attempt), then the pipeline start time
(``REPLAY_PIPELINE_STARTED``, inherited by every stage). A waiter is an flock-held marker file,
so a killed waiter stops counting at once. ``capacity`` is the number of leases. Leases never
preempt a running job.
"""

from __future__ import annotations

import fcntl
import json
import os
import time
from collections.abc import Generator
from contextlib import contextmanager
from pathlib import Path

DEFAULT_CAPACITY = 4
MAX_CAPACITY = 16
POLL_SECONDS = 5.0
MAX_SETTINGS_BYTES = 4096
STARTED_VARIABLE = "REPLAY_PIPELINE_STARTED"
PROCESS_STARTED = time.time_ns()  # rank of a job started outside a pipeline


def root() -> Path:
    """The lease directory of this checkout, wherever the caller runs from (src/replay/..)."""
    return Path(__file__).resolve().parents[2] / "edits" / "work" / "gpu-leases"


def started() -> int:
    """This pipeline's start time (ns), the queue order of its recording."""
    value = os.environ.get(STARTED_VARIABLE, "")
    return int(value) if value.isdigit() else PROCESS_STARTED


def settings(directory: Path) -> tuple[int, tuple[str, ...]]:
    """Capacity and priority replay IDs (first is served first); a missing file: defaults."""
    path = directory / "settings.json"
    if not path.exists():
        return DEFAULT_CAPACITY, ()
    data = path.read_bytes()
    if len(data) > MAX_SETTINGS_BYTES:
        raise ValueError("GPU lease settings are too large")
    raw: object = json.loads(data.decode("utf-8"))
    if not isinstance(raw, dict) or not set(raw) <= {"capacity", "priority"}:
        raise ValueError("GPU lease settings take only capacity and priority")
    capacity = raw.get("capacity", DEFAULT_CAPACITY)
    priority = raw.get("priority", [])
    if type(capacity) is not int or not 1 <= capacity <= MAX_CAPACITY:
        raise ValueError("GPU lease capacity must be an integer from 1 to 16")
    if not isinstance(priority, list) or not all(isinstance(item, str) for item in priority):
        raise ValueError("GPU lease priority must list replay IDs")
    return capacity, tuple(priority)


def _rank(replay_id: str, start: int, priority: tuple[str, ...]) -> tuple[int, int]:
    return (priority.index(replay_id) if replay_id in priority else len(priority), start)


def _held(path: Path) -> bool:
    """True while another process holds the waiter marker; a stale marker is removed."""
    try:
        descriptor = os.open(path, os.O_RDONLY)
    except FileNotFoundError:
        return False
    try:
        fcntl.flock(descriptor, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        return True
    finally:
        os.close(descriptor)
    path.unlink(missing_ok=True)
    return False


def _free_slot(directory: Path, capacity: int) -> int | None:
    for slot in range(capacity):
        descriptor = os.open(directory / f"slot-{slot}.lock", os.O_RDWR | os.O_CREAT, 0o600)
        try:
            fcntl.flock(descriptor, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            os.close(descriptor)
            continue
        return descriptor
    return None


class Waiter:
    """One job's place in the queue for a lease; ``close`` (or acquiring) leaves the queue."""

    def __init__(self, directory: Path, replay_id: str, *, start: int | None = None) -> None:
        if not replay_id or "." in replay_id or "/" in replay_id:
            raise ValueError("A replay ID must be a non-empty name without dots or slashes")
        self.directory, self.replay_id = directory, replay_id
        self.start = started() if start is None else start
        self.path = directory / "waiting" / f"{replay_id}.{self.start}.{os.getpid()}.{id(self)}"
        self.descriptor: int | None = None

    def try_acquire(self) -> int | None:
        """A held lease's descriptor, or None while it is taken or owed to a recording ahead."""
        capacity, priority = settings(self.directory)
        if self.descriptor is None:
            self.path.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
            self.descriptor = os.open(self.path, os.O_RDWR | os.O_CREAT, 0o600)
            fcntl.flock(self.descriptor, fcntl.LOCK_EX)
        mine = _rank(self.replay_id, self.start, priority)
        for marker in self.path.parent.iterdir():
            replay, _, rest = marker.name.partition(".")
            start = rest.partition(".")[0]
            if (
                replay != self.replay_id
                and start.isdigit()
                and _rank(replay, int(start), priority) < mine
                and _held(marker)
            ):
                return None
        lease = _free_slot(self.directory, capacity)
        if lease is not None:
            self.close()
        return lease

    def close(self) -> None:
        if self.descriptor is not None:
            self.path.unlink(missing_ok=True)
            os.close(self.descriptor)
            self.descriptor = None


def release(descriptor: int) -> None:
    os.close(descriptor)  # closing the only descriptor drops the flock


@contextmanager
def lease(
    directory: Path, replay_id: str, *, poll_seconds: float = POLL_SECONDS
) -> Generator[None]:
    """Hold one GPU lease for the duration of the block, waiting for its turn."""
    waiter = Waiter(directory, replay_id)
    try:
        while (descriptor := waiter.try_acquire()) is None:
            time.sleep(poll_seconds)
    finally:
        waiter.close()
    try:
        yield
    finally:
        release(descriptor)
