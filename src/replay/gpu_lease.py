"""Machine-wide GPU leases shared by every recording pipeline and cleanup chunk job.

Per-pipeline slots cannot see other pipelines, and cleanup chunk jobs (the largest GPU user)
had no GPU bound at all, so two recordings could run a dozen GPU jobs that only slowed each
other. Each lease here is an exclusive ``flock`` on one slot file. ``flock`` (unlike POSIX
record locks) also excludes threads of one process that open the file separately, and the
kernel drops it when its holder exits, so a killed job never leaks a lease.

Recordings are served one at a time: a waiter leaves free leases to the waiters of recordings
ranked ahead of it, so the leading recording finishes first and the others only use leases it
leaves idle (its serial stages, its memory waits, free leases beyond its waiters' count). Rank
is the position in ``priority`` (replay IDs in ``settings.json``, read at every attempt), then
``REPLAY_PIPELINE_STARTED`` (inherited by every stage): a recording pipeline sets it from its
broadcast time (``newest_first``), a job started outside a pipeline ranks by its own start
time. A waiter is an flock-held marker file, so a killed waiter stops counting at once.
``capacity`` is the number of leases. Leases never preempt a running job. ``holders.jsonl``
logs every acquire and release (recording, step, pid, wait and hold seconds) so an idle or a
long-waited lease can be explained afterwards.

A waiter that names its ``job`` is orchestrated by ``gpu_orchestra`` while that is enabled: it
then also needs the job's GPU units, priority follows the job's class before the recording rank,
and the units appear in ``holders.jsonl``. Slots and waiter markers keep their format, so
processes running earlier code still see every orchestrated job.

``lease-urgent.json`` beside the lease directories (written by ``recording_priority``) names
urgent recordings: while one of them waits for any lease, no other recording is granted a new
lease in any directory (running jobs keep theirs).
"""

from __future__ import annotations

import fcntl
import json
import os
import sys
import threading
import time
from collections.abc import Generator
from contextlib import contextmanager
from pathlib import Path

from replay import gpu_orchestra

DEFAULT_CAPACITY = 4
MAX_CAPACITY = 16
POLL_SECONDS = 5.0
MAX_SETTINGS_BYTES = 4096
STARTED_VARIABLE = "REPLAY_PIPELINE_STARTED"
PROCESS_STARTED = time.time_ns()  # rank of a job started outside a pipeline
# newest_first mirrors broadcast times below every current clock reading (valid until 2065), so
# the newest broadcast ranks first and recording pipelines rank ahead of loose jobs.
MIRROR_NS = 3 * 10**18
# shortcut: one ~150-byte line per acquire or release, never rotated; rotate it at megabytes.
HOLDERS = "holders.jsonl"
URGENT = "lease-urgent.json"
MAX_URGENT_BYTES = 4096
# This process's held leases by descriptor: (directory, replay ID, step, acquired monotonic).
_holders: dict[int, tuple[Path, str, str, float]] = {}
_grants: dict[int, gpu_orchestra.Grant] = {}  # the GPU units an orchestrated lease holds
_holders_lock = threading.Lock()


def root() -> Path:
    """The lease directory of this checkout, wherever the caller runs from (src/replay/..)."""
    return Path(__file__).resolve().parents[2] / "edits" / "work" / "gpu-leases"


def started() -> int:
    """This pipeline's start time (ns), the queue order of its recording."""
    value = os.environ.get(STARTED_VARIABLE, "")
    return int(value) if value.isdigit() else PROCESS_STARTED


def newest_first(broadcast_ns: int) -> int:
    """A rank (``REPLAY_PIPELINE_STARTED``) that serves later broadcasts first."""
    if not 0 < broadcast_ns < MIRROR_NS:
        raise ValueError("A broadcast time must be a positive epoch in nanoseconds before 2065")
    return MIRROR_NS - broadcast_ns


def settings(directory: Path) -> tuple[int, tuple[str, ...]]:
    """Capacity and priority replay IDs (first is served first); a missing file: defaults.

    ``manual: true`` only tells ``lease_priority`` to leave a hand-set priority alone.
    """
    path = directory / "settings.json"
    if not path.exists():
        return DEFAULT_CAPACITY, ()
    data = path.read_bytes()
    if len(data) > MAX_SETTINGS_BYTES:
        raise ValueError("GPU lease settings are too large")
    raw: object = json.loads(data.decode("utf-8"))
    if not isinstance(raw, dict) or not set(raw) <= {"capacity", "priority", "manual"}:
        raise ValueError("GPU lease settings take only capacity, priority and manual")
    if type(raw.get("manual", False)) is not bool:
        raise ValueError("GPU lease manual must be true or false")
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


def marker_held(path: Path) -> bool:
    """True while a process holds this waiter marker (a stale one is removed)."""
    return _held(path)


def urgent_waiting(work: Path, replay_id: str) -> bool:
    """True while an urgent recording other than ``replay_id`` waits for any lease."""
    path = work / URGENT
    if not path.is_file() or path.stat().st_size > MAX_URGENT_BYTES:
        return False
    raw: object = json.loads(path.read_text(encoding="utf-8"))
    urgent = raw.get("urgent") if isinstance(raw, dict) else None
    if not isinstance(urgent, list) or replay_id in urgent:
        return False
    names = {str(item) for item in urgent}
    for folder in work.glob("*-leases/waiting"):
        for marker in folder.iterdir():
            if marker.name.partition(".")[0] in names and _held(marker):
                return True
    return False


def _free_slots(directory: Path, capacity: int, wanted: int) -> list[tuple[int, int]]:
    """(slot, locked descriptor) of up to ``wanted`` free leases."""
    taken: list[tuple[int, int]] = []
    for slot in range(capacity):
        if len(taken) == wanted:
            break
        descriptor = os.open(directory / f"slot-{slot}.lock", os.O_RDWR | os.O_CREAT, 0o600)
        try:
            fcntl.flock(descriptor, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            os.close(descriptor)
            continue
        taken.append((slot, descriptor))
    return taken


def _log(directory: Path, row: dict[str, object]) -> None:
    """Append one holder line; a failed write is reported, never lets a lease leak or stick."""
    line = json.dumps({"t": round(time.time(), 3), **row}) + "\n"
    try:
        with (directory / HOLDERS).open("a", encoding="utf-8") as stream:
            stream.write(line)  # one short O_APPEND write: concurrent holders do not interleave
    except OSError as error:
        print(f"GPU lease holder log not written: {error}", file=sys.stderr)


class Waiter:
    """One job's place in the queue for a lease; ``close`` (or acquiring) leaves the queue."""

    def __init__(
        self,
        directory: Path,
        replay_id: str,
        *,
        start: int | None = None,
        step: str = "",
        job: str = "",
        units: float | None = None,
    ) -> None:
        if not replay_id or "." in replay_id or "/" in replay_id:
            raise ValueError("A replay ID must be a non-empty name without dots or slashes")
        self.directory, self.replay_id, self.step = directory, replay_id, step
        self.start = started() if start is None else start
        self.path = directory / "waiting" / f"{replay_id}.{self.start}.{os.getpid()}.{id(self)}"
        self.descriptor: int | None = None
        self.created = time.monotonic()
        self.job, self.units = job, units
        self.ticket: gpu_orchestra.Ticket | None = None

    def try_acquire(self) -> int | None:
        """A held lease's descriptor, or None while every free one is owed to waiters ahead."""
        capacity, priority = settings(self.directory)
        if self.descriptor is None:
            self.path.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
            self.descriptor = os.open(self.path, os.O_RDWR | os.O_CREAT, 0o600)
            fcntl.flock(self.descriptor, fcntl.LOCK_EX)
        mine = _rank(self.replay_id, self.start, priority)
        if urgent_waiting(self.directory.parent, self.replay_id):
            return None  # an urgent recording waits: no new grant to anyone else
        if self.job and gpu_orchestra.enabled():
            return self._orchestrated(capacity, mine)
        ahead = 0
        for marker in self.path.parent.iterdir():
            replay, _, rest = marker.name.partition(".")
            start = rest.partition(".")[0]
            if (
                replay != self.replay_id
                and start.isdigit()
                and _rank(replay, int(start), priority) < mine
                and _held(marker)
            ):
                ahead += 1
        # Work-conserving: a lease is taken only while one stays free for every waiter ahead.
        free = _free_slots(self.directory, capacity, ahead + 1)
        if len(free) <= ahead:
            for _, descriptor in free:
                os.close(descriptor)
            return None
        (_, lease), *spare = free
        for _, descriptor in spare:
            os.close(descriptor)
        return self._granted(lease, None)

    def _orchestrated(self, capacity: int, rank: tuple[int, int]) -> int | None:
        """Units from ``gpu_orchestra`` plus one slot; the class order replaces the rank rule."""
        if self.ticket is None:
            self.ticket = gpu_orchestra.Ticket(
                gpu_orchestra.root(), self.job, self.replay_id, rank=rank[0], start=rank[1],
                weight=self.units,
            )  # fmt: skip

        def slot() -> tuple[int, int] | None:
            free = _free_slots(self.directory, capacity, 1)
            return free[0] if free else None

        taken = self.ticket.try_take(leases=self.directory, capacity=capacity, take_slot=slot)
        if taken is None:
            return None
        grant, lease = taken
        return self._granted(lease, grant)

    def _granted(self, lease: int, grant: gpu_orchestra.Grant | None) -> int:
        self.close()
        with _holders_lock:
            _holders[lease] = (self.directory, self.replay_id, self.step, time.monotonic())
            if grant is not None:
                _grants[lease] = grant
        _log(
            self.directory,
            {"event": "acquire", "recording": self.replay_id, "step": self.step,
             "pid": os.getpid(), "waitedSeconds": round(time.monotonic() - self.created, 3),
             "units": None if grant is None else grant.weight},
        )  # fmt: skip
        return lease

    def close(self) -> None:
        if self.descriptor is not None:
            self.path.unlink(missing_ok=True)
            os.close(self.descriptor)
            self.descriptor = None
        if self.ticket is not None:
            self.ticket.close()


def release(descriptor: int) -> None:
    with _holders_lock:
        holder = _holders.pop(descriptor, None)
        grant = _grants.pop(descriptor, None)
    if grant is not None:
        grant.release()
    os.close(descriptor)  # closing the only descriptor drops the flock
    if holder is not None:
        directory, replay_id, step, acquired = holder
        _log(
            directory,
            {"event": "release", "recording": replay_id, "step": step, "pid": os.getpid(),
             "heldSeconds": round(time.monotonic() - acquired, 3)},
        )  # fmt: skip


@contextmanager
def lease(
    directory: Path,
    replay_id: str,
    *,
    poll_seconds: float = POLL_SECONDS,
    step: str = "",
    job: str = "",
    units: float | None = None,
) -> Generator[None]:
    """Hold one GPU lease (and ``job``'s GPU units when orchestrated) for the block."""
    waiter = Waiter(directory, replay_id, step=step, job=job, units=units)
    try:
        while (descriptor := waiter.try_acquire()) is None:
            time.sleep(poll_seconds)
    finally:
        waiter.close()
    try:
        yield
    finally:
        release(descriptor)
