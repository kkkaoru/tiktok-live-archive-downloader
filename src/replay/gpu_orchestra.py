"""GPU orchestration of this repository's video jobs: weighted GPU units, critical-path priority.

``gpu_lease`` counts GPU jobs; jobs differ widely in how much of the GPU they use (two STTN
cleanup chunk streams already saturate it: one chunk per 30 s at 2 and at 3 streams), so a flat
count either idles the GPU or crowds it. Here each job class has a weight in GPU units and a job
is admitted while the running weights plus its own fit ``budget`` (1.0, plus ``overcommit``).

Priority is per recording first: the recording rank of ``gpu_lease`` (the ranked queue of
``recording_priority``: urgent, then newest broadcast or a manual order), then within a recording
the job class: cleanup chunks (the first-run critical path), the short caption-path steps
(onsets, compose), the steps with slack (yolo-track, separation, asr), and the repository's own
benchmarks (``run``). A waiter's class improves by one every ``agingSeconds`` it waits.
Admission is greedy in that order: a waiter is admitted when the units the waiters ahead of it
can use now still leave room for it, so a lower job uses units no higher waiter fits into
(work-conserving). A recording ranked below another active one (holding or waiting for units)
borrows only ``shortJobs`` (chunk-sized: cleanup chunks, onsets), so a higher recording is never
blocked long by a job that cannot be preempted. Once the top waiter has waited ``starveSeconds``,
nothing behind it starts until it fits. Running jobs are never preempted. The media engine
(compose, finish) and Compressor keep their own ``gpu_lease`` directories; they are not units.

Compatibility with orchestrators that run the earlier ``gpu_lease`` code: an orchestrated job
also holds one of the ``gpu_lease`` slots (so earlier code still sees it), and every slot held
without an orchestrated marker counts as ``legacyWeight`` units. Orchestration is used only while
``settings.json`` says ``"enabled": true``; otherwise ``gpu_lease`` behaves as before.

Ledger: ``held/`` and ``waiting/`` hold flock-held marker files (a killed job's marker frees at
once); one ledger lock makes check-and-take atomic between processes.
"""

from __future__ import annotations

import argparse
import fcntl
import json
import os
import re
import subprocess
import sys
import time
from collections.abc import Callable, Generator, Sequence
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path

CLASSES = {
    "cleanup-chunk": 0,
    "onsets": 1,
    "compose": 1,
    "yolo-track": 2,
    "separation": 2,
    "asr": 2,
    "bench": 3,
}
# GPU units per job. The STTN chunk is measured (2 streams saturate the GPU); the others are
# conservative estimates until measured (see edits/work/pipeline-speed-20261024 for the method).
DEFAULT_WEIGHTS = {
    "cleanup-chunk": 0.5,
    "onsets": 0.25,
    "compose": 0.5,
    "yolo-track": 0.5,
    "separation": 0.5,
    "asr": 0.5,
    "bench": 0.5,
}
MILLI = 1000
EPSILON = 1e-9
BUDGET = 1.0
LEGACY_WEIGHT = 0.5
AGING_SECONDS = 600.0
STARVE_SECONDS = 1800.0
POLL_SECONDS = 5.0
MAX_SETTINGS_BYTES = 4096
SHORT_JOBS = ("cleanup-chunk", "onsets")
SETTINGS_KEYS = frozenset(
    {"enabled", "budget", "overcommit", "weights", "legacyWeight", "agingSeconds",
     "starveSeconds", "shortJobs"}
)  # fmt: skip
HELD = re.compile(
    r"^(?P<job>[a-z-]+)\.(?P<owner>[A-Za-z0-9_-]+)\.(?P<rank>[0-9]+)\.(?P<start>[0-9]+)"
    r"\.[0-9]+\.[0-9]+\.(?P<milli>[0-9]+)\.(?P<slot>[0-9]+)$"
)
WAITING = re.compile(
    r"^(?P<job>[a-z-]+)\.(?P<owner>[A-Za-z0-9_-]+)\.(?P<rank>[0-9]+)\.(?P<start>[0-9]+)"
    r"\.(?P<since>[0-9]+)\.[0-9]+\.[0-9]+\.(?P<milli>[0-9]+)$"
)


@dataclass(frozen=True, slots=True)
class Settings:
    enabled: bool
    budget: float
    weights: dict[str, float]
    legacy_weight: float
    aging: float
    starve: float
    short: frozenset[str] = frozenset(SHORT_JOBS)


@dataclass(frozen=True, slots=True)
class Held:
    job: str
    owner: str
    rank: int
    start: int
    weight: float
    slot: int


@dataclass(frozen=True, slots=True)
class Waiting:
    job: str
    owner: str
    rank: int
    start: int
    since_ns: int
    weight: float
    path: Path

    def key(self, now_ns: int, aging: float) -> tuple[int, int, int, int]:
        """Recording rank and start, then the effective (aged) class, then waiting order."""
        steps = int((now_ns - self.since_ns) / 1e9 / aging)
        return (self.rank, self.start, max(0, CLASSES[self.job] - steps), self.since_ns)


def root() -> Path:
    return Path(__file__).resolve().parents[2] / "edits" / "work" / "gpu-orchestra"


def _units(value: object, name: str, upper: float) -> float:
    if isinstance(value, bool) or not isinstance(value, int | float) or not 0 < value <= upper:
        raise ValueError(f"GPU orchestra {name} must be a number in (0, {upper}]")
    return float(value)


def settings(directory: Path) -> Settings:
    """The orchestration settings; a missing file means disabled with the defaults."""
    path = directory / "settings.json"
    raw: object = {}
    if path.exists():
        data = path.read_bytes()
        if len(data) > MAX_SETTINGS_BYTES:
            raise ValueError("GPU orchestra settings are too large")
        raw = json.loads(data.decode("utf-8"))
    if not isinstance(raw, dict) or not set(raw) <= SETTINGS_KEYS:
        raise ValueError("GPU orchestra settings take " + ", ".join(sorted(SETTINGS_KEYS)))
    enabled = raw.get("enabled", False)
    if type(enabled) is not bool:
        raise ValueError("GPU orchestra enabled must be true or false")
    weights = raw.get("weights", {})
    if not isinstance(weights, dict) or not set(weights) <= set(CLASSES):
        raise ValueError("GPU orchestra weights name only known jobs")
    short = raw.get("shortJobs", list(SHORT_JOBS))
    if not isinstance(short, list) or not set(short) <= set(CLASSES):
        raise ValueError("GPU orchestra shortJobs name only known jobs")
    overcommit = raw.get("overcommit", 0)
    if overcommit != 0:
        overcommit = _units(overcommit, "overcommit", 1.0)
    return Settings(
        enabled,
        _units(raw.get("budget", BUDGET), "budget", 4.0) + float(overcommit),
        {
            job: _units(weights.get(job, weight), job, 4.0)
            for job, weight in DEFAULT_WEIGHTS.items()
        },
        _units(raw.get("legacyWeight", LEGACY_WEIGHT), "legacyWeight", 4.0),
        _units(raw.get("agingSeconds", AGING_SECONDS), "agingSeconds", 86400.0),
        _units(raw.get("starveSeconds", STARVE_SECONDS), "starveSeconds", 86400.0),
        frozenset(str(job) for job in short),
    )


def enabled(directory: Path | None = None) -> bool:
    return settings(root() if directory is None else directory).enabled


@contextmanager
def _ledger(directory: Path) -> Generator[None]:
    directory.mkdir(mode=0o700, parents=True, exist_ok=True)
    descriptor = os.open(directory / "ledger.lock", os.O_RDWR | os.O_CREAT, 0o600)
    try:
        fcntl.flock(descriptor, fcntl.LOCK_EX)
        yield
    finally:
        os.close(descriptor)


def _live(path: Path) -> bool:
    """True while some process holds the marker; a stale marker is removed."""
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


def _hold(path: Path) -> int:
    path.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
    descriptor = os.open(path, os.O_RDWR | os.O_CREAT, 0o600)
    fcntl.flock(descriptor, fcntl.LOCK_EX)
    return descriptor


def held(directory: Path) -> list[Held]:
    """Live orchestrated jobs."""
    folder = directory / "held"
    if not folder.is_dir():
        return []
    found = [(HELD.match(path.name), path) for path in folder.iterdir()]
    return [
        Held(match["job"], match["owner"], int(match["rank"]), int(match["start"]),
             int(match["milli"]) / MILLI, int(match["slot"]))
        for match, path in found
        if match and _live(path)
    ]  # fmt: skip


def waiting(directory: Path) -> list[Waiting]:
    folder = directory / "waiting"
    if not folder.is_dir():
        return []
    rows = []
    for path in folder.iterdir():
        match = WAITING.match(path.name)
        if match and match["job"] in CLASSES and _live(path):
            rows.append(
                Waiting(match["job"], match["owner"], int(match["rank"]), int(match["start"]),
                        int(match["since"]), int(match["milli"]) / MILLI, path)
            )  # fmt: skip
    return rows


def legacy_slots(leases: Path, capacity: int, owned: set[int]) -> int:
    """``gpu_lease`` slots held by jobs without an orchestrated marker (earlier code)."""
    count = 0
    for slot in range(capacity):
        if slot in owned:
            continue
        path = leases / f"slot-{slot}.lock"
        if not path.exists():
            continue
        descriptor = os.open(path, os.O_RDWR)
        try:
            fcntl.flock(descriptor, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            count += 1
        finally:
            os.close(descriptor)
    return count


def admitted(
    me: Waiting,
    *,
    running: float,
    queue: Sequence[Waiting],
    current: Settings,
    now_ns: int,
    busy: Sequence[tuple[int, int]] = (),
) -> bool:
    """Greedy admission in priority order: is ``me`` given units in this pass?

    ``busy`` are the (rank, start) of recordings holding units. A job heavier than the whole
    budget still runs alone on an idle GPU.
    """
    free = current.budget - running
    top = min([*busy, *((item.rank, item.start) for item in queue)], default=(0, 0))
    for item in sorted(queue, key=lambda entry: entry.key(now_ns, current.aging)):
        borrowing = (item.rank, item.start) > top and item.job not in current.short
        fits = not borrowing and (item.weight <= free + EPSILON or free >= current.budget - EPSILON)
        if item.path == me.path:
            return fits
        if fits:
            free -= item.weight
        elif (now_ns - item.since_ns) / 1e9 >= current.starve:
            return False  # a starving waiter ahead: nothing behind it starts until it fits
    return False


class Grant:
    """Units held by one admitted job; ``release`` (or the process ending) frees them."""

    def __init__(self, descriptor: int, path: Path, weight: float) -> None:
        self.descriptor, self.path, self.weight = descriptor, path, weight

    def release(self) -> None:
        if self.descriptor >= 0:
            self.path.unlink(missing_ok=True)
            os.close(self.descriptor)
            self.descriptor = -1


class Ticket:
    """One job's place in the orchestra queue (its waiting marker is held while it waits)."""

    def __init__(
        self,
        directory: Path,
        job: str,
        owner: str,
        *,
        rank: int,
        start: int,
        weight: float | None = None,
    ) -> None:
        if job not in CLASSES or not re.fullmatch(r"[A-Za-z0-9_-]+", owner):
            raise ValueError("Unknown orchestrated job or owner")
        self.directory, self.job, self.owner = directory, job, owner
        self.rank, self.start = rank, max(start, 0)
        self.weight = settings(directory).weights[job] if weight is None else weight
        self.since = time.time_ns()
        milli = round(self.weight * MILLI)
        name = f"{job}.{owner}.{rank}.{max(start, 0)}.{self.since}.{os.getpid()}.{id(self)}.{milli}"
        self.me = Waiting(job, owner, rank, max(start, 0), self.since, milli / MILLI,
                          directory / "waiting" / name)  # fmt: skip
        self.descriptor: int | None = None

    def try_take(
        self, *, leases: Path, capacity: int, take_slot: Callable[[], tuple[int, int] | None]
    ) -> tuple[Grant, int] | None:
        """Units and one ``gpu_lease`` slot (its descriptor), or None while not admitted."""
        if self.descriptor is None:
            self.descriptor = _hold(self.me.path)
        with _ledger(self.directory):
            current = settings(self.directory)
            jobs = held(self.directory)
            owned = {job.slot for job in jobs}
            running = sum(job.weight for job in jobs)
            running += current.legacy_weight * legacy_slots(leases, capacity, owned)
            if not admitted(
                self.me, running=running, queue=waiting(self.directory), current=current,
                now_ns=time.time_ns(), busy=[(job.rank, job.start) for job in jobs],
            ):  # fmt: skip
                return None
            taken = take_slot()
            if taken is None:
                return None
            slot, lease = taken
            milli = round(self.weight * MILLI)
            name = (
                f"{self.job}.{self.owner}.{self.rank}.{self.start}.{os.getpid()}.{id(self)}"
                f".{milli}.{slot}"
            )
            path = self.directory / "held" / name
            grant = Grant(_hold(path), path, self.weight)
        self.close()
        return grant, lease

    def close(self) -> None:
        if self.descriptor is not None:
            self.me.path.unlink(missing_ok=True)
            os.close(self.descriptor)
            self.descriptor = None


def status(directory: Path, leases: Path, capacity: int) -> dict[str, object]:
    current = settings(directory)
    jobs = held(directory)
    owned = {job.slot for job in jobs}
    legacy = legacy_slots(leases, capacity, owned)
    now = time.time_ns()
    queue = sorted(waiting(directory), key=lambda item: item.key(now, current.aging))
    return {
        "enabled": current.enabled,
        "budget": current.budget,
        "unitsUsed": round(sum(job.weight for job in jobs) + legacy * current.legacy_weight, 3),
        "running": [
            {"job": job.job, "owner": job.owner, "rank": job.rank, "units": job.weight}
            for job in jobs
        ],
        "legacySlotsHeld": legacy,
        "waiting": [
            {
                "job": item.job,
                "owner": item.owner,
                "units": item.weight,
                "rank": item.rank,
                "class": item.key(now, current.aging)[2],
                "waitedSeconds": round((now - item.since_ns) / 1e9, 1),
            }
            for item in queue
        ],  # fmt: skip
    }


def main(argv: Sequence[str] | None = None) -> int:
    from replay import gpu_lease  # deferred: gpu_lease builds on this module

    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    commands.add_parser("status")
    runner = commands.add_parser("run", help="run a benchmark as the lowest GPU class")
    runner.add_argument("--units", type=float, default=DEFAULT_WEIGHTS["bench"])
    runner.add_argument("--name", default="bench")
    runner.add_argument("command", nargs=argparse.REMAINDER)
    args = parser.parse_args(argv)
    leases = gpu_lease.root()
    if args.command == "status":
        capacity, _ = gpu_lease.settings(leases)
        print(json.dumps(status(root(), leases, capacity), indent=1))
        return 0
    command = [part for part in args.command if part != "--"]
    if not command:
        parser.error("run needs a command")
    with gpu_lease.lease(leases, "bench", step=args.name, job="bench", units=args.units):
        return subprocess.call(command)


if __name__ == "__main__":
    sys.exit(main())
