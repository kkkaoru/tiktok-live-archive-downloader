"""Machine-wide memory reservations for GPU/ML steps and cleanup chunks, plus per-process caps.

Free memory alone admitted work that grew later: two recordings' jobs swapped a 48 GB machine
into watchdog resets (10-18, 10-19, 10-22). Each job now reserves its peak footprint before it
starts and holds the reservation until it ends. A job is admitted while the reservations held
across all pipelines plus its own fit the budget (physical memory minus ``headroomGiB`` for the
OS and other apps), the free-memory level is at least ``MEMORY_FREE_MIN`` and the kernel's
memory pressure is normal. With nothing reserved a job is admitted unless pressure is critical,
so one job larger than the budget still runs alone. Nothing running is ever stopped.

Other processes count too (another session's ~22-32 GB model server is a fact of life): a job
starts only while the measured free memory (``kern.memorystatus_level`` of physical memory) covers
it, with or without reservations held. Pressure counts as critical, so nothing starts, while the
machine pages in faster than
``SWAP_IN_LIMIT`` (``vm_stat`` Swapins + Pageins, per minute between samples at least
``SWAP_SAMPLE_SECONDS`` apart, shared by every caller in ``vm-sample.json``): the 10-24 thrash
paged in 8 GB/min while normal running stays under about 1 GB/min, and a level or pressure reading
alone reacted only after the machine was already swapping.

A reservation is an flock-held marker file, as in ``gpu_lease``, so a killed job releases it at
once; one ledger lock makes check-and-reserve atomic between processes. A job reserves the
larger of its estimate below and the largest physical footprint measured on earlier runs
(``footprints.json``) times ``MARGIN``; ``settings.json`` may set ``budgetGiB``,
``headroomGiB``, ``peaksGiB`` per job and ``hardCaps``.

The footprint is each process's lifetime maximum ``ri_phys_footprint`` (``proc_pid_rusage``),
summed over the job's process group. Unlike ``ps`` resident size it includes Metal/MPS
allocations: a 2 GiB MPS tensor showed 0.30 GiB resident and 2.19 GiB footprint, and a cap
derived from resident sizes failed a legitimate onsets job (10-24). A sample that cannot read a
live process's footprint is not recorded at all.

``cap_environment`` turns a peak measured on ``MIN_CAP_RUNS`` runs into a hard torch MPS
allocation cap (``PYTORCH_MPS_HIGH_WATERMARK_RATIO``, never 0, which would disable it) of at least
``CAP_FACTOR`` times the measured footprint and never below the reservation: an allocation
beyond it raises instead of swapping. MLX 0.32 offers only a soft limit (it still swaps), so MLX
jobs are bounded by their reservation only.
"""

from __future__ import annotations

import ctypes
import fcntl
import json
import os
import re
import subprocess
import threading
import time
from collections.abc import Callable, Generator
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path

GIB = 2**30
# Estimated peaks (GiB) until a run of the job has measured one. A cleanup chunk's peak physical
# footprint was 5.61 GB at 900 frames (2026-10-24; 7 GiB keeps a 1.25 margin), CPU separation
# ~5 GB per worker and Metal ASR ~4 GB (2026-10-16); YOLO inference, wav2vec2 onsets and the
# native compose are estimates.
DEFAULT_PEAKS_GIB = {
    "cleanup-chunk": 7.0,
    "separation": 10.0,
    "asr": 6.0,
    "yolo-track": 4.0,
    "onsets": 4.0,
    "compose": 6.0,
    # bun clips-mac.ts loads about 22 GB of local models (two swap thrashes on 10-24 ran it
    # outside any reservation); a measured footprint replaces this once recorded.
    "clips": 24.0,
}
# Placeholders until measured: the measured footprint (x MARGIN) then replaces them outright.
PLACEHOLDER_PEAKS = frozenset({"clips"})
MARGIN = 1.25
CAP_FACTOR = 2.0  # a hard cap keeps this much room above the largest measured footprint
# A hard cap needs this many complete measured runs: peaks grow with recording length.
MIN_CAP_RUNS = 2
MEASURED = "footprints.json"  # physical footprints only (ps resident sizes were undercounts)
RUSAGE_INFO_V4 = 4
ESRCH = 3
HEADROOM_GIB = 10.0  # OS, apps and the other agents' processes
MEMORY_FREE_MIN = 20  # kern.memorystatus_level, as in workflow_parallel
PRESSURE_NORMAL = 1
PRESSURE_CRITICAL = 4
SWAP_IN_LIMIT = 2 * 1024**3  # bytes per minute (thrash 8 GB/min, normal < 1 GB/min, 10-24)
SWAP_SAMPLE_SECONDS = 10.0
SWAP_SAMPLE_MAX_SECONDS = 600.0  # an older sample says nothing about now
VM_STAT = ("/usr/bin/vm_stat",)
PAGED_IN = re.compile(r"^(?:Pageins|Swapins):\s+(\d+)\.$", re.MULTILINE)
PAGE_SIZE = re.compile(r"page size of (\d+) bytes")
POLL_SECONDS = 15.0
SAMPLE_SECONDS = 10.0
MAX_SETTINGS_BYTES = 4096
SETTINGS_KEYS = frozenset({"budgetGiB", "headroomGiB", "peaksGiB", "hardCaps"})
# torch's MPS ceiling scales the device's recommended working set (0.78 of memory measured here).
MPS_RECOMMENDED_SHARE = 0.78
MARKER = re.compile(
    r"^(?P<job>[a-z-]+)\.(?P<owner>[A-Za-z0-9_-]+)\.[0-9]+\.[0-9]+\.(?P<bytes>[0-9]+)$"
)

Pressure = Callable[[], tuple[int, int]]  # (free level %, pressure level)


@dataclass(frozen=True, slots=True)
class Settings:
    budget: int
    peaks: dict[str, int]
    hard_caps: bool
    known: frozenset[str]  # jobs whose peak was measured or configured, not estimated
    caps: dict[str, int]  # hard-cap bytes of the known jobs
    total: int = 0  # physical memory; 0 skips the measured free-memory check


def root() -> Path:
    """The reservation directory of this checkout (src/replay/..)."""
    return Path(__file__).resolve().parents[2] / "edits" / "work" / "memory-reservations"


def physical() -> int:
    value = subprocess.run(
        ["/usr/sbin/sysctl", "-n", "hw.memsize"], capture_output=True, text=True, check=False
    ).stdout.strip()
    if not value.isdigit():
        raise ValueError("hw.memsize is unreadable")
    return int(value)


def _sysctl(name: str) -> int | None:
    value = subprocess.run(
        ["/usr/sbin/sysctl", "-n", name], capture_output=True, text=True, check=False
    ).stdout.strip()
    return int(value) if value.isdigit() else None


def pressure() -> tuple[int, int]:
    """(free-memory level, pressure level); unreadable counts as none free and critical.

    The pressure level is critical while the swap-in rate is too high: starting a job into that
    thrash only deepens it, so even a job alone waits.
    """
    level = _sysctl("kern.memorystatus_level")
    state = _sysctl("kern.memorystatus_vm_pressure_level")
    measured = PRESSURE_CRITICAL if state is None else state
    if swap_in_rate() >= SWAP_IN_LIMIT:
        measured = PRESSURE_CRITICAL
    return (0 if level is None else level, measured)


def paged_in_bytes(text: str) -> int | None:
    """Swapins + Pageins in bytes from ``vm_stat`` output; None when unreadable."""
    size, counts = PAGE_SIZE.search(text), PAGED_IN.findall(text)
    if size is None or len(counts) != 2:
        return None
    return (int(counts[0]) + int(counts[1])) * int(size.group(1))


def _vm_stat() -> str:
    return subprocess.run(VM_STAT, capture_output=True, text=True, check=False).stdout


def swap_in_rate(
    directory: Path | None = None,
    *,
    read: Callable[[], str] = _vm_stat,
    clock: Callable[[], float] = time.time,
) -> float:
    """Bytes paged in per minute, machine-wide, since a shared earlier sample (inf if unreadable).

    0 until a sample ``SWAP_SAMPLE_SECONDS`` old exists; a newer call returns the rate measured
    last, so callers polling every few seconds all see one stable rate.
    """
    place = root() if directory is None else directory
    total, now = paged_in_bytes(read()), clock()
    if total is None:
        return float("inf")
    place.mkdir(mode=0o700, parents=True, exist_ok=True)
    path = place / "vm-sample.json"
    descriptor = os.open(place / "vm-sample.lock", os.O_RDWR | os.O_CREAT, 0o600)
    try:
        fcntl.flock(descriptor, fcntl.LOCK_EX)
        try:
            previous = json.loads(path.read_text(encoding="utf-8"))
            at, paged, rate = float(previous["t"]), int(previous["bytes"]), float(previous["rate"])
        except (OSError, ValueError, KeyError, TypeError):
            at, paged, rate = now - SWAP_SAMPLE_MAX_SECONDS - 1, total, 0.0  # no usable sample
        elapsed = now - at
        if 0 <= elapsed < SWAP_SAMPLE_SECONDS:
            return rate
        fresh = SWAP_SAMPLE_SECONDS <= elapsed <= SWAP_SAMPLE_MAX_SECONDS
        rate = max(total - paged, 0) / elapsed * 60 if fresh else 0.0
        staged = path.with_name(f".vm-sample.{os.getpid()}")
        staged.write_text(json.dumps({"t": now, "bytes": total, "rate": rate}), encoding="utf-8")
        os.replace(staged, path)
        return rate
    finally:
        os.close(descriptor)


def _gib(value: object, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, int | float) or not 0 < value <= 1024:
        raise ValueError(f"Memory reservation {name} must be a positive GiB number")
    return float(value)


def settings(directory: Path, total: int) -> Settings:
    """The budget, the per-job peaks (bytes) and whether hard caps apply."""
    path = directory / "settings.json"
    raw: object = {}
    if path.exists():
        data = path.read_bytes()
        if len(data) > MAX_SETTINGS_BYTES:
            raise ValueError("Memory reservation settings are too large")
        raw = json.loads(data.decode("utf-8"))
    if not isinstance(raw, dict) or not set(raw) <= SETTINGS_KEYS:
        raise ValueError("Memory reservation settings take " + ", ".join(sorted(SETTINGS_KEYS)))
    headroom = _gib(raw.get("headroomGiB", HEADROOM_GIB), "headroomGiB")
    budget = (
        _gib(raw["budgetGiB"], "budgetGiB") * GIB if "budgetGiB" in raw else total - headroom * GIB
    )
    configured = raw.get("peaksGiB", {})
    if not isinstance(configured, dict):
        raise ValueError("peaksGiB must map jobs to GiB")
    measured = _measured(directory)
    peaks = {
        job: int(
            _gib(configured[job], job) * GIB
            if job in configured
            else measured[job][0] * MARGIN
            if job in PLACEHOLDER_PEAKS and job in measured
            else max(gib * GIB, measured.get(job, (0, 0))[0] * MARGIN)
        )
        for job, gib in DEFAULT_PEAKS_GIB.items()
    }
    hard = raw.get("hardCaps", True)
    if type(hard) is not bool:
        raise ValueError("hardCaps must be true or false")
    known = frozenset({*configured, *measured} & set(DEFAULT_PEAKS_GIB))
    caps = {
        job: int(max(peaks[job], CAP_FACTOR * measured[job][0]))
        for job in known
        if job in measured and measured[job][1] >= MIN_CAP_RUNS
    }
    return Settings(int(budget), peaks, hard, known, caps, total)


def _measured(directory: Path) -> dict[str, tuple[int, int]]:
    """Each job's (largest footprint, complete measured runs)."""
    path = directory / MEASURED
    if not path.exists():
        return {}
    raw: object = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise ValueError(f"{MEASURED} must map jobs to their peak and runs")
    found: dict[str, tuple[int, int]] = {}
    for job, value in raw.items():
        peak = value.get("peak") if isinstance(value, dict) else None
        runs = value.get("runs") if isinstance(value, dict) else None
        if type(peak) is int and type(runs) is int and peak > 0 and runs > 0:
            found[str(job)] = (peak, runs)
    return found


def record_peak(directory: Path, job: str, peak: int) -> None:
    """Count one complete measured run of a job and keep its largest peak (atomically)."""
    if peak <= 0:
        return
    with _ledger(directory):
        measured = _measured(directory)
        largest, runs = measured.get(job, (0, 0))
        measured[job] = (max(largest, peak), runs + 1)
        rows = {name: {"peak": value, "runs": count} for name, (value, count) in measured.items()}
        temporary = directory / f".footprints.{os.getpid()}.tmp"
        temporary.write_text(json.dumps(rows, sort_keys=True) + "\n", encoding="utf-8")
        temporary.replace(directory / MEASURED)


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
    """True while a process holds this reservation; a stale marker is removed."""
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


def reserved(directory: Path) -> int:
    """Bytes reserved by live jobs on this machine."""
    held = directory / "held"
    if not held.is_dir():
        return 0
    total = 0
    for marker in held.iterdir():
        found = MARKER.match(marker.name)
        if found and _live(marker):
            total += int(found["bytes"])
    return total


class Reservation:
    """One job's reservation; ``try_reserve`` holds it, ``release`` (or exit) returns it."""

    def __init__(self, directory: Path, job: str, owner: str, size: int) -> None:
        if job not in DEFAULT_PEAKS_GIB or not re.fullmatch(r"[A-Za-z0-9_-]+", owner):
            raise ValueError("Unknown reserved job or owner")
        self.directory, self.job, self.size = directory, job, size
        name = f"{job}.{owner}.{os.getpid()}.{id(self)}.{size}"
        self.path = directory / "held" / name
        self.descriptor: int | None = None

    def try_reserve(self, current: Settings, measure: Pressure | None = None) -> bool:
        with _ledger(self.directory):
            held = reserved(self.directory)
            level, state = (pressure if measure is None else measure)()
            fits = held + self.size <= current.budget and level >= MEMORY_FREE_MIN
            # What other processes use shows only in the measured free memory.
            free = level * current.total // 100
            covered = current.total == 0 or free >= min(self.size, current.budget)
            if (
                state >= PRESSURE_CRITICAL
                or not covered
                or (held and (not fits or state > PRESSURE_NORMAL))
            ):
                return False
            self.path.parent.mkdir(mode=0o700, exist_ok=True)
            self.descriptor = os.open(self.path, os.O_RDWR | os.O_CREAT, 0o600)
            fcntl.flock(self.descriptor, fcntl.LOCK_EX)
            return True

    def release(self) -> None:
        if self.descriptor is not None:
            self.path.unlink(missing_ok=True)
            os.close(self.descriptor)
            self.descriptor = None


@contextmanager
def reserve(
    job: str,
    owner: str,
    *,
    directory: Path | None = None,
    measure: Pressure | None = None,
    poll_seconds: float = POLL_SECONDS,
    sleep: Callable[[float], None] = time.sleep,
) -> Generator[Settings]:
    """Wait for and hold a reservation of the job's peak for the block."""
    place = root() if directory is None else directory
    current = settings(place, physical())
    reservation = Reservation(place, job, owner, current.peaks[job])
    while not reservation.try_reserve(current, measure):
        sleep(poll_seconds)
        current = settings(place, physical())
    try:
        yield current
    finally:
        reservation.release()


def cap_environment(job: str, current: Settings, total: int) -> dict[str, str]:
    """Environment capping a torch MPS process at ``CAP_FACTOR`` times its measured footprint.

    Only a job with ``MIN_CAP_RUNS`` complete measured runs is capped: a cap from an estimate or
    one short run could fail a job that simply needs more, and a failed run would never measure
    a larger peak.
    """
    if not current.hard_caps or job not in current.caps:
        return {}
    ratio = max(current.caps[job] / (MPS_RECOMMENDED_SHARE * total), 0.01)  # 0 lifts the cap
    value = f"{ratio:.4f}"
    return {"PYTORCH_MPS_HIGH_WATERMARK_RATIO": value, "PYTORCH_MPS_LOW_WATERMARK_RATIO": value}


class _RusageV4(ctypes.Structure):
    """``struct rusage_info_v4``: a UUID, then 64-bit counters (only two are read)."""

    _fields_ = [("uuid", ctypes.c_uint8 * 16), *((f"c{i}", ctypes.c_uint64) for i in range(40))]


FOOTPRINT, LIFETIME_MAX_FOOTPRINT = "c7", "c28"  # ri_phys_footprint, ri_lifetime_max_...


def footprint(pid: int) -> int | None:
    """A live process's lifetime maximum physical footprint (GPU memory included), 0 when it
    has exited, None when it cannot be read."""
    library = ctypes.CDLL("/usr/lib/libproc.dylib", use_errno=True)
    info = _RusageV4()
    if library.proc_pid_rusage(pid, RUSAGE_INFO_V4, ctypes.byref(info)) != 0:
        return 0 if ctypes.get_errno() == ESRCH else None
    return max(int(getattr(info, FOOTPRINT)), int(getattr(info, LIFETIME_MAX_FOOTPRINT)))


def group_footprint(
    match: Callable[[str], bool], read: Callable[[int], int | None] = footprint
) -> int | None:
    """Summed footprints of the process groups of the processes ``match`` finds; None when any
    live member's footprint is unreadable (such a sample is never recorded)."""
    listing = subprocess.run(
        ["/bin/ps", "-axo", "pid=,pgid=,args="], capture_output=True, text=True, check=False
    ).stdout
    rows = [line.split(None, 2) for line in listing.splitlines()]
    rows = [row for row in rows if len(row) == 3 and row[0].isdigit()]
    groups = {row[1] for row in rows if match(row[2])}
    total = 0
    for row in rows:
        if row[1] not in groups:
            continue
        value = read(int(row[0]))
        if value is None:
            return None
        total += value
    return total


@contextmanager
def measuring(
    directory: Path,
    job: str,
    match: Callable[[str], bool],
    *,
    interval: float = SAMPLE_SECONDS,
    sample: Callable[[Callable[[str], bool]], int | None] = group_footprint,
) -> Generator[None]:
    """Sample the job's process-group footprint; record its peak when the block ends, unless a
    sample was incomplete."""
    stop = threading.Event()
    peak = [0]
    complete = [True]

    def loop() -> None:
        while not stop.wait(interval):
            value = sample(match)
            if value is None:
                complete[0] = False
            else:
                peak[0] = max(peak[0], value)

    thread = threading.Thread(target=loop, name=f"peak-{job}", daemon=True)
    thread.start()
    try:
        yield
    finally:
        stop.set()
        thread.join()
        if complete[0]:
            record_peak(directory, job, peak[0])
