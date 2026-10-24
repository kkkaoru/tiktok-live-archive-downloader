"""Recording-first priority: one ranked queue of recordings drives every machine-wide lease.

The rank lives beside the queue file in ``<queue>.order.json`` (the queue file itself keeps the
format the running queue runner validates): ``mode`` ``auto`` ranks the queued recordings by
broadcast time, newest first (the validated ``start_time`` of the dated original); ``pinned``
keeps a manual ``order`` and places recordings added later after it by date. ``urgent``
recordings rank above everything, the most recent request first, and leave the list once they are
done; ``paused`` ones are not launched and rank after every other recording.

``publish`` writes the rank as the ``priority`` of every lease directory (the GPU, Compressor and
compose leases; a ``manual`` settings file is left alone) and the urgent list to
``lease-urgent.json``. ``gpu_lease`` reads both at every attempt, so a change takes effect at the
next lease decision with no restart; while an urgent recording waits for any lease, no other
recording is granted a new one (running jobs are never stopped).

Edits are made under the queue's lock (the one ``recording_queue add`` takes) and written
atomically, so they are safe beside the live queue runner.
"""

from __future__ import annotations

import fcntl
import json
import os
import statistics
import time
from collections.abc import Callable, Generator, Sequence
from contextlib import contextmanager
from dataclasses import dataclass, replace
from pathlib import Path

from replay import gpu_lease
from replay.recordings import record

MODES = ("auto", "pinned")
ORDER_KEYS = frozenset({"version", "mode", "order", "urgent", "paused"})
LEASE_DIRECTORIES = ("gpu-leases", "compressor-leases", "compose-leases")
URGENT = gpu_lease.URGENT
DRAIN_REASONS = ("drain", "relaunch", "abandon")


@dataclass(frozen=True, slots=True)
class Order:
    mode: str = "auto"
    order: tuple[str, ...] = ()
    urgent: tuple[str, ...] = ()
    paused: tuple[str, ...] = ()


def order_path(queue: Path) -> Path:
    return queue.with_name(queue.stem + ".order.json")


def _ids(value: object, name: str) -> tuple[str, ...]:
    if not isinstance(value, list) or not all(isinstance(item, str) for item in value):
        raise ValueError(f"Queue order {name} must list recording IDs")
    return tuple(value)


def load(queue: Path) -> Order:
    path = order_path(queue)
    if not path.exists():
        return Order()
    raw = record(json.loads(path.read_text(encoding="utf-8")))
    if not set(raw) <= ORDER_KEYS or raw.get("version") != 1 or raw.get("mode") not in MODES:
        raise ValueError(f"Unsupported queue order {path.name}")
    return Order(
        str(raw["mode"]),
        _ids(raw.get("order", []), "order"),
        _ids(raw.get("urgent", []), "urgent"),
        _ids(raw.get("paused", []), "paused"),
    )


def _write(path: Path, value: object) -> None:
    staged = path.with_name(f".{path.name}.{os.getpid()}.partial")
    staged.write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8")
    staged.replace(path)


@contextmanager
def locked(queue: Path) -> Generator[None]:
    """The queue's lock: the queue runner's ``add`` and every order edit take it."""
    with queue.with_name(f"{queue.name}.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        yield


def save(queue: Path, order: Order) -> None:
    _write(
        order_path(queue),
        {"version": 1, "mode": order.mode, "order": list(order.order),
         "urgent": list(order.urgent), "paused": list(order.paused)},
    )  # fmt: skip


def enqueue(queue: Path, replay_id: str) -> None:
    """Add a recording to the queue file (caller holds the lock)."""
    data = record(json.loads(queue.read_text(encoding="utf-8")))
    recordings = data.get("recordings")
    if not isinstance(recordings, list):
        raise ValueError("The queue lists no recordings")
    if replay_id not in recordings:
        _write(queue, {**data, "recordings": [*recordings, replay_id]})


def newest(broadcasts: dict[str, int]) -> Callable[[str], tuple[int, str]]:
    return lambda replay_id: (-broadcasts.get(replay_id, 0), replay_id)


def ranked(order: Order, broadcasts: dict[str, int]) -> list[str]:
    """Every queued recording (``broadcasts``' keys) in rank order."""
    queued = set(broadcasts)
    urgent = [item for item in order.urgent if item in queued]
    rest = sorted(queued - set(urgent), key=newest(broadcasts))
    if order.mode == "pinned":
        pinned = [item for item in order.order if item in rest]
        rest = pinned + [item for item in rest if item not in pinned]
    active = [item for item in rest if item not in order.paused]
    return [*urgent, *active, *[item for item in rest if item in order.paused]]


def moved(current: Sequence[str], replay_id: str, *, to: int) -> list[str]:
    """``current`` with ``replay_id`` at 1-based position ``to`` (clamped)."""
    rest = [item for item in current if item != replay_id]
    at = min(max(to, 1), len(rest) + 1) - 1
    return [*rest[:at], replay_id, *rest[at:]]


def publish(order: Order, ranking: Sequence[str], work: Path) -> None:
    """Write the rank to every lease directory's priority and the urgent list."""
    priority = [item for item in ranking if item not in order.paused]
    for name in LEASE_DIRECTORIES:
        directory = work / name
        path = directory / "settings.json"
        if not path.exists():
            continue
        raw = record(json.loads(path.read_text(encoding="utf-8")))
        if raw.get("manual") is True:
            continue
        capacity, _ = gpu_lease.settings(directory)
        kept: list[str] = []
        for key in priority:
            if len(json.dumps({"capacity": capacity, "priority": [*kept, key]})) > 4000:
                break
            kept.append(key)
        _write(path, {"capacity": capacity, "priority": kept})
    work.mkdir(mode=0o700, parents=True, exist_ok=True)
    _write(work / URGENT, {"urgent": [item for item in order.urgent if item in ranking]})


def holders(directory: Path, now: float | None = None) -> list[dict[str, object]]:
    """Jobs holding a lease of ``directory`` now, with the step's typical hold time left."""
    path = directory / gpu_lease.HOLDERS
    if not path.exists():
        return []
    held: dict[tuple[object, object, object], dict[str, object]] = {}
    durations: dict[str, list[float]] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        row = record(json.loads(line))
        key = (row.get("pid"), row.get("recording"), row.get("step"))
        if row.get("event") == "acquire":
            held[key] = row
        elif row.get("event") == "release":
            held.pop(key, None)
            seconds = row.get("heldSeconds")
            if isinstance(seconds, int | float):
                durations.setdefault(str(row.get("step")), []).append(float(seconds))
    moment = time.time() if now is None else now
    found = []
    for (pid, recording, step), row in held.items():
        if not isinstance(pid, int) or not _alive(pid):
            continue
        elapsed = moment - float(str(row.get("t", moment)))
        typical = durations.get(str(step))
        left = None if not typical else max(0.0, statistics.median(typical) - elapsed)
        found.append(
            {
                "recording": recording,
                "step": step,
                "heldSeconds": round(elapsed),
                "expectedRemainingSeconds": None if left is None else round(left),
            }
        )
    return found


def _alive(pid: int) -> bool:
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return True


def waiting_on(work: Path, replay_id: str) -> list[str]:
    """The lease directories where ``replay_id`` has a live waiter."""
    found = []
    for name in LEASE_DIRECTORIES:
        folder = work / name / "waiting"
        if folder.is_dir() and any(
            marker.name.partition(".")[0] == replay_id and gpu_lease.marker_held(marker)
            for marker in folder.iterdir()
        ):
            found.append(name)
    return found


def drain_path(work: Path, replay_id: str) -> Path:
    return work / "drain" / f"{replay_id}.json"


def drain(work: Path, replay_id: str, reason: str) -> None:
    """Ask the recording's runners to stop at their next chunk or step boundary."""
    if reason not in DRAIN_REASONS:
        raise ValueError(f"A drain reason is one of {', '.join(DRAIN_REASONS)}")
    path = drain_path(work, replay_id)
    path.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
    _write(path, {"reason": reason, "t": round(time.time(), 1)})


def undrain(work: Path, replay_id: str) -> None:
    drain_path(work, replay_id).unlink(missing_ok=True)


def drained(work: Path, replay_id: str) -> str | None:
    """The drain reason while the recording is being drained, else None."""
    path = drain_path(work, replay_id)
    if not path.is_file():
        return None
    reason = record(json.loads(path.read_text(encoding="utf-8"))).get("reason")
    return str(reason) if reason in DRAIN_REASONS else "drain"


def without(order: Order, finished: set[str]) -> Order:
    """The order with finished recordings dropped from the urgent list."""
    return replace(order, urgent=tuple(item for item in order.urgent if item not in finished))
