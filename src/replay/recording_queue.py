"""Keep a few recording pipelines busy, newest broadcast first, from one persistent queue file.

``run --queue QUEUE.json --defaults DEFAULTS.json`` polls the queue (read at every tick, so an
operator or the replay watcher edits it live) and launches ``recording_pipeline run`` detached
(``recording_pipeline.dispatch``, one tmux server per recording) while fewer than ``slots``
queued recordings run. Stopping or restarting the runner never stops a pipeline. ``add QUEUE ID``
appends a recording (the watcher's hook).

The queue holds no state of its own; each tick derives it from the recording's files:

- ``done``: every pipeline step has its receipt (the ``summary`` step passes the completion gate);
- ``running``: the pipeline lock is held (lane-launched runs count too), or the latest launch has
  no status while its tmux window lives (``run`` prepares, possibly for minutes, before locking);
  a launch a reboot cut off has neither and is launched again;
- ``held``: ``MAX_FAILURES`` launches failed since the shared code last changed; editing the code
  (a fix) makes it ready again, and a held recording never blocks the ones behind it;
- ``unbound``: no edit configuration yet (a download that has not completed);
- ``error``: its state could not be read (reported in the row; the others continue);
- ``ready``: launched in broadcast order, newest first, so a failed recording is retried in its
  place instead of falling behind.

Queue file: ``{"version": 1, "slots": 1..3 | "auto", "paused": bool, "recordings": [ID, ...]}``.
Launch order and every machine-wide lease follow one recording rank (``recording_priority``):
urgent recordings, then newest broadcast first or a pinned manual order; paused ones last and
never launched. ``list``, ``move``, ``pause``, ``resume``, ``pin-order``, ``auto-order`` and
``urgent`` change it mid-run; ``drain`` asks a recording's runners to stop at their next chunk or
step boundary (``undrain`` clears it), and a drained recording is not launched;
``run --order-leases`` (and every such command) publishes it to the
lease directories, which read it at their next decision.
``"auto"`` admits every ready recording, at most one per tick, while memory admits more work
(``memory_admits``): the GPU leases and the per-job memory reservations of ``memory_admission``
then meter the running pipelines instead of a fixed number of them.
``paused`` stops new launches (running pipelines continue), e.g. for a quality gate.
"""

import argparse
import fcntl
import json
import os
import subprocess
import sys
import time
from collections.abc import Callable, Sequence
from datetime import datetime
from pathlib import Path

from replay import gpu_lease, memory_admission
from replay import recording_pipeline as pipelines
from replay import recording_priority as priority
from replay.core import ReplayError
from replay.recording_steps import recording_name
from replay.recordings import JST, record
from replay.stages import StageError, execution_lock

MAX_SLOTS = 3  # 3 pipelines plus a make check froze the 48 GB Mac (2026-10-22)
MAX_FAILURES = 2
AUTO_SLOTS = 0  # load_queue's value for "auto"
PRIORITY_COMMANDS = ("list", "move", "pause", "resume", "pin-order", "auto-order", "urgent")
DEFAULT_QUEUE = Path("edits/work/recording-pipeline/queue.json")
DEFAULT_DEFAULTS = Path("edits/work/recording-pipeline/DEFAULTS-v1.json")
# Validated broadcast times: a YouTube edit config hashes its whole original on every load
# (~90 s each), which every tick and priority command (under the queue lock) used to repeat.
BROADCASTS = Path("edits/work/recording-pipeline/broadcasts.json")
POLL_SECONDS = 60.0
QUEUE_KEYS = frozenset({"version", "slots", "paused", "recordings"})
DISK_FLOOR_EXIT = 4  # scripts/tmux-job.sh refuses a launch below the free-space floor
STEP_NAMES = (*pipelines.ORDER, "retire-intermediates")
# A failure of one recording or tick is reported and retried; it never stops the runner.
FAILURES = (ReplayError, StageError, OSError, ValueError)

Launch = Callable[[str], str]


def load_queue(path: Path) -> tuple[int, bool, tuple[str, ...]]:
    """Slots, paused and the queued recording IDs (validated recording names)."""
    data = record(json.loads(path.read_text(encoding="utf-8")))
    slots, paused, recordings = data.get("slots"), data.get("paused", False), data.get("recordings")
    if not set(data) <= QUEUE_KEYS or data.get("version") != 1:
        raise ReplayError("Unsupported recording queue")
    if slots == "auto":
        slots = AUTO_SLOTS
    elif type(slots) is not int or not 1 <= slots <= MAX_SLOTS:
        raise ReplayError(f"Queue slots must be an integer from 1 to {MAX_SLOTS} or auto")
    if not isinstance(paused, bool) or not isinstance(recordings, list):
        raise ReplayError("Queue paused must be a boolean and recordings a list")
    for item in recordings:
        recording_name(str(item))  # raises for anything but a recording ID
    return slots, paused, tuple(str(item) for item in recordings)


def add(path: Path, replay_id: str) -> bool:
    """Append a recording to the queue (atomically, under a lock); False when already queued."""
    recording_name(replay_id)
    with path.with_name(f"{path.name}.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        recordings = load_queue(path)[2]
        if replay_id in recordings:
            return False
        data = record(json.loads(path.read_text(encoding="utf-8")))
        staged = path.with_name(f".{path.name}.{os.getpid()}")
        staged.write_text(
            json.dumps({**data, "recordings": [*recordings, replay_id]}, indent=2) + "\n",
            encoding="utf-8",
        )
        os.replace(staged, path)
    return True


def code_changed(checkout: Path) -> float:
    """When the shared code last changed (newest ``src/replay`` file)."""
    return max(path.stat().st_mtime for path in (checkout / "src/replay").rglob("*.py"))


def launches(layout: pipelines.Layout) -> list[str]:
    """The recording's dispatched launch names, oldest first."""
    found: list[str] = []
    for number in range(1, pipelines.MAX_LAUNCHES + 1):
        name = f"pipeline-v{number}"
        if not (layout.directory / f"{name}.launch").exists():
            break
        found.append(name)
    return found


UNHOLD = "unhold"  # beside a recording's launches: failures before its mtime are resolved


def unhold(layout: pipelines.Layout) -> None:
    """Mark the recording's earlier failed launches resolved (their statuses stay as evidence)."""
    (layout.directory / UNHOLD).touch(mode=0o600)


def resolved(layout: pipelines.Layout) -> float:
    """When its failures were last marked resolved (0 when never)."""
    marker = layout.directory / UNHOLD
    return marker.stat().st_mtime if marker.exists() else 0.0


def failures(layout: pipelines.Layout, since: float) -> int:
    """Launches that exited nonzero after the code last changed."""
    statuses = (layout.directory / f"{name}.status" for name in launches(layout))
    return sum(
        1
        for status in statuses
        if status.is_file()
        and status.stat().st_mtime > since
        and status.read_text(encoding="utf-8").strip()
        not in ("0", str(pipelines.DRAINED_EXIT))  # a drained run did not fail
    )


def window_alive(directory: Path, name: str) -> bool:
    """The launch's tmux window exists (the socket is relative: macOS limits its path length)."""
    windows = subprocess.run(
        ["tmux", "-f", "/dev/null", "-S", "tmux.sock", "list-windows", "-a", "-F",
         "#{window_name}"],
        cwd=directory, capture_output=True, text=True, check=False,
    ).stdout  # fmt: skip
    return name in windows.split()


def running(layout: pipelines.Layout) -> bool:
    if not layout.directory.is_dir():
        return False
    names = launches(layout)
    pending = bool(names) and not (layout.directory / f"{names[-1]}.status").exists()
    if pending and window_alive(layout.directory, names[-1]):
        return True
    with execution_lock(layout.directory, "pipeline") as free:
        return not free


def done(layout: pipelines.Layout) -> bool:
    return all((layout.steps / f"{name}.json").is_file() for name in STEP_NAMES)


def state(
    replay_id: str, defaults: dict[str, object], *, checkout: Path, since: float
) -> tuple[str, int]:
    """The recording's state and its broadcast time (ns; 0 while unbound)."""
    edit_root = checkout / "edits" / recording_name(replay_id)
    try:
        pipelines.bind(replay_id, str(defaults["python"]), checkout)
        broadcast = cached_broadcast(edit_root, checkout)
    except (ReplayError, OSError, subprocess.CalledProcessError):
        return "unbound", 0
    layout = pipelines.layout_of(replay_id, defaults, checkout)
    if running(layout):
        return "running", broadcast
    if done(layout):
        return "done", broadcast
    if failures(layout, max(since, resolved(layout))) >= MAX_FAILURES:
        return "held", broadcast
    return "ready", broadcast


def cached_broadcast(edit_root: Path, checkout: Path) -> int:
    """``pipelines.broadcast``, validated again only when the edit config or its source changed.

    The cache key is the size and mtime of ``edit-config.json`` and of the source it names.
    """
    config = edit_root / "edit-config.json"
    if not config.is_file():
        return pipelines.broadcast(edit_root)
    source = Path(str(record(json.loads(config.read_text(encoding="utf-8"))).get("source")))
    key = [[stat.st_size, stat.st_mtime_ns] for stat in (config.stat(), source.stat())]
    path = checkout / BROADCASTS
    try:
        cache = record(json.loads(path.read_text(encoding="utf-8")))
    except (OSError, ValueError, ReplayError):
        cache = {}
    entry = cache.get(edit_root.name)
    if isinstance(entry, dict) and entry.get("key") == key and type(entry.get("ns")) is int:
        return int(entry["ns"])
    value = pipelines.broadcast(edit_root)
    path.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
    staged = path.with_name(f".{path.name}.{os.getpid()}")
    staged.write_text(
        json.dumps({**cache, edit_root.name: {"key": key, "ns": value}}, indent=1) + "\n",
        encoding="utf-8",
    )
    os.replace(staged, path)
    return value


def swap_thrashing() -> bool:
    return memory_admission.swap_in_rate() >= memory_admission.SWAP_IN_LIMIT


def layouts(
    rows: Sequence[dict[str, object]], defaults: dict[str, object], checkout: Path
) -> list[pipelines.Layout]:
    """The queued recordings' pipeline layouts (a row whose layout is invalid is left out)."""
    found = []
    for row in rows:
        try:
            found.append(pipelines.layout_of(str(row["recording"]), defaults, checkout))
        except FAILURES:
            continue
    return found


def volume_retries(layouts: Sequence[pipelines.Layout], day: str) -> int:
    """Steps rerun after a volume detach (``step_retry`` class ``volume``) on ``day`` (local)."""
    count = 0
    for layout in layouts:
        for record_path in (layout.steps / "logs").glob("*.failure.json"):
            stamp = record_path.name.removesuffix(".failure.json").rsplit("-", 1)[-1]
            if not stamp.isdigit():
                continue
            if datetime.fromtimestamp(int(stamp) / 1e9).strftime("%Y-%m-%d") != day:
                continue
            try:
                row = record(json.loads(record_path.read_text(encoding="utf-8")))
            except (OSError, ValueError, ReplayError):
                continue
            count += row.get("class") == "volume" and row.get("retried") is True
    return count


def newest_broadcast(row: dict[str, object]) -> int:
    return -int(str(row["broadcast"]))


def memory_admits() -> bool:
    """Room for one more recording: normal pressure, free memory, and reservations to spare."""
    directory = memory_admission.root()
    current = memory_admission.settings(directory, memory_admission.physical())
    level, state = memory_admission.pressure()
    smallest = min(current.peaks.values())
    return (
        state == memory_admission.PRESSURE_NORMAL
        and level >= memory_admission.MEMORY_FREE_MIN
        and memory_admission.reserved(directory) + smallest <= current.budget
    )


def tick(
    queue: Path,
    defaults: dict[str, object],
    *,
    checkout: Path,
    launch: Launch,
    admits: Callable[[], bool] = memory_admits,
    thrashing: Callable[[], bool] | None = None,
) -> list[dict[str, object]]:
    """Launch ready recordings into free slots; return every queued recording's row.

    While the machine pages in too fast (``swap_thrashing``) running pipelines continue but at
    most one runs: nothing is launched beside a running one until the rate recovers.
    """
    slots, paused, recordings = load_queue(queue)
    since = code_changed(checkout)
    rows: list[dict[str, object]] = []
    for replay_id in recordings:
        try:
            found, broadcast = state(replay_id, defaults, checkout=checkout, since=since)
        except FAILURES as error:
            rows.append({"recording": replay_id, "state": "error", "broadcast": 0,
                         "error": f"{type(error).__name__}: {error}"})  # fmt: skip
            continue
        rows.append({"recording": replay_id, "state": found, "broadcast": broadcast})
    order = priority.load(queue)
    ranking = priority.ranked(
        order, {str(row["recording"]): int(str(row["broadcast"])) for row in rows}
    )
    rows.sort(key=lambda row: ranking.index(str(row["recording"])))
    running = sum(1 for row in rows if row["state"] == "running")
    free = slots - running
    if slots == AUTO_SLOTS:
        free = 1 if any(row["state"] == "ready" for row in rows) and admits() else 0
    if free > 0 and (swap_thrashing if thrashing is None else thrashing)():
        free = min(free, 1 - running)
    for row in rows:
        if paused or free <= 0 or row["state"] != "ready" or row["recording"] in order.paused:
            continue
        if priority.drained(gpu_lease.root().parent, str(row["recording"])) is not None:
            row["state"] = "drained"
            continue
        try:
            row["launch"] = launch(str(row["recording"]))
        except (subprocess.CalledProcessError, *FAILURES) as error:
            row["state"] = "launch-refused"
            if getattr(error, "returncode", None) == DISK_FLOOR_EXIT:
                break  # below the free-space floor: launch nothing more this tick
            continue
        row["state"], free = "launched", free - 1
    return rows


def launcher(defaults_path: Path, defaults: dict[str, object], checkout: Path) -> Launch:
    def launch(replay_id: str) -> str:
        layout = pipelines.layout_of(replay_id, defaults, checkout)
        layout.directory.mkdir(mode=0o700, parents=True, exist_ok=True)
        python = str(defaults["python"])
        return pipelines.dispatch(
            layout, replay_id, defaults_path.resolve(), python=python, checkout=checkout
        )

    return launch


def order_leases(queue: Path, rows: Sequence[dict[str, object]]) -> None:
    """Publish the recording rank to the leases (dropping finished urgent recordings).

    A failure is reported and retried at the next tick.
    """
    try:
        broadcasts = {str(row["recording"]): int(str(row["broadcast"])) for row in rows}
        finished = {str(row["recording"]) for row in rows if row["state"] == "done"}
        with priority.locked(queue):
            order = priority.without(priority.load(queue), finished)
            priority.save(queue, order)
            priority.publish(order, priority.ranked(order, broadcasts), gpu_lease.root().parent)
    except FAILURES as error:
        print(f"Lease order not refreshed ({type(error).__name__}): {error}", file=sys.stderr)


def queue_rows(queue: Path, defaults: dict[str, object], checkout: Path) -> list[dict[str, object]]:
    """Each queued recording's state and broadcast time (no launch)."""
    since = code_changed(checkout)
    rows: list[dict[str, object]] = []
    for replay_id in load_queue(queue)[2]:
        try:
            found, broadcast = state(replay_id, defaults, checkout=checkout, since=since)
        except FAILURES:
            found, broadcast = "error", 0
        rows.append({"recording": replay_id, "state": found, "broadcast": broadcast})
    return rows


def current_step(steps: Path) -> str | None:
    """The newest started step without a receipt."""
    logs = sorted((steps / "logs").glob("*.run.log"), key=lambda path: path.stat().st_mtime)
    for log in reversed(logs):
        name = log.name.removesuffix(".run.log").rpartition("-")[0]
        if not (steps / f"{name}.json").exists():
            return name
    return None


def change_order(
    args: argparse.Namespace, ranking: list[str], order: priority.Order
) -> priority.Order:
    """The order after one ``move``/``pause``/``resume``/``pin-order``/``auto-order``/``urgent``."""
    regular = [item for item in ranking if item not in order.urgent]
    if args.command == "auto-order":
        return priority.Order("auto", (), order.urgent, order.paused)
    if args.command == "pin-order":
        return priority.Order("pinned", tuple(regular), order.urgent, order.paused)
    if args.command == "pause":
        return priority.Order(
            order.mode, order.order, order.urgent, (*order.paused, args.replay_id)
        )
    if args.command == "resume":
        kept = tuple(item for item in order.paused if item != args.replay_id)
        return priority.Order(order.mode, order.order, order.urgent, kept)
    if args.command == "urgent":
        rest = tuple(item for item in order.urgent if item != args.replay_id)
        return priority.Order(order.mode, order.order, (args.replay_id, *rest), order.paused)
    if args.replay_id not in regular:
        raise ReplayError(f"{args.replay_id} is not queued (or is urgent)")
    others = [item for item in regular if item != args.replay_id]
    to = 1 if args.top else args.to
    if args.before is not None or args.after is not None:
        anchor = args.before if args.before is not None else args.after
        if anchor not in others:
            raise ReplayError(f"{anchor} is not queued")
        to = others.index(anchor) + (1 if args.before is not None else 2)
    if to is None:
        raise ReplayError("move needs --to, --top, --before or --after")
    placed = priority.moved(regular, args.replay_id, to=to)
    return priority.Order("pinned", tuple(placed), order.urgent, order.paused)


def prioritize(args: argparse.Namespace, checkout: Path) -> list[dict[str, object]]:
    """Apply one priority command under the queue lock, publish the rank and list the queue."""
    defaults = pipelines.load_defaults(args.defaults)
    work = gpu_lease.root().parent
    if args.command == "urgent":
        recording_name(args.replay_id)
        with priority.locked(args.queue):
            priority.enqueue(args.queue, args.replay_id)
    rows = queue_rows(args.queue, defaults, checkout)  # outside the lock: it reads every recording
    with priority.locked(args.queue):
        broadcasts = {str(row["recording"]): int(str(row["broadcast"])) for row in rows}
        finished = {str(row["recording"]) for row in rows if row["state"] == "done"}
        order = priority.without(priority.load(args.queue), finished)
        if args.command != "list":
            order = change_order(args, priority.ranked(order, broadcasts), order)
        ranking = priority.ranked(order, broadcasts)
        priority.save(args.queue, order)
        if args.command != "list":
            priority.publish(order, ranking, work)
    listed = []
    by_id = {str(row["recording"]): row for row in rows}
    for rank, replay_id in enumerate(ranking, start=1):
        row = by_id[replay_id]
        layout = pipelines.layout_of(replay_id, defaults, checkout)
        broadcast = int(str(row["broadcast"]))
        dated = datetime.fromtimestamp(broadcast / 1e9, JST).isoformat() if broadcast else None
        waiting = priority.waiting_on(work, replay_id)
        entry: dict[str, object] = {
            "rank": rank, "recording": replay_id, "state": row["state"], "broadcastJST": dated,
            "urgent": replay_id in order.urgent, "paused": replay_id in order.paused,
            "step": current_step(layout.steps) if layout.steps.is_dir() else None,
            "waitingFor": waiting,
            "drained": priority.drained(work, replay_id),
        }  # fmt: skip
        if replay_id in order.urgent:
            # Who holds what the urgent recording waits for, and their typical time left.
            entry["blockedBy"] = [
                holder for name in waiting for holder in priority.holders(work / name)
            ]
        listed.append(entry)
    return listed


def main(argv: Sequence[str] | None = None, *, sleep: Callable[[float], None] = time.sleep) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    runner = commands.add_parser("run")
    runner.add_argument("--queue", type=Path, required=True)
    runner.add_argument("--defaults", type=Path, required=True)
    runner.add_argument("--poll", type=float, default=POLL_SECONDS)
    runner.add_argument("--once", action="store_true")
    runner.add_argument(
        "--order-leases",
        action="store_true",
        help="each tick, publish the recording rank to the machine-wide leases",
    )
    for name in ("drain", "undrain"):
        draining = commands.add_parser(name)
        draining.add_argument("replay_id")
        if name == "drain":
            draining.add_argument("--reason", choices=priority.DRAIN_REASONS, default="drain")
    holding = commands.add_parser("unhold")
    holding.add_argument("replay_id")
    holding.add_argument("--defaults", type=Path, default=DEFAULT_DEFAULTS)
    adding = commands.add_parser("add")
    adding.add_argument("queue", type=Path)
    adding.add_argument("replay_id")
    for name in PRIORITY_COMMANDS:
        command = commands.add_parser(name)
        command.add_argument("--queue", type=Path, default=DEFAULT_QUEUE)
        command.add_argument("--defaults", type=Path, default=DEFAULT_DEFAULTS)
        if name in ("move", "pause", "resume", "urgent"):
            command.add_argument("replay_id")
        if name == "move":
            command.add_argument("--to", type=int)
            command.add_argument("--top", action="store_true")
            command.add_argument("--before")
            command.add_argument("--after")
    args = parser.parse_args(argv)
    checkout = Path.cwd().resolve()
    try:
        if args.command == "add":
            print("added" if add(args.queue, args.replay_id) else "already queued")
            return 0
        if args.command == "unhold":
            unhold(
                pipelines.layout_of(
                    args.replay_id, pipelines.load_defaults(args.defaults), checkout
                )
            )
            print(f"unheld {args.replay_id}")
            return 0
        if args.command in ("drain", "undrain"):
            recording_name(args.replay_id)
            work = gpu_lease.root().parent
            if args.command == "drain":
                priority.drain(work, args.replay_id, args.reason)
            else:
                priority.undrain(work, args.replay_id)
            print(f"{args.command}ed {args.replay_id}")
            return 0
        if args.command in PRIORITY_COMMANDS:
            print(json.dumps(prioritize(args, checkout), indent=1, ensure_ascii=False))
            return 0
        defaults = pipelines.load_defaults(args.defaults)
    except FAILURES as error:
        print(f"Queue stopped ({type(error).__name__}): {error}", file=sys.stderr)
        return 1
    launch = launcher(args.defaults, defaults, checkout)
    while True:
        try:
            rows = tick(args.queue, defaults, checkout=checkout, launch=launch)
        except FAILURES as error:  # e.g. a queue file caught mid-edit: retried next tick
            print(f"Queue tick failed ({type(error).__name__}): {error}", file=sys.stderr)
            if args.once:
                return 1
        else:
            if args.order_leases:
                order_leases(args.queue, rows)
            today = datetime.now().strftime("%Y-%m-%d")
            print(
                json.dumps({"at": round(time.time()), "recordings": rows,
                            "volumeRetriesToday": volume_retries(
                                layouts(rows, defaults, checkout), today)}),
                flush=True,
            )  # fmt: skip
            if args.once:
                return 0
        sleep(args.poll)


if __name__ == "__main__":
    raise SystemExit(main())
