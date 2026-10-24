"""Order the GPU lease priority shortest-remaining-GPU-work first (``settings.json``).

The lease order is live configuration, read by ``gpu_lease`` at every attempt and bound into no
receipt, so rewriting it changes only which waiter is served next. A recording's remaining GPU
work is the mean measured wall of each GPU step it has no receipt for, plus the measured mean
cleanup scaled by its chunks still without a receipt. Serving the least remaining work first
lowered the simulated mean completion from 7.57 h to 5.59 h for 7 recordings (makespan +0.5 h,
bottleneck survey 2026-10-24). Recordings with a ``summary`` receipt are done and dropped;
``capacity`` is kept. Without ``--apply`` the order is only printed.
"""

from __future__ import annotations

import argparse
import json
import os
from collections.abc import Sequence
from pathlib import Path

from replay import gpu_lease
from replay.recordings import record

GPU_WORK_STEPS = ("separation", "asr", "yolo-track", "onsets", "cleanup", "compose")
CLEANUP_REVISION = "p1"


def step_means(recordings: Path) -> dict[str, float]:
    """Mean measured ``wallSeconds`` of each GPU step over every recording's receipts."""
    walls: dict[str, list[float]] = {step: [] for step in GPU_WORK_STEPS}
    for step in GPU_WORK_STEPS:
        for receipt in recordings.glob(f"*/steps/{step}.json"):
            wall = record(json.loads(receipt.read_text(encoding="utf-8"))).get("wallSeconds")
            if isinstance(wall, int | float) and wall > 0:
                walls[step].append(float(wall))
    return {step: sum(values) / len(values) if values else 0.0 for step, values in walls.items()}


def cleanup_left(work: Path, name: str) -> float:
    """The share of a recording's cleanup chunks without a receipt (1.0 before configs)."""
    clean = work / f"clean-{name}-{CLEANUP_REVISION}"
    configs = list((clean / "configs").glob("chunk-*.json"))
    if not configs:
        return 1.0
    finished = sum(
        (clean / "state" / path.stem / "chunk-receipt.json").exists() for path in configs
    )
    return (len(configs) - finished) / len(configs)


def remaining(recording: Path, means: dict[str, float], work: Path) -> float:
    """Estimated GPU seconds a recording still needs."""
    total = 0.0
    for step in GPU_WORK_STEPS:
        if (recording / "steps" / f"{step}.json").exists():
            continue
        share = cleanup_left(work, recording.name) if step == "cleanup" else 1.0
        total += means[step] * share
    return total


def lease_key(name: str) -> str:
    """The ID a recording leases under: ``<id>`` for TikTok, ``youtube-<id>`` for YouTube."""
    return name.removeprefix("replay-")


def ordered(recordings: Path, work: Path) -> list[tuple[str, float]]:
    """Unfinished recordings, least remaining GPU work first (then by name)."""
    means = step_means(recordings)
    rows = [
        (lease_key(path.name), remaining(path, means, work))
        for path in sorted(recordings.iterdir())
        if (path / "steps").is_dir() and not (path / "steps" / "summary.json").exists()
    ]
    return sorted(rows, key=lambda row: (row[1], row[0]))


def apply(directory: Path, priority: list[str]) -> list[str]:
    """Write the priority (as many as the settings size allows), keeping the capacity."""
    capacity, _ = gpu_lease.settings(directory)
    kept: list[str] = []
    for key in priority:
        text = json.dumps({"capacity": capacity, "priority": [*kept, key]})
        if len(text.encode()) > gpu_lease.MAX_SETTINGS_BYTES:
            break
        kept.append(key)
    path = directory / "settings.json"
    temporary = path.with_name(f".settings.{os.getpid()}.tmp")
    temporary.write_text(json.dumps({"capacity": capacity, "priority": kept}) + "\n", "utf-8")
    temporary.replace(path)  # gpu_lease never reads a half-written file
    return kept


def refresh(directory: Path, recordings: Path, work: Path) -> list[str] | None:
    """Rewrite the priority when the order changed; None while ``manual`` pins a hand-set one."""
    path = directory / "settings.json"
    if path.exists() and record(json.loads(path.read_text(encoding="utf-8"))).get("manual") is True:
        return None
    priority = [key for key, _ in ordered(recordings, work)]
    current = list(gpu_lease.settings(directory)[1])
    return current if current == priority else apply(directory, priority)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--recordings", type=Path, default=Path("edits/work/recordings"))
    parser.add_argument("--work", type=Path, default=Path("edits/work"))
    parser.add_argument("--leases", type=Path, default=gpu_lease.root())
    parser.add_argument("--apply", action="store_true", help="write settings.json")
    args = parser.parse_args(argv)
    rows = ordered(args.recordings, args.work)
    for key, seconds in rows:
        print(f"{key}\t{seconds:.0f}")
    if args.apply:
        print(f"priority: {len(apply(args.leases, [key for key, _ in rows]))} recordings")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
