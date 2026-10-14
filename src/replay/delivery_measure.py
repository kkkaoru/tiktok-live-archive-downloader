"""Supervise all measured-caption jobs of one recording until they finish.

A single long-running process repeatedly (1) plans jobs whose foreground
windows have become available, (2) keeps a bounded pool of measurement
processes busy, and (3) preserves a failed attempt as
`measurement-NNNN.failed-K` and retries it, up to a fixed attempt budget. One
failing job never stops the others, unlike a sequential stage workflow.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from pathlib import Path

from replay import delivery_captions as captions
from replay.core import ReplayError

POLL_SECONDS = 10.0
MAX_ATTEMPTS = 2

Spawn = Callable[[Sequence[str], Path], subprocess.Popen[bytes]]


@dataclass(frozen=True, slots=True)
class Inputs:
    tokens: Path
    cuts: Path
    foreground: tuple[Path, ...]
    window_list: Path | None
    settings: captions.Settings


@dataclass
class Progress:
    running: dict[int, subprocess.Popen[bytes]] = field(default_factory=dict)
    attempts: dict[int, int] = field(default_factory=dict)
    failed: set[int] = field(default_factory=set)


def _spawn(command: Sequence[str], log: Path) -> subprocess.Popen[bytes]:
    with log.open("ab") as stream:
        return subprocess.Popen(
            list(command), stdin=subprocess.DEVNULL, stdout=stream, stderr=subprocess.STDOUT
        )


def plan_ready(directory: Path, inputs: Inputs) -> tuple[int, int]:
    """Write manifests for newly ready jobs; return (ready manifests, total jobs)."""
    tokens, changes = captions.prepare_tokens(inputs.tokens, rate=inputs.settings.rate)
    extras = captions.read_window_list(inputs.window_list)
    windows = captions.window_index(inputs.foreground, extras)
    segments = captions.partition(
        captions.read_cuts(inputs.cuts), tokens, windows, rate=inputs.settings.rate
    )
    captions.write_jobs(directory, segments, inputs.settings, workers=1)
    record = directory / "token-changes.json"
    if not record.exists():
        captions._write_new(record, changes)
    total = sum(len(segment.jobs) for segment in segments)
    return len(list(directory.glob("manifest-[0-9][0-9][0-9][0-9].json"))), total


def _done(directory: Path, index: int) -> bool:
    return (directory / f"measurement-{index:04d}" / "candidate.json").is_file()


def _pending(directory: Path, progress: Progress) -> list[int]:
    indices = [
        int(path.stem.removeprefix("manifest-"))
        for path in sorted(directory.glob("manifest-[0-9][0-9][0-9][0-9].json"))
    ]
    return [
        index
        for index in indices
        if not _done(directory, index)
        and index not in progress.running
        and index not in progress.failed
    ]


def _reap(directory: Path, progress: Progress) -> None:
    for index, process in list(progress.running.items()):
        code = process.poll()
        if code is None:
            continue
        del progress.running[index]
        if code == 0 and _done(directory, index):
            continue
        attempts = progress.attempts.get(index, 0) + 1
        progress.attempts[index] = attempts
        output = directory / f"measurement-{index:04d}"
        if output.exists():
            output.rename(output.with_name(f"{output.name}.failed-s{attempts}"))
        if attempts >= MAX_ATTEMPTS:
            progress.failed.add(index)


def supervise(
    directory: Path,
    inputs: Inputs,
    *,
    workers: int,
    spawn: Spawn = _spawn,
    sleep: Callable[[float], None] = time.sleep,
) -> dict[str, object]:
    """Run until every job has a candidate or no further progress is possible."""
    directory = directory.absolute()
    progress = Progress()
    logs = directory / "supervisor-logs"
    while True:
        ready, total = plan_ready(directory, inputs)
        logs.mkdir(mode=0o700, exist_ok=True)
        _reap(directory, progress)
        for index in _pending(directory, progress)[: max(workers - len(progress.running), 0)]:
            stage = captions._measure_stage(
                directory,
                index,
                minimum_top=inputs.settings.minimum_top,
                unknown_top=inputs.settings.unknown_top,
            )
            command = stage["command"]
            if not isinstance(command, list):
                raise ReplayError("Invalid measurement command")
            progress.running[index] = spawn(
                [str(item) for item in command], logs / f"{index:04d}.log"
            )
        done = sum(1 for index in range(total) if _done(directory, index))
        print(
            json.dumps(
                {"done": done, "ready": ready, "total": total, "failed": sorted(progress.failed)}
            ),
            flush=True,
        )
        if not progress.running and done + len(progress.failed) == total:
            return {"done": done, "total": total, "failed": sorted(progress.failed)}
        sleep(POLL_SECONDS)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("tokens", type=Path)
    parser.add_argument("cuts", type=Path)
    parser.add_argument("directory", type=Path)
    parser.add_argument("--foreground", type=Path, action="append", default=[])
    parser.add_argument("--window-list", type=Path)
    parser.add_argument("--native", type=Path, required=True)
    parser.add_argument("--calibration", type=Path, required=True)
    parser.add_argument("--rate", type=int, default=30)
    parser.add_argument("--minimum-top", type=float)
    parser.add_argument("--unknown-top", type=float)
    parser.add_argument("--workers", type=int, default=4)
    args = parser.parse_args(argv)
    settings = captions.Settings(
        args.rate,
        args.native.resolve(strict=True),
        args.calibration.resolve(strict=True),
        (0, 0, 1080, 1920),
        540,
        24,
        args.minimum_top,
        args.unknown_top,
    )
    inputs = Inputs(args.tokens, args.cuts, tuple(args.foreground), args.window_list, settings)
    try:
        result = supervise(args.directory, inputs, workers=args.workers)
    except (ReplayError, OSError, ValueError) as error:
        print(f"DELIVERY_MEASURE_ERROR {error}", file=sys.stderr)
        return 1
    print(json.dumps(result))
    return 0 if not result["failed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
