"""Collect every foreground window of a plan with a supervised worker pool.

The plan written by `replay.delivery_media foreground-plan` has one config per
128-frame window. This supervisor runs `replay.foreground_collection observe`
(through `replay.digest_store_run`, whose digest store is shared by every window
process) for every window without a receipt, keeps a bounded number running,
preserves a failed attempt as `evidence-NNNN.failed-sK` and retries it once.
One failing window never stops the others, and windows already observed by any
earlier run are skipped.
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

from replay import digest_store_run

POLL_SECONDS = 5.0
MAX_ATTEMPTS = 2

Spawn = Callable[[Sequence[str], Path], subprocess.Popen[bytes]]


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


def windows(directory: Path) -> list[int]:
    return [
        int(path.stem) for path in sorted((directory / "configs").glob("[0-9][0-9][0-9][0-9].json"))
    ]


def evidence(directory: Path, index: int) -> Path:
    return directory / f"evidence-{index:04d}"


def observed(directory: Path, index: int) -> bool:
    return (evidence(directory, index) / "provenance.json").is_file()


def command(directory: Path, index: int) -> list[str]:
    return [
        sys.executable,
        "-m",
        "replay.digest_store_run",
        str(digest_store_run.default_store()),
        "replay.foreground_collection",
        "observe",
        str(directory / "configs" / f"{index:04d}.json"),
        str(evidence(directory, index)),
    ]


def _reap(directory: Path, progress: Progress) -> None:
    for index, process in list(progress.running.items()):
        code = process.poll()
        if code is None:
            continue
        del progress.running[index]
        if code == 0 and observed(directory, index):
            continue
        attempts = progress.attempts.get(index, 0) + 1
        progress.attempts[index] = attempts
        output = evidence(directory, index)
        if output.exists():
            output.rename(output.with_name(f"{output.name}.failed-s{attempts}"))
        if attempts >= MAX_ATTEMPTS:
            progress.failed.add(index)


def supervise(
    directory: Path,
    *,
    workers: int,
    spawn: Spawn = _spawn,
    sleep: Callable[[float], None] = time.sleep,
) -> dict[str, object]:
    """Run until every window is observed or has exhausted its attempts."""
    directory = directory.absolute()
    logs = directory / "supervisor-logs"
    logs.mkdir(mode=0o700, exist_ok=True)
    progress = Progress()
    all_windows = windows(directory)
    while True:
        _reap(directory, progress)
        pending = [
            index
            for index in all_windows
            if not observed(directory, index)
            and index not in progress.running
            and index not in progress.failed
            and not evidence(directory, index).exists()
        ]
        for index in pending[: max(workers - len(progress.running), 0)]:
            progress.running[index] = spawn(command(directory, index), logs / f"{index:04d}.log")
        done = sum(1 for index in all_windows if observed(directory, index))
        print(
            json.dumps(
                {"done": done, "total": len(all_windows), "failed": sorted(progress.failed)}
            ),
            flush=True,
        )
        blocked = [
            index
            for index in all_windows
            if not observed(directory, index)
            and index not in progress.failed
            and index not in progress.running
        ]
        if not progress.running and not pending and len(blocked) == 0:
            return {"done": done, "total": len(all_windows), "failed": sorted(progress.failed)}
        if not progress.running and not pending:
            return {
                "done": done,
                "total": len(all_windows),
                "failed": sorted(progress.failed),
                "busyElsewhere": blocked,
            }
        sleep(POLL_SECONDS)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory", type=Path)
    parser.add_argument("--workers", type=int, default=4)
    args = parser.parse_args(argv)
    result = supervise(args.directory, workers=args.workers)
    print(json.dumps(result))
    return 0 if not result["failed"] and "busyElsewhere" not in result else 1


if __name__ == "__main__":
    raise SystemExit(main())
