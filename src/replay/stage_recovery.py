"""Recover a stopped workflow so its unfinished stages can run again.

`replay.stages.run_stage` never retries a stage whose receipt says `running`:
an interrupted or failed mutation must be inspected. For generated, idempotent
delivery workflows the safe recovery is mechanical, so this module performs it
without deleting anything: the old state directory is kept as `<state>.failed-K`,
completed receipts are carried into a fresh state directory, and the partial
outputs of every unfinished stage are preserved by renaming them aside (the
stage's own output directory when no other stage writes there, else the file).
Only run it while no runner owns the workflow.
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from collections import Counter
from pathlib import Path

from replay.core import ReplayError
from replay.recordings import record

MAX_RECEIPT_BYTES = 1_048_576


def _receipt(path: Path) -> dict[str, object]:
    if path.is_symlink() or path.stat().st_size > MAX_RECEIPT_BYTES:
        raise ReplayError(f"Invalid stage receipt: {path}")
    return record(json.loads(path.read_text(encoding="utf-8")))


def _stages(workflow: Path) -> list[dict[str, object]]:
    rows = record(json.loads(workflow.read_text(encoding="utf-8"))).get("stages")
    if not isinstance(rows, list):
        raise ReplayError("Workflow has no stage list")
    return [record(row) for row in rows]


def _outputs(stage: dict[str, object]) -> list[Path]:
    values = stage.get("outputs")
    if not isinstance(values, list):
        raise ReplayError("Stage has no output list")
    return [Path(str(value)) for value in values]


def _aside(path: Path, round_number: int) -> Path:
    target = path.with_name(f"{path.name}.failed-{round_number}")
    while target.exists() or target.is_symlink():
        round_number += 1
        target = path.with_name(f"{path.name}.failed-{round_number}")
    path.rename(target)
    return target


def unfinished(workflow: Path, state: Path) -> list[str]:
    """Names of stages whose receipt in `state` is still `running`."""
    if not state.is_dir():
        return []
    names = {str(stage.get("name")) for stage in _stages(workflow)}
    return sorted(
        path.stem
        for path in state.glob("*.json")
        if path.stem in names and _receipt(path).get("status") == "running"
    )


def recover(workflow: Path, state: Path) -> list[Path]:
    """Make unfinished stages runnable again; return every preserved path."""
    stopped = set(unfinished(workflow, state))
    if not stopped:
        return []
    rounds = len(list(state.parent.glob(f"{state.name}.failed-*")))
    stages = _stages(workflow)
    outputs = [output for stage in stages for output in _outputs(stage)]
    owners = Counter(output.parent.resolve() for output in outputs)
    inputs = [workflow.absolute(), workflow.resolve()]
    for stage in stages:
        values = stage.get("inputs")
        if not isinstance(values, list):
            raise ReplayError("Stage has no input list")
        for value in values:
            path = Path(str(value))
            inputs.extend((path.absolute(), path.resolve()))
    preserved = [_aside(state, rounds)]
    old = preserved[0]
    state.mkdir(mode=0o700)
    for receipt in old.glob("*.json"):
        if _receipt(receipt).get("status") == "complete":
            shutil.copy2(receipt, state / receipt.name)
    for stage in stages:
        if str(stage.get("name")) not in stopped:
            continue
        for output in _outputs(stage):
            parent = output.parent.resolve()
            exclusive = (
                owners[parent] == 1
                and not any(
                    path.is_relative_to(parent) or parent.is_relative_to(path) for path in inputs
                )
                and not any(
                    other.resolve().is_relative_to(parent) for other in outputs if other != output
                )
            )
            target = output.parent if exclusive else output
            if target.exists():
                preserved.append(_aside(target, rounds))
    return preserved


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("workflow", type=Path)
    parser.add_argument("--state", type=Path, required=True)
    args = parser.parse_args(argv)
    try:
        for path in recover(args.workflow, args.state):
            print(path)
    except (ReplayError, OSError, ValueError) as error:
        print(f"STAGE_RECOVERY_ERROR {error}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
