"""Re-pin one edited controller file in submitted Compressor receipts, keeping history.

Version-2 receipts pin the controller code (`compressor_batch.py`) by size, times
and SHA-256, so editing it invalidates every receipt written by an earlier copy.
`python -m replay.compressor_repin STATE_DIR OLD_SHA256 [--apply]` replaces only
that pin, only where the recorded digest equals OLD_SHA256 and every other
runtime pin still matches the live file; the replaced pin is kept in
`runtimePinHistory`. Default is a dry run.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from replay import compressor_batch as batch
from replay.core import ReplayError

CONTROLLER = "compressor_batch.py"


def repin(state: Path, old_sha256: str, *, apply: bool) -> list[str]:
    changed: list[str] = []
    live: dict[str, dict[str, object]] = {}

    def current(path: Path) -> dict[str, object]:
        key = str(path.absolute())
        if key not in live:
            live[key] = batch._pin(path)
        return live[key]

    for receipt in sorted(state.glob("alpha-*.json")):
        saved = json.loads(receipt.read_text(encoding="utf-8"))
        pins = saved.get("runtimeInputs")
        command = saved.get("command")
        if saved.get("submissionVersion") != 2 or not isinstance(pins, list):
            continue
        if not isinstance(command, list) or not command or not isinstance(command[0], str):
            raise ReplayError(f"Invalid command in {receipt.name}")
        paths = batch.runtime_paths(Path(command[0]))
        stale = [
            index
            for index, (path, pin) in enumerate(zip(paths, pins, strict=True))
            if batch._portable(pin) != batch._portable(current(path))
        ]
        if not stale:
            continue
        if len(stale) != 1 or paths[stale[0]].name != CONTROLLER:
            raise ReplayError(f"{receipt.name}: unexpected changed runtime pins {stale}")
        if pins[stale[0]].get("sha256") != old_sha256:
            raise ReplayError(f"{receipt.name}: controller digest is not {old_sha256}")
        if apply:
            saved.setdefault("runtimePinHistory", []).append(
                {"replaced": pins[stale[0]], "reason": "controller code edited after submit"}
            )
            pins[stale[0]] = current(paths[stale[0]])
            batch._write(receipt, saved)
        changed.append(receipt.name)
    return changed


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("state", type=Path)
    parser.add_argument("old_sha256")
    parser.add_argument("--apply", action="store_true")
    args = parser.parse_args(argv)
    try:
        names = repin(args.state, args.old_sha256, apply=args.apply)
    except (ReplayError, OSError, ValueError) as error:
        print(f"COMPRESSOR_REPIN_ERROR {error}", file=sys.stderr)
        return 1
    print(f"{'repinned' if args.apply else 'would repin'} {len(names)} receipts")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
