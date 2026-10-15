"""Derive a Metal ASR pipeline config for another recording from a reference config.

Everything except the separation plan comes from the reference (tools, model, timeouts), but a
reference config also pins the digest of every code file the producer imports. Copying it as it is
keeps digests of files that have changed since, and the producer then stops with `ASR input/code
changed`. This tool replaces the plan pin and re-reads the digest of every pinned file.
"""

import argparse
import hashlib
import json
import os
import sys
from pathlib import Path

from replay.core import ReplayError

ROOT = Path(__file__).resolve().parents[2]


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def derive(reference: Path, plan: Path, output: Path) -> dict[str, object]:
    """The reference config with `plan` pinned and every pinned code file's digest refreshed."""
    if output.exists() or output.is_symlink():
        raise ReplayError("Refusing to overwrite the output")
    if not plan.is_file() or not plan.is_absolute():
        raise ReplayError("The separation plan must be an existing absolute path")
    loaded: object = json.loads(reference.read_text(encoding="utf-8"))
    if not isinstance(loaded, dict):
        raise ReplayError("The reference config must be a JSON object")
    config: dict[str, object] = loaded
    config["separationPlan"] = {"path": str(plan), "sha256": sha256(plan)}
    pinned = config.get("metalCodeInputs")
    if not isinstance(pinned, dict) or not pinned:
        raise ReplayError("The reference config pins no code files")
    refreshed: dict[str, str] = {}
    for name in pinned:
        path = Path(name)
        if not path.is_file():
            raise ReplayError("A pinned code file is missing")
        refreshed[name] = sha256(path)
    config["metalCodeInputs"] = refreshed
    with output.open("x", encoding="utf-8") as stream:
        os.fchmod(stream.fileno(), 0o600)
        json.dump(config, stream, indent=1)
    return config


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("reference", type=Path)
    parser.add_argument("plan", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args(argv)
    derive(args.reference, args.plan, args.output)
    print(args.output)
    return 0


if __name__ == "__main__":
    sys.exit(main())
