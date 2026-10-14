"""Read-only ownership diagnostics; not measured caption or delivery acceptance."""

import argparse
import json
import re
from dataclasses import asdict
from pathlib import Path

from replay.artifact_json import _json, caption_reflow_json
from replay.caption_io import _array
from replay.caption_linguistics import (
    ASSEMBLED_GROUP_OWNER,
    atomic_tokens,
    recognition_continuity,
)
from replay.caption_measured_job import _token
from replay.caption_timing import CaptionPolicy
from replay.recordings import record
from replay.stages import file_digest

MAX_TOKEN_FILES = 4096
MAX_TOKENS = 16384
TOKEN_FILE = re.compile(r"tokens(?:-[0-9]+)?\.json")


def audit_tokens(path: Path, *, include_units: bool = False) -> dict[str, object]:
    """Preserve input clocks and report canonical owners without rewriting tokens."""
    digest = file_digest(path)
    raw = _json(path)
    if raw.get("version") != 1 or raw.get("clock") != "output":
        raise ValueError("Expected version 1 output-clock tokens")
    values = _array(raw.get("tokens"), MAX_TOKENS)
    if not 0 < len(values) <= MAX_TOKENS:
        raise ValueError("Token audit count exceeds budget")
    words = [_token(value) for value in values]
    changed = sum(recognition_continuity(word.utterance) != word.utterance for word in words)
    assembled = sum(
        ASSEMBLED_GROUP_OWNER.fullmatch(word.utterance.rpartition(":cut-")[0]) is not None
        for word in words
    )
    result: dict[str, object] = {
        "path": str(path.resolve()),
        "sha256": digest,
        "tokens": len(words),
        "assembledGroupTokens": assembled,
        "normalizedOwnerTokens": changed,
        "originalOwners": len({word.utterance for word in words}),
        "continuityOwners": len({recognition_continuity(word.utterance) for word in words}),
    }
    if include_units:
        result["lexicalUnits"] = [
            asdict(word)
            for word in atomic_tokens(
                words, maximum_gap=CaptionPolicy().maximum_gap, complete_phrases=True
            )
        ]
    if file_digest(path) != digest:
        raise ValueError("Token audit input changed")
    return result


def audit_reflow(path: Path) -> dict[str, object]:
    """Inspect only the exact token files adopted by the pinned reflow inputs."""
    digest = file_digest(path)
    inputs = record(caption_reflow_json(path).get("inputs"))
    token_inputs = {
        name: pin for name, pin in inputs.items() if TOKEN_FILE.fullmatch(Path(name).name)
    }
    if not 0 < len(token_inputs) <= MAX_TOKEN_FILES:
        raise ValueError("Reflow token input count exceeds budget")
    reports = []
    for name, pin in sorted(token_inputs.items()):
        report = audit_tokens(Path(name))
        if report["sha256"] != pin:
            raise ValueError("Reflow token input pin mismatch")
        reports.append(report)
    if file_digest(path) != digest:
        raise ValueError("Reflow audit input changed")
    return {"path": str(path.resolve()), "sha256": digest, "files": reports}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", type=Path)
    parser.add_argument("--reflow", action="store_true")
    parser.add_argument("--include-units", action="store_true")
    args = parser.parse_args(argv)
    if args.reflow and args.include_units:
        parser.error("Use --include-units with one token input, not a full reflow")
    result = (
        audit_reflow(args.input)
        if args.reflow
        else audit_tokens(args.input, include_units=args.include_units)
    )
    print(
        json.dumps(
            {"version": 1, "measuredLayoutVerified": False, "audit": result},
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
