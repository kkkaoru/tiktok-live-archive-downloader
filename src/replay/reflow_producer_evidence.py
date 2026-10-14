"""Bind historical producer bytes to immutable measured-layout artifacts.

Saved code is evidence only and is never imported or executed. Media, tokens,
measurements and every other input retain their current-file hash requirement.
"""

import argparse
import json
from pathlib import Path

from replay.core import ReplayError
from replay.foreground_evidence import _bound_bytes
from replay.recordings import record
from replay.stages import file_digest

MAX_MANIFEST_BYTES = 1024 * 1024
PRODUCER_NAMES = frozenset({"caption_native_paging.py", "caption_native_reflow.py"})


def _producer(value: object) -> Path:
    if not isinstance(value, str):
        raise ReplayError("Expected producer source path")
    path = Path(value)
    if (
        path.name not in PRODUCER_NAMES
        or path != Path(__file__).resolve().parent / path.name
        or path.resolve() != path
    ):
        raise ReplayError("Historical evidence is restricted to layout producer code")
    return path


def read(path: Path, *, artifact: Path, inputs: dict[str, object]) -> dict[str, str]:
    """Verify exact artifact and preserved producer bytes, returning code-only pins."""
    raw = record(json.loads(_bound_bytes(path, file_digest(path), MAX_MANIFEST_BYTES)))
    bound = record(raw.get("artifact"))
    rows = raw.get("producers")
    if (
        raw.get("version") != 1
        or bound.get("path") != str(artifact.resolve())
        or bound.get("sha256") != file_digest(artifact)
        or not isinstance(rows, list)
        or not 1 <= len(rows) <= len(PRODUCER_NAMES)
    ):
        raise ReplayError("Producer evidence does not bind this measured artifact")
    pins: dict[str, str] = {}
    for value in rows:
        row = record(value)
        source = str(_producer(row.get("sourcePath")))
        digest = row.get("sha256")
        evidence = record(row.get("evidence"))
        saved = evidence.get("path")
        if (
            source in pins
            or not isinstance(digest, str)
            or inputs.get(source) != digest
            or evidence.get("sha256") != digest
            or not isinstance(saved, str)
            or Path(saved).resolve() != Path(saved)
            or saved == source
            or file_digest(Path(saved)) != digest
        ):
            raise ReplayError("Historical producer bytes or input binding changed")
        pins[source] = digest
    return pins


def prepare(*, artifact: Path, source: Path, evidence: Path, output: Path) -> None:
    """Create an exclusive manifest for previously preserved producer bytes."""
    source = _producer(str(source.absolute()))
    raw = record(json.loads(_bound_bytes(artifact, file_digest(artifact), 64 * 1024 * 1024)))
    inputs = record(raw.get("inputs"))
    digest = file_digest(evidence)
    if inputs.get(str(source)) != digest or evidence.resolve() == source:
        raise ReplayError("Saved producer does not match artifact input")
    document = {
        "version": 1,
        "artifact": {"path": str(artifact.resolve()), "sha256": file_digest(artifact)},
        "producers": [
            {
                "sourcePath": str(source),
                "sha256": digest,
                "evidence": {"path": str(evidence.resolve()), "sha256": digest},
            }
        ],
    }
    with output.open("x", encoding="utf-8") as handle:
        json.dump(document, handle, indent=2)
        handle.write("\n")


def evidence_inputs(path: Path, *, artifact: Path, inputs: dict[str, object]) -> dict[str, str]:
    """Return current manifest/archive pins after verifying the code-only exception."""
    read(path, artifact=artifact, inputs=inputs)
    digest = file_digest(path)
    raw = record(json.loads(_bound_bytes(path, digest, MAX_MANIFEST_BYTES)))
    rows = raw.get("producers")
    if not isinstance(rows, list):
        raise ReplayError("Expected verified producer evidence rows")
    pins = {str(path.resolve()): digest}
    for value in rows:
        evidence = record(record(value).get("evidence"))
        saved, expected = evidence.get("path"), evidence.get("sha256")
        if (
            not isinstance(saved, str)
            or not isinstance(expected, str)
            or file_digest(Path(saved)) != expected
        ):
            raise ReplayError("Producer evidence changed during input collection")
        pins[saved] = expected
    return pins


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("artifact", type=Path)
    parser.add_argument("source", type=Path)
    parser.add_argument("evidence", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    prepare(artifact=args.artifact, source=args.source, evidence=args.evidence, output=args.output)


if __name__ == "__main__":
    main()
