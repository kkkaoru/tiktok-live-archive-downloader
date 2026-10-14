"""Stabilize derived FCP title Y while preserving text, clocks and raw evidence."""

import argparse
import hashlib
import json
from fractions import Fraction
from math import isfinite
from pathlib import Path
from xml.etree import ElementTree as ET

from replay.artifact_json import _path
from replay.core import ReplayError
from replay.fcp_tracking_caption import MAX_TRACK_BYTES, _xml
from replay.stages import file_digest


def _time(value: str | None) -> Fraction:
    if value is None or not value.endswith("s"):
        raise ReplayError("Caption key requires an FCP rational clock")
    try:
        result = Fraction(value[:-1])
    except (ValueError, ZeroDivisionError) as error:
        raise ReplayError("Invalid caption clock") from error
    if result < 0:
        raise ReplayError("Caption clock must be nonnegative")
    return result


def stabilize_caption_y(root: ET.Element) -> None:
    """Hold each generated caption at its own first displayed-frame Y.

    No placement state passes between cues. Raw tracking and caption clocks stay
    unchanged; this derived anchor policy does not certify native painted bounds.
    """
    for title in root.findall(".//title"):
        _stabilize_title(title)


def _position(key: ET.Element) -> float:
    vector = (key.get("value") or "").split()
    try:
        values = tuple(float(part) for part in vector)
    except ValueError as error:
        raise ReplayError("Invalid caption position") from error
    if len(values) != 2 or values[0] != 0 or not all(isfinite(v) for v in values):
        raise ReplayError("Expected centered finite caption position")
    return values[1]


def _stabilize_title(title: ET.Element) -> None:
    local = _time(title.get("start"))
    duration = _time(title.get("duration"))
    keys = title.findall("adjust-transform/param[@name='position']/keyframeAnimation/keyframe")
    if not keys:
        return
    times = tuple(_time(key.get("time")) for key in keys)
    values = tuple(_position(key) for key in keys)
    if duration <= 0 or times[0] != local:
        raise ReplayError("Caption position requires an exact displayed onset")
    if any(not local <= time <= local + duration for time in times):
        raise ReplayError("Caption position key exceeds its title")
    if any(right <= left for left, right in zip(times[:-1], times[1:], strict=True)):
        raise ReplayError("Caption position clocks must strictly increase")
    for key in keys:
        key.set("value", f"0 {values[0]:.12g}")


def export_candidate(source: Path, output: Path, *, name: str) -> None:
    """Copy a bounded existing XML for native comparison; never edit its media."""
    source, output = _path(str(source)), _path(str(output))
    receipt = output.with_suffix(".stability.json")
    if output.exists() or receipt.exists() or not name.strip():
        raise ReplayError("Expected a named new stabilization candidate")
    if source.stat().st_size > MAX_TRACK_BYTES:
        raise ReplayError("Candidate XML exceeds bounded input size")
    raw = source.read_bytes()
    root = _xml(raw, root_tag="fcpxml")
    projects = root.findall("./library/event/project")
    if len(projects) != 1:
        raise ReplayError("Expected exactly one candidate project")
    stabilize_caption_y(root)
    projects[0].set("name", name)
    encoded = ET.tostring(root, encoding="utf-8", xml_declaration=True)
    with output.open("xb") as handle:
        handle.write(encoded)
    with receipt.open("x", encoding="utf-8") as handle:
        json.dump(
            {
                "version": 1,
                "candidateOnly": True,
                "nativeRenderVerified": False,
                "rawObservationsModified": False,
                "verticalPolicy": "cue-onset-anchor-hold",
                "source": {"path": str(source), "sha256": hashlib.sha256(raw).hexdigest()},
                "output": {"path": str(output), "sha256": hashlib.sha256(encoded).hexdigest()},
                "code": {str(path): file_digest(path) for path in (Path(__file__),)},
            },
            handle,
            indent=2,
        )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--name", required=True)
    args = parser.parse_args()
    export_candidate(args.source, args.output, name=args.name)


if __name__ == "__main__":
    main()
