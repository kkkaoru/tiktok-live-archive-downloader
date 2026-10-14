"""Export pinned measured captions as bounded Motion stage plans, not render approval."""

import argparse
import json
from fractions import Fraction
from pathlib import Path
from typing import TypedDict

from replay.caption_measured_job import _pairs
from replay.caption_native_layout import NativeCaptionLayout
from replay.caption_timing import display_time
from replay.fcp_measured import MAX_CANDIDATE_BYTES, MAX_PART_BINDINGS, read_measured_captions
from replay.fcp_project import frame_span
from replay.foreground_evidence import _bound_bytes
from replay.motion_caption import MAX_SEQUENCE_SENTENCES
from replay.recordings import record
from replay.stages import file_digest
from replay.tracking_observations import _checked_output

MAX_COMBINED_BINDINGS = 128 * MAX_PART_BINDINGS


class Placement(TypedDict):
    text: str
    startFrame: int
    endFrame: int
    x: float
    y: float


def _verify_bindings(raw: dict[str, object]) -> None:
    bindings = raw.get("inputs")
    calibration = record(raw.get("renderCalibration"))
    if not isinstance(bindings, dict) or not 1 <= len(bindings) <= MAX_COMBINED_BINDINGS:
        raise ValueError("Expected bounded combined-caption provenance")
    name, digest = calibration.get("path"), calibration.get("sha256")
    if not isinstance(name, str) or not isinstance(digest, str) or bindings.get(name) != digest:
        raise ValueError("Measured captions lack a bound common calibration")
    for name, digest in bindings.items():
        if not isinstance(name, str) or not isinstance(digest, str):
            raise ValueError("Invalid measured-caption input binding")
        path = Path(name)
        if not path.is_absolute() or file_digest(_checked_output(path)) != digest:
            raise ValueError("Measured-caption input fingerprint changed")


def export_plans(
    candidate: Path,
    output: Path,
    *,
    sha256: str,
    width: int,
    height: int,
    rate: int,
    frames: int,
    expected_text: str,
) -> Path:
    """Preserve frame-visible text/origins, refusing hidden or overlapping stages.

    Each document covers only its local time interval, avoiding repeated full-length
    alpha renders. Empty gaps remain empty; frame sampling uses the existing FCP
    rule start <= frame time < end. No recognition clock or layout is changed.
    """
    if (
        any(type(value) is not int for value in (width, height, rate, frames))
        or not 1 <= min(width, height) <= max(width, height) <= 8192
        or not 1 <= rate <= 60
        or not 1 <= frames <= rate * 120
        or not expected_text.strip()
    ):
        raise ValueError("Invalid bounded Motion stage canvas or template text")
    candidate, output = _checked_output(candidate), _checked_output(output)
    if output.exists():
        raise FileExistsError(output)
    raw = record(
        json.loads(_bound_bytes(candidate, sha256, MAX_CANDIDATE_BYTES), object_pairs_hook=_pairs)
    )
    _verify_bindings(raw)
    captions = read_measured_captions(
        candidate,
        sha256=sha256,
        width=width,
        height=height,
        rate=rate,
        duration=Fraction(frames, rate),
    )
    placements: list[Placement] = []
    previous_end = 0
    for item in captions:
        span = frame_span(display_time(item.cue.start), display_time(item.cue.end), rate)
        if span is None:
            raise ValueError("Measured caption has no display frame; refusing text loss")
        start, end = int(span[0] * rate), int(span[1] * rate)
        if start < previous_end or end > frames:
            raise ValueError("Measured caption frame intervals overlap or exceed the timeline")
        placements.append(
            {
                "text": item.cue.text,
                "startFrame": start,
                "endFrame": end,
                "x": item.x - width / 2,
                "y": height / 2 - item.y,
            }
        )
        previous_end = end
    _verify_bindings(raw)
    if file_digest(candidate) != sha256:
        raise ValueError("Measured candidate changed during planning")
    output.mkdir(mode=0o700)
    parts: list[dict[str, object]] = []
    for index in range(0, len(placements), MAX_SEQUENCE_SENTENCES):
        group = placements[index : index + MAX_SEQUENCE_SENTENCES]
        offset = group[0]["startFrame"]
        last = group[-1]["endFrame"]
        local: list[Placement] = [
            {
                **entry,
                "startFrame": entry["startFrame"] - offset,
                "endFrame": entry["endFrame"] - offset,
            }
            for entry in group
        ]
        path = output / f"part-{len(parts):04d}.json"
        NativeCaptionLayout._save(
            path,
            {
                "frames": last - offset,
                "width": width,
                "height": height,
                "rate": rate,
                "expectedText": expected_text,
                "placements": local,
            },
        )
        parts.append(
            {
                "path": str(path),
                "sha256": file_digest(path),
                "offsetFrame": offset,
                "frames": last - offset,
                "stages": len(group),
            }
        )
    result = output / "manifest.json"
    NativeCaptionLayout._save(
        result,
        {
            "version": 1,
            "candidateOnly": True,
            "candidate": {"path": str(candidate), "sha256": sha256},
            "renderCalibration": raw["renderCalibration"],
            "frames": frames,
            "rate": rate,
            "parts": parts,
            "motionRenderVerified": False,
            "deliveryComplete": False,
        },
    )
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("candidate", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--sha256", required=True)
    parser.add_argument("--width", type=int, required=True)
    parser.add_argument("--height", type=int, required=True)
    parser.add_argument("--rate", type=int, required=True)
    parser.add_argument("--frames", type=int, required=True)
    parser.add_argument("--expected-text", required=True)
    args = parser.parse_args()
    export_plans(
        args.candidate,
        args.output,
        sha256=args.sha256,
        width=args.width,
        height=args.height,
        rate=args.rate,
        frames=args.frames,
        expected_text=args.expected_text,
    )


if __name__ == "__main__":
    main()
