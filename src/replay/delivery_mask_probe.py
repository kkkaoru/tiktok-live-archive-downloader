"""Render at most eight NEW source-mask frames with writer-bound lossless evidence.

Not reconstruction of a historical encoder input or temporal/readability approval.
Use the current shared producer, never a historical implementation snapshot.
"""

import argparse
import json
import os
from fractions import Fraction
from pathlib import Path

from replay.core import ReplayError
from replay.delivery_mask_quality import digest, integer, local_path, read_json, rows
from replay.recordings import record
from replay.subtitle_candidate import render_candidate
from replay.subtitle_clock import VideoFrameClock
from replay.subtitle_plan import repair_spans


def probe(producer: Path, output: Path, *, sha256: str, start: int, count: int = 8) -> Path:
    """Preserve actual PTS, ROI membership and the producer's global softness ramp."""
    local_path(str(producer))
    local_path(str(output))
    if digest(producer) != sha256:
        raise ReplayError("Producer configuration changed")
    if type(start) is not int or start < 0 or type(count) is not int or not 1 <= count <= 8:
        raise ReplayError("Probe requires a nonnegative start and one to eight frames")
    data = read_json(producer)
    clock = record(data.get("clock"))
    pts = rows(clock.get("pts"))
    if start + count > len(pts):
        raise ReplayError("Probe exceeds source clock")
    time_base = clock.get("timeBase")
    if not isinstance(time_base, str):
        raise ReplayError("Missing source time base")
    scale = Fraction(time_base)
    if scale <= 0:
        raise ReplayError("Invalid source time base")
    original_clock = VideoFrameClock(
        tuple(integer(value) for value in pts), scale, integer(clock.get("lastDuration"))
    )
    spans = repair_spans(
        data.get("masks"),
        frames=len(pts),
        width=integer(data.get("width")),
        height=integer(data.get("height")),
        source_clock=original_clock,
    )
    masks: list[dict[str, object]] = []
    for frame in range(start, start + count):
        span = next((s for s in spans if s.start_frame <= frame < s.end_frame), None)
        if span is None:
            raise ReplayError("Probe frame lacks planned mask; select an active interval")
        end = (
            original_clock.pts[frame + 1]
            if frame + 1 < len(pts)
            else (original_clock.pts[-1] + original_clock.last_duration)
        )
        masks.append(
            {
                "startSeconds": float(
                    (original_clock.pts[frame] - original_clock.pts[start]) * scale
                ),
                "endSeconds": float((end - original_clock.pts[start]) * scale),
                "region": {"x": span.x, "y": span.y, "width": span.width, "height": span.height},
            }
        )
    data.update(
        clock={
            "pts": pts[start : start + count],
            "timeBase": time_base,
            "lastDuration": (
                integer(pts[start + count]) - integer(pts[start + count - 1])
                if start + count < len(pts)
                else integer(clock.get("lastDuration"))
            ),
        },
        masks=masks,
        rampStartFrame=integer(data.get("rampStartFrame", 0)) + start,
        rampTotalFrames=integer(data.get("rampTotalFrames", len(pts))),
        evidenceFrames=list(range(count)),
        outputDirectory=str(output / "candidate"),
        workDirectory=str(output / "work"),
        probeParent={"path": str(producer), "sha256": sha256},
        probeImplementation={
            "path": str(Path(__file__).resolve()),
            "sha256": digest(Path(__file__)),
        },
    )
    output.mkdir(mode=0o700)
    config = output / "probe-config.json"
    with config.open("x", encoding="utf-8") as stream:
        os.fchmod(stream.fileno(), 0o600)
        json.dump(data, stream, indent=2, allow_nan=False)
    result = render_candidate(config)
    if digest(producer) != sha256:
        raise ReplayError("Producer changed during probe; candidate is not approved")
    return result


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("producer", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--sha256", required=True)
    parser.add_argument("--start-frame", type=int, required=True)
    parser.add_argument("--count", type=int, default=8)
    args = parser.parse_args(argv)
    probe(args.producer, args.output, sha256=args.sha256, start=args.start_frame, count=args.count)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
