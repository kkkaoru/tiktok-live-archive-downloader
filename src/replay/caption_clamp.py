"""Measure a caption job whose subject leaves no room above the head.

`replay.caption_measured_job` places captions above the highest foreground
point of its window and refuses a window where the subject reaches the top of
the frame. For such windows the delivery policy keeps captions in an upper band
instead: the observed support top is floored at `--minimum-top`, so captions
sit at that band and may overlap the subject there. Everything else - tokens,
clocks, measured widths, POS line breaks and Motion calibration - is the
unchanged measured job. A `clamp.json` record beside the candidate states the
floor, so the fallback is never mistaken for an above-head placement.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from collections.abc import Callable
from fractions import Fraction
from pathlib import Path

from replay import caption_measured_job, digest_memo, foreground_geometry, stages
from replay.caption_source_support import OutputSourceSupport
from replay.core import ReplayError
from replay.foreground_observations import ForegroundFrame

MAX_MINIMUM_TOP = 1920
UNKNOWN_SUPPORT = "Required foreground support is unknown"


def clamped(floor: float) -> type[OutputSourceSupport]:
    """An OutputSourceSupport whose top is never above `floor`."""

    class ClampedSupport(OutputSourceSupport):
        def __call__(self, start: Fraction, end: Fraction) -> float:
            return max(OutputSourceSupport.__call__(self, start, end), floor)

    return ClampedSupport


def unknown_frames(default: float) -> Callable[[ForegroundFrame], float]:
    """Per-frame support top where a frame without an observed subject is `default`.

    Applying the default per frame keeps every interval minimum monotonic, so a
    run never reports less room than its enclosing window.
    """
    original = foreground_geometry._support_top

    def support_top(frame: ForegroundFrame) -> float:
        try:
            return original(frame)
        except ReplayError as error:
            if str(error) != UNKNOWN_SUPPORT:
                raise
            return default

    return support_top


def run(
    manifest: Path,
    directory: Path,
    *,
    sha256: str,
    minimum_top: float,
    unknown_top: float | None = None,
) -> Path:
    if not 0 < minimum_top <= MAX_MINIMUM_TOP:
        raise ValueError("Minimum support top must lie inside the canvas")
    digest_memo.install(stages.file_digest)
    caption_measured_job.__dict__["OutputSourceSupport"] = clamped(minimum_top)
    if unknown_top is not None:
        foreground_geometry.__dict__["_support_top"] = unknown_frames(unknown_top)
    candidate = caption_measured_job.plan(manifest, directory, manifest_sha256=sha256)
    record = directory / "clamp.json"
    with record.open("x", encoding="utf-8") as stream:
        os.fchmod(stream.fileno(), 0o600)
        json.dump(
            {
                "minimumSupportTop": minimum_top,
                "unknownSupportTop": unknown_top,
                "reason": "subject reaches the frame top; captions held in the upper band",
                "abovePlacementApproved": False,
            },
            stream,
        )
    return candidate


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("manifest", type=Path)
    parser.add_argument("directory", type=Path)
    parser.add_argument("--manifest-sha256", required=True)
    parser.add_argument("--minimum-top", type=float, required=True)
    parser.add_argument("--unknown-top", type=float)
    args = parser.parse_args(argv)
    try:
        run(
            args.manifest,
            args.directory,
            sha256=args.manifest_sha256,
            minimum_top=args.minimum_top,
            unknown_top=args.unknown_top,
        )
    except (ValueError, OSError, ReplayError) as error:
        print(f"CAPTION_CLAMP_ERROR {error}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
