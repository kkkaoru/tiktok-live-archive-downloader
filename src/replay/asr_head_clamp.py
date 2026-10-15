"""Clamp a label that starts inside the recording's partial first video frame.

When the decoded audio starts part-way through a video frame, the first whole frame begins slightly
later (``first_frame / fps``), so audio before that boundary cannot be delivered. A recognized word
that begins in that sub-frame head but continues into the delivered range is not removed: only its
onset is, and the label is rewritten to start at the first delivered boundary. The original start,
the boundary and the dropped length are kept as provenance and the input file is never changed.
A label that would be removed entirely, a point label, or a label that starts before the decoded
audio is refused, so this is never a way to drop or invent speech. Because the label starts at or
after the audio start, the dropped part is always shorter than one frame.
"""

import argparse
import json
import sys
from collections.abc import Sequence
from decimal import ROUND_CEILING, Decimal
from fractions import Fraction
from math import ceil
from pathlib import Path

from replay.core import ReplayError

MAX_LABELS = 1_000_000
DIGITS = Decimal("0.000000001")


def not_below(value: Fraction) -> float:
    """The smallest 9-digit decimal at or above `value`, so a rounded start never undershoots it."""
    exact = Decimal(value.numerator) / Decimal(value.denominator)
    return float(exact.quantize(DIGITS, rounding=ROUND_CEILING))


def first_boundary(audio_start: Fraction, *, fps: int) -> Fraction:
    """Start (seconds) of the first whole video frame at or after the audio start."""
    if type(fps) is not int or not 1 <= fps <= 240 or audio_start < 0:
        raise ReplayError("Invalid frame rate or audio start")
    return Fraction(ceil(audio_start * fps), fps)


def clamp_head(
    labels: Sequence[dict[str, object]],
    *,
    audio_start: Fraction,
    offset: Fraction,
    fps: int,
) -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    """Return (labels, clamps). `offset` adds the audio start to the label clock (source clock).

    Labels are on the input-PCM clock; ``offset + start`` is the source time compared against the
    first delivered boundary.
    """
    if not 0 < len(labels) <= MAX_LABELS:
        raise ReplayError("Invalid label count")
    boundary = first_boundary(audio_start, fps=fps)
    result: list[dict[str, object]] = []
    clamps: list[dict[str, object]] = []
    for index, label in enumerate(labels):
        start, end = Fraction(str(label["start"])), Fraction(str(label["end"]))
        source_start, source_end = offset + start, offset + end
        if source_start >= boundary:
            result.append(dict(label))
            continue
        if source_start < audio_start:
            raise ReplayError("A label starts before the decoded audio")
        if source_end <= boundary or start == end:
            raise ReplayError("A label would be removed entirely at the head")
        dropped = boundary - source_start
        changed = dict(label)
        changed["start"] = not_below(boundary - offset)
        changed["headClamp"] = {
            "originalStart": label["start"],
            "firstDeliveredBoundary": float(boundary),
            "droppedSeconds": float(dropped),
            "rule": "sub-frame head onset not delivered; label kept from the first whole frame",
        }
        result.append(changed)
        clamps.append({"index": index, "text": label["text"], "droppedSeconds": float(dropped)})
    return result, clamps


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("labels", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--audio-start", required=True, help="seconds, e.g. 0.901")
    parser.add_argument("--fps", type=int, required=True)
    args = parser.parse_args(argv)
    if args.output.exists() or args.output.is_symlink():
        raise ReplayError("Refusing to overwrite the output")
    document = json.loads(args.labels.read_text(encoding="utf-8"))
    start = Fraction(args.audio_start)
    labels, clamps = clamp_head(document["labels"], audio_start=start, offset=start, fps=args.fps)
    document = dict(document) | {
        "labels": labels,
        "headClamps": clamps,
        "parentLabelsFile": str(args.labels),
    }
    args.output.write_text(json.dumps(document, ensure_ascii=False, indent=1), encoding="utf-8")
    print(json.dumps({"clamped": len(clamps), "clamps": clamps}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    sys.exit(main())
