"""Residual-caption audit of a repaired chunk from OCR of every Nth repaired frame.

Vision reports unreadable repair smear as fixed low-confidence guesses (0.3 and 0.5) and real
text with confidence 1. A repaired frame therefore fails the audit when any line at or above
``MIN_CONFIDENCE`` overlaps that frame's search region. Failing frames name the chunk intervals
that must escalate to full OCR. A passing audit samples frames; it proves no absence.
"""

from collections.abc import Mapping, Sequence
from dataclasses import dataclass

from replay.caption_io import FrameObservation
from replay.subtitle_region_fill import Region

MIN_CONFIDENCE = 0.9
AUDIT_STRIDE = 10


@dataclass(frozen=True, slots=True)
class Residual:
    frame: int
    text: str
    confidence: float
    box: tuple[float, float, float, float] | None = None


def audit_indices(count: int, stride: int = AUDIT_STRIDE) -> tuple[int, ...]:
    """Every stride-th frame plus the last one."""
    if count < 1 or stride < 1:
        raise ValueError("Frame count and stride must be positive")
    return tuple(sorted({*range(0, count, stride), count - 1}))


def _touches(left: float, top: float, width: float, height: float, region: Region) -> bool:
    return (
        left < region[2]
        and left + width > region[0]
        and top < region[3]
        and top + height > region[1]
    )


def residual_lines(
    observed: Mapping[int, FrameObservation],
    regions: Sequence[Region | None],
    *,
    minimum_confidence: float = MIN_CONFIDENCE,
) -> tuple[Residual, ...]:
    """Confident lines of the repaired frames that overlap the frame's search region."""
    found: list[Residual] = []
    for index, frame in sorted(observed.items()):
        region = regions[index]
        if region is None:
            continue
        for line in frame.detections:
            box = line.rectangle
            if line.confidence >= minimum_confidence and _touches(
                box.x, box.y, box.width, box.height, region
            ):
                found.append(
                    Residual(
                        index, line.text, line.confidence, (box.x, box.y, box.width, box.height)
                    )
                )
    return tuple(found)


def escalation_frames(residuals: Sequence[Residual], count: int, stride: int) -> tuple[int, ...]:
    """Frames of the audit intervals around every failing audit frame (to rerun with full OCR)."""
    frames: set[int] = set()
    for item in residuals:
        frames.update(range(max(0, item.frame - stride + 1), min(count, item.frame + stride)))
    return tuple(sorted(frames))
