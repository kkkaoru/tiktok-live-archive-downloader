"""Fill per-frame text boxes from OCR of selected frames only (a sampled, not raw, observation).

Only the sampled frames are recognised. Every other frame receives the union of the boxes of
its two bracketing samples. The result is a conservative mask candidate, never a raw OCR
reading, and the receipt says so together with the exact sampling rule and sampled indices.
"""

from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

FILL_RULE = "bracketing-sample-union-v1"
MAD_SCALE = 6.0  # trigger when a frame delta exceeds the median by this many MADs
COVER = 0.7  # a true box counts as covered when one filled box overlaps >= 70% of its area


@dataclass(frozen=True, slots=True)
class Box:
    x: float
    y: float
    width: float
    height: float

    @property
    def area(self) -> float:
        return max(self.width, 0.0) * max(self.height, 0.0)

    def overlap(self, other: "Box") -> float:
        width = min(self.x + self.width, other.x + other.width) - max(self.x, other.x)
        height = min(self.y + self.height, other.y + other.height) - max(self.y, other.y)
        return max(width, 0.0) * max(height, 0.0)


@dataclass(frozen=True, slots=True)
class Coverage:
    total: int
    missed: int
    true_area: float
    extra_area: float


def sample_indices(count: int, stride: int, triggers: Iterable[int] = ()) -> tuple[int, ...]:
    """First frame, every stride-th frame, the last frame and any change-trigger frames."""
    if count < 1 or stride < 1:
        raise ValueError("Frame count and stride must be positive")
    extra = tuple(triggers)
    if any(not 0 <= index < count for index in extra):
        raise ValueError("Trigger frame outside the interval")
    return tuple(sorted({*range(0, count, stride), count - 1, *extra}))


def _bracket(sampled: Sequence[int], frame: int) -> tuple[int, int]:
    previous = max(index for index in sampled if index <= frame)
    following = min(index for index in sampled if index >= frame)
    return previous, following


def fill(count: int, samples: Mapping[int, tuple[Box, ...]]) -> tuple[tuple[Box, ...], ...]:
    """Per-frame boxes: a sampled frame keeps its own boxes, others the bracketing union."""
    sampled = sorted(samples)
    if not sampled or sampled[0] != 0 or sampled[-1] != count - 1:
        raise ValueError("Samples must include the first and last frame")
    filled: list[tuple[Box, ...]] = []
    for frame in range(count):
        previous, following = _bracket(sampled, frame)
        if previous == following:
            filled.append(samples[frame])
            continue
        filled.append(samples[previous] + samples[following])
    return tuple(filled)


def _covered(box: Box, predicted: Sequence[Box]) -> bool:
    return any(box.overlap(other) >= COVER * box.area for other in predicted)


def coverage(truth: Sequence[tuple[Box, ...]], predicted: Sequence[tuple[Box, ...]]) -> Coverage:
    """Compare filled boxes with full per-frame OCR: missed true lines and extra masked area."""
    if len(truth) != len(predicted):
        raise ValueError("Truth and prediction cover different frame counts")
    total = missed = 0
    true_area = extra_area = 0.0
    for actual, filled in zip(truth, predicted, strict=True):
        total += len(actual)
        missed += sum(not _covered(box, filled) for box in actual)
        true_area += sum(box.area for box in actual)
        extra_area += sum(
            box.area
            for box in filled
            if not any(box.overlap(other) >= COVER * box.area for other in actual)
        )
    return Coverage(total, missed, true_area, extra_area)


def receipt(
    count: int, stride: int, sampled: Sequence[int], triggers: Sequence[int]
) -> dict[str, object]:
    """Honest provenance: which frames were read, by which rule, and that the rest is filled."""
    return {
        "version": 1,
        "kind": "sampled-ocr-observation",
        "rawOcr": False,
        "interpolated": True,
        "fillRule": FILL_RULE,
        "frames": count,
        "stride": stride,
        "sampledFrames": list(sampled),
        "triggerFrames": list(triggers),
        "qualityApproved": False,
    }


def change_triggers(bands: NDArray[np.uint8], scale: float = MAD_SCALE) -> tuple[int, ...]:
    """Frames bracketing an abrupt change of the mask band, from the measured delta distribution.

    The threshold is derived from the interval itself (median plus scale median-absolute
    deviations of consecutive mean absolute differences), so no stream-specific constant exists.
    A change between frame i-1 and i adds both frames, because both sides need an OCR reading.
    """
    if bands.ndim < 3 or bands.shape[0] < 2:
        raise ValueError("Expected at least two stacked band frames")
    flat = bands.reshape(bands.shape[0], -1).astype(np.float32)
    deltas = np.abs(np.diff(flat, axis=0)).mean(axis=1)
    median = float(np.median(deltas))
    spread = float(np.median(np.abs(deltas - median)))
    hits = np.flatnonzero(deltas > median + scale * spread)
    return tuple(sorted({int(i) for hit in hits for i in (hit, hit + 1)}))
