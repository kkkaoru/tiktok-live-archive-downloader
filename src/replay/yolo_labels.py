"""YOLO label geometry and time-block splits for the avatar-face / source-caption detector.

Pure helpers only: pixel rectangles become normalized YOLO rows, rows are validated against the
canvas, and frames are split into train/validation sets by contiguous time blocks so that two
neighbouring frames of one moment never land on both sides of the split. Nothing here reads media,
Vision output or OCR output; callers hand in already verified rectangles.
"""

from collections.abc import Sequence
from dataclasses import dataclass
from math import isfinite

from replay.core import ReplayError

FACE = 0
CAPTION = 1
CLASS_NAMES = ("avatar_face", "source_caption")
MAX_FRAMES = 10_000_000


@dataclass(frozen=True, slots=True)
class PixelBox:
    """A rectangle in canvas pixels: left/top origin, positive width and height."""

    x: float
    y: float
    width: float
    height: float


def yolo_row(label: int, box: PixelBox, *, canvas: tuple[int, int]) -> str:
    """One normalized `class cx cy w h` row; the box must lie wholly inside the canvas."""
    width, height = canvas
    values = (box.x, box.y, box.width, box.height)
    if label not in (FACE, CAPTION) or width <= 0 or height <= 0:
        raise ReplayError("Invalid YOLO label or canvas")
    if not all(isfinite(value) for value in values) or box.width <= 0 or box.height <= 0:
        raise ReplayError("Invalid YOLO box")
    if box.x < 0 or box.y < 0 or box.x + box.width > width or box.y + box.height > height:
        raise ReplayError("YOLO box leaves the canvas")
    center_x = (box.x + box.width / 2) / width
    center_y = (box.y + box.height / 2) / height
    return (
        f"{label} {center_x:.6f} {center_y:.6f} {box.width / width:.6f} {box.height / height:.6f}"
    )


def clip_to_canvas(box: PixelBox, *, canvas: tuple[int, int]) -> PixelBox | None:
    """Intersect a box with the canvas; None when nothing visible remains."""
    width, height = canvas
    left, top = max(box.x, 0.0), max(box.y, 0.0)
    right, bottom = min(box.x + box.width, float(width)), min(box.y + box.height, float(height))
    if right <= left or bottom <= top:
        return None
    return PixelBox(left, top, right - left, bottom - top)


def line_box(corners: Sequence[float]) -> PixelBox:
    """A `[left, top, right, bottom]` line box (the cleanup admission format) as a PixelBox."""
    if len(corners) != 4:
        raise ReplayError("Expected left, top, right, bottom")
    left, top, right, bottom = corners
    return PixelBox(float(left), float(top), float(right) - float(left), float(bottom) - float(top))


def split_by_time_blocks(
    frames: int, *, block: int, validation_every: int, gap: int
) -> tuple[tuple[int, ...], tuple[int, ...]]:
    """Frame indices for (train, validation) with a guard gap between them.

    Every `validation_every`-th block of `block` frames is validation. Training frames within `gap`
    frames of any validation frame are dropped, so a validation frame is always at least `gap + 1`
    frames away from every training frame and near-duplicate neighbours cannot leak across.
    """
    if not 0 < frames <= MAX_FRAMES or block < 1 or validation_every < 2 or gap < 0:
        raise ReplayError("Invalid time-block split")
    validation = [
        index
        for index in range(frames)
        if (index // block) % validation_every == validation_every - 1
    ]
    held = set(validation)
    guarded = {
        near for index in validation for near in range(index - gap, index + gap + 1) if near >= 0
    }
    train = [index for index in range(frames) if index not in held and index not in guarded]
    return tuple(train), tuple(validation)
