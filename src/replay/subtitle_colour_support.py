"""Caption-colour supplement for learned glyph masks, limited to detected text lines.

The learned detector can miss small caption characters (``<< X >>`` markers, edge strokes).
Inside the original frame's recognised line boxes, caption-fill pixels are added to the glyph
mask; scene pixels of the same hue outside those boxes are never touched. One exception covers a
caption line that scrolls in or out of the band: it is clipped by the band edge, so the detector
cannot recognise it. A caption-coloured component that touches the band's top or bottom row, is
no taller than one recognised line and lines up horizontally with a recognised box is such a
clipped line and is keyed as well. The supplement is a mask input only: it proves neither text
absence nor repair quality.
"""

from collections.abc import Callable, Sequence
from dataclasses import dataclass

import cv2
import numpy as np
from numpy.typing import NDArray

from replay.subtitle_inpaint import Image

Box = tuple[int, int, int, int]
Detector = Callable[[Sequence[Image]], tuple[Image, ...]]
Components = tuple[int, NDArray[np.int32], NDArray[np.int32]]  # count, labels, stats

HUE_RANGE = (150, 172)
MIN_SATURATION = 90
MIN_VALUE = 170
BOX_MARGIN = 10
OUTLINE_KERNEL = 5
MINIMUM_ALIGNED_SHARE = 0.5


@dataclass(frozen=True, slots=True)
class FrameBoxes:
    """One active frame: ROI origin on the canvas and recognised line boxes in canvas pixels."""

    origin: tuple[int, int]
    boxes: tuple[Box, ...]


def pink(image: Image) -> Image:
    """Boolean-valued uint8 (0/1) mask of caption-fill pixels of a BGR image."""
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    hue, saturation, value = hsv[..., 0], hsv[..., 1], hsv[..., 2]
    inside = (hue >= HUE_RANGE[0]) & (hue <= HUE_RANGE[1])
    return (inside & (saturation >= MIN_SATURATION) & (value >= MIN_VALUE)).astype(np.uint8)


def box_mask(shape: tuple[int, int], origin: tuple[int, int], boxes: Sequence[Box]) -> Image:
    """Margin-widened line boxes as a 0/1 mask over an ROI of ``shape`` (height, width)."""
    mask = np.zeros(shape, dtype=np.uint8)
    for left, top, right, bottom in boxes:
        x0, y0 = max(0, left - origin[0] - BOX_MARGIN), max(0, top - origin[1] - BOX_MARGIN)
        x1, y1 = right - origin[0] + BOX_MARGIN, bottom - origin[1] + BOX_MARGIN
        mask[y0 : max(y0, y1), x0 : max(x0, x1)] = 1
    return mask


def components(mask: Image) -> Components:
    """8-connected components of a 0/1 caption-colour mask: count, labels and stats."""
    count, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
    return int(count), np.asarray(labels, dtype=np.int32), np.asarray(stats, dtype=np.int32)


def clipped_lines(roi: Image, frame: FrameBoxes, found: Components | None = None) -> Image:
    """0/1 mask of caption-coloured components clipped by the band edge beside a recognised line.

    A component qualifies when it touches the top or bottom row of ``roi``, is no taller than the
    tallest recognised line and at least half of its width lies within the horizontal span of one
    recognised box (widened by the box margin).
    """
    height, width = roi.shape[:2]
    result = np.zeros((height, width), dtype=np.uint8)
    if not frame.boxes:
        return result
    line_height = max(bottom - top for _, top, _, bottom in frame.boxes)
    spans = [
        (max(0, left - frame.origin[0] - BOX_MARGIN), right - frame.origin[0] + BOX_MARGIN)
        for left, _, right, _ in frame.boxes
    ]
    count, labels, stats = components(pink(roi)) if found is None else found
    keep = np.zeros(count, dtype=np.uint8)
    for label in range(1, count):
        x, y, w, h = (int(stats[label, k]) for k in range(4))
        clipped = y == 0 or y + h == height
        aligned = max(max(0, min(x + w, end) - max(x, start)) for start, end in spans)
        if clipped and h <= line_height and aligned >= MINIMUM_ALIGNED_SHARE * w:
            keep[label] = 1
    # One lookup over the labels instead of a full-ROI comparison per kept component.
    return np.asarray(keep[labels], dtype=np.uint8)


def crossing_lines(roi: Image, frame: FrameBoxes, found: Components | None = None) -> Image:
    """0/1 mask of caption-coloured components that cross out of the recognised line boxes.

    A scrolling line is only partly recognised: the detector reads the part inside its zone and
    the rest of the glyphs lie just above or below the boxes. A caption-coloured component that
    overlaps a recognised box is keyed in full when it stays within one recognised line height
    above the topmost box and below the lowest one; scene pixels farther away or not connected
    to a recognised line are never added.
    """
    height, width = roi.shape[:2]
    result = np.zeros((height, width), dtype=np.uint8)
    if not frame.boxes:
        return result
    line_height = max(bottom - top for _, top, _, bottom in frame.boxes)
    top = max(0, min(box[1] for box in frame.boxes) - frame.origin[1] - line_height)
    bottom = max(top, max(box[3] for box in frame.boxes) - frame.origin[1] + line_height)
    inside = box_mask(roi.shape[:2], frame.origin, frame.boxes)
    count, labels, _ = components(pink(roi)) if found is None else found
    # Per component, in one pass each: does it touch a box, and does it reach the rows outside
    # one line height around the boxes (instead of a full-ROI comparison per component).
    touches = np.bincount(labels.ravel(), weights=inside.ravel(), minlength=count) > 0
    escapes = np.zeros(count, dtype=bool)
    escapes[labels[:top]] = True
    escapes[labels[bottom:]] = True
    keep = touches & ~escapes
    keep[0] = False
    return np.asarray(keep[labels], dtype=np.uint8)


def key_mask(roi: Image, frame: FrameBoxes) -> Image:
    """Caption-coloured pixels (plus their outline) inside the frame's line boxes, as 0/255."""
    kernel = np.ones((OUTLINE_KERNEL, OUTLINE_KERNEL), np.uint8)
    pinked = pink(roi)  # one colour conversion and one labelling serve every step below
    found = components(pinked)
    edge = np.maximum(clipped_lines(roi, frame, found), crossing_lines(roi, frame, found))
    allowed = np.maximum(
        box_mask(roi.shape[:2], frame.origin, frame.boxes),
        np.asarray(cv2.dilate(edge, kernel), dtype=np.uint8),
    )
    fill = np.minimum(pinked, allowed)
    grown = np.minimum(np.asarray(cv2.dilate(fill, kernel), dtype=np.uint8), allowed)
    return np.asarray(grown * 255, dtype=np.uint8)


class ColourKeyDetector:
    """Wrap a batch detector, adding colour-keyed pixels for each active frame in order.

    The renderer calls the detector sequentially over the active frames, so the i-th ROI seen
    belongs to ``frames[i]``. A mismatch in count fails instead of keying the wrong frame.
    """

    def __init__(self, inner: Detector, frames: Sequence[FrameBoxes]) -> None:
        self._inner = inner
        self._frames = frames
        self._next = 0

    @property
    def consumed(self) -> int:
        return self._next

    def detect(self, rois: Sequence[Image]) -> tuple[Image, ...]:
        base = self._inner(rois)
        if self._next + len(rois) > len(self._frames):
            raise ValueError("More ROIs than planned active frames")
        merged = tuple(
            np.maximum(mask, key_mask(roi, frame))
            for mask, roi, frame in zip(
                base, rois, self._frames[self._next : self._next + len(rois)], strict=True
            )
        )
        self._next += len(rois)
        return merged
