"""Local OCR-derived subtitle masks; sampling is evidence, not exhaustive detection."""

from dataclasses import dataclass
from math import ceil, floor, isfinite


@dataclass(frozen=True, slots=True)
class TextRectangle:
    x: float
    y: float
    width: float
    height: float

    def __post_init__(self) -> None:
        if not all(isfinite(value) for value in (self.x, self.y, self.width, self.height)):
            raise ValueError("Non-finite OCR rectangle")
        if min(self.x, self.y) < 0 or min(self.width, self.height) <= 0:
            raise ValueError("Invalid OCR rectangle")


@dataclass(frozen=True, slots=True)
class TextDetection:
    text: str
    confidence: float
    rectangle: TextRectangle

    def __post_init__(self) -> None:
        if not isfinite(self.confidence) or not 0 <= self.confidence <= 1:
            raise ValueError("Invalid OCR confidence")
        if len(self.text) > 8192:
            raise ValueError("OCR text budget exceeded")


@dataclass(frozen=True, slots=True)
class MaskPolicy:
    search_top: float = 0.45
    search_bottom: float = 0.95
    minimum_confidence: float = 0.3
    vertical_line_padding: float = 1.5
    horizontal_padding: int = 24

    def __post_init__(self) -> None:
        if not 0 <= self.search_top < self.search_bottom <= 1:
            raise ValueError("Invalid normalized subtitle search area")
        if not 0 <= self.minimum_confidence <= 1 or not 0 <= self.vertical_line_padding <= 2:
            raise ValueError("Invalid subtitle confidence/padding policy")
        if not 0 <= self.horizontal_padding <= 100:
            raise ValueError("Invalid subtitle horizontal padding")


def subtitle_rectangles(
    detections: tuple[TextDetection, ...],
    *,
    width: int,
    height: int,
    policy: MaskPolicy | None = None,
) -> tuple[TextRectangle, ...]:
    """Locate full lines, including spill beyond any previous fixed blur band.

    The configurable broad search area is an explicit heuristic. Numeric-only UI
    counters are excluded; other text inside the area remains a candidate. Padding
    includes clipped/scrolling neighboring lines. This does not identify semantic
    subtitles universally or establish coverage between sampled frames.
    """
    if not 1 <= width <= 8192 or not 1 <= height <= 8192 or width * height > 16_000_000:
        raise ValueError("Invalid mask canvas")
    if len(detections) > 64:
        raise ValueError("OCR line budget exceeded")
    selected = MaskPolicy() if policy is None else policy
    rectangles: list[TextRectangle] = []
    for detection in detections:
        box = detection.rectangle
        if box.x + box.width > width + 0.001 or box.y + box.height > height + 0.001:
            raise ValueError("OCR rectangle exceeds its declared canvas")
        center = (box.y + box.height / 2) / height
        if not selected.search_top <= center <= selected.search_bottom:
            continue
        if detection.confidence < selected.minimum_confidence:
            continue
        if not any(character.isalpha() for character in detection.text):
            continue
        vertical = ceil(box.height * selected.vertical_line_padding)
        left = max(0, floor(box.x) - selected.horizontal_padding)
        top = max(0, floor(box.y) - vertical)
        right = min(width, ceil(box.x + box.width) + selected.horizontal_padding)
        bottom = min(height, ceil(box.y + box.height) + vertical)
        rectangles.append(TextRectangle(left, top, right - left, bottom - top))
    return _merge_rectangles(rectangles)


def _merge_rectangles(rectangles: list[TextRectangle]) -> tuple[TextRectangle, ...]:
    merged: list[TextRectangle] = []
    for initial in rectangles:
        current = initial
        index = 0
        while index < len(merged):
            other = merged[index]
            if (
                current.x <= other.x + other.width
                and other.x <= current.x + current.width
                and current.y <= other.y + other.height
                and other.y <= current.y + current.height
            ):
                left, top = min(current.x, other.x), min(current.y, other.y)
                right = max(current.x + current.width, other.x + other.width)
                bottom = max(current.y + current.height, other.y + other.height)
                current = TextRectangle(left, top, right - left, bottom - top)
                merged.pop(index)
                index = 0
            else:
                index += 1
        merged.append(current)
    return tuple(sorted(merged, key=lambda rectangle: (rectangle.y, rectangle.x)))
