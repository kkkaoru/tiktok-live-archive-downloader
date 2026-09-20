"""Learn source-specific subtitle bands; retain sampled evidence and semantic limits."""

from collections import defaultdict
from dataclasses import dataclass
from math import floor

from .caption_io import FrameObservation
from .caption_masks import TextDetection


@dataclass(frozen=True, slots=True)
class SubtitleBands:
    width: int
    height: int
    primary: tuple[int, ...]
    secondary: tuple[int, ...]


def japanese_letters(text: str) -> bool:
    return (
        sum(
            "ぁ" <= character <= "ゖ" or "ァ" <= character <= "ヺ" or "一" <= character <= "鿿"
            for character in text
        )
        >= 2
    )


def _band(line: TextDetection) -> int:
    return floor((line.rectangle.y + line.rectangle.height / 2) / 32)


def _eligible(line: TextDetection, *, width: int, height: int) -> bool:
    box = line.rectangle
    if box.x + box.width > width + 0.001 or box.y + box.height > height + 0.001:
        raise ValueError("Profile detection exceeds canvas")
    return (
        line.confidence >= 0.3
        and 16 <= box.height <= 96
        and any(character.isalpha() for character in line.text)
    )


def learn_subtitle_bands(frames: tuple[FrameObservation, ...]) -> SubtitleBands:
    """Seed variable centered Japanese lines and nearby lower translated lines.

    This is a geometry/script heuristic, not semantic or exhaustive subtitle
    detection. An OCR search ROI is not itself a mask. Repeated static labels and
    variable Roman text above the Japanese bands cannot seed subtitle masks.
    """
    if not frames or len(frames) > 50000:
        raise ValueError("Invalid subtitle profile observation budget")
    width, height = frames[0].width, frames[0].height
    if not 1 <= width <= 8192 or not 1 <= height <= 8192:
        raise ValueError("Invalid profile canvas")
    counts: dict[int, int] = defaultdict(int)
    texts: dict[int, set[str]] = defaultdict(set)
    jp_counts: dict[int, int] = defaultdict(int)
    jp_texts: dict[int, set[str]] = defaultdict(set)
    for frame in frames:
        if (frame.width, frame.height) != (width, height) or len(frame.detections) > 64:
            raise ValueError("Inconsistent profile observations")
        for line in frame.detections:
            eligible = _eligible(line, width=width, height=height)
            box = line.rectangle
            if (
                not eligible
                or box.width < width / 10
                or box.x >= width * 0.55
                or box.x + box.width <= width * 0.45
            ):
                continue
            band = _band(line)
            counts[band] += 1
            if len(texts[band]) < 5:
                texts[band].add(line.text)
            if japanese_letters(line.text):
                jp_counts[band] += 1
                if len(jp_texts[band]) < 5:
                    jp_texts[band].add(line.text)
    primary = tuple(
        sorted(band for band in jp_counts if jp_counts[band] >= 10 and len(jp_texts[band]) >= 5)
    )
    if not primary:
        raise ValueError("No variable Japanese subtitle bands; refuse a blanket mask")
    secondary = tuple(
        sorted(
            band
            for band in counts
            if band not in primary
            and counts[band] >= 10
            and len(texts[band]) >= 5
            and any(0 < band - source <= 6 for source in primary)
        )
    )
    return SubtitleBands(width, height, primary, secondary)


def _linked(left: TextDetection, right: TextDetection) -> bool:
    first, second = left.rectangle, right.rectangle
    vertical = abs(first.y + first.height / 2 - second.y - second.height / 2)
    horizontal = abs(first.x + first.width / 2 - second.x - second.width / 2)
    return 0 < vertical <= max(first.height, second.height) * 1.6 and horizontal <= max(
        80, min(first.width, second.width) / 4
    )


def profiled_detections(
    frame: FrameObservation, profile: SubtitleBands
) -> tuple[TextDetection, ...]:
    """Unusual upper/lower lines require a concurrent, close caption-line chain."""
    if (frame.width, frame.height) != (profile.width, profile.height) or len(frame.detections) > 64:
        raise ValueError("Frame differs from learned profile")
    bands = set(profile.primary + profile.secondary)
    if not bands:
        raise ValueError("Empty subtitle band profile")
    candidates = tuple(
        line for line in frame.detections if _eligible(line, width=frame.width, height=frame.height)
    )
    accepted = [line for line in candidates if _band(line) in bands]
    pending = [
        line
        for line in candidates
        if min(bands) - 1 <= _band(line) <= max(bands) + 2 and line not in accepted
    ]
    while pending:
        additions = [line for line in pending if any(_linked(line, anchor) for anchor in accepted)]
        if not additions:
            break
        accepted.extend(additions)
        pending = [line for line in pending if line not in additions]
    return tuple(line for line in candidates if line in accepted)
