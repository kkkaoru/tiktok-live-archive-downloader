"""Learn source-specific subtitle bands; retain sampled evidence and semantic limits."""

from collections import Counter, defaultdict
from dataclasses import dataclass
from difflib import SequenceMatcher
from math import floor

from .caption_io import FrameObservation
from .caption_masks import TextDetection

# A fixed on-screen label sits in one (x, width) cell and OCR only misreads it: at least half of a
# band's Japanese lines share a cell and most of those read as variants of one text (measured:
# labels 0.95-1.0, caption bands 0.05-0.33 of lines at least 0.75 similar to the commonest reading).
LABEL_CELL = 24
LABEL_CELL_SHARE = 0.5
LABEL_VARIANT_SHARE = 0.8
LABEL_SIMILARITY = 0.75


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
    detection. An OCR search ROI is not itself a mask. Repeated labels, including
    changing decimal counters/spacing, cannot seed subtitle masks. Roman text
    above the Japanese bands cannot seed them either; detected text is untouched.
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
    cells: dict[int, list[tuple[tuple[int, int], str]]] = defaultdict(list)
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
            # Counters changing within a fixed label are not lexical subtitle variety.
            text = "".join(
                character
                for character in line.text
                if not character.isdecimal() and not character.isspace()
            )
            counts[band] += 1
            if len(texts[band]) < 5:
                texts[band].add(text)
            if japanese_letters(line.text):
                jp_counts[band] += 1
                cells[band].append(
                    ((round(box.x / LABEL_CELL), round(box.width / LABEL_CELL)), text)
                )
                if len(jp_texts[band]) < 5:
                    jp_texts[band].add(text)
    labels = {band for band, lines in cells.items() if _fixed_label(lines)}
    primary = tuple(
        sorted(
            band
            for band in jp_counts
            if jp_counts[band] >= 10 and len(jp_texts[band]) >= 5 and band not in labels
        )
    )
    if not primary:
        raise ValueError("No variable Japanese subtitle bands; refuse a blanket mask")
    secondary = tuple(
        sorted(
            band
            for band in counts
            if band not in primary
            and band not in labels
            and counts[band] >= 10
            and len(texts[band]) >= 5
            and any(0 < band - source <= 6 for source in primary)
        )
    )
    return SubtitleBands(width, height, primary, secondary)


def _fixed_label(lines: list[tuple[tuple[int, int], str]]) -> bool:
    """Most lines in one geometry cell, read as variants of one text: a label, not captions."""
    cell, count = Counter(key for key, _ in lines).most_common(1)[0]
    if count < len(lines) * LABEL_CELL_SHARE:
        return False
    texts = [text for key, text in lines if key == cell]
    reading = Counter(texts).most_common(1)[0][0]
    variants = sum(
        SequenceMatcher(None, reading, text).ratio() >= LABEL_SIMILARITY for text in texts
    )
    return variants >= len(texts) * LABEL_VARIANT_SHARE


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
