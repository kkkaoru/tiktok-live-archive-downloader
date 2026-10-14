"""Lossless line-boundary legality, independent of morphology and recognition time.

This filter never invents a split: callers supply POS/recognition-supported
candidates. Rejected punctuation stays in the original intact candidate.
"""

from dataclasses import dataclass
from itertools import pairwise
from unicodedata import category

from replay.caption_linguistics import PROHIBITED_LINE_END, PROHIBITED_LINE_START

MAX_CHARACTERS = 8192
MAX_LINES = 64


@dataclass(frozen=True, slots=True)
class LineLegality:
    """A candidate judgment, never a rewritten text or a quality approval."""

    legal: bool
    reason: str


def assess_lines(text: str, lines: tuple[str, ...]) -> LineLegality:
    """Preserve every character; reject only invalid proposed line boundaries.

    A whole punctuation-only utterance remains valid and visible. Punctuation
    cannot become a separate line when splitting a larger utterance. Symbols
    such as emoji are not silently classified as punctuation or discarded.
    """
    if (
        not text
        or len(text) > MAX_CHARACTERS
        or any(c in text for c in "\n\r\0")
        or not 1 <= len(lines) <= MAX_LINES
        or "".join(lines) != text
    ):
        raise ValueError("Invalid or nonconserving caption line candidate")
    if any(not line.strip() for line in lines):
        return LineLegality(False, "empty-line")
    if len(lines) == 1:
        return LineLegality(True, "intact-text")
    if any(
        all(category(c).startswith(("P", "M", "Z")) or c.isspace() for c in line) for line in lines
    ):
        return LineLegality(False, "punctuation-only-line")
    for left, right in pairwise(lines):
        first, last = right.lstrip()[0], left.rstrip()[-1]
        if first in PROHIBITED_LINE_START or category(first) in {"Pe", "Pf", "Mn", "Mc", "Me"}:
            return LineLegality(False, "prohibited-line-start")
        if last in PROHIBITED_LINE_END or category(last) in {"Ps", "Pi"}:
            return LineLegality(False, "prohibited-line-end")
    return LineLegality(True, "legal-boundaries")
