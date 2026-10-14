"""Wrap caption phrases to a measured safe width without new timings.

Uses the shared linguistic wrapper with an East-Asian-width advance estimate
at the render font size (full-width kana/kanji/CJK punctuation count one em;
half-width kana and Latin count half; other Latin counts six tenths, a
conservative overestimate so true renderer widths fit inside the wrapped
lines). No video-ID, date, duration or resolution branches; the caller
supplies the safe width in render pixels and the font size.
"""

from __future__ import annotations

import unicodedata
from typing import Final

from replay.caption_linguistics import CaptionLayoutError, wrap_measured_lines

LINE_HEIGHT_RATIO: Final[float] = 1.25
HALF_WIDTH_RATIO: Final[float] = 0.5
NARROW_RATIO: Final[float] = 0.6


def _advance_ratio(character: str) -> float:
    if unicodedata.east_asian_width(character) in ("W", "F", "A"):
        return 1.0
    if "\u3000" <= character <= "\u303f":
        return 1.0
    if unicodedata.east_asian_width(character) == "H":
        return HALF_WIDTH_RATIO
    return NARROW_RATIO


def measure_em(text: str, font_size: float) -> tuple[float, float]:
    """Estimate render size; overestimates narrow Latin so true widths fit.

    Multi-line values measure the widest line and stack line heights, so the
    shared wrapper can validate candidate layouts.
    """
    if type(font_size) is not float or not font_size > 0:
        raise ValueError("Font size must be a positive float")
    lines = text.split("\n")
    width = max(sum(_advance_ratio(character) for character in line) for line in lines) * font_size
    return width, font_size * LINE_HEIGHT_RATIO * len(lines)


def _char_split_lines(text: str, font_size: float, max_width_px: float) -> list[str]:
    """Greedily pack characters into lines within the safe width."""
    lines: list[str] = []
    current = ""
    current_width = 0.0
    for character in text:
        advance = _advance_ratio(character) * font_size
        if current and current_width + advance > max_width_px:
            lines.append(current)
            current = ""
            current_width = 0.0
        current += character
        current_width += advance
    if current:
        lines.append(current)
    return lines


def wrap_report(
    text: str, *, font_size: float = 104.0, max_width_px: float = 1161.0, max_lines: int = 3
) -> dict[str, object]:
    """Wrap a phrase to the safe width with a generic last-resort chain.

    Rules depend only on width, character count and character class: (1) the
    shared linguistic wrapper decides at phrase boundaries; (2) character
    boundaries greedily pack advances when the phrase has no usable phrase
    structure (repetitions, overlong single tokens); (3) a still-growing
    phrase beyond the line budget rolls its oldest characters out so the
    newest words stay inside the safe area. Every dropped prefix character
    is reported; clocks never change here.
    """
    if type(max_width_px) is not float or not max_width_px > 0:
        raise ValueError("Maximum width must be a positive float")
    width, height = measure_em(text, font_size)
    if width <= max_width_px and text.count("\n") + 1 <= max_lines:
        return {
            "text": text,
            "fallback": False,
            "reason": "already-fit",
            "dropped": "",
            "lines": [
                {"line": line, "width": measure_em(line, font_size)[0]} for line in text.split("\n")
            ],
        }
    try:
        wrapped = wrap_measured_lines(
            text.replace("\n", ""),
            measure=lambda value: measure_em(value, font_size),
            width=max_width_px,
            height=font_size * LINE_HEIGHT_RATIO * max_lines,
            maximum_lines=max_lines,
        )
        lines = wrapped.split("\n")
        return {
            "text": wrapped,
            "fallback": False,
            "reason": "linguistic",
            "dropped": "",
            "lines": [{"line": line, "width": measure_em(line, font_size)[0]} for line in lines],
        }
    except CaptionLayoutError:
        pass
    dropped = ""
    remaining = text.replace("\n", "")
    while True:
        lines = _char_split_lines(remaining, font_size, max_width_px)
        if 1 <= len(lines) <= max_lines and all(
            measure_em(line, font_size)[0] <= max_width_px for line in lines
        ):
            break
        if len(remaining) <= 1:
            raise CaptionLayoutError("Roll-up cannot fit a single character")
        # Roll-up: send the oldest characters out of the safe area.
        dropped += remaining[0]
        remaining = remaining[1:]
    return {
        "text": "\n".join(lines),
        "fallback": True,
        "reason": "character-boundary",
        "dropped": dropped,
        "lines": [{"line": line, "width": measure_em(line, font_size)[0]} for line in lines],
    }


def wrap_phrase(
    text: str, *, font_size: float = 104.0, max_width_px: float = 1161.0, max_lines: int = 3
) -> str:
    """Wrap one phrase to measured lines; never invents new display times."""
    report = wrap_report(text, font_size=font_size, max_width_px=max_width_px, max_lines=max_lines)
    return str(report["text"])
