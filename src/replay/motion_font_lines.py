"""Bounded whole-line contexts for CoreText selection, not Motion raster proof."""

import unicodedata
from collections.abc import Iterable

from replay.caption_style_defaults import MAX_CAPTION_LINES
from replay.core import ReplayError

SCOPE = "coretext-per-line-v1"


def placement_lines(texts: Iterable[str]) -> tuple[str, ...]:
    """Preserve each actual line, including spaces and prefix context; never join tokens.

    NativeCaptionLayout and CaptionTextMeasurement measure newline-separated lines.
    FontIdentityProbe shapes CTLine, not a paragraph; Motion equivalence is unproven.
    """
    result: dict[str, None] = {}
    for text in texts:
        lines = text.split("\n")
        if not 1 <= len(lines) <= MAX_CAPTION_LINES:
            raise ReplayError("Font context exceeds the selected caption line limit")
        for line in lines:
            if (
                not line
                or len(line.encode("utf-16-le")) // 2 > 120
                or any(
                    unicodedata.category(character) in {"Cc", "Cf", "Zl", "Zp"}
                    for character in line
                )
            ):
                raise ReplayError("Font context contains unsupported line text")
            result[line] = None
    if (
        not result
        or len(result) > 128
        or sum(len(line.encode("utf-16-le")) // 2 for line in result) > 4096
    ):
        raise ReplayError("Font contexts exceed one bounded native request")
    return tuple(result)
