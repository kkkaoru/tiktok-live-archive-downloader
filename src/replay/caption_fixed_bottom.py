"""Fixed-bottom caption style and placement for horizontal 16:9 video.

This is the shared, video-agnostic companion to the head-tracked TikTok path:
same style-20 appearance at twice the YouTube single size, same linguistic
segmentation upstream, but the caption sits at a fixed bottom-centre slot and
never follows a face, so no Vision observation is required. All geometry
derives from the canvas and caption_style_defaults; there are no video-ID,
date, duration or resolution branches.
"""

from __future__ import annotations

from dataclasses import dataclass
from fractions import Fraction
from typing import Final

from replay.caption_style_defaults import (
    CAPTION_SCREEN_INSET_PX,
    ENTRANCE_DURATION_DENOMINATOR,
    ENTRANCE_DURATION_NUMERATOR,
    ENTRANCE_START_Y_SCALE,
    FONT_FACE,
    FONT_FAMILY,
    GRADIENT_BOTTOM_HEX,
    GRADIENT_BOTTOM_START,
    GRADIENT_DIRECTION,
    GRADIENT_TOP_HEX,
    GRADIENT_TOP_HOLD,
    HEAD_CAPTION_REFERENCE_HEIGHT,
    INNER_BLACK_STROKE_WIDTH,
    MAX_CAPTION_LINES,
    OUTER_BLACK_STROKE_WIDTH,
    REFERENCE_FONT_SIZE,
    SHADOW_BLUR_STD_DEVIATION,
    SHADOW_OFFSET_X,
    SHADOW_OFFSET_Y,
    WHITE_STROKE_WIDTH,
)

# YouTube single size is 52 at PlayRes 1920x1080; the fixed-bottom style uses
# twice that, expressed against the style-20 reference font size.
FIXED_BOTTOM_FONT_SIZE: Final[float] = 104.0
FONT_SCALE: Final[float] = FIXED_BOTTOM_FONT_SIZE / REFERENCE_FONT_SIZE

MAX_LINES: Final[int] = MAX_CAPTION_LINES


@dataclass(frozen=True)
class FixedBottomAppearance:
    """Style-20 appearance scaled to the fixed-bottom font size."""

    font_family: str = FONT_FAMILY
    font_face: str = FONT_FACE
    font_size: float = FIXED_BOTTOM_FONT_SIZE
    gradient_top: str = GRADIENT_TOP_HEX
    gradient_bottom: str = GRADIENT_BOTTOM_HEX
    gradient_top_hold: float = GRADIENT_TOP_HOLD
    gradient_bottom_start: float = GRADIENT_BOTTOM_START
    gradient_direction: str = GRADIENT_DIRECTION
    white_stroke: float = WHITE_STROKE_WIDTH * FONT_SCALE
    outer_black_stroke: float = OUTER_BLACK_STROKE_WIDTH * FONT_SCALE
    inner_black_stroke: float = INNER_BLACK_STROKE_WIDTH * FONT_SCALE
    shadow_offset_x: float = SHADOW_OFFSET_X * FONT_SCALE
    shadow_offset_y: float = SHADOW_OFFSET_Y * FONT_SCALE
    shadow_blur: float = SHADOW_BLUR_STD_DEVIATION * FONT_SCALE
    entrance_start_y_scale: float = ENTRANCE_START_Y_SCALE
    entrance_duration: Fraction = Fraction(
        ENTRANCE_DURATION_NUMERATOR, ENTRANCE_DURATION_DENOMINATOR
    )


@dataclass(frozen=True)
class FixedBottomSlot:
    """Bottom slot on a 16:9 canvas; x is the centre, bottom the baseline."""

    x: float
    bottom: float
    max_width: float
    max_lines: int = MAX_LINES


def bottom_slot(canvas_width: int, canvas_height: int) -> FixedBottomSlot:
    """Fixed bottom-centre slot scaled proportionally from the reference canvas."""
    scale = canvas_height / HEAD_CAPTION_REFERENCE_HEIGHT
    return _slot(canvas_width, canvas_height, exclude_left=CAPTION_SCREEN_INSET_PX * scale)


def chat_aware_slot(
    canvas_width: int, canvas_height: int, *, exclude_left: float
) -> FixedBottomSlot:
    """Bottom slot clearing a measured left-column overlay (chat/panel).

    exclude_left is the measured overlay right edge plus margin, in canvas
    pixels; the caption block centres in [exclude_left, width - inset].
    No video-ID, date, duration or resolution branches.
    """
    return _slot(canvas_width, canvas_height, exclude_left=exclude_left)


def _slot(canvas_width: int, canvas_height: int, *, exclude_left: float) -> FixedBottomSlot:
    if (
        type(canvas_width) is not int
        or type(canvas_height) is not int
        or canvas_width <= 0
        or canvas_height <= 0
    ):
        raise ValueError("Canvas must be positive integer pixels")
    if canvas_width * 9 != canvas_height * 16:
        raise ValueError("Fixed-bottom placement requires a 16:9 canvas")
    if not isinstance(exclude_left, (int, float)) or exclude_left < 0:
        raise ValueError("Overlay exclusion must be a nonnegative width")
    scale = canvas_height / HEAD_CAPTION_REFERENCE_HEIGHT
    inset = CAPTION_SCREEN_INSET_PX * scale
    right = canvas_width - inset
    if not exclude_left + inset < right:
        raise ValueError("Overlay exclusion leaves no caption width")
    return FixedBottomSlot(
        x=(exclude_left + right) / 2,
        bottom=canvas_height - inset,
        max_width=right - exclude_left,
    )
