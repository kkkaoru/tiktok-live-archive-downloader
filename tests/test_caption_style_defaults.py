"""Pin the selected preview's colour intent and reference geometry."""

import pytest

from replay import caption_style_defaults as style


def test_selected_variable_font_identity() -> None:
    assert style.FONT_FILE_NAME == "NotoSansJP-VariableFont_wght.ttf"
    assert style.FONT_FAMILY == "Noto Sans JP"
    assert style.FONT_FACE == "Bold"
    assert style.FONT_POSTSCRIPT_NAME == "NotoSansJP-Thin_Bold"
    assert style.FONT_WEIGHT_AXIS == 2003265652
    assert style.FONT_WEIGHT == 700.0


def test_head_gap_is_explicit_shared_editing_intent() -> None:
    assert style.HEAD_CAPTION_GAP_PX == 24.0


def test_selected_colour_and_gradient() -> None:
    assert style.DEFAULT_CAPTION_STYLE_ID == 20
    assert style.COLOR_SPACE == "sRGB"
    assert style.GRADIENT_TOP_HEX == "#2FB0EE"
    assert style.GRADIENT_BOTTOM_HEX == "#E3F6FF"
    assert style.GRADIENT_DARK_HOLD == 0.75
    assert style.GRADIENT_DIRECTION == "top_to_bottom"
    assert style.GRADIENT_INTERPOLATION == "sRGB"


def test_reference_stroke_geometry() -> None:
    assert style.REFERENCE_FONT_SIZE == 88.0
    assert style.WHITE_HEX == "#FFFFFF"
    assert style.BLACK_HEX == "#000000"
    assert style.WHITE_STROKE_WIDTH == 28.8
    assert pytest.approx(36.0 * 0.8) == style.WHITE_STROKE_WIDTH
    assert style.OUTER_BLACK_STROKE_WIDTH == 34.8
    assert style.INNER_BLACK_STROKE_WIDTH == 4.0
    assert style.STROKE_LINE_JOIN == "round"
    assert pytest.approx(12.4) == (style.WHITE_STROKE_WIDTH - style.INNER_BLACK_STROKE_WIDTH) / 2
    assert pytest.approx(3.0) == (style.OUTER_BLACK_STROKE_WIDTH - style.WHITE_STROKE_WIDTH) / 2


def test_reference_shadow() -> None:
    assert style.SHADOW_COLOR_HEX == "#000000"
    assert style.SHADOW_OFFSET_X == 0.0
    assert style.SHADOW_OFFSET_Y == 7.0
    assert style.SHADOW_BLUR_STD_DEVIATION == 4.0
    assert style.SHADOW_OPACITY == 0.6
