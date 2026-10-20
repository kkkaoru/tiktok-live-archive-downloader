"""Fixed-bottom placement uses canvas proportions only; no head observation."""

from collections.abc import Callable
from fractions import Fraction

import pytest

from replay import caption_fixed_bottom as fixed
from replay.caption_style_defaults import HEAD_CAPTION_REFERENCE_HEIGHT


def test_reference_canvas_slot() -> None:
    slot = fixed.bottom_slot(1920, 1080)
    assert slot.x == 960
    assert slot.bottom == pytest.approx(1080 - 24 * 1080 / HEAD_CAPTION_REFERENCE_HEIGHT)
    assert slot.max_width == pytest.approx(1920 - 2 * 24 * 1080 / HEAD_CAPTION_REFERENCE_HEIGHT)
    assert slot.max_lines == 3


def test_source_canvas_slot_scales_proportionally() -> None:
    slot = fixed.bottom_slot(1280, 720)
    assert slot.x == 640
    assert slot.bottom == pytest.approx(720 - 24 * 720 / HEAD_CAPTION_REFERENCE_HEIGHT)
    assert slot.max_width == pytest.approx(1280 - 2 * 24 * 720 / HEAD_CAPTION_REFERENCE_HEIGHT)


def test_appearance_is_style20_at_twice_youtube_size() -> None:
    appearance = fixed.FixedBottomAppearance()
    assert appearance.font_size == 104.0
    assert appearance.font_family == "Noto Sans JP"
    assert appearance.font_face == "Bold"
    assert appearance.gradient_top == "#97D8F7"
    assert appearance.gradient_bottom == "#2FB0EE"
    assert appearance.gradient_top_hold == 0.10
    assert appearance.gradient_bottom_start == 0.30
    assert appearance.white_stroke == pytest.approx(20.16 * 104 / 88)
    assert appearance.outer_black_stroke == pytest.approx(26.16 * 104 / 88)
    assert appearance.inner_black_stroke == pytest.approx(4.0 * 104 / 88)
    assert appearance.shadow_offset_y == pytest.approx(16.0 * 104 / 88)
    assert appearance.shadow_blur == pytest.approx(8.0 * 104 / 88)
    assert appearance.entrance_duration == Fraction(1, 15)
    assert appearance.entrance_start_y_scale == 0.96


def test_chat_aware_slot_clears_measured_overlay() -> None:
    slot = fixed.chat_aware_slot(1280, 720, exclude_left=497.0)
    assert slot.x == pytest.approx((497.0 + 1280 - 9.0) / 2)
    assert slot.bottom == pytest.approx(720 - 9.0)
    assert slot.max_width == pytest.approx(1280 - 9.0 - 497.0)
    assert slot.max_lines == 3


def test_chat_aware_slot_rejects_bad_exclusion() -> None:
    with pytest.raises(ValueError, match="[Ee]xclusion"):
        fixed.chat_aware_slot(1280, 720, exclude_left=-1.0)
    with pytest.raises(ValueError, match="no caption width"):
        fixed.chat_aware_slot(1280, 720, exclude_left=1271.0)


def test_non_16_by_9_and_bad_canvas_rejected() -> None:
    with pytest.raises(ValueError, match="16:9"):
        fixed.bottom_slot(1080, 1920)
    with pytest.raises(ValueError, match="16:9"):
        fixed.bottom_slot(1920, 1200)
    for width, height in [(0, 1080), (1920, 0), (-1920, 1080)]:
        with pytest.raises(ValueError, match="[Cc]anvas"):
            fixed.bottom_slot(width, height)
    unchecked: Callable[..., object] = fixed.bottom_slot
    with pytest.raises(ValueError, match="[Cc]anvas"):
        unchecked(1920.0, 1080)
