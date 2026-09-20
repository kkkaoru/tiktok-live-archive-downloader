import math

import pytest

from replay.caption_masks import MaskPolicy, TextDetection, TextRectangle, subtitle_rectangles


def test_multiline_subtitles_above_old_band_are_covered() -> None:
    result = subtitle_rectangles(
        (
            TextDetection("前の行", 0.8, TextRectangle(0, 1260, 1080, 20)),
            TextDetection("現在の字幕", 0.9, TextRectangle(0, 1295, 1080, 30)),
            TextDetection("続きです", 0.9, TextRectangle(380, 1340, 260, 30)),
        ),
        width=1080,
        height=1920,
    )
    assert result == (TextRectangle(0, 1230, 1080, 185),)


def test_search_area_excludes_top_ui_without_narrow_fixed_band() -> None:
    assert subtitle_rectangles(
        (
            TextDetection("フォローしてね", 1, TextRectangle(100, 590, 200, 30)),
            TextDetection("字幕", 0.9, TextRectangle(200, 1500, 200, 30)),
            TextDetection("99%", 1, TextRectangle(100, 1000, 100, 30)),
            TextDetection("曖昧", 0.2, TextRectangle(100, 1200, 100, 30)),
        ),
        width=1080,
        height=1920,
    ) == (TextRectangle(176, 1455, 248, 120),)


def test_configuration_can_find_top_subtitles_without_changing_source_code() -> None:
    assert subtitle_rectangles(
        (TextDetection("上の字幕", 1, TextRectangle(5, 5, 100, 20)),),
        width=1080,
        height=1920,
        policy=MaskPolicy(search_top=0, search_bottom=0.4),
    ) == (TextRectangle(0, 0, 129, 55),)


def test_padding_clamps_to_canvas_and_disjoint_boxes_stay_local() -> None:
    assert subtitle_rectangles(
        (
            TextDetection("左", 1, TextRectangle(0, 60, 10, 10)),
            TextDetection("右", 1, TextRectangle(90, 90, 10, 10)),
        ),
        width=100,
        height=100,
        policy=MaskPolicy(search_bottom=1, horizontal_padding=2, vertical_line_padding=1),
    ) == (TextRectangle(0, 50, 12, 30), TextRectangle(88, 80, 12, 20))
    assert subtitle_rectangles((), width=100, height=100) == ()


def test_transitive_intersections_merge_regardless_of_input_order() -> None:
    assert subtitle_rectangles(
        (
            TextDetection("左", 1, TextRectangle(0, 50, 20, 10)),
            TextDetection("右", 1, TextRectangle(40, 50, 20, 10)),
            TextDetection("橋", 1, TextRectangle(20, 50, 20, 10)),
        ),
        width=100,
        height=100,
        policy=MaskPolicy(horizontal_padding=0, vertical_line_padding=0),
    ) == (TextRectangle(0, 50, 60, 10),)


@pytest.mark.parametrize(
    ("x", "y", "width", "height"),
    [(math.nan, 0, 1, 1), (0, 0, math.inf, 1), (-1, 0, 1, 1), (0, 0, 0, 1)],
)
def test_invalid_rectangles_are_refused(x: float, y: float, width: float, height: float) -> None:
    with pytest.raises(ValueError):
        TextRectangle(x, y, width, height)


@pytest.mark.parametrize("confidence", [math.nan, -0.1, 1.1])
def test_invalid_confidence_is_refused(confidence: float) -> None:
    with pytest.raises(ValueError):
        TextDetection("字幕", confidence, TextRectangle(0, 0, 1, 1))


def test_ocr_budgets_and_canvas_mismatch_are_refused() -> None:
    with pytest.raises(ValueError, match="text budget"):
        TextDetection("あ" * 8193, 1, TextRectangle(0, 0, 1, 1))
    with pytest.raises(ValueError, match="line budget"):
        subtitle_rectangles(
            (TextDetection("字幕", 1, TextRectangle(0, 0, 1, 1)),) * 65,
            width=100,
            height=100,
        )
    with pytest.raises(ValueError, match="canvas"):
        subtitle_rectangles(
            (TextDetection("字幕", 1, TextRectangle(99, 99, 2, 2)),), width=100, height=100
        )
    with pytest.raises(ValueError, match="canvas"):
        subtitle_rectangles((), width=8192, height=8192)


@pytest.mark.parametrize(
    ("top", "bottom", "confidence", "vertical", "horizontal"),
    [
        (0.9, 0.1, 0.3, 1.5, 24),
        (0, 1, math.nan, 1.5, 24),
        (0, 1, 0.3, 3, 24),
        (0, 1, 0.3, 1.5, 101),
    ],
)
def test_invalid_mask_policy_is_refused(
    top: float, bottom: float, confidence: float, vertical: float, horizontal: int
) -> None:
    with pytest.raises(ValueError):
        MaskPolicy(top, bottom, confidence, vertical, horizontal)
