import json
from pathlib import Path

import pytest

from replay.caption_projection import CutSpan
from replay.caption_timing import CaptionCue
from replay.core import ReplayError
from replay.vertical_captions import (
    STYLE_RECORD,
    HeadSample,
    PlacedCaption,
    display_text,
    growth_runs,
    head_sample,
    highest_head,
    layout,
    load_head_track,
    place,
    render_ass,
    settle,
    source_seconds,
    top_limit,
)

CUTS = (CutSpan(0, 300, 0), CutSpan(600, 900, 300))


def test_face_box_wins_and_figure_box_is_the_fallback() -> None:
    face = {"t": 1, "faceTop": 600, "faceHeight": 100, "bodyTop": 300}
    assert head_sample(face) == HeadSample(1.0, 520.0)
    assert head_sample({"t": 2, "faceTop": None, "bodyTop": 450}) == HeadSample(2.0, 450.0)
    assert head_sample({"t": 3}) == HeadSample(3.0, None)
    with pytest.raises(ReplayError, match="time"):
        head_sample({"t": -1})
    with pytest.raises(ReplayError, match="face top"):
        head_sample({"t": 1, "faceTop": True})


def test_head_track_file_validation(tmp_path: Path) -> None:
    path = tmp_path / "track.json"
    path.write_text(json.dumps({"frame": [1080, 1920], "samples": [{"t": 0}, {"t": 0.5}]}))
    assert load_head_track(path) == (HeadSample(0.0, None), HeadSample(0.5, None))
    path.write_text(json.dumps({"frame": [1920, 1080], "samples": [{"t": 0}]}))
    with pytest.raises(ReplayError, match="vertical output frame"):
        load_head_track(path)
    path.write_text(json.dumps({"frame": [1080, 1920], "samples": []}))
    with pytest.raises(ReplayError, match="bounded sample list"):
        load_head_track(path)
    path.write_text(json.dumps({"frame": [1080, 1920], "samples": [{"t": 1}, {"t": 1}]}))
    with pytest.raises(ReplayError, match="not ordered"):
        load_head_track(path)


def test_output_time_maps_back_over_removed_source() -> None:
    assert source_seconds(5.0, CUTS) == 5.0
    assert source_seconds(12.0, CUTS) == 22.0
    assert source_seconds(25.0, CUTS) == 30.0


def test_highest_head_ignores_undetected_frames_and_one_outlier() -> None:
    samples = tuple(HeadSample(float(index), 800.0 - index) for index in range(10)) + (
        HeadSample(10.0, None),
        HeadSample(11.0, 100.0),
    )
    clocks = [sample.seconds for sample in samples]
    assert highest_head(samples, clocks, 0.0, 11.0) == 791.0
    assert highest_head(samples, clocks, 9.5, 10.5) is None


def test_growth_runs_join_only_contiguous_extensions() -> None:
    cues = (
        CaptionCue("今日は", 0, 0.5),
        CaptionCue("今日は\nいい", 0.5, 1.0),
        CaptionCue("天気", 1.0, 1.5),
        CaptionCue("天気だね", 2.0, 2.5),
    )
    assert growth_runs(cues) == [[0, 1], [2], [3]]


def test_caption_bottom_sits_above_the_head_between_gauges_and_frame_edge() -> None:
    assert place(900.0) == 860
    assert place(200.0) == 545
    assert place(1950.0) == 1800


def test_height_rises_at_once_but_drops_only_for_a_clear_fall() -> None:
    assert settle(860, None) == 860
    assert settle(845, 860) == 860
    assert settle(830, 860) == 830
    assert settle(1000, 860) == 860
    assert settle(1030, 860) == 1030


def test_layout_keeps_a_still_head_still_and_never_covers_a_rising_head() -> None:
    cues = (
        CaptionCue("一", 0.0, 0.5),
        CaptionCue("一二", 0.5, 2.0),
        CaptionCue("三", 3.0, 3.5),
        CaptionCue("四", 6.0, 6.5),
        CaptionCue("五", 12.0, 12.5),
    )
    samples = (
        HeadSample(0.0, 900.0),
        HeadSample(1.5, 800.0),
        HeadSample(3.0, 790.0),
        HeadSample(6.0, 1300.0),
        HeadSample(12.0, None),
    )
    assert layout(cues, CUTS, samples) == (760, 760, 760, 1260, 1260)


def test_layout_without_any_detection_uses_the_legacy_centre_line() -> None:
    cues = (CaptionCue("こんにちは", 0.0, 0.5),)
    assert layout(cues, CUTS, (HeadSample(0.0, None),)) == (802,)


def test_strict_headroom_preserves_rim_clearance_and_rises_without_tolerance() -> None:
    cues = (CaptionCue("一", 0.0, 0.5), CaptionCue("二", 2.0, 2.5))
    samples = (HeadSample(0.0, 900.0), HeadSample(2.0, 895.0))
    assert layout(cues, CUTS, samples, require_headroom=True) == (851, 846)


def test_strict_headroom_uses_highest_observation_for_the_whole_growth_run() -> None:
    cues = (CaptionCue("一", 0.0, 0.5), CaptionCue("一二", 0.5, 6.0))
    samples = tuple(HeadSample(index * 0.5, 900.0) for index in range(10)) + (
        HeadSample(5.0, 800.0),
    )
    assert layout(cues, CUTS, samples, require_headroom=True) == (751, 751)


@pytest.mark.parametrize("top", [None, 590.0])
def test_strict_headroom_refuses_missing_detection_or_insufficient_space(top: float | None) -> None:
    cues = (CaptionCue("こんにちは", 0.0, 0.5),)
    with pytest.raises(ReplayError, match="no observed head|cannot fit above"):
        layout(cues, CUTS, (HeadSample(0.0, top),), require_headroom=True)


@pytest.mark.parametrize("top", [None, 0.0, 590.0])
def test_explicit_top_overlap_clamps_instead_of_reframing(top: float | None) -> None:
    cues = (CaptionCue("こんにちは", 0.0, 0.5),)
    assert layout(cues, CUTS, (HeadSample(0.0, top),), allow_top_overlap=True) == (554,)


@pytest.mark.parametrize("top", [None, 0.0, 590.0])
def test_larger_caption_block_preserves_upper_limit(top: float | None) -> None:
    cues = (CaptionCue("こんにちは", 0.0, 0.5),)
    assert layout(
        cues, CUTS, (HeadSample(0.0, top),), allow_top_overlap=True, caption_scale=1.3
    ) == (607,)


@pytest.mark.parametrize("scale", [True, 0.0, -1.0, 4.01, float("nan"), float("inf")])
def test_invalid_caption_scale_is_rejected(scale: float) -> None:
    with pytest.raises(ReplayError, match="Invalid caption scale"):
        layout((), CUTS, (), caption_scale=scale)


def test_larger_caption_cannot_exceed_canvas() -> None:
    with pytest.raises(ReplayError, match="exceeds the available canvas"):
        layout((), CUTS, (), 1700, caption_scale=1.3)


def test_larger_non_strict_caption_uses_scaled_height() -> None:
    cues = (CaptionCue("一", 0.0, 0.5),)
    assert layout(cues, CUTS, (HeadSample(0.0, 10),), caption_scale=1.3) == (595,)
    assert layout(cues, CUTS, (), caption_scale=1.3) == (827,)


def test_top_overlap_still_uses_highest_head_and_immediate_upward_motion() -> None:
    cues = (CaptionCue("一", 0.0, 0.5), CaptionCue("二", 2.0, 2.5))
    samples = (HeadSample(0.0, 900.0), HeadSample(2.0, 895.0))
    assert layout(cues, CUTS, samples, allow_top_overlap=True) == (851, 846)


def test_strict_headroom_retains_downward_stability_and_frame_limit() -> None:
    cues = (CaptionCue("一", 0.0, 0.5), CaptionCue("二", 2.0, 2.5), CaptionCue("三", 4.0, 4.5))
    samples = (HeadSample(0.0, 900.0), HeadSample(2.0, 1000.0), HeadSample(4.0, 2000.0))
    assert layout(cues, CUTS, samples, require_headroom=True) == (851, 851, 1800)


def test_display_text_breaks_after_punctuation_and_balances_long_lines() -> None:
    assert display_text("今日は晴れです、明日は雨") == "今日は晴れです、\n明日は雨"
    assert display_text("あいうえおかきくけこさしすせそたちつてとな") == (
        "あいうえおかきくけこさ\nしすせそたちつてとな"
    )
    assert display_text("みんなの元に届いてる\nかな") == "みんなの元に届いてるかな"


def test_ass_uses_the_single_speaker_blue_cyan_style_centred() -> None:
    assert STYLE_RECORD["fill"] == "#005A9C"
    result = render_ass([PlacedCaption("こんにちは", 0.5, 1.5, 860)])
    assert "PlayResX: 1080\nPlayResY: 1920" in result
    assert "Style: vertical,Hiragino Sans,66," in result
    assert (
        "Dialogue: 0,0:00:00.50,0:00:01.50,vertical,,0,0,0,,"
        "{\\an2\\pos(540,860)\\bord9\\shad0\\3c&H000000&\\1a&HFF&}こんにちは"
    ) in result
    assert (
        "Dialogue: 1,0:00:00.50,0:00:01.50,vertical,,0,0,0,,"
        "{\\an2\\pos(540,860)\\bord6\\shad0\\3c&HFFFFFF&\\1a&HFF&}こんにちは"
    ) in result
    assert (
        "Dialogue: 2,0:00:00.50,0:00:01.50,vertical,,0,0,0,,"
        "{\\an2\\pos(540,860)\\bord0\\shad0\\1c&H9C5A00&\\clip(0,0,1080,1920)}こんにちは"
    ) in result


def survey(*frames: list[tuple[float, float, float, float]]) -> dict[str, object]:
    return {
        "data": {
            "structuredContent": {
                "measurement": [
                    {
                        "width": 540,
                        "height": 960,
                        "lines": [
                            {"text": "x", "region": {"x": x, "y": y, "width": w, "height": h}}
                            for x, y, w, h in boxes
                        ],
                    }
                    for boxes in frames
                ]
            }
        }
    }


def test_fixed_text_over_the_caption_column_lowers_the_top_limit() -> None:
    banner = (170.0, 178.0, 316.0, 26.0)
    counter = (55.0, 290.0, 110.0, 22.0)
    moving = [(200.0, 100.0 * index + 40.0, 50.0, 20.0) for index in range(5)]
    assert (
        top_limit(
            survey(
                [banner, counter, moving[0]],
                [banner, counter, moving[1]],
                [(172.0, 179.0, 316.0, 26.0), counter, moving[2]],
                [moving[3]],
                [moving[4]],
            )
        )
        == 426
    )
    assert top_limit(survey([counter], [counter], [counter])) == 380
    assert place(200.0, 426) == 591


def test_survey_shape_is_validated() -> None:
    with pytest.raises(ReplayError, match="measurement list"):
        top_limit({"data": {"structuredContent": {}}})
    with pytest.raises(ReplayError, match="frame is malformed"):
        top_limit({"data": {"structuredContent": {"measurement": [{"width": 1}]}}})
    incomplete = {
        "data": {
            "structuredContent": {
                "measurement": [
                    {"width": 540, "height": 960, "lines": [{"region": {"x": 1, "y": 1}}]}
                ]
            }
        }
    }
    with pytest.raises(ReplayError, match="incomplete"):
        top_limit(incomplete)
