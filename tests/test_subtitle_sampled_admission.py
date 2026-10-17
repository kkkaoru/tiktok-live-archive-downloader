"""Sampled admission fills regions conservatively, keeps unknown frames and records its rule."""

from fractions import Fraction

import pytest

from replay import subtitle_sampled_admission as admission
from replay.caption_io import FrameObservation
from replay.caption_masks import MaskPolicy, TextDetection, TextRectangle
from replay.subtitle_cleanup_plan import CaptionSearchProfile
from replay.subtitle_clock import VideoFrameClock
from replay.subtitle_profile import SubtitleBands
from replay.subtitle_region_fill import Region

WIDTH = 1080
HEIGHT = 1920
CLOCK = VideoFrameClock(tuple(range(0, 10 * 3000, 3000)), Fraction(1, 90000), 3000)
BANDS = SubtitleBands(WIDTH, HEIGHT, primary=(42,), secondary=())
PROFILE = CaptionSearchProfile(BANDS, (0, 1283, 1077, 167))
POLICY = MaskPolicy()
LINE = TextDetection("こんにちは世界です", 0.9, TextRectangle(200, 1330, 600, 48))


def _frame(index: int, lines: tuple[TextDetection, ...]) -> FrameObservation:
    seconds = float(CLOCK.pts[index] * CLOCK.time_base)
    return FrameObservation(seconds, seconds, WIDTH, HEIGHT, lines)


def test_admission_brackets_sampled_regions_over_unsampled_frames() -> None:
    sampled = {0: _frame(0, (LINE,)), 9: _frame(9, (LINE,))}
    result = admission.admission(
        sampled=sampled,
        clock=CLOCK,
        profile=PROFILE,
        policy=POLICY,
        stride=9,
        triggers=(),
        detected=[None] * 10,
        radius=2,
    )
    frames = result["frames"]
    assert isinstance(frames, list)
    assert len(frames) == 10
    assert frames[0]["sampled"] is True
    assert frames[4]["sampled"] is False
    assert frames[4]["classification"] == "overlay-subtitle"
    assert result["unknownSourcePTS"] == []
    sampling = result["sampling"]
    assert isinstance(sampling, dict)
    assert sampling["rawOcr"] is False
    assert sampling["sampledFrames"] == [0, 9]


def test_admission_keeps_frames_without_any_region_unknown() -> None:
    sampled = {0: _frame(0, ()), 9: _frame(9, ())}
    result = admission.admission(
        sampled=sampled,
        clock=CLOCK,
        profile=PROFILE,
        policy=POLICY,
        stride=9,
        triggers=(),
        detected=[None] * 10,
        radius=1,
    )
    assert result["masks"] == []
    assert result["unknownSourcePTS"] == list(CLOCK.pts)


def test_admission_unites_detector_regions_into_unsampled_frames() -> None:
    sampled = {0: _frame(0, ()), 9: _frame(9, ())}
    detected: list[Region | None] = [None] * 10
    detected[5] = (100, 1300, 900, 1450)
    result = admission.admission(
        sampled=sampled,
        clock=CLOCK,
        profile=PROFILE,
        policy=POLICY,
        stride=9,
        triggers=(),
        detected=detected,
        radius=1,
    )
    masks = result["masks"]
    assert isinstance(masks, list)
    assert len(masks) == 3
    assert masks[0]["region"] == {"x": 100, "y": 1300, "width": 800, "height": 150}


def test_admission_rejects_a_sampled_frame_with_a_wrong_clock() -> None:
    wrong = FrameObservation(1.0, 1.0, WIDTH, HEIGHT, (LINE,))
    with pytest.raises(ValueError, match="clock differs"):
        admission.admission(
            sampled={0: wrong, 9: _frame(9, ())},
            clock=CLOCK,
            profile=PROFILE,
            policy=POLICY,
            stride=9,
            triggers=(),
            detected=[None] * 10,
            radius=1,
        )


def test_admission_rejects_detector_regions_of_the_wrong_length() -> None:
    with pytest.raises(ValueError, match="do not cover"):
        admission.admission(
            sampled={0: _frame(0, ()), 9: _frame(9, ())},
            clock=CLOCK,
            profile=PROFILE,
            policy=POLICY,
            stride=9,
            triggers=(),
            detected=[None] * 3,
            radius=1,
        )


def test_admission_rejects_a_region_below_the_detector_minimum() -> None:
    sampled = {0: _frame(0, ()), 9: _frame(9, ())}
    detected: list[Region | None] = [None] * 10
    detected[5] = (100, 1300, 104, 1450)
    with pytest.raises(ValueError, match="budget"):
        admission.admission(
            sampled=sampled,
            clock=CLOCK,
            profile=PROFILE,
            policy=POLICY,
            stride=9,
            triggers=(),
            detected=detected,
            radius=0,
        )


def test_line_boxes_keep_text_touching_the_caption_zone_from_neighbours_within_the_radius() -> None:
    other = TextDetection("ありがとう", 0.9, TextRectangle(100.4, 1400.2, 50.2, 20.0))
    far = TextDetection("遠い", 0.9, TextRectangle(10, 100, 40, 20))
    blank = TextDetection("  ", 0.9, TextRectangle(200, 1330, 50, 20))
    sampled = {0: _frame(0, (LINE, far, blank)), 9: _frame(9, (other,))}
    result = admission.admission(
        sampled=sampled,
        clock=CLOCK,
        profile=PROFILE,
        policy=MaskPolicy(),
        stride=9,
        triggers=(),
        detected=[None] * 10,
        radius=5,
    )
    frames = result["frames"]
    assert isinstance(frames, list)
    assert frames[4]["lineBoxes"] == [[200, 1330, 800, 1378], [100, 1400, 151, 1421]]


def test_line_boxes_ignore_sampled_frames_beyond_the_radius() -> None:
    sampled = {0: _frame(0, (LINE,)), 9: _frame(9, (LINE,))}
    result = admission.admission(
        sampled=sampled,
        clock=CLOCK,
        profile=PROFILE,
        policy=MaskPolicy(),
        stride=9,
        triggers=(),
        detected=[None] * 10,
        radius=3,
    )
    frames = result["frames"]
    assert isinstance(frames, list)
    assert (frames[4]["lineBoxes"], frames[1]["lineBoxes"]) == ([], [[200, 1330, 800, 1378]])


def test_frame_without_sampled_neighbour_text_has_no_line_boxes() -> None:
    sampled = {0: _frame(0, ()), 9: _frame(9, ())}
    result = admission.admission(
        sampled=sampled,
        clock=CLOCK,
        profile=PROFILE,
        policy=MaskPolicy(),
        stride=9,
        triggers=(),
        detected=[(0, 1300, 1000, 1400)] + [None] * 9,
        radius=9,
    )
    frames = result["frames"]
    assert isinstance(frames, list)
    assert frames[0]["lineBoxes"] == []


def test_region_also_covers_recognised_lines_that_the_selection_ignored() -> None:
    faint = TextDetection("ありがとう", 0.3, TextRectangle(100, 1400, 50, 20))
    detected: list[Region | None] = [None] * 10
    detected[4] = (400, 1390, 500, 1410)
    result = admission.admission(
        sampled={0: _frame(0, (faint,)), 9: _frame(9, ())},
        clock=CLOCK,
        profile=PROFILE,
        policy=MaskPolicy(),
        stride=9,
        triggers=(),
        detected=detected,
        radius=5,
    )
    frames = result["frames"]
    assert isinstance(frames, list)
    assert frames[4]["region"] == {"x": 94, "y": 1380, "width": 406, "height": 60}
    assert frames[9]["region"] == {"x": 400, "y": 1390, "width": 100, "height": 20}
    assert result["policy"] == "sampled-ocr-plus-detector-regions-v5"


def test_line_reach_grows_the_lines_box_by_the_median_line_height_each_way() -> None:
    lines = [[10, 100, 60, 130], [20, 140, 80, 165], [15, 170, 70, 215]]
    assert admission._line_reach(lines) == [(10, 70, 80, 245)]


def test_line_reach_ignores_vertical_boxes() -> None:
    assert admission._line_reach([[10, 100, 60, 130], [700, 20, 720, 300]]) == [(10, 70, 60, 160)]


def test_line_reach_is_empty_when_no_box_is_horizontal() -> None:
    assert admission._line_reach([[700, 20, 720, 300]]) == []


def test_line_reach_is_empty_without_lines() -> None:
    assert admission._line_reach([]) == []


def test_line_boxes_skip_vertical_interface_text_inside_the_zone() -> None:
    menu = TextDetection("自分のアバター", 0.9, TextRectangle(760, 1300, 24, 130))
    sampled = {0: _frame(0, (LINE, menu)), 9: _frame(9, ())}
    result = admission.admission(
        sampled=sampled,
        clock=CLOCK,
        profile=PROFILE,
        policy=MaskPolicy(),
        stride=9,
        triggers=(),
        detected=[None] * 10,
        radius=5,
    )
    frames = result["frames"]
    assert isinstance(frames, list)
    assert frames[2]["lineBoxes"] == [[200, 1330, 800, 1378]]


def test_within_zone_clips_a_region_stretched_by_interface_text() -> None:
    zone: Region = (0, 1300, 1077, 1440)
    assert admission._within_zone((0, 1030, 1080, 1647), zone) == (0, 1230, 1080, 1510)


def test_within_zone_keeps_a_region_already_near_the_zone() -> None:
    zone: Region = (0, 1300, 1077, 1440)
    assert admission._within_zone((100, 1290, 900, 1450), zone) == (100, 1290, 900, 1450)
