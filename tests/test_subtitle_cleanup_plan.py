from fractions import Fraction

import pytest

from replay.caption_io import FrameObservation
from replay.caption_masks import MaskPolicy, TextDetection, TextRectangle
from replay.core import ReplayError
from replay.subtitle_cleanup_plan import (
    CaptionSearchProfile,
    admission,
    clock_chunks,
    frame_region,
    search_profile,
)
from replay.subtitle_clock import VideoFrameClock
from replay.subtitle_profile import SubtitleBands


@pytest.fixture
def observations() -> tuple[FrameObservation, ...]:
    return tuple(
        FrameObservation(
            float(index),
            float(index),
            100,
            100,
            (TextDetection("字幕" + word, 0.9, TextRectangle(0, 40, 100, 20)),),
        )
        for index, word in enumerate("春夏秋冬雨雲花風月空")
    )


def test_chunk_boundaries_keep_nonzero_vfr_clock() -> None:
    chunks = clock_chunks(
        {"pts": [100, 103, 109, 111], "timeBase": "1/1000", "lastDuration": 9},
        maximum_frames=2,
    )
    assert tuple(chunk.pts for chunk in chunks) == ((100, 103), (109, 111))
    assert tuple(chunk.last_duration for chunk in chunks) == (6, 9)


@pytest.mark.parametrize(
    "value",
    [
        None,
        {},
        {"pts": []},
        {"pts": [True]},
        {"pts": [-1]},
        {"pts": [0], "timeBase": 1, "lastDuration": 1},
        {"pts": [0], "timeBase": "1/30", "lastDuration": False},
        {"pts": [0], "timeBase": "1/30", "lastDuration": 0},
        {"pts": [2, 1], "timeBase": "1/30", "lastDuration": 1},
        {"pts": [2, 2], "timeBase": "1/30", "lastDuration": 1},
    ],
)
def test_invalid_global_clock(value: object) -> None:
    with pytest.raises((ValueError, ReplayError)):
        clock_chunks(value, maximum_frames=2)


@pytest.mark.parametrize("budget", [True, 0, 10001])
def test_invalid_chunk_budget(budget: int) -> None:
    with pytest.raises(ValueError):
        clock_chunks({"pts": [0], "timeBase": "1/30", "lastDuration": 1}, maximum_frames=budget)


def test_profile_is_measured_and_retains_canvas_edges(
    observations: tuple[FrameObservation, ...],
) -> None:
    profile = search_profile(observations)
    assert profile.measured_envelope == (0, 40, 100, 20)
    assert profile.bands.primary == (1,)


def test_frame_regions_retain_learned_spill_when_one_line_is_missed() -> None:
    profile = CaptionSearchProfile(SubtitleBands(300, 600, (10,), (11,)), (0, 320, 300, 64))
    frame = FrameObservation(
        0, 0, 300, 600, (TextDetection("字幕", 0.9, TextRectangle(110, 320, 80, 20)),)
    )
    assert frame_region(frame, profile, policy=MaskPolicy(vertical_line_padding=0)) == (
        {"x": 0, "y": 320, "width": 300, "height": 64},
        [0],
    )


def test_nontext_frame_is_unknown_not_absence(observations: tuple[FrameObservation, ...]) -> None:
    profile = search_profile(observations)
    frames = (observations[0], FrameObservation(1, 1, 100, 100, ()))
    plan = admission(frames, VideoFrameClock((0, 1), Fraction(1), 1), profile, policy=MaskPolicy())
    assert plan["unknownSourcePTS"] == [1]
    assert plan["removalQualityApproved"] is False
    assert plan["unknownFramesKeptUnmodified"] is True


def test_all_recognized_frames_are_kept_on_their_clock(
    observations: tuple[FrameObservation, ...],
) -> None:
    plan = admission(
        observations[:2],
        VideoFrameClock((0, 1), Fraction(1), 1),
        search_profile(observations),
        policy=MaskPolicy(),
    )
    assert plan["unknownSourcePTS"] == []
    assert plan["masks"] == [
        {
            "startSeconds": 0.0,
            "endSeconds": 1.0,
            "region": {"x": 0, "y": 10, "width": 100, "height": 80},
        },
        {
            "startSeconds": 1.0,
            "endSeconds": 2.0,
            "region": {"x": 0, "y": 10, "width": 100, "height": 80},
        },
    ]


def test_missing_observation_rejected(observations: tuple[FrameObservation, ...]) -> None:
    with pytest.raises(ValueError, match="cover"):
        admission(
            observations[:1],
            VideoFrameClock((0, 1), Fraction(1), 1),
            search_profile(observations),
            policy=MaskPolicy(),
        )


def test_wrong_actual_time_rejected(observations: tuple[FrameObservation, ...]) -> None:
    frame = FrameObservation(0, 1, 100, 100, observations[0].detections)
    with pytest.raises(ValueError, match="clock"):
        admission(
            (frame,),
            VideoFrameClock((0,), Fraction(1), 1),
            search_profile(observations),
            policy=MaskPolicy(),
        )


def test_search_policy_exclusion_does_not_fabricate_support(
    observations: tuple[FrameObservation, ...],
) -> None:
    assert (
        frame_region(
            observations[0],
            search_profile(observations),
            policy=MaskPolicy(search_top=0.7, search_bottom=1),
        )
        is None
    )


def test_large_measured_region_is_not_implicitly_scaled() -> None:
    profile = CaptionSearchProfile(SubtitleBands(4000, 4000, (50,), ()), (0, 1500, 4000, 1000))
    frame = FrameObservation(
        0, 0, 4000, 4000, (TextDetection("字幕", 0.9, TextRectangle(1900, 1600, 200, 20)),)
    )
    with pytest.raises(ValueError, match="budget"):
        frame_region(frame, profile, policy=MaskPolicy(search_top=0))
