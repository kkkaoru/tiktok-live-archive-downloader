from fractions import Fraction

import pytest

from replay.core import ReplayError
from replay.head_observations import tracking_frames
from replay.subtitle_clock import VideoFrameClock
from replay.vertical_captions import Box


@pytest.fixture
def frame() -> dict[str, object]:
    return {
        "presentationValue": 1,
        "presentationTimescale": 2,
        "width": 100,
        "height": 200,
        "method": "detection",
        "detectionAttempted": True,
        "faces": [{"region": {"x": 10, "y": 20, "width": 30, "height": 40}, "confidence": 0.9}],
    }


def test_exact_native_clocks_tracking_and_missing_geometry(frame: dict[str, object]) -> None:
    result = tracking_frames(
        [
            frame,
            {
                **frame,
                "presentationValue": 16,
                "presentationTimescale": 30,
                "method": "tracking",
                "detectionAttempted": False,
            },
            {
                **frame,
                "presentationValue": 17,
                "presentationTimescale": 30,
                "method": "missing",
                "faces": [],
            },
        ],
        VideoFrameClock((15, 16, 17), Fraction(1, 30), 1),
        width=100,
        height=200,
    )
    assert tuple(item.time for item in result) == (
        Fraction(1, 2),
        Fraction(8, 15),
        Fraction(17, 30),
    )
    assert tuple(item.method for item in result) == ("detection", "tracking", "missing")
    assert result[0].faces[0].box == Box(10, 20, 40, 60)
    assert result[0].faces[0].confidence == 0.9
    assert result[1].detection_attempted is False
    assert result[2].faces == ()


def test_lost_track_rejections_remain_explicit_without_geometry(frame: dict[str, object]) -> None:
    result = tracking_frames(
        [
            frame,
            {
                **frame,
                "presentationValue": 16,
                "presentationTimescale": 30,
                "method": "missing",
                "faces": [],
                "outOfFrameTrackingObservations": 1,
            },
        ],
        VideoFrameClock((15, 16), Fraction(1, 30), 1),
        width=100,
        height=200,
    )
    assert result[0].out_of_frame_tracking_observations == 0
    assert result[1].out_of_frame_tracking_observations == 1
    assert result[1].faces == ()
    assert result[1].method == "missing"


def test_rejected_track_requires_actual_redetection_attempt(frame: dict[str, object]) -> None:
    with pytest.raises(ReplayError, match="rejection evidence"):
        tracking_frames(
            [
                frame,
                {
                    **frame,
                    "presentationValue": 16,
                    "presentationTimescale": 30,
                    "method": "tracking",
                    "detectionAttempted": False,
                    "outOfFrameTrackingObservations": 1,
                },
            ],
            VideoFrameClock((15, 16), Fraction(1, 30), 1),
            width=100,
            height=200,
        )


def test_detector_rejections_do_not_fill_missing_geometry(frame: dict[str, object]) -> None:
    result = tracking_frames(
        [{**frame, "method": "missing", "faces": [], "outOfFrameDetectionObservations": 1}],
        VideoFrameClock((15,), Fraction(1, 30), 1),
        width=100,
        height=200,
    )
    assert result[0].out_of_frame_detection_observations == 1
    assert result[0].out_of_frame_tracking_observations == 0
    assert result[0].faces == ()


def test_detector_rejections_require_actual_attempt(frame: dict[str, object]) -> None:
    with pytest.raises(ReplayError, match="rejection evidence"):
        tracking_frames(
            [
                frame,
                {
                    **frame,
                    "presentationValue": 16,
                    "presentationTimescale": 30,
                    "method": "tracking",
                    "detectionAttempted": False,
                    "outOfFrameDetectionObservations": 1,
                },
            ],
            VideoFrameClock((15, 16), Fraction(1, 30), 1),
            width=100,
            height=200,
        )


def test_tracking_can_survive_an_empty_redetection(frame: dict[str, object]) -> None:
    result = tracking_frames(
        [
            frame,
            {**frame, "presentationValue": 16, "presentationTimescale": 30, "method": "tracking"},
        ],
        VideoFrameClock((15, 16), Fraction(1, 30), 1),
        width=100,
        height=200,
    )
    assert result[1].method == "tracking"
    assert result[1].detection_attempted is True


@pytest.mark.parametrize(
    "changes",
    [
        {"outOfFrameDetectionObservations": True},
        {"outOfFrameDetectionObservations": -1},
        {"outOfFrameDetectionObservations": 33},
        {"outOfFrameDetectionObservations": None},
        {"outOfFrameTrackingObservations": True},
        {"outOfFrameTrackingObservations": -1},
        {"outOfFrameTrackingObservations": 65},
        {"outOfFrameTrackingObservations": 0.5},
        {"outOfFrameTrackingObservations": None},
        {"presentationValue": True},
        {"presentationValue": -1},
        {"presentationValue": 10**100},
        {"presentationTimescale": 0},
        {"presentationTimescale": 1.5},
        {"presentationTimescale": True},
        {"presentationTimescale": 1_000_000_001},
        {"presentationValue": 2},
        {"width": True},
        {"height": True},
        {"width": 101},
        {"height": 201},
        {"method": "guessed"},
        {"method": []},
        {"method": "tracking"},
        {"detectionAttempted": 1},
        {"detectionAttempted": False},
        {"method": "missing"},
        {"faces": []},
        {"faces": None},
        {"faces": [None]},
        {"faces": [{}] * 33},
        {"faces": [{"confidence": 0.3}]},
        {"faces": [{"confidence": 1.01}]},
        {"faces": [{"confidence": float("nan")}]},
        {"faces": [{"confidence": 1, "region": {"x": 90, "y": 20, "width": 30, "height": 40}}]},
    ],
)
def test_rejects_inconsistent_or_unbounded_native_data(
    frame: dict[str, object], changes: dict[str, object]
) -> None:
    with pytest.raises(ReplayError):
        tracking_frames(
            [frame | changes], VideoFrameClock((15,), Fraction(1, 30), 1), width=100, height=200
        )


@pytest.mark.parametrize("raw", [None, {}, [], [None], [{}, {}]])
def test_rejects_missing_extra_or_malformed_frames(raw: object) -> None:
    with pytest.raises(ReplayError):
        tracking_frames(raw, VideoFrameClock((15,), Fraction(1, 30), 1), width=100, height=200)


@pytest.mark.parametrize(
    "width,height",
    [(True, 200), (100, False), (0, 200), (100, 0), (16385, 200), (100, 16385), (8192, 8192)],
)
def test_requires_verified_bounded_display_dimensions(
    frame: dict[str, object], width: int, height: int
) -> None:
    with pytest.raises(ReplayError):
        tracking_frames(
            [frame], VideoFrameClock((15,), Fraction(1, 30), 1), width=width, height=height
        )


def test_refuses_duplicate_frames_instead_of_deduplicating(frame: dict[str, object]) -> None:
    with pytest.raises(ReplayError, match="verified source clock"):
        tracking_frames(
            [frame, frame], VideoFrameClock((15, 16), Fraction(1, 30), 1), width=100, height=200
        )


def test_missing_is_preserved_without_a_default_face(frame: dict[str, object]) -> None:
    result = tracking_frames(
        [{**frame, "faces": [], "method": "missing"}],
        VideoFrameClock((15,), Fraction(1, 30), 1),
        width=100,
        height=200,
    )
    assert result[0].faces == ()
    assert result[0].time == Fraction(1, 2)


def test_refuses_frame_budget_overrun(frame: dict[str, object]) -> None:
    with pytest.raises(ReplayError, match="budget"):
        tracking_frames(
            [frame] * 7201,
            VideoFrameClock(tuple(range(7201)), Fraction(1, 120), 1),
            width=100,
            height=200,
        )


def test_refuses_duration_budget_overrun(frame: dict[str, object]) -> None:
    with pytest.raises(ReplayError, match="budget"):
        tracking_frames(
            [frame], VideoFrameClock((15,), Fraction(1, 30), 3630), width=100, height=200
        )
