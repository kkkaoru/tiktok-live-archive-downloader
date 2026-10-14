import pytest

from replay.core import ReplayError
from replay.head_observations import HeadObservation, head_frames
from replay.vertical_captions import HeadSample


def frame(**changes: object) -> dict[str, object]:
    result: dict[str, object] = {
        "requestedTimeSeconds": 2.5,
        "actualTimeSeconds": 2.48,
        "width": 540,
        "height": 960,
        "faces": [{"x": 100, "y": 300, "width": 100, "height": 100}],
        "bodies": [{"x": 50, "y": 200, "width": 300, "height": 700}],
    }
    result.update(changes)
    return result


def test_face_geometry_scales_to_canvas_using_actual_clock() -> None:
    envelope = {
        "ok": True,
        "data": {"structuredContent": {"sourceModified": False, "measurement": [frame()]}},
    }
    assert head_frames(envelope) == (HeadObservation(2.5, HeadSample(2.48, 440.0)),)


def test_letterbox_offset_and_missing_detection_remain_explicit() -> None:
    assert head_frames([frame(width=960, height=540, faces=[], bodies=[])], native=True) == (
        HeadObservation(2.5, HeadSample(2.48, None)),
    )
    result = head_frames(
        [
            frame(
                width=960,
                height=540,
                bodies=[],
                faces=[{"x": 100, "y": 200, "width": 100, "height": 100}],
            )
        ],
        native=True,
    )
    assert result == (HeadObservation(2.5, HeadSample(2.48, 791.25)),)


def test_largest_face_wins_and_tallest_body_is_used_only_without_faces() -> None:
    faces = [
        {"x": 0, "y": 100, "width": 20, "height": 20},
        {"x": 100, "y": 300, "width": 100, "height": 100},
    ]
    bodies = [
        {"x": 0, "y": 50, "width": 20, "height": 50},
        {"x": 100, "y": 200, "width": 100, "height": 400},
    ]
    assert head_frames([frame(faces=faces, bodies=bodies)], native=True)[0].sample.top == 440.0
    assert head_frames([frame(faces=[], bodies=bodies)], native=True)[0].sample.top == 400.0


@pytest.mark.parametrize("value", [None, [], [{}] * 9])
def test_bad_batch_is_rejected(value: object) -> None:
    with pytest.raises(ReplayError, match="batch"):
        head_frames(value, native=True)


@pytest.mark.parametrize(
    "changes",
    [
        {"width": True},
        {"height": 0},
        {"width": 16385},
        {"width": 4097, "height": 4096},
        {"actualTimeSeconds": -1},
        {"requestedTimeSeconds": 86401},
        {"actualTimeSeconds": True},
        {"actualTimeSeconds": float("nan")},
        {"requestedTimeSeconds": 10**1000},
        {"faces": None},
        {"bodies": [{}] * 33},
        {"faces": [{"x": 0, "y": 0, "width": 541, "height": 1}]},
        {"faces": [{"x": 0, "y": 0, "width": 1, "height": 0}]},
        {"faces": [{"x": -1, "y": 0, "width": 1, "height": 1}]},
    ],
)
def test_external_geometry_and_clocks_are_validated(changes: dict[str, object]) -> None:
    with pytest.raises(ReplayError):
        head_frames([frame(**changes)], native=True)


def test_native_arrays_are_not_accepted_as_executor_success() -> None:
    with pytest.raises(ValueError, match="object"):
        head_frames([frame()])
