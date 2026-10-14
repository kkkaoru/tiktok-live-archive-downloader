from fractions import Fraction

import pytest

from replay.core import ReplayError
from replay.foreground_observations import ForegroundFrame, ForegroundMask, foreground_frames
from replay.vertical_captions import Box


@pytest.fixture
def row() -> dict[str, object]:
    return {
        "requested": {"value": 1001, "timescale": 30000},
        "actual": {"value": 3003, "timescale": 90000},
        "width": 16,
        "height": 16,
        "observationPresent": True,
        "instanceLabels": [1, 3],
        "unionMask": {
            "width": 16,
            "height": 16,
            "nonzeroPixels": 20,
            "nonzeroBounds": {"x": 3, "y": 10, "width": 5, "height": 4},
        },
        "maskThreshold8Bit": 1,
        "captionPlacementApproved": False,
    }


def test_exact_rational_clock_and_top_left_pixel_extent(row: dict[str, object]) -> None:
    assert foreground_frames([row], times=(Fraction(1001, 30000),), width=16, height=16) == (
        ForegroundFrame(Fraction(1001, 30000), (1, 3), ForegroundMask(20, Box(3, 10, 8, 14))),
    )


def test_absence_and_empty_labels_remain_distinct_unknowns(row: dict[str, object]) -> None:
    absent = row | {"observationPresent": False, "instanceLabels": None, "unionMask": None}
    empty = row | {
        "requested": {"value": 1, "timescale": 1},
        "actual": {"value": 1, "timescale": 1},
        "instanceLabels": [],
        "unionMask": None,
    }
    assert foreground_frames(
        [absent, empty], times=(Fraction(1001, 30000), Fraction(1)), width=16, height=16
    ) == (
        ForegroundFrame(Fraction(1001, 30000), None, None),
        ForegroundFrame(Fraction(1), (), None),
    )


def test_present_instance_with_empty_quantized_mask_is_not_subject_absence(
    row: dict[str, object],
) -> None:
    row["unionMask"] = {"width": 16, "height": 16, "nonzeroPixels": 0}
    assert foreground_frames([row], times=(Fraction(1001, 30000),), width=16, height=16) == (
        ForegroundFrame(Fraction(1001, 30000), (1, 3), ForegroundMask(0, None)),
    )


@pytest.mark.parametrize(
    "width,height",
    [(True, 16), (16, False), (0, 16), (16, -1), (16385, 1), (1, 16385), (4097, 4097)],
)
def test_native_source_geometry_budget(width: int, height: int) -> None:
    with pytest.raises(ReplayError, match="geometry budget"):
        foreground_frames([], times=(Fraction(0),), width=width, height=height)


@pytest.mark.parametrize(
    "times",
    [
        [],
        (),
        (0.5,),
        (Fraction(-1),),
        (Fraction(86401),),
        (Fraction(1, 1000000001),),
        (Fraction(1), Fraction(1)),
        (Fraction(2), Fraction(1)),
        (Fraction(0),) * 9,
    ],
)
def test_invalid_expected_clocks(times: tuple[Fraction, ...]) -> None:
    with pytest.raises(ReplayError, match="expected foreground sample clocks"):
        foreground_frames([], times=times, width=16, height=16)


@pytest.mark.parametrize("raw", [None, {}, [], [None, None]])
def test_missing_or_extra_frame_is_not_filled(raw: object) -> None:
    with pytest.raises(ReplayError, match="frame count"):
        foreground_frames(raw, times=(Fraction(0),), width=16, height=16)


@pytest.mark.parametrize(
    "patch",
    [
        {"requested": {"value": 1000, "timescale": 30000}},
        {"actual": {"value": 1000, "timescale": 30000}},
        {"actual": {"value": -1, "timescale": 30000}},
        {"actual": {"value": 1, "timescale": 0}},
        {"actual": {"value": 1, "timescale": 1000000001}},
        {"actual": {"value": True, "timescale": 30000}},
        {"actual": {"value": 1001, "timescale": 30000.0}},
        {"actual": None},
        {"width": True},
        {"width": 17},
        {"height": 17},
        {"captionPlacementApproved": True},
        {"captionPlacementApproved": None},
        {"maskThreshold8Bit": 2},
        {"maskThreshold8Bit": True},
        {"observationPresent": 1},
        {"observationPresent": None},
        {"observationPresent": False},
        {"observationPresent": False, "instanceLabels": None},
        {"instanceLabels": None},
        {"instanceLabels": [False]},
        {"instanceLabels": [0]},
        {"instanceLabels": [-1]},
        {"instanceLabels": [2**63]},
        {"instanceLabels": [1, 1]},
        {"instanceLabels": [3, 1]},
        {"instanceLabels": []},
        {"unionMask": None},
    ],
)
def test_malformed_observations_fail_closed(
    row: dict[str, object], patch: dict[str, object]
) -> None:
    with pytest.raises(ReplayError):
        foreground_frames([row | patch], times=(Fraction(1001, 30000),), width=16, height=16)


@pytest.mark.parametrize(
    "mask",
    [
        {"width": 15, "height": 16, "nonzeroPixels": 0},
        {"width": 16, "height": 15, "nonzeroPixels": 0},
        {"width": 16, "height": 16, "nonzeroPixels": -1},
        {"width": 16, "height": 16, "nonzeroPixels": 257},
        {"width": 16, "height": 16, "nonzeroPixels": True},
        {"width": 16, "height": 16, "nonzeroPixels": 1},
        {"width": 16, "height": 16, "nonzeroPixels": 0, "nonzeroBounds": {}},
    ],
)
def test_mask_counts_and_dimensions_are_validated(
    row: dict[str, object], mask: dict[str, object]
) -> None:
    with pytest.raises(ReplayError):
        foreground_frames(
            [row | {"unionMask": mask}], times=(Fraction(1001, 30000),), width=16, height=16
        )


@pytest.mark.parametrize(
    "bounds",
    [
        {"x": -1, "y": 0, "width": 5, "height": 4},
        {"x": 0, "y": -1, "width": 5, "height": 4},
        {"x": 12, "y": 0, "width": 5, "height": 4},
        {"x": 0, "y": 13, "width": 5, "height": 4},
        {"x": 0, "y": 0, "width": 0, "height": 4},
        {"x": 0, "y": 0, "width": 5, "height": 0},
        {"x": 0, "y": 0, "width": 4, "height": 4},
        {"x": 0.5, "y": 0, "width": 5, "height": 4},
    ],
)
def test_mask_bounds_are_not_clipped_or_fabricated(
    row: dict[str, object], bounds: dict[str, object]
) -> None:
    mask = {"width": 16, "height": 16, "nonzeroPixels": 20, "nonzeroBounds": bounds}
    with pytest.raises(ReplayError):
        foreground_frames(
            [row | {"unionMask": mask}], times=(Fraction(1001, 30000),), width=16, height=16
        )


def test_sparse_bbox_area_is_not_claimed_as_foreground_pixel_coverage(
    row: dict[str, object],
) -> None:
    row["unionMask"] = {
        "width": 16,
        "height": 16,
        "nonzeroPixels": 2,
        "nonzeroBounds": {"x": 0, "y": 0, "width": 16, "height": 16},
    }
    assert foreground_frames([row], times=(Fraction(1001, 30000),), width=16, height=16) == (
        ForegroundFrame(Fraction(1001, 30000), (1, 3), ForegroundMask(2, Box(0, 0, 16, 16))),
    )
