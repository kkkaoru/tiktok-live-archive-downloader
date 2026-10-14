from dataclasses import replace
from fractions import Fraction

import pytest

from replay.core import ReplayError
from replay.foreground_geometry import interval_top, place_envelope
from replay.foreground_observations import ForegroundFrame, ForegroundMask
from replay.subtitle_clock import VideoFrameClock
from replay.vertical_captions import Box


@pytest.fixture
def frames() -> tuple[ForegroundFrame, ...]:
    return (
        ForegroundFrame(Fraction(0), (1,), ForegroundMask(10, Box(0, 900, 20, 1000))),
        ForegroundFrame(Fraction(1, 10), (1,), ForegroundMask(10, Box(0, 700, 20, 1000))),
        ForegroundFrame(Fraction(7, 10), (1,), ForegroundMask(10, Box(0, 900, 20, 1000))),
    )


@pytest.fixture
def clock() -> VideoFrameClock:
    return VideoFrameClock((0, 1, 7), Fraction(1, 10), 2)


@pytest.mark.parametrize(
    "start,end",
    [
        (Fraction(0), Fraction(9, 10)),
        (Fraction(1, 20), Fraction(7, 10)),
        (Fraction(1, 10), Fraction(7, 10)),
    ],
)
def test_interval_retains_high_excursion(
    frames: tuple[ForegroundFrame, ...], clock: VideoFrameClock, start: Fraction, end: Fraction
) -> None:
    assert interval_top(frames, clock=clock, start=start, end=end) == 700


@pytest.mark.parametrize(
    "start,end",
    [
        (Fraction(0), Fraction(1, 10)),
        (Fraction(7, 10), Fraction(9, 10)),
        (Fraction(4, 5), Fraction(9, 10)),
    ],
)
def test_exact_half_open_vfr_extent(
    frames: tuple[ForegroundFrame, ...], clock: VideoFrameClock, start: Fraction, end: Fraction
) -> None:
    assert interval_top(frames, clock=clock, start=start, end=end) == 900


@pytest.mark.parametrize(
    "middle",
    [
        ForegroundFrame(Fraction(1, 10), None, None),
        ForegroundFrame(Fraction(1, 10), (), None),
        ForegroundFrame(Fraction(1, 10), (1,), ForegroundMask(0, None)),
        ForegroundFrame(Fraction(1, 10), (1,), ForegroundMask(10, None)),
    ],
)
def test_required_unknown_is_not_discarded(
    frames: tuple[ForegroundFrame, ...], clock: VideoFrameClock, middle: ForegroundFrame
) -> None:
    altered = (frames[0], middle, frames[2])
    with pytest.raises(ReplayError, match="unknown"):
        interval_top(altered, clock=clock, start=Fraction(0), end=Fraction(9, 10))
    assert interval_top(altered, clock=clock, start=Fraction(7, 10), end=Fraction(9, 10)) == 900


def test_missing_middle_record_and_misaligned_batch_fail(
    frames: tuple[ForegroundFrame, ...], clock: VideoFrameClock
) -> None:
    with pytest.raises(ReplayError, match="complete supplied clock"):
        interval_top((frames[0], frames[2]), clock=clock, start=Fraction(0), end=Fraction(9, 10))
    with pytest.raises(ReplayError, match="complete supplied clock"):
        interval_top(
            (frames[0], frames[1], replace(frames[2], time=Fraction(8, 10))),
            clock=clock,
            start=Fraction(0),
            end=Fraction(9, 10),
        )


@pytest.mark.parametrize(
    "start,end",
    [
        (Fraction(-1), Fraction(1)),
        (Fraction(0), Fraction(1)),
        (Fraction(1, 10), Fraction(1, 10)),
        (Fraction(1, 10), Fraction(0)),
    ],
)
def test_interval_outside_verified_extents_fails(
    frames: tuple[ForegroundFrame, ...], clock: VideoFrameClock, start: Fraction, end: Fraction
) -> None:
    with pytest.raises(ReplayError, match="outside"):
        interval_top(frames, clock=clock, start=start, end=end)


def test_cut_sides_are_independently_evaluated(
    frames: tuple[ForegroundFrame, ...], clock: VideoFrameClock
) -> None:
    altered = (frames[0], replace(frames[1], labels=None, mask=None), frames[2])
    assert interval_top(altered, clock=clock, start=Fraction(0), end=Fraction(1, 10)) == 900
    assert interval_top(altered, clock=clock, start=Fraction(7, 10), end=Fraction(9, 10)) == 900
    with pytest.raises(ReplayError, match="unknown"):
        interval_top(altered, clock=clock, start=Fraction(0), end=Fraction(9, 10))


def test_origin_relative_envelope_with_clearance() -> None:
    assert place_envelope(
        Box(-40, -20, 40, 10),
        region=Box(0, 10, 200, 200),
        center_x=100,
        foreground_top=90,
        clearance=10,
    ) == (100, 70)


def test_stabilization_only_inside_feasible_range() -> None:
    assert place_envelope(
        Box(-40, -20, 40, 10),
        region=Box(0, 10, 200, 200),
        center_x=100,
        foreground_top=90,
        clearance=10,
        previous_y=50,
    ) == (100, 50)
    assert place_envelope(
        Box(-40, -20, 40, 10),
        region=Box(0, 10, 200, 200),
        center_x=100,
        foreground_top=90,
        clearance=10,
        previous_y=-100,
    ) == (100, 30)
    assert place_envelope(
        Box(-40, -20, 40, 10),
        region=Box(0, 10, 200, 200),
        center_x=100,
        foreground_top=90,
        clearance=10,
        previous_y=1000,
    ) == (100, 70)


def test_actual_motion_leading_boundary_is_not_estimated_font_height() -> None:
    assert place_envelope(
        Box(-20, 0, 20, 26), region=Box(0, 0, 100, 100), center_x=50, foreground_top=26, clearance=0
    ) == (50, 0)
    with pytest.raises(ReplayError, match="cannot fit above"):
        place_envelope(
            Box(-20, 0, 20, 26),
            region=Box(0, 0, 100, 100),
            center_x=50,
            foreground_top=25,
            clearance=0,
        )
    assert place_envelope(
        Box(-20, 0, 20, 12), region=Box(0, 0, 100, 100), center_x=50, foreground_top=25, clearance=0
    ) == (50, 13)


def test_asymmetric_animation_and_width_cannot_be_clamped() -> None:
    assert place_envelope(
        Box(-10, -5, 20, 5), region=Box(0, 0, 100, 100), center_x=50, foreground_top=50, clearance=0
    ) == (50, 45)
    with pytest.raises(ReplayError, match="fixed horizontal"):
        place_envelope(
            Box(-10, -5, 51, 5),
            region=Box(0, 0, 100, 100),
            center_x=50,
            foreground_top=50,
            clearance=0,
        )
    with pytest.raises(ReplayError, match="fixed horizontal"):
        place_envelope(
            Box(-51, -5, 10, 5),
            region=Box(0, 0, 100, 100),
            center_x=50,
            foreground_top=50,
            clearance=0,
        )


def test_canvas_bottom_remains_a_constraint() -> None:
    assert place_envelope(
        Box(-10, -10, 10, 10),
        region=Box(0, 0, 100, 100),
        center_x=50,
        foreground_top=1000,
        clearance=0,
    ) == (50, 90)


@pytest.mark.parametrize(
    "box", [Box(0, 0, 0, 10), Box(0, 10, 10, 0), Box(0, float("nan"), 10, 10), Box(0, 0, 20000, 10)]
)
def test_invalid_envelope_fails(box: Box) -> None:
    with pytest.raises(ReplayError, match="rectangle"):
        place_envelope(
            box, region=Box(0, 0, 100, 100), center_x=50, foreground_top=100, clearance=0
        )


@pytest.mark.parametrize(
    "top,gap,previous",
    [
        (float("nan"), 0, None),
        (-1, 0, None),
        (100, -1, None),
        (100, float("inf"), None),
        (100, 0, float("nan")),
        (True, 0, None),
    ],
)
def test_invalid_constraints_fail(top: float, gap: float, previous: float | None) -> None:
    with pytest.raises(ReplayError, match="constraints"):
        place_envelope(
            Box(-10, -10, 10, 10),
            region=Box(0, 0, 100, 100),
            center_x=50,
            foreground_top=top,
            clearance=gap,
            previous_y=previous,
        )


def test_interval_top_rejects_invalid_support_rectangle(
    clock: VideoFrameClock, frames: tuple[ForegroundFrame, ...]
) -> None:
    middle = replace(frames[1], mask=ForegroundMask(10, Box(0, -10, 10, 10)))
    with pytest.raises(ReplayError, match="rectangle"):
        interval_top(
            (frames[0], middle, frames[2]), clock=clock, start=Fraction(0), end=Fraction(9, 10)
        )
