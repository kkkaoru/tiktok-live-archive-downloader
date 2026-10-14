from fractions import Fraction

import pytest

from replay.caption_style_defaults import HEAD_CAPTION_REFERENCE_HEIGHT, HEAD_CAPTION_Y_DEADBAND_PX
from replay.caption_y_stability import CaptionYStability
from replay.core import ReplayError


def test_small_reversals_hold_but_real_motion_tracks_with_bounded_error() -> None:
    state = CaptionYStability(6)
    assert state.sample(Fraction(0), 100) == 100
    assert state.sample(Fraction(1, 30), 103) == 100
    assert state.sample(Fraction(2, 30), 97) == 100
    assert state.sample(Fraction(3, 30), 140) == 134
    assert state.sample(Fraction(4, 30), 160) == 154
    assert state.sample(Fraction(5, 30), 100) == 106


def test_shared_policy_uses_six_pixels_at_reference_canvas() -> None:
    assert HEAD_CAPTION_Y_DEADBAND_PX == 6
    assert HEAD_CAPTION_REFERENCE_HEIGHT == 1920


@pytest.mark.parametrize("rate", [24, 30, 60, 120])
def test_no_frame_count_dependent_lag(rate: int) -> None:
    state = CaptionYStability(6)
    assert state.sample(Fraction(0), 0) == 0
    assert state.sample(Fraction(1, rate), 100) == 94


def test_zero_band_preserves_input_and_equal_clock_is_supported() -> None:
    state = CaptionYStability(0)
    assert state.sample(Fraction(0), 1) == 1
    assert state.sample(Fraction(0), 2) == 2


@pytest.mark.parametrize("value", [-1.0, float("nan"), float("inf")])
def test_invalid_band(value: float) -> None:
    with pytest.raises(ReplayError):
        CaptionYStability(value)


@pytest.mark.parametrize("time,value", [(Fraction(-1), 1.0), (Fraction(0), float("nan"))])
def test_invalid_sample(time: Fraction, value: float) -> None:
    with pytest.raises(ReplayError):
        CaptionYStability(6).sample(time, value)


def test_backwards_clock_rejected() -> None:
    state = CaptionYStability(6)
    state.sample(Fraction(1), 100)
    with pytest.raises(ReplayError, match="backwards"):
        state.sample(Fraction(0), 100)
