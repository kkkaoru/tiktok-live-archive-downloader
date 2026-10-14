from __future__ import annotations

from fractions import Fraction

import pytest

from replay import frame_grid
from replay.core import ReplayError

REAL = {"rate": 30, "media_frames": 3601, "media_ticks": 10_802_880, "timebase": 90_000}


def test_real_failing_request_is_valid_on_the_frame_grid() -> None:
    chosen = frame_grid.select(**REAL, start_seconds=113.53733333333334, duration_seconds=6.5)
    assert (chosen.start, chosen.frames, chosen.end) == (3406, 195, 3601)
    assert frame_grid.tail_shortfall_ticks(3601, 10_802_880, 30, 90_000) == 120


def test_exact_fractions_and_whole_media() -> None:
    chosen = frame_grid.select(
        **{**REAL, "media_ticks": 10_803_000},
        start_seconds=Fraction(0),
        duration_seconds=Fraction(3601, 30),
    )
    assert (chosen.start, chosen.frames) == (0, 3601)


def test_real_overrun_of_one_frame_still_fails() -> None:
    with pytest.raises(ReplayError, match="exceeds the 3601"):
        frame_grid.select(**REAL, start_seconds=113.57, duration_seconds=6.5)


def test_media_shorter_than_half_a_frame_or_longer_than_grid_fails() -> None:
    with pytest.raises(ReplayError, match="differs from the frame grid"):
        frame_grid.select(
            **{**REAL, "media_ticks": 10_802_880 - 3000}, start_seconds=0, duration_seconds=1
        )
    with pytest.raises(ReplayError, match="differs from the frame grid"):
        frame_grid.select(
            **{**REAL, "media_ticks": 10_803_100}, start_seconds=0, duration_seconds=1
        )


def test_invalid_inputs_fail() -> None:
    with pytest.raises(ReplayError, match="positive"):
        frame_grid.select(**{**REAL, "rate": 0}, start_seconds=0, duration_seconds=1)
    with pytest.raises(ReplayError, match="cover a frame"):
        frame_grid.select(**REAL, start_seconds=0, duration_seconds=0)
    with pytest.raises(ReplayError, match="at or after zero"):
        frame_grid.select(**REAL, start_seconds=-1, duration_seconds=1)
