from fractions import Fraction
from itertools import chain

import pytest

from replay.tracking_plan import tracking_windows


def test_vfr_windows_use_following_measured_pts_and_exact_tail() -> None:
    windows = tracking_windows(
        (0, 900, 1190, 1200, 1900, 2400), time_base=Fraction(1, 10), last_duration=10
    )
    assert len(windows) == 3
    assert windows[0].pts == (0, 900, 1190)
    assert windows[0].last_duration == 10
    assert windows[1].pts == (1200, 1900)
    assert windows[1].last_duration == 500
    assert windows[2].pts == (2400,)
    assert windows[2].last_duration == 10
    assert windows[2].time_base == Fraction(1, 10)


def test_full_source_over_clock_limit_keeps_every_frame_in_bounded_windows() -> None:
    windows = tracking_windows(tuple(range(50001)), time_base=Fraction(1, 1000), last_duration=1)
    assert len(windows) == 7
    assert len(windows[0].pts) == 7200
    assert windows[0].pts[:2] == (0, 1)
    assert windows[0].pts[-1] == 7199
    assert windows[1].pts[0] == 7200
    assert windows[-1].pts[0] == 43200
    assert windows[-1].pts[-2:] == (49999, 50000)
    assert len(tuple(chain.from_iterable(window.pts for window in windows))) == 50001
    assert windows[-1].last_duration == 1


def test_non_unit_time_base_is_not_rounded_to_a_nominal_frame_rate() -> None:
    windows = tracking_windows((0, 3596, 3597), time_base=Fraction(1001, 30000), last_duration=1)
    assert len(windows) == 2
    assert windows[0].pts == (0,)
    assert windows[0].last_duration == 3596
    assert windows[1].pts == (3596, 3597)
    assert windows[1].last_duration == 1


def test_source_offset_and_single_frame_are_preserved() -> None:
    windows = tracking_windows((301,), time_base=Fraction(1, 90000), last_duration=1014)
    assert len(windows) == 1
    assert windows[0].pts == (301,)
    assert windows[0].last_duration == 1014
    assert windows[0].time_base == Fraction(1, 90000)


def test_seeded_vfr_windows_share_one_complete_seed_frame() -> None:
    windows = tracking_windows(
        (10, 15, 30, 40, 55, 70),
        time_base=Fraction(1),
        last_duration=5,
        seed_pts=(10, 30, 40, 55),
        maximum_seconds=Fraction(40),
    )
    assert tuple(window.pts for window in windows) == ((10, 15, 30), (30, 40, 55), (55, 70))
    assert tuple(window.last_duration for window in windows) == (10, 15, 5)


def test_seed_overlap_counts_against_frame_budget_for_long_source() -> None:
    windows = tracking_windows(
        tuple(range(50001)),
        time_base=Fraction(1, 1000),
        last_duration=1,
        seed_pts=tuple(range(0, 50001, 7000)),
    )
    assert tuple(len(window.pts) for window in windows) == (
        7001,
        7001,
        7001,
        7001,
        7001,
        7001,
        7001,
        1001,
    )
    assert tuple(window.pts[0] for window in windows) == (
        0,
        7000,
        14000,
        21000,
        28000,
        35000,
        42000,
        49000,
    )
    assert windows[-1].pts[-1] == 50000


def test_seeded_single_frame_and_non_unit_clock() -> None:
    assert (
        tracking_windows(
            (5,),
            time_base=Fraction(2, 3),
            last_duration=3,
            seed_pts=(5,),
            maximum_seconds=Fraction(2),
        )[0].last_duration
        == 3
    )


@pytest.mark.parametrize("seeds", [(), (1,), (0, 0), (0, 2, 1), (False,), (0, 1.5), [0]])
def test_invalid_face_seed_vectors(seeds: tuple[int, ...]) -> None:
    with pytest.raises(ValueError, match="Face seeds"):
        tracking_windows((0, 1, 2), time_base=Fraction(1), last_duration=1, seed_pts=seeds)


@pytest.mark.parametrize("seeds", [(0, 1), (0, 5)])
def test_off_grid_seed_is_not_rounded(seeds: tuple[int, ...]) -> None:
    with pytest.raises(ValueError, match="match decoded"):
        tracking_windows((0, 2, 4), time_base=Fraction(1), last_duration=1, seed_pts=seeds)


@pytest.mark.parametrize("budget", [Fraction(0), Fraction(-1), Fraction(121), 90, True])
def test_invalid_operational_budget(budget: Fraction) -> None:
    with pytest.raises(ValueError, match="duration budget"):
        tracking_windows((0,), time_base=Fraction(1), last_duration=1, maximum_seconds=budget)


@pytest.mark.parametrize("tail", [1, 5])
def test_seed_start_fits_but_full_seed_extent_does_not(tail: int) -> None:
    with pytest.raises(ValueError, match="No advancing face seed"):
        tracking_windows(
            (0, 89, 90),
            time_base=Fraction(1),
            last_duration=tail,
            seed_pts=(0, 90),
            maximum_seconds=Fraction(90),
        )


def test_terminal_extent_requires_seeded_split() -> None:
    windows = tracking_windows(
        (0, 60, 89),
        time_base=Fraction(1),
        last_duration=2,
        seed_pts=(0, 60),
        maximum_seconds=Fraction(90),
    )
    assert tuple(window.pts for window in windows) == ((0, 60), (60, 89))
    assert tuple(window.last_duration for window in windows) == (29, 2)


@pytest.mark.parametrize("pts", [(), (-1, 2), (0, 0), (2, 1), (False, 1), (0, 1.5), [0, 1]])
def test_invalid_source_pts_are_rejected(pts: tuple[int, ...]) -> None:
    with pytest.raises(ValueError, match="Source PTS"):
        tracking_windows(pts, time_base=Fraction(1), last_duration=1)


@pytest.mark.parametrize("time_base", [Fraction(0), Fraction(-1), Fraction(1, 1000000001), 0.5])
def test_existing_time_base_contract_is_not_weakened(time_base: Fraction) -> None:
    with pytest.raises(ValueError, match="time base"):
        tracking_windows((0,), time_base=time_base, last_duration=1)


@pytest.mark.parametrize("last_duration", [0, -1, True, 0.5])
def test_terminal_extent_must_be_measured_positive_ticks(last_duration: int) -> None:
    with pytest.raises(ValueError, match="frame clocks"):
        tracking_windows((0,), time_base=Fraction(1), last_duration=last_duration)


def test_source_extent_over_existing_clock_limit_is_rejected() -> None:
    with pytest.raises(ValueError, match="time base or extent"):
        tracking_windows((86400,), time_base=Fraction(1), last_duration=1)


@pytest.mark.parametrize("pts,last_duration", [((0, 121), 1), ((0,), 121)])
def test_a_frame_longer_than_native_budget_is_not_truncated(
    pts: tuple[int, ...], last_duration: int
) -> None:
    with pytest.raises(ValueError, match="frame extent exceeds"):
        tracking_windows(pts, time_base=Fraction(1), last_duration=last_duration)
