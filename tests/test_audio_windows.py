"""Synthetic sample-domain regression cases, without recording fixtures."""

import pytest

from replay.audio_windows import AudioWindow, overlap_samples, plan_audio_windows


def test_short_input_has_one_unpadded_window() -> None:
    assert plan_audio_windows(1, window_samples=10, stride_samples=8) == (AudioWindow(0, 1),)


def test_exact_window_has_no_redundant_tail() -> None:
    assert plan_audio_windows(10, window_samples=10, stride_samples=8) == (AudioWindow(0, 10),)


def test_one_sample_beyond_window_keeps_the_tail() -> None:
    assert plan_audio_windows(11, window_samples=10, stride_samples=8) == (
        AudioWindow(0, 10),
        AudioWindow(8, 3),
    )


def test_exact_two_window_coverage_avoids_contained_third_window() -> None:
    assert plan_audio_windows(18, window_samples=10, stride_samples=8) == (
        AudioWindow(0, 10),
        AudioWindow(8, 10),
    )


def test_tail_shorter_than_nominal_overlap_is_not_planned_again() -> None:
    assert plan_audio_windows(17, window_samples=10, stride_samples=8) == (
        AudioWindow(0, 10),
        AudioWindow(8, 9),
    )


def test_no_overlap_and_final_partial_window() -> None:
    assert plan_audio_windows(21, window_samples=10, stride_samples=10) == (
        AudioWindow(0, 10),
        AudioWindow(10, 10),
        AudioWindow(20, 1),
    )


def test_join_overlap_matches_source_samples() -> None:
    assert overlap_samples(AudioWindow(0, 10), AudioWindow(8, 3)) == 2


def test_touching_windows_have_zero_overlap() -> None:
    assert overlap_samples(AudioWindow(0, 10), AudioWindow(10, 3)) == 0


@pytest.mark.parametrize("value", [0, -1, True])
def test_invalid_sample_count(value: int) -> None:
    with pytest.raises(ValueError, match="positive integers"):
        plan_audio_windows(value, window_samples=10, stride_samples=8)


@pytest.mark.parametrize("value", [0, -1, True])
def test_invalid_window_size(value: int) -> None:
    with pytest.raises(ValueError, match="positive integers"):
        plan_audio_windows(20, window_samples=value, stride_samples=8)


@pytest.mark.parametrize("value", [0, -1, True])
def test_invalid_stride(value: int) -> None:
    with pytest.raises(ValueError, match="positive integers"):
        plan_audio_windows(20, window_samples=10, stride_samples=value)


@pytest.mark.parametrize("value", [0, -1, True])
def test_invalid_budget(value: int) -> None:
    with pytest.raises(ValueError, match="positive integers"):
        plan_audio_windows(20, window_samples=10, stride_samples=8, maximum_windows=value)


def test_stride_cannot_leave_gaps() -> None:
    with pytest.raises(ValueError, match="uncovered"):
        plan_audio_windows(20, window_samples=10, stride_samples=11)


def test_window_budget_is_checked_before_allocation() -> None:
    with pytest.raises(ValueError, match="budget"):
        plan_audio_windows(20, window_samples=10, stride_samples=8, maximum_windows=2)


def test_exact_budget_is_allowed() -> None:
    assert plan_audio_windows(11, window_samples=10, stride_samples=8, maximum_windows=2) == (
        AudioWindow(0, 10),
        AudioWindow(8, 3),
    )


@pytest.mark.parametrize("start", [-1, True])
def test_invalid_window_start(start: int) -> None:
    with pytest.raises(ValueError, match="nonnegative"):
        AudioWindow(start, 10)


@pytest.mark.parametrize("count", [0, -1, True])
def test_invalid_window_count(count: int) -> None:
    with pytest.raises(ValueError, match="positive"):
        AudioWindow(0, count)


@pytest.mark.parametrize(
    "right",
    [AudioWindow(0, 11), AudioWindow(0, 10), AudioWindow(8, 2), AudioWindow(8, 1)],
)
def test_nonadvancing_or_contained_windows_are_rejected(right: AudioWindow) -> None:
    with pytest.raises(ValueError, match="advance"):
        overlap_samples(AudioWindow(0, 10), right)


def test_reversed_windows_are_rejected() -> None:
    with pytest.raises(ValueError, match="advance"):
        overlap_samples(AudioWindow(8, 10), AudioWindow(0, 10))


def test_gap_between_windows_is_rejected() -> None:
    with pytest.raises(ValueError, match="uncovered"):
        overlap_samples(AudioWindow(0, 10), AudioWindow(11, 10))
