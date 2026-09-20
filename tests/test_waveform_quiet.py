"""Sample-clock validation and independent quiet evidence, never ASR inference."""

import pytest

from replay.caption_projection import MeasuredQuietSpan
from replay.waveform_quiet import measured_quiet_runs


def test_consecutive_full_quiet_windows_merge() -> None:
    result = measured_quiet_runs(
        [
            "frame:0 pts:0 pts_time:0\n",
            "lavfi.astats.Overall.Peak_level=-61\n",
            "frame:1 pts:882 pts_time:0.020000\n",
            "lavfi.astats.Overall.Peak_level=-inf\n",
        ],
        sample_count=1764,
    )
    assert result == (MeasuredQuietSpan(0, 0.04),)


def test_short_tail_is_not_padded_to_quiet() -> None:
    result = measured_quiet_runs(
        [
            "frame:0 pts:0 pts_time:0",
            "lavfi.astats.Overall.Peak_level=-61",
            "frame:1 pts:882 pts_time:0.02",
            "lavfi.astats.Overall.Peak_level=-inf",
        ],
        sample_count=1000,
    )
    assert result == (MeasuredQuietSpan(0, 0.02),)


def test_threshold_equality_is_not_below_threshold() -> None:
    assert (
        measured_quiet_runs(
            ["frame:0 pts:0 pts_time:0", "lavfi.astats.Overall.Peak_level=-60"],
            sample_count=882,
        )
        == ()
    )


def test_loud_gap_separates_quiet_runs() -> None:
    result = measured_quiet_runs(
        [
            "frame:0 pts:0 pts_time:0",
            "lavfi.astats.Overall.Peak_level=-61",
            "frame:1 pts:882 pts_time:0.02",
            "lavfi.astats.Overall.Peak_level=-10",
            "frame:2 pts:1764 pts_time:0.04",
            "lavfi.astats.Overall.Peak_level=-62",
        ],
        sample_count=2646,
    )
    assert result == (MeasuredQuietSpan(0, 0.02), MeasuredQuietSpan(0.04, 0.06))


@pytest.mark.parametrize("count", [0, -1, True])
def test_invalid_sample_count(count: int) -> None:
    with pytest.raises(ValueError, match="policy"):
        measured_quiet_runs([], sample_count=count)


@pytest.mark.parametrize("rate", [0, 7999, 192001, True])
def test_invalid_sample_rate(rate: int) -> None:
    with pytest.raises(ValueError, match="policy"):
        measured_quiet_runs([], sample_count=1, sample_rate=rate)


@pytest.mark.parametrize("window", [0, 44101, True])
def test_invalid_window(window: int) -> None:
    with pytest.raises(ValueError, match="policy"):
        measured_quiet_runs([], sample_count=1, window_samples=window)


@pytest.mark.parametrize("threshold", [float("nan"), float("inf"), -161, 1, False])
def test_invalid_threshold(threshold: float) -> None:
    with pytest.raises(ValueError, match="policy"):
        measured_quiet_runs([], sample_count=1, threshold_dbfs=threshold)


@pytest.mark.parametrize("minimum", [float("nan"), 0, 2, True])
def test_invalid_minimum(minimum: float) -> None:
    with pytest.raises(ValueError, match="policy"):
        measured_quiet_runs([], sample_count=1, minimum_seconds=minimum)


def test_measurement_budget() -> None:
    with pytest.raises(ValueError, match="budget"):
        measured_quiet_runs([], sample_count=1_100_001, window_samples=1)


@pytest.mark.parametrize(
    "header",
    ["bad", "x" * 161, "frame:1 pts:0 pts_time:0", "frame:0 pts:1 pts_time:0"],
)
def test_invalid_header(header: str) -> None:
    with pytest.raises(ValueError, match="clock"):
        measured_quiet_runs([header], sample_count=882)


@pytest.mark.parametrize("peak", ["", "x" * 161, "wrong-key=-90"])
def test_missing_peak(peak: str) -> None:
    with pytest.raises(ValueError, match="Missing"):
        measured_quiet_runs(["frame:0 pts:0 pts_time:0", peak], sample_count=882)


@pytest.mark.parametrize("peak", ["nan", "inf", "-1e999", "not-a-number"])
def test_invalid_peak(peak: str) -> None:
    with pytest.raises(ValueError):
        measured_quiet_runs(
            ["frame:0 pts:0 pts_time:0", f"lavfi.astats.Overall.Peak_level={peak}"],
            sample_count=882,
        )


def test_extra_frame_rejected() -> None:
    with pytest.raises(ValueError, match="extra"):
        measured_quiet_runs(
            [
                "frame:0 pts:0 pts_time:0",
                "lavfi.astats.Overall.Peak_level=-61",
                "frame:1 pts:882 pts_time:0.02",
            ],
            sample_count=882,
        )


def test_missing_frame_rejected() -> None:
    with pytest.raises(ValueError, match="incomplete"):
        measured_quiet_runs([], sample_count=882)
