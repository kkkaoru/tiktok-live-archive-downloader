from fractions import Fraction

import pytest

from replay.audio_clock import AudioFrameClock
from replay.caption_io import MLXWordGroup


def test_native_gap_not_flattened_and_boundary_sides_preserved() -> None:
    clock = AudioFrameClock((50, 150, 400), (100, 100, 100), 10000, Fraction(1, 10000))
    assert clock.sample_ends == (100, 200, 300)
    assert clock.source_time(0) == Fraction(1, 200)
    assert clock.source_time(0.02) == Fraction(1, 25)
    assert clock.source_time(0.02, end=True) == Fraction(1, 40)
    assert clock.source_time(0.03) == Fraction(1, 20)
    assert clock.source_interval(0.015, 0.025) == (Fraction(1, 50), Fraction(9, 200))


def test_probe_decode_samples_not_packet_durations() -> None:
    clock = AudioFrameClock.from_probe(
        {
            "streams": [{"sample_rate": "48000", "time_base": "1/48000"}],
            "frames": [{"pts": 26880, "nb_samples": 1024}, {"pts": 27840, "nb_samples": 1024}],
        }
    )
    assert clock.source_time(0.032) == Fraction(443, 750)
    assert clock.source_interval(0.001, 0.02) == (Fraction(561, 1000), Fraction(29, 50))


@pytest.mark.parametrize("point", [float("nan"), float("inf"), -0.1, 0.031])
def test_out_of_support_clock_rejected(point: float) -> None:
    clock = AudioFrameClock((0,), (300,), 10000, Fraction(1, 10000))
    with pytest.raises(ValueError):
        clock.source_time(point)


def test_projection_retains_unresolved_points_and_original_provenance() -> None:
    clock = AudioFrameClock((50, 150, 400), (100, 100, 100), 10000, Fraction(1, 10000))
    groups = (
        MLXWordGroup(
            text="日本語",
            start=0.005,
            end=0.015,
            source_segments=(0,),
            piece_count=2,
            minimum_probability=0.9,
        ),
        MLXWordGroup(
            text="未確定",
            start=0.02,
            end=0.02,
            source_segments=(1,),
            piece_count=1,
            minimum_probability=0.8,
        ),
    )
    result = clock.project_groups(groups)
    assert result == (
        MLXWordGroup(
            text="日本語",
            start=0.01,
            end=0.02,
            source_segments=(0,),
            piece_count=2,
            minimum_probability=0.9,
        ),
        MLXWordGroup(
            text="未確定",
            start=0.04,
            end=0.04,
            source_segments=(1,),
            piece_count=1,
            minimum_probability=0.8,
        ),
    )
    assert groups[1].start == 0.02
    with pytest.raises(ValueError, match="onsets"):
        clock.project_groups((groups[1], groups[0]))


def test_overlap_cannot_invent_positive_word_support() -> None:
    clock = AudioFrameClock((0, 1), (100, 100), 10000, Fraction(1, 10000))
    with pytest.raises(ValueError, match="overlap"):
        clock.source_interval(0.009, 0.011)
    with pytest.raises(ValueError, match="positive support"):
        clock.source_interval(0.01, 0.01)


@pytest.mark.parametrize(
    "value",
    [
        {},
        {"streams": [], "frames": []},
        {"streams": [{}], "frames": []},
        {"streams": [{}], "frames": [{}]},
        {"streams": [{"sample_rate": "true"}], "frames": [{}]},
        {"streams": [{"sample_rate": "48000"}], "frames": [{}]},
        {"streams": [{"sample_rate": "48000", "time_base": "1/48000"}], "frames": [{}]},
        {
            "streams": [{"sample_rate": "48000", "time_base": "1/48000"}],
            "frames": [{"pts": False, "nb_samples": 1024}],
        },
        {
            "streams": [{"sample_rate": "48000", "time_base": "-1/48000"}],
            "frames": [{"pts": 0, "nb_samples": 1024}],
        },
    ],
)
def test_invalid_probe_rejected(value: object) -> None:
    with pytest.raises(ValueError):
        AudioFrameClock.from_probe(value)


@pytest.mark.parametrize(
    ("pts", "samples", "rate", "base"),
    [
        ((), (), 48000, Fraction(1, 48000)),
        ((0,), (), 48000, Fraction(1, 48000)),
        ((0,), (0,), 48000, Fraction(1, 48000)),
        ((-1,), (1024,), 48000, Fraction(1, 48000)),
        ((0, 0), (1024, 1024), 48000, Fraction(1, 48000)),
        ((0,), (1024,), 1, Fraction(1, 48000)),
        ((0,), (1024,), 48000, Fraction(0)),
    ],
)
def test_invalid_clock_rejected(
    pts: tuple[int, ...], samples: tuple[int, ...], rate: int, base: Fraction
) -> None:
    with pytest.raises(ValueError):
        AudioFrameClock(pts, samples, rate, base)
