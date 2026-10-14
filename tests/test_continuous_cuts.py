"""Synthetic native clocks; selecting samples must not repair their timing."""

from fractions import Fraction

import pytest

from replay.audio_clock import AudioFrameClock
from replay.continuous_cuts import retained_pcm_spans


def test_native_gap_and_missing_tail_do_not_add_silence() -> None:
    clock = AudioFrameClock((0, 150), (800, 800), 8000, Fraction(1, 1000))
    assert retained_pcm_spans(
        clock, ((Fraction(0), Fraction(1)),), output_rate=8000, output_samples=1600
    ) == ((0, 1600),)


def test_native_overlap_keeps_distinct_original_samples() -> None:
    clock = AudioFrameClock((0, 50), (800, 800), 8000, Fraction(1, 1000))
    assert retained_pcm_spans(
        clock,
        ((Fraction(3, 50), Fraction(9, 100)),),
        output_rate=8000,
        output_samples=1600,
    ) == ((480, 720), (880, 1120))


def test_full_overlap_selection_preserves_all_samples() -> None:
    clock = AudioFrameClock((0, 50), (800, 800), 8000, Fraction(1, 1000))
    assert retained_pcm_spans(
        clock, ((Fraction(0), Fraction(3, 20)),), output_rate=8000, output_samples=1600
    ) == ((0, 1600),)


def test_real_cut_omits_only_selected_sample_gap() -> None:
    clock = AudioFrameClock((0,), (800,), 8000, Fraction(1, 1000))
    assert retained_pcm_spans(
        clock,
        ((Fraction(0), Fraction(1, 25)), (Fraction(3, 50), Fraction(1, 10))),
        output_rate=8000,
        output_samples=800,
    ) == ((0, 320), (480, 800))


def test_fixed_resampling_and_outward_edge_rounding() -> None:
    clock = AudioFrameClock((0,), (800,), 8000, Fraction(1, 1000))
    assert retained_pcm_spans(
        clock,
        ((Fraction(1, 100000), Fraction(2, 100000)),),
        output_rate=12000,
        output_samples=1200,
    ) == ((0, 1),)


def test_fractional_resample_eof_is_bounded_by_actual_samples() -> None:
    clock = AudioFrameClock((0,), (801,), 8000, Fraction(1, 1000))
    assert retained_pcm_spans(
        clock, ((Fraction(0), Fraction(1)),), output_rate=12000, output_samples=1201
    ) == ((0, 1201),)


def test_touching_retained_ranges_merge_without_duplicate_samples() -> None:
    clock = AudioFrameClock((0,), (800,), 8000, Fraction(1, 1000))
    assert retained_pcm_spans(
        clock,
        ((Fraction(0), Fraction(1, 25)), (Fraction(1, 25), Fraction(2, 25))),
        output_rate=8000,
        output_samples=800,
    ) == ((0, 640),)


def test_presentation_gap_has_no_samples_to_invent() -> None:
    clock = AudioFrameClock((0, 150), (800, 800), 8000, Fraction(1, 1000))
    with pytest.raises(ValueError, match="no decoded audio"):
        retained_pcm_spans(
            clock,
            ((Fraction(11, 100), Fraction(14, 100)),),
            output_rate=8000,
            output_samples=1600,
        )


@pytest.mark.parametrize(
    "rate,samples", [(True, 800), (7999, 800), (192001, 800), (8000, True), (8000, 0), (8000, 802)]
)
def test_changed_or_invalid_continuous_clock_is_refused(rate: int, samples: int) -> None:
    clock = AudioFrameClock((0,), (800,), 8000, Fraction(1, 1000))
    with pytest.raises(ValueError, match="Invalid uncorrected PCM"):
        retained_pcm_spans(
            clock, ((Fraction(0), Fraction(1)),), output_rate=rate, output_samples=samples
        )


@pytest.mark.parametrize(
    "intervals",
    [
        (),
        ((Fraction(-1), Fraction(1)),),
        ((Fraction(1), Fraction(1)),),
        ((Fraction(0), Fraction(21601)),),
        ((Fraction(0), Fraction(2)), (Fraction(1), Fraction(3))),
    ],
)
def test_invalid_ranges_are_refused(intervals: tuple[tuple[Fraction, Fraction], ...]) -> None:
    clock = AudioFrameClock((0,), (800,), 8000, Fraction(1, 1000))
    with pytest.raises(ValueError, match="Invalid uncorrected PCM"):
        retained_pcm_spans(clock, intervals, output_rate=8000, output_samples=800)
