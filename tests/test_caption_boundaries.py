import math

import pytest

from replay.caption_boundaries import AudioSpan, BoundaryPolicy, protect_boundaries
from replay.caption_projection import CutSpan
from replay.caption_timing import SpokenToken


def test_weak_word_onset_is_kept_without_relaxing_speech_classification() -> None:
    cores = (AudioSpan(1, 2), AudioSpan(4, 5))
    result = protect_boundaries(
        cores,
        (SpokenToken("子音から始まる語", 0.94, 1.2, "a"), SpokenToken("次の語", 4, 4.5, "b")),
        (AudioSpan(0.8, 0.9), AudioSpan(2.2, 3.5)),
        source_frames=180,
    )
    assert result.cuts == (CutSpan(27, 60, 0), CutSpan(120, 150, 33))
    assert result.speech_cores is cores
    assert result.added_context_frames == 3
    assert result.unresolved_words == ()


def test_frame_quantization_cannot_round_away_the_start_of_speech() -> None:
    result = protect_boundaries((AudioSpan(0.999, 1.999),), (), (), source_frames=90)
    assert result.cuts == (CutSpan(29, 60, 0),)
    assert result.added_context_frames == 0


def test_no_blanket_padding_or_hallucinated_words_in_silence() -> None:
    result = protect_boundaries(
        (AudioSpan(1, 2),),
        (SpokenToken("無音中の幻覚", 3, 4, "unconfirmed"),),
        (AudioSpan(0, 0.5), AudioSpan(0.98, 0.99)),
        source_frames=150,
    )
    assert result.cuts == (CutSpan(30, 60, 0),)
    assert result.added_context_frames == 0


def test_end_of_a_confirmed_word_is_not_truncated() -> None:
    result = protect_boundaries(
        (AudioSpan(1, 2),), (SpokenToken("語尾", 1.5, 2.09, "a"),), (), source_frames=90
    )
    assert result.cuts == (CutSpan(30, 63, 0),)
    assert result.added_context_frames == 3


def test_exact_extension_at_large_timestamp_survives_bound_rounding() -> None:
    result = protect_boundaries(
        (AudioSpan(7200, 7200.4),),
        (SpokenToken("ending", 7200.2, 7200.8, "a"),),
        (),
        source_frames=216060,
    )
    assert result.cuts == (CutSpan(216000, 216024, 0),)
    assert result.unresolved_words == ()
    assert result.added_context_frames == 12


def test_exact_start_extension_survives_subtraction_rounding() -> None:
    result = protect_boundaries(
        (AudioSpan(7200.8, 7201.6),),
        (SpokenToken("onset", 7200.4, 7201, "a"),),
        (),
        source_frames=216060,
    )
    assert result.cuts == (CutSpan(216012, 216048, 0),)
    assert result.unresolved_words == ()
    assert result.added_context_frames == 12


def test_rounding_compensation_rejects_a_microsecond_end_overrun() -> None:
    result = protect_boundaries(
        (AudioSpan(7200, 7200.4),),
        (SpokenToken("late", 7200.2, 7200.800001, "a"),),
        (),
        source_frames=216060,
    )
    assert result.cuts == (CutSpan(216000, 216012, 0),)
    assert result.unresolved_words == (SpokenToken("late", 7200.2, 7200.800001, "a"),)
    assert result.added_context_frames == 0


def test_rounding_compensation_rejects_a_microsecond_start_overrun() -> None:
    result = protect_boundaries(
        (AudioSpan(7200.8, 7201.6),),
        (SpokenToken("early", 7200.399999, 7201, "a"),),
        (),
        source_frames=216060,
    )
    assert result.cuts == (CutSpan(216024, 216048, 0),)
    assert result.unresolved_words == (SpokenToken("early", 7200.399999, 7201, "a"),)
    assert result.added_context_frames == 0


def test_zero_extension_policy_does_not_add_a_rounding_allowance() -> None:
    result = protect_boundaries(
        (AudioSpan(1, 2),),
        (SpokenToken("outside", 1.5, math.nextafter(2, math.inf), "a"),),
        (),
        source_frames=90,
        policy=BoundaryPolicy(maximum_word_extension=0),
    )
    assert result.cuts == (CutSpan(30, 60, 0),)
    assert result.unresolved_words == (SpokenToken("outside", 1.5, 2.0000000000000004, "a"),)
    assert result.added_context_frames == 0


def test_guard_overlap_merges_only_touching_retained_intervals() -> None:
    result = protect_boundaries(
        (AudioSpan(1, 1.49), AudioSpan(1.5, 2)),
        (SpokenToken("連続した語", 1.4, 1.7, "a"),),
        (),
        source_frames=90,
    )
    assert result.cuts == (CutSpan(30, 60, 0),)
    assert result.added_context_frames == 0


def test_coarse_alignment_is_reported_instead_of_retaining_the_entire_gap() -> None:
    word = SpokenToken("長すぎる認識", 0, 5, "a")
    result = protect_boundaries((AudioSpan(1, 2), AudioSpan(3, 4)), (word,), (), source_frames=180)
    assert result.cuts == (CutSpan(30, 60, 0), CutSpan(90, 120, 30))
    assert result.unresolved_words == (word,)
    assert result.added_context_frames == 0


@pytest.mark.parametrize(("start", "end"), [(math.nan, 1), (-1, 1), (0, 0)])
def test_invalid_boundary_span_is_rejected(start: float, end: float) -> None:
    with pytest.raises(ValueError):
        AudioSpan(start, end)


@pytest.mark.parametrize(
    ("extension", "guard", "quiet"), [(2, 0.08, 0.02), (0.4, math.nan, 0.02), (0.4, 0.08, 0.2)]
)
def test_boundary_policy_is_bounded(extension: float, guard: float, quiet: float) -> None:
    with pytest.raises(ValueError):
        BoundaryPolicy(extension, guard, quiet)


def test_invalid_timeline_and_word_evidence_are_rejected() -> None:
    with pytest.raises(ValueError, match="frame clock"):
        protect_boundaries((), (), (), source_frames=0)
    with pytest.raises(ValueError, match="budget"):
        protect_boundaries((AudioSpan(0, 1),) * 3001, (), (), source_frames=90)
    with pytest.raises(ValueError, match="nonoverlapping"):
        protect_boundaries((AudioSpan(0, 2), AudioSpan(1, 3)), (), (), source_frames=90)
    with pytest.raises(ValueError, match="Word timing"):
        protect_boundaries((), (SpokenToken("範囲外", 0, 4, "a"),), (), source_frames=90)
    with pytest.raises(ValueError, match="Word timing"):
        protect_boundaries(
            (), (SpokenToken("後", 1, 2, "a"), SpokenToken("前", 0, 1, "a")), (), source_frames=90
        )


def test_overlap_work_budget_stops_pathological_alignment() -> None:
    cores = tuple(AudioSpan(float(i), i + 0.5) for i in range(6))
    words = (SpokenToken("長すぎる認識", 0, 6, "a"),) * 100_000
    with pytest.raises(ValueError, match="work budget"):
        protect_boundaries(cores, words, (), source_frames=300)
