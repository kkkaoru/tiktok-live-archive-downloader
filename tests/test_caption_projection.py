import math

import pytest

from replay.caption_projection import (
    CutSpan,
    MeasuredQuietSpan,
    TokenProjection,
    project_tokens,
    project_tokens_over_quiet,
)
from replay.caption_timing import SpokenToken, make_captions


def test_jump_cut_does_not_carry_old_recognition_into_new_utterance() -> None:
    source = (
        SpokenToken("前の話", 0.1, 0.9, "chunk"),
        SpokenToken("除外する話", 1.2, 1.8, "chunk"),
        SpokenToken("新しい話", 2, 2.8, "chunk"),
    )
    result = project_tokens(source, (CutSpan(0, 30, 0), CutSpan(60, 90, 30)))
    assert result.tokens == (
        SpokenToken("前の話", 0.1, 0.9, "chunk:cut-0"),
        SpokenToken("新しい話", 1, 1.7999999999999998, "chunk:cut-1"),
    )
    assert result.omitted == (source[1],)
    assert result.boundary_conflicts == ()
    assert tuple(cue.text for cue in make_captions(result.tokens)) == ("前の話", "新しい話")


def test_coarse_old_phrase_crossing_cut_is_reported_not_copied() -> None:
    token = SpokenToken("古い長い認識文を持ち越さない", 0.5, 2.5, "chunk")
    assert project_tokens((token,), (CutSpan(30, 60, 0),)) == TokenProjection((), (), (token,))


def test_only_one_frame_of_quantization_tolerance_is_allowed() -> None:
    result = project_tokens((SpokenToken("境界", 0.98, 2.02, "chunk"),), (CutSpan(30, 60, 0),))
    assert result.tokens == (SpokenToken("境界", 0, 1, "chunk:cut-0"),)
    assert result.boundary_conflicts == ()


def test_before_first_cut_and_empty_timeline_remain_reviewable() -> None:
    token = SpokenToken("除外", 0, 0.5, "chunk")
    assert project_tokens((token,), (CutSpan(30, 60, 0),)) == TokenProjection((), (token,), ())
    assert project_tokens((token,), ()) == TokenProjection((), (token,), ())
    assert project_tokens((), ()) == TokenProjection((), (), ())


@pytest.mark.parametrize(
    "cuts",
    [
        (CutSpan(0, 30, 1),),
        (CutSpan(0, 30, 0), CutSpan(20, 40, 30)),
        (CutSpan(0, 30, 0), CutSpan(60, 90, 31)),
    ],
)
def test_noncontiguous_or_overlapping_timeline_is_rejected(cuts: tuple[CutSpan, ...]) -> None:
    with pytest.raises(ValueError, match="timeline"):
        project_tokens((), cuts)


def test_invalid_tokens_and_projection_budget_are_rejected() -> None:
    with pytest.raises(ValueError, match="ordered"):
        project_tokens((SpokenToken("後", 1, 2, "a"), SpokenToken("前", 0, 1, "a")), ())
    with pytest.raises(ValueError, match="budget"):
        project_tokens((), (), fps=0)


@pytest.mark.parametrize(("start", "end", "output"), [(0, 0, 0), (-1, 30, 0), (0, 30, -1)])
def test_cut_spans_validate_positions(start: int, end: int, output: int) -> None:
    with pytest.raises(ValueError):
        CutSpan(start, end, output)


def test_measured_internal_quiet_maps_one_label_without_character_splitting() -> None:
    result = project_tokens_over_quiet(
        (SpokenToken("one whole word", 0, 3, "a"),),
        (CutSpan(0, 30, 0), CutSpan(60, 90, 30)),
        (MeasuredQuietSpan(1, 1.5), MeasuredQuietSpan(1.5, 2)),
    )
    assert result == TokenProjection(
        (SpokenToken("one whole word", 0, 2, "a:quiet-cuts-0-1"),), (), ()
    )


def test_only_measured_quiet_may_be_removed_from_native_interval_edges() -> None:
    result = project_tokens_over_quiet(
        (SpokenToken("word", 0.5, 3.5, "a"),),
        (CutSpan(30, 90, 0),),
        (MeasuredQuietSpan(0.5, 1), MeasuredQuietSpan(3, 3.5)),
    )
    assert result == TokenProjection((SpokenToken("word", 0, 2, "a:quiet-cuts-0-0"),), (), ())


def test_quiet_projection_does_not_assume_an_unmeasured_prefix_is_silence() -> None:
    result = project_tokens_over_quiet(
        (SpokenToken("word", 0.5, 2, "a"),), (CutSpan(30, 60, 0),), ()
    )
    assert result == TokenProjection((), (), (SpokenToken("word", 0.5, 2, "a"),))


def test_unmeasured_tail_and_partial_quiet_evidence_remain_conflicts() -> None:
    result = project_tokens_over_quiet(
        (SpokenToken("word", 0, 3, "a"),),
        (CutSpan(0, 30, 0), CutSpan(60, 90, 30)),
        (MeasuredQuietSpan(1, 1.9),),
    )
    assert result == TokenProjection((), (), (SpokenToken("word", 0, 3, "a"),))
    assert project_tokens_over_quiet(
        (SpokenToken("tail", 0, 2, "a"),), (CutSpan(0, 30, 0),), ()
    ) == TokenProjection((), (), (SpokenToken("tail", 0, 2, "a"),))


def test_wholly_unretained_nonquiet_words_are_not_silently_omitted() -> None:
    assert project_tokens_over_quiet(
        (SpokenToken("missing", 1, 2, "a"),), (CutSpan(0, 15, 0),), ()
    ) == TokenProjection((), (), (SpokenToken("missing", 1, 2, "a"),))
    assert project_tokens_over_quiet(
        (SpokenToken("quiet alternative", 1, 2, "a"),), (), (MeasuredQuietSpan(1, 2),)
    ) == TokenProjection((), (SpokenToken("quiet alternative", 1, 2, "a"),), ())


def test_wholly_retained_word_needs_no_quiet_certificate() -> None:
    assert project_tokens_over_quiet(
        (SpokenToken("kept", 2, 2.5, "a"),),
        (CutSpan(0, 30, 0), CutSpan(60, 90, 30)),
        (),
    ) == TokenProjection((SpokenToken("kept", 1, 1.5, "a:quiet-cuts-1-1"),), (), ())


@pytest.mark.parametrize(("start", "end"), [(math.nan, 1), (0, math.inf), (-1, 1), (1, 1)])
def test_measured_quiet_intervals_are_finite_and_positive(start: float, end: float) -> None:
    with pytest.raises(ValueError, match="quiet interval"):
        MeasuredQuietSpan(start, end)


def test_quiet_projection_validates_order_and_budgets() -> None:
    with pytest.raises(ValueError, match="ordered"):
        project_tokens_over_quiet((), (), (MeasuredQuietSpan(1, 3), MeasuredQuietSpan(2, 4)))
    with pytest.raises(ValueError, match="budget"):
        project_tokens_over_quiet((), (), (MeasuredQuietSpan(0, 1),) * 100_001)
    with pytest.raises(ValueError, match="budget"):
        project_tokens_over_quiet((), (), (), fps=0)
    with pytest.raises(ValueError, match="ordered"):
        project_tokens_over_quiet(
            (SpokenToken("later", 2, 3, "a"), SpokenToken("earlier", 0, 1, "a")), (), ()
        )


def test_quiet_projection_bounds_intersection_work() -> None:
    with pytest.raises(ValueError, match="work budget"):
        project_tokens_over_quiet(
            (SpokenToken("long", 0, 11, "a"),) * 100_000,
            (
                CutSpan(0, 30, 0),
                CutSpan(60, 90, 30),
                CutSpan(120, 150, 60),
                CutSpan(180, 210, 90),
                CutSpan(240, 270, 120),
                CutSpan(300, 330, 150),
            ),
            (
                MeasuredQuietSpan(1, 2),
                MeasuredQuietSpan(3, 4),
                MeasuredQuietSpan(5, 6),
                MeasuredQuietSpan(7, 8),
                MeasuredQuietSpan(9, 10),
            ),
        )
