"""Exact-label replacement must use real alternative clocks or remain unresolved."""

import pytest

from replay import caption_repair
from replay.caption_repair import ClockLabel, canonical_text, repair_native_points


def test_point_uses_existing_positive_native_clock() -> None:
    original = (
        ClockLabel("a", 0.1, 0.2, "old"),
        ClockLabel("b", 0.3, 0.3, "old"),
        ClockLabel("c", 0.5, 0.6, "old"),
    )
    alternative = (ClockLabel("b", 0.25, 0.35, "new"),)
    result = repair_native_points(original, (alternative,), source_duration=1)
    assert result.labels[1] == ClockLabel("b", 0.25, 0.35, "new")
    assert result.unresolved_original_indices == ()
    assert result.repairs[0].original_start == 1
    assert result.repairs[0].original_end == 2
    assert original[1].end == 0.3


def test_whole_block_can_change_tokenization_without_apportioning_clocks() -> None:
    original = (
        ClockLabel("x", 0, 0.45, "old"),
        ClockLabel("a", 0.5, 1, "old"),
        ClockLabel("b", 1, 1, "old"),
        ClockLabel("c", 1.1, 1.2, "old"),
    )
    result = repair_native_points(
        original, ((ClockLabel("ab", 0.5, 1.04, "new"),),), source_duration=2
    )
    assert result.labels == (
        ClockLabel("x", 0, 0.45, "old"),
        ClockLabel("ab", 0.5, 1.04, "new"),
        ClockLabel("c", 1.1, 1.2, "old"),
    )
    assert result.repairs[0].original_start == 1
    assert result.repairs[0].original_end == 3


@pytest.mark.parametrize(
    "alternative",
    [
        ClockLabel("B", 0.25, 0.35, "new"),
        ClockLabel("b。", 0.25, 0.35, "new"),
        ClockLabel("b", 0.3, 0.3, "new"),
        ClockLabel("b", 0.1, 0.35, "new"),
        ClockLabel("b", 0.25, 0.55, "new"),
    ],
)
def test_spelling_punctuation_points_and_neighbors_not_forced(alternative: ClockLabel) -> None:
    original = (
        ClockLabel("a", 0.1, 0.2, "old"),
        ClockLabel("b", 0.3, 0.3, "old"),
        ClockLabel("c", 0.5, 0.6, "old"),
    )
    result = repair_native_points(original, ((alternative,),), source_duration=1)
    assert result.unresolved_original_indices == (1,)
    assert result.labels[1] == ClockLabel("b", 0.3, 0.3, "old")


def test_whitespace_equivalence_is_only_text_normalization() -> None:
    assert canonical_text(" あ b\n") == "あb"
    result = repair_native_points(
        (ClockLabel(" b ", 0.3, 0.3, "old"),),
        ((ClockLabel("b", 0.25, 0.35, "new"),),),
        source_duration=1,
    )
    assert result.labels == (ClockLabel("b", 0.25, 0.35, "new"),)


def test_wrong_repetition_outside_point_neighborhood_is_not_selected() -> None:
    result = repair_native_points(
        (ClockLabel("b", 1, 1, "old"),), ((ClockLabel("b", 4, 4.1, "new"),),), source_duration=5
    )
    assert result.unresolved_original_indices == (0,)


def test_later_matching_occurrence_is_found() -> None:
    result = repair_native_points(
        (ClockLabel("b", 1, 1, "old"),),
        ((ClockLabel("b", 0, 0.1, "new"), ClockLabel("b", 0.95, 1.05, "new")),),
        source_duration=2,
    )
    assert result.labels == (ClockLabel("b", 0.95, 1.05, "new"),)


def test_no_alternatives_preserves_points() -> None:
    result = repair_native_points((ClockLabel("b", 1, 1, "old"),), ((),), source_duration=2)
    assert result.unresolved_original_indices == (0,)


def test_decreasing_alternative_ends_are_not_adopted() -> None:
    result = repair_native_points(
        (ClockLabel("bc", 0.5, 0.5, "old"),),
        ((ClockLabel("b", 0.4, 0.6, "new"), ClockLabel("c", 0.5, 0.55, "new")),),
        source_duration=1,
    )
    assert result.unresolved_original_indices == (0,)


def test_repair_covering_two_points_is_not_duplicated() -> None:
    original = (ClockLabel("a", 0.1, 0.1, "old"), ClockLabel("b", 0.2, 0.2, "old"))
    result = repair_native_points(
        original, ((ClockLabel("ab", 0.1, 0.3, "new"),),), source_duration=1
    )
    assert len(result.repairs) == 1
    assert result.unresolved_original_indices == ()


def test_adjacent_repair_respects_already_replaced_neighbor() -> None:
    original = (ClockLabel("a", 1, 1, "old"), ClockLabel("b", 1.2, 1.2, "old"))
    alternatives = ((ClockLabel("a", 0.9, 1.15, "new"), ClockLabel("b", 1.1, 1.3, "new")),)
    result = repair_native_points(original, alternatives, source_duration=2)
    assert len(result.repairs) == 1
    assert result.unresolved_original_indices == (1,)


@pytest.mark.parametrize(
    "text,provenance",
    [(" ", "x"), ("a" * 2049, "x"), ("a\0", "x"), ("a", ""), ("a", "x" * 257), ("a", "x\0")],
)
def test_invalid_label_text(text: str, provenance: str) -> None:
    with pytest.raises(ValueError, match="provenance or text"):
        ClockLabel(text, 0, 1, provenance)


@pytest.mark.parametrize("start,end", [(-1, 0), (1, 0), (float("nan"), 1), (0, float("inf"))])
def test_invalid_label_clock(start: float, end: float) -> None:
    with pytest.raises(ValueError, match="clock"):
        ClockLabel("a", start, end, "old")


@pytest.mark.parametrize("duration", [0, float("nan"), 86401, True])
def test_invalid_source_duration(duration: float) -> None:
    with pytest.raises(ValueError, match="budget"):
        repair_native_points((), (), source_duration=duration)


@pytest.mark.parametrize("maximum", [0, 33, True])
def test_invalid_block_budget(maximum: int) -> None:
    with pytest.raises(ValueError, match="budget"):
        repair_native_points((), (), source_duration=1, maximum_block_labels=maximum)


def test_original_count_budget() -> None:
    with pytest.raises(ValueError, match="budget"):
        repair_native_points((ClockLabel("a", 0, 1, "old"),) * 100001, (), source_duration=1)


def test_alternative_count_budget() -> None:
    with pytest.raises(ValueError, match="budget"):
        repair_native_points((), ((),) * 257, source_duration=1)


def test_alternative_label_budget() -> None:
    with pytest.raises(ValueError, match="budget"):
        repair_native_points((), ((ClockLabel("a", 0, 1, "new"),) * 100001,), source_duration=1)


def test_text_budget() -> None:
    with pytest.raises(ValueError, match="text exceeds"):
        repair_native_points((ClockLabel("a" * 2048, 0, 1, "old"),) * 4000, (), source_duration=1)


def test_unordered_onsets_rejected() -> None:
    with pytest.raises(ValueError, match="ordered"):
        repair_native_points(
            (ClockLabel("a", 1, 1, "old"), ClockLabel("b", 0, 0.5, "old")), (), source_duration=2
        )


def test_source_overrun_rejected() -> None:
    with pytest.raises(ValueError, match="source duration"):
        repair_native_points((ClockLabel("a", 0, 2, "old"),), (), source_duration=1)


def test_search_budget_is_enforced() -> None:
    original = tuple(ClockLabel("a", index, index, "old") for index in range(550))
    with pytest.raises(ValueError, match="search exceeds"):
        repair_native_points(original, ((ClockLabel("z", 0, 600, "new"),),), source_duration=600)


def test_internal_text_conservation_guard(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        caption_repair,
        "_match",
        lambda *_args, **_kwargs: (ClockLabel("different", 0.4, 0.6, "broken"),),
    )
    with pytest.raises(ValueError, match="changed transcript"):
        repair_native_points(
            (ClockLabel("a", 0.5, 0.5, "old"),),
            ((ClockLabel("a", 0, 1, "new"),),),
            source_duration=1,
        )
