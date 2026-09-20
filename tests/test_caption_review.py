"""Context selection uses whole native labels and shared context, never guessed times."""

import pytest

from replay.caption_repair import ClockLabel
from replay.caption_review import select_native_context, spoken_text


@pytest.fixture
def original() -> tuple[ClockLabel, ...]:
    return (
        ClockLabel("導入", 0, 0.5, "old"),
        ClockLabel("古い", 0.8, 1, "old"),
        ClockLabel("ことば", 1.2, 1.2, "old"),
        ClockLabel("末尾", 1.5, 1.8, "old"),
        ClockLabel("別文", 2, 2.4, "old"),
    )


@pytest.fixture
def alternative() -> tuple[ClockLabel, ...]:
    return (
        ClockLabel("導入", 0.1, 0.5, "new"),
        ClockLabel("新しい", 0.8, 1, "new"),
        ClockLabel("ことば!", 1.1, 1.3, "new"),
        ClockLabel("末尾", 1.5, 1.8, "new"),
        ClockLabel("別文", 2, 2.4, "new"),
    )


def test_shared_anchors_select_native_text_without_forcing_spelling(
    original: tuple[ClockLabel, ...], alternative: tuple[ClockLabel, ...]
) -> None:
    result = select_native_context(
        original,
        alternative,
        target_start=2,
        target_end=3,
        context_start=0,
        context_end=3,
        source_duration=4,
        required_text=("ことば",),
    )
    assert result is not None
    assert (result.original_start, result.original_end) == (1, 3)
    assert result.replacement == (
        ClockLabel("新しい", 0.8, 1, "new"),
        ClockLabel("ことば!", 1.1, 1.3, "new"),
    )
    assert original[2].end == 1.2


def test_anchor_normalization_does_not_transliterate() -> None:
    assert spoken_text(" R.I.P！ あ\n") == "RIPあ"
    assert spoken_text("アイ") == "アイ"


def test_missing_required_word_remains_unresolved(
    original: tuple[ClockLabel, ...], alternative: tuple[ClockLabel, ...]
) -> None:
    assert (
        select_native_context(
            original,
            alternative,
            target_start=2,
            target_end=3,
            context_start=0,
            context_end=3,
            source_duration=4,
            required_text=("欠落",),
        )
        is None
    )


def test_native_whole_word_may_contain_point_without_invented_character_clock() -> None:
    result = select_native_context(
        (ClockLabel("I.", 0.5, 0.5, "old"),),
        (ClockLabel("RIP", 0.4, 0.6, "new"),),
        target_start=0,
        target_end=1,
        context_start=0,
        context_end=1,
        source_duration=1,
        required_text=("I.",),
    )
    assert result is not None
    assert result.replacement == (ClockLabel("RIP", 0.4, 0.6, "new"),)


def test_no_safe_anchor_is_not_guessed(
    original: tuple[ClockLabel, ...], alternative: tuple[ClockLabel, ...]
) -> None:
    assert (
        select_native_context(
            original,
            alternative,
            target_start=2,
            target_end=3,
            context_start=0.05,
            context_end=3,
            source_duration=4,
        )
        is None
    )


def test_crossing_old_left_seam_expands_instead_of_clipping(
    original: tuple[ClockLabel, ...], alternative: tuple[ClockLabel, ...]
) -> None:
    donor = alternative[:1] + (ClockLabel("新しい", 0.4, 1, "new"),) + alternative[2:]
    result = select_native_context(
        original,
        donor,
        target_start=2,
        target_end=3,
        context_start=0,
        context_end=3,
        source_duration=4,
    )
    assert result is not None
    assert result.original_start == 0
    assert result.replacement[1].start == 0.4


def test_crossing_old_right_seam_expands_instead_of_clipping(
    original: tuple[ClockLabel, ...], alternative: tuple[ClockLabel, ...]
) -> None:
    donor = alternative[:2] + (ClockLabel("ことば!", 1.1, 1.7, "new"),) + alternative[3:]
    result = select_native_context(
        original,
        donor,
        target_start=2,
        target_end=3,
        context_start=0,
        context_end=3,
        source_duration=4,
    )
    assert result is not None
    assert result.original_end == 4


@pytest.mark.parametrize(
    "donor",
    [
        (),
        (ClockLabel("a", 0.5, 0.5, "new"),),
        (ClockLabel("a", 0, 3, "new"),),
        (ClockLabel("a" * 37, 0.4, 0.6, "new"),),
        (ClockLabel("a!b!c!", 0.4, 0.6, "new"),),
        (ClockLabel("a", 0.2, 0.3, "new"), ClockLabel("b", 0.2, 0.4, "new")),
        (ClockLabel("a", 0.2, 0.5, "new"), ClockLabel("b", 0.3, 0.4, "new")),
    ],
)
def test_invalid_display_alignment_does_not_become_a_caption(donor: tuple[ClockLabel, ...]) -> None:
    assert (
        select_native_context(
            (ClockLabel("a", 0.5, 0.5, "old"),),
            donor,
            target_start=0,
            target_end=1,
            context_start=0,
            context_end=4,
            source_duration=4,
        )
        is None
    )


def test_point_outside_measured_context_remains_unresolved() -> None:
    assert (
        select_native_context(
            (ClockLabel("a", 0.1, 0.1, "old"),),
            (ClockLabel("a", 0.4, 0.6, "new"),),
            target_start=0,
            target_end=1,
            context_start=0.2,
            context_end=1,
            source_duration=1,
        )
        is None
    )


@pytest.mark.parametrize(
    "start,end,duration", [(float("nan"), 1, 1), (1, 0, 1), (0, 61, 61), (0, 1, 86401)]
)
def test_invalid_bounds(start: float, end: float, duration: float) -> None:
    with pytest.raises(ValueError, match="bounds or budget"):
        select_native_context(
            (ClockLabel("a", 0.5, 0.5, "old"),),
            (),
            target_start=0,
            target_end=1,
            context_start=start,
            context_end=end,
            source_duration=duration,
        )


def test_target_index_budget() -> None:
    with pytest.raises(ValueError, match="bounds or budget"):
        select_native_context(
            (ClockLabel("a", 0.5, 0.5, "old"),),
            (),
            target_start=True,
            target_end=1,
            context_start=0,
            context_end=1,
            source_duration=1,
        )


def test_alternative_count_budget() -> None:
    with pytest.raises(ValueError, match="bounds or budget"):
        select_native_context(
            (ClockLabel("a", 0.5, 0.5, "old"),),
            (ClockLabel("a", 0.4, 0.6, "new"),) * 1001,
            target_start=0,
            target_end=1,
            context_start=0,
            context_end=1,
            source_duration=1,
        )


def test_alternative_text_budget() -> None:
    with pytest.raises(ValueError, match="bounds or budget"):
        select_native_context(
            (ClockLabel("a", 0.5, 0.5, "old"),),
            (ClockLabel("a" * 2000, 0.4, 0.6, "new"),) * 17,
            target_start=0,
            target_end=1,
            context_start=0,
            context_end=1,
            source_duration=1,
        )


def test_unordered_alternative() -> None:
    with pytest.raises(ValueError, match="ordered"):
        select_native_context(
            (ClockLabel("a", 0.5, 0.5, "old"),),
            (ClockLabel("a", 0.6, 0.7, "new"), ClockLabel("b", 0.4, 0.5, "new")),
            target_start=0,
            target_end=1,
            context_start=0,
            context_end=1,
            source_duration=1,
        )


def test_source_overrun() -> None:
    with pytest.raises(ValueError, match="exceeds source"):
        select_native_context(
            (ClockLabel("a", 0.5, 0.5, "old"),),
            (ClockLabel("a", 0.4, 1.1, "new"),),
            target_start=0,
            target_end=1,
            context_start=0,
            context_end=1,
            source_duration=1,
        )


def test_context_overrun() -> None:
    with pytest.raises(ValueError, match="measured context"):
        select_native_context(
            (ClockLabel("a", 0.5, 0.5, "old"),),
            (ClockLabel("a", 0.1, 0.6, "new"),),
            target_start=0,
            target_end=1,
            context_start=0.2,
            context_end=1,
            source_duration=1,
        )


def test_anchor_search_budget() -> None:
    original = (ClockLabel("同じ", 0, 1, "old"),) * 70 + (ClockLabel("a", 1, 1, "old"),)
    with pytest.raises(ValueError, match="anchor budget"):
        select_native_context(
            original,
            (ClockLabel("同じ", 0, 1, "new"),) * 70,
            target_start=70,
            target_end=71,
            context_start=0,
            context_end=2,
            source_duration=2,
        )
