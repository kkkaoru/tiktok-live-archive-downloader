"""An actual short provider phrase is not interpolated per-character timing."""

import pytest

from replay.caption_repair import ClockLabel
from replay.caption_review import native_phrase_fallback
from replay.caption_timing import CaptionPolicy


def test_whole_native_phrase_clock_preserves_zero_duration_fragment() -> None:
    original = (ClockLabel("えぇ", 0, 0.54, "word"), ClockLabel("っ", 0.54, 0.54, "point"))
    phrase = ClockLabel("えぇっ", 0, 0.54, "actual-native-segment")
    result = native_phrase_fallback(
        original,
        phrase,
        first=0,
        end=2,
        source_duration=1,
        policy=CaptionPolicy(maximum_characters=18, maximum_lines=1),
    )
    assert result is not None
    assert result.replacement == (phrase,)
    assert original[1].start == original[1].end


@pytest.mark.parametrize(
    "phrase",
    [
        ClockLabel("違う文字", 0, 0.54, "native"),
        ClockLabel("えぇっ", 0.54, 0.54, "native"),
        ClockLabel("えぇっ", 0, 3, "native"),
    ],
)
def test_no_forced_text_or_coarse_clock(phrase: ClockLabel) -> None:
    original = (ClockLabel("えぇ", 0, 0.54, "word"), ClockLabel("っ", 0.54, 0.54, "point"))
    assert native_phrase_fallback(original, phrase, first=0, end=2, source_duration=4) is None


def test_original_neighbors_are_not_retimed() -> None:
    original = (
        ClockLabel("前", 0, 0.5, "word"),
        ClockLabel("えぇっ", 0.6, 0.6, "point"),
        ClockLabel("後", 0.8, 1, "word"),
    )
    assert (
        native_phrase_fallback(
            original, ClockLabel("えぇっ", 0.4, 0.7, "native"), first=1, end=2, source_duration=1
        )
        is None
    )
    assert (
        native_phrase_fallback(
            original, ClockLabel("えぇっ", 0.5, 0.9, "native"), first=1, end=2, source_duration=1
        )
        is None
    )


@pytest.mark.parametrize("text", ["あ" * 37, "あ!い!う!"])
def test_display_budget_is_not_relaxed(text: str) -> None:
    original = (ClockLabel(text, 0.5, 0.5, "point"),)
    assert (
        native_phrase_fallback(
            original, ClockLabel(text, 0.1, 0.6, "native"), first=0, end=1, source_duration=1
        )
        is None
    )


def test_invalid_index() -> None:
    with pytest.raises(ValueError, match="bounds"):
        native_phrase_fallback(
            (), ClockLabel("あ", 0, 1, "native"), first=0, end=1, source_duration=1
        )


def test_invalid_original_order() -> None:
    with pytest.raises(ValueError, match="original phrase clocks"):
        native_phrase_fallback(
            (ClockLabel("あ", 1, 1, "old"), ClockLabel("い", 0, 0.5, "old")),
            ClockLabel("あい", 0, 1, "native"),
            first=0,
            end=2,
            source_duration=2,
        )


def test_original_overrun() -> None:
    with pytest.raises(ValueError, match="original phrase clocks"):
        native_phrase_fallback(
            (ClockLabel("あ", 0, 2, "old"),),
            ClockLabel("あ", 0, 1, "native"),
            first=0,
            end=1,
            source_duration=1,
        )
