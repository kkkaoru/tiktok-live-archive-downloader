"""Synthetic line legality without private transcripts or model loading."""

import pytest

from replay.caption_line_legality import assess_lines


@pytest.mark.parametrize("punctuation", ["?", "？！", "。", "）", "」", "】", "…"])
def test_sentence_tail_never_becomes_a_line(punctuation: str) -> None:
    result = assess_lines("字幕" + punctuation, ("字幕", punctuation))
    assert result.legal is False
    assert result.reason == "punctuation-only-line"


@pytest.mark.parametrize(
    "tail", ["？次", "）次", "】次", "ゃ次", "ー次", "\u3099次", "\u0301next", "  。次"]
)
def test_prohibited_start_attaches_to_prior_text(tail: str) -> None:
    assert assess_lines("前" + tail, ("前", tail)).reason == "prohibited-line-start"


@pytest.mark.parametrize("opening", ["「", "（", "[", "【", "“"])
def test_opening_bracket_stays_with_following_text(opening: str) -> None:
    assert (
        assess_lines("前" + opening + "後", ("前" + opening, "後")).reason == "prohibited-line-end"
    )


def test_legitimate_lines_preserve_punctuation_and_emoji() -> None:
    assert assess_lines("前？後。", ("前？", "後。")).legal is True
    assert assess_lines("前「後」", ("前", "「後」")).legal is True
    assert assess_lines("前😀", ("前", "😀")).legal is True
    assert assess_lines("?!", ("?!",)).reason == "intact-text"


def test_three_lines_and_empty_line() -> None:
    assert assess_lines("甲乙丙", ("甲", "乙", "丙")).legal is True
    assert assess_lines("甲 乙", ("甲", " ", "乙")).reason == "empty-line"


@pytest.mark.parametrize(
    "text,lines",
    [
        ("", ("",)),
        ("ab", ("a",)),
        ("ab", ("a", "c")),
        ("a\nb", ("a\nb",)),
        ("x" * 8193, ("x" * 8193,)),
        ("a" * 65, ("a",) * 65),
    ],
)
def test_bounded_exact_text_contract(text: str, lines: tuple[str, ...]) -> None:
    with pytest.raises(ValueError):
        assess_lines(text, lines)
