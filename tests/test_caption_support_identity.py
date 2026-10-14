"""Exact original clocks and ambiguous aliases, without ASR or media fixtures."""

from dataclasses import replace
from fractions import Fraction

import pytest

from replay.caption_support_identity import (
    OriginalFragment,
    SupportAlias,
    SupportIdentity,
    bind_aliases,
    support_key,
)


def test_outer_whitespace_only_and_no_timestamp_change() -> None:
    original = OriginalFragment(" 12 34!", Fraction(7, 3), Fraction(8, 3), "window:cut", 10)
    identity = SupportIdentity("a" * 64, "cut1", (original,))
    result = bind_aliases((SupportAlias(" 12 34!", identity), SupportAlias("12 34! ", identity)))
    assert result["12 34!"].fragments[0] is original
    assert result["12 34!"].fragments[0].start == Fraction(7, 3)
    assert result["12 34!"].fragments[0].text == " 12 34!"
    assert support_key("\t字幕 内部 \u3000") == "字幕 内部"


@pytest.mark.parametrize("change", ["time", "owner", "source", "cut", "index"])
def test_collision_rejects_different_full_identity(change: str) -> None:
    fragment = OriginalFragment("語", Fraction(1), Fraction(2), "owner", 1)
    original = SupportIdentity("a" * 64, "cut", (fragment,))
    variants = {
        "time": replace(original, fragments=(replace(fragment, end=Fraction(3)),)),
        "owner": replace(original, fragments=(replace(fragment, owner="other"),)),
        "source": replace(original, input_sha256="b" * 64),
        "cut": replace(original, cut_identity="different"),
        "index": replace(original, fragments=(replace(fragment, source_token_index=2),)),
    }
    with pytest.raises(ValueError, match="Ambiguous"):
        bind_aliases((SupportAlias("語", original), SupportAlias(" 語", variants[change])))


def test_internal_space_and_punctuation_are_not_removed() -> None:
    original = SupportIdentity(
        "a" * 64, "cut", (OriginalFragment("12 12!", Fraction(1), Fraction(2), "owner", 1),)
    )
    with pytest.raises(ValueError, match="recognized text"):
        bind_aliases((SupportAlias("1212!", original),))
    with pytest.raises(ValueError, match="recognized text"):
        bind_aliases((SupportAlias("12 12", original),))


@pytest.mark.parametrize("text", ["", "  ", "x\0", "x" * 8193])
def test_invalid_key(text: str) -> None:
    with pytest.raises(ValueError):
        support_key(text)


def test_invalid_models_and_budget() -> None:
    with pytest.raises(ValueError, match="fragment"):
        OriginalFragment("x", Fraction(1), Fraction(1), "owner", 0)
    with pytest.raises(ValueError, match="source/cut"):
        SupportIdentity("bad", "cut", ())
    identity = SupportIdentity(
        "a" * 64, "cut", (OriginalFragment("x", Fraction(1), Fraction(2), "owner", 0),)
    )
    with pytest.raises(ValueError, match="budget"):
        bind_aliases((SupportAlias("x", identity),) * 16385)
