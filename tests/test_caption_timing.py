import math

import pytest

from replay.caption_timing import (
    CaptionCue,
    CaptionPolicy,
    SpokenToken,
    make_captions,
    owned_tokens,
    punctuation_lines,
)


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("こんにちは、今日は晴れです。次へ！", "こんにちは、\n今日は晴れです。\n次へ！"),
        ("  、。こんにちは。\n", "こんにちは。"),
        ("本当！？次です。", "本当！？\n次です。"),
        ("価格は3.5ドル、7,000円。", "価格は3.5ドル、\n7,000円。"),
        ("こんにちは。』次です。", "こんにちは。』\n次です。"),
        ("前半\n後半", "前半\n後半"),
        ("。！？", ""),
        ("「最初、 次。」", "「最初、\n次。」"),
        ("", ""),
    ],
)
def test_punctuation_breaks_without_dangling_prefix(text: str, expected: str) -> None:
    assert punctuation_lines(text) == expected


def test_only_spoken_words_appear_and_new_utterance_is_clean() -> None:
    tokens = (
        SpokenToken("今日は、", 0.5, 1, "first"),
        SpokenToken("編集", 1.04, 1.5, "first"),
        SpokenToken("します。", 1.55, 2, "first"),
        SpokenToken("別の話", 3, 3.5, "second"),
    )
    assert make_captions(tokens) == (
        CaptionCue("今日は、", 0.5, 1.04),
        CaptionCue("今日は、\n編集", 1.04, 1.55),
        CaptionCue("今日は、\n編集します。", 1.55, 2.08),
        CaptionCue("別の話", 3, 3.58),
    )


def test_same_utterance_does_not_carry_text_across_silence() -> None:
    assert make_captions(
        (SpokenToken("前の話", 0, 0.5, "a"), SpokenToken("新しい話", 1, 1.5, "a"))
    ) == (CaptionCue("前の話", 0, 0.58), CaptionCue("新しい話", 1, 1.58))


def test_sentence_end_resets_context_at_next_onset() -> None:
    assert make_captions(
        (SpokenToken("最初。", 0, 0.5, "a"), SpokenToken("次です。", 0.52, 1, "a"))
    ) == (CaptionCue("最初。", 0, 0.52), CaptionCue("次です。", 0.52, 1.08))


def test_overlap_is_clipped_without_repeating_old_cue_on_restart() -> None:
    assert make_captions((SpokenToken("前", 0, 1, "a"), SpokenToken("新", 0.8, 1.2, "b"))) == (
        CaptionCue("前", 0, 0.8),
        CaptionCue("新", 0.8, 1.28),
    )


def test_punctuation_only_tokens_do_not_create_phantom_cues() -> None:
    assert make_captions(
        (
            SpokenToken("、", 0, 0.1, "a"),
            SpokenToken("こんにちは", 0.2, 0.5, "a"),
            SpokenToken("。", 0.5, 0.6, "a"),
            SpokenToken("！", 1, 1.1, "b"),
        )
    ) == (CaptionCue("こんにちは。", 0.2, 0.58),)
    assert make_captions(()) == ()


def test_exact_duplicate_versus_legitimate_repeated_word() -> None:
    result = make_captions(
        (
            SpokenToken("そう", 0, 0.3, "a"),
            SpokenToken("そう", 0, 0.3, "a"),
            SpokenToken("そう", 0.3, 0.6, "a"),
        )
    )
    assert tuple(cue.text for cue in result) == ("そう", "そうそう")
    assert tuple(cue.start for cue in result) == pytest.approx((0, 0.3))
    assert tuple(cue.end for cue in result) == pytest.approx((0.3, 0.68))


def test_context_budget_resets_without_using_future_words() -> None:
    result = make_captions(
        (
            SpokenToken("第一、", 0, 0.2, "a"),
            SpokenToken("第二、", 0.2, 0.4, "a"),
            SpokenToken("第三", 0.4, 0.6, "a"),
        )
    )
    assert tuple(cue.text for cue in result) == ("第一、", "第一、\n第二、", "第三")
    assert tuple(cue.start for cue in result) == pytest.approx((0, 0.2, 0.4))
    assert tuple(cue.end for cue in result) == pytest.approx((0.2, 0.4, 0.68))
    assert make_captions(
        (SpokenToken("四文字語", 0, 0.2, "a"), SpokenToken("次", 0.2, 0.4, "a")),
        policy=CaptionPolicy(maximum_characters=4),
    ) == (CaptionCue("四文字語", 0, 0.2), CaptionCue("次", 0.2, 0.48000000000000004))


def test_time_budget_resets_even_when_speech_is_continuous() -> None:
    assert make_captions(
        (SpokenToken("前", 0, 0.5, "a"), SpokenToken("次", 0.5, 1, "a")),
        policy=CaptionPolicy(maximum_context_seconds=0.5),
    ) == (CaptionCue("前", 0, 0.5), CaptionCue("次", 0.5, 1.08))


def test_overlap_chunk_ownership_is_half_open_not_global_text_deduplication() -> None:
    assert owned_tokens(
        (SpokenToken("前", 0, 1, "chunk"), SpokenToken("中", 1, 2, "chunk")),
        offset=10,
        owner_start=10.5,
        owner_end=11.5,
    ) == (SpokenToken("前", 10, 11, "chunk"),)


@pytest.mark.parametrize(
    "tokens",
    [
        (SpokenToken("後", 1, 2, "a"), SpokenToken("前", 0, 0.5, "a")),
        (SpokenToken("重複", 0, 1, "a"), SpokenToken("曖昧", 0, 1, "a")),
        (SpokenToken("長すぎる未分割の文章", 0, 5, "a"),),
        (SpokenToken("あ" * 37, 0, 1, "a"),),
        (SpokenToken("一、二、三。", 0, 1, "a"),),
    ],
)
def test_unsupported_native_timing_is_rejected(tokens: tuple[SpokenToken, ...]) -> None:
    with pytest.raises(ValueError):
        make_captions(tokens)


@pytest.mark.parametrize(
    ("text", "start", "end", "utterance"),
    [
        ("", 0, 1, "a"),
        ("\0", 0, 1, "a"),
        ("あ", math.nan, 1, "a"),
        ("あ", 0, math.inf, "a"),
        ("あ", -1, 1, "a"),
        ("あ", 0, 0, "a"),
        ("あ", 0, 1, ""),
    ],
)
def test_tokens_validate_boundaries(text: str, start: float, end: float, utterance: str) -> None:
    with pytest.raises(ValueError):
        SpokenToken(text, start, end, utterance)


@pytest.mark.parametrize(
    ("hold", "gap", "duration", "characters", "lines"),
    [
        (math.nan, 0.25, 2.8, 36, 2),
        (0.5, 0.1, 2.8, 36, 2),
        (0, 0.25, 6, 36, 2),
        (0, 0.25, 2.8, 3, 2),
        (0, 0.25, 2.8, 36, 4),
    ],
)
def test_policy_validates_budgets(
    hold: float, gap: float, duration: float, characters: int, lines: int
) -> None:
    with pytest.raises(ValueError):
        CaptionPolicy(hold, gap, duration, characters, lines)


@pytest.mark.parametrize(("offset", "start", "end"), [(math.nan, 0, 1), (1, 0, 2), (0, 2, 1)])
def test_invalid_ownership_is_rejected(offset: float, start: float, end: float) -> None:
    with pytest.raises(ValueError):
        owned_tokens((), offset=offset, owner_start=start, owner_end=end)
