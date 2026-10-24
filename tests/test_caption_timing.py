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


def test_measured_policy_allows_three_lines_without_retiming(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        "replay.caption_linguistics._atomic_ends", lambda text: frozenset({len(text)})
    )
    monkeypatch.setattr(
        "replay.caption_linguistics.semantic_break", lambda previous, following: False
    )
    monkeypatch.setattr(
        "replay.caption_linguistics.boundaries", lambda text: ((len(text),), (len(text),))
    )
    words = (SpokenToken("abcdef", 0, 1, "one"), SpokenToken("gh", 1, 2, "two"))
    result = make_captions(
        words,
        policy=CaptionPolicy(
            maximum_lines=3,
            linguistic_boundaries=True,
            hold_seconds=0,
            measured_layout=lambda text: "\n".join(
                text[index : index + 2] for index in range(0, len(text), 2)
            ),
        ),
    )
    assert [(cue.text, cue.start, cue.end) for cue in result] == [
        ("ab\ncd\nef", 0, 1),
        ("gh", 1, 2),
    ]


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


def test_frame_budget_preserves_short_word_before_width_reset() -> None:
    cues = make_captions(
        (
            SpokenToken("ab", 0, 0.96, "a"),
            SpokenToken("c", 0.96, 0.98, "a"),
            SpokenToken("d", 0.98, 0.99, "a"),
            SpokenToken("ef", 0.99, 1.1, "a"),
        ),
        policy=CaptionPolicy(maximum_characters=4, frame_rate=30),
    )
    assert [cue.text for cue in cues] == ["ab", "abc", "d", "def"]
    assert [cue.start for cue in cues] == [0, 0.96, 0.98, 0.99]
    assert [cue.text for cue in cues if cue.start <= 1 < cue.end] == ["def"]


def test_smooth_frame_horizon_preempts_a_one_frame_final_prefix() -> None:
    cues = make_captions(
        (
            SpokenToken("ab", 0, 0.94, "a"),
            SpokenToken("c", 0.94, 0.99, "a"),
            SpokenToken("de", 0.99, 1.1, "a"),
        ),
        policy=CaptionPolicy(maximum_characters=4, frame_rate=30, reset_lookahead_frames=2),
    )
    assert [cue.text for cue in cues] == ["ab", "c", "cde"]
    assert [cue.start for cue in cues] == [0, 0.94, 0.99]


@pytest.mark.parametrize("third_onset", [0.95, 0.98])
def test_anticipated_reset_does_not_hide_the_preceding_short_prefix(third_onset: float) -> None:
    cues = make_captions(
        (
            SpokenToken("ab", 0, 0.94, "a"),
            SpokenToken("c", 0.94, third_onset, "a"),
            SpokenToken("d", third_onset, 0.99, "a"),
            SpokenToken("e", 0.99, 1.1, "a"),
        ),
        policy=CaptionPolicy(maximum_characters=4, frame_rate=30, reset_lookahead_frames=2),
    )
    assert [cue.text for cue in cues] == ["ab", "c", "cd", "cde"]
    assert [cue.start for cue in cues] == [0, 0.94, third_onset, 0.99]


def test_frame_horizon_uses_the_render_clock_not_decimal_float_noise() -> None:
    cues = make_captions(
        (
            SpokenToken("ab", 0, 109 / 150, "a"),
            SpokenToken("c", 109 / 150, 23 / 30, "a"),
            SpokenToken("de", 23 / 30, 0.9, "a"),
        ),
        policy=CaptionPolicy(maximum_characters=4, frame_rate=30, reset_lookahead_frames=2),
    )
    assert [cue.text for cue in cues] == ["ab", "c", "cde"]
    assert [cue.start for cue in cues] == [0, 0.7266666666666667, 0.7666666666666667]


def test_rewind_does_not_count_float_noise_as_an_extra_visible_frame() -> None:
    cues = make_captions(
        (
            SpokenToken("ab", 0, 0.71, "a"),
            SpokenToken("c", 0.71, 23 / 30, "a"),
            SpokenToken("d", 23 / 30, 0.79, "a"),
            SpokenToken("e", 0.79, 0.9, "a"),
        ),
        policy=CaptionPolicy(maximum_characters=4, frame_rate=30, reset_lookahead_frames=2),
    )
    assert [cue.text for cue in cues] == ["ab", "c", "cd", "cde"]
    assert [cue.start for cue in cues] == [0, 0.71, 0.7666666666666667, 0.79]


def test_real_subframe_time_after_the_horizon_is_not_pulled_back() -> None:
    cues = make_captions(
        (
            SpokenToken("ab", 0, 109 / 150, "a"),
            SpokenToken("c", 109 / 150, 0.7667, "a"),
            SpokenToken("de", 0.7667, 0.9, "a"),
        ),
        policy=CaptionPolicy(maximum_characters=4, frame_rate=30, reset_lookahead_frames=2),
    )
    assert [cue.text for cue in cues] == ["ab", "abc", "de"]


def test_frame_rewind_cannot_expand_the_character_budget() -> None:
    cues = make_captions(
        (
            SpokenToken("aa", 0.94, 0.95, "a"),
            SpokenToken("bb", 0.95, 0.99, "a"),
            SpokenToken("cc", 0.99, 1.1, "a"),
        ),
        policy=CaptionPolicy(maximum_characters=4, frame_rate=30, reset_lookahead_frames=2),
    )
    assert [cue.text for cue in cues] == ["aa", "aabb", "cc"]


@pytest.mark.parametrize("lookahead", [0, 31, True])
def test_reset_lookahead_is_bounded_integer(lookahead: int) -> None:
    with pytest.raises(ValueError, match="Invalid caption reset lookahead"):
        CaptionPolicy(frame_rate=30, reset_lookahead_frames=lookahead)


def test_reset_lookahead_requires_a_frame_rate() -> None:
    with pytest.raises(ValueError, match="Invalid caption reset lookahead"):
        CaptionPolicy(reset_lookahead_frames=2)


def test_frame_budget_preserves_short_word_before_lifetime_reset() -> None:
    cues = make_captions(
        (
            SpokenToken("A", 0.19, 1.973, "a"),
            SpokenToken("B", 1.973, 1.993, "a"),
            SpokenToken("C", 1.993, 2.1, "a"),
        ),
        policy=CaptionPolicy(maximum_context_seconds=1.8, frame_rate=30),
    )
    assert [cue.text for cue in cues] == ["A", "B", "BC"]
    assert [cue.start for cue in cues] == [0.19, 1.973, 1.993]
    assert [cue.text for cue in cues if cue.start <= 2 < cue.end] == ["BC"]


@pytest.mark.parametrize(
    "following", [SpokenToken("C", 0.99, 1.1, "b"), SpokenToken("CDEF", 0.99, 1.1, "a")]
)
def test_frame_budget_does_not_bridge_ownership_or_oversized_pair(following: SpokenToken) -> None:
    cues = make_captions(
        (SpokenToken("AA", 0, 0.98, "a"), SpokenToken("B", 0.98, 0.99, "a"), following),
        policy=CaptionPolicy(maximum_characters=4, frame_rate=30),
    )
    assert [cue.text for cue in cues[:2]] == ["AA", "AAB"]
    assert cues[-1].text == following.text


def test_frame_budget_does_not_bridge_sentence_end() -> None:
    cues = make_captions(
        (
            SpokenToken("A", 0, 0.98, "a"),
            SpokenToken("B!", 0.98, 0.99, "a"),
            SpokenToken("C", 0.99, 1.1, "a"),
        ),
        policy=CaptionPolicy(frame_rate=30),
    )
    assert [cue.text for cue in cues] == ["A", "AB!", "C"]


@pytest.mark.parametrize("frame_rate", [0, 121, True])
def test_frame_rate_is_bounded_integer(frame_rate: int) -> None:
    with pytest.raises(ValueError, match="Invalid caption frame rate"):
        CaptionPolicy(frame_rate=frame_rate)


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
        (
            SpokenToken(
                "あいうえおかきくけこさしすせそたちつてとなにぬねのはひふへほまみむめもやゆ",
                0,
                1,
                "a",
            ),
        ),
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


def test_held_note_run_is_not_captioned() -> None:
    tokens = (
        SpokenToken("こんにちは", 0.0, 0.8, "a:quiet-cuts-0-0"),
        SpokenToken("ー" * 86, 1.0, 3.0, "a:quiet-cuts-0-0"),
        SpokenToken("次です", 3.2, 4.0, "a:quiet-cuts-0-0"),
    )

    cues = make_captions(tokens)

    assert [cue.text for cue in cues] == ["こんにちは", "次です"]
    assert (cues[0].start, cues[1].start) == (0.0, 3.2)


def test_recognizer_repetition_loops_are_not_captioned() -> None:
    tokens = (
        SpokenToken("こんにちは", 0.0, 0.8, "a:quiet-cuts-0-0"),
        SpokenToken("ッハ" * 19, 1.0, 1.02, "a:quiet-cuts-0-0"),
        SpokenToken("F" * 54, 1.5, 1.9, "a:quiet-cuts-0-0"),
        SpokenToken("次です", 3.2, 4.0, "a:quiet-cuts-0-0"),
    )

    cues = make_captions(tokens)

    assert [cue.text for cue in cues] == ["こんにちは", "次です"]


def test_short_repeated_laughter_is_still_captioned() -> None:
    tokens = (SpokenToken("ハハハ", 0.0, 0.7, "a:quiet-cuts-0-0"),)

    assert [cue.text for cue in make_captions(tokens)] == ["ハハハ"]


def test_prolonged_mark_inside_a_word_is_captioned() -> None:
    tokens = (SpokenToken("コーヒー", 0.0, 0.7, "a:quiet-cuts-0-0"),)

    assert [cue.text for cue in make_captions(tokens)] == ["コーヒー"]


def test_word_widened_by_marks_beyond_the_budget_is_still_rejected() -> None:
    tokens = (SpokenToken("あ" + "ー" * 30, 0.0, 0.7, "a:quiet-cuts-0-0"),)

    with pytest.raises(ValueError, match="exceeds display budget"):
        make_captions(tokens, policy=CaptionPolicy(maximum_characters=18, maximum_lines=1))


def test_rolling_width_keeps_recent_complete_words_instead_of_a_lone_particle() -> None:
    tokens = (
        SpokenToken("あさって", 0, 0.5, "owner"),
        SpokenToken("あさって", 0.5, 1, "owner"),
        SpokenToken("じゃない", 1, 1.2, "owner"),
        SpokenToken("って", 1.2, 1.5, "owner"),
    )
    cues = make_captions(
        tokens, policy=CaptionPolicy(maximum_characters=12, retain_recent_words=True)
    )
    assert tuple(cue.text for cue in cues) == (
        "あさって",
        "あさってあさって",
        "あさってじゃない",
        "あさってじゃないって",
    )
    assert cues[-1] == CaptionCue("あさってじゃないって", 1.2, 1.58)


def test_rolling_context_discards_expired_words_without_clearing_recent_words() -> None:
    cues = make_captions(
        (
            SpokenToken("甲", 0, 0.25, "owner"),
            SpokenToken("乙", 0.25, 0.5, "owner"),
            SpokenToken("丙", 0.5, 0.75, "owner"),
        ),
        policy=CaptionPolicy(maximum_context_seconds=0.5, retain_recent_words=True),
    )
    assert tuple(cue.text for cue in cues) == ("甲", "甲乙", "乙丙")


def test_rolling_context_respects_line_budget_and_hard_ownership_boundary() -> None:
    cues = make_captions(
        (
            SpokenToken("第一、", 0, 0.2, "owner"),
            SpokenToken("第二、", 0.2, 0.4, "owner"),
            SpokenToken("第三", 0.4, 0.6, "owner"),
            SpokenToken("新しい話", 0.6, 0.8, "next-owner"),
        ),
        policy=CaptionPolicy(retain_recent_words=True),
    )
    assert tuple(cue.text for cue in cues) == (
        "第一、",
        "第一、\n第二、",
        "第二、\n第三",
        "新しい話",
    )


def test_uncertain_region_is_blank_without_placeholder_or_context_carry() -> None:
    tokens = (
        SpokenToken("前", 0, 0.5, "same"),
        SpokenToken("不確かな認識", 0.51, 0.55, "same"),
        SpokenToken("後", 0.56, 0.9, "same"),
    )
    assert make_captions(tokens, excluded_regions=((0.5, 0.56),)) == (
        CaptionCue("前", 0, 0.5),
        CaptionCue("後", 0.56, 0.98),
    )
    assert tokens[1] == SpokenToken("不確かな認識", 0.51, 0.55, "same")


def test_uncertain_coarse_interval_is_not_split_into_invented_words() -> None:
    assert make_captions((SpokenToken("不確か", 1, 7, "owner"),), excluded_regions=((1, 7),)) == ()
    with pytest.raises(ValueError, match="too coarse"):
        make_captions((SpokenToken("不確か", 1, 7, "owner"),))


def test_exclusion_does_not_clip_a_partially_supported_recognized_word() -> None:
    assert make_captions(
        (SpokenToken("重なる語", 0, 1, "owner"), SpokenToken("次", 2, 2.5, "owner")),
        excluded_regions=((0.5, 1.5),),
    ) == (CaptionCue("次", 2, 2.58),)


def test_multiple_excluded_regions_preserve_boundary_ownership_and_tail() -> None:
    assert make_captions(
        (
            SpokenToken("甲", 0, 0.25, "owner"),
            SpokenToken("乙", 0.5, 0.75, "owner"),
            SpokenToken("丙", 1, 1.25, "owner"),
        ),
        excluded_regions=((0.25, 0.5), (0.75, 1)),
    ) == (
        CaptionCue("甲", 0, 0.25),
        CaptionCue("乙", 0.5, 0.75),
        CaptionCue("丙", 1, 1.33),
    )


@pytest.mark.parametrize(
    "regions",
    [
        ((-1.0, 1.0),),
        ((0.0, 0.0),),
        ((0.0, math.inf),),
        ((math.nan, 1.0),),
        ((False, 1.0),),
        ((0.0, True),),
        ((0.0, 86401.0),),
        ((0.0, 2.0), (1.0, 3.0)),
        ((2.0, 3.0), (0.0, 1.0)),
    ],
)
def test_excluded_regions_validate_evidence_bounds(
    regions: tuple[tuple[float, float], ...],
) -> None:
    with pytest.raises(ValueError, match="excluded caption regions"):
        make_captions((), excluded_regions=regions)


def test_exclusion_budget_and_unsorted_tokens_still_fail() -> None:
    with pytest.raises(ValueError, match="region budget"):
        make_captions((), excluded_regions=((0.0, 1.0),) * 100001)
    with pytest.raises(ValueError, match="not ordered"):
        make_captions(
            (SpokenToken("後", 2, 3, "a"), SpokenToken("前", 0, 1, "a")),
            excluded_regions=((0, 1),),
        )


def test_trailing_particle_stays_with_its_words_instead_of_switching_alone() -> None:
    cues = make_captions(
        (
            SpokenToken("今日は", 0, 0.3, "a"),
            SpokenToken("楽しかった", 0.3, 0.8, "a"),
            SpokenToken("よ", 0.95, 1.1, "a"),
        ),
        policy=CaptionPolicy(maximum_characters=8),
    )
    assert cues == (
        CaptionCue("今日は", 0, 0.3),
        CaptionCue("楽しかった", 0.3, 0.95),
        CaptionCue("楽しかったよ", 0.95, 1.1800000000000002),
    )


def test_particle_tail_bridges_a_short_pause_but_not_a_long_one() -> None:
    assert make_captions(
        (SpokenToken("そうだ", 0, 0.3, "a"), SpokenToken("よね", 0.5, 0.7, "a"))
    ) == (CaptionCue("そうだ", 0, 0.5), CaptionCue("そうだよね", 0.5, 0.7799999999999999))
    assert make_captions(
        (SpokenToken("そうだ", 0, 0.3, "a"), SpokenToken("よね", 0.6, 0.8, "a"))
    ) == (CaptionCue("そうだ", 0, 0.38), CaptionCue("よね", 0.6, 0.88))


def test_fragmented_word_attaches_only_its_end_and_sentence_end_is_not_a_tail() -> None:
    fragments = make_captions(
        (
            SpokenToken("今日", 0, 0.2, "a"),
            SpokenToken("あり", 0.2, 0.3, "a"),
            SpokenToken("がと", 0.3, 0.4, "a"),
        ),
        policy=CaptionPolicy(maximum_characters=4),
    )
    assert tuple(cue.text for cue in fragments) == ("今日", "あり", "ありがと")
    sentence = make_captions(
        (SpokenToken("終わり。", 0, 0.3, "a"), SpokenToken("ね", 0.3, 0.4, "a"))
    )
    assert tuple(cue.text for cue in sentence) == ("終わり。", "ね")
    other_speaker = make_captions(
        (SpokenToken("本当", 0, 0.3, "a"), SpokenToken("よ", 0.3, 0.4, "b"))
    )
    assert tuple(cue.text for cue in other_speaker) == ("本当", "よ")


def test_tail_overrun_is_opt_in_for_renderers_that_wrap_evenly() -> None:
    tokens = (
        SpokenToken("気づいたんじゃないの", 0, 0.8, "a"),
        SpokenToken("かな?", 0.8, 1.1, "a"),
    )
    assert tuple(
        cue.text for cue in make_captions(tokens, policy=CaptionPolicy(maximum_characters=12))
    ) == ("気づいたんじゃないの", "かな?")
    assert tuple(
        cue.text
        for cue in make_captions(
            tokens, policy=CaptionPolicy(maximum_characters=12, tail_overrun=True)
        )
    ) == ("気づいたんじゃないの", "気づいたんじゃないのかな?")
    split_tail = (
        SpokenToken("気づいたんじゃない", 0, 0.6, "a"),
        SpokenToken("の", 0.6, 0.8, "a"),
        SpokenToken("かな?", 0.8, 1.1, "a"),
    )
    assert tuple(
        cue.text
        for cue in make_captions(
            split_tail, policy=CaptionPolicy(maximum_characters=12, tail_overrun=True)
        )
    ) == ("気づいたんじゃない", "気づいたんじゃないの", "気づいたんじゃないのかな?")


def test_particle_after_a_comma_starts_the_next_caption() -> None:
    assert tuple(
        cue.text
        for cue in make_captions(
            (SpokenToken("そうだ、", 0, 0.3, "a"), SpokenToken("ね", 0.3, 0.4, "a")),
            policy=CaptionPolicy(maximum_characters=4, tail_overrun=True),
        )
    ) == ("そうだ、", "ね")


def test_scoped_policy_requires_complete_callback_contract() -> None:
    with pytest.raises(ValueError, match="Scoped caption fitting"):
        CaptionPolicy(scoped_layout=lambda text, context: text)


def test_lexical_provenance_retains_following_punctuation_without_retiming() -> None:
    from replay.caption_timing import lexical_token_sources

    words, indices = lexical_token_sources(
        (
            SpokenToken("珍", 0, 0.2, "u"),
            SpokenToken("しい", 0.2, 0.4, "u"),
            SpokenToken("…", 0.4, 0.5, "u"),
            SpokenToken("ー", 1, 1.2, "other"),
            SpokenToken("次", 2, 2.2, "other"),
        ),
        word_mark_gap=0.25,
    )
    assert words == [
        SpokenToken("珍", 0, 0.2, "u"),
        SpokenToken("しい…", 0.2, 0.4, "u"),
        SpokenToken("次", 2, 2.2, "other"),
    ]
    assert indices == [(0,), (1, 2), (4,)]


def test_scoped_final_layout_receives_each_real_occurrence(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        "replay.caption_linguistics._atomic_ends", lambda text: frozenset({len(text)})
    )
    calls: list[tuple[str, tuple[SpokenToken, ...]]] = []

    def scoped(text: str, context: tuple[SpokenToken, ...]) -> str:
        calls.append((text, context))
        return text

    cues = make_captions(
        (SpokenToken("語", 0, 0.2, "one"), SpokenToken("語", 2, 2.3, "two")),
        policy=CaptionPolicy(
            linguistic_boundaries=True,
            measured_layout=lambda text: text,
            scoped_layout=scoped,
            scoped_indivisible=lambda text, context: False,
        ),
    )
    assert tuple(cue.start for cue in cues) == (0, 2)
    assert calls[-2:] == [
        ("語", (SpokenToken("語", 0, 0.2, "one"),)),
        ("語", (SpokenToken("語", 2, 2.3, "two"),)),
    ]


def test_a_recognizer_loop_split_across_tokens_is_dropped_but_short_repeats_stay() -> None:
    from replay.caption_timing import SpokenToken, lexical_token_sources

    texts = ("ラ", "ララ", "ラララ", "ララ", "ラ", "なん", "はは", "はは")
    sung = tuple(
        SpokenToken("ラ", 10 + index * 0.3, 10.2 + index * 0.3, "a") for index in range(10)
    )
    assert [word.text for word in lexical_token_sources(sung)[0]] == ["ラ"] * 10
    tokens = tuple(
        SpokenToken(text, index * 0.01, index * 0.01 + 0.01, "a")
        for index, text in enumerate(texts)
    )
    words, indices = lexical_token_sources(tokens)
    assert [word.text for word in words] == ["なん", "はは", "はは"][: len(words)]
    assert indices[0] == (5,)


def test_a_loop_run_inside_one_token_is_dropped_and_its_tail_kept() -> None:
    from replay.caption_timing import SpokenToken, lexical_token_sources

    tokens = (
        SpokenToken("ラ" * 100 + "なん", 0.0, 0.45, "a"),
        SpokenToken("こぉぉぉぉぉぉぉぉぉ", 0.5, 1.0, "a"),
    )
    words, _ = lexical_token_sources(tokens)
    assert [(word.text, word.start, word.end) for word in words] == [
        ("なん", 0.0, 0.45),
        ("こぉぉぉぉぉぉぉぉぉ", 0.5, 1.0),
    ]


def test_an_elongation_run_is_cut_after_two_caption_lines_but_a_short_one_stays() -> None:
    from replay.caption_timing import SpokenToken, lexical_token_sources

    dense = (
        SpokenToken("ふぉ", 39.5, 39.8, "a"),
        SpokenToken("ぉ" * 6, 39.8, 40.0, "a"),
        SpokenToken("ぉ" * 40, 40.0, 40.1, "a"),
        SpokenToken("ぉ" * 15, 40.1, 41.0, "a"),
    )
    held = (
        SpokenToken("こぉ", 0.0, 0.3, "a"),
        SpokenToken("ぉぉぉ", 0.3, 1.0, "a"),
        SpokenToken("ぉぉぉぉぉ", 1.0, 2.0, "a"),
    )
    assert [word.text for word in lexical_token_sources(dense)[0]] == ["ふぉ", "ぉ" * 6]
    assert len(lexical_token_sources(held)[0]) == 3
