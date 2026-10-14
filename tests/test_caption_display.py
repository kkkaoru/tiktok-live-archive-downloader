import pytest

from replay.caption_display import native_interval, owned_captions, unretained_frame_prefixes
from replay.caption_projection import CutSpan
from replay.caption_timing import CaptionCue, CaptionPolicy, SpokenToken


def test_hidden_prefixes_survive_in_a_later_displayed_native_extension() -> None:
    assert (
        unretained_frame_prefixes(
            (
                SpokenToken("a", 0.001, 0.002, "a"),
                SpokenToken("b", 0.002, 0.004, "a"),
                SpokenToken("c", 0.004, 0.2, "a"),
            ),
            policy=CaptionPolicy(frame_rate=30),
        )
        == ()
    )


def test_hidden_prefix_is_not_borrowed_from_another_owner() -> None:
    assert unretained_frame_prefixes(
        (SpokenToken("a", 0.001, 0.002, "first"), SpokenToken("ab", 0.002, 0.2, "second")),
        policy=CaptionPolicy(frame_rate=30),
    ) == (0,)


def test_reset_with_coincidentally_repeated_text_does_not_retain_a_hidden_prefix() -> None:
    assert unretained_frame_prefixes(
        (SpokenToken("abc", 0.001, 0.002, "a"), SpokenToken("abcd", 0.002, 0.2, "a")),
        policy=CaptionPolicy(frame_rate=30, maximum_characters=4),
    ) == (0,)


def test_a_wholly_unshown_chain_is_not_retained() -> None:
    assert unretained_frame_prefixes(
        (
            SpokenToken("a", 0.001, 0.002, "a"),
            SpokenToken("b", 0.002, 0.004, "a"),
            SpokenToken("c", 0.004, 0.01, "a"),
        ),
        policy=CaptionPolicy(frame_rate=30, hold_seconds=0),
    ) == (0, 1, 2)


def test_frame_prefix_accounting_requires_an_explicit_rate() -> None:
    with pytest.raises(ValueError, match="requires an output rate"):
        unretained_frame_prefixes((), policy=CaptionPolicy())
    assert unretained_frame_prefixes((), policy=CaptionPolicy(frame_rate=30)) == ()


def test_half_open_endpoint() -> None:
    assert native_interval(0, 1, first_frame=0, end_frame=60) == (0, 0.9999833333333333)


def test_empty_numerical_intersection() -> None:
    assert native_interval(10.099999999999909, 10.3, first_frame=0, end_frame=303) is None


def test_large_clock_mask_start_is_not_after_its_native_frame() -> None:
    assert native_interval(
        2603.5333333333333,
        2605.5333333333333,
        first_frame=77870,
        end_frame=78188,
        frame_end_tick=False,
    ) == (7.866666666666666, 9.866666666666667)


def test_adjacent_masks_have_no_tick_hole() -> None:
    left = native_interval(
        2601.5333333333333,
        2603.5333333333333,
        first_frame=77870,
        end_frame=78188,
        frame_end_tick=False,
    )
    right = native_interval(
        2603.5333333333333,
        2605.5333333333333,
        first_frame=77870,
        end_frame=78188,
        frame_end_tick=False,
    )
    assert left == (5.866666666666666, 7.866666666666666)
    assert right == (7.866666666666666, 9.866666666666667)


def test_title_start_is_also_frame_exact_but_retains_endpoint_tick() -> None:
    assert native_interval(
        2603.5333333333333, 2605.5333333333333, first_frame=77870, end_frame=78188
    ) == (7.866666666666666, 9.86665)


def test_unchanged_nonframe_clock() -> None:
    assert native_interval(0.01, 0.02, first_frame=0, end_frame=60) == (0.01, 0.02)


def test_disjoint_interval() -> None:
    assert native_interval(4, 5, first_frame=0, end_frame=60) is None


@pytest.mark.parametrize("start,end", [(float("nan"), 1), (2, 1), (-1, 1)])
def test_invalid_interval(start: float, end: float) -> None:
    with pytest.raises(ValueError, match="Invalid display"):
        native_interval(start, end, first_frame=0, end_frame=60)


def test_invalid_part() -> None:
    with pytest.raises(ValueError, match="render part"):
        native_interval(0, 1, first_frame=2, end_frame=1)


def test_real_subtick_interval_is_not_silently_discarded() -> None:
    with pytest.raises(ValueError, match="tick budget"):
        native_interval(0.01, 0.010001, first_frame=0, end_frame=60)


def test_tiny_nonframe_area_is_not_rounding_at_a_frame_boundary() -> None:
    with pytest.raises(ValueError, match="unrepresentable"):
        native_interval(0.01, 0.01000000001, first_frame=0, end_frame=60)


def test_hold_cannot_bleed_into_next_edit() -> None:
    assert owned_captions(
        (SpokenToken("話", 0.8, 0.99, "a:quiet-cuts-0-0"),), (CutSpan(0, 30, 0),)
    ) == (CaptionCue("話", 0.8, 1.0),)


@pytest.mark.parametrize("owner", ["missing", "a:quiet-cuts-x-0", "a:quiet-cuts-0-2"])
def test_missing_or_invalid_ownership(owner: str) -> None:
    with pytest.raises(ValueError, match="ownership"):
        owned_captions((SpokenToken("話", 0.8, 0.99, owner),), (CutSpan(0, 30, 0),))


def test_word_outside_owned_cut() -> None:
    with pytest.raises(ValueError, match="exceeds"):
        owned_captions((SpokenToken("話", 0.8, 1.01, "a:quiet-cuts-0-0"),), (CutSpan(0, 30, 0),))


def test_accepted_display_violations_permit_one_punctuation_break() -> None:
    token = SpokenToken("ン、テテ", 0.8, 0.99, "a:quiet-cuts-0-0")
    cuts = (CutSpan(0, 30, 0),)
    with pytest.raises(ValueError, match="punctuation line budget"):
        owned_captions((token,), cuts)
    assert owned_captions((token,), cuts, accept_coarse_timing=True) == (
        CaptionCue("ン、\nテテ", 0.8, 1.0),
    )
    assert owned_captions((token,), cuts, candidate_only=True) == (
        CaptionCue("ン、\nテテ", 0.8, 1.0),
    )
    with pytest.raises(ValueError, match="punctuation line budget"):
        owned_captions(
            (SpokenToken("一、二、三。", 0.8, 0.99, "a:quiet-cuts-0-0"),),
            cuts,
            candidate_only=True,
        )


def test_candidate_keeps_coarse_native_clock_and_wraps_text_only() -> None:
    token = SpokenToken("一二三四五六七八九十百千万億兆京垓秭穣溝", 0.2, 3.5, "a:quiet-cuts-0-0")
    cuts = (CutSpan(0, 150, 0),)
    with pytest.raises(ValueError, match="display budget"):
        owned_captions((token,), cuts)
    assert owned_captions((token,), cuts, candidate_only=True) == (
        CaptionCue("一二三四五六七八九十百千万億兆京垓秭\n穣溝", 0.2, 3.58),
    )


def test_fine_candidate_words_share_the_two_line_candidate_budget() -> None:
    cuts = (CutSpan(0, 150, 0),)
    tokens = (
        SpokenToken("一二三四五六七八九十", 0.2, 0.6, "a:quiet-cuts-0-0"),
        SpokenToken("百千万億兆京垓秭穣溝", 0.6, 1.0, "a:quiet-cuts-0-0"),
    )
    assert owned_captions(tokens, cuts) == (
        CaptionCue("一二三四五六七八九十", 0.2, 0.6),
        CaptionCue("百千万億兆京垓秭穣溝", 0.6, 1.08),
    )
    assert owned_captions(tokens, cuts, candidate_only=True) == (
        CaptionCue("一二三四五六七八九十", 0.2, 0.6),
        CaptionCue("一二三四五六七八九十百千万億兆京垓秭\n穣溝", 0.6, 1.08),
    )


def test_candidate_does_not_fabricate_fine_timing_or_widen_normal_cues() -> None:
    cuts = (CutSpan(0, 240, 0),)
    with pytest.raises(ValueError, match="too coarse"):
        owned_captions((SpokenToken("長い", 1, 7, "a:quiet-cuts-0-0"),), cuts)
    assert owned_captions(
        (SpokenToken("長い", 1, 7, "a:quiet-cuts-0-0"),), cuts, candidate_only=True
    ) == (CaptionCue("長い", 1, 7.08),)
    assert owned_captions(
        (SpokenToken("短い", 0.2, 0.4, "a:quiet-cuts-0-0"),), cuts, candidate_only=True
    ) == (CaptionCue("短い", 0.2, 0.48000000000000004),)


def test_candidate_does_not_hide_overlong_native_phrase() -> None:
    with pytest.raises(ValueError, match="display budget"):
        owned_captions(
            (
                SpokenToken(
                    "一二三四五六七八九十百千万億兆京垓秭穣溝澗正載極恒河沙阿僧祇那由他不可思議",
                    0.1,
                    0.3,
                    "a:quiet-cuts-0-0",
                ),
            ),
            (CutSpan(0, 30, 0),),
            candidate_only=True,
        )


def test_punctuation_is_not_a_new_owned_word() -> None:
    assert owned_captions(
        (
            SpokenToken("話", 0.7, 0.8, "a:quiet-cuts-0-0"),
            SpokenToken("。", 0.8, 0.9, "a:quiet-cuts-0-0"),
        ),
        (CutSpan(0, 30, 0),),
    ) == (CaptionCue("話。", 0.7, 0.88),)


def test_explicit_rolling_policy_preserves_words_and_clips_hold_at_cut() -> None:
    assert owned_captions(
        (
            SpokenToken("昨日", 0.1, 0.3, "a:quiet-cuts-0-0"),
            SpokenToken("今日", 0.3, 0.6, "a:quiet-cuts-0-0"),
            SpokenToken("明日", 0.6, 0.95, "a:quiet-cuts-0-0"),
            SpokenToken("次", 1.0, 1.2, "a:quiet-cuts-1-1"),
        ),
        (CutSpan(0, 30, 0), CutSpan(60, 90, 30)),
        display_policy=CaptionPolicy(
            maximum_characters=4, hold_seconds=0.2, retain_recent_words=True
        ),
    ) == (
        CaptionCue("昨日", 0.1, 0.3),
        CaptionCue("昨日今日", 0.3, 0.6),
        CaptionCue("今日明日", 0.6, 1.0),
        CaptionCue("次", 1.0, 1.4),
    )


@pytest.mark.parametrize("accept_coarse,candidate", [(True, False), (False, True)])
def test_explicit_policy_cannot_silently_override_coarse_modes(
    accept_coarse: bool, candidate: bool
) -> None:
    with pytest.raises(ValueError, match="cannot be combined"):
        owned_captions(
            (),
            (),
            accept_coarse_timing=accept_coarse,
            candidate_only=candidate,
            display_policy=CaptionPolicy(),
        )
