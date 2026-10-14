"""Real Rust Vibrato with a tiny synthetic dictionary; no private/model fixtures."""

import hashlib
import json
import sys
from pathlib import Path

import pytest
from vibrato import Vibrato

from replay import caption_linguistics
from replay.caption_display import unretained_frame_prefixes
from replay.caption_linguistics import (
    CaptionLayoutError,
    atomic_tokens,
    attached_tokens,
    boundaries,
    wrap_two_lines,
)
from replay.caption_linguistics import _tokenizer as load_tokenizer
from replay.caption_timing import CaptionPolicy, SpokenToken, make_captions


@pytest.fixture(autouse=True)
def synthetic_dictionary(monkeypatch: pytest.MonkeyPatch) -> Vibrato:
    tokenizer = Vibrato.from_textdict(
        (
            "今日,0,0,0,名詞,一般\nは,0,0,0,助詞,係助詞\n青い,0,0,0,形容詞,自立\n"
            "空,0,0,0,名詞,一般\nを,0,0,0,助詞,格助詞\n見,0,0,0,動詞,自立\n"
            "ながら,0,0,0,助詞,接続助詞\n、,0,0,0,記号,読点\nお,0,0,0,接頭詞,名詞接続\n"
            "茶,0,0,0,名詞,一般\n飲み,0,0,0,動詞,自立\nまし,0,0,0,助動詞,*\n"
            "た,0,0,0,助動詞,*\n。,0,0,0,記号,句点\nありがとう,0,0,0,感動詞,*\n"
            "ござい,0,0,0,助動詞,*\nます,0,0,0,助動詞,*\nか,0,0,0,助詞,終助詞\n"
            "abcdefghijklmnop,0,0,0,名詞,一般\nqrstuvwxyzabcd,0,0,0,名詞,一般\n"
            "ハト,0,0,0,名詞,一般\nミン,0,0,0,名詞,一般\n大好き,0,0,0,名詞,一般\n"
            "フォロワー,0,0,0,名詞,一般\nさん,0,0,0,名詞,接尾,人名\n"
            "あいうえお,0,0,0,名詞,一般\n"
            "おはよう,0,0,0,感動詞,*\nこんにちは,0,0,0,感動詞,*\n"
            "キ,0,0,0,名詞,一般,*,*,*,*,キ,キ,キ\nャ,0,0,0,名詞,一般,*,*,*,*,ャ,ャ,ャ\n"
            "バーチャル,0,0,0,名詞,一般,*,*,*,*,バーチャル,バーチャル,バーチャル\n"
            "デスクトップ,0,0,0,名詞,一般,*,*,*,*,デスクトップ,デスクトップ,デスクトップ\n"
            "クラウド,0,0,0,名詞,固有名詞,*,*,*,*,クラウド,クラウド,クラウド\n"
            "ペガサス,0,0,0,名詞,固有名詞,*,*,*,*,ペガサス,ペガサス,ペガサス\n"
            "ちゃん,0,0,0,名詞,接尾,人名,*,*,*,ちゃん,チャン,チャン\n"
        ),
        "1 1\n0 0 0\n",
        "DEFAULT 0 1 0\nSPACE 0 1 0\n0x0020 SPACE\n",
        "DEFAULT,0,0,10000,名詞,一般\nSPACE,0,0,10000,記号,空白\n",
        max_grouping_len=1,
    )
    boundaries.cache_clear()
    caption_linguistics._atomic_ends.cache_clear()
    caption_linguistics.semantic_break.cache_clear()
    load_tokenizer.cache_clear()
    monkeypatch.setattr(caption_linguistics, "_tokenizer", lambda: tokenizer)
    monkeypatch.setattr(caption_linguistics, "_bounded_tokenizer", lambda: tokenizer)
    return tokenizer


def test_real_pos_attaches_particles_auxiliaries_and_prefixes() -> None:
    assert boundaries("今日は青い空を見ながら、お茶を飲みました。") == (
        (3, 5, 7, 12, 15, 21),
        (2, 3, 5, 6, 7, 8, 11, 12, 13, 14, 15, 17, 19, 20, 21),
    )


@pytest.mark.parametrize(
    "text,expected",
    [
        ("かぬりさん", (5,)),
        ("今日はかぬりさん", (3, 8)),
        ("おはようかぬりさん", (4, 9)),
    ],
)
def test_person_suffix_protects_kana_name_leading_particle_homograph(
    text: str, expected: tuple[int, ...]
) -> None:
    assert boundaries(text)[0] == expected


@pytest.mark.parametrize("text", ["飲みかぬりさん", "か、ぬりさん", "か茶さん", "かさん"])
def test_person_suffix_does_not_absorb_actual_function_word_or_punctuation(text: str) -> None:
    assert caption_linguistics._honorific_names(text) == ()


@pytest.mark.parametrize(
    "owner,start", [("000000000000:1:cut-2", 0.1), ("000000000000:1:cut-1", 0.5)]
)
def test_honorific_name_inference_does_not_cross_cut_or_gap(owner: str, start: float) -> None:
    words = [
        SpokenToken("か", 0, 0.1, "000000000000:0:cut-1"),
        SpokenToken("ぬりさん", start, start + 0.2, owner),
    ]
    assert [
        word.text for word in atomic_tokens(words, maximum_gap=0.25, complete_phrases=True)
    ] == ["か", "ぬりさん"]


def test_adjacent_katakana_morphemes_remain_one_display_word() -> None:
    assert boundaries("ハトミン大好きありがとう") == ((4, 7, 12), (4, 7, 12))
    assert wrap_two_lines("ハトミン大好き", width=4) == "ハトミン\n大好き"
    with pytest.raises(CaptionLayoutError):
        wrap_two_lines("ハトミン", width=3)


def test_known_katakana_words_allow_a_real_lexical_boundary() -> None:
    assert boundaries("バーチャルデスクトップ") == ((5, 11), (5, 11))
    assert wrap_two_lines("バーチャルデスクトップ", width=6) == "バーチャル\nデスクトップ"


def test_dictionary_missing_is_explicit(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(caption_linguistics, "DICTIONARY_PATH", tmp_path / "absent")
    with pytest.raises(ValueError, match="Install the pinned Vibrato"):
        load_tokenizer()


def test_changed_dictionary_is_rejected(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    path = tmp_path / "changed.dic"
    path.write_bytes(b"changed")
    monkeypatch.setattr(caption_linguistics, "DICTIONARY_PATH", path)
    with pytest.raises(ValueError, match="pinned checksum"):
        load_tokenizer()


def test_dictionary_load_checks_bytes_once(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    synthetic_dictionary: Vibrato,
) -> None:
    path = tmp_path / "synthetic.dic"
    path.write_bytes(b"synthetic")
    monkeypatch.setattr(caption_linguistics, "DICTIONARY_PATH", path)
    monkeypatch.setattr(
        caption_linguistics, "DICTIONARY_SHA256", hashlib.sha256(b"synthetic").hexdigest()
    )
    calls: list[tuple[bytes, bool]] = []

    def construct(data: bytes, *, ignore_space: bool, max_grouping_len: int) -> Vibrato:
        assert max_grouping_len == 1
        calls.append((data, ignore_space))
        return synthetic_dictionary

    monkeypatch.setattr(caption_linguistics, "Vibrato", construct)
    assert load_tokenizer().tokenize_to_surfaces("今日は") == ["今日", "は"]
    assert load_tokenizer().tokenize_to_surfaces("空を") == ["空", "を"]
    assert calls == [(b"synthetic", False)]


def test_malformed_features_rejected(monkeypatch: pytest.MonkeyPatch) -> None:
    tokenizer = Vibrato.from_textdict(
        "空,0,0,0,名詞\n",
        "1 1\n0 0 0\n",
        "DEFAULT 0 1 0\n",
        "DEFAULT,0,0,10000,名詞\n",
    )
    monkeypatch.setattr(caption_linguistics, "_tokenizer", lambda: tokenizer)
    with pytest.raises(ValueError, match="Invalid Vibrato feature"):
        boundaries("空")


def test_phrase_break_and_text_retention() -> None:
    assert (
        wrap_two_lines("今日は青い空を見ながら、お茶を飲みました。", width=12)
        == "今日は青い空を見ながら、\nお茶を飲みました。"
    )
    assert wrap_two_lines("今日は") == "今日は"


def test_morpheme_break_when_one_phrase_exceeds_line() -> None:
    assert wrap_two_lines("ありがとうございます", width=5) == "ありがとう\nございます"


@pytest.mark.parametrize("text", ["", "a\0b", "a" * 8193])
def test_invalid_morphological_input(text: str) -> None:
    with pytest.raises(ValueError, match="Invalid Japanese"):
        boundaries(text)


@pytest.mark.parametrize(
    "text,width",
    [("今日\nは", 18), ("今日\rは", 18), ("今日は", 0), ("今日は", True), ("今日は", 81)],
)
def test_invalid_layout(text: str, width: int) -> None:
    with pytest.raises(ValueError, match="Invalid linguistic"):
        wrap_two_lines(text, width=width)


def test_no_character_splitting_or_budget_waiver() -> None:
    with pytest.raises(CaptionLayoutError, match="No intact morpheme"):
        wrap_two_lines("abcdefghijk", width=6)
    with pytest.raises(CaptionLayoutError, match="two-line display"):
        wrap_two_lines("今日は青い空を見ました", width=3)


def test_original_ownership_and_gap_are_hard_boundaries() -> None:
    words = [
        SpokenToken("今日", 0.0, 0.2, "a"),
        SpokenToken("は", 0.2, 0.3, "a"),
        SpokenToken("青い", 0.3, 0.5, "a"),
        SpokenToken("空", 0.5, 0.6, "a"),
        SpokenToken("を", 0.6, 0.7, "a"),
        SpokenToken("見", 0.7, 0.8, "a"),
        SpokenToken("ます", 0.8, 1.0, "b"),
        SpokenToken("か", 1.5, 1.6, "b"),
    ]
    before = tuple(words)
    assert attached_tokens(words, maximum_gap=0.25) == [
        False,
        True,
        False,
        False,
        True,
        False,
        False,
        False,
    ]
    assert tuple(words) == before
    assert attached_tokens([], maximum_gap=0.25) == []


def test_caption_switch_moves_before_noun_not_its_particle() -> None:
    words = (
        SpokenToken("今日", 0.0, 0.2, "a"),
        SpokenToken("は", 0.2, 0.3, "a"),
        SpokenToken("空", 0.3, 0.4, "a"),
        SpokenToken("を", 0.4, 0.5, "a"),
    )
    result = make_captions(
        words, policy=CaptionPolicy(maximum_characters=4, linguistic_boundaries=True)
    )
    assert [cue.text for cue in result] == ["今日", "今日は", "空", "空を"]
    assert [cue.start for cue in result] == [0.0, 0.2, 0.3, 0.4]


def test_reset_uses_intact_line_budget_not_just_total_characters() -> None:
    words = (
        SpokenToken("abcdefghijklmnop", 0.0, 0.2, "a"),
        SpokenToken("あいうえお", 0.2, 0.4, "a"),
        SpokenToken("qrstuvwxyzabcd", 0.4, 0.6, "a"),
    )
    result = make_captions(words, policy=CaptionPolicy(linguistic_boundaries=True))
    assert [cue.text for cue in result] == [
        "abcdefghijklmnop",
        "abcdefghijklmnopあいうえお",
        "qrstuvwxyzabcd",
    ]
    assert [cue.start for cue in result] == [0.0, 0.2, 0.4]
    assert [cue.end for cue in result] == [0.2, 0.4, 0.6799999999999999]


def test_frame_lookahead_accounts_for_the_following_particle() -> None:
    words = (
        SpokenToken("今日", 0.0, 0.31, "a"),
        SpokenToken("は", 0.31, 0.32, "a"),
        SpokenToken("空", 0.32, 0.5, "a"),
        SpokenToken("を", 0.5, 0.7, "a"),
    )
    policy = CaptionPolicy(maximum_characters=4, linguistic_boundaries=True, frame_rate=30)
    assert unretained_frame_prefixes(words, policy=policy) == ()
    assert [cue.start for cue in make_captions(words, policy=policy)] == [0.0, 0.31, 0.32, 0.5]


def test_oversized_future_phrase_does_not_erase_each_prefix() -> None:
    words = (
        SpokenToken("飲", 0.0, 0.1, "a"),
        SpokenToken("み", 0.1, 0.2, "a"),
        SpokenToken("まし", 0.2, 0.4, "a"),
        SpokenToken("まし", 0.4, 0.6, "a"),
        SpokenToken("まし", 0.6, 0.8, "a"),
        SpokenToken("まし", 0.8, 1.0, "a"),
        SpokenToken("まし", 1.0, 1.2, "a"),
        SpokenToken("まし", 1.2, 1.4, "a"),
    )
    result = make_captions(
        words, policy=CaptionPolicy(maximum_characters=8, linguistic_boundaries=True)
    )
    assert [cue.text for cue in result[:2]] == ["飲み", "飲みまし"]
    assert [cue.start for cue in result] == [0.0, 0.2, 0.4, 0.6, 0.8, 1.0, 1.2]


def test_atomic_unknown_word_keeps_first_onset_and_following_word_clock() -> None:
    words = (
        SpokenToken("ハ", 0, 0.4, "a"),
        SpokenToken("ト", 0.4, 0.8, "a"),
        SpokenToken("ミ", 0.8, 1.2, "a"),
        SpokenToken("ン", 1.2, 1.6, "a"),
        SpokenToken("大好き", 1.6, 2, "a"),
    )
    assert [
        (c.text, c.start, c.end)
        for c in make_captions(words, policy=CaptionPolicy(linguistic_boundaries=True))
    ] == [("ハトミン", 0, 1.6), ("ハトミン大好き", 1.6, 2.08)]


def test_atomic_chunk_boundary_is_not_a_cut_and_preserves_long_vowel() -> None:
    words = (
        SpokenToken("フ", 0, 0.2, "/synthetic/0001.raw.json:cut-7"),
        SpokenToken("ォ", 0.24, 0.3, "/synthetic/0002.raw.json:cut-7"),
        SpokenToken("ロ", 0.3, 0.4, "/synthetic/0002.raw.json:cut-7"),
        SpokenToken("ワ", 0.4, 0.5, "/synthetic/0002.raw.json:cut-7"),
        SpokenToken("ー", 0.5, 0.6, "/synthetic/0002.raw.json:cut-7"),
        SpokenToken("空", 0.6, 0.8, "/synthetic/0002.raw.json:cut-7"),
    )
    assert [
        (c.text, c.start)
        for c in make_captions(words, policy=CaptionPolicy(linguistic_boundaries=True))
    ] == [("フォロワー", 0), ("フォロワー空", 0.6)]


def test_assembled_fragment_indices_do_not_create_utterance_boundaries() -> None:
    words = [
        SpokenToken("フ", 0, 0.2, "000000000000:10:cut-7"),
        SpokenToken("ォ", 0.2, 0.3, "000000000000:11:cut-7"),
        SpokenToken("ロ", 0.3, 0.4, "000000000000:12:cut-7"),
        SpokenToken("ワ", 0.4, 0.5, "000000000000:13:cut-7"),
        SpokenToken("ー", 0.5, 0.6, "000000000000:14:cut-7"),
        SpokenToken("さん", 0.6, 0.7, "000000000000:15:cut-7"),
    ]
    assert atomic_tokens(words, maximum_gap=0.25) == [
        SpokenToken("フォロワーさん", 0, 0.7, "assembled:cut-7")
    ]
    assert words[0] == SpokenToken("フ", 0, 0.2, "000000000000:10:cut-7")


@pytest.mark.parametrize(
    "owner",
    [
        "000000000000:11:cut-8",
        "region:1:000000000000:11:cut-7",
        "source:other:000000000000:11:cut-7",
        "scene:other:000000000000:11:cut-7",
        "speaker:000000000000:11:cut-7",
        "00000000000:11:cut-7",
        "000000000000:11:quiet-cuts-7-8",
    ],
)
def test_assembled_cut_context_and_unknown_owner_boundaries_remain(owner: str) -> None:
    words = [
        SpokenToken("ハト", 0, 0.2, "000000000000:10:cut-7"),
        SpokenToken("ミン", 0.2, 0.4, owner),
    ]
    assert [token.text for token in atomic_tokens(words, maximum_gap=0.25)] == ["ハト", "ミン"]


def test_assembled_continuity_preserves_region_and_real_recognition_gap() -> None:
    assert (
        caption_linguistics.recognition_continuity("region:2:speaker:000000000000:17:cut-7")
        == "region:2:speaker:assembled:cut-7"
    )
    words = [
        SpokenToken("ハト", 0, 0.2, "000000000000:10:cut-7"),
        SpokenToken("ミン", 0.5, 0.7, "000000000000:11:cut-7"),
    ]
    assert [token.text for token in atomic_tokens(words, maximum_gap=0.25)] == ["ハト", "ミン"]


def test_assembled_window_ownership_does_not_split_a_word() -> None:
    words = [
        SpokenToken("ハト", 0, 0.2, "source:a:scene:b:speaker:c:000000000000:10:cut-7"),
        SpokenToken("ミン", 0.22, 0.4, "source:a:scene:b:speaker:c:000000480000:11:cut-7"),
    ]
    assert atomic_tokens(words, maximum_gap=0.25) == [
        SpokenToken("ハトミン", 0, 0.4, "source:a:scene:b:speaker:c:assembled:cut-7")
    ]
    assert words == [
        SpokenToken("ハト", 0, 0.2, "source:a:scene:b:speaker:c:000000000000:10:cut-7"),
        SpokenToken("ミン", 0.22, 0.4, "source:a:scene:b:speaker:c:000000480000:11:cut-7"),
    ]


def test_assembled_window_phrase_retains_particle_and_independent_greeting() -> None:
    words = [
        SpokenToken("飲み", 0, 0.12, "000000000000:10:cut-7"),
        SpokenToken("まし", 0.12, 0.2, "000000000000:11:cut-7"),
        SpokenToken("た", 0.2, 0.295, "000000000000:12:cut-7"),
        SpokenToken("か", 0.312, 0.332, "000000480000:1:cut-7"),
        SpokenToken("ありがとう", 0.332, 0.6, "000000480000:2:cut-7"),
    ]
    assert atomic_tokens(words, maximum_gap=0.25, complete_phrases=True) == [
        SpokenToken("飲みましたか", 0, 0.332, "assembled:cut-7"),
        SpokenToken("ありがとう", 0.332, 0.6, "assembled:cut-7"),
    ]
    cues = make_captions(
        tuple(words),
        policy=CaptionPolicy(frame_rate=30, linguistic_boundaries=True, complete_phrases=True),
    )
    assert [(cue.text, cue.start) for cue in cues] == [
        ("飲みましたか", 0),
        ("ありがとう", 0.332),
    ]
    assert words[-2] == SpokenToken("か", 0.312, 0.332, "000000480000:1:cut-7")


def test_assembled_window_continuity_still_rejects_a_real_gap() -> None:
    words = [
        SpokenToken("ハト", 0, 0.2, "000000000000:10:cut-7"),
        SpokenToken("ミン", 0.5, 0.7, "000000480000:11:cut-7"),
    ]
    assert [token.text for token in atomic_tokens(words, maximum_gap=0.25)] == ["ハト", "ミン"]


@pytest.mark.parametrize(
    "owner", ["/synthetic/0002.raw.json:cut-8", "/other/0002.raw.json:cut-7", "unknown"]
)
def test_atomic_never_joins_different_cut_or_source(owner: str) -> None:
    words = [
        SpokenToken("ハト", 0, 0.2, "/synthetic/0001.raw.json:cut-7"),
        SpokenToken("ミン", 0.2, 0.4, owner),
    ]
    assert [token.text for token in atomic_tokens(words, maximum_gap=0.25)] == ["ハト", "ミン"]


def test_atomic_keeps_real_gap_and_coarse_token_timing() -> None:
    words = [SpokenToken("今日は", 0, 0.2, "a"), SpokenToken("青い空", 1, 2, "a")]
    assert atomic_tokens(words, maximum_gap=0.25) == [
        SpokenToken("今日は", 0, 0.2, "a"),
        SpokenToken("青い空", 1, 2, "a"),
    ]


def test_honorific_is_kept_with_its_name_without_absorbing_next_word() -> None:
    words = [
        SpokenToken("フォロワー", 0, 0.4, "a"),
        SpokenToken("さん", 0.4, 0.5, "a"),
        SpokenToken("空", 0.5, 0.7, "a"),
    ]
    assert [
        (word.text, word.start, word.end) for word in atomic_tokens(words, maximum_gap=0.25)
    ] == [("フォロワーさん", 0, 0.5), ("空", 0.5, 0.7)]


def test_atomic_known_loanword_compound_keeps_independent_onsets() -> None:
    words = [
        SpokenToken("バー", 0, 0.2, "a"),
        SpokenToken("チャル", 0.2, 0.4, "a"),
        SpokenToken("デスク", 0.4, 0.6, "a"),
        SpokenToken("トップ", 0.6, 0.8, "a"),
    ]
    assert [(word.text, word.start) for word in atomic_tokens(words, maximum_gap=0.25)] == [
        ("バーチャル", 0),
        ("デスクトップ", 0.4),
    ]


def test_nonlexical_long_notes_still_removed_with_linguistic_mode() -> None:
    words = (
        SpokenToken("空", 0, 0.2, "a"),
        SpokenToken("ーーーー", 0.2, 0.4, "a"),
        SpokenToken("ー", 1, 1.2, "a"),
    )
    assert [
        (cue.text, cue.start)
        for cue in make_captions(words, policy=CaptionPolicy(linguistic_boundaries=True))
    ] == [("空", 0)]


def test_independent_interjections_and_final_particles_start_new_pages() -> None:
    words = (
        SpokenToken("おはよう", 0, 0.2, "a"),
        SpokenToken("おはよう", 0.2, 0.4, "a"),
        SpokenToken("か", 0.4, 0.5, "a"),
        SpokenToken("こんにちは", 0.5, 0.7, "a"),
        SpokenToken("か", 0.7, 0.8, "a"),
        SpokenToken("今日", 0.8, 1, "a"),
    )
    assert [
        (c.text, c.start)
        for c in make_captions(words, policy=CaptionPolicy(linguistic_boundaries=True))
    ] == [
        ("おはよう", 0),
        ("おはよう", 0.2),
        ("おはようか", 0.4),
        ("こんにちは", 0.5),
        ("こんにちはか", 0.7),
        ("今日", 0.8),
    ]


def test_complete_phrases_bind_inflection_and_particle_at_known_onset() -> None:
    words = (
        SpokenToken("今日", 0, 0.2, "a"),
        SpokenToken("は", 0.2, 0.3, "a"),
        SpokenToken("飲", 0.3, 0.4, "a"),
        SpokenToken("み", 0.4, 0.5, "a"),
        SpokenToken("まし", 0.5, 0.6, "a"),
        SpokenToken("た", 0.6, 0.7, "a"),
        SpokenToken("か", 0.7, 0.8, "a"),
        SpokenToken("こんにちは", 0.8, 1, "a"),
    )
    result = make_captions(
        words, policy=CaptionPolicy(linguistic_boundaries=True, complete_phrases=True)
    )
    assert [(cue.text, cue.start, cue.end) for cue in result] == [
        ("今日は", 0, 0.3),
        ("今日は飲みましたか", 0.3, 0.8),
        ("こんにちは", 0.8, 1.08),
    ]


def test_complete_phrase_cannot_fall_back_to_character_overflow() -> None:
    with pytest.raises(ValueError, match="Native token exceeds display budget"):
        make_captions(
            (SpokenToken("ありがとうございます", 0, 1, "a"),),
            policy=CaptionPolicy(
                linguistic_boundaries=True, complete_phrases=True, maximum_characters=6
            ),
        )


def test_complete_phrases_require_morphology() -> None:
    with pytest.raises(ValueError, match="requires linguistic"):
        CaptionPolicy(complete_phrases=True)


def test_phrase_small_kana_cannot_be_a_known_dictionary_boundary() -> None:
    words = [
        SpokenToken("キ", 0, 0.2, "a"),
        SpokenToken("ャ", 0.2, 0.4, "a"),
        SpokenToken("空", 0.4, 0.6, "a"),
    ]
    assert [
        (t.text, t.start) for t in atomic_tokens(words, maximum_gap=0.25, complete_phrases=True)
    ] == [("キャ", 0), ("空", 0.4)]


def test_slow_phrase_uses_existing_morpheme_onsets_without_coarse_token_error() -> None:
    words = (
        SpokenToken("飲み", 0, 1, "a"),
        SpokenToken("まし", 1, 2, "a"),
        SpokenToken("た", 2, 3, "a"),
        SpokenToken("か", 3, 4, "a"),
    )
    assert [
        (c.text, c.start)
        for c in make_captions(
            words, policy=CaptionPolicy(linguistic_boundaries=True, complete_phrases=True)
        )
    ] == [("飲み", 0), ("飲みまし", 1), ("飲みました", 2), ("飲みましたか", 3)]


def test_slow_protected_word_keeps_fine_recognition_provenance() -> None:
    words = (
        SpokenToken("フ", 0, 0.5, "a"),
        SpokenToken("ォロ", 0.5, 2.9, "a"),
        SpokenToken("ワー", 2.9, 3.3, "a"),
        SpokenToken("さん", 3.3, 3.8, "a"),
    )
    policy = CaptionPolicy(linguistic_boundaries=True, complete_phrases=True)
    cues = make_captions(words, policy=policy)
    assert [(cue.text, cue.start, cue.end) for cue in cues] == [("フォロワーさん", 0, 3.88)]
    assert words[1].end == 2.9
    with pytest.raises(ValueError, match="too coarse"):
        make_captions((SpokenToken("フォロワーさん", 0, 3.8, "a"),), policy=policy)


def test_oversized_protected_compound_uses_known_lexical_onsets() -> None:
    words = [
        SpokenToken("クラウド", 0, 0.6, "a"),
        SpokenToken("ペガサス", 0.6, 1.1, "a"),
        SpokenToken("さん", 1.1, 1.3, "a"),
    ]
    normal = atomic_tokens(words, maximum_gap=0.25, complete_phrases=True)
    assert [token.text for token in normal] == ["クラウドペガサスさん"]
    fitted = atomic_tokens(
        words,
        maximum_gap=0.25,
        complete_phrases=True,
        accept_phrase=lambda token: len(token.text) <= 6,
    )
    assert [(token.text, token.start, token.end) for token in fitted] == [
        ("クラウド", 0, 0.6),
        ("ペガサスさん", 0.6, 1.3),
    ]
    # A coarse token gives no internal onset, even if morphology finds a boundary.
    assert [
        token.text
        for token in atomic_tokens(
            [SpokenToken("クラウドペガサス", 0, 1, "a")],
            maximum_gap=0.25,
            complete_phrases=True,
            accept_phrase=lambda token: len(token.text) <= 4,
        )
    ] == ["クラウドペガサス"]
    assert [
        token.text
        for token in atomic_tokens(
            [SpokenToken("ハト", 0, 0.5, "a"), SpokenToken("ミン", 0.5, 1, "a")],
            maximum_gap=0.25,
            complete_phrases=True,
            accept_phrase=lambda token: len(token.text) <= 3,
        )
    ] == ["ハトミン"]


def test_repeated_unknown_compound_preserves_whole_repeated_units() -> None:
    text = "ポヨデスクトップポヨ"
    ends = caption_linguistics._overflow_ends(text)
    assert 2 in ends and len(text) - 2 in ends
    # A repeated phonetic unit is retained whole even with a short vocal middle.
    assert caption_linguistics._overflow_ends("ポヨポポポヨ") == frozenset({2, 4, 6})
    words = [
        SpokenToken("ポ", 0, 0.2, "a"),
        SpokenToken("ヨ", 0.2, 0.4, "a"),
        SpokenToken("デスクトップ", 0.4, 1.6, "a"),
        SpokenToken("ポヨ", 1.6, 2, "a"),
    ]
    result = atomic_tokens(
        words,
        maximum_gap=0.25,
        complete_phrases=True,
        accept_phrase=lambda token: len(token.text) <= 6,
    )
    assert [(token.text, token.start, token.end) for token in result] == [
        ("ポヨ", 0, 0.4),
        ("デスクトップ", 0.4, 1.6),
        ("ポヨ", 1.6, 2),
    ]


def test_morphology_diagnostic_cli(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.setattr(sys, "argv", ["caption_linguistics", "フォロワーさん"])
    caption_linguistics.main()
    data = json.loads(capsys.readouterr().out)
    assert data["atomic"] == [7]
    assert "".join(item["text"] for item in data["tokens"]) == "フォロワーさん"


def test_bounded_unknown_analysis_exposes_only_adjacent_known_anchors(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    unknown = Vibrato.from_textdict(
        "無,0,0,0,名詞,一般\n",
        "1 1\n0 0 0\n",
        "DEFAULT 0 1 0\n",
        "DEFAULT,0,0,10000,名詞,固有名詞,組織,*,*,*,*\n",
    )
    monkeypatch.setattr(caption_linguistics, "_tokenizer", lambda: unknown)
    words = [SpokenToken("クラウド", 0, 1, "a"), SpokenToken("ペガサス", 1, 2, "a")]
    assert [
        (item.text, item.start)
        for item in atomic_tokens(
            words,
            maximum_gap=0.25,
            complete_phrases=True,
            accept_phrase=lambda token: len(token.text) <= 4,
        )
    ] == [("クラウド", 0), ("ペガサス", 1)]
    for text in ("クラウドポヨペガサス", "クラウドさんペガサス"):
        assert caption_linguistics._overflow_ends(text) == frozenset({len(text)})


def test_repetition_edges_keep_small_kana_and_long_vowels_together() -> None:
    assert caption_linguistics._overflow_ends("ギャポーギャポー") == frozenset({4, 8})
    assert 9 in caption_linguistics._overflow_ends(" ポヨデスクトップポヨスピ")
    words = [SpokenToken("ギャポー", 0, 0.6, "a"), SpokenToken("ギャポー", 0.6, 1.3, "a")]
    assert [
        (token.text, token.start)
        for token in atomic_tokens(
            words,
            maximum_gap=0.25,
            complete_phrases=True,
            accept_phrase=lambda token: len(token.text) <= 4,
        )
    ] == [("ギャポー", 0), ("ギャポー", 0.6)]


def test_raw_chunk_punctuation_keeps_text_and_sentence_reset() -> None:
    words = (
        SpokenToken("今日", 0, 0.2, "/source/0001.raw.json:cut-1"),
        SpokenToken("。", 0.2, 0.21, "/source/0002.raw.json:cut-1"),
        SpokenToken("空", 0.21, 0.4, "/source/0002.raw.json:cut-1"),
    )
    assert [
        (c.text, c.start)
        for c in make_captions(
            words, policy=CaptionPolicy(linguistic_boundaries=True, complete_phrases=True)
        )
    ] == [("今日。", 0), ("空", 0.21)]


def test_overlapping_recognized_long_mark_survives() -> None:
    words = (
        SpokenToken("フォロワ", 0, 0.4, "a"),
        SpokenToken("ー", 0.3, 0.5, "a"),
    )
    assert [
        (c.text, c.start, c.end)
        for c in make_captions(words, policy=CaptionPolicy(linguistic_boundaries=True))
    ] == [("フォロワー", 0, 0.58)]


def test_frame_audit_uses_same_aggregated_units_as_planner() -> None:
    words = (
        SpokenToken("ハ", 0, 0.1, "a"),
        SpokenToken("ト", 0.1, 0.2, "a"),
        SpokenToken("ミ", 0.2, 0.3, "a"),
        SpokenToken("ン", 0.3, 0.4, "a"),
    )
    assert (
        unretained_frame_prefixes(
            words,
            policy=CaptionPolicy(linguistic_boundaries=True, complete_phrases=True, frame_rate=30),
        )
        == ()
    )


def test_single_native_token_cannot_be_split_at_invented_times() -> None:
    words = (SpokenToken("zyxwvutsrqponmlkjihg", 0.0, 0.2, "a"),)
    with pytest.raises(ValueError, match="Native token exceeds display budget"):
        make_captions(words, policy=CaptionPolicy(linguistic_boundaries=True))


@pytest.mark.parametrize("lines,overrun", [(1, False), (3, False), (2, True)])
def test_linguistic_policy_requires_supported_fixed_layout(lines: int, overrun: bool) -> None:
    with pytest.raises(ValueError, match="two fixed-width lines"):
        CaptionPolicy(linguistic_boundaries=True, maximum_lines=lines, tail_overrun=overrun)


@pytest.mark.parametrize("flag", [1, None, "true"])
def test_policy_requires_boolean(flag: bool) -> None:
    with pytest.raises(ValueError, match="must be a boolean"):
        CaptionPolicy(linguistic_boundaries=flag)


def test_three_measured_phrase_lines_preserve_text() -> None:
    def measure(text: str) -> tuple[float, float]:
        lines = text.split("\n")
        return float(max(map(len, lines))), float(len(lines))

    assert (
        caption_linguistics.wrap_measured_lines(
            "今日は青い空を", measure=measure, width=3, height=3, maximum_lines=3
        )
        == "今日は\n青い\n空を"
    )
    with pytest.raises(CaptionLayoutError):
        caption_linguistics.wrap_measured_lines(
            "今日は青い空を", measure=measure, width=3, height=2, maximum_lines=3
        )


def test_measured_polite_chain_wraps_without_inventing_a_time() -> None:
    assert (
        caption_linguistics.wrap_measured_lines(
            "ありがとうございましたー!",
            measure=lambda text: (
                float(max(map(len, text.split("\n")))),
                float(len(text.split("\n"))),
            ),
            width=8,
            height=2,
        )
        == "ありがとう\nございましたー!"
    )


def test_rhythmic_unit_preserves_partial_tail_before_width_overflow() -> None:
    words = [
        SpokenToken("キ", 0, 0.1, "a"),
        SpokenToken("ャ", 0.1, 0.2, "a"),
        SpokenToken("ポ", 0.2, 0.3, "a"),
        SpokenToken("キ", 0.3, 0.4, "a"),
        SpokenToken("ャ", 0.4, 0.5, "a"),
        SpokenToken("ポ", 0.5, 0.6, "a"),
        SpokenToken("キ", 0.6, 0.7, "a"),
    ]
    assert [
        (token.text, token.start, token.end)
        for token in atomic_tokens(
            words,
            maximum_gap=0.25,
            complete_phrases=True,
            accept_phrase=lambda token: True,
        )
    ] == [("キャポ", 0, 0.3), ("キャポキ", 0.3, 0.7)]
    assert caption_linguistics._rhythmic_ends("ッパッパ", frozenset({4})) == frozenset({4})
    assert caption_linguistics._rhythmic_ends("キャキャ", frozenset({4})) == frozenset({4})
    assert (
        caption_linguistics.wrap_measured_lines(
            "キャポキャポキ",
            measure=lambda text: (
                float(max(map(len, text.split("\n")))),
                float(len(text.split("\n"))),
            ),
            width=4,
            height=2,
        )
        == "キャポ\nキャポキ"
    )


def test_overflow_honorific_protects_unknown_name_not_previous_compound() -> None:
    words = [
        SpokenToken("ペガサス", 0, 0.4, "a"),
        SpokenToken("ヌ", 0.4, 0.6, "a"),
        SpokenToken("ゼ", 0.6, 0.8, "a"),
        SpokenToken("ちゃん", 0.8, 1.0, "a"),
    ]
    assert [
        (token.text, token.start, token.end)
        for token in atomic_tokens(
            words,
            maximum_gap=0.25,
            complete_phrases=True,
            accept_phrase=lambda token: len(token.text) <= 5,
        )
    ] == [("ペガサス", 0, 0.4), ("ヌゼちゃん", 0.4, 1.0)]
    assert caption_linguistics._overflow_ends("ヌゼちゃん") == frozenset({5})


def test_unknown_compound_uses_existing_balanced_speech_edges() -> None:
    words = [
        SpokenToken("ヌ", 0, 0.2, "a"),
        SpokenToken("クラウド", 0.2, 0.6, "a"),
        SpokenToken("ゼ", 0.6, 0.8, "a"),
        SpokenToken("ペガサス", 0.8, 1.2, "a"),
    ]
    assert [
        (token.text, token.start, token.end)
        for token in atomic_tokens(
            words,
            maximum_gap=0.25,
            complete_phrases=True,
            accept_phrase=lambda token: len(token.text) <= 6,
        )
    ] == [("ヌクラウド", 0, 0.6), ("ゼペガサス", 0.6, 1.2)]


def test_unknown_compound_keeps_coarse_clock_and_small_kana() -> None:
    words = [SpokenToken("ヌクラウドゼペガサス", 0, 1.2, "a")]
    assert (
        caption_linguistics._recognition_overflow(words, lambda token: len(token.text) <= 6) is None
    )
    assert (
        caption_linguistics._recognition_overflow(
            [SpokenToken("ペガサス", 0, 0.5, "a"), SpokenToken("ャヌ", 0.5, 1, "a")],
            lambda token: len(token.text) <= 4,
        )
        is None
    )
    assert (
        caption_linguistics._recognition_overflow(
            [SpokenToken("ヌゼ", 0, 0.5, "a"), SpokenToken("ポロ", 0.5, 1, "a")],
            lambda token: len(token.text) <= 2,
        )
        is None
    )


def test_unknown_compound_surrounding_spaces_preserve_text_and_existing_edges() -> None:
    words = [
        SpokenToken(" ヌ", 0, 0.2, "a"),
        SpokenToken("クラウド", 0.2, 0.6, "a"),
        SpokenToken("ゼ", 0.6, 0.8, "a"),
        SpokenToken("ペガサス ", 0.8, 1.2, "a"),
    ]
    before = tuple(words)
    result = caption_linguistics._recognition_overflow(
        words, lambda token: len(token.text.strip()) <= 6
    )
    assert result is not None
    assert [(token.text, token.start, token.end) for token in result] == [
        (" ヌクラウド", 0, 0.6),
        ("ゼペガサス ", 0.6, 1.2),
    ]
    assert "".join(token.text for token in result) == "".join(token.text for token in words)
    assert tuple(words) == before


def test_space_is_not_lexical_support_or_permission_to_split_known_words() -> None:
    assert (
        caption_linguistics._recognition_overflow(
            [SpokenToken(" ヌ", 0, 0.2, "a"), SpokenToken("クラウド ", 0.2, 0.6, "a")],
            lambda token: len(token.text.strip()) <= 4,
        )
        is None
    )
    assert (
        caption_linguistics._recognition_overflow(
            [SpokenToken(" クラ", 0, 0.2, "a"), SpokenToken("ウド ", 0.2, 0.6, "a")],
            lambda token: len(token.text.strip()) <= 2,
        )
        is None
    )
    assert (
        caption_linguistics._recognition_overflow(
            [SpokenToken(" ヌクラウド ", 0, 0.6, "a"), SpokenToken("ゼペガサス ", 0.6, 1.2, "a")],
            lambda token: len(token.text.strip()) <= 6,
        )
        is None
    )


@pytest.mark.parametrize("maximum_lines", [0, 4, True])
def test_measured_line_limit_is_bounded(maximum_lines: int) -> None:
    with pytest.raises(ValueError, match="Invalid measured"):
        caption_linguistics.wrap_measured_lines(
            "今日は",
            measure=lambda text: (3, 1),
            width=3,
            height=3,
            maximum_lines=maximum_lines,
        )


def test_three_line_search_has_a_resource_budget(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        caption_linguistics, "boundaries", lambda text: (tuple(range(1, len(text))), ())
    )
    with pytest.raises(CaptionLayoutError, match="candidate budget"):
        caption_linguistics.wrap_measured_lines(
            "a" * 365,
            measure=lambda text: (float(max(map(len, text.split("\n")))), 1),
            width=2,
            height=3,
            maximum_lines=3,
        )


def test_measured_fitting_line_does_not_prepare_unused_splits() -> None:
    prepared: list[tuple[str, ...]] = []
    assert (
        caption_linguistics.wrap_measured_lines(
            "今日は",
            measure=lambda text: (float(len(text)), 1.0),
            width=10,
            height=2,
            prepare_lines=prepared.append,
        )
        == "今日は"
    )
    assert prepared == []


def test_measured_batch_failure_does_not_fall_back_to_individual_calls() -> None:
    def fail_preparation(lines: tuple[str, ...]) -> None:
        raise RuntimeError("Synthetic batch failure")

    with pytest.raises(RuntimeError, match="Synthetic batch failure"):
        caption_linguistics.wrap_measured_lines(
            "今日は青い空です",
            measure=lambda text: (float(len(text)), 1.0),
            width=4,
            height=2,
            prepare_lines=fail_preparation,
        )
