"""Stable caption line breaks: shown text keeps its breaks, or the page switches."""

import pytest

from replay import caption_linguistics, caption_timing
from replay.caption_linguistics import CaptionLayoutError
from replay.caption_stable_lines import (
    BLOCKED,
    allowed,
    breaks,
    constrained,
    displayed_prefix,
    kept,
    kept_layout,
)
from replay.caption_timing import CaptionPolicy, SpokenToken


def measured_wrap(width: int) -> CaptionPolicy:
    def wrap(text: str) -> str:
        return caption_linguistics.wrap_measured_lines(
            text,
            measure=lambda value: (
                max(len(line) for line in value.split("\n")),
                float(len(value.split("\n"))),
            ),
            width=width,
            height=2,
            maximum_lines=2,
        )

    return CaptionPolicy(
        frame_rate=30,
        linguistic_boundaries=True,
        complete_phrases=True,
        measured_layout=wrap,
        maximum_lines=2,
        maximum_characters=80,
    )


def spoken(*words: str) -> tuple[SpokenToken, ...]:
    return tuple(
        SpokenToken(word, index * 0.5, (index + 1) * 0.5, "u") for index, word in enumerate(words)
    )


def test_free_layout_rebreaks_text_already_shown() -> None:
    cues = caption_timing.make_captions(
        spoken("私の", "ことを", "配信中に", "男よりも", "私の"), policy=measured_wrap(10)
    )
    assert [cue.text for cue in cues] == [
        "私の",
        "私のことを",
        "私のことを配信中に",
        "私のことを\n配信中に男よりも",
        "私のことを配信中に\n男よりも私の",
    ]


def test_kept_layout_extends_shown_text_without_moving_its_breaks() -> None:
    with kept():
        cues = caption_timing.make_captions(
            spoken("私の", "ことを", "配信中に", "男よりも", "私の"), policy=measured_wrap(10)
        )
    assert [cue.text for cue in cues] == [
        "私の",
        "私のことを",
        "私のことを配信中に",
        "私のことを配信中に\n男よりも",
        "私のことを配信中に\n男よりも私の",
    ]


def test_a_growth_that_would_rebreak_shown_text_starts_a_new_page() -> None:
    with kept():
        cues = caption_timing.make_captions(
            spoken("ずっと", "好きで", "いてくれる", "人が", "多い"), policy=measured_wrap(8)
        )
    assert [cue.text for cue in cues] == [
        "ずっと",
        "ずっと好きで",
        "ずっと好きで\nいてくれる",
        "ずっと好きで\nいてくれる人が",
        "多い",
    ]


def test_the_pinned_functions_are_restored_after_the_block() -> None:
    layout = vars(caption_timing)["_measured_layout"]
    wrap = vars(caption_linguistics)["wrap_measured_lines"]
    with kept():
        assert vars(caption_linguistics)["wrap_measured_lines"] is not wrap
    assert vars(caption_timing)["_measured_layout"] is layout
    assert vars(caption_linguistics)["wrap_measured_lines"] is wrap


def test_breaks_are_offsets_in_the_text_without_newlines() -> None:
    assert breaks("ab\ncde\nf") == (2, 5)
    assert breaks("abc") == ()


def test_only_breaks_inside_the_shown_text_must_match() -> None:
    assert allowed("ab\ncd\nef", (4, (2,))) is True
    assert allowed("abcd\nef", (4, ())) is True
    assert allowed("a\nbcdef", (4, ())) is False
    assert allowed("abcdef", (4, (2,))) is False


def test_constrained_measure_blocks_only_full_layouts_that_rebreak() -> None:
    measure = constrained(lambda value: (1.0, 1.0), "abcdef", (4, (2,)))
    assert measure("abcd\nef") == (BLOCKED, BLOCKED)
    assert measure("ab\ncdef") == (1.0, 1.0)
    assert measure("abcd") == (1.0, 1.0)


def test_a_page_whose_shown_prefix_does_not_fit_cannot_grow_and_failures_are_cached() -> None:
    calls: list[str] = []

    def original(text: str, policy: CaptionPolicy, context: tuple[SpokenToken, ...] = ()) -> str:
        calls.append(text)
        if text == "あいう":
            raise CaptionLayoutError("too wide")
        return text

    layout = kept_layout(original)
    policy = CaptionPolicy()
    tokens = spoken("あいう", "えお")
    with pytest.raises(CaptionLayoutError, match="too wide"):
        layout("あいうえお", policy, tokens)
    with pytest.raises(CaptionLayoutError, match="too wide"):
        layout("あいう", policy, tokens[:1])
    assert calls == ["あいう"]


def test_a_fallback_after_a_failed_constrained_wrap_does_not_count() -> None:
    def fallback_wrap(text: str) -> str:
        try:
            return caption_linguistics.wrap_measured_lines(
                text, measure=lambda value: (float(len(value)), 1.0), width=4, height=1
            )
        except CaptionLayoutError:
            return text  # a scaled indivisible word, as caption_indivisible_fit does

    with kept():
        layout = vars(caption_timing)["_measured_layout"]
        policy = CaptionPolicy(
            frame_rate=30, linguistic_boundaries=True, complete_phrases=True,
            measured_layout=fallback_wrap, maximum_lines=2, maximum_characters=80,
        )  # fmt: skip
        tokens = spoken("あいう", "えお")
        assert layout("あいう", policy, tokens[:1]) == "あいう"
        with pytest.raises(CaptionLayoutError, match="no measured layout"):
            layout("あいうえお", policy, tokens)


def test_the_displayed_prefix_ignores_look_ahead_tokens_past_the_text() -> None:
    tokens = spoken("ちょっと", "待って", "クリ", "が")
    assert [token.text for token in displayed_prefix("ちょっと待ってクリ", tokens)] == [
        "ちょっと",
        "待って",
    ]
    assert displayed_prefix("ちょっと", tokens) == ()
    assert displayed_prefix("別の文", tokens) == ()
