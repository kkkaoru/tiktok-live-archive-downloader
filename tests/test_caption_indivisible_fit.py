"""Fine speech may fit uniformly; coarse speech never gains invented timing."""

from collections.abc import Iterable
from fractions import Fraction

import pytest
from test_caption_linguistics import synthetic_dictionary as synthetic_dictionary

from replay.caption_indivisible_fit import IndivisibleFit
from replay.caption_linguistics import CaptionLayoutError
from replay.caption_native_paging import assemble, paged_cues
from replay.caption_scale import scale_fields
from replay.caption_timing import CaptionCue, CaptionPolicy, ScaledCaptionCue, SpokenToken


def test_outer_space_alias_preserves_fine_original_anchors() -> None:
    tokens = (
        SpokenToken(" あ", 0, 0.2, "u"),
        SpokenToken("いうえおあ", 0.2, 0.4, "u"),
        SpokenToken("いうえお", 0.4, 0.6, "u"),
    )
    fit = IndivisibleFit(tokens, layout=Measurements(), maximum_height=20)
    assert fit.eligible("あいうえおあいうえお") is True
    result = fit.wrap("あいうえおあいうえお")
    assert result == "あいうえお\nあいうえお"
    assert fit.evidence[result].fragments == 3
    assert fit.evidence[result].maximum_fragment_seconds == 0.2
    assert tokens[0] == SpokenToken(" あ", 0, 0.2, "u")


class Measurements:
    width = 100.0
    expected_point_size: float | None = 20.0

    def prepare_lines(self, texts: Iterable[str]) -> None:
        tuple(texts)

    def measure(self, text: str) -> tuple[float, float]:
        return max(map(len, text.split("\n"))) * 20.0, len(text.split("\n")) * 20.0


def test_repeated_outer_space_words_keep_each_occurrence_evidence() -> None:
    tokens = (
        SpokenToken(" こ", 0, 0.2, "same"),
        SpokenToken("ぉぉぉぉ", 0.2, 0.4, "same"),
        SpokenToken("ぉぉぉぉぉぉ", 0.4, 0.6, "same"),
        SpokenToken("こ", 2, 2.5, "same"),
        SpokenToken("ぉぉぉぉ", 2.5, 3, "same"),
        SpokenToken("ぉぉぉぉぉぉ", 3, 3.5, "same"),
    )
    cues, hidden, widened = paged_cues(tokens, layout=Measurements(), rate=30, duration=Fraction(4))
    assert hidden == widened == 0
    assert len(cues) == 2
    assert isinstance(cues[0], ScaledCaptionCue)
    assert isinstance(cues[1], ScaledCaptionCue)
    assert (cues[0].text, cues[0].start) == ("こぉぉぉぉぉぉぉぉぉぉ", 0)
    assert cues[0].end == pytest.approx(0.68)
    assert (cues[1].text, cues[1].start, cues[1].end) == ("こぉぉぉぉぉぉぉぉぉぉ", 2, 3.58)
    assert cues[0].scale.maximum_fragment_seconds == pytest.approx(0.2)
    assert cues[1].scale.maximum_fragment_seconds == 0.5


def test_scoped_identity_rejects_ambiguous_context_and_wrong_owner() -> None:
    tokens = (
        SpokenToken(" こ", 0, 0.2, "same"),
        SpokenToken("ぉぉぉぉぉぉ", 0.2, 0.4, "same"),
        SpokenToken("こ", 2, 2.5, "same"),
        SpokenToken("ぉぉぉぉぉぉ", 2.5, 3, "same"),
    )
    fit = IndivisibleFit(tokens, layout=Measurements(), maximum_height=20)
    with pytest.raises(ValueError, match="Ambiguous"):
        fit.eligible_scoped("こぉぉぉぉぉぉ", (SpokenToken("こぉぉぉぉぉぉ", 0, 3, "same"),))
    assert (
        fit.eligible_scoped("こぉぉぉぉぉぉ", (SpokenToken("こぉぉぉぉぉぉ", 0, 0.4, "other"),))
        is False
    )
    assert fit.occurrences[1].identity.fragments[0].source_token_index == 2
    assert fit.occurrences[1].identity.fragments[0].text == "こ"


def test_punctuation_identity_uses_original_index_not_containment() -> None:
    tokens = (
        SpokenToken(" こ", 0, 0.2, "u"),
        SpokenToken("ぉぉぉぉぉぉ", 0.2, 0.4, "u"),
        SpokenToken("…", 0.4, 0.5, "u"),
    )
    fit = IndivisibleFit(tokens, layout=Measurements(), maximum_height=20)
    assert len(fit.occurrences) == 1
    assert fit.occurrences[0].unit.end == 0.4
    assert tuple(item.source_token_index for item in fit.occurrences[0].identity.fragments) == (
        0,
        1,
        2,
    )
    assert fit.occurrences[0].identity.fragments[-1].end == Fraction(1, 2)
    assert fit.occurrences[0].identity.fragments[-1].text == "…"


def test_fine_indivisible_word_uses_measured_scale_and_retains_full_text() -> None:
    tokens = tuple(
        SpokenToken(text, index * 0.4, (index + 1) * 0.4, "u")
        for index, text in enumerate("こぉぉぉぉぉぉぉぉぉぉ")
    )
    cues, hidden, widened = paged_cues(tokens, layout=Measurements(), rate=30, duration=Fraction(5))
    assert hidden == widened == 0
    assert len(cues) == 1
    cue = cues[0]
    assert isinstance(cue, ScaledCaptionCue)
    assert (cue.text, cue.start, cue.end) == ("こぉぉぉぉぉぉぉぉぉぉ", 0, 4.48)
    assert cue.scale.factor == pytest.approx(5 / 11)
    assert cue.scale.fragments == 11
    assert assemble(
        (
            {
                "captions": [
                    {
                        "text": cue.text,
                        "start": cue.start,
                        "end": cue.end,
                        **scale_fields(cue.scale),
                    }
                ]
            },
        ),
        rate=30,
    )[0]["indivisibleScale"] == {
        "width": 220.0,
        "height": 20.0,
        "available_width": 100.0,
        "available_height": 20.0,
        "fragments": 11,
        "maximum_fragment_seconds": pytest.approx(0.4),
    }


def test_coarse_word_and_splittable_phrase_do_not_gain_scale() -> None:
    coarse = (SpokenToken("こぉぉぉぉぉぉぉ", 0, 4, "u"),)
    with pytest.raises(ValueError, match="timing is too coarse"):
        paged_cues(coarse, layout=Measurements(), rate=30, duration=Fraction(5))
    fit = IndivisibleFit(
        (SpokenToken("今日は", 0, 0.3, "u"), SpokenToken("青い", 0.3, 0.6, "u")),
        layout=Measurements(),
        maximum_height=None,
    )
    assert fit.eligible("今日は青い") is False
    with pytest.raises(CaptionLayoutError):
        fit.wrap("abcdefghijklmnop")


def test_cross_window_particle_has_real_phrase_frames_before_a_greeting() -> None:
    tokens = (
        SpokenToken("飲み", 0, 0.12, "000000000000:10:cut-7"),
        SpokenToken("まし", 0.12, 0.2, "000000000000:11:cut-7"),
        SpokenToken("た", 0.2, 0.295, "000000000000:12:cut-7"),
        SpokenToken("か", 0.312, 0.332, "000000480000:1:cut-7"),
        SpokenToken("ありがとう", 0.332, 0.6, "000000480000:2:cut-7"),
    )
    layout = Measurements()
    layout.width = 240
    cues, hidden, widened = paged_cues(tokens, layout=layout, rate=30, duration=Fraction(1))
    assert cues == (
        CaptionCue("飲みましたか", 0, 0.332),
        CaptionCue("ありがとう", 0.332, 0.6799999999999999),
    )
    assert hidden == 0
    assert widened == 0
    assert tokens[-2] == SpokenToken("か", 0.312, 0.332, "000000480000:1:cut-7")


def test_true_cut_keeps_density_guard_for_a_subframe_particle() -> None:
    tokens = (
        SpokenToken("飲み", 0, 0.12, "000000000000:10:cut-7"),
        SpokenToken("まし", 0.12, 0.2, "000000000000:11:cut-7"),
        SpokenToken("た", 0.2, 0.295, "000000000000:12:cut-7"),
        SpokenToken("か", 0.312, 0.332, "000000480000:1:cut-8"),
        SpokenToken("ありがとう", 0.332, 0.6, "000000480000:2:cut-8"),
    )
    layout = Measurements()
    layout.width = 240
    with pytest.raises(ValueError, match="density exceeds one-frame display-delay budget"):
        paged_cues(tokens, layout=layout, rate=30, duration=Fraction(1))


def test_normal_fitting_unit_remains_unscaled_and_height_uses_same_factor() -> None:
    tokens = (SpokenToken("こ", 0, 0.3, "u"), SpokenToken("ぉ", 0.3, 0.6, "u"))
    fit = IndivisibleFit(tokens, layout=Measurements(), maximum_height=None)
    assert fit.wrap("こぉ") == "こぉ"
    assert fit.evidence == {}
    short = IndivisibleFit(tokens, layout=Measurements(), maximum_height=10)
    assert short.wrap("こぉ") == "こぉ"
    assert short.evidence["こぉ"].factor == 0.5


def test_scale_policy_requires_measurement() -> None:
    with pytest.raises(ValueError, match="requires measured"):
        CaptionPolicy(indivisible_text=lambda text: True)


def test_fine_unit_over_character_budget_is_preserved() -> None:
    tokens = tuple(
        SpokenToken(text, index * 0.1, (index + 1) * 0.1, "u")
        for index, text in enumerate("こ" + "ぉ" * 81)
    )
    cues, _, _ = paged_cues(tokens, layout=Measurements(), rate=30, duration=Fraction(9))
    assert len(cues) == 1
    assert isinstance(cues[0], ScaledCaptionCue)
    assert len(cues[0].text) == 82
    assert cues[0].text.startswith("こぉ")


def test_coarse_fragment_prevents_fine_support() -> None:
    fit = IndivisibleFit(
        (SpokenToken("こ", 0, 3, "u"), SpokenToken("ぉ", 3, 3.2, "u")),
        layout=Measurements(),
        maximum_height=None,
    )
    assert fit.eligible("こぉ") is False


def test_continuous_small_vowel_run_is_not_discarded_as_recognition_noise() -> None:
    tokens = (
        SpokenToken("こ", 0, 0.2, "u"),
        SpokenToken("ぉ" * 10, 0.2, 1, "u"),
        SpokenToken("ぉ", 1, 1.2, "u"),
    )
    cues, _, _ = paged_cues(tokens, layout=Measurements(), rate=30, duration=Fraction(2))
    assert len(cues) == 1
    assert cues[0].text == "こぉぉぉぉぉぉぉぉぉぉぉ"
    assert cues[0].start == 0
    assert cues[0].end == 1.28


@pytest.mark.parametrize("owner,start", [("another", 0.2), ("u", 0.6)])
def test_isolated_vowel_noise_does_not_cross_owner_or_gap(owner: str, start: float) -> None:
    tokens = (SpokenToken("こ", 0, 0.2, "u"), SpokenToken("ぉ" * 10, start, start + 0.1, owner))
    cues, _, _ = paged_cues(tokens, layout=Measurements(), rate=30, duration=Fraction(2))
    assert [cue.text for cue in cues] == ["こ"]


def test_small_vowel_alone_does_not_authorize_repetition_noise() -> None:
    from replay.caption_timing import _lexical_tokens

    tokens = (SpokenToken("ぉ", 0, 0.1, "u"), SpokenToken("ぉ" * 20, 0.1, 0.3, "u"))
    assert [token.text for token in _lexical_tokens(tokens, word_mark_gap=0.25)] == ["ぉ"]


def test_vowel_chain_finds_real_base_without_crossing_owner() -> None:
    from replay.caption_timing import _lexical_tokens

    tokens = (
        SpokenToken("こ", 0, 0.1, "a"),
        SpokenToken("ぉ", 0.1, 0.2, "b"),
        SpokenToken("ぉ" * 20, 0.2, 0.3, "b"),
    )
    assert [token.text for token in _lexical_tokens(tokens, word_mark_gap=0.25)] == ["こ", "ぉ"]
