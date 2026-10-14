"""Unaligned lexical boundaries produce simultaneous text, never new clocks."""

from collections.abc import Iterable
from fractions import Fraction

import pytest
from test_caption_indivisible_fit import Measurements
from test_caption_linguistics import synthetic_dictionary as synthetic_dictionary

from replay.caption_anchor_fit import anchored_groups, fit_group
from replay.caption_linguistics import CaptionLayoutError
from replay.caption_native_paging import paged_cues
from replay.caption_scale import read_scale, scale_fields
from replay.caption_timing import ScaledCaptionCue, SpokenToken


def test_crossing_asr_anchors_fit_simultaneously_without_intermediate_words() -> None:
    tokens = (
        SpokenToken("あ", 1, 1.1, "u"),
        SpokenToken("いうえおあ", 1.1, 2.1, "u"),
        SpokenToken("いうえお", 2.1, 2.5, "u"),
    )
    cues, _, _ = paged_cues(
        tokens, layout=Measurements(), rate=30, duration=Fraction(3), maximum_height=20
    )
    assert len(cues) == 1
    cue = cues[0]
    assert isinstance(cue, ScaledCaptionCue)
    assert (cue.text, cue.start, cue.end) == ("あいうえお\nあいうえお", 1, 2.58)
    assert cue.scale.factor == 0.5
    assert cue.scale.provenance == "recognition-anchor-group"
    assert read_scale(scale_fields(cue.scale)) == cue.scale


@pytest.mark.parametrize(
    "tokens",
    [
        [SpokenToken("あいうえお", 0, 0.2, "u"), SpokenToken("あいうえお", 0.2, 0.4, "u")],
        [SpokenToken("あ", 0, 3, "u"), SpokenToken("いうえおあいうえお", 3, 3.1, "u")],
        [SpokenToken("あ", 0, 0.2, "a"), SpokenToken("いうえおあいうえお", 0.2, 0.4, "b")],
        [SpokenToken("あ", 0, 0.2, "u"), SpokenToken("いうえおあいうえお", 1, 1.2, "u")],
    ],
)
def test_safe_boundaries_coarse_input_and_discontinuity_are_not_group_scaled(
    tokens: list[SpokenToken],
) -> None:
    assert anchored_groups(tokens) == {}


def test_normal_height_keeps_unscaled_simultaneous_group() -> None:
    tokens = (
        SpokenToken("あ", 0, 0.2, "u"),
        SpokenToken("いうえおあ", 0.2, 0.4, "u"),
        SpokenToken("いうえお", 0.4, 0.6, "u"),
    )
    cues, _, _ = paged_cues(
        tokens, layout=Measurements(), rate=30, duration=Fraction(1), maximum_height=40
    )
    assert [cue.text for cue in cues] == ["あいうえお\nあいうえお"]
    assert not isinstance(cues[0], ScaledCaptionCue)


def test_coarse_token_still_rejected_before_layout() -> None:
    with pytest.raises(ValueError, match="timing is too coarse"):
        paged_cues(
            (SpokenToken("あいうえおあいうえお", 0, 3, "u"),),
            layout=Measurements(),
            rate=30,
            duration=Fraction(4),
            maximum_height=20,
        )


def test_long_word_has_no_anchor_group_provenance() -> None:
    assert anchored_groups([SpokenToken("こ", 0, 0.2, "u"), SpokenToken("ぉ", 0.2, 0.4, "u")]) == {}


def test_group_can_use_single_line_when_multiline_cannot_fit_width() -> None:
    text, scale = fit_group(
        "あいうえおあいうえお", layout=Measurements(), maximum_height=10, support=(3, 0.4)
    )
    assert text == "あいうえおあいうえお"
    assert scale is not None
    assert scale.factor == 0.5


def test_joint_overwidth_rows_preserve_original_recognition_clock() -> None:
    tokens = (
        SpokenToken("あ", 1, 1.1, "u"),
        SpokenToken("いうえおあ", 1.1, 2.1, "u"),
        SpokenToken("いうえお", 2.1, 2.5, "u"),
    )
    layout = Measurements()
    layout.width = 50
    cues, hidden, widened = paged_cues(
        tokens, layout=layout, rate=30, duration=Fraction(3), maximum_height=40
    )
    assert len(cues) == 1
    cue = cues[0]
    assert isinstance(cue, ScaledCaptionCue)
    assert (cue.text, cue.start, cue.end) == ("あいうえお\nあいうえお", 1, 2.58)
    assert (cue.scale.width, cue.scale.height, cue.scale.factor) == (100, 40, 0.5)
    assert (cue.scale.fragments, cue.scale.maximum_fragment_seconds) == (3, 1)
    assert cue.scale.provenance == "recognition-anchor-group"
    assert hidden == widened == 0
    assert tokens == (
        SpokenToken("あ", 1, 1.1, "u"),
        SpokenToken("いうえおあ", 1.1, 2.1, "u"),
        SpokenToken("いうえお", 2.1, 2.5, "u"),
    )


def test_default_height_jointly_fits_three_measured_rows() -> None:
    layout = Measurements()
    layout.width = 50
    text, scale = fit_group(
        "あいうえおあいうえおあいうえお",
        layout=layout,
        maximum_height=None,
        support=(4, 0.4),
    )
    assert text == "あいうえお\nあいうえお\nあいうえお"
    assert scale is not None
    assert (scale.width, scale.height, scale.available_height, scale.factor) == (100, 60, 60, 0.5)
    assert (scale.fragments, scale.maximum_fragment_seconds) == (4, 0.4)


def test_group_uses_rendered_multiline_height_not_inferred_line_count() -> None:
    class WholeHeight(Measurements):
        def measure(self, text: str) -> tuple[float, float]:
            return (200, 20) if "\n" not in text else (100, 90)

    text, scale = fit_group(
        "あいうえおあいうえお",
        layout=WholeHeight(),
        maximum_height=30,
        support=(3, 0.4),
    )
    assert text == "あいうえおあいうえお"
    assert scale is not None
    assert (scale.width, scale.height, scale.factor) == (200, 20, 0.5)


def test_legitimate_joint_factor_one_has_no_scale_evidence() -> None:
    text, scale = fit_group(
        "あいうえおあいうえお", layout=Measurements(), maximum_height=40, support=(3, 0.4)
    )
    assert text == "あいうえお\nあいうえお"
    assert scale is None


def test_group_candidate_limit_does_not_accept_partial_search() -> None:
    class ObservedMeasurements(Measurements):
        def __init__(self) -> None:
            self.measures = 0
            self.preparations = 0

        def prepare_lines(self, texts: Iterable[str]) -> None:
            self.preparations += 1

        def measure(self, text: str) -> tuple[float, float]:
            self.measures += 1
            return 200, 20

    layout = ObservedMeasurements()
    with pytest.raises(CaptionLayoutError, match="candidate budget"):
        fit_group(
            "あいうえお" * 50,
            layout=layout,
            maximum_height=40,
            support=(3, 0.4),
        )
    assert layout.measures == 0
    assert layout.preparations == 0


def test_default_height_baseline_is_the_only_measure_before_candidate_refusal() -> None:
    class ObservedMeasurements(Measurements):
        def __init__(self) -> None:
            self.measures = 0
            self.preparations = 0

        def prepare_lines(self, texts: Iterable[str]) -> None:
            self.preparations += 1

        def measure(self, text: str) -> tuple[float, float]:
            self.measures += 1
            return 200, 20

    layout = ObservedMeasurements()
    with pytest.raises(CaptionLayoutError, match="candidate budget"):
        fit_group(
            "あいうえお" * 50,
            layout=layout,
            maximum_height=None,
            support=(3, 0.4),
        )
    assert layout.measures == 1
    assert layout.preparations == 0


def test_production_indivisible_path_retains_long_atomic_text_and_clock() -> None:
    tokens = (
        SpokenToken("こ", 1, 1.1, "u"),
        SpokenToken("ぉ" * 150, 1.1, 2.1, "u"),
    )
    cues, hidden, widened = paged_cues(
        tokens, layout=Measurements(), rate=30, duration=Fraction(3), maximum_height=20
    )
    assert len(cues) == 1
    cue = cues[0]
    assert isinstance(cue, ScaledCaptionCue)
    assert cue.text == (
        "こぉぉぉぉぉぉぉぉぉぉぉぉぉぉぉぉぉぉぉぉぉぉぉぉぉぉぉぉぉぉぉぉぉぉ"
        "ぉぉぉぉぉぉぉぉぉぉぉぉぉぉぉぉぉぉぉぉぉぉぉぉぉぉぉぉぉぉぉぉぉぉぉ"
        "ぉぉぉぉぉぉぉぉぉぉぉぉぉぉぉぉぉぉぉぉぉぉぉぉぉぉぉぉぉぉぉぉぉぉぉ"
        "ぉぉぉぉぉぉぉぉぉぉぉぉぉぉぉぉぉぉぉぉぉぉぉぉぉぉぉぉぉぉぉぉぉぉぉ"
        "ぉぉぉぉぉぉぉぉぉぉぉ"
    )
    assert (cue.start, cue.end) == (1, 2.18)
    assert (cue.scale.width, cue.scale.height) == (3020, 20)
    assert cue.scale.provenance == "indivisible-word"
    assert hidden == widened == 0
