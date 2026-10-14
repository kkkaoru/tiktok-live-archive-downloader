"""Measured font geometry and phrase boundaries drive wrapping and native-clock resets."""

from functools import partial

import pytest

from replay import caption_linguistics, caption_timing
from replay.caption_linguistics import CaptionLayoutError, wrap_measured_lines
from replay.caption_timing import CaptionPolicy, SpokenToken, make_captions


def measure(text: str) -> tuple[float, float]:
    """Synthetic proportional renderer: two wide glyphs and one narrow glyph."""
    widths = {"あ": 9, "い": 3, "う": 9}
    return max(sum(widths[c] for c in line) for line in text.split("\n")), 10 * len(
        text.split("\n")
    )


@pytest.fixture(autouse=True)
def morphology(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        caption_linguistics,
        "boundaries",
        lambda text: (
            tuple(range(1, len(text) + 1)),
            tuple(range(1, len(text) + 1)),
        ),
    )


def test_narrow_text_fits_without_count_based_break() -> None:
    assert wrap_measured_lines("いいい", measure=measure, width=10, height=20) == "いいい"


def test_wide_text_wraps_at_measured_boundary() -> None:
    assert wrap_measured_lines("あい", measure=measure, width=10, height=20) == "あ\nい"


def test_two_line_height_must_fit() -> None:
    with pytest.raises(CaptionLayoutError):
        wrap_measured_lines("あい", measure=measure, width=10, height=19)


def test_no_character_split_inside_a_phrase(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(caption_linguistics, "boundaries", lambda text: ((2,), (1, 2)))
    with pytest.raises(CaptionLayoutError):
        wrap_measured_lines("あい", measure=measure, width=10, height=20)


@pytest.mark.parametrize("width,height", [(0, 20), (10, 0), (float("nan"), 20), (10, float("inf"))])
def test_invalid_region(width: float, height: float) -> None:
    with pytest.raises(ValueError, match="region"):
        wrap_measured_lines("あ", measure=measure, width=width, height=height)


@pytest.mark.parametrize("bounds", [(float("nan"), 10), (-1, 10), (1, 0), (1, float("inf"))])
def test_invalid_renderer_result(bounds: tuple[float, float]) -> None:
    with pytest.raises(ValueError, match="measurement"):
        wrap_measured_lines("あ", measure=lambda text: bounds, width=10, height=20)


def test_measured_switch_preserves_observed_token_times(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        caption_linguistics, "attached_tokens", lambda words, maximum_gap: [False] * len(words)
    )
    layout = partial(wrap_measured_lines, measure=measure, width=10, height=20)
    cues = make_captions(
        (
            SpokenToken("あ", 0, 0.9, "speech"),
            SpokenToken("い", 0.95, 1.85, "speech"),
            SpokenToken("う", 1.9, 2.8, "speech"),
        ),
        policy=CaptionPolicy(linguistic_boundaries=True, measured_layout=layout),
    )
    assert tuple(cue.text for cue in cues) == ("あ", "あ\nい", "う")
    assert tuple(cue.start for cue in cues) == (0, 0.95, 1.9)
    assert tuple(round(cue.end, 2) for cue in cues) == (0.95, 1.9, 2.88)


def test_oversized_native_token_is_not_shrunk_or_retimed() -> None:
    layout = partial(wrap_measured_lines, measure=measure, width=8, height=20)
    with pytest.raises(ValueError, match="finer alignment"):
        make_captions(
            (SpokenToken("あ", 0, 0.5, "speech"),),
            policy=CaptionPolicy(linguistic_boundaries=True, measured_layout=layout),
        )


def test_measurement_cannot_rewrite_transcript() -> None:
    with pytest.raises(ValueError, match="changed text"):
        make_captions(
            (SpokenToken("あ", 0, 0.5, "speech"),),
            policy=CaptionPolicy(linguistic_boundaries=True, measured_layout=lambda text: "う"),
        )


def test_measured_layout_requires_pos() -> None:
    with pytest.raises(ValueError, match="linguistic"):
        CaptionPolicy(measured_layout=lambda text: text)


def test_missing_layout_is_explicit() -> None:
    with pytest.raises(ValueError, match="Missing measured"):
        caption_timing._measured_layout("あ", CaptionPolicy())
