"""Safe wrap keeps short phrases intact and wraps long ones linguistically."""

import pytest

from replay.caption_linguistics import CaptionLayoutError
from replay.caption_safe_wrap import measure_em, wrap_phrase, wrap_report


def test_short_phrase_unchanged() -> None:
    assert wrap_phrase("どうもー!") == "どうもー!"


def test_long_phrase_wraps_within_three_lines() -> None:
    text = "なんとフィアージュのウィオンリ君だけ"
    wrapped = wrap_phrase(text)
    assert 2 <= wrapped.count("\n") + 1 <= 3
    for line in wrapped.split("\n"):
        width, _ = measure_em(line, 104.0)
        assert width <= 1161.0
    assert wrapped.replace("\n", "") == text


def test_mixed_alphanumeric_overestimates_safely() -> None:
    width, height = measure_em("VR123", 104.0)
    assert width == pytest.approx((0.6 * 2 + 0.6 * 3) * 104.0)
    assert height == pytest.approx(104.0 * 1.25)
    assert wrap_phrase("VRチャッターのカシスミルクです").count("\n") + 1 <= 3


def test_full_width_punctuation_counts_one_em() -> None:
    width, _ = measure_em("、「」", 104.0)
    assert width == pytest.approx(3 * 104.0)


def test_char_fallback_splits_runs_and_long_tokens() -> None:
    report = wrap_report("は" * 16)
    assert report["fallback"] is True
    assert report["reason"] == "character-boundary"
    assert str(report["text"]).replace("\n", "") == "は" * 16
    assert len(str(report["text"]).split("\n")) <= 3
    long_token = "あ" * 30
    linguistic = wrap_report(long_token)
    assert linguistic["fallback"] is False
    assert str(linguistic["text"]).replace("\n", "") == long_token
    assert len(str(linguistic["text"]).split("\n")) <= 3
    lines = linguistic["lines"]
    assert isinstance(lines, list)
    assert all(float(entry["width"]) <= 1161.0 for entry in lines)


def test_linguistic_path_reports_no_fallback() -> None:
    report = wrap_report("どうもー!")
    assert report["fallback"] is False
    assert report["text"] == "どうもー!"


def test_overwide_multiline_flat_then_linguistic_wrap() -> None:
    # An over-wide placement that already carries a line break is flattened
    # and re-laid out linguistically; no roll-up needed here.
    report = wrap_report("性を避けて生活するって無理じゃない?\nって思うけどね")
    assert report["fallback"] is False
    assert report["dropped"] == ""
    lines = str(report["text"]).split("\n")
    assert len(lines) <= 3
    flat = str(report["text"]).replace("\n", "")
    assert flat == "性を避けて生活するって無理じゃない?って思うけどね"


def test_overlong_single_token_rejected_without_rollup_fit() -> None:
    # Even after rolling out every drop-able prefix, 3 lines at 10px cannot
    # hold the tail: the loop bottoms out at one character.
    with pytest.raises(CaptionLayoutError, match="[Rr]oll-up"):
        wrap_phrase("あ" * 40, max_width_px=10.0, max_lines=3)


def test_invalid_inputs_rejected() -> None:
    with pytest.raises(ValueError, match="[Ww]idth"):
        wrap_phrase("あ", max_width_px=0.0)
    with pytest.raises(ValueError, match="[Ff]ont"):
        measure_em("あ", 0.0)
    with pytest.raises(CaptionLayoutError, match="[Rr]oll-up"):
        wrap_phrase("あ" * 200, max_width_px=10.0, max_lines=3)
