"""Synthetic POS and measurement regressions; no native rendering or private media."""

from collections.abc import Callable

import pytest
from test_caption_linguistics import synthetic_dictionary as synthetic_dictionary

from replay import caption_joint_fit
from replay.caption_joint_fit import JointFitBudget, joint_fit
from replay.caption_linguistics import CaptionLayoutError


def test_orphan_punctuation_is_not_measured(monkeypatch: pytest.MonkeyPatch) -> None:
    text = "カタカナ?"
    monkeypatch.setattr(caption_joint_fit, "boundaries", lambda _: ((len(text) - 1,), ()))
    measured: list[str] = []

    def dimensions(value: str) -> tuple[float, float]:
        measured.append(value)
        return (989, 158) if value == text else (947, 278.12)

    fit = joint_fit(text, measure=dimensions, available_width=972, available_height=316)
    assert measured == ["カタカナ?"]
    assert fit.text == "カタカナ?"
    assert fit.factor == 972 / 989


@pytest.fixture
def measure() -> Callable[[str], tuple[float, float]]:
    def dimensions(text: str) -> tuple[float, float]:
        lines = text.split("\n")
        return float(max(map(len, lines)) * 100), float(len(lines) * 20)

    return dimensions


def test_multiline_wider_than_unscaled_region_keeps_larger_factor(
    measure: Callable[[str], tuple[float, float]],
) -> None:
    fit = joint_fit(
        "あいうえおあいうえお",
        measure=measure,
        available_width=300,
        available_height=40,
    )
    assert fit.text == "あいうえお\nあいうえお"
    assert (fit.width, fit.height, fit.factor, fit.line_count) == (500, 40, 0.6, 2)
    assert (fit.candidate_count, fit.measurement_units) == (2, 2)
    assert fit.readability_accepted is False


def test_three_rows_are_jointly_measured(
    measure: Callable[[str], tuple[float, float]],
) -> None:
    fit = joint_fit(
        "あいうえおあいうえおあいうえお",
        measure=measure,
        available_width=300,
        available_height=60,
        budget=JointFitBudget(maximum_candidates=4, maximum_measurements=4),
    )
    assert fit.text == "あいうえお\nあいうえお\nあいうえお"
    assert (fit.width, fit.height, fit.factor, fit.line_count) == (500, 60, 0.6, 3)
    assert fit.candidate_count == 4


def test_height_can_make_single_row_better(
    measure: Callable[[str], tuple[float, float]],
) -> None:
    fit = joint_fit(
        "あいうえおあいうえお",
        measure=measure,
        available_width=300,
        available_height=10,
    )
    assert fit.text == "あいうえおあいうえお"
    assert (fit.factor, fit.line_count) == (0.3, 1)


def test_measured_whole_height_is_not_inferred_from_single_line() -> None:
    def dimensions(text: str) -> tuple[float, float]:
        return {
            "あいうえおあいうえお": (1000, 20),
            "あいうえお\nあいうえお": (500, 90),
        }[text]

    fit = joint_fit(
        "あいうえおあいうえお",
        measure=dimensions,
        available_width=300,
        available_height=30,
    )
    assert fit.text == "あいうえお\nあいうえお"
    assert (fit.width, fit.height) == (500, 90)
    assert fit.factor == pytest.approx(0.3333333333333333)


def test_equal_factors_prefer_original_single_row(
    measure: Callable[[str], tuple[float, float]],
) -> None:
    fit = joint_fit(
        "あいうえおあいうえお",
        measure=measure,
        available_width=1500,
        available_height=60,
    )
    assert fit.text == "あいうえおあいうえお"
    assert (fit.factor, fit.line_count) == (1, 1)


def test_equal_scaled_factors_preserve_narrower_group_layout(
    measure: Callable[[str], tuple[float, float]],
) -> None:
    fit = joint_fit(
        "あいうえおあいうえお",
        measure=measure,
        available_width=500,
        available_height=20,
    )
    assert fit.text == "あいうえお\nあいうえお"
    assert (fit.factor, fit.line_count) == (0.5, 2)


def test_particles_are_not_detached(
    measure: Callable[[str], tuple[float, float]],
) -> None:
    fit = joint_fit(
        "今日は青い空を",
        measure=measure,
        available_width=300,
        available_height=40,
    )
    assert fit.text == "今日は\n青い空を"
    assert fit.factor == 0.75


def test_literal_rhythmic_units_reuse_existing_protection(
    measure: Callable[[str], tuple[float, float]],
) -> None:
    fit = joint_fit(
        "ハトハトハト",
        measure=measure,
        available_width=200,
        available_height=60,
    )
    assert fit.text == "ハト\nハト\nハト"
    assert fit.factor == 1


def test_unknown_name_and_honorific_remain_intact(
    measure: Callable[[str], tuple[float, float]],
) -> None:
    fit = joint_fit(
        "クラウドペガサスちゃん",
        measure=measure,
        available_width=100,
        available_height=60,
    )
    assert fit.text == "クラウドペガサスちゃん"
    assert fit.candidate_count == 1
    assert fit.readability_accepted is False


def test_known_loanwords_split_only_at_their_real_boundary(
    measure: Callable[[str], tuple[float, float]],
) -> None:
    fit = joint_fit(
        "バーチャルデスクトップ",
        measure=measure,
        available_width=500,
        available_height=40,
    )
    assert fit.text == "バーチャル\nデスクトップ"
    assert fit.factor == pytest.approx(0.8333333333333334)


def test_caller_verified_atomic_unit_prevents_incidental_pos_splits(
    measure: Callable[[str], tuple[float, float]],
) -> None:
    fit = joint_fit(
        "あいうえおあいうえお",
        measure=measure,
        available_width=300,
        available_height=40,
        indivisible=True,
        budget=JointFitBudget(maximum_candidates=1, maximum_measurements=1),
    )
    assert fit.text == "あいうえおあいうえお"
    assert (fit.factor, fit.candidate_count, fit.measurement_units) == (0.3, 1, 1)


def test_unknown_alphanumeric_word_is_not_split_by_count(
    measure: Callable[[str], tuple[float, float]],
) -> None:
    fit = joint_fit(
        "abcdefghijkxyz",
        measure=measure,
        available_width=300,
        available_height=60,
    )
    assert fit.text == "abcdefghijkxyz"
    assert fit.candidate_count == 1


def test_atomic_eighty_two_units_are_retained_without_readability_claim(
    measure: Callable[[str], tuple[float, float]],
) -> None:
    fit = joint_fit(
        "こぉぉぉぉぉぉぉぉぉぉぉぉぉぉぉぉぉぉぉぉぉぉぉぉぉぉぉぉぉぉぉぉぉぉぉぉぉぉぉぉぉぉぉぉぉぉぉぉぉぉぉぉぉぉぉぉぉぉぉぉぉぉぉぉぉぉぉぉぉぉぉぉぉぉぉぉぉぉぉぉぉ",
        measure=measure,
        available_width=300,
        available_height=60,
        indivisible=True,
    )
    assert fit.text == (
        "こぉぉぉぉぉぉぉぉぉぉぉぉぉぉぉぉぉぉぉぉぉぉぉぉぉぉぉぉぉぉぉぉぉぉ"
        "ぉぉぉぉぉぉぉぉぉぉぉぉぉぉぉぉぉぉぉぉぉぉぉぉぉぉぉぉぉぉぉぉぉぉぉ"
        "ぉぉぉぉぉぉぉぉぉぉぉぉ"
    )
    assert (fit.width, fit.height, fit.candidate_count) == (8200, 20, 1)
    assert fit.factor == pytest.approx(0.036585365853658534)
    assert fit.readability_accepted is False


def test_atomic_one_hundred_fifty_one_units_are_retained(
    measure: Callable[[str], tuple[float, float]],
) -> None:
    fit = joint_fit(
        "うわぁぁぁぁぁぁぁぁぁぁぁぁぁぁぁぁぁぁぁぁぁぁぁぁぁぁぁぁぁぁぁぁぁぁぁぁぁぁぁぁぁぁぁぁぁぁぁぁぁぁぁぁぁぁぁぁぁぁぁぁぁぁぁぁぁぁぁぁぁぁぁぁぁぁぁぁぁぁぁぁぁぁぁぁぁぁぁぁぁぁぁぁぁぁぁぁぁぁぁぁぁぁぁぁぁぁぁぁぁぁぁぁぁぁぁぁぁぁぁぁぁぁぁぁぁぁぁぁぁぁぁぁぁぁぁぁぁぁぁぁぁぁぁぁぁぁぁぁぁ",
        measure=measure,
        available_width=300,
        available_height=60,
        indivisible=True,
    )
    assert fit.text == (
        "うわぁぁぁぁぁぁぁぁぁぁぁぁぁぁぁぁぁぁぁぁぁぁぁぁぁぁぁぁぁぁぁぁぁ"
        "ぁぁぁぁぁぁぁぁぁぁぁぁぁぁぁぁぁぁぁぁぁぁぁぁぁぁぁぁぁぁぁぁぁぁぁ"
        "ぁぁぁぁぁぁぁぁぁぁぁぁぁぁぁぁぁぁぁぁぁぁぁぁぁぁぁぁぁぁぁぁぁぁぁ"
        "ぁぁぁぁぁぁぁぁぁぁぁぁぁぁぁぁぁぁぁぁぁぁぁぁぁぁぁぁぁぁぁぁぁぁぁ"
        "ぁぁぁぁぁぁぁぁぁぁぁ"
    )
    assert (fit.width, fit.height, fit.candidate_count) == (15100, 20, 1)
    assert fit.readability_accepted is False


def test_single_line_limit_does_not_attempt_extra_candidates(
    measure: Callable[[str], tuple[float, float]],
) -> None:
    fit = joint_fit(
        "あいうえおあいうえお",
        measure=measure,
        available_width=300,
        available_height=60,
        maximum_lines=1,
        budget=JointFitBudget(maximum_candidates=1, maximum_measurements=1),
    )
    assert fit.text == "あいうえおあいうえお"
    assert fit.candidate_count == 1


def test_prepare_lines_is_unique_bounded_and_measures_complete_candidates() -> None:
    preparations: list[tuple[str, ...]] = []
    calls: list[str] = []

    def dimensions(text: str) -> tuple[float, float]:
        calls.append(text)
        return (1000, 20) if "\n" not in text else (500, 40)

    fit = joint_fit(
        "あいうえおあいうえお",
        measure=dimensions,
        prepare_lines=preparations.append,
        available_width=300,
        available_height=40,
        budget=JointFitBudget(maximum_candidates=2, maximum_measurements=4),
    )
    assert preparations == [("あいうえおあいうえお", "あいうえお")]
    assert calls == ["あいうえおあいうえお", "あいうえお\nあいうえお"]
    assert fit.measurement_units == 4


def test_candidate_overflow_refuses_before_any_renderer_call() -> None:
    calls: list[str] = []
    preparations: list[tuple[str, ...]] = []

    def dimensions(text: str) -> tuple[float, float]:
        calls.append(text)
        return 100, 20

    with pytest.raises(CaptionLayoutError, match="candidate budget"):
        joint_fit(
            "あいうえおあいうえおあいうえお",
            measure=dimensions,
            prepare_lines=preparations.append,
            available_width=300,
            available_height=60,
            budget=JointFitBudget(maximum_candidates=3),
        )
    assert calls == []
    assert preparations == []


def test_measurement_overflow_includes_preparation_and_has_no_side_effect() -> None:
    calls: list[str] = []
    preparations: list[tuple[str, ...]] = []

    def dimensions(text: str) -> tuple[float, float]:
        calls.append(text)
        return 100, 20

    with pytest.raises(CaptionLayoutError, match="measurement budget"):
        joint_fit(
            "あいうえおあいうえお",
            measure=dimensions,
            prepare_lines=preparations.append,
            available_width=300,
            available_height=60,
            budget=JointFitBudget(maximum_measurements=3),
        )
    assert calls == []
    assert preparations == []


def test_blank_candidate_row_is_excluded_without_trimming_source(
    measure: Callable[[str], tuple[float, float]], monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(caption_joint_fit, "boundaries", lambda text: ((1, 2, 3), (1, 2, 3)))
    fit = joint_fit("あ い", measure=measure, available_width=100, available_height=60)
    assert fit.text == "あ\n い"
    assert fit.candidate_count == 3


@pytest.mark.parametrize("text", ["", "   ", "a\nb", "a\rb", "a\0b", "a" * 8193])
def test_invalid_source_is_rejected_before_measurement(
    text: str, measure: Callable[[str], tuple[float, float]]
) -> None:
    with pytest.raises(ValueError, match="fit input"):
        joint_fit(text, measure=measure, available_width=300, available_height=60)


@pytest.mark.parametrize(
    "width,height",
    [
        (0, 60),
        (-1, 60),
        (float("nan"), 60),
        (float("inf"), 60),
        (True, 60),
        (300, 0),
        (300, -1),
        (300, float("nan")),
        (300, float("inf")),
        (300, True),
    ],
)
def test_invalid_available_dimensions(
    width: float, height: float, measure: Callable[[str], tuple[float, float]]
) -> None:
    with pytest.raises(ValueError, match="fit input"):
        joint_fit("今日は", measure=measure, available_width=width, available_height=height)


@pytest.mark.parametrize("maximum_lines", [0, 4, True])
def test_invalid_line_limits(
    maximum_lines: int, measure: Callable[[str], tuple[float, float]]
) -> None:
    with pytest.raises(ValueError, match="fit input"):
        joint_fit(
            "今日は",
            measure=measure,
            available_width=300,
            available_height=60,
            maximum_lines=maximum_lines,
        )


@pytest.mark.parametrize(
    "candidates,measurements",
    [(0, 1), (65537, 1), (True, 1), (1, 0), (1, 262145), (1, True)],
)
def test_invalid_budgets(candidates: int, measurements: int) -> None:
    with pytest.raises(ValueError, match="fit budget"):
        JointFitBudget(candidates, measurements)


@pytest.mark.parametrize(
    "dimensions",
    [
        (0, 20),
        (-1, 20),
        (float("nan"), 20),
        (float("inf"), 20),
        (True, 20),
        (100, 0),
        (100, -1),
        (100, float("nan")),
        (100, float("inf")),
        (100, True),
    ],
)
def test_invalid_rendered_dimensions(dimensions: tuple[float, float]) -> None:
    with pytest.raises(ValueError, match="rendered joint"):
        joint_fit(
            "今日は",
            measure=lambda text: dimensions,
            available_width=300,
            available_height=60,
        )


def test_positive_dimensions_cannot_hide_scale_underflow() -> None:
    with pytest.raises(ValueError, match="underflow"):
        joint_fit(
            "今日は",
            measure=lambda text: (1e308, 20),
            available_width=1e-320,
            available_height=60,
        )


def test_measurement_error_is_preserved() -> None:
    def dimensions(text: str) -> tuple[float, float]:
        raise ValueError("native measurement unavailable")

    with pytest.raises(ValueError, match="native measurement unavailable"):
        joint_fit("今日は", measure=dimensions, available_width=300, available_height=60)
