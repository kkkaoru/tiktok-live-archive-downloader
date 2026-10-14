import json
import sys
from fractions import Fraction
from pathlib import Path

import pytest
from test_caption_native_layout import layout as layout
from test_caption_native_paging import fixture_index
from test_caption_native_reflow import exact as exact

from replay import caption_measure_prefetch as prefetch
from replay.caption_measure_prefetch import PrefetchBudget, plan_prefetch, prediction_coverage
from replay.caption_native_layout import NativeCaptionLayout
from replay.caption_native_paging import paged_cues
from replay.caption_timing import CaptionPolicy, SpokenToken


def test_prefix_and_reserved_tail() -> None:
    tokens = (
        SpokenToken("猫", 0, 0.1, "a"),
        SpokenToken("が", 0.1, 0.2, "a"),
        SpokenToken("来た", 0.2, 0.3, "a"),
    )
    result = plan_prefetch(tokens, policy=CaptionPolicy(linguistic_boundaries=True))
    assert "猫が" in result.texts
    assert "猫が来た" in result.texts
    assert result.exhausted is False
    coverage = prediction_coverage(result, ("猫が", "猫が", "未知"))
    assert coverage.requested == 3
    assert coverage.predicted == 2
    assert coverage.missing == ("未知",)
    assert coverage.unused > 0


@pytest.mark.parametrize(
    "budget", [PrefetchBudget(visits=1), PrefetchBudget(strings=1), PrefetchBudget(utf16_units=1)]
)
def test_budget_falls_back_without_claiming_completeness(budget: PrefetchBudget) -> None:
    result = plan_prefetch(
        (SpokenToken("ab", 0, 0.1, "u"), SpokenToken("cd", 0.1, 0.2, "u")),
        policy=CaptionPolicy(),
        budget=budget,
    )
    assert result.exhausted
    assert result.visits <= budget.visits
    assert result.utf16_units <= budget.utf16_units
    assert len(result.texts) <= budget.strings


def test_duplicate_discovery_is_bounded() -> None:
    tokens = tuple(SpokenToken("a", index / 10, (index + 1) / 10, "u") for index in range(4096))
    result = plan_prefetch(tokens, policy=CaptionPolicy(), budget=PrefetchBudget(visits=10))
    assert result.visits == 10
    assert result.texts == ("a",)
    assert result.exhausted


def test_eligibility_utf16_and_utterance_boundaries() -> None:
    tokens = (
        SpokenToken("😀", 0, 0.1, "a"),
        SpokenToken("b", 0.1, 0.2, "b"),
        SpokenToken("x\n", 0.2, 0.3, "c"),
        SpokenToken("😀" * 61, 0.3, 0.4, "d"),
    )
    result = plan_prefetch(tokens, policy=CaptionPolicy(maximum_characters=80))
    assert result.texts == ("😀", "b")
    assert result.utf16_units == 3


def test_lexical_normalization_and_character_cap() -> None:
    tokens = (
        SpokenToken("cat", 0, 0.1, "u"),
        SpokenToken("。", 0.1, 0.2, "u"),
        SpokenToken("dog", 0.2, 0.3, "u"),
    )
    result = plan_prefetch(tokens, policy=CaptionPolicy(maximum_characters=4))
    assert result.texts == ("cat。", "dog")
    assert plan_prefetch((), policy=CaptionPolicy()).texts == ()


@pytest.mark.parametrize("value", [0, -1, 65537, True])
def test_invalid_budget(value: int) -> None:
    with pytest.raises(ValueError, match="budget"):
        PrefetchBudget(visits=value)


def test_input_limits() -> None:
    with pytest.raises(ValueError, match="input"):
        plan_prefetch((SpokenToken("a", 0, 1, "u"),) * 4097, policy=CaptionPolicy())
    with pytest.raises(ValueError, match="input"):
        plan_prefetch((SpokenToken("a" * 100, 0, 1, "u"),) * 4096, policy=CaptionPolicy())
    with pytest.raises(ValueError, match="observed"):
        prediction_coverage(plan_prefetch((), policy=CaptionPolicy()), ("a",) * 65537)


@pytest.mark.parametrize("width", [24, 48, 120])
def test_warming_preserves_exact_cues(exact: NativeCaptionLayout, width: int) -> None:
    exact.width = width
    tokens = tuple(
        SpokenToken(text, i * 0.2, (i + 1) * 0.2, "u") for i, text in enumerate("abcdefghij")
    )
    plan = plan_prefetch(
        tokens,
        policy=CaptionPolicy(linguistic_boundaries=True, maximum_characters=80, frame_rate=30),
    )
    before = paged_cues(tokens, layout=exact, rate=30, duration=Fraction(3))
    original_requests = exact._batch
    exact._cache.clear()
    exact.prepare_lines(plan.texts)
    after = paged_cues(tokens, layout=exact, rate=30, duration=Fraction(3))
    assert after == before
    assert exact._batch - original_requests < original_requests


def test_saved_evaluation_cli(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    fixture_index(tmp_path)
    request = tmp_path / "batch-000000.request.json"
    request.write_text(json.dumps({"texts": ["ab", "missing"]}), encoding="utf-8")
    output = tmp_path / "audit.json"
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "prefetch",
            str(tmp_path),
            "--segment",
            "0",
            "--requests",
            str(tmp_path),
            "--output",
            str(output),
        ],
    )
    prefetch.main()
    result = json.loads(output.read_text(encoding="utf-8"))
    assert result["coverage"] == {
        "requested": 2,
        "predicted": 1,
        "unused": 0,
        "missing": ["missing"],
    }
    assert result["nativeExecution"] is False
    assert result["productionCacheModified"] is False
    with pytest.raises(FileExistsError):
        prefetch.main()


@pytest.mark.parametrize("segment", [-1, 512, True])
def test_bad_segment(tmp_path: Path, segment: int) -> None:
    with pytest.raises(ValueError, match="segment"):
        prefetch.evaluate_saved(tmp_path, segment=segment, requests=tmp_path, output=tmp_path / "o")


@pytest.mark.parametrize("rows", [None, [], [{"segment": 1}]])
def test_bad_index(tmp_path: Path, rows: object) -> None:
    (tmp_path / "segments.json").write_text(json.dumps({"segments": rows}), encoding="utf-8")
    with pytest.raises(ValueError):
        prefetch.evaluate_saved(tmp_path, segment=0, requests=tmp_path, output=tmp_path / "o")


@pytest.mark.parametrize("texts", [None, [1]])
def test_bad_saved_texts(tmp_path: Path, texts: object) -> None:
    fixture_index(tmp_path)
    (tmp_path / "batch-000000.request.json").write_text(
        json.dumps({"texts": texts}), encoding="utf-8"
    )
    with pytest.raises(ValueError, match="strings"):
        prefetch.evaluate_saved(tmp_path, segment=0, requests=tmp_path, output=tmp_path / "o")


def test_empty_requests_and_document_budget(tmp_path: Path) -> None:
    fixture_index(tmp_path)
    with pytest.raises(ValueError, match="bounded"):
        prefetch.evaluate_saved(tmp_path, segment=0, requests=tmp_path, output=tmp_path / "o")
    (tmp_path / "segments.json").write_text("", encoding="utf-8")
    with pytest.raises(ValueError, match="document"):
        prefetch.evaluate_saved(tmp_path, segment=0, requests=tmp_path, output=tmp_path / "o")


def test_request_count_budget(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    fixture_index(tmp_path)
    (tmp_path / "batch-000000.request.json").write_text('{"texts": ["a", "b"]}', encoding="utf-8")
    monkeypatch.setattr(prefetch, "MAX_LOG_REQUESTS", 1)
    with pytest.raises(ValueError, match="Too many"):
        prefetch.evaluate_saved(tmp_path, segment=0, requests=tmp_path, output=tmp_path / "o")


def test_batch_forecast_limits_and_repeated_requests() -> None:
    plan = prefetch.PrefetchPlan(tuple(str(index) for index in range(17)), 17, 24, False)
    assert prefetch.forecast_batches(
        plan, (("0",), ("missing",), ("missing",))
    ) == prefetch.BatchForecast(3, 2, 1, 3)
    large = prefetch.PrefetchPlan(("x" * 120, "y" * 120), 2, 240, False)
    assert prefetch.forecast_batches(large, ()) == prefetch.BatchForecast(0, 2, 0, 2)
    assert prefetch.forecast_batches(
        prefetch.PrefetchPlan((), 0, 0, False), ()
    ) == prefetch.BatchForecast(0, 0, 0, 0)
