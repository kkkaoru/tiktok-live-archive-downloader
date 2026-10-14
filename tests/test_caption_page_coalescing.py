from dataclasses import replace
from fractions import Fraction

import pytest
from test_caption_linguistics import synthetic_dictionary as synthetic_dictionary

from replay.caption_linguistics import boundaries
from replay.caption_page_coalescing import (
    MeasuredLayout,
    MeasuredStage,
    PagePolicy,
    RecognizedToken,
    coalesce,
)
from replay.core import ReplayError


@pytest.fixture
def stages() -> tuple[MeasuredStage, ...]:
    first = RecognizedToken("t1", "私は", Fraction(0), Fraction(1))
    second = RecognizedToken("t2", "猫が", Fraction(1), Fraction(2))
    third = RecognizedToken("t3", "好き", Fraction(2), Fraction(3))
    return (
        MeasuredStage(
            "s1",
            "私は",
            Fraction(0),
            Fraction(1),
            "speech",
            "cut",
            (first,),
            MeasuredLayout("m1", "style", 100, (40,)),
            False,
        ),
        MeasuredStage(
            "s2",
            "私は\n猫が",
            Fraction(1),
            Fraction(2),
            "speech",
            "cut",
            (first, second),
            MeasuredLayout("m2", "style", 100, (40, 40)),
            False,
        ),
        MeasuredStage(
            "s3",
            "私は\n猫が\n好き",
            Fraction(2),
            Fraction(3),
            "speech",
            "cut",
            (first, second, third),
            MeasuredLayout("m3", "style", 100, (40, 40, 40)),
            True,
        ),
    )


def test_final_measured_page_records_advanced_display_not_changed_asr(
    stages: tuple[MeasuredStage, ...],
) -> None:
    result = coalesce(stages, policy=PagePolicy(Fraction(4), "synthetic-pos"))
    assert result.original_stage_count == 3
    assert len(result.pages) == 1
    page = result.pages[0]
    assert page.text == "私は\n猫が\n好き"
    assert (page.start, page.end) == (Fraction(0), Fraction(3))
    assert page.layout.evidence == "m3"
    assert page.layout.line_widths == (40, 40, 40)
    assert page.source_stages == stages
    assert [
        (change.token.identifier, change.original_stage_onset, change.page_onset)
        for change in page.display_changes
    ] == [("t2", Fraction(1), Fraction(0)), ("t3", Fraction(2), Fraction(0))]
    assert [(change.token.start, change.token.end) for change in page.display_changes] == [
        (Fraction(1), Fraction(2)),
        (Fraction(2), Fraction(3)),
    ]
    assert result.policy.recognition_clocks_changed is False
    assert result.policy.progressive_growth_preserved is False
    assert result.candidate_only is True


@pytest.mark.parametrize("text", ["hello ", "مرحبا", "👩🏽‍💻", "猫の"])
def test_multilingual_text_is_not_normalized_or_split(
    stages: tuple[MeasuredStage, ...], text: str
) -> None:
    token = replace(stages[0].tokens[0], text=text)
    first = replace(stages[0], text=text, tokens=(token,))
    last = replace(
        stages[1],
        text=text + "\n猫が",
        tokens=(token, stages[1].tokens[1]),
        phrase_boundary_after=True,
    )
    plan = coalesce((first, last), policy=PagePolicy(Fraction(4), "synthetic-pos"))
    assert len(plan.pages) == 1
    assert plan.pages[0].text == text + "\n猫が"
    assert plan.pages[0].source_stages[0].tokens[0].text == text


def test_partial_phrase_is_never_new_page(stages: tuple[MeasuredStage, ...]) -> None:
    plan = coalesce(stages[:2], policy=PagePolicy(Fraction(4), "synthetic-pos"))
    assert len(plan.pages) == 2
    assert plan.pages[0].text == "私は"
    assert plan.pages[1].text == "私は\n猫が"
    assert plan.pages[1].display_changes == ()


def test_caller_duration_can_break_only_at_existing_phrase_boundary(
    stages: tuple[MeasuredStage, ...],
) -> None:
    second = replace(stages[1], phrase_boundary_after=True)
    plan = coalesce((stages[0], second, stages[2]), policy=PagePolicy(Fraction(2), "synthetic-pos"))
    assert len(plan.pages) == 2
    assert (plan.pages[0].start, plan.pages[0].end, plan.pages[0].text) == (
        Fraction(0),
        Fraction(2),
        "私は\n猫が",
    )
    assert (plan.pages[1].start, plan.pages[1].end) == (Fraction(2), Fraction(3))


def test_budget_inside_phrase_preserves_all_original_stages(
    stages: tuple[MeasuredStage, ...],
) -> None:
    plan = coalesce(stages, policy=PagePolicy(Fraction(2), "synthetic-pos"))
    assert len(plan.pages) == 3
    assert [page.text for page in plan.pages] == ["私は", "私は\n猫が", "私は\n猫が\n好き"]


@pytest.mark.parametrize("kind", ["cut", "utterance", "gap", "style", "region"])
def test_ownership_gaps_and_measurement_geometry_are_hard_boundaries(
    stages: tuple[MeasuredStage, ...], kind: str
) -> None:
    second = stages[1]
    if kind == "cut":
        second = replace(second, cut="different-cut")
    elif kind == "utterance":
        second = replace(second, utterance="different-speaker")
    elif kind == "gap":
        second = replace(second, start=Fraction(11, 10))
    elif kind == "style":
        second = replace(second, layout=replace(second.layout, style="other-font"))
    else:
        second = replace(second, layout=replace(second.layout, region_width=120))
    plan = coalesce(
        (stages[0], replace(second, phrase_boundary_after=True)),
        policy=PagePolicy(Fraction(4), "synthetic-pos"),
    )
    assert len(plan.pages) == 2
    assert plan.pages[1].display_changes == ()


def test_reset_is_not_dropped_as_a_prefix(stages: tuple[MeasuredStage, ...]) -> None:
    token = RecognizedToken("other", "次の話", Fraction(1), Fraction(2))
    second = replace(
        stages[1],
        text="次の話",
        tokens=(token,),
        layout=MeasuredLayout("other", "style", 100, (80,)),
        phrase_boundary_after=True,
    )
    plan = coalesce((stages[0], second), policy=PagePolicy(Fraction(4), "synthetic-pos"))
    assert [page.text for page in plan.pages] == ["私は", "次の話"]


@pytest.mark.parametrize(
    "kind",
    [
        "blank",
        "overflow",
        "four-lines",
        "missing-width",
        "missing-pos",
        "missing-measurement",
        "text-loss",
        "duplicate-token",
        "clock",
        "long-stage",
        "nan",
    ],
)
def test_rejects_unverified_or_lossy_input(stages: tuple[MeasuredStage, ...], kind: str) -> None:
    first = stages[0]
    policy = PagePolicy(Fraction(4), "synthetic-pos")
    if kind == "blank":
        first = replace(first, text="私は\n")
    elif kind == "overflow":
        first = replace(first, layout=replace(first.layout, line_widths=(101,)))
    elif kind == "four-lines":
        first = replace(first, text="私\nは\n猫\nだ")
    elif kind == "missing-width":
        first = replace(first, layout=replace(first.layout, line_widths=()))
    elif kind == "missing-pos":
        policy = replace(policy, phrase_evidence="")
    elif kind == "missing-measurement":
        first = replace(first, layout=replace(first.layout, evidence=""))
    elif kind == "text-loss":
        first = replace(first, text="猫")
    elif kind == "duplicate-token":
        first = replace(first, text="私は私は", tokens=(first.tokens[0], first.tokens[0]))
    elif kind == "clock":
        first = replace(first, tokens=(replace(first.tokens[0], start=Fraction(1, 2)),))
    elif kind == "long-stage":
        first = replace(first, end=Fraction(5))
    else:
        first = replace(first, layout=replace(first.layout, region_width=float("nan")))
    with pytest.raises(ReplayError):
        coalesce((first,), policy=policy)


@pytest.mark.parametrize(
    "policy",
    [
        PagePolicy(Fraction(0), "pos"),
        PagePolicy(Fraction(4), "pos", maximum_lines=4),
        PagePolicy(Fraction(4), "pos", recognition_clocks_changed=True),
        PagePolicy(Fraction(4), "pos", progressive_growth_preserved=True),
        PagePolicy(Fraction(4), "pos", display_mode="progressive"),
    ],
)
def test_policy_cannot_claim_original_progressive_behavior(
    stages: tuple[MeasuredStage, ...], policy: PagePolicy
) -> None:
    with pytest.raises(ReplayError, match="policy"):
        coalesce(stages, policy=policy)


@pytest.mark.parametrize("kind", ["overlap", "identifier"])
def test_refuses_overlap_and_duplicate_stages(stages: tuple[MeasuredStage, ...], kind: str) -> None:
    second = (
        replace(stages[1], identifier="s1")
        if kind == "identifier"
        else replace(stages[0], identifier="overlap")
    )
    with pytest.raises(ReplayError, match="overlap"):
        coalesce((stages[0], second), policy=PagePolicy(Fraction(4), "synthetic-pos"))


def test_empty_proposal_is_explicit() -> None:
    plan = coalesce((), policy=PagePolicy(Fraction(4), "synthetic-pos"))
    assert plan.pages == ()
    assert plan.original_stage_count == 0


def test_real_synthetic_pos_keeps_particle_with_its_noun() -> None:
    phrase_ends, _ = boundaries("今日は青い")
    assert phrase_ends == (3, 5)
    noun = RecognizedToken("noun", "今日", Fraction(0), Fraction(1))
    particle = RecognizedToken("particle", "は", Fraction(1), Fraction(2))
    adjective = RecognizedToken("adjective", "青い", Fraction(2), Fraction(3))
    first = MeasuredStage(
        "s1",
        "今日",
        Fraction(0),
        Fraction(1),
        "u",
        "c",
        (noun,),
        MeasuredLayout("m1", "font", 100, (40,)),
        2 in phrase_ends,
    )
    second = MeasuredStage(
        "s2",
        "今日は",
        Fraction(1),
        Fraction(2),
        "u",
        "c",
        (noun, particle),
        MeasuredLayout("m2", "font", 100, (60,)),
        3 in phrase_ends,
    )
    third = MeasuredStage(
        "s3",
        "今日は\n青い",
        Fraction(2),
        Fraction(3),
        "u",
        "c",
        (noun, particle, adjective),
        MeasuredLayout("m3", "font", 100, (60, 40)),
        5 in phrase_ends,
    )
    plan = coalesce((first, second, third), policy=PagePolicy(Fraction(2), "synthetic-vibrato-pos"))
    assert [page.text for page in plan.pages] == ["今日は", "今日は\n青い"]
    assert plan.pages[0].display_changes[0].token.text == "は"
    assert plan.pages[0].display_changes[0].token.start == Fraction(1)
    assert plan.pages[0].display_changes[0].page_onset == Fraction(0)
