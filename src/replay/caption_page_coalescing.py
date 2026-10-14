"""Independent, opt-in planning for measured prefix pages, without rendering.

A page displays its final measured text at its first stage's display onset.
This deliberately advances later words' display times. Recognition clocks stay
unchanged; word-by-word reveal and 1-to-2-to-3 growth are NOT preserved. No current
workflow calls this utility. Supplied measurement and POS evidence is retained,
not independently certified here.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from fractions import Fraction

from replay.caption_style_defaults import MAX_CAPTION_LINES
from replay.core import ReplayError


@dataclass(frozen=True, slots=True)
class RecognizedToken:
    identifier: str
    text: str
    start: Fraction
    end: Fraction


@dataclass(frozen=True, slots=True)
class MeasuredLayout:
    """Caller-supplied native measurement identity and exact displayed line widths."""

    evidence: str
    style: str
    region_width: float
    line_widths: tuple[float, ...]


@dataclass(frozen=True, slots=True)
class MeasuredStage:
    identifier: str
    text: str
    start: Fraction
    end: Fraction
    utterance: str
    cut: str
    tokens: tuple[RecognizedToken, ...]
    layout: MeasuredLayout
    phrase_boundary_after: bool


@dataclass(frozen=True, slots=True)
class PagePolicy:
    """Explicit caller budget, never a replay-specific or inferred time constant."""

    maximum_duration: Fraction
    phrase_evidence: str
    maximum_lines: int = MAX_CAPTION_LINES
    display_mode: str = "final-measured-prefix-at-page-onset"
    recognition_clocks_changed: bool = False
    progressive_growth_preserved: bool = False


@dataclass(frozen=True, slots=True)
class DisplayChange:
    token: RecognizedToken
    original_stage_onset: Fraction
    page_onset: Fraction


@dataclass(frozen=True, slots=True)
class CaptionPage:
    text: str
    start: Fraction
    end: Fraction
    layout: MeasuredLayout
    source_stages: tuple[MeasuredStage, ...]
    display_changes: tuple[DisplayChange, ...]


@dataclass(frozen=True, slots=True)
class PagePlan:
    policy: PagePolicy
    pages: tuple[CaptionPage, ...]
    original_stage_count: int
    candidate_only: bool = True


def _clock(start: Fraction, end: Fraction) -> bool:
    return isinstance(start, Fraction) and isinstance(end, Fraction) and 0 <= start < end


def _validate_stage(stage: MeasuredStage, policy: PagePolicy) -> None:
    lines = stage.text.split("\n")
    layout = stage.layout
    if (
        not stage.identifier
        or not stage.utterance
        or not stage.cut
        or not _clock(stage.start, stage.end)
        or stage.end - stage.start > policy.maximum_duration
        or type(stage.phrase_boundary_after) is not bool
        or not 1 <= len(lines) <= policy.maximum_lines
        or any(not line.strip() or "\r" in line for line in lines)
        or not layout.evidence
        or not layout.style
        or not math.isfinite(layout.region_width)
        or layout.region_width <= 0
        or len(layout.line_widths) != len(lines)
        or any(
            not math.isfinite(width) or not 0 < width <= layout.region_width
            for width in layout.line_widths
        )
        or not stage.tokens
        or len({token.identifier for token in stage.tokens}) != len(stage.tokens)
        or "".join(stage.text.split("\n")) != "".join(token.text for token in stage.tokens)
    ):
        raise ReplayError("Invalid measured caption stage or incomplete text provenance")
    previous = Fraction(0)
    for token in stage.tokens:
        if (
            not token.identifier
            or not token.text
            or not _clock(token.start, token.end)
            or token.start < previous
            or token.start > stage.start
        ):
            raise ReplayError("Invalid or not-yet-recognized input token clock")
        previous = token.start


def _continues(previous: MeasuredStage, following: MeasuredStage) -> bool:
    return (
        previous.end == following.start
        and previous.utterance == following.utterance
        and previous.cut == following.cut
        and previous.layout.style == following.layout.style
        and previous.layout.region_width == following.layout.region_width
        and len(following.tokens) > len(previous.tokens)
        and following.tokens[: len(previous.tokens)] == previous.tokens
    )


def _stage_onsets(stage: MeasuredStage) -> dict[str, Fraction]:
    return {token.identifier: stage.start for token in stage.tokens}


def _page(stages: tuple[MeasuredStage, ...]) -> CaptionPage:
    first, last = stages[0], stages[-1]
    onsets: dict[str, Fraction] = {}
    for stage in stages:
        onsets = _stage_onsets(stage) | onsets
    changes = tuple(
        DisplayChange(token, onsets[token.identifier], first.start)
        for token in last.tokens
        if onsets[token.identifier] != first.start
    )
    return CaptionPage(last.text, first.start, last.end, last.layout, stages, changes)


def _flush(stages: tuple[MeasuredStage, ...]) -> tuple[CaptionPage, ...]:
    # Never create a grouped page that ends inside a supplied phrase. Keep the
    # original stages if a gap, cut, reset or explicit duration budget prevents it.
    if len(stages) > 1 and stages[-1].phrase_boundary_after:
        return (_page(stages),)
    return tuple(_page((stage,)) for stage in stages)


def coalesce(stages: tuple[MeasuredStage, ...], *, policy: PagePolicy) -> PagePlan:
    """Preserve every input stage and token in a non-rendering page proposal.

    Only exact contiguous prefix growth may merge. Cuts, utterance owners, gaps,
    resets, style changes and caller duration budgets break a group. The final
    stage supplies the unchanged measured line layout, and every advanced display
    onset is recorded separately from its unchanged recognition timestamps.
    """
    if (
        not isinstance(policy.maximum_duration, Fraction)
        or policy.maximum_duration <= 0
        or not policy.phrase_evidence
        or type(policy.maximum_lines) is not int
        or not 1 <= policy.maximum_lines <= MAX_CAPTION_LINES
        or policy.display_mode != "final-measured-prefix-at-page-onset"
        or policy.recognition_clocks_changed is not False
        or policy.progressive_growth_preserved is not False
    ):
        raise ReplayError("Invalid explicit caption page policy")
    pages: list[CaptionPage] = []
    pending: list[MeasuredStage] = []
    identifiers: set[str] = set()
    prior: MeasuredStage | None = None
    for stage in stages:
        _validate_stage(stage, policy)
        if stage.identifier in identifiers or (prior is not None and prior.end > stage.start):
            raise ReplayError("Caption stages overlap, are unordered, or repeat an identifier")
        identifiers.add(stage.identifier)
        if pending and (
            not _continues(pending[-1], stage)
            or stage.end - pending[0].start > policy.maximum_duration
        ):
            pages.extend(_flush(tuple(pending)))
            pending.clear()
        pending.append(stage)
        prior = stage
    pages.extend(_flush(tuple(pending)))
    return PagePlan(policy, tuple(pages), len(stages))
