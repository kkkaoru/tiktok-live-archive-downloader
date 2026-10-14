"""Fit simultaneous lexical groups whose original ASR anchors cannot split them."""

from bisect import bisect_left, bisect_right

from replay.caption_joint_fit import DEFAULT_MEASUREMENT_BUDGET, JointFitBudget, joint_fit
from replay.caption_linguistics import _atomic_ends, atomic_tokens, recognition_continuity
from replay.caption_measure_cache import CaptionMeasurer
from replay.caption_scale import FOCUSED_TIMING_SECONDS, CaptionScale
from replay.caption_style_defaults import MAX_CAPTION_LINES
from replay.caption_timing import CaptionPolicy, SpokenToken

BASELINE_MEASUREMENT_UNITS = 1


def anchored_groups(words: list[SpokenToken]) -> dict[str, tuple[int, float]]:
    """Record only fine original fragments with no safe internal lexical onset."""
    starts = tuple(word.start for word in words)
    result: dict[str, tuple[int, float]] = {}
    for unit in atomic_tokens(words, maximum_gap=CaptionPolicy().maximum_gap):
        ends = _atomic_ends(unit.text)
        if ends == frozenset({len(unit.text)}):
            continue
        members = words[bisect_left(starts, unit.start) : bisect_right(starts, unit.end)]
        members = [
            word
            for word in members
            if word.end <= unit.end and recognition_continuity(word.utterance) == unit.utterance
        ]
        if len(members) < 2 or "".join(word.text for word in members) != unit.text:
            continue
        maximum = max(word.end - word.start for word in members)
        offset = 0
        anchors: set[int] = set()
        for word in members[:-1]:
            offset += len(word.text)
            anchors.add(offset)
        if maximum <= FOCUSED_TIMING_SECONDS and not anchors.intersection(ends):
            result[unit.text] = len(members), maximum
    return result


def fit_group(
    text: str,
    *,
    layout: CaptionMeasurer,
    maximum_height: float | None,
    support: tuple[int, float],
) -> tuple[str, CaptionScale | None]:
    """Compare bounded POS-safe simultaneous layouts, without inventing timing."""
    available_height = maximum_height
    budget = JointFitBudget()
    if available_height is None:
        available_height = layout.measure(text)[1] * MAX_CAPTION_LINES
        budget = JointFitBudget(
            maximum_measurements=DEFAULT_MEASUREMENT_BUDGET - BASELINE_MEASUREMENT_UNITS
        )
    fitted = joint_fit(
        text,
        measure=layout.measure,
        available_width=layout.width,
        available_height=available_height,
        prepare_lines=layout.prepare_lines,
        budget=budget,
    )
    if fitted.factor == 1.0:
        return fitted.text, None
    return fitted.text, CaptionScale(
        fitted.width,
        fitted.height,
        layout.width,
        available_height,
        support[0],
        support[1],
        provenance="recognition-anchor-group",
    )
