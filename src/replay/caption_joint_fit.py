"""Jointly choose a POS-safe simultaneous layout and measured uniform scale."""

from collections.abc import Callable, Iterator
from dataclasses import dataclass
from itertools import chain, combinations, islice, pairwise
from math import isfinite
from typing import Literal

from replay.caption_line_legality import assess_lines
from replay.caption_linguistics import (
    MAX_LAYOUT_CANDIDATES,
    MAX_TEXT_LENGTH,
    CaptionLayoutError,
    _rhythmic_ends,
    boundaries,
)
from replay.caption_style_defaults import MAX_CAPTION_LINES

Measurement = Callable[[str], tuple[float, float]]
LinePreparation = Callable[[tuple[str, ...]], None]
DEFAULT_CANDIDATE_BUDGET = 1024
DEFAULT_MEASUREMENT_BUDGET = 4096
MAX_MEASUREMENT_BUDGET = MAX_LAYOUT_CANDIDATES * (MAX_CAPTION_LINES + 1)


@dataclass(frozen=True, slots=True)
class JointFitBudget:
    """Bound proposed layouts and submitted measurement/preparation units."""

    maximum_candidates: int = DEFAULT_CANDIDATE_BUDGET
    maximum_measurements: int = DEFAULT_MEASUREMENT_BUDGET

    def __post_init__(self) -> None:
        if (
            type(self.maximum_candidates) is not int
            or not 1 <= self.maximum_candidates <= MAX_LAYOUT_CANDIDATES
            or type(self.maximum_measurements) is not int
            or not 1 <= self.maximum_measurements <= MAX_MEASUREMENT_BUDGET
        ):
            raise ValueError("Invalid joint caption fit budget")


@dataclass(frozen=True, slots=True)
class JointCaptionFit:
    """Measured geometry is a candidate, never native readability acceptance."""

    text: str
    width: float
    height: float
    factor: float
    line_count: int
    candidate_count: int
    measurement_units: int
    readability_accepted: Literal[False] = False


DEFAULT_BUDGET = JointFitBudget()


def _positive(value: float) -> bool:
    return not isinstance(value, bool) and isfinite(value) and value > 0


def _splits(points: tuple[int, ...], maximum_lines: int) -> Iterator[tuple[int, ...]]:
    return chain.from_iterable(combinations(points, count) for count in range(1, maximum_lines))


def _layouts(text: str, maximum_lines: int, budget: JointFitBudget) -> tuple[str, ...]:
    phrase_ends, _ = boundaries(text)
    points = tuple(
        sorted(
            point for point in _rhythmic_ends(text, frozenset(phrase_ends)) if 0 < point < len(text)
        )
    )
    candidates = [text]
    attempted = 1
    for split in islice(_splits(points, maximum_lines), budget.maximum_candidates):
        attempted += 1
        if attempted > budget.maximum_candidates:
            raise CaptionLayoutError("Joint caption candidate budget exceeded")
        lines = tuple(text[start:end] for start, end in pairwise((0, *split, len(text))))
        if assess_lines(text, lines).legal:
            candidates.append("\n".join(lines))
    return tuple(candidates)


def _rank(candidate: JointCaptionFit) -> tuple[float, float, float]:
    if candidate.factor == 1:
        return candidate.factor, -float(candidate.line_count), -candidate.width
    return candidate.factor, -candidate.width, -float(candidate.line_count)


def joint_fit(
    text: str,
    *,
    measure: Measurement,
    available_width: float,
    available_height: float,
    maximum_lines: int = MAX_CAPTION_LINES,
    indivisible: bool = False,
    prepare_lines: LinePreparation | None = None,
    budget: JointFitBudget = DEFAULT_BUDGET,
) -> JointCaptionFit:
    """Measure all bounded legitimate layouts, then maximize their uniform fit.

    Only newlines at existing phrase/rhythmic boundaries are added. The callable
    measures each whole rendered candidate, including line advance and paint.
    Preparation submits unique single lines once, within the same unit budget.
    Caller-verified indivisible units retain exactly one intact layout. No source
    characters, clocks, font, palette or placement are modified here.
    Budget overflow fails before submitting any measurement, rather than silently
    selecting from an incomplete search. Timing eligibility belongs to the caller.
    """
    if (
        not text.strip()
        or len(text) > MAX_TEXT_LENGTH
        or any(character in text for character in "\r\n\0")
        or not _positive(available_width)
        or not _positive(available_height)
        or type(maximum_lines) is not int
        or not 1 <= maximum_lines <= MAX_CAPTION_LINES
        or type(indivisible) is not bool
    ):
        raise ValueError("Invalid joint caption fit input")
    candidates = (text,) if indivisible else _layouts(text, maximum_lines, budget)
    unique_lines = tuple(dict.fromkeys(chain.from_iterable(row.split("\n") for row in candidates)))
    units = len(candidates) + (len(unique_lines) if prepare_lines is not None else 0)
    if units > budget.maximum_measurements:
        raise CaptionLayoutError("Joint caption measurement budget exceeded")
    if prepare_lines is not None:
        prepare_lines(unique_lines)
    measured: list[JointCaptionFit] = []
    for candidate in candidates:
        width, height = measure(candidate)
        if not _positive(width) or not _positive(height):
            raise ValueError("Invalid rendered joint caption dimensions")
        factor = min(1.0, available_width / width, available_height / height)
        if not _positive(factor):
            raise ValueError("Joint caption fit factor underflow")
        measured.append(
            JointCaptionFit(
                candidate,
                width,
                height,
                factor,
                candidate.count("\n") + 1,
                len(candidates),
                units,
            )
        )
    return max(measured, key=_rank)
