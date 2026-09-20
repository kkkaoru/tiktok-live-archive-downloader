"""Conservative point repair using complete, identical-text native-clock alternatives."""

from dataclasses import dataclass
from itertools import pairwise
from math import isfinite


@dataclass(frozen=True, slots=True)
class ClockLabel:
    text: str
    start: float
    end: float
    provenance: str

    def __post_init__(self) -> None:
        if (
            not self.text.strip()
            or len(self.text) > 2048
            or "\0" in self.text
            or not self.provenance
            or len(self.provenance) > 256
            or "\0" in self.provenance
        ):
            raise ValueError("Invalid recognition label provenance or text")
        if not isfinite(self.start) or not isfinite(self.end) or not 0 <= self.start <= self.end:
            raise ValueError("Invalid native recognition clock")


@dataclass(frozen=True, slots=True)
class ClockRepair:
    original_start: int
    original_end: int
    replacement: tuple[ClockLabel, ...]


@dataclass(frozen=True, slots=True)
class RepairedLabels:
    labels: tuple[ClockLabel, ...]
    repairs: tuple[ClockRepair, ...]
    unresolved_original_indices: tuple[int, ...]


def canonical_text(text: str) -> str:
    """Whitespace-only equivalence, never spelling or punctuation correction."""
    return "".join(text.split())


def _ordered(labels: tuple[ClockLabel, ...]) -> None:
    if any(right.start < left.start for left, right in pairwise(labels)):
        raise ValueError("Recognition onsets must be ordered")


def _match(
    pattern: str,
    alternative: tuple[ClockLabel, ...],
    *,
    lower: float,
    upper: float,
    point: float,
) -> tuple[ClockLabel, ...] | None:
    boundaries = [0]
    for label in alternative:
        boundaries.append(boundaries[-1] + len(canonical_text(label.text)))
    indices = {offset: index for index, offset in enumerate(boundaries)}
    text = "".join(canonical_text(label.text) for label in alternative)
    offset = text.find(pattern)
    while offset >= 0:
        end_offset = offset + len(pattern)
        if offset in indices and end_offset in indices:
            selected = alternative[indices[offset] : indices[end_offset]]
            if (
                selected
                and selected[0].start >= lower
                and selected[-1].end <= upper
                and selected[0].start - 0.25 <= point <= selected[-1].end + 0.25
                and all(label.end > label.start for label in selected)
                and all(right.end >= left.end for left, right in pairwise(selected))
            ):
                return selected
        offset = text.find(pattern, offset + 1)
    return None


def repair_native_points(
    original: tuple[ClockLabel, ...],
    alternatives: tuple[tuple[ClockLabel, ...], ...],
    *,
    source_duration: float,
    maximum_block_labels: int = 32,
) -> RepairedLabels:
    """Replace whole label blocks, never stretch a point into a neighboring clock.

    Alternative boundaries and text must come directly from an independent local
    recognition result. Whitespace-equivalent text must match a complete original
    block and complete alternative labels. Unchanged external neighbors must still
    fit, and the block envelope may differ by at most 0.25 seconds. This is a
    selection bound, not permission to adjust a clock. Unresolved points remain
    in the result, not silently omitted as silence.
    This is a timing-conservation policy, not a speech-accuracy assertion.
    """
    if (
        isinstance(source_duration, bool)
        or not isfinite(source_duration)
        or not 0 < source_duration <= 86400
        or type(maximum_block_labels) is not int
        or not 1 <= maximum_block_labels <= 32
        or len(original) > 100_000
        or len(alternatives) > 256
        or sum(len(part) for part in alternatives) > 100_000
    ):
        raise ValueError("Clock repair budget exceeded")
    if (
        sum(len(label.text.encode("utf-8")) for part in (original, *alternatives) for label in part)
        > 8_000_000
    ):
        raise ValueError("Clock repair text exceeds budget")
    _ordered(original)
    for part in alternatives:
        _ordered(part)
    if any(label.end > source_duration for part in (original, *alternatives) for label in part):
        raise ValueError("Recognition exceeds source duration")
    choices = tuple((part, max(label.end for label in part)) for part in alternatives if part)
    if not choices:
        return RepairedLabels(
            original,
            (),
            tuple(index for index, label in enumerate(original) if label.start == label.end),
        )
    repairs: list[ClockRepair] = []
    unresolved: list[int] = []
    work = 0
    for point_index, label in enumerate(original):
        if label.end > label.start or any(
            repair.original_start <= point_index < repair.original_end for repair in repairs
        ):
            continue
        found: ClockRepair | None = None
        for size in range(1, min(maximum_block_labels, len(original)) + 1):
            for left in range(
                max(0, point_index - size + 1), min(point_index + 1, len(original) - size + 1)
            ):
                right = left + size
                if any(
                    left < prior.original_end and prior.original_start < right for prior in repairs
                ):
                    continue
                lower = original[left - 1].end if left else 0.0
                upper = original[right].start if right < len(original) else source_duration
                for prior in repairs:
                    if prior.original_end == left:
                        lower = prior.replacement[-1].end
                    if prior.original_start == right:
                        upper = prior.replacement[0].start
                pattern = "".join(canonical_text(item.text) for item in original[left:right])
                for part, alternative_end in choices:
                    work += 1
                    if work > 250_000:
                        raise ValueError("Clock repair search exceeds budget")
                    if alternative_end < lower or part[0].start > upper:
                        continue
                    matched = _match(pattern, part, lower=lower, upper=upper, point=label.start)
                    if matched is not None and (
                        abs(matched[0].start - original[left].start) <= 0.25
                        and abs(matched[-1].end - max(item.end for item in original[left:right]))
                        <= 0.25
                    ):
                        found = ClockRepair(left, right, matched)
                        break
                if found is not None:
                    break
            if found is not None:
                break
        if found is None:
            unresolved.append(point_index)
        else:
            repairs.append(found)
    repairs.sort(key=lambda repair: repair.original_start)
    result: list[ClockLabel] = []
    cursor = 0
    for repair in repairs:
        result.extend(original[cursor : repair.original_start])
        result.extend(repair.replacement)
        cursor = repair.original_end
    result.extend(original[cursor:])
    if "".join(canonical_text(label.text) for label in original) != "".join(
        canonical_text(label.text) for label in result
    ):
        raise ValueError("Clock repair changed transcript text")
    _ordered(tuple(result))
    remaining = tuple(
        index
        for index in unresolved
        if not any(repair.original_start <= index < repair.original_end for repair in repairs)
    )
    return RepairedLabels(tuple(result), tuple(repairs), remaining)
