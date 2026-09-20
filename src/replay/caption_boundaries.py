"""Protect confirmed speech at edit boundaries without relaxing VAD classification."""

from bisect import bisect_left, bisect_right
from dataclasses import dataclass
from math import ceil, floor, inf, isfinite, nextafter

from replay.caption_projection import CutSpan
from replay.caption_timing import SpokenToken


@dataclass(frozen=True, slots=True)
class AudioSpan:
    start: float
    end: float

    def __post_init__(self) -> None:
        if not isfinite(self.start) or not isfinite(self.end) or not 0 <= self.start < self.end:
            raise ValueError("Invalid audio boundary span")


@dataclass(frozen=True, slots=True)
class BoundaryPolicy:
    maximum_word_extension: float = 0.4
    maximum_attack_guard: float = 0.08
    minimum_quiet_run: float = 0.02

    def __post_init__(self) -> None:
        if not 0 <= self.maximum_word_extension <= 1:
            raise ValueError("Invalid word boundary extension")
        if not 0 <= self.maximum_attack_guard <= 0.2 or not 0.005 <= self.minimum_quiet_run <= 0.1:
            raise ValueError("Invalid acoustic boundary guard")


@dataclass(frozen=True, slots=True)
class ProtectedCuts:
    cuts: tuple[CutSpan, ...]
    speech_cores: tuple[AudioSpan, ...]
    unresolved_words: tuple[SpokenToken, ...]
    added_context_frames: int


def protect_boundaries(
    speech_cores: tuple[AudioSpan, ...],
    words: tuple[SpokenToken, ...],
    quiet_runs: tuple[AudioSpan, ...],
    *,
    source_frames: int,
    fps: int = 30,
    policy: BoundaryPolicy | None = None,
) -> ProtectedCuts:
    """Keep VAD decisions immutable; change only the mechanical edit boundaries.

    Only words overlapping an already accepted speech core can protect a boundary.
    A hallucination in a silent gap cannot create a new retained interval. Short
    pre-onset guards require observed quiet-run evidence, not blanket pre-roll.
    Quantization is outward: flooring starts and ceiling ends, never rounding a
    weak consonant away. Coarse conflicting words are reported, not silently used
    to retain arbitrary non-speech. Quiet runs must come from the unchanged detector.
    """
    if not 1 <= fps <= 120 or not 1 <= source_frames <= fps * 21600:
        raise ValueError("Invalid frame clock")
    if len(speech_cores) > 3000 or len(words) > 100_000 or len(quiet_runs) > 100_000:
        raise ValueError("Boundary planning budget exceeded")
    duration = source_frames / fps
    _validate_spans(speech_cores, duration)
    _validate_spans(quiet_runs, duration)
    starts, ends = _word_bounds(words, duration)
    selected = BoundaryPolicy() if policy is None else policy
    quiet_ends = [
        run.end for run in quiet_runs if run.end - run.start >= selected.minimum_quiet_run
    ]
    intervals: list[tuple[int, int]] = []
    baseline: list[tuple[int, int]] = []
    unresolved: list[SpokenToken] = []
    seen_conflicts: set[SpokenToken] = set()
    comparisons = 0
    for core in speech_cores:
        start, end = core.start, core.end
        _append_interval(baseline, floor(start * fps), min(source_frames, ceil(end * fps)))
        first, stop = bisect_right(ends, core.start), bisect_left(starts, core.end)
        comparisons += stop - first
        if comparisons > 500_000:
            raise ValueError("Boundary overlap work budget exceeded")
        earliest = core.start - selected.maximum_word_extension
        latest = core.end + selected.maximum_word_extension
        if selected.maximum_word_extension > 0:
            # Compensate one ULP of bound arithmetic, not an audio-clock tick.
            earliest = nextafter(earliest, -inf)
            latest = nextafter(latest, inf)
        for word in words[first:stop]:
            if word.start < earliest or word.end > latest:
                if word not in seen_conflicts:
                    unresolved.append(word)
                    seen_conflicts.add(word)
                continue
            start, end = min(start, word.start), max(end, word.end)
        boundary = start
        quiet_index = bisect_right(quiet_ends, start) - 1
        if quiet_index >= 0 and start - quiet_ends[quiet_index] <= selected.maximum_attack_guard:
            boundary = quiet_ends[quiet_index]
        _append_interval(
            intervals, max(0, floor(boundary * fps)), min(source_frames, ceil(end * fps))
        )
    cursor = 0
    cuts: list[CutSpan] = []
    for start_frame, end_frame in intervals:
        cuts.append(CutSpan(start_frame, end_frame, cursor))
        cursor += end_frame - start_frame
    original_frames = sum(end - start for start, end in baseline)
    return ProtectedCuts(tuple(cuts), speech_cores, tuple(unresolved), cursor - original_frames)


def _validate_spans(spans: tuple[AudioSpan, ...], duration: float) -> None:
    previous_end = 0.0
    for span in spans:
        if span.start < previous_end or span.end > duration:
            raise ValueError("Boundary evidence must be ordered, nonoverlapping and inside source")
        previous_end = span.end


def _word_bounds(
    words: tuple[SpokenToken, ...], duration: float
) -> tuple[list[float], list[float]]:
    starts: list[float] = []
    ends: list[float] = []
    for word in words:
        if word.end > duration or (starts and (word.start < starts[-1] or word.end < ends[-1])):
            raise ValueError("Word timing must be ordered and inside the source")
        starts.append(word.start)
        ends.append(word.end)
    return starts, ends


def _append_interval(intervals: list[tuple[int, int]], start: int, end: int) -> None:
    if intervals and start <= intervals[-1][1]:
        previous = intervals.pop()
        intervals.append((previous[0], max(end, previous[1])))
    else:
        intervals.append((start, end))
