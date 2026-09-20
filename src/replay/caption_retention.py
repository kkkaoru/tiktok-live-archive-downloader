"""Explicit nonquiet retention, separate from immutable classifier speech cores."""

from bisect import bisect_right
from dataclasses import dataclass
from math import ceil, floor

from .caption_boundaries import AudioSpan
from .caption_projection import CutSpan, MeasuredQuietSpan


@dataclass(frozen=True, slots=True)
class RetentionRequest:
    support_index: int
    start: float
    end: float
    source_start_frame: int
    source_end_frame: int


@dataclass(frozen=True, slots=True)
class RetentionPlan:
    cuts: tuple[CutSpan, ...]
    requests: tuple[RetentionRequest, ...]
    added_frames: int


def _span_key(span: AudioSpan) -> tuple[float, float]:
    return span.start, span.end


def _union(spans: tuple[AudioSpan, ...]) -> tuple[AudioSpan, ...]:
    result: list[AudioSpan] = []
    for span in sorted(spans, key=_span_key):
        if result and span.start <= result[-1].end:
            prior = result.pop()
            span = AudioSpan(prior.start, max(prior.end, span.end))
        result.append(span)
    return tuple(result)


def _missing(
    span: AudioSpan, covered: tuple[AudioSpan, ...], ends: tuple[float, ...]
) -> tuple[AudioSpan, ...]:
    index = bisect_right(ends, span.start)
    cursor = span.start
    missing: list[AudioSpan] = []
    while index < len(covered) and covered[index].start < span.end:
        current = covered[index]
        if current.start > cursor:
            missing.append(AudioSpan(cursor, min(current.start, span.end)))
        cursor = max(cursor, current.end)
        index += 1
    if cursor < span.end:
        missing.append(AudioSpan(cursor, span.end))
    return tuple(missing)


def retain_nonquiet_support(
    base: tuple[CutSpan, ...],
    supports: tuple[AudioSpan, ...],
    quiet: tuple[MeasuredQuietSpan, ...],
    *,
    source_frames: int,
    fps: int = 30,
) -> RetentionPlan:
    """Report every added nonquiet support portion; never reclassify it as a core.

    Supports must come from explicit recognition/original-speech evidence. Only
    independently measured quiet may be excluded. ASR absence is not accepted as
    quiet input. Output rounding is outward and cannot truncate a weak onset.
    """
    if (
        type(source_frames) is not int
        or type(fps) is not int
        or not 1 <= fps <= 120
        or not 1 <= source_frames <= fps * 21600
        or len(base) > 3000
        or len(supports) > 100_000
        or len(quiet) > 100_000
    ):
        raise ValueError("Retention budget exceeded")
    duration = source_frames / fps
    previous_end = cursor = 0
    for cut in base:
        if (
            cut.source_start_frame < previous_end
            or cut.source_end_frame > source_frames
            or cut.output_start_frame != cursor
        ):
            raise ValueError("Invalid base cut timeline")
        previous_end = cut.source_end_frame
        cursor += cut.duration_frames
    if any(span.end > duration for span in supports) or any(span.end > duration for span in quiet):
        raise ValueError("Retention evidence exceeds source")
    covered = _union(
        tuple(AudioSpan(cut.source_start_frame / fps, cut.source_end_frame / fps) for cut in base)
        + tuple(AudioSpan(span.start, span.end) for span in quiet)
    )
    ends = tuple(span.end for span in covered)
    requests: list[RetentionRequest] = []
    for index, span in enumerate(supports):
        for missing in _missing(span, covered, ends):
            requests.append(
                RetentionRequest(
                    index,
                    missing.start,
                    missing.end,
                    floor(missing.start * fps),
                    min(source_frames, ceil(missing.end * fps)),
                )
            )
            if len(requests) > 100_000:
                raise ValueError("Retention request budget exceeded")
    intervals = [(cut.source_start_frame, cut.source_end_frame) for cut in base]
    intervals.extend((request.source_start_frame, request.source_end_frame) for request in requests)
    merged: list[tuple[int, int]] = []
    for start, end in sorted(intervals):
        if merged and start <= merged[-1][1]:
            previous = merged.pop()
            start, end = previous[0], max(previous[1], end)
        merged.append((start, end))
    cuts: list[CutSpan] = []
    offset = 0
    for start, end in merged:
        cuts.append(CutSpan(start, end, offset))
        offset += end - start
    if len(cuts) > 3000:
        raise ValueError("Retained timeline exceeds native cut budget")
    return RetentionPlan(tuple(cuts), tuple(requests), offset - cursor)
