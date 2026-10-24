"""Conservative normalized-frame cuts from measured quiet, not inferred silence."""

from math import ceil, floor, isfinite

from .caption_projection import CutSpan, MeasuredQuietSpan
from .caption_repair import ClockLabel


def _protected(
    labels: tuple[ClockLabel, ...], *, frames: int, fps: int, guard: float
) -> list[tuple[int, int]]:
    intervals: list[tuple[int, int]] = []
    for label in labels:
        if label.end > frames / fps:
            raise ValueError("Recognition exceeds the planned clock")
        left = max(0, min(frames - 1, floor((label.start - guard) * fps)))
        right = min(frames, max(left + 1, ceil((label.end + guard) * fps)))
        intervals.append((left, right))
    merged: list[tuple[int, int]] = []
    for left, right in sorted(intervals):
        if merged and left <= merged[-1][1]:
            left, previous = merged.pop()
            right = max(previous, right)
        merged.append((left, right))
    return merged


def quiet_cut_frames(
    quiet: tuple[MeasuredQuietSpan, ...],
    labels: tuple[ClockLabel, ...],
    *,
    source_frames: int,
    fps: int = 30,
    minimum_cut_seconds: float = 0.3,
    speech_guard_seconds: float = 0.08,
) -> tuple[CutSpan, ...]:
    """Remove only measured quiet interiors outside conservative ASR protection.

    Caller supplies a verified NORMALIZED output grid, not original VFR ordinals,
    and binds quiet/recognition evidence to the same source-presentation clock.
    Quiet is rounded inward; recognized intervals and their guard round outward.
    Even an unresolved point protects its containing frame. These protection
    extents are NOT replacement recognition clocks or a point repair. All other
    unclassified sound/video stays; this is not a VAD or non-speech certificate.
    """
    if (
        type(source_frames) is not int
        or type(fps) is not int
        or not 1 <= fps <= 120
        or not 1 <= source_frames <= fps * 21600
        or len(quiet) > 100_000
        or len(labels) > 100_000
        or isinstance(minimum_cut_seconds, bool)
        or not isfinite(minimum_cut_seconds)
        or not 0.02 <= minimum_cut_seconds <= 10
        or isinstance(speech_guard_seconds, bool)
        or not isfinite(speech_guard_seconds)
        or not 0 <= speech_guard_seconds <= 1
    ):
        raise ValueError("Invalid quiet-cut policy or budget")
    protected = _protected(labels, frames=source_frames, fps=fps, guard=speech_guard_seconds)
    minimum = ceil(minimum_cut_seconds * fps)
    removed: list[tuple[int, int]] = []
    previous_end = 0.0
    index = 0
    for span in quiet:
        # Decoded audio may outlast the last video frame by under a frame (14 ms on a YouTube
        # source); quiet there has no frame to cut.
        if span.start < previous_end or span.end > (source_frames + 1) / fps:
            raise ValueError("Quiet evidence must be ordered and inside the planned clock")
        previous_end = span.end
        cursor, end = ceil(span.start * fps), min(floor(span.end * fps), source_frames)
        while index < len(protected) and protected[index][1] <= cursor:
            index += 1
        while index < len(protected) and protected[index][0] < end:
            left, right = protected[index]
            if left - cursor >= minimum:
                removed.append((cursor, left))
            cursor = max(cursor, right)
            if right >= end:
                break
            index += 1
        if end - cursor >= minimum:
            removed.append((cursor, end))
    cuts: list[CutSpan] = []
    source_cursor = output_cursor = 0
    for left, right in [*removed, (source_frames, source_frames)]:
        if source_cursor < left:
            cuts.append(CutSpan(source_cursor, left, output_cursor))
            output_cursor += left - source_cursor
        source_cursor = right
    if not cuts:
        raise ValueError("Quiet cuts would remove the entire recording")
    return tuple(cuts)
