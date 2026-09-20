"""Project actual timed lexical units through a frame-quantized cut timeline."""

from bisect import bisect_right
from dataclasses import dataclass
from math import isfinite

from replay.caption_timing import SpokenToken


@dataclass(frozen=True, slots=True)
class CutSpan:
    source_start_frame: int
    source_end_frame: int
    output_start_frame: int

    def __post_init__(self) -> None:
        if not 0 <= self.source_start_frame < self.source_end_frame:
            raise ValueError("Invalid source cut span")
        if self.output_start_frame < 0:
            raise ValueError("Invalid output cut position")

    @property
    def duration_frames(self) -> int:
        return self.source_end_frame - self.source_start_frame


@dataclass(frozen=True, slots=True)
class MeasuredQuietSpan:
    """Caller-supplied waveform evidence, never inferred from an empty transcript."""

    start: float
    end: float

    def __post_init__(self) -> None:
        if not isfinite(self.start) or not isfinite(self.end) or not 0 <= self.start < self.end:
            raise ValueError("Invalid measured quiet interval")


@dataclass(frozen=True, slots=True)
class TokenProjection:
    tokens: tuple[SpokenToken, ...]
    omitted: tuple[SpokenToken, ...]
    boundary_conflicts: tuple[SpokenToken, ...]


def project_tokens(
    tokens: tuple[SpokenToken, ...], cuts: tuple[CutSpan, ...], *, fps: int = 30
) -> TokenProjection:
    """Never duplicate a complete old phrase on either side of a jump cut.

    A token's midpoint selects at most one retained span. Material extending more
    than one quantization frame outside that span is reported as a boundary
    conflict and not displayed. No text or word times are proportionally split.
    Raw/omitted/conflicting tokens must remain available for review.
    """
    if not 1 <= fps <= 120 or len(cuts) > 3000 or len(tokens) > 100_000:
        raise ValueError("Caption projection budget exceeded")
    starts = _cut_starts(cuts, fps=fps)
    projected: list[SpokenToken] = []
    omitted: list[SpokenToken] = []
    conflicts: list[SpokenToken] = []
    previous_start = 0.0
    for token in tokens:
        if token.start < previous_start:
            raise ValueError("Source tokens must be ordered")
        previous_start = token.start
        midpoint = (token.start + token.end) / 2
        index = bisect_right(starts, midpoint) - 1
        if index < 0 or midpoint >= cuts[index].source_end_frame / fps:
            omitted.append(token)
            continue
        selected = cuts[index]
        source_start = selected.source_start_frame / fps
        source_end = selected.source_end_frame / fps
        tolerance = 1 / fps + 1e-9
        if token.start < source_start - tolerance or token.end > source_end + tolerance:
            conflicts.append(token)
            continue
        start = max(source_start, token.start)
        end = min(source_end, token.end)
        offset = selected.output_start_frame / fps - source_start
        projected.append(
            SpokenToken(token.text, start + offset, end + offset, f"{token.utterance}:cut-{index}")
        )
    return TokenProjection(tuple(projected), tuple(omitted), tuple(conflicts))


def project_tokens_over_quiet(
    tokens: tuple[SpokenToken, ...],
    cuts: tuple[CutSpan, ...],
    quiet_runs: tuple[MeasuredQuietSpan, ...],
    *,
    fps: int = 30,
) -> TokenProjection:
    """Map a whole label once across cuts only when every removed part is measured quiet.

    This changes neither cuts nor VAD classification. Returned clocks are display
    times on the edited timeline, not replacements for raw recognition clocks.
    No characters are apportioned among fragments. Missing nonquiet coverage is a
    conflict, including wholly unretained words. Omitted quiet words remain in the
    result for review. Callers must retain raw clocks and independent measurement
    provenance; ASR gaps or low sound-classification scores are not quiet evidence.
    """
    if not 1 <= fps <= 120 or len(cuts) > 3000 or len(tokens) > 100_000:
        raise ValueError("Caption projection budget exceeded")
    starts = _cut_starts(cuts, fps=fps)
    quiet = _quiet_ranges(quiet_runs)
    quiet_starts = [span.start for span in quiet]
    projected: list[SpokenToken] = []
    omitted: list[SpokenToken] = []
    conflicts: list[SpokenToken] = []
    previous_start = 0.0
    remaining = 500_000
    for token in tokens:
        if token.start < previous_start:
            raise ValueError("Source tokens must be ordered")
        previous_start = token.start
        result, work = _quiet_token(
            token,
            cuts=cuts,
            starts=starts,
            quiet=quiet,
            quiet_starts=quiet_starts,
            fps=fps,
            remaining=remaining,
        )
        remaining -= work
        projected.extend(result.tokens)
        omitted.extend(result.omitted)
        conflicts.extend(result.boundary_conflicts)
    return TokenProjection(tuple(projected), tuple(omitted), tuple(conflicts))


def _cut_starts(cuts: tuple[CutSpan, ...], *, fps: int) -> list[float]:
    cursor = 0
    previous_end = 0
    for cut in cuts:
        if cut.source_start_frame < previous_end or cut.output_start_frame != cursor:
            raise ValueError("Cuts must be source-ordered with a contiguous output timeline")
        cursor += cut.duration_frames
        previous_end = cut.source_end_frame
    return [cut.source_start_frame / fps for cut in cuts]


def _quiet_ranges(spans: tuple[MeasuredQuietSpan, ...]) -> list[MeasuredQuietSpan]:
    if len(spans) > 100_000:
        raise ValueError("Quiet evidence budget exceeded")
    result: list[MeasuredQuietSpan] = []
    for span in spans:
        if result and span.start < result[-1].end:
            raise ValueError("Quiet evidence must be ordered and nonoverlapping")
        if result and span.start == result[-1].end:
            previous = result.pop()
            result.append(MeasuredQuietSpan(previous.start, span.end))
        else:
            result.append(span)
    return result


def _quiet_covers(
    start: float,
    end: float,
    *,
    quiet: list[MeasuredQuietSpan],
    quiet_starts: list[float],
) -> bool:
    if start == end:
        return True
    index = bisect_right(quiet_starts, start) - 1
    return index >= 0 and quiet[index].end >= end


def _quiet_token(
    token: SpokenToken,
    *,
    cuts: tuple[CutSpan, ...],
    starts: list[float],
    quiet: list[MeasuredQuietSpan],
    quiet_starts: list[float],
    fps: int,
    remaining: int,
) -> tuple[TokenProjection, int]:
    index = max(0, bisect_right(starts, token.start) - 1)
    cursor = token.start
    first: float | None = None
    last: float | None = None
    first_index = index
    last_index = index
    work = 0
    while index < len(cuts) and starts[index] < token.end:
        work += 1
        if work > remaining:
            raise ValueError("Quiet projection work budget exceeded")
        cut = cuts[index]
        start = max(token.start, starts[index])
        end = min(token.end, cut.source_end_frame / fps)
        if start < end:
            if not _quiet_covers(cursor, start, quiet=quiet, quiet_starts=quiet_starts):
                return TokenProjection((), (), (token,)), work
            offset = cut.output_start_frame / fps - starts[index]
            if first is None:
                first = start + offset
                first_index = index
            last = end + offset
            last_index = index
            cursor = end
        index += 1
    if not _quiet_covers(cursor, token.end, quiet=quiet, quiet_starts=quiet_starts):
        return TokenProjection((), (), (token,)), work
    if first is None or last is None:
        return TokenProjection((), (token,), ()), work
    mapped = SpokenToken(
        token.text, first, last, f"{token.utterance}:quiet-cuts-{first_index}-{last_index}"
    )
    return TokenProjection((mapped,), (), ()), work
