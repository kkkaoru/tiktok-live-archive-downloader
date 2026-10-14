"""Map exact output intervals to retained source support without assuming source FPS."""

from dataclasses import dataclass
from fractions import Fraction
from pathlib import Path

from replay.foreground_evidence import ForegroundWindow
from replay.foreground_geometry import interval_top
from replay.subtitle_clock import MAX_CLOCK_SECONDS, MAX_CLOCK_TIMESCALE

MAX_WINDOWS = 32
MAX_CUTS = 3000
MAX_OUTPUT_SECONDS = 120


@dataclass(frozen=True, slots=True)
class RetainedSourceSpan:
    """An independently verified, unit-speed time translation; never frame ordinals."""

    source_start: Fraction
    output_start: Fraction
    duration: Fraction

    def __post_init__(self) -> None:
        values = (self.source_start, self.output_start, self.duration)
        if (
            any(not isinstance(value, Fraction) for value in values)
            or any(value < 0 or value.denominator > MAX_CLOCK_TIMESCALE for value in values)
            or not 0 < self.duration <= MAX_OUTPUT_SECONDS
            or max(self.source_start, self.output_start) + self.duration > MAX_CLOCK_SECONDS
        ):
            raise ValueError("Retained source spans require bounded exact rational clocks")

    @property
    def source_end(self) -> Fraction:
        return self.source_start + self.duration

    @property
    def output_end(self) -> Fraction:
        return self.output_start + self.duration


def _extent(window: ForegroundWindow) -> tuple[Fraction, Fraction]:
    clock = window.job.clock
    return clock.pts[0] * clock.time_base, (clock.pts[-1] + clock.last_duration) * clock.time_base


def _identity(window: ForegroundWindow) -> tuple[Path, str, Path, str, int, int]:
    job = window.job
    return job.source, job.source_hash, job.native, job.native_hash, job.width, job.height


@dataclass(frozen=True, slots=True)
class OutputSourceSupport:
    """A callable for measured_plan, using independently pinned read_window results.

    Windows must be source-ordered/disjoint and share source, native producer and
    canvas identities. Gaps remain unknown. Cuts are ordered, unit-speed and
    contiguous in output time; removed source intervals are never queried.
    This verifies arithmetic/coverage only, not that a media editor actually used
    these cuts. Bind that correspondence independently, and reverify receipts
    before publishing. No pixel transform, source FPS or semantic approval is inferred.
    """

    windows: tuple[ForegroundWindow, ...]
    cuts: tuple[RetainedSourceSpan, ...]

    def __post_init__(self) -> None:
        if (
            not isinstance(self.windows, tuple)
            or not 1 <= len(self.windows) <= MAX_WINDOWS
            or not isinstance(self.cuts, tuple)
            or not 1 <= len(self.cuts) <= MAX_CUTS
        ):
            raise ValueError("Bounded foreground windows and retained cuts are required")
        previous_end = Fraction(0)
        identity = _identity(self.windows[0])
        for window in self.windows:
            start, end = _extent(window)
            if _identity(window) != identity or start < previous_end:
                raise ValueError("Foreground windows overlap or have conflicting identities")
            previous_end = end
        source_end = Fraction(0)
        output_end = self.cuts[0].output_start
        for cut in self.cuts:
            if cut.source_start < source_end or cut.output_start != output_end:
                raise ValueError("Retained cuts must be source-ordered and output-contiguous")
            source_end, output_end = cut.source_end, cut.output_end
        if output_end - self.cuts[0].output_start > MAX_OUTPUT_SECONDS:
            raise ValueError("Output support window exceeds its duration budget")

    def _source_top(self, start: Fraction, end: Fraction) -> float:
        cursor = start
        tops: list[float] = []
        for window in self.windows:
            first, last = _extent(window)
            if last <= cursor:
                continue
            if first > cursor:
                break
            stop = min(end, last)
            tops.append(interval_top(window.frames, clock=window.job.clock, start=cursor, end=stop))
            cursor = stop
            if cursor == end:
                return min(tops)
        raise ValueError("Required source interval lacks complete foreground clock coverage")

    def __call__(self, start: Fraction, end: Fraction) -> float:
        if (
            not isinstance(start, Fraction)
            or not isinstance(end, Fraction)
            or max(start.denominator, end.denominator) > MAX_CLOCK_TIMESCALE
            or not self.cuts[0].output_start <= start < end <= self.cuts[-1].output_end
        ):
            raise ValueError("Requested output interval is outside the retained clock")
        tops: list[float] = []
        for cut in self.cuts:
            left, right = max(start, cut.output_start), min(end, cut.output_end)
            if left < right:
                offset = cut.source_start - cut.output_start
                tops.append(self._source_top(left + offset, right + offset))
        return min(tops)
