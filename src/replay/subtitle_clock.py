"""Bounded presentation clocks for native FFmpeg VFR repair windows."""

from bisect import bisect_left
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from fractions import Fraction
from itertools import pairwise
from math import isfinite
from tempfile import NamedTemporaryFile

MAX_CLOCK_FRAMES = 50_000
MAX_CLOCK_SECONDS = 86_400
MAX_CLOCK_TIMESCALE = 1_000_000_000
CLOCK_TOLERANCE_SECONDS = 0.000001
FRAME_SAMPLE_FRACTION = Fraction(1, 4)


@dataclass(frozen=True, slots=True)
class VideoFrameClock:
    """Verified consecutive source PTS, not an average-fps approximation.

    The caller binds these clocks to the unchanged, zero-origin source. PTS are
    absolute source ticks. last_duration is the terminal presentation extent:
    use the next source PTS at a split. At EOF, verify the terminal decoded
    duration against complete packet support, not a contradictory container header.
    Masks use local decoded-frame indices. Output PTS are rebased to zero.
    """

    pts: tuple[int, ...]
    time_base: Fraction
    last_duration: int

    def __post_init__(self) -> None:
        if (
            not isinstance(self.pts, tuple)
            or not 1 <= len(self.pts) <= MAX_CLOCK_FRAMES
            or any(type(value) is not int or value < 0 for value in self.pts)
            or any(right <= left for left, right in pairwise(self.pts))
            or type(self.last_duration) is not int
            or self.last_duration <= 0
        ):
            raise ValueError("Invalid presentation frame clocks")
        if (
            not isinstance(self.time_base, Fraction)
            or self.time_base <= 0
            or self.time_base.denominator > MAX_CLOCK_TIMESCALE
            or (self.pts[-1] + self.last_duration) * self.time_base > MAX_CLOCK_SECONDS
        ):
            raise ValueError("Invalid presentation time base or extent")

    def verify_terminal_packet_support(
        self, *, packet_pts: tuple[int, ...], packet_durations: tuple[int, ...]
    ) -> None:
        """Verify a complete decoded tail against its original encoded packets.

        Packet order may differ from presentation order. Require one packet per
        decoded frame and the exact final presentation duration; never trim a
        frame to fit a shorter declared stream duration. The caller must bind
        both observations to the same unchanged source and complete EOF tail.
        This establishes input presentation support, not repaired-output clocks.
        """
        if (
            not isinstance(packet_pts, tuple)
            or not isinstance(packet_durations, tuple)
            or len(packet_pts) != len(self.pts)
            or len(packet_durations) != len(packet_pts)
            or any(type(value) is not int or value < 0 for value in packet_pts)
            or any(type(value) is not int or value <= 0 for value in packet_durations)
            or tuple(sorted(packet_pts)) != self.pts
        ):
            raise ValueError("Encoded packets do not match the complete decoded tail")
        terminal = packet_pts.index(self.pts[-1])
        if packet_durations[terminal] != self.last_duration:
            raise ValueError("Terminal decoded duration differs from its encoded packet")

    def observation_times(self) -> tuple[float, ...]:
        """Local interior requests; never subtract large floating source clocks."""
        following = (*self.pts[1:], self.pts[-1] + self.last_duration)
        return tuple(
            float((value - self.pts[0] + (end - value) * FRAME_SAMPLE_FRACTION) * self.time_base)
            for value, end in zip(self.pts, following, strict=True)
        )

    def observed_indices(self, actual_times: tuple[float, ...]) -> tuple[int, ...]:
        """Verify distinct actual frames on this clock, rejecting ambiguous matches."""
        if not actual_times or len(actual_times) > MAX_CLOCK_FRAMES:
            raise ValueError("Invalid actual frame observation count")
        times = tuple(float((value - self.pts[0]) * self.time_base) for value in self.pts)
        end = float((self.pts[-1] - self.pts[0] + self.last_duration) * self.time_base)
        indices: set[int] = set()
        for time in actual_times:
            if isinstance(time, bool) or not isfinite(time) or not 0 <= time < end:
                raise ValueError("Invalid actual frame clock")
            index = bisect_left(times, time - CLOCK_TOLERANCE_SECONDS)
            if index == len(times) or abs(times[index] - time) > CLOCK_TOLERANCE_SECONDS:
                raise ValueError("Actual frame clock is outside the verified presentation grid")
            if index + 1 < len(times) and abs(times[index + 1] - time) <= CLOCK_TOLERANCE_SECONDS:
                raise ValueError("Ambiguous actual frame clock")
            indices.add(index)
        return tuple(sorted(indices))

    @contextmanager
    def encoding_options(self) -> Iterator[tuple[str, ...]]:
        """Keep a private filter alive through encoding; never replace media.

        A balanced lookup is logarithmic per frame. A filter file avoids command
        argument limits. No output -r: it contradicts passthrough timing.
        """
        numerator, denominator = self.time_base.numerator, self.time_base.denominator
        local_pts = tuple((value - self.pts[0]) * numerator for value in self.pts)
        expression = _timestamp_expression(local_pts).replace(",", r"\,")
        with NamedTemporaryFile(mode="w", encoding="utf-8", suffix=".filter") as handle:
            handle.write(f"settb=1/{denominator},setpts={expression}")
            handle.flush()
            yield (
                "-/filter:v",
                handle.name,
                "-fps_mode",
                "passthrough",
                "-enc_time_base",
                f"1:{denominator}",
                "-video_track_timescale",
                str(denominator),
                "-bsf:v",
                "setts=duration='if(eq(N,"
                f"{len(self.pts) - 1}),{self.last_duration * numerator},DURATION)'",
            )


def concat_durations(durations: tuple[Fraction, ...], *, time_base: Fraction) -> tuple[str, ...]:
    """FFconcat microsecond durations without accumulating per-part rounding.

    Callers verify each encoded part and use the actual common mux time base.
    This only preserves join offsets: independently verify every joined PTS,
    terminal extent, and copied audio/subtitle packet after muxing.
    """
    if (
        not isinstance(durations, tuple)
        or not 1 <= len(durations) <= MAX_CLOCK_FRAMES
        or not isinstance(time_base, Fraction)
        or time_base <= 0
        or time_base.denominator > MAX_CLOCK_TIMESCALE
    ):
        raise ValueError("Invalid concat clock budget")
    total = Fraction(0)
    previous_us = 0
    result: list[str] = []
    for duration in durations:
        if (
            not isinstance(duration, Fraction)
            or duration <= 0
            or (duration / time_base).denominator != 1
        ):
            raise ValueError("Concat duration is outside the mux clock grid")
        total += duration
        end_us = round(total * 1_000_000)
        delta_us = end_us - previous_us
        if (
            total > MAX_CLOCK_SECONDS
            or delta_us <= 0
            or abs(Fraction(end_us, 1_000_000) - total) >= time_base / 2
        ):
            raise ValueError("Concat microsecond offsets cannot preserve the mux clock")
        result.append(f"{delta_us // 1_000_000}.{delta_us % 1_000_000:06d}")
        previous_us = end_us
    return tuple(result)


def _timestamp_expression(pts: tuple[int, ...], *, start: int = 0) -> str:
    if len(pts) == 1:
        return str(pts[0])
    middle = len(pts) // 2
    left = _timestamp_expression(pts[:middle], start=start)
    right = _timestamp_expression(pts[middle:], start=start + middle)
    return f"if(lt(N,{start + middle}),{left},{right})"
