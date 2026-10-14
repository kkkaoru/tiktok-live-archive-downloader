"""Map continuous decoded-audio sample clocks back to native presentation times."""

from bisect import bisect_left, bisect_right
from dataclasses import dataclass, field, replace
from fractions import Fraction
from itertools import accumulate, pairwise
from math import isfinite

from replay.caption_io import MLXWordGroup
from replay.recordings import record

MAX_AUDIO_FRAMES = 1_000_000


@dataclass(frozen=True, slots=True)
class AudioFrameClock:
    """Decoded frame sample counts, not packet durations or an inferred constant shift.

    Extraction without asynchronous resampling concatenates decoded samples. Native
    frames may contain timestamp gaps or overlap. Boundary endpoints select the
    corresponding side of the boundary; no samples, silence or word times are made up.
    Caller must bind the probe and decoded recognition WAV to the unchanged source.
    """

    pts: tuple[int, ...]
    samples: tuple[int, ...]
    sample_rate: int
    time_base: Fraction
    sample_ends: tuple[int, ...] = field(init=False)

    def __post_init__(self) -> None:
        if (
            not 1 <= len(self.pts) <= MAX_AUDIO_FRAMES
            or len(self.pts) != len(self.samples)
            or type(self.sample_rate) is not int
            or not 8000 <= self.sample_rate <= 192000
            or not isinstance(self.time_base, Fraction)
            or not 0 < self.time_base <= 1
            or any(type(value) is not int or value < 0 for value in self.pts)
            or any(type(value) is not int or not 1 <= value <= 65536 for value in self.samples)
            or any(right <= left for left, right in pairwise(self.pts))
        ):
            raise ValueError("Invalid native decoded-audio clock")
        object.__setattr__(self, "sample_ends", tuple(accumulate(self.samples)))

    @classmethod
    def from_probe(cls, value: object) -> "AudioFrameClock":
        data = record(value)
        streams, frames = data.get("streams"), data.get("frames")
        if not isinstance(streams, list) or len(streams) != 1:
            raise ValueError("Expected exactly one probed audio stream")
        if not isinstance(frames, list) or not 1 <= len(frames) <= MAX_AUDIO_FRAMES:
            raise ValueError("Expected bounded decoded audio frames")
        stream = record(streams[0])
        rate, base = stream.get("sample_rate"), stream.get("time_base")
        if not isinstance(rate, str) or not rate.isascii() or not rate.isdecimal():
            raise ValueError("Missing native audio sample rate")
        if not isinstance(base, str) or len(base) > 64:
            raise ValueError("Missing rational native audio time base")
        pts: list[int] = []
        samples: list[int] = []
        for item in frames:
            frame = record(item)
            point, count = frame.get("pts"), frame.get("nb_samples")
            if type(point) is not int or type(count) is not int:
                raise ValueError("Missing integral decoded frame clock or sample count")
            pts.append(point)
            samples.append(count)
        return cls(tuple(pts), tuple(samples), int(rate), Fraction(base))

    def source_time(self, seconds: float, *, end: bool = False) -> Fraction:
        """Preserve the native source coordinate of a continuous PCM timestamp."""
        if not isfinite(seconds):
            raise ValueError("Recognition clock must be finite")
        position = Fraction(str(seconds)) * self.sample_rate
        if not 0 <= position <= self.sample_ends[-1]:
            raise ValueError("Recognition clock exceeds decoded sample support")
        search = bisect_left if end else bisect_right
        index = min(search(self.sample_ends, position), len(self.pts) - 1)
        prior = self.sample_ends[index - 1] if index else 0
        return self.pts[index] * self.time_base + (position - prior) / self.sample_rate

    def source_interval(self, start: float, end: float) -> tuple[Fraction, Fraction]:
        if not start < end:
            raise ValueError("Recognition interval must have positive support")
        first, last = self.source_time(start), self.source_time(end, end=True)
        if last <= first:
            raise ValueError("Native frame overlap invalidates the word interval")
        return first, last

    def project_groups(self, groups: tuple[MLXWordGroup, ...]) -> tuple[MLXWordGroup, ...]:
        """Retain ASR provenance and unresolved points; never fabricate a word span."""
        result: list[MLXWordGroup] = []
        for group in groups:
            if group.start == group.end:
                start = end = self.source_time(group.start)
            else:
                start, end = self.source_interval(group.start, group.end)
            result.append(replace(group, start=float(start), end=float(end)))
        if any(right.start < left.start for left, right in pairwise(result)):
            raise ValueError("Native presentation clock reverses word onsets")
        return tuple(result)
