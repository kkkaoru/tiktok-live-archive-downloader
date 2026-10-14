"""Select uncorrected PCM samples using observed native presentation cut ranges."""

import math
import wave
from bisect import bisect_left, bisect_right
from fractions import Fraction
from itertools import pairwise
from pathlib import Path

from replay.audio_clock import AudioFrameClock

BLOCK_FRAMES = 65536


def retained_pcm_spans(
    clock: AudioFrameClock,
    intervals: tuple[tuple[Fraction, Fraction], ...],
    *,
    output_rate: int,
    output_samples: int,
) -> tuple[tuple[int, int], ...]:
    """Return ordered, merged half-open PCM selections, never padding or retiming.

    Native timestamp gaps have no samples; overlapping native frames keep their
    distinct decoded samples. Only the fixed sample-rate conversion is projected.
    Outward sample rounding protects cut edges, with duplicates merged. Output
    duration can differ from the video cut duration. This does not approve sync,
    silence classification, speech protection, or a completed delivery.
    """
    if (
        type(output_rate) is not int
        or not 8000 <= output_rate <= 192000
        or type(output_samples) is not int
        or output_samples <= 0
        or abs(
            Fraction(output_samples)
            - Fraction(clock.sample_ends[-1] * output_rate, clock.sample_rate)
        )
        > 1
        or not 1 <= len(intervals) <= 100_000
        or any(
            not isinstance(start, Fraction)
            or not isinstance(end, Fraction)
            or not 0 <= start < end <= 21600
            for start, end in intervals
        )
        or any(right[0] < left[1] for left, right in pairwise(intervals))
    ):
        raise ValueError("Invalid uncorrected PCM clock or ordered native cut ranges")
    starts = tuple(start for start, _ in intervals)
    ends = tuple(end for _, end in intervals)
    result: list[tuple[int, int]] = []
    consumed = 0
    for pts, samples in zip(clock.pts, clock.samples, strict=True):
        first = pts * clock.time_base
        last = first + Fraction(samples, clock.sample_rate)
        for index in range(bisect_right(ends, first), bisect_left(starts, last)):
            start, end = intervals[index]
            origin = Fraction(consumed, clock.sample_rate)
            low = max(0, math.floor((origin + max(start, first) - first) * output_rate))
            high = min(output_samples, math.ceil((origin + min(end, last) - first) * output_rate))
            if low < high:
                if result and low <= result[-1][1]:
                    result[-1] = (result[-1][0], max(result[-1][1], high))
                else:
                    result.append((low, high))
        consumed += samples
    if not result:
        raise ValueError("Selected native ranges contain no decoded audio samples")
    return tuple(result)


def write_pcm_spans(source: Path, output: Path, spans: tuple[tuple[int, int], ...]) -> int:
    """Copy selected PCM16 frames byte-for-byte; no fades, padding or resampling.

    Partial outputs remain on failure. A caller must bind these selections to
    the intended source, model result and measured clock before using them.
    """
    if source.is_symlink() or not source.is_file():
        raise ValueError("PCM selection needs a regular source")
    if output.exists() or output.is_symlink() or not output.parent.is_dir():
        raise ValueError("PCM selection needs a new output")
    before = source.stat()
    with wave.open(str(source), "rb") as reader:
        if reader.getsampwidth() != 2 or reader.getcomptype() != "NONE":
            raise ValueError("PCM selection supports uncompressed PCM16 only")
        if (
            not 1 <= len(spans) <= 100_000
            or any(
                type(start) is not int
                or type(end) is not int
                or not 0 <= start < end <= reader.getnframes()
                for start, end in spans
            )
            or any(right[0] < left[1] for left, right in pairwise(spans))
        ):
            raise ValueError("Invalid ordered PCM sample selections")
        with output.open("xb") as handle, wave.open(handle, "wb") as writer:
            writer.setparams(reader.getparams())
            for start, end in spans:
                reader.setpos(start)
                remaining = end - start
                while remaining:
                    count = min(remaining, BLOCK_FRAMES)
                    raw = reader.readframes(count)
                    if len(raw) != count * reader.getnchannels() * 2:
                        raise ValueError("Selected source PCM is truncated")
                    writer.writeframesraw(raw)
                    remaining -= count
    after = source.stat()
    if (before.st_ino, before.st_size, before.st_mtime_ns) != (
        after.st_ino,
        after.st_size,
        after.st_mtime_ns,
    ):
        raise ValueError("Source PCM changed during selection")
    return sum(end - start for start, end in spans)
