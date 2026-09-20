"""Independent quiet evidence from sample-indexed stereo FFmpeg peak measurements."""

import re
from collections.abc import Iterable
from math import ceil, isfinite, isnan

from replay.caption_projection import MeasuredQuietSpan

HEADER = re.compile(r"frame:(\d+)\s+pts:(\d+)\s+pts_time:[0-9.e+\-]+")
PEAK_PREFIX = "lavfi.astats.Overall.Peak_level="


def measured_quiet_runs(
    lines: Iterable[str],
    *,
    sample_count: int,
    sample_rate: int = 44100,
    window_samples: int = 882,
    threshold_dbfs: float = -60,
    minimum_seconds: float = 0.02,
) -> tuple[MeasuredQuietSpan, ...]:
    """Require every frame and its exact PTS; never use rounded pts_time or ASR.

    The caller must measure the maximum absolute sample across both channels,
    without downmixing (opposite-phase speech could otherwise cancel). Negative
    infinity denotes measured digital zero. A short partial final window is not
    promoted to the minimum quiet duration or padded with invented silence.
    """
    if (
        type(sample_count) is not int
        or sample_count <= 0
        or type(sample_rate) is not int
        or not 8000 <= sample_rate <= 192000
        or type(window_samples) is not int
        or not 1 <= window_samples <= sample_rate
        or isinstance(threshold_dbfs, bool)
        or not isfinite(threshold_dbfs)
        or not -160 <= threshold_dbfs <= 0
        or isinstance(minimum_seconds, bool)
        or not isfinite(minimum_seconds)
        or not 0.005 <= minimum_seconds <= 1
    ):
        raise ValueError("Invalid waveform measurement policy")
    count = (sample_count + window_samples - 1) // window_samples
    if count > 1_100_000:
        raise ValueError("Waveform measurement count exceeds budget")
    minimum_samples = ceil(minimum_seconds * sample_rate)
    quiet: list[MeasuredQuietSpan] = []
    iterator = iter(lines)
    observed = 0
    for header in iterator:
        match = HEADER.fullmatch(header.strip()) if len(header) <= 160 else None
        if match is None or int(match[1]) != observed or int(match[2]) != observed * window_samples:
            raise ValueError("Waveform frame order or exact sample clock is invalid")
        if observed >= count:
            raise ValueError("Unexpected extra waveform frame")
        peak_line = next(iterator, "").strip()
        if len(peak_line) > 160 or not peak_line.startswith(PEAK_PREFIX):
            raise ValueError("Missing waveform peak measurement")
        raw_peak = peak_line[len(PEAK_PREFIX) :]
        peak = float(raw_peak)
        if isnan(peak) or (not isfinite(peak) and raw_peak != "-inf"):
            raise ValueError("Invalid waveform peak")
        start = observed * window_samples
        end = min(sample_count, start + window_samples)
        if peak < threshold_dbfs and end - start >= minimum_samples:
            beginning = start / sample_rate
            if quiet and quiet[-1].end == beginning:
                beginning = quiet.pop().start
            quiet.append(MeasuredQuietSpan(beginning, end / sample_rate))
        observed += 1
    if observed != count:
        raise ValueError("Waveform evidence is incomplete")
    return tuple(quiet)
