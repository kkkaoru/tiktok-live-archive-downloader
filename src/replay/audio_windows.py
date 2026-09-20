"""Sample-exact bounded windows for overlapping offline audio processing."""

from dataclasses import dataclass

DEFAULT_MAXIMUM_WINDOWS = 10_000


@dataclass(frozen=True)
class AudioWindow:
    """A positive half-open source sample range, not a speech decision."""

    start_sample: int
    sample_count: int

    def __post_init__(self) -> None:
        if type(self.start_sample) is not int or self.start_sample < 0:
            raise ValueError("Window start must be a nonnegative integer")
        if type(self.sample_count) is not int or self.sample_count <= 0:
            raise ValueError("Window sample count must be a positive integer")

    @property
    def end_sample(self) -> int:
        return self.start_sample + self.sample_count


def plan_audio_windows(
    sample_count: int,
    *,
    window_samples: int,
    stride_samples: int,
    maximum_windows: int = DEFAULT_MAXIMUM_WINDOWS,
) -> tuple[AudioWindow, ...]:
    """Cover every input sample without redundant fully contained tail windows."""
    values = (sample_count, window_samples, stride_samples, maximum_windows)
    if any(type(value) is not int or value <= 0 for value in values):
        raise ValueError("Audio window parameters must be positive integers")
    if stride_samples > window_samples:
        raise ValueError("Stride must not leave uncovered samples")
    remaining = max(0, sample_count - window_samples)
    count = 1 + (remaining + stride_samples - 1) // stride_samples
    if count > maximum_windows:
        raise ValueError("Audio window count exceeds its budget")
    return tuple(
        AudioWindow(start, min(window_samples, sample_count - start))
        for start in range(0, count * stride_samples, stride_samples)
    )


def overlap_samples(left: AudioWindow, right: AudioWindow) -> int:
    """Validate forward coverage and return the actual adjacent overlap."""
    if right.start_sample <= left.start_sample or right.end_sample <= left.end_sample:
        raise ValueError("Audio windows must advance both start and end")
    overlap = left.end_sample - right.start_sample
    if overlap < 0:
        raise ValueError("Audio windows leave uncovered samples")
    return overlap
