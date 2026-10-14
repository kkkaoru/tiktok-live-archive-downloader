"""Partition verified full-source PTS without inventing clocks or widening budgets."""

from bisect import bisect_right
from fractions import Fraction
from itertools import pairwise

from replay.head_observations import MAX_TRACKING_FRAMES, MAX_TRACKING_SECONDS
from replay.subtitle_clock import VideoFrameClock


def tracking_windows(
    pts: tuple[int, ...],
    *,
    time_base: Fraction,
    last_duration: int,
    seed_pts: tuple[int, ...] | None = None,
    maximum_seconds: Fraction = Fraction(MAX_TRACKING_SECONDS),
) -> tuple[VideoFrameClock, ...]:
    """Use measured next-frame boundaries, and the independently verified EOF extent.

    The caller binds the complete source clock and terminal duration to its media.
    A long source is not one oversized VideoFrameClock: every returned clock keeps
    that type's existing limits and the stricter native tracking budgets. Splitting
    neither proves tracking continuity across windows nor approves caption geometry.

    Explicit seed_pts must be exact observed face-frame ticks, not rounded sample
    requests. Seeded windows overlap exactly one complete seed frame. Each start
    still needs successful native detection; candidates do not prove tracking.
    """
    if (
        not isinstance(pts, tuple)
        or not pts
        or any(type(point) is not int or point < 0 for point in pts)
        or any(right <= left for left, right in pairwise(pts))
    ):
        raise ValueError("Source PTS must be nonnegative, integer and strictly increasing")
    # Validate the time base and terminal extent with the existing clock contract.
    VideoFrameClock((pts[-1],), time_base, last_duration)
    if not isinstance(maximum_seconds, Fraction) or not 0 < maximum_seconds <= MAX_TRACKING_SECONDS:
        raise ValueError("Invalid tracking duration budget")
    seeds = _seed_indices(pts, seed_pts) if seed_pts is not None else None
    boundaries = (*pts, pts[-1] + last_duration)
    windows: list[VideoFrameClock] = []
    start = 0
    while start < len(pts):
        latest_boundary = pts[start] + maximum_seconds / time_base
        end = min(start + MAX_TRACKING_FRAMES, bisect_right(boundaries, latest_boundary) - 1)
        if end <= start:
            raise ValueError("A measured frame extent exceeds the tracking duration budget")
        next_start = end
        if seeds is not None and end < len(pts):
            next_start = seeds[bisect_right(seeds, end - 1) - 1]
            if next_start <= start:
                raise ValueError("No advancing face seed within the tracking budget")
            end = next_start + 1
        windows.append(VideoFrameClock(pts[start:end], time_base, boundaries[end] - pts[end - 1]))
        start = next_start
    return tuple(windows)


def _seed_indices(pts: tuple[int, ...], seeds: tuple[int, ...]) -> tuple[int, ...]:
    if (
        not isinstance(seeds, tuple)
        or not seeds
        or any(type(seed) is not int for seed in seeds)
        or any(right <= left for left, right in pairwise(seeds))
        or seeds[0] != pts[0]
    ):
        raise ValueError("Face seeds must be ordered integer PTS including the first frame")
    indices = tuple(bisect_right(pts, seed) - 1 for seed in seeds)
    if any(index < 0 or pts[index] != seed for index, seed in zip(indices, seeds, strict=True)):
        raise ValueError("Face seeds must match decoded source PTS exactly")
    return indices
