"""Frame-index selection on a constant-rate grid, tolerant only of sub-frame container rounding.

A native clip request is valid when its integer frames fit the media's frame
count. Comparing float seconds against the container duration rejects valid
requests because the last frame's container length can be a few ticks short of
the grid. This module keeps the start and frame count exact, accepts a tail
shortfall of at most half a frame, and still rejects any real over-run.
"""

from __future__ import annotations

from dataclasses import dataclass
from fractions import Fraction

from replay.core import ReplayError


@dataclass(frozen=True, slots=True)
class FrameSelection:
    start: int
    frames: int

    @property
    def end(self) -> int:
        return self.start + self.frames


def tail_shortfall_ticks(media_frames: int, media_ticks: int, rate: int, timebase: int) -> int:
    """Ticks by which the container is shorter than `media_frames` grid frames."""
    return media_frames * timebase // rate - media_ticks


def select(
    *,
    rate: int,
    media_frames: int,
    media_ticks: int,
    timebase: int,
    start_seconds: Fraction | float,
    duration_seconds: Fraction | float,
) -> FrameSelection:
    if min(rate, media_frames, media_ticks, timebase) <= 0:
        raise ReplayError("Frame grid inputs must be positive")
    start = round(Fraction(start_seconds) * rate)
    frames = round(Fraction(duration_seconds) * rate)
    if start < 0 or frames <= 0:
        raise ReplayError("Frame selection must start at or after zero and cover a frame")
    shortfall = tail_shortfall_ticks(media_frames, media_ticks, rate, timebase)
    if shortfall < 0 or 2 * shortfall > timebase // rate:
        raise ReplayError(f"Media length differs from the frame grid by {shortfall} ticks")
    if start + frames > media_frames:
        raise ReplayError(
            f"Frame selection {start}+{frames} exceeds the {media_frames} available frames"
        )
    return FrameSelection(start, frames)
