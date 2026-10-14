"""Exact interval support and measured rectangle placement, not semantic approval.

Callers independently bind source clocks, foreground receipts and the complete
origin-relative Motion envelope (text, style, canvas, template and every animation
state). These pure operations neither establish those bindings nor identify heads.
Legacy head sampling and its missing-data fallbacks are deliberately not used.
"""

import math
from bisect import bisect_left, bisect_right
from fractions import Fraction

from replay.core import ReplayError
from replay.foreground_observations import MAX_DIMENSION, ForegroundFrame
from replay.subtitle_clock import VideoFrameClock
from replay.vertical_captions import Box


def interval_top(
    frames: tuple[ForegroundFrame, ...],
    *,
    clock: VideoFrameClock,
    start: Fraction,
    end: Fraction,
) -> float:
    """Minimum observed support top across a half-open SOURCE interval.

    The supplied clock must independently describe consecutive source frames.
    Include the frame containing a mid-frame start, exclude a frame starting at
    end, and use the verified terminal duration. Absent/empty support is unknown,
    not permission to place a caption. Resolve retained cut intervals separately;
    never pass an interval spanning a removed portion as an output-time interval.
    """
    times = tuple(tick * clock.time_base for tick in clock.pts)
    terminal = (clock.pts[-1] + clock.last_duration) * clock.time_base
    if (
        not isinstance(start, Fraction)
        or not isinstance(end, Fraction)
        or not times[0] <= start < end <= terminal
    ):
        raise ReplayError("Foreground interval is outside the exact source clock")
    if (
        not isinstance(frames, tuple)
        or any(not isinstance(frame.time, Fraction) for frame in frames)
        or tuple(frame.time for frame in frames) != times
    ):
        raise ReplayError("Foreground records do not cover the complete supplied clock")
    first, last = bisect_right(times, start) - 1, bisect_left(times, end)
    tops = tuple(_support_top(frame) for frame in frames[first:last])
    return min(tops)


def _support_top(frame: ForegroundFrame) -> float:
    mask = frame.mask
    if not frame.labels or mask is None or mask.nonzero_pixels <= 0 or mask.bounds is None:
        raise ReplayError("Required foreground support is unknown")
    _check_box(mask.bounds, relative=False)
    return mask.bounds.top


def _finite(value: float) -> bool:
    return not isinstance(value, bool) and math.isfinite(value)


def _check_box(box: Box, *, relative: bool) -> None:
    values = (box.left, box.top, box.right, box.bottom)
    minimum = -MAX_DIMENSION if relative else 0
    if (
        any(not _finite(value) or not minimum <= value <= MAX_DIMENSION for value in values)
        or box.left >= box.right
        or box.top >= box.bottom
    ):
        raise ReplayError("Invalid measured caption or foreground rectangle")


def place_envelope(
    envelope: Box,
    *,
    region: Box,
    center_x: float,
    foreground_top: float,
    clearance: float,
    previous_y: float | None = None,
) -> tuple[float, float]:
    """Return a fixed-X origin whose COMPLETE envelope fits above supplied support.

    Envelope coordinates are relative to the calibrated Motion origin, including
    borders, shadows and animation extrema. Region is an explicit canvas rectangle,
    not the legacy gauge-derived limit. Retain a previous Y only if it still fits;
    otherwise select the closest feasible Y (or the lowest feasible first origin).
    An impossible rectangle raises, never shrinks, clips or declares approval.
    Model support is not a guarantee of whole-subject coverage or head identity.
    """
    _check_box(envelope, relative=True)
    _check_box(region, relative=False)
    if (
        not _finite(center_x)
        or not _finite(foreground_top)
        or not 0 <= foreground_top <= MAX_DIMENSION
        or not _finite(clearance)
        or not 0 <= clearance <= MAX_DIMENSION
        or (previous_y is not None and not _finite(previous_y))
    ):
        raise ReplayError("Invalid measured placement constraints")
    if center_x + envelope.left < region.left or center_x + envelope.right > region.right:
        raise ReplayError("Measured caption cannot fit at the fixed horizontal origin")
    lowest = region.top - envelope.top
    highest = min(region.bottom, foreground_top - clearance) - envelope.bottom
    if lowest > highest:
        raise ReplayError("Measured caption envelope cannot fit above foreground support")
    y = highest if previous_y is None else min(max(previous_y, lowest), highest)
    return center_x, y
