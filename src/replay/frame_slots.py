"""Which source frame each constant-rate output slot shows, tolerant of timestamp quantization.

The compose reads every output slot ``j`` at its middle, ``(j + 1/2) / rate`` of source time,
so slot ``j`` shows the latest frame with ``pts <= (j + 1/2) / rate``: frame ``i`` first shows
at ``ceil(x_i - 1/2)`` with ``x_i = pts_i * rate``. Timestamps quantized to a millisecond (FLV)
put two frames into one slot now and then and leave a neighbouring slot empty: one frame is
dropped and its neighbour repeats.

``first_slots`` keeps that slot for every frame and resolves only collisions: the earlier frame
moves back a slot, or else the later one forward, while the frame stays within half a slot
plus the clock's quantum of its own time; a frame that fits nowhere is a real excess and is
dropped. Real gaps are kept (nothing moves further than that bound), and a clock without
collisions keeps today's map exactly.
"""

from collections import Counter
from collections.abc import Sequence
from dataclasses import dataclass
from fractions import Fraction
from functools import reduce
from math import ceil, gcd

# The step values that cover this share of all steps define the clock's quantization.
DOMINANT_SHARE = Fraction(99, 100)
# A corrected reading keeps this many slots from a frame boundary: two ticks of the native
# 1/60000 s timescale at 30 fps, well above its rounding.
MARGIN = Fraction(1, 1000)


def quantum(pts: Sequence[int]) -> int:
    """Timestamp quantization in time-base ticks: the GCD of the dominant steps' deviations.

    Steps outside the values covering ``DOMINANT_SHARE`` of all steps (gaps, a lead-in at
    another rate) are left out, so they cannot shrink the quantum. 0 for a constant step.
    """
    counts = Counter(later - earlier for earlier, later in zip(pts, pts[1:], strict=False))
    ranked = counts.most_common()
    if not ranked:
        return 0
    modal, covered, dominant = ranked[0][0], 0, []
    for step, count in ranked:
        if covered >= DOMINANT_SHARE * sum(counts.values()):
            break
        dominant.append(step)
        covered += count
    return reduce(gcd, (abs(step - modal) for step in dominant), 0)


def first_slots(pts: Sequence[int], time_base: Fraction, rate: int) -> list[int | None]:
    """Each frame's first output slot, or None for a frame no slot can show.

    A forward pass gives each frame the earliest slot within ``1/2 + quantum`` of its time
    after the previous one (the most frames any order-keeping assignment can show); a backward
    pass then moves each one as late as its successor allows, but never past today's
    mid-slot reading, so a clock without collisions keeps today's slots exactly.
    """
    tolerance = Fraction(1, 2) + quantum(pts) * time_base * rate
    times = [value * time_base * rate for value in pts]
    slots: list[int | None] = []
    last: int | None = None
    for time in times:
        slot = ceil(time - tolerance)
        if slot <= time - tolerance:
            slot += 1  # the window is open at both ends
        if last is not None:
            slot = max(slot, last + 1)
        if slot - time < tolerance:
            slots.append(slot)
            last = slot
        else:
            slots.append(None)
    following: int | None = None
    for index in range(len(slots) - 1, -1, -1):
        earliest = slots[index]
        if earliest is None:
            continue
        today = ceil(times[index] - Fraction(1, 2))
        latest = today if following is None else min(today, following - 1)
        slots[index] = max(earliest, latest)
        following = slots[index]
    return slots


@dataclass(frozen=True, slots=True)
class Run:
    """Output slots ``[start, end)`` read at ``(slot + 1/2 + phase) / rate`` of source time."""

    start: int
    end: int
    phase: Fraction


def phase_windows(
    pts: Sequence[int], time_base: Fraction, rate: int
) -> list[tuple[Fraction, Fraction]]:
    """Per output slot, the reading phases ``[low, high)`` (in slots) that show its frame.

    Slot ``j`` read at ``j + 1/2 + phase`` shows its assigned frame ``A`` exactly when the
    reading falls in ``[x_A, x_(A+1))``; a dropped frame still bounds its predecessor.
    """
    slots = first_slots(pts, time_base, rate)
    times = [value * time_base * rate for value in pts]
    shown = [(slot, index) for index, slot in enumerate(slots) if slot is not None]
    windows: list[tuple[Fraction, Fraction]] = []
    cursor, assigned = 0, shown[0][1]
    for slot in range(shown[-1][0] + 1):
        while cursor < len(shown) and shown[cursor][0] <= slot:
            assigned = shown[cursor][1]
            cursor += 1
        middle = slot + Fraction(1, 2)
        following = times[assigned + 1] if assigned + 1 < len(times) else middle + 1
        windows.append((times[assigned] - middle, following - middle))
    return windows


def runs(windows: Sequence[tuple[Fraction, Fraction]], start: int, end: int) -> list[Run]:
    """Fewest runs covering slots ``[start, end)``, each with one phase all its slots accept.

    A run keeps phase 0 (today's mid-slot reading) whenever every slot in it accepts that;
    otherwise it takes the phase of the common window nearest 0, ``MARGIN`` inside its edges.
    Slots past the windows (beyond the last frame) accept any phase.
    """
    result: list[Run] = []
    slot = start
    while slot < end:
        low, high = _window(windows, slot)
        stop = slot + 1
        while stop < end:
            next_low, next_high = _window(windows, stop)
            low_, high_ = max(low, next_low), min(high, next_high)
            if not (low_ <= 0 < high_ or high_ - low_ > 2 * MARGIN):
                break
            low, high, stop = low_, high_, stop + 1
        phase = (
            Fraction(0) if low <= 0 < high else min(max(low + MARGIN, Fraction(0)), high - MARGIN)
        )
        result.append(Run(slot, stop, phase))
        slot = stop
    return result


def _window(windows: Sequence[tuple[Fraction, Fraction]], slot: int) -> tuple[Fraction, Fraction]:
    return windows[slot] if slot < len(windows) else (Fraction(-1, 2), Fraction(1, 2))
