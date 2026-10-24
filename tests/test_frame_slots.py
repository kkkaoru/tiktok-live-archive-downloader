"""Output slots keep every source frame through millisecond timestamp jitter, and only then."""

from bisect import bisect_right
from fractions import Fraction
from math import ceil

from replay import frame_slots

MS = Fraction(1, 1000)


def shown(pts: list[int], time_base: Fraction) -> list[int]:
    """The frame each slot shows when every run reads at its phase (the compose's reading)."""
    times = [value * time_base * 30 for value in pts]
    windows = frame_slots.phase_windows(pts, time_base, 30)
    return [
        bisect_right(times, slot + Fraction(1, 2) + run.phase) - 1
        for run in frame_slots.runs(windows, 0, len(windows))
        for slot in range(run.start, run.end)
    ]


def test_the_quantum_comes_from_the_dominant_steps_only() -> None:
    lead_in = [3000 * index for index in range(10)]
    body = [lead_in[-1] + 3015 * index + (45 if index % 2 else 0) for index in range(1, 2001)]
    gap = [body[-1] + 540000]
    assert frame_slots.quantum(lead_in) == 0
    assert frame_slots.quantum(lead_in + body + gap) == 90
    assert frame_slots.quantum([0]) == 0


def test_millisecond_rounded_30_fps_keeps_todays_map() -> None:
    pts = [round(Fraction(1000 * index, 30)) for index in range(90)]
    assert frame_slots.first_slots(pts, MS, 30) == [
        ceil(value * 30 * MS - Fraction(1, 2)) for value in pts
    ]
    assert frame_slots.first_slots(pts, MS, 30) == list(range(90))
    windows = frame_slots.phase_windows(pts, MS, 30)
    assert frame_slots.runs(windows, 0, 90) == [frame_slots.Run(0, 90, Fraction(0))]


def test_flv_steps_near_slot_middles_drop_nothing() -> None:
    # 33.5 ms average (29.85 fps) from 33/34 ms steps, starting half a slot in: today's mid-slot
    # readings fall between frames that land in one slot and drop one of each pair.
    pts = [17]
    for index in range(299):
        pts.append(pts[-1] + (33 if index % 2 else 34))
    today = [ceil(value * 30 * MS - Fraction(1, 2)) for value in pts]
    assert len(set(today)) == 299
    slots = frame_slots.first_slots(pts, MS, 30)
    assert None not in slots
    kept = [slot for slot in slots if slot is not None]
    assert kept == sorted(set(kept))
    frames = shown(pts, MS)
    assert sorted(set(frames)) == list(range(300))
    assert len(frames) - len(set(frames)) == len(frames) - 300


def test_a_real_gap_repeats_the_frame_before_it() -> None:
    pts = [round(Fraction(1000 * index, 30)) for index in range(30)]
    pts += [6000 + round(Fraction(1000 * index, 30)) for index in range(30)]
    slots = frame_slots.first_slots(pts, MS, 30)
    assert slots == list(range(30)) + list(range(180, 210))
    frames = shown(pts, MS)
    assert frames[29:181] == [29] * 151 + [30]


def test_more_frames_than_slots_drop_the_excess() -> None:
    pts = [round(Fraction(1000 * index, 60)) for index in range(60)]
    slots = frame_slots.first_slots(pts, MS, 30)
    assert slots.count(None) == 29
    assert sorted(set(shown(pts, MS))) == [
        index for index, slot in enumerate(slots) if slot is not None
    ]


def test_runs_keep_phase_zero_or_take_the_nearest_shared_phase() -> None:
    windows = [
        (Fraction(-1, 4), Fraction(1, 4)),
        (Fraction(1, 10), Fraction(3, 5)),
        (Fraction(1, 5), Fraction(2, 5)),
        (Fraction(-3, 5), Fraction(-2, 5)),
    ]
    assert frame_slots.runs(windows, 0, 6) == [
        frame_slots.Run(0, 3, Fraction(1, 5) + frame_slots.MARGIN),
        frame_slots.Run(3, 6, Fraction(-2, 5) - frame_slots.MARGIN),
    ]
    assert frame_slots.runs(windows, 0, 1) == [frame_slots.Run(0, 1, Fraction(0))]
    assert frame_slots.runs(windows, 4, 6) == [frame_slots.Run(4, 6, Fraction(0))]
