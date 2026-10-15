"""One fixed vertical origin per caption run, from the person's worst-case top edge.

A run is a chain of cues where each cue extends the previous text and starts within one and a
half frames of the previous end, so the same words stay on screen while they grow. The origin
(paragraph bottom, fixed centre X) is chosen once for the whole run and is never moved while the
run is displayed. Unlike face-following placement, a run that does not fit above the person is a
hard error: nothing shrinks, clips or silently moves.
"""

from collections.abc import Callable, Sequence
from dataclasses import dataclass
from fractions import Fraction
from math import isfinite

from replay.core import ReplayError
from replay.fcp_tracking_caption import TrackingPosition
from replay.foreground_geometry import place_envelope
from replay.vertical_captions import Box

EXTENSION_TOLERANCE_FRAMES = Fraction(3, 2)
MAX_CUES = 1_000_000

Envelope = Callable[[str], Box]
PersonTop = Callable[[Fraction, Fraction], float]


@dataclass(frozen=True, slots=True)
class RunCue:
    """A page of text and its half-open display interval in seconds."""

    text: str
    start: Fraction
    end: Fraction


@dataclass(frozen=True, slots=True)
class FixedCue:
    """A cue with its run index and the single origin shared by the whole run."""

    cue: RunCue
    run: int
    x: float
    y: float


def _words(text: str) -> str:
    return text.replace("\n", "")


def extends(previous: RunCue, following: RunCue, *, rate: int) -> bool:
    """True when `following` grows `previous` without a visible break between them."""
    before, after = _words(previous.text), _words(following.text)
    gap = abs(following.start - previous.end)
    return (
        len(after) > len(before)
        and after.startswith(before)
        and gap <= EXTENSION_TOLERANCE_FRAMES / rate
    )


def split_runs(cues: Sequence[RunCue], *, rate: int) -> tuple[tuple[int, ...], ...]:
    """Group consecutive cue indices into runs; every cue belongs to exactly one run."""
    if not 1 <= rate <= 240 or not 0 < len(cues) <= MAX_CUES:
        raise ReplayError("Invalid caption run input")
    runs: list[list[int]] = [[0]]
    for index in range(1, len(cues)):
        previous, following = cues[index - 1], cues[index]
        if following.start < previous.start or following.end <= following.start:
            raise ReplayError("Caption cues must be ordered with positive duration")
        if extends(previous, following, rate=rate):
            runs[-1].append(index)
        else:
            runs.append([index])
    return tuple(tuple(run) for run in runs)


def worst_person_top(series: Sequence[TrackingPosition], start: Fraction, end: Fraction) -> float:
    """Smallest person-top Y over the samples bracketing [start, end].

    Observations bracketing the interval are included because the person may move between samples.
    """
    if not series or start >= end or series[0].time > start or series[-1].time < end:
        raise ReplayError("Person-top series does not cover the caption interval")
    first = max(i for i, point in enumerate(series) if point.time <= start)
    last = min(i for i, point in enumerate(series) if point.time >= end)
    window = series[first : last + 1]
    top = min(point.y for point in window)
    if not isfinite(top):
        raise ReplayError("Non-finite person top")
    return top


@dataclass(frozen=True, slots=True)
class PersonTopEstimate:
    """The worst-case person top for a run and whether it was observed or held.

    `held` means no observation fell inside the run and the nearest observations before and after
    were used; the caller records such runs as derived, never as observations.
    """

    top: float
    held: bool


def person_top_with_hold(
    series: Sequence[TrackingPosition],
    start: Fraction,
    end: Fraction,
    *,
    maximum_drift: float,
) -> PersonTopEstimate:
    """Observed worst-case top inside [start, end]; otherwise hold the nearest observations.

    `series` holds only observed samples. When none falls inside the interval, the nearest observed
    sample before and after are used and the estimate is marked held. If they differ by more than
    `maximum_drift` pixels, or either side is missing, the run fails rather than guessing.
    """
    if maximum_drift < 0 or not isfinite(maximum_drift) or start >= end:
        raise ReplayError("Invalid person-top hold request")
    inside = [point for point in series if start <= point.time <= end]
    if inside:
        return PersonTopEstimate(min(point.y for point in inside), False)
    before = [point for point in series if point.time < start]
    after = [point for point in series if point.time > end]
    if not before or not after:
        raise ReplayError("No observed person on one side of an unobserved caption run")
    near_before, near_after = before[-1], after[0]
    if abs(near_before.y - near_after.y) > maximum_drift:
        raise ReplayError("Observed person top differs too much across an unobserved caption run")
    return PersonTopEstimate(min(near_before.y, near_after.y), True)


def fixed_origins(
    cues: Sequence[RunCue],
    *,
    rate: int,
    region: Box,
    center_x: float,
    clearance: float,
    envelope: Envelope,
    person_top: PersonTop,
) -> tuple[FixedCue, ...]:
    """Choose one origin per run so every cue of the run fits above the worst-case person top."""
    result: list[FixedCue] = []
    for number, run in enumerate(split_runs(cues, rate=rate)):
        start, end = cues[run[0]].start, cues[run[-1]].end
        top = person_top(start, end)
        feasible = [
            place_envelope(
                envelope(cues[i].text),
                region=region,
                center_x=center_x,
                foreground_top=top,
                clearance=clearance,
            )[1]
            for i in run
        ]
        origin = min(feasible)
        for i in run:
            _, kept = place_envelope(
                envelope(cues[i].text),
                region=region,
                center_x=center_x,
                foreground_top=top,
                clearance=clearance,
                previous_y=origin,
            )
            if kept != origin:
                raise ReplayError("A caption run does not fit at one fixed origin above the person")
            result.append(FixedCue(cues[i], number, center_x, origin))
    return tuple(result)


def candidate_document(
    fixed: Sequence[FixedCue],
    *,
    canvas: tuple[int, int],
    rate: int,
    coordinate_system: str,
    inputs: dict[str, str],
    calibration: tuple[str, str],
) -> dict[str, object]:
    """Candidate-only measured-caption document; approvals stay false for the caller to earn.

    `calibration` is the (path, sha256) of the render calibration; downstream readers require it
    to be one of the bound inputs with the same digest, so a mismatch is refused here already.
    """
    path, digest = calibration
    if inputs.get(path) != digest:
        raise ReplayError("The render calibration must be bound among the document inputs")
    return {
        "version": 1,
        "candidateOnly": True,
        "inputs": inputs,
        "renderCalibration": {"path": path, "sha256": digest},
        "canvas": list(canvas),
        "coordinateSystem": coordinate_system,
        "policy": {"frame_rate": rate},
        "candidates": [
            {
                "cue": {
                    "text": item.cue.text,
                    "start": float(item.cue.start),
                    "end": float(item.cue.end),
                },
                "x": item.x,
                "y": item.y,
            }
            for item in fixed
        ],
        "editorCorrespondenceVerified": False,
        "completeMotionSupportVerified": False,
        "motionRenderVerified": False,
        "semanticPlacementApproved": False,
        "captionGeometryApproved": False,
        "aaaContrastApproved": False,
        "deliveryComplete": False,
    }
