"""Bound a final display cell without extending physical media observations."""

from fractions import Fraction
from math import ceil

from replay.core import ReplayError


def terminal_cell_end(*, physical_end: Fraction, frame_duration: Fraction) -> Fraction:
    if physical_end <= 0 or not 0 < frame_duration <= 1:
        raise ReplayError("Expected positive physical extent and bounded display frame")
    return ceil(physical_end / frame_duration) * frame_duration


def validate_terminal_cell(
    *,
    start: Fraction,
    end: Fraction,
    physical_end: Fraction,
    frame_duration: Fraction,
    allow_terminal_cell: bool = False,
) -> bool:
    """Return whether the explicit interval uses the final partial display cell.

    Only a grid-aligned exclusive end after an actual valid frame start may
    exceed physical end, and only when explicitly enabled. Interior missing
    observations are outside this contract and remain unknown.
    """
    edge = terminal_cell_end(physical_end=physical_end, frame_duration=frame_duration)
    if not 0 <= start < end or not isinstance(allow_terminal_cell, bool):
        raise ReplayError("Invalid display interval or terminal-cell policy")
    if end <= physical_end:
        return False
    if (
        not allow_terminal_cell
        or end != edge
        or max(start, end - frame_duration) >= physical_end
        or not 0 < end - physical_end < frame_duration
    ):
        raise ReplayError("Display interval exceeds physical coverage beyond its final valid cell")
    return True


def terminal_cell_receipt(*, physical_end: Fraction, frame_duration: Fraction) -> dict[str, object]:
    edge = terminal_cell_end(physical_end=physical_end, frame_duration=frame_duration)
    return {
        "policy": "last-valid-display-cell",
        "physicalEnd": str(physical_end),
        "displayGridEnd": str(edge),
        "derivedDisplayTail": str(edge - physical_end),
        "frameDuration": str(frame_duration),
        "observedCoverageExtended": False,
    }
