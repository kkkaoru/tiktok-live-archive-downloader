"""Approximate head-near caption placement from measured face scale.

This is derived geometry, not hair detection. Callers retain face observation or
hold provenance and supply the actual rendered caption footprint, including
outlines and animation excursions. X and bottom alignment are caller invariants.
"""

from dataclasses import dataclass
from fractions import Fraction
from math import isfinite
from typing import Literal

from replay.caption_style_defaults import CAPTION_SCREEN_INSET_PX, HEAD_CAPTION_GAP_PX
from replay.core import ReplayError


@dataclass(frozen=True, slots=True)
class FaceSizeObservation:
    time: Fraction
    width: float
    height: float
    observed: bool = True


@dataclass(frozen=True, slots=True)
class FaceVerticalExtent:
    top: float
    bottom: float


@dataclass(frozen=True, slots=True)
class HeadNearPlacement:
    estimated_head_top: float
    caption_bottom: float | None
    head_extension_ratio: float
    reason: Literal["scale-derived", "canvas-fitted", "insufficient-space-above-face"]


def current_face_heights(
    observations: tuple[FaceSizeObservation, ...],
    *,
    reference: FaceSizeObservation,
) -> tuple[float, ...]:
    """Derive each scale from its own geometry, without retaining older sizes.

    The larger current axis ratio protects against expression/yaw contraction
    along only one axis. It never carries an older maximum into a new sample.
    Missing provenance remains on the input; a derived size is not a detection.
    """
    if (
        not observations
        or not reference.observed
        or any(
            isinstance(v, bool) or not isfinite(v) or v <= 0
            for v in (reference.width, reference.height)
        )
    ):
        raise ReplayError("Expected ordered scale observations and reset times")
    heights: list[float] = []
    previous: FaceSizeObservation | None = None
    for item in observations:
        if (
            not isinstance(item.time, Fraction)
            or not isinstance(item.observed, bool)
            or any(
                isinstance(v, bool) or not isfinite(v) or v <= 0 for v in (item.width, item.height)
            )
            or (previous is not None and item.time <= previous.time)
        ):
            raise ReplayError("Expected finite positive face sizes with increasing exact clocks")
        heights.append(max(item.height, item.width * reference.height / reference.width))
        previous = item
    return tuple(heights)


def caption_only_bottom(*, caption_height: float, canvas_height: float) -> float:
    """Fit the measured block at the screen top without consulting a face position.

    This screen-only candidate does not prove face clearance or head proximity.
    """
    values = (caption_height, canvas_height)
    if (
        any(isinstance(value, bool) or not isfinite(value) for value in values)
        or not 0 < caption_height <= canvas_height
    ):
        raise ReplayError("Expected bounded screen-safe caption geometry")
    inset = min(CAPTION_SCREEN_INSET_PX, (canvas_height - caption_height) / 2)
    return caption_height + inset


def place_near_head(
    *,
    face: FaceVerticalExtent,
    reference_face: FaceVerticalExtent,
    reference_head_top: float,
    caption_height: float,
    canvas_height: float,
    gap: float = HEAD_CAPTION_GAP_PX,
    estimated_face_height: float | None = None,
) -> HeadNearPlacement:
    """Scale a reference head extension, then fit a bottom-aligned caption.

    A smaller face reduces the reference pixel offset proportionally. A fitted
    result may approach the face more than the estimated hair top; it remains a
    candidate with explicit provenance. No room above the face returns no Y,
    allowing the caller to try measured reflow without inventing an observation.
    """
    values = (
        face.top,
        face.bottom,
        reference_face.top,
        reference_face.bottom,
        reference_head_top,
        caption_height,
        canvas_height,
        gap,
    )
    if (
        any(isinstance(value, bool) or not isfinite(value) for value in values)
        or not 0 < caption_height <= canvas_height
        or not 0 <= face.top < face.bottom <= canvas_height
        or not 0
        <= reference_head_top
        <= reference_face.top
        < reference_face.bottom
        <= canvas_height
        or gap < 0
    ):
        raise ReplayError("Expected finite face, reference head and rendered caption geometry")
    ratio = (reference_face.top - reference_head_top) / (reference_face.bottom - reference_face.top)
    scale_height = (
        face.bottom - face.top if estimated_face_height is None else estimated_face_height
    )
    if (
        isinstance(scale_height, bool)
        or not isfinite(scale_height)
        or not 0 < scale_height <= canvas_height
    ):
        raise ReplayError("Expected bounded derived face height")
    head = face.top - ratio * scale_height
    upper_bottom = face.top - gap
    if upper_bottom < caption_height:
        return HeadNearPlacement(head, None, ratio, "insufficient-space-above-face")
    desired = head - gap
    bottom = min(max(desired, caption_height), upper_bottom)
    return HeadNearPlacement(
        head, bottom, ratio, "scale-derived" if bottom == desired else "canvas-fitted"
    )
