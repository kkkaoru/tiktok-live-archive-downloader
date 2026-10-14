"""Sampled mask coverage and independent post-render OCR, not perceptual approval."""

from bisect import bisect_right
from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from replay.caption_io import FrameObservation
from replay.subtitle_plan import ObservedMask, number
from replay.subtitle_profile import SubtitleBands, profiled_detections

PixelSupport = NDArray[np.bool_]
MAX_SUPPORT_PIXELS = 16_000_000


@dataclass(frozen=True, slots=True)
class PixelSupportAudit:
    source_pixels: int
    output_pixels: int
    persistent_pixels: int
    source_outside_roi: int
    output_outside_roi: int


@dataclass(frozen=True, slots=True)
class MaskObservationAudit:
    sampled_frames: int
    subtitle_lines: int
    uncovered_lines: int


@dataclass(frozen=True, slots=True)
class OutputTextAudit:
    sampled_frames: int
    source_subtitle_lines: int
    output_readable_lines: int
    frames_with_readable_text: int
    maximum_actual_time_delta: float


def audit_pixel_support(
    source: PixelSupport, output: PixelSupport, *, allowed_roi: PixelSupport
) -> PixelSupportAudit:
    """Count independently supplied evidence pixels inside/outside the repair ROI.

    Callers align actual decoded frames and localize the evidence independently.
    These counts neither classify pixels as subtitles nor certify naturalness.
    No detector, confidence threshold or quality-approval flag is hidden here.
    """
    if (
        any(
            mask.dtype != np.bool_ or mask.ndim != 2 or mask.shape != source.shape
            for mask in (source, output, allowed_roi)
        )
        or not 0 < source.size <= MAX_SUPPORT_PIXELS
    ):
        raise ValueError("Invalid pixel-support audit masks")
    return PixelSupportAudit(
        int(np.count_nonzero(source)),
        int(np.count_nonzero(output)),
        int(np.count_nonzero(source & output)),
        int(np.count_nonzero(source & ~allowed_roi)),
        int(np.count_nonzero(output & ~allowed_roi)),
    )


def audit_mask_observations(
    frames: tuple[FrameObservation, ...],
    masks: tuple[ObservedMask, ...],
    *,
    profile: SubtitleBands,
) -> MaskObservationAudit:
    """Check full line-box containment at actual decoded, half-open mask times.

    Missing masks and partially covered lines remain explicit failures. The
    caller verifies file/source provenance. This cannot certify unsampled frames.
    """
    if not 1 <= len(frames) <= 50000 or len(masks) > 50000:
        raise ValueError("Invalid observation audit budget")
    previous_end = 0.0
    for mask in masks:
        if not previous_end <= number(mask.start) < number(mask.end) <= 86400:
            raise ValueError("Mask audit requires ordered nonoverlapping intervals")
        box = mask.rectangle
        if box.x + box.width > profile.width or box.y + box.height > profile.height:
            raise ValueError("Audit mask exceeds source canvas")
        previous_end = mask.end
    starts = tuple(mask.start for mask in masks)
    lines = uncovered = 0
    for frame in frames:
        time = number(frame.actual_time)
        if not 0 <= time <= 86400:
            raise ValueError("Invalid observation audit clock")
        index = bisect_right(starts, time) - 1
        active = masks[index] if index >= 0 and time < masks[index].end else None
        detected = profiled_detections(frame, profile)
        lines += len(detected)
        uncovered += sum(
            active is None
            or line.rectangle.x < active.rectangle.x
            or line.rectangle.y < active.rectangle.y
            or line.rectangle.x + line.rectangle.width > active.rectangle.x + active.rectangle.width
            or line.rectangle.y + line.rectangle.height
            > active.rectangle.y + active.rectangle.height
            for line in detected
        )
    return MaskObservationAudit(len(frames), lines, uncovered)


def audit_output_text(
    source: tuple[FrameObservation, ...],
    output: tuple[FrameObservation, ...],
    *,
    profile: SubtitleBands,
) -> OutputTextAudit:
    """Count readable output lines in SOURCE-learned bands at matching requests.

    Decode held-frame clocks remain distinct and their difference is reported,
    never silently equated. A zero count is only sampled OCR absence, not proof
    of invisible fragments, temporal stability, or a fully restored background.
    """
    if not 1 <= len(source) <= 50000 or len(source) != len(output):
        raise ValueError("Incomplete paired OCR observations")
    source_lines = output_lines = readable_frames = 0
    previous_request, maximum_delta = -1.0, 0.0
    for original, candidate in zip(source, output, strict=True):
        requested = number(original.requested_time)
        if (
            not 0 <= requested <= 86400
            or requested <= previous_request
            or requested != number(candidate.requested_time)
        ):
            raise ValueError("Paired OCR requests differ or are duplicated")
        previous_request = requested
        before, after = number(original.actual_time), number(candidate.actual_time)
        if not 0 <= before <= 86400 or not 0 <= after <= 86400:
            raise ValueError("Invalid paired OCR decoded clock")
        maximum_delta = max(maximum_delta, abs(before - after))
        source_lines += len(profiled_detections(original, profile))
        detected = profiled_detections(candidate, profile)
        output_lines += len(detected)
        readable_frames += bool(detected)
    return OutputTextAudit(len(source), source_lines, output_lines, readable_frames, maximum_delta)
