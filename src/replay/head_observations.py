"""Validate sampled Vision head geometry and fit it to the caption canvas.

The largest face is selected, as in the existing single-subject FFmpeg workflow;
without a face the tallest whole-person box is used. This is geometric selection,
not identity recognition or proof of head clearance on unobserved frames.
"""

import math
from dataclasses import dataclass
from enum import StrEnum
from fractions import Fraction

from replay.caption_io import executor_measurement
from replay.core import ReplayError
from replay.recordings import record
from replay.subtitle_clock import VideoFrameClock
from replay.vertical_captions import FRAME, Box, HeadSample, head_sample


@dataclass(frozen=True, slots=True)
class HeadObservation:
    requested_time: float
    sample: HeadSample


@dataclass(frozen=True, slots=True)
class TrackedFace:
    box: Box
    confidence: float


class TrackingMethod(StrEnum):
    DETECTION = "detection"
    TRACKING = "tracking"
    MISSING = "missing"


@dataclass(frozen=True, slots=True)
class TrackingFrame:
    time: Fraction
    method: TrackingMethod
    detection_attempted: bool
    faces: tuple[TrackedFace, ...]
    out_of_frame_tracking_observations: int = 0
    out_of_frame_detection_observations: int = 0


# At most 32 active tracks plus 32 freshly seeded tracks in one frame.
MAX_TRACKING_REJECTIONS = 64
MAX_DETECTION_REJECTIONS = 32
MAX_TRACKING_FRAMES = 7200
MAX_TRACKING_SECONDS = 120
MAX_TRACKING_TIMESCALE = 1_000_000_000
MIN_TRACKING_CONFIDENCE = 0.3


def _number(value: object) -> float:
    if isinstance(value, bool) or not isinstance(value, (float, int)):
        raise ReplayError("Invalid head observation number")
    try:
        result = float(value)
    except OverflowError as error:
        raise ReplayError("Head observation number exceeds its budget") from error
    if not math.isfinite(result):
        raise ReplayError("Non-finite head observation number")
    return result


def _boxes(value: object, width: int, height: int) -> tuple[Box, ...]:
    if not isinstance(value, list) or len(value) > 32:
        raise ReplayError("Invalid head rectangle list")
    result: list[Box] = []
    for item in value:
        region = record(item)
        x, y, w, h = (_number(region.get(key)) for key in ("x", "y", "width", "height"))
        if min(x, y) < 0 or min(w, h) <= 0 or x + w > width or y + h > height:
            raise ReplayError("Head rectangle exceeds its source frame")
        result.append(Box(x, y, x + w, y + h))
    return tuple(result)


def validate_tracking_window(clock: VideoFrameClock, *, width: int, height: int) -> None:
    """Check the same resource/geometry bounds before collection and during parsing."""
    if (
        not 1 <= len(clock.pts) <= MAX_TRACKING_FRAMES
        or (clock.pts[-1] - clock.pts[0] + clock.last_duration) * clock.time_base
        > MAX_TRACKING_SECONDS
        or type(width) is not int
        or type(height) is not int
        or not 1 <= width <= 16384
        or not 1 <= height <= 16384
        or width * height > 16777216
    ):
        raise ReplayError("Invalid tracking window or geometry budget")


def tracking_frames(
    value: object, clock: VideoFrameClock, *, width: int, height: int
) -> tuple[TrackingFrame, ...]:
    """Match every native rational PTS to an independently verified source clock.

    Missing geometry stays empty. This is not source provenance, identity, hair
    geometry or caption clearance; callers verify those separately.
    """
    validate_tracking_window(clock, width=width, height=height)
    if not isinstance(value, list) or len(value) != len(clock.pts):
        raise ReplayError("Invalid tracking frame count")
    result: list[TrackingFrame] = []
    for item, expected in zip(value, clock.pts, strict=True):
        frame = record(item)
        ticks, scale = frame.get("presentationValue"), frame.get("presentationTimescale")
        if (
            type(ticks) is not int
            or type(scale) is not int
            or not 1 <= scale <= MAX_TRACKING_TIMESCALE
            or not 0 <= ticks <= 86400 * scale
            or Fraction(ticks, scale) != expected * clock.time_base
            or type(frame.get("width")) is not int
            or type(frame.get("height")) is not int
            or frame.get("width") != width
            or frame.get("height") != height
        ):
            raise ReplayError("Tracking frame differs from verified source clock or geometry")
        raw_method = frame.get("method")
        if not isinstance(raw_method, str):
            raise ReplayError("Invalid tracking observation method")
        try:
            method = TrackingMethod(raw_method)
        except ValueError as error:
            raise ReplayError("Invalid tracking observation method") from error
        attempted = frame.get("detectionAttempted")
        if (
            type(attempted) is not bool
            or (not result and (not attempted or method == "tracking"))
            or (method == "detection" and not attempted)
        ):
            raise ReplayError("Invalid tracking observation method")
        faces = _tracked_faces(frame.get("faces"), width, height)
        if (method == "missing") != (not faces):
            raise ReplayError("Tracking method contradicts observed geometry")
        # Legacy successful responses had no rejected boxes: they failed instead.
        rejected = _rejection_count(
            frame.get("outOfFrameTrackingObservations", 0),
            maximum=MAX_TRACKING_REJECTIONS,
            detection_attempted=attempted,
        )
        rejected_detections = _rejection_count(
            frame.get("outOfFrameDetectionObservations", 0),
            maximum=MAX_DETECTION_REJECTIONS,
            detection_attempted=attempted,
        )
        result.append(
            TrackingFrame(
                Fraction(ticks, scale), method, attempted, faces, rejected, rejected_detections
            )
        )
    return tuple(result)


def _rejection_count(value: object, *, maximum: int, detection_attempted: bool) -> int:
    if (
        type(value) is not int
        or not 0 <= value <= maximum
        or (value > 0 and not detection_attempted)
    ):
        raise ReplayError("Invalid out-of-frame rejection evidence")
    return value


def _tracked_faces(value: object, width: int, height: int) -> tuple[TrackedFace, ...]:
    if not isinstance(value, list) or len(value) > 32:
        raise ReplayError("Invalid tracked face list")
    result: list[TrackedFace] = []
    for item in value:
        face = record(item)
        confidence = _number(face.get("confidence"))
        if not MIN_TRACKING_CONFIDENCE < confidence <= 1:
            raise ReplayError("Invalid tracking confidence")
        box = _boxes([face.get("region")], width, height)[0]
        result.append(TrackedFace(box, confidence))
    return tuple(result)


def head_frames(value: object, *, native: bool = False) -> tuple[HeadObservation, ...]:
    """Parse one bounded batch, retaining actual rather than requested sample clocks.

    Native input is the raw CLI array; the default requires an Executor envelope.
    Input rectangles are display-oriented source pixels. Fit/letterbox projection
    matches the existing vertical canvas and changes height only, not caption X.
    Empty detections remain None; held decoded times are not silently synthesized.
    """
    raw = value if native else executor_measurement(value)
    if not isinstance(raw, list) or not 1 <= len(raw) <= 8:
        raise ReplayError("Invalid head observation batch")
    result: list[HeadObservation] = []
    for item in raw:
        frame = record(item)
        requested = _number(frame.get("requestedTimeSeconds"))
        actual = _number(frame.get("actualTimeSeconds"))
        width, height = frame.get("width"), frame.get("height")
        if (
            type(width) is not int
            or type(height) is not int
            or not 1 <= width <= 16384
            or not 1 <= height <= 16384
            or width * height > 16777216
            or not 0 <= requested <= 86400
            or not 0 <= actual <= 86400
        ):
            raise ReplayError("Invalid head frame geometry or clock")
        faces = _boxes(frame.get("faces"), width, height)
        bodies = _boxes(frame.get("bodies"), width, height)
        face = max(faces, key=lambda box: box.right - box.left, default=None)
        body = max(bodies, key=lambda box: box.bottom - box.top, default=None)
        scale = min(FRAME[0] / width, FRAME[1] / height)
        offset_y = (FRAME[1] - height * scale) / 2
        sample = head_sample(
            {
                "t": actual,
                "faceTop": offset_y + face.top * scale if face is not None else None,
                "faceHeight": (face.bottom - face.top) * scale if face is not None else None,
                "bodyTop": offset_y + body.top * scale if body is not None else None,
            }
        )
        result.append(HeadObservation(requested, sample))
    return tuple(result)
