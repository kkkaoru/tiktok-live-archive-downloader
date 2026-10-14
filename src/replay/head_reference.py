"""Select a measured face reference and derive approximate head geometry."""

from bisect import bisect_right
from fractions import Fraction
from math import isfinite
from statistics import median

from replay.caption_style_defaults import HEAD_CAPTION_GAP_PX, HEAD_EXTENSION_FACE_HEIGHT_RATIO
from replay.core import ReplayError
from replay.fcp_head_near import FaceScaleSeries
from replay.fcp_tracking_caption import TrackingCalibration


def automatic_reference(faces: FaceScaleSeries) -> TrackingCalibration:
    """Use a representative observed scale; never select held/missing geometry.

    ROI is a full-width vertical reference band, not a detected face rectangle.
    The head location is a shared heuristic, including explicit canvas clamping.
    """
    gaps = faces.tops.derived_ranges
    starts = tuple(gap.start for gap in gaps)
    observed = []
    for top, bottom in zip(faces.tops.positions, faces.bottoms.positions, strict=True):
        previous = bisect_right(starts, top.time) - 1
        if previous < 0 or top.time >= gaps[previous].end:
            observed.append((top, bottom))
    if not observed:
        raise ReplayError("Automatic head reference requires an observed face")
    typical_height = median(bottom.y - top.y for top, bottom in observed)
    top, bottom = min(observed, key=lambda pair: abs(pair[1].y - pair[0].y - typical_height))
    height = bottom.y - top.y
    return TrackingCalibration(
        top.time,
        max(0.0, top.y - HEAD_EXTENSION_FACE_HEIGHT_RATIO * height),
        1,
        0,
        HEAD_CAPTION_GAP_PX,
        faces.tops.height,
        (0.0, top.y, float(faces.tops.width), height),
    )


def resolve_reference(config: dict[str, object], faces: FaceScaleSeries) -> TrackingCalibration:
    """Retain explicit references or select the opt-in shared automatic estimate."""
    policy = config.get("headReferencePolicy", "explicit")
    if policy == "automatic-face-relative":
        if any(key in config for key in ("referenceTime", "referenceHeadTop", "referenceROI")):
            raise ReplayError("Automatic reference cannot mix explicit calibration")
        return automatic_reference(faces)
    if policy != "explicit":
        raise ReplayError("Unknown head reference policy")
    time = Fraction(str(config.get("referenceTime")))
    extent = faces.observed_reference(time)
    head, roi = config.get("referenceHeadTop"), config.get("referenceROI")
    if (
        isinstance(head, bool)
        or not isinstance(head, (int, float))
        or not isfinite(head)
        or not 0 <= head <= extent.top
        or not isinstance(roi, list)
        or len(roi) != 4
        or any(
            isinstance(v, bool) or not isinstance(v, (int, float)) or not isfinite(v) for v in roi
        )
    ):
        raise ReplayError("Expected finite explicit head and reference rectangle")
    coordinates = tuple(float(v) for v in roi if isinstance(v, (int, float)))
    if min(coordinates) < 0 or min(coordinates[2:]) <= 0:
        raise ReplayError("Expected positive reference rectangle")
    return TrackingCalibration(
        time,
        float(head),
        1,
        0,
        HEAD_CAPTION_GAP_PX,
        faces.tops.height,
        (coordinates[0], coordinates[1], coordinates[2], coordinates[3]),
    )
