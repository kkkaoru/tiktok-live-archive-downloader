"""Transport independently measured interior upper pixels, never missing faces.

Two disjoint measured tile bands must agree under the existing CPU texture
limits. A partial object rectangle, historical height or confidence alone is
never a head seed. This bounded diagnostic does not change caption placement.
"""

import argparse
import json
import math
import resource
from collections.abc import Iterable
from dataclasses import asdict, dataclass
from fractions import Fraction
from itertools import chain, pairwise
from pathlib import Path
from time import monotonic

import numpy as np

from replay import head_contour_track as tracker
from replay import head_upper_texture as upper
from replay.artifact_json import MAX_JSON_BYTES as MAX_APPROX_JSON_BYTES
from replay.artifact_json import _json as read_approximate_json
from replay.caption_style_defaults import (
    HEAD_CAPTION_GAP_PX,
    HEAD_CAPTION_REFERENCE_HEIGHT,
    HEAD_CAPTION_Y_DEADBAND_PX,
)
from replay.delivery_mask_quality import integer, rows
from replay.head_image_support import measured_upper_cells
from replay.head_observations import MIN_TRACKING_CONFIDENCE
from replay.head_seed import ContourTrackMetrics, Coordinates
from replay.head_texture_track import (
    Matrix,
    TextureFit,
    TextureMotion,
    derive_contour_motion,
    foreground_features,
    measured_region_owners,
    measured_tile_partition,
    select_current_local_features,
    select_current_stable_features,
)
from replay.recordings import record
from replay.stages import _uncached_file_digest
from replay.subtitle_clock import VideoFrameClock
from replay.subtitle_inpaint import Image

PROTOCOL = "derived-partial-upper-pixels-v1"
APPROX_PROTOCOL = "derived-face-motion-approximate-head-v1"
UNCERTAIN_PROTOCOL = "derived-face-motion-uncertain-head-v1"
CORROBORATED_PROTOCOL = "derived-face-motion-corroborated-head-v1"
SPATIAL_PROTOCOL = "derived-face-motion-spatial-stable-head-v1"
SPATIAL_POLICY = "current-supported-spatial-replenishment-v1"
LOCAL_PROTOCOL = "derived-face-motion-measured-local-head-v1"
LOCAL_POLICY = "predeclared-measured-region-spatial-v1"
LOCAL_HEAD_POLICY = "predeclared-local-face-target-with-independent-corroboration-v1"
SPATIAL_MODELS_PER_RECORD = 32
MAX_SPATIAL_MODEL_RECORDS = 16
MAX_SPATIAL_MODEL_TOTAL_BYTES = 120 * 1024 * 1024
MAX_FRAMES = 256
MAX_SECONDS = 10
MIN_FEATURES = 12
RESET_REASONS = frozenset({"source-cut", "scene", "zoom", "geometry"})
Region = tuple[int, int, int, int]
Face = tuple[float, float, float, float]


@dataclass(frozen=True, slots=True)
class PartialPixelSeed:
    contour: tracker.ContourSeed
    upper_regions: tuple[Region, ...]
    anchor_regions: tuple[Region, ...]
    native_status: str


@dataclass(frozen=True, slots=True)
class PixelObservation:
    source_pts: int
    face_observed: bool
    provenance: str
    candidate_upper_y: float | None
    upper_features: int
    upper_supported: int
    upper_inliers: int
    anchor_features: int
    anchor_supported: int
    anchor_inliers: int
    maximum_band_disagreement: float | None
    reason: str | None


@dataclass(frozen=True, slots=True)
class FaceMotionObservation:
    source_pts: int
    face_observed: bool
    estimated_upper_y: float | None
    maximum_model_disagreement: float | None
    models: tuple[dict[str, object], ...]
    feature_refresh: dict[str, object]
    reason: str | None


@dataclass(frozen=True, slots=True)
class ApproximateModelDecision:
    accepted: bool
    reason: str | None
    transferred_controls: Coordinates | None
    audit: dict[str, object]


@dataclass(frozen=True, slots=True)
class ModelPairDecision:
    accepted: bool
    reason: str | None
    transferred_controls: tuple[Coordinates, Coordinates] | None
    estimated_upper_y: float | None
    audit: dict[str, object]


def target_y_budget(height: int) -> float:
    """Allocate at most one third of the shared intended head clearance.

    This editing-policy allocation is not a physical hair-error calibration and
    is independent of the existing model agreement and caption deadband.
    """
    if not 16 <= height <= 16384:
        raise ValueError("Bounded canvas height is required")
    return HEAD_CAPTION_GAP_PX * height / HEAD_CAPTION_REFERENCE_HEIGHT / 3


def approximate_model(
    fitted: TextureFit,
    controls: Coordinates,
    *,
    maximum_target_y_uncertainty_px: float,
    width: int,
    height: int,
) -> ApproximateModelDecision:
    """Fit a distinct approximate target-Y model from actual supported pixels.

    The robust basis remains the original exact fit's inliers. A conditional
    perturbation budget controls a separate consensus; no legacy decision is
    changed. The L1 operator gain covers every real apex control, including
    extrapolation beyond the distributed facial support.
    """
    budget = float(maximum_target_y_uncertainty_px)
    if not math.isfinite(budget) or not 0 < budget <= target_y_budget(height):
        raise ValueError("Target Y budget must fit its explicit clearance allocation")
    if (
        not 16 <= width <= 16384
        or controls.ndim != 2
        or controls.shape[1] != 2
        or not 3 <= len(controls) <= 256
        or not np.isfinite(controls).all()
    ):
        raise ValueError("Bounded actual target controls are required")
    source = fitted.source.astype(np.float64)
    current = fitted.metrics.points.astype(np.float64)
    supported = fitted.metrics.supported
    basis = np.asarray(fitted.inliers, np.bool_) & supported
    count = int(np.count_nonzero(basis))
    source_span = np.ptp(source, axis=0)
    basis_span = np.ptp(source[basis], axis=0) if count else np.zeros(2)
    reason: str | None = None
    matrix: Matrix | None = None
    moved: Coordinates | None = None
    gain: float | None = None
    limit: float | None = None
    bound: float | None = None
    scale: float | None = None
    rotation: float | None = None
    residual: tuple[float | None, ...] = tuple(None for _ in source)
    consensus = np.zeros(len(source), np.bool_)
    consensus_span = np.zeros(2)
    if fitted.matrix is None or count < MIN_FEATURES:
        reason = "approximate_basis_unavailable"
    elif np.any(source_span < 8) or np.any(basis_span < source_span * 0.5):
        reason = "approximate_basis_clustered"
    else:
        old, new = source[basis], current[basis]
        center, destination = old.mean(axis=0), new.mean(axis=0)
        centered, displaced = old - center, new - destination
        denominator = float(np.sum(centered * centered))
        real = float(np.sum(centered * displaced)) / denominator
        imaginary = (
            float(np.sum(centered[:, 0] * displaced[:, 1] - centered[:, 1] * displaced[:, 0]))
            / denominator
        )
        linear = np.array(((real, -imaginary), (imaginary, real)), np.float64)
        offset = destination - center @ linear.T
        matrix = ((real, -imaginary, float(offset[0])), (imaginary, real, float(offset[1])))
        target = controls.astype(np.float64) - center
        # Each current correspondence has bounded vector perturbation. Sum the
        # Euclidean row-coefficient norms for each target's output Y.
        dx, dy = target[:, 0, None], target[:, 1, None]
        x, y = centered[:, 0], centered[:, 1]
        coeff_u = (dy * x - dx * y) / denominator
        coeff_v = 1 / count + (dy * y + dx * x) / denominator
        gain = float(np.sum(np.hypot(coeff_u, coeff_v), axis=1).max())
        limit = budget / gain
        errors = np.linalg.norm(source @ linear.T + offset - current, axis=1)
        residual = tuple(float(v) if np.isfinite(v) else None for v in errors)
        consensus = supported & (errors <= limit)
        accepted_count = int(np.count_nonzero(consensus))
        consensus_span = np.ptp(source[consensus], axis=0) if accepted_count else np.zeros(2)
        scale, rotation = (
            math.hypot(real, imaginary),
            abs(math.degrees(math.atan2(imaginary, real))),
        )
        transformed = np.asarray(controls @ linear.T + offset, np.float32)
        bound = float(errors[consensus].max()) * gain if accepted_count else None
        if (
            accepted_count < MIN_FEATURES
            or accepted_count / int(np.count_nonzero(supported)) < 0.75
        ):
            reason = "approximate_consensus_insufficient"
        elif np.any(consensus_span < source_span * 0.5):
            reason = "approximate_consensus_clustered"
        elif not 0.9 <= scale <= 1.1 or rotation > 15:
            reason = "approximate_deformation_exceeded"
        elif np.any(transformed < 0) or np.any(transformed >= (width, height)):
            reason = "approximate_target_clipped"
        else:
            moved = transformed
    supported_count = int(np.count_nonzero(supported))
    consensus_count = int(np.count_nonzero(consensus))
    audit: dict[str, object] = {
        "version": 1,
        "accepted": reason is None,
        "reason": reason,
        "matrix": matrix,
        "basisInliers": basis.tolist(),
        "basisCount": count,
        "consensusInliers": consensus.tolist(),
        "consensusCount": consensus_count,
        "consensusRatio": consensus_count / supported_count if supported_count else None,
        "minimumInliers": MIN_FEATURES,
        "minimumInlierRatio": 0.75,
        "sourceSpan": source_span.tolist(),
        "basisSpan": basis_span.tolist(),
        "consensusSpan": consensus_span.tolist(),
        "minimumSourceSpanPx": 8,
        "minimumSupportSpanRatio": 0.5,
        "residuals": residual,
        "targetSensitivityGain": gain,
        "localResidualLimitPx": limit,
        "conditionalBoundPx": bound,
        "maximumTargetYModelUncertaintyPx": budget,
        "geometry": {
            "scale": scale,
            "rotationDegrees": rotation,
            "scaleLimits": [0.9, 1.1],
            "rotationLimitDegrees": 15,
        },
        "transferredControls": None if moved is None else moved.tolist(),
        "notTrueHairErrorGuarantee": True,
        "uncertaintyScope": "one-current-correspondence-transition",
        "accumulatedHairErrorBounded": False,
    }
    return ApproximateModelDecision(reason is None, reason, moved, audit)


def _coordinates(value: object) -> Coordinates:
    items = rows(value)
    if not 3 <= len(items) <= 256 or any(len(rows(item)) != 2 for item in items):
        raise ValueError("Bounded two-dimensional model coordinates are required")
    return np.asarray(tuple(tuple(_number(v) for v in rows(item)) for item in items), np.float32)


def _booleans(value: object, length: int) -> np.ndarray[tuple[int], np.dtype[np.bool_]]:
    items = rows(value)
    if len(items) != length or any(type(item) is not bool for item in items):
        raise ValueError("Exact correspondence status flags are required")
    return np.asarray(items, np.bool_)


def _same_audit(actual: object, expected: object) -> bool:
    if isinstance(expected, dict):
        supplied = record(actual)
        return set(supplied) == set(expected) and all(
            _same_audit(supplied[key], value) for key, value in expected.items()
        )
    if isinstance(expected, (tuple, list)):
        supplied_rows = rows(actual)
        return len(supplied_rows) == len(expected) and all(
            _same_audit(one, two) for one, two in zip(supplied_rows, expected, strict=True)
        )
    if isinstance(expected, float):
        return math.isclose(_number(actual), expected, rel_tol=1e-6, abs_tol=1e-5)
    return type(actual) is type(expected) and actual == expected


def _serialized_model(
    value: object, *, width: int, height: int
) -> tuple[dict[str, object], TextureFit, Coordinates]:
    raw = record(value)
    source, current, controls = (
        _coordinates(raw.get(k)) for k in ("sourceFeatures", "currentFeatures", "sourceControls")
    )
    count = len(source)
    if len(current) != count or count < MIN_FEATURES:
        raise ValueError("Approximate model correspondence counts differ")
    forward, backward, canvas, supported, inliers = (
        _booleans(raw.get(k), count)
        for k in ("forwardStatus", "backwardStatus", "canvasStatus", "supported", "inliers")
    )
    roundtrip, appearance = (
        np.asarray(tuple(np.nan if v is None else _number(v) for v in rows(raw.get(k))), np.float32)
        for k in ("roundtripError", "appearanceError")
    )
    if (
        len(roundtrip) != count
        or len(appearance) != count
        or np.any(roundtrip < 0)
        or np.any(appearance < 0)
        or np.any(source < 8)
        or np.any(source >= (width - 8, height - 8))
        or not np.array_equal(
            canvas, np.all((current >= 8) & (current < (width - 8, height - 8)), axis=1)
        )
        or not np.array_equal(
            supported,
            forward
            & backward
            & canvas
            & np.isfinite(roundtrip)
            & np.isfinite(appearance)
            & (roundtrip <= 0.5)
            & (appearance <= 12),
        )
    ):
        raise ValueError("Approximate model pixel support differs from original guards")
    matrix_rows = rows(raw.get("fitMatrix"))
    if len(matrix_rows) != 2 or any(len(rows(item)) != 3 for item in matrix_rows):
        raise ValueError("Actual original fit matrix is required")
    first, second = (tuple(_number(v) for v in rows(item)) for item in matrix_rows)
    matrix: Matrix = ((first[0], first[1], first[2]), (second[0], second[1], second[2]))
    if not math.isclose(matrix[0][0], matrix[1][1], abs_tol=1e-8) or not math.isclose(
        matrix[0][1], -matrix[1][0], abs_tol=1e-8
    ):
        raise ValueError("Original matrix must describe a similarity")
    transform = np.asarray(matrix)
    residual = np.linalg.norm(source @ transform[:, :2].T + transform[:, 2] - current, axis=1)
    if not np.array_equal(inliers, supported & (residual <= 1.5)) or not _same_audit(
        raw.get("residuals"), tuple(float(v) for v in residual)
    ):
        raise ValueError("Original exact residual evidence differs")
    span = np.ptp(source, axis=0)
    inlier_span = np.ptp(source[inliers], axis=0) if np.any(inliers) else np.zeros(2)
    metrics = ContourTrackMetrics(
        current, supported, forward, backward, canvas, roundtrip, appearance
    )
    fitted = TextureFit(
        source,
        metrics,
        matrix,
        tuple(bool(v) for v in inliers),
        tuple(float(v) for v in residual),
        (float(span[0]), float(span[1])),
        (float(inlier_span[0]), float(inlier_span[1])),
        None,
    )
    return raw, fitted, controls


def _validated_model(
    value: object, *, maximum_target_y_uncertainty_px: float, width: int, height: int
) -> tuple[TextureFit, Coordinates, ApproximateModelDecision]:
    raw, fitted, controls = _serialized_model(value, width=width, height=height)
    decision = approximate_model(
        fitted,
        controls,
        maximum_target_y_uncertainty_px=maximum_target_y_uncertainty_px,
        width=width,
        height=height,
    )
    if not _same_audit(raw.get("approximateHeadContract"), decision.audit):
        raise ValueError(
            "Approximate serialized contract differs from its actual correspondence decision"
        )
    return fitted, controls, decision


def validate_approximate_model(
    value: object, *, maximum_target_y_uncertainty_px: float, width: int, height: int
) -> ApproximateModelDecision:
    """Recompute serialized supported-model decisions, without refitting images."""
    return _validated_model(
        value,
        maximum_target_y_uncertainty_px=maximum_target_y_uncertainty_px,
        width=width,
        height=height,
    )[2]


def _corroborating_model(
    fitted: TextureFit,
    controls: Coordinates,
    decision: ApproximateModelDecision,
    *,
    width: int,
    height: int,
) -> ApproximateModelDecision:
    """Corroborate a target using an independent distributed robust pixel basis.

    This does not certify its whole-cohort consensus. The original strict basis
    and its failures remain visible; a complete primary is separately required.
    """
    audit = decision.audit
    count = integer(audit["basisCount"])
    basis = np.asarray(fitted.inliers, np.bool_) & fitted.metrics.supported
    matrix_value = audit["matrix"]
    reason: str | None = None
    moved: Coordinates | None = None
    bound: float | None = None
    original_max: float | None = None
    if matrix_value is None or fitted.matrix is None or count < MIN_FEATURES:
        reason = "corroboration_basis_unavailable"
    elif np.any(np.ptp(fitted.source[basis], axis=0) < np.ptp(fitted.source, axis=0) * 0.5):
        reason = "corroboration_basis_clustered"
    else:
        original_max = max(
            _number(v) for v, selected in zip(fitted.residuals, basis, strict=True) if selected
        )
        residual = rows(
            list(audit["residuals"])
            if isinstance(audit["residuals"], tuple)
            else audit["residuals"]
        )
        bound = max(
            _number(v) for v, selected in zip(residual, basis, strict=True) if selected
        ) * _number(audit["targetSensitivityGain"])
        geometry = record(audit["geometry"])
        original = np.asarray(fitted.matrix, np.float64)
        original_scale = math.hypot(float(original[0, 0]), float(original[1, 0]))
        original_rotation = abs(
            math.degrees(math.atan2(float(original[1, 0]), float(original[0, 0])))
        )
        transform = np.asarray(matrix_value, np.float64)
        transformed = np.asarray(controls @ transform[:, :2].T + transform[:, 2], np.float32)
        if original_max > 1.5:
            reason = "corroboration_original_basis_residual_exceeded"
        elif (
            not 0.9 <= original_scale <= 1.1
            or original_rotation > 15
            or not 0.9 <= _number(geometry["scale"]) <= 1.1
            or _number(geometry["rotationDegrees"]) > 15
        ):
            reason = "corroboration_deformation_exceeded"
        elif bound > _number(audit["maximumTargetYModelUncertaintyPx"]):
            reason = "corroboration_target_budget_exceeded"
        elif np.any(transformed < 0) or np.any(transformed >= (width, height)):
            reason = "corroboration_target_clipped"
        else:
            moved = transformed
    proof: dict[str, object] = {
        "version": 1,
        "accepted": reason is None,
        "reason": reason,
        "role": "independent-distributed-target-corroboration",
        "wholeCohortAccepted": decision.accepted,
        "wholeCohortReason": decision.reason,
        "wholeCohortConsensusRatio": audit["consensusRatio"],
        "basisCount": count,
        "minimumBasisCount": MIN_FEATURES,
        "basisSpan": audit["basisSpan"],
        "sourceSpan": audit["sourceSpan"],
        "minimumSupportSpanRatio": 0.5,
        "originalBasisResidualLimitPx": 1.5,
        "originalBasisMaximumResidualPx": original_max,
        "conditionalBasisTargetBoundPx": bound,
        "maximumTargetYModelUncertaintyPx": audit["maximumTargetYModelUncertaintyPx"],
        "transferredControls": None if moved is None else moved.tolist(),
        "notWholeCohortApproval": True,
        "notTrueHairErrorGuarantee": True,
    }
    return ApproximateModelDecision(reason is None, reason, moved, proof)


def _model_pair(
    fits: tuple[TextureFit, TextureFit],
    controls: tuple[Coordinates, Coordinates],
    decisions: tuple[ApproximateModelDecision, ApproximateModelDecision],
    *,
    width: int,
    height: int,
) -> ModelPairDecision:
    corroborators = tuple(
        _corroborating_model(fitted, points, decision, width=width, height=height)
        for fitted, points, decision in zip(fits, controls, decisions, strict=True)
    )
    eligible = tuple(i for i, decision in enumerate(decisions) if decision.accepted)
    primary: int | None = None
    corroborator: int | None = None
    reason: str | None = None
    transferred: tuple[Coordinates, Coordinates] | None = None
    estimate: float | None = None
    difference: float | None = None
    if not eligible:
        reason = "complete_primary_unavailable"
    elif set(map(tuple, fits[0].source)).intersection(map(tuple, fits[1].source)):
        reason = "corroboration_cohorts_overlap"
    else:
        primary = max(eligible, key=lambda i: _number(decisions[i].audit["consensusRatio"]))
        corroborator = 1 - primary
        first = decisions[primary].transferred_controls
        second = corroborators[corroborator].transferred_controls
        if first is None or second is None:
            reason = corroborators[corroborator].reason or "corroboration_unavailable"
        else:
            difference = float(np.linalg.norm(first - second, axis=1).max())
            if difference > HEAD_CAPTION_Y_DEADBAND_PX:
                reason = "competing_corroborated_target"
            else:
                transferred = (first, second) if primary == 0 else (second, first)
                estimate = float(first[:, 1].min())
    proof: dict[str, object] = {
        "version": 1,
        "accepted": reason is None,
        "reason": reason,
        "primaryIndex": primary,
        "corroboratorIndex": corroborator,
        "primaryPolicy": "complete-conditional-target-y-model-v1",
        "corroborationPolicy": "independent-distributed-strict-basis-target-v1",
        "completeModelAccepted": [decision.accepted for decision in decisions],
        "corroborators": [decision.audit for decision in corroborators],
        "maximumControlsDisagreementPx": difference,
        "modelAgreementLimitPx": HEAD_CAPTION_Y_DEADBAND_PX,
        "estimatedUpperY": estimate,
        "currentPixelsRequired": True,
    }
    return ModelPairDecision(reason is None, reason, transferred, estimate, proof)


def validate_model_pair(
    value: object, *, maximum_target_y_uncertainty_px: float, width: int, height: int
) -> ModelPairDecision:
    """Recompute the distinct full-primary/target-corroboration role decision."""
    models = rows(value)
    if len(models) != 2:
        raise ValueError("One complete primary and independent corroborator are required")
    first, second = (
        _validated_model(
            model,
            maximum_target_y_uncertainty_px=maximum_target_y_uncertainty_px,
            width=width,
            height=height,
        )
        for model in models
    )
    return _model_pair(
        (first[0], second[0]),
        (first[1], second[1]),
        (first[2], second[2]),
        width=width,
        height=height,
    )


def validate_clock(clock: VideoFrameClock) -> None:
    """Retain the physical frame clock and finite diagnostic scope."""
    if (
        not 1 <= len(clock.pts) <= MAX_FRAMES
        or not 0 < clock.time_base <= 1
        or clock.last_duration <= 0
        or (clock.pts[-1] - clock.pts[0] + clock.last_duration) * clock.time_base > MAX_SECONDS
        or any(right <= left for left, right in pairwise(clock.pts))
    ):
        raise ValueError("Partial pixel diagnostic clock exceeds its finite budget")


def detector_seed(value: object, *, clock: VideoFrameClock) -> tuple[Face, tuple[bool, ...]]:
    """Use an actual accepted detection; preserve later missing-face flags."""
    validate_clock(clock)
    frames = rows(value)
    if len(frames) < len(clock.pts):
        raise ValueError("Detector evidence does not cover the exact pixel clock")
    flags: list[bool] = []
    for pts, item in zip(clock.pts, frames, strict=False):
        frame = record(item)
        face = record(frame.get("faceFrame"))
        if (
            Fraction(
                integer(face.get("presentationValue")), integer(face.get("presentationTimescale"))
            )
            != pts * clock.time_base
            or type(frame.get("faceObserved")) is not bool
            or frame["faceObserved"] != bool(rows(face.get("faces")))
        ):
            raise ValueError("Detector/current face evidence clock or status differs")
        flags.append(frame["faceObserved"] is True)
    first = record(frames[0])
    face_frame = record(first["faceFrame"])
    detections = [
        record(item)
        for item in rows(first.get("objects"))
        if record(item).get("stage") == "detection"
        and record(item).get("disposition") == "accepted"
    ]
    if face_frame.get("method") != "detection" or len(detections) != 1 or not flags[0]:
        raise ValueError("One actual accepted detector seed is required")
    region = record(detections[0].get("displayTopLeft"))
    values = tuple(_number(region.get(key)) for key in ("x", "y", "width", "height"))
    faces = rows(face_frame.get("faces"))
    confidence = _number(detections[0].get("confidence"))
    if (
        len(faces) != 1
        or record(record(faces[0]).get("region")) != region
        or _number(record(faces[0]).get("confidence")) != confidence
        or not MIN_TRACKING_CONFIDENCE < confidence <= 1
    ):
        raise ValueError("Detector seed must equal the actual accepted face rectangle")
    return (values[0], values[1], values[2], values[3]), tuple(flags)


def seed_request(*, source: Path, clock: VideoFrameClock, face: Face) -> dict[str, object]:
    """Outward rounding associates a component; it does not estimate the head."""
    validate_clock(clock)
    x, y, w, h = face
    if min(x, y) < 0 or min(w, h) <= 0 or not all(math.isfinite(v) for v in face):
        raise ValueError("Current detector rectangle must be finite and in frame")
    left, top = math.floor(x), math.floor(y)
    presentation = {"value": clock.pts[0], "timescale": clock.time_base.denominator}
    if clock.time_base.numerator != 1:
        raise ValueError("Native pixel seed requires an exact reciprocal time base")
    return {
        "path": str(source),
        "samples": [presentation],
        "headFace": {
            "presentation": presentation,
            "x": left,
            "y": top,
            "width": math.ceil(x + w) - left,
            "height": math.ceil(y + h) - top,
            "minimumFaceCoverage": 0.5,
        },
    }


def read_seed(
    response: object, *, clock: VideoFrameClock, width: int, height: int, face: Face
) -> PartialPixelSeed | None:
    """Associate measured upper pixels with independent measured face-interior tiles."""
    validate_clock(clock)
    start = clock.pts[0] * clock.time_base
    measured = measured_upper_cells(
        response, times=(start,), frame_ends=(start + clock.time_base,), width=width, height=height
    )
    if measured[0].visible_top is None:
        return None
    head = record(record(rows(response)[0]).get("headTop"))
    points = np.asarray(
        tuple(
            (integer(record(p).get("x")), integer(record(p).get("y")))
            for p in rows(head["upperContour"])
        ),
        np.float32,
    )
    regions = tuple(
        _region(item, width=width, height=height) for item in rows(head.get("interiorRegions"))
    )
    if len(regions) > tracker.MAX_CONTOUR_POINTS:
        raise ValueError("Measured pixel regions exceed the native tile budget")
    x, y, w, h = face
    eligible = tuple(r for r in regions if x <= r[0] and r[0] + r[2] <= x + w and r[1] + r[3] <= y)
    columns: dict[tuple[int, int], list[Region]] = {}
    for region in eligible:
        columns.setdefault((region[0], region[2]), []).append(region)
    high = tuple(min(items, key=lambda r: r[1]) for items in columns.values())
    # The second model comes from actual occupied face-interior pixels, not a
    # second sparse band above the face, which can omit the available facial
    # texture. The accepted same-PTS detector associates these pixels; it does
    # not label any subsequent missing frame as an observed face.
    low = tuple(
        r
        for r in regions
        if x <= r[0] and r[0] + r[2] <= x + w and y <= r[1] and r[1] + r[3] <= y + h
    )
    if not high or not low:
        return None
    selected = np.asarray(
        [p for p in points if any(r[0] <= p[0] < r[0] + r[2] for r in high)], np.float32
    )
    if (
        len(selected) < 3
        or float(selected[:, 1].min()) != measured[0].visible_top
        or np.any(selected < tracker.LK_MARGIN_PIXELS)
        or np.any(selected >= (width - tracker.LK_MARGIN_PIXELS, height - tracker.LK_MARGIN_PIXELS))
    ):
        return None
    contour = tracker.ContourSeed(start, width, height, selected, high)
    return PartialPixelSeed(contour, high, low, str(head["status"]))


def track_pixels(
    seed: PartialPixelSeed,
    *,
    clock: VideoFrameClock,
    face_flags: tuple[bool, ...],
    frames: Iterable[tuple[int, Image]],
    resets: tuple[tuple[int, str], ...] = (),
) -> tuple[PixelObservation, ...]:
    """Require current image support and band agreement; never hold or reseed."""
    validate_clock(clock)
    if seed.contour.time != clock.pts[0] * clock.time_base or len(face_flags) != len(clock.pts):
        raise ValueError("Pixel seed or original face flags differ from the source clock")
    if any(pts not in clock.pts or reason not in RESET_REASONS for pts, reason in resets) or any(
        left[0] >= right[0] for left, right in pairwise(resets)
    ):
        raise ValueError("Pixel resets require exact ordered source boundaries")
    reset_map = dict(resets)
    first: Image | None = None
    upper_features = np.empty((0, 2), np.float32)
    anchor_features = np.empty((0, 2), np.float32)
    observations: list[PixelObservation] = []
    for index, (pts, image) in enumerate(frames):
        if index >= len(clock.pts) or pts != clock.pts[index]:
            raise ValueError("Decoded current pixel PTS differs")
        if image.dtype != np.uint8 or image.shape != (seed.contour.height, seed.contour.width, 3):
            raise ValueError("Decoded current pixel geometry differs")
        reason = reset_map.get(pts)
        if index and (pts - clock.pts[index - 1]) * clock.time_base > 1:
            reason = "unsupported-frame-extent"
        disagreement: float | None = None
        candidate: float | None = None
        upper_supported = upper_inliers = anchor_supported = anchor_inliers = 0
        if first is None:
            first = image
            upper_features = foreground_features(image, seed.upper_regions)
            anchor_features = foreground_features(image, seed.anchor_regions)
            upper_supported = upper_inliers = len(upper_features)
            anchor_supported = anchor_inliers = len(anchor_features)
            if min(len(upper_features), len(anchor_features)) < MIN_FEATURES:
                reason = "independent_texture_unavailable"
            if reason is None:
                candidate = float(seed.contour.points[:, 1].min())
        elif reason is None:
            upper_motion = derive_contour_motion(first, image, upper_features, seed.contour.points)
            anchor_motion = derive_contour_motion(
                first, image, anchor_features, seed.contour.points
            )
            upper_supported = int(np.count_nonzero(upper_motion.metrics.supported))
            anchor_supported = int(np.count_nonzero(anchor_motion.metrics.supported))
            upper_inliers, anchor_inliers = sum(upper_motion.inliers), sum(anchor_motion.inliers)
            reason = upper_motion.reason or anchor_motion.reason
            if upper_motion.contour is not None and anchor_motion.contour is not None:
                disagreement = float(
                    np.linalg.norm(upper_motion.contour - anchor_motion.contour, axis=1).max()
                )
                if disagreement > HEAD_CAPTION_Y_DEADBAND_PX:
                    reason = "competing_upper_and_head_motion"
                elif reason is None:
                    candidate = float(upper_motion.contour[:, 1].min())
        observations.append(
            PixelObservation(
                pts,
                face_flags[index],
                "measured-partial-pixel-seed" if index == 0 else "current-pixel-derived-candidate",
                candidate,
                len(upper_features),
                upper_supported,
                upper_inliers,
                len(anchor_features),
                anchor_supported,
                anchor_inliers,
                disagreement,
                reason,
            )
        )
        if reason is not None:
            return tuple(observations)
    if len(observations) != len(clock.pts):
        raise ValueError("Current pixel decoder ended before its declared clock")
    return tuple(observations)


def _finite_errors(values: Iterable[float]) -> list[float | None]:
    return [float(value) if math.isfinite(value) else None for value in values]


def _model_audit(
    features: Coordinates, points: Coordinates, motion: TextureMotion
) -> dict[str, object]:
    """Expose actual correspondence checks and the existing complete model decision.

    The existing API does not expose a rejected affine matrix. Such deformation
    values are explicitly unavailable, never inferred as a pass from inlier counts.
    For accepted models, the transferred measured control points expose the
    similarity's scale/rotation without fitting another motion backend.
    """
    supported = int(np.count_nonzero(motion.metrics.supported))
    inliers = np.asarray(motion.inliers, np.bool_)
    count = int(np.count_nonzero(inliers))
    source_span = np.ptp(features, axis=0)
    inlier_span = np.ptp(features[inliers], axis=0) if count else np.zeros(2)
    accepted = motion.reason is None and motion.contour is not None
    scale: float | None = None
    rotation: float | None = None
    if accepted and motion.contour is not None:
        original_delta, moved_delta = points[-1] - points[0], motion.contour[-1] - motion.contour[0]
        denominator = float(np.dot(original_delta, original_delta))
        if denominator <= 0:
            raise ValueError("Measured approximate upper controls must be distinct")
        real = float(np.dot(original_delta, moved_delta)) / denominator
        imaginary = (
            float(original_delta[0] * moved_delta[1] - original_delta[1] * moved_delta[0])
            / denominator
        )
        scale, rotation = (
            math.hypot(real, imaginary),
            abs(math.degrees(math.atan2(imaginary, real))),
        )
    return {
        "acceptedByExistingMotionContract": accepted,
        "reason": motion.reason,
        "sourceFeatures": features.tolist(),
        "currentFeatures": motion.features.tolist(),
        "forwardStatus": motion.metrics.forward_status.tolist(),
        "backwardStatus": motion.metrics.backward_status.tolist(),
        "canvasStatus": motion.metrics.in_bounds.tolist(),
        "roundtripError": _finite_errors(motion.metrics.roundtrip_error),
        "roundtripLimitPx": 0.5,
        "appearanceError": _finite_errors(motion.metrics.appearance_error),
        "appearanceLimit": 12,
        "supported": motion.metrics.supported.tolist(),
        "supportedCount": supported,
        "inliers": list(motion.inliers),
        "inlierCount": count,
        "inlierRatio": count / supported if supported else None,
        "residuals": list(motion.residuals),
        "minimumInliers": MIN_FEATURES,
        "minimumInlierRatio": 0.75,
        "sourceSpan": source_span.tolist(),
        "inlierSpan": inlier_span.tolist(),
        "minimumSourceSpanPx": 8,
        "minimumInlierSpanRatio": 0.5,
        "deformationGuard": {
            "status": "passed-by-existing-contract"
            if accepted
            else "not-exposed-for-rejected-model",
            "scaleFromAcceptedControls": scale,
            "rotationDegreesFromAcceptedControls": rotation,
            "scaleLimits": [0.9, 1.1],
            "rotationLimitDegrees": 15,
            "aggregateFailureReason": motion.reason,
        },
        "transferredControls": None if motion.contour is None else motion.contour.tolist(),
        "completeModelRequired": True,
    }


def face_feature_cohorts(image: Image, seed: PartialPixelSeed) -> tuple[Coordinates, Coordinates]:
    """Split actual occupied face-interior texture into disjoint distributed sets."""
    return _cohorts(foreground_features(image, seed.anchor_regions))


def _cohorts(features: Coordinates) -> tuple[Coordinates, Coordinates]:
    ordered = features[np.lexsort((features[:, 1], features[:, 0]))]
    return ordered[::2].copy(), ordered[1::2].copy()


def _similarity(controls: Coordinates, moved: Coordinates) -> tuple[float, float, Coordinates]:
    """Recover an accepted similarity from the same measured control endpoints."""
    delta, transported = controls[-1] - controls[0], moved[-1] - moved[0]
    denominator = float(np.dot(delta, delta))
    if denominator <= 0:
        raise ValueError("Accepted face support controls must be distinct")
    real = float(np.dot(delta, transported)) / denominator
    imaginary = float(delta[0] * transported[1] - delta[1] * transported[0]) / denominator
    linear = np.array([[real, -imaginary], [imaginary, real]], np.float32)
    offset = np.asarray(moved[0] - controls[0] @ linear.T, np.float32)
    return real, imaginary, offset


def _projected_interior(
    regions: tuple[Region, ...], controls: Coordinates, moved: Coordinates
) -> tuple[tuple[Region, ...], Coordinates]:
    """Take an inward-rounded square inside each transported measured tile.

    These are derived geometry, not a fresh foreground mask. No rectangular
    envelope around the whole face is filled; original gaps remain excluded.
    """
    real, imaginary, offset = _similarity(controls, moved)
    linear = np.array([[real, -imaginary], [imaginary, real]], np.float32)
    scale_squared = real * real + imaginary * imaginary
    divisor = abs(real) + abs(imaginary)
    if divisor == 0:
        raise ValueError("Accepted face support cannot have zero scale")
    projected: list[Region] = []
    corners: list[Coordinates] = []
    for x, y, w, h in regions:
        center = np.asarray((x + w / 2, y + h / 2), np.float32) @ linear.T + offset
        half = min(w, h) * scale_squared / (2 * divisor)
        left, top = math.ceil(float(center[0]) - half), math.ceil(float(center[1]) - half)
        right, bottom = math.floor(float(center[0]) + half), math.floor(float(center[1]) + half)
        projected.append((left, top, max(0, right - left), max(0, bottom - top)))
        corners.append(
            np.asarray(((x, y), (x + w, y), (x + w, y + h), (x, y + h)), np.float32) @ linear.T
            + offset
        )
    return tuple(projected), np.concatenate(corners)


def refresh_face_features(
    image: Image,
    seed: PartialPixelSeed,
    controls: Coordinates,
    moved: tuple[Coordinates, Coordinates],
    *,
    continuing: tuple[Coordinates, Coordinates] | None = None,
    spatial_stable: bool = False,
    local_support: bool = False,
) -> tuple[tuple[Coordinates, Coordinates], dict[str, object]]:
    """Select current texture only inside both fully accepted support projections.

    Calling requires complete dual-model acceptance and agreement. These current
    features must pass the unchanged correspondence/model guards next frame.
    Neither refreshed features nor transported tiles are new detected faces.
    """
    if local_support:
        if not spatial_stable or continuing is None:
            raise ValueError("Local refresh requires current supported spatial endpoints")
        return _refresh_local_features(image, seed, controls, moved, continuing=continuing)
    first, first_corners = _projected_interior(seed.anchor_regions, controls, moved[0])
    second, second_corners = _projected_interior(seed.anchor_regions, controls, moved[1])
    difference = float(np.linalg.norm(first_corners - second_corners, axis=1).max())
    height, width = image.shape[:2]
    shared: list[Region] = []
    for one, two in zip(first, second, strict=True):
        margin = tracker.LK_MARGIN_PIXELS
        x, y = max(one[0], two[0], margin), max(one[1], two[1], margin)
        right = min(one[0] + one[2], two[0] + two[2], width - margin)
        bottom = min(one[1] + one[3], two[1] + two[3], height - margin)
        if right > x and bottom > y:
            shared.append((x, y, right - x, bottom - y))
    reason: str | None = None
    if difference > HEAD_CAPTION_Y_DEADBAND_PX:
        reason = "competing_face_interior_geometry"
    elif not shared:
        reason = "current_face_interior_unavailable"
    # foreground_features supplies the existing bounded CPU feature selector.
    # Its mask comes from explicit derived tiles here, not native remeasurement.
    cohorts = (
        (np.empty((0, 2), np.float32), np.empty((0, 2), np.float32))
        if reason is not None
        else _cohorts(foreground_features(image, tuple(shared)))
    )
    selection: dict[str, object] | None = None
    if spatial_stable:
        if continuing is None:
            raise ValueError("Spatial refresh requires accepted current correspondence endpoints")
        if reason is None:
            current = select_current_stable_features(image, tuple(shared), continuing=continuing)
            cohorts, selection = current.cohorts, current.audit()
    if reason is None and min(len(cohorts[0]), len(cohorts[1])) < MIN_FEATURES:
        reason = "current_face_texture_unavailable"
    audit: dict[str, object] = {
        "status": "current-pixel-features-selected" if reason is None else "unsupported",
        "reason": reason,
        "supportPolicy": "intersection-of-two-accepted-measured-interior-projections-v1",
        "derivedSupportRegions": shared,
        "originalMeasuredInteriorRegions": seed.anchor_regions,
        "maximumInteriorProjectionDisagreementPx": difference,
        "modelAgreementLimitPx": HEAD_CAPTION_Y_DEADBAND_PX,
        "selectedFeatureCounts": [len(group) for group in cohorts],
        "nativeInteriorRemeasured": False,
        "newFaceObservation": False,
        "nextFrameRequiresCompleteModels": True,
    }
    if spatial_stable:
        audit["spatialSelection"] = selection
    return cohorts, audit


def _refresh_local_features(
    image: Image,
    seed: PartialPixelSeed,
    controls: Coordinates,
    moved: tuple[Coordinates, Coordinates],
    *,
    continuing: tuple[Coordinates, Coordinates],
) -> tuple[tuple[Coordinates, Coordinates], dict[str, object]]:
    """Project each measured owner with its independently validated current model."""
    split, owners = measured_region_owners(image, seed.anchor_regions)
    first, first_corners = _projected_interior(seed.anchor_regions, controls, moved[0])
    second, second_corners = _projected_interior(seed.anchor_regions, controls, moved[1])
    height, width = image.shape[:2]
    projected = tuple(
        _clipped_local_region(one if owner == 0 else two, width=width, height=height)
        for one, two, owner in zip(first, second, owners, strict=True)
    )
    reason: str | None = None
    selection: dict[str, object] | None = None
    ownership_error: str | None = None
    cohorts = (np.empty((0, 2), np.float32), np.empty((0, 2), np.float32))
    try:
        current = select_current_local_features(
            image, projected, tile_owners=owners, seed_split_y=split, continuing=continuing
        )
    except ValueError as exc:
        reason, ownership_error = "current_local_ownership_unsupported", str(exc)
    else:
        cohorts, selection = current.cohorts, current.audit()
        if min(len(cohorts[0]), len(cohorts[1])) < MIN_FEATURES:
            reason = "current_local_texture_unavailable"
            selection = None
    return cohorts, {
        "status": "current-pixel-features-selected" if reason is None else "unsupported",
        "reason": reason,
        "supportPolicy": "predeclared-owner-current-interior-projections-v1",
        "derivedSupportRegions": projected,
        "originalMeasuredInteriorRegions": seed.anchor_regions,
        "originalTileOwners": owners,
        "seedAreaMedianY": split,
        "ownershipError": ownership_error,
        "maximumCrossOwnerProjectionDifferencePx": float(
            np.linalg.norm(first_corners - second_corners, axis=1).max()
        ),
        "crossOwnerFullInteriorAgreementRequired": False,
        "currentTargetAgreementRequired": True,
        "modelAgreementLimitPx": HEAD_CAPTION_Y_DEADBAND_PX,
        "selectedFeatureCounts": [len(group) for group in cohorts],
        "spatialSelection": selection,
        "nativeInteriorRemeasured": False,
        "newFaceObservation": False,
        "nextFrameRequiresCompleteModels": False,
        "nextFrameRequiresCompletePrimaryAndCorroboration": True,
    }


def _clipped_local_region(region: Region, *, width: int, height: int) -> Region:
    x, y, w, h = region
    margin = tracker.LK_MARGIN_PIXELS
    left, top = max(x, margin), max(y, margin)
    right, bottom = min(x + w, width - margin), min(y + h, height - margin)
    return left, top, max(0, right - left), max(0, bottom - top)


def _continued_current_features(motion: TextureMotion) -> Coordinates:
    """Keep current independently valid strict-basis endpoints after acceptance."""
    if motion.fit is None:
        raise ValueError("Spatial refresh requires actual accepted fit evidence")
    selected = np.asarray(motion.fit.inliers, np.bool_) & motion.fit.metrics.supported
    return motion.fit.metrics.points[selected].copy()


def track_face_motion(
    seed: PartialPixelSeed,
    *,
    clock: VideoFrameClock,
    face_flags: tuple[bool, ...],
    frames: Iterable[tuple[int, Image]],
    resets: tuple[tuple[int, str], ...] = (),
    maximum_target_y_uncertainty_px: float | None = None,
    corroborated: bool = False,
    spatial_stable: bool = False,
    local_support: bool = False,
) -> tuple[FaceMotionObservation, ...]:
    """Transport an associated approximate apex using current facial pixels only.

    This is a separate derived policy. It does not turn the failed current upper
    contour into measured hair or turn original missing frames into detected faces.
    Both disjoint cohorts use current texture support. The opt-in corroborated
    policy requires one complete primary and distributed independent corroboration;
    the original policies require both complete models before feature refresh.
    A rejected model/reset ends the scope; no held position or detector reseed occurs.
    """
    validate_clock(clock)
    if local_support and not spatial_stable:
        raise ValueError("Measured local models require the distinct spatial role policy")
    if spatial_stable and not corroborated:
        raise ValueError("Spatial selection requires the distinct primary/corroboration policy")
    if corroborated and maximum_target_y_uncertainty_px is None:
        raise ValueError("Corroboration requires an explicit target Y budget")
    if (
        seed.contour.time != clock.pts[0] * clock.time_base
        or len(face_flags) != len(clock.pts)
        or not face_flags[0]
    ):
        raise ValueError("Approximate seed and original face clock differ")
    if any(pts not in clock.pts or reason not in RESET_REASONS for pts, reason in resets) or any(
        left[0] >= right[0] for left, right in pairwise(resets)
    ):
        raise ValueError("Approximate resets require exact ordered source boundaries")
    apex = float(seed.contour.points[:, 1].min())
    controls = seed.contour.points[seed.contour.points[:, 1] <= apex + HEAD_CAPTION_Y_DEADBAND_PX]
    if len(controls) < 3:
        raise ValueError("Measured approximate apex requires at least three real control points")
    previous: Image | None = None
    cohorts = (np.empty((0, 2), np.float32), np.empty((0, 2), np.float32))
    moved = (controls.copy(), controls.copy())
    observations: list[FaceMotionObservation] = []
    reset_map = dict(resets)
    for index, (pts, image) in enumerate(frames):
        if index >= len(clock.pts) or pts != clock.pts[index]:
            raise ValueError("Approximate decoded pixel PTS differs")
        if image.dtype != np.uint8 or image.shape != (seed.contour.height, seed.contour.width, 3):
            raise ValueError("Approximate decoded pixel geometry differs")
        reason = reset_map.get(pts)
        if index and (pts - clock.pts[index - 1]) * clock.time_base > 1:
            reason = "unsupported-frame-extent"
        audits: tuple[dict[str, object], ...] = ()
        difference: float | None = None
        estimate: float | None = None
        refresh: dict[str, object] = {"status": "not-attempted", "reason": reason}
        pair: ModelPairDecision | None = None
        if previous is None:
            if not local_support:
                cohorts = face_feature_cohorts(image, seed)
            initial_selection: dict[str, object] | None = None
            if spatial_stable:
                if local_support:
                    split, owners = measured_region_owners(image, seed.anchor_regions)
                    local = select_current_local_features(
                        image, seed.anchor_regions, tile_owners=owners, seed_split_y=split
                    )
                    cohorts, initial_selection = local.cohorts, local.audit()
                else:
                    selection = select_current_stable_features(image, seed.anchor_regions)
                    cohorts, initial_selection = selection.cohorts, selection.audit()
            refresh = {
                "status": "original-measured-seed",
                "selectedFeatureCounts": [len(group) for group in cohorts],
                "nativeInteriorRemeasured": False,
                "newFaceObservation": False,
            }
            if spatial_stable:
                refresh["spatialSelection"] = initial_selection
            if min(len(cohorts[0]), len(cohorts[1])) < MIN_FEATURES:
                reason = "independent_face_texture_unavailable"
            if reason is None:
                estimate = apex
        elif reason is None:
            first = derive_contour_motion(previous, image, cohorts[0], moved[0])
            second = derive_contour_motion(previous, image, cohorts[1], moved[1])
            audits = (
                _model_audit(cohorts[0], moved[0], first),
                _model_audit(cohorts[1], moved[1], second),
            )
            current = (first.contour, second.contour)
            reason = first.reason or second.reason
            if maximum_target_y_uncertainty_px is not None:
                if first.fit is None or second.fit is None:
                    raise ValueError("Approximate target Y requires actual typed fit evidence")
                decisions = tuple(
                    approximate_model(
                        motion.fit,
                        points,
                        maximum_target_y_uncertainty_px=maximum_target_y_uncertainty_px,
                        width=seed.contour.width,
                        height=seed.contour.height,
                    )
                    for motion, points in ((first, moved[0]), (second, moved[1]))
                    if motion.fit is not None
                )
                current = (decisions[0].transferred_controls, decisions[1].transferred_controls)
                reason = decisions[0].reason or decisions[1].reason
                audits = tuple(
                    {
                        **audit,
                        "sourceControls": points.tolist(),
                        "fitMatrix": motion.fit.matrix if motion.fit is not None else None,
                        "approximateHeadContract": decision.audit,
                    }
                    for audit, motion, points, decision in zip(
                        audits, (first, second), moved, decisions, strict=True
                    )
                )
                if corroborated:
                    pair = _model_pair(
                        (first.fit, second.fit),
                        moved,
                        (decisions[0], decisions[1]),
                        width=seed.contour.width,
                        height=seed.contour.height,
                    )
                    current = pair.transferred_controls or (None, None)
                    reason = pair.reason
            if current[0] is not None and current[1] is not None:
                difference = float(np.linalg.norm(current[0] - current[1], axis=1).max())
                if difference > HEAD_CAPTION_Y_DEADBAND_PX:
                    reason = "competing_face_motion"
                elif reason is None:
                    moved = (current[0], current[1])
                    if spatial_stable:
                        cohorts, refresh = refresh_face_features(
                            image,
                            seed,
                            controls,
                            moved,
                            continuing=(
                                _continued_current_features(first),
                                _continued_current_features(second),
                            ),
                            spatial_stable=True,
                            local_support=local_support,
                        )
                        refresh["selectionHistoryPTS"] = clock.pts[index - 1]
                        refresh["continuedBasisOriginalIndices"] = [
                            np.flatnonzero(
                                np.asarray(motion.fit.inliers, np.bool_)
                                & motion.fit.metrics.supported
                            ).tolist()
                            for motion in (first, second)
                            if motion.fit is not None
                        ]
                    else:
                        cohorts, refresh = refresh_face_features(image, seed, controls, moved)
                    refresh_reason = refresh["reason"]
                    reason = None if refresh_reason is None else str(refresh_reason)
                    if reason is None:
                        estimate = (
                            pair.estimated_upper_y
                            if pair is not None
                            else float((current[0][:, 1].min() + current[1][:, 1].min()) / 2)
                        )
        if refresh["status"] == "not-attempted":
            refresh["reason"] = reason
        refresh["sourcePTS"] = pts
        if pair is not None:
            refresh["modelPairDecision"] = pair.audit
            refresh["nextFrameRequiresCompleteModels"] = False
            refresh["nextFrameRequiresCompletePrimaryAndCorroboration"] = True
        observations.append(
            FaceMotionObservation(
                pts, face_flags[index], estimate, difference, audits, refresh, reason
            )
        )
        if reason is not None:
            return tuple(observations)
        previous = image
    if len(observations) != len(clock.pts):
        raise ValueError("Approximate current pixel decoder ended before its clock")
    return tuple(observations)


def approximate_association(
    value: object, *, source_hash: str, clock: VideoFrameClock, face: Face, inputs: dict[str, str]
) -> dict[str, object]:
    """Bind an explicit approximate seed association; no semantic segmentation claim."""
    path = tracker._reference(value, inputs)
    observation = record(tracker._read(path))
    operation = record(observation.get("operation"))
    visual = record(observation.get("visualObservation"))
    source = record(observation.get("source"))
    if (
        observation.get("version") != 1
        or source.get("expectedSHA256FromExistingBoundConfig") != source_hash
        or source.get("identityBeforeAfterEqual") is not True
        or operation.get("commandExit") != 0
        or operation.get("framesWritten") != 1
        or operation.get("showinfoFirstPTS") != clock.pts[0]
        or operation.get("timeBase") != str(clock.time_base)
        or visual.get("directlyViewed") is not True
        or visual.get("pixelPerfectHairSegmentation") is not False
        or tuple(rows(visual.get("acceptedOriginalDetectorRectangle"))) != face
    ):
        raise ValueError("Approximate association must use the actual source-bound seed image")
    image = operation.get("image")
    if not isinstance(image, str):
        raise ValueError("Approximate association image is required")
    image_path = Path(image).absolute()
    if (
        image_path != image_path.resolve()
        or not image_path.is_relative_to(Path.cwd().resolve() / "edits" / "work")
        or not 0 < image_path.stat().st_size <= 16 * 1024 * 1024
    ):
        raise ValueError("Approximate association requires a bounded canonical work image")
    tracker._reference({"path": str(image_path), "sha256": operation.get("imageSHA256")}, inputs)
    return {
        "kind": "source-image-associated-approximate-upper",
        "observation": str(path),
        "image": str(image_path),
        "seedPTS": clock.pts[0],
        "approximateAssociationOnly": True,
        "semanticHeadVerified": False,
    }


def _number(value: object) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
        raise ValueError("Finite pixel geometry is required")
    return float(value)


def _region(value: object, *, width: int, height: int) -> Region:
    raw = record(value)
    if set(raw) != {"x", "y", "width", "height"}:
        raise ValueError("Native occupied tiles require exact fields")
    x, y, w, h = (integer(raw[k]) for k in ("x", "y", "width", "height"))
    if not 0 <= x < x + w <= width or not 0 <= y < y + h <= height:
        raise ValueError("Measured occupied tile exceeds the source canvas")
    return x, y, w, h


def _spatial_guard_defaults() -> dict[str, object]:
    """Immutable meaning of the new report's repeated guard references."""
    return {
        "model": {
            "roundtripLimitPx": 0.5,
            "appearanceLimit": 12,
            "minimumInliers": MIN_FEATURES,
            "minimumInlierRatio": 0.75,
            "minimumSourceSpanPx": 8,
            "minimumInlierSpanRatio": 0.5,
            "completeModelRequired": True,
        },
        "approximateHeadContract": {
            "minimumInliers": MIN_FEATURES,
            "minimumInlierRatio": 0.75,
            "minimumSourceSpanPx": 8,
            "minimumSupportSpanRatio": 0.5,
            "notTrueHairErrorGuarantee": True,
            "uncertaintyScope": "one-current-correspondence-transition",
            "accumulatedHairErrorBounded": False,
        },
    }


def _compact_guard_fields(value: object, name: str) -> dict[str, object]:
    raw = record(value)
    defaults = record(_spatial_guard_defaults()[name])
    if any(not _same_audit(raw.get(key), expected) for key, expected in defaults.items()):
        raise ValueError("Spatial model guard defaults differ from their original evidence")
    return {
        **{key: item for key, item in raw.items() if key not in defaults},
        "sharedGuardRef": name,
    }


def _expand_guard_fields(value: object, name: str) -> dict[str, object]:
    raw = record(value)
    defaults = record(_spatial_guard_defaults()[name])
    if raw.get("sharedGuardRef") != name or set(raw).intersection(defaults):
        raise ValueError("Spatial guard reference must be exact and unshadowed")
    return {**defaults, **{key: item for key, item in raw.items() if key != "sharedGuardRef"}}


def _compact_spatial_model(value: object) -> dict[str, object]:
    model = _compact_guard_fields(value, "model")
    model["approximateHeadContract"] = _compact_guard_fields(
        model["approximateHeadContract"], "approximateHeadContract"
    )
    return model


def _expand_spatial_model(value: object) -> dict[str, object]:
    model = _expand_guard_fields(value, "model")
    model["approximateHeadContract"] = _expand_guard_fields(
        model["approximateHeadContract"], "approximateHeadContract"
    )
    return model


def _compact_spatial_observations(
    observations: tuple[FaceMotionObservation, ...] | tuple[PixelObservation, ...],
) -> list[dict[str, object]]:
    return [_compact_spatial_observation(item) for item in observations]


def _compact_spatial_observation(
    item: FaceMotionObservation | PixelObservation,
) -> dict[str, object]:
    if not isinstance(item, FaceMotionObservation):
        raise ValueError("Spatial reports require actual face motion observations")
    observation: dict[str, object] = asdict(item)
    observation["models"] = [_compact_spatial_model(model) for model in item.models]
    refresh = record(observation["feature_refresh"])
    if "originalMeasuredInteriorRegions" in refresh:
        refresh.pop("originalMeasuredInteriorRegions")
        refresh["originalMeasuredInteriorRegionsRef"] = "seedAnchorRegions"
    return observation


def _write_spatial_models(report: dict[str, object], output: Path) -> None:
    """Store each actual model once in separately byte-bounded typed records."""
    observations = tuple(record(item) for item in rows(report["observations"]))
    models = tuple(chain.from_iterable(rows(item["models"]) for item in observations))
    if len(models) > MAX_SPATIAL_MODEL_RECORDS * SPATIAL_MODELS_PER_RECORD:
        raise ValueError("Spatial model count exceeds its bounded records")
    references: list[dict[str, object]] = []
    total = 0
    for start in range(0, len(models), SPATIAL_MODELS_PER_RECORD):
        encoded = json.dumps(
            {
                "version": 1,
                "protocol": LOCAL_PROTOCOL
                if report.get("protocol") == LOCAL_PROTOCOL
                else SPATIAL_PROTOCOL,
                "models": models[start : start + SPATIAL_MODELS_PER_RECORD],
            },
            separators=(",", ":"),
            allow_nan=False,
        )
        size = len(encoded.encode("utf-8"))
        total += size
        if size > MAX_APPROX_JSON_BYTES or total > MAX_SPATIAL_MODEL_TOTAL_BYTES:
            raise ValueError("Spatial model record exceeds its bounded JSON artifact budget")
        path = output / f"models-{len(references):04d}.json"
        with path.open("x", encoding="utf-8") as handle:
            handle.write(encoded)
        path.chmod(0o600)
        references.append({"path": str(path), "sha256": _uncached_file_digest(path)})
    index = 0
    for observation in observations:
        count = len(rows(observation["models"]))
        observation["models"] = [
            {"record": item // SPATIAL_MODELS_PER_RECORD, "index": item % SPATIAL_MODELS_PER_RECORD}
            for item in range(index, index + count)
        ]
        index += count
    report["spatialModelRecords"] = references
    report["spatialModelArtifactRoot"] = str(output)


def _read_spatial_models(report: dict[str, object]) -> tuple[tuple[object, ...], ...]:
    root_name = report.get("spatialModelArtifactRoot")
    if not isinstance(root_name, str):
        raise ValueError("Spatial model records require their exact artifact root")
    root = Path(root_name)
    if (
        not root.is_absolute()
        or root != root.resolve()
        or not root.is_relative_to(Path.cwd().resolve() / "edits" / "work")
    ):
        raise ValueError("Spatial model artifact root must be canonical repository work")
    references = rows(report.get("spatialModelRecords"))
    if len(references) > MAX_SPATIAL_MODEL_RECORDS:
        raise ValueError("Spatial model records exceed their count budget")
    result: list[tuple[object, ...]] = []
    total = 0
    for index, value in enumerate(references):
        reference = record(value)
        path = root / f"models-{index:04d}.json"
        if reference.get("path") != str(path):
            raise ValueError("Spatial model record name or root differs")
        bound = tracker._reference(reference, {})
        total += bound.stat().st_size
        if total > MAX_SPATIAL_MODEL_TOTAL_BYTES:
            raise ValueError("Spatial model records exceed their total byte budget")
        raw = record(read_approximate_json(bound))
        if (
            set(raw) != {"version", "protocol", "models"}
            or type(raw["version"]) is not int
            or raw["version"] != 1
            or raw["protocol"]
            != (LOCAL_PROTOCOL if report.get("protocol") == LOCAL_PROTOCOL else SPATIAL_PROTOCOL)
        ):
            raise ValueError("Unsupported spatial model record")
        models = rows(raw["models"])
        if not 1 <= len(models) <= SPATIAL_MODELS_PER_RECORD:
            raise ValueError("Spatial model record count differs")
        result.append(tuple(models))
    return tuple(result)


def _referenced_spatial_model(
    value: object, models: tuple[tuple[object, ...], ...]
) -> dict[str, object]:
    reference = record(value)
    if set(reference) != {"record", "index"}:
        raise ValueError("Spatial model reference requires exact coordinates")
    record_index, index = integer(reference["record"]), integer(reference["index"])
    if not 0 <= record_index < len(models) or not 0 <= index < len(models[record_index]):
        raise ValueError("Spatial model reference exceeds its record")
    return _expand_spatial_model(models[record_index][index])


def _spatial_selection_counts(value: object, length: int) -> tuple[int, ...]:
    counts = tuple(integer(item) for item in rows(value))
    if len(counts) != length or any(item < 0 for item in counts):
        raise ValueError("Spatial selection counts require exact nonnegative cells")
    return counts


def _validate_spatial_selection(observation: dict[str, object], previous_pts: int | None) -> None:
    refresh = record(observation["feature_refresh"])
    if refresh.get("status") not in {"original-measured-seed", "current-pixel-features-selected"}:
        if refresh.get("spatialSelection") is not None:
            raise ValueError("Rejected spatial motion cannot refresh current features")
        return
    selection = record(refresh.get("spatialSelection"))
    if (
        selection.get("policy") != SPATIAL_POLICY
        or selection.get("selectionBeforeNextFrame") is not True
        or selection.get("nextFrameRevalidatesEverySelectedPoint") is not True
        or selection.get("newFaceObservation") is not False
        or (previous_pts is not None and refresh.get("selectionHistoryPTS") != previous_pts)
    ):
        raise ValueError("Spatial selection must preserve its current pre-frame history")
    counts = _spatial_selection_counts(selection.get("selectedFeatureCounts"), 2)
    continued = _spatial_selection_counts(selection.get("cellSelectedContinued"), 4)
    fresh = _spatial_selection_counts(selection.get("cellSelectedFresh"), 4)
    area = _spatial_selection_counts(selection.get("cellEligiblePixels"), 4)
    old_candidates = _spatial_selection_counts(selection.get("cellContinuedCandidates"), 4)
    new_candidates = _spatial_selection_counts(selection.get("cellFreshCandidates"), 4)
    _spatial_selection_counts(selection.get("areaMedianSplit"), 2)
    if any(
        old > available_old or new > available_new or (pixels == 0 and old + new != 0)
        for old, new, available_old, available_new, pixels in zip(
            continued, fresh, old_candidates, new_candidates, area, strict=True
        )
    ):
        raise ValueError("Spatial selected points require available current occupied support")
    if max(counts) > 128 or sum(counts) != sum(continued) + sum(fresh):
        raise ValueError("Spatial selected counts exceed their unchanged feature budget")
    origins = tuple(rows(item) for item in rows(selection.get("selectedOrigins")))
    if len(origins) != 2 or tuple(len(group) for group in origins) != counts:
        raise ValueError("Spatial selected origins differ from their current cohorts")
    _validate_spatial_origins(origins)


def _spatial_origin(value: object) -> tuple[str, int]:
    raw = rows(value)
    if (
        len(raw) != 2
        or not isinstance(raw[0], str)
        or raw[0] not in {"fresh", "continued-0", "continued-1"}
    ):
        raise ValueError("Spatial origin requires its exact current support identity")
    index = integer(raw[1])
    if not 0 <= index < 256:
        raise ValueError("Spatial origin index exceeds its current feature budget")
    return raw[0], index


def _validate_spatial_origins(origins: tuple[list[object], ...]) -> None:
    flattened = tuple(chain.from_iterable(origins))
    identities = tuple(_spatial_origin(item) for item in flattened)
    if len(set(identities)) != len(identities):
        raise ValueError("Spatial independent cohorts may not share selected origins")
    for cohort, group in enumerate(origins):
        if any(_spatial_origin(item)[0] not in {"fresh", f"continued-{cohort}"} for item in group):
            raise ValueError("Spatial selected origin is not current independent support")


def spatial_observations(
    value: object, *, local_support: bool = False
) -> tuple[dict[str, object], ...]:
    """Losslessly expand the distinct spatial report's typed shared references."""
    report = record(value)
    if (
        report.get("protocol") != (LOCAL_PROTOCOL if local_support else SPATIAL_PROTOCOL)
        or report.get("featureSelectionPolicy") != SPATIAL_POLICY
        or not _same_audit(report.get("spatialModelGuardDefaults"), _spatial_guard_defaults())
    ):
        raise ValueError("Spatial report policy or guard references differ")
    result: list[dict[str, object]] = []
    models = _read_spatial_models(report)
    used: list[tuple[int, int]] = []
    for value in rows(report.get("observations")):
        observation, references = _expand_spatial_observation(
            value, models, report["seedAnchorRegions"]
        )
        used.extend(references)
        _validate_spatial_selection(
            observation, None if not result else integer(result[-1]["source_pts"])
        )
        result.append(observation)
    if local_support:
        _validate_local_material(report, tuple(result))
    expected = tuple(
        chain.from_iterable(
            tuple((record_index, index) for index in range(len(group)))
            for record_index, group in enumerate(models)
        )
    )
    if tuple(used) != expected:
        raise ValueError("Spatial model references must consume all records once in source order")
    return tuple(result)


def _validate_local_material(
    report: dict[str, object], observations: tuple[dict[str, object], ...]
) -> None:
    if report.get("localSupportPolicy") != LOCAL_POLICY:
        raise ValueError("Explicit predeclared local report policy is required")
    original = tuple(
        tuple(integer(v) for v in rows(tile)) for tile in rows(report.get("seedAnchorRegions"))
    )
    if any(len(tile) != 4 for tile in original):
        raise ValueError("Local measured tile geometry differs")
    regions = tuple((tile[0], tile[1], tile[2], tile[3]) for tile in original)
    geometry = record(report.get("localGeometry"))
    if not regions and not observations:
        return
    split, expected_owners = measured_tile_partition(
        regions, width=integer(geometry.get("width")), height=integer(geometry.get("height"))
    )
    prior_selection: dict[str, object] | None = None
    for observation in observations:
        refresh = record(observation["feature_refresh"])
        models = rows(observation.get("models"))
        if (
            prior_selection is not None
            and models
            and not _same_audit(
                [record(model).get("sourceFeatures") for model in models],
                prior_selection["currentSelectedFeatures"],
            )
        ):
            raise ValueError("Local current correspondences differ from prior selected pixels")
        if refresh.get("status") not in {
            "original-measured-seed",
            "current-pixel-features-selected",
        }:
            prior_selection = None
            continue
        selection = record(refresh.get("spatialSelection"))
        support = record(selection.get("predeclaredLocalSupport"))
        owners = tuple(integer(v) for v in rows(support.get("originalTileOwners")))
        if (
            support.get("policy") != LOCAL_POLICY
            or support.get("ownershipSelectedBeforeFit") is not True
            or support.get("fitDependentSelection") is not False
            or support.get("maximumFeaturesPerOwner") != 128
            or owners != expected_owners
            or support.get("seedAreaMedianY") != split
        ):
            raise ValueError("Local models require fixed measured tile ownership")
        current_regions = (
            regions
            if refresh["status"] == "original-measured-seed"
            else tuple(_local_region(tile) for tile in rows(refresh.get("derivedSupportRegions")))
        )
        if len(current_regions) != len(regions):
            raise ValueError("Local current projection lost original tile identities")
        if refresh["status"] == "current-pixel-features-selected" and (
            refresh.get("supportPolicy") != "predeclared-owner-current-interior-projections-v1"
            or refresh.get("originalTileOwners") != list(expected_owners)
            or refresh.get("seedAreaMedianY") != split
            or refresh.get("crossOwnerFullInteriorAgreementRequired") is not False
            or refresh.get("currentTargetAgreementRequired") is not True
            or refresh.get("modelAgreementLimitPx") != HEAD_CAPTION_Y_DEADBAND_PX
            or refresh.get("nextFrameRequiresCompletePrimaryAndCorroboration") is not True
        ):
            raise ValueError("Local projection must preserve its distinct supported role policy")
        _validate_local_selected(selection, support, current_regions, owners)
        prior_selection = selection


def _local_region(value: object) -> Region:
    values = tuple(integer(v) for v in rows(value))
    if len(values) != 4:
        raise ValueError("Local projected tile requires exact geometry")
    return values[0], values[1], values[2], values[3]


def _validate_local_selected(
    selection: dict[str, object],
    support: dict[str, object],
    regions: tuple[Region, ...],
    owners: tuple[int, ...],
) -> None:
    _validate_spatial_selection(
        {
            "feature_refresh": {
                "status": "original-measured-seed",
                "spatialSelection": support.get("originalSelection"),
            }
        },
        None,
    )
    groups = tuple(
        _selected_coordinates(value) for value in rows(selection.get("currentSelectedFeatures"))
    )
    originals = tuple(
        _selected_coordinates(value) for value in rows(support.get("originalSelectedFeatures"))
    )
    references = tuple(rows(value) for value in rows(support.get("originalSelectedIndices")))
    counts = tuple(integer(value) for value in rows(selection.get("selectedFeatureCounts")))
    origins = tuple(rows(value) for value in rows(selection.get("selectedOrigins")))
    original_origins = tuple(
        rows(value) for value in rows(record(support["originalSelection"])["selectedOrigins"])
    )
    if (
        len(groups) != 2
        or len(originals) != 2
        or len(references) != 2
        or tuple(len(group) for group in groups) != counts
    ):
        raise ValueError("Local selected material counts differ")
    flattened = tuple(chain.from_iterable(_local_reference_rows(group) for group in references))
    excluded = _local_reference_rows(rows(support.get("ownerBudgetExcludedIndices")))
    expected = tuple(
        chain.from_iterable(
            tuple((cohort, index) for index in range(len(group)))
            for cohort, group in enumerate(originals)
        )
    )
    if len(set((*flattened, *excluded))) != len(expected) or set((*flattened, *excluded)) != set(
        expected
    ):
        raise ValueError("Local selection may not omit original selected material")
    for owner, (group, reference, group_origins) in enumerate(
        zip(groups, references, origins, strict=True)
    ):
        source_references = _local_reference_rows(reference)
        _validate_local_group(group, source_references, originals, regions, owners, owner)
        if any(
            not _same_audit(origin, original_origins[cohort][index])
            for origin, (cohort, index) in zip(group_origins, source_references, strict=True)
        ):
            raise ValueError("Local regrouping changed an original current point identity")
    if any(
        not any(
            len(group) == 128
            and any(
                identity == owner
                and tile[0] <= originals[cohort][index][0] < tile[0] + tile[2]
                and tile[1] <= originals[cohort][index][1] < tile[1] + tile[3]
                for tile, identity in zip(regions, owners, strict=True)
            )
            for owner, group in enumerate(groups)
        )
        for cohort, index in excluded
    ):
        raise ValueError("Local original material may only be excluded by its pre-fit owner cap")


def _selected_coordinates(value: object) -> Coordinates:
    items = rows(value)
    if len(items) > 128 or any(len(rows(item)) != 2 for item in items):
        raise ValueError("Local selected pixels exceed their bounded coordinate budget")
    return np.asarray(
        tuple(tuple(_number(v) for v in rows(item)) for item in items), np.float32
    ).reshape(-1, 2)


def _local_reference_rows(values: list[object]) -> tuple[tuple[int, int], ...]:
    pairs = tuple(tuple(integer(v) for v in rows(value)) for value in values)
    if any(len(pair) != 2 or pair[0] > 1 or pair[1] >= 128 for pair in pairs):
        raise ValueError("Local source indices exceed original selected cohorts")
    return tuple((pair[0], pair[1]) for pair in pairs)


def _validate_local_group(
    group: Coordinates,
    references: tuple[tuple[int, int], ...],
    originals: tuple[Coordinates, ...],
    regions: tuple[Region, ...],
    owners: tuple[int, ...],
    owner: int,
) -> None:
    if len(group) != len(references):
        raise ValueError("Local point references differ from selected pixels")
    for point, (cohort, index) in zip(group, references, strict=True):
        if index >= len(originals[cohort]) or not np.array_equal(point, originals[cohort][index]):
            raise ValueError("Local selected point is not original current material")
        matches = tuple(
            identity
            for tile, identity in zip(regions, owners, strict=True)
            if tile[0] <= point[0] < tile[0] + tile[2] and tile[1] <= point[1] < tile[1] + tile[3]
        )
        if matches != (owner,):
            raise ValueError("Local point lacks its unique predeclared measured owner")


def _expand_spatial_observation(
    value: object,
    models: tuple[tuple[object, ...], ...],
    seed_regions: object,
) -> tuple[dict[str, object], tuple[tuple[int, int], ...]]:
    observation = record(value).copy()
    references = rows(observation.get("models"))
    observation["models"] = [_referenced_spatial_model(model, models) for model in references]
    used = tuple(
        (integer(record(item)["record"]), integer(record(item)["index"])) for item in references
    )
    refresh = record(observation["feature_refresh"]).copy()
    if "originalMeasuredInteriorRegionsRef" in refresh:
        if (
            refresh.pop("originalMeasuredInteriorRegionsRef") != "seedAnchorRegions"
            or "originalMeasuredInteriorRegions" in refresh
        ):
            raise ValueError("Spatial interior reference differs from its genuine seed")
        refresh["originalMeasuredInteriorRegions"] = seed_regions
    observation["feature_refresh"] = refresh
    return observation, used


def _validate_spatial_policy(value: object) -> None:
    policy = record(value)
    if (
        set(policy) != {"version", "kind"}
        or type(policy["version"]) is not int
        or policy["version"] != 1
        or policy["kind"] != SPATIAL_POLICY
    ):
        raise ValueError("Explicit current spatial feature policy is required")


def run(
    config: Path,
    output: Path,
    *,
    approximate: bool = False,
    approximate_target_y: bool = False,
    corroborated: bool = False,
    spatial_stable: bool = False,
    local_support: bool = False,
) -> Path:
    """Read pinned actual evidence and emit a bounded, unapproved diagnostic."""
    started = monotonic()
    root = Path.cwd().resolve()
    if (
        output != output.resolve()
        or not output.is_relative_to(root / "edits" / "work")
        or output.exists()
    ):
        raise ValueError("New canonical repository work output is required")
    raw = record(tracker._read(config))
    spatial_stable = spatial_stable or local_support
    corroborated = corroborated or spatial_stable
    approximate_target_y = approximate_target_y or corroborated
    approximate = approximate or approximate_target_y
    extra = {"approximateAssociation"} if approximate else set()
    if approximate_target_y:
        extra.add("approximateYPolicy")
    if spatial_stable:
        extra.add("spatialFeaturePolicy")
        _validate_spatial_policy(raw.get("spatialFeaturePolicy"))
    if local_support:
        extra.add("localFeaturePolicy")
        _validate_local_policy(raw.get("localFeaturePolicy"))
    if (
        set(raw)
        != extra
        | {
            "version",
            "source",
            "nativeExecutable",
            "nativeRequest",
            "nativeResponse",
            "detectorRequest",
            "detectorResponse",
            "ffmpeg",
            "clock",
            "width",
            "height",
            "resets",
        }
        or type(raw["version"]) is not int
        or raw["version"] != 1
    ):
        raise ValueError("Unsupported partial pixel configuration")
    inputs = {str(config): _uncached_file_digest(config)}
    source = tracker._reference(raw["source"], inputs)
    tracker._reference(raw["nativeExecutable"], inputs)
    native_request = tracker._reference(raw["nativeRequest"], inputs)
    native_response = tracker._reference(raw["nativeResponse"], inputs)
    detector_request = tracker._reference(raw["detectorRequest"], inputs)
    detector_response = tracker._reference(raw["detectorResponse"], inputs)
    ffmpeg = tracker._reference(raw["ffmpeg"], inputs)
    clock_raw = record(raw["clock"])
    base = clock_raw.get("timeBase")
    if not isinstance(base, str) or set(clock_raw) != {"pts", "timeBase", "lastDuration"}:
        raise ValueError("Exact original pixel clock is required")
    clock = VideoFrameClock(
        tuple(integer(p) for p in rows(clock_raw["pts"])),
        Fraction(base),
        integer(clock_raw["lastDuration"]),
    )
    validate_clock(clock)
    width, height = integer(raw["width"]), integer(raw["height"])
    budget: float | None = None
    if approximate_target_y:
        policy = record(raw.get("approximateYPolicy"))
        if (
            set(policy) != {"version", "maximumTargetYModelUncertaintyPx"}
            or type(policy["version"]) is not int
            or policy["version"] != 1
        ):
            raise ValueError("Explicit target Y approximation policy is required")
        budget = _number(policy["maximumTargetYModelUncertaintyPx"])
        if not 0 < budget <= target_y_budget(height):
            raise ValueError("Target Y budget exceeds its shared clearance allocation")
    if record(tracker._read(detector_request)).get("path") != str(source):
        raise ValueError("Detector request source binding differs")
    face, flags = detector_seed(tracker._read(detector_response), clock=clock)
    association = (
        approximate_association(
            raw["approximateAssociation"],
            source_hash=inputs[str(source)],
            clock=clock,
            face=face,
            inputs=inputs,
        )
        if approximate
        else None
    )
    if tracker._read(native_request) != seed_request(source=source, clock=clock, face=face):
        raise ValueError("Native request must use the actual same-PTS detector")
    seed = read_seed(
        tracker._read(native_response), clock=clock, width=width, height=height, face=face
    )
    paths = (*upper._code_paths(), Path(__file__).resolve())
    inputs.update({str(path): _uncached_file_digest(path) for path in paths})
    resets = tuple(
        (integer(record(r).get("sourcePTS")), str(record(r).get("reason")))
        for r in rows(raw["resets"])
    )
    observations: tuple[FaceMotionObservation, ...] | tuple[PixelObservation, ...] = ()
    if seed is not None:
        if approximate:
            observations = track_face_motion(
                seed,
                clock=clock,
                face_flags=flags,
                frames=tracker._decode(source, clock, seed.contour, ffmpeg),
                resets=resets,
                maximum_target_y_uncertainty_px=budget,
                corroborated=corroborated,
                spatial_stable=spatial_stable,
                local_support=local_support,
            )
        else:
            observations = track_pixels(
                seed,
                clock=clock,
                face_flags=flags,
                frames=tracker._decode(source, clock, seed.contour, ffmpeg),
                resets=resets,
            )
    upper._pins(inputs)
    complete = len(observations) == len(clock.pts) and observations[-1].reason is None
    failure = (
        "measured_independent_upper_seed_unavailable" if seed is None else observations[-1].reason
    )
    output.mkdir(mode=0o700)
    report = {
        "protocol": LOCAL_PROTOCOL
        if local_support
        else SPATIAL_PROTOCOL
        if spatial_stable
        else CORROBORATED_PROTOCOL
        if corroborated
        else UNCERTAIN_PROTOCOL
        if approximate_target_y
        else APPROX_PROTOCOL
        if approximate
        else PROTOCOL,
        "inputs": inputs,
        "requestedFrames": len(clock.pts),
        "observedFrames": len(observations),
        "completeTrack": complete,
        "nativeStatus": None if seed is None else seed.native_status,
        "seedUpperRegions": [] if seed is None else seed.upper_regions,
        "seedAnchorRegions": [] if seed is None else seed.anchor_regions,
        "seedAnchorPolicy": "same-pts-measured-face-interior-v1",
        "seedDetectorRegion": face,
        "observations": [asdict(o) for o in observations],
        "measurementRequestReason": failure,
        "elapsedSeconds": monotonic() - started,
        "processPeakRSSBytes": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
        "derivedPolicySupport": True,
        "semanticHeadVerified": False,
        "headNearVerified": False,
        "captionPlacementApproved": False,
        "qualityApproved": False,
        "independentPixelReplayVerified": False,
    }
    if approximate:
        report.update(
            {
                "approximateAssociation": association,
                "seedAnchorPolicy": "two-disjoint-current-face-interior-models-v1",
                "currentMeasuredUpperPixels": False,
                "originalFaceFlagsPreserved": True,
                "modelAgreementLimitPx": HEAD_CAPTION_Y_DEADBAND_PX,
                "estimatedHeadPolicy": "derived-face-pixel-motion-approximation",
                "featureRefreshPolicy": "dual-accepted-current-interior-only-v1",
            }
        )
    if approximate_target_y:
        report.update(
            {
                "estimatedHeadPolicy": "bounded-face-pixel-target-y-approximation-v1",
                "featureRefreshPolicy": "dual-accepted-approximate-current-interior-only-v1",
                "targetYModelUncertainty": {
                    "configuredBudgetPx": budget,
                    "kind": "conditional-correspondence-perturbation-bound",
                    "allocationPolicy": "one-third-of-scaled-caption-head-gap-v1",
                    "styleHeadGapPx": HEAD_CAPTION_GAP_PX,
                    "styleReferenceHeight": HEAD_CAPTION_REFERENCE_HEIGHT,
                    "maximumAllocatedBudgetPx": target_y_budget(height),
                    "isTrueHairPositionErrorGuarantee": False,
                    "isCaptionYDeadband": False,
                    "uncertaintyScope": "one-current-correspondence-transition",
                    "accumulatedHairErrorBounded": False,
                },
            }
        )
    if corroborated:
        report.update(
            {
                "estimatedHeadPolicy": "primary-face-target-with-independent-corroboration-v1",
                "featureRefreshPolicy": "primary-corroborated-current-interior-only-v1",
            }
        )
    if spatial_stable:
        report["featureSelectionPolicy"] = SPATIAL_POLICY
        report["spatialModelGuardDefaults"] = _spatial_guard_defaults()
        report["observations"] = _compact_spatial_observations(observations)
        _write_spatial_models(report, output)
    if local_support:
        report.update(
            {
                "localSupportPolicy": LOCAL_POLICY,
                "localGeometry": {"width": width, "height": height},
                "estimatedHeadPolicy": LOCAL_HEAD_POLICY,
                "featureRefreshPolicy": "predeclared-owner-current-interior-projections-v1",
            }
        )
    path = output / "partial-pixel-track.json"
    with path.open("x", encoding="utf-8") as handle:
        if approximate:
            encoded = json.dumps(report, separators=(",", ":"), allow_nan=False)
            if len(encoded.encode("utf-8")) > MAX_APPROX_JSON_BYTES:
                raise ValueError("Approximate model audit exceeds its bounded JSON artifact budget")
            handle.write(encoded)
        else:
            json.dump(report, handle, indent=2, allow_nan=False)
    path.chmod(0o600)
    return path


def verify(
    config: Path,
    output: Path,
    *,
    approximate: bool = False,
    approximate_target_y: bool = False,
    corroborated: bool = False,
    spatial_stable: bool = False,
    local_support: bool = False,
) -> None:
    """Verify the diagnostic clock/status, without inventing independent pixel replay."""
    spatial_stable = spatial_stable or local_support
    corroborated = corroborated or spatial_stable
    approximate_target_y = approximate_target_y or corroborated
    approximate = approximate or approximate_target_y
    raw = record(tracker._read(config))
    expected = rows(record(raw["clock"])["pts"])
    path = output / "partial-pixel-track.json"
    report = read_approximate_json(path) if approximate else record(tracker._read(path))
    if spatial_stable and report.get("spatialModelArtifactRoot") != str(output):
        raise ValueError("Spatial verifier requires its exact output artifact root")
    observations = (
        list(spatial_observations(report, local_support=local_support))
        if spatial_stable
        else rows(report.get("observations"))
    )
    if spatial_stable:
        _validate_spatial_policy(raw.get("spatialFeaturePolicy"))
    if local_support:
        _validate_local_policy(raw.get("localFeaturePolicy"))
        if report.get("localGeometry") != {"width": raw["width"], "height": raw["height"]}:
            raise ValueError("Local report geometry differs from its producer input")
    if (
        report.get("protocol")
        != (
            LOCAL_PROTOCOL
            if local_support
            else SPATIAL_PROTOCOL
            if spatial_stable
            else CORROBORATED_PROTOCOL
            if corroborated
            else UNCERTAIN_PROTOCOL
            if approximate_target_y
            else APPROX_PROTOCOL
            if approximate
            else PROTOCOL
        )
        or report.get("seedAnchorPolicy")
        != (
            "two-disjoint-current-face-interior-models-v1"
            if approximate
            else "same-pts-measured-face-interior-v1"
        )
        or report.get("requestedFrames") != len(expected)
        or report.get("observedFrames") != len(observations)
        or len(observations) > len(expected)
        or any(
            record(o).get("source_pts") != pts
            for o, pts in zip(observations, expected, strict=False)
        )
        or any(
            report.get(k) is not False
            for k in (
                "semanticHeadVerified",
                "headNearVerified",
                "captionPlacementApproved",
                "qualityApproved",
                "independentPixelReplayVerified",
            )
        )
        or report.get("completeTrack")
        != (
            len(observations) == len(expected)
            and bool(observations)
            and record(observations[-1]).get("reason") is None
        )
    ):
        raise ValueError("Partial pixel diagnostic contradicts clock or unapproved status")
    if approximate and (
        report.get("currentMeasuredUpperPixels") is not False
        or report.get("originalFaceFlagsPreserved") is not True
        or report.get("estimatedHeadPolicy")
        != (
            LOCAL_HEAD_POLICY
            if local_support
            else "primary-face-target-with-independent-corroboration-v1"
            if corroborated
            else "bounded-face-pixel-target-y-approximation-v1"
            if approximate_target_y
            else "derived-face-pixel-motion-approximation"
        )
        or report.get("featureRefreshPolicy")
        != (
            "predeclared-owner-current-interior-projections-v1"
            if local_support
            else "primary-corroborated-current-interior-only-v1"
            if corroborated
            else "dual-accepted-approximate-current-interior-only-v1"
            if approximate_target_y
            else "dual-accepted-current-interior-only-v1"
        )
        or any(
            record(o).get("estimated_upper_y") is not None
            and index > 0
            and (
                len(rows(record(o).get("models"))) != 2
                or record(record(o).get("feature_refresh")).get("status")
                != "current-pixel-features-selected"
                or record(record(o).get("feature_refresh")).get("sourcePTS")
                != record(o).get("source_pts")
                or record(record(o).get("feature_refresh")).get("newFaceObservation") is not False
                or record(record(o).get("feature_refresh")).get("nativeInteriorRemeasured")
                is not False
                or (
                    not corroborated
                    and any(
                        (
                            record(record(m).get("approximateHeadContract")).get("accepted")
                            is not True
                            if approximate_target_y
                            else record(m).get("acceptedByExistingMotionContract") is not True
                            or record(m).get("reason") is not None
                        )
                        for m in rows(record(o).get("models"))
                    )
                )
            )
            for index, o in enumerate(observations)
        )
    ):
        raise ValueError("Approximate head policy requires both complete current face models")
    if approximate_target_y:
        budget = _number(
            record(raw.get("approximateYPolicy")).get("maximumTargetYModelUncertaintyPx")
        )
        uncertainty = record(report.get("targetYModelUncertainty"))
        if (
            uncertainty.get("configuredBudgetPx") != budget
            or uncertainty.get("isTrueHairPositionErrorGuarantee") is not False
            or uncertainty.get("isCaptionYDeadband") is not False
        ):
            raise ValueError(
                "Approximate uncertainty must preserve its explicit conditional budget"
            )
        for item in observations:
            observation = record(item)
            models = rows(observation.get("models"))
            if corroborated and models:
                if any(record(model).get("fitMatrix") is None for model in models):
                    if (
                        observation.get("estimated_upper_y") is not None
                        or record(
                            record(observation.get("feature_refresh")).get("modelPairDecision")
                        ).get("accepted")
                        is not False
                    ):
                        raise ValueError("Missing current fit may not supply approximate Y")
                    continue
                pair = validate_model_pair(
                    models,
                    maximum_target_y_uncertainty_px=budget,
                    width=integer(raw["width"]),
                    height=integer(raw["height"]),
                )
                if not _same_audit(
                    record(observation.get("feature_refresh")).get("modelPairDecision"), pair.audit
                ):
                    raise ValueError("Serialized primary/corroboration decision differs")
                estimate = observation.get("estimated_upper_y")
                if estimate is not None and (
                    not pair.accepted or estimate != pair.estimated_upper_y
                ):
                    raise ValueError("Approximate Y requires a complete primary and corroboration")
                continue
            for model in rows(observation.get("models")):
                # Rejected missing-fit evidence remains a failure, not usable Y.
                if record(model).get("fitMatrix") is not None:
                    decision = validate_approximate_model(
                        model,
                        maximum_target_y_uncertainty_px=budget,
                        width=integer(raw["width"]),
                        height=integer(raw["height"]),
                    )
                    if observation.get("estimated_upper_y") is not None and not decision.accepted:
                        raise ValueError(
                            "Approximate Y requires a complete supported uncertainty decision"
                        )
    if approximate:
        clock = record(raw["clock"])
        base = clock.get("timeBase")
        if not isinstance(base, str):
            raise ValueError("Approximate verifier requires its exact original clock")
        _, flags = detector_seed(
            tracker._read(Path(str(record(raw["detectorResponse"])["path"]))),
            clock=VideoFrameClock(
                tuple(integer(p) for p in expected), Fraction(base), integer(clock["lastDuration"])
            ),
        )
        if any(
            record(o).get("face_observed") is not flag
            for o, flag in zip(observations, flags, strict=False)
        ):
            raise ValueError("Approximate output changed original face observations")
    if local_support:
        _verify_local_controls(raw, report, tuple(record(item) for item in observations))


def _validate_local_policy(value: object) -> None:
    policy = record(value)
    if (
        set(policy) != {"version", "kind"}
        or type(policy["version"]) is not int
        or policy["version"] != 1
        or policy["kind"] != LOCAL_POLICY
    ):
        raise ValueError("Explicit predeclared measured local feature policy is required")


def _verify_local_controls(
    raw: dict[str, object], report: dict[str, object], observations: tuple[dict[str, object], ...]
) -> None:
    """Bind each model's controls to the real seed and prior supported transition."""
    clock_raw = record(raw["clock"])
    clock = VideoFrameClock(
        tuple(integer(pts) for pts in rows(clock_raw["pts"])),
        Fraction(str(clock_raw["timeBase"])),
        integer(clock_raw["lastDuration"]),
    )
    face, _ = detector_seed(
        tracker._read(Path(str(record(raw["detectorResponse"])["path"]))), clock=clock
    )
    seed = read_seed(
        tracker._read(Path(str(record(raw["nativeResponse"])["path"]))),
        clock=clock,
        width=integer(raw["width"]),
        height=integer(raw["height"]),
        face=face,
    )
    if seed is None:
        if observations:
            raise ValueError("Local observations require their actual measured seed")
        return
    if not _same_audit(report.get("seedAnchorRegions"), seed.anchor_regions):
        raise ValueError("Local tile ownership differs from actual native measured support")
    apex = float(seed.contour.points[:, 1].min())
    controls = seed.contour.points[seed.contour.points[:, 1] <= apex + HEAD_CAPTION_Y_DEADBAND_PX]
    moved = (controls, controls.copy())
    rejected = False
    for observation in observations:
        if rejected:
            raise ValueError("Local tracking may not continue after rejection or reset")
        models = rows(observation["models"])
        if models:
            if not _same_audit(
                [record(model).get("sourceControls") for model in models],
                [group.tolist() for group in moved],
            ):
                raise ValueError("Local model controls differ from the actual supported chain")
            if all(record(model).get("fitMatrix") is not None for model in models):
                pair = validate_model_pair(
                    models,
                    maximum_target_y_uncertainty_px=_number(
                        record(raw["approximateYPolicy"])["maximumTargetYModelUncertaintyPx"]
                    ),
                    width=seed.contour.width,
                    height=seed.contour.height,
                )
                if pair.transferred_controls is not None:
                    moved = pair.transferred_controls
        refresh = record(observation["feature_refresh"])
        if refresh.get("status") == "current-pixel-features-selected":
            _, owners = measured_tile_partition(
                seed.anchor_regions, width=seed.contour.width, height=seed.contour.height
            )
            first, _ = _projected_interior(seed.anchor_regions, controls, moved[0])
            second, _ = _projected_interior(seed.anchor_regions, controls, moved[1])
            projected = tuple(
                _clipped_local_region(
                    one if owner == 0 else two,
                    width=seed.contour.width,
                    height=seed.contour.height,
                )
                for one, two, owner in zip(first, second, owners, strict=True)
            )
            if not _same_audit(refresh.get("derivedSupportRegions"), projected):
                raise ValueError("Local current support differs from its owner's model")
        elif observation.get("estimated_upper_y") not in {None, apex}:
            raise ValueError("Local initial position differs from its actual seed")
        rejected = observation["reason"] is not None


def bind_native(config: Path, output: Path) -> None:
    """Bind one newly produced native response into an exclusive consumer config."""
    root = Path.cwd().resolve() / "edits" / "work"
    if output != output.resolve() or not output.is_relative_to(root) or output.exists():
        raise ValueError("Exclusive repository work binding output is required")
    raw = record(tracker._read(config))
    reference = record(raw.get("nativeResponse"))
    if set(reference) != {"path", "sha256"} or reference["sha256"] is not None:
        raise ValueError("Native output binding requires an explicit unbound producer output")
    name = reference["path"]
    if not isinstance(name, str):
        raise ValueError("Native producer output path is required")
    path = Path(name)
    if not path.is_absolute() or path != path.resolve() or not path.is_relative_to(root):
        raise ValueError("Native producer response must be a canonical repository work file")
    tracker._read(path)
    raw["nativeResponse"] = {"path": str(path), "sha256": _uncached_file_digest(path)}
    with output.open("x", encoding="utf-8") as handle:
        json.dump(raw, handle, indent=2, allow_nan=False)
    output.chmod(0o600)


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("config", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--verify", action="store_true")
    parser.add_argument("--bind-native", action="store_true")
    parser.add_argument("--approx-face-motion", action="store_true")
    parser.add_argument("--approx-target-y", action="store_true")
    parser.add_argument("--approx-corroborated-y", action="store_true")
    parser.add_argument("--approx-spatial-stable-y", action="store_true")
    parser.add_argument("--approx-local-support-y", action="store_true")
    args = parser.parse_args(argv)
    if args.verify and args.bind_native:
        parser.error("Choose one diagnostic operation")
    if (
        args.approx_face_motion
        or args.approx_target_y
        or args.approx_corroborated_y
        or args.approx_spatial_stable_y
        or args.approx_local_support_y
    ) and args.bind_native:
        parser.error("Approximation consumes an already bound native seed")
    if (
        sum(
            (
                args.approx_face_motion,
                args.approx_target_y,
                args.approx_corroborated_y,
                args.approx_spatial_stable_y,
                args.approx_local_support_y,
            )
        )
        > 1
    ):
        parser.error("Choose one distinct approximate policy")
    if args.bind_native:
        bind_native(args.config, args.output)
    elif args.verify:
        verify(
            args.config,
            args.output,
            approximate=args.approx_face_motion,
            approximate_target_y=args.approx_target_y,
            corroborated=args.approx_corroborated_y,
            spatial_stable=args.approx_spatial_stable_y,
            local_support=args.approx_local_support_y,
        )
    else:
        run(
            args.config,
            args.output,
            approximate=args.approx_face_motion,
            approximate_target_y=args.approx_target_y,
            corroborated=args.approx_corroborated_y,
            spatial_stable=args.approx_spatial_stable_y,
            local_support=args.approx_local_support_y,
        )


if __name__ == "__main__":
    main()
