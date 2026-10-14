"""Bounded face-Y candidates from current pixel seeds, never head observations."""

import json
from bisect import bisect_left, bisect_right
from dataclasses import dataclass
from fractions import Fraction
from itertools import pairwise
from math import hypot, isclose, isfinite
from pathlib import Path
from typing import Literal
from xml.etree import ElementTree as ET

from replay import head_partial_pixel_track as motion
from replay import producer_code_evidence
from replay.artifact_json import MAX_JSON_BYTES, _json, _pin
from replay.caption_style_defaults import (
    HEAD_CAPTION_GAP_PX,
    HEAD_CAPTION_REFERENCE_HEIGHT,
    HEAD_CAPTION_Y_DEADBAND_PX,
)
from replay.caption_y_stability import CaptionYStability
from replay.core import ReplayError
from replay.delivery_mask_quality import integer, rows
from replay.fcp_caption_srt import _time
from replay.fcp_head_near import FaceScaleSeries
from replay.fcp_project import rational
from replay.fcp_screen_stack import ScreenStack
from replay.fcp_visible_arc import VisibleArc
from replay.foreground_evidence import _bound_bytes
from replay.head_near_placement import caption_only_bottom
from replay.recordings import record
from replay.stages import file_digest
from replay.subtitle_clock import VideoFrameClock
from replay.tracking_observations import TrackingJob

MAX_CELLS = 262_144
MAX_SCOPE_SECONDS = 120
MAX_BLOCKS = 20_000
Policy = Literal[
    "pixel-seed-candidate",
    "face-y-derived-candidate",
    "measurement-required",
    "missing-face",
    "face-motion-derived-candidate",
]
Reason = Literal[
    "scope-start",
    "face-y-change",
    "face-shape-change",
    "face-x-change",
    "zoom-reset",
    "missing-face",
]
FaceBox = tuple[float, float, float, float]
MOTION_POLICY = "derived-face-pixel-motion-approximation"
UNCERTAIN_MOTION_POLICY = "bounded-face-pixel-target-y-approximation-v1"
MOTION_POLICIES = frozenset({MOTION_POLICY, UNCERTAIN_MOTION_POLICY})
Point = tuple[float, float]
Controls = tuple[Point, ...]
ControlPair = tuple[Controls, Controls]
MAX_MOTION_INPUTS = 512
ARCHIVED_CONSUMERS = ("head_approx_placement.py", "fcp_head_refresh.py")
MOTION_AUTHORITIES = (
    "source",
    "nativeExecutable",
    "nativeRequest",
    "nativeResponse",
    "detectorRequest",
    "detectorResponse",
    "ffmpeg",
    "approximateAssociation",
)
MOTION_CONFIG_FIELDS = frozenset(
    (*MOTION_AUTHORITIES, "version", "clock", "width", "height", "resets")
)
MOTION_CODE_ROLES = (
    "head_partial_pixel_track.py",
    "head_upper_texture.py",
    "head_texture_track.py",
)


@dataclass(frozen=True, slots=True)
class CurrentFaceCell:
    start: Fraction
    end: Fraction
    face: FaceBox | None


@dataclass(frozen=True, slots=True)
class MeasurementRequest:
    source_frame_start: Fraction
    source_frame_end: Fraction
    unsupported_start: Fraction
    unsupported_end: Fraction
    face: FaceBox | None
    reason: Reason

    def encoded(self) -> dict[str, object]:
        return {
            "sourceFrameStart": str(self.source_frame_start),
            "sourceFrameEnd": str(self.source_frame_end),
            "unsupportedStart": str(self.unsupported_start),
            "unsupportedEnd": str(self.unsupported_end),
            "face": self.face,
            "reason": self.reason,
            "kind": "current-face-observation" if self.face is None else "current-upper-pixels",
        }


@dataclass(frozen=True, slots=True)
class ApproxCell:
    start: Fraction
    end: Fraction
    face_time: Fraction
    top: float | None
    seed_time: Fraction | None
    policy: Policy
    face_observed: bool
    raw_top: float | None = None

    def encoded(self) -> dict[str, object]:
        result: dict[str, object] = {
            "start": str(self.start),
            "end": str(self.end),
            "currentFaceTime": str(self.face_time),
            "currentFaceObserved": self.face_observed,
            "estimatedUpperY": self.top,
            "seedTime": None if self.seed_time is None else str(self.seed_time),
            "seedAge": None if self.seed_time is None else str(self.face_time - self.seed_time),
            "policy": self.policy,
            "measuredCurrentUpperPixels": self.policy == "pixel-seed-candidate",
            "semanticHeadVerified": False,
            "headNearVerified": False,
        }
        if self.policy == "face-motion-derived-candidate":
            result["rawEstimatedUpperY"] = self.raw_top
        return result


@dataclass(frozen=True, slots=True)
class ApproxPlan:
    cells: tuple[ApproxCell, ...]
    requests: tuple[MeasurementRequest, ...]
    policy: str = "current-face-y-offset-candidate"

    def at(self, time: Fraction, *, ending: bool = False) -> ApproxCell | None:
        search = bisect_left if ending else bisect_right
        index = search(self.cells, time, key=lambda cell: cell.start) - 1
        if index < 0 or not (
            self.cells[index].start < time <= self.cells[index].end
            if ending
            else self.cells[index].start <= time < self.cells[index].end
        ):
            return None
        return self.cells[index]

    def encoded(self) -> dict[str, object]:
        return {
            "version": 1,
            "policy": self.policy,
            "candidateOnly": True,
            "headNearVerified": False,
            "semanticHeadVerified": False,
            "measurementRequests": [request.encoded() for request in self.requests],
            "cells": [cell.encoded() for cell in self.cells],
        }


@dataclass(frozen=True, slots=True)
class DerivedMotion:
    plan: ApproxPlan
    inputs: dict[str, str]

    def receipt(self) -> dict[str, object]:
        return {
            "protocol": _motion_protocol(self.plan.policy),
            "inputs": self.inputs,
            "candidateOnly": True,
            "measuredCurrentUpperPixels": False,
            "semanticHeadVerified": False,
            "headNearVerified": False,
            "qualityApproved": False,
        }


def _motion_protocol(policy: str) -> str:
    if policy not in MOTION_POLICIES:
        raise ReplayError("Expected an explicit derived head policy")
    return motion.APPROX_PROTOCOL if policy == MOTION_POLICY else motion.UNCERTAIN_PROTOCOL


def _number(value: object) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not isfinite(value):
        raise ReplayError("Derived head geometry requires finite numbers")
    return float(value)


def _points(
    value: object, *, minimum: int, width: int, height: int
) -> tuple[tuple[float, float], ...]:
    points = rows(value)
    if not minimum <= len(points) <= motion.MAX_FRAMES:
        raise ReplayError("Derived motion requires bounded current point geometry")
    result: list[tuple[float, float]] = []
    for point in points:
        pair = rows(point)
        if len(pair) != 2:
            raise ReplayError("Derived motion requires two-dimensional points")
        x, y = _number(pair[0]), _number(pair[1])
        if not 0 <= x < width or not 0 <= y < height:
            raise ReplayError("Derived motion point exceeds its source canvas")
        result.append((x, y))
    return tuple(result)


def _accepted_model(value: object, *, width: int, height: int) -> tuple[tuple[float, float], ...]:
    """Check actual reported support, rather than accepting a model's pass flag alone."""
    model = record(value)
    supported = rows(model.get("supported"))
    inliers = rows(model.get("inliers"))
    residuals = rows(model.get("residuals"))
    if (
        model.get("acceptedByExistingMotionContract") is not True
        or model.get("reason") is not None
        or model.get("completeModelRequired") is not True
        or not motion.MIN_FEATURES <= len(supported) <= motion.MAX_FRAMES
        or len(inliers) != len(supported)
        or len(residuals) != len(supported)
        or any(type(item) is not bool for item in (*supported, *inliers))
        or model.get("supportedCount") != supported.count(True)
        or model.get("inlierCount") != inliers.count(True)
        or inliers.count(True) < motion.MIN_FEATURES
        or supported.count(True) == 0
        or _number(model.get("inlierRatio")) != inliers.count(True) / supported.count(True)
        or _number(model.get("inlierRatio")) < 0.75
        or model.get("minimumInliers") != motion.MIN_FEATURES
        or model.get("minimumInlierRatio") != 0.75
    ):
        raise ReplayError("Derived head requires complete accepted current motion support")
    for supported_point, inlier, residual in zip(supported, inliers, residuals, strict=True):
        if inlier is True and (supported_point is not True or not 0 <= _number(residual) <= 1.5):
            raise ReplayError("Derived motion inliers contradict their current residuals")
    spans = rows(model.get("sourceSpan")), rows(model.get("inlierSpan"))
    if (
        any(len(span) != 2 for span in spans)
        or model.get("minimumSourceSpanPx") != 8
        or model.get("minimumInlierSpanRatio") != 0.5
        or any(_number(v) < 8 for v in spans[0])
        or any(_number(b) < _number(a) * 0.5 for a, b in zip(*spans, strict=True))
    ):
        raise ReplayError("Derived motion requires distributed current support")
    deformation = record(model.get("deformationGuard"))
    if (
        deformation.get("status") != "passed-by-existing-contract"
        or deformation.get("aggregateFailureReason") is not None
        or deformation.get("scaleLimits") != [0.9, 1.1]
        or deformation.get("rotationLimitDegrees") != 15
        or not 0.9 <= _number(deformation.get("scaleFromAcceptedControls")) <= 1.1
        or not 0 <= _number(deformation.get("rotationDegreesFromAcceptedControls")) <= 15
    ):
        raise ReplayError("Derived motion exceeds its unchanged deformation limits")
    for errors, statuses, limit in (
        ("roundtripError", "backwardStatus", 0.5),
        ("appearanceError", "forwardStatus", 12),
    ):
        values, flags = rows(model.get(errors)), rows(model.get(statuses))
        canvas = rows(model.get("canvasStatus"))
        if (
            len(values) != len(supported)
            or len(flags) != len(supported)
            or len(canvas) != len(supported)
            or any(type(flag) is not bool for flag in (*flags, *canvas))
            or any(
                ok is True
                and (flag is not True or inside is not True or not 0 <= _number(v) <= limit)
                for ok, flag, inside, v in zip(supported, flags, canvas, values, strict=True)
            )
        ):
            raise ReplayError("Derived motion support contradicts current pixel errors")
    if model.get("roundtripLimitPx") != 0.5 or model.get("appearanceLimit") != 12:
        raise ReplayError("Derived motion changed its existing pixel error limits")
    source = _points(
        model.get("sourceFeatures"), minimum=motion.MIN_FEATURES, width=width, height=height
    )
    current = rows(model.get("currentFeatures"))
    if len(source) != len(supported) or len(current) != len(source):
        raise ReplayError("Derived motion point counts differ from model support")
    # Failed correspondences can be nonfinite; only supported current points are usable.
    _points(
        [point for point, ok in zip(current, supported, strict=True) if ok],
        minimum=motion.MIN_FEATURES,
        width=width,
        height=height,
    )
    return _points(model.get("transferredControls"), minimum=3, width=width, height=height)


def _supported_estimate(
    observation: dict[str, object],
    *,
    first: bool,
    width: int,
    height: int,
    uncertainty_budget: float | None = None,
    previous_controls: ControlPair | None = None,
) -> tuple[float | None, ControlPair | None]:
    top, reason = observation.get("estimated_upper_y"), observation.get("reason")
    if top is None:
        if not isinstance(reason, str) or not reason:
            raise ReplayError("Unknown derived head requires an explicit rejection reason")
        return None, None
    estimate = _number(top)
    refresh = record(observation.get("feature_refresh"))
    if (
        reason is not None
        or not 0 < estimate < height
        or refresh.get("sourcePTS") != observation.get("source_pts")
        or refresh.get("nativeInteriorRemeasured") is not False
        or refresh.get("newFaceObservation") is not False
        or any(integer(n) < motion.MIN_FEATURES for n in rows(refresh.get("selectedFeatureCounts")))
        or len(rows(refresh.get("selectedFeatureCounts"))) != 2
    ):
        raise ReplayError("Derived head requires supported current feature refresh")
    models = rows(observation.get("models"))
    if first:
        if refresh.get("status") != "original-measured-seed" or models:
            raise ReplayError("Initial derived head must use the original measured seed")
        return estimate, previous_controls
    if (
        len(models) != 2
        or refresh.get("status") != "current-pixel-features-selected"
        or refresh.get("reason") is not None
        or refresh.get("supportPolicy")
        != "intersection-of-two-accepted-measured-interior-projections-v1"
        or refresh.get("nextFrameRequiresCompleteModels") is not True
        or refresh.get("modelAgreementLimitPx") != HEAD_CAPTION_Y_DEADBAND_PX
        or not 0
        <= _number(refresh.get("maximumInteriorProjectionDisagreementPx"))
        <= HEAD_CAPTION_Y_DEADBAND_PX
        or not 0
        <= _number(observation.get("maximum_model_disagreement"))
        <= HEAD_CAPTION_Y_DEADBAND_PX
    ):
        raise ReplayError("Derived head requires both agreeing current interior models")
    if uncertainty_budget is None:
        first_controls = _accepted_model(models[0], width=width, height=height)
        second_controls = _accepted_model(models[1], width=width, height=height)
    else:
        if previous_controls is None:
            raise ReplayError("Approximate uncertainty requires source-bound control continuity")
        first_features = _points(
            record(models[0]).get("sourceFeatures"),
            minimum=motion.MIN_FEATURES,
            width=width,
            height=height,
        )
        second_features = _points(
            record(models[1]).get("sourceFeatures"),
            minimum=motion.MIN_FEATURES,
            width=width,
            height=height,
        )
        if set(first_features) & set(second_features):
            raise ReplayError("Approximate models require disjoint current feature cohorts")
        first_controls = _uncertain_model(
            models[0],
            previous=previous_controls[0],
            budget=uncertainty_budget,
            width=width,
            height=height,
        )
        second_controls = _uncertain_model(
            models[1],
            previous=previous_controls[1],
            budget=uncertainty_budget,
            width=width,
            height=height,
        )
    if len(first_controls) != len(second_controls):
        raise ReplayError("Derived motion models transported different control points")
    difference = max(
        hypot(a[0] - b[0], a[1] - b[1])
        for a, b in zip(first_controls, second_controls, strict=True)
    )
    apex = (
        min(point[1] for point in first_controls) + min(point[1] for point in second_controls)
    ) / 2
    if not isclose(
        difference, _number(observation["maximum_model_disagreement"]), abs_tol=1e-4
    ) or not isclose(estimate, apex, abs_tol=1e-4):
        raise ReplayError("Derived head estimate contradicts its actual transferred controls")
    return estimate, (first_controls, second_controls)


def _uncertain_model(
    value: object, *, previous: Controls, budget: float, width: int, height: int
) -> Controls:
    """Use the producer's public numerical verifier, preserving its original model decision."""
    decision = motion.validate_approximate_model(
        value, maximum_target_y_uncertainty_px=budget, width=width, height=height
    )
    model = record(value)
    if (
        not decision.accepted
        or decision.reason is not None
        or decision.transferred_controls is None
        or _points(model.get("sourceControls"), minimum=3, width=width, height=height) != previous
    ):
        raise ReplayError("Approximate uncertainty lost its source-bound current control chain")
    return _points(decision.transferred_controls.tolist(), minimum=3, width=width, height=height)


def derived_plan(
    report: dict[str, object],
    *,
    clock: VideoFrameClock,
    flags: tuple[bool, ...],
    width: int,
    height: int,
    resets: tuple[Fraction, ...] = (),
    uncertainty_budget: float | None = None,
    seed_controls: Controls | None = None,
) -> ApproxPlan:
    """Keep rejected suffixes unknown and raw missing-face flags independent of candidate Y."""
    motion.validate_clock(clock)
    observations = rows(report.get("observations"))
    policy = UNCERTAIN_MOTION_POLICY if uncertainty_budget is not None else MOTION_POLICY
    if uncertainty_budget is not None:
        metadata = record(report.get("targetYModelUncertainty"))
        if (
            not 0 < _number(uncertainty_budget) <= motion.target_y_budget(height)
            or (
                bool(observations)
                and (seed_controls is None or not 3 <= len(seed_controls) <= motion.MAX_FRAMES)
            )
            or metadata
            != {
                "configuredBudgetPx": uncertainty_budget,
                "kind": "conditional-correspondence-perturbation-bound",
                "allocationPolicy": "one-third-of-scaled-caption-head-gap-v1",
                "styleHeadGapPx": HEAD_CAPTION_GAP_PX,
                "styleReferenceHeight": HEAD_CAPTION_REFERENCE_HEIGHT,
                "maximumAllocatedBudgetPx": motion.target_y_budget(height),
                "isTrueHairPositionErrorGuarantee": False,
                "isCaptionYDeadband": False,
                "uncertaintyScope": "one-current-correspondence-transition",
                "accumulatedHairErrorBounded": False,
            }
            or metadata.get("isTrueHairPositionErrorGuarantee") is not False
            or metadata.get("isCaptionYDeadband") is not False
            or metadata.get("accumulatedHairErrorBounded") is not False
        ):
            raise ReplayError(
                "Expected explicit bounded target-Y uncertainty without a hair guarantee"
            )
        if seed_controls is not None:
            seed_controls = _points(
                [list(point) for point in seed_controls], minimum=3, width=width, height=height
            )
            if observations and record(observations[0]).get("estimated_upper_y") not in (
                None,
                min(point[1] for point in seed_controls),
            ):
                raise ReplayError("Approximate initial Y differs from its original apex controls")
    if (
        report.get("protocol") != _motion_protocol(policy)
        or report.get("requestedFrames") != len(clock.pts)
        or report.get("observedFrames") != len(observations)
        or len(observations) > len(clock.pts)
        or len(flags) != len(clock.pts)
        or any(type(flag) is not bool for flag in flags)
        or type(width) is not int
        or not 0 < width <= 16384
        or type(height) is not int
        or not 0 < height <= 16384
        or report.get("derivedPolicySupport") is not True
        or report.get("modelAgreementLimitPx") != HEAD_CAPTION_Y_DEADBAND_PX
        or report.get("currentMeasuredUpperPixels") is not False
        or report.get("originalFaceFlagsPreserved") is not True
        or report.get("estimatedHeadPolicy") != policy
        or report.get("featureRefreshPolicy")
        != (
            "dual-accepted-current-interior-only-v1"
            if uncertainty_budget is None
            else "dual-accepted-approximate-current-interior-only-v1"
        )
        or report.get("seedAnchorPolicy") != "two-disjoint-current-face-interior-models-v1"
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
        or any(time not in {pts * clock.time_base for pts in clock.pts} for time in resets)
        or any(a >= b for a, b in pairwise(resets))
    ):
        raise ReplayError(
            "Derived head report contradicts its clock, budget or approximation status"
        )
    result: list[ApproxCell] = []
    stopped = False
    controls = None if seed_controls is None else (seed_controls, seed_controls)
    for index, pts in enumerate(clock.pts):
        start = pts * clock.time_base
        end = (
            clock.pts[index + 1] if index + 1 < len(clock.pts) else pts + clock.last_duration
        ) * clock.time_base
        top = None
        if index < len(observations):
            item = record(observations[index])
            if (
                type(item.get("source_pts")) is not int
                or item.get("source_pts") != pts
                or item.get("face_observed") is not flags[index]
            ):
                raise ReplayError("Derived head changed original PTS or raw face flags")
            top, controls = _supported_estimate(
                item,
                first=index == 0,
                width=width,
                height=height,
                uncertainty_budget=uncertainty_budget,
                previous_controls=controls,
            )
            if top is not None and (
                stopped
                or start in resets
                or (index > 0 and (pts - clock.pts[index - 1]) * clock.time_base > 1)
            ):
                raise ReplayError("Derived head cannot propagate after a rejected model or reset")
            stopped = stopped or top is None
        result.append(
            ApproxCell(
                start,
                end,
                start,
                top,
                clock.pts[0] * clock.time_base if top is not None else None,
                "face-motion-derived-candidate",
                flags[index],
                top,
            )
        )
    complete = bool(observations) and len(observations) == len(clock.pts) and not stopped
    if report.get("completeTrack") is not complete:
        raise ReplayError("Derived head completeness differs from actual supported observations")
    failure = report.get("measurementRequestReason")
    expected = (
        record(observations[-1]).get("reason")
        if observations
        else "measured_independent_upper_seed_unavailable"
    )
    if failure != expected or (not complete and not isinstance(failure, str)):
        raise ReplayError("Derived head rejection differs from its actual producer outcome")
    return ApproxPlan(tuple(result), (), policy)


def motion_scope(plan: ApproxPlan, *, start: Fraction, end: Fraction, height: int) -> ApproxPlan:
    """Stabilize candidate Y once per supported scope, resetting at unknowns and source cuts."""
    if (
        plan.policy not in MOTION_POLICIES
        or type(height) is not int
        or not 0 < height <= 16384
        or not plan.cells
        or len(plan.cells) > motion.MAX_FRAMES
        or not plan.cells[0].start <= start < end <= plan.cells[-1].end
        or any(
            cell.start >= cell.end or cell.policy != "face-motion-derived-candidate"
            for cell in plan.cells
        )
        or any(left.end != right.start for left, right in pairwise(plan.cells))
    ):
        raise ReplayError("Derived head comparison exceeds its physical source clock")
    state = CaptionYStability(HEAD_CAPTION_Y_DEADBAND_PX * height / HEAD_CAPTION_REFERENCE_HEIGHT)
    cells: list[ApproxCell] = []
    for cell in plan.cells:
        if cell.end <= start or cell.start >= end:
            continue
        left, right = max(start, cell.start), min(end, cell.end)
        if cell.top is None:
            state = CaptionYStability(state.deadband)
            top = None
        else:
            if not 0 < _number(cell.top) < height:
                raise ReplayError("Derived head exceeds its source canvas")
            top = state.sample(left, cell.top)
        cells.append(
            ApproxCell(
                left,
                right,
                cell.face_time,
                top,
                cell.seed_time,
                cell.policy,
                cell.face_observed,
                cell.raw_top,
            )
        )
    return ApproxPlan(tuple(cells), (), plan.policy)


def read_motion_binding(value: object, *, job: TrackingJob) -> DerivedMotion:
    """Bind original producer bytes separately from current downstream consumer code."""
    binding = record(value)
    policy = binding.get("policy")
    if (
        not isinstance(policy, str)
        or policy not in MOTION_POLICIES
        or set(binding)
        - {
            "policy",
            "config",
            "report",
            "producerEvidence",
        }
    ):
        raise ReplayError("Expected explicit derived face-motion config and report")
    config, report_path = _pin(binding.get("config")), _pin(binding.get("report"))
    raw, report = _json(config), _json(report_path)
    fields = MOTION_CONFIG_FIELDS | (
        {"approximateYPolicy"} if policy == UNCERTAIN_MOTION_POLICY else set()
    )
    if set(raw) != fields or type(raw.get("version")) is not int or raw["version"] != 1:
        raise ReplayError("Expected original version-one face-motion configuration")
    if report_path.name != "partial-pixel-track.json":
        raise ReplayError("Expected the original face-motion producer report")
    source = record(raw.get("source"))
    if (source.get("path"), source.get("sha256"), raw.get("width"), raw.get("height")) != (
        str(job.source),
        job.source_hash,
        job.width,
        job.height,
    ):
        raise ReplayError("Derived head belongs to another source or canvas")
    inputs = {str(config): file_digest(config), str(report_path): file_digest(report_path)}
    producer = {
        name: producer_code_evidence._digest(digest)
        for name, digest in record(report.get("inputs")).items()
    }
    if (
        not 1 <= len(producer) <= MAX_MOTION_INPUTS
        or producer.get(str(config)) != inputs[str(config)]
    ):
        raise ReplayError("Derived head requires its original bounded producer inputs")
    if any(str(Path(__file__).with_name(name)) not in producer for name in MOTION_CODE_ROLES):
        raise ReplayError("Derived head omitted its original motion producer code")
    archives: dict[str, producer_code_evidence.CodeArchive] = {}
    if "producerEvidence" in binding:
        descriptor = _pin(binding["producerEvidence"])
        evidence = producer_code_evidence.read_archives(
            descriptor,
            file_digest(descriptor),
            protocol=_motion_protocol(policy),
            source_names=ARCHIVED_CONSUMERS,
            root=Path(__file__).resolve().parents[2],
            binding_fields=frozenset({"measurement"}),
        )
        if record(evidence.document.get("measurement")) != {
            "config": binding["config"],
            "report": binding["report"],
        }:
            raise ReplayError("Producer code archive binds another face-motion observation")
        archives = evidence.archives
        inputs.update(evidence.inputs)
    for name, digest in producer.items():
        saved = archives.get(name)
        if saved is not None and saved.sha256 != digest:
            raise ReplayError("Historical face-motion code archive differs from producer input")
        path = Path(name) if saved is None else saved.path
        checked = _pin({"path": str(path), "sha256": digest})
        inputs[str(checked)] = digest
    for name in MOTION_AUTHORITIES:
        path = _pin(raw.get(name))
        if producer.get(str(path)) != file_digest(path):
            raise ReplayError("Derived head omitted an original authority input")
    data = record(raw.get("clock"))
    if set(data) != {"pts", "timeBase", "lastDuration"} or not isinstance(
        data.get("timeBase"), str
    ):
        raise ReplayError("Derived head requires its physical rational clock")
    clock = VideoFrameClock(
        tuple(integer(p) for p in rows(data["pts"])),
        Fraction(str(data["timeBase"])),
        integer(data["lastDuration"]),
    )
    motion.validate_clock(clock)
    first_index = bisect_left(job.clock.pts, clock.pts[0])
    final_index = bisect_left(job.clock.pts, clock.pts[-1])
    final_duration = (
        job.clock.pts[final_index + 1] - clock.pts[-1]
        if final_index + 1 < len(job.clock.pts)
        else job.clock.last_duration
    )
    if (
        clock.time_base != job.clock.time_base
        or job.clock.pts[first_index : final_index + 1] != clock.pts
        or clock.last_duration != final_duration
        or (clock.pts[-1] + clock.last_duration) * clock.time_base
        > (job.clock.pts[-1] + job.clock.last_duration) * job.clock.time_base
    ):
        raise ReplayError("Derived head clock differs from original physical source frames")
    detector = _pin(raw["detectorResponse"])
    native = _pin(raw["nativeResponse"])
    face, flags = motion.detector_seed(
        json.loads(_bound_bytes(detector, file_digest(detector), MAX_JSON_BYTES)), clock=clock
    )
    if _json(_pin(raw["detectorRequest"])).get("path") != str(job.source) or _json(
        _pin(raw["nativeRequest"])
    ) != motion.seed_request(source=job.source, clock=clock, face=face):
        raise ReplayError("Derived head native/detector requests differ from the actual source")
    association = motion.approximate_association(
        raw["approximateAssociation"],
        source_hash=job.source_hash,
        clock=clock,
        face=face,
        inputs=inputs,
    )
    if report.get("approximateAssociation") != association or report.get(
        "seedDetectorRegion"
    ) != list(face):
        raise ReplayError("Derived head changed its original source-image association")
    if any(
        producer.get(str(association[key])) != inputs.get(str(association[key]))
        for key in ("observation", "image")
    ):
        raise ReplayError("Derived head omitted original source-image evidence")
    seed = motion.read_seed(
        json.loads(_bound_bytes(native, file_digest(native), MAX_JSON_BYTES)),
        clock=clock,
        width=job.width,
        height=job.height,
        face=face,
    )
    observations = rows(report.get("observations"))
    if seed is None:
        expected_upper: list[object] = []
        expected_anchor: list[object] = []
        status = None
    else:
        expected_upper = [list(region) for region in seed.upper_regions]
        expected_anchor = [list(region) for region in seed.anchor_regions]
        status = seed.native_status
    if (
        bool(observations) != (seed is not None)
        or report.get("nativeStatus") != status
        or report.get("seedUpperRegions") != expected_upper
        or report.get("seedAnchorRegions") != expected_anchor
        or (
            seed is not None
            and record(observations[0]).get("estimated_upper_y")
            not in (None, float(seed.contour.points[:, 1].min()))
        )
    ):
        raise ReplayError("Derived head initial estimate differs from its actual native seed")
    reset_rows = tuple(record(item) for item in rows(raw.get("resets")))
    if any(
        set(item) != {"sourcePTS", "reason"}
        or not isinstance(item.get("reason"), str)
        or str(item["reason"]) not in motion.RESET_REASONS
        for item in reset_rows
    ):
        raise ReplayError("Derived head requires original explicit source reset boundaries")
    resets = tuple(integer(item["sourcePTS"]) * clock.time_base for item in reset_rows)
    uncertainty_budget = None
    seed_controls = None
    if policy == UNCERTAIN_MOTION_POLICY:
        uncertainty = record(raw["approximateYPolicy"])
        if (
            set(uncertainty) != {"version", "maximumTargetYModelUncertaintyPx"}
            or type(uncertainty.get("version")) is not int
            or uncertainty["version"] != 1
        ):
            raise ReplayError("Expected version-one explicit target-Y uncertainty input")
        uncertainty_budget = _number(uncertainty["maximumTargetYModelUncertaintyPx"])
        if seed is not None:
            apex = float(seed.contour.points[:, 1].min())
            points = seed.contour.points[
                seed.contour.points[:, 1] <= apex + HEAD_CAPTION_Y_DEADBAND_PX
            ]
            seed_controls = _points(points.tolist(), minimum=3, width=job.width, height=job.height)
    plan = derived_plan(
        report,
        clock=clock,
        flags=flags,
        width=job.width,
        height=job.height,
        resets=resets,
        uncertainty_budget=uncertainty_budget,
        seed_controls=seed_controls,
    )
    return DerivedMotion(plan, inputs)


def current_cells(
    faces: FaceScaleSeries, *, start: Fraction, end: Fraction
) -> tuple[CurrentFaceCell, ...]:
    """Keep raw frame extents and missing flags; ignore derived scale envelopes."""
    positions = faces.tops.positions
    if (
        not faces.sizes
        or not positions[0].time <= start < end <= faces.tops.coverage_end
        or len(positions) > MAX_CELLS
    ):
        raise ReplayError("Approximate placement requires paired current face cells")
    left = bisect_right(positions, start, key=lambda point: point.time) - 1
    right = bisect_left(positions, end, key=lambda point: point.time)
    result: list[CurrentFaceCell] = []
    for index in range(left, right):
        position, size = positions[index], faces.sizes[index]
        next_time = (
            positions[index + 1].time if index + 1 < len(positions) else faces.tops.coverage_end
        )
        observed = not any(
            gap.start <= position.time < gap.end for gap in faces.tops.derived_ranges
        )
        if observed != size.observed:
            raise ReplayError("Current face-size and missing-observation provenance differ")
        result.append(
            CurrentFaceCell(
                position.time,
                next_time,
                (position.x, position.y, size.width, faces.bottoms.positions[index].y - position.y)
                if observed
                else None,
            )
        )
    return tuple(result)


def scoped_plan(
    faces: FaceScaleSeries,
    *,
    pixels: VisibleArc,
    start: Fraction,
    end: Fraction,
    resets: tuple[Fraction, ...] = (),
) -> ApproxPlan:
    """Reset on source cuts and bounded scope ends; do not hold a seed indefinitely."""
    source = current_cells(faces, start=start, end=end)
    cells: list[ApproxCell] = []
    requests: list[MeasurementRequest] = []
    index, cursor = 0, start
    while cursor < end:
        limit = min(cursor + MAX_SCOPE_SECONDS, end)
        stop = (
            len(source) if limit == end else bisect_right(source, limit, key=lambda cell: cell.end)
        )
        if stop == index:
            raise ReplayError("Approximate frame cell exceeds its bounded source scope")
        boundary = min(source[stop - 1].end, end)
        plan = derive(
            source[index:stop],
            pixels=pixels,
            start=cursor,
            end=boundary,
            width=faces.tops.width,
            height=faces.tops.height,
            resets=tuple(time for time in resets if cursor < time < boundary),
        )
        cells.extend(plan.cells)
        requests.extend(plan.requests)
        index, cursor = stop, boundary
    return ApproxPlan(tuple(cells), tuple(requests))


def _validate(
    faces: tuple[CurrentFaceCell, ...],
    *,
    start: Fraction,
    end: Fraction,
    width: int,
    height: int,
    resets: tuple[Fraction, ...],
) -> None:
    if (
        not 1 <= len(faces) <= MAX_CELLS
        or type(width) is not int
        or type(height) is not int
        or not 0 < width <= 16384
        or not 0 < height <= 16384
        or width * height > 16_777_216
        or not isinstance(start, Fraction)
        or not isinstance(end, Fraction)
        or not 0 <= start < end <= 86400
        or end - start > MAX_SCOPE_SECONDS
    ):
        raise ReplayError("Approximate placement requires bounded consecutive source cells")
    for cell in faces:
        if (
            not isinstance(cell.start, Fraction)
            or not isinstance(cell.end, Fraction)
            or not 0 <= cell.start < cell.end <= 86400
            or cell.end - cell.start > 1
        ):
            raise ReplayError("Approximate placement requires actual bounded frame cells")
        if cell.face is not None:
            x, y, w, h = cell.face
            if any(isinstance(v, bool) or not isfinite(v) for v in cell.face) or not (
                0 <= x < x + w <= width and 0 <= y < y + h <= height
            ):
                raise ReplayError("Approximate placement face geometry exceeds its canvas")
    if (
        not faces[0].start <= start < faces[0].end
        or not faces[-1].start < end <= faces[-1].end
        or any(left.end != right.start for left, right in pairwise(faces))
        or any(not isinstance(time, Fraction) or not start < time < end for time in resets)
        or any(left >= right for left, right in pairwise(resets))
        or not set(resets).issubset(cell.start for cell in faces)
    ):
        raise ReplayError("Approximate placement requires exact consecutive cells and reset clocks")


def _change(current: FaceBox, seed: FaceBox, tolerance: float) -> Reason | None:
    if abs(current[1] - seed[1]) > tolerance:
        return "face-y-change"
    if max(abs(current[2] - seed[2]), abs(current[3] - seed[3])) > tolerance:
        return "face-shape-change"
    return "face-x-change" if abs(current[0] - seed[0]) > tolerance else None


def derive(
    faces: tuple[CurrentFaceCell, ...],
    *,
    pixels: VisibleArc,
    start: Fraction,
    end: Fraction,
    width: int,
    height: int,
    resets: tuple[Fraction, ...] = (),
) -> ApproxPlan:
    """Request fresh seeds on change; no nearest old seed or face-size conversion.

    A retained seed applies only inside this explicit <=120-second source scope,
    while every derived cell requires its own current accepted face. A seed's
    bound is operational, not proof of unchanged hair/pose between measurements.
    """
    _validate(faces, start=start, end=end, width=width, height=height, resets=resets)
    tolerance = HEAD_CAPTION_Y_DEADBAND_PX * height / HEAD_CAPTION_REFERENCE_HEIGHT
    seed: tuple[Fraction, FaceBox, float] | None = None
    reason: Reason = "scope-start"
    cells: list[ApproxCell] = []
    requests: list[MeasurementRequest] = []
    reset_index = 0
    for face in faces:
        left, right = max(start, face.start), min(end, face.end)
        if reset_index < len(resets) and face.start == resets[reset_index]:
            seed, reason = None, "zoom-reset"
            reset_index += 1
        if face.face is None:
            seed, reason = None, "missing-face"
        elif seed is not None:
            invalid = _change(face.face, seed[1], tolerance)
            if invalid is not None:
                seed, reason = None, invalid
        policy: Policy = "missing-face" if face.face is None else "face-y-derived-candidate"
        if seed is None and face.face is not None:
            measured = pixels.at(face.start)
            if measured is not None and measured.visible_top is not None:
                if (
                    measured.start != face.start
                    or measured.end != face.end
                    or not isfinite(measured.visible_top)
                    or not 0 <= measured.visible_top <= face.face[1]
                    or measured.supported < 3
                ):
                    raise ReplayError(
                        "Approximate seed must bind current supported same-PTS pixels"
                    )
                seed = (face.start, face.face, measured.visible_top)
                policy = "pixel-seed-candidate"
            else:
                policy = "measurement-required"
        top = None if seed is None or face.face is None else seed[2] + face.face[1] - seed[1][1]
        if top is not None and face.face is not None and not 0 <= top <= face.face[1]:
            seed, top, policy, reason = None, None, "measurement-required", "face-y-change"
        if seed is None:
            pending = None if not requests else requests[-1].face
            if face.face is not None and pending is not None and cells and cells[-1].top is None:
                changed = _change(face.face, pending, tolerance)
                if changed is not None:
                    reason = changed
            request = MeasurementRequest(face.start, face.end, left, right, face.face, reason)
            if (
                requests
                and cells[-1].top is None
                and requests[-1].reason == reason
                and (requests[-1].face is None) == (face.face is None)
                and (
                    face.face is None
                    or pending is None
                    or _change(face.face, pending, tolerance) is None
                )
                and requests[-1].unsupported_end == left
            ):
                old = requests.pop()
                request = MeasurementRequest(
                    old.source_frame_start,
                    old.source_frame_end,
                    old.unsupported_start,
                    right,
                    old.face,
                    old.reason,
                )
            requests.append(request)
        cells.append(
            ApproxCell(
                left,
                right,
                face.start,
                top,
                None if seed is None else seed[0],
                policy,
                face.face is not None,
            )
        )
    return ApproxPlan(tuple(cells), tuple(requests))


def unknown_stacks(
    plan: ApproxPlan, blocks: tuple[tuple[Fraction, Fraction, float], ...]
) -> tuple[ScreenStack, ...]:
    """Reserve measured height across each head-unknown run without changing face flags."""
    if not 1 <= len(blocks) <= MAX_BLOCKS or any(
        not isinstance(left, Fraction)
        or not isinstance(right, Fraction)
        or not 0 <= left < right <= 86400
        or isinstance(h, bool)
        or not isfinite(h)
        or h <= 0
        for left, right, h in blocks
    ):
        raise ReplayError("Expected bounded measured caption blocks")
    ranges: list[tuple[Fraction, Fraction]] = []
    for cell in plan.cells:
        if cell.top is None:
            if ranges and ranges[-1][1] == cell.start:
                ranges[-1] = (ranges[-1][0], cell.end)
            else:
                ranges.append((cell.start, cell.end))
    return tuple(
        ScreenStack(left, right, max(heights))
        for left, right in ranges
        if (heights := tuple(h for a, b, h in blocks if a < right and b > left))
    )


def title_samples(
    title: ET.Element,
    *,
    plan: ApproxPlan,
    caption_height: float,
    canvas_height: int,
    stacks: tuple[ScreenStack, ...],
) -> tuple[tuple[str, str], ...]:
    """Centered bottom keys; missing seeds use explicit bottom-stacked screen layout."""
    start, duration = _time(title.attrib, "offset"), _time(title.attrib, "duration")
    end, local = start + duration, _time(title.attrib, "start")
    if (
        title.tag != "title"
        or type(canvas_height) is not int
        or not 0 < canvas_height <= 16384
        or isinstance(caption_height, bool)
        or not 0 < caption_height <= canvas_height
        or not duration > 0
    ):
        raise ReplayError("Expected bounded approximate caption geometry")
    left = bisect_right(plan.cells, start, key=lambda cell: cell.start)
    right = bisect_left(plan.cells, end, key=lambda cell: cell.start)
    times = (start, *(cell.start for cell in plan.cells[left:right]), end)
    if any(a.end != b.start for a, b in pairwise(plan.cells[max(0, left - 1) : right])):
        raise ReplayError("Approximate title cannot cross unsupported source-clock holes")
    result: list[tuple[str, str]] = []
    for time in times:
        cell = plan.at(time, ending=time == end)
        if cell is None:
            raise ReplayError("Approximate title exceeds declared source scope")
        if cell.top is None:
            reserved = max(
                (
                    stack.height
                    for stack in stacks
                    if stack.start <= time < stack.end or stack.start <= time == stack.end == end
                ),
                default=caption_height,
            )
            if reserved < caption_height:
                raise ReplayError("Approximate unknown stack is smaller than the caption")
            bottom = caption_only_bottom(caption_height=reserved, canvas_height=canvas_height)
        else:
            bottom = cell.top - HEAD_CAPTION_GAP_PX
            if not caption_height <= bottom <= canvas_height:
                raise ReplayError("Approximate caption requires measured headroom paging")
        result.append(
            (rational(local + time - start), f"0 {50 - bottom / canvas_height * 100:.12g}")
        )
    return tuple(result)
