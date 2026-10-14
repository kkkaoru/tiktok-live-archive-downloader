import json
from copy import deepcopy
from dataclasses import replace
from fractions import Fraction
from pathlib import Path
from xml.etree import ElementTree as ET

import numpy as np
import pytest
from test_fcp_head_near import paired as paired

from replay import head_approx_placement as placement
from replay import head_partial_pixel_track as motion
from replay.core import ReplayError
from replay.delivery_mask_quality import rows
from replay.fcp_head_near import FaceScaleSeries
from replay.fcp_screen_stack import ScreenStack
from replay.fcp_tracking_caption import MissingFaceRange
from replay.fcp_visible_arc import ArcCell, VisibleArc
from replay.head_contour_track import ContourSeed
from replay.head_near_placement import FaceSizeObservation
from replay.head_seed import ContourTrackMetrics
from replay.head_texture_track import TextureFit
from replay.recordings import record
from replay.stages import file_digest
from replay.subtitle_clock import VideoFrameClock
from replay.tracking_observations import TrackingJob


@pytest.fixture
def accepted_motion_model() -> dict[str, object]:
    return {
        "acceptedByExistingMotionContract": True,
        "reason": None,
        "completeModelRequired": True,
        "sourceFeatures": [[20 + i * 2, 30 + i * 2] for i in range(12)],
        "currentFeatures": [[20 + i * 2, 36 + i * 2] for i in range(12)],
        "supported": [True] * 12,
        "supportedCount": 12,
        "inliers": [True] * 12,
        "inlierCount": 12,
        "inlierRatio": 1,
        "residuals": [0] * 12,
        "minimumInliers": 12,
        "minimumInlierRatio": 0.75,
        "sourceSpan": [22, 22],
        "inlierSpan": [22, 22],
        "minimumSourceSpanPx": 8,
        "minimumInlierSpanRatio": 0.5,
        "forwardStatus": [True] * 12,
        "backwardStatus": [True] * 12,
        "canvasStatus": [True] * 12,
        "roundtripError": [0] * 12,
        "roundtripLimitPx": 0.5,
        "appearanceError": [0] * 12,
        "appearanceLimit": 12,
        "deformationGuard": {
            "status": "passed-by-existing-contract",
            "aggregateFailureReason": None,
            "scaleFromAcceptedControls": 1,
            "rotationDegreesFromAcceptedControls": 0,
            "scaleLimits": [0.9, 1.1],
            "rotationLimitDegrees": 15,
        },
        "transferredControls": [[100, 706], [200, 708], [300, 707]],
    }


@pytest.fixture
def derived_report(accepted_motion_model: dict[str, object]) -> dict[str, object]:
    return {
        "protocol": "derived-face-motion-approximate-head-v1",
        "requestedFrames": 3,
        "observedFrames": 2,
        "completeTrack": False,
        "currentMeasuredUpperPixels": False,
        "originalFaceFlagsPreserved": True,
        "estimatedHeadPolicy": "derived-face-pixel-motion-approximation",
        "featureRefreshPolicy": "dual-accepted-current-interior-only-v1",
        "seedAnchorPolicy": "two-disjoint-current-face-interior-models-v1",
        "derivedPolicySupport": True,
        "modelAgreementLimitPx": 6,
        "semanticHeadVerified": False,
        "headNearVerified": False,
        "captionPlacementApproved": False,
        "qualityApproved": False,
        "independentPixelReplayVerified": False,
        "measurementRequestReason": None,
        "observations": [
            {
                "source_pts": 0,
                "face_observed": True,
                "estimated_upper_y": 700,
                "maximum_model_disagreement": None,
                "models": [],
                "reason": None,
                "feature_refresh": {
                    "status": "original-measured-seed",
                    "selectedFeatureCounts": [12, 12],
                    "nativeInteriorRemeasured": False,
                    "newFaceObservation": False,
                    "sourcePTS": 0,
                },
            },
            {
                "source_pts": 1,
                "face_observed": False,
                "estimated_upper_y": 706,
                "maximum_model_disagreement": 0,
                "models": [deepcopy(accepted_motion_model), deepcopy(accepted_motion_model)],
                "reason": None,
                "feature_refresh": {
                    "status": "current-pixel-features-selected",
                    "reason": None,
                    "supportPolicy": (
                        "intersection-of-two-accepted-measured-interior-projections-v1"
                    ),
                    "selectedFeatureCounts": [12, 12],
                    "nativeInteriorRemeasured": False,
                    "newFaceObservation": False,
                    "sourcePTS": 1,
                    "nextFrameRequiresCompleteModels": True,
                    "modelAgreementLimitPx": 6,
                    "maximumInteriorProjectionDisagreementPx": 0,
                },
            },
        ],
    }


@pytest.fixture
def uncertain_motion_model() -> dict[str, object]:
    source = np.array(
        [
            [100, 760],
            [200, 760],
            [300, 760],
            [100, 800],
            [200, 800],
            [300, 800],
            [100, 840],
            [200, 840],
            [300, 840],
            [150, 780],
            [250, 780],
            [200, 820],
            [150, 760],
            [250, 760],
            [150, 840],
            [250, 840],
            [100, 820],
        ],
        dtype=np.float32,
    )
    current = source + np.array([0, 6], dtype=np.float32)
    current[12:, 1] += 1.75
    supported = np.ones(17, dtype=np.bool_)
    metrics = ContourTrackMetrics(
        current,
        supported,
        supported,
        supported,
        supported,
        np.zeros(17, dtype=np.float32),
        np.zeros(17, dtype=np.float32),
    )
    fitted = TextureFit(
        source,
        metrics,
        ((1, 0, 0), (0, 1, 6)),
        (True,) * 12 + (False,) * 5,
        (0,) * 12 + (1.75,) * 5,
        (200, 80),
        (200, 80),
        "motion_support_insufficient",
    )
    controls = np.array([[100, 700], [200, 702], [300, 701]], dtype=np.float32)
    decision = motion.approximate_model(
        fitted,
        controls,
        maximum_target_y_uncertainty_px=8,
        width=1080,
        height=1920,
    )
    return {
        "acceptedByExistingMotionContract": False,
        "reason": "motion_support_insufficient",
        "sourceFeatures": source.tolist(),
        "currentFeatures": current.tolist(),
        "sourceControls": controls.tolist(),
        "fitMatrix": [[1, 0, 0], [0, 1, 6]],
        "supported": [True] * 17,
        "forwardStatus": [True] * 17,
        "backwardStatus": [True] * 17,
        "canvasStatus": [True] * 17,
        "roundtripError": [0] * 17,
        "appearanceError": [0] * 17,
        "inliers": [True] * 12 + [False] * 5,
        "residuals": [0] * 12 + [1.75] * 5,
        "approximateHeadContract": record(json.loads(json.dumps(decision.audit))),
    }


@pytest.fixture
def uncertain_report(
    derived_report: dict[str, object], uncertain_motion_model: dict[str, object]
) -> dict[str, object]:
    second_model = deepcopy(uncertain_motion_model)
    source = np.asarray(rows(second_model["sourceFeatures"]), np.float32) + [20, 0]
    current = np.asarray(rows(second_model["currentFeatures"]), np.float32) + [20, 0]
    flags = np.ones(17, np.bool_)
    fitted = TextureFit(
        np.asarray(source, np.float32),
        ContourTrackMetrics(
            np.asarray(current, np.float32),
            flags,
            flags,
            flags,
            flags,
            np.zeros(17, np.float32),
            np.zeros(17, np.float32),
        ),
        ((1, 0, 0), (0, 1, 6)),
        (True,) * 12 + (False,) * 5,
        (0,) * 12 + (1.75,) * 5,
        (200, 80),
        (200, 80),
        "motion_support_insufficient",
    )
    decision = motion.approximate_model(
        fitted,
        np.array([[100, 700], [200, 702], [300, 701]], np.float32),
        maximum_target_y_uncertainty_px=8,
        width=1080,
        height=1920,
    )
    second_model.update(
        {
            "sourceFeatures": source.tolist(),
            "currentFeatures": current.tolist(),
            "approximateHeadContract": record(json.loads(json.dumps(decision.audit))),
        }
    )
    derived_report.update(
        {
            "protocol": "derived-face-motion-uncertain-head-v1",
            "requestedFrames": 2,
            "completeTrack": True,
            "estimatedHeadPolicy": "bounded-face-pixel-target-y-approximation-v1",
            "featureRefreshPolicy": "dual-accepted-approximate-current-interior-only-v1",
            "targetYModelUncertainty": {
                "configuredBudgetPx": 8,
                "kind": "conditional-correspondence-perturbation-bound",
                "allocationPolicy": "one-third-of-scaled-caption-head-gap-v1",
                "styleHeadGapPx": 24,
                "styleReferenceHeight": 1920,
                "maximumAllocatedBudgetPx": 8,
                "isTrueHairPositionErrorGuarantee": False,
                "isCaptionYDeadband": False,
                "uncertaintyScope": "one-current-correspondence-transition",
                "accumulatedHairErrorBounded": False,
            },
        }
    )
    record(rows(derived_report["observations"])[1])["models"] = [
        deepcopy(uncertain_motion_model),
        second_model,
    ]
    return derived_report


def test_explicit_uncertainty_preserves_legacy_rejection_and_raw_missing_flag(
    uncertain_report: dict[str, object],
) -> None:
    plan = placement.derived_plan(
        uncertain_report,
        clock=VideoFrameClock((0, 1), Fraction(1, 30), 1),
        flags=(True, False),
        width=1080,
        height=1920,
        uncertainty_budget=8,
        seed_controls=((100, 700), (200, 702), (300, 701)),
    )
    assert tuple((cell.top, cell.face_observed) for cell in plan.cells) == (
        (700, True),
        (706, False),
    )
    assert plan.policy == "bounded-face-pixel-target-y-approximation-v1"
    model = record(rows(record(rows(uncertain_report["observations"])[1])["models"])[0])
    assert model["acceptedByExistingMotionContract"] is False
    assert model["reason"] == "motion_support_insufficient"
    assert record(model["approximateHeadContract"])["accepted"] is True
    assert (
        placement.DerivedMotion(plan, {}).receipt()["protocol"]
        == "derived-face-motion-uncertain-head-v1"
    )


@pytest.mark.parametrize(
    "fault",
    [
        "budget",
        "nan",
        "metadata",
        "hair",
        "legacy-protocol",
        "deadband",
        "seed-controls",
        "source-chain",
        "matrix",
        "consensus",
        "bound",
        "duplicate-model",
        "accumulated-guarantee",
        "invalid-seed",
        "changed-apex",
    ],
)
def test_uncertainty_never_relabels_legacy_evidence_or_skips_its_numeric_proof(
    uncertain_report: dict[str, object],
    fault: str,
) -> None:
    metadata = record(uncertain_report["targetYModelUncertainty"])
    model = record(rows(record(rows(uncertain_report["observations"])[1])["models"])[0])
    if fault == "metadata":
        metadata["configuredBudgetPx"] = 6
    elif fault == "hair":
        metadata["isTrueHairPositionErrorGuarantee"] = True
    elif fault == "legacy-protocol":
        uncertain_report["protocol"] = "derived-face-motion-approximate-head-v1"
    elif fault == "deadband":
        metadata["isCaptionYDeadband"] = True
    elif fault == "source-chain":
        model["sourceControls"] = [[100, 699], [200, 701], [300, 700]]
    elif fault == "matrix":
        model["fitMatrix"] = [[1, 0, 0], [0, 1, 5]]
    elif fault == "consensus":
        record(model["approximateHeadContract"])["consensusCount"] = 12
    elif fault == "bound":
        record(model["approximateHeadContract"])["conditionalBoundPx"] = 0
    elif fault == "duplicate-model":
        record(rows(uncertain_report["observations"])[1])["models"] = [model, deepcopy(model)]
    elif fault == "accumulated-guarantee":
        metadata["accumulatedHairErrorBounded"] = True
    elif fault == "changed-apex":
        record(rows(uncertain_report["observations"])[0])["estimated_upper_y"] = 699
    budget = float("nan") if fault == "nan" else 8.0
    seed = ((100.0, float("nan")), (200.0, 702.0), (300.0, 701.0))
    if fault != "invalid-seed":
        seed = ((100.0, 700.0), (200.0, 702.0), (300.0, 701.0))
    with pytest.raises((ReplayError, ValueError)):
        placement.derived_plan(
            uncertain_report,
            clock=VideoFrameClock((0, 1), Fraction(1, 30), 1),
            flags=(True, False),
            width=1080,
            height=1920,
            uncertainty_budget=9 if fault == "budget" else budget,
            seed_controls=None if fault == "seed-controls" else seed,
        )


def test_numerically_valid_model_still_requires_the_actual_prior_controls(
    uncertain_motion_model: dict[str, object],
) -> None:
    with pytest.raises(ReplayError, match="control chain"):
        placement._uncertain_model(
            uncertain_motion_model,
            previous=((100, 699), (200, 701), (300, 700)),
            budget=8,
            width=1080,
            height=1920,
        )


def test_uncertainty_numeric_rejection_remains_unknown_not_held(
    uncertain_motion_model: dict[str, object],
) -> None:
    model = uncertain_motion_model
    current = np.asarray(rows(model["currentFeatures"]), dtype=np.float32)
    current[12:, 1] += 20
    source = np.asarray(rows(model["sourceFeatures"]), dtype=np.float32)
    flags = np.ones(17, dtype=np.bool_)
    metrics = ContourTrackMetrics(
        current, flags, flags, flags, flags, np.zeros(17, np.float32), np.zeros(17, np.float32)
    )
    fitted = TextureFit(
        source,
        metrics,
        ((1, 0, 0), (0, 1, 6)),
        (True,) * 12 + (False,) * 5,
        (0,) * 12 + (21.75,) * 5,
        (200, 80),
        (200, 80),
        "motion_support_insufficient",
    )
    decision = motion.approximate_model(
        fitted,
        np.array([[100, 700], [200, 702], [300, 701]], np.float32),
        maximum_target_y_uncertainty_px=8,
        width=1080,
        height=1920,
    )
    model.update(
        {
            "currentFeatures": current.tolist(),
            "residuals": [0] * 12 + [21.75] * 5,
            "approximateHeadContract": record(json.loads(json.dumps(decision.audit))),
        }
    )
    with pytest.raises(ReplayError, match="control chain"):
        placement._uncertain_model(
            model, previous=((100, 700), (200, 702), (300, 701)), budget=8, width=1080, height=1920
        )


def test_derived_models_keep_raw_missing_flag_and_unobserved_suffix_unknown(
    derived_report: dict[str, object],
) -> None:
    # A genuine producer prefix must end with a rejection, rather than silently omit a frame.
    second = record(rows(derived_report["observations"])[1])
    second.update({"estimated_upper_y": None, "reason": "model-inlier-ratio"})
    derived_report["measurementRequestReason"] = "model-inlier-ratio"
    plan = placement.derived_plan(
        derived_report,
        clock=VideoFrameClock((0, 1, 2), Fraction(1, 30), 1),
        flags=(True, False, False),
        width=1080,
        height=1920,
    )
    assert tuple((cell.top, cell.face_observed) for cell in plan.cells) == (
        (700, True),
        (None, False),
        (None, False),
    )
    assert plan.cells[1].encoded()["measuredCurrentUpperPixels"] is False
    assert plan.encoded()["policy"] == "derived-face-pixel-motion-approximation"
    assert plan.encoded()["semanticHeadVerified"] is False


def test_complete_derived_estimate_keeps_raw_missing_without_measured_claim(
    derived_report: dict[str, object],
) -> None:
    derived_report.update({"requestedFrames": 2, "completeTrack": True})
    plan = placement.derived_plan(
        derived_report,
        clock=VideoFrameClock((0, 1), Fraction(1, 30), 1),
        flags=(True, False),
        width=1080,
        height=1920,
    )
    assert tuple((cell.top, cell.raw_top, cell.face_observed) for cell in plan.cells) == (
        (700, 700, True),
        (706, 706, False),
    )
    assert plan.cells[1].encoded()["rawEstimatedUpperY"] == 706
    assert placement.DerivedMotion(plan, {}).receipt() == {
        "protocol": "derived-face-motion-approximate-head-v1",
        "inputs": {},
        "candidateOnly": True,
        "measuredCurrentUpperPixels": False,
        "semanticHeadVerified": False,
        "headNearVerified": False,
        "qualityApproved": False,
    }


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("protocol", "measured-head"),
        ("observedFrames", 1),
        ("requestedFrames", 256),
        ("currentMeasuredUpperPixels", True),
        ("originalFaceFlagsPreserved", False),
        ("headNearVerified", True),
        ("completeTrack", True),
        ("modelAgreementLimitPx", 20),
        ("measurementRequestReason", "other"),
        ("derivedPolicySupport", False),
    ],
)
def test_derived_report_rejects_status_clock_and_approval_changes(
    derived_report: dict[str, object],
    field: str,
    value: object,
) -> None:
    derived_report[field] = value
    with pytest.raises(ReplayError):
        placement.derived_plan(
            derived_report,
            clock=VideoFrameClock((0, 1, 2), Fraction(1, 30), 1),
            flags=(True, False, False),
            width=1080,
            height=1920,
        )


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("source_pts", 2),
        ("face_observed", True),
        ("estimated_upper_y", float("nan")),
        ("estimated_upper_y", -1),
        ("estimated_upper_y", 1920),
        ("estimated_upper_y", None),
        ("estimated_upper_y", 708),
        ("maximum_model_disagreement", 7),
        ("models", []),
        ("reason", "reset"),
    ],
)
def test_derived_observations_require_actual_matching_controls(
    derived_report: dict[str, object],
    field: str,
    value: object,
) -> None:
    derived_report.update({"requestedFrames": 2, "completeTrack": True})
    record(rows(derived_report["observations"])[1])[field] = value
    with pytest.raises(ReplayError):
        placement.derived_plan(
            derived_report,
            clock=VideoFrameClock((0, 1), Fraction(1, 30), 1),
            flags=(True, False),
            width=1080,
            height=1920,
        )


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("acceptedByExistingMotionContract", False),
        ("reason", "inlier-ratio"),
        ("supportedCount", 11),
        ("inliers", [True] * 11),
        ("inlierRatio", 0.5),
        ("residuals", [-1] * 12),
        ("residuals", [2] * 12),
        ("sourceSpan", [7, 22]),
        ("inlierSpan", [10, 22]),
        ("backwardStatus", [False] * 12),
        ("appearanceError", [13] * 12),
        ("roundtripError", [float("inf")] * 12),
        ("roundtripLimitPx", 1),
        ("currentFeatures", []),
        ("sourceFeatures", [[20, 30]] * 11),
        ("transferredControls", [[20, 30]] * 2),
        ("transferredControls", [[20, -1], [30, 700], [40, 701]]),
    ],
)
def test_model_pass_flag_never_replaces_current_pixel_support(
    accepted_motion_model: dict[str, object],
    field: str,
    value: object,
) -> None:
    accepted_motion_model[field] = value
    with pytest.raises(ReplayError):
        placement._accepted_model(accepted_motion_model, width=1080, height=1920)


@pytest.mark.parametrize(
    "fault", ["scale", "rotation", "status", "span-shape", "point-shape", "status-shape"]
)
def test_model_geometry_and_deformation_are_bounded(
    accepted_motion_model: dict[str, object],
    fault: str,
) -> None:
    deformation = record(accepted_motion_model["deformationGuard"])
    if fault == "scale":
        deformation["scaleFromAcceptedControls"] = 1.2
    elif fault == "rotation":
        deformation["rotationDegreesFromAcceptedControls"] = 16
    elif fault == "status":
        deformation["aggregateFailureReason"] = "rejected"
    elif fault == "span-shape":
        accepted_motion_model["sourceSpan"] = [20]
    elif fault == "point-shape":
        accepted_motion_model["sourceFeatures"] = [[20, 30, 40]] * 12
    else:
        accepted_motion_model["canvasStatus"] = [1] * 12
    with pytest.raises(ReplayError):
        placement._accepted_model(accepted_motion_model, width=1080, height=1920)


@pytest.mark.parametrize(
    "fault",
    [
        "refresh",
        "refresh-clock",
        "refresh-count",
        "models-disagree",
        "seed",
        "reset",
        "clock-gap",
        "width",
        "empty",
    ],
)
def test_derived_reset_unknown_and_seed_boundaries(
    derived_report: dict[str, object],
    fault: str,
) -> None:
    clock = VideoFrameClock((0, 1), Fraction(1, 30), 1)
    derived_report.update({"requestedFrames": 2, "completeTrack": True})
    observations = rows(derived_report["observations"])
    second = record(observations[1])
    refresh = record(second["feature_refresh"])
    if fault == "refresh":
        refresh["nativeInteriorRemeasured"] = True
    elif fault == "refresh-clock":
        refresh["sourcePTS"] = 0
    elif fault == "refresh-count":
        refresh["selectedFeatureCounts"] = [12, 11]
    elif fault == "models-disagree":
        record(rows(second["models"])[1])["transferredControls"] = [
            [100, 706],
            [200, 708],
            [300, 708],
        ]
    elif fault == "seed":
        record(observations[0])["models"] = second["models"]
    elif fault == "clock-gap":
        clock = VideoFrameClock((0, 60), Fraction(1, 30), 1)
        second["source_pts"] = 60
        refresh["sourcePTS"] = 60
    elif fault == "empty":
        derived_report.update({"observations": [], "observedFrames": 0, "completeTrack": False})
    with pytest.raises(ReplayError):
        placement.derived_plan(
            derived_report,
            clock=clock,
            flags=(True, False),
            width=0 if fault == "width" else 1080,
            height=1920,
            resets=(Fraction(1, 30),) if fault == "reset" else (),
        )


def test_unknown_seed_and_explicit_reset_keep_all_later_cells_unknown(
    derived_report: dict[str, object],
) -> None:
    derived_report.update(
        {
            "observations": [],
            "observedFrames": 0,
            "measurementRequestReason": "measured_independent_upper_seed_unavailable",
        }
    )
    plan = placement.derived_plan(
        derived_report,
        clock=VideoFrameClock((0, 1, 2), Fraction(1, 30), 1),
        flags=(True, False, False),
        width=1080,
        height=1920,
        resets=(Fraction(1, 30),),
    )
    assert tuple(cell.top for cell in plan.cells) == (None, None, None)


def test_rejected_model_cannot_resume_on_a_later_frame(derived_report: dict[str, object]) -> None:
    observations = rows(derived_report["observations"])
    recovered = deepcopy(record(observations[1]))
    recovered["source_pts"] = 2
    record(recovered["feature_refresh"])["sourcePTS"] = 2
    record(observations[1]).update({"estimated_upper_y": None, "reason": "scene"})
    derived_report.update({"observations": [*observations, recovered], "observedFrames": 3})
    with pytest.raises(ReplayError, match="propagate"):
        placement.derived_plan(
            derived_report,
            clock=VideoFrameClock((0, 1, 2), Fraction(1, 30), 1),
            flags=(True, False, False),
            width=1080,
            height=1920,
        )


def test_stability_suppresses_small_reversal_but_follows_real_motion_and_resets() -> None:
    plan = placement.ApproxPlan(
        tuple(
            placement.ApproxCell(
                Fraction(i),
                Fraction(i + 1),
                Fraction(i),
                top,
                Fraction(0),
                "face-motion-derived-candidate",
                False,
                top,
            )
            for i, top in enumerate((700, 702, 698, 720, None, 680))
        ),
        (),
        "derived-face-pixel-motion-approximation",
    )
    scoped = placement.motion_scope(plan, start=Fraction(0), end=Fraction(6), height=1920)
    assert tuple(cell.top for cell in scoped.cells) == (700, 700, 700, 714, None, 680)
    assert tuple(cell.raw_top for cell in scoped.cells) == (700, 702, 698, 720, None, 680)
    cut = placement.motion_scope(plan, start=Fraction(3), end=Fraction(4), height=1920)
    assert cut.cells[0].top == 720


@pytest.mark.parametrize("fault", ["coverage", "gap", "policy", "top", "height", "empty"])
def test_stability_rejects_invalid_scope(fault: str) -> None:
    first = placement.ApproxCell(
        Fraction(0),
        Fraction(1),
        Fraction(0),
        700,
        Fraction(0),
        "face-motion-derived-candidate",
        True,
        700,
    )
    second = replace(
        first,
        start=Fraction(2) if fault == "gap" else Fraction(1),
        end=Fraction(2),
        top=float("nan") if fault == "top" else 706,
    )
    plan = placement.ApproxPlan(
        () if fault == "empty" else (first, second),
        (),
        "other" if fault == "policy" else "derived-face-pixel-motion-approximation",
    )
    with pytest.raises(ReplayError):
        placement.motion_scope(
            plan,
            start=Fraction(0),
            end=Fraction(3 if fault == "coverage" else 2),
            height=0 if fault == "height" else 1920,
        )


@pytest.fixture
def motion_binding(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    derived_report: dict[str, object],
) -> tuple[dict[str, object], TrackingJob]:
    source = tmp_path / "source.mov"
    source.write_bytes(b"synthetic source")
    image = tmp_path / "source-image.png"
    image.write_bytes(b"synthetic image, never decoded")
    clock = VideoFrameClock((0, 1), Fraction(1, 30), 1)
    face = (100.0, 900.0, 200.0, 400.0)
    native = tmp_path / "native.json"
    native.write_text("[]", encoding="utf-8")
    detector = tmp_path / "detector.json"
    detector.write_text("[]", encoding="utf-8")
    native_request = tmp_path / "native-request.json"
    native_request.write_text(
        json.dumps(motion.seed_request(source=source, clock=clock, face=face)),
        encoding="utf-8",
    )
    detector_request = tmp_path / "detector-request.json"
    detector_request.write_text(json.dumps({"path": str(source)}), encoding="utf-8")
    executable = tmp_path / "executable"
    executable.write_bytes(b"synthetic tool, never executed")
    association = tmp_path / "association.json"
    association.write_text(
        json.dumps(
            {
                "version": 1,
                "source": {
                    "expectedSHA256FromExistingBoundConfig": file_digest(source),
                    "identityBeforeAfterEqual": True,
                },
                "operation": {
                    "commandExit": 0,
                    "framesWritten": 1,
                    "showinfoFirstPTS": 0,
                    "timeBase": "1/30",
                    "image": str(image),
                    "imageSHA256": file_digest(image),
                },
                "visualObservation": {
                    "directlyViewed": True,
                    "pixelPerfectHairSegmentation": False,
                    "acceptedOriginalDetectorRectangle": list(face),
                },
            }
        ),
        encoding="utf-8",
    )
    paths = {
        "source": source,
        "nativeExecutable": executable,
        "nativeRequest": native_request,
        "nativeResponse": native,
        "detectorRequest": detector_request,
        "detectorResponse": detector,
        "ffmpeg": executable,
        "approximateAssociation": association,
    }
    config = tmp_path / "motion-config.json"
    config.write_text(
        json.dumps(
            {
                "version": 1,
                "width": 1080,
                "height": 1920,
                "resets": [],
                "clock": {"pts": [0, 1], "timeBase": "1/30", "lastDuration": 1},
                **{
                    key: {"path": str(path), "sha256": file_digest(path)}
                    for key, path in paths.items()
                },
            }
        ),
        encoding="utf-8",
    )
    seed = motion.PartialPixelSeed(
        ContourSeed(
            Fraction(0),
            1080,
            1920,
            np.array([[100, 700], [200, 702], [300, 701]], dtype=np.float32),
            ((100, 800, 20, 20),),
        ),
        ((100, 800, 20, 20),),
        ((100, 1000, 20, 20),),
        "clippedSupport",
    )
    monkeypatch.setattr(motion, "detector_seed", lambda *args, **kwargs: (face, (True, False)))
    monkeypatch.setattr(motion, "read_seed", lambda *args, **kwargs: seed)
    producer = {str(path): file_digest(path) for path in (*paths.values(), config, image)}
    producer.update(
        {
            str(Path(placement.__file__).with_name(name)): file_digest(
                Path(placement.__file__).with_name(name)
            )
            for name in placement.MOTION_CODE_ROLES
        }
    )
    derived_report.update(
        {
            "requestedFrames": 2,
            "completeTrack": True,
            "inputs": producer,
            "nativeStatus": "clippedSupport",
            "seedUpperRegions": [[100, 800, 20, 20]],
            "seedAnchorRegions": [[100, 1000, 20, 20]],
            "seedDetectorRegion": list(face),
            "approximateAssociation": {
                "kind": "source-image-associated-approximate-upper",
                "observation": str(association),
                "image": str(image),
                "seedPTS": 0,
                "approximateAssociationOnly": True,
                "semanticHeadVerified": False,
            },
        }
    )
    report = tmp_path / "partial-pixel-track.json"
    report.write_text(json.dumps(derived_report), encoding="utf-8")
    binding: dict[str, object] = {
        "policy": "derived-face-pixel-motion-approximation",
        "config": {"path": str(config), "sha256": file_digest(config)},
        "report": {"path": str(report), "sha256": file_digest(report)},
    }
    job = TrackingJob(
        config=config,
        config_hash=file_digest(config),
        source=source,
        native=executable,
        source_hash=file_digest(source),
        native_hash=file_digest(executable),
        clock=clock,
        width=1080,
        height=1920,
        interval=0.5,
    )
    return binding, job


def test_motion_binding_pins_original_authorities_and_keeps_current_reader_separate(
    motion_binding: tuple[dict[str, object], TrackingJob],
) -> None:
    binding, job = motion_binding
    result = placement.read_motion_binding(binding, job=job)
    assert tuple((cell.top, cell.face_observed) for cell in result.plan.cells) == (
        (700, True),
        (706, False),
    )
    assert result.receipt()["qualityApproved"] is False
    assert str(job.source) in result.inputs


@pytest.mark.parametrize("fault", ["none", "budget", "version", "extent", "physical-gap"])
def test_uncertain_binding_pins_explicit_budget_and_complete_physical_source_scope(
    motion_binding: tuple[dict[str, object], TrackingJob],
    uncertain_report: dict[str, object],
    fault: str,
) -> None:
    binding, job = motion_binding
    config_path = Path(str(record(binding["config"])["path"]))
    report_path = Path(str(record(binding["report"])["path"]))
    config = record(json.loads(config_path.read_text(encoding="utf-8")))
    original_report = record(json.loads(report_path.read_text(encoding="utf-8")))
    original_report.update(uncertain_report)
    config["approximateYPolicy"] = {
        "version": 2 if fault == "version" else 1,
        "maximumTargetYModelUncertaintyPx": 9 if fault == "budget" else 8,
    }
    if fault == "extent":
        job = replace(job, clock=VideoFrameClock((0, 1), Fraction(1, 30), 2))
    elif fault == "physical-gap":
        job = replace(job, clock=VideoFrameClock((0, 1, 2), Fraction(1, 30), 1))
        record(config["clock"])["pts"] = [0, 2]
    config_path.write_text(json.dumps(config), encoding="utf-8")
    record(binding["config"])["sha256"] = file_digest(config_path)
    record(original_report["inputs"])[str(config_path)] = file_digest(config_path)
    report_path.write_text(json.dumps(original_report), encoding="utf-8")
    record(binding["report"])["sha256"] = file_digest(report_path)
    binding["policy"] = "bounded-face-pixel-target-y-approximation-v1"
    if fault != "none":
        with pytest.raises((ReplayError, ValueError)):
            placement.read_motion_binding(binding, job=job)
        return
    result = placement.read_motion_binding(binding, job=job)
    assert tuple((cell.top, cell.face_observed) for cell in result.plan.cells) == (
        (700, True),
        (706, False),
    )
    assert result.receipt()["protocol"] == "derived-face-motion-uncertain-head-v1"
    assert result.receipt()["measuredCurrentUpperPixels"] is False
    assert result.receipt()["qualityApproved"] is False


@pytest.mark.parametrize(
    "fault",
    [
        "policy",
        "extra",
        "config-version",
        "source",
        "canvas",
        "report-name",
        "config-pin",
        "code-omitted",
        "authority-omitted",
        "hash",
        "clock",
        "detector-request",
        "native-request",
        "association",
        "image-omitted",
        "seed",
        "reset",
    ],
)
def test_motion_binding_rejects_bad_source_clock_and_provenance(
    motion_binding: tuple[dict[str, object], TrackingJob],
    fault: str,
) -> None:
    binding, job = motion_binding
    config_path = Path(str(record(binding["config"])["path"]))
    report_path = Path(str(record(binding["report"])["path"]))
    config = record(json.loads(config_path.read_text(encoding="utf-8")))
    report = record(json.loads(report_path.read_text(encoding="utf-8")))
    inputs = record(report["inputs"])
    if fault == "policy":
        binding["policy"] = "measured-head"
    elif fault == "extra":
        binding["approval"] = True
    elif fault == "config-version":
        config["version"] = 2
    elif fault == "source":
        job = replace(job, source_hash="f" * 64)
    elif fault == "canvas":
        config["width"] = 1920
    elif fault == "report-name":
        report_path = report_path.with_name("other.json")
    elif fault == "config-pin":
        inputs[str(config_path)] = "f" * 64
    elif fault == "code-omitted":
        inputs.pop(str(Path(placement.__file__).with_name("head_partial_pixel_track.py")))
    elif fault == "authority-omitted":
        inputs.pop(str(job.source))
    elif fault == "hash":
        job.source.write_bytes(b"changed")
    elif fault == "clock":
        job = replace(job, clock=VideoFrameClock((0, 1), Fraction(1, 60), 1))
    elif fault == "detector-request":
        request = Path(str(record(config["detectorRequest"])["path"]))
        request.write_text(json.dumps({"path": "another source"}), encoding="utf-8")
        record(config["detectorRequest"])["sha256"] = file_digest(request)
        inputs[str(request)] = file_digest(request)
    elif fault == "native-request":
        request = Path(str(record(config["nativeRequest"])["path"]))
        request.write_text("{}", encoding="utf-8")
        record(config["nativeRequest"])["sha256"] = file_digest(request)
        inputs[str(request)] = file_digest(request)
    elif fault == "association":
        record(report["approximateAssociation"])["seedPTS"] = 1
    elif fault == "image-omitted":
        inputs.pop(str(record(report["approximateAssociation"])["image"]))
    elif fault == "seed":
        report["nativeStatus"] = "acceptedSemanticHead"
    else:
        config["resets"] = [{"sourcePTS": 1, "reason": "guessed"}]
    config_path.write_text(json.dumps(config), encoding="utf-8")
    record(binding["config"])["sha256"] = file_digest(config_path)
    if fault != "config-pin":
        inputs[str(config_path)] = file_digest(config_path)
    report_path.write_text(json.dumps(report), encoding="utf-8")
    binding["report"] = {"path": str(report_path), "sha256": file_digest(report_path)}
    with pytest.raises((ReplayError, ValueError)):
        placement.read_motion_binding(binding, job=job)


@pytest.mark.parametrize("fault", ["none", "context", "digest", "archive-changed"])
def test_old_consumer_archive_is_exact_data_not_an_executable_backend(
    motion_binding: tuple[dict[str, object], TrackingJob],
    tmp_path: Path,
    fault: str,
) -> None:
    binding, job = motion_binding
    report_path = Path(str(record(binding["report"])["path"]))
    report = record(json.loads(report_path.read_text(encoding="utf-8")))
    original = str(Path(placement.__file__).resolve())
    archive = tmp_path / "original-consumer.py.txt"
    archive.write_bytes(b"original producer bytes, never imported or executed")
    digest = file_digest(archive)
    record(report["inputs"])[original] = "f" * 64 if fault == "digest" else digest
    report_path.write_text(json.dumps(report), encoding="utf-8")
    binding["report"] = {"path": str(report_path), "sha256": file_digest(report_path)}
    descriptor = tmp_path / "producer-code.json"
    descriptor.write_text(
        json.dumps(
            {
                "version": 1,
                "protocol": "derived-face-motion-approximate-head-v1",
                "archives": [{"sourcePath": original, "path": str(archive), "sha256": digest}],
                "measurement": {
                    "config": binding["config"],
                    "report": {} if fault == "context" else binding["report"],
                },
            }
        ),
        encoding="utf-8",
    )
    binding["producerEvidence"] = {"path": str(descriptor), "sha256": file_digest(descriptor)}
    if fault == "archive-changed":
        archive.write_bytes(b"changed bytes")
    if fault != "none":
        with pytest.raises((ReplayError, ValueError)):
            placement.read_motion_binding(binding, job=job)
        return
    result = placement.read_motion_binding(binding, job=job)
    assert result.plan.cells[1].face_observed is False
    assert str(archive) in result.inputs
    assert original not in result.inputs


@pytest.fixture
def faces() -> tuple[placement.CurrentFaceCell, ...]:
    return (
        placement.CurrentFaceCell(Fraction(0), Fraction(1), (100, 900, 200, 400)),
        placement.CurrentFaceCell(Fraction(1), Fraction(2), (101, 904, 202, 401)),
        placement.CurrentFaceCell(Fraction(2), Fraction(3), (100, 902, 201, 402)),
        placement.CurrentFaceCell(Fraction(3), Fraction(4), (100, 900, 200, 400)),
    )


@pytest.fixture
def pixels() -> VisibleArc:
    return VisibleArc((ArcCell(Fraction(0), Fraction(1), 700, 3, 0),), {})


def test_raw_current_cells_ignore_existing_scale_and_preserve_missing(
    paired: FaceScaleSeries,
) -> None:
    gap = MissingFaceRange(Fraction(1), Fraction(2), 30, Fraction(0), "last-known-hold", "missing")
    source = replace(
        paired,
        tops=replace(paired.tops, derived_ranges=(gap,)),
        bottoms=replace(paired.bottoms, derived_ranges=(gap,)),
        sizes=(
            FaceSizeObservation(Fraction(0), 200, 400),
            FaceSizeObservation(Fraction(1), 100, 100, False),
        ),
    )
    assert placement.current_cells(source, start=Fraction(0), end=Fraction(3, 2)) == (
        placement.CurrentFaceCell(Fraction(0), Fraction(1), (0, 900, 200, 400)),
        placement.CurrentFaceCell(Fraction(1), Fraction(2), None),
    )


@pytest.mark.parametrize("fault", ["sizes", "outside", "provenance", "budget"])
def test_face_consumer_requires_paired_raw_coverage(
    paired: FaceScaleSeries,
    monkeypatch: pytest.MonkeyPatch,
    fault: str,
) -> None:
    source = replace(
        paired,
        sizes=()
        if fault == "sizes"
        else (
            FaceSizeObservation(Fraction(0), 200, 400, fault != "provenance"),
            FaceSizeObservation(Fraction(1), 100, 100),
        ),
    )
    if fault == "budget":
        monkeypatch.setattr(placement, "MAX_CELLS", 1)
    with pytest.raises(ReplayError):
        placement.current_cells(
            source, start=Fraction(0), end=Fraction(3 if fault == "outside" else 2)
        )


def test_scope_end_requires_a_new_seed_and_keeps_partial_terminal_frame(
    paired: FaceScaleSeries,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source = replace(
        paired,
        sizes=(
            FaceSizeObservation(Fraction(0), 200, 400),
            FaceSizeObservation(Fraction(1), 100, 100),
        ),
    )
    monkeypatch.setattr(placement, "MAX_SCOPE_SECONDS", 1)
    plan = placement.scoped_plan(
        source,
        pixels=VisibleArc((ArcCell(Fraction(0), Fraction(1), 700, 3, 0),), {}),
        start=Fraction(0),
        end=Fraction(3, 2),
    )
    assert tuple(cell.top for cell in plan.cells) == (700, None)
    assert plan.cells[1].end == Fraction(3, 2)
    assert plan.requests[0].reason == "scope-start"
    monkeypatch.setattr(placement, "MAX_SCOPE_SECONDS", Fraction(1, 2))
    with pytest.raises(ReplayError, match="scope"):
        placement.scoped_plan(source, pixels=VisibleArc((), {}), start=Fraction(0), end=Fraction(2))


def test_current_seed_only_translates_y_and_keeps_candidate_provenance(
    faces: tuple[placement.CurrentFaceCell, ...], pixels: VisibleArc
) -> None:
    plan = placement.derive(
        faces, pixels=pixels, start=Fraction(0), end=Fraction(4), width=1080, height=1920
    )
    assert tuple(cell.top for cell in plan.cells) == (700, 704, 702, 700)
    assert tuple(cell.policy for cell in plan.cells) == (
        "pixel-seed-candidate",
        "face-y-derived-candidate",
        "face-y-derived-candidate",
        "face-y-derived-candidate",
    )
    assert plan.requests == ()
    assert plan.cells[1].encoded() == {
        "start": "1",
        "end": "2",
        "currentFaceTime": "1",
        "currentFaceObserved": True,
        "estimatedUpperY": 704,
        "seedTime": "0",
        "seedAge": "1",
        "policy": "face-y-derived-candidate",
        "measuredCurrentUpperPixels": False,
        "semanticHeadVerified": False,
        "headNearVerified": False,
    }
    assert plan.at(Fraction(-1)) is None
    assert plan.at(Fraction(4)) is None
    assert plan.at(Fraction(4), ending=True) == plan.cells[3]
    assert plan.at(Fraction(0), ending=True) is None
    assert plan.encoded()["headNearVerified"] is False


@pytest.mark.parametrize(
    ("face", "reason"),
    [
        ((100, 907, 200, 400), "face-y-change"),
        ((100, 900, 200, 450), "face-shape-change"),
        ((100, 900, 250, 400), "face-shape-change"),
        ((107, 900, 200, 400), "face-x-change"),
    ],
)
def test_change_requests_measurement_instead_of_scaling_old_head(
    faces: tuple[placement.CurrentFaceCell, ...],
    pixels: VisibleArc,
    face: placement.FaceBox,
    reason: placement.Reason,
) -> None:
    plan = placement.derive(
        (faces[0], replace(faces[1], face=face), replace(faces[2], face=face)),
        pixels=pixels,
        start=Fraction(0),
        end=Fraction(3),
        width=1080,
        height=1920,
    )
    assert tuple(cell.top for cell in plan.cells) == (700, None, None)
    assert len(plan.requests) == 1
    assert plan.requests[0].reason == reason
    assert plan.requests[0].unsupported_start == Fraction(1)
    assert plan.requests[0].unsupported_end == Fraction(3)
    assert plan.requests[0].encoded()["kind"] == "current-upper-pixels"
    assert plan.cells[1].face_observed is True


def test_actual_scale_and_motion_adopt_new_same_pts_pixels_without_old_size_multiplier(
    faces: tuple[placement.CurrentFaceCell, ...],
) -> None:
    plan = placement.derive(
        (faces[0], replace(faces[1], face=(120, 1000, 400, 800))),
        pixels=VisibleArc(
            (
                ArcCell(Fraction(0), Fraction(1), 700, 3, 0),
                ArcCell(Fraction(1), Fraction(2), 650, 5, 2),
            ),
            {},
        ),
        start=Fraction(0),
        end=Fraction(2),
        width=1080,
        height=1920,
    )
    assert tuple(cell.top for cell in plan.cells) == (700, 650)
    assert plan.cells[1].seed_time == Fraction(1)
    assert plan.cells[1].policy == "pixel-seed-candidate"
    assert plan.requests == ()


def test_reset_and_missing_face_cannot_relabel_old_pixels_as_current(
    faces: tuple[placement.CurrentFaceCell, ...], pixels: VisibleArc
) -> None:
    plan = placement.derive(
        (faces[0], faces[1], replace(faces[2], face=None), faces[3]),
        pixels=pixels,
        start=Fraction(0),
        end=Fraction(4),
        width=1080,
        height=1920,
        resets=(Fraction(1),),
    )
    assert tuple(cell.top for cell in plan.cells) == (700, None, None, None)
    assert tuple(request.reason for request in plan.requests) == (
        "zoom-reset",
        "missing-face",
        "missing-face",
    )
    assert plan.requests[1].encoded()["kind"] == "current-face-observation"
    assert plan.cells[2].encoded()["currentFaceObserved"] is False
    assert plan.cells[3].encoded()["currentFaceObserved"] is True
    assert plan.cells[3].seed_time is None


def test_new_scope_discards_previous_seed_and_recovery_uses_exact_new_frame(
    faces: tuple[placement.CurrentFaceCell, ...],
) -> None:
    plan = placement.derive(
        faces[1:],
        pixels=VisibleArc((ArcCell(Fraction(2), Fraction(3), 690, 4, 1),), {}),
        start=Fraction(1),
        end=Fraction(4),
        width=1080,
        height=1920,
    )
    assert tuple(cell.top for cell in plan.cells) == (None, 690, 688)
    assert plan.requests[0].reason == "scope-start"
    assert plan.cells[2].seed_time == Fraction(2)


def test_resolution_scales_deadband_not_the_head_extension() -> None:
    plan = placement.derive(
        (
            placement.CurrentFaceCell(Fraction(0), Fraction(1), (10, 450, 100, 200)),
            placement.CurrentFaceCell(Fraction(1), Fraction(2), (10, 454, 100, 200)),
        ),
        pixels=VisibleArc((ArcCell(Fraction(0), Fraction(1), 350, 3, 0),), {}),
        start=Fraction(0),
        end=Fraction(2),
        width=540,
        height=960,
    )
    assert plan.cells[1].top is None
    assert plan.requests[0].reason == "face-y-change"


def test_unmeasured_scope_requests_changes_without_inventing_an_anchor() -> None:
    plan = placement.derive(
        (
            placement.CurrentFaceCell(Fraction(0), Fraction(1), (100, 900, 200, 400)),
            placement.CurrentFaceCell(Fraction(1), Fraction(2), (100, 910, 200, 400)),
            placement.CurrentFaceCell(Fraction(2), Fraction(3), (100, 912, 200, 400)),
        ),
        pixels=VisibleArc((), {}),
        start=Fraction(0),
        end=Fraction(3),
        width=1080,
        height=1920,
    )
    assert tuple(cell.top for cell in plan.cells) == (None, None, None)
    assert tuple(request.reason for request in plan.requests) == ("scope-start", "face-y-change")
    assert plan.requests[1].source_frame_start == Fraction(1)
    assert plan.requests[1].unsupported_end == Fraction(3)


def test_negative_derived_upper_position_requires_fresh_measurement() -> None:
    plan = placement.derive(
        (
            placement.CurrentFaceCell(Fraction(0), Fraction(1), (100, 900, 200, 400)),
            placement.CurrentFaceCell(Fraction(1), Fraction(2), (100, 896, 200, 400)),
        ),
        pixels=VisibleArc((ArcCell(Fraction(0), Fraction(1), 2, 3, 0),), {}),
        start=Fraction(0),
        end=Fraction(2),
        width=1080,
        height=1920,
    )
    assert plan.cells[1].top is None
    assert plan.cells[1].policy == "measurement-required"
    assert plan.requests[0].reason == "face-y-change"


@pytest.mark.parametrize(
    "cell",
    [
        ArcCell(Fraction(0), Fraction(2), 700, 3, 0),
        ArcCell(Fraction(0), Fraction(1), 901, 3, 0),
        ArcCell(Fraction(0), Fraction(1), float("nan"), 3, 0),
        ArcCell(Fraction(0), Fraction(1), 700, 2, 0),
    ],
)
def test_seed_requires_same_pts_frame_end_and_supported_canvas_pixels(
    faces: tuple[placement.CurrentFaceCell, ...],
    cell: ArcCell,
) -> None:
    with pytest.raises(ReplayError, match="same-PTS"):
        placement.derive(
            faces,
            pixels=VisibleArc((cell,), {}),
            start=Fraction(0),
            end=Fraction(4),
            width=1080,
            height=1920,
        )


@pytest.mark.parametrize(
    "fault", ["empty", "width", "end", "hole", "long-cell", "reset", "order", "face", "age"]
)
def test_source_scope_and_current_geometry_are_bounded(
    faces: tuple[placement.CurrentFaceCell, ...],
    pixels: VisibleArc,
    fault: str,
) -> None:
    selected = () if fault == "empty" else faces
    if fault == "hole":
        selected = (faces[0], faces[2], faces[3])
    if fault == "long-cell":
        selected = (replace(faces[0], end=Fraction(2)), faces[2], faces[3])
    if fault == "face":
        selected = (replace(faces[0], face=(100, 900, 2000, 400)), *faces[1:])
    resets: tuple[Fraction, ...] = (Fraction(1, 2),) if fault == "reset" else ()
    if fault == "order":
        resets = (Fraction(2), Fraction(1))
    with pytest.raises(ReplayError):
        placement.derive(
            selected,
            pixels=pixels,
            start=Fraction(0),
            end=Fraction(121) if fault == "age" else Fraction(5 if fault == "end" else 4),
            width=0 if fault == "width" else 1080,
            height=1920,
            resets=resets,
        )


@pytest.mark.parametrize(
    ("height", "text"), [(100, "甲"), (220, "甲\n乙"), (340, "甲\n乙\n丙"), (100, "丁")]
)
def test_known_head_bottom_is_identical_for_one_two_three_one_rows(
    faces: tuple[placement.CurrentFaceCell, ...],
    pixels: VisibleArc,
    height: float,
    text: str,
) -> None:
    plan = placement.derive(
        (faces[0],), pixels=pixels, start=Fraction(0), end=Fraction(1), width=1080, height=1920
    )
    title = ET.fromstring('<title offset="0s" start="3600s" duration="1s"/>')
    ET.SubElement(title, "text").text = text
    before = ET.tostring(title)
    assert placement.title_samples(
        title,
        plan=plan,
        caption_height=height,
        canvas_height=1920,
        stacks=(),
    ) == (("3600s", "0 14.7916666667"), ("3601s", "0 14.7916666667"))
    assert ET.tostring(title) == before


@pytest.mark.parametrize(("start", "height"), [(0, 100), (1, 220), (2, 340), (3, 100)])
def test_unknown_head_reserves_largest_measured_stack_without_fake_observation(
    faces: tuple[placement.CurrentFaceCell, ...],
    start: int,
    height: float,
) -> None:
    plan = placement.derive(
        tuple(replace(cell, face=None) for cell in faces),
        pixels=VisibleArc((), {}),
        start=Fraction(0),
        end=Fraction(4),
        width=1080,
        height=1920,
    )
    stacks = placement.unknown_stacks(
        plan,
        (
            (Fraction(0), Fraction(1), 100),
            (Fraction(1), Fraction(2), 220),
            (Fraction(2), Fraction(3), 340),
            (Fraction(3), Fraction(4), 100),
        ),
    )
    assert stacks == (ScreenStack(Fraction(0), Fraction(4), 340),)
    title = ET.Element("title", offset=f"{start}s", start="3600s", duration="1s")
    assert placement.title_samples(
        title,
        plan=plan,
        caption_height=height,
        canvas_height=1920,
        stacks=stacks,
    ) == (("3600s", "0 31.0416666667"), ("3601s", "0 31.0416666667"))
    assert plan.requests[0].encoded()["kind"] == "current-face-observation"


@pytest.mark.parametrize(
    "fault",
    [
        "no-block",
        "height",
        "clock",
        "title",
        "duration",
        "outside",
        "small-stack",
        "headroom",
        "hole",
    ],
)
def test_caption_bounds_do_not_hide_missing_measurements(
    faces: tuple[placement.CurrentFaceCell, ...],
    pixels: VisibleArc,
    fault: str,
) -> None:
    plan = placement.derive(
        faces, pixels=pixels, start=Fraction(0), end=Fraction(4), width=1080, height=1920
    )
    title = ET.fromstring('<title offset="0s" start="3600s" duration="4s"/>')
    if fault in ("no-block", "height", "clock"):
        with pytest.raises(ReplayError):
            placement.unknown_stacks(
                plan,
                ()
                if fault == "no-block"
                else (
                    (
                        Fraction(1),
                        Fraction(0) if fault == "clock" else Fraction(2),
                        float("nan") if fault == "height" else 100,
                    ),
                ),
            )
        return
    if fault == "title":
        title.tag = "gap"
    if fault == "duration":
        title.set("duration", "0s")
    if fault == "outside":
        title.set("offset", "4s")
    if fault == "small-stack":
        plan = replace(plan, cells=tuple(replace(cell, top=None) for cell in plan.cells))
    if fault == "hole":
        plan = replace(plan, cells=(plan.cells[0], plan.cells[2], plan.cells[3]))
    with pytest.raises(ReplayError):
        placement.title_samples(
            title,
            plan=plan,
            caption_height=800 if fault == "headroom" else 100,
            canvas_height=1920,
            stacks=(ScreenStack(Fraction(0), Fraction(4), 50),),
        )
