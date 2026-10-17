"""Sampled eligibility accepts only honest, internally consistent sampled-OCR evidence."""

import copy
from fractions import Fraction

import pytest

from replay import subtitle_sampled_eligibility as eligibility
from replay import subtitle_yolo_admission as yolo_admission
from replay.caption_io import FrameObservation
from replay.caption_masks import MaskPolicy, TextDetection, TextRectangle
from replay.subtitle_cleanup_plan import CaptionSearchProfile
from replay.subtitle_clock import VideoFrameClock
from replay.subtitle_profile import SubtitleBands
from replay.subtitle_sampled_admission import admission
from replay.subtitle_stream import SubtitleSpan

WIDTH = 1080
HEIGHT = 1920
SOURCE = "a" * 64
PINS = {"source": SOURCE, "native": "b" * 64}
CLOCK = VideoFrameClock(tuple(range(0, 10 * 3000, 3000)), Fraction(1, 90000), 3000)
PROFILE = CaptionSearchProfile(
    SubtitleBands(WIDTH, HEIGHT, primary=(42,), secondary=()), (0, 1283, 1077, 167)
)
LINE = TextDetection("こんにちは世界です", 0.9, TextRectangle(200, 1330, 600, 48))


def _observation(index: int) -> FrameObservation:
    seconds = float(CLOCK.pts[index] * CLOCK.time_base)
    return FrameObservation(seconds, seconds, WIDTH, HEIGHT, (LINE,))


def _admission(detected: list[tuple[int, int, int, int] | None]) -> dict[str, object]:
    return admission(
        sampled={0: _observation(0), 9: _observation(9)},
        clock=CLOCK,
        profile=PROFILE,
        policy=MaskPolicy(),
        stride=9,
        triggers=(),
        detected=detected,
        radius=2,
    )


def _receipt(plan: dict[str, object]) -> dict[str, object]:
    attempt = {
        "stride": 9,
        "directory": "attempt-sampled",
        "sampling": plan["sampling"],
        "auditedFrames": 2,
        "residualLines": [],
        "passed": True,
    }
    return {
        "version": 1,
        "kind": "sampled-ocr-repair-chunk",
        "attempts": [attempt],
        "accepted": "attempt-sampled",
        "escalated": False,
        "regionsFromSampling": True,
        "pins": PINS,
        "qualityApproved": False,
        "deliveryComplete": False,
    }


def _section(parent: dict[str, object], key: str) -> dict[str, object]:
    value = parent[key]
    if not isinstance(value, dict):
        raise TypeError(key)
    return value


def _rows(parent: dict[str, object], key: str) -> list[dict[str, object]]:
    value = parent[key]
    if not isinstance(value, list):
        raise TypeError(key)
    return value


def _check(receipt: dict[str, object], plan: dict[str, object]) -> tuple[SubtitleSpan, ...]:
    return eligibility.validate_sampled_eligibility(
        receipt, plan, source_sha256=SOURCE, clock=CLOCK, width=WIDTH, height=HEIGHT, pins=PINS
    )


def test_consistent_evidence_returns_the_covering_span() -> None:
    plan = _admission([None] * 10)
    spans = _check(_receipt(plan), plan)
    assert [(s.start_frame, s.end_frame) for s in spans] == [(0, 10)]


def test_all_unknown_chunk_is_valid_and_has_no_spans() -> None:
    plan = admission(
        sampled={
            0: FrameObservation(0.0, 0.0, WIDTH, HEIGHT, ()),
            9: FrameObservation(
                float(CLOCK.pts[9] * CLOCK.time_base),
                float(CLOCK.pts[9] * CLOCK.time_base),
                WIDTH,
                HEIGHT,
                (),
            ),
        },
        clock=CLOCK,
        profile=PROFILE,
        policy=MaskPolicy(),
        stride=9,
        triggers=(),
        detected=[None] * 10,
        radius=1,
    )
    assert _check(_receipt(plan), plan) == ()


def test_raw_ocr_claim_is_rejected() -> None:
    plan = _admission([None] * 10)
    _section(plan, "sampling")["rawOcr"] = True
    with pytest.raises(ValueError, match="honestly"):
        _check(_receipt(plan), plan)


def test_wide_sample_gap_is_rejected() -> None:
    plan = _admission([None] * 10)
    _section(plan, "sampling")["stride"] = 3
    with pytest.raises(ValueError, match="gap wider"):
        _check(_receipt(plan), plan)


def test_sampled_frames_must_include_first_and_last_frame() -> None:
    plan = _admission([None] * 10)
    _section(plan, "sampling")["sampledFrames"] = [3, 9]
    with pytest.raises(ValueError, match="first and last"):
        _check(_receipt(plan), plan)


def test_non_integer_sampled_frames_are_rejected() -> None:
    plan = _admission([None] * 10)
    _section(plan, "sampling")["sampledFrames"] = [0, "9"]
    with pytest.raises(ValueError, match="Invalid sampled"):
        _check(_receipt(plan), plan)


def test_failed_audit_is_not_acceptable() -> None:
    plan = _admission([None] * 10)
    receipt = _receipt(plan)
    _rows(receipt, "attempts")[0]["residualLines"] = [{"frame": 3}]
    with pytest.raises(ValueError, match="residual-text audit"):
        _check(receipt, plan)


def test_accepted_attempt_must_be_the_last() -> None:
    plan = _admission([None] * 10)
    receipt = _receipt(plan)
    attempts = _rows(receipt, "attempts")
    attempts.append(copy.deepcopy(attempts[0]) | {"directory": "attempt-full"})
    receipt["escalated"] = True
    with pytest.raises(ValueError, match="residual-text audit"):
        _check(receipt, plan)


def test_pins_must_match_the_declared_source() -> None:
    plan = _admission([None] * 10)
    receipt = _receipt(plan)
    receipt["pins"] = {"source": "c" * 64}
    with pytest.raises(ValueError, match="pins"):
        _check(receipt, plan)


def test_delivery_flag_must_stay_false() -> None:
    plan = _admission([None] * 10)
    receipt = _receipt(plan)
    receipt["deliveryComplete"] = True
    with pytest.raises(ValueError, match="approval flags"):
        _check(receipt, plan)


def test_missing_accepted_attempt_is_rejected() -> None:
    plan = _admission([None] * 10)
    receipt = _receipt(plan)
    receipt["accepted"] = None
    with pytest.raises(ValueError, match="unique accepted"):
        _check(receipt, plan)


def test_attempt_count_is_bounded() -> None:
    plan = _admission([None] * 10)
    receipt = _receipt(plan)
    receipt["attempts"] = []
    with pytest.raises(ValueError, match="one or two"):
        _check(receipt, plan)


def test_sampling_must_match_the_accepted_attempt() -> None:
    plan = _admission([None] * 10)
    receipt = _receipt(plan)
    _rows(receipt, "attempts")[0]["sampling"] = {"kind": "other"}
    with pytest.raises(ValueError, match="differs from the accepted"):
        _check(receipt, plan)


def test_admission_policy_must_match() -> None:
    plan = _admission([None] * 10)
    receipt = _receipt(plan)
    plan["policy"] = "other"
    with pytest.raises(ValueError, match="policy"):
        _check(receipt, plan)


def test_frame_record_pts_must_match_the_clock() -> None:
    plan = _admission([None] * 10)
    receipt = _receipt(plan)
    _rows(plan, "frames")[4]["sourcePTS"] = 1
    with pytest.raises(ValueError, match="record 4"):
        _check(receipt, plan)


def test_sampled_flag_must_match_the_sampled_frames() -> None:
    plan = _admission([None] * 10)
    receipt = _receipt(plan)
    _rows(plan, "frames")[4]["sampled"] = True
    with pytest.raises(ValueError, match="record 4"):
        _check(receipt, plan)


def test_frame_count_must_match_the_clock() -> None:
    plan = _admission([None] * 10)
    receipt = _receipt(plan)
    _rows(plan, "frames").pop()
    with pytest.raises(ValueError, match="frame records"):
        _check(receipt, plan)


def test_unknown_list_must_match_records() -> None:
    plan = _admission([None] * 10)
    receipt = _receipt(plan)
    plan["unknownSourcePTS"] = [0]
    with pytest.raises(ValueError, match="Unknown source PTS"):
        _check(receipt, plan)


def test_non_object_receipt_is_rejected() -> None:
    plan = _admission([None] * 10)
    with pytest.raises(ValueError, match="chunk receipt"):
        eligibility.validate_sampled_eligibility(
            [], plan, source_sha256=SOURCE, clock=CLOCK, width=WIDTH, height=HEIGHT, pins=PINS
        )


def _yolo_plan() -> dict[str, object]:
    boxes: list[list[tuple[float, float, float, float]]] = [[(200.0, 1330.0, 800.0, 1378.0)]] * 4
    return yolo_admission.admission(boxes=boxes + [[]] * 6, clock=CLOCK, profile=PROFILE, radius=0)


def test_a_yolo_planned_admission_is_eligible_under_its_own_rules() -> None:
    plan = _yolo_plan()
    spans = _check(_receipt(plan), plan)
    assert [(s.start_frame, s.end_frame) for s in spans] == [(0, 4)]


def test_a_yolo_sampling_receipt_under_the_ocr_policy_is_rejected() -> None:
    plan = _yolo_plan()
    plan["policy"] = "sampled-ocr-plus-detector-regions-v5"
    with pytest.raises(ValueError, match="policy and sampling kind disagree"):
        _check(_receipt(plan), plan)


def test_a_yolo_admission_claiming_ocr_or_partial_detection_is_rejected() -> None:
    plan = _yolo_plan()
    _section(plan, "sampling")["detectedFrames"] = 9
    with pytest.raises(ValueError, match="honestly describe YOLO detection"):
        _check(_receipt(plan), plan)


def test_a_yolo_frame_claiming_ocr_sampling_is_rejected() -> None:
    plan = _yolo_plan()
    _rows(plan, "frames")[0]["sampled"] = True
    with pytest.raises(ValueError, match="differs from clock or repair spans"):
        _check(_receipt(plan), plan)
