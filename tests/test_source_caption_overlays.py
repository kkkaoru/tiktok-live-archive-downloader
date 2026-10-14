"""Pinned alpha audit reuse never turns into a raw-video overlay allowlist."""

import json
from pathlib import Path

import pytest

from replay.core import ReplayError
from replay.fcp_part_assembly import _reference
from replay.source_caption_overlays import validate_overlays


@pytest.fixture
def evidence(tmp_path: Path) -> tuple[dict[str, object], dict[str, object], Path]:
    alpha = tmp_path / "caption.mov"
    alpha.write_bytes(b"synthetic audited alpha")
    source = tmp_path / "caption.moti"
    source.write_bytes(b"synthetic title")
    plan = tmp_path / "plan.json"
    plan.write_text(json.dumps({"frames": 30, "rate": 30}))
    state = tmp_path / "state.json"
    state.write_text("{}")
    part: dict[str, object] = {
        "alpha": _reference(alpha),
        "source": _reference(source),
        "plan": _reference(plan),
        "producerState": _reference(state),
        "frames": 30,
        "historicalSubmissionBound": True,
        "motionSourcePlanStyleVerified": True,
        "checks": {"boundsAndGapsVerified": True, "gapsVerified": True},
        "measurement": {
            "fullDecodeVerified": True,
            "decodedFrames": 30,
            "clock": {"frames": 30, "start": "0", "end": "1", "exactCFR": True},
        },
    }
    audit: dict[str, object] = {
        "kind": "delivery-caption-audit",
        "alphaOnlyAccepted": True,
        "measurements": {"rate": 30, "parts": [part]},
    }
    return audit, part, alpha


def check(
    tmp_path: Path,
    audit: dict[str, object],
    alpha: Path,
    *,
    role: str = "motion-caption-alpha",
    start: float = 0,
    duration: float = 1,
) -> None:
    path = tmp_path / "audit.json"
    path.write_text(json.dumps(audit))
    validate_overlays(
        [
            {
                "sourcePath": str(alpha),
                "selection": {"startSeconds": start, "durationSeconds": duration, "rate": 1},
            }
        ],
        config={"captionOverlays": [{"role": role, "audit": _reference(path)}]},
        clean=tmp_path / "clean.mov",
    )


def test_existing_alpha_only_audit_allows_timed_caption_overlay(
    tmp_path: Path, evidence: tuple[dict[str, object], dict[str, object], Path]
) -> None:
    audit, _, alpha = evidence
    check(tmp_path, audit, alpha, start=0.25, duration=0.5)


@pytest.mark.parametrize("field", ["alphaOnlyAccepted", "kind"])
def test_unapproved_or_wrong_audit_cannot_authorize_video(
    tmp_path: Path, evidence: tuple[dict[str, object], dict[str, object], Path], field: str
) -> None:
    audit, _, alpha = evidence
    audit[field] = False
    with pytest.raises(ReplayError):
        check(tmp_path, audit, alpha)


@pytest.mark.parametrize(
    "field", ["historicalSubmissionBound", "motionSourcePlanStyleVerified", "frames"]
)
def test_missing_producer_or_clock_proof_fails(
    tmp_path: Path, evidence: tuple[dict[str, object], dict[str, object], Path], field: str
) -> None:
    audit, part, alpha = evidence
    part[field] = False
    with pytest.raises(ReplayError):
        check(tmp_path, audit, alpha)


@pytest.mark.parametrize(
    "mode", ["raw", "hash", "role", "negative", "past-end", "nan", "empty", "no-rate"]
)
def test_overlay_substitution_and_invalid_clock_fail(
    tmp_path: Path, evidence: tuple[dict[str, object], dict[str, object], Path], mode: str
) -> None:
    audit, _, alpha = evidence
    if mode == "raw":
        alpha = tmp_path / "original.mp4"
    if mode == "hash":
        alpha.write_bytes(b"changed raw video")
    if mode == "empty":
        audit["measurements"] = {"rate": 30, "parts": []}
    if mode == "no-rate":
        audit["measurements"] = {"rate": None, "parts": []}
    with pytest.raises(ReplayError):
        check(
            tmp_path,
            audit,
            alpha,
            role="raw-video" if mode == "role" else "motion-caption-alpha",
            start=-1 if mode == "negative" else 0,
            duration=2 if mode == "past-end" else float("nan") if mode == "nan" else 1,
        )


def test_invalid_evidence_list_fails(tmp_path: Path) -> None:
    with pytest.raises(ReplayError, match="list"):
        validate_overlays([], config={"captionOverlays": {}}, clean=tmp_path / "clean.mov")
