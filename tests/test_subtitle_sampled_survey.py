"""The survey binds sampled-repair audits to the assembled payloads and discloses limits."""

import hashlib
import json
from pathlib import Path

import pytest

from replay import subtitle_sampled_survey as survey
from replay.recordings import record


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value), encoding="utf-8")


def _attempt(name: str, *, passed: bool, lines: list[dict[str, object]]) -> dict[str, object]:
    return {
        "directory": name,
        "auditedFrames": 181,
        "auditStride": 5,
        "residualLines": lines,
        "passed": passed,
    }


def _chunk(root: Path, index: int, attempts: list[dict[str, object]]) -> str:
    directory = root / f"chunk-{index:05d}"
    receipt = root / f"receipt-{index}.json"
    _write(receipt, {"attempts": attempts, "accepted": attempts[-1]["directory"]})
    receipt.write_text(
        json.dumps(
            {
                "attempts": attempts,
                "accepted": attempts[-1]["directory"],
                "escalated": len(attempts) > 1,
            }
        ),
        encoding="utf-8",
    )
    _write(
        directory / "chunk.json",
        {"sampledReceipt": {"path": str(receipt), "sha256": _sha(receipt)}},
    )
    media = root / f"media-{index}.mp4"
    media.write_bytes(f"video-{index}".encode())
    _write(
        directory / "repair" / "candidate.json",
        {"mediaOutput": str(media), "videoSHA256": _sha(media)},
    )
    return str(media)


def _world(root: Path, *, escalate: bool = False) -> tuple[Path, Path]:
    clean = [_attempt("attempt-sampled", passed=True, lines=[])]
    bad = [
        _attempt("attempt-sampled", passed=False, lines=[{"frame": 580, "text": "x"}]),
        _attempt("attempt-full", passed=True, lines=[]),
    ]
    media = [_chunk(root, 0, clean), _chunk(root, 1, bad if escalate else clean)]
    _write(root / "plan.json", {"chunks": 2})
    assembly = root / "receipt.json"
    _write(
        assembly,
        {
            "exactPresentationClockVerified": True,
            "compressedPacketsIdentical": True,
            "inputs": {item: _sha(Path(item)) for item in media},
        },
    )
    return root / "plan.json", assembly


def _summary(path: Path) -> dict[str, object]:
    return record(json.loads(path.read_text(encoding="utf-8")))


def test_clean_audits_make_a_no_subtitle_summary(tmp_path: Path) -> None:
    plan, assembly = _world(tmp_path)
    summary = _summary(survey.survey(plan, assembly, tmp_path / "out"))
    assert (summary["masks"], summary["observedFrames"], summary["observedTextFrames"]) == (
        0,
        362,
        0,
    )
    assert summary["noSourceSubtitlesDetected"] is True
    assert summary["humanReviewed"] is False
    assert summary["qualityApproved"] is False


def test_escalation_and_rejected_hits_are_disclosed(tmp_path: Path) -> None:
    plan, assembly = _world(tmp_path, escalate=True)
    summary = _summary(survey.survey(plan, assembly, tmp_path / "out"))
    detail = summary["survey"]
    assert isinstance(detail, dict)
    assert (detail["escalatedChunks"], detail["rejectedAttemptHits"]) == ([1], 1)


def test_confident_residual_in_the_accepted_attempt_blocks_the_claim(tmp_path: Path) -> None:
    plan, assembly = _world(tmp_path)
    receipt = tmp_path / "receipt-0.json"
    data = json.loads(receipt.read_text("utf-8"))
    data["attempts"][0]["residualLines"] = [{"frame": 3}]
    data["attempts"][0]["passed"] = True
    receipt.write_text(json.dumps(data), encoding="utf-8")
    chunk = tmp_path / "chunk-00000/chunk.json"
    _write(chunk, {"sampledReceipt": {"path": str(receipt), "sha256": _sha(receipt)}})
    summary = _summary(survey.survey(plan, assembly, tmp_path / "out"))
    assert (summary["observedTextFrames"], summary["noSourceSubtitlesDetected"]) == (1, False)


def test_changed_receipt_is_rejected(tmp_path: Path) -> None:
    plan, assembly = _world(tmp_path)
    (tmp_path / "receipt-0.json").write_text("{}", encoding="utf-8")
    with pytest.raises(ValueError, match="changed"):
        survey.survey(plan, assembly, tmp_path / "out")


def test_media_the_assembly_did_not_verify_is_rejected(tmp_path: Path) -> None:
    plan, assembly = _world(tmp_path)
    data = json.loads(assembly.read_text("utf-8"))
    data["inputs"] = {}
    assembly.write_text(json.dumps(data), encoding="utf-8")
    with pytest.raises(ValueError, match="assembly verified"):
        survey.survey(plan, assembly, tmp_path / "out")


def test_unproven_assembly_is_rejected(tmp_path: Path) -> None:
    plan, assembly = _world(tmp_path)
    data = json.loads(assembly.read_text("utf-8"))
    data["compressedPacketsIdentical"] = False
    assembly.write_text(json.dumps(data), encoding="utf-8")
    with pytest.raises(ValueError, match="exact clock"):
        survey.survey(plan, assembly, tmp_path / "out")


def test_plan_without_chunks_is_rejected(tmp_path: Path) -> None:
    plan, assembly = _world(tmp_path)
    _write(plan, {"chunks": 0})
    with pytest.raises(ValueError, match="chunk count"):
        survey.survey(plan, assembly, tmp_path / "out")


def test_failed_accepted_attempt_is_rejected(tmp_path: Path) -> None:
    plan, assembly = _world(tmp_path)
    receipt = tmp_path / "receipt-1.json"
    data = json.loads(receipt.read_text("utf-8"))
    data["attempts"][0]["passed"] = False
    receipt.write_text(json.dumps(data), encoding="utf-8")
    _write(
        tmp_path / "chunk-00001/chunk.json",
        {"sampledReceipt": {"path": str(receipt), "sha256": _sha(receipt)}},
    )
    with pytest.raises(ValueError, match="passed accepted"):
        survey.survey(plan, assembly, tmp_path / "out")


def test_malformed_attempts_and_counts_are_rejected(tmp_path: Path) -> None:
    plan, assembly = _world(tmp_path)
    receipt = tmp_path / "receipt-0.json"
    receipt.write_text(json.dumps({"attempts": 5, "accepted": "x"}), encoding="utf-8")
    _write(
        tmp_path / "chunk-00000/chunk.json",
        {"sampledReceipt": {"path": str(receipt), "sha256": _sha(receipt)}},
    )
    with pytest.raises(ValueError, match="lacks attempts"):
        survey.survey(plan, assembly, tmp_path / "out")


def test_negative_audit_count_is_rejected(tmp_path: Path) -> None:
    plan, assembly = _world(tmp_path)
    receipt = tmp_path / "receipt-0.json"
    data = json.loads(receipt.read_text("utf-8"))
    data["attempts"][0]["auditedFrames"] = -1
    receipt.write_text(json.dumps(data), encoding="utf-8")
    _write(
        tmp_path / "chunk-00000/chunk.json",
        {"sampledReceipt": {"path": str(receipt), "sha256": _sha(receipt)}},
    )
    with pytest.raises(ValueError, match="nonnegative"):
        survey.survey(plan, assembly, tmp_path / "out")


def test_residual_record_must_be_a_list(tmp_path: Path) -> None:
    plan, assembly = _world(tmp_path)
    receipt = tmp_path / "receipt-0.json"
    data = json.loads(receipt.read_text("utf-8"))
    data["attempts"][0]["residualLines"] = None
    receipt.write_text(json.dumps(data), encoding="utf-8")
    _write(
        tmp_path / "chunk-00000/chunk.json",
        {"sampledReceipt": {"path": str(receipt), "sha256": _sha(receipt)}},
    )
    with pytest.raises(ValueError, match="residual line records"):
        survey.survey(plan, assembly, tmp_path / "out")
