"""Source-caption survey evidence for a source assembled from sampled-OCR repair chunks.

The completion gate accepts "no source captions" only from a survey of the media that is
actually rendered. Each chunk's accepted attempt was audited by Vision OCR of every Nth
repaired frame; the assembly is a verified stream copy of exactly those chunk payloads. This
binds those audits to the assembled source by hash and states the sampling and its limits.
Rejected attempts and escalations are disclosed. It never approves quality or removes review.
"""

import json
import os
from dataclasses import dataclass
from pathlib import Path

from replay.local_asr import fingerprint
from replay.recordings import record

REASON = "sampled-repair-audit-no-confident-residual-text"
MIN_CONFIDENCE = 0.9


@dataclass(frozen=True, slots=True)
class ChunkAudit:
    """What one chunk's accepted audit observed, plus every attempt that was rejected."""

    index: int
    audited_frames: int
    text_frames: int
    stride: int
    escalated: bool
    rejected_hits: int


def _read(path: Path) -> dict[str, object]:
    return record(json.loads(path.read_text(encoding="utf-8")))


def _pinned(reference: object) -> dict[str, object]:
    pin = record(reference)
    path = Path(str(pin.get("path")))
    if fingerprint(path) != pin.get("sha256"):
        raise ValueError(f"Pinned survey input changed: {path.name}")
    return _read(path)


def _attempts(receipt: dict[str, object]) -> list[dict[str, object]]:
    attempts = receipt.get("attempts")
    if not isinstance(attempts, list):
        raise ValueError("Chunk receipt lacks attempts")
    return [record(item) for item in attempts]


def _count(value: object) -> int:
    if type(value) is not int or value < 0:
        raise ValueError("Expected a nonnegative audit count")
    return value


def _lines(attempt: dict[str, object]) -> list[object]:
    lines = attempt.get("residualLines")
    if not isinstance(lines, list):
        raise ValueError("Attempt lacks residual line records")
    return lines


def _chunk(index: int, directory: Path, assembled: dict[str, str]) -> ChunkAudit:
    """Read one chunk's pinned receipt and check its media against the assembly bindings."""
    receipt = _pinned(_read(directory / "chunk.json").get("sampledReceipt"))
    candidate = _read(directory / "repair" / "candidate.json")
    if assembled.get(str(candidate.get("mediaOutput"))) != candidate.get("videoSHA256"):
        raise ValueError("Chunk media is not the payload the assembly verified")
    attempts = _attempts(receipt)
    accepted = [a for a in attempts if a.get("directory") == receipt.get("accepted")]
    if len(accepted) != 1 or accepted[0].get("passed") is not True:
        raise ValueError("Chunk has no passed accepted attempt")
    rejected = sum(len(_lines(a)) for a in attempts if a.get("passed") is not True)
    return ChunkAudit(
        index=index,
        audited_frames=_count(accepted[0].get("auditedFrames")),
        text_frames=len({str(record(line).get("frame")) for line in _lines(accepted[0])}),
        stride=_count(accepted[0].get("auditStride")),
        escalated=len(attempts) > 1,
        rejected_hits=rejected,
    )


def _summary(rows: list[ChunkAudit], assembly: Path) -> dict[str, object]:
    observed = sum(row.audited_frames for row in rows)
    text = sum(row.text_frames for row in rows)
    return {
        "masks": 0,
        "reason": REASON,
        "observedFrames": observed,
        "observedTextFrames": text,
        "humanReviewed": False,
        "noSourceSubtitlesDetected": text == 0,
        "evidence": "inputs.json",
        "survey": {
            "kind": "sampled-repair-audit",
            "ocr": "Apple Vision, every audit-stride-th repaired frame plus the last of each chunk",
            "minimumConfidence": MIN_CONFIDENCE,
            "scope": "confident lines overlapping the frame's source-caption search region",
            "auditStrides": sorted({row.stride for row in rows}),
            "chunks": len(rows),
            "escalatedChunks": [row.index for row in rows if row.escalated],
            "rejectedAttemptHits": sum(row.rejected_hits for row in rows),
            "assemblyReceipt": {"path": str(assembly), "sha256": fingerprint(assembly)},
            "limits": "Sampled frames only; isolated residual text between them is not excluded.",
        },
        "qualityApproved": False,
    }


def survey(plan: Path, assembly: Path, output: Path) -> Path:
    """Write ``summary.json`` and ``inputs.json`` under ``output`` (created, never replaced)."""
    plan_data = _read(plan)
    proof = _read(assembly)
    if (
        proof.get("exactPresentationClockVerified") is not True
        or proof.get("compressedPacketsIdentical") is not True
    ):
        raise ValueError("Assembly does not prove an exact clock and identical payloads")
    bound = {key: str(value) for key, value in record(proof.get("inputs")).items()}
    count = plan_data.get("chunks")
    if type(count) is not int or count < 1:
        raise ValueError("Plan lacks a chunk count")
    rows = [_chunk(i, plan.parent / f"chunk-{i:05d}", bound) for i in range(count)]
    output.mkdir(mode=0o700, parents=True)
    summary = output / "summary.json"
    for name, value in (("inputs.json", bound), ("summary.json", _summary(rows, assembly))):
        descriptor = os.open(output / name, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)
        with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
            json.dump(value, stream, indent=2, sort_keys=True)
    return summary
