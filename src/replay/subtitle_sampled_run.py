"""Sampled-OCR repair of one chunk with an audit gate and escalation to full OCR.

Attempt 1 uses the configured stride. The repaired video is audited by OCR of every Nth
frame; a confident residual line escalates to an attempt that recognises every frame (the
non-sampled admission). Each attempt keeps its own directory and nothing is overwritten.
The receipt states that regions came from sampling and that quality is not approved.
"""

import json
import os
import time
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from itertools import pairwise
from pathlib import Path

from replay.caption_io import FrameObservation
from replay.subtitle_colour_residual import GATE, flagged
from replay.subtitle_region_fill import Region
from replay.subtitle_repair_audit import Residual, audit_indices, residual_lines

FULL_STRIDE = 1
AUDIT_STRIDE = 5


@dataclass(frozen=True, slots=True)
class Collaborators:
    """Injectable work: planning, rendering and auditing of one attempt."""

    plan: Callable[[int, Path], dict[str, object]]
    render: Callable[[dict[str, object], Path], Path]
    audit: Callable[[Path, Sequence[int]], Mapping[int, FrameObservation]]
    colour: Callable[[dict[str, object], Path, Sequence[int]], Sequence[int]] | None = None
    caption_coloured: Callable[[Sequence[Residual]], Sequence[bool]] | None = None
    # Registers the residual lines a failed full attempt left, as repair boxes for one more try.
    residual: Callable[[Sequence[Residual], int], None] | None = None


def _regions(plan: Mapping[str, object]) -> list[Region | None]:
    frames = plan["frames"]
    if not isinstance(frames, list):
        raise ValueError("Admission lacks per-frame entries")
    regions: list[Region | None] = []
    for entry in frames:
        box = entry.get("region") if isinstance(entry, dict) else None
        if isinstance(box, dict):
            regions.append((box["x"], box["y"], box["x"] + box["width"], box["y"] + box["height"]))
        else:
            regions.append(None)
    return regions


def _stacked(region: Region) -> Region:
    """The region with its own height added above and below it.

    A caption block taller than the profiled band stacks its extra lines there; only
    caption-coloured lines among them fail the audit, so scene text beside it stays interface.
    """
    left, top, right, bottom = region
    return left, top - (bottom - top), right, bottom + (bottom - top)


def _attempt(
    stride: int, directory: Path, tools: Collaborators, audit_stride: int
) -> dict[str, object]:
    directory.mkdir(mode=0o700)
    marks = [time.monotonic()]
    plan = tools.plan(stride, directory)
    marks.append(time.monotonic())
    video = tools.render(plan, directory)
    marks.append(time.monotonic())
    regions = _regions(plan)
    indices = audit_indices(len(regions), audit_stride)
    observed = tools.audit(video, indices)
    residuals = residual_lines(
        observed, [None if box is None else _stacked(box) for box in regions]
    )
    interface: list[Residual] = []
    if tools.caption_coloured is not None and residuals:
        flags = tools.caption_coloured(residuals)
        interface = [item for item, coloured in zip(residuals, flags, strict=True) if not coloured]
        residuals = tuple(item for item, coloured in zip(residuals, flags, strict=True) if coloured)
    counts: Sequence[int] = () if tools.colour is None else tools.colour(plan, video, indices)
    marks.append(time.monotonic())
    over = flagged(counts, gate=GATE)
    return {
        "stride": stride,
        "directory": directory.name,
        "video": video.name,
        "sampling": plan.get("sampling"),
        "stageSeconds": plan.get("stageSeconds"),
        # Wall time of plan, render and audit (with the colour gate): where a chunk's time goes.
        "phaseSeconds": dict(
            zip(
                ("plan", "render", "audit"),
                (round(b - a, 3) for a, b in pairwise(marks)),
                strict=True,
            )
        ),
        "auditStride": audit_stride,
        "auditedFrames": len(observed),
        "residualLines": [_residual(item) for item in residuals],
        "interfaceText": [_residual(item) for item in interface],
        "colourResidual": None
        if tools.colour is None
        else {
            "gate": GATE,
            "flaggedFrames": [indices[i] for i in over],
            "maximum": max(counts, default=0),
        },
        "passed": not residuals and not over,
    }


def _residual(item: Residual) -> dict[str, object]:
    return {
        "frame": item.frame,
        "text": item.text,
        "confidence": item.confidence,
        "box": None if item.box is None else list(item.box),
    }


def _residuals(attempt: Mapping[str, object]) -> list[Residual]:
    lines = attempt["residualLines"]
    if not isinstance(lines, list):
        raise ValueError("Attempt lacks residual lines")
    return [
        Residual(int(line["frame"]), str(line["text"]), float(line["confidence"]),
                 (float(box[0]), float(box[1]), float(box[2]), float(box[3])))
        for line in lines
        if isinstance(line, dict) and isinstance(box := line.get("box"), list)
    ]  # fmt: skip


def run(
    output: Path,
    tools: Collaborators,
    *,
    stride: int = 6,
    audit_stride: int = AUDIT_STRIDE,
    pins: Mapping[str, str] | None = None,
) -> dict[str, object]:
    """Attempt the sampled repair, escalate once to full OCR on a failed audit, then once more
    with the residual boxes when a collaborator takes them; write the receipt."""
    if stride <= FULL_STRIDE or audit_stride < 1:
        raise ValueError("Sampled stride must exceed 1 and the audit stride must be positive")
    output.mkdir(mode=0o700, parents=True, exist_ok=True)
    attempts = [_attempt(stride, output / "attempt-sampled", tools, audit_stride)]
    if not attempts[0]["passed"]:
        attempts.append(_attempt(FULL_STRIDE, output / "attempt-full", tools, audit_stride))
    left = _residuals(attempts[-1])
    if not attempts[-1]["passed"] and tools.residual is not None and left:
        # OCR found what the plan missed (a glyph neither detector nor colour key covered):
        # repair those boxes too, once, and audit the result like any attempt.
        tools.residual(left, audit_stride)
        attempts.append(_attempt(FULL_STRIDE, output / "attempt-residual", tools, audit_stride))
    accepted = next((item for item in reversed(attempts) if item["passed"]), None)
    receipt: dict[str, object] = {
        "version": 1,
        "kind": "sampled-ocr-repair-chunk",
        "attempts": attempts,
        "accepted": None if accepted is None else accepted["directory"],
        "escalated": len(attempts) > 1,
        "regionsFromSampling": True,
        "pins": dict(pins or {}),
        "qualityApproved": False,
        "deliveryComplete": False,
    }
    destination = output / "chunk-receipt.json"
    descriptor = os.open(destination, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)
    with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
        json.dump(receipt, stream, indent=2, sort_keys=True)
    if accepted is None:
        raise ValueError("Audit failed after escalation; receipt retained, nothing accepted")
    return receipt
