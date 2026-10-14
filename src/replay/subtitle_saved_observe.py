"""Observe bounded pinned PNGs; no media decode, OCR, inference or repair.

Config version 1 supplies clock {pts, timeBase, lastDuration}, receipt references,
and 1..8 frames with sourcePTS, source/candidate references, glyph/repair lists
(unioned supports), and optional named comparisons {name, image}. All references
are absolute checkout-local {path, sha256}. Clocks/provenance are caller-supplied
historical evidence, not reverified source-media observations or quality approval.
"""

import argparse
import json
import os
import struct
import sys
from dataclasses import replace
from fractions import Fraction
from pathlib import Path
from typing import Literal

import cv2
import numpy as np

from replay.core import ReplayError
from replay.delivery_mask_quality import (
    bind,
    digest,
    integer,
    local_path,
    read_json,
    reference,
    rows,
)
from replay.recordings import record
from replay.subtitle_clock import VideoFrameClock
from replay.subtitle_inpaint import Image
from replay.subtitle_quality_observe import (
    MAX_PIXELS,
    ObservationFrame,
    OriginalContext,
    Support,
    candidate_pair_change,
    temporal_observation,
)

MAX_FRAMES = 8
MAX_PNG_BYTES = 16 * 1024 * 1024
MAX_REFERENCES = 16
MODELS: tuple[Literal["translation", "local-affine"], ...] = ("translation", "local-affine")


def _bounded(value: object, *, maximum: int = MAX_REFERENCES) -> list[object]:
    items = rows(value)
    if not 1 <= len(items) <= maximum:
        raise ReplayError("Empty or excessive saved evidence list")
    return items


def _image(value: object, bindings: dict[str, str], *, mask: bool = False) -> Image:
    path = reference(value, bindings)
    if not 33 <= path.stat().st_size <= MAX_PNG_BYTES:
        raise ReplayError("PNG byte budget exceeded")
    data = path.read_bytes()
    if data[:8] != b"\x89PNG\r\n\x1a\n" or data[12:16] != b"IHDR":
        raise ReplayError("Expected PNG")
    width, height = struct.unpack(">II", data[16:24])
    if min(width, height) < 8 or width * height > MAX_PIXELS:
        raise ReplayError("PNG canvas budget exceeded")
    image = cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_UNCHANGED)
    if (
        image is None
        or image.dtype != np.uint8
        or image.shape != ((height, width) if mask else (height, width, 3))
    ):
        raise ReplayError("Expected unchanged uint8 grayscale support or BGR image")
    if mask and np.any((image != 0) & (image != 255)):
        raise ReplayError("Support must be binary")
    return np.asarray(image, dtype=np.uint8)


def _support(value: object, bindings: dict[str, str], shape: tuple[int, ...]) -> Support:
    result = np.zeros(shape, dtype=np.bool_)
    for ref in _bounded(value):
        image = _image(ref, bindings, mask=True)
        if image.shape != shape:
            raise ReplayError("Support canvas mismatch")
        result |= image > 0
    return result


def _frame(
    raw: dict[str, object], clock: VideoFrameClock, bindings: dict[str, str]
) -> ObservationFrame:
    source = _image(raw.get("source"), bindings)
    candidate = _image(raw.get("candidate"), bindings)
    glyph = _support(raw.get("glyph"), bindings, source.shape[:2])
    repair = _support(raw.get("repair"), bindings, source.shape[:2])
    if candidate.shape != source.shape or np.any(glyph & ~repair):
        raise ReplayError("Candidate canvas or glyph support mismatch")
    return ObservationFrame(
        OriginalContext(source, repair, integer(raw.get("sourcePTS")) * clock.time_base),
        candidate,
        glyph,
        repair,
        None,
        None,
    )


def _metrics(image: Image, frame: ObservationFrame) -> dict[str, object]:
    if image.shape != frame.original.image.shape:
        raise ReplayError("Comparison canvas mismatch")
    pixels = image.astype(np.float32)
    support = frame.repair
    gx = np.max(np.abs(pixels[:, 1:] - pixels[:, :-1]), axis=2)
    gy = np.max(np.abs(pixels[1:] - pixels[:-1]), axis=2)
    boundary = np.concatenate(
        (gx[support[:, 1:] != support[:, :-1]], gy[support[1:] != support[:-1]])
    )
    return {
        "outsideChangedPixels": int(
            np.count_nonzero(np.any(image != frame.original.image, axis=2) & ~support)
        ),
        "boundaryAdjacentCodeDifference": {
            "count": int(boundary.size),
            "mean": float(boundary.mean()) if boundary.size else None,
            "p95": float(np.quantile(boundary, 0.95)) if boundary.size else None,
            "max": float(boundary.max()) if boundary.size else None,
        },
    }


def _state(
    frame: ObservationFrame,
    following: ObservationFrame | None,
    *,
    clock: VideoFrameClock,
    model: Literal["translation", "local-affine"],
) -> dict[str, object]:
    pair = candidate_pair_change(frame, following, clock=clock, registration_model=model)
    temporal = temporal_observation(frame, registration_model=model)
    reg = pair.registration
    return {
        "model": model,
        "repairPixels": pair.repair_pixels,
        "unknownPixels": int(np.count_nonzero(pair.unknown)),
        "changedPixels": int(np.count_nonzero(pair.changed)),
        "changedFraction": pair.changed_fraction,
        "registration": None
        if reg is None
        else {
            "tracked": reg.tracked_features,
            "globalTranslationInlierFraction": reg.inlier_fraction,
            "displacement": reg.displacement,
            "localModelsTested": reg.local_models_tested,
            "localModelsAccepted": reg.local_models_accepted,
            "knownSourcePixels": int(np.count_nonzero(reg.known)),
            "contextGeometryPixels": int(np.count_nonzero(reg.context_geometry)),
        },
        "twoSidedOriginalDonors": {
            "evaluated": temporal.evaluated,
            "unknownPixels": int(np.count_nonzero(temporal.unknown)),
            "unstablePixels": int(np.count_nonzero(temporal.unstable)),
        },
    }


def observe_saved(config: Path, *, sha256: str) -> dict[str, object]:
    """Measure caller-bound saved evidence and rehash inputs after observation."""
    bindings: dict[str, str] = {}
    bind(config, sha256, bindings)
    raw = read_json(config)
    if raw.get("version") != 1:
        raise ReplayError("Unsupported saved observation version")
    for receipt in _bounded(raw.get("receipts")):
        reference(receipt, bindings)
    source_clock = record(raw.get("clock"))
    base = source_clock.get("timeBase")
    if not isinstance(base, str):
        raise ReplayError("Expected rational time base")
    clock = VideoFrameClock(
        tuple(integer(p) for p in _bounded(source_clock.get("pts"), maximum=MAX_FRAMES)),
        Fraction(base),
        integer(source_clock.get("lastDuration")),
    )
    entries = tuple(record(row) for row in _bounded(raw.get("frames"), maximum=MAX_FRAMES))
    if tuple(integer(row.get("sourcePTS")) for row in entries) != clock.pts:
        raise ReplayError("Saved frame PTS differ from supplied clock")
    # Bind the shared implementation before any measurement; never historical scripts.
    for name in (
        "subtitle_saved_observe.py",
        "subtitle_quality_observe.py",
        "subtitle_registration.py",
        "subtitle_clock.py",
        "delivery_mask_quality.py",
    ):
        path = Path(__file__).resolve().with_name(name)
        bind(path, digest(path), bindings)
    frames = tuple(_frame(row, clock, bindings) for row in entries)
    results: list[dict[str, object]] = []
    for index, (frame, entry) in enumerate(zip(frames, entries, strict=True)):
        following = frames[index + 1] if index + 1 < len(frames) else None
        frame = replace(
            frame,
            previous=frames[index - 1].original if index else None,
            following=following.original if following else None,
        )
        comparisons = rows(entry.get("comparisons", []))
        if len(comparisons) > MAX_REFERENCES:
            raise ReplayError("Excessive comparison list")
        metrics = [
            {"name": "source", **_metrics(frame.original.image, frame)},
            {"name": "candidate", **_metrics(frame.candidate, frame)},
        ]
        for comparison in comparisons:
            item = record(comparison)
            label = item.get("name")
            if not isinstance(label, str) or not 1 <= len(label) <= 128:
                raise ReplayError("Missing bounded comparison name")
            metrics.append({"name": label, **_metrics(_image(item.get("image"), bindings), frame)})
        results.append(
            {
                "sourcePTS": clock.pts[index],
                "metrics": metrics,
                "states": [_state(frame, following, clock=clock, model=model) for model in MODELS],
            }
        )
    after = {path: digest(Path(path)) for path in bindings}
    if after != bindings:
        raise ReplayError("Evidence changed during observation")
    return {
        "version": 1,
        "kind": "saved-png-observation",
        "inputsBefore": bindings,
        "inputsAfter": after,
        "clock": source_clock,
        "frames": results,
        "sourceMediaReverified": False,
        "qualityApproved": False,
        "newMediaDecode": 0,
        "newOCR": 0,
        "newDB": 0,
        "newRepair": 0,
        "limitations": (
            "Caller-supplied historical clocks/receipts; hidden background unknown. "
            "Context-conditioned candidate variation is not artifact classification. "
            "Boundary differences are not structural truth. "
            "Missing support is unknown, never stable."
        ),
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("config", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--sha256", required=True)
    args = parser.parse_args(argv)
    try:
        output = local_path(str(args.output.absolute()))
        if output.exists():
            raise ReplayError("Refusing to overwrite observation")
        report = observe_saved(args.config.absolute(), sha256=args.sha256)
        with output.open("x", encoding="utf-8") as handle:
            os.fchmod(handle.fileno(), 0o600)
            json.dump(report, handle, indent=2, allow_nan=False)
            handle.write("\n")
    except (ReplayError, ValueError, OSError, ZeroDivisionError) as error:
        print(f"SAVED_OBSERVATION_ERROR: {error}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
