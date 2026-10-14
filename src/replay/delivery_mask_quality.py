"""Fail-closed source-mask accounting and bounded lossless support measurements.

This does not reconstruct lost repair support from MP4 or approve a delivery.
Execution observations must come from the compositor, not this projected plan.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import resource
import shutil
import subprocess
import sys
from bisect import bisect_right
from collections.abc import Callable, Sequence
from dataclasses import asdict, dataclass
from fractions import Fraction
from pathlib import Path
from zipfile import BadZipFile, ZipFile

import cv2
import numpy as np
from numpy.typing import NDArray

from replay.core import ReplayError
from replay.delivery_source_assembly import Chunk, load_chunks
from replay.recordings import record
from replay.subtitle_audit import audit_pixel_support
from replay.subtitle_clock import VideoFrameClock
from replay.subtitle_detector import LearnedTextDetector
from replay.subtitle_inpaint import Image, InpaintSettings, repair_glyphs, unchanged_outside_support
from replay.subtitle_plan import repair_spans
from replay.subtitle_retry import RepairEvidence, RepairMethod, decide_repair

MAX_JSON_BYTES = 256 * 1024 * 1024
MAX_FRAMES = 1_000_000
MAX_BATCH = 8
MAX_BATCH_PIXELS = 16_000_000
Support = NDArray[np.bool_]
Detector = Callable[[Sequence[Image]], tuple[Image, ...]]


@dataclass(frozen=True)
class LosslessFrame:
    """Caller must bind these pixels, supports and OCR to equal actual PTS.

    Instability is independently localized temporal evidence, not a frame diff
    guessed here. Its provenance/registration is a required upstream input.
    """

    source: Image
    candidate: Image
    original_glyph: Support
    repair_roi: Support
    unstable: Support
    readable: bool
    source_pts: Fraction
    candidate_pts: Fraction


def measure_batch(
    frames: Sequence[LosslessFrame], *, detect: Detector
) -> tuple[RepairEvidence, ...]:
    """Reuse one detector batch, original glyph denominator and strict GLYPH policy.

    Pure measurements only: these do not establish extraction/media provenance.
    LearnedTextDetector.detect is compatible; loading/downloading is not done here.
    """
    if not 1 <= len(frames) <= MAX_BATCH or sum(f.source.size for f in frames) > MAX_BATCH_PIXELS:
        raise ValueError("Lossless evidence batch budget exceeded")
    for frame in frames:
        if (
            frame.source.dtype != np.uint8
            or frame.source.ndim != 3
            or frame.source.shape[-1] != 3
            or frame.candidate.dtype != np.uint8
            or frame.candidate.shape != frame.source.shape
            or frame.source_pts != frame.candidate_pts
            or frame.source_pts < 0
            or type(frame.readable) is not bool
        ):
            raise ValueError("Lossless frame canvas or actual PTS mismatch")
        audit_pixel_support(frame.original_glyph, frame.unstable, allowed_roi=frame.repair_roi)
        if frame.original_glyph.shape != frame.source.shape[:2]:
            raise ValueError("Support canvas mismatch")
    detected = detect(tuple(frame.candidate for frame in frames))
    if len(detected) != len(frames):
        raise ValueError("Incomplete independent detection batch")
    results: list[RepairEvidence] = []
    for frame, output in zip(frames, detected, strict=True):
        if output.dtype != np.uint8 or output.shape != frame.original_glyph.shape:
            raise ValueError("Invalid independent detector support")
        audit = audit_pixel_support(frame.original_glyph, output > 0, allowed_roi=frame.repair_roi)
        area = int(np.count_nonzero(frame.repair_roi))
        unstable = int(np.count_nonzero(frame.unstable))
        valid = (
            audit.source_pixels > 0
            and area > 0
            and audit.source_outside_roi == 0
            and not np.any(frame.unstable & ~frame.repair_roi)
            and np.array_equal(frame.source[~frame.repair_roi], frame.candidate[~frame.repair_roi])
        )
        # Count ALL candidate evidence inside the ROI, not just original overlap.
        residual = int(np.count_nonzero((output > 0) & frame.repair_roi))
        results.append(
            RepairEvidence(
                valid,
                min(1.0, residual / audit.source_pixels) if audit.source_pixels else None,
                unstable / area if area and unstable <= area else None,
                frame.readable,
            )
        )
    return tuple(results)


def summarize_quality(evidence: Sequence[RepairEvidence], *, expected: int) -> dict[str, object]:
    """No cosmetic fallback and no partial/empty coverage acceptance."""
    complete = len(evidence) == expected and expected > 0
    return {
        "measuredFrames": len(evidence),
        "expectedFrames": expected,
        "technicallyInvalidFrames": sum(not item.technically_valid for item in evidence),
        "readableFrames": sum(item.readable_text is True for item in evidence),
        "maximumResidualFraction": max(
            (item.residual_fraction for item in evidence if item.residual_fraction is not None),
            default=None,
        ),
        "maximumUnstableFraction": max(
            (item.unstable_fraction for item in evidence if item.unstable_fraction is not None),
            default=None,
        ),
        "complete": complete,
        "accepted": complete
        and all(decide_repair(RepairMethod.GLYPH, item).accepted for item in evidence),
    }


def local_path(value: object) -> Path:
    if not isinstance(value, str) or not value or any(c in value for c in "\x00\n\r"):
        raise ReplayError("Invalid local path")
    path = Path(value)
    if (
        not path.is_absolute()
        or path != path.resolve()
        or not path.is_relative_to(Path.cwd().resolve())
    ):
        raise ReplayError(f"Unsafe path: {path}")
    return path


def digest(path: Path) -> str:
    with path.open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


def read_json(path: Path) -> dict[str, object]:
    if not 0 < path.stat().st_size <= MAX_JSON_BYTES:
        raise ReplayError(f"JSON byte budget exceeded: {path}")
    return record(json.loads(path.read_text(encoding="utf-8")))


def integer(value: object) -> int:
    if type(value) is not int or value < 0:
        raise ReplayError("Expected nonnegative integer")
    return value


def rows(value: object) -> list[object]:
    if not isinstance(value, list) or len(value) > MAX_FRAMES:
        raise ReplayError("Missing or excessive frame array")
    return list(value)


def bind(path: Path, pin: str, bindings: dict[str, str]) -> None:
    local_path(str(path))
    if digest(path) != pin:
        raise ReplayError(f"Changed input: {path}")
    bindings[str(path)] = pin


def reference(value: object, bindings: dict[str, str]) -> Path:
    raw = record(value)
    path = local_path(raw.get("path"))
    pin = raw.get("sha256")
    if not isinstance(pin, str):
        raise ReplayError("Missing input SHA256")
    bind(path, pin, bindings)
    return path


def _load(manifest: Path, pin: str, bindings: dict[str, str]) -> tuple[Chunk, ...]:
    # Check media scope BEFORE the existing loader opens/hashes indexed media.
    bind(manifest, pin, bindings)
    data = read_json(manifest)
    index = read_json(reference(data.get("index"), bindings))
    for row in rows(index.get("ranges")):
        local_path(record(row).get("media"))
    for raw in rows(data.get("chunks")):
        refs = record(raw)
        producer = read_json(reference(refs.get("producer"), bindings))
        local_path(producer.get("outputDirectory"))
        reference(refs.get("candidate"), bindings)
    return load_chunks(manifest, sha256=pin, bindings=bindings)


def _execution_masks(
    manifest: Path, chunks: tuple[Chunk, ...], bindings: dict[str, str]
) -> tuple[dict[int, int], ...]:
    result: list[dict[int, int]] = []
    for raw, chunk in zip(rows(read_json(manifest).get("chunks")), chunks, strict=True):
        refs = record(raw)
        producer = read_json(reference(refs.get("producer"), bindings))
        receipt_path = reference(refs.get("candidate"), bindings)
        receipt = read_json(receipt_path)
        spans = repair_spans(
            producer.get("masks"),
            frames=len(chunk.clock.pts),
            width=integer(producer.get("width")),
            height=integer(producer.get("height")),
            source_clock=chunk.clock,
        )
        expected = [f for span in spans for f in range(span.start_frame, span.end_frame)]
        active = [integer(f) for f in rows(receipt.get("activeSourceFrames"))]
        counts = [integer(f) for f in rows(receipt.get("detectedSupportPixels"))]
        rendered = record(receipt.get("render"))
        if (
            active != expected
            or len(active) != len(counts)
            or integer(receipt.get("emptyDetections")) != counts.count(0)
            or integer(rendered.get("frames")) != len(chunk.clock.pts)
        ):
            raise ReplayError(f"Mask execution differs from producer plan: {receipt_path}")
        result.append(dict(zip(active, counts, strict=True)))
    return tuple(result)


def _project(
    cuts: dict[str, object], chunks: tuple[Chunk, ...], rate: int
) -> list[dict[str, object]]:
    if not 1 <= rate <= 240:
        raise ReplayError("Invalid render rate")
    starts = [c.clock.pts[0] * c.clock.time_base for c in chunks]
    projected: list[dict[str, object]] = []
    previous = 0
    for raw in rows(cuts.get("cuts")):
        cut = record(raw)
        start, end = integer(cut.get("source_start_frame")), integer(cut.get("source_end_frame"))
        if (
            start < previous
            or end <= start
            or len(projected) + end - start > MAX_FRAMES
            or integer(cut.get("output_start_frame")) != len(projected)
        ):
            raise ReplayError("Protected cuts overlap, are noncontiguous, or exceed budget")
        previous = end
        for frame in range(start, end):
            time = Fraction(frame, rate)
            index = bisect_right(starts, time) - 1
            if index < 0:
                raise ReplayError("Selected clock precedes source")
            clock = chunks[index].clock
            source_frame = bisect_right(clock.pts, time / clock.time_base) - 1
            if time >= (clock.pts[-1] + clock.last_duration) * clock.time_base:
                raise ReplayError("Selected clock exceeds source")
            # Do not infer native compositor held-frame/rounding behavior.
            if clock.pts[source_frame] * clock.time_base != time:
                raise ReplayError(
                    "Non-exact native selection PTS; actual compositor frame mapping required"
                )
            projected.append(
                {
                    "outputFrame": len(projected),
                    "chunk": index,
                    "sourceFrame": source_frame,
                    "sourcePTS": clock.pts[source_frame],
                    "candidatePTS": clock.pts[source_frame] - clock.pts[0],
                    "timeBase": str(clock.time_base),
                }
            )
    if not projected:
        raise ReplayError("Empty protected selection")
    return projected


def account(request: Path, *, sha256: str) -> dict[str, object]:
    """Check pinned plans/receipts; missing actual execution/quality stays failed.

    Request v1 has renderConfig/assemblyManifest references and optional execution.
    All references are absolute checkout-local {path,sha256} pairs.
    """
    bindings: dict[str, str] = {}
    bind(request, sha256, bindings)
    raw = read_json(request)
    if integer(raw.get("version")) != 1:
        raise ReplayError("Unsupported mask accounting request")
    config_path = reference(raw.get("renderConfig"), bindings)
    manifest = reference(raw.get("assemblyManifest"), bindings)
    config = read_json(config_path)
    cuts_path = local_path(config.get("cuts"))
    reference(raw.get("protectedCuts"), bindings)
    if str(cuts_path) not in bindings:
        raise ReplayError("Render config cuts are not pinned protected cuts")
    chunks = _load(manifest, bindings[str(manifest)], bindings)
    for module in (
        "delivery_mask_quality.py",
        "delivery_source_assembly.py",
        "subtitle_plan.py",
        "subtitle_clock.py",
        "subtitle_audit.py",
        "subtitle_retry.py",
    ):
        implementation = Path(__file__).resolve().with_name(module)
        bind(implementation, digest(implementation), bindings)
    render_index = reference(raw.get("renderIndex"), bindings)
    if str(render_index) != config.get("sourceChunks"):
        raise ReplayError("Render config source index mismatch")
    manifest_index = record(read_json(manifest).get("index"))
    if str(render_index) != manifest_index.get("path"):
        assembled = read_json(reference(raw.get("assemblyReceipt"), bindings))
        upstream = record(assembled.get("inputs"))
        if any(
            upstream.get(name) != pin
            for name, pin in bindings.items()
            if name == str(manifest) or name in {str(c.media) for c in chunks}
        ):
            raise ReplayError("Assembly receipt does not bind source manifest/media")
        output = reference(assembled.get("output"), bindings)
        index_rows = rows(read_json(render_index).get("ranges"))
        if len(index_rows) != 1:
            raise ReplayError("Expected one assembled source range")
        row = record(index_rows[0])
        if (
            assembled.get("indexSHA256") != bindings[str(render_index)]
            or assembled.get("exactPresentationClockVerified") is not True
            or assembled.get("compressedPacketsIdentical") is not True
            or row.get("media") != str(output)
            or row.get("sha256") != bindings[str(output)]
            or row.get("frames") != sum(len(c.clock.pts) for c in chunks)
            or row.get("sourceFirstPTS") != chunks[0].clock.pts[0]
            or row.get("sourceEndPTS") != chunks[-1].clock.pts[-1] + chunks[-1].clock.last_duration
            or row.get("timeBase") != str(chunks[0].clock.time_base)
        ):
            raise ReplayError("Assembly receipt/index/clock mismatch")
    masks = _execution_masks(manifest, chunks, bindings)
    projected = _project(read_json(cuts_path), chunks, integer(config.get("rate")))
    selected = {(integer(row["chunk"]), integer(row["sourceFrame"])) for row in projected}
    masked = sorted((c, f) for c, f in selected if f in masks[c])
    empty = sum(masks[c][f] == 0 for c, f in masked)
    limitations = [
        "Missing input: lossless pre-encode candidate ROI pixels, original glyph support, "
        "repair ROI and independently registered temporal support for every selected masked frame; "
        "candidate MP4 and detectedSupportPixels cannot reconstruct these exactly.",
        "Missing input: independent readable-text observations "
        "at paired actual source/candidate PTS.",
        "Source quality_measured/quality_passed flags are not promoted by accounting.",
        "Source-mask candidate only; final video/compositor receipt chain remains unverified.",
    ]
    execution_matches = False
    if raw.get("execution") is not None:
        execution = read_json(reference(raw.get("execution"), bindings))
        if (
            execution.get("renderConfigSHA256") != bindings[str(config_path)]
            or execution.get("assemblyManifestSHA256") != bindings[str(manifest)]
            or execution.get("frames") != projected
        ):
            raise ReplayError("Actual execution observation clock/coverage/binding mismatch")
        execution_matches = True
        limitations.append(
            "Execution rows match; observation provenance is not independently verified."
        )
    else:
        limitations.append(
            "Missing input: actual native compositor source-frame observations (execution)."
        )
    for name, pin in tuple(bindings.items()):
        bind(Path(name), pin, bindings)
    return {
        "version": 1,
        "kind": "delivery-mask-quality",
        "inputs": [
            {"path": name, "sha256": pin, "bytes": Path(name).stat().st_size}
            for name, pin in sorted(bindings.items())
        ],
        "checks": {
            "recordedMaskExecutionMatchesPlan": True,
            "executionRowsMatchProtectedCuts": execution_matches,
            "actualCompositorProvenanceVerified": False,
            "selectedEmptyZero": empty == 0,
            "selectedMaskFramesPositive": len(masked) > 0,
            "qualityComplete": False,
            "losslessEvidenceVerified": False,
        },
        "measurements": {
            "selectedFrames": len(projected),
            "selectedNativeFrames": len(selected),
            "sourceFrames": sum(len(c.clock.pts) for c in chunks),
            "removedNativeFrames": sum(len(c.clock.pts) for c in chunks) - len(selected),
            "selectedMaskFrames": len(masked),
            "selectedEmpty": empty,
            "measuredFrames": 0,
            "missingMaskFrames": len(masked),
            "extraMaskFrames": 0,
            "technicallyInvalidFrames": None,
            "readableFrames": None,
            "maximumResidualFraction": None,
            "maximumUnstableFraction": None,
            "coverageDenominator": "all-selected-native-mask-frames-including-empty-detections",
            "producerPolicy": "unverified-per-producer-runtime",
            "finalChainVerified": False,
            "projectedFrames": projected,
            "selectedMaskedFrames": [{"chunk": c, "sourceFrame": f} for c, f in masked],
            "thresholds": {
                "residual": 0.005,
                "unstable": 0.02,
                "readableFrames": 0,
                "technicalInvalid": 0,
                "selectedEmpty": 0,
                "coverage": 1.0,
            },
        },
        "accepted": False,
        "limitations": limitations,
    }


def _clock(value: object) -> VideoFrameClock:
    raw = record(value)
    base = raw.get("timeBase")
    if not isinstance(base, str):
        raise ReplayError("Missing rational source time base")
    return VideoFrameClock(
        tuple(integer(p) for p in rows(raw.get("pts"))),
        Fraction(base),
        integer(raw.get("lastDuration")),
    )


def read_capture_pixels(path: Path) -> dict[str, Image]:
    """Read bounded writer arrays, without reconstructing pixels from encoded media."""
    with ZipFile(path) as archive:
        if (
            sorted(archive.namelist()) != ["candidate.npy", "glyph.npy", "repair.npy", "source.npy"]
            or sum(item.file_size for item in archive.infolist()) > 128 * 1024 * 1024
        ):
            raise ReplayError("Invalid or excessive capture archive")
    with np.load(path, allow_pickle=False) as archive:
        result = {
            key: np.asarray(archive[key]) for key in ("source", "candidate", "glyph", "repair")
        }
    if any(value.dtype != np.uint8 for value in result.values()):
        raise ReplayError("Capture must contain uint8 pixels")
    source = result["source"]
    if (
        source.ndim != 3
        or source.shape[-1] != 3
        or not 0 < source.size <= MAX_BATCH_PIXELS
        or result["candidate"].shape != source.shape
        or any(result[key].shape != source.shape[:2] for key in ("glyph", "repair"))
    ):
        raise ReplayError("Capture array canvases differ or exceed budget")
    return result


def observation_source(producer: dict[str, object], authorization: object) -> Path:
    """An explicit request pin permits only this read-only, resolved source file."""
    value = producer.get("source")
    if authorization is None:
        return local_path(value)
    allowed = record(authorization)
    if (
        not isinstance(value, str)
        or not value
        or any(c in value for c in "\x00\n\r")
        or allowed.get("path") != value
        or allowed.get("sha256") != producer.get("sourceSHA256")
    ):
        raise ReplayError("Source authorization differs from producer")
    path = Path(value)
    if not path.is_absolute() or path != path.resolve() or not path.is_file():
        raise ReplayError("Authorized source must be a resolved regular file")
    return path


def observe_capture(request: Path, *, sha256: str, directory: Path) -> dict[str, object]:
    """New bounded observations, not historical reconstruction or full acceptance.

    Request references producer/candidate/contextProducer. The latter supplies the
    unchanged original clock including immediate neighbors; no inferred FPS. One
    decode window and bounded detector/OCR batches are used. Never loop this CLI
    over a full recording: capture streaming requires a separate producer contract.
    """
    # Deferred to keep the pure batch metric API usable by the observation module.
    from replay import subtitle_quality_observe as observation

    if local_path(str(directory)).exists():
        raise ReplayError("Refusing to overwrite observation directory")
    bindings: dict[str, str] = {}
    bind(request, sha256, bindings)
    raw = read_json(request)
    if raw.get("version") != 1:
        raise ReplayError("Unsupported observation request")
    producer_path = reference(raw.get("producer"), bindings)
    candidate_path = reference(raw.get("candidate"), bindings)
    context_path = reference(raw.get("contextProducer"), bindings)
    producer, candidate, context = (
        read_json(p) for p in (producer_path, candidate_path, context_path)
    )
    repair_method = raw.get("repairMethod")
    if repair_method is not None and (
        not isinstance(repair_method, str)
        or repair_method not in (RepairMethod.GLYPH, RepairMethod.EXPANDED)
    ):
        raise ReplayError("Bounded comparison requires a non-softened glyph method")
    source = observation_source(producer, raw.get("authorizedSource"))
    native_name = producer.get("nativeExecutable")
    if not isinstance(native_name, str) or not Path(native_name).is_absolute():
        raise ReplayError("Invalid native executable path")
    native = local_path(str(Path(native_name).resolve()))
    ffmpeg_name = shutil.which("ffmpeg")
    if ffmpeg_name is None:
        raise ReplayError("Missing FFmpeg")
    ffmpeg = Path(ffmpeg_name).resolve()
    inputs = record(candidate.get("inputs"))
    if inputs.get(str(producer_path)) != bindings[str(producer_path)]:
        raise ReplayError("Candidate does not bind producer")
    for name, pin in inputs.items():
        if name != native_name and Path(name) not in (ffmpeg, source):
            local_path(name)
        if not isinstance(pin, str) or digest(Path(name)) != pin:
            raise ReplayError("Changed producer input")
        bindings[name] = pin
    for module in (Path(__file__).resolve(), Path(observation.__file__).resolve()):
        bind(module, digest(module), bindings)
    required = [producer_path, Path(str(producer.get("source"))), ffmpeg]
    required.extend(
        Path(__file__).with_name(f"subtitle_{name}.py").resolve()
        for name in (
            "candidate",
            "prepared",
            "detector",
            "stream",
            "repair",
            "inpaint",
            "video",
            "evidence",
            "clock",
            "plan",
            "retry",
        )
    )
    if any(str(path) not in inputs for path in required):
        raise ReplayError("Incomplete producer implementation/tool pins")
    if candidate.get("evidenceRuntime") != {
        "numpy": np.__version__,
        "opencv": cv2.__version__,
    }:
        raise ReplayError("Capture runtime differs from observer")
    if inputs.get(str(source)) != producer.get("sourceSHA256"):
        raise ReplayError("Producer source fingerprint mismatch")
    if context.get("source") != str(source) or context.get("sourceSHA256") != inputs.get(
        str(source)
    ):
        raise ReplayError("Context source differs from producer")
    video = local_path(str(candidate_path.parent / "candidate.mp4"))
    pin = candidate.get("videoSHA256")
    if not isinstance(pin, str):
        raise ReplayError("Missing encoded candidate binding")
    bind(video, pin, bindings)
    capture = record(candidate.get("losslessEvidence"))
    entries = tuple(record(row) for row in rows(capture.get("frames")))
    if (
        capture.get("capturePoint") != "after-encoder-pipe-write-before-lossy-encode"
        or capture.get("pixelFormat") != "bgr24"
        or not 1 <= len(entries) <= MAX_BATCH
    ):
        raise ReplayError("Unsupported writer capture")
    clock, original_clock = _clock(producer.get("clock")), _clock(context.get("clock"))
    indices = tuple(integer(row.get("frame")) for row in entries)
    if indices != tuple(range(indices[0], indices[0] + len(indices))) or indices[-1] >= len(
        clock.pts
    ):
        raise ReplayError("Observation capture must be consecutive")
    if producer.get("evidenceFrames") != list(indices) or record(candidate.get("render")).get(
        "frames"
    ) != len(clock.pts):
        raise ReplayError("Capture selection or encoder completion differs from producer")
    positions = tuple(original_clock.pts.index(clock.pts[index]) for index in indices)
    if (
        positions != tuple(range(positions[0], positions[0] + len(positions)))
        or clock.time_base != original_clock.time_base
    ):
        raise ReplayError("Capture/context clocks differ")
    start, end = max(0, positions[0] - 1), min(len(original_clock.pts), positions[-1] + 2)
    extent = (
        original_clock.last_duration
        if end == len(original_clock.pts)
        else original_clock.pts[end] - original_clock.pts[end - 1]
    )
    width, height = integer(context.get("width")), integer(context.get("height"))
    if producer.get("width") != width or producer.get("height") != height:
        raise ReplayError("Producer/context canvases differ")
    decoded = observation.decode_context(
        source,
        clock=VideoFrameClock(original_clock.pts[start:end], original_clock.time_base, extent),
        width=width,
        height=height,
        ffmpeg=ffmpeg,
    )
    regions = tuple(record(row.get("region")) for row in entries)
    left = max(0, min(integer(region.get("x")) for region in regions) - 96)
    top = max(0, min(integer(region.get("y")) for region in regions) - 96)
    right = min(width, max(integer(r.get("x")) + integer(r.get("width")) for r in regions) + 96)
    bottom = min(height, max(integer(r.get("y")) + integer(r.get("height")) for r in regions) + 96)
    cropped = tuple(image[top:bottom, left:right].copy() for image in decoded)
    selection = record(producer.get("learnedTextDetector"))
    model = reference({"path": selection.get("model"), "sha256": selection.get("sha256")}, bindings)
    if selection.get("backend") != "ppocrv3-db" or inputs.get(str(model)) != bindings[str(model)]:
        raise ReplayError("Unbound residual detector")
    detector = LearnedTextDetector(model, sha256=bindings[str(model)])
    excluded = detector.detect(cropped[:8]) + (
        detector.detect(cropped[8:]) if len(cropped) > 8 else ()
    )
    frames: list[observation.ObservationFrame] = []
    actual_methods: list[str] = []
    repaired_pixels: list[dict[str, Image]] = []
    for entry, index, position, region in zip(entries, indices, positions, regions, strict=True):
        if (
            entry.get("sourcePTS") != clock.pts[index]
            or entry.get("candidatePTS") != clock.pts[index] - clock.pts[0]
            or entry.get("timeBase") != str(clock.time_base)
        ):
            raise ReplayError("Capture actual PTS mismatch")
        pixels = read_capture_pixels(reference(entry, bindings))
        x, y, w, h = (integer(region.get(key)) for key in ("x", "y", "width", "height"))
        if not left <= x < x + w <= right or not top <= y < y + h <= bottom:
            raise ReplayError("Capture region leaves context")
        before = cropped[position - start]
        area = (slice(y - top, y - top + h), slice(x - left, x - left + w))
        if (
            pixels["source"].shape != (h, w, 3)
            or pixels["candidate"].shape != (h, w, 3)
            or pixels["glyph"].shape != (h, w)
            or pixels["repair"].shape != (h, w)
            or not np.array_equal(before[area], pixels["source"])
        ):
            raise ReplayError("Producer pixels differ from exact-PTS source context")
        glyph, repair = np.zeros(before.shape[:2], np.bool_), np.zeros(before.shape[:2], np.bool_)
        glyph[area], repair[area] = pixels["glyph"] > 0, pixels["repair"] > 0
        if entry.get("glyphPixels") != int(np.count_nonzero(glyph)) or entry.get(
            "repairPixels"
        ) != int(np.count_nonzero(repair)):
            raise ReplayError("Capture support counts differ from writer receipt")
        methods = rows(entry.get("methods"))
        if not methods or not all(method in tuple(RepairMethod) for method in methods):
            raise ReplayError("Missing or unknown writer repair method")
        actual_methods.append(str(methods[-1]) if repair_method is None else repair_method)
        if repair_method is not None:
            repaired, support = repair_glyphs(
                pixels["source"],
                text_mask=pixels["glyph"],
                settings=InpaintSettings(expansion=2 if repair_method == RepairMethod.GLYPH else 4),
            )
            if not unchanged_outside_support(pixels["source"], repaired, support):
                raise ReplayError("Bounded repair changed pixels outside support")
            pixels = {**pixels, "candidate": repaired, "repair": support}
            repair[area] = support > 0
        repaired_pixels.append(pixels)
        after = before.copy()
        after[area] = pixels["candidate"]

        def neighbor(offset: int, position: int) -> observation.OriginalContext | None:
            slot = position - start + offset
            if not 0 <= slot < len(cropped):
                return None
            return observation.OriginalContext(
                cropped[slot],
                excluded[slot] > 0,
                original_clock.pts[position + offset] * original_clock.time_base,
            )

        current = observation.OriginalContext(
            before,
            (excluded[position - start] > 0) | repair | glyph,
            clock.pts[index] * clock.time_base,
        )
        frames.append(
            observation.ObservationFrame(
                current, after, glyph, repair, neighbor(-1, position), neighbor(1, position)
            )
        )
    if inputs.get(native_name) != producer.get("nativeSHA256") or digest(native) != inputs.get(
        native_name
    ):
        raise ReplayError("Unbound OCR executable")
    bind(native, digest(native), bindings)
    local_path(str(directory)).mkdir(mode=0o700)
    results = observation.observe_batch(
        tuple(frames), detect=detector.detect, executable=native, work_directory=directory
    )
    supports: list[dict[str, object]] = []
    stored_bytes = 0
    for entry, result, position, pixels in zip(
        entries, results, positions, repaired_pixels, strict=True
    ):
        arrays: dict[str, NDArray[np.bool_] | NDArray[np.float32]] = {
            "unstable": result.temporal.unstable,
            "unknown": result.temporal.unknown,
        }
        registrations: list[dict[str, object]] = []
        for label, correspondence in (
            ("previous", result.temporal.previous),
            ("following", result.temporal.following),
        ):
            if correspondence is not None:
                arrays[f"{label}_known"] = correspondence.known
                arrays[f"{label}_xy"] = correspondence.donor_xy
                registrations.append(
                    {
                        "neighbor": label,
                        "sourcePTS": original_clock.pts[
                            position + (-1 if label == "previous" else 1)
                        ],
                        "timeBase": str(original_clock.time_base),
                        "trackedFeatures": correspondence.tracked_features,
                        "inlierFraction": correspondence.inlier_fraction,
                        "displacement": correspondence.displacement,
                    }
                )
        path = directory / f"support-{integer(entry['frame']):08d}.npz"
        with path.open("xb") as handle:
            os.fchmod(handle.fileno(), 0o600)
            np.savez(
                handle,
                candidate=pixels["candidate"],
                glyph=pixels["glyph"],
                repair=pixels["repair"],
                unstable=arrays["unstable"],
                unknown=arrays["unknown"],
                previous_known=arrays.get("previous_known", np.empty((0, 0), np.bool_)),
                previous_xy=arrays.get("previous_xy", np.empty((0, 0, 2), np.float32)),
                following_known=arrays.get("following_known", np.empty((0, 0), np.bool_)),
                following_xy=arrays.get("following_xy", np.empty((0, 0, 2), np.float32)),
            )
        stored_bytes += path.stat().st_size
        supports.append({"path": str(path), "sha256": digest(path), "registrations": registrations})
    for name, expected in bindings.items():
        if digest(Path(name)) != expected:
            raise ReplayError("Observation input changed during execution")
    return {
        "version": 1,
        "kind": "source-mask-bounded-observation",
        "repairMethod": repair_method,
        "actualMethods": actual_methods,
        "comparisonScope": "lossless-ROI-only-not-encoded-producer"
        if repair_method
        else "writer-capture",
        "qualityMethod": str(RepairMethod.GLYPH),
        "accepted": False,
        "inputs": bindings,
        "sampleQuality": summarize_quality(
            tuple(result.evidence for result in results), expected=len(entries)
        ),
        "unknownPixels": [int(np.count_nonzero(result.temporal.unknown)) for result in results],
        "frameEvidence": [asdict(result.evidence) for result in results],
        "readableLines": [result.readable_lines for result in results],
        "ocrObservations": [asdict(result.ocr) for result in results],
        "ocrClock": "image-local-zero; sourcePTS supplies media correspondence",
        "contextOrigin": [left, top],
        "sourcePTS": [row["sourcePTS"] for row in entries],
        "fullCoverageVerified": False,
        "actualCompositorProvenanceVerified": False,
        "supports": supports,
        "storedSupportBytes": stored_bytes,
        "decodedContextBytes": sum(image.nbytes for image in decoded),
        "croppedContextBytes": sum(image.nbytes for image in cropped),
        "processLifetimeMaxRSS": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
        "childLifetimeMaxRSS": resource.getrusage(resource.RUSAGE_CHILDREN).ru_maxrss,
        "maxRSSUnit": "bytes" if sys.platform == "darwin" else "KiB",
        "maxRSSScope": "process-lifetime-high-water-not-simultaneous-job-peak",
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("request", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--sha256", required=True)
    parser.add_argument(
        "--observe",
        action="store_true",
        help="Measure one bounded writer capture; not full delivery acceptance",
    )
    args = parser.parse_args(argv)
    try:
        output = local_path(str(args.output))
        if output.exists():
            raise ReplayError("Refusing to overwrite mask quality evidence")
        try:
            result = (
                observe_capture(
                    args.request, sha256=args.sha256, directory=output.with_suffix(".observations")
                )
                if args.observe
                else account(args.request, sha256=args.sha256)
            )
        except (ReplayError, ValueError, OSError, BadZipFile, subprocess.SubprocessError) as error:
            result = {
                "version": 1,
                "kind": "delivery-mask-quality",
                "inputs": [],
                "checks": {"accounting": False},
                "measurements": {},
                "accepted": False,
                "limitations": [str(error)],
            }
        with output.open("x", encoding="utf-8") as handle:
            os.fchmod(handle.fileno(), 0o600)
            json.dump(result, handle, indent=2, allow_nan=False)
            handle.write("\n")
    except (ReplayError, ValueError, OSError) as error:
        print(f"MASK_QUALITY_ERROR: {error}", file=sys.stderr)
        return 1
    return 0 if result["accepted"] is True else 1


if __name__ == "__main__":
    raise SystemExit(main())
