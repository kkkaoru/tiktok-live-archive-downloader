"""Render a bounded, explicit-detector source-caption candidate; never approve it."""

import argparse
import hashlib
import json
import os
import shutil
from dataclasses import asdict
from fractions import Fraction
from pathlib import Path

import cv2
import numpy as np

from . import disk_budget, subtitle_detector, subtitle_eligibility, subtitle_prepared
from .delivery_source_assembly import candidate_media_path
from .local_asr import fingerprint
from .recordings import record
from .subtitle_clock import VideoFrameClock
from .subtitle_evidence import RepairCapture
from .subtitle_inpaint import Image
from .subtitle_plan import repair_spans
from .subtitle_video import VideoRepairJob, render_repaired_video
from .subtitle_vision import vision_text_masks

MAX_CONFIG_BYTES = 8 * 1024 * 1024


def _integer(value: object) -> int:
    if type(value) is not int:
        raise ValueError("Expected an integer")
    return value


def _path(value: object) -> Path:
    if not isinstance(value, str) or not Path(value).is_absolute():
        raise ValueError("Expected an absolute local path")
    return Path(value)


def _reviewed_glyph(image: Image, item: dict[str, object], inputs: dict[str, str]) -> Image:
    if hashlib.sha256(image.tobytes()).hexdigest() != item.get("sourceROISHA256"):
        raise ValueError("Reviewed glyph source pixels differ from canonical decode")
    path = subtitle_eligibility._reference(item.get("mask"), inputs)
    mask = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if mask is None or mask.shape != image.shape[:2] or not np.all((mask == 0) | (mask == 255)):
        raise ValueError("Invalid reviewed glyph support pixels")
    return np.asarray(mask, dtype=np.uint8)


def render_candidate(config: Path) -> Path:
    """Keep source/clock/detector bindings, frame-local measured support and failures.

    The output is a video-only intermediate. Mask counts are observations, not
    residual-text/readability/temporal measurements. No completion gate is passed
    here; independently verify output clocks, decode and removal quality before
    incorporating it into an edited delivery. Existing files are never replaced.
    """
    if config.is_symlink() or not config.is_file() or config.stat().st_size > MAX_CONFIG_BYTES:
        raise ValueError("Expected bounded regular candidate configuration")
    data = record(json.loads(config.read_text(encoding="utf-8")))
    source, output = _path(data.get("source")), _path(data.get("outputDirectory"))
    native = _path(data.get("nativeExecutable"))
    work = _path(data.get("workDirectory"))
    prepared = data.get("prepareVisionContext", False)
    if not isinstance(prepared, bool) or (prepared and data.get("visionTextRegions") is not True):
        raise ValueError("Prepared observation requires explicit Vision text-region support")
    # Detector changes require explicit configuration. No discovery, installation,
    # permission change, automatic fallback or model download occurs.
    if (
        output.is_symlink()
        or work.is_symlink()
        or not output.resolve().is_relative_to(Path.cwd().resolve())
        or not work.resolve().is_relative_to(Path.cwd().resolve())
        or not os.access(native, os.X_OK)
        or not isinstance(data.get("visionTextRegions"), bool)
    ):
        raise ValueError("Invalid local candidate storage or detector selection")
    video = candidate_media_path(data)
    if video.exists() or video.is_symlink():
        raise FileExistsError("Candidate media already exists; preserve the prior output")
    clock_data = record(data.get("clock"))
    pts = clock_data.get("pts")
    time_base = clock_data.get("timeBase")
    if not isinstance(pts, list) or not isinstance(time_base, str):
        raise ValueError("Missing decoded presentation clock")
    clock = VideoFrameClock(
        tuple(_integer(value) for value in pts),
        Fraction(time_base),
        _integer(clock_data.get("lastDuration")),
    )
    width, height = _integer(data.get("width")), _integer(data.get("height"))
    if "mediaOutput" in data:
        volume = disk_budget.measure(video.parent)
        reserve = len(clock.pts) * width * height * 3
        if volume.free - volume.floor < reserve:
            raise ValueError("Insufficient media-volume capacity for the complete repair candidate")
    spans = repair_spans(
        data.get("masks"), frames=len(clock.pts), width=width, height=height, source_clock=clock
    )
    inputs: dict[str, str] = {}
    source_pin = data.get("sourceSHA256")
    if not isinstance(source_pin, str):
        raise ValueError("Missing source fingerprint")
    subtitle_eligibility.validate_eligibility(
        data.get("sourceEligibility"),
        source_sha256=source_pin,
        clock=clock,
        width=width,
        height=height,
        spans=spans,
        bindings=inputs,
    )
    for path in (
        source,
        native,
        config,
        Path(__file__),
        Path(subtitle_prepared.__file__),
        Path(subtitle_eligibility.__file__),
        Path(__file__).with_name("delivery_source_assembly.py"),
        Path(disk_budget.__file__),
    ):
        pin = fingerprint(path)
        if str(path) in inputs and inputs[str(path)] != pin:
            raise ValueError("Eligibility input changed before rendering")
        inputs[str(path)] = pin
    if inputs[str(source)] != data.get("sourceSHA256") or inputs[str(native)] != data.get(
        "nativeSHA256"
    ):
        raise ValueError("Source or native executable fingerprint changed")
    for key in ("probeParent", "probeImplementation"):
        if key in data:
            ref = record(data[key])
            path = _path(ref.get("path"))
            if path != path.resolve() or not path.is_relative_to(Path.cwd().resolve()):
                raise ValueError("Probe provenance must be checkout-local")
            inputs[str(path)] = fingerprint(path)
            if inputs[str(path)] != ref.get("sha256"):
                raise ValueError("Probe provenance changed")
    context_frames = _integer(data.get("contextFrames", 0))
    registered: list[dict[str, object]] = []
    donors_verified = False
    if context_frames:
        ref = record(data.get("registeredGlyphSupport"))
        support_path = subtitle_eligibility._reference(ref, inputs)
        support_data = record(json.loads(support_path.read_text(encoding="utf-8")))
        subtitle_eligibility._reference(support_data.get("reviewEvidence"), inputs)
        items = support_data.get("frames")
        if (
            support_data.get("sourceSHA256") != source_pin
            or support_data.get("timeBase") != str(clock.time_base)
            or not isinstance(support_data.get("glyphsComplete"), bool)
            or not isinstance(items, list)
            or len(items) != sum(span.end_frame - span.start_frame for span in spans)
        ):
            raise ValueError("Temporal candidates require source-bound glyph review")
        donors_verified = support_data["glyphsComplete"] is True
        registered = [record(item) for item in items]
        active_pts = tuple(
            clock.pts[frame] for span in spans for frame in range(span.start_frame, span.end_frame)
        )
        if tuple(item.get("sourcePTS") for item in registered) != active_pts:
            raise ValueError("Reviewed glyph clocks differ from source")
        for item in registered:
            subtitle_eligibility._reference(item.get("mask"), inputs)
    detector: subtitle_detector.LearnedTextDetector | None = None
    selection = data.get("learnedTextDetector")
    if context_frames and (selection is not None or prepared):
        raise ValueError("Reviewed temporal masks cannot be mixed with detector preparation")
    if selection is not None:
        selected = record(selection)
        digest = selected.get("sha256")
        if (
            prepared
            or data["visionTextRegions"] is not True
            or selected.get("backend") != "ppocrv3-db"
            or not isinstance(digest, str)
        ):
            raise ValueError("Invalid explicit learned detector selection")
        model = _path(selected.get("model"))
        detector = subtitle_detector.LearnedTextDetector(model, sha256=digest)
        inputs[str(model)] = digest
        inputs[str(Path(subtitle_detector.__file__))] = fingerprint(
            Path(subtitle_detector.__file__)
        )
    capture: RepairCapture | None = None
    evidence_frames = data.get("evidenceFrames")
    if evidence_frames is not None:
        ffmpeg = shutil.which("ffmpeg")
        if ffmpeg is None:
            raise ValueError("Missing evidence-bound decoder/encoder")
        implementation = Path(__file__).resolve().parent
        for name in (
            "subtitle_evidence",
            "subtitle_stream",
            "subtitle_video",
            "subtitle_repair",
            "subtitle_inpaint",
            "subtitle_retry",
            "subtitle_clock",
            "subtitle_plan",
            "subtitle_quality_observe",
            "subtitle_registration",
        ):
            path = implementation / f"{name}.py"
            inputs[str(path)] = fingerprint(path)
        binary = Path(ffmpeg).resolve()
        inputs[str(binary)] = fingerprint(binary)
    output.mkdir(mode=0o700)
    if evidence_frames is not None:
        capture = RepairCapture(output / "lossless-evidence", evidence_frames, clock)
    work.mkdir(mode=0o700, parents=True, exist_ok=True)
    counts: list[int] = []
    observed_lines: list[int] = []
    clipped_lines: list[int] = []

    def detect(images: tuple[Image, ...]) -> tuple[Image, ...]:
        if registered:
            selected = registered[len(counts) : len(counts) + len(images)]
            masks = tuple(
                _reviewed_glyph(image, item, inputs)
                for image, item in zip(images, selected, strict=True)
            )
        elif detector is not None:
            masks = detector.detect(images)
        elif prepared:
            observed = subtitle_prepared.prepared_text_masks(
                images, executable=native, work_directory=work
            )
            masks = observed.masks
            observed_lines.extend(observed.observed_lines)
            clipped_lines.extend(observed.clipped_lines)
        else:
            masks = vision_text_masks(
                images,
                executable=native,
                work_directory=work,
                text_regions=data["visionTextRegions"] is True,
            )
        counts.extend(int(np.count_nonzero(mask)) for mask in masks)
        return masks

    hardware = data.get("hardwareEncode", True)
    if not isinstance(hardware, bool):
        raise ValueError("Hardware encoder selection must be boolean")
    job = VideoRepairJob(
        source,
        video,
        width,
        height,
        len(clock.pts),
        spans,
        source_clock=clock,
        hardware_encode=hardware,
        hardware_hevc=hardware,
        context_frames=context_frames,
        context_donors_verified=donors_verified,
        decode_log=(output / "candidate.mp4.decode.log" if "mediaOutput" in data else None),
        ramp_start_frame=_integer(data.get("rampStartFrame", 0)),
        ramp_total_frames=(
            _integer(data["rampTotalFrames"]) if "rampTotalFrames" in data else None
        ),
    )
    receipt = render_repaired_video(job, detect_batch=detect, evidence=capture)
    if len(counts) != sum(span.end_frame - span.start_frame for span in spans):
        raise ValueError("Incomplete frame-local detection accounting")
    if any(fingerprint(Path(name)) != digest for name, digest in inputs.items()):
        raise ValueError("Candidate inputs changed during rendering")
    backend = "ppocrv3-db" if detector is not None else "vision"
    if registered:
        backend = "reviewed-glyph-support"
    result = {
        "inputs": inputs,
        "videoSHA256": fingerprint(video),
        "render": asdict(receipt),
        "visionTextRegions": data["visionTextRegions"] if detector is None else None,
        "prepareVisionContext": prepared,
        "learnedTextDetector": selection,
        "detectionBackend": backend,
        "registeredGlyphSupport": data.get("registeredGlyphSupport"),
        "learnedGridSizes": detector.grid_sizes if detector is not None else [],
        "learnedSquareContextRegions": detector.square_context_regions
        if detector is not None
        else [],
        "learnedObservedRegions": detector.observed_regions if detector is not None else [],
        "learnedCanvasIntersections": detector.intersected_regions if detector is not None else [],
        "detectedSupportPixels": counts,
        "observedContextLines": observed_lines,
        "clippedContextLines": clipped_lines,
        "contextClippingIsNotGlyphOrQualityApproval": True,
        "activeSourceFrames": [
            frame for span in spans for frame in range(span.start_frame, span.end_frame)
        ],
        "emptyDetections": counts.count(0),
        "emptyDetectionIsNotAbsence": True,
        "sourceEligibility": data["sourceEligibility"],
        "eligibilityIsNotRepairQuality": True,
        "sourceStart": str(clock.pts[0] * clock.time_base),
        "contextFrames": job.context_frames,
        "donorGlyphSupportVerified": donors_verified,
        "inactiveFramesAreUnreviewedNotApproved": True,
        "unsupportedPixelsRetained": bool(job.context_frames),
        "geometrySupportIsNotRestorationApproval": True,
        "qualityApproval": False,
        "fullDecodeVerified": False,
        "deliveryComplete": False,
    }
    if capture is not None:
        result["losslessEvidence"] = capture.manifest()
        result["evidenceRuntime"] = {"numpy": np.__version__, "opencv": cv2.__version__}
    if "mediaOutput" in data:
        result["mediaOutput"] = str(video)
    path = output / "candidate.json"
    with path.open("x", encoding="utf-8") as stream:
        os.fchmod(stream.fileno(), 0o600)
        json.dump(result, stream, indent=2, allow_nan=False)
    return path


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("config", type=Path)
    args = parser.parse_args(argv)
    render_candidate(args.config)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
