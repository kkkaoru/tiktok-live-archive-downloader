"""Validate explicit source-subtitle eligibility before any destructive candidate work.

OCR strings, learned bands and DB polygons are observations, not semantic approval.
This consumes trusted local classification evidence; it does not classify clothing,
logos or subtitles itself, certify absence, or measure repair quality.
"""

import json
import os
from pathlib import Path

from .caption_io import FrameObservation, native_ocr_frames
from .local_asr import fingerprint
from .recordings import is_record
from .subtitle_clock import VideoFrameClock
from .subtitle_stream import SubtitleSpan

MAX_BYTES = 8 * 1024 * 1024
MAX_EXECUTABLE_BYTES = 256 * 1024 * 1024
MAX_FRAMES = 36_000


def _object(value: object) -> dict[str, object]:
    if not is_record(value):
        raise ValueError("Missing source eligibility evidence object")
    return value


def _reference(value: object, bindings: dict[str, str], *, executable: bool = False) -> Path:
    ref = _object(value)
    name, pin = ref.get("path"), ref.get("sha256")
    if not isinstance(name, str) or not isinstance(pin, str):
        raise ValueError("Missing pinned source eligibility evidence")
    path = Path(name)
    if (
        not path.is_absolute()
        or path != path.resolve()
        or (not executable and not path.is_relative_to(Path.cwd().resolve()))
        or not path.is_file()
        or not 0 < path.stat().st_size <= (MAX_EXECUTABLE_BYTES if executable else MAX_BYTES)
        or (executable and not os.access(path, os.X_OK))
    ):
        raise ValueError("Eligibility requires local evidence or a pinned installed executable")
    if fingerprint(path) != pin or (str(path) in bindings and bindings[str(path)] != pin):
        raise ValueError("Source eligibility evidence fingerprint changed")
    bindings[str(path)] = pin
    return path


def _integer(value: object) -> int:
    if type(value) is not int:
        raise ValueError("Expected eligibility integer")
    return value


def _lines(value: object, observation: FrameObservation) -> tuple[int, ...]:
    if not isinstance(value, list) or not 1 <= len(value) <= 64:
        raise ValueError("Subtitle eligibility requires actual recognized lines")
    indices = tuple(_integer(item) for item in value)
    if indices != tuple(sorted(set(indices))) or not 0 <= indices[0] <= indices[-1] < len(
        observation.detections
    ):
        raise ValueError("Invalid eligibility OCR line selection")
    return indices


def _observations(
    value: object, *, producer: dict[str, object], bindings: dict[str, str]
) -> tuple[FrameObservation, ...]:
    """Read pinned collector batches without weakening the eight-frame parser cap.

    Each batch binds the same collector provenance. Global observation indices
    refer to concatenated batch order; duplicates and seek substitutions fail.
    """
    if isinstance(value, list):
        return native_ocr_frames(value)
    manifest = _object(value)
    batches = manifest.get("batches")
    if (
        type(manifest.get("version")) is not int
        or manifest["version"] != 1
        or manifest.get("kind") != "source-caption-ocr-batches"
        or manifest.get("sourceSHA256") != producer.get("sourceSHA256")
        or not isinstance(batches, list)
        or not 1 <= len(batches) <= MAX_FRAMES
    ):
        raise ValueError("Invalid bounded source OCR batch manifest")
    observations: list[FrameObservation] = []
    previous = -1.0
    for item in batches:
        batch = _object(item)
        provenance = _reference(batch.get("ocrProvenance"), bindings)
        if json.loads(provenance.read_text(encoding="utf-8")) != producer:
            raise ValueError("OCR batch collector provenance differs")
        path = _reference(batch.get("ocr"), bindings)
        frames = native_ocr_frames(json.loads(path.read_text(encoding="utf-8")))
        if not frames or len(observations) + len(frames) > MAX_FRAMES:
            raise ValueError("Invalid total OCR observation budget")
        for frame in frames:
            if frame.requested_time != frame.actual_time or frame.actual_time <= previous:
                raise ValueError("OCR batch clocks are substituted, repeated or unordered")
            previous = frame.actual_time
        observations.extend(frames)
    return tuple(observations)


def validate_eligibility(
    value: object,
    *,
    source_sha256: str,
    clock: VideoFrameClock,
    width: int,
    height: int,
    spans: tuple[SubtitleSpan, ...],
    bindings: dict[str, str],
) -> None:
    """Require same-frame semantic evidence for every requested active repair.

    A classification file must independently document why the selected text is an
    overlay subtitle, not merely quote an OCR score. Its hash is bound here, not
    its truth certified. Non-subtitle/unknown records never silently remove a
    requested frame. Unmodified/absence coverage needs a separate contract.
    """
    path = _reference(value, bindings)
    data = _object(json.loads(path.read_text(encoding="utf-8")))
    if (
        type(data.get("version")) is not int
        or data["version"] != 1
        or data.get("kind") != "source-caption-eligibility"
        or data.get("sourceSHA256") != source_sha256
        or data.get("canvas") != [width, height]
        or data.get("timeBase") != str(clock.time_base)
    ):
        raise ValueError("Source eligibility identity, canvas or clock mismatch")
    # Installed tools may be outside the checkout; evidence and writes may not.
    _reference(data.get("decoder"), bindings, executable=True)
    _reference(data.get("codec"), bindings)
    _reference(data.get("classificationEvidence"), bindings)
    provenance = _reference(data.get("ocrProvenance"), bindings)
    producer = _object(json.loads(provenance.read_text(encoding="utf-8")))
    recognizer_name = producer.get("nativeExecutable")
    if (
        producer.get("sourceSHA256") != source_sha256
        or producer.get("measurementTool") != "video_text_recognize"
        or producer.get("fullFrame") is not True
        or not isinstance(recognizer_name, str)
    ):
        raise ValueError("Eligibility OCR provenance differs from source or operation")
    recognizer = Path(recognizer_name)
    if not recognizer.is_absolute():
        raise ValueError("Eligibility recognizer must be an absolute path")
    _reference(
        {"path": str(recognizer.resolve()), "sha256": producer.get("nativeSHA256")},
        bindings,
        executable=True,
    )
    ocr = _reference(data.get("ocr"), bindings)
    observations = _observations(
        json.loads(ocr.read_text(encoding="utf-8")), producer=producer, bindings=bindings
    )
    frames = data.get("frames")
    if not isinstance(frames, list) or not 1 <= len(frames) <= MAX_FRAMES:
        raise ValueError("Missing bounded eligibility frame records")
    entries: dict[int, dict[str, object]] = {}
    previous = -1
    for item in frames:
        entry = _object(item)
        pts = _integer(entry.get("sourcePTS"))
        if pts <= previous:
            raise ValueError("Eligibility PTS must be unique and ordered")
        entries[pts] = entry
        previous = pts
    for span in spans:
        _validate_span(
            span,
            entries=entries,
            observations=observations,
            clock=clock,
            width=width,
            height=height,
        )


def _validate_span(
    span: SubtitleSpan,
    *,
    entries: dict[int, dict[str, object]],
    observations: tuple[FrameObservation, ...],
    clock: VideoFrameClock,
    width: int,
    height: int,
) -> None:
    for frame in range(span.start_frame, span.end_frame):
        pts = clock.pts[frame]
        entry = entries.get(pts)
        if entry is None or entry.get("classification") != "overlay-subtitle":
            raise ValueError(f"Unresolved source subtitle eligibility at PTS {pts}")
        _validate_frame(
            entry,
            observations=observations,
            time=float(pts * clock.time_base),
            span=span,
            width=width,
            height=height,
        )


def _validate_frame(
    entry: dict[str, object],
    *,
    observations: tuple[FrameObservation, ...],
    time: float,
    span: SubtitleSpan,
    width: int,
    height: int,
) -> None:
    index = _integer(entry.get("observation"))
    if not 0 <= index < len(observations):
        raise ValueError("Missing eligibility OCR observation")
    observed = observations[index]
    if (
        observed.actual_time != time
        or observed.requested_time != time
        or (observed.width, observed.height) != (width, height)
    ):
        raise ValueError("Eligibility OCR differs from actual source frame")
    region = _object(entry.get("region"))
    box = tuple(_integer(region.get(key)) for key in ("x", "y", "width", "height"))
    if box != (span.x, span.y, span.width, span.height):
        raise ValueError("Eligibility region differs from requested repair")
    for index in _lines(entry.get("lines"), observed):
        line = observed.detections[index]
        rect = line.rectangle
        if (
            not line.text.strip()
            or rect.x < span.x
            or rect.y < span.y
            or rect.x + rect.width > span.x + span.width
            or rect.y + rect.height > span.y + span.height
        ):
            raise ValueError("Eligibility text is empty or outside the repair region")
