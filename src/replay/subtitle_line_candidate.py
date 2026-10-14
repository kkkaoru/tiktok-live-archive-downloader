"""One saved-frame GLYPH diagnostic from pinned, explicitly reviewed line ink.

Does not decode/re-hash source media, infer ink approval, encode video, retry or
measure temporal quality. A partial ink-core candidate is not subtitle removal.
"""

import argparse
import json
import os
from fractions import Fraction
from pathlib import Path

import cv2
import numpy as np

from . import subtitle_line_support
from .caption_io import native_ocr_frames
from .caption_masks import TextRectangle
from .delivery_mask_quality import bind, integer, local_path, read_json, reference, rows
from .local_asr import fingerprint
from .recordings import record
from .subtitle_clock import VideoFrameClock
from .subtitle_eligibility import validate_eligibility
from .subtitle_inpaint import Image
from .subtitle_plan import repair_spans
from .subtitle_quality_diagnostic import _png


def _image(path: Path, *, width: int, height: int, gray: bool = False) -> Image:
    if path.stat().st_size > 16 * 1024 * 1024:
        raise ValueError("Excessive saved-frame PNG")
    image = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE if gray else cv2.IMREAD_COLOR)
    shape = (height, width) if gray else (height, width, 3)
    if image is None or image.shape != shape or image.dtype != np.uint8:
        raise ValueError("Saved frame or support dimensions differ")
    return np.asarray(image, dtype=np.uint8)


def render_saved_candidate(config: Path, review: Path, directory: Path) -> Path:
    """Consume trusted local pixel-review evidence, preserving all input bindings."""
    local_path(str(config))
    local_path(str(review))
    local_path(str(directory))
    if directory.exists():
        raise FileExistsError("Refusing to overwrite saved candidate")
    bindings: dict[str, str] = {}
    bind(config, fingerprint(config), bindings)
    bind(review, fingerprint(review), bindings)
    for name in (
        "subtitle_line_support",
        "subtitle_ink_topology",
        "subtitle_inpaint",
        "subtitle_eligibility",
        "subtitle_quality_diagnostic",
    ):
        path = Path(__file__).resolve().with_name(f"{name}.py")
        bind(path, fingerprint(path), bindings)
    implementation = Path(__file__).resolve()
    bind(implementation, fingerprint(implementation), bindings)
    data, reviewed = read_json(config), read_json(review)
    localization_only = reviewed.get("version") == 2
    if localization_only:
        if (
            reviewed.get("kind") != "reviewed-line-ink-support"
            or reviewed.get("reviewerType") != "agent"
            or reviewed.get("ocrIsLocalizationOnly") is not True
        ):
            raise ValueError("Missing explicit pixel-permission review")
        reference(reviewed.get("segmentationEvidence"), bindings)
    if (
        type(reviewed.get("version")) is not int
        or reviewed["version"] not in (1, 2)
        or (not localization_only and reviewed.get("kind") != "reviewed-line-ink-core")
        or reviewed.get("configSHA256") != bindings[str(config)]
        or reviewed.get("inkClassified") is not True
        or reviewed.get("repairSupportReviewed") is not True
        or reviewed.get("fullCaptionSupport") is not False
    ):
        raise ValueError("Missing bound partial ink-core review")
    clock_data = record(data.get("clock"))
    pts = rows(clock_data.get("pts"))
    if len(pts) != 1 or not isinstance(clock_data.get("timeBase"), str):
        raise ValueError("Saved candidate requires exactly one source PTS")
    clock = VideoFrameClock(
        (integer(pts[0]),),
        Fraction(str(clock_data["timeBase"])),
        integer(clock_data.get("lastDuration")),
    )
    width, height = integer(data.get("width")), integer(data.get("height"))
    spans = repair_spans(
        data.get("masks"), frames=1, width=width, height=height, source_clock=clock
    )
    source_pin = data.get("sourceSHA256")
    if (
        not isinstance(source_pin, str)
        or len(spans) != 1
        or integer(reviewed.get("sourcePTS")) != clock.pts[0]
    ):
        raise ValueError("Saved candidate requires one bound active region")
    validate_eligibility(
        data.get("sourceEligibility"),
        source_sha256=source_pin,
        clock=clock,
        width=width,
        height=height,
        spans=spans,
        bindings=bindings,
    )
    eligibility = read_json(reference(data.get("sourceEligibility"), bindings))
    entries = tuple(record(item) for item in rows(eligibility.get("frames")))
    entry = next(item for item in entries if item.get("sourcePTS") == clock.pts[0])
    observations = native_ocr_frames(
        json.loads(reference(eligibility.get("ocr"), bindings).read_text(encoding="utf-8"))
    )
    observed = observations[integer(entry.get("observation"))]
    selected = tuple(integer(item) for item in rows(entry.get("lines")))
    span = spans[0]
    region = TextRectangle(span.x, span.y, span.width, span.height)
    envelopes = subtitle_line_support.line_masks(observed.detections, region=region)
    source_path = reference(reviewed.get("sourcePNG"), bindings)
    source = _image(source_path, width=width, height=height)
    image = source[span.y : span.y + span.height, span.x : span.x + span.width]
    ink, allowed = (
        np.zeros(image.shape[:2], dtype=np.uint8),
        np.zeros(image.shape[:2], dtype=np.uint8),
    )
    line_reviews = tuple(record(item) for item in rows(reviewed.get("lines")))
    if tuple(integer(item.get("line")) for item in line_reviews) != selected:
        raise ValueError("Reviewed ink must retain each selected OCR line separately")
    for item, index in zip(line_reviews, selected, strict=True):
        line_ink = _image(
            reference(item.get("ink"), bindings), width=span.width, height=span.height, gray=True
        )
        line_allowed = _image(
            reference(item.get("allowed"), bindings),
            width=span.width,
            height=span.height,
            gray=True,
        )
        attributed = subtitle_line_support.attribute_components(
            line_ink if localization_only else line_allowed,
            envelopes=envelopes,
            selected=(index,),
            localization_only=localization_only,
        )
        if not attributed or any(row.selected_line != index for row in attributed):
            raise ValueError("Reviewed repair is unknown or belongs to another line")
        if np.any((allowed != 0) & (line_allowed != 0)):
            raise ValueError("Reviewed line supports overlap")
        # Validate each line before merging; never infer ownership from a union.
        subtitle_line_support.reviewed_support(
            image,
            ink=line_ink,
            allowed=line_allowed,
            envelopes=envelopes,
            selected=(index,),
            localization_only=localization_only,
        )
        ink |= line_ink
        allowed |= line_allowed
    candidate, actual = subtitle_line_support.repair_reviewed_ink(
        image,
        ink=ink,
        allowed=allowed,
        envelopes=envelopes,
        selected=selected,
        localization_only=localization_only,
    )
    full = source.copy()
    full[span.y : span.y + span.height, span.x : span.x + span.width] = candidate
    full_allowed = np.zeros(source.shape[:2], dtype=np.uint8)
    full_allowed[span.y : span.y + span.height, span.x : span.x + span.width] = actual
    outside_changes = int(np.count_nonzero(np.any(source != full, axis=2) & (full_allowed == 0)))
    if outside_changes:
        raise ValueError("Full frame changed outside reviewed support")
    directory.mkdir(mode=0o700)
    images = {
        name: _png(directory / f"{name}.png", pixels)
        for name, pixels in (
            ("source", source),
            ("candidate", full),
            ("source-roi", image),
            ("candidate-roi", candidate),
            ("ink", ink),
            ("repair", actual),
        )
    }
    for name, pin in tuple(bindings.items()):
        # Eligibility separately validates pinned, installed read-only executables.
        if fingerprint(Path(name)) != pin:
            raise ValueError("Saved candidate inputs changed during processing")
    receipt = directory / "candidate.json"
    with receipt.open("x", encoding="utf-8") as handle:
        os.fchmod(handle.fileno(), 0o600)
        json.dump(
            {
                "version": 1,
                "kind": "saved-line-ink-candidate",
                "inputs": bindings,
                "images": images,
                "sourcePTS": clock.pts[0],
                "timeBase": str(clock.time_base),
                "historicalSourceSHA256": source_pin,
                "sourceMediaReverified": False,
                "sourcePNGReverified": True,
                "method": "GLYPH",
                "algorithm": "telea",
                "expansion": 2,
                "softness": 0,
                "selectedLines": selected,
                "ocrIsLocalizationOnly": localization_only,
                "inkPixels": int(np.count_nonzero(ink)),
                "repairPixels": int(np.count_nonzero(actual)),
                "changedPixels": int(np.count_nonzero(np.any(image != candidate, axis=2))),
                "changedPixelsOutsidePermission": outside_changes,
                "fullCaptionSupport": False,
                "qualityApproved": False,
                "temporalMeasured": False,
                "deliveryComplete": False,
            },
            handle,
            indent=2,
            allow_nan=False,
        )
    return receipt


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("config", type=Path)
    parser.add_argument("review", type=Path)
    parser.add_argument("directory", type=Path)
    args = parser.parse_args(argv)
    render_saved_candidate(args.config, args.review, args.directory)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
