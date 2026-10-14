"""Export pinned writer ROIs losslessly; no decode, repair or quality approval."""

import argparse
import json
import os
import sys
from pathlib import Path
from zipfile import BadZipFile

import cv2
import numpy as np

from replay.core import ReplayError
from replay.delivery_mask_quality import (
    MAX_BATCH,
    Detector,
    bind,
    digest,
    integer,
    local_path,
    read_capture_pixels,
    read_json,
    reference,
    rows,
)
from replay.recordings import record
from replay.subtitle_detector import LearnedTextDetector
from replay.subtitle_inpaint import Image


def _png(path: Path, pixels: Image) -> dict[str, object]:
    ok, encoded = cv2.imencode(".png", pixels)
    if not ok:
        raise ReplayError("Lossless PNG encoding failed")
    decoded = cv2.imdecode(encoded, cv2.IMREAD_UNCHANGED)
    if decoded is None or not np.array_equal(decoded, pixels):
        raise ReplayError("Lossless PNG roundtrip differs")
    with path.open("xb") as handle:
        os.fchmod(handle.fileno(), 0o600)
        handle.write(encoded.tobytes())
    return {"path": str(path), "sha256": digest(path), "roundtripIdentical": True}


def export_capture(
    receipt: Path,
    directory: Path,
    *,
    sha256: str,
    detect: Detector | None = None,
) -> dict[str, object]:
    """Inspect saved evidence only; a historical source binding is NOT reverified.

    Optional independent detection runs on original pixels, not candidate pixels.
    Its support never replaces the producer glyph denominator. PNGs retain source
    dimensions and BGR-to-PNG channel semantics; no resize or contrast treatment.
    """
    local_path(str(directory))
    bindings: dict[str, str] = {}
    bind(receipt, sha256, bindings)
    capture = record(read_json(receipt).get("losslessEvidence"))
    entries = tuple(record(row) for row in rows(capture.get("frames")))
    if capture.get("pixelFormat") != "bgr24" or not 1 <= len(entries) <= MAX_BATCH:
        raise ReplayError("Expected one to eight BGR writer captures")
    pixels = tuple(read_capture_pixels(reference(row, bindings)) for row in entries)
    if len({integer(row.get("frame")) for row in entries}) != len(entries):
        raise ReplayError("Duplicate capture frame")
    if directory.exists():
        raise ReplayError("Refusing to overwrite diagnostic directory")
    directory.mkdir(mode=0o700)
    detected = detect(tuple(item["source"] for item in pixels)) if detect else None
    if detected is not None and (
        len(detected) != len(pixels)
        or any(
            mask.dtype != np.uint8 or mask.shape != item["glyph"].shape
            for mask, item in zip(detected, pixels, strict=True)
        )
    ):
        raise ReplayError("Invalid independent source detection")
    frames: list[dict[str, object]] = []
    for index, (entry, arrays) in enumerate(zip(entries, pixels, strict=True)):
        frame = integer(entry.get("frame"))
        source_glyph = arrays["glyph"] > 0
        source_repair = arrays["repair"] > 0
        if entry.get("glyphPixels") != int(np.count_nonzero(source_glyph)) or entry.get(
            "repairPixels"
        ) != int(np.count_nonzero(source_repair)):
            raise ReplayError("Writer support counts differ")
        images = dict(arrays)
        if detected is not None:
            images["independent-source"] = detected[index]
        frames.append(
            {
                "frame": frame,
                "sourcePTS": entry.get("sourcePTS"),
                "timeBase": entry.get("timeBase"),
                "region": entry.get("region"),
                "glyphPixels": int(np.count_nonzero(source_glyph)),
                "repairPixels": int(np.count_nonzero(source_repair)),
                "sourceEqualsCandidate": bool(
                    np.array_equal(arrays["source"], arrays["candidate"])
                ),
                "independentSourcePixels": int(np.count_nonzero(detected[index]))
                if detected is not None
                else None,
                "images": {
                    key: _png(directory / f"{frame:08d}-{key}.png", image)
                    for key, image in images.items()
                },
            }
        )
    for name, pin in tuple(bindings.items()):
        bind(Path(name), pin, bindings)
    return {
        "version": 1,
        "kind": "saved-source-mask-roi-diagnostic",
        "inputs": bindings,
        "frames": frames,
        "sourceMediaReverified": False,
        "clockAndRegion": "copied-from-pinned-writer-receipt-not-new-observations",
        "qualityApproved": False,
        "limitations": "Empty detection is not absence; original hidden background is unknown.",
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("receipt", type=Path)
    parser.add_argument("directory", type=Path)
    parser.add_argument("--sha256", required=True)
    parser.add_argument("--detector", type=Path)
    parser.add_argument("--detector-sha256")
    args = parser.parse_args(argv)
    try:
        if (args.detector is None) != (args.detector_sha256 is None):
            raise ReplayError("Supply both detector path and pin")
        detector = None
        if args.detector is not None:
            model = local_path(str(args.detector))
            detector = LearnedTextDetector(model, sha256=args.detector_sha256)
        report = export_capture(
            args.receipt,
            args.directory,
            sha256=args.sha256,
            detect=detector.detect if detector is not None else None,
        )
        report["implementationSHA256"] = digest(Path(__file__))
        report["detector"] = (
            {"path": str(args.detector), "sha256": args.detector_sha256}
            if detector is not None
            else None
        )
        if detector is not None and digest(args.detector) != args.detector_sha256:
            raise ReplayError("Detector changed during diagnosis")
        with (args.directory / "diagnostic.json").open("x", encoding="utf-8") as handle:
            os.fchmod(handle.fileno(), 0o600)
            json.dump(report, handle, indent=2, allow_nan=False)
            handle.write("\n")
    except (ReplayError, ValueError, OSError, BadZipFile) as error:
        print(f"MASK_DIAGNOSTIC_ERROR: {error}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
