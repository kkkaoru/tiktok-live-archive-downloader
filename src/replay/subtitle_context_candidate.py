"""Sample bounded original PNG donors with explicitly reviewed clearance, never approve."""

import argparse
import json
from dataclasses import asdict
from fractions import Fraction
from pathlib import Path

import cv2
import numpy as np

from replay.delivery_mask_quality import bind, integer, read_json, reference, rows
from replay.recordings import record
from replay.subtitle_quality_observe import OriginalContext, register
from replay.subtitle_repair import repair_registered_context
from replay.subtitle_saved_observe import _image


def restore_context(config: Path, output: Path, *, sha256: str) -> Path:
    """Keep unknown pixels unchanged; clearance is a supplied review, not OCR absence.

    Each frame supplies sourcePTS, image and excluded PNG references. Target exclusion
    is reviewed glyph support; donor exclusion includes ALL unreviewed pixels. The
    pinned review must explicitly describe original-frame clearance and its scope.
    Geometry/photometry checks remain conditional evidence, not hidden-pixel truth.
    """
    if output != output.resolve() or not output.is_relative_to(Path.cwd().resolve()):
        raise ValueError("Expected absolute checkout-local output")
    bindings: dict[str, str] = {}
    bind(config, sha256, bindings)
    data = read_json(config)
    review = read_json(reference(data.get("review"), bindings))
    if (
        data.get("version") != 1
        or review.get("reviewerKind") != "agent-visual-review"
        or review.get("originalDonorClearanceReviewed") is not True
        or not isinstance(review.get("scope"), str)
        or not review["scope"]
    ):
        raise ValueError("Explicit scoped original-donor review required")
    base = data.get("timeBase")
    if not isinstance(base, str) or Fraction(base) <= 0:
        raise ValueError("Expected positive source time base")
    entries = rows(data.get("frames"))
    if not 3 <= len(entries) <= 8:
        raise ValueError("Expected target and two to seven original donors")
    contexts: list[OriginalContext] = []
    for entry in entries:
        frame = record(entry)
        image = _image(frame.get("image"), bindings)
        excluded = _image(frame.get("excluded"), bindings, mask=True)
        if excluded.shape != image.shape[:2]:
            raise ValueError("Exclusion canvas differs")
        contexts.append(
            OriginalContext(image, excluded != 0, integer(frame.get("sourcePTS")) * Fraction(base))
        )
    target, *donors = contexts
    if (
        len({frame.pts for frame in contexts}) != len(contexts)
        or not any(frame.pts < target.pts for frame in donors)
        or not any(frame.pts > target.pts for frame in donors)
    ):
        raise ValueError("Distinct bracketing source PTS required")
    geometry: list[dict[str, object]] = []
    for donor in donors:
        exclusion = target.excluded | donor.excluded
        fitted = register(
            OriginalContext(target.image, exclusion, target.pts),
            OriginalContext(donor.image, exclusion, donor.pts),
            model="local-affine",
        )
        geometry.append(
            {
                "donorPTS": str(donor.pts),
                "tracks": fitted.tracked_features,
                "translationInliers": fitted.inlier_fraction,
                "localModelsTested": fitted.local_models_tested,
                "localModelsAccepted": fitted.local_models_accepted,
                "geometryPixels": int(np.count_nonzero(fitted.context_geometry)),
                "neighborhoodDiagnostics": (
                    asdict(fitted.local_diagnostics)
                    if fitted.local_diagnostics is not None
                    else None
                ),
            }
        )
    result = repair_registered_context(target, donors=tuple(donors))
    for path, pin in tuple(bindings.items()):
        bind(Path(path), pin, bindings)
    output.mkdir(mode=0o700)
    for name, pixels in (
        ("candidate", result.image),
        ("support", result.support),
        ("unknown", result.unsupported),
    ):
        if pixels is None or not cv2.imwrite(str(output / f"{name}.png"), pixels):
            raise OSError("Cannot write context evidence")
        (output / f"{name}.png").chmod(0o600)
    receipt = output / "context.json"
    with receipt.open("x", encoding="utf-8") as stream:
        json.dump(
            {
                "inputs": bindings,
                "forwardLocalGeometry": geometry,
                "sourcePTS": [str(frame.pts) for frame in contexts],
                "targetPixels": int(np.count_nonzero(target.excluded)),
                "supportedPixels": int(np.count_nonzero(result.support)),
                "unknownPixels": int(np.count_nonzero(target.excluded & (result.support == 0))),
                "fullCaptionSupport": False,
                "sourceClockIndependentlyVerified": False,
                "sourceMediaReverified": False,
                "qualityApproved": False,
                "reviewerKind": "agent-visual-review",
            },
            stream,
            indent=2,
            allow_nan=False,
        )
    receipt.chmod(0o600)
    return receipt


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("config", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--sha256", required=True)
    args = parser.parse_args(argv)
    restore_context(args.config, args.output, sha256=args.sha256)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
