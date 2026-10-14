"""Review a source-bound head contour seed; propagate only explicit reviewed seeds."""

import argparse
import hashlib
import json
from dataclasses import dataclass
from fractions import Fraction
from pathlib import Path

import cv2
import numpy as np
from numpy.typing import NDArray

from replay.delivery_mask_quality import integer, read_json, reference, rows
from replay.local_asr import fingerprint
from replay.recordings import record
from replay.subtitle_inpaint import Image

Coordinates = NDArray[np.float32]


@dataclass(frozen=True, slots=True)
class ContourTrackMetrics:
    points: Coordinates
    supported: NDArray[np.bool_]
    forward_status: NDArray[np.bool_]
    backward_status: NDArray[np.bool_]
    in_bounds: NDArray[np.bool_]
    roundtrip_error: Coordinates
    appearance_error: Coordinates


def propagate_contour(
    source: Image, donor: Image, points: Coordinates
) -> tuple[Coordinates, NDArray[np.bool_]]:
    """Return independently supported correspondences, preserving the original API."""
    metrics = measure_contour_tracks(source, donor, points)
    return metrics.points, metrics.supported


def measure_contour_tracks(source: Image, donor: Image, points: Coordinates) -> ContourTrackMetrics:
    """Independently check each seed's LK roundtrip and observed edge appearance.

    These are point correspondences, not segmentation or subject confidence.
    Failed points remain unknown; a partial contour cannot supply a head top.
    """
    if (
        source.dtype != np.uint8
        or donor.dtype != np.uint8
        or source.ndim != 3
        or source.shape[-1] != 3
        or donor.shape != source.shape
        or not 256 <= source.shape[0] * source.shape[1] <= 16_000_000
        or points.dtype != np.float32
        or points.ndim != 2
        or points.shape[1] != 2
        or not 3 <= len(points) <= 256
        or not np.isfinite(points).all()
        or np.any(points < 8)
        or np.any(points >= (source.shape[1] - 8, source.shape[0] - 8))
    ):
        raise ValueError("Invalid contour images or seed coordinates")
    gray, other = cv2.cvtColor(source, cv2.COLOR_BGR2GRAY), cv2.cvtColor(donor, cv2.COLOR_BGR2GRAY)
    tracks = points.reshape(-1, 1, 2)
    forward, status, _ = cv2.calcOpticalFlowPyrLK(
        gray, other, tracks, tracks.copy(), winSize=(15, 15), maxLevel=1
    )
    if forward is None or status is None:
        return _missing_tracks(points)
    backward, reverse, _ = cv2.calcOpticalFlowPyrLK(
        other, gray, forward, forward.copy(), winSize=(15, 15), maxLevel=1
    )
    if backward is None or reverse is None:
        return _missing_tracks(points, status.ravel() > 0)
    xy = np.asarray(forward.reshape(-1, 2), np.float32)
    forward_status, backward_status = status.ravel() > 0, reverse.ravel() > 0
    valid = forward_status & backward_status
    valid &= np.isfinite(xy).all(axis=1)
    roundtrip = np.asarray(np.linalg.norm(backward.reshape(-1, 2) - points, axis=1), np.float32)
    valid &= roundtrip <= 0.5
    in_bounds = np.all((xy >= 8) & (xy < (source.shape[1] - 8, source.shape[0] - 8)), axis=1)
    valid &= in_bounds
    source_samples = cv2.remap(gray, points[:, :1], points[:, 1:], cv2.INTER_LINEAR)
    donor_samples = cv2.remap(other, xy[:, :1], xy[:, 1:], cv2.INTER_LINEAR)
    appearance = np.asarray(
        np.abs(source_samples.astype(np.int16) - donor_samples.astype(np.int16)).ravel(),
        np.float32,
    )
    valid &= appearance <= 12
    return ContourTrackMetrics(
        xy, valid, forward_status, backward_status, in_bounds, roundtrip, appearance
    )


def _missing_tracks(
    points: Coordinates, forward: NDArray[np.bool_] | None = None
) -> ContourTrackMetrics:
    missing = np.zeros(len(points), np.bool_)
    errors = np.full(len(points), np.inf, np.float32)
    return ContourTrackMetrics(
        points.copy(),
        missing,
        missing if forward is None else forward,
        missing,
        missing,
        errors,
        errors,
    )


def review_seed(config: Path, output: Path) -> Path:
    """Emit an actual image for review; never infer approval from a face/person box."""
    root = Path.cwd().resolve()
    if output != output.resolve() or not output.is_relative_to(root) or output.exists():
        raise ValueError("Expected new checkout-local seed output directory")
    if config.stat().st_size > 1024 * 1024:
        raise ValueError("Seed configuration exceeds byte budget")
    config_pin = fingerprint(config)
    data = record(json.loads(config.read_text(encoding="utf-8")))
    ref = record(data.get("sourcePNG"))
    source = Path(str(ref.get("path")))
    if (
        source != source.resolve()
        or not source.is_relative_to(root)
        or not 0 < source.stat().st_size <= 16 * 1024 * 1024
        or fingerprint(source) != ref.get("sha256")
    ):
        raise ValueError("Seed source image binding differs")
    image = cv2.imread(str(source), cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError("Cannot decode seed source image")
    pixels = np.asarray(image, dtype=np.uint8)
    points = np.asarray(data.get("points"), dtype=np.float32)
    # Validate geometry without claiming same-image tracking as an observation.
    propagate_contour(pixels, pixels, points)
    candidate = pixels.copy()
    cv2.polylines(candidate, [np.rint(points).astype(np.int32)], False, (0, 255, 0), 2)
    output.mkdir(mode=0o700)
    path = output / "seed-review.png"
    if not cv2.imwrite(str(path), candidate):
        raise OSError("Cannot write seed review image")
    path.chmod(0o600)
    if fingerprint(source) != ref.get("sha256") or fingerprint(config) != config_pin:
        raise ValueError("Seed inputs changed during image generation")
    report = {
        "sourcePNG": ref,
        "configSHA256": config_pin,
        "sourceClockIndependentlyVerified": False,
        "sourcePTS": data.get("sourcePTS"),
        "timeBase": data.get("timeBase"),
        "sourceMediaSHA256": data.get("sourceMediaSHA256"),
        "seedPoints": points.tolist(),
        "candidateTop": float(np.min(points[:, 1])),
        "seedReviewed": False,
        "headContourComplete": False,
        "independentHeadObservations": 0,
        "placementApproved": False,
        "imageSHA256": hashlib.sha256(path.read_bytes()).hexdigest(),
        "nextStep": (
            "Review full outer hair contour and occlusion before propagate_contour; "
            "require every point supported and independently inspect propagated boundaries."
        ),
    }
    receipt = output / "seed-review.json"
    with receipt.open("x", encoding="utf-8") as stream:
        json.dump(report, stream, indent=2, allow_nan=False)
    receipt.chmod(0o600)
    return receipt


def propagate_reviewed_arc(config: Path, output: Path) -> Path:
    """Track a scoped visible arc, retaining failed points and no whole-head approval."""
    bindings: dict[str, str] = {}
    data = read_json(config)
    config_pin = fingerprint(config)
    review = read_json(reference(data.get("visibleArcReview"), bindings))
    if (
        review.get("reviewerKind") != "agent-visual-review"
        or review.get("visibleArcReviewed") is not True
        or not isinstance(review.get("scope"), str)
        or not review["scope"]
    ):
        raise ValueError("Explicit visible-arc review required")
    donors = rows(data.get("donors"))
    base = data.get("timeBase")
    if not 1 <= len(donors) <= 8 or not isinstance(base, str) or Fraction(base) <= 0:
        raise ValueError("Invalid bounded arc clocks")
    source_pts = integer(data.get("sourcePTS"))
    # Reuse source validation and unapproved seed imagery; this does not review a full head.
    review_seed(config, output)
    source_path = reference(data.get("sourcePNG"), bindings)
    source = cv2.imread(str(source_path), cv2.IMREAD_COLOR)
    if source is None:
        raise ValueError("Cannot decode arc source")
    points = np.asarray(data.get("points"), np.float32)
    observations: list[dict[str, object]] = []
    seen = {source_pts}
    for entry in donors:
        frame = record(entry)
        pts = integer(frame.get("sourcePTS"))
        if pts in seen or pts < 0:
            raise ValueError("Distinct nonnegative arc PTS required")
        seen.add(pts)
        path = reference(frame.get("sourcePNG"), bindings)
        if not 0 < path.stat().st_size <= 16 * 1024 * 1024:
            raise ValueError("Arc image exceeds byte budget")
        donor = cv2.imread(str(path), cv2.IMREAD_COLOR)
        if donor is None:
            raise ValueError("Cannot decode arc donor")
        moved, valid = propagate_contour(
            np.asarray(source, np.uint8), np.asarray(donor, np.uint8), points
        )
        observations.append(
            {
                "sourcePTS": pts,
                "points": [point.tolist() if np.isfinite(point).all() else None for point in moved],
                "supported": valid.tolist(),
                "supportedPoints": int(np.count_nonzero(valid)),
                "unknownPoints": int(np.count_nonzero(~valid)),
                "independentlyReviewedPropagation": False,
            }
        )
    if fingerprint(config) != config_pin:
        raise ValueError("Arc config changed during propagation")
    for path_text, pin in bindings.items():
        if fingerprint(Path(path_text)) != pin:
            raise ValueError("Arc image or review changed during propagation")
    receipt = output / "arc-tracks.json"
    with receipt.open("x", encoding="utf-8") as stream:
        json.dump(
            {
                "configSHA256": config_pin,
                "inputs": bindings,
                "timeBase": base,
                "seedPTS": source_pts,
                "sourceClockIndependentlyVerified": False,
                "visibleArcReviewed": True,
                "seedReviewed": False,
                "headContourComplete": False,
                "placementApproved": False,
                "independentHeadObservations": 0,
                "observations": observations,
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
    parser.add_argument("--propagate-reviewed-arc", action="store_true")
    args = parser.parse_args(argv)
    if args.propagate_reviewed_arc:
        propagate_reviewed_arc(args.config, args.output)
    else:
        review_seed(args.config, args.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
