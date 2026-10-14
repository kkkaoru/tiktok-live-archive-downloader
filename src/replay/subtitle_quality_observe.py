"""Bounded registered temporal observations; unknown pixels never become stable.

Coordinates are top-left pixels in equal-sized original context canvases. The
default subpixel-translation model is deliberately conservative. Explicit local
affine neighborhoods need held-out support; unsupported rotation, deformation,
occlusion and subtitle-covered donors remain unknown.
This module does not certify extraction or compositor provenance.
"""

import re
import subprocess
from dataclasses import dataclass
from fractions import Fraction
from pathlib import Path
from typing import Literal

import cv2
import numpy as np
from numpy.typing import NDArray

from replay.delivery_mask_quality import Detector, LosslessFrame, measure_batch
from replay.subtitle_audit import audit_pixel_support
from replay.subtitle_clock import VideoFrameClock
from replay.subtitle_inpaint import Image
from replay.subtitle_prepared import ImageContext, image_observations
from replay.subtitle_registration import GeometryDiagnostics, local_affine_geometry
from replay.subtitle_retry import RepairEvidence

Support = NDArray[np.bool_]
Coordinates = NDArray[np.float32]
MAX_PIXELS = 2_000_000
MIN_FEATURES = 8
ROUNDTRIP_PIXELS = 0.5
MODEL_ERROR_PIXELS = 0.5
MIN_INLIER_FRACTION = 0.9
PHOTO_CODES = 12
MIN_TEXTURE_CODES = 8


@dataclass(frozen=True)
class OriginalContext:
    image: Image
    excluded: Support
    pts: Fraction


@dataclass(frozen=True)
class Correspondence:
    """Target-to-donor coordinates, including explicit unmapped support."""

    donor_xy: Coordinates
    known: Support
    sampled: Image
    tracked_features: int
    inlier_fraction: float
    displacement: tuple[float, float] | None
    context_geometry: Support
    # Context-supported geometry, NOT observed hidden-background correspondence.
    local_models_tested: int = 0
    local_models_accepted: int = 0
    local_diagnostics: GeometryDiagnostics | None = None


@dataclass(frozen=True)
class CandidatePairChange:
    """Apparent candidate variation, never RepairEvidence or background truth.

    A context model cannot see occlusion or deformation hidden by text.
    Natural light/motion and artificial flicker can produce identical changes.
    Keeping this separate prevents promoting conditional geometry to approval.
    """

    changed: Support
    unknown: Support
    repair_pixels: int
    registration: Correspondence | None

    @property
    def changed_fraction(self) -> float | None:
        if not self.repair_pixels or np.any(self.unknown):
            return None
        return float(np.count_nonzero(self.changed) / self.repair_pixels)


@dataclass(frozen=True)
class TemporalObservation:
    unstable: Support
    unknown: Support
    previous: Correspondence | None
    following: Correspondence | None

    @property
    def evaluated(self) -> bool:
        return not bool(np.any(self.unknown))


@dataclass(frozen=True)
class ObservationFrame:
    original: OriginalContext
    candidate: Image
    glyph: Support
    repair: Support
    previous: OriginalContext | None
    following: OriginalContext | None


@dataclass(frozen=True)
class QualityObservation:
    evidence: RepairEvidence
    temporal: TemporalObservation
    readable_lines: int
    clipped_lines: int
    ocr: ImageContext


def decode_context(
    source: Path,
    *,
    clock: VideoFrameClock,
    width: int,
    height: int,
    ffmpeg: Path,
) -> tuple[Image, ...]:
    """Decode ONE consecutive window with exact observed PTS, never per frame.

    At most ten frames (eight targets plus neighbors), 120MiB raw BGR, 60 seconds.
    Uses the same preroll/select/passthrough contract as subtitle_video; showinfo
    checks actual decoder PTS, rather than assigning the requested clock to bytes.
    This decoder is shared by bounded observations, not a full-recording scheduler.
    """
    size = width * height * 3
    if (
        min(width, height) < 8
        or not 1 <= len(clock.pts) <= 10
        or not 0 < size * len(clock.pts) <= 120 * 1024 * 1024
    ):
        raise ValueError("Context decode budget exceeded")
    result = subprocess.run(
        [
            str(ffmpeg),
            "-v",
            "info",
            "-xerror",
            "-nostdin",
            "-copyts",
            "-noaccurate_seek",
            "-ss",
            str(max(0, int(clock.pts[0] * clock.time_base) - 1)),
            "-i",
            str(source),
            "-map",
            "0:v:0",
            "-an",
            "-vf",
            f"settb={clock.time_base},select=gte(pts\\,{clock.pts[0]}),trim=end_frame={len(clock.pts)},showinfo",
            "-frames:v",
            str(len(clock.pts)),
            "-fps_mode",
            "passthrough",
            "-enc_time_base",
            "filter",
            "-pix_fmt",
            "bgr24",
            "-f",
            "rawvideo",
            "pipe:1",
        ],
        check=True,
        capture_output=True,
        timeout=60,
    )
    observed = tuple(
        int(value) for value in re.findall(rb"\bn:\s*\d+\s+pts:\s*(-?\d+)\s", result.stderr)
    )
    if observed != clock.pts or len(result.stdout) != size * len(clock.pts):
        raise ValueError("Decoded context pixels or actual PTS differ from source clock")
    return tuple(
        np.frombuffer(result.stdout, np.uint8, count=size, offset=index * size).reshape(
            height, width, 3
        )
        for index in range(len(clock.pts))
    )


def _validate(context: OriginalContext) -> None:
    image = context.image
    if (
        image.dtype != np.uint8
        or image.ndim != 3
        or image.shape[-1] != 3
        or not 64 <= image.shape[0] * image.shape[1] <= MAX_PIXELS
        or min(image.shape[:2]) < 8
        or context.pts < 0
    ):
        raise ValueError("Invalid bounded original context")
    audit_pixel_support(context.excluded, context.excluded, allowed_roi=context.excluded)
    if context.excluded.shape != image.shape[:2]:
        raise ValueError("Context exclusion canvas mismatch")


def register(
    original: OriginalContext,
    donor: OriginalContext,
    *,
    model: Literal["translation", "local-affine"] = "translation",
) -> Correspondence:
    """Track known-context corners bidirectionally, fit subpixel translation.

    LK uses 15px windows and one pyramid reduction. Exclusions are dilated by 32px
    so tracking windows cannot learn glyphs. At least eight corners, 90% model
    inliers, <=0.5px roundtrip/model errors and textured known pixels are required.
    The explicit local-affine model instead requires independent fit/held-out
    partitions in bounded neighborhoods (see subtitle_registration); it is never
    an automatic fallback. inlier_fraction still reports the global translation
    hypothesis; local model counts and pixel support are separate diagnostics.
    Only supported corner hulls can support measurements. Per-pixel
    known-source photometric mismatch is dilated by 3px to mark local motion
    unknown. Donor exclusions and image boundaries are never zero-filled evidence.
    """
    if model not in ("translation", "local-affine"):
        raise ValueError("Unsupported context registration model")
    _validate(original)
    _validate(donor)
    if original.image.shape != donor.image.shape or original.pts == donor.pts:
        raise ValueError("Registration requires distinct clocks and equal canvases")
    height, width = original.image.shape[:2]
    yy, xx = np.indices((height, width), dtype=np.float32)
    xy = np.stack((xx, yy), axis=-1)
    empty = np.zeros((height, width), dtype=np.bool_)
    failed = Correspondence(xy, empty, donor.image.copy(), 0, 0.0, None, empty)
    gray = cv2.cvtColor(original.image, cv2.COLOR_BGR2GRAY)
    other = cv2.cvtColor(donor.image, cv2.COLOR_BGR2GRAY)
    excluded = cv2.dilate(original.excluded.astype(np.uint8), np.ones((65, 65), np.uint8))
    allowed = (excluded == 0).astype(np.uint8) * 255
    allowed[:32] = allowed[-32:] = 0
    allowed[:, :32] = allowed[:, -32:] = 0
    if not np.any(allowed) or float(np.std(gray[allowed > 0])) < MIN_TEXTURE_CODES:
        return failed
    corners = cv2.goodFeaturesToTrack(gray, 256, 0.01, 8, mask=allowed, blockSize=5)
    if corners is None or len(corners) < MIN_FEATURES:
        return failed
    forward, status, _ = cv2.calcOpticalFlowPyrLK(
        gray, other, corners, corners.copy(), winSize=(15, 15), maxLevel=1
    )
    if forward is None or status is None:
        return failed
    backward, reverse_status, _ = cv2.calcOpticalFlowPyrLK(
        other, gray, forward, forward.copy(), winSize=(15, 15), maxLevel=1
    )
    if backward is None or reverse_status is None:
        return failed
    points = corners.reshape(-1, 2)
    targets = forward.reshape(-1, 2)
    valid = (
        (status.ravel() > 0)
        & (reverse_status.ravel() > 0)
        & (np.linalg.norm(backward.reshape(-1, 2) - points, axis=1) <= ROUNDTRIP_PIXELS)
        & np.isfinite(targets).all(axis=1)
        & (targets[:, 0] >= 32)
        & (targets[:, 0] < width - 32)
        & (targets[:, 1] >= 32)
        & (targets[:, 1] < height - 32)
    )
    safe_targets = np.rint(np.clip(np.nan_to_num(targets), (0, 0), (width - 1, height - 1))).astype(
        np.int32
    )
    donor_excluded = cv2.dilate(donor.excluded.astype(np.uint8), np.ones((65, 65), np.uint8))
    valid &= donor_excluded[safe_targets[:, 1], safe_targets[:, 0]] == 0
    if np.count_nonzero(valid) < MIN_FEATURES:
        return failed
    # Rounding each axis can introduce sqrt(0.5) pixels of model error by itself.
    delta = np.median((targets - points)[valid], axis=0)
    inliers = valid & (np.linalg.norm(targets - points - delta, axis=1) <= MODEL_ERROR_PIXELS)
    confidence = float(np.count_nonzero(inliers) / len(points))
    displacement = None
    tested = accepted = 0
    diagnostics = None
    hull = np.zeros((height, width), np.uint8)
    if model == "local-affine":
        local = local_affine_geometry(
            np.asarray(points, dtype=np.float32),
            targets=np.asarray(targets, dtype=np.float32),
            valid=valid,
            shape=(height, width),
        )
        xy = local.donor_xy
        hull = local.supported.astype(np.uint8)
        tested, accepted = local.tested, local.accepted
        diagnostics = local.diagnostics
    else:
        if np.count_nonzero(inliers) < MIN_FEATURES or confidence < MIN_INLIER_FRACTION:
            return Correspondence(
                xy, empty, donor.image.copy(), len(points), confidence, None, empty
            )
        displacement = float(delta[0]), float(delta[1])
        xy += delta
        cv2.fillConvexPoly(hull, cv2.convexHull(np.rint(points[inliers]).astype(np.int32)), 1)
    map_x, map_y = xy[:, :, 0].copy(), xy[:, :, 1].copy()
    sampled = np.asarray(cv2.remap(donor.image, map_x, map_y, cv2.INTER_LINEAR), dtype=np.uint8)
    donor_unknown = (
        cv2.remap(
            donor.excluded.astype(np.float32),
            map_x,
            map_y,
            cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=1,
        )
        > 0
    )
    mismatch = np.max(np.abs(original.image.astype(np.int16) - sampled.astype(np.int16)), axis=2)
    bad = (mismatch > PHOTO_CODES) & ~original.excluded & ~donor_unknown
    bad = cv2.dilate(bad.astype(np.uint8), np.ones((7, 7), np.uint8)) > 0
    known = (hull > 0) & ~donor_unknown & ~bad
    in_bounds = (map_x >= 0) & (map_x < width - 1) & (map_y >= 0) & (map_y < height - 1)
    geometry = (hull > 0) & in_bounds & ~bad
    return Correspondence(
        xy,
        known & in_bounds,
        sampled,
        len(points),
        confidence,
        displacement,
        geometry,
        tested,
        accepted,
        diagnostics,
    )


def _validate_frame(frame: ObservationFrame) -> None:
    _validate(frame.original)
    audit_pixel_support(frame.glyph, frame.repair, allowed_roi=frame.original.excluded)
    if (
        frame.candidate.dtype != np.uint8
        or frame.candidate.shape != frame.original.image.shape
        or frame.repair.shape != frame.original.excluded.shape
        or np.any(frame.repair & ~frame.original.excluded)
    ):
        raise ValueError("Invalid candidate context or repair support")


def candidate_pair_change(
    frame: ObservationFrame,
    following: ObservationFrame | None,
    *,
    clock: VideoFrameClock,
    registration_model: Literal["translation", "local-affine"] = "translation",
) -> CandidatePairChange:
    """Measure real candidate-to-candidate changes under original-context motion.

    Only adjacent, observed clock entries are permitted, including frame 0->1.
    Register original nonrepair context, never repaired content. Missing pairs,
    failed registration and unsupported pixels stay unknown, not zero. The full
    target repair area is the denominator; empty area is unmeasured. This does
    NOT classify changes as artifacts: hidden motion/lighting remains ambiguous.
    """
    _validate_frame(frame)
    times = tuple(pts * clock.time_base for pts in clock.pts)
    if frame.original.pts not in times:
        raise ValueError("Target is absent from observed clock")
    area = int(np.count_nonzero(frame.repair))
    if following is None:
        return CandidatePairChange(np.zeros_like(frame.repair), frame.repair.copy(), area, None)
    _validate_frame(following)
    index = times.index(frame.original.pts)
    if index + 1 >= len(times) or following.original.pts != times[index + 1]:
        raise ValueError("Pair is not adjacent in observed clock")
    correspondence = register(frame.original, following.original, model=registration_model)
    xy = correspondence.donor_xy
    sampled = cv2.remap(following.candidate, xy[:, :, 0], xy[:, :, 1], cv2.INTER_LINEAR)
    difference = np.max(np.abs(frame.candidate.astype(np.int16) - sampled.astype(np.int16)), axis=2)
    supported = frame.repair & correspondence.context_geometry
    return CandidatePairChange(
        supported & (difference > PHOTO_CODES),
        frame.repair & ~supported,
        area,
        correspondence,
    )


def temporal_observation(
    frame: ObservationFrame,
    *,
    registration_model: Literal["translation", "local-affine"] = "translation",
) -> TemporalObservation:
    """Use BOTH immediate original neighbors; require their mapped colors agree.

    This historical metric compares candidate colors to ORIGINAL donors, not
    candidate-to-candidate variation. Keep its gate behavior until an independent
    artifact/occlusion contract is established; pair change is diagnostic only.
    Instability denominator is the complete repair support, never its known subset.
    Candidate disagreement >12 BGR codes with either mutually agreeing original
    donor is unstable. Donor disagreement is unknown (natural motion, not failure).
    Missing boundary neighbors leave the entire repair support unevaluated.
    """
    _validate_frame(frame)
    if frame.previous is None or frame.following is None:
        return TemporalObservation(np.zeros_like(frame.repair), frame.repair.copy(), None, None)
    if not frame.previous.pts < frame.original.pts < frame.following.pts:
        raise ValueError("Neighbor presentation clocks are unordered")
    before = register(frame.original, frame.previous, model=registration_model)
    after = register(frame.original, frame.following, model=registration_model)
    agreement = (
        np.max(np.abs(before.sampled.astype(np.int16) - after.sampled.astype(np.int16)), axis=2)
        <= PHOTO_CODES
    )
    known = frame.repair & before.known & after.known & agreement
    difference = np.maximum(
        np.max(np.abs(frame.candidate.astype(np.int16) - before.sampled.astype(np.int16)), axis=2),
        np.max(np.abs(frame.candidate.astype(np.int16) - after.sampled.astype(np.int16)), axis=2),
    )
    return TemporalObservation(
        known & (difference > PHOTO_CODES), frame.repair & ~known, before, after
    )


def observe_batch(
    frames: tuple[ObservationFrame, ...],
    *,
    detect: Detector,
    executable: Path,
    work_directory: Path,
) -> tuple[QualityObservation, ...]:
    """One independent native OCR batch and one residual detector batch, no flags.

    OCR absence means only absence observed by Vision accurate ja/en recognition.
    Any nonblank recognized line counts, without a confidence-based disappearance.
    Clipped lines invalidate the measurement. Input provenance is a separate gate.
    """
    if not 1 <= len(frames) <= 8:
        raise ValueError("Supply one to eight observation frames")
    temporal = tuple(temporal_observation(frame) for frame in frames)
    ocr = image_observations(
        tuple(frame.candidate for frame in frames),
        executable=executable,
        work_directory=work_directory,
    )
    readable = tuple(
        sum(bool(line.text.strip()) for line in row.observation.detections) for row in ocr
    )
    lossless = tuple(
        LosslessFrame(
            frame.original.image,
            frame.candidate,
            frame.glyph,
            frame.repair,
            time.unstable,
            count > 0,
            frame.original.pts,
            frame.original.pts,
        )
        for frame, time, count in zip(frames, temporal, readable, strict=True)
    )
    measured = measure_batch(lossless, detect=detect)
    return tuple(
        QualityObservation(
            RepairEvidence(
                metric.technically_valid and time.evaluated and row.clipped_lines == 0,
                metric.residual_fraction,
                metric.unstable_fraction if time.evaluated else None,
                metric.readable_text,
            ),
            time,
            count,
            row.clipped_lines,
            row,
        )
        for metric, time, count, row in zip(measured, temporal, readable, ocr, strict=True)
    )
