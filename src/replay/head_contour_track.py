"""Track measured foreground contour seeds without inventing face detections."""

import argparse
import json
import re
from collections.abc import Iterable, Iterator
from dataclasses import asdict, dataclass
from fractions import Fraction
from itertools import pairwise
from pathlib import Path

import numpy as np

from replay.delivery_mask_quality import integer, rows
from replay.head_seed import Coordinates, measure_contour_tracks
from replay.head_texture_track import derive_contour_motion, foreground_features
from replay.local_asr import fingerprint
from replay.recordings import record
from replay.subtitle_clock import VideoFrameClock
from replay.subtitle_inpaint import Image
from replay.subtitle_quality_observe import decode_context

MAX_JSON_BYTES = 2 * 1024 * 1024
MAX_TRACK_FRAMES = 3600
MAX_TRACK_SECONDS = 120
MAX_CONTOUR_POINTS = 256
DECODE_BATCH_FRAMES = 10
LK_MARGIN_PIXELS = 8


@dataclass(frozen=True, slots=True)
class ContourSeed:
    time: Fraction
    width: int
    height: int
    points: Coordinates
    interior_regions: tuple[tuple[int, int, int, int], ...] = ()


@dataclass(frozen=True, slots=True)
class ContourObservation:
    source_pts: int
    provenance: str
    supported_points: int
    unknown_points: int
    points: tuple[tuple[float, float] | None, ...]
    candidate_top: float | None
    reason: str | None
    supported: tuple[bool, ...]
    roundtrip_error: tuple[float | None, ...] | None
    appearance_error: tuple[float | None, ...] | None
    forward_status: tuple[bool, ...] | None
    backward_status: tuple[bool, ...] | None
    in_bounds: tuple[bool, ...] | None


@dataclass(frozen=True, slots=True)
class DerivedContourObservation:
    source_pts: int
    provenance: str
    candidate_top: float | None
    points: tuple[tuple[float, float], ...]
    feature_count: int
    inlier_count: int
    maximum_inlier_residual: float | None
    reason: str | None
    input_feature_count: int
    independently_supported_features: int


def _time(value: object) -> Fraction:
    raw = record(value)
    value, scale = integer(raw.get("value")), integer(raw.get("timescale"))
    if not 1 <= scale <= 1_000_000_000 or value > 86400 * scale:
        raise ValueError("Invalid contour presentation clock")
    return Fraction(value, scale)


def contour_seed(response: object, *, clock: VideoFrameClock) -> ContourSeed:
    """Accept measured support pixels only; never derive seeds from a rectangle."""
    frames = rows(response)
    if len(frames) != 1:
        raise ValueError("Exactly one measured contour seed is required")
    frame = record(frames[0])
    expected = clock.pts[0] * clock.time_base
    if _time(frame.get("actual")) != expected or _time(frame.get("requested")) != expected:
        raise ValueError("Contour seed and decoded source clock differ")
    width, height = integer(frame.get("width")), integer(frame.get("height"))
    if min(width, height) < 16 or width * height > 16_000_000:
        raise ValueError("Invalid contour canvas")
    head = record(frame.get("headTop"))
    if (
        frame.get("captionPlacementApproved") is not False
        or head.get("status") != "candidate"
        or head.get("semanticHeadVerified") is not False
        or head.get("captionPlacementApproved") is not False
        or head.get("contourScope") != "upper-component-in-face-x-range"
    ):
        raise ValueError("Unapproved measured foreground contour required")
    points = rows(head.get("upperContour"))
    if not 3 <= len(points) <= MAX_CONTOUR_POINTS:
        raise ValueError("Measured contour requires 3..256 points")
    coordinates = tuple(
        (integer(record(point).get("x")), integer(record(point).get("y"))) for point in points
    )
    if (
        len(set(coordinates)) != len(coordinates)
        or any(x >= width or y >= height for x, y in coordinates)
        or any(right[0] <= left[0] for left, right in pairwise(coordinates))
    ):
        raise ValueError("Contour pixels must be distinct, ordered and inside the source")
    regions = tuple(
        tuple(integer(record(region).get(key)) for key in ("x", "y", "width", "height"))
        for region in rows(head.get("interiorRegions", []))
    )
    return ContourSeed(
        expected,
        width,
        height,
        np.asarray(coordinates, np.float32),
        tuple((r[0], r[1], r[2], r[3]) for r in regions),
    )


def track_contour(
    seed: ContourSeed, clock: VideoFrameClock, frames: Iterable[tuple[int, Image]]
) -> tuple[ContourObservation, ...]:
    """Stop at the first unsupported point; no hold or automatic reseeding policy."""
    if (
        seed.time != clock.pts[0] * clock.time_base
        or seed.points.dtype != np.float32
        or seed.points.ndim != 2
        or seed.points.shape[1] != 2
        or not 3 <= len(seed.points) <= MAX_CONTOUR_POINTS
        or not np.isfinite(seed.points).all()
        or len(clock.pts) > MAX_TRACK_FRAMES
        or (clock.pts[-1] - clock.pts[0]) * clock.time_base > MAX_TRACK_SECONDS
    ):
        raise ValueError("Contour track clock or duration exceeds its seed contract")
    points = seed.points.copy()
    observations: list[ContourObservation] = []
    previous: Image | None = None
    for index, (pts, image) in enumerate(frames):
        if index >= len(clock.pts) or pts != clock.pts[index]:
            raise ValueError("Decoded contour frame clock differs")
        if image.dtype != np.uint8 or image.shape != (seed.height, seed.width, 3):
            raise ValueError("Decoded contour frame geometry differs")
        reason: str | None = None
        metrics = None
        if previous is None:
            valid = np.all(
                (points >= LK_MARGIN_PIXELS)
                & (points < (seed.width - LK_MARGIN_PIXELS, seed.height - LK_MARGIN_PIXELS)),
                axis=1,
            )
            provenance = "measured-foreground-contour"
            if not bool(valid.all()):
                reason = "seed_outside_lk_support"
        else:
            metrics = measure_contour_tracks(previous, image, points)
            points, valid = metrics.points, metrics.supported
            provenance = "tracked-foreground-contour"
            if not bool(valid.all()):
                reason = "lk_roundtrip_or_appearance_unsupported"
        observations.append(
            ContourObservation(
                pts,
                provenance,
                int(np.count_nonzero(valid)),
                int(np.count_nonzero(~valid)),
                tuple(
                    (float(point[0]), float(point[1])) if np.isfinite(point).all() else None
                    for point in points
                ),
                float(np.min(points[:, 1])) if reason is None else None,
                reason,
                tuple(bool(value) for value in valid),
                _finite_errors(metrics.roundtrip_error) if metrics is not None else None,
                _finite_errors(metrics.appearance_error) if metrics is not None else None,
                tuple(bool(value) for value in metrics.forward_status)
                if metrics is not None
                else None,
                tuple(bool(value) for value in metrics.backward_status)
                if metrics is not None
                else None,
                tuple(bool(value) for value in metrics.in_bounds) if metrics is not None else None,
            )
        )
        if reason is not None:
            return tuple(observations)
        previous = image
    if len(observations) != len(clock.pts):
        raise ValueError("Decoded contour sequence is incomplete")
    return tuple(observations)


def _finite_errors(values: Coordinates) -> tuple[float | None, ...]:
    return tuple(float(value) if np.isfinite(value) else None for value in values)


def track_texture_contour(
    seed: ContourSeed, clock: VideoFrameClock, frames: Iterable[tuple[int, Image]]
) -> tuple[DerivedContourObservation, ...]:
    """Compare each frame to the measured seed, avoiding cumulative fit drift.

    The same measured texture anchors every comparison. A failed anchor comparison
    stops the bounded track; it never becomes a held or fabricated detection.
    """
    if (
        seed.time != clock.pts[0] * clock.time_base
        or len(clock.pts) > MAX_TRACK_FRAMES
        or (clock.pts[-1] - clock.pts[0]) * clock.time_base > MAX_TRACK_SECONDS
    ):
        raise ValueError("Texture contour clock exceeds bounded contract")
    observations: list[DerivedContourObservation] = []
    previous: Image | None = None
    points = seed.points.copy()
    features = np.empty((0, 2), np.float32)
    for index, (pts, image) in enumerate(frames):
        if (
            index >= len(clock.pts)
            or pts != clock.pts[index]
            or image.shape != (seed.height, seed.width, 3)
        ):
            raise ValueError("Texture frame clock or geometry differs")
        reason: str | None = None
        residual: float | None = None
        if previous is None:
            features = foreground_features(image, seed.interior_regions)
            count = len(features)
            provenance = "measured-foreground-contour"
            if count < 12:
                reason = "insufficient_foreground_texture"
            input_count = supported_count = count
        else:
            input_count = len(features)
            motion = derive_contour_motion(previous, image, features, seed.points)
            supported_count = int(np.count_nonzero(motion.metrics.supported))
            reason = motion.reason
            count = sum(motion.inliers)
            provenance = "texture-derived-foreground-contour"
            values = [
                error
                for error, valid in zip(motion.residuals, motion.inliers, strict=True)
                if valid and error is not None
            ]
            residual = max(values) if values else None
            if motion.contour is not None:
                points = motion.contour
        observations.append(
            DerivedContourObservation(
                pts,
                provenance,
                float(points[:, 1].min()) if reason is None else None,
                tuple((float(p[0]), float(p[1])) for p in points) if reason is None else (),
                len(features),
                count,
                residual,
                reason,
                input_count,
                supported_count,
            )
        )
        if reason is not None:
            return tuple(observations)
        if previous is None:
            previous = image
    if len(observations) != len(clock.pts):
        raise ValueError("Decoded texture sequence is incomplete")
    return tuple(observations)


def _read(path: Path) -> object:
    if not 0 < path.stat().st_size <= MAX_JSON_BYTES:
        raise ValueError("Contour JSON byte budget exceeded")
    return json.loads(path.read_text(encoding="utf-8"))


def _reference(value: object, bindings: dict[str, str]) -> Path:
    raw = record(value)
    name, pin = raw.get("path"), raw.get("sha256")
    if (
        not isinstance(name, str)
        or not isinstance(pin, str)
        or re.fullmatch(r"[a-f0-9]{64}", pin) is None
    ):
        raise ValueError("Explicit contour input path and SHA256 required")
    path = Path(name)
    if not path.is_absolute() or path != path.resolve() or not path.is_file():
        raise ValueError("Contour input must be a canonical regular file")
    if fingerprint(path) != pin:
        raise ValueError("Contour input fingerprint differs")
    bindings[name] = pin
    return path


def _decode(
    source: Path, clock: VideoFrameClock, seed: ContourSeed, ffmpeg: Path
) -> Iterator[tuple[int, Image]]:
    for start in range(0, len(clock.pts), DECODE_BATCH_FRAMES):
        end = min(len(clock.pts), start + DECODE_BATCH_FRAMES)
        last = clock.pts[end] - clock.pts[end - 1] if end < len(clock.pts) else clock.last_duration
        part = VideoFrameClock(clock.pts[start:end], clock.time_base, last)
        images = decode_context(
            source, clock=part, width=seed.width, height=seed.height, ffmpeg=ffmpeg
        )
        yield from zip(part.pts, images, strict=True)


def run(config: Path, output: Path) -> Path:
    """Read-only media analysis with pinned native seed and verified decoded PTS."""
    root = Path.cwd().resolve()
    if output != output.resolve() or not output.is_relative_to(root) or output.exists():
        raise ValueError("Expected a new checkout-local contour output directory")
    config_pin = fingerprint(config)
    raw = record(_read(config))
    if type(raw.get("version")) is not int or raw.get("version") != 1:
        raise ValueError("Unsupported contour request version")
    bindings: dict[str, str] = {str(config): config_pin}
    source = _reference(raw.get("source"), bindings)
    request_path = _reference(raw.get("nativeRequest"), bindings)
    response_path = _reference(raw.get("nativeResponse"), bindings)
    _reference(raw.get("nativeExecutable"), bindings)
    ffmpeg = _reference(raw.get("ffmpeg"), bindings)
    clock_raw = record(raw.get("clock"))
    base = clock_raw.get("timeBase")
    if not isinstance(base, str):
        raise ValueError("Explicit contour time base required")
    clock = VideoFrameClock(
        tuple(integer(p) for p in rows(clock_raw.get("pts"))),
        Fraction(base),
        integer(clock_raw.get("lastDuration")),
    )
    request = record(_read(request_path))
    requested_samples = rows(request.get("samples"))
    if (
        request.get("path") != str(source)
        or len(requested_samples) != 1
        or _time(requested_samples[0]) != clock.pts[0] * clock.time_base
        or _time(record(request.get("headFace")).get("presentation"))
        != clock.pts[0] * clock.time_base
    ):
        raise ValueError("Native contour seed source binding differs")
    seed = contour_seed(_read(response_path), clock=clock)
    method = raw.get("trackingMethod", "contour-lk")
    observations: tuple[ContourObservation, ...] | tuple[DerivedContourObservation, ...]
    if method == "foreground-texture":
        observations = track_texture_contour(seed, clock, _decode(source, clock, seed, ffmpeg))
    elif method == "contour-lk":
        observations = track_contour(seed, clock, _decode(source, clock, seed, ffmpeg))
    else:
        raise ValueError("Unknown contour tracking method")
    for name, pin in bindings.items():
        if fingerprint(Path(name)) != pin:
            raise ValueError("Contour inputs changed during tracking")
    output.mkdir(mode=0o700)
    receipt = output / "contour-tracks.json"
    with receipt.open("x", encoding="utf-8") as stream:
        json.dump(
            {
                "version": 1,
                "trackingMethod": method,
                "inputs": bindings,
                "timeBase": str(clock.time_base),
                "seedPTS": clock.pts[0],
                "requestedFrames": len(clock.pts),
                "observedFrames": len(observations),
                "completeTrack": len(observations) == len(clock.pts)
                and observations[-1].reason is None,
                "semanticHeadVerified": False,
                "captionPlacementApproved": False,
                "headContourComplete": False,
                "observations": [asdict(observation) for observation in observations],
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
    args = parser.parse_args(argv)
    run(args.config, args.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
