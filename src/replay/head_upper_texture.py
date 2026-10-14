"""Bounded texture transport of measured partial upper pixels, never head detection.

The seed's occupied tiles are restricted to the observed face's horizontal span
and the pixels above that same-frame face. Missing support, source boundaries or
failed image correspondences request another measurement; no rectangle or held
position replaces them. Existing complete-contour acceptance remains unchanged.
"""

import argparse
import ast
import json
import math
import resource
import sys
from collections.abc import Iterable
from dataclasses import asdict, dataclass
from fractions import Fraction
from itertools import islice, pairwise
from pathlib import Path
from time import monotonic

import numpy as np

from replay import head_contour_track as tracker
from replay import head_image_retained as retained
from replay import tracking_observations as storage
from replay.caption_style_defaults import HEAD_CAPTION_Y_DEADBAND_PX
from replay.delivery_mask_quality import integer, rows
from replay.head_image_plan import MAX_CELLS, ImageFrameCell
from replay.head_image_support import measured_upper_cells
from replay.producer_code_evidence import _unique_object
from replay.recordings import record
from replay.stages import _uncached_file_digest
from replay.subtitle_clock import VideoFrameClock
from replay.subtitle_inpaint import Image

ROOT = Path(__file__).resolve().parents[2]
PROTOCOL = "approx-upper-texture-v1"
RESET_REASONS = frozenset({"source-cut", "zoom-reset"})


@dataclass(frozen=True, slots=True)
class ApproxTextureSeed:
    contour: tracker.ContourSeed
    cell: ImageFrameCell
    native_status: str
    response_sha256: str


@dataclass(frozen=True, slots=True)
class MeasurementRequest:
    start: Fraction
    end: Fraction
    reason: str
    face_observed: bool

    def encoded(self) -> dict[str, object]:
        return {
            "start": str(self.start),
            "end": str(self.end),
            "reason": self.reason,
            "faceObserved": self.face_observed,
        }


@dataclass(frozen=True, slots=True)
class UpperTextureTrack:
    observations: tuple[tracker.DerivedContourObservation, ...]
    request: MeasurementRequest | None


def read_seed(evidence: retained.RetainedUpperPixels) -> ApproxTextureSeed | None:
    """Use the first exact cell's native partial pixels, retaining its clip status."""
    cell = evidence.job.cells[0]
    if cell.face is None:
        return None
    batch = next((batch for batch in evidence.batches if cell in batch.cells), None)
    if batch is None:
        raise ValueError("Measured seed cell is absent from retained raw batches")
    raw = retained.read_response(batch)
    measured = measured_upper_cells(
        raw,
        times=tuple(c.start for c in batch.cells),
        frame_ends=tuple(c.end for c in batch.cells),
        width=evidence.job.width,
        height=evidence.job.height,
    )
    index = batch.cells.index(cell)
    upper = measured[index].visible_top
    if upper is None:
        return None
    frame = record(rows(raw)[index])
    head = record(frame.get("headTop"))
    coordinates = tuple(
        (integer(record(point).get("x")), integer(record(point).get("y")))
        for point in rows(head.get("upperContour"))
    )
    regions = rows(head.get("interiorRegions"))
    if len(regions) > tracker.MAX_CONTOUR_POINTS:
        raise ValueError("Occupied upper texture tiles exceed the measured region budget")
    x, y, width, _ = cell.face
    selected: list[tuple[int, int, int, int]] = []
    for region in regions:
        tile = record(region)
        if set(tile) != {"x", "y", "width", "height"}:
            raise ValueError("Occupied texture tiles require exact native pixel extents")
        tx, ty, tw, th = (integer(tile[key]) for key in ("x", "y", "width", "height"))
        if not 0 <= tx < tx + tw <= evidence.job.width or not (
            0 <= ty < ty + th <= evidence.job.height
        ):
            raise ValueError("Occupied texture tile exceeds the source canvas")
        if x <= tx and tx + tw <= x + width and upper <= ty and ty + th <= y:
            selected.append((tx, ty, tw, th))
    if not selected:
        return None
    points = np.asarray(coordinates, np.float32)
    if np.any(points < tracker.LK_MARGIN_PIXELS) or np.any(
        points
        >= (
            evidence.job.width - tracker.LK_MARGIN_PIXELS,
            evidence.job.height - tracker.LK_MARGIN_PIXELS,
        )
    ):
        return None
    return ApproxTextureSeed(
        tracker.ContourSeed(
            cell.start,
            evidence.job.width,
            evidence.job.height,
            points,
            tuple(selected),
        ),
        cell,
        str(head["status"]),
        batch.response_sha256,
    )


def source_clock(cells: tuple[ImageFrameCell, ...], time_base: Fraction) -> VideoFrameClock:
    """Represent retained frame cells exactly; never infer or interpolate a PTS."""
    if not 1 <= len(cells) <= MAX_CELLS or time_base <= 0:
        raise ValueError("Expected bounded independent texture source cells")
    ticks = tuple(cell.start / time_base for cell in cells)
    duration = (cells[-1].end - cells[-1].start) / time_base
    if any(tick.denominator != 1 for tick in (*ticks, duration)) or any(
        not 0 <= cell.start < cell.end <= 86400 for cell in cells
    ):
        raise ValueError("Texture source cells do not have exact native integer ticks")
    if any(left.end > right.start for left, right in pairwise(cells)):
        raise ValueError("Texture source cells overlap or reverse")
    return VideoFrameClock(tuple(t.numerator for t in ticks), time_base, duration.numerator)


def track_upper(
    seed: ApproxTextureSeed,
    cells: tuple[ImageFrameCell, ...],
    time_base: Fraction,
    frames: Iterable[tuple[int, Image]],
    *,
    resets: tuple[tuple[Fraction, str], ...] = (),
) -> UpperTextureTrack:
    """Transport current measured pixels only while image support remains valid."""
    source_clock(cells, time_base)
    if seed.cell != cells[0] or seed.contour.time != cells[0].start:
        raise ValueError("Approximate texture seed must match the first source cell")
    starts = {cell.start for cell in cells}
    if any(time not in starts or reason not in RESET_REASONS for time, reason in resets) or any(
        left[0] >= right[0] for left, right in pairwise(resets)
    ):
        raise ValueError("Texture resets require ordered exact source-cell boundaries")
    reset_map = dict(resets)
    request = None
    limit = len(cells)
    for index, cell in enumerate(cells):
        reason = (
            "missing-current-face"
            if cell.face is None
            else "unsupported-frame-extent"
            if cell.end - cell.start > 1
            else "noncontiguous-source-clock"
            if index and cells[index - 1].end != cell.start
            else reset_map.get(cell.start)
            if index
            else None
        )
        if reason is not None:
            limit = index
            request = MeasurementRequest(cell.start, cell.end, reason, cell.face is not None)
            break
    if limit == 0:
        return UpperTextureTrack((), request)
    clock = source_clock(cells[:limit], time_base)
    observations = tracker.track_texture_contour(seed.contour, clock, islice(frames, limit))
    if observations[-1].reason is not None:
        cell = cells[len(observations) - 1]
        request = MeasurementRequest(
            cell.start, cell.end, observations[-1].reason, cell.face is not None
        )
    return UpperTextureTrack(observations, request)


def _code_paths() -> tuple[Path, ...]:
    """Pin the actual current replay import closure, without executing archived code."""
    pending = ["head_upper_texture"]
    found: set[str] = set()
    while pending:
        name = pending.pop()
        if name in found:
            continue
        found.add(name)
        path = ROOT / "src" / "replay" / f"{name}.py"
        tree = ast.parse(path.read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            if isinstance(node, ast.ImportFrom) and node.module == "replay":
                pending.extend(alias.name for alias in node.names)
            elif isinstance(node, ast.ImportFrom) and node.module is not None:
                if node.module.startswith("replay."):
                    pending.append(node.module.removeprefix("replay."))
            elif isinstance(node, ast.Import):
                pending.extend(
                    alias.name.removeprefix("replay.")
                    for alias in node.names
                    if alias.name.startswith("replay.")
                )
    return tuple(ROOT / "src" / "replay" / f"{name}.py" for name in sorted(found))


def _pins(inputs: dict[str, str]) -> None:
    if any(_uncached_file_digest(Path(path)) != digest for path, digest in inputs.items()):
        raise ValueError("Protected texture input bytes changed")


def _measurement_request(
    track: UpperTextureTrack, cells: tuple[ImageFrameCell, ...], errors: tuple[float | None, ...]
) -> MeasurementRequest | None:
    disagreement = next(
        (
            index
            for index, error in enumerate(errors)
            if error is None or error > HEAD_CAPTION_Y_DEADBAND_PX
        ),
        None,
    )
    if disagreement is not None and track.observations[disagreement].reason is None:
        cell = cells[disagreement]
        return MeasurementRequest(
            cell.start, cell.end, "measured-upper-disagreement", cell.face is not None
        )
    return track.request


def run(config: Path, output: Path) -> Path:
    """Decode bounded retained cells and report support/error; this approves no media."""
    started = monotonic()
    config = storage._checked_output(config)
    output = storage._checked_output(output)
    work = ROOT / "edits" / "work"
    if not config.is_relative_to(work) or not output.is_relative_to(work) or output.exists():
        raise ValueError("Texture config/output require existing/new repository work paths")
    config_hash = _uncached_file_digest(config)
    raw = record(
        json.loads(storage._read(config, storage.CONFIG_BYTES), object_pairs_hook=_unique_object)
    )
    if (
        set(raw)
        != {
            "version",
            "retainedImage",
            "sourceSHA256",
            "width",
            "height",
            "timeBase",
            "ffmpeg",
            "resets",
        }
        or type(raw["version"]) is not int
        or raw["version"] != 1
    ):
        raise ValueError("Invalid approximate upper-texture configuration")
    width, height = integer(raw["width"]), integer(raw["height"])
    evidence = retained.read_image_binding(
        raw["retainedImage"],
        source_hash=storage._digest(raw["sourceSHA256"]),
        width=width,
        height=height,
    )
    base = raw["timeBase"]
    if not isinstance(base, str):
        raise ValueError("Texture native time base must be explicit")
    time_base = Fraction(base)
    clock = source_clock(evidence.job.cells, time_base)
    inputs = dict(evidence.inputs)
    inputs[str(config)] = config_hash
    ffmpeg = tracker._reference(raw["ffmpeg"], inputs)
    paths = (*_code_paths(), ROOT / "uv.lock", Path(sys.executable).resolve())
    inputs.update({str(path): _uncached_file_digest(path) for path in paths})
    resets = tuple(
        (Fraction(str(record(row).get("time"))), str(record(row).get("reason")))
        for row in rows(raw["resets"])
    )
    _pins(inputs)
    seed = read_seed(evidence)
    if seed is None:
        cell = evidence.job.cells[0]
        track = UpperTextureTrack(
            (),
            MeasurementRequest(
                cell.start, cell.end, "measured-upper-texture-unavailable", cell.face is not None
            ),
        )
    else:
        track = track_upper(
            seed,
            evidence.job.cells,
            time_base,
            tracker._decode(evidence.job.source, clock, seed.contour, ffmpeg),
            resets=resets,
        )
    rows_out: list[dict[str, object]] = []
    errors: list[float] = []
    all_errors: list[float | None] = []
    for cell, upper, observation in zip(
        evidence.job.cells, evidence.arc.cells, track.observations, strict=False
    ):
        if (upper.start, upper.end) != (cell.start, cell.end):
            raise ValueError("Retained native comparison cell clock differs")
        error = (
            abs(observation.candidate_top - upper.visible_top)
            if (observation.candidate_top is not None and upper.visible_top is not None)
            else None
        )
        if error is not None:
            errors.append(error)
        all_errors.append(error)
        rows_out.append(
            {
                "cell": cell.encoded(),
                "observation": asdict(observation),
                "measuredUpperPixelY": upper.visible_top,
                "absoluteErrorPixels": error,
            }
        )
    complete = len(track.observations) == len(clock.pts) and track.request is None
    accepted = (
        complete and len(errors) == len(clock.pts) and max(errors) <= HEAD_CAPTION_Y_DEADBAND_PX
    )
    request = _measurement_request(track, evidence.job.cells, tuple(all_errors))
    _pins(inputs)
    output.mkdir(mode=0o700)
    result = output / "upper-texture.json"
    storage._write(
        result,
        {
            "version": 1,
            "protocol": PROTOCOL,
            "inputs": inputs,
            "timeBase": str(time_base),
            "requestedFrames": len(clock.pts),
            "rows": rows_out,
            "completeTrack": complete,
            "probeAccepted": accepted,
            "maximumAbsoluteErrorPixels": max(errors) if errors else None,
            "errorBudgetPixels": HEAD_CAPTION_Y_DEADBAND_PX,
            "seedNativeStatus": seed.native_status if seed is not None else None,
            "seedResponseSHA256": seed.response_sha256 if seed is not None else None,
            "measurementRequest": request.encoded() if request else None,
            "elapsedSeconds": monotonic() - started,
            "processPeakRSSBytes": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
            "rssMeaning": "macOS process peak RSS, not exclusive physical allocation",
            "decodeBatchMaximumFrames": tracker.DECODE_BATCH_FRAMES,
            "decodeBatchMaximumBytes": 120 * 1024 * 1024,
            "newSegmentationCalls": 0,
            "derivedPolicySupport": True,
            "semanticHeadVerified": False,
            "headNearVerified": False,
            "qualityApproved": False,
            "deliveryComplete": False,
        },
    )
    return result


def verify(config: Path, output: Path) -> None:
    """Check bound source cells and numerical diagnostics, without pixel replay/approval."""
    raw = record(
        json.loads(storage._read(config, storage.CONFIG_BYTES), object_pairs_hook=_unique_object)
    )
    result = record(
        json.loads(
            storage._read(output / "upper-texture.json", tracker.MAX_JSON_BYTES),
            object_pairs_hook=_unique_object,
        )
    )
    evidence = retained.read_image_binding(
        raw.get("retainedImage"),
        source_hash=storage._digest(raw.get("sourceSHA256")),
        width=integer(raw.get("width")),
        height=integer(raw.get("height")),
    )
    inputs = {
        path: storage._digest(digest) for path, digest in record(result.get("inputs")).items()
    }
    if inputs.get(str(config)) != _uncached_file_digest(config) or any(
        inputs.get(path) != digest for path, digest in evidence.inputs.items()
    ):
        raise ValueError("Texture diagnostic dependency roles or configuration pin differ")
    if any(str(path) not in inputs for path in _code_paths()):
        raise ValueError("Texture diagnostic current helper closure is incomplete")
    _pins(inputs)
    clock = source_clock(evidence.job.cells, Fraction(str(raw.get("timeBase"))))
    entries = rows(result.get("rows"))
    if (
        result.get("protocol") != PROTOCOL
        or type(result.get("version")) is not int
        or result.get("version") != 1
        or integer(result.get("requestedFrames")) != len(clock.pts)
        or len(entries) > len(clock.pts)
    ):
        raise ValueError("Texture diagnostic protocol or requested frame count differs")
    errors: list[float] = []
    supported = 0
    for index, item in enumerate(entries):
        row = record(item)
        observation = record(row.get("observation"))
        point_y = tuple(
            float(record_value[1]) for record_value in _points(observation.get("points"))
        )
        top = observation.get("candidate_top")
        measured = evidence.arc.cells[index].visible_top
        error = None
        if top is not None:
            if (
                isinstance(top, bool)
                or not isinstance(top, (float, int))
                or not math.isfinite(top)
                or not point_y
                or top != min(point_y)
                or observation.get("reason") is not None
            ):
                raise ValueError("Derived texture top contradicts its supported contour pixels")
            supported += 1
            if measured is not None:
                error = abs(top - measured)
                errors.append(error)
        if (
            row.get("cell") != evidence.job.cells[index].encoded()
            or integer(observation.get("source_pts")) != clock.pts[index]
            or row.get("measuredUpperPixelY") != measured
            or row.get("absoluteErrorPixels") != error
        ):
            raise ValueError("Texture diagnostic source clock or comparison error differs")
    complete = len(entries) == len(clock.pts) and supported == len(clock.pts)
    accepted = (
        complete and len(errors) == len(clock.pts) and max(errors) <= HEAD_CAPTION_Y_DEADBAND_PX
    )
    if (
        result.get("completeTrack") is not complete
        or result.get("probeAccepted") is not accepted
        or result.get("maximumAbsoluteErrorPixels") != (max(errors) if errors else None)
        or result.get("errorBudgetPixels") != HEAD_CAPTION_Y_DEADBAND_PX
    ):
        raise ValueError("Texture diagnostic coverage/error acceptance differs")
    if (
        any(
            result.get(flag) is not False
            for flag in (
                "semanticHeadVerified",
                "headNearVerified",
                "qualityApproved",
                "deliveryComplete",
            )
        )
        or result.get("newSegmentationCalls") != 0
    ):
        raise ValueError("Texture diagnostics cannot certify head, caption or delivery quality")
    _pins(inputs)


def _points(value: object) -> tuple[tuple[float, float], ...]:
    items = rows(value)
    if len(items) > tracker.MAX_CONTOUR_POINTS:
        raise ValueError("Derived texture contour exceeds its point budget")
    points = []
    for item in items:
        numbers = rows(item)
        if len(numbers) != 2:
            raise ValueError("Derived texture contour requires finite pixel pairs")
        points.append((_pixel_number(numbers[0]), _pixel_number(numbers[1])))
    return tuple(points)


def _pixel_number(value: object) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
        raise ValueError("Derived texture contour requires finite pixel pairs")
    return float(value)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("config", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--verify", action="store_true")
    args = parser.parse_args()
    if args.verify:
        verify(args.config, args.output)
        print(
            "Verified bound numerical texture diagnostics; "
            "pixel replay and head quality remain unverified"
        )
    else:
        print(run(args.config, args.output))


if __name__ == "__main__":
    main()
