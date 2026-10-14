"""Read observed FCP object-tracker positions and project calibrated caption Y.

Results are candidates, not evidence that an ROI follows a head. No head position,
coordinate scale, confidence or approval is inferred from Apple's tracking data.
"""

from __future__ import annotations

import argparse
import json
import math
import plistlib
import re
from bisect import bisect_left
from dataclasses import dataclass, replace
from fractions import Fraction
from itertools import pairwise
from pathlib import Path
from typing import Literal
from urllib.parse import unquote, urlsplit
from xml.etree import ElementTree as ET

from replay.caption_style_defaults import HEAD_CAPTION_GAP_PX
from replay.core import ReplayError
from replay.fcp_project import rational
from replay.head_observations import tracking_frames
from replay.subtitle_clock import VideoFrameClock
from replay.tracking_observations import RESPONSE_BYTES, _read, load_job, verify

MAX_TRACK_BYTES = 16 * 1024 * 1024
MAX_TRACK_SAMPLES = 200_000
MAX_COORDINATE = 100_000


@dataclass(frozen=True, slots=True)
class TrackingPosition:
    time: Fraction
    x: float
    y: float


@dataclass(frozen=True, slots=True)
class TrackingCalibration:
    """Explicit measured geometry in output pixels; positive source Y points down."""

    reference_time: Fraction
    source_head_top: float
    tracker_y_to_pixels: float
    caption_local_bottom: float
    gap: float
    canvas_height: int
    roi: tuple[float, float, float, float]


@dataclass(frozen=True, slots=True)
class FacePositionSeries:
    """Face positions; optional candidate derivations never become observations."""

    positions: tuple[TrackingPosition, ...]
    coverage_end: Fraction
    width: int
    height: int
    anchor: Literal["top", "bottom"] = "top"
    derived_ranges: tuple[MissingFaceRange, ...] = ()


@dataclass(frozen=True, slots=True)
class MissingFaceRange:
    start: Fraction
    end: Fraction
    frames: int
    derived_from: Fraction
    method: Literal["leading-observation-backfill", "last-known-hold"]
    reason: Literal["missing", "ambiguous"]


@dataclass(frozen=True, slots=True)
class FaceObservation:
    time: Fraction
    position: TrackingPosition | None
    reason: Literal["missing", "ambiguous"] | None = None


@dataclass(frozen=True, slots=True)
class FaceObservationSeries:
    samples: tuple[FaceObservation, ...]
    coverage_end: Fraction
    width: int
    height: int
    anchor: Literal["top", "bottom"] = "top"


def face_observation_series(
    observations: object,
    clock: VideoFrameClock,
    *,
    width: int,
    height: int,
    anchor: Literal["top", "bottom"] = "top",
) -> FaceObservationSeries:
    """Preserve every verified frame, including failed or ambiguous detection."""
    if anchor not in ("top", "bottom"):
        raise ReplayError("Unknown face anchor")
    frames = tracking_frames(observations, clock, width=width, height=height)
    samples = tuple(
        FaceObservation(
            frame.time,
            TrackingPosition(
                frame.time,
                frame.faces[0].box.left,
                frame.faces[0].box.top if anchor == "top" else frame.faces[0].box.bottom,
            )
            if len(frame.faces) == 1
            else None,
            None if len(frame.faces) == 1 else ("missing" if not frame.faces else "ambiguous"),
        )
        for frame in frames
    )
    return FaceObservationSeries(
        samples, (clock.pts[-1] + clock.last_duration) * clock.time_base, width, height, anchor
    )


def resolve_face_observations(
    series: FaceObservationSeries, *, missing_policy: Literal["reject", "hold"] = "reject"
) -> FacePositionSeries:
    """Explicit candidate holds retain missing provenance and all frame times."""
    if missing_policy not in ("reject", "hold"):
        raise ReplayError("Unknown missing-face policy")
    observed = tuple(sample.position for sample in series.samples if sample.position is not None)
    if not observed:
        raise ReplayError("No observed face exists; candidate positions cannot be invented")
    if missing_policy == "reject" and len(observed) != len(series.samples):
        raise ReplayError("Face-driven caption candidates require one observed face per frame")
    if (
        not series.samples
        or series.coverage_end <= series.samples[-1].time
        or any(b.time <= a.time for a, b in pairwise(series.samples))
    ):
        raise ReplayError("Face observation clock must be ordered with exclusive coverage")
    previous = observed[0]
    points: list[TrackingPosition] = []
    gaps: list[MissingFaceRange] = []
    for index, sample in enumerate(series.samples):
        if sample.position is not None:
            previous = sample.position
            points.append(previous)
            continue
        if sample.reason is None:
            raise ReplayError("Missing face samples require their observed failure reason")
        end = (
            series.samples[index + 1].time
            if index + 1 < len(series.samples)
            else series.coverage_end
        )
        method: Literal["leading-observation-backfill", "last-known-hold"] = (
            "leading-observation-backfill" if sample.time < previous.time else "last-known-hold"
        )
        points.append(TrackingPosition(sample.time, previous.x, previous.y))
        gap = MissingFaceRange(sample.time, end, 1, previous.time, method, sample.reason)
        if (
            gaps
            and gaps[-1].end == gap.start
            and gaps[-1].derived_from == gap.derived_from
            and gaps[-1].method == gap.method
            and gaps[-1].reason == gap.reason
        ):
            gaps[-1] = replace(gaps[-1], end=end, frames=gaps[-1].frames + 1)
        else:
            gaps.append(gap)
    return FacePositionSeries(
        tuple(points), series.coverage_end, series.width, series.height, series.anchor, tuple(gaps)
    )


def merge_face_observations(
    windows: tuple[FaceObservationSeries, ...],
    *,
    seed_overlap: Literal["reject", "next"] = "reject",
) -> FaceObservationSeries:
    """Join raw windows before deriving positions, including wholly lost windows."""
    if not windows or any(len(window.samples) < 2 for window in windows):
        raise ReplayError("Expected bounded nonempty face observation windows")
    first = windows[0]
    previous_end = first.samples[0].time
    samples: list[FaceObservation] = []
    for window in windows:
        if (
            (window.width, window.height, window.anchor)
            != (first.width, first.height, first.anchor)
            or window.coverage_end <= window.samples[-1].time
            or any(b.time <= a.time for a, b in pairwise(window.samples))
        ):
            raise ReplayError("Face windows have incompatible geometry or frame clocks")
        if window.samples[0].time != previous_end:
            if (
                seed_overlap != "next"
                or not samples
                or window.samples[0].time != samples[-1].time
                or window.samples[1].time != previous_end
            ):
                raise ReplayError(
                    "Face observation windows have an unexplained clock gap or overlap"
                )
            samples.pop()
        samples.extend(window.samples)
        previous_end = window.coverage_end
    return FaceObservationSeries(
        tuple(samples), previous_end, first.width, first.height, first.anchor
    )


def face_position_series(
    observations: object,
    clock: VideoFrameClock,
    *,
    width: int,
    height: int,
    anchor: Literal["top", "bottom"] = "top",
    missing_policy: Literal["reject", "hold"] = "reject",
) -> FacePositionSeries:
    """Normalize actual face tops, never hair tops; ambiguous or lost faces fail."""
    return resolve_face_observations(
        face_observation_series(observations, clock, width=width, height=height, anchor=anchor),
        missing_policy=missing_policy,
    )


def read_verified_face_observations(
    config: Path, evidence: Path, *, anchor: Literal["top", "bottom"] = "top"
) -> FaceObservationSeries:
    verify(config, evidence)
    job = load_job(config)
    return face_observation_series(
        json.loads(_read(evidence / "response.json", RESPONSE_BYTES)),
        job.clock,
        width=job.width,
        height=job.height,
        anchor=anchor,
    )


def read_verified_face_positions(config: Path, evidence: Path) -> FacePositionSeries:
    """Reuse an existing native response only after its full domain provenance check."""
    verify(config, evidence)
    job = load_job(config)
    return face_position_series(
        json.loads(_read(evidence / "response.json", RESPONSE_BYTES)),
        job.clock,
        width=job.width,
        height=job.height,
    )


def _xml(raw: bytes, *, root_tag: str) -> ET.Element:
    if len(raw) > MAX_TRACK_BYTES:
        raise ReplayError("Tracking XML exceeds size limit")
    raw = raw.replace(b"<!DOCTYPE fcpxml>", b"")
    if re.search(rb"<!\s*(?:DOCTYPE|ENTITY)\b", raw, re.IGNORECASE):
        raise ReplayError("Tracking XML declarations are unsupported")
    try:
        root = ET.fromstring(raw)
    except ET.ParseError as error:
        raise ReplayError("Invalid tracking XML") from error
    if root.tag != root_tag:
        raise ReplayError("Unexpected tracking XML root")
    return root


def _time(value: str | None) -> Fraction:
    if value is None or not re.fullmatch(r"\d+(?:/[1-9]\d*)?s", value):
        raise ReplayError("Expected nonnegative rational FCP time")
    return Fraction(value[:-1])


def parse_tracking_positions(
    raw: bytes, *, frame_duration: Fraction
) -> tuple[TrackingPosition, ...]:
    """Read only the observed parameter-container/200/201 position channel."""
    if not 0 < frame_duration <= 1:
        raise ReplayError("Invalid tracking frame duration")
    root = _xml(raw, root_tag="parameter-container")
    channels = root.findall("param[@key='200']/param[@key='201']/keyframeAnimation")
    if len(channels) != 1:
        raise ReplayError("Expected exactly one observed tracker position channel")
    keys = channels[0].findall("keyframe")
    if not 2 <= len(keys) <= MAX_TRACK_SAMPLES:
        raise ReplayError("Expected bounded tracking samples")
    result: list[TrackingPosition] = []
    for key in keys:
        time = _time(key.get("time"))
        vector = (key.get("value") or "").split()
        try:
            values = tuple(float(value) for value in vector)
        except ValueError as error:
            raise ReplayError("Invalid tracker position vector") from error
        if len(values) != 2 or any(not math.isfinite(v) or abs(v) > MAX_COORDINATE for v in values):
            raise ReplayError("Tracker positions require two finite bounded coordinates")
        if time / frame_duration != int(time / frame_duration) or (
            result and time - result[-1].time != frame_duration
        ):
            raise ReplayError("Tracking samples must be consecutive on the explicit frame clock")
        result.append(TrackingPosition(time, values[0], values[1]))
    return tuple(result)


def read_native_tracker(
    bundle: Path, *, tracker_id: str, frame_duration: Fraction
) -> tuple[TrackingPosition, ...]:
    """Resolve one explicitly selected tracker and its observed, contained plist."""
    root = _xml((bundle / "Info.fcpxml").read_bytes(), root_tag="fcpxml")
    shapes = [node for node in root.findall(".//tracking-shape") if node.get("id") == tracker_id]
    if len(shapes) != 1:
        raise ReplayError("Tracker selection must identify exactly one tracking shape")
    locators = [
        node
        for node in root.findall("resources/locator")
        if node.get("id") == shapes[0].get("dataLocator")
    ]
    if len(locators) != 1:
        raise ReplayError("Tracker locator is missing or ambiguous")
    relative = Path(locators[0].get("url") or "")
    if relative.is_absolute() or relative.parts[:3] != ("Contents", "Effects", "ObjectTracking"):
        raise ReplayError("Unsupported tracker locator path")
    path = (bundle / relative).resolve()
    if not path.is_relative_to(bundle.resolve()) or path.stat().st_size > MAX_TRACK_BYTES:
        raise ReplayError("Tracker plist escapes bundle or exceeds size limit")
    try:
        payload: object = plistlib.loads(path.read_bytes())
    except (plistlib.InvalidFileException, ValueError) as error:
        raise ReplayError("Invalid tracker plist") from error
    if (
        not isinstance(payload, dict)
        or payload.get("type") != "fcpxmld.data.effects.object-tracking"
        or payload.get("id") != path.stem
        or not isinstance(payload.get("data"), bytes)
    ):
        raise ReplayError("Unsupported native tracker plist envelope")
    return parse_tracking_positions(payload["data"], frame_duration=frame_duration)


def _sample_y(samples: tuple[TrackingPosition, ...], time: Fraction) -> float:
    if not samples[0].time <= time <= samples[-1].time:
        raise ReplayError("Caption or reference time lies outside observed tracking")
    index = bisect_left(samples, time, key=lambda point: point.time)
    point = samples[index]
    if point.time == time:
        return point.y
    previous = samples[index - 1]
    weight = float((time - previous.time) / (point.time - previous.time))
    return previous.y + (point.y - previous.y) * weight


def title_position_samples(
    title: ET.Element,
    samples: tuple[TrackingPosition, ...],
    *,
    calibration: TrackingCalibration,
    tracker_source_start: Fraction,
    observed_end: Fraction | None = None,
) -> tuple[tuple[str, str], ...]:
    """Project observed Y to existing cue periods, returning FCP time/vector text.

    Input title offset is source time in its immediate asset-clip parent. Tracker
    time zero corresponds to explicit tracker_source_start. Existing title
    positioning is not added: local-bottom calibration already includes geometry.
    Output X is always zero. This does not serialize an unobserved transform key.
    """
    values = (
        calibration.source_head_top,
        calibration.tracker_y_to_pixels,
        calibration.caption_local_bottom,
        calibration.gap,
        *calibration.roi,
    )
    if (
        len(samples) < 2
        or any(
            point.time < 0
            or any(not math.isfinite(v) or abs(v) > MAX_COORDINATE for v in (point.x, point.y))
            for point in samples
        )
        or any(following.time <= previous.time for previous, following in pairwise(samples))
        or (
            observed_end is None
            and any(
                following.time - previous.time != samples[1].time - samples[0].time
                for previous, following in pairwise(samples)
            )
        )
        or len(calibration.roi) != 4
        or any(
            isinstance(v, bool) or not math.isfinite(v) or abs(v) > MAX_COORDINATE for v in values
        )
        or not 0 < calibration.canvas_height <= MAX_COORDINATE
        or not 0 <= calibration.source_head_top <= calibration.canvas_height
        or calibration.tracker_y_to_pixels == 0
        or calibration.gap < 0
        or min(calibration.roi) < 0
        or min(calibration.roi[2:]) <= 0
        or tracker_source_start < 0
    ):
        raise ReplayError("Explicit finite measured caption calibration is required")
    start = _time(title.get("offset")) - tracker_source_start
    duration = _time(title.get("duration"))
    local_start = _time(title.get("start"))
    if title.tag != "title" or duration <= 0:
        raise ReplayError("Expected a positive-duration cue title")
    end = start + duration
    reference_y = _sample_y(samples, calibration.reference_time)
    frame_duration = samples[1].time - samples[0].time
    coverage_end = samples[-1].time + frame_duration if observed_end is None else observed_end
    if not samples[-1].time < coverage_end or end > coverage_end:
        raise ReplayError("Caption end lies outside observed tracking frame coverage")
    # A title's end is exclusive. The final observed frame covers its complete
    # frame interval; no invented position at the next, invisible frame is needed.
    last_key = min(end, samples[-1].time)
    times = sorted(
        {start, *(point.time for point in samples if start < point.time < last_key), last_key}
    )
    result: list[tuple[str, str]] = []
    for time in times:
        head_top = (
            calibration.source_head_top
            + (_sample_y(samples, time) - reference_y) * calibration.tracker_y_to_pixels
        )
        desired_bottom = head_top - calibration.gap
        if not 0 <= desired_bottom <= calibration.canvas_height:
            raise ReplayError("Observed caption bottom leaves the canvas")
        y = calibration.canvas_height / 2 - desired_bottom - calibration.caption_local_bottom
        result.append(
            (rational(local_start + time - start), f"0 {y / calibration.canvas_height * 100:.12g}")
        )
    return tuple(result)


def apply_tracker_to_titles(
    fcpxml: bytes,
    samples: tuple[TrackingPosition, ...],
    *,
    tracker_id: str | None,
    title_ref: str,
    calibration: TrackingCalibration,
    tracker_source_start: Fraction,
    observed_end: Fraction | None = None,
    covered_titles_only: bool = False,
    asset_ref: str | None = None,
    source_path: Path | None = None,
) -> bytes:
    """Replace selected titles' position using the native param name=position schema.

    Selection is restricted to direct titles in the one asset-clip owning the
    tracker. Existing position is replaced, never added; reveal, style and clocks
    are preserved. Nested timelines and time-remapped clips require another clock
    adapter and are refused. Output is a candidate for native import verification.
    """
    root = _xml(fcpxml, root_tag="fcpxml")
    if (tracker_id is None) == (asset_ref is None):
        raise ReplayError("Select exactly one tracker ID or asset reference")
    if asset_ref is not None and source_path is None:
        raise ReplayError("Asset selection requires verified source binding")
    owners = [
        clip
        for clip in root.findall(".//asset-clip")
        if (
            asset_ref is not None
            and clip.get("ref") == asset_ref
            and any(title.get("ref") == title_ref for title in clip.findall("title"))
        )
        or (
            tracker_id is not None
            and any(
                shape.get("id") == tracker_id
                for shape in clip.findall("object-tracker/tracking-shape")
            )
        )
    ]
    if len(owners) != 1 or owners[0].find("timeMap") is not None:
        raise ReplayError("Expected one tracker-owning asset-clip without time remapping")
    if source_path is not None:
        assets = [
            node
            for node in root.findall("resources/asset")
            if node.get("id") == owners[0].get("ref")
        ]
        if len(assets) != 1:
            raise ReplayError("Expected one source asset resource")
        representations = assets[0].findall("media-rep[@kind='original-media']")
        if len(representations) != 1:
            raise ReplayError("Expected one original-media source binding")
        url = urlsplit(representations[0].get("src") or "")
        if (
            url.scheme != "file"
            or url.netloc not in {"", "localhost"}
            or url.query
            or url.fragment
            or Path(unquote(url.path)).resolve(strict=True) != source_path.resolve(strict=True)
        ):
            raise ReplayError("FCP asset differs from verified face-tracking source")
    parents = {child: parent for parent in root.iter() for child in parent}
    ancestor = parents.get(owners[0])
    while ancestor is not None:
        if ancestor.tag in {"clip", "ref-clip", "mc-clip", "sync-clip", "asset-clip"} or (
            ancestor.find("timeMap") is not None
        ):
            raise ReplayError("Nested or time-remapped tracker timelines are unsupported")
        ancestor = parents.get(ancestor)
    titles = [title for title in owners[0].findall("title") if title.get("ref") == title_ref]
    if not titles:
        raise ReplayError("No direct caption titles match the explicit reference")
    applied = 0
    for title in titles:
        start = _time(title.get("offset")) - tracker_source_start
        end = start + _time(title.get("duration"))
        coverage_end = observed_end or samples[-1].time + samples[1].time - samples[0].time
        if covered_titles_only and (start < samples[0].time or end > coverage_end):
            owners[0].remove(title)
            continue
        points = title_position_samples(
            title,
            samples,
            calibration=calibration,
            tracker_source_start=tracker_source_start,
            observed_end=observed_end,
        )
        transforms = title.findall("adjust-transform")
        if len(transforms) > 1:
            raise ReplayError("Ambiguous caption transform")
        transform = transforms[0] if transforms else ET.SubElement(title, "adjust-transform")
        if (
            transform.get("scale", "1 1") != "1 1"
            or transform.get("rotation", "0") != "0"
            or transform.get("anchor", "0 0") != "0 0"
            or any(child.get("name") != "position" for child in transform)
        ):
            raise ReplayError("Caption transform requires identity scale, rotation and anchor")
        transform.attrib.pop("position", None)
        for previous in transform.findall("param[@name='position']"):
            transform.remove(previous)
        param = ET.SubElement(transform, "param", name="position")
        animation = ET.SubElement(param, "keyframeAnimation")
        for time, vector in points:
            ET.SubElement(animation, "keyframe", time=time, value=vector, curve="linear")
        applied += 1
    if applied == 0:
        raise ReplayError("No caption titles lie completely inside observed coverage")
    return ET.tostring(root, encoding="unicode", xml_declaration=True).encode("utf-8")


def main() -> None:
    """Write a new bounded bundle retaining selected tracker bytes and relative URL."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("bundle", type=Path)
    parser.add_argument("output", type=Path)
    selection = parser.add_mutually_exclusive_group(required=True)
    selection.add_argument("--tracker-id")
    selection.add_argument("--asset-ref")
    parser.add_argument("--title-ref", required=True)
    parser.add_argument("--frame-duration", type=_time, required=True)
    parser.add_argument("--tracker-source-start", type=_time, required=True)
    parser.add_argument("--reference-time", type=_time, required=True)
    parser.add_argument("--head-top", type=float, required=True)
    parser.add_argument("--y-to-pixels", type=float, required=True)
    parser.add_argument("--local-bottom", type=float, required=True)
    parser.add_argument("--gap", type=float, default=HEAD_CAPTION_GAP_PX)
    parser.add_argument("--canvas-height", type=int, required=True)
    parser.add_argument("--roi", type=float, nargs=4, required=True)
    parser.add_argument("--face-config", type=Path)
    parser.add_argument("--face-evidence", type=Path)
    parser.add_argument("--covered-titles-only", action="store_true")
    args = parser.parse_args()
    if (args.face_config is None) != (args.face_evidence is None):
        raise ReplayError("Face config and evidence must be supplied together")
    observed_end = None
    source_path = None
    if args.face_config is not None:
        series = read_verified_face_positions(args.face_config, args.face_evidence)
        if (
            args.y_to_pixels != 1
            or args.tracker_source_start != 0
            or args.canvas_height != series.height
        ):
            raise ReplayError(
                "Native face positions require source clocks and unscaled output pixels"
            )
        samples, observed_end = series.positions, series.coverage_end
        source_path = load_job(args.face_config).source
    else:
        if args.asset_ref is not None:
            raise ReplayError("Asset selection requires native face evidence")
        samples = read_native_tracker(
            args.bundle, tracker_id=args.tracker_id, frame_duration=args.frame_duration
        )
    input_xml = args.bundle if args.bundle.is_file() else args.bundle / "Info.fcpxml"
    raw = input_xml.read_bytes()
    root = _xml(raw, root_tag="fcpxml")
    locators = root.findall("resources/locator")
    if len(locators) > 1 or (args.face_config is None and len(locators) != 1):
        raise ReplayError("Bundle copy supports exactly one observed tracker locator")
    relative = Path(locators[0].get("url") or "") if locators else Path()
    calibration = TrackingCalibration(
        args.reference_time,
        args.head_top,
        args.y_to_pixels,
        args.local_bottom,
        args.gap,
        args.canvas_height,
        (args.roi[0], args.roi[1], args.roi[2], args.roi[3]),
    )
    result = apply_tracker_to_titles(
        raw,
        samples,
        tracker_id=args.tracker_id,
        title_ref=args.title_ref,
        calibration=calibration,
        tracker_source_start=args.tracker_source_start,
        observed_end=observed_end,
        covered_titles_only=args.covered_titles_only,
        asset_ref=args.asset_ref,
        source_path=source_path,
    )
    if args.face_config is not None:
        # The rejected FCP tracker was used only to select its owning clip.
        # Do not retain its unrelated observations in a native-face candidate.
        candidate = _xml(result, root_tag="fcpxml")
        resources = candidate.find("resources")
        if resources is None:
            raise ReplayError("Expected FCP resources")
        for locator in candidate.findall("resources/locator"):
            resources.remove(locator)
        for clip in candidate.findall(".//asset-clip"):
            for tracker in clip.findall("object-tracker"):
                if any(shape.get("id") == args.tracker_id for shape in tracker):
                    clip.remove(tracker)
        result = ET.tostring(candidate, encoding="unicode", xml_declaration=True).encode("utf-8")
    args.output.mkdir()
    if args.face_config is None:
        destination = args.output / relative
        destination.parent.mkdir(parents=True)
        with destination.open("xb") as handle:
            handle.write((args.bundle / relative).read_bytes())
    with (args.output / "Info.fcpxml").open("xb") as handle:
        handle.write(result)


if __name__ == "__main__":
    main()
