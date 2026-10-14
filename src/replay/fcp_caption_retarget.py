"""Retarget existing measured ASR stages to observed face-relative bottom origins.

Reuses caption text, recognition clocks and explicit cuts. It never recognizes
speech. Optional scale-aware placement derives approximate head position from
paired face extents and an explicit reference, not hair detection.
Processed output-clock voice can be connected; generated
projects require separately verified local source-caption repair for delivery.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
from bisect import bisect_left, bisect_right
from collections.abc import Callable
from dataclasses import dataclass, replace
from fractions import Fraction
from math import ceil, floor, isfinite
from pathlib import Path
from typing import Literal
from xml.etree import ElementTree as ET

from replay import reflow_producer_evidence
from replay.caption_scale import CaptionScale, read_scale
from replay.caption_style_defaults import (
    HEAD_CAPTION_GAP_PX,
    HEAD_CAPTION_REFERENCE_HEIGHT,
    HEAD_CAPTION_Y_DEADBAND_PX,
    MAX_CAPTION_LINES,
    REFERENCE_FONT_SIZE,
)
from replay.caption_timing import display_time
from replay.core import ReplayError
from replay.delivery_captions import Cut as SourceCut
from replay.delivery_captions import read_cuts
from replay.fcp_caption_stability import stabilize_caption_y
from replay.fcp_head_near import FaceScaleSeries
from replay.fcp_head_near import title_samples as head_title_samples
from replay.fcp_measured import CaptionAnchor, _boundary_hold, read_measured_captions
from replay.fcp_pre_layout import geometry_factor, unique_style_ids
from replay.fcp_project import (
    Caption,
    CaptionStyle,
    CaptionTemplate,
    Cut,
    MediaInfo,
    Timeline,
    build_fcpxml,
    caption_template,
    probe_media,
    rational,
)
from replay.fcp_screen_stack import screen_stacks
from replay.fcp_tracking_caption import (
    FaceObservationSeries,
    FacePositionSeries,
    TrackingCalibration,
    TrackingPosition,
    _sample_y,
    merge_face_observations,
    read_verified_face_observations,
    read_verified_face_positions,
    resolve_face_observations,
    title_position_samples,
)
from replay.foreground_evidence import _bound_bytes
from replay.frame_cell_extent import validate_terminal_cell
from replay.motion_stages import _integer
from replay.recordings import record
from replay.stages import file_digest
from replay.tracking_observations import TrackingJob, load_job

ROOT = Path(__file__).resolve().parents[2]
MAX_JSON_BYTES = 64 * 1024 * 1024
CLIP_TRAILERS = frozenset(
    {
        "marker",
        "chapter-marker",
        "rating",
        "keyword",
        "analysis-marker",
        "audio-channel-source",
        "filter-video",
        "filter-video-mask",
        "filter-audio",
        "metadata",
    }
)


@dataclass(frozen=True, slots=True)
class RetainedSpan:
    """One selected source cut and its old/new output clocks."""

    source_start: Fraction
    original_start: Fraction
    output_start: Fraction
    duration: Fraction


@dataclass(frozen=True, slots=True)
class Cue:
    start: Fraction
    end: Fraction
    text: str
    scale: CaptionScale | None = None


def selected_spans(
    cuts: tuple[SourceCut, ...],
    *,
    rate: int,
    source_window: tuple[Fraction, Fraction] | None,
    output_window: tuple[Fraction, Fraction] | None = None,
) -> tuple[RetainedSpan, ...]:
    """Intersect only an explicitly requested source window, on whole frame cells."""
    if type(rate) is not int or not 1 <= rate <= 60 or not cuts:
        raise ReplayError("Expected bounded frame rate and retained cuts")
    if source_window is not None and not 0 <= source_window[0] < source_window[1]:
        raise ReplayError("Invalid selected source window")
    if output_window is not None and (
        source_window is not None or not 0 <= output_window[0] < output_window[1]
    ):
        raise ReplayError("Select one valid source or output window")
    spans: list[RetainedSpan] = []
    cursor = Fraction(0)
    for cut in cuts:
        start, end = cut.source_start, cut.source_end
        if source_window is not None:
            start = max(start, ceil(source_window[0] * rate))
            end = min(end, floor(source_window[1] * rate))
        if output_window is not None:
            start = max(start, cut.source_start + ceil(output_window[0] * rate) - cut.output_start)
            end = min(end, cut.source_start + floor(output_window[1] * rate) - cut.output_start)
        if end <= start:
            continue
        duration = Fraction(end - start, rate)
        spans.append(
            RetainedSpan(
                Fraction(start, rate),
                Fraction(cut.output_start + start - cut.source_start, rate),
                cursor,
                duration,
            )
        )
        cursor += duration
    if not spans:
        raise ReplayError("Selected source window contains no retained frames")
    return tuple(spans)


def project_cues(cues: tuple[Cue, ...], spans: tuple[RetainedSpan, ...]) -> tuple[Caption, ...]:
    """Split recognition stages only at retained-cut/window boundaries."""
    result: list[Caption] = []
    for span in spans:
        for cue in cues:
            start = max(cue.start, span.original_start)
            end = min(cue.end, span.original_start + span.duration)
            if end <= start:
                continue
            result.append(
                Caption(
                    span.output_start + start - span.original_start,
                    span.output_start + end - span.original_start,
                    540,
                    0,
                    tuple(cue.text.split("\n")),
                    "#FFFFFF",
                    motion_origin_y=0,
                    uniform_scale=1.0 if cue.scale is None else cue.scale.factor,
                )
            )
    if not result:
        raise ReplayError("Selected source window contains no recognized caption stages")
    return tuple(result)


def build(
    cues: tuple[Cue, ...],
    spans: tuple[RetainedSpan, ...],
    *,
    source: Path,
    media: MediaInfo,
    template: CaptionTemplate,
    faces: FacePositionSeries,
    calibration: TrackingCalibration,
    rate: int,
    name: str,
    voice: tuple[Path, MediaInfo] | None = None,
    scale_faces: FaceScaleSeries | None = None,
    measure_caption: Callable[[str], tuple[float, float]] | None = None,
    terminal_frame_cell: bool = False,
) -> bytes:
    """Build per-stage Motion titles and replace static Y with observed animation."""
    if (faces.width, faces.height) != (1080, 1920) or calibration.caption_local_bottom != 0:
        raise ReplayError("Expected calibrated 1080x1920 bottom-origin template geometry")
    if template.anchor is not CaptionAnchor.BOTTOM:
        raise ReplayError("Retargeting requires an explicitly verified bottom template")
    if (scale_faces is None) != (measure_caption is None) or (
        scale_faces is not None and faces != scale_faces.bottoms
    ):
        raise ReplayError("Scale-aware retargeting requires paired faces and actual text bounds")
    if any(gap.start <= calibration.reference_time < gap.end for gap in faces.derived_ranges):
        raise ReplayError("Caption calibration reference must be an observed face frame")
    if terminal_frame_cell and scale_faces is None:
        raise ReplayError("Terminal-cell placement requires explicit paired scale geometry")
    if any(span.source_start < faces.positions[0].time for span in spans):
        raise ReplayError("Selected cuts exceed observed face coverage")
    for span in spans:
        validate_terminal_cell(
            start=span.source_start,
            end=span.source_start + span.duration,
            physical_end=faces.coverage_end,
            frame_duration=Fraction(1, rate),
            allow_terminal_cell=terminal_frame_cell,
        )
    timeline = Timeline(
        rate,
        faces.width,
        faces.height,
        tuple(Cut(source, span.source_start, span.output_start, span.duration) for span in spans),
        (),
        project_cues(cues, spans),
        CaptionStyle(REFERENCE_FONT_SIZE, "#FFFFFF", 0, "#FFFFFF", 0, "#000000"),
    )
    xml, _ = build_fcpxml(
        timeline,
        name=name,
        sources={source: media},
        template=template,
        voice=voice,
        voice_offsets=None if voice is None else tuple(span.original_start for span in spans),
    )
    root = ET.fromstring(xml)
    _split_titles_at_cuts(root, spans)
    stacks = (
        screen_stacks(
            root,
            gaps=faces.derived_ranges,
            measured_height=lambda title: (
                measure_caption(title.findtext("text", ""))[1]
                * geometry_factor(title, template.pre_layout_geometry)
            ),
            canvas_height=faces.height,
        )
        if measure_caption is not None
        else {}
    )
    times = tuple(point.time for point in faces.positions)
    reference_y = _sample_y(faces.positions, calibration.reference_time)
    for title in root.findall(".//asset-clip/title"):
        start = Fraction(title.attrib["offset"][:-1])
        end = start + Fraction(title.attrib["duration"][:-1])
        left = max(0, bisect_right(times, start) - 1)
        right = min(len(times), bisect_left(times, end) + 1)
        left = max(0, min(left, right - 2))
        local = faces.positions[left:right]
        adjusted = replace(
            calibration,
            reference_time=local[0].time,
            source_head_top=calibration.source_head_top
            + (local[0].y - reference_y) * calibration.tracker_y_to_pixels,
        )
        if scale_faces is not None and measure_caption is not None:
            text = title.find("text")
            if text is None:
                raise ReplayError("Generated title lacks its caption text")
            width, height = measure_caption("".join(text.itertext()))
            scale = geometry_factor(title, template.pre_layout_geometry)
            width, height = width * scale, height * scale
            if not 0 < width <= faces.width:
                raise ReplayError("Measured caption width exceeds the canvas")
            points = head_title_samples(
                title,
                faces=scale_faces,
                calibration=calibration,
                caption_height=height,
                frame_duration=Fraction(1, rate),
                allow_terminal_cell=terminal_frame_cell,
                screen_stacks=stacks[title],
            )
        else:
            points = title_position_samples(
                title,
                local,
                calibration=adjusted,
                tracker_source_start=Fraction(0),
                observed_end=faces.coverage_end,
            )
        transform = title.find("adjust-transform")
        if transform is None:
            raise ReplayError("Generated title lacks its transform")
        transform.attrib.pop("position", None)
        animation = ET.SubElement(
            ET.SubElement(transform, "param", name="position"), "keyframeAnimation"
        )
        for time, vector in points:
            ET.SubElement(animation, "keyframe", time=time, value=vector, curve="linear")
    stabilize_caption_y(root)
    unique_style_ids(root)
    return ET.tostring(root, encoding="unicode", xml_declaration=True).encode("utf-8")


def merge_faces(
    series: tuple[FacePositionSeries, ...], *, seed_overlap: Literal["reject", "next"] = "reject"
) -> FacePositionSeries:
    """Join windows; replace one shared seed only under explicit next-window policy."""
    if not series or any(len(item.positions) < 2 for item in series):
        raise ReplayError("Expected nonempty observed face windows")
    if any(item.derived_ranges for item in series):
        raise ReplayError("Merge raw face observations before deriving candidate positions")
    first = series[0]
    previous_end = first.positions[0].time
    points: list[TrackingPosition] = []
    for item in series:
        if (item.width, item.height, item.anchor) != (first.width, first.height, first.anchor):
            raise ReplayError("Face windows must be consecutive on the same canvas")
        if item.positions[0].time != previous_end:
            if (
                seed_overlap != "next"
                or not points
                or item.positions[0].time != points[-1].time
                or item.positions[1].time != previous_end
            ):
                raise ReplayError("Face windows must be consecutive on the same canvas")
            points.pop()
        points.extend(item.positions)
        previous_end = item.coverage_end
    return FacePositionSeries(tuple(points), previous_end, first.width, first.height, first.anchor)


def read_face_windows(
    references: list[tuple[Path, Path]],
    *,
    seed_overlap: Literal["reject", "next"] = "reject",
    missing_policy: Literal["reject", "hold"] = "reject",
    face_anchor: Literal["top", "bottom"] = "top",
) -> tuple[TrackingJob, FacePositionSeries]:
    if not 1 <= len(references) <= 512:
        raise ReplayError("Expected bounded native face window references")
    first = load_job(references[0][0])
    series: list[FacePositionSeries] = []
    observations: list[FaceObservationSeries] = []
    for config, evidence in references:
        job = load_job(config)
        if job.source != first.source or job.source_hash != first.source_hash:
            raise ReplayError("Face windows do not share their pinned original")
        if missing_policy == "reject" and face_anchor == "top":
            series.append(read_verified_face_positions(config, evidence))
        else:
            observations.append(
                read_verified_face_observations(config, evidence, anchor=face_anchor)
            )
    if observations:
        return first, resolve_face_observations(
            merge_face_observations(tuple(observations), seed_overlap=seed_overlap),
            missing_policy=missing_policy,
        )
    return first, merge_faces(tuple(series), seed_overlap=seed_overlap)


def _split_titles_at_cuts(root: ET.Element, spans: tuple[RetainedSpan, ...]) -> None:
    """Undo cross-cut coalescing so one title never tracks across a source jump."""
    clips = root.findall("library/event/project/sequence/spine/asset-clip")
    pending: list[tuple[ET.Element, Fraction, Fraction]] = []
    for clip, span in zip(clips, spans, strict=True):
        for title in clip.findall("title"):
            start = span.output_start + Fraction(title.attrib["offset"][:-1]) - span.source_start
            pending.append((title, start, start + Fraction(title.attrib["duration"][:-1])))
            clip.remove(title)
    for title, start, end in pending:
        _append_title_pieces(title, start=start, end=end, clips=clips, spans=spans)


def _append_title_pieces(
    title: ET.Element,
    *,
    start: Fraction,
    end: Fraction,
    clips: list[ET.Element],
    spans: tuple[RetainedSpan, ...],
) -> None:
    for clip, span in zip(clips, spans, strict=True):
        left, right = max(start, span.output_start), min(end, span.output_start + span.duration)
        if left < right:
            piece = copy.deepcopy(title)
            piece.set("offset", rational(span.source_start + left - span.output_start))
            piece.set("duration", rational(right - left))
            insertion = next(
                (index for index, child in enumerate(clip) if child.tag in CLIP_TRAILERS), len(clip)
            )
            clip.insert(insertion, piece)


def read_cues(
    directory: Path, spans: tuple[RetainedSpan, ...], *, rate: int, source: Path, source_sha256: str
) -> tuple[Cue, ...]:
    """Read only relevant measured segments, preserving explicit offsets and holds."""
    index = record(json.loads((directory / "segments.json").read_text(encoding="utf-8")))
    entries = index.get("segments")
    if index.get("rate") != rate or not isinstance(entries, list):
        raise ReplayError("Caption segment frame clock differs")
    cues: list[Cue] = []
    earliest = min(span.original_start for span in spans)
    latest = max(span.original_start + span.duration for span in spans)
    for entry in entries:
        row = record(entry)
        offset = Fraction(_integer(row.get("offsetFrame")), rate)
        frames = _integer(row.get("frames"))
        if frames <= 0 or offset < 0:
            raise ReplayError("Invalid caption segment extent")
        if offset >= latest or offset + Fraction(frames, rate) <= earliest:
            continue
        path = directory / f"combined-{_integer(row.get('segment')):04d}.json"
        digest = file_digest(path)
        raw = record(json.loads(_bound_bytes(path, digest, MAX_JSON_BYTES)))
        if record(raw.get("inputs")).get(str(source)) != source_sha256:
            raise ReplayError("ASR candidate and native face evidence source differ")
        members = read_measured_captions(
            path, sha256=digest, width=1080, height=1920, rate=rate, duration=Fraction(frames, rate)
        )
        shifted = tuple(
            replace(
                item,
                cue=replace(
                    item.cue,
                    start=float(display_time(item.cue.start) + offset),
                    end=float(display_time(item.cue.end) + offset),
                ),
            )
            for item in members
        )
        if cues:
            previous = replace(
                shifted[0],
                cue=replace(
                    shifted[0].cue,
                    start=float(cues[-1].start),
                    end=float(cues[-1].end),
                    text=cues[-1].text,
                ),
            )
            trimmed = _boundary_hold(previous, shifted[0], record(raw.get("policy")))
            cues[-1] = replace(cues[-1], end=display_time(trimmed.cue.end))
        cues.extend(
            Cue(display_time(item.cue.start), display_time(item.cue.end), item.cue.text)
            for item in shifted
        )
    if not cues or any(b.start < a.end for a, b in zip(cues, cues[1:], strict=False)):
        raise ReplayError("Caption segments are empty or overlap beyond their declared hold")
    return tuple(cues)


def _seconds(value: object) -> Fraction:
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not isfinite(value):
        raise ReplayError("Reflow times and dimensions must be finite numbers")
    return display_time(float(value))


def read_reflow(
    path: Path,
    spans: tuple[RetainedSpan, ...],
    *,
    source: Path,
    source_sha256: str,
    producer_evidence: Path | None = None,
) -> tuple[Cue, ...]:
    """Consume verified horizontal layout only, never inherited caption origins."""
    raw = record(json.loads(_bound_bytes(path, file_digest(path), MAX_JSON_BYTES)))
    layout = record(raw.get("layout"))
    if (
        raw.get("version") != 1
        or raw.get("candidateOnly") is not True
        or layout.get("version") != 1
        or layout.get("scope") != "horizontal-phrase-layout-only"
        or layout.get("recognitionClocksChanged") is not False
        or layout.get("originsRequireRetargeting") is not True
        or layout.get("maximumLines") != MAX_CAPTION_LINES
        or layout.get("widthMeasurement")
        != "native-point-size-checked-glyph-bounds-including-border-rim"
        or layout.get("boundaries") != "pinned-vibrato-ipadic-phrase-boundaries"
        or _seconds(layout.get("pointSize")) <= 0
        or not 0 < _seconds(layout.get("paragraphWidth")) <= 1080
    ):
        raise ReplayError("Unsupported measured reflow contract")
    inputs = record(raw.get("inputs"))
    if inputs.get(str(source)) != source_sha256:
        raise ReplayError("Reflow and observed source differ")
    historical = (
        {}
        if producer_evidence is None
        else reflow_producer_evidence.read(producer_evidence, artifact=path, inputs=inputs)
    )
    for name, digest in inputs.items():
        if not isinstance(digest, str) or (
            name != str(source)
            and historical.get(name) != digest
            and file_digest(Path(name)) != digest
        ):
            raise ReplayError(f"Reflow input hash changed: {Path(name).name}")
    coverage = record(raw.get("coverage"))
    left, right = _seconds(coverage.get("start")), _seconds(coverage.get("end"))
    if not 0 <= left < right or any(
        span.original_start < left or span.original_start + span.duration > right for span in spans
    ):
        raise ReplayError("Requested output exceeds measured reflow coverage")
    rows = raw.get("captions")
    if not isinstance(rows, list) or len(rows) > 1_000_000:
        raise ReplayError("Expected bounded measured reflow captions")
    empty_spans = _empty_recognition_spans(raw.get("emptyRecognitionSpans", []), left, right)
    if not rows and (
        not empty_spans
        or empty_spans[0][0] != left
        or empty_spans[-1][1] != right
        or any(a[1] != b[0] for a, b in zip(empty_spans, empty_spans[1:], strict=False))
    ):
        raise ReplayError("Empty reflow requires complete zero-token coverage")
    cues: list[Cue] = []
    for item in rows:
        row = record(item)
        start, end, text = _seconds(row.get("start")), _seconds(row.get("end")), row.get("text")
        if (
            not isinstance(text, str)
            or not text
            or len(text) > 2048
            or "\0" in text
            or not 1 <= len(text.split("\n")) <= MAX_CAPTION_LINES
            or any(not line.strip() for line in text.split("\n"))
            or not left <= start < end <= right
            or (cues and start < cues[-1].end)
        ):
            raise ReplayError("Invalid measured reflow caption")
        if any(start < blank_end and blank_start < end for blank_start, blank_end in empty_spans):
            raise ReplayError("Caption overlaps a declared zero-token span")
        cues.append(Cue(start, end, text, read_scale(row)))
    return tuple(cues)


def _empty_recognition_spans(
    value: object, left: Fraction, right: Fraction
) -> tuple[tuple[Fraction, Fraction], ...]:
    """Recognized-token absence does not assert silence or speech accuracy."""
    if not isinstance(value, list) or len(value) > 512:
        raise ReplayError("Expected bounded empty recognition spans")
    result: list[tuple[Fraction, Fraction]] = []
    for entry in value:
        row = record(entry)
        start, end = _seconds(row.get("start")), _seconds(row.get("end"))
        if (
            row.get("classification") != "no-recognized-caption-tokens"
            or type(row.get("recognizedTokens")) is not int
            or row.get("recognizedTokens") != 0
            or type(row.get("measuredGlyphs")) is not int
            or row.get("measuredGlyphs") != 0
            or not left <= start < end <= right
            or (result and start < result[-1][1])
        ):
            raise ReplayError("Invalid empty recognition span evidence")
        result.append((start, end))
    return tuple(result)


def face_references(path: Path) -> list[tuple[Path, Path]]:
    """Read an ordered collector index; missing evidence still fails verification."""
    raw = record(json.loads(_bound_bytes(path, file_digest(path), MAX_JSON_BYTES)))
    rows = raw.get("windows")
    if not isinstance(rows, list) or not 1 <= len(rows) <= 512:
        raise ReplayError("Expected bounded face window index")
    result: list[tuple[Path, Path]] = []
    for index, value in enumerate(rows):
        row = record(value)
        config, evidence = row.get("config"), row.get("evidence")
        if (
            row.get("index") != index
            or not isinstance(config, str)
            or not isinstance(evidence, str)
        ):
            raise ReplayError("Face window index must contain ordered configuration/evidence paths")
        result.append((Path(config), Path(evidence)))
    return result


def tracking_receipt(
    faces: FacePositionSeries, spans: tuple[RetainedSpan, ...]
) -> dict[str, object]:
    ranges: list[dict[str, object]] = []
    for gap in faces.derived_ranges:
        projected: list[dict[str, str]] = []
        for span in spans:
            left = max(gap.start, span.source_start)
            right = min(gap.end, span.source_start + span.duration)
            if left < right:
                projected.append(
                    {
                        "start": str(span.output_start + left - span.source_start),
                        "end": str(span.output_start + right - span.source_start),
                    }
                )
        ranges.append(
            {
                "sourceStart": str(gap.start),
                "sourceEnd": str(gap.end),
                "frames": gap.frames,
                "derivedFromObservedTime": str(gap.derived_from),
                "method": gap.method,
                "reason": gap.reason,
                "outputRanges": projected,
            }
        )
    derived = sum(gap.frames for gap in faces.derived_ranges)
    return {
        "version": 1,
        "candidateOnly": True,
        "formalTrackingAccepted": False,
        "faceAnchor": faces.anchor,
        "observedFrames": len(faces.positions) - derived,
        "derivedFrames": derived,
        "unknownSourceRanges": ranges,
        "allCaptionStagesRetained": True,
        "captionYStabilization": {
            "method": "bounded-spatial-hysteresis-v1",
            "referencePixels": HEAD_CAPTION_Y_DEADBAND_PX,
            "referenceHeight": HEAD_CAPTION_REFERENCE_HEIGHT,
            "rawObservationsModified": False,
            "actualRenderVerified": False,
        },
        "limitation": "Derived holds/backfills are placement estimates, not face observations; "
        "face-box displacement is not direct hair-top tracking",
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("captions", type=Path)
    parser.add_argument("cuts", type=Path)
    parser.add_argument("template", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--face-config", type=Path)
    parser.add_argument("--face-evidence", type=Path)
    parser.add_argument("--face-index", type=Path)
    parser.add_argument("--face-anchor", choices=("top", "bottom"), default="top")
    parser.add_argument("--missing-face-policy", choices=("reject", "hold"), default="reject")
    parser.add_argument(
        "--check", action="store_true", help="Validate and build in memory without publication"
    )
    parser.add_argument(
        "--face-window",
        type=Path,
        nargs=2,
        action="append",
        help="Ordered native CONFIG EVIDENCE pairs for full-source coverage",
    )
    parser.add_argument("--media-source", type=Path, help="Byte-identical existing managed source")
    parser.add_argument("--source-window", type=Fraction, nargs=2)
    parser.add_argument(
        "--output-window",
        type=Fraction,
        nargs=2,
        help="Select an exact interval of the original cut output clock",
    )
    parser.add_argument(
        "--reflow", type=Path, help="Measured horizontal layout on original output clock"
    )
    parser.add_argument("--producer-evidence", type=Path)
    parser.add_argument("--seed-overlap", choices=("reject", "next"), default="reject")
    parser.add_argument("--voice", type=Path, help="Already cut, output-clock processed voice")
    parser.add_argument("--reference-time", type=Fraction, required=True)
    parser.add_argument("--head-top", type=float, required=True)
    parser.add_argument("--roi", type=float, nargs=4, required=True)
    parser.add_argument("--rate", type=int, default=30)
    parser.add_argument("--name", required=True)
    args = parser.parse_args()
    output = args.output.absolute()
    if output.resolve() != output or not output.is_relative_to(ROOT) or output.suffix != ".fcpxml":
        raise ReplayError("Candidate output must be an unlinked checkout FCPXML path")
    if output.exists():
        raise FileExistsError(output)
    if args.face_index is not None:
        if any(
            value is not None for value in (args.face_window, args.face_config, args.face_evidence)
        ):
            raise ReplayError("Choose one face window reference form")
        references = face_references(args.face_index)
    elif args.face_window is not None:
        if args.face_config is not None or args.face_evidence is not None:
            raise ReplayError("Choose one face window reference form")
        references = [(pair[0], pair[1]) for pair in args.face_window]
    elif args.face_config is not None and args.face_evidence is not None:
        references = [(args.face_config, args.face_evidence)]
    else:
        raise ReplayError("Native face configuration and evidence are required")
    if (
        args.source_window is None
        and args.output_window is None
        and (args.voice is None or args.reflow is None)
    ):
        raise ReplayError("Full-length candidates require measured reflow, processed voice")
    overlap: Literal["reject", "next"] = "next" if args.seed_overlap == "next" else "reject"
    missing: Literal["reject", "hold"] = "hold" if args.missing_face_policy == "hold" else "reject"
    anchor: Literal["top", "bottom"] = "bottom" if args.face_anchor == "bottom" else "top"
    receipt_path = output.with_suffix(".tracking.json")
    if (missing == "hold" or anchor == "bottom") and receipt_path.exists():
        raise FileExistsError(receipt_path)
    job, faces = read_face_windows(
        references, seed_overlap=overlap, missing_policy=missing, face_anchor=anchor
    )
    media_source = job.source if args.media_source is None else args.media_source.absolute()
    if media_source != job.source and file_digest(media_source) != job.source_hash:
        raise ReplayError("Managed source differs from the observed original")
    window = None if args.source_window is None else (args.source_window[0], args.source_window[1])
    output_window = (
        None if args.output_window is None else (args.output_window[0], args.output_window[1])
    )
    spans = selected_spans(
        read_cuts(args.cuts), rate=args.rate, source_window=window, output_window=output_window
    )
    cues = (
        read_cues(
            args.captions, spans, rate=args.rate, source=job.source, source_sha256=job.source_hash
        )
        if args.reflow is None
        else read_reflow(
            args.reflow,
            spans,
            source=job.source,
            source_sha256=job.source_hash,
            producer_evidence=args.producer_evidence,
        )
    )
    calibration = TrackingCalibration(
        args.reference_time, args.head_top, 1, 0, HEAD_CAPTION_GAP_PX, faces.height, tuple(args.roi)
    )
    result = build(
        cues,
        spans,
        source=media_source,
        media=probe_media(media_source),
        template=caption_template(args.template.absolute(), anchor=CaptionAnchor.BOTTOM),
        faces=faces,
        calibration=calibration,
        rate=args.rate,
        name=args.name,
        voice=None if args.voice is None else (args.voice.absolute(), probe_media(args.voice)),
    )
    digest = hashlib.sha256(result).hexdigest()
    if not args.check:
        if missing == "hold" or anchor == "bottom":
            receipt = tracking_receipt(faces, spans)
            receipt.update(
                {
                    "fcpxmlSHA256": digest,
                    "missingPolicy": missing,
                    "seedOverlapPolicy": overlap,
                    "inputs": {
                        str(path.resolve()): file_digest(path)
                        for config, evidence in references
                        for path in (
                            config,
                            evidence / "response.json",
                            evidence / "provenance.json",
                        )
                    },
                }
            )
            with receipt_path.open("x", encoding="utf-8") as stream:
                json.dump(receipt, stream, ensure_ascii=False)
        with output.open("xb") as stream:
            stream.write(result)
    print(
        json.dumps(
            {
                "candidateOnly": True,
                "output": str(output),
                "sha256": digest,
                "checkOnly": args.check,
                "retainedCuts": len(spans),
                "captionStages": len(ET.fromstring(result).findall(".//title")),
                "sourceWindow": None if window is None else [str(value) for value in window],
                "originalOutputWindow": None
                if output_window is None
                else [str(value) for value in output_window],
                "deliveryComplete": False,
                "processedVoice": args.voice is not None,
                "sourceCaptionRepairRequired": True,
                "seedOverlapPolicy": overlap,
                "missingFacePolicy": missing,
                "faceAnchor": anchor,
                "derivedPositionFrames": sum(gap.frames for gap in faces.derived_ranges),
            }
        )
    )


if __name__ == "__main__":
    main()
