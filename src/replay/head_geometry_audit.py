"""Audit exact source-face estimates and serialized caption Y without rendering."""

import argparse
import json
from bisect import bisect_right
from dataclasses import replace
from fractions import Fraction
from pathlib import Path
from xml.etree import ElementTree as ET

from replay.artifact_json import _json, _path, _pin
from replay.core import ReplayError
from replay.fcp_caption_retarget import face_references
from replay.fcp_caption_srt import _time
from replay.fcp_head_near import FaceScaleSeries, read_face_scale
from replay.fcp_head_quality import geometry_at
from replay.fcp_tracking_caption import TrackingCalibration, TrackingPosition, _sample_y
from replay.foreground_evidence import _bound_bytes
from replay.head_near_placement import FaceVerticalExtent, current_face_heights, place_near_head
from replay.head_reference import resolve_reference
from replay.stages import file_digest
from replay.tracking_observations import TrackingJob, load_job

MAX_XML_BYTES = 64 * 1024 * 1024
MAX_SAMPLES = 64
MAX_AUDIT_WINDOWS = 9


def caption_y(title: ET.Element, source_time: Fraction) -> float:
    """Read the serialized linear position at the matching Motion content clock."""
    transform = title.find("adjust-transform")
    if transform is None:
        raise ReplayError("Caption lacks a position transform")
    static = transform.get("position")
    keys = transform.findall("param[@name='position']/keyframeAnimation/keyframe")
    if bool(keys) == (static is not None):
        raise ReplayError("Expected exactly one caption position representation")
    if static is not None:
        return _vector_y(static)
    positions = tuple(
        TrackingPosition(_time(key.attrib, "time"), 0, _vector_y(key.get("value", "")))
        for key in keys
    )
    if any(a.time >= b.time for a, b in zip(positions, positions[1:], strict=False)):
        raise ReplayError("Expected ordered caption position keys")
    content_time = _time(title.attrib, "start") + source_time - _time(title.attrib, "offset")
    if not positions[0].time <= content_time <= positions[-1].time:
        raise ReplayError("Caption sample is outside serialized key coverage")
    return _sample_y(positions, content_time)


def _vector_y(value: str) -> float:
    parts = value.split()
    if len(parts) != 2:
        raise ReplayError("Expected a centered caption position vector")
    try:
        x, y = (float(part) for part in parts)
    except ValueError as error:
        raise ReplayError("Invalid caption position vector") from error
    if x != 0 or not -50 <= y <= 50:
        raise ReplayError("Expected a finite centered in-canvas caption anchor")
    return y


def sample_geometry(
    root: ET.Element,
    *,
    output_time: Fraction,
    faces: FaceScaleSeries,
    calibration: TrackingCalibration,
    reference_face: FaceVerticalExtent | None = None,
) -> dict[str, object]:
    """Keep frame mapping, raw geometry, derived placement and XML Y distinct."""
    clip, source_time = _source_at(root, output_time)
    titles = [
        title
        for title in clip.findall("title")
        if _time(title.attrib, "offset")
        <= source_time
        < _time(title.attrib, "offset") + _time(title.attrib, "duration")
    ]
    if len(titles) != 1:
        raise ReplayError("Expected exactly one caption at the sampled source clock")
    title = titles[0]
    texts = ["".join(text.itertext()) for text in title.findall("text")]
    if len(texts) != 3 or len(set(texts)) != 1 or not texts[0]:
        raise ReplayError("Expected matching three-layer caption text")
    quality = geometry_at(faces.tops, source_time)
    face = faces.extent(source_time)
    index = bisect_right(faces.sizes, source_time, key=lambda item: item.time) - 1
    if index < 0:
        raise ReplayError("Sample lacks paired source-face size evidence")
    size = faces.sizes[index]
    estimated_height = faces.estimated_height(source_time)
    reference = (
        faces.observed_reference(calibration.reference_time)
        if reference_face is None
        else reference_face
    )
    estimate = place_near_head(
        face=face,
        reference_face=reference,
        reference_head_top=calibration.source_head_top,
        caption_height=1,
        canvas_height=calibration.canvas_height,
        gap=calibration.gap,
        estimated_face_height=estimated_height,
    )
    y = caption_y(title, source_time)
    return {
        "outputTime": str(output_time),
        "sourceTime": str(source_time),
        "title": title.get("name"),
        "text": texts[0],
        "lineCount": len(texts[0].split("\n")),
        "observedFace": quality.observed,
        "heldAge": str(quality.age),
        "faceTop": face.top,
        "faceBottom": face.bottom,
        "faceHeightAtSample": face.bottom - face.top,
        "scaleObservationTime": str(size.time),
        "scaleObservationWidth": size.width,
        "scaleObservationHeight": size.height,
        "estimatedScaleHeight": estimated_height,
        "estimatedHeadTop": estimate.estimated_head_top if quality.observed else None,
        "estimatedUnfittedBottom": estimate.estimated_head_top - calibration.gap
        if quality.observed
        else None,
        "serializedY": y,
        "serializedBottom": (50 - y) * calibration.canvas_height / 100,
        "headNearVerified": False,
        "nativePaintVerified": False,
    }


def _source_at(root: ET.Element, output_time: Fraction) -> tuple[ET.Element, Fraction]:
    clips = [
        clip
        for clip in root.findall("./library/event/project/sequence/spine/asset-clip")
        if _time(clip.attrib, "offset")
        <= output_time
        < _time(clip.attrib, "offset") + _time(clip.attrib, "duration")
    ]
    if len(clips) != 1:
        raise ReplayError("Expected exactly one source clip at the sampled output clock")
    clip = clips[0]
    source_time = output_time - _time(clip.attrib, "offset") + _time(clip.attrib, "start")
    return clip, source_time


def _window_index(jobs: tuple[TrackingJob, ...], time: Fraction) -> int:
    matches = [
        index
        for index, job in enumerate(jobs)
        if job.clock.pts[0] * job.clock.time_base
        <= time
        < (job.clock.pts[-1] + job.clock.last_duration) * job.clock.time_base
    ]
    if not matches:
        raise ReplayError("Audit sample lacks verified tracking-window coverage")
    return matches[-1]


def _bounded_samples(
    root: ET.Element,
    *,
    refresh: dict[str, object],
    index: Path,
    times: tuple[Fraction, ...],
    pins: dict[str, str],
) -> list[dict[str, object]]:
    if "referenceTime" not in refresh:
        raise ReplayError("Bounded geometry audit requires an explicit observed reference")
    references = face_references(index)
    jobs = tuple(load_job(config) for config, _evidence in references)
    reference_index = _window_index(jobs, Fraction(str(refresh["referenceTime"])))
    sample_indices = tuple(_window_index(jobs, _source_at(root, time)[1]) for time in times)
    selected = sorted({reference_index, *sample_indices})
    if len(selected) > MAX_AUDIT_WINDOWS:
        raise ReplayError("Geometry audit exceeds its verified-window budget")
    series: dict[int, FaceScaleSeries] = {}
    for number in selected:
        config, evidence = references[number]
        pins.update(
            {
                str(path): file_digest(path)
                for path in (config, evidence / "response.json", evidence / "provenance.json")
            }
        )
        job, series[number] = read_face_scale(
            [references[number]], missing_policy="hold", seed_overlap="next"
        )
        if (job.source, job.source_hash) != (
            jobs[reference_index].source,
            jobs[reference_index].source_hash,
        ):
            raise ReplayError("Audit windows do not share one pinned source")
    calibration = resolve_reference(refresh, series[reference_index])
    reference = series[reference_index].observed_reference(calibration.reference_time)
    reference_size = next(
        size for size in series[reference_index].sizes if size.time == calibration.reference_time
    )
    rows: list[dict[str, object]] = []
    for time, number in zip(times, sample_indices, strict=True):
        faces = series[number]
        heights = current_face_heights(faces.sizes, reference=reference_size)
        faces = replace(
            faces,
            scale_heights=tuple(
                TrackingPosition(size.time, 0, height)
                for size, height in zip(faces.sizes, heights, strict=True)
            ),
        )
        rows.append(
            sample_geometry(
                root,
                output_time=time,
                faces=faces,
                calibration=calibration,
                reference_face=reference,
            )
        )
    return rows


def audit(config_path: Path, output: Path) -> dict[str, object]:
    config = _json(config_path)
    frames = config.get("frames")
    if (
        config.get("version") != 1
        or not isinstance(frames, list)
        or not 1 <= len(frames) <= MAX_SAMPLES
        or any(type(frame) is not int or frame < 0 for frame in frames)
        or len(set(frames)) != len(frames)
    ):
        raise ReplayError("Expected bounded distinct integer output-frame samples")
    refresh_path, xml = _pin(config.get("refresh")), _pin(config.get("xml"))
    refresh = _json(refresh_path)
    index = _pin(refresh.get("faceIndex"))
    code = tuple(
        Path(__file__).with_name(name)
        for name in (
            "head_geometry_audit.py",
            "fcp_head_near.py",
            "head_near_placement.py",
            "head_reference.py",
        )
    )
    pins = {str(path): file_digest(path) for path in (config_path, refresh_path, xml, index, *code)}
    root = ET.fromstring(_bound_bytes(xml, pins[str(xml)], MAX_XML_BYTES))
    sequence = root.find("./library/event/project/sequence")
    if sequence is None:
        raise ReplayError("Expected one source-mapped sequence")
    formats = root.findall(f"./resources/format[@id='{sequence.get('format')}']")
    if len(formats) != 1:
        raise ReplayError("Expected one referenced output frame format")
    duration = _time(formats[0].attrib, "frameDuration")
    if duration <= 0:
        raise ReplayError("Expected a positive output frame duration")
    samples = _bounded_samples(
        root,
        refresh=refresh,
        index=index,
        times=tuple(frame * duration for frame in frames),
        pins=pins,
    )
    report: dict[str, object] = {
        "version": 1,
        "inputs": pins,
        "candidateOnly": True,
        "samples": samples,
    }
    if any(file_digest(Path(path)) != digest for path, digest in pins.items()):
        raise ReplayError("Geometry audit input changed during collection")
    with _path(str(output.absolute())).open("x", encoding="utf-8") as stream:
        json.dump(report, stream, ensure_ascii=False, indent=2)
        stream.write("\n")
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("config", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    audit(args.config, args.output)


if __name__ == "__main__":
    main()
