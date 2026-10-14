"""Replace measured captions in a bounded clean XML window, reusing its position track."""

import argparse
import copy
import hashlib
import json
import math
from bisect import bisect_right
from fractions import Fraction
from pathlib import Path
from xml.etree import ElementTree as ET

from replay.artifact_json import _json, _path, _pin
from replay.caption_scale import CaptionScale
from replay.core import ReplayError
from replay.fcp_caption_retarget import Cue, RetainedSpan, read_reflow
from replay.fcp_caption_srt import _time, caption_cues
from replay.fcp_measured import CaptionAnchor
from replay.fcp_output_window import window
from replay.fcp_partition import _merged, _required, _stamp
from replay.fcp_pre_layout import (
    GeometryValues,
    apply_geometry,
    require_production_geometry,
    unique_style_ids,
)
from replay.fcp_project import caption_template, frame_span
from replay.recordings import record
from replay.stages import file_digest

PositionTrack = tuple[tuple[Fraction, float, float], ...]
TITLE_START = Fraction(3600)


def _track(titles: list[ET.Element]) -> PositionTrack:
    points: dict[Fraction, tuple[float, float]] = {}
    for title in titles:
        if any(
            child.tag not in {"param", "text", "text-style-def", "adjust-transform"}
            for child in title
        ):
            raise ReplayError("Unsupported source title structure")
        offset, start = _time(title.attrib, "offset"), _time(title.attrib, "start")
        frames = title.findall(
            "./adjust-transform/param[@name='position']/keyframeAnimation/keyframe"
        )
        if not frames:
            raise ReplayError("Source title has no reusable position keyframes")
        for frame in frames:
            values = frame.get("value", "").split()
            if len(values) != 2 or frame.get("curve") != "linear":
                raise ReplayError("Expected linear two-dimensional title positions")
            x, y = float(values[0]), float(values[1])
            if not math.isfinite(x) or not math.isfinite(y) or x != 0:
                raise ReplayError("Position track must be finite and horizontally centered")
            time = offset + _time(frame.attrib, "time") - start
            previous = points.get(time)
            if previous is not None and not math.isclose(previous[1], y, abs_tol=1e-8):
                raise ReplayError("Existing title positions conflict at the same source time")
            points[time] = (x, y)
    return tuple((time, *vector) for time, vector in sorted(points.items()))


def _position(track: PositionTrack, time: Fraction) -> tuple[float, float]:
    if not track or not track[0][0] <= time <= track[-1][0]:
        raise ReplayError("New caption exceeds the existing position track; observations required")
    index = bisect_right(tuple(point[0] for point in track), time) - 1
    left = track[index]
    if left[0] == time:
        return left[1], left[2]
    right = track[index + 1]
    ratio = float((time - left[0]) / (right[0] - left[0]))
    return left[1] + ratio * (right[1] - left[1]), left[2] + ratio * (right[2] - left[2])


def _title(
    prototype: ET.Element,
    track: PositionTrack,
    *,
    start: Fraction,
    end: Fraction,
    text: str,
    scale: CaptionScale | None = None,
    pre_layout_geometry: GeometryValues = (),
) -> ET.Element:
    result = copy.deepcopy(prototype)
    result.set("offset", _stamp(start))
    result.set("start", _stamp(TITLE_START))
    result.set("duration", _stamp(end - start))
    nodes = result.findall("text")
    definitions = {item.get("id") for item in result.findall("text-style-def")}
    if len(nodes) != 3 or any(
        any(
            child.tag != "text-style" or not child.get("ref") or child.get("ref") not in definitions
            for child in node
        )
        for node in nodes
    ):
        raise ReplayError("Expected three plain template text layers")
    for node in nodes:
        node.clear()
        node.text = text
    transform = _required(result, "./adjust-transform")
    transform.clear()
    factor = 1.0 if scale is None else scale.factor
    require_production_geometry(pre_layout_geometry, factor)
    apply_geometry(result, pre_layout_geometry, factor)
    animation = ET.SubElement(
        ET.SubElement(transform, "param", name="position"), "keyframeAnimation"
    )
    samples = (start, *(time for time, _, _ in track if start < time < end), end)
    for time in samples:
        x, y = _position(track, time)
        ET.SubElement(
            animation,
            "keyframe",
            time=_stamp(TITLE_START + time - start),
            value=f"{x:.12g} {y:.12g}",
            curve="linear",
        )
    return result


def replace_captions(
    root: ET.Element,
    cues: tuple[Cue, ...],
    *,
    global_start: Fraction,
    rate: int,
    pre_layout_geometry: GeometryValues = (),
) -> ET.Element:
    """Replace title text/clock only; old stage boundaries never select new text."""
    result = copy.deepcopy(root)
    spine = _required(result, "./library/event/project/sequence/spine")
    for clip in spine:
        clip_start = _time(clip.attrib, "start")
        output_start = global_start + _time(clip.attrib, "offset")
        output_end = output_start + _time(clip.attrib, "duration")
        titles = clip.findall("title")
        applicable = [
            (span, cue.text, cue.scale)
            for cue in cues
            if (span := frame_span(max(cue.start, output_start), min(cue.end, output_end), rate))
            is not None
        ]
        if applicable and not titles:
            raise ReplayError("New caption has no existing placement evidence in its source clip")
        track = _track(titles)
        insertion = list(clip).index(titles[0]) if titles else 0
        for title in titles:
            clip.remove(title)
        for span, text, scale in applicable:
            title = _title(
                titles[0],
                track,
                start=clip_start + span[0] - output_start,
                end=clip_start + span[1] - output_start,
                text=text,
                scale=scale,
                pre_layout_geometry=pre_layout_geometry,
            )
            title.set("name", f"Measured caption {_stamp(span[0])}")
            clip.insert(insertion, title)
            insertion += 1
    unique_style_ids(result)
    return result


def run(config_path: Path, output: Path) -> None:
    config = _json(config_path)
    if config.get("version") != 1:
        raise ReplayError("Expected caption window configuration version one")
    semantic = _pin(config.get("semantic"))
    lineage_path = _pin(config.get("lineage"))
    lineage = _json(lineage_path)
    if (
        lineage.get("outputXML") != config.get("semantic")
        or lineage.get("captionClockEqual") is not True
    ):
        raise ReplayError("Caption window must bind its verified clean semantic lineage")
    reflow = _pin(config.get("reflow"))
    source = record(config.get("recognitionSource"))
    source_path = _path(source.get("path"))
    source_sha = source.get("sha256")
    if not isinstance(source_sha, str) or len(source_sha) != 64:
        raise ReplayError("Expected the existing verified recognition source identity")
    start = Fraction(str(config.get("start")))
    end = Fraction(str(config.get("end")))
    caption_cues(semantic)
    root = window(ET.parse(semantic).getroot(), start, end)
    sequence = _required(root, "./library/event/project/sequence")
    formats = [
        item
        for item in root.findall("./resources/format")
        if item.get("id") == sequence.get("format")
    ]
    if len(formats) != 1:
        raise ReplayError("Expected one frame format")
    rate = 1 / _time(formats[0].attrib, "frameDuration")
    if rate.denominator != 1:
        raise ReplayError("Caption window requires an integer source frame rate")
    cues = read_reflow(
        reflow,
        (RetainedSpan(Fraction(0), start, Fraction(0), end - start),),
        source=source_path,
        source_sha256=source_sha,
    )
    template_path = _pin(config["template"]) if "template" in config else None
    geometry: GeometryValues = ()
    if template_path is not None:
        template = caption_template(template_path, anchor=CaptionAnchor.BOTTOM)
        geometry = template.pre_layout_geometry
        refs = {title.get("ref") for title in root.findall(".//asset-clip/title")}
        for ref in refs:
            effects = [item for item in root.findall("./resources/effect") if item.get("id") == ref]
            if len(effects) != 1:
                raise ReplayError("Caption template resource is missing or ambiguous")
            effects[0].attrib.update(name=template.name, uid=template.uid, src=template.src)
    patched = replace_captions(
        root, cues, global_start=start, rate=rate.numerator, pre_layout_geometry=geometry
    )
    name = config.get("name")
    if not isinstance(name, str) or not name:
        raise ReplayError("Expected a candidate project name")
    _required(patched, "./library/event/project").set("name", name)
    content = ET.tostring(patched, encoding="utf-8", xml_declaration=True)
    output = _path(str(output))
    receipt_path = output.with_suffix(".caption-window.json")
    if output.exists() or receipt_path.exists():
        raise ReplayError("Caption window output already exists")
    with output.open("xb") as handle:
        handle.write(content)
    expected = [
        (span[0] - start, span[1] - start, cue.text)
        for cue in cues
        if (span := frame_span(max(start, cue.start), min(end, cue.end), rate.numerator))
        is not None
    ]
    if caption_cues(output) != _merged(expected):
        raise ReplayError("New caption text or exact output clock differs from measured reflow")
    receipt = {
        "version": 1,
        "candidateOnly": True,
        "nativeRenderVerified": False,
        "sourceCaptionQualityAccepted": False,
        "placementPolicy": "reuse-existing-source-clock-position-track",
        "mediaRehashed": False,
        "start": str(start),
        "end": str(end),
        "inputs": {
            str(path): file_digest(path)
            for path in (
                config_path,
                semantic,
                lineage_path,
                reflow,
                *((template_path,) if template_path is not None else ()),
            )
        },
        "outputXML": {"path": str(output), "sha256": hashlib.sha256(content).hexdigest()},
        "recognitionSource": source,
        "inheritedCleanLineage": str(lineage_path),
    }
    with receipt_path.open("x", encoding="utf-8") as handle:
        json.dump(receipt, handle, ensure_ascii=False, indent=2)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("config", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    run(args.config, args.output)


if __name__ == "__main__":
    main()
