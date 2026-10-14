"""Trim a direct FCP output window while preserving connected source clocks."""

import argparse
import copy
from fractions import Fraction
from pathlib import Path
from xml.etree import ElementTree as ET
from xml.parsers import expat

from replay.caption_font_supplier import MAX_FCPXML_BYTES
from replay.core import ReplayError
from replay.fcp_caption_srt import Cue, _Collector, _time, caption_cues
from replay.fcp_partition import _clip_check, _merged, _required, _stamp
from replay.stages import file_digest

MAX_XML_BYTES = MAX_FCPXML_BYTES
MAX_XML_ELEMENTS = 2_000_000
TIMED_CHILDREN = frozenset({"asset-clip", "audio", "video", "title"})
STATIC_CHILDREN = frozenset(
    {
        "adjust-conform",
        "adjust-transform",
        "adjust-crop",
        "adjust-volume",
        "filter-video",
        "filter-audio",
        "metadata",
        "audio-channel-source",
    }
)


def _cues(root: ET.Element) -> tuple[Cue, ...]:
    if sum(1 for _ in root.iter()) > MAX_XML_ELEMENTS:
        raise ReplayError("FCP output window exceeds element budget")
    content = ET.tostring(root, encoding="utf-8")
    if not 0 < len(content) <= MAX_XML_BYTES:
        raise ReplayError("FCP output window exceeds byte budget")
    collector = _Collector()
    parser = expat.ParserCreate()
    parser.StartElementHandler = collector.start
    parser.CharacterDataHandler = collector.data
    parser.EndElementHandler = collector.end
    parser.Parse(content, True)
    ordered = sorted(collector.cues)
    if (
        collector.sequence_count != 1
        or collector.duration != collector.clip_end
        or not ordered
        or any(left[1] > right[0] for left, right in zip(ordered, ordered[1:], strict=False))
    ):
        raise ReplayError("Expected complete direct caption timeline")
    return _merged(ordered)


def _check_connected(clip: ET.Element) -> None:
    for child in clip:
        if child.tag == "audio-channel-source" and child.get("active") != "0":
            raise ReplayError("Only inactive source audio metadata is supported")
        if child.tag in STATIC_CHILDREN:
            if any(
                item.tag in TIMED_CHILDREN or item.tag in {"spine", "timeMap"}
                for item in child.iter()
            ):
                raise ReplayError("Unsupported nested timed content")
            continue
        if child.tag not in TIMED_CHILDREN:
            raise ReplayError(f"Unsupported primary clip child: {child.tag}")
        if any(
            item is not child and (item.tag in TIMED_CHILDREN or item.tag in {"spine", "timeMap"})
            for item in child.iter()
        ):
            raise ReplayError("Unsupported nested connected media")


def _trim_connected(clip: ET.Element, lower: Fraction, upper: Fraction) -> None:
    for child in list(clip):
        if child.tag in STATIC_CHILDREN:
            continue
        start = _time(child.attrib, "offset")
        end = start + _time(child.attrib, "duration")
        kept_start, kept_end = max(start, lower), min(end, upper)
        if kept_start >= kept_end:
            clip.remove(child)
            continue
        child.set("offset", _stamp(kept_start))
        child.set("duration", _stamp(kept_end - kept_start))
        # Advance the connected item's content clock, including Motion's usual
        # 3600-second title origin. Existing animation/keyframes remain intact.
        child.set("start", _stamp(_time(child.attrib, "start") + kept_start - start))


def window(root: ET.Element, start: Fraction, end: Fraction) -> ET.Element:
    """Return a copy; trim only the selected clock without resetting animation."""
    original_cues = _cues(root)
    sequence = _required(root, "./library/event/project/sequence")
    duration = _time(sequence.attrib, "duration")
    if (
        not isinstance(start, Fraction)
        or not isinstance(end, Fraction)
        or not 0 <= start < end <= duration
    ):
        raise ReplayError("Expected exact bounded output window")
    formats = [
        item
        for item in root.findall("./resources/format")
        if item.get("id") == sequence.get("format")
    ]
    if len(formats) != 1:
        raise ReplayError("Expected one referenced frame format")
    frame = _time(formats[0].attrib, "frameDuration")
    if frame <= 0 or any((value / frame).denominator != 1 for value in (start, end)):
        raise ReplayError("Output window must align exactly to video frames")
    result = copy.deepcopy(root)
    project = _required(result, "./library/event/project")
    project.attrib.pop("uid", None)
    target = _required(project, "./sequence")
    spine = _required(target, "./spine")
    for clip in list(spine):
        _clip_check(clip, frame=frame)
        _check_connected(clip)
        offset = _time(clip.attrib, "offset")
        clip_end = offset + _time(clip.attrib, "duration")
        kept_start, kept_end = max(start, offset), min(end, clip_end)
        if kept_start >= kept_end:
            spine.remove(clip)
            continue
        source_start = _time(clip.attrib, "start") + kept_start - offset
        source_end = source_start + kept_end - kept_start
        _trim_connected(clip, source_start, source_end)
        clip.set("offset", _stamp(kept_start - start))
        clip.set("start", _stamp(source_start))
        clip.set("duration", _stamp(kept_end - kept_start))
    target.set("duration", _stamp(end - start))
    expected = _merged(
        [
            (max(left, start) - start, min(right, end) - start, text)
            for left, right, text in original_cues
            if left < end and right > start
        ]
    )
    if _cues(result) != expected:
        raise ReplayError("Output window changed caption text or exact display clock")
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("start", type=Fraction)
    parser.add_argument("end", type=Fraction)
    args = parser.parse_args()
    digest = file_digest(args.source)
    caption_cues(args.source)
    result = window(ET.parse(args.source).getroot(), args.start, args.end)
    if file_digest(args.source) != digest:
        raise ReplayError("Source FCPXML changed during window extraction")
    with args.output.open("xb") as handle:
        handle.write(ET.tostring(result, encoding="utf-8", xml_declaration=True))


if __name__ == "__main__":
    main()
