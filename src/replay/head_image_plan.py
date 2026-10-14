"""Plan bounded current upper-pixel measurements from verified source frame cells."""

import argparse
import json
import math
from dataclasses import dataclass
from fractions import Fraction
from itertools import pairwise
from pathlib import Path
from xml.etree import ElementTree as ET

from replay import tracking_observations as storage
from replay.artifact_json import _json, _pin
from replay.core import ReplayError
from replay.fcp_caption_retarget import face_references
from replay.fcp_caption_srt import _time
from replay.foreground_evidence import _bound_bytes
from replay.head_observations import TrackingFrame, tracking_frames
from replay.recordings import record
from replay.stages import file_digest
from replay.vertical_captions import Box

MAX_CELLS = 128
MAX_WINDOWS = 9
MAX_XML_BYTES = 64 * 1024 * 1024
CODE_FILES = (
    "head_image_plan.py",
    "artifact_json.py",
    "fcp_caption_retarget.py",
    "fcp_caption_srt.py",
    "foreground_evidence.py",
    "head_observations.py",
    "vertical_captions.py",
    "subtitle_clock.py",
    "tracking_observations.py",
    "recordings.py",
    "stages.py",
    "core.py",
)


@dataclass(frozen=True, slots=True)
class ImageFrameCell:
    start: Fraction
    end: Fraction
    face: tuple[int, int, int, int] | None

    def request_face(self) -> dict[str, object] | None:
        if self.face is None:
            return None
        x, y, width, height = self.face
        return {
            "presentation": {"value": self.start.numerator, "timescale": self.start.denominator},
            "x": x,
            "y": y,
            "width": width,
            "height": height,
            "minimumFaceCoverage": 0.5,
        }

    def encoded(self) -> dict[str, object]:
        return {"start": str(self.start), "end": str(self.end), "face": self.request_face()}


def image_cells(value: object, *, width: int, height: int) -> tuple[ImageFrameCell, ...]:
    """Read exact independent cells; missing faces are never synthesized or held."""
    if not isinstance(value, list) or not 1 <= len(value) <= MAX_CELLS:
        raise ReplayError("Current-image collection exceeds its frame-cell budget")
    result: list[ImageFrameCell] = []
    for item in value:
        raw = record(item)
        if set(raw) != {"start", "end", "face"}:
            raise ReplayError("Unexpected current-image frame-cell fields")
        start, end = _fraction(raw["start"]), _fraction(raw["end"])
        if not 0 <= start < end <= 86400 or end - start > 1:
            raise ReplayError("Invalid current-image source-frame cell")
        face = raw["face"]
        coordinates: tuple[int, int, int, int] | None = None
        if face is not None:
            data = record(face)
            numbers = tuple(data.get(key) for key in ("x", "y", "width", "height"))
            if (
                set(data) != {"presentation", "x", "y", "width", "height", "minimumFaceCoverage"}
                or any(type(number) is not int for number in numbers)
                or data["presentation"]
                != {"value": start.numerator, "timescale": start.denominator}
                or type(data["minimumFaceCoverage"]) is not float
                or data["minimumFaceCoverage"] != 0.5
            ):
                raise ReplayError("Current-image face must match its exact source-frame clock")
            x, y, w, h = (storage._integer(number) for number in numbers)
            if not 0 <= x < x + w <= width or not 0 <= y < y + h <= height:
                raise ReplayError("Current-image face exceeds source geometry")
            coordinates = (x, y, w, h)
        result.append(ImageFrameCell(start, end, coordinates))
    if any(left.end > right.start for left, right in pairwise(result)):
        raise ReplayError("Current-image frame cells must be ordered and independent")
    return tuple(result)


def _fraction(value: object) -> Fraction:
    if not isinstance(value, str) or len(value) > 32:
        raise ReplayError("Expected bounded exact source-frame clock text")
    try:
        result = Fraction(value)
    except (ValueError, ZeroDivisionError) as error:
        raise ReplayError("Invalid exact source-frame clock") from error
    if result.denominator > 1_000_000_000:
        raise ReplayError("Source-frame clock exceeds its timescale budget")
    return result


def caption_spans(root: ET.Element) -> tuple[tuple[Fraction, Fraction], ...]:
    """Keep title offsets in the direct source clock, across real source cuts."""
    clips = root.findall("./library/event/project/sequence/spine/asset-clip")
    if not clips or any(clip.find("timeMap") is not None for clip in clips):
        raise ReplayError("Current-image plan requires direct source-mapped clips")
    spans: list[tuple[Fraction, Fraction]] = []
    for clip in clips:
        start = _time(clip.attrib, "start")
        end = start + _time(clip.attrib, "duration")
        for title in clip.findall("title"):
            left = _time(title.attrib, "offset")
            right = left + _time(title.attrib, "duration")
            if not start <= left < right <= end:
                raise ReplayError("Current-image title exceeds its source clip")
            spans.append((left, right))
    if not spans or len(spans) > 512:
        raise ReplayError("Expected bounded caption-covered source spans")
    return tuple(spans)


def _roi(box: Box) -> tuple[int, int, int, int]:
    x, y = math.floor(box.left), math.floor(box.top)
    return x, y, math.ceil(box.right) - x, math.ceil(box.bottom) - y


def planned_cells(
    frames: tuple[TrackingFrame, ...],
    ends: tuple[Fraction, ...],
    spans: tuple[tuple[Fraction, Fraction], ...],
) -> tuple[ImageFrameCell, ...]:
    """Bind each intersecting actual frame to its raw same-frame single face."""
    if len(frames) != len(ends) or any(
        frame.time >= end for frame, end in zip(frames, ends, strict=True)
    ):
        raise ReplayError("Tracking frame-cell ends differ from the source clock")
    cells = tuple(
        ImageFrameCell(
            frame.time,
            end,
            _roi(frame.faces[0].box) if len(frame.faces) == 1 else None,
        )
        for frame, end in zip(frames, ends, strict=True)
        if any(frame.time < right and left < end for left, right in spans)
    )
    if not cells or len(cells) > MAX_CELLS:
        raise ReplayError("Caption-covered cells exceed the current-image frame budget")
    for left, right in spans:
        cursor = left
        for cell in cells:
            if cell.end <= cursor:
                continue
            if cell.start > cursor:
                break
            cursor = cell.end
            if cursor >= right:
                break
        if cursor < right:
            raise ReplayError("Caption source clock lacks complete actual frame-cell coverage")
    return cells


def plan(config: Path, output: Path) -> dict[str, object]:
    """Verify only intersecting windows and publish a pinned measurement config."""
    config = storage._path(str(config.absolute()))
    raw = _json(config)
    if (
        set(raw) != {"version", "xml", "faceIndex", "native", "sandbox"}
        or type(raw["version"]) is not int
        or raw["version"] != 1
    ):
        raise ReplayError("Invalid current-image planning configuration")
    xml, index, native, sandbox = (
        _pin(raw[key]) for key in ("xml", "faceIndex", "native", "sandbox")
    )
    inputs = {str(path): file_digest(path) for path in (config, xml, index, native, sandbox)}
    root = ET.fromstring(_bound_bytes(xml, inputs[str(xml)], MAX_XML_BYTES))
    spans = caption_spans(root)
    references = face_references(index)
    jobs = tuple((storage.load_job(cfg), evidence) for cfg, evidence in references)
    selected = [
        (job, evidence)
        for job, evidence in jobs
        if any(
            job.clock.pts[0] * job.clock.time_base < right
            and left < (job.clock.pts[-1] + job.clock.last_duration) * job.clock.time_base
            for left, right in spans
        )
    ]
    if not 1 <= len(selected) <= MAX_WINDOWS:
        raise ReplayError("Current-image plan exceeds its verified-window budget")
    first = selected[0][0]
    all_frames: dict[Fraction, tuple[TrackingFrame, Fraction]] = {}
    for job, evidence in selected:
        if (job.source, job.source_hash, job.width, job.height) != (
            first.source,
            first.source_hash,
            first.width,
            first.height,
        ):
            raise ReplayError("Current-image windows must share one pinned source and geometry")
        storage.verify(job.config, evidence)
        paths = (
            job.config,
            job.native,
            evidence / "request.json",
            evidence / "response.json",
            evidence / "provenance.json",
        )
        inputs.update({str(path): file_digest(path) for path in paths})
        frames = tracking_frames(
            json.loads(storage._read(evidence / "response.json", storage.RESPONSE_BYTES)),
            job.clock,
            width=job.width,
            height=job.height,
        )
        ends = (
            *tuple(frame.time for frame in frames[1:]),
            (job.clock.pts[-1] + job.clock.last_duration) * job.clock.time_base,
        )
        all_frames.update(
            (frame.time, (frame, end)) for frame, end in zip(frames, ends, strict=True)
        )
    ordered = sorted(all_frames.values(), key=lambda item: item[0].time)
    cells = planned_cells(
        tuple(item[0] for item in ordered), tuple(item[1] for item in ordered), spans
    )
    inputs[str(first.source)] = first.source_hash
    inputs.update(
        {
            str(path): file_digest(path)
            for path in (Path(__file__).with_name(name) for name in CODE_FILES)
        }
    )
    if any(file_digest(Path(path)) != digest for path, digest in inputs.items()):
        raise ReplayError("Current-image plan input changed during verification")
    result: dict[str, object] = {
        "version": 1,
        "source": str(first.source),
        "sourceSHA256": first.source_hash,
        "nativeExecutable": str(native),
        "nativeSHA256": inputs[str(native)],
        "sandbox": str(sandbox),
        "sandboxSHA256": inputs[str(sandbox)],
        "width": first.width,
        "height": first.height,
        "cells": [cell.encoded() for cell in cells],
        "inputs": inputs,
    }
    storage._write(storage._checked_output(output), result)
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("config", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    result = plan(args.config, args.output)
    cells = result["cells"]
    assert isinstance(cells, list)
    print(f"Planned {len(cells)} actual frame cells; placement remains unapproved")


if __name__ == "__main__":
    main()
