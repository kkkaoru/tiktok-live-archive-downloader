"""Reference bounded repository-local Motion title documents, never alpha movies.

The ~/Titles.localized UID and file-URL convention comes from FCP's own exports.
A repository-local title tree is a staging package, NOT an installed template.
FCP import/resource resolution and native playback must be checked separately.
Bind each Motion text object explicitly, in document order, without style overrides.
This matches the observed FCP instance structure, not proof of template resolution.
"""

from __future__ import annotations

import argparse
import json
from fractions import Fraction
from pathlib import Path
from xml.etree import ElementTree as ET

from replay.fcp_project import MOTION_TEMPLATES_DIRECTORY, TITLE_START, rational
from replay.foreground_evidence import _bound_bytes
from replay.motion_caption import (
    MAX_CAPTION_TEXT_UNITS,
    MAX_SEQUENCE_SENTENCES,
    MAX_XML_CHARACTERS,
    _parse,
)
from replay.motion_stages import _integer, _text
from replay.recordings import record

ROOT = Path(__file__).resolve().parents[2]


def _local(path: Path) -> Path:
    absolute = path.absolute()
    if absolute.resolve() != absolute or not absolute.is_relative_to(ROOT):
        raise ValueError("Motion title paths must stay in the unlinked checkout")
    return absolute


def _title_texts(root: ET.Element) -> tuple[str, ...]:
    nodes = root.findall("scene/layer/scenenode[@factoryID='16']")
    if not 1 <= len(nodes) <= 3 * MAX_SEQUENCE_SENTENCES:
        raise ValueError("Expected bounded flat Motion title text objects")
    texts: list[str] = []
    for node in nodes:
        bodies = node.findall("parameter[@id='2']/parameter[@id='369']/text")
        if len(bodies) != 1 or len(node.findall(".//text")) != 1:
            raise ValueError("Expected one Motion text body per title object")
        body = bodies[0]
        if list(body) or not body.text or len(body.text) > MAX_CAPTION_TEXT_UNITS:
            raise ValueError("Expected a bounded plain Motion title text body")
        texts.append(body.text)
    if len(root.findall(".//scenenode[@factoryID='16']")) != len(nodes):
        raise ValueError("Nested Motion title text objects are not supported")
    return tuple(texts)


def _reference(
    raw: object, *, canvas: tuple[int, int, int]
) -> tuple[Path, str, int, int, tuple[str, ...]]:
    item = record(raw)
    if set(item) != {"path", "sha256", "offsetFrame", "frames"}:
        raise ValueError("Invalid Motion title reference")
    path = _local(Path(_text(item["path"])))
    offset, frames = _integer(item["offsetFrame"]), _integer(item["frames"])
    if (
        path.suffix != ".moti"
        or not path.is_file()
        or not 0 < path.stat().st_size <= MAX_XML_CHARACTERS
        or MOTION_TEMPLATES_DIRECTORY not in path.parts
    ):
        raise ValueError("Expected a pinned Motion title in a staged template tree")
    relative = path.parts[path.parts.index(MOTION_TEMPLATES_DIRECTORY) + 1 :]
    if len(relative) < 4 or relative[0] != "Titles.localized":
        raise ValueError("Expected a title category, title directory and .moti file")
    root = _parse(_bound_bytes(path, _text(item["sha256"]), MAX_XML_CHARACTERS).decode("utf-8"))
    settings = root.find("scene/sceneSettings")
    if settings is None:
        raise ValueError("Motion title has no scene settings")
    expected = dict(zip(("width", "height", "frameRate"), map(str, canvas), strict=True))
    expected.update(duration=str(frames), NTSC="0")
    if (
        any(settings.findtext(key) != value for key, value in expected.items())
        or root.findtext("template/flags") != "1"
        or offset < 0
        or not 1 <= frames <= canvas[2] * 120
    ):
        raise ValueError("Motion title canvas, duration or title contract differs")
    return path, "~/" + "/".join(relative), offset, frames, _title_texts(root)


def build(plan: object) -> str:
    """Make a caption-only candidate timeline with exact document-length titles."""
    raw = record(plan)
    if set(raw) != {"version", "name", "width", "height", "rate", "frames", "titles"}:
        raise ValueError("Invalid Motion title timeline fields")
    width, height, rate, frames = (
        _integer(raw[key]) for key in ("width", "height", "rate", "frames")
    )
    name = _text(raw["name"])
    titles = raw["titles"]
    if (
        type(raw["version"]) is not int
        or raw["version"] != 1
        or not 1 <= min(width, height) <= max(width, height) <= 8192
        or not 1 <= rate <= 60
        or not 1 <= frames <= rate * 120
        or not name.strip()
        or not isinstance(titles, list)
        or not 1 <= len(titles) <= 32
    ):
        raise ValueError("Invalid bounded Motion title timeline")
    root = ET.Element("fcpxml", version="1.11")
    resources = ET.SubElement(root, "resources")
    ET.SubElement(
        resources,
        "format",
        id="r1",
        frameDuration=rational(Fraction(1, rate)),
        width=str(width),
        height=str(height),
        colorSpace="1-1-1 (Rec. 709)",
    )
    library = ET.SubElement(root, "library")
    event = ET.SubElement(library, "event", name=name)
    project = ET.SubElement(event, "project", name=name)
    duration = rational(Fraction(frames, rate))
    sequence = ET.SubElement(
        project, "sequence", format="r1", duration=duration, tcStart="0s", tcFormat="NDF"
    )
    spine = ET.SubElement(sequence, "spine")
    gap = ET.SubElement(
        spine, "gap", name="Caption-only candidate", offset="0s", start="0s", duration=duration
    )
    previous_end = 0
    for index, title in enumerate(titles):
        path, uid, offset, count, texts = _reference(title, canvas=(width, height, rate))
        if offset < previous_end or offset + count > frames:
            raise ValueError("Motion titles overlap or exceed the timeline")
        identifier = f"t{index}"
        ET.SubElement(
            resources, "effect", id=identifier, name=path.stem, uid=uid, src=path.as_uri()
        )
        instance = ET.SubElement(
            gap,
            "title",
            ref=identifier,
            name=path.stem,
            lane="1",
            offset=rational(Fraction(offset, rate)),
            start=rational(TITLE_START),
            duration=rational(Fraction(count, rate)),
        )
        for text in texts:
            ET.SubElement(instance, "text").text = text
        previous_end = offset + count
    ET.indent(root)
    return (
        '<?xml version="1.0" encoding="UTF-8"?>\n<!DOCTYPE fcpxml>\n'
        + ET.tostring(root, encoding="unicode")
        + "\n"
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("plan", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    plan, output = _local(args.plan), _local(args.output)
    if not plan.is_file() or not 0 < plan.stat().st_size <= MAX_XML_CHARACTERS:
        raise ValueError("Expected a bounded title timeline plan")
    result = build(json.loads(plan.read_text(encoding="utf-8")))
    with output.open("x", encoding="utf-8") as stream:
        stream.write(result)


if __name__ == "__main__":
    main()
