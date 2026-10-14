"""Synchronize a copied Motion title's project ranges with an explicit frame clock.

This does not retime objects or certify FCP playback. Existing
scene duration and rate must already match the requested clock. Only stale
timeRange/playRange metadata is normalized; source and installed titles are
never overwritten. Explicit linear playback removes the natively verified
Project Loop End marker (type 7), preserves Poster Frame (type 8), and rejects
other marker types rather than guessing their effect on timing.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from xml.etree import ElementTree as ET

from replay.core import ReplayError
from replay.motion_caption import MAX_XML_CHARACTERS, _one, _parse

ROOT = Path(__file__).resolve().parents[2]
MAX_RATE = 60
MAX_SECONDS = 120
PROJECT_LOOP_END = "7"
POSTER_FRAME = "8"


def _linear_markers(scene: ET.Element) -> None:
    sets = scene.findall("timemarkerset")
    if not sets:
        return
    if len(sets) != 1:
        raise ReplayError("Expected at most one Motion marker set")
    markers = sets[0]
    for marker in list(markers):
        if marker.tag != "timemarker":
            raise ReplayError("Unsupported Motion marker structure")
        kind = _one(marker, "type").text
        if kind not in {PROJECT_LOOP_END, POSTER_FRAME}:
            raise ReplayError("Unverified Motion marker type prevents linear playback")
        if kind == PROJECT_LOOP_END:
            markers.remove(marker)


def normalize_ranges(text: str, *, frames: int, rate: int, linear_playback: bool = False) -> str:
    """Return a candidate with zero-origin project ranges, preserving all content.

    Accept only integral non-NTSC clocks and exact sceneSettings agreement, so a
    metadata repair cannot silently change the duration of caption animation.
    """
    if (
        type(frames) is not int
        or type(rate) is not int
        or not 1 <= rate <= MAX_RATE
        or not 1 <= frames <= rate * MAX_SECONDS
    ):
        raise ReplayError("Expected a bounded integral Motion frame clock")
    root = _parse(text)
    if _one(root, "template/flags").text != "1":
        raise ReplayError("Expected a Motion title template")
    scene = _one(root, "scene")
    settings = _one(scene, "sceneSettings")
    expected = {"duration": str(frames), "frameRate": str(rate), "NTSC": "0"}
    if any(_one(settings, key).text != value for key, value in expected.items()):
        raise ReplayError("Explicit frame clock differs from Motion scene settings")
    if type(linear_playback) is not bool:
        raise ReplayError("Linear playback must be an explicit boolean")
    if linear_playback:
        _linear_markers(scene)
    for name in ("timeRange", "playRange"):
        span = _one(scene, name)
        if set(span.attrib) != {"offset", "duration"} or list(span):
            raise ReplayError("Unsupported Motion project range structure")
        span.set("offset", "0 1 1 0")
        span.set("duration", f"{frames} {rate} 1 0")
    return (
        '<?xml version="1.0" encoding="UTF-8"?>\n<!DOCTYPE ozxmlscene>\n'
        + ET.tostring(root, encoding="unicode")
        + "\n"
    )


def _local(path: Path) -> Path:
    absolute = path.absolute()
    if absolute.resolve() != absolute or not absolute.is_relative_to(ROOT):
        raise ReplayError("Motion timing paths must stay in the unlinked checkout")
    if absolute.suffix != ".moti":
        raise ReplayError("Motion timing paths must use .moti")
    return absolute


def prepare(
    source: Path, output: Path, *, frames: int, rate: int, linear_playback: bool = False
) -> None:
    """Write only a new repository-local candidate, never modify the source."""
    source, output = _local(source), _local(output)
    if not source.is_file() or not 0 < source.stat().st_size <= MAX_XML_CHARACTERS:
        raise ReplayError("Expected a bounded Motion source file")
    result = normalize_ranges(
        source.read_text(encoding="utf-8"),
        frames=frames,
        rate=rate,
        linear_playback=linear_playback,
    )
    with output.open("x", encoding="utf-8") as stream:
        stream.write(result)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--frames", type=int, required=True)
    parser.add_argument("--rate", type=int, required=True)
    parser.add_argument("--linear-playback", action="store_true")
    args = parser.parse_args()
    prepare(
        args.source,
        args.output,
        frames=args.frames,
        rate=args.rate,
        linear_playback=args.linear_playback,
    )


if __name__ == "__main__":
    main()
