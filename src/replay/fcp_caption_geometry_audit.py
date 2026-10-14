"""Read pinned caption geometry revisions without changing text, clocks or media."""

import argparse
import copy
import json
import math
import re
from fractions import Fraction
from pathlib import Path
from xml.etree import ElementTree as ET

from replay.artifact_json import _json, _path, _pin
from replay.core import ReplayError
from replay.fcp_caption_srt import _time, caption_cues
from replay.fcp_head_refresh import (
    MAX_REFRESH_TITLES,
    _without_positions,
    partition_impact,
    placement_changes,
)
from replay.foreground_evidence import _bound_bytes
from replay.stages import file_digest

MAX_XML_BYTES = 64 * 1024 * 1024


def _position_x(value: str | None) -> str:
    parts = (value or "").split()
    if len(parts) != 2 or any(
        len(part) > 64
        or re.fullmatch(r"[+-]?(?:[0-9]+(?:\.[0-9]*)?|\.[0-9]+)(?:[eE][+-]?[0-9]{1,3})?", part)
        is None
        or not math.isfinite(float(part))
        for part in parts
    ):
        raise ReplayError("Expected a bounded finite position vector")
    return str(Fraction(parts[0]))


def _horizontal(transform: ET.Element) -> bytes:
    """Ignore Y sampling while preserving X curves and unknown channel metadata."""
    result = ET.Element("horizontal")
    values: list[str] = []
    simple = True
    if "position" in transform.attrib:
        value = _position_x(transform.get("position"))
        values.append(value)
        result.set("position", value)
    params = transform.findall("param[@name='position']")
    if len(params) > 1:
        raise ReplayError("Expected one caption position channel")
    for source in params:
        param = copy.deepcopy(source)
        simple = param.attrib == {"name": "position"} and len(param) == 1
        if "value" in param.attrib:
            value = _position_x(param.get("value"))
            values.append(value)
            param.set("value", value)
        animations = param.findall("keyframeAnimation")
        simple = simple and len(animations) == 1 and not animations[0].attrib
        for animation in animations:
            keys = animation.findall("keyframe")
            simple = simple and bool(keys) and len(keys) == len(animation)
            previous: Fraction | None = None
            for key in keys:
                time = _time(key.attrib, "time")
                if previous is not None and time <= previous:
                    raise ReplayError("Expected increasing caption position keys")
                previous = time
                value = _position_x(key.get("value"))
                values.append(value)
                key.set("value", value)
                simple = (
                    simple
                    and key.attrib.keys() == {"time", "value", "curve"}
                    and key.get("curve") == "linear"
                    and not len(key)
                )
        result.append(param)
    if simple and values and len(set(values)) == 1:
        return ET.tostring(ET.Element("constant-x", {"value": values[0]}))
    return ET.tostring(result)


def _structure(root: ET.Element) -> tuple[bytes, tuple[bytes, ...]]:
    canonical = copy.deepcopy(root)
    projects = canonical.findall("./library/event/project")
    if len(projects) != 1:
        raise ReplayError("Expected one comparison project")
    projects[0].attrib.pop("name", None)
    horizontal = tuple(
        _horizontal(transform)
        for transform in canonical.findall(".//asset-clip/title/adjust-transform")
    )
    return _without_positions(canonical), horizontal


def _caption_rows(root: ET.Element, changes: list[dict[str, object]]) -> list[dict[str, object]]:
    titles = root.findall(".//asset-clip/title")
    if not 1 <= len(titles) <= MAX_REFRESH_TITLES or len(titles) != len(changes):
        raise ReplayError("Expected bounded matching caption geometry rows")
    rows: list[dict[str, object]] = []
    for title, change in zip(titles, changes, strict=True):
        text = "".join(title.findall("text")[0].itertext())
        lines = text.split("\n")
        if not 1 <= len(lines) <= 3 or any(not line.strip() for line in lines):
            raise ReplayError("Expected nonempty one-to-three caption rows")
        transform = title.find("adjust-transform")
        if transform is None:
            raise ReplayError("Expected existing caption transform")
        rows.append(
            {
                **change,
                "text": text,
                "lineCount": len(lines),
                "position": transform.get("position"),
                "keys": [
                    dict(key.attrib)
                    for key in transform.findall(
                        "./param[@name='position']/keyframeAnimation/keyframe"
                    )
                ],
                "outerScale": transform.get("scale", "1 1"),
            }
        )
    return rows


def audit(config_path: Path, output: Path) -> dict[str, object]:
    """Compare all title transforms and optionally bind reuse to an exact partition."""
    config = _json(config_path)
    if config.get("version") != 1:
        raise ReplayError("Expected geometry audit version 1")
    before, after = _pin(config.get("before")), _pin(config.get("after"))
    partition = None if config.get("partition") is None else _pin(config["partition"])
    code = tuple(
        Path(__file__).with_name(name)
        for name in ("fcp_caption_geometry_audit.py", "fcp_head_refresh.py", "fcp_caption_srt.py")
    )
    inputs = {str(p): file_digest(p) for p in (config_path, before, after, *code)}
    if partition is not None:
        inputs[str(partition)] = file_digest(partition)
    left = ET.fromstring(_bound_bytes(before, inputs[str(before)], MAX_XML_BYTES))
    right = ET.fromstring(_bound_bytes(after, inputs[str(after)], MAX_XML_BYTES))
    if caption_cues(before) != caption_cues(after) or _structure(left) != _structure(right):
        raise ReplayError("Caption audit requires unchanged text, clocks, media and non-Y geometry")
    changes = placement_changes(left, right)
    impact = (
        [] if partition is None else partition_impact(partition, before, changes, start=Fraction(0))
    )
    report: dict[str, object] = {
        "version": 1,
        "inputs": inputs,
        "captionMediaClockUnchanged": True,
        "changedTitles": sum(row["changed"] is True for row in changes),
        "captions": _caption_rows(right, changes),
        "partitionImpact": impact,
        "nativeRenderVerified": False,
        "headNearVerified": False,
        "candidateOnly": True,
    }
    if any(file_digest(Path(p)) != digest for p, digest in inputs.items()):
        raise ReplayError("Geometry audit input changed")
    destination = _path(str(output.absolute()))
    with destination.open("x", encoding="utf-8") as stream:
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
