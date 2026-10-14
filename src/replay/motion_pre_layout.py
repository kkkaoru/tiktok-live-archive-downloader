"""Publish synchronized style geometry before Motion paragraph layout.

The generated controls need native save/readback and render verification. They
are separate scalar controls consumed together by one scale policy, not a rig.
"""

import argparse
import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path
from xml.etree import ElementTree as ET

from replay import caption_style_defaults as defaults
from replay.core import ReplayError
from replay.motion_caption import MAX_XML_CHARACTERS, _attribute, _one, _parse
from replay.motion_entrance import validate_entrance_transform
from replay.motion_title_fcpxml import _local
from replay.stages import file_digest


@dataclass(frozen=True)
class GeometryControl:
    name: str
    key: str
    base: float
    object: str
    channel: str


LAYERS = (
    ("Caption Keyline Face", defaults.INNER_BLACK_STROKE_WIDTH),
    ("Caption Face", defaults.WHITE_STROKE_WIDTH),
    ("Caption Outer Rim", defaults.OUTER_BLACK_STROKE_WIDTH),
)


def _scalar(node: ET.Element, path: str) -> float:
    target = _one(node, path)
    try:
        value = float(_attribute(target, "value"))
    except ValueError as error:
        raise ReplayError("Invalid Motion geometry scalar") from error
    if list(target) or not math.isfinite(value) or value <= 0:
        raise ReplayError("Expected a positive static Motion geometry scalar")
    return value


def geometry_controls(document: str) -> tuple[GeometryControl, ...]:
    """Read observed style20 scalar controls without changing text or layout."""
    root = _parse(document)
    if root.get("version") != "5.15" or _one(root, "template/flags").text != "1":
        raise ReplayError("Expected observed Motion 5.15 title")
    layer = _one(root, "scene/layer")
    nodes = layer.findall("scenenode[@factoryID='16']")
    if len(nodes) != 3 or {node.get("name") for node in nodes} != {name for name, _ in LAYERS}:
        raise ReplayError("Expected three style20 text layers")
    controls: list[GeometryControl] = []
    font_size: float | None = None
    for name, stroke in LAYERS:
        node = _one(layer, f"scenenode[@name='{name}']")
        validate_entrance_transform(_one(node, "parameter[@id='1']/parameter[@id='100']"))
        layout = _one(node, "parameter[@id='2']")
        if (
            _one(layout, "parameter[@id='314']").get("value") != "1"
            or _one(layout, "parameter[@id='373']/parameter[@id='2']").get("value") != "2"
            or _one(layout, "parameter[@id='319']/parameter[@id='322']").get("value") != "0"
        ):
            raise ReplayError("Expected bottom paragraph and non-fading entrance")
        style = _one(node, "style")
        if _one(style, "parameter[@id='83']/font").text != defaults.FONT_POSTSCRIPT_NAME:
            raise ReplayError("Motion font differs from selected caption defaults")
        size = _scalar(style, "parameter[@id='3']")
        if font_size is not None and size != font_size:
            raise ReplayError("Text layers use different font sizes")
        font_size = size
        geometry = [
            ("Font Size", "3", size),
            ("Outline Width", "30/36", stroke * size / defaults.REFERENCE_FONT_SIZE / 2),
        ]
        if name == "Caption Outer Rim":
            geometry.extend(
                [
                    (
                        "Shadow Distance",
                        "21/27",
                        defaults.SHADOW_OFFSET_Y * size / defaults.REFERENCE_FONT_SIZE,
                    ),
                    (
                        "Shadow Blur X",
                        "21/75/1",
                        defaults.SHADOW_BLUR_STD_DEVIATION * size / defaults.REFERENCE_FONT_SIZE,
                    ),
                    (
                        "Shadow Blur Y",
                        "21/75/2",
                        defaults.SHADOW_BLUR_STD_DEVIATION * size / defaults.REFERENCE_FONT_SIZE,
                    ),
                ]
            )
        for label, channel, expected in geometry:
            path = "/".join(f"parameter[@id='{part}']" for part in channel.split("/"))
            base = _scalar(style, path)
            if not math.isclose(base, expected, rel_tol=1e-9, abs_tol=1e-9):
                raise ReplayError("Motion style geometry differs from caption defaults")
            style_channel = f"5/{_attribute(style, 'id')}/{channel}"
            controls.append(
                GeometryControl(
                    f"{name} {label}",
                    f"9999/{_attribute(layer, 'id')}/{_attribute(node, 'id')}/{style_channel}",
                    base,
                    _attribute(node, "id"),
                    f"./{style_channel}",
                )
            )
    return tuple(controls)


def publish_geometry(document: str) -> str:
    """Add observed published scalar targets; preserve all existing geometry."""
    controls = geometry_controls(document)
    root = _parse(document)
    published = _one(root, "scene/publishSettings")
    targets = published.findall("target")
    for control in controls:
        if any(
            target.get("name") == control.name
            or (target.get("object"), target.get("channel")) == (control.object, control.channel)
            for target in targets
        ):
            raise ReplayError("Motion geometry is already published")
        ET.SubElement(
            published, "target", object=control.object, channel=control.channel, name=control.name
        )
    return (
        '<?xml version="1.0" encoding="UTF-8"?>\n<!DOCTYPE ozxmlscene>\n'
        + ET.tostring(root, encoding="unicode")
        + "\n"
    )


def scaled_values(document: str, factor: float) -> tuple[tuple[str, str, float], ...]:
    """Return every synchronized scalar; outer FCP transform must remain identity."""
    if isinstance(factor, bool) or not math.isfinite(factor) or not 0 < factor <= 1:
        raise ReplayError("Expected a finite positive caption fit factor at most one")
    controls = geometry_controls(document)
    root = _parse(document)
    published = _one(root, "scene/publishSettings")
    for control in controls:
        if (
            len(
                [
                    target
                    for target in published.findall("target")
                    if target.attrib
                    == {"object": control.object, "channel": control.channel, "name": control.name}
                ]
            )
            != 1
        ):
            raise ReplayError("Missing or ambiguous published geometry control")
    return tuple((item.name, item.key, item.base * factor) for item in controls)


def group_controls(document: str, *, published: bool = True) -> tuple[GeometryControl, ...]:
    """Validate zero-origin caption group and its synchronized X/Y scale contract."""
    geometry_controls(document)
    root = _parse(document)
    layer = _one(root, "scene/layer")
    for node in layer.findall("scenenode[@factoryID='16']"):
        transform = _one(node, "parameter[@id='1']/parameter[@id='100']")
        if transform.find("parameter[@id='101']") is not None:
            raise ReplayError("Group fitting requires default-zero child position")
        bottom = node.find("parameter[@id='2']/parameter[@id='326']")
        if bottom is not None and (list(bottom) or bottom.get("value") not in {"0", "0.0"}):
            raise ReplayError("Group fitting requires zero paragraph bottom")
    controls = tuple(
        GeometryControl(
            f"Caption Group Scale {axis}",
            f"9999/{_attribute(layer, 'id')}/1/100/105/{identity}",
            1.0,
            _attribute(layer, "id"),
            f"./1/100/105/{identity}",
        )
        for axis, identity in (("X", "1"), ("Y", "2"))
    )
    info = _one(layer, "parameter[@id='1']")
    transforms = info.findall("parameter[@id='100']")
    if not published:
        if transforms:
            raise ReplayError("Expected default-zero untransformed caption group")
        return controls
    # Native Motion Save omits the complete default-identity transform while
    # preserving the published axis targets. Non-default transforms stay explicit.
    if transforms:
        transform = _one(info, "parameter[@id='100']")
        scale = _one(transform, "parameter[@id='105']")
        if len(transform) != 1 or len(scale) != 2:
            raise ReplayError("Unsupported caption group transform")
        for identity in ("1", "2"):
            if _scalar(scale, f"parameter[@id='{identity}']") != 1:
                raise ReplayError("Expected identity caption group scale")
    targets = _one(root, "scene/publishSettings").findall("target")
    for control in controls:
        matches = [target for target in targets if target.get("name") == control.name]
        if len(matches) != 1 or matches[0].attrib != {
            "name": control.name,
            "object": control.object,
            "channel": control.channel,
        }:
            raise ReplayError("Missing or ambiguous published group scale")
    return controls


def publish_group_scale(document: str) -> str:
    """Add group scale before canvas compositing, preserving every text subtree."""
    controls = group_controls(document, published=False)
    root = _parse(document)
    layer = _one(root, "scene/layer")
    targets = _one(root, "scene/publishSettings")
    if any(target.get("name") in {item.name for item in controls} for target in targets):
        raise ReplayError("Group scale is already published")
    transform = ET.SubElement(
        _one(layer, "parameter[@id='1']"),
        "parameter",
        name="Transform",
        id="100",
        flags="8589938704",
    )
    scale = ET.SubElement(transform, "parameter", name="Scale", id="105", flags="8589938704")
    for axis, identity, control in zip(("X", "Y"), ("1", "2"), controls, strict=True):
        ET.SubElement(
            scale, "parameter", name=axis, id=identity, flags="8606711824", default="1", value="1"
        )
        ET.SubElement(
            targets, "target", name=control.name, object=control.object, channel=control.channel
        )
    result = (
        '<?xml version="1.0" encoding="UTF-8"?>\n<!DOCTYPE ozxmlscene>\n'
        + ET.tostring(root, encoding="unicode")
        + "\n"
    )
    group_controls(result)
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("receipt", type=Path, nargs="?")
    parser.add_argument("--inspect", action="store_true")
    parser.add_argument("--group-scale", action="store_true")
    args = parser.parse_args()
    if args.inspect:
        if args.receipt is not None:
            raise ReplayError("Inspection takes only source and output receipt")
        source, output = map(_local, (args.source, args.output))
        if not 0 < source.stat().st_size <= MAX_XML_CHARACTERS:
            raise ReplayError("Motion source exceeds budget")
        document = source.read_text(encoding="utf-8")
        if args.group_scale:
            controls = group_controls(document)
        else:
            scaled_values(document, 1.0)
            controls = geometry_controls(document)
        with output.open("x", encoding="utf-8") as stream:
            json.dump(
                {
                    "version": 1,
                    "source": {"path": str(source), "sha256": file_digest(source)},
                    "controls": [asdict(item) for item in controls],
                    "renderVerified": False,
                },
                stream,
                indent=2,
            )
        return
    if args.receipt is None:
        raise ReplayError("Template preparation requires an output receipt")
    source, output, receipt = map(_local, (args.source, args.output, args.receipt))
    if output.exists() or receipt.exists() or len({source, output, receipt}) != 3:
        raise ReplayError("Expected distinct new Motion template and receipt paths")
    if not 0 < source.stat().st_size <= MAX_XML_CHARACTERS:
        raise ReplayError("Motion source exceeds budget")
    source_hash = file_digest(source)
    source_document = source.read_text(encoding="utf-8")
    document = (
        publish_group_scale(source_document)
        if args.group_scale
        else publish_geometry(source_document)
    )
    controls = group_controls(document) if args.group_scale else geometry_controls(document)
    with output.open("x", encoding="utf-8") as stream:
        stream.write(document)
    with receipt.open("x", encoding="utf-8") as stream:
        json.dump(
            {
                "version": 1,
                "source": {"path": str(source), "sha256": source_hash},
                "output": {"path": str(output), "sha256": file_digest(output)},
                "controls": [asdict(item) for item in controls],
                "nativeVerified": False,
                "renderVerified": False,
            },
            stream,
            indent=2,
        )


if __name__ == "__main__":
    main()
