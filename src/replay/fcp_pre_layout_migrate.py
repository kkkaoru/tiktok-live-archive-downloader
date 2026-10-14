"""Migrate immutable caption geometry to a pinned pre-layout Motion template."""

import argparse
import copy
import math
from pathlib import Path
from xml.etree import ElementTree as ET

from replay.artifact_json import _json, _path, _pin
from replay.core import ReplayError
from replay.fcp_measured import CaptionAnchor
from replay.fcp_part_assembly import _save
from replay.fcp_pre_layout import apply_geometry, geometry_factor, unique_style_ids
from replay.fcp_project import CaptionTemplate, caption_template
from replay.fcpxml_chapters import MAX_XML_BYTES, MAX_XML_ELEMENTS
from replay.stages import file_digest


def migrate(
    root: ET.Element, template: CaptionTemplate, *, template_only_diagnostic: bool = False
) -> ET.Element:
    """Preserve text, clocks, media and placement; replace only measured scaling."""
    if not template.pre_layout_geometry:
        raise ReplayError("Migration requires a published pre-layout caption template")
    result = copy.deepcopy(root)
    titles = result.findall(".//title")
    if not titles:
        raise ReplayError("Expected caption titles for geometry migration")
    refs = {title.get("ref") for title in titles}
    for ref in refs:
        matches = [node for node in result.findall("./resources/effect") if node.get("id") == ref]
        if ref is None or len(matches) != 1:
            raise ReplayError("Caption effect resource is missing or ambiguous")
        matches[0].attrib.update(name=template.name, uid=template.uid, src=template.src)
    if template_only_diagnostic:
        return result
    for title in titles:
        texts = title.findall("text")
        transform = title.find("adjust-transform")
        if (
            len(texts) != 3
            or any(any(child.tag != "text-style" for child in node) for node in texts)
            or transform is None
        ):
            raise ReplayError("Expected three plain caption layers and an explicit transform")
        try:
            values = tuple(float(item) for item in transform.get("scale", "1 1").split())
        except ValueError as error:
            raise ReplayError("Invalid source caption scale") from error
        if len(values) != 2 or values[0] != values[1] or not math.isfinite(values[0]):
            raise ReplayError("Expected a uniform source caption scale")
        factor = (
            geometry_factor(title, template.pre_layout_geometry)
            if title.findall("text-style-def")
            or any(
                node.get("key") in {key for _, key, _ in template.pre_layout_geometry}
                for node in title.findall("param")
            )
            else values[0]
        )
        apply_geometry(title, template.pre_layout_geometry, factor)
        geometry_factor(title, template.pre_layout_geometry)
    unique_style_ids(result)
    return result


def run(config_path: Path, output: Path) -> None:
    output = _path(str(output))
    config = _json(config_path)
    if config.get("version") != 1:
        raise ReplayError("Expected pre-layout migration version one")
    source, template_path = _pin(config.get("sourceXML")), _pin(config.get("template"))
    dependencies = (
        config_path.resolve(),
        Path(__file__).resolve(),
        Path(__file__).with_name("fcp_pre_layout.py"),
        Path(__file__).with_name("fcp_project.py"),
        Path(__file__).with_name("motion_pre_layout.py"),
        Path(__file__).with_name("caption_style_defaults.py"),
    )
    inputs = {str(path): file_digest(path) for path in dependencies}
    if not 0 < source.stat().st_size <= MAX_XML_BYTES:
        raise ReplayError("Caption XML exceeds its bounded read budget")
    root = ET.fromstring(source.read_bytes())
    if sum(1 for _ in root.iter()) > MAX_XML_ELEMENTS:
        raise ReplayError("Caption XML exceeds its element budget")
    template = caption_template(template_path, anchor=CaptionAnchor.BOTTOM)
    mode = config.get("mode", "pre-layout-geometry")
    if not isinstance(mode, str) or mode not in {"pre-layout-geometry", "template-only-diagnostic"}:
        raise ReplayError("Unknown caption migration mode")
    result = migrate(root, template, template_only_diagnostic=mode == "template-only-diagnostic")
    name = config.get("name")
    projects = result.findall("./library/event/project")
    if not isinstance(name, str) or not name or len(projects) != 1:
        raise ReplayError("Expected one explicitly named candidate project")
    projects[0].set("name", name)
    receipt = output.with_suffix(".geometry-migration.json")
    if output.exists() or receipt.exists():
        raise ReplayError("Geometry migration outputs must be new")
    content = b'<?xml version="1.0" encoding="UTF-8"?>\n<!DOCTYPE fcpxml>\n' + ET.tostring(result)
    _pin(config.get("sourceXML"))
    _pin(config.get("template"))
    if any(file_digest(Path(path)) != digest for path, digest in inputs.items()):
        raise ReplayError("Geometry migration producer changed during preparation")
    with output.open("xb") as stream:
        stream.write(content)
    _save(
        receipt,
        {
            "version": 1,
            "candidateOnly": True,
            "nativeRenderVerified": False,
            "headNearVerified": False,
            "placementPolicy": "source-position-track-unchanged",
            "mode": mode,
            "sourceXML": config["sourceXML"],
            "template": config["template"],
            "outputXML": {"path": str(output.resolve()), "sha256": file_digest(output)},
            "inputs": inputs,
        },
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("config", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    run(args.config, args.output)


if __name__ == "__main__":
    main()
