"""Apply observed Motion paragraph bottom alignment to a copied title project.

Geometry, text, appearance and reveal curves remain unchanged. This does not
convert a full-text Type On animation into incremental text states: callers must
already supply the visible prefix for each state. Native rendering is required
to establish the resulting baseline; XML controls alone are not that evidence.
"""

from __future__ import annotations

import argparse
import math
from pathlib import Path
from xml.etree import ElementTree as ET

from replay.core import ReplayError
from replay.motion_caption import MAX_XML_CHARACTERS, _one, _parameter, _parse

ENUM_FLAGS = "8590000144"
GROUP_FLAGS = "8589938704"
PARAGRAPH_ALIGNMENT_FLAGS = "8606777360"
MARGIN_FLAGS = "8606711824"
DISABLED_MARGIN_FLAGS = "8606744592"
MAX_PARAGRAPH_UNITS = 10_000


def _control(parent: ET.Element, *, identifier: str, flags: str, value: str | None) -> ET.Element:
    matches = parent.findall(f"parameter[@id='{identifier}']")
    if len(matches) > 1:
        raise ReplayError(f"Duplicate Motion layout control: {identifier}")
    if matches:
        node = matches[0]
        if node.get("flags") != flags or node.find("curve") is not None:
            raise ReplayError(f"Unsupported Motion layout control: {identifier}")
    else:
        node = ET.SubElement(parent, "parameter", id=identifier, flags=flags)
    if value is not None:
        node.set("value", value)
    return node


def _paragraph_box(controls: ET.Element, *, width: float, height: float, bottom: float) -> None:
    for identifier, value in (
        ("323", -width / 2),
        ("324", width / 2),
        ("325", bottom + height),
        ("326", bottom),
    ):
        existing = controls.findall(f"parameter[@id='{identifier}']")
        if len(existing) == 1 and existing[0].get("flags") == DISABLED_MARGIN_FLAGS:
            existing[0].set("flags", MARGIN_FLAGS)
        _control(controls, identifier=identifier, flags=MARGIN_FLAGS, value=str(float(value)))


def bottom_align_caption_text(
    template_xml: str,
    *,
    width: float,
    height: float,
    bottom: float = 0,
    anchor_y: float | None = None,
) -> str:
    """Center every text paragraph and bottom-align it, preserving local geometry.

    Uses Layout Method 314=1 and Alignment 373/Vertical 2=2 observed in a
    native Motion 6.3 save. Paragraph 401=1 is the existing centered-text schema.
    Explicit bounds use observed margin controls 323 through 326 in local Motion
    units. The box is centered on local X=0 and its lower edge is ``bottom``.
    Callers derive width from the safe canvas region and height from measured
    line height and line cap. Rejects animated controls and blank-line padding.
    An explicit anchor_y replaces static per-state Y offsets with one common
    origin. Animated Y must instead be preserved or moved to a parent by callers.
    """
    if (
        any(
            isinstance(value, bool) or not math.isfinite(value) for value in (width, height, bottom)
        )
        or not 0 < width <= MAX_PARAGRAPH_UNITS
        or not 0 < height <= MAX_PARAGRAPH_UNITS
        or abs(bottom) > MAX_PARAGRAPH_UNITS
        or abs(bottom + height) > MAX_PARAGRAPH_UNITS
    ):
        raise ReplayError("Paragraph bounds require positive finite bounded width and height")
    if anchor_y is not None and (
        isinstance(anchor_y, bool)
        or not math.isfinite(anchor_y)
        or abs(anchor_y) > MAX_PARAGRAPH_UNITS
    ):
        raise ReplayError("Caption anchor Y must be finite and bounded")
    root = _parse(template_xml)
    if root.get("version") not in {"5.13", "5.15"}:
        raise ReplayError("Only observed Motion ozml 5.13/5.15 text layouts are supported")
    nodes = root.findall(".//scenenode[@factoryID='16']")
    if not nodes:
        raise ReplayError("Motion project has no text nodes")
    for node in nodes:
        if anchor_y is not None:
            transform = _control(
                _parameter(node, "1"), identifier="100", flags=GROUP_FLAGS, value=None
            )
            position = _control(transform, identifier="101", flags=GROUP_FLAGS, value=None)
            _control(position, identifier="2", flags=MARGIN_FLAGS, value=str(float(anchor_y)))
        controls = _parameter(node, "2")
        text = _one(controls, "parameter[@id='369']/text").text
        if text is None or any(not line.strip() for line in text.split("\n")):
            raise ReplayError("Caption states must contain text without blank-line padding")
        paragraphs = node.findall("paragraphStyle")
        if not paragraphs:
            raise ReplayError("Motion text node has no paragraph style")
        for paragraph in paragraphs:
            _control(paragraph, identifier="401", flags=PARAGRAPH_ALIGNMENT_FLAGS, value="1")
        _control(controls, identifier="314", flags=ENUM_FLAGS, value="1")
        alignment = _control(controls, identifier="373", flags=GROUP_FLAGS, value=None)
        _control(alignment, identifier="2", flags=ENUM_FLAGS, value="2")
        _paragraph_box(controls, width=width, height=height, bottom=bottom)
        cache = _one(node, "paragraphMarginsCached")
        cache.attrib.clear()
        cache.set("cached", "0")
    result = ET.tostring(root, encoding="unicode")
    if len(result) > MAX_XML_CHARACTERS:
        raise ReplayError("Bottom-aligned Motion document exceeds the XML size limit")
    return result


def main() -> None:
    """Write a new candidate; never overwrite a source or existing destination."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("template", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument(
        "--width", type=float, required=True, help="Safe width in local Motion units"
    )
    parser.add_argument(
        "--height", type=float, required=True, help="Measured line height times line cap"
    )
    parser.add_argument(
        "--bottom", type=float, default=0, help="Local lower edge relative to text origin"
    )
    parser.add_argument("--anchor-y", type=float, help="Common static Y origin for all text states")
    args = parser.parse_args()
    result = bottom_align_caption_text(
        args.template.read_text(encoding="utf-8"),
        width=args.width,
        height=args.height,
        bottom=args.bottom,
        anchor_y=args.anchor_y,
    )
    with args.output.open("x", encoding="utf-8") as handle:
        handle.write(result)


if __name__ == "__main__":
    main()
