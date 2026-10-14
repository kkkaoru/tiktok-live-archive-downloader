"""Copy a bottom-anchored title with a synchronized, non-fading Motion scale entrance.

Uses the scale channel saved by Motion 6.3 (ozml 5.15). This is an experimental
template adapter, not evidence that FCP preserves entrance timing on short titles.
"""

from __future__ import annotations

import argparse
import copy
import math
from fractions import Fraction
from pathlib import Path
from xml.etree import ElementTree as ET

from replay import caption_style_defaults as defaults
from replay.core import ReplayError
from replay.motion_caption import MAX_XML_CHARACTERS, _one, _parse, _project_duration, _reveal_curve
from replay.motion_template_timing import _local

GROUP_FLAGS = "8589938704"
ANIMATED_FLAGS = "8589934608"
STATIC_FLAGS = "8606711824"
VERTICAL_IDENTITY = "1.0"


def _curve() -> ET.Element:
    curve = _reveal_curve(
        (
            (Fraction(0), Fraction(str(defaults.MOTION_ENTRANCE_START_SCALE))),
            (
                Fraction(
                    defaults.MOTION_ENTRANCE_DURATION_NUMERATOR,
                    defaults.MOTION_ENTRANCE_DURATION_DENOMINATOR,
                ),
                Fraction(1),
            ),
        ),
        smooth=False,
    )
    # Scale has no Type On min/max bounds. Native Save retains default/value 1.
    curve.set("default", "1")
    curve.set("value", "1")
    curve.remove(_one(curve, "min"))
    curve.remove(_one(curve, "max"))
    return curve


def _same_curve(actual: ET.Element, expected: ET.Element) -> bool:
    if actual.tag != expected.tag or set(actual.attrib) != set(expected.attrib):
        return False
    if any(actual.get(key) != value for key, value in expected.attrib.items()):
        return False
    if len(actual) != len(expected):
        return False
    if list(actual):
        return all(_same_curve(left, right) for left, right in zip(actual, expected, strict=True))
    try:
        if actual.tag == "time":
            left = (actual.text or "").split()
            right = (expected.text or "").split()
            return (
                len(left) == 4
                and left[2:] == ["1", "0"]
                and Fraction(int(left[0]), int(left[1])) == Fraction(int(right[0]), int(right[1]))
            )
        return math.isclose(float(actual.text or ""), float(expected.text or ""), abs_tol=1e-12)
    except (ValueError, ZeroDivisionError):
        return False


def validate_entrance_transform(transform: ET.Element) -> None:
    """Accept horizontal entrance with invariant vertical scale and optional position.

    The caller validates position independently. Missing position means native
    default zero; unknown rotation, anchor or other scale axes remain unsupported.
    """
    children = transform.findall("parameter")
    if (
        {child.get("id") for child in children} not in ({"105"}, {"101", "105"})
        or len(children) != len({child.get("id") for child in children})
        or any(child.tag not in {"parameter", "foldFlags"} for child in transform)
    ):
        raise ReplayError("Unsupported Motion entrance transform")
    scale = _one(transform, "parameter[@id='105']")
    axes = scale.findall("parameter")
    if (
        scale.get("flags") != GROUP_FLAGS
        or any(child.tag not in {"parameter", "foldFlags"} for child in scale)
        or len(axes) != 2
        or [axis.attrib for axis in axes]
        != [
            {"name": "X", "id": "1", "flags": ANIMATED_FLAGS},
            {
                "name": "Y",
                "id": "2",
                "flags": STATIC_FLAGS,
                "default": "1",
                "value": VERTICAL_IDENTITY,
            },
        ]
    ):
        raise ReplayError("Motion entrance requires horizontal pop and fixed Y scale")
    if len(axes[0]) != 1 or not _same_curve(_one(axes[0], "curve"), _curve()) or len(axes[1]):
        raise ReplayError("Unsupported Motion entrance curve")


def animate_entrance(template_xml: str) -> str:
    """Add the shared entrance to all three text layers without changing style.

    Only default-zero origins and previously unscaled layers are accepted. Opacity,
    position, text and final size stay unchanged. Templates must disable Type On fading.
    """
    root = _parse(template_xml)
    if root.get("version") != "5.15" or _one(root, "template/flags").text != "1":
        raise ReplayError("Expected the observed ozml 5.15 Motion title")
    if _project_duration(root) <= Fraction(
        defaults.MOTION_ENTRANCE_DURATION_NUMERATOR,
        defaults.MOTION_ENTRANCE_DURATION_DENOMINATOR,
    ):
        raise ReplayError("Motion title is shorter than its entrance")
    nodes = root.findall("scene/layer/scenenode[@factoryID='16']")
    if len(nodes) != 3:
        raise ReplayError("Motion entrance requires three synchronized text layers")
    for node in nodes:
        properties = _one(node, "parameter[@id='1']")
        if properties.find("parameter[@id='100']") is not None:
            raise ReplayError("Motion entrance requires default-zero untransformed layers")
        controls = _one(node, "parameter[@id='2']")
        if (
            _one(controls, "parameter[@id='314']").get("value") != "1"
            or _one(controls, "parameter[@id='373']/parameter[@id='2']").get("value") != "2"
            or _one(controls, "parameter[@id='319']/parameter[@id='322']").get("value") != "0"
        ):
            raise ReplayError("Motion entrance requires bottom layout without fade")
        bottom = controls.find("parameter[@id='326']")
        if bottom is not None and bottom.get("value") not in {"0", "0.0"}:
            raise ReplayError("Motion entrance requires zero paragraph bottom")
        transform = ET.SubElement(
            properties, "parameter", name="Transform", id="100", flags=GROUP_FLAGS
        )
        scale = ET.SubElement(transform, "parameter", name="Scale", id="105", flags=GROUP_FLAGS)
        ET.SubElement(scale, "foldFlags").text = "11"
        axis = ET.SubElement(scale, "parameter", name="X", id="1", flags=ANIMATED_FLAGS)
        axis.append(copy.deepcopy(_curve()))
        ET.SubElement(
            scale,
            "parameter",
            name="Y",
            id="2",
            flags=STATIC_FLAGS,
            default="1",
            value=VERTICAL_IDENTITY,
        )
    return (
        '<?xml version="1.0" encoding="UTF-8"?>\n<!DOCTYPE ozxmlscene>\n'
        + ET.tostring(root, encoding="unicode")
        + "\n"
    )


def upgrade_entrance(template_xml: str) -> str:
    """Migrate only the previously observed uniform entrance, never other motion.

    Removing the complete validated old transform and using the normal builder
    preserves all unrelated text, paint, layout, opacity and published controls.
    """
    root = _parse(template_xml)
    nodes = root.findall("scene/layer/scenenode[@factoryID='16']")
    if len(nodes) != 3:
        raise ReplayError("Entrance migration requires three text layers")
    for node in nodes:
        properties = _one(node, "parameter[@id='1']")
        transform = _one(properties, "parameter[@id='100']")
        if len(transform) != 1 or transform[0].get("id") != "105":
            raise ReplayError("Migration refuses position, anchor or unknown transforms")
        scale = transform[0]
        axes = scale.findall("parameter")
        if (
            scale.get("flags") != GROUP_FLAGS
            or len(scale) != 3
            or scale[0].tag != "foldFlags"
            or scale[0].text != "11"
            or [axis.attrib for axis in axes]
            != [
                {"name": "X", "id": "1", "flags": ANIMATED_FLAGS},
                {"name": "Y", "id": "2", "flags": ANIMATED_FLAGS},
            ]
            or any(
                len(axis) != 1 or not _same_curve(_one(axis, "curve"), _curve()) for axis in axes
            )
        ):
            raise ReplayError("Migration requires exact observed uniform entrance")
        properties.remove(transform)
    return animate_entrance(ET.tostring(root, encoding="unicode"))


def preserve_neutral(template_xml: str) -> str:
    """Copy the observed two-layer neutral template without adding animation.

    This retains its existing paint and layout; it is not converted to the
    three-layer native FCP contract and supplies no native rendering approval.
    """
    root = _parse(template_xml)
    nodes = root.findall("scene/layer/scenenode[@factoryID='16']")
    layers = root.findall("scene/layer")
    if (
        root.get("version") != "5.13"
        or len(nodes) != 2
        or len(layers) != 1
        or len(root.findall(".//scenenode[@factoryID='16']")) != 2
        or root.findall(".//behavior")
    ):
        raise ReplayError("Expected observed neutral two-layer template")
    for node in [*layers, *nodes]:
        properties = _one(node, "parameter[@id='1']")
        if properties.find("parameter[@id='100']") is not None:
            raise ReplayError("Neutral template must not animate or transform its origin")
    return template_xml


def prepare(source: Path, output: Path, *, upgrade: bool = False, neutral: bool = False) -> None:
    source, output = _local(source), _local(output)
    if not source.is_file() or not 0 < source.stat().st_size <= MAX_XML_CHARACTERS:
        raise ReplayError("Expected a bounded Motion entrance source")
    if upgrade and neutral:
        raise ReplayError("Choose one template migration contract")
    producer = preserve_neutral if neutral else upgrade_entrance if upgrade else animate_entrance
    result = producer(source.read_text(encoding="utf-8"))
    if len(result) > MAX_XML_CHARACTERS:
        raise ReplayError("Animated Motion document exceeds XML size limit")
    with output.open("x", encoding="utf-8") as stream:
        stream.write(result)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--upgrade-uniform", action="store_true")
    parser.add_argument("--preserve-neutral", action="store_true")
    args = parser.parse_args()
    prepare(args.source, args.output, upgrade=args.upgrade_uniform, neutral=args.preserve_neutral)


if __name__ == "__main__":
    main()
