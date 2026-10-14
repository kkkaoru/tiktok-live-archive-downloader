"""Fit template-painted paragraphs with an explicitly validated scale location."""

import math
from xml.etree import ElementTree as ET

from replay.core import ReplayError

GeometryValues = tuple[tuple[str, str, float], ...]


def paragraph_values(document: str) -> GeometryValues:
    """Read actual template margins after validating its shared style contract."""
    from replay.motion_caption import _attribute, _one, _parse
    from replay.motion_pre_layout import geometry_controls, group_controls

    geometry_controls(document)
    root = _parse(document)
    layer = _one(root, "scene/layer")
    values: list[tuple[str, str, float]] = []
    for name in ("Caption Keyline Face", "Caption Face", "Caption Outer Rim"):
        node = _one(layer, f"scenenode[@name='{name}']")
        for channel, label in (("323", "Left Margin"), ("324", "Right Margin")):
            scalar = _one(node, f"parameter[@id='2']/parameter[@id='{channel}']")
            try:
                value = float(_attribute(scalar, "value"))
            except ValueError as error:
                raise ReplayError("Invalid template paragraph margin") from error
            if len(scalar):
                raise ReplayError("Template paragraph margins must be static")
            key = f"9999/{_attribute(layer, 'id')}/{_attribute(node, 'id')}/2/{channel}"
            values.append((f"{name} {label}", key, value))
    if any(
        target.get("name") in {"Caption Group Scale X", "Caption Group Scale Y"}
        for target in root.findall("scene/publishSettings/target")
    ):
        values.extend(
            (control.name, control.key, control.base)
            for control in group_controls(document, published=True)
        )
    result = tuple(values)
    _controls(result)
    return result


def _controls(values: GeometryValues) -> None:
    if len(values) not in (6, 8) or len({key for _, key, _ in values}) != len(values):
        raise ReplayError("Expected six unique paragraph margin controls")
    if len(values) == 8:
        x, y = values[6:]
        if (
            x[0] != "Caption Group Scale X"
            or y[0] != "Caption Group Scale Y"
            or not x[1].endswith("/1/100/105/1")
            or y[1] != x[1][:-1] + "2"
            or x[2] != 1
            or y[2] != 1
        ):
            raise ReplayError("Expected identity published parent group scale axes")
    widths: list[float] = []
    for left, right in zip(values[:6:2], values[1:6:2], strict=True):
        if (
            not left[0]
            or not right[0]
            or not left[1].endswith("/2/323")
            or right[1] != left[1][:-3] + "324"
            or not math.isfinite(left[2])
            or not math.isfinite(right[2])
            or not left[2] < 0 < right[2]
            or left[2] != -right[2]
        ):
            raise ReplayError("Expected centered finite paired paragraph margins")
        widths.append(right[2] - left[2])
    if len(set(widths)) != 1:
        raise ReplayError("Paragraph layers must have equal measured widths")


def _plain(title: ET.Element) -> None:
    texts = title.findall("text")
    if title.findall("text-style-def") or len(texts) != 3 or any(len(text) for text in texts):
        raise ReplayError("Paragraph fitting requires three plain template text layers")


def require_production_geometry(values: GeometryValues, factor: float) -> None:
    """Reject the diagnosed canvas-clipped outer fit in production consumers."""
    if factor != 1:
        if len(values) != 8:
            raise ReplayError("Scaled production captions require native internal group geometry")
        _controls(values)


def apply_geometry(title: ET.Element, values: GeometryValues, factor: float) -> None:
    """Preserve font/paint/animation and title origin; scale measured bounds once."""
    if isinstance(factor, bool) or not math.isfinite(factor) or not 0 < factor <= 1:
        raise ReplayError("Expected a bounded paragraph fit factor")
    if not values:
        if factor != 1:
            raise ReplayError("Scaled captions require published paragraph geometry")
        if title.findall("param") or title.findall("text-style-def"):
            raise ReplayError("Existing caption geometry requires a template contract")
    else:
        _controls(values)
        _plain(title)
    transform = title.find("adjust-transform")
    if transform is None:
        raise ReplayError("Caption geometry requires an existing placement transform")
    transform.attrib.pop("scale", None)
    keys = {key for _, key, _ in values}
    group_key = values[6][1].rsplit("/", 1)[0] if len(values) == 8 else None
    if group_key is not None:
        keys.add(group_key)
    for node in title.findall("param"):
        if node.get("key") in keys:
            title.remove(node)
    if factor == 1:
        return
    if len(values) == 6:
        transform.set("scale", f"{factor:.15g} {factor:.15g}")
    for index, (name, key, base) in enumerate(values[:6]):
        value = base / factor
        if not math.isfinite(value):
            raise ReplayError("Expanded paragraph margin is not finite")
        title.insert(index, ET.Element("param", name=name, key=key, value=f"{value:.15g}"))
    if group_key is not None:
        title.insert(
            6, ET.Element("param", name="調整", key=group_key, value=f"{factor:.15g} {factor:.15g}")
        )


def geometry_factor(title: ET.Element, values: GeometryValues) -> float:
    """Verify paragraph expansion and exactly one internal or diagnostic outer fit."""
    transform = title.find("adjust-transform")
    if transform is None:
        raise ReplayError("Missing caption placement transform")
    try:
        scale = tuple(float(value) for value in transform.get("scale", "1 1").split())
    except ValueError as error:
        raise ReplayError("Invalid caption outer scale") from error
    if (
        len(scale) != 2
        or scale[0] != scale[1]
        or not math.isfinite(scale[0])
        or not 0 < scale[0] <= 1
    ):
        raise ReplayError("Expected uniform bounded caption outer scale")
    factor = scale[0]
    if not values:
        if factor != 1:
            raise ReplayError("Scaled caption lacks a paragraph contract")
        return factor
    _controls(values)
    _plain(title)
    if len(values) == 8:
        if factor != 1:
            raise ReplayError("Internal group fitting requires identity outer scale")
        group_key = values[6][1].rsplit("/", 1)[0]
        leaf_keys = {key for _, key, _ in values[6:]}
        if any(node.get("key") in leaf_keys for node in title.findall("param")):
            raise ReplayError("Internal scale must use the native parent vector")
        nodes = [node for node in title.findall("param") if node.get("key") == group_key]
        if len(nodes) > 1 or (nodes and len(nodes[0])):
            raise ReplayError("Ambiguous internal group scale")
        try:
            factors = (
                tuple(float(value) for value in nodes[0].attrib["value"].split())
                if nodes
                else (1.0, 1.0)
            )
        except (KeyError, ValueError) as error:
            raise ReplayError("Invalid internal group scale") from error
        if (
            len(factors) != 2
            or factors[0] != factors[1]
            or not math.isfinite(factors[0])
            or not 0 < factors[0] <= 1
        ):
            raise ReplayError("Expected uniform bounded internal group scale")
        factor = factors[0]
    for _, key, base in values[:6]:
        matches = [node for node in title.findall("param") if node.get("key") == key]
        if factor == 1 and not matches:
            continue
        if len(matches) != 1 or len(matches[0]):
            raise ReplayError("Missing or ambiguous paragraph margin")
        try:
            value = float(matches[0].attrib["value"])
        except (KeyError, ValueError) as error:
            raise ReplayError("Invalid paragraph margin value") from error
        if not math.isclose(value * factor, base, rel_tol=1e-12, abs_tol=1e-12):
            raise ReplayError("Paragraph width and outer scale disagree")
    return factor


def unique_style_ids(root: ET.Element) -> None:
    """Keep imported native local references unambiguous after title cloning."""
    for number, title in enumerate(root.iter("title")):
        mapping: dict[str, str] = {}
        for index, definition in enumerate(title.findall("text-style-def")):
            old = definition.get("id")
            if old is None or old in mapping:
                raise ReplayError("Caption style identifiers are missing or ambiguous")
            mapping[old] = f"caption-style-{number}-{index}"
            definition.set("id", mapping[old])
        for span in title.findall("text/text-style"):
            old = span.get("ref")
            if old is None or old not in mapping:
                raise ReplayError("Caption text lacks its local style definition")
            span.set("ref", mapping[old])
