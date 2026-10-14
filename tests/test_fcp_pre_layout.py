import copy
from xml.etree import ElementTree as ET

import pytest
from test_motion_pre_layout import source as source

from replay.core import ReplayError
from replay.fcp_pre_layout import (
    GeometryValues,
    apply_geometry,
    geometry_factor,
    paragraph_values,
    unique_style_ids,
)


@pytest.mark.parametrize("fault", ["none", "group", "number", "animated", "mismatch"])
def test_actual_template_margin_readback(source: str, fault: str) -> None:
    root = ET.fromstring(source)
    layer = root.find("scene/layer")
    assert layer is not None
    ET.SubElement(layer, "parameter", id="1")
    for node in root.findall("scene/layer/scenenode"):
        layout = node.find("parameter[@id='2']")
        assert layout is not None
        ET.SubElement(layout, "parameter", id="323", value="-486")
        ET.SubElement(layout, "parameter", id="324", value="486")
    scalar = root.find(".//parameter[@id='323']")
    assert scalar is not None
    if fault == "number":
        scalar.set("value", "bad")
    elif fault == "animated":
        ET.SubElement(scalar, "keyframe")
    elif fault == "mismatch":
        scalar.set("value", "-400")
    document = ET.tostring(root, encoding="unicode")
    if fault == "group":
        from replay.motion_pre_layout import publish_group_scale

        document = publish_group_scale(document)
        values = paragraph_values(document)
        assert len(values) == 8
        assert values[-1] == ("Caption Group Scale Y", "9999/100/1/100/105/2", 1)
    elif fault != "none":
        with pytest.raises(ReplayError):
            paragraph_values(document)
    else:
        values = paragraph_values(document)
        assert len(values) == 6
        assert values[0][1:] == ("9999/100/200/2/323", -486)


@pytest.fixture
def geometry() -> GeometryValues:
    return tuple(
        (f"layer{layer} {channel}", f"9999/1/{layer}/2/{channel}", value)
        for layer in range(3)
        for channel, value in (("323", -486.0), ("324", 486.0))
    )


@pytest.fixture
def title() -> ET.Element:
    return ET.fromstring(
        "<title><text>whole</text><text>whole</text><text>whole</text>"
        '<adjust-transform position="0 10"/></title>'
    )


@pytest.fixture
def internal_geometry(geometry: GeometryValues) -> GeometryValues:
    return (
        *geometry,
        ("Caption Group Scale X", "9999/1/1/100/105/1", 1),
        ("Caption Group Scale Y", "9999/1/1/100/105/2", 1),
    )


@pytest.mark.parametrize("factor", [1.0, 0.142084490571554, 0.0773577397532837])
def test_internal_scale_before_canvas_keeps_outer_origin(
    internal_geometry: GeometryValues, title: ET.Element, factor: float
) -> None:
    apply_geometry(title, internal_geometry, factor)
    transform = title.find("adjust-transform")
    assert transform is not None and transform.attrib == {"position": "0 10"}
    assert geometry_factor(title, internal_geometry) == pytest.approx(factor)
    assert len(title.findall("param")) == (0 if factor == 1 else 7)
    if factor != 1:
        group = title.findall("param")[6]
        assert group.get("key") == "9999/1/1/100/105"
        assert group.get("value") == f"{factor:.15g} {factor:.15g}"
    assert title.findall("text-style-def") == []
    assert [text.text for text in title.findall("text")] == ["whole"] * 3
    apply_geometry(title, internal_geometry, 1)
    assert title.findall("param") == []


@pytest.mark.parametrize(
    "fault",
    ["outer", "duplicate", "animated", "invalid", "absent", "unequal", "nan", "missing", "leaf"],
)
def test_internal_scale_readback_rejects_partial_or_double_fit(
    internal_geometry: GeometryValues, title: ET.Element, fault: str
) -> None:
    apply_geometry(title, internal_geometry, 0.5)
    node = title.findall("param")[6]
    if fault == "outer":
        transform = title.find("adjust-transform")
        assert transform is not None
        transform.set("scale", ".5 .5")
    elif fault == "duplicate":
        title.append(copy.deepcopy(node))
    elif fault == "animated":
        ET.SubElement(node, "keyframeAnimation")
    elif fault == "absent":
        node.attrib.pop("value")
    elif fault == "missing":
        title.remove(node)
    elif fault == "leaf":
        node.set("key", internal_geometry[6][1])
    else:
        node.set("value", {"invalid": "bad", "unequal": ".7 .5", "nan": "nan nan"}[fault])
    with pytest.raises(ReplayError):
        geometry_factor(title, internal_geometry)


def test_invalid_internal_template_contract(
    internal_geometry: GeometryValues, title: ET.Element
) -> None:
    invalid = (*internal_geometry[:7], ("Caption Group Scale Y", internal_geometry[7][1], 0.9))
    with pytest.raises(ReplayError, match="identity"):
        apply_geometry(title, invalid, 0.5)


@pytest.mark.parametrize("factor", [1.0, 0.142084490571554, 0.0773577397532837])
def test_expand_before_scale_preserves_text_paint_origin(
    geometry: GeometryValues, factor: float, title: ET.Element
) -> None:
    apply_geometry(title, geometry, factor)
    assert title.findall("text-style-def") == []
    assert [text.text for text in title.findall("text")] == ["whole"] * 3
    assert len(title.findall("param")) == (0 if factor == 1 else 6)
    transform = title.find("adjust-transform")
    assert transform is not None and transform.get("position") == "0 10"
    assert geometry_factor(title, geometry) == pytest.approx(factor)
    if factor != 1:
        for node, (_, key, base) in zip(title.findall("param"), geometry, strict=True):
            assert node.get("key") == key
            assert float(node.attrib["value"]) * factor == pytest.approx(base)
    apply_geometry(title, geometry, 1)
    assert title.findall("param") == []
    assert transform.attrib == {"position": "0 10"}
    assert geometry_factor(title, geometry) == 1


def test_no_contract_cannot_keep_scaled_prototype(
    geometry: GeometryValues, title: ET.Element
) -> None:
    apply_geometry(title, (), 1)
    assert geometry_factor(title, ()) == 1
    with pytest.raises(ReplayError, match="published"):
        apply_geometry(title, (), 0.5)
    apply_geometry(title, geometry, 0.5)
    with pytest.raises(ReplayError, match="template contract"):
        apply_geometry(title, (), 1)
    with pytest.raises(ReplayError, match="contract"):
        geometry_factor(title, ())


@pytest.mark.parametrize("factor", [0, 2, float("nan"), True, 5e-324])
def test_invalid_factor(geometry: GeometryValues, title: ET.Element, factor: float) -> None:
    with pytest.raises(ReplayError):
        apply_geometry(title, geometry, factor)


@pytest.mark.parametrize("fault", ["count", "duplicate", "center", "width", "nan", "key", "name"])
def test_invalid_margin_contract(geometry: GeometryValues, title: ET.Element, fault: str) -> None:
    values = list(geometry)
    if fault == "count":
        values.pop()
    elif fault == "duplicate":
        values[2] = values[0]
    elif fault == "center":
        values[0] = (*values[0][:2], -400)
    elif fault == "width":
        values[0] = (*values[0][:2], -400)
        values[1] = (*values[1][:2], 400)
    elif fault == "nan":
        values[0] = (*values[0][:2], float("nan"))
    elif fault == "key":
        values[0] = (values[0][0], "wrong", values[0][2])
    else:
        values[0] = ("", values[0][1], values[0][2])
    with pytest.raises(ReplayError):
        apply_geometry(title, tuple(values), 0.5)


@pytest.mark.parametrize(
    "fault", ["missing", "duplicate", "animated", "invalid", "absent", "wrong"]
)
def test_margin_readback_guards(geometry: GeometryValues, title: ET.Element, fault: str) -> None:
    apply_geometry(title, geometry, 0.5)
    node = title.findall("param")[0]
    if fault == "missing":
        title.remove(node)
    elif fault == "duplicate":
        title.append(copy.deepcopy(node))
    elif fault == "animated":
        ET.SubElement(node, "keyframeAnimation")
    elif fault == "absent":
        node.attrib.pop("value")
    else:
        node.set("value", "bad" if fault == "invalid" else "-1")
    with pytest.raises(ReplayError):
        geometry_factor(title, geometry)


@pytest.mark.parametrize("scale", ["bad", "nan nan", "0 0", ".5 .6", "1"])
def test_invalid_outer_scale(title: ET.Element, scale: str) -> None:
    transform = title.find("adjust-transform")
    assert transform is not None
    transform.set("scale", scale)
    with pytest.raises(ReplayError):
        geometry_factor(title, ())


def test_missing_transform_and_nonplain_rejected(
    geometry: GeometryValues, title: ET.Element
) -> None:
    title.remove(title.findall("adjust-transform")[0])
    with pytest.raises(ReplayError, match="placement"):
        apply_geometry(title, geometry, 1)
    with pytest.raises(ReplayError, match="placement"):
        geometry_factor(title, geometry)
    ET.SubElement(title, "text-style-def")
    with pytest.raises(ReplayError, match="plain"):
        apply_geometry(title, geometry, 1)


def test_imported_style_ids_are_not_duplicated() -> None:
    title = ET.fromstring(
        '<title><text><text-style ref="s">x</text-style></text><text-style-def id="s"/></title>'
    )
    root = ET.Element("root")
    root.extend((title, copy.deepcopy(title)))
    unique_style_ids(root)
    assert len({node.get("id") for node in root.findall(".//text-style-def")}) == 2
    span = title.find("text/text-style")
    assert span is not None
    span.set("ref", "absent")
    with pytest.raises(ReplayError):
        unique_style_ids(root)


def test_duplicate_local_style_definition_rejected() -> None:
    title = ET.fromstring('<title><text-style-def id="s"/><text-style-def id="s"/></title>')
    with pytest.raises(ReplayError):
        unique_style_ids(title)
