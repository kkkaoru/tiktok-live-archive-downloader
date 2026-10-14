import json
import sys
from pathlib import Path
from xml.etree import ElementTree as ET

import pytest

from replay import motion_pre_layout as layout
from replay.core import ReplayError
from replay.motion_entrance import animate_entrance


def node(name: str, identity: int, width: float, shadow: str = "") -> str:
    return f'''<scenenode name="{name}" id="{identity}" factoryID="16">
    <parameter id="1"/><style id="{identity + 1}">
    <parameter id="83"><font>NotoSansJP-Thin_Bold</font></parameter>
    <parameter id="3" value="88"/><parameter id="30">
    <parameter id="36" value="{width}"/></parameter>
    {shadow}</style><parameter id="2"><parameter id="314" value="1"/>
    <parameter id="373"><parameter id="2" value="2"/></parameter>
    <parameter id="319"><parameter id="322" value="0"/></parameter>
    <parameter id="369"><text>合成字幕</text></parameter></parameter></scenenode>'''


@pytest.fixture
def source() -> str:
    shadow = """<parameter id="21"><parameter id="27" value="7"/>
    <parameter id="75"><parameter id="1" value="4"/>
    <parameter id="2" value="4"/></parameter></parameter>"""
    return animate_entrance(
        '<ozml version="5.15"><template><flags>1</flags></template><scene><publishSettings/>'
        "<sceneSettings><duration>240</duration><frameRate>30</frameRate><NTSC>0</NTSC></sceneSettings>"
        '<layer id="100">'
        + node("Caption Keyline Face", 200, 2)
        + node("Caption Face", 300, 14.4)
        + node("Caption Outer Rim", 400, 17.4, shadow)
        + "</layer></scene></ozml>"
    )


@pytest.fixture
def group_source(source: str) -> str:
    return source.replace('<layer id="100">', '<layer id="100"><parameter id="1"/>')


def test_group_scale_preserves_text_and_styles(group_source: str) -> None:
    result = layout.publish_group_scale(group_source)
    before, after = ET.fromstring(group_source), ET.fromstring(result)
    assert [ET.tostring(node) for node in before.findall("scene/layer/scenenode")] == [
        ET.tostring(node) for node in after.findall("scene/layer/scenenode")
    ]
    controls = layout.group_controls(result)
    assert [(item.name, item.key, item.base) for item in controls] == [
        ("Caption Group Scale X", "9999/100/1/100/105/1", 1.0),
        ("Caption Group Scale Y", "9999/100/1/100/105/2", 1.0),
    ]
    with pytest.raises(ReplayError, match="untransformed"):
        layout.publish_group_scale(result)


@pytest.mark.parametrize(
    "mutation", ["position", "bottom", "scale", "extra", "missing", "duplicate", "wrong"]
)
def test_group_scale_rejects_unsupported_geometry(group_source: str, mutation: str) -> None:
    root = ET.fromstring(layout.publish_group_scale(group_source))
    transform = root.find("scene/layer/parameter[@id='1']/parameter[@id='100']")
    targets = root.find("scene/publishSettings")
    child = root.find("scene/layer/scenenode")
    assert transform is not None and targets is not None and child is not None
    if mutation == "position":
        child_transform = child.find("parameter[@id='1']/parameter[@id='100']")
        assert child_transform is not None
        ET.SubElement(child_transform, "parameter", id="101", value="1")
    elif mutation == "bottom":
        obj = child.find("parameter[@id='2']")
        assert obj is not None
        ET.SubElement(obj, "parameter", id="326", value="1")
    elif mutation == "scale":
        transform[0][0].set("value", "0.5")
    elif mutation == "extra":
        ET.SubElement(transform, "parameter", id="101", value="1")
    elif mutation == "missing":
        targets.remove(targets[0])
    elif mutation == "duplicate":
        ET.SubElement(targets, "target", targets[0].attrib)
    else:
        targets[0].set("channel", "wrong")
    with pytest.raises(ReplayError):
        layout.group_controls(ET.tostring(root, encoding="unicode"))


def test_group_scale_rejects_existing_target(group_source: str) -> None:
    source = group_source.replace(
        "<publishSettings />",
        '<publishSettings><target name="Caption Group Scale X"/></publishSettings>',
    )
    with pytest.raises(ReplayError, match="already"):
        layout.publish_group_scale(source)


def test_native_save_omits_identity_group_transform(group_source: str) -> None:
    root = ET.fromstring(layout.publish_group_scale(group_source))
    info = root.find("scene/layer/parameter[@id='1']")
    assert info is not None
    info.remove(info[0])
    controls = layout.group_controls(ET.tostring(root, encoding="unicode"))
    assert [(item.name, item.base) for item in controls] == [
        ("Caption Group Scale X", 1.0),
        ("Caption Group Scale Y", 1.0),
    ]


def test_group_scale_cli(
    group_source: str, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    source, output, receipt, inspection = (
        tmp_path / "source.moti",
        tmp_path / "new.moti",
        tmp_path / "receipt.json",
        tmp_path / "inspect.json",
    )
    source.write_text(group_source, encoding="utf-8")
    monkeypatch.setattr(layout, "_local", lambda value: value)
    monkeypatch.setattr(
        sys, "argv", ["layout", str(source), str(output), str(receipt), "--group-scale"]
    )
    layout.main()
    monkeypatch.setattr(
        sys, "argv", ["layout", str(output), str(inspection), "--group-scale", "--inspect"]
    )
    layout.main()
    assert (
        json.loads(inspection.read_text(encoding="utf-8"))["controls"][0]["key"]
        == "9999/100/1/100/105/1"
    )


@pytest.mark.parametrize("characters,factor", [(12, 1.0), (82, 0.142), (151, 0.07)])
def test_published_dimensions_are_pre_layout_and_synchronized(
    source: str, characters: int, factor: float
) -> None:
    source = source.replace("合成字幕", "あ" * characters)
    result = layout.publish_geometry(source)
    old, new = ET.fromstring(source), ET.fromstring(result)
    old_layer, new_layer = old.find("scene/layer"), new.find("scene/layer")
    assert old_layer is not None and new_layer is not None
    assert ET.tostring(old_layer) == ET.tostring(new_layer)
    values = layout.scaled_values(result, factor)
    assert len(values) == 9
    assert values[0] == ("Caption Keyline Face Font Size", "9999/100/200/5/201/3", 88 * factor)
    assert values[2][2] == values[4][2] == 88 * factor
    assert values[-1][2] == values[-2][2] == 4 * factor
    assert new.findtext(".//text") == "あ" * characters
    with pytest.raises(ReplayError, match="already"):
        layout.publish_geometry(result)


@pytest.mark.parametrize("factor", [0, -1, 2, True, float("nan"), float("inf")])
def test_invalid_scale(source: str, factor: float) -> None:
    with pytest.raises(ReplayError):
        layout.scaled_values(layout.publish_geometry(source), factor)


@pytest.mark.parametrize(
    "old,new",
    [
        ('version="5.15"', 'version="5.13"'),
        ('factoryID="16"', 'factoryID="15"'),
        ('id="314" value="1"', 'id="314" value="0"'),
        ("NotoSansJP-Thin_Bold", "Other"),
        ('value="14.4"', 'value="15"'),
        ('value="14.4"', 'value="nan"'),
        ('value="14.4"', 'value="bad"'),
    ],
)
def test_rejects_changed_contract(source: str, old: str, new: str) -> None:
    with pytest.raises(ReplayError):
        layout.publish_geometry(source.replace(old, new))


def test_mixed_sizes_and_missing_publication(source: str) -> None:
    with pytest.raises(ReplayError, match="different"):
        layout.publish_geometry(
            source.replace('value="88"', 'value="44"', 1).replace(
                'id="36" value="2"', 'id="36" value="1"', 1
            )
        )
    with pytest.raises(ReplayError, match="Missing"):
        layout.scaled_values(source, 1.0)


def test_cli_exclusive_output(source: str, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    original, output, receipt = (
        tmp_path / "source.moti",
        tmp_path / "new.moti",
        tmp_path / "receipt.json",
    )
    original.write_text(source, encoding="utf-8")
    monkeypatch.setattr(layout, "_local", lambda value: value)
    monkeypatch.setattr(sys, "argv", ["pre-layout", str(original), str(output), str(receipt)])
    layout.main()
    assert json.loads(receipt.read_text(encoding="utf-8"))["nativeVerified"] is False
    with pytest.raises(ReplayError, match="distinct"):
        layout.main()


def test_cli_input_budget(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    source = tmp_path / "empty.moti"
    source.touch()
    monkeypatch.setattr(layout, "_local", lambda value: value)
    monkeypatch.setattr(
        sys, "argv", ["pre-layout", str(source), str(tmp_path / "new"), str(tmp_path / "receipt")]
    )
    with pytest.raises(ReplayError, match="budget"):
        layout.main()


@pytest.mark.parametrize("mode", ["valid", "extra", "empty", "missing"])
def test_inspect_saved_contract(
    source: str, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, mode: str
) -> None:
    template = tmp_path / "native-saved.moti"
    template.write_text(
        "" if mode == "empty" else layout.publish_geometry(source), encoding="utf-8"
    )
    output = tmp_path / "readback.json"
    argv = ["pre-layout", str(template), str(output)]
    if mode != "missing":
        argv.append("--inspect")
    if mode == "extra":
        argv.append(str(tmp_path / "extra"))
    monkeypatch.setattr(layout, "_local", lambda value: value)
    monkeypatch.setattr(sys, "argv", argv)
    if mode == "valid":
        layout.main()
        assert len(json.loads(output.read_text(encoding="utf-8"))["controls"]) == 9
    else:
        with pytest.raises(ReplayError):
            layout.main()
