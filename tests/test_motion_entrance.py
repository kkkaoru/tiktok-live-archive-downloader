import copy
import sys
from pathlib import Path
from xml.etree import ElementTree as ET

import pytest

from replay import motion_entrance as entrance
from replay import motion_template_timing
from replay.core import ReplayError

NODE = """<scenenode factoryID="16"><parameter id="1"/>
<style><parameter id="3" value="82.9558"/></style>
<parameter id="2"><parameter id="314" value="1"/>
<parameter id="373"><parameter id="2" value="2"/></parameter>
<parameter id="319"><parameter id="322" value="0"/></parameter>
<parameter id="369"><text>合成字幕</text></parameter></parameter></scenenode>"""
SOURCE = (
    '<ozml version="5.15"><template><flags>1</flags></template><scene>'
    "<sceneSettings><duration>240</duration><frameRate>30</frameRate><NTSC>0</NTSC>"
    "</sceneSettings><layer>" + NODE * 3 + "</layer></scene><!--intent--></ozml>"
)


def test_horizontal_entrance_preserves_style_text_opacity() -> None:
    result = entrance.animate_entrance(SOURCE)
    root = ET.fromstring(result)
    transforms = root.findall(".//scenenode/parameter/parameter[@id='100']")
    assert len(transforms) == 3
    assert ET.tostring(transforms[0]) == ET.tostring(transforms[1]) == ET.tostring(transforms[2])
    entrance.validate_entrance_transform(transforms[0])
    assert root.findall(".//curve/keypoint/time")[0].text == "0 1 1 0"
    assert root.findall(".//curve/keypoint/time")[1].text == "1 5 1 0"
    assert root.findall(".//curve/keypoint/value")[0].text == "0.78"
    assert root.findall(".//curve/keypoint/value")[1].text == "1.0"
    font_size = root.find(".//style/parameter")
    assert font_size is not None
    assert font_size.get("value") == "82.9558"
    assert root.findtext(".//text") == "合成字幕"
    assert "<!--intent-->" in result
    assert root.find(".//parameter[@id='101']") is None
    assert root.find(".//curve/min") is None
    vertical = transforms[0].find("parameter/parameter[@id='2']")
    assert vertical is not None and vertical.get("value") == "1.0" and len(vertical) == 0


@pytest.mark.parametrize(
    "old,new",
    [
        ('version="5.15"', 'version="5.13"'),
        ("<flags>1</flags>", "<flags>0</flags>"),
        ("<duration>240</duration>", "<duration>1</duration>"),
        ('factoryID="16"', 'factoryID="15"'),
        ('<parameter id="1"/>', '<parameter id="1"><parameter id="100"/></parameter>'),
        ('id="314" value="1"', 'id="314" value="0"'),
        ('id="322" value="0"', 'id="322" value="1"'),
        ('<parameter id="369">', '<parameter id="326" value="2"/><parameter id="369">'),
    ],
)
def test_rejects_nonzero_origin_fading_layout_or_unsupported_clock(old: str, new: str) -> None:
    with pytest.raises(ReplayError):
        entrance.animate_entrance(SOURCE.replace(old, new))


@pytest.mark.parametrize(
    "old,new",
    [
        ('id="105"', 'id="103"'),
        ('name="Y" id="2"', 'name="X" id="1"'),
        ("<value>0.78</value>", "<value>0.5</value>"),
        ("<time>1 5 1 0</time>", "<time>1 2 1 0</time>"),
        ("<time>1 5 1 0</time>", "<time>1 0 1 0</time>"),
        ("<value>0.78</value>", "<value>bad</value>"),
        ('<keypoint interpolation="1"', '<keypoint interpolation="2"'),
        ("<numberOfKeypoints>2</numberOfKeypoints>", "<numberOfKeypoints>3</numberOfKeypoints>"),
        ('<curve type="1"', '<curve unknown="x" type="1"'),
        (
            "<numberOfKeypoints>2</numberOfKeypoints>",
            "<numberOfKeypoints>2</numberOfKeypoints><unknown/>",
        ),
    ],
)
def test_validator_refuses_modified_animation(old: str, new: str) -> None:
    root = ET.fromstring(entrance.animate_entrance(SOURCE).replace(old, new))
    transform = root.find(".//parameter[@id='100']")
    assert transform is not None
    with pytest.raises(ReplayError):
        entrance.validate_entrance_transform(transform)


def test_native_rational_time_and_float_serialization() -> None:
    result = entrance.animate_entrance(SOURCE).replace("1 5 1 0", "160 800 1 0")
    result = result.replace("<value>0.78</value>", "<value>0.78000000000000003</value>")
    transform = ET.fromstring(result).find(".//parameter[@id='100']")
    assert transform is not None
    entrance.validate_entrance_transform(transform)


def test_new_file_cli_and_nonoverwriting(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(motion_template_timing, "ROOT", tmp_path)
    source, output = tmp_path / "source.moti", tmp_path / "output.moti"
    source.write_text(SOURCE, encoding="utf-8")
    monkeypatch.setattr(sys, "argv", ["entrance", str(source), str(output)])
    entrance.main()
    assert "0.78" in output.read_text(encoding="utf-8")
    assert source.read_text(encoding="utf-8") == SOURCE
    with pytest.raises(FileExistsError):
        entrance.main()
    with pytest.raises(ReplayError, match="bounded"):
        entrance.prepare(tmp_path / "missing.moti", output)
    monkeypatch.setattr(entrance, "MAX_XML_CHARACTERS", len(SOURCE.encode("utf-8")) + 1)
    with pytest.raises(ReplayError, match="exceeds"):
        entrance.prepare(source, tmp_path / "too-big.moti")


@pytest.mark.parametrize("text", ["犬", "羽根\n広がる", "ぁー\n長音\n三行"])
def test_upgrade_preserves_all_non_transform_content(text: str) -> None:
    expected = entrance.animate_entrance(SOURCE.replace("合成字幕", text))
    root = ET.fromstring(expected)
    scales = root.findall(".//scenenode/parameter/parameter/parameter[@id='105']")

    def legacy(scale: ET.Element) -> None:
        vertical = scale.find("parameter[@id='2']")
        horizontal = scale.find("parameter[@id='1']")
        assert vertical is not None and horizontal is not None
        scale.remove(vertical)
        replacement = copy.deepcopy(horizontal)
        replacement.set("id", "2")
        replacement.set("name", "Y")
        scale.append(replacement)

    tuple(map(legacy, scales))
    migrated = entrance.upgrade_entrance(ET.tostring(root, encoding="unicode"))
    assert ET.tostring(ET.fromstring(migrated)) == ET.tostring(ET.fromstring(expected))
    assert ET.fromstring(migrated).find(".//parameter[@id='101']") is None


def test_upgrade_rejects_nonlegacy_and_missing_layers() -> None:
    with pytest.raises(ReplayError, match="uniform"):
        entrance.upgrade_entrance(entrance.animate_entrance(SOURCE))
    with pytest.raises(ReplayError, match="three"):
        entrance.upgrade_entrance("<ozml><scene/></ozml>")
    altered = entrance.animate_entrance(SOURCE).replace(
        '<parameter name="Scale"', '<parameter id="101"/><parameter name="Scale"'
    )
    with pytest.raises(ReplayError, match="position"):
        entrance.upgrade_entrance(altered)


def test_neutral_two_layer_preserves_bytes_and_rejects_transform() -> None:
    source = (
        '<ozml version="5.13"><scene><layer><parameter id="1"/>'
        + NODE * 2
        + "</layer></scene></ozml>"
    )
    assert entrance.preserve_neutral(source) == source
    with pytest.raises(ReplayError, match="two-layer"):
        entrance.preserve_neutral(SOURCE)
    with pytest.raises(ReplayError, match="origin"):
        entrance.preserve_neutral(
            source.replace(
                '<parameter id="1"/>', '<parameter id="1"><parameter id="100"/></parameter>'
            )
        )


def test_exclusive_migration_modes(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(motion_template_timing, "ROOT", tmp_path)
    source = tmp_path / "source.moti"
    source.write_text(SOURCE)
    with pytest.raises(ReplayError, match="one template"):
        entrance.prepare(source, tmp_path / "result.moti", upgrade=True, neutral=True)
