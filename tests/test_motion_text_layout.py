"""Synthetic observed layout controls, without Apple template assets."""

import runpy
import sys
from pathlib import Path
from xml.etree import ElementTree as ET

import pytest

from replay.core import ReplayError
from replay.motion_text_layout import bottom_align_caption_text

SOURCE = """<ozml version="5.15"><scene><layer>
<scenenode factoryID="16"><paragraphMarginsCached cached="1" left="0" right="416"/>
<paragraphStyle id="12"/><style id="13"><parameter id="3" value="82"/></style>
<parameter id="1" value="240"/><parameter id="2">
<parameter id="369"><text>最初の行
次の行</text></parameter><parameter id="319"><parameter id="321" value="1"/></parameter>
</parameter><timing in="0 1 1 0" out="2 1 1 0"/>
</scenenode></layer></scene></ozml>"""


@pytest.mark.parametrize("version", ["5.13", "5.15"])
def test_bottom_alignment_preserves_text_geometry_style_and_reveal(version: str) -> None:
    source = SOURCE.replace('version="5.15"', f'version="{version}"')
    result = bottom_align_caption_text(source, width=800, height=360)
    root = ET.fromstring(result)
    node = root.find(".//scenenode")
    assert node is not None
    assert node.findtext("parameter[@id='2']/parameter[@id='369']/text") == "最初の行\n次の行"
    assert node.findall("parameter[@id='1']")[0].attrib == {"id": "1", "value": "240"}
    assert node.findall("style/parameter")[0].attrib == {"id": "3", "value": "82"}
    assert node.findall("parameter[@id='2']/parameter[@id='319']/parameter")[0].attrib == {
        "id": "321",
        "value": "1",
    }
    assert node.findall("paragraphMarginsCached")[0].attrib == {"cached": "0"}
    assert node.findall("paragraphStyle/parameter")[0].attrib == {
        "id": "401",
        "flags": "8606777360",
        "value": "1",
    }
    assert node.findall("parameter[@id='2']/parameter[@id='314']")[0].attrib == {
        "id": "314",
        "flags": "8590000144",
        "value": "1",
    }
    assert node.findall("parameter[@id='2']/parameter[@id='373']/parameter")[0].attrib == {
        "id": "2",
        "flags": "8590000144",
        "value": "2",
    }
    assert node.findall("parameter[@id='2']/parameter[@id='323']")[0].get("value") == "-400.0"
    assert node.findall("parameter[@id='2']/parameter[@id='324']")[0].get("value") == "400.0"
    assert node.findall("parameter[@id='2']/parameter[@id='325']")[0].get("value") == "360.0"
    assert node.findall("parameter[@id='2']/parameter[@id='326']")[0].get("value") == "0.0"
    assert bottom_align_caption_text(result, width=800, height=360) == result


@pytest.mark.parametrize(
    ("old", "new", "message"),
    [
        ('version="5.15"', 'version="6"', "Only observed"),
        ('factoryID="16"', 'factoryID="1"', "no text nodes"),
        ('<paragraphStyle id="12"/>', "", "no paragraph style"),
        ("最初の行\n次の行", "\n最初の行", "blank-line padding"),
        ("最初の行\n次の行", "", "blank-line padding"),
        (
            '<paragraphStyle id="12"/>',
            '<paragraphStyle><parameter id="401"/><parameter id="401"/></paragraphStyle>',
            "Duplicate",
        ),
        (
            '<paragraphStyle id="12"/>',
            '<paragraphStyle><parameter id="401" flags="1"/></paragraphStyle>',
            "Unsupported",
        ),
        (
            '<paragraphStyle id="12"/>',
            '<paragraphStyle><parameter id="401" flags="8606777360">'
            "<curve/></parameter></paragraphStyle>",
            "Unsupported",
        ),
    ],
)
def test_rejects_unknown_or_animated_layout(old: str, new: str, message: str) -> None:
    with pytest.raises(ReplayError, match=message):
        bottom_align_caption_text(SOURCE.replace(old, new), width=800, height=360)


def test_updates_every_prefix_state_and_preserves_distinct_text() -> None:
    root = ET.fromstring(SOURCE)
    layer = root.find("scene/layer")
    assert layer is not None
    second = ET.fromstring(SOURCE).find(".//scenenode")
    assert second is not None
    text = second.find("parameter[@id='2']/parameter[@id='369']/text")
    assert text is not None
    text.text = "最初の行"
    layer.append(second)
    result = ET.fromstring(
        bottom_align_caption_text(ET.tostring(root, encoding="unicode"), width=800, height=360)
    )
    assert len(result.findall(".//parameter[@id='373']/parameter[@value='2']")) == 2
    assert result.findall(".//text")[0].text == "最初の行\n次の行"
    assert result.findall(".//text")[1].text == "最初の行"


def test_cli_writes_new_file_and_refuses_overwrite(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    source = tmp_path / "source.moti"
    output = tmp_path / "output.moti"
    source.write_text(SOURCE, encoding="utf-8")
    monkeypatch.setattr(
        sys,
        "argv",
        ["motion_text_layout", str(source), str(output), "--width", "800", "--height", "360"],
    )
    monkeypatch.delitem(sys.modules, "replay.motion_text_layout")
    runpy.run_module("replay.motion_text_layout", run_name="__main__")
    assert 'value="2"' in output.read_text(encoding="utf-8")
    assert source.read_text(encoding="utf-8") == SOURCE
    with pytest.raises(FileExistsError):
        runpy.run_module("replay.motion_text_layout", run_name="__main__")


def test_replaces_disabled_zero_width_margins_with_explicit_box() -> None:
    source = SOURCE.replace(
        '<parameter id="369">',
        '<parameter id="324" flags="8606744592" value="0"/><parameter id="369">',
    )
    root = ET.fromstring(bottom_align_caption_text(source, width=1000, height=400, bottom=-20))
    assert root.findall(".//parameter[@id='324']")[0].attrib == {
        "id": "324",
        "flags": "8606711824",
        "value": "500.0",
    }
    assert root.findall(".//parameter[@id='325']")[0].get("value") == "380.0"
    assert root.findall(".//parameter[@id='326']")[0].get("value") == "-20.0"


@pytest.mark.parametrize(
    ("width", "height", "bottom"),
    [
        (0, 100, 0),
        (100, 0, 0),
        (float("nan"), 100, 0),
        (100, 100, float("inf")),
        (True, 100, 0),
        (10001, 100, 0),
        (100, 10001, 0),
        (100, 100, 10001),
        (100, 100, 9999),
    ],
)
def test_rejects_invalid_bounds(width: float, height: float, bottom: float) -> None:
    with pytest.raises(ReplayError, match="Paragraph bounds"):
        bottom_align_caption_text(SOURCE, width=width, height=height, bottom=bottom)


def test_explicit_anchor_replaces_static_y_preserving_x() -> None:
    source = SOURCE.replace(
        '<parameter id="1" value="240"/>',
        '<parameter id="1"><parameter id="100" flags="8589938704">'
        '<parameter id="101" flags="8589938704">'
        '<parameter id="1" flags="8606711824" value="10"/>'
        '<parameter id="2" flags="8606711824" value="480"/>'
        "</parameter></parameter></parameter>",
    )
    result = ET.fromstring(bottom_align_caption_text(source, width=800, height=360, anchor_y=240))
    assert result.findall(".//parameter[@id='101']/parameter[@id='2']")[0].get("value") == "240.0"
    assert result.findall(".//parameter[@id='101']/parameter[@id='1']")[0].get("value") == "10"


def test_anchor_rejects_animated_y_without_modifying_input() -> None:
    source = SOURCE.replace(
        '<parameter id="1" value="240"/>',
        '<parameter id="1"><parameter id="100" flags="8589938704">'
        '<parameter id="101" flags="8589938704">'
        '<parameter id="2" flags="8606711824"><curve value="480"/></parameter>'
        "</parameter></parameter></parameter>",
    )
    with pytest.raises(ReplayError, match="Unsupported Motion layout control: 2"):
        bottom_align_caption_text(source, width=800, height=360, anchor_y=240)


@pytest.mark.parametrize("anchor_y", [True, float("nan"), float("inf"), 10001])
def test_rejects_invalid_anchor(anchor_y: float) -> None:
    with pytest.raises(ReplayError, match="anchor Y"):
        bottom_align_caption_text(SOURCE, width=800, height=360, anchor_y=anchor_y)
