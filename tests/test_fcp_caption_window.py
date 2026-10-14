import json
import sys
from fractions import Fraction
from pathlib import Path
from xml.etree import ElementTree as ET

import pytest
from test_fcp_pre_layout import geometry as geometry
from test_fcp_pre_layout import internal_geometry as internal_geometry

from replay import fcp_caption_window as patch
from replay.caption_scale import CaptionScale
from replay.core import ReplayError
from replay.fcp_caption_retarget import Cue
from replay.fcp_pre_layout import GeometryValues, geometry_factor
from replay.stages import file_digest


@pytest.fixture
def root() -> ET.Element:
    return ET.fromstring("""<fcpxml><resources><format id="f" frameDuration="1/30s"/></resources>
    <library><event><project name="old"><sequence format="f" duration="1s"><spine>
    <asset-clip offset="0s" start="10s" duration="1s" ref="clean" srcEnable="video">
    <asset-clip name="Voice" offset="10s" start="100s" duration="1s" ref="voice"/>
    <title ref="motion" offset="10s" start="3600s" duration="1/2s">
    <text>partial</text><text>partial</text><text>partial</text><adjust-transform>
    <param name="position"><keyframeAnimation>
    <keyframe time="3600s" value="0 10" curve="linear"/>
    <keyframe time="7201/2s" value="0 15" curve="linear"/>
    </keyframeAnimation></param></adjust-transform></title>
    <title ref="motion" offset="21/2s" start="3600s" duration="1/2s">
    <text>old boundary</text><text>old boundary</text><text>old boundary</text><adjust-transform>
    <param name="position"><keyframeAnimation>
    <keyframe time="3600s" value="0 15" curve="linear"/>
    <keyframe time="7201/2s" value="0 20" curve="linear"/>
    </keyframeAnimation></param></adjust-transform></title>
    <audio-channel-source srcCh="1, 2" active="0"/>
    </asset-clip></spine></sequence></project></event></library></fcpxml>""")


def test_new_atomic_caption_does_not_reuse_old_stage_boundaries(root: ET.Element) -> None:
    before = ET.tostring(root)
    result = patch.replace_captions(
        root, (Cue(Fraction(0), Fraction(1), "whole word"),), global_start=Fraction(0), rate=30
    )
    titles = result.findall(".//title")
    assert len(titles) == 1
    clip = result.find(".//spine/asset-clip")
    assert clip is not None
    assert [node.tag for node in clip] == ["asset-clip", "title", "audio-channel-source"]
    assert [node.text for node in titles[0].findall("text")] == ["whole word"] * 3
    assert titles[0].get("offset") == "10/1s"
    assert titles[0].get("duration") == "1/1s"
    assert [node.attrib for node in titles[0].findall(".//keyframe")] == [
        {"time": "3600/1s", "value": "0 10", "curve": "linear"},
        {"time": "7201/2s", "value": "0 15", "curve": "linear"},
        {"time": "3601/1s", "value": "0 20", "curve": "linear"},
    ]
    assert ET.tostring(root) == before
    new_voice = result.find(".//asset-clip/asset-clip")
    old_voice = root.find(".//asset-clip/asset-clip")
    assert new_voice is not None and old_voice is not None
    assert ET.tostring(new_voice) == ET.tostring(old_voice)


def test_scaled_window_keeps_bottom_position_and_replaces_old_scale_once(
    root: ET.Element, internal_geometry: GeometryValues
) -> None:
    geometry = internal_geometry
    scale = CaptionScale(400, 40, 100, 40, 4, 0.25)
    result = patch.replace_captions(
        root,
        (Cue(Fraction(0), Fraction(1), "whole word", scale),),
        global_start=Fraction(0),
        rate=30,
        pre_layout_geometry=geometry,
    )
    transform = result.find(".//title/adjust-transform")
    assert transform is not None
    assert transform.get("scale") is None
    title = result.find(".//title")
    assert title is not None
    assert geometry_factor(title, geometry) == 0.25
    keyframe = transform.find(".//keyframe")
    assert keyframe is not None
    assert keyframe.get("value") == "0 10"


def test_scaled_production_rejects_old_outer_contract(
    root: ET.Element, geometry: GeometryValues
) -> None:
    scale = CaptionScale(400, 40, 100, 40, 4, 0.25)
    with pytest.raises(ReplayError, match="native internal"):
        patch.replace_captions(
            root,
            (Cue(Fraction(0), Fraction(1), "word", scale),),
            global_start=Fraction(0),
            rate=30,
            pre_layout_geometry=geometry,
        )


def test_interpolated_position_preserves_source_clock(root: ET.Element) -> None:
    result = patch.replace_captions(
        root, (Cue(Fraction(1, 4), Fraction(3, 4), "new"),), global_start=Fraction(0), rate=30
    )
    title = result.find(".//title")
    assert title is not None
    assert title.get("offset") == "154/15s"
    keyframe = title.find(".//keyframe")
    assert keyframe is not None
    assert keyframe.get("value") == "0 12.6666666667"


@pytest.mark.parametrize("change", ["curve", "x", "nan", "conflict", "nested", "unsupported"])
def test_invalid_reused_position_is_rejected(root: ET.Element, change: str) -> None:
    frame = root.find(".//keyframe")
    title = root.find(".//title")
    assert frame is not None and title is not None
    if change == "curve":
        frame.set("curve", "smooth")
    elif change == "x":
        frame.set("value", "1 10")
    elif change == "nan":
        frame.set("value", "0 nan")
    elif change == "conflict":
        other = root.findall(".//title")[1].find(".//keyframe")
        assert other is not None
        other.set("value", "0 99")
    elif change == "nested":
        text = title.find("text")
        assert text is not None
        ET.SubElement(text, "text-style")
    else:
        ET.SubElement(title, "filter-video")
    with pytest.raises(ReplayError):
        patch.replace_captions(
            root, (Cue(Fraction(0), Fraction(1), "new"),), global_start=Fraction(0), rate=30
        )


def test_position_extrapolation_refused() -> None:
    with pytest.raises(ReplayError, match="observations required"):
        patch._position(((Fraction(1), 0, 1),), Fraction(0))


@pytest.fixture
def config(root: ET.Element, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    semantic = tmp_path / "semantic.fcpxml"
    ET.ElementTree(root).write(semantic)
    semantic_ref = {"path": str(semantic), "sha256": file_digest(semantic)}
    lineage = tmp_path / "lineage.json"
    lineage.write_text(json.dumps({"outputXML": semantic_ref, "captionClockEqual": True}))
    reflow = tmp_path / "reflow.json"
    reflow.write_text("{}")
    config_path = tmp_path / "config.json"
    config_path.write_text(
        json.dumps(
            {
                "version": 1,
                "semantic": semantic_ref,
                "lineage": {"path": str(lineage), "sha256": file_digest(lineage)},
                "reflow": {"path": str(reflow), "sha256": file_digest(reflow)},
                "recognitionSource": {"path": str(tmp_path / "original.mov"), "sha256": "a" * 64},
                "start": "0",
                "end": "1",
                "name": "new",
            }
        )
    )
    monkeypatch.setattr(
        patch, "read_reflow", lambda *args, **kwargs: (Cue(Fraction(0), Fraction(1), "whole"),)
    )
    return config_path


def test_cli_binds_immutable_inputs(config: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    output = config.parent / "new.fcpxml"
    monkeypatch.setattr(sys, "argv", ["window", str(config), str(output)])
    patch.main()
    receipt = json.loads(output.with_suffix(".caption-window.json").read_text())
    assert receipt["candidateOnly"] is True
    assert receipt["mediaRehashed"] is False
    assert receipt["nativeRenderVerified"] is False
    assert receipt["outputXML"]["sha256"] == file_digest(output)
    with pytest.raises(ReplayError, match="already exists"):
        patch.run(config, output)


@pytest.mark.parametrize(
    "field,value",
    [("version", 2), ("name", ""), ("recognitionSource", {"path": "/source", "sha256": 0})],
)
def test_bad_configuration(config: Path, field: str, value: object) -> None:
    raw = json.loads(config.read_text())
    raw[field] = value
    config.write_text(json.dumps(raw))
    with pytest.raises(ReplayError):
        patch.run(config, config.parent / "out.fcpxml")
