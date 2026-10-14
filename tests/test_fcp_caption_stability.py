import json
import sys
from copy import deepcopy
from pathlib import Path
from xml.etree import ElementTree as ET

import pytest

from replay import fcp_caption_stability as stability
from replay.core import ReplayError
from replay.fcp_caption_stability import stabilize_caption_y


def _document() -> ET.Element:
    return ET.fromstring("""<fcpxml><asset-clip>
      <title offset="10s" start="5s" duration="1s"><text>一行</text>
        <adjust-transform><param name="position"><keyframeAnimation>
          <keyframe time="5s" value="0 20" curve="linear"/>
          <keyframe time="11/2s" value="0 20.1" curve="linear"/>
          <keyframe time="6s" value="0 19.9" curve="linear"/>
        </keyframeAnimation></param></adjust-transform></title>
      <title offset="11s" start="5s" duration="1s"><text>次の語</text>
        <adjust-transform><param name="position"><keyframeAnimation>
          <keyframe time="5s" value="0 20.2" curve="linear"/>
          <keyframe time="6s" value="0 23" curve="linear"/>
        </keyframeAnimation></param></adjust-transform></title>
    </asset-clip></fcpxml>""")


def test_each_cue_holds_own_onset_preserving_text_time_x_and_animation() -> None:
    root = _document()
    stabilize_caption_y(root)
    assert [key.get("value") for key in root.findall(".//keyframe")] == [
        "0 20",
        "0 20",
        "0 20",
        "0 20.2",
        "0 20.2",
    ]
    assert [key.get("time") for key in root.findall(".//keyframe")] == [
        "5s",
        "11/2s",
        "6s",
        "5s",
        "6s",
    ]
    assert [key.get("curve") for key in root.findall(".//keyframe")] == ["linear"] * 5
    assert [title.findtext("text") for title in root.findall(".//title")] == ["一行", "次の語"]


def test_gap_resets_without_blending_disconnected_observations() -> None:
    root = _document()
    root.findall(".//title")[1].set("offset", "20s")
    stabilize_caption_y(root)
    assert root.findall(".//keyframe")[3].get("value") == "0 20.2"


def test_source_cut_resets_and_multiline_text_is_unchanged() -> None:
    root = _document()
    clip = root.find("asset-clip")
    assert clip is not None
    second = deepcopy(clip)
    second.findall("title")[0].findall("text")[0].text = "一行\n二行\n三行"
    second.findall(".//keyframe")[0].set("value", "0 30")
    root.append(second)
    stabilize_caption_y(root)
    assert root.findall("asset-clip")[1].findall(".//keyframe")[0].get("value") == "0 30"
    assert root.findall("asset-clip")[1].findtext("title/text") == "一行\n二行\n三行"


def test_new_taller_title_cannot_inherit_a_higher_position() -> None:
    root = _document()
    root.findall(".//keyframe")[3].set("value", "0 19.9")
    root.findall(".//keyframe")[4].set("value", "0 19.8")
    stabilize_caption_y(root)
    assert root.findall(".//keyframe")[3].get("value") == "0 19.9"


def test_one_two_three_one_lines_keep_a_shared_anchor_without_blank_padding() -> None:
    root = ET.Element("fcpxml")
    clip = ET.SubElement(root, "asset-clip")
    texts = ("一行", "一行\n二行", "一行\n二行\n三行", "次の一行")
    for index, text in enumerate(texts):
        title = ET.SubElement(
            clip, "title", offset=f"{index}s", start="3600s", duration="1s", ref="bottom-title"
        )
        for _ in range(3):
            ET.SubElement(title, "text").text = text
        animation = ET.SubElement(
            ET.SubElement(ET.SubElement(title, "adjust-transform"), "param", name="position"),
            "keyframeAnimation",
        )
        for time, y in (("3600s", "20"), ("7201/2s", "20.1"), ("3601s", "20")):
            ET.SubElement(animation, "keyframe", time=time, value=f"0 {y}", curve="linear")
    expected = deepcopy(root)
    stabilize_caption_y(root)
    assert {key.get("value") for key in root.findall(".//keyframe")} == {"0 20"}
    for key in expected.findall(".//keyframe"):
        key.set("value", "0 20")
    assert ET.tostring(root) == ET.tostring(expected)
    assert [node.text for node in root.findall(".//title/text")][::3] == list(texts)


def test_title_without_positions_is_preserved() -> None:
    root = ET.fromstring(
        '<fcpxml><asset-clip><title offset="0s" start="0s" duration="1s"/></asset-clip></fcpxml>'
    )
    stabilize_caption_y(root)
    assert root.find(".//title") is not None


@pytest.mark.parametrize("value", ["bad", "1 20", "0 nan", "0 1 2"])
def test_bad_vectors_fail(value: str) -> None:
    root = _document()
    root.findall(".//keyframe")[0].set("value", value)
    with pytest.raises(ReplayError):
        stabilize_caption_y(root)


@pytest.mark.parametrize("value", ["bad", "xs", "1/0s", "-1s", "30s"])
def test_bad_key_clocks_fail(value: str) -> None:
    root = _document()
    root.findall(".//keyframe")[0].set("time", value)
    with pytest.raises(ReplayError):
        stabilize_caption_y(root)


def test_cli_writes_new_pinned_candidate(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    root = _document()
    clip = root.find("asset-clip")
    assert clip is not None
    root.remove(clip)
    project = ET.SubElement(ET.SubElement(ET.SubElement(root, "library"), "event"), "project")
    project.append(clip)
    source = tmp_path / "source.fcpxml"
    source.write_bytes(ET.tostring(root))
    output = tmp_path / "candidate.fcpxml"
    monkeypatch.setattr(sys, "argv", ["stabilize", str(source), str(output), "--name", "Stable"])
    stability.main()
    saved_project = ET.parse(output).find(".//project")
    assert saved_project is not None
    assert saved_project.get("name") == "Stable"
    assert ET.parse(output).findall(".//keyframe")[1].get("value") == "0 20"
    assert ET.parse(source).findall(".//keyframe")[1].get("value") == "0 20.1"
    receipt = json.loads(output.with_suffix(".stability.json").read_text())
    assert receipt["nativeRenderVerified"] is False
    assert receipt["rawObservationsModified"] is False
    with pytest.raises(ReplayError, match="new stabilization"):
        stability.export_candidate(source, output, name="Stable")


def test_export_requires_one_project_and_bounded_input(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    source = tmp_path / "source.fcpxml"
    source.write_text("<fcpxml/>", encoding="utf-8")
    with pytest.raises(ReplayError, match="exactly one"):
        stability.export_candidate(source, tmp_path / "candidate.fcpxml", name="Stable")
    monkeypatch.setattr(stability, "MAX_TRACK_BYTES", 1)
    with pytest.raises(ReplayError, match="bounded"):
        stability.export_candidate(source, tmp_path / "candidate.fcpxml", name="Stable")


@pytest.mark.parametrize("text", ["一行", "一行\n二行", "一行\n二行\n三行", "次の一行"])
def test_onset_middle_last_anchor_is_exact_for_any_line_count(text: str) -> None:
    root = _document()
    node = root.find(".//title/text")
    assert node is not None
    node.text = text
    root.findall(".//keyframe")[1].set("value", "0 -10")
    root.findall(".//keyframe")[2].set("value", "0 35")
    stabilize_caption_y(root)
    assert [key.get("value") for key in root.findall(".//title")[0].findall(".//keyframe")] == [
        "0 20",
        "0 20",
        "0 20",
    ]
    assert node.text == text


@pytest.mark.parametrize("clock", ["11/2s", "5s"])
def test_missing_onset_or_duplicate_clock_is_rejected(clock: str) -> None:
    root = _document()
    keys = root.findall(".//title")[0].findall(".//keyframe")
    if clock == "11/2s":
        keys[0].set("time", clock)
    else:
        keys[1].set("time", clock)
    with pytest.raises(ReplayError):
        stabilize_caption_y(root)


def test_hold_applies_to_gap_titles_and_is_idempotent() -> None:
    root = _document()
    clip = root.find("asset-clip")
    assert clip is not None
    clip.tag = "gap"
    stabilize_caption_y(root)
    once = ET.tostring(root)
    stabilize_caption_y(root)
    assert ET.tostring(root) == once
    assert [key.get("value") for key in root.findall(".//keyframe")] == [
        "0 20",
        "0 20",
        "0 20",
        "0 20.2",
        "0 20.2",
    ]
