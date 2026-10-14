import sys
from fractions import Fraction
from pathlib import Path
from xml.etree import ElementTree as ET

import pytest
from test_fcp_partition import source as source

from replay import fcp_output_window as output_window
from replay.core import ReplayError
from replay.fcp_caption_srt import caption_cues


def test_partial_source_title_audio_and_keyframes_are_preserved(source: Path) -> None:
    root = ET.parse(source).getroot()
    crop = ET.SubElement(root.findall(".//spine/asset-clip")[0], "adjust-crop", {"mode": "trim"})
    ET.SubElement(crop, "trim-rect", {"top": "25", "bottom": "5", "left": "4", "right": "3"})
    crop_before = ET.tostring(crop)
    before = ET.tostring(root)
    result = output_window.window(root, Fraction(1, 3), Fraction(5, 3))
    clips = result.findall(".//spine/asset-clip")
    assert [(item.get("offset"), item.get("start"), item.get("duration")) for item in clips] == [
        ("0/1s", "301/3s", "2/3s"),
        ("2/3s", "200/1s", "2/3s"),
    ]
    first_title = clips[0].findall("title")[0]
    assert first_title.get("start") == "10801/3s"
    assert first_title.get("offset") == "301/3s"
    assert first_title.findall(".//keyframe")[0].attrib == {"time": "3600s", "value": "2"}
    assert clips[0].findall("asset-clip")[1].get("start") == "16/3s"
    assert result.findall(".//sequence")[0].get("duration") == "4/3s"
    assert ET.tostring(result.findall(".//adjust-crop")[0]) == crop_before
    assert output_window._cues(result) == ((Fraction(0), Fraction(4, 3), "same\nphrase"),)
    assert ET.tostring(root) == before


def test_trim_a_title_that_started_inside_parent(source: Path) -> None:
    root = ET.parse(source).getroot()
    title = root.findall(".//title")[0]
    title.set("offset", "501/5s")
    title.set("duration", "2/5s")
    result = output_window.window(root, Fraction(1, 3), Fraction(6, 5))
    trimmed = result.findall(".//title")[0]
    assert trimmed.get("start") == "54002/15s"
    assert trimmed.get("duration") == "4/15s"
    assert trimmed.findall(".//keyframe")[0].get("time") == "3600s"


def test_inactive_audio_channel_metadata_is_preserved(source: Path) -> None:
    root = ET.parse(source).getroot()
    metadata = ET.SubElement(
        root.findall(".//spine/asset-clip")[0],
        "audio-channel-source",
        {"srcCh": "1, 2", "role": "dialogue.dialogue-1", "active": "0"},
    )
    result = output_window.window(root, Fraction(1, 3), Fraction(5, 3))
    assert ET.tostring(result.findall(".//audio-channel-source")[0]) == ET.tostring(metadata)
    metadata.set("active", "1")
    with pytest.raises(ReplayError, match="inactive source audio"):
        output_window.window(root, Fraction(1, 3), Fraction(5, 3))


def test_early_children_and_whole_clips_are_removed(source: Path) -> None:
    root = ET.parse(source).getroot()
    first = root.findall(".//spine/asset-clip")[0]
    first.findall("title")[0].set("duration", "1/5s")
    first.findall("asset-clip")[0].set("duration", "1/5s")
    result = output_window.window(root, Fraction(1, 2), Fraction(3, 2))
    assert len(result.findall(".//title")) == 1
    assert len(result.findall(".//spine/asset-clip")[0].findall("asset-clip")) == 1
    assert (
        len(output_window.window(root, Fraction(1), Fraction(2)).findall(".//spine/asset-clip"))
        == 1
    )


@pytest.mark.parametrize(
    "start,end",
    [
        (Fraction(-1), Fraction(1)),
        (Fraction(1), Fraction(1)),
        (Fraction(0), Fraction(3)),
        (Fraction(1, 7), Fraction(1)),
    ],
)
def test_invalid_window(source: Path, start: Fraction, end: Fraction) -> None:
    with pytest.raises(ReplayError):
        output_window.window(ET.parse(source).getroot(), start, end)


@pytest.mark.parametrize(
    "change", ["timeMap", "nested", "unknown", "gap", "format", "zero-frame", "static-nested"]
)
def test_unsupported_structures_fail(source: Path, change: str) -> None:
    root = ET.parse(source).getroot()
    clip = root.findall(".//spine/asset-clip")[0]
    if change == "timeMap":
        ET.SubElement(clip, "timeMap")
    elif change == "nested":
        ET.SubElement(clip.findall("asset-clip")[0], "audio")
    elif change == "unknown":
        ET.SubElement(clip, "unknown-adjustment")
    elif change == "gap":
        root.findall(".//spine/asset-clip")[1].set("offset", "3s")
    elif change == "format":
        root.findall(".//sequence")[0].set("format", "missing")
    elif change == "zero-frame":
        root.findall("./resources/format")[0].set("frameDuration", "0s")
    else:
        ET.SubElement(ET.SubElement(clip, "metadata"), "audio")
    with pytest.raises(ReplayError):
        output_window.window(root, Fraction(0), Fraction(1))


@pytest.mark.parametrize("limit", ["MAX_XML_BYTES", "MAX_XML_ELEMENTS"])
def test_bounded_xml(source: Path, monkeypatch: pytest.MonkeyPatch, limit: str) -> None:
    monkeypatch.setattr(output_window, limit, 1)
    with pytest.raises(ReplayError, match="budget"):
        output_window.window(ET.parse(source).getroot(), Fraction(0), Fraction(1))


def test_cli_exact_cues_and_no_overwrite(source: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    target = source.with_name("window.fcpxml")
    monkeypatch.setattr(sys, "argv", ["window", str(source), str(target), "1/3", "5/3"])
    output_window.main()
    assert caption_cues(target) == ((Fraction(0), Fraction(4, 3), "same\nphrase"),)
    with pytest.raises(FileExistsError):
        output_window.main()


def test_unsupported_nesting_outside_window_is_rejected(source: Path) -> None:
    root = ET.parse(source).getroot()
    child = root.findall(".//spine/asset-clip")[0].findall("asset-clip")[0]
    ET.SubElement(child, "audio")
    with pytest.raises(ReplayError, match="nested"):
        output_window.window(root, Fraction(1), Fraction(2))


def test_full_semantic_document_is_not_limited_by_short_import_budget(source: Path) -> None:
    root = ET.parse(source).getroot()
    metadata = ET.SubElement(root.findall(".//spine/asset-clip")[0], "metadata")
    metadata.text = "x" * (8 * 1024 * 1024)
    result = output_window.window(root, Fraction(1), Fraction(2))
    assert output_window._cues(result) == ((Fraction(0), Fraction(1), "same\nphrase"),)
