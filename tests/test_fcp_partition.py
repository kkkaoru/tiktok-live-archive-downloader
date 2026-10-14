import json
import sys
from fractions import Fraction
from pathlib import Path
from xml.etree import ElementTree as ET

import pytest

from replay import fcp_partition as partition
from replay.core import ReplayError


@pytest.fixture
def source(tmp_path: Path) -> Path:
    path = tmp_path / "full.fcpxml"
    path.write_text(
        """<fcpxml version="1.11"><resources>
      <format id="f" frameDuration="1/30s" width="1080" height="1920"/>
      </resources><library><event name="event"><project name="full" uid="original">
      <sequence format="f" duration="2s" tcStart="0s"><spine>
      <asset-clip offset="0s" start="100s" duration="1s" ref="video">
        <asset-clip name="mask" offset="100s" start="100s" duration="1s" ref="video">
          <filter-video ref="blur"/>
        </asset-clip>
        <asset-clip name="Voice" offset="100s" start="5s" duration="1s" ref="audio"/>
        <title offset="100s" start="3600s" duration="1s" ref="title">
          <param name="Y" key="k"><keyframeAnimation><keyframe time="3600s" value="2"/>
          <keyframe time="3601s" value="3"/></keyframeAnimation></param>
          <text>same\nphrase</text><text>same\nphrase</text><text>same\nphrase</text>
        </title>
      </asset-clip>
      <asset-clip offset="1s" start="200s" duration="1s" ref="video">
        <asset-clip name="Voice" offset="200s" start="6s" duration="1s" ref="audio"/>
        <title offset="200s" start="3600s" duration="1s" ref="title">
          <text>same\nphrase</text><text>same\nphrase</text><text>same\nphrase</text>
        </title>
      </asset-clip></spine></sequence></project></event></library></fcpxml>""",
        encoding="utf-8",
    )
    return path


def test_preserves_source_audio_titles_and_animation_at_all_endpoints(source: Path) -> None:
    original = source.read_bytes()
    parts = partition.partition(source, target_duration=Fraction(1), maximum_titles=1)
    assert [(part.start, part.duration, part.clips, part.titles) for part in parts] == [
        (Fraction(0), Fraction(1), 1, 1),
        (Fraction(1), Fraction(1), 1, 1),
    ]
    first = ET.fromstring(parts[0].content)
    second = ET.fromstring(parts[1].content)
    keyframe = first.find(".//keyframe")
    second_clip = second.find(".//spine/asset-clip")
    title = second.find(".//title")
    voice = second.find(".//asset-clip/asset-clip")
    assert (
        keyframe is not None and second_clip is not None and title is not None and voice is not None
    )
    assert keyframe.attrib == {"time": "3600s", "value": "2"}
    assert second_clip.attrib == {
        "offset": "0/1s",
        "start": "200s",
        "duration": "1s",
        "ref": "video",
    }
    assert title.get("offset") == "200s"
    assert voice.get("start") == "6s"
    original_second = ET.fromstring(original).findall(".//spine/asset-clip")[1]
    original_second.set("offset", "0/1s")
    assert ET.tostring(second_clip) == ET.tostring(original_second)
    assert source.read_bytes() == original


def test_write_manifest_exact_global_cues_and_cli(
    source: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    output = source.parent / "parts"
    monkeypatch.setattr(
        sys,
        "argv",
        ["partition", str(source), str(output), "--target-duration", "1", "--maximum-titles", "1"],
    )
    partition.main()
    manifest = json.loads((output / "manifest.json").read_text())
    assert manifest["globalCaptionCuesEqual"] is True
    assert manifest["duration"] == "2/1s"
    assert manifest["parts"][1]["globalStart"] == "1/1s"
    with pytest.raises(ReplayError, match="new canonical"):
        partition.write_parts(source, output, target_duration=Fraction(1), maximum_titles=1)


def test_duration_target_does_not_cut_a_long_existing_clip(source: Path) -> None:
    assert len(partition.partition(source, target_duration=Fraction(1, 2), maximum_titles=1)) == 2


def test_fractional_frame_endpoints_are_preserved_without_rounding(source: Path) -> None:
    source.write_text(
        source.read_text(encoding="utf-8")
        .replace('frameDuration="1/30s"', 'frameDuration="1001/30000s"')
        .replace('duration="2s"', 'duration="1001/15000s"')
        .replace('duration="1s"', 'duration="1001/30000s"')
        .replace('offset="1s"', 'offset="1001/30000s"'),
        encoding="utf-8",
    )
    parts = partition.partition(source, target_duration=Fraction(1001, 30000), maximum_titles=1)
    assert [(item.start, item.start + item.duration) for item in parts] == [
        (Fraction(0), Fraction(1001, 30000)),
        (Fraction(1001, 30000), Fraction(1001, 15000)),
    ]


@pytest.mark.parametrize("duration,titles", [(0, 1), (3601, 1), (1, 0), (1, 10001)])
def test_invalid_budget(source: Path, duration: int, titles: int) -> None:
    with pytest.raises(ReplayError, match="bounded"):
        partition.partition(source, target_duration=Fraction(duration), maximum_titles=titles)


@pytest.mark.parametrize("change", ["format", "frame", "audio", "offgrid"])
def test_invalid_clocks_rejected(source: Path, change: str) -> None:
    tree = ET.parse(source)
    root = tree.getroot()
    sequence = root.find(".//sequence")
    format_node = root.find(".//format")
    nested = root.find(".//asset-clip/asset-clip")
    assert sequence is not None and format_node is not None and nested is not None
    if change == "format":
        sequence.set("format", "missing")
    elif change == "frame":
        format_node.set("frameDuration", "0s")
    elif change == "audio":
        nested.set("duration", "2s")
    else:
        format_node.set("frameDuration", "2/3s")
    tree.write(source)
    with pytest.raises(ReplayError):
        partition.partition(source, target_duration=Fraction(1), maximum_titles=1)


def test_element_and_part_budgets(source: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(partition, "MAX_PARTS", 1)
    with pytest.raises(ReplayError, match="count"):
        partition.partition(source, target_duration=Fraction(1), maximum_titles=1)
    monkeypatch.setattr(partition, "MAX_PARTS", 10)
    monkeypatch.setattr(partition, "MAX_XML_ELEMENTS", 1)
    with pytest.raises(ReplayError, match="XML budget"):
        partition.partition(source, target_duration=Fraction(1), maximum_titles=1)


def test_required_structure() -> None:
    with pytest.raises(ReplayError, match="structure"):
        partition._required(ET.Element("empty"), "missing")


def test_library_locations_are_explicit_distinct_file_urls_without_bundle_creation(
    source: Path,
) -> None:
    directory = source.parent / "library destinations"
    directory.mkdir()
    parts = partition.partition(
        source,
        target_duration=Fraction(1),
        maximum_titles=1,
        library_directory=directory,
        library_name_prefix="Part",
    )
    first = ET.fromstring(parts[0].content).find("library")
    second = ET.fromstring(parts[1].content).find("library")
    assert first is not None and second is not None
    assert first.get("location") == (directory / "Part-0001.fcpbundle").as_uri()
    assert second.get("location") == (directory / "Part-0002.fcpbundle").as_uri()
    assert list(directory.iterdir()) == []
    with pytest.raises(ReplayError, match="canonical absolute"):
        partition.partition(
            source,
            target_duration=Fraction(1),
            maximum_titles=1,
            library_directory=directory / "missing",
        )
    with pytest.raises(ReplayError, match="prefix"):
        partition.partition(
            source, target_duration=Fraction(1), maximum_titles=1, library_name_prefix="../unsafe"
        )


def test_title_budget_never_splits_one_clip(source: Path) -> None:
    tree = ET.parse(source)
    root = tree.getroot()
    clip = root.findall(".//spine/asset-clip")[0]
    title = clip.find("title")
    assert title is not None
    title.set("duration", "1/2s")
    other = ET.fromstring(ET.tostring(title))
    other.set("offset", "201/2s")
    clip.append(other)
    tree.write(source)
    with pytest.raises(ReplayError, match="alone"):
        partition.partition(source, target_duration=Fraction(1), maximum_titles=1)
