import sys
from pathlib import Path
from xml.etree import ElementTree as ET

import pytest

from replay import motion_template_timing as timing
from replay.core import ReplayError


@pytest.fixture
def template() -> str:
    return (
        '<ozml version="5.15"><template><flags>1</flags></template><scene>'
        "<sceneSettings><duration>240</duration><frameRate>30</frameRate>"
        "<NTSC>0</NTSC></sceneSettings>"
        '<timeRange offset="1 1 1 0" duration="1001 100 1 0"/>'
        '<playRange offset="0 1 1 0" duration="1001 100 1 0"/>'
        "<timemarkerset><timemarker><inpoint>1 30 1 0</inpoint>"
        "<type>7</type></timemarker></timemarkerset>"
        '<layer><scenenode factoryID="16"><timing in="120 30 1 0" '
        'out="179 30 1 0" offset="120 30 1 0"/>'
        '<parameter id="2"><parameter id="369"><text>First\nSecond\nThird</text>'
        '</parameter><parameter id="321"><keyframe><time>0 1 1 0</time>'
        "<value>1</value></keyframe></parameter></parameter>"
        "</scenenode></layer></scene><!--preserve intent--></ozml>"
    )


def test_exact_ranges_preserve_content_markers_and_local_animation(template: str) -> None:
    result = timing.normalize_ranges(template, frames=240, rate=30)
    root = ET.fromstring(result)
    time_range = root.find("scene/timeRange")
    play_range = root.find("scene/playRange")
    node_timing = root.find("scene/layer/scenenode/timing")
    assert time_range is not None and play_range is not None and node_timing is not None
    assert time_range.attrib == {"offset": "0 1 1 0", "duration": "240 30 1 0"}
    assert play_range.attrib == {"offset": "0 1 1 0", "duration": "240 30 1 0"}
    assert root.findtext("scene/timemarkerset/timemarker/type") == "7"
    assert root.findtext("scene/timemarkerset/timemarker/inpoint") == "1 30 1 0"
    assert root.findtext("scene/layer/scenenode/parameter/parameter/text") == "First\nSecond\nThird"
    assert node_timing.attrib == {"in": "120 30 1 0", "out": "179 30 1 0", "offset": "120 30 1 0"}
    assert root.findtext(".//keyframe/time") == "0 1 1 0"
    assert root.findtext(".//keyframe/value") == "1"
    assert "<!--preserve intent-->" in result
    assert timing.normalize_ranges(result, frames=240, rate=30) == result


def test_different_integral_clock(template: str) -> None:
    source = template.replace("<duration>240</duration>", "<duration>75</duration>")
    source = source.replace("<frameRate>30</frameRate>", "<frameRate>25</frameRate>")
    result = ET.fromstring(timing.normalize_ranges(source, frames=75, rate=25))
    time_range = result.find("scene/timeRange")
    assert time_range is not None
    assert time_range.get("duration") == "75 25 1 0"


@pytest.mark.parametrize("frames,rate", [(0, 30), (240, 0), (240, 61), (3601, 30), (True, 30)])
def test_invalid_explicit_clock(template: str, frames: int, rate: int) -> None:
    with pytest.raises(ReplayError, match="clock"):
        timing.normalize_ranges(template, frames=frames, rate=rate)


@pytest.mark.parametrize(
    "old,new",
    [
        ("<duration>240</duration>", "<duration>239</duration>"),
        ("<frameRate>30</frameRate>", "<frameRate>25</frameRate>"),
        ("<NTSC>0</NTSC>", "<NTSC>1</NTSC>"),
        ("<flags>1</flags>", "<flags>0</flags>"),
        ('<playRange offset="0 1 1 0" duration="1001 100 1 0"/>', ""),
        ("<timeRange offset=", '<timeRange unexpected="1" offset='),
        (
            '<timeRange offset="1 1 1 0" duration="1001 100 1 0"/>',
            '<timeRange offset="0 1 1 0" duration="1 1 1 0"><unknown/></timeRange>',
        ),
    ],
)
def test_no_silent_retiming_or_unknown_structure(template: str, old: str, new: str) -> None:
    with pytest.raises(ReplayError):
        timing.normalize_ranges(template.replace(old, new), frames=240, rate=30)


def test_cli_nonoverwriting_candidate(
    template: str, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(timing, "ROOT", tmp_path)
    source, output = tmp_path / "source.moti", tmp_path / "candidate.moti"
    source.write_text(template, encoding="utf-8")
    monkeypatch.setattr(
        sys, "argv", ["timing", str(source), str(output), "--frames", "240", "--rate", "30"]
    )
    timing.main()
    assert '<timeRange offset="0 1 1 0" duration="240 30 1 0"' in output.read_text(encoding="utf-8")
    assert source.read_text(encoding="utf-8") == template
    with pytest.raises(FileExistsError):
        timing.main()
    with pytest.raises(FileExistsError):
        timing.prepare(source, source, frames=240, rate=30)


def test_refuses_linked_and_external_paths(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    root = tmp_path / "checkout"
    root.mkdir()
    monkeypatch.setattr(timing, "ROOT", root)
    link = root / "link"
    link.symlink_to(tmp_path, target_is_directory=True)
    with pytest.raises(ReplayError, match="unlinked"):
        timing.prepare(link / "a.moti", root / "b.moti", frames=240, rate=30)
    with pytest.raises(ReplayError, match="unlinked"):
        timing.prepare(tmp_path / "a.moti", root / "b.moti", frames=240, rate=30)
    with pytest.raises(ReplayError, match=".moti"):
        timing.prepare(root / "a.xml", root / "b.moti", frames=240, rate=30)
    with pytest.raises(ReplayError, match="bounded"):
        timing.prepare(root / "missing.moti", root / "b.moti", frames=240, rate=30)


def test_refuses_oversize_source(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(timing, "ROOT", tmp_path)
    monkeypatch.setattr(timing, "MAX_XML_CHARACTERS", 5)
    source = tmp_path / "source.moti"
    source.write_text("123456", encoding="utf-8")
    with pytest.raises(ReplayError, match="bounded"):
        timing.prepare(source, tmp_path / "output.moti", frames=240, rate=30)


def test_linear_playback_removes_loop_and_preserves_poster(template: str) -> None:
    template = template.replace(
        "</timemarkerset>",
        "<timemarker><inpoint>0 30 1 0</inpoint><type>8</type></timemarker></timemarkerset>",
    )
    root = ET.fromstring(
        timing.normalize_ranges(template, frames=240, rate=30, linear_playback=True)
    )
    assert [node.text for node in root.findall("scene/timemarkerset/timemarker/type")] == ["8"]
    assert root.findtext("scene/timemarkerset/timemarker/inpoint") == "0 30 1 0"
    assert root.findtext("scene/layer/scenenode/parameter/parameter/text") == "First\nSecond\nThird"


@pytest.mark.parametrize("replacement", ["<type>99</type>", "<type>6</type>", "<type/>"])
def test_linear_playback_refuses_unknown_marker(template: str, replacement: str) -> None:
    with pytest.raises(ReplayError, match="Unverified"):
        timing.normalize_ranges(
            template.replace("<type>7</type>", replacement),
            frames=240,
            rate=30,
            linear_playback=True,
        )


def test_linear_playback_accepts_no_markers(template: str) -> None:
    root = ET.fromstring(template)
    scene = root.find("scene")
    markers = root.find("scene/timemarkerset")
    assert scene is not None and markers is not None
    scene.remove(markers)
    result = timing.normalize_ranges(
        ET.tostring(root, encoding="unicode"), frames=240, rate=30, linear_playback=True
    )
    assert "timemarkerset" not in result


@pytest.mark.parametrize(
    "replacement", ["<timemarkerset/><timemarkerset>", "<timemarkerset><unknown/>"]
)
def test_linear_playback_rejects_ambiguous_structure(template: str, replacement: str) -> None:
    with pytest.raises(ReplayError):
        timing.normalize_ranges(
            template.replace("<timemarkerset>", replacement),
            frames=240,
            rate=30,
            linear_playback=True,
        )


def test_linear_cli_is_explicit(
    template: str, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(timing, "ROOT", tmp_path)
    source, output = tmp_path / "source.moti", tmp_path / "linear.moti"
    source.write_text(template, encoding="utf-8")
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "timing",
            str(source),
            str(output),
            "--frames",
            "240",
            "--rate",
            "30",
            "--linear-playback",
        ],
    )
    timing.main()
    assert "<type>7</type>" not in output.read_text(encoding="utf-8")
    assert "<type>7</type>" in source.read_text(encoding="utf-8")
