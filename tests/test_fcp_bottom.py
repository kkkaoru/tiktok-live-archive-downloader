import json
from dataclasses import replace
from fractions import Fraction
from pathlib import Path
from xml.etree import ElementTree as ET

import pytest
from test_motion_entrance import SOURCE

from replay import fcp_project, motion_entrance
from replay.caption_style_defaults import MOTION_ENTRANCE_START_SCALE
from replay.core import ReplayError
from replay.fcp_measured import CaptionAnchor, read_measured_captions
from replay.fcp_project import (
    Caption,
    CaptionStyle,
    Cut,
    MediaInfo,
    Timeline,
    build_fcpxml,
    caption_template,
)
from replay.stages import file_digest


@pytest.fixture
def title(tmp_path: Path) -> Path:
    path = tmp_path / "Motion Templates.localized/Titles.localized/Category/Title/Title.moti"
    path.parent.mkdir(parents=True)
    node = (
        '<scenenode factoryID="16"><parameter id="1"><parameter id="100">'
        '<parameter id="101"><parameter id="1" value="0"/>'
        '<parameter id="2" value="0"/></parameter></parameter></parameter>'
        '<parameter id="2"><parameter id="314" value="1"/>'
        '<parameter id="373"><parameter id="2" value="2"/></parameter>'
        '<parameter id="326" value="0"/></parameter></scenenode>'
    )
    path.write_text(
        "<ozml><scene><timemarkerset><timemarker><type>8</type></timemarker>"
        "</timemarkerset><layer>" + node * 3 + "</layer></scene></ozml>",
        encoding="utf-8",
    )
    return path


@pytest.fixture
def bottom_candidate(tmp_path: Path) -> Path:
    path = tmp_path / "bottom.json"
    path.write_text(
        json.dumps(
            {
                "version": 1,
                "candidateOnly": True,
                "canvas": [1080, 1920],
                "coordinateSystem": "top-left canvas; measured paragraph bottom origin",
                "policy": {"frame_rate": 30},
                "candidates": [
                    {"cue": {"text": "a\nb\nc", "start": 0, "end": 1}, "x": 540, "y": 600}
                ],
            }
        ),
        encoding="utf-8",
    )
    return path


def test_native_zero_position_with_motion_entrance(title: Path) -> None:
    title.write_text(motion_entrance.animate_entrance(SOURCE), encoding="utf-8")
    template = caption_template(title, anchor=CaptionAnchor.BOTTOM)
    assert template is not None
    title.write_text(
        title.read_text(encoding="utf-8").replace(
            f"<value>{MOTION_ENTRANCE_START_SCALE}</value>", "<value>0.5</value>"
        ),
        encoding="utf-8",
    )
    with pytest.raises(ReplayError, match="entrance curve"):
        caption_template(title, anchor=CaptionAnchor.BOTTOM)


def test_bottom_contract_preserves_three_lines(bottom_candidate: Path) -> None:
    result = read_measured_captions(
        bottom_candidate,
        sha256=file_digest(bottom_candidate),
        width=1080,
        height=1920,
        rate=30,
        duration=Fraction(1),
        anchor=CaptionAnchor.BOTTOM,
    )
    assert result[0].cue.text == "a\nb\nc"
    assert result[0].y == 600
    with pytest.raises(ValueError, match="coordinate contract"):
        read_measured_captions(
            bottom_candidate,
            sha256=file_digest(bottom_candidate),
            width=1080,
            height=1920,
            rate=30,
            duration=Fraction(1),
        )


@pytest.mark.parametrize(
    "old,new",
    [
        ('"x": 540', '"x": 541'),
        ("a\\nb\\nc", "a\\nb\\nc\\nd"),
        ("measured paragraph bottom origin", "calibrated text origin"),
    ],
)
def test_bottom_refuses_offcenter_four_lines_and_top_relabel(
    bottom_candidate: Path, old: str, new: str
) -> None:
    bottom_candidate.write_text(
        bottom_candidate.read_text(encoding="utf-8").replace(old, new), encoding="utf-8"
    )
    with pytest.raises(ValueError):
        read_measured_captions(
            bottom_candidate,
            sha256=file_digest(bottom_candidate),
            width=1080,
            height=1920,
            rate=30,
            duration=Fraction(1),
            anchor=CaptionAnchor.BOTTOM,
        )


@pytest.mark.parametrize("lines", [("a",), ("a", "b"), ("a", "b", "c")])
def test_bottom_stage_uses_same_origin_without_line_correction(
    title: Path, lines: tuple[str, ...]
) -> None:
    template = caption_template(title, anchor=CaptionAnchor.BOTTOM)
    timeline = Timeline(
        30,
        1080,
        1920,
        (Cut(Path("/source.mp4"), Fraction(0), Fraction(0), Fraction(1)),),
        (),
        (Caption(Fraction(0), Fraction(1), 540, 999, lines, "#FFFFFF", motion_origin_y=600),),
        CaptionStyle(88, "#FFFFFF", 1, "#FFFFFF", 1, "#000000"),
    )
    result, report = build_fcpxml(
        timeline,
        name="bottom",
        sources={Path("/source.mp4"): MediaInfo(Fraction(1), True, False)},
        template=template,
    )
    root = ET.fromstring(result)
    transform = root.find(".//title/adjust-transform")
    assert transform is not None
    assert transform.get("position") == "0 18.75"
    assert len(root.findall(".//title/text")) == 3
    assert report.captions == 1
    with pytest.raises(ReplayError, match="measured centered origins"):
        build_fcpxml(
            replace(timeline, captions=(replace(timeline.captions[0], motion_origin_y=None),)),
            name="bad",
            sources={},
            template=template,
        )


@pytest.mark.parametrize(
    "old,new",
    [
        ("<type>8</type>", "<type>7</type>"),
        ('id="314" value="1"', 'id="314" value="0"'),
        ('id="326" value="0"', 'id="326" value="5"'),
        ('id="1" value="0"', 'id="1" value="1"'),
        ('<parameter id="314" value="1"/>', ""),
        ('<parameter id="101">', '<parameter id="777">'),
    ],
)
def test_bottom_template_rejects_wrong_geometry_or_loop(title: Path, old: str, new: str) -> None:
    title.write_text(title.read_text(encoding="utf-8").replace(old, new), encoding="utf-8")
    with pytest.raises(ReplayError):
        caption_template(title, anchor=CaptionAnchor.BOTTOM)


def test_bottom_template_requires_three_nodes(title: Path) -> None:
    root = ET.fromstring(title.read_text(encoding="utf-8"))
    layer = root.find("scene/layer")
    assert layer is not None
    layer.remove(layer[0])
    title.write_text(ET.tostring(root, encoding="unicode"), encoding="utf-8")
    with pytest.raises(ReplayError, match="three Motion"):
        caption_template(title, anchor=CaptionAnchor.BOTTOM)


def test_per_stage_bottom_does_not_require_published_reveal_controls(title: Path) -> None:
    title.write_text(
        title.read_text(encoding="utf-8").replace(
            "<scene>",
            '<scene><publishSettings><target name="Caption Reveal Face" '
            'object="3" channel="./2/319/321"/></publishSettings>',
        ),
        encoding="utf-8",
    )
    assert caption_template(title, anchor=CaptionAnchor.BOTTOM).reveal_keys is None


def test_native_save_omits_default_zero_position_and_margin(title: Path) -> None:
    source = (
        title.read_text(encoding="utf-8")
        .replace(
            '<parameter id="100"><parameter id="101"><parameter id="1" value="0"/>'
            '<parameter id="2" value="0"/></parameter></parameter>',
            "",
        )
        .replace('<parameter id="326" value="0"/>', "")
    )
    title.write_text(source, encoding="utf-8")
    assert caption_template(title, anchor=CaptionAnchor.BOTTOM).text_objects == 3


def test_cli_bottom_requires_measured_contract_and_forwards_it(
    title: Path,
    bottom_candidate: Path,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    plan = tmp_path / "plan.json"
    plan.write_text("[]", encoding="utf-8")
    timeline = Timeline(
        30,
        1080,
        1920,
        (Cut(Path("/source.mp4"), Fraction(0), Fraction(0), Fraction(1)),),
        (),
        (),
        CaptionStyle(88, "#FFFFFF", 1, "#FFFFFF", 1, "#000000"),
    )

    def load(plan: object, directory: Path) -> Timeline:
        return timeline

    def probe(path: Path) -> MediaInfo:
        return MediaInfo(Fraction(1), True, False)

    monkeypatch.setattr(fcp_project, "load_timeline", load)
    output = tmp_path / "bottom.fcpxml"
    args = [
        str(plan),
        str(output),
        "--name",
        "bottom",
        "--caption-anchor",
        "bottom",
        "--caption-template",
        str(title),
    ]
    with pytest.raises(ReplayError, match="Bottom mode requires"):
        fcp_project.main(args, probe=probe)
    assert not output.exists()
    assert (
        fcp_project.main(
            args
            + [
                "--measured-captions",
                str(bottom_candidate),
                "--measured-captions-sha256",
                file_digest(bottom_candidate),
            ],
            probe=probe,
        )
        == 0
    )
    root = ET.fromstring(output.read_bytes())
    assert root.findtext(".//title/text") == "a\nb\nc"
    transform = root.find(".//title/adjust-transform")
    assert transform is not None
    assert transform.get("position") == "0 18.75"
