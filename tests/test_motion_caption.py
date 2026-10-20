"""Synthetic Motion structure, not bundled Apple template content."""

import hashlib
import json
import runpy
import sys
from dataclasses import replace
from fractions import Fraction
from pathlib import Path
from xml.etree import ElementTree as ET

import pytest

from replay import motion_reveal_stages, motion_stages
from replay.core import ReplayError
from replay.motion_caption import (
    CaptionAppearance,
    CaptionPlacement,
    add_nonfade_entrance,
    animate_caption_template,
    build_caption_sequence,
    build_caption_template,
    position_caption_template,
    replace_caption_text,
    restyle_caption_template,
)

BASE = """<ozml version="5.13"><scene>
<sceneSettings><width>1920</width><height>1080</height></sceneSettings>
<publishSettings><version>2</version></publishSettings>
<layer id="999100001"><scenenode name="Title" id="999100100" factoryID="16">
<paragraphStyle id="999100120"/>
<style id="999100110">
<parameter id="83"><font>Example</font></parameter><parameter id="3" value="63"/>
<parameter id="14" flags="4176">
<parameter id="16">
<parameter id="1" value="1"/><parameter id="2" value="1"/><parameter id="3" value="1"/>
</parameter><parameter id="17"><parameter id="1">
<parameter id="999100010" factoryID="4">
<parameter id="1"><curve value="0"/></parameter>
<parameter id="2"><curve value="0.5"/></parameter><parameter id="3">
<parameter id="1"><curve value="1"/></parameter>
<parameter id="2"><curve value="1"/></parameter>
<parameter id="3"><curve value="1"/></parameter>
</parameter></parameter>
<parameter id="999100011" factoryID="4">
<parameter id="1"><curve value="1"/></parameter>
<parameter id="2"><curve value="0.5"/></parameter><parameter id="3">
<parameter id="1"><curve value="1"/></parameter>
<parameter id="2"><curve value="1"/></parameter>
<parameter id="3"><curve value="1"/></parameter>
</parameter></parameter>
</parameter></parameter>
</parameter>
<parameter id="21" flags="36944"/><parameter id="30" flags="36944"/>
<parameter id="400" value="999100110"/>
</style><styleRun style="999100110" offset="0" length="5"/>
</scenenode></layer></scene></ozml>"""

REFERENCE = """<ozml><scene><style>
<parameter id="14">
<parameter name="Fill with" id="15" flags="65616" value="0"/>
<parameter name="Color" id="16">
<parameter id="1" value="1"/><parameter id="2" value="1"/><parameter id="3" value="1"/>
</parameter>
</parameter>
<parameter id="21">
<parameter id="23">
<parameter id="1" value="1"/><parameter id="2" value="1"/><parameter id="3" value="1"/>
</parameter>
<parameter id="26" value="0.75"/><parameter id="27" value="5"/><parameter id="29" value="0"/>
<parameter id="75"><parameter id="1" value="0"/><parameter id="2" value="0"/></parameter>
</parameter>
<parameter id="30">
<parameter name="Color" id="32">
<parameter id="1" value="1"/><parameter id="2" value="1"/><parameter id="3" value="1"/>
</parameter>
<parameter id="35" value="0.7"/><parameter id="36" value="1"/><parameter id="50" value="1"/>
</parameter>
</style></scene></ozml>"""


TYPE_ON_BASE = BASE.replace(
    '<paragraphStyle id="999100120"/>',
    '<paragraphStyle id="999100120"/><parameter id="2" flags="8589938704">'
    '<parameter id="351" value="1"/><parameter id="323" value="0"/></parameter>',
)


def test_restyle_preserves_non_style_content_and_updates_both_objects() -> None:
    original = build_caption_template(TYPE_ON_BASE, REFERENCE, type_on=True)
    appearance = CaptionAppearance(
        top_srgb=(36 / 255, 85 / 255, 106 / 255),
        bottom_srgb=(40 / 255, 89 / 255, 109 / 255),
        font_size=50.72635135135135,
        white_border=6.1425,
        black_rim=3.07125,
        top=(0.01, 0.06, 0.1),
        bottom=(0.02, 0.085, 0.12),
    )
    changed = restyle_caption_template(original, appearance=appearance)
    root = ET.fromstring(changed)
    expected = ET.fromstring(
        build_caption_template(TYPE_ON_BASE, REFERENCE, appearance=appearance, type_on=True)
    )
    assert ET.tostring(root) == ET.tostring(expected)
    assert original == build_caption_template(TYPE_ON_BASE, REFERENCE, type_on=True)
    size = root.find(".//scenenode[@name='Caption Face']/style/parameter[@id='3']")
    assert size is not None
    assert size.get("value") == "50.72635135135135"


def test_restyle_cli_writes_new_file_only(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    import replay.motion_caption as module

    source = tmp_path / "source.moti"
    output = tmp_path / "styled.moti"
    original = build_caption_template(TYPE_ON_BASE, REFERENCE, type_on=True)
    source.write_text(original, encoding="utf-8")
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "motion_caption",
            str(source),
            str(output),
            "--font-size",
            "50",
            "--border",
            "6",
            "--rim",
            "3",
            "--fill-srgb",
            "#24556A",
            "--fill",
            "0.02",
            "0.085",
            "0.12",
            "--border-color",
            "0",
            "0",
            "0",
        ],
    )
    runpy.run_path(module.__file__, run_name="__main__")
    result = output.read_text(encoding="utf-8")
    assert result == restyle_caption_template(
        original,
        appearance=CaptionAppearance(
            top_srgb=(36 / 255, 85 / 255, 106 / 255),
            bottom_srgb=(36 / 255, 85 / 255, 106 / 255),
            font_size=50.0,
            white_border=6.0,
            black_rim=3.0,
            top=(0.02, 0.085, 0.12),
            bottom=(0.02, 0.085, 0.12),
            border_color=(0.0, 0.0, 0.0),
        ),
    )
    with pytest.raises(FileExistsError):
        module.main()
    assert output.read_text(encoding="utf-8") == result
    assert source.read_text(encoding="utf-8") == original


def test_restyle_rejects_invalid_appearance_and_layout() -> None:
    original = build_caption_template(BASE, REFERENCE)
    with pytest.raises(ReplayError, match="positive"):
        restyle_caption_template(
            original,
            appearance=CaptionAppearance(
                top_srgb=(36 / 255, 85 / 255, 106 / 255),
                bottom_srgb=(40 / 255, 89 / 255, 109 / 255),
                font_size=0,
            ),
        )
    with pytest.raises(ReplayError, match="two generated"):
        restyle_caption_template(
            BASE,
            appearance=CaptionAppearance(
                top_srgb=(36 / 255, 85 / 255, 106 / 255),
                bottom_srgb=(40 / 255, 89 / 255, 109 / 255),
            ),
        )
    with pytest.raises(ReplayError, match="5.13"):
        restyle_caption_template(
            original.replace('version="5.13"', 'version="5.99"'),
            appearance=CaptionAppearance(
                top_srgb=(36 / 255, 85 / 255, 106 / 255),
                bottom_srgb=(40 / 255, 89 / 255, 109 / 255),
            ),
        )


def test_restyle_retains_output_size_budget(monkeypatch: pytest.MonkeyPatch) -> None:
    original = build_caption_template(BASE, REFERENCE)
    monkeypatch.setattr("replay.motion_caption.MAX_XML_CHARACTERS", len(original))
    with pytest.raises(ReplayError, match="Restyled Motion"):
        restyle_caption_template(
            original,
            appearance=CaptionAppearance(
                top_srgb=(36 / 255, 85 / 255, 106 / 255),
                bottom_srgb=(40 / 255, 89 / 255, 109 / 255),
                top=(1 / 30, 1 / 30, 1 / 30),
            ),
        )


def test_bright_cyan_is_rejected_even_with_a_dark_border() -> None:
    appearance = CaptionAppearance(
        top_srgb=(184 / 255, 239 / 255, 1.0),
        bottom_srgb=(184 / 255, 239 / 255, 1.0),
        top=(184 / 255, 239 / 255, 1.0),
        bottom=(184 / 255, 239 / 255, 1.0),
        border_color=(0.0, 0.0, 0.0),
    )
    with pytest.raises(ReplayError, match="white contrast"):
        build_caption_template(BASE, REFERENCE, appearance=appearance)


@pytest.mark.parametrize("color", [(1.1, 0.0, 0.0), (0.0, -0.1, 0.0), (0.0, 0.0, float("nan"))])
def test_invalid_border_colors_rejected(color: tuple[float, float, float]) -> None:
    with pytest.raises(ReplayError):
        build_caption_template(
            BASE,
            REFERENCE,
            appearance=CaptionAppearance(
                top_srgb=(36 / 255, 85 / 255, 106 / 255),
                bottom_srgb=(40 / 255, 89 / 255, 109 / 255),
                border_color=color,
            ),
        )


@pytest.fixture
def text_template() -> str:
    base = TYPE_ON_BASE.replace(
        '<parameter id="351" value="1"/>',
        '<parameter id="369"><text>Title</text></parameter><parameter id="351" value="1"/>',
    )
    base = base.replace("<paragraphStyle", '<parameter id="1"/><paragraphStyle', 1)
    return build_caption_template(base, REFERENCE, type_on=True)


@pytest.fixture
def sequence_template(text_template: str) -> str:
    xml = text_template.replace(
        "</sceneSettings>",
        "<duration>300</duration><frameRate>30</frameRate><NTSC>0</NTSC></sceneSettings>",
    )
    root = ET.fromstring(xml)
    layer = root.find("scene/layer")
    assert layer is not None
    face, rim = layer.findall("scenenode")
    ET.SubElement(layer, "timing", offset="0 1 1 0")
    ET.SubElement(face, "timing", offset="0 1 1 0")
    ET.SubElement(rim, "timing", offset="0 1 1 0")
    return ET.tostring(root, encoding="unicode")


@pytest.fixture
def stage_plan() -> dict[str, object]:
    return {
        "frames": 30,
        "width": 1080,
        "height": 1920,
        "rate": 30,
        "expectedText": "Title",
        "placements": [
            {"text": "字", "startFrame": 0, "endFrame": 1, "x": 0, "y": 100},
            {"text": "字幕", "startFrame": 1, "endFrame": 30, "x": 0, "y": 100},
        ],
    }


def test_visible_stages_have_their_own_complete_text_and_clock(
    sequence_template: str,
    stage_plan: dict[str, object],
) -> None:
    root = ET.fromstring(motion_stages.prepare(sequence_template, stage_plan))
    assert [text.text for text in root.findall("scene/layer/scenenode//text")] == [
        "字",
        "字",
        "字幕",
        "字幕",
    ]
    assert [timing.get("out") for timing in root.findall("scene/layer/scenenode/timing")] == [
        "0 30 1 0",
        "0 30 1 0",
        "29 30 1 0",
        "29 30 1 0",
    ]
    assert [
        fade.get("value")
        for fade in root.findall(
            "scene/layer/scenenode/parameter[@id='2']/parameter[@id='319']/parameter[@id='322']"
        )
    ] == ["0", "0", "0", "0"]


def test_stage_entrance_is_visible_uniform_scale_without_fade(
    sequence_template: str,
    stage_plan: dict[str, object],
) -> None:
    prepared = motion_stages.prepare(sequence_template, stage_plan)
    root = ET.fromstring(prepared)
    curves = root.findall(
        "scene/layer/scenenode/parameter[@id='1']/parameter[@id='100']/"
        "parameter[@id='105']/parameter/curve"
    )

    assert len(curves) == 8
    assert [
        node.get("name")
        for node in root.findall(
            "scene/layer/scenenode/parameter[@id='1']/parameter[@id='100']/"
            "parameter[@id='105']/parameter"
        )
    ] == ["X", "Y"] * 4
    for curve in curves:
        keyframes = curve.findall("keypoint")
        assert [
            Fraction(
                int((item.findtext("time") or "").split()[0]),
                int((item.findtext("time") or "").split()[1]),
            )
            for item in keyframes
        ] == [Fraction(0), Fraction(1, 5)]
        assert [float(item.findtext("value") or "nan") for item in keyframes] == [0.78, 1.0]
    assert [
        fade.get("value")
        for fade in root.findall(
            "scene/layer/scenenode/parameter[@id='2']/parameter[@id='319']/parameter[@id='322']"
        )
    ] == ["0"] * 4
    with pytest.raises(ReplayError, match="already animated"):
        add_nonfade_entrance(prepared, rate=30)


def test_low_resolution_motion_plan_uses_reference_canvas_without_moving_titles(
    sequence_template: str,
    stage_plan: dict[str, object],
) -> None:
    plan = stage_plan | {
        "width": 1280,
        "height": 720,
        "placements": [{"text": "どう", "startFrame": 0, "endFrame": 11, "x": 884, "y": 9}],
    }
    root = ET.fromstring(motion_stages.prepare(sequence_template, plan))

    assert root.findtext("scene/sceneSettings/width") == "1920"
    assert root.findtext("scene/sceneSettings/height") == "1080"
    assert [body.text for body in root.findall(".//text")] == ["どう", "どう"]
    normalized_sizes = [
        size.get("value") for size in root.findall("scene/layer/scenenode/style/parameter[@id='3']")
    ]
    direct = ET.fromstring(
        build_caption_sequence(
            sequence_template,
            (
                CaptionPlacement(
                    "どう",
                    0,
                    11,
                    ((Fraction(0), Fraction(1)), (Fraction(11, 30), Fraction(1))),
                    884,
                    9,
                ),
            ),
            expected_text="Title",
            frames=30,
            width=1280,
            height=720,
            rate=30,
            match_fcp_scale=True,
            smooth=False,
        )
    )
    assert normalized_sizes == [
        size.get("value")
        for size in direct.findall("scene/layer/scenenode/style/parameter[@id='3']")
    ]
    positions = root.findall(
        "scene/layer/scenenode/parameter[@id='1']/parameter[@id='100']/parameter[@id='101']"
    )
    assert [
        [(item.get("name"), item.get("value")) for item in position.findall("parameter")]
        for position in positions
    ] == [
        [("X", "884.0"), ("Y", "9.0")],
        [("X", "884.0"), ("Y", "9.0")],
    ]


def test_low_resolution_portrait_motion_plan_uses_reference_canvas(
    sequence_template: str,
    stage_plan: dict[str, object],
) -> None:
    plan = stage_plan | {
        "width": 720,
        "height": 1280,
        "placements": [{"text": "字", "startFrame": 0, "endFrame": 1, "x": 0, "y": 100}],
    }

    root = ET.fromstring(motion_stages.prepare(sequence_template, plan))

    assert root.findtext("scene/sceneSettings/width") == "1080"
    assert root.findtext("scene/sceneSettings/height") == "1920"


@pytest.mark.parametrize(
    ("coordinate_space", "source_x", "source_y", "expected_x", "expected_y"),
    [
        ("canvas-top-left", 884, 640, 366, -420),
        ("canvas-bottom-left", 884, 80, 366, -420),
    ],
)
def test_motion_stage_converts_declared_canvas_coordinates_after_normalization(
    sequence_template: str,
    stage_plan: dict[str, object],
    coordinate_space: str,
    source_x: int,
    source_y: int,
    expected_x: int,
    expected_y: int,
) -> None:
    plan = stage_plan | {
        "width": 1280,
        "height": 720,
        "coordinateSpace": coordinate_space,
        "placements": [
            {
                "text": "どう",
                "startFrame": 0,
                "endFrame": 11,
                "x": source_x,
                "y": source_y,
            }
        ],
    }

    root = ET.fromstring(motion_stages.prepare(sequence_template, plan))
    positions = root.findall(
        "scene/layer/scenenode/parameter[@id='1']/parameter[@id='100']/parameter[@id='101']"
    )

    assert [
        [(item.get("name"), item.get("value")) for item in position.findall("parameter")]
        for position in positions
    ] == [
        [("X", str(float(expected_x))), ("Y", str(float(expected_y)))],
        [("X", str(float(expected_x))), ("Y", str(float(expected_y)))],
    ]


@pytest.mark.parametrize(
    "patch",
    [
        {"extra": True},
        {"coordinateSpace": "unknown"},
        {"coordinateSpace": []},
        {"rate": 0},
        {"rate": True},
        {"frames": 1.0},
        {"expectedText": 42},
        {"placements": []},
        {"placements": "invalid"},
        {"placements": [{}]},
        {"placements": [{"text": 5, "startFrame": 0, "endFrame": 1, "x": 0, "y": 0}]},
        {"placements": [{"text": "A", "startFrame": 0, "endFrame": 1, "x": True, "y": 0}]},
        {"placements": [{"text": "A", "startFrame": 0, "endFrame": 1, "x": 0, "y": "x"}]},
        {"placements": [{"text": "A", "startFrame": 0, "endFrame": 1, "x": 0, "y": float("inf")}]},
    ],
)
def test_stage_plan_rejects_invalid_fields(
    sequence_template: str,
    stage_plan: dict[str, object],
    patch: dict[str, object],
) -> None:
    with pytest.raises(ReplayError):
        motion_stages.prepare(sequence_template, stage_plan | patch)


def test_stage_cli_preserves_inputs_and_refuses_overwrite(
    sequence_template: str,
    stage_plan: dict[str, object],
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    template, plan, output = (
        tmp_path / "template.moti",
        tmp_path / "plan.json",
        tmp_path / "out.moti",
    )
    template.write_text(sequence_template, encoding="utf-8")
    plan.write_text(json.dumps(stage_plan), encoding="utf-8")
    monkeypatch.setattr(sys, "argv", ["motion_stages", str(template), str(plan), str(output)])
    runpy.run_path(motion_stages.__file__, run_name="__main__")
    assert output.is_file()
    with pytest.raises(FileExistsError):
        motion_stages.main()
    assert template.read_text(encoding="utf-8") == sequence_template
    assert json.loads(plan.read_text(encoding="utf-8")) == stage_plan
    monkeypatch.setattr(
        sys, "argv", ["motion_stages", str(tmp_path / "missing"), str(plan), str(output)]
    )
    with pytest.raises(ReplayError, match="bounded regular"):
        motion_stages.main()


def test_one_frame_constant_pop_in_holds_identical_keys(sequence_template: str) -> None:
    placement = CaptionPlacement(
        "A",
        0,
        1,
        ((Fraction(0), Fraction(1, 2)), (Fraction(1, 30), Fraction(1, 2))),
        0,
        0,
    )
    xml = build_caption_sequence(
        sequence_template,
        (placement,),
        expected_text="Title",
        frames=1,
        width=1080,
        height=1920,
        smooth=False,
    )
    nodes = ET.fromstring(xml).findall("scene/layer/scenenode")
    assert len(nodes) == 2
    assert [t.get("out") for t in ET.fromstring(xml).findall("scene/layer/scenenode/timing")] == [
        "0 30 1 0",
        "0 30 1 0",
    ]


@pytest.mark.parametrize("smooth,last", [(True, Fraction(1, 2)), (False, Fraction(1))])
def test_exclusive_end_cannot_add_fade_or_late_reveal(
    sequence_template: str,
    smooth: bool,
    last: Fraction,
) -> None:
    placement = CaptionPlacement(
        "A",
        0,
        1,
        ((Fraction(0), Fraction(1, 2)), (Fraction(1, 30), last)),
        0,
        0,
    )
    with pytest.raises(ReplayError):
        build_caption_sequence(
            sequence_template,
            (placement,),
            expected_text="Title",
            frames=1,
            width=1080,
            height=1920,
            smooth=smooth,
        )


def test_sequence_explicit_fcp_scale_changes_only_scalar_geometry(sequence_template: str) -> None:
    source = (
        sequence_template.replace('value="27.87162162162162"', 'value="20"')
        .replace('value="3.375"', 'value="3"')
        .replace('value="5.0625"', 'value="5"')
        .replace('value="0.5625"', 'value="1"')
        .replace('value="1.125"', 'value="2"')
    )
    placement = CaptionPlacement(
        "AB", 0, 30, ((Fraction(0), Fraction(0)), (Fraction(1, 2), Fraction(1))), 10, 20
    )
    document = build_caption_sequence(
        source,
        (placement,),
        expected_text="Title",
        frames=30,
        width=1080,
        height=2160,
        match_fcp_scale=True,
    )
    nodes = ET.fromstring(document).findall("scene/layer/scenenode")
    assert [n.findall("style/parameter[@id='3']")[0].get("value") for n in nodes] == [
        "40.0",
        "40.0",
    ]
    assert [
        n.findall("style/parameter[@id='30']/parameter[@id='36']")[0].get("value") for n in nodes
    ] == ["6.0", "10.0"]
    shadow = nodes[1].find("style/parameter[@id='21']")
    assert shadow is not None
    assert shadow.findall("parameter[@id='27']")[0].get("value") == "2.0"
    assert [c.get("value") for c in shadow.findall("parameter[@id='75']/parameter")] == [
        "4.0",
        "4.0",
    ]
    default = ET.fromstring(
        build_caption_sequence(
            source,
            (placement,),
            expected_text="Title",
            frames=30,
            width=1080,
            height=2160,
        )
    )
    # Compare scaled controls independently from unchanged text and output origins.
    changed = ET.fromstring(document)
    paths = (
        ".//style/parameter[@id='3']",
        ".//style/parameter[@id='30']/parameter[@id='36']",
        ".//style/parameter[@id='21']/parameter[@id='27']",
        ".//style/parameter[@id='21']/parameter[@id='75']/parameter",
    )
    default_values = [c.get("value") for path in paths for c in default.findall(path)]
    changed_values = [c.get("value") for path in paths for c in changed.findall(path)]
    assert default_values == ["20", "20", "3", "5", "1", "2", "2"]
    assert changed_values == ["40.0", "40.0", "6.0", "10.0", "2.0", "4.0", "4.0"]
    assert [n.findtext("parameter[@id='2']/parameter[@id='369']/text") for n in nodes] == [
        "AB",
        "AB",
    ]
    assert [
        c.get("value")
        for n in nodes
        for c in n.findall("parameter[@id='1']/parameter[@id='100']/parameter[@id='101']/parameter")
    ] == ["10", "20", "10", "20"]
    assert 'value="20"' in source


@pytest.mark.parametrize("height", ["bad", "0", "-1"])
def test_sequence_fcp_scale_rejects_invalid_reference_height(
    sequence_template: str, height: str
) -> None:
    source = sequence_template.replace("<height>1080</height>", f"<height>{height}</height>")
    placement = CaptionPlacement(
        "AB", 0, 30, ((Fraction(0), Fraction(0)), (Fraction(1, 2), Fraction(1))), 0, 0
    )
    with pytest.raises(ReplayError, match="reference canvas height"):
        build_caption_sequence(
            source,
            (placement,),
            expected_text="Title",
            frames=30,
            width=1080,
            height=1920,
            match_fcp_scale=True,
        )


@pytest.mark.parametrize("value", ["bad", "nan", "inf", "-1", "10001", "6000"])
def test_sequence_fcp_scale_rejects_invalid_or_oversized_geometry(
    sequence_template: str, value: str
) -> None:
    source = sequence_template.replace('value="27.87162162162162"', f'value="{value}"')
    placement = CaptionPlacement(
        "AB", 0, 30, ((Fraction(0), Fraction(0)), (Fraction(1, 2), Fraction(1))), 0, 0
    )
    with pytest.raises(ReplayError, match="geometry"):
        build_caption_sequence(
            source,
            (placement,),
            expected_text="Title",
            frames=30,
            width=1080,
            height=2160,
            match_fcp_scale=True,
        )


@pytest.mark.parametrize("scale", [1, None, "true"])
def test_sequence_fcp_scale_requires_boolean(sequence_template: str, scale: bool) -> None:
    with pytest.raises(ReplayError, match="must be a boolean"):
        build_caption_sequence(
            sequence_template,
            (),
            expected_text="Title",
            frames=30,
            width=1080,
            height=1920,
            match_fcp_scale=scale,
        )


@pytest.mark.parametrize("smooth", [True, False])
def test_sequence_uses_explicit_support_and_local_keys(
    sequence_template: str, smooth: bool
) -> None:
    first = CaptionPlacement(
        "AB", 0, 30, ((Fraction(0), Fraction(0)), (Fraction(1, 2), Fraction(1))), 0, 0
    )
    second = replace(first, text="日本", start_frame=45, end_frame=75, x=100, y=200)
    result = build_caption_sequence(
        sequence_template,
        (first, second),
        expected_text="Title",
        frames=90,
        width=1080,
        height=1920,
        smooth=smooth,
    )
    root = ET.fromstring(result)
    fade_values = [node.get("value") for node in root.findall(".//parameter[@id='322']")]
    assert fade_values == (["1", "1", "1", "1"] if smooth else ["0", "0", "0", "0"])
    assert root.findtext("scene/sceneSettings/duration") == "90"
    assert root.findtext("scene/sceneSettings/NTSC") == "0"
    parent_timing = root.find("scene/layer/timing")
    assert parent_timing is not None
    assert parent_timing.attrib == {"offset": "0 1 1 0", "in": "0 1 1 0", "out": "89 30 1 0"}
    face1, rim1, face2, rim2 = root.findall("scene/layer/scenenode")
    assert len({face1.get("id"), rim1.get("id"), face2.get("id"), rim2.get("id")}) == 4
    face_timing = face2.find("timing")
    rim_timing = rim2.find("timing")
    assert face_timing is not None and rim_timing is not None
    assert face_timing.attrib == {"offset": "45 30 1 0", "in": "45 30 1 0", "out": "74 30 1 0"}
    assert ET.tostring(face_timing) == ET.tostring(rim_timing)
    assert face2.findtext("parameter[@id='2']/parameter[@id='369']/text") == "日本"
    assert (
        face2.findtext(
            "parameter[@id='2']/parameter[@id='319']/parameter[@id='321']/curve/keypoint/time"
        )
        == "0 1 1 0"
    )
    assert "<text>Title</text>" in sequence_template


@pytest.mark.parametrize("smooth", [1, None, "false"])
def test_sequence_rejects_non_boolean_animation(sequence_template: str, smooth: bool) -> None:
    with pytest.raises(ReplayError, match="smoothing must be a boolean"):
        build_caption_sequence(
            sequence_template,
            (),
            expected_text="Title",
            frames=30,
            width=1080,
            height=1920,
            smooth=smooth,
        )


@pytest.mark.parametrize(
    "frames,width,height,rate",
    [(0, 1080, 1920, 30), (90, 0, 1920, 30), (90, 1080, 1920, 0), (7201, 1080, 1920, 60)],
)
def test_sequence_invalid_settings(
    sequence_template: str, frames: int, width: int, height: int, rate: int
) -> None:
    placement = CaptionPlacement("AB", 0, 30, ((Fraction(0), Fraction(1)),), 0, 0)
    with pytest.raises(ReplayError, match="bounded Motion"):
        build_caption_sequence(
            sequence_template,
            (placement,),
            expected_text="Title",
            frames=frames,
            width=width,
            height=height,
            rate=rate,
        )


def test_sequence_compacts_sparse_ids_without_expanding_the_allowed_range(
    sequence_template: str,
) -> None:
    root = ET.fromstring(sequence_template)
    root.set("id", "999199999")
    placements = tuple(
        CaptionPlacement(
            "AB",
            index * 30,
            (index + 1) * 30,
            ((Fraction(0), Fraction(0)), (Fraction(1, 2), Fraction(1))),
            0,
            0,
        )
        for index in range(17)
    )
    output = ET.fromstring(
        build_caption_sequence(
            ET.tostring(root, encoding="unicode"),
            placements,
            expected_text="Title",
            frames=510,
            width=1080,
            height=1920,
        )
    )
    nodes = output.findall("scene/layer/scenenode")
    assert len(nodes) == 34
    assert len({node.get("id") for node in nodes}) == 34
    assert max(int(node.attrib["id"]) for node in nodes) < 1_000_000_000
    style = nodes[-1].find("style")
    style_run = nodes[-1].find("styleRun")
    assert style is not None and style_run is not None
    assert style_run.get("style") == style.get("id")
    root.set("id", "999999999")
    with pytest.raises(ReplayError, match="ID range exhausted"):
        build_caption_sequence(
            ET.tostring(root, encoding="unicode"),
            placements,
            expected_text="Title",
            frames=510,
            width=1080,
            height=1920,
        )


def test_sequence_refuses_overlap_or_key_outside_support(sequence_template: str) -> None:
    placement = CaptionPlacement(
        "AB", 0, 30, ((Fraction(0), Fraction(0)), (Fraction(2), Fraction(1))), 0, 0
    )
    with pytest.raises(ReplayError, match="sentence support"):
        build_caption_sequence(
            sequence_template,
            (placement,),
            expected_text="Title",
            frames=90,
            width=1080,
            height=1920,
        )
    placement = replace(
        placement, samples=((Fraction(0), Fraction(0)), (Fraction(1, 2), Fraction(1)))
    )
    with pytest.raises(ReplayError, match="overlapping"):
        build_caption_sequence(
            sequence_template,
            (placement, placement),
            expected_text="Title",
            frames=90,
            width=1080,
            height=1920,
        )


def test_position_sets_both_origins_without_changing_text(text_template: str) -> None:
    result = position_caption_template(text_template, x=200, y=100)
    root = ET.fromstring(result)
    positions = root.findall(
        "scene/layer/scenenode/parameter[@id='1']/parameter[@id='100']/parameter[@id='101']"
    )
    assert [
        [child.attrib for child in position.findall("parameter")] for position in positions
    ] == [
        [
            {"name": "X", "id": "1", "flags": "8606711824", "default": "0", "value": "200"},
            {"name": "Y", "id": "2", "flags": "8606711824", "default": "0", "value": "100"},
        ],
        [
            {"name": "X", "id": "1", "flags": "8606711824", "default": "0", "value": "200"},
            {"name": "Y", "id": "2", "flags": "8606711824", "default": "0", "value": "100"},
        ],
    ]
    assert [body.text for body in root.findall(".//text")] == ["Title", "Title"]
    assert ET.fromstring(text_template).findall(".//parameter[@id='100']") == []


@pytest.mark.parametrize(
    ("x", "y"), [(float("nan"), 0), (0, float("inf")), (10001, 0), (0, -10001), (True, 0)]
)
def test_position_rejects_invalid_coordinates(text_template: str, x: float, y: float) -> None:
    with pytest.raises(ReplayError, match="finite"):
        position_caption_template(text_template, x=x, y=y)


def test_position_refuses_an_existing_transform(text_template: str) -> None:
    positioned = position_caption_template(text_template, x=0, y=0)
    with pytest.raises(ReplayError, match="already has a transform"):
        position_caption_template(positioned, x=1, y=1)


def test_position_checks_serialized_budget(
    text_template: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr("replay.motion_caption.MAX_XML_CHARACTERS", len(text_template) + 10)
    with pytest.raises(ReplayError, match="Positioned Motion document"):
        position_caption_template(text_template, x=0, y=0)


def test_text_copy_preserves_combining_marks_and_both_styles(text_template: str) -> None:
    result = replace_caption_text(text_template, expected_text="Title", text="日本語\nAか\u3099B")
    root = ET.fromstring(result)
    assert [body.text for body in root.findall(".//text")] == [
        "日本語\nAか\u3099B",
        "日本語\nAか\u3099B",
    ]
    assert [run.get("length") for run in root.findall(".//styleRun")] == ["8", "8"]
    assert [run.get("style") for run in root.findall(".//styleRun")] == ["999100230", "999100110"]
    assert [body.text for body in ET.fromstring(text_template).findall(".//text")] == [
        "Title",
        "Title",
    ]


def test_text_copy_escapes_xml_metacharacters(text_template: str) -> None:
    root = ET.fromstring(replace_caption_text(text_template, expected_text="Title", text="<&>"))
    assert [body.text for body in root.findall(".//text")] == ["<&>", "<&>"]
    assert [run.get("length") for run in root.findall(".//styleRun")] == ["3", "3"]


@pytest.mark.parametrize("text", ["", "a" * 121, "\x00", "\r", "\t", "\ud800", "\uffff", "😀"])
def test_text_copy_rejects_unverified_or_invalid_text(text_template: str, text: str) -> None:
    with pytest.raises(ReplayError, match="BMP units"):
        replace_caption_text(text_template, expected_text="Title", text=text)


@pytest.mark.parametrize(
    ("old", "new", "message"),
    [
        ('version="5.13"', 'version="5.15"', "5.13"),
        ('name="Caption Face"', 'name="Other"', "two generated"),
        ("<text>Title</text>", "<text>Wrong</text>", "does not match"),
        ("<text>Title</text>", "<text>Title<character/></text>", "does not match"),
        ("<text>Title</text>", "<text>Title</text><text>extra</text>", "structure"),
        ("<text>Title</text>", "<text>Title</text><character/>", "does not match"),
        ("<text>Title</text>", "<text>Title</text><kerning/>", "does not match"),
        ("<text>Title</text>", '<text>Title</text><object value="84"/>', "does not match"),
        (
            "<text>Title</text>",
            '<text>Title</text><parameter name="Kerning" value="0"/>',
            "does not match",
        ),
        ('offset="0"', 'offset="1"', "does not match"),
        ('length="5"', 'length="4"', "does not match"),
        ('style="999100230"', 'style="wrong"', "does not match"),
        (
            'id="321" flags="8589934608" default="1" value="1"',
            'id="321" flags="8589934608" default="1" value="0"',
            "before generating",
        ),
    ],
)
def test_text_copy_refuses_changed_contract(
    text_template: str, old: str, new: str, message: str
) -> None:
    assert old in text_template
    with pytest.raises(ReplayError, match=message):
        replace_caption_text(text_template.replace(old, new, 1), expected_text="Title", text="字幕")


def test_text_copy_refuses_stale_expected_body(text_template: str) -> None:
    with pytest.raises(ReplayError, match="does not match"):
        replace_caption_text(text_template, expected_text="Other", text="字幕")


def test_text_copy_checks_serialized_output_budget(
    text_template: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr("replay.motion_caption.MAX_XML_CHARACTERS", len(text_template) + 10)
    with pytest.raises(ReplayError, match="Updated Motion document"):
        replace_caption_text(text_template, expected_text="Title", text="&" * 120)


def test_template_has_two_text_objects_and_remapped_references() -> None:
    root = ET.fromstring(build_caption_template(BASE, REFERENCE))
    upper, lower = root.findall("scene/layer/scenenode")
    assert upper.attrib == {"name": "Caption Face", "id": "999100220", "factoryID": "16"}
    assert lower.attrib == {"name": "Caption Outer Rim", "id": "999100100", "factoryID": "16"}
    assert upper.findall("styleRun")[0].attrib["style"] == "999100230"
    assert upper.findall("paragraphStyle")[0].attrib["id"] == "999100240"
    assert upper.findall("style/parameter[@id='400']")[0].attrib["value"] == "999100110"
    assert root.findall("scene/publishSettings/target")[0].attrib == {
        "object": "999100220",
        "channel": "./5/999100230/14/15",
        "name": "Caption Fill Mode",
    }
    assert (
        root.findall(".//scenenode[@factoryID='16']")[1].findall("style")[0].attrib["id"]
        == "999100110"
    )
    assert "Caption Face" not in BASE


def test_template_keeps_blue_gradient_white_border_black_rim_and_shadow() -> None:
    root = ET.fromstring(build_caption_template(BASE, REFERENCE))
    upper, lower = root.findall("scene/layer/scenenode/style")
    assert upper.findall("parameter[@id='83']/font")[0].text == "HiraginoSans-W6"
    assert float(upper.findall("parameter[@id='3']")[0].attrib["value"]) == pytest.approx(
        27.8716216216
    )
    assert upper.findall("parameter[@id='14']/parameter[@id='15']")[0].attrib["value"] == "1"
    first, last = upper.findall(
        "parameter[@id='14']/parameter[@id='17']/parameter[@id='1']/parameter"
    )
    assert first.findall("parameter[@id='1']/curve")[0].attrib["value"] == "0.15"
    assert first.findall("parameter[@id='2']/curve")[0].attrib["value"] == "0.8"
    assert (
        first.findall("parameter[@id='3']/parameter[@id='2']/curve")[0].attrib["value"] == "0.055"
    )
    assert last.findall("parameter[@id='1']/curve")[0].attrib["value"] == "0.7"
    assert last.findall("parameter[@id='3']/parameter[@id='3']/curve")[0].attrib["value"] == "0.12"
    assert upper.findall("parameter[@id='30']")[0].attrib["flags"] == "4176"
    assert (
        upper.findall("parameter[@id='30']/parameter[@id='32']/parameter[@id='1']")[0].attrib[
            "value"
        ]
        == "1.0"
    )
    assert upper.findall("parameter[@id='30']/parameter[@id='36']")[0].attrib["value"] == "3.375"
    assert upper.findall("parameter[@id='30']/parameter[@id='50']")[0].attrib["value"] == "0.0"
    assert len(lower.findall("parameter[@id='14']/parameter[@id='16']")) == 1
    assert (
        lower.findall("parameter[@id='14']/parameter[@id='16']/parameter[@id='1']")[0].attrib[
            "value"
        ]
        == "0"
    )
    assert lower.findall("parameter[@id='30']/parameter[@id='36']")[0].attrib["value"] == "5.0625"
    assert (
        lower.findall("parameter[@id='30']/parameter[@id='32']/parameter[@id='3']")[0].attrib[
            "value"
        ]
        == "0.0"
    )
    assert lower.findall("parameter[@id='21']")[0].attrib["flags"] == "4176"
    assert (
        lower.findall("parameter[@id='21']/parameter[@id='23']/parameter[@id='1']")[0].attrib[
            "value"
        ]
        == "0"
    )
    assert lower.findall("parameter[@id='14']/parameter[@id='15']")[0].attrib["value"] == "0"
    assert lower.findall("parameter[@id='21']/parameter[@id='26']")[0].attrib["value"] == "0.1"
    assert lower.findall("parameter[@id='21']/parameter[@id='27']")[0].attrib["value"] == "0.5625"
    assert (
        lower.findall("parameter[@id='21']/parameter[@id='75']/parameter[@id='2']")[0].attrib[
            "value"
        ]
        == "1.125"
    )


@pytest.mark.parametrize(
    ("appearance", "message"),
    [
        (
            replace(
                CaptionAppearance(
                    top_srgb=(36 / 255, 85 / 255, 106 / 255),
                    bottom_srgb=(40 / 255, 89 / 255, 109 / 255),
                ),
                font_size=float("nan"),
            ),
            "finite",
        ),
        (
            replace(
                CaptionAppearance(
                    top_srgb=(36 / 255, 85 / 255, 106 / 255),
                    bottom_srgb=(40 / 255, 89 / 255, 109 / 255),
                ),
                white_border=0,
            ),
            "positive",
        ),
        (
            replace(
                CaptionAppearance(
                    top_srgb=(36 / 255, 85 / 255, 106 / 255),
                    bottom_srgb=(40 / 255, 89 / 255, 109 / 255),
                ),
                black_rim=-1,
            ),
            "positive",
        ),
        (
            replace(
                CaptionAppearance(
                    top_srgb=(36 / 255, 85 / 255, 106 / 255),
                    bottom_srgb=(40 / 255, 89 / 255, 109 / 255),
                ),
                shadow_opacity=1.01,
            ),
            "RGB",
        ),
        (
            replace(
                CaptionAppearance(
                    top_srgb=(36 / 255, 85 / 255, 106 / 255),
                    bottom_srgb=(40 / 255, 89 / 255, 109 / 255),
                ),
                top=(0, -1, 0),
            ),
            "RGB",
        ),
        (
            replace(
                CaptionAppearance(
                    top_srgb=(36 / 255, 85 / 255, 106 / 255),
                    bottom_srgb=(40 / 255, 89 / 255, 109 / 255),
                ),
                top_location=0.8,
            ),
            "ordered",
        ),
        (
            replace(
                CaptionAppearance(
                    top_srgb=(36 / 255, 85 / 255, 106 / 255),
                    bottom_srgb=(40 / 255, 89 / 255, 109 / 255),
                ),
                midpoint=0,
            ),
            "ordered",
        ),
        (
            replace(
                CaptionAppearance(
                    top_srgb=(36 / 255, 85 / 255, 106 / 255),
                    bottom_srgb=(40 / 255, 89 / 255, 109 / 255),
                ),
                shadow_blur=-1,
            ),
            "shadow geometry",
        ),
        (
            replace(
                CaptionAppearance(
                    top_srgb=(36 / 255, 85 / 255, 106 / 255),
                    bottom_srgb=(40 / 255, 89 / 255, 109 / 255),
                ),
                shadow_distance=1081,
            ),
            "shadow geometry",
        ),
        (
            replace(
                CaptionAppearance(
                    top_srgb=(36 / 255, 85 / 255, 106 / 255),
                    bottom_srgb=(40 / 255, 89 / 255, 109 / 255),
                ),
                shadow_opacity=True,
            ),
            "finite",
        ),
    ],
)
def test_invalid_appearance_is_rejected(appearance: CaptionAppearance, message: str) -> None:
    with pytest.raises(ReplayError, match=message):
        build_caption_template(BASE, REFERENCE, appearance=appearance)


@pytest.mark.parametrize(
    ("source", "message"),
    [
        ("x" * (4 * 1024 * 1024 + 1), "size limit"),
        ('<!DOCTYPE ozxmlscene SYSTEM "file:///etc/passwd">' + BASE, "declarations"),
        ('<!ENTITY x "secret">' + BASE, "declarations"),
        ("<ozml>", "Invalid Motion XML"),
        ("<fcpxml/>", "ozml document"),
        (BASE.replace('version="5.13"', 'version="99"'), "5.13"),
        (BASE.replace("<width>1920</width>", "<width>720</width>"), "reference canvas"),
        (BASE.replace("<height>1080</height>", ""), "structure"),
        (BASE.replace('factoryID="4"', 'factoryID="5"'), "two Motion gradient"),
        (BASE.replace('flags="4176"', ""), "Missing Motion attribute"),
        (BASE.replace("999100", "123100"), "Missing generated"),
        (BASE.replace("999100120", "999999999"), "range exhausted"),
    ],
)
def test_unsupported_or_unsafe_templates_are_rejected(source: str, message: str) -> None:
    with pytest.raises(ReplayError, match=message):
        build_caption_template(source, REFERENCE)


def test_missing_control_reference_is_rejected() -> None:
    with pytest.raises(ReplayError, match="reference style"):
        build_caption_template(BASE, "<ozml/>")


def test_empty_motion_declaration_is_accepted() -> None:
    result = build_caption_template("<!DOCTYPE ozxmlscene>" + BASE, REFERENCE)
    assert ET.fromstring(result).attrib["version"] == "5.13"


def test_default_candidate_bytes_are_stable_without_type_on() -> None:
    result = build_caption_template(BASE, REFERENCE)
    assert hashlib.sha256(result.encode()).hexdigest() == (
        "aac18be8416a541ecf91460535149d61d417356e79b495c65bb30e0eb8cc8f02"
    )
    assert 'id="319"' not in build_caption_template(TYPE_ON_BASE, REFERENCE, type_on=False)


def test_type_on_adds_full_reveal_controls_to_both_text_objects() -> None:
    source = TYPE_ON_BASE
    root = ET.fromstring(build_caption_template(source, REFERENCE, type_on=True))
    face, rim = root.findall("scene/layer/scenenode")
    face_351, face_group, face_323 = face.findall("parameter[@id='2']/parameter")
    rim_351, rim_group, rim_323 = rim.findall("parameter[@id='2']/parameter")
    assert face_351.attrib == {"id": "351", "value": "1"}
    assert face_323.attrib == {"id": "323", "value": "0"}
    assert rim_351.attrib == {"id": "351", "value": "1"}
    assert rim_323.attrib == {"id": "323", "value": "0"}
    assert face_group.attrib == {"name": "Type On", "id": "319", "flags": "8589938704"}
    assert rim_group.attrib == {"name": "Type On", "id": "319", "flags": "8589938704"}
    face_end, face_fade = face_group.findall("parameter")
    rim_end, rim_fade = rim_group.findall("parameter")
    assert face_end.attrib == {
        "name": "End",
        "id": "321",
        "flags": "8589934608",
        "default": "1",
        "value": "1",
    }
    assert face_fade.attrib == {
        "name": "Fade In",
        "id": "322",
        "flags": "8606711824",
        "default": "1",
        "value": "0",
    }
    assert rim_end.attrib == {
        "name": "End",
        "id": "321",
        "flags": "8589934608",
        "default": "1",
        "value": "1",
    }
    assert rim_fade.attrib == {
        "name": "Fade In",
        "id": "322",
        "flags": "8606711824",
        "default": "1",
        "value": "0",
    }
    fill, reveal_face, reveal_rim = root.findall("scene/publishSettings/target")
    assert fill.attrib == {
        "object": "999100220",
        "channel": "./5/999100230/14/15",
        "name": "Caption Fill Mode",
    }
    assert reveal_face.attrib == {
        "object": "999100220",
        "channel": "./2/319/321",
        "name": "Caption Reveal Face",
    }
    assert reveal_rim.attrib == {
        "object": "999100100",
        "channel": "./2/319/321",
        "name": "Caption Reveal Rim",
    }
    assert 'id="319"' not in source


def test_type_on_publication_follows_remapped_object_ids() -> None:
    source = TYPE_ON_BASE.replace("999100", "999300")
    root = ET.fromstring(build_caption_template(source, REFERENCE, type_on=True))
    fill, reveal_face, reveal_rim = root.findall("scene/publishSettings/target")
    assert fill.attrib == {
        "object": "999300220",
        "channel": "./5/999300230/14/15",
        "name": "Caption Fill Mode",
    }
    assert reveal_face.attrib == {
        "object": "999300220",
        "channel": "./2/319/321",
        "name": "Caption Reveal Face",
    }
    assert reveal_rim.attrib == {
        "object": "999300100",
        "channel": "./2/319/321",
        "name": "Caption Reveal Rim",
    }


@pytest.mark.parametrize(
    ("source", "message"),
    [
        (BASE, "structure"),
        (TYPE_ON_BASE.replace('<parameter id="351" value="1"/>', ""), "structure"),
        (
            TYPE_ON_BASE.replace(
                '<parameter id="323" value="0"/>',
                '<parameter id="319"><parameter id="321" value="1"/></parameter>'
                '<parameter id="323" value="0"/>',
            ),
            "already has Type On",
        ),
        (
            TYPE_ON_BASE.replace(
                '<parameter id="323" value="0"/>',
                '<parameter id="319"/><parameter id="319"/><parameter id="323" value="0"/>',
            ),
            "already has Type On",
        ),
        (TYPE_ON_BASE.replace('version="5.13"', 'version="5.15"'), "5.13"),
    ],
)
def test_type_on_rejects_unsupported_or_conflicting_objects(source: str, message: str) -> None:
    with pytest.raises(ReplayError, match=message):
        build_caption_template(source, REFERENCE, type_on=True)


ANIMATION_TEMPLATE = build_caption_template(
    TYPE_ON_BASE.replace(
        "</sceneSettings>",
        "<duration>300</duration><frameRate>30</frameRate><NTSC>0</NTSC></sceneSettings>",
    ),
    REFERENCE,
    type_on=True,
)
REVEAL_PATH = "scene/layer/scenenode/parameter[@id='2']/parameter[@id='319']"
ANIMATION_SAMPLES = (
    (Fraction(0), Fraction(1, 10)),
    (Fraction(3, 10), Fraction(1, 10)),
    (Fraction(1, 3), Fraction(1, 2)),
)


def test_motion_animation_uses_same_exact_clock_on_both_objects() -> None:
    result = animate_caption_template(ANIMATION_TEMPLATE, samples=ANIMATION_SAMPLES)
    root = ET.fromstring(result)
    face, rim = root.findall(REVEAL_PATH)
    assert ET.tostring(face) == ET.tostring(rim)
    end = face.findall("parameter[@id='321']")[0]
    assert "value" not in end.attrib
    curve = end.findall("curve")[0]
    assert curve.attrib == {"type": "1", "default": "0.1", "value": "0.1"}
    assert curve.findtext("numberOfKeypoints") == "3"
    assert curve.findtext("min") == "0" and curve.findtext("max") == "1"
    first, second, third = curve.findall("keypoint")
    assert [point.findtext("time") for point in (first, second, third)] == [
        "0 1 1 0",
        "3 10 1 0",
        "1 3 1 0",
    ]
    assert [point.findtext("value") for point in (first, second, third)] == ["0.1", "0.1", "0.5"]
    assert first.attrib == {"interpolation": "1", "flags": "0"}
    assert first.findtext("inputTangentTime") is None
    assert first.findtext("outputTangentTime") == "0.1"
    assert first.findtext("outputTangentValue") == "0.0"
    assert second.findtext("inputTangentTime") == "-0.1"
    assert float(second.findtext("outputTangentTime", "")) == pytest.approx(1 / 90)
    assert float(second.findtext("outputTangentValue", "")) == pytest.approx(2 / 15)
    assert float(third.findtext("inputTangentValue", "")) == pytest.approx(-2 / 15)
    assert third.findtext("outputTangentTime") == second.findtext("outputTangentTime")
    # Reversing only the known curve/fade changes restores the entire template tree.
    for group in (face, rim):
        control = group.findall("parameter[@id='321']")[0]
        control.remove(control.findall("curve")[0])
        control.set("value", "1")
        assert group.findall("parameter[@id='322']")[0].get("value") == "0"
    assert ET.canonicalize(ET.tostring(root, encoding="unicode")) == ET.canonicalize(
        ANIMATION_TEMPLATE
    )


def test_motion_fade_is_explicit_and_both_objects_stay_synchronized() -> None:
    root = ET.fromstring(
        animate_caption_template(
            ANIMATION_TEMPLATE,
            samples=ANIMATION_SAMPLES,
            fade_in=True,
        )
    )
    assert [node.get("value") for node in root.findall(REVEAL_PATH + "/parameter[@id='322']")] == [
        "1",
        "1",
    ]
    assert all(
        node.get("value") == "0"
        for node in ET.fromstring(ANIMATION_TEMPLATE).findall(REVEAL_PATH + "/parameter[@id='322']")
    )


def test_motion_smoothing_retains_sparse_keys_and_zero_slope_handles() -> None:
    root = ET.fromstring(
        animate_caption_template(
            ANIMATION_TEMPLATE,
            samples=(
                (Fraction(0), Fraction(0)),
                (Fraction(4, 15), Fraction(1, 2)),
                (Fraction(1, 2), Fraction(1, 2)),
                (Fraction(23, 30), Fraction(1)),
            ),
            fade_in=True,
            smooth=True,
        )
    )
    face, rim = root.findall(REVEAL_PATH)
    assert ET.tostring(face) == ET.tostring(rim)
    curve = face.findall("parameter[@id='321']/curve")[0]
    assert curve.findtext("numberOfKeypoints") == "4"
    points = curve.findall("keypoint")
    assert [point.findtext("time") for point in points] == [
        "0 1 1 0",
        "4 15 1 0",
        "1 2 1 0",
        "23 30 1 0",
    ]
    assert [point.findtext("value") for point in points] == ["0.0", "0.5", "0.5", "1.0"]
    assert [point.findtext("inputTangentValue") for point in points] == [None, "0.0", "0.0", "0.0"]
    assert [point.findtext("outputTangentValue") for point in points] == [
        "0.0",
        "0.0",
        "0.0",
        "0.0",
    ]
    assert [float(point.findtext("outputTangentTime", "")) for point in points] == pytest.approx(
        [4 / 45, 7 / 90, 4 / 45, 4 / 45]
    )
    assert face.findall("parameter[@id='322']")[0].get("value") == "1"


@pytest.mark.parametrize(
    ("fade_in", "smooth", "message"),
    [(1, False, "Fade In"), (False, 1, "smoothing"), (True, "yes", "smoothing")],
)
def test_motion_animation_refuses_non_boolean_options(
    fade_in: bool, smooth: bool, message: str
) -> None:
    with pytest.raises(ReplayError, match=message):
        animate_caption_template(
            ANIMATION_TEMPLATE, samples=ANIMATION_SAMPLES, fade_in=fade_in, smooth=smooth
        )


def test_motion_animation_preserves_ntsc_clock_without_rounding() -> None:
    samples = ((Fraction(0), Fraction(0)), (Fraction(1001, 100), Fraction(1)))
    with pytest.raises(ReplayError, match="duration"):
        animate_caption_template(ANIMATION_TEMPLATE, samples=samples)
    result = animate_caption_template(
        ANIMATION_TEMPLATE.replace("<NTSC>0</NTSC>", "<NTSC>1</NTSC>"),
        samples=samples,
    )
    assert "1001 100 1 0" in result
    assert "<duration>300</duration>" in result


@pytest.mark.parametrize(
    ("samples", "message"),
    [
        ((), "between 2"),
        (((Fraction(0), Fraction(0)),), "between 2"),
        (ANIMATION_SAMPLES * 4000, "10000"),
        (((Fraction(1), Fraction(0)), (Fraction(2), Fraction(1))), "start at zero"),
        (((Fraction(0), Fraction(0)), (Fraction(0), Fraction(1))), "increasing"),
        (((Fraction(0), Fraction(0)), (Fraction(-1), Fraction(1))), "increasing"),
        (((Fraction(0), Fraction(-1)), (Fraction(1), Fraction(1))), "zero and one"),
        (((Fraction(0), Fraction(0)), (Fraction(1), Fraction(2))), "zero and one"),
        (((Fraction(0), Fraction(0)), (Fraction(1, 2**31), Fraction(1))), "CMTime"),
        (((Fraction(0), Fraction(0)), (Fraction(2**63), Fraction(1))), "CMTime"),
    ],
)
def test_motion_animation_refuses_invalid_samples(
    samples: tuple[tuple[Fraction, Fraction], ...],
    message: str,
) -> None:
    with pytest.raises(ReplayError, match=message):
        animate_caption_template(ANIMATION_TEMPLATE, samples=samples)


@pytest.mark.parametrize(
    ("old", "new", "message"),
    [
        ('version="5.13"', 'version="5.15"', "5.13"),
        ("<duration>300</duration>", "<duration>bad</duration>", "Invalid Motion project clock"),
        ("<frameRate>30</frameRate>", "<frameRate>0/0</frameRate>", "Invalid Motion project clock"),
        ("<duration>300</duration>", "<duration>0</duration>", "Unsupported Motion project clock"),
        (
            "<frameRate>30</frameRate>",
            "<frameRate>241</frameRate>",
            "Unsupported Motion project clock",
        ),
        ("<NTSC>0</NTSC>", "<NTSC>2</NTSC>", "Unsupported Motion project clock"),
        ("Caption Face", "Another title", "two generated"),
        ('factoryID="16"', 'factoryID="99"', "two generated"),
        (
            'id="321" flags="8589934608" default="1" value="1"',
            'id="321" flags="8589934608" default="1" value="0.5"',
            "already animated",
        ),
        (
            'id="322" flags="8606711824" default="1" value="0"',
            'id="322" flags="8606711824" default="1" value="1"',
            "already animated",
        ),
    ],
)
def test_motion_animation_refuses_unsupported_or_modified_templates(
    old: str,
    new: str,
    message: str,
) -> None:
    assert old in ANIMATION_TEMPLATE
    with pytest.raises(ReplayError, match=message):
        animate_caption_template(ANIMATION_TEMPLATE.replace(old, new), samples=ANIMATION_SAMPLES)


def test_motion_animation_does_not_replace_existing_curves() -> None:
    result = animate_caption_template(ANIMATION_TEMPLATE, samples=ANIMATION_SAMPLES)
    with pytest.raises(ReplayError, match="already animated"):
        animate_caption_template(result, samples=ANIMATION_SAMPLES)


def test_motion_animation_limits_expanded_document_size() -> None:
    samples = tuple((Fraction(i, 1000), Fraction(i, 10_000)) for i in range(10_000))
    with pytest.raises(ReplayError, match="XML size limit"):
        animate_caption_template(ANIMATION_TEMPLATE, samples=samples)


def test_a_growing_caption_becomes_one_revealing_title(
    sequence_template: str, stage_plan: dict[str, object]
) -> None:
    root = ET.fromstring(motion_reveal_stages.prepare(sequence_template, stage_plan))
    assert [text.text for text in root.findall("scene/layer/scenenode//text")] == ["字幕", "字幕"]
    assert [timing.get("out") for timing in root.findall("scene/layer/scenenode/timing")] == [
        "29 30 1 0",
        "29 30 1 0",
    ]


def test_the_entrance_is_added_only_when_the_style_enables_it(
    sequence_template: str, stage_plan: dict[str, object], monkeypatch: pytest.MonkeyPatch
) -> None:
    without = motion_reveal_stages.prepare(sequence_template, stage_plan)
    monkeypatch.setattr(motion_reveal_stages, "MOTION_ENTRANCE", True)
    with_entrance = motion_reveal_stages.prepare(sequence_template, stage_plan)
    assert with_entrance != without
    assert len(with_entrance) > len(without)


def reveal_stage(text: str, start: int, end: int, y: float = 100.0) -> motion_reveal_stages.Stage:
    return motion_reveal_stages.Stage(text, start, end, 0.0, y)


def test_chains_break_on_resets_gaps_moves_and_rebalanced_lines() -> None:
    stages = [
        reveal_stage("あ", 0, 2),
        reveal_stage("あい", 2, 4),
        reveal_stage("う", 4, 6),
        reveal_stage("うえ", 7, 9),
        reveal_stage("うえお", 9, 10, y=50.0),
        reveal_stage("か\nき", 10, 12),
        reveal_stage("かき\nく", 12, 14),
    ]
    assert [[s.text for s in c] for c in motion_reveal_stages.chains(stages)] == [
        ["あ", "あい"],
        ["う"],
        ["うえ"],
        ["うえお"],
        ["か\nき"],
        ["かき\nく"],
    ]


def test_reveal_keys_show_each_stage_on_its_own_frames() -> None:
    chain = [reveal_stage("あ", 0, 3), reveal_stage("あい", 3, 5), reveal_stage("あいう", 5, 9)]
    assert motion_reveal_stages.reveal(chain, 30) == (
        (Fraction(0), Fraction(1, 6)),
        (Fraction(2, 30), Fraction(1, 6)),
        (Fraction(3, 30), Fraction(3, 6)),
        (Fraction(4, 30), Fraction(3, 6)),
        (Fraction(5, 30), Fraction(5, 6)),
    )
    assert motion_reveal_stages.reveal([reveal_stage("あ", 0, 3)], 30) == (
        (Fraction(0), Fraction(1, 2)),
        (Fraction(3, 30), Fraction(1, 2)),
    )


def test_a_stage_splitting_a_composed_character_is_refused() -> None:
    with pytest.raises(ReplayError, match="composed character"):
        motion_reveal_stages.visible("か", [2])


@pytest.mark.parametrize(
    "change",
    [
        {"rate": 0},
        {"coordinateSpace": "screen"},
        {"placements": []},
        {"extra": 1},
        {"placements": [{"text": "a", "startFrame": 0, "endFrame": 1, "x": True, "y": 0}]},
        {"placements": [{"text": "a", "startFrame": 0, "endFrame": 1, "x": 0, "y": "0"}]},
        {"placements": [{"text": "a", "startFrame": 0, "endFrame": 1}]},
    ],
)
def test_invalid_reveal_plans_are_refused(
    sequence_template: str, stage_plan: dict[str, object], change: dict[str, object]
) -> None:
    with pytest.raises(ReplayError):
        motion_reveal_stages.prepare(sequence_template, {**stage_plan, **change})


def test_too_many_titles_are_refused(
    sequence_template: str, stage_plan: dict[str, object], monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(motion_reveal_stages, "MAX_SEQUENCE_SENTENCES", 1)
    plan = {**stage_plan, "placements": [
        {"text": "字", "startFrame": 0, "endFrame": 1, "x": 0, "y": 100},
        {"text": "幕", "startFrame": 1, "endFrame": 30, "x": 0, "y": 100},
    ]}  # fmt: skip
    with pytest.raises(ReplayError, match="Too many"):
        motion_reveal_stages.prepare(sequence_template, plan)


def test_the_reveal_cli_writes_once(
    sequence_template: str, stage_plan: dict[str, object], tmp_path: Path
) -> None:
    (tmp_path / "t.moti").write_text(sequence_template, encoding="utf-8")
    (tmp_path / "p.json").write_text(json.dumps(stage_plan), encoding="utf-8")
    arguments = [str(tmp_path / "t.moti"), str(tmp_path / "p.json"), str(tmp_path / "o.moti")]
    assert motion_reveal_stages.main(arguments) == 0
    assert "字幕" in (tmp_path / "o.moti").read_text(encoding="utf-8")
    with pytest.raises(ReplayError, match="overwrite"):
        motion_reveal_stages.main(arguments)
