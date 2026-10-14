"""Synthetic stage finishing; no private footage or native rendering."""

import sys
from pathlib import Path
from xml.etree import ElementTree as ET

import pytest
from test_motion_caption import sequence_template as sequence_template
from test_motion_caption import stage_plan as stage_plan
from test_motion_caption import text_template as text_template

from replay import motion_finish, motion_stages
from replay.core import ReplayError
from replay.motion_finish import FinishStyle, finish_sequence


def test_three_layers_keep_stage_clocks_and_text(
    sequence_template: str, stage_plan: dict[str, object]
) -> None:
    original = motion_stages.prepare(sequence_template, stage_plan)
    result = finish_sequence(
        original,
        style=FinishStyle(
            top=(0.012, 0.055, 0.08),
            top_srgb=(36 / 255, 85 / 255, 106 / 255),
            bottom_srgb=(40 / 255, 89 / 255, 109 / 255),
            bottom=(0.02, 0.085, 0.12),
        ),
    )
    root = ET.fromstring(result)
    nodes = root.findall("scene/layer/scenenode")
    assert [node.get("name") for node in nodes] == [
        "Caption Keyline Face",
        "Caption Face",
        "Caption Outer Rim",
        "Caption Keyline Face",
        "Caption Face",
        "Caption Outer Rim",
    ]
    assert [node.findtext("parameter[@id='2']/parameter[@id='369']/text") for node in nodes] == [
        "字",
        "字",
        "字",
        "字幕",
        "字幕",
        "字幕",
    ]
    assert [node.findall("timing")[0].get("in") for node in nodes] == [
        "0 30 1 0",
        "0 30 1 0",
        "0 30 1 0",
        "1 30 1 0",
        "1 30 1 0",
        "1 30 1 0",
    ]
    widths = root.findall("scene/layer/scenenode/style/parameter[@id='30']/parameter[@id='36']")
    assert [item.get("value") for item in widths] == [
        "2.0",
        "10.92",
        "16.38",
        "2.0",
        "10.92",
        "16.38",
    ]
    first, white, rim = nodes[:3]
    assert ET.tostring(first.findall("parameter[@id='2']")[0]) == ET.tostring(
        white.findall("parameter[@id='2']")[0]
    )
    stops = first.findall(
        "style/parameter[@id='14']/parameter[@id='17']/parameter[@id='1']/parameter"
    )
    assert [item.findall("parameter[@id='1']/curve")[0].get("value") for item in stops] == [
        "0.8",
        "1.0",
    ]
    assert [
        item.get("value") for item in stops[1].findall("parameter[@id='3']/parameter/curve")
    ] == ["0.02", "0.085", "0.12"]
    assert [
        item.get("value")
        for item in white.findall("style/parameter[@id='30']/parameter[@id='32']/parameter")
    ] == ["1", "1", "1"]
    assert rim.findall("style/parameter[@id='21']/parameter[@id='26']")[0].get("value") == "0.4"
    assert rim.findall("style/parameter[@id='21']/parameter[@id='27']")[0].get("value") == "3.0"
    assert len({node.get("id") for node in nodes}) == 6
    assert first.findall("styleRun")[0].get("style") == first.findall("style")[0].get("id")
    assert original == motion_stages.prepare(sequence_template, stage_plan)
    with pytest.raises(ReplayError, match="ordered"):
        finish_sequence(
            result,
            style=FinishStyle(
                top=(0.012, 0.055, 0.08),
                top_srgb=(36 / 255, 85 / 255, 106 / 255),
                bottom_srgb=(40 / 255, 89 / 255, 109 / 255),
                bottom=(0.02, 0.085, 0.12),
            ),
        )


@pytest.mark.parametrize("keyline", [0.0, -1.0, 10.92, float("nan"), True])
def test_invalid_keyline(sequence_template: str, keyline: float) -> None:
    with pytest.raises(ReplayError, match="Keyline"):
        finish_sequence(
            sequence_template,
            style=FinishStyle(
                top=(0.012, 0.055, 0.08),
                top_srgb=(36 / 255, 85 / 255, 106 / 255),
                bottom_srgb=(40 / 255, 89 / 255, 109 / 255),
                bottom=(0.02, 0.085, 0.12),
                keyline=keyline,
            ),
        )


@pytest.mark.parametrize("start", [-0.1, 1.0, float("nan")])
def test_invalid_gradient(sequence_template: str, start: float) -> None:
    with pytest.raises(ReplayError):
        finish_sequence(
            sequence_template,
            style=FinishStyle(
                top=(0.012, 0.055, 0.08),
                top_srgb=(36 / 255, 85 / 255, 106 / 255),
                bottom_srgb=(40 / 255, 89 / 255, 109 / 255),
                bottom=(0.02, 0.085, 0.12),
                bright_start=start,
            ),
        )


def test_unsupported_version(sequence_template: str) -> None:
    with pytest.raises(ReplayError, match="5.13"):
        finish_sequence(
            sequence_template.replace('version="5.13"', 'version="9"'),
            style=FinishStyle(
                top=(0.012, 0.055, 0.08),
                top_srgb=(36 / 255, 85 / 255, 106 / 255),
                bottom_srgb=(40 / 255, 89 / 255, 109 / 255),
                bottom=(0.02, 0.085, 0.12),
            ),
        )


def test_mismatched_animation(sequence_template: str) -> None:
    root = ET.fromstring(sequence_template)
    face = root.findall("scene/layer/scenenode")[0]
    face.findall("timing")[0].set("in", "1 30 1 0")
    with pytest.raises(ReplayError, match="synchronized"):
        finish_sequence(
            ET.tostring(root, encoding="unicode"),
            style=FinishStyle(
                top=(0.012, 0.055, 0.08),
                top_srgb=(36 / 255, 85 / 255, 106 / 255),
                bottom_srgb=(40 / 255, 89 / 255, 109 / 255),
                bottom=(0.02, 0.085, 0.12),
            ),
        )


def test_wrong_pair_order(sequence_template: str) -> None:
    with pytest.raises(ReplayError, match="ordered"):
        finish_sequence(
            sequence_template.replace("Caption Face", "Other"),
            style=FinishStyle(
                top=(0.012, 0.055, 0.08),
                top_srgb=(36 / 255, 85 / 255, 106 / 255),
                bottom_srgb=(40 / 255, 89 / 255, 109 / 255),
                bottom=(0.02, 0.085, 0.12),
            ),
        )


def test_expanded_size(sequence_template: str, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(motion_finish, "MAX_XML_CHARACTERS", len(sequence_template) + 10)
    with pytest.raises(ReplayError, match="size limit"):
        finish_sequence(
            sequence_template,
            style=FinishStyle(
                top=(0.012, 0.055, 0.08),
                top_srgb=(36 / 255, 85 / 255, 106 / 255),
                bottom_srgb=(40 / 255, 89 / 255, 109 / 255),
                bottom=(0.02, 0.085, 0.12),
            ),
        )


def test_invalid_original_gradient(sequence_template: str) -> None:
    root = ET.fromstring(sequence_template)
    group = root.findall(
        "scene/layer/scenenode/style/parameter[@id='14']/parameter[@id='17']/parameter[@id='1']"
    )[0]
    group.remove(group[0])
    with pytest.raises(ReplayError, match="two original"):
        finish_sequence(
            ET.tostring(root, encoding="unicode"),
            style=FinishStyle(
                top=(0.012, 0.055, 0.08),
                top_srgb=(36 / 255, 85 / 255, 106 / 255),
                bottom_srgb=(40 / 255, 89 / 255, 109 / 255),
                bottom=(0.02, 0.085, 0.12),
            ),
        )


def test_invalid_original_color(sequence_template: str) -> None:
    root = ET.fromstring(sequence_template)
    channel = root.findall(
        "scene/layer/scenenode/style/parameter[@id='14']/parameter[@id='17']/parameter[@id='1']/parameter/parameter[@id='3']/parameter[@id='1']/curve"
    )[0]
    channel.set("value", "not-a-number")
    with pytest.raises(ReplayError, match="colour"):
        finish_sequence(
            ET.tostring(root, encoding="unicode"),
            style=FinishStyle(
                top=(0.012, 0.055, 0.08),
                top_srgb=(36 / 255, 85 / 255, 106 / 255),
                bottom_srgb=(40 / 255, 89 / 255, 109 / 255),
                bottom=(0.02, 0.085, 0.12),
            ),
        )


def test_private_paragraph_ids_are_not_control_differences(sequence_template: str) -> None:
    root = ET.fromstring(sequence_template)
    face, rim = root.findall("scene/layer/scenenode/parameter[@id='2']")
    ET.SubElement(face, "parameter", id="999199015", value="1")
    ET.SubElement(rim, "parameter", id="999169573", value="1")
    result = finish_sequence(
        ET.tostring(root, encoding="unicode"),
        style=FinishStyle(
            top=(0.012, 0.055, 0.08),
            top_srgb=(36 / 255, 85 / 255, 106 / 255),
            bottom_srgb=(40 / 255, 89 / 255, 109 / 255),
            bottom=(0.02, 0.085, 0.12),
        ),
    )
    assert len(ET.fromstring(result).findall("scene/layer/scenenode")) == 3
    rim[-1].set("value", "2")
    with pytest.raises(ReplayError, match="synchronized"):
        finish_sequence(
            ET.tostring(root, encoding="unicode"),
            style=FinishStyle(
                top=(0.012, 0.055, 0.08),
                top_srgb=(36 / 255, 85 / 255, 106 / 255),
                bottom_srgb=(40 / 255, 89 / 255, 109 / 255),
                bottom=(0.02, 0.085, 0.12),
            ),
        )


def test_cli_explicit_soft_colour_and_wider_white_band(
    sequence_template: str, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    source, output = tmp_path / "source.moti", tmp_path / "soft.moti"
    source.write_text(sequence_template, encoding="utf-8")
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "motion-finish",
            "--top-srgb",
            "#24556A",
            "--bottom-srgb",
            "#28596D",
            str(source),
            str(output),
            "--top",
            "0.012",
            "0.055",
            "0.08",
            "--bottom",
            "0.02",
            "0.085",
            "0.12",
            "--border",
            "16.38",
        ],
    )
    motion_finish.main()
    root = ET.fromstring(output.read_text(encoding="utf-8"))
    nodes = root.findall("scene/layer/scenenode")
    stops = nodes[0].findall(
        "style/parameter[@id='14']/parameter[@id='17']/parameter[@id='1']/parameter"
    )
    assert [
        item.get("value") for item in stops[0].findall("parameter[@id='3']/parameter/curve")
    ] == ["0.012", "0.055", "0.08"]
    assert [
        item.get("value") for item in stops[1].findall("parameter[@id='3']/parameter/curve")
    ] == ["0.02", "0.085", "0.12"]
    assert [
        item.get("value")
        for item in root.findall(
            "scene/layer/scenenode/style/parameter[@id='30']/parameter[@id='36']"
        )
    ] == ["2.0", "16.38", "21.84"]
    assert source.read_text(encoding="utf-8") == sequence_template


@pytest.mark.parametrize("top", [(float("nan"), 1.0, 1.0), (-0.1, 0.8, 0.9), (0.7, 0.8, 1.1)])
def test_explicit_colour_is_validated(
    sequence_template: str, top: tuple[float, float, float]
) -> None:
    with pytest.raises(ReplayError):
        finish_sequence(
            sequence_template,
            style=FinishStyle(
                top_srgb=(36 / 255, 85 / 255, 106 / 255),
                bottom_srgb=(40 / 255, 89 / 255, 109 / 255),
                bottom=(0.02, 0.085, 0.12),
                top=top,
            ),
        )


@pytest.mark.parametrize("colors", [("0.7", "0.83", "0.87"), ("0.86", "0.93", "0.95")])
def test_cli_rejects_previous_low_contrast_palette_before_writing(
    sequence_template: str,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    colors: tuple[str, str, str],
) -> None:
    source, output = tmp_path / "source.moti", tmp_path / "rejected.moti"
    source.write_text(sequence_template, encoding="utf-8")
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "motion-finish",
            "--top-srgb",
            "#24556A",
            "--bottom-srgb",
            "#B3D4DE",
            str(source),
            str(output),
            "--top",
            "0.012",
            "0.055",
            "0.08",
            "--bottom",
            *colors,
        ],
    )
    with pytest.raises(ReplayError, match="below 7:1"):
        motion_finish.main()
    assert not output.exists()
    assert source.read_text(encoding="utf-8") == sequence_template


def test_cli_sets_all_three_font_objects(
    sequence_template: str, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    source, output = tmp_path / "source.moti", tmp_path / "font.moti"
    source.write_text(sequence_template, encoding="utf-8")
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "motion-finish",
            "--top-srgb",
            "#24556A",
            "--bottom-srgb",
            "#28596D",
            str(source),
            str(output),
            "--top",
            "0.012",
            "0.055",
            "0.08",
            "--bottom",
            "0.02",
            "0.085",
            "0.12",
            "--font-name",
            "NotoSansJP-Bold",
            "--font-size",
            "100.65702127659574",
        ],
    )
    motion_finish.main()
    root = ET.fromstring(output.read_text(encoding="utf-8"))
    assert [
        node.text for node in root.findall("scene/layer/scenenode/style/parameter[@id='83']/font")
    ] == ["NotoSansJP-Bold", "NotoSansJP-Bold", "NotoSansJP-Bold"]
    assert [
        node.get("value") for node in root.findall("scene/layer/scenenode/style/parameter[@id='3']")
    ] == ["100.65702127659574", "100.65702127659574", "100.65702127659574"]
    assert source.read_text(encoding="utf-8") == sequence_template


@pytest.mark.parametrize(
    ("name", "size"),
    [
        (None, 100.0),
        ("NotoSansJP-Bold", None),
        ("", 100.0),
        ("bad font/path", 100.0),
        ("NotoSansJP-Bold", 0.0),
        ("NotoSansJP-Bold", float("nan")),
    ],
)
def test_font_override_requires_valid_name_and_size(
    sequence_template: str, name: str | None, size: float | None
) -> None:
    with pytest.raises(ReplayError):
        finish_sequence(
            sequence_template,
            style=FinishStyle(
                top=(0.012, 0.055, 0.08),
                top_srgb=(36 / 255, 85 / 255, 106 / 255),
                bottom_srgb=(40 / 255, 89 / 255, 109 / 255),
                bottom=(0.02, 0.085, 0.12),
                font_name=name,
                font_size=size,
            ),
        )


def test_missing_objects() -> None:
    with pytest.raises(ReplayError, match="bounded pairs"):
        finish_sequence(
            '<ozml version="5.13"><scene><layer/></scene></ozml>',
            style=FinishStyle(
                top=(0.012, 0.055, 0.08),
                top_srgb=(36 / 255, 85 / 255, 106 / 255),
                bottom_srgb=(40 / 255, 89 / 255, 109 / 255),
                bottom=(0.02, 0.085, 0.12),
            ),
        )


def test_missing_generated_ids(sequence_template: str, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(motion_finish, "_generated_ids", lambda node: set())
    with pytest.raises(ReplayError, match="Missing generated"):
        finish_sequence(
            sequence_template,
            style=FinishStyle(
                top=(0.012, 0.055, 0.08),
                top_srgb=(36 / 255, 85 / 255, 106 / 255),
                bottom_srgb=(40 / 255, 89 / 255, 109 / 255),
                bottom=(0.02, 0.085, 0.12),
            ),
        )


def test_id_budget(sequence_template: str, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(motion_finish, "GENERATED_ID_MAX", 999_000_000)
    with pytest.raises(ReplayError, match="ID range"):
        finish_sequence(
            sequence_template,
            style=FinishStyle(
                top=(0.012, 0.055, 0.08),
                top_srgb=(36 / 255, 85 / 255, 106 / 255),
                bottom_srgb=(40 / 255, 89 / 255, 109 / 255),
                bottom=(0.02, 0.085, 0.12),
            ),
        )


def test_cli_preserves_input_and_refuses_overwrite(
    sequence_template: str, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    source, output = tmp_path / "source.moti", tmp_path / "result.moti"
    source.write_text(sequence_template, encoding="utf-8")
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "motion-finish",
            "--top-srgb",
            "#24556A",
            "--bottom-srgb",
            "#28596D",
            str(source),
            str(output),
            "--top",
            "0.012",
            "0.055",
            "0.08",
            "--bottom",
            "0.02",
            "0.085",
            "0.12",
        ],
    )
    motion_finish.main()
    assert output.read_text(encoding="utf-8") == finish_sequence(
        sequence_template,
        style=FinishStyle(
            top=(0.012, 0.055, 0.08),
            top_srgb=(36 / 255, 85 / 255, 106 / 255),
            bottom_srgb=(40 / 255, 89 / 255, 109 / 255),
            bottom=(0.02, 0.085, 0.12),
        ),
    )
    assert source.read_text(encoding="utf-8") == sequence_template
    with pytest.raises(FileExistsError):
        motion_finish.main()
    output.unlink()
    source.unlink()
    source.symlink_to(tmp_path / "missing")
    with pytest.raises(ReplayError, match="bounded regular"):
        motion_finish.main()
