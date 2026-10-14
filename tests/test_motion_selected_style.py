"""Selected default candidates retain measured contrast, not an AAA approval."""

import json
import sys
from dataclasses import replace
from pathlib import Path
from xml.etree import ElementTree as ET

import pytest
from test_motion_caption import sequence_template as sequence_template
from test_motion_caption import text_template as text_template

from replay import motion_finish
from replay.core import ReplayError
from replay.motion_caption import COLOUR_INTENT_MARKER
from replay.motion_finish import finish_sequence, selected_finish_style


def test_reference_geometry_and_explicit_raw_controls() -> None:
    style = selected_finish_style(
        top=(0.1, 0.2, 0.3),
        bottom=(0.4, 0.5, 0.6),
        font_name="NotoSansJP-Thin_Bold",
        font_size=88.0,
        stroke_scale=0.5,
        shadow_scale=1.0,
    )
    assert style.top == (0.1, 0.2, 0.3)
    assert style.bottom == (0.4, 0.5, 0.6)
    assert style.top_srgb == (47 / 255, 176 / 255, 238 / 255)
    assert style.bottom_srgb == (227 / 255, 246 / 255, 1.0)
    assert style.bright_start == 0.75
    assert style.midpoint == 0.5
    assert style.keyline == 2.0
    assert style.white_border == 14.4
    assert style.black_rim == pytest.approx(3.0)
    assert style.shadow_opacity == 0.6
    assert style.shadow_distance == 7.0
    assert style.shadow_blur == 4.0
    assert style.intent_policy == "selected-default"


@pytest.mark.parametrize("scale", [0.0, -1.0, float("nan"), float("inf"), True, 1000.0])
def test_invalid_reference_scale(scale: float) -> None:
    with pytest.raises(ReplayError):
        selected_finish_style(
            top=(0.1, 0.2, 0.3),
            bottom=(0.4, 0.5, 0.6),
            font_name="NotoSansJP-Thin_Bold",
            font_size=88.0,
            stroke_scale=scale,
            shadow_scale=scale,
        )


@pytest.mark.parametrize(
    "field,value",
    [
        ("top_srgb", (0.1, 0.2, 0.3)),
        ("bottom_srgb", (0.4, 0.5, 0.6)),
        ("top_location", 0.6),
        ("bottom_location", 0.9),
        ("midpoint", 0.8),
        ("intent_policy", "unknown"),
    ],
)
def test_selected_policy_does_not_waive_arbitrary_intent(field: str, value: object) -> None:
    appearance = selected_finish_style(
        top=(0.1, 0.2, 0.3),
        bottom=(0.4, 0.5, 0.6),
        font_name="NotoSansJP-Thin_Bold",
        font_size=88.0,
        stroke_scale=0.5,
        shadow_scale=1.0,
    ).appearance()
    object.__setattr__(appearance, field, value)
    with pytest.raises(ReplayError, match="adopted|Unknown"):
        appearance.validate()


def test_strict_contrast_policy_still_rejects_selected_palette() -> None:
    appearance = selected_finish_style(
        top=(0.1, 0.2, 0.3),
        bottom=(0.4, 0.5, 0.6),
        font_name="NotoSansJP-Thin_Bold",
        font_size=88.0,
        stroke_scale=0.5,
        shadow_scale=1.0,
    ).appearance()
    with pytest.raises(ReplayError, match="below 7:1"):
        replace(appearance, intent_policy="white-aaa").validate()


def test_finished_annotation_does_not_claim_aaa(sequence_template: str) -> None:
    document = finish_sequence(
        sequence_template,
        style=selected_finish_style(
            top=(0.1, 0.2, 0.3),
            bottom=(0.4, 0.5, 0.6),
            font_name="NotoSansJP-Thin_Bold",
            font_size=88.0,
            stroke_scale=0.5,
            shadow_scale=1.0,
        ),
    )
    root = ET.fromstring(document, parser=ET.XMLParser(target=ET.TreeBuilder(insert_comments=True)))
    comment = next(child for child in root if not isinstance(child.tag, str))
    payload = json.loads((comment.text or "").removeprefix(COLOUR_INTENT_MARKER))
    assert payload["selectedStyle"] == 20
    assert payload["intentPolicy"] == "selected-default"
    assert payload["requiredWhiteContrast"] is None
    assert payload["selectedWhiteContrast"]["top"] == pytest.approx(2.4516641403549544)
    assert payload["selectedWhiteContrast"]["bottom"] == pytest.approx(1.1115530534139368)
    assert payload["renderedContrastVerified"] is False
    assert payload["renderedColourVerified"] is False
    assert payload["motionControls"]["top"] == [0.1, 0.2, 0.3]
    assert payload["motionControls"]["topLocation"] == 0.75
    assert payload["motionControls"]["midpoint"] == 0.5
    assert len(root.findall("scene/layer/scenenode")) == 3


def test_cli_uses_selected_font_and_style_defaults_without_overrides(
    sequence_template: str,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source, output = tmp_path / "input.moti", tmp_path / "output.moti"
    source.write_text(sequence_template, encoding="utf-8")
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "motion-finish",
            str(source),
            str(output),
            "--top",
            "0.1",
            "0.2",
            "0.3",
            "--bottom",
            "0.4",
            "0.5",
            "0.6",
            "--font-size",
            "88",
            "--stroke-scale",
            "0.5",
            "--shadow-scale",
            "1",
        ],
    )
    motion_finish.main()
    assert '"selectedStyle": 20' in output.read_text(encoding="utf-8")
    assert "NotoSansJP-Thin_Bold" in output.read_text(encoding="utf-8")
    assert source.read_text(encoding="utf-8") == sequence_template
    with pytest.raises(FileExistsError):
        motion_finish.main()


def test_selected_style_rejects_static_or_filename_font_selection() -> None:
    with pytest.raises(ReplayError, match="PostScript"):
        selected_finish_style(
            top=(0.1, 0.2, 0.3),
            bottom=(0.4, 0.5, 0.6),
            font_name="NotoSansJP-VariableFont_wght",
            font_size=88,
            stroke_scale=0.5,
            shadow_scale=1,
        )


@pytest.mark.parametrize(
    "extra",
    [
        ["--top-srgb", "#24556A"],
        ["--border", "18"],
        ["--top-srgb", "#24556A", "--bottom-srgb", "#28596D"],
    ],
)
def test_cli_rejects_partial_or_mixed_style_configuration(
    sequence_template: str,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    extra: list[str],
) -> None:
    source, output = tmp_path / "input.moti", tmp_path / "output.moti"
    source.write_text(sequence_template, encoding="utf-8")
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "motion-finish",
            str(source),
            str(output),
            "--top",
            "0.1",
            "0.2",
            "0.3",
            "--bottom",
            "0.4",
            "0.5",
            "0.6",
            "--font-name",
            "NotoSansJP-Thin_Bold",
            "--font-size",
            "88",
            "--stroke-scale",
            "0.5",
            "--shadow-scale",
            "1",
            *extra,
        ],
    )
    with pytest.raises(SystemExit, match="2"):
        motion_finish.main()
    assert not output.exists()
