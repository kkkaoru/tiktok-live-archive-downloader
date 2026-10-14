"""Explicit intent and candidate annotations are not native-render approval."""

import json
import sys
from dataclasses import replace
from xml.etree import ElementTree as ET

import pytest
from test_motion_caption import BASE, REFERENCE
from test_motion_caption import text_template as text_template

from replay import motion_caption, motion_finish
from replay.core import ReplayError
from replay.motion_caption import (
    COLOUR_INTENT_MARKER,
    DEFAULT_APPEARANCE,
    CaptionAppearance,
    build_caption_template,
    replace_caption_text,
    restyle_caption_template,
)
from replay.motion_finish import FinishStyle


@pytest.mark.parametrize("missing", ["top_srgb", "bottom_srgb"])
def test_both_intent_stops_are_mandatory(missing: str) -> None:
    appearance = replace(DEFAULT_APPEARANCE)
    object.__setattr__(appearance, missing, None)
    with pytest.raises(ReplayError, match="Explicit sRGB intent"):
        build_caption_template(BASE, REFERENCE, appearance=appearance)


def test_no_implicit_constructor_intent() -> None:
    with pytest.raises(ReplayError, match="Explicit sRGB intent"):
        CaptionAppearance().validate()


@pytest.mark.parametrize("stop", ["top_srgb", "bottom_srgb"])
def test_dark_raw_controls_cannot_excuse_bright_intent(stop: str) -> None:
    appearance = replace(DEFAULT_APPEARANCE)
    object.__setattr__(appearance, stop, (0.7, 0.83, 0.87))
    with pytest.raises(ReplayError, match="below 7:1"):
        build_caption_template(BASE, REFERENCE, appearance=appearance)


def test_raw_controls_are_not_claimed_to_be_srgb() -> None:
    document = build_caption_template(
        BASE,
        REFERENCE,
        appearance=replace(DEFAULT_APPEARANCE, top=(0.8, 0.8, 0.8), bottom=(0.9, 0.9, 0.9)),
    )
    root = ET.fromstring(document, parser=ET.XMLParser(target=ET.TreeBuilder(insert_comments=True)))
    comment = next(child for child in root if not isinstance(child.tag, str))
    payload = json.loads((comment.text or "").removeprefix(COLOUR_INTENT_MARKER))
    assert payload["motionControls"]["top"] == [0.8, 0.8, 0.8]
    assert payload["intent"]["space"] == "sRGB"
    assert payload["intent"]["top"] == [36 / 255, 85 / 255, 106 / 255]
    assert payload["requiredWhiteContrast"] == 7.0
    assert payload["renderedContrastVerified"] is False


def test_raw_control_component_count() -> None:
    appearance = replace(DEFAULT_APPEARANCE)
    object.__setattr__(appearance, "top", (0.1, 0.2))
    with pytest.raises(ReplayError, match="three components"):
        appearance.validate()


def test_annotation_is_replaced_on_restyle_and_preserved_on_text_change(text_template: str) -> None:
    restyled = restyle_caption_template(
        text_template,
        appearance=replace(DEFAULT_APPEARANCE, top_srgb=(0.0, 0.0, 0.0)),
    )
    changed = replace_caption_text(restyled, expected_text="Title", text="別の字幕")
    assert changed.count(COLOUR_INTENT_MARKER) == 1
    root = ET.fromstring(changed, parser=ET.XMLParser(target=ET.TreeBuilder(insert_comments=True)))
    comment = next(child for child in root if not isinstance(child.tag, str))
    payload = json.loads((comment.text or "").removeprefix(COLOUR_INTENT_MARKER))
    assert payload["intent"]["top"] == [0.0, 0.0, 0.0]
    assert payload["renderedContrastVerified"] is False


def test_finish_cannot_inherit_unbound_top_control() -> None:
    with pytest.raises(ReplayError, match="inherited colour is unbound"):
        FinishStyle(bottom=(0.1, 0.2, 0.3), top_srgb=(0, 0, 0), bottom_srgb=(0, 0, 0)).appearance()


def test_finish_requires_explicit_intent() -> None:
    with pytest.raises(ReplayError, match="Explicit sRGB intent"):
        FinishStyle(top=(0.1, 0.2, 0.3), bottom=(0.1, 0.2, 0.3)).appearance()


def test_restyle_cli_does_not_infer_intent(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "restyle",
            "unused.moti",
            "unused-out.moti",
            "--font-size",
            "30",
            "--border",
            "6",
            "--rim",
            "3",
            "--fill",
            "0.1",
            "0.2",
            "0.3",
        ],
    )
    with pytest.raises(SystemExit, match="2"):
        motion_caption.main()


def test_finish_cli_does_not_infer_intent(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "finish",
            "unused.moti",
            "unused-out.moti",
            "--top",
            "0.1",
            "0.2",
            "0.3",
            "--bottom",
            "0.1",
            "0.2",
            "0.3",
        ],
    )
    with pytest.raises(SystemExit, match="2"):
        motion_finish.main()
