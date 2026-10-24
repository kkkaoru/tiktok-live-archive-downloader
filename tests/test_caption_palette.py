"""A recording may recolour the default caption gradient; nothing else of the style changes."""

import json
from pathlib import Path

import pytest

from replay import caption_palette as palette
from replay.core import ReplayError

GREEN = {"gradientTop": "#B6DF9E", "gradientBottom": "#6DBE3C"}
STAMPED = (
    '<x><!--replay-caption-colour-v1 {"intent": {"bottom": [0.1, 0.2, 0.3], "space": "sRGB", '
    '"top": [0.4, 0.5, 0.6]}, "intentPolicy": "selected-default", "motionControls": {}, '
    '"selectedStyle": 20}--><y/></x>'
)


def test_the_intent_is_optional_and_strict(tmp_path: Path) -> None:
    assert palette.intent(tmp_path) is None
    (tmp_path / "caption-colors.json").write_text(json.dumps({"version": 1, **GREEN}), "utf-8")
    assert palette.intent(tmp_path) == GREEN
    (tmp_path / "caption-colors.json").write_text(json.dumps(GREEN), "utf-8")
    with pytest.raises(ReplayError, match="caption-colors.json must be"):
        palette.intent(tmp_path)


@pytest.mark.parametrize(
    ("value", "message"),
    [
        ({"gradientTop": "#B6DF9E"}, "exactly gradientTop and gradientBottom"),
        ("#B6DF9E", "exactly gradientTop and gradientBottom"),
        ({"gradientTop": "#b6df9e", "gradientBottom": "#6DBE3C"}, "upper-case"),
        ({"gradientTop": "#6DBE3C", "gradientBottom": "#B6DF9E"}, "brighter"),
    ],
)
def test_malformed_palettes_are_refused(value: object, message: str) -> None:
    with pytest.raises(ReplayError, match=message):
        palette.validated(value)


def test_the_finish_controls_take_the_palette_and_keep_everything_else() -> None:
    finish = ["--top", "0.5", "0.5", "0.5", "--bottom", "0.1", "0.1", "0.1", "--font-size", "82"]
    assert palette.finish_arguments(finish, GREEN) == [
        "--top", "0.7137254901960784", "0.8745098039215686", "0.6196078431372549",
        "--bottom", "0.42745098039215684", "0.7450980392156863", "0.23529411764705882",
        "--font-size", "82",
    ]  # fmt: skip
    with pytest.raises(ReplayError, match="--top with three values once"):
        palette.finish_arguments(["--top", "1"], GREEN)


def test_captions_settings_are_unchanged_without_a_palette() -> None:
    defaults: dict[str, object] = {
        "motionFinish": ["--top", "1", "1", "1", "--bottom", "0", "0", "0"]
    }
    assert palette.captions_settings(defaults, None) == {"version": 1, **defaults}
    recoloured = palette.captions_settings(defaults, GREEN)
    assert recoloured["palette"] == GREEN
    assert recoloured["motionFinish"] == palette.finish_arguments(
        ["--top", "1", "1", "1", "--bottom", "0", "0", "0"], GREEN
    )
    with pytest.raises(ReplayError, match="list of arguments"):
        palette.captions_settings({"motionFinish": "x"}, GREEN)


def test_a_finished_part_is_restamped_with_the_colours_it_renders() -> None:
    text = palette.restamp(STAMPED, GREEN)
    payload = json.loads(text.split("replay-caption-colour-v1 ", 1)[1].split("-->", 1)[0])
    assert payload["intent"] == {
        "bottom": [0.42745098039215684, 0.7450980392156863, 0.23529411764705882],
        "space": "sRGB",
        "top": [0.7137254901960784, 0.8745098039215686, 0.6196078431372549],
    }
    assert payload["intentPolicy"] == "recording-palette"
    assert payload["palette"] == GREEN
    assert round(payload["selectedWhiteContrast"]["bottom"], 2) == 2.31
    assert text.endswith("--><y/></x>")
    with pytest.raises(ReplayError, match="one colour intent"):
        palette.restamp("<x/>", GREEN)
