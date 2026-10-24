"""Native caption render requests built from synthetic Motion part plans and reveal records."""

import json
from pathlib import Path

import pytest

from replay.caption_native_request import main, request, style, titles
from replay.core import ReplayError


def plan(**changes: object) -> dict[str, object]:
    return {
        "frames": 60,
        "width": 1080,
        "height": 1920,
        "rate": 30,
        "expectedText": "Title",
        "placements": [
            {"text": "ちょっと", "startFrame": 0, "endFrame": 10, "x": 0.0, "y": 420.0},
            {"text": "ちょっと待って", "startFrame": 10, "endFrame": 30, "x": 0.0, "y": 420.0},
            {"text": "東京", "startFrame": 40, "endFrame": 60, "x": -37.25, "y": -200.0},
        ],
        **changes,
    }


GROWING = {"text": "ちょっと待って", "startFrame": 0, "endFrame": 30,
           "graphemeFrames": [0, 2, 4, 6, 10, 15, 20]}  # fmt: skip
SINGLE = {"text": "東京", "startFrame": 40, "endFrame": 60, "graphemeFrames": [40, 50]}
REVEAL = {"offsetFrame": 0, "titles": [GROWING, SINGLE]}


def test_style_twenty_is_converted_to_output_pixels_with_motion_measurements() -> None:
    assert style(font_size=82.5, stroke_scale=0.5, shadow_scale=1.0) == {
        "fontName": "NotoSansJP-Thin_Bold",
        "fontSize": 82.5,
        "faceGrow": 0.3,
        "keyline": 1.443,
        "whiteBorder": 10.08,
        "rim": 13.08,
        "shadowOpacity": 0.7,
        "shadowOffsetY": 16.0,
        "shadowSigma": pytest.approx(5.656854),
        "gradientTop": "#97D8F7",
        "gradientBottom": "#2FB0EE",
        "topLocation": pytest.approx(0.2688),
        "bottomLocation": pytest.approx(0.3944),
        "outputGamma": 1.961,
    }


def test_a_recording_palette_replaces_only_the_gradient_colours() -> None:
    recoloured = style(
        font_size=82.5, stroke_scale=0.5, shadow_scale=1.0,
        palette={"gradientTop": "#F7C8D8", "gradientBottom": "#E04A7A"},
    )  # fmt: skip
    assert (recoloured["gradientTop"], recoloured["gradientBottom"], recoloured["rim"]) == (
        "#F7C8D8",
        "#E04A7A",
        13.08,
    )


def test_style_refuses_a_nonpositive_scale() -> None:
    with pytest.raises(ReplayError, match="finite and positive"):
        style(font_size=82.5, stroke_scale=0.0, shadow_scale=1.0)


def test_each_growth_chain_becomes_one_title_at_its_motion_position() -> None:
    assert titles(plan(), REVEAL) == [
        {"text": "ちょっと待って", "startFrame": 0, "endFrame": 30,
         "graphemeFrames": [0, 2, 4, 6, 10, 15, 20], "x": 0.0, "y": 420.0},
        {"text": "東京", "startFrame": 40, "endFrame": 60, "graphemeFrames": [40, 50],
         "x": -37.25, "y": -200.0},
    ]  # fmt: skip


def test_top_left_canvas_coordinates_are_moved_to_the_motion_origin() -> None:
    document = plan(
        coordinateSpace="canvas-top-left",
        placements=[{"text": "東京", "startFrame": 40, "endFrame": 60, "x": 540.0, "y": 760.0}],
    )
    assert titles(document, {"titles": [SINGLE]}) == [
        {"text": "東京", "startFrame": 40, "endFrame": 60, "graphemeFrames": [40, 50],
         "x": 0.0, "y": 200.0},
    ]  # fmt: skip


def test_a_reveal_record_of_another_part_is_refused() -> None:
    with pytest.raises(ReplayError, match="does not match"):
        titles(plan(), {"titles": [SINGLE, GROWING]})


def test_a_reveal_record_with_a_different_title_count_is_refused() -> None:
    with pytest.raises(ReplayError, match="does not match"):
        titles(plan(), {"titles": [GROWING]})


def test_a_canvas_motion_would_rescale_is_refused() -> None:
    with pytest.raises(ReplayError, match="reference canvas"):
        titles(plan(width=540, height=960), REVEAL)


def test_an_unknown_coordinate_space_is_refused() -> None:
    with pytest.raises(ReplayError, match="coordinate space"):
        titles(plan(coordinateSpace="screen"), REVEAL)


def test_placements_must_be_a_list() -> None:
    with pytest.raises(ReplayError, match="must be an array"):
        titles(plan(placements={}), REVEAL)


def test_the_request_carries_the_canvas_clock_and_output(tmp_path: Path) -> None:
    document = request(
        plan(), REVEAL, tmp_path / "native.mov", font_size=82.5, stroke_scale=0.5,
        shadow_scale=1.0,
    )  # fmt: skip
    assert (document["output"], document["width"], document["height"]) == (
        str(tmp_path / "native.mov"),
        1080,
        1920,
    )
    assert (document["rate"], document["frames"]) == (30, 60)
    assert document["titles"] == titles(plan(), REVEAL)


def test_main_writes_the_request_once(tmp_path: Path) -> None:
    (tmp_path / "plan.json").write_text(json.dumps(plan()), encoding="utf-8")
    (tmp_path / "reveal.json").write_text(json.dumps(REVEAL), encoding="utf-8")
    arguments = [
        str(tmp_path / "plan.json"), str(tmp_path / "reveal.json"), str(tmp_path / "out.mov"),
        str(tmp_path / "request.json"), "--font-size", "82.5", "--stroke-scale", "0.5",
        "--shadow-scale", "1.0",
    ]  # fmt: skip
    assert main(arguments) == 0
    written = json.loads((tmp_path / "request.json").read_text(encoding="utf-8"))
    assert written["titles"][1]["text"] == "東京"
    with pytest.raises(FileExistsError):
        main(arguments)


def test_main_takes_a_validated_palette(tmp_path: Path) -> None:
    (tmp_path / "plan.json").write_text(json.dumps(plan()), encoding="utf-8")
    (tmp_path / "reveal.json").write_text(json.dumps(REVEAL), encoding="utf-8")
    arguments = [
        str(tmp_path / "plan.json"), str(tmp_path / "reveal.json"), str(tmp_path / "out.mov"),
        str(tmp_path / "request.json"), "--font-size", "82.5", "--stroke-scale", "0.5",
        "--shadow-scale", "1.0", "--palette",
    ]  # fmt: skip
    assert main([*arguments, "#F7C8D8", "#E04A7A"]) == 0
    written = json.loads((tmp_path / "request.json").read_text(encoding="utf-8"))
    assert written["style"]["gradientBottom"] == "#E04A7A"
    (tmp_path / "request.json").unlink()
    with pytest.raises(ReplayError, match="brighter"):
        main([*arguments, "#E04A7A", "#F7C8D8"])
