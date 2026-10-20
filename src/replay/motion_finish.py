"""Finish prepared Motion stages with a white band, black edges and a light toe.

This is a render candidate, not an installed editable FCP title or a contrast
certificate. Existing two-object producers and their bound inputs stay unchanged.
"""

import argparse
import copy
import math
import re
from dataclasses import dataclass, replace
from pathlib import Path
from xml.etree import ElementTree as ET

from replay import caption_style_defaults as defaults
from replay.caption_contrast import _hex_color
from replay.caption_font_identity import require_selected_name
from replay.core import ReplayError
from replay.motion_caption import (
    GENERATED_ID_MAX,
    MAX_SEQUENCE_SENTENCES,
    MAX_XML_CHARACTERS,
    CaptionAppearance,
    ColourIntentPolicy,
    _attribute,
    _generated_ids,
    _gradient,
    _one,
    _outline,
    _parameter,
    _parse,
    _remap_tree,
    _sequence_tree_ids,
    _shadow,
    _stamp_colour_intent,
)


@dataclass(frozen=True)
class FinishStyle:
    """Geometry is in the already prepared document's units; RGB is Motion RGB."""

    bottom: tuple[float, float, float]
    bright_start: float = 0.8  # where the top colour starts blending into the bottom one
    blend_end: float = 1.0  # where the bottom colour is reached
    keyline: float = 2.0
    white_border: float = 10.92
    black_rim: float = 5.46
    shadow_opacity: float = 0.4
    shadow_distance: float = 3.0
    shadow_blur: float = 3.0
    top: tuple[float, float, float] | None = None
    top_srgb: tuple[float, float, float] | None = None
    bottom_srgb: tuple[float, float, float] | None = None
    font_name: str | None = None
    font_size: float | None = None
    midpoint: float = 0.8
    intent_policy: ColourIntentPolicy = "white-aaa"

    def appearance(self) -> CaptionAppearance:
        if self.top is None:
            raise ReplayError(
                "Replace the top Motion control explicitly; inherited colour is unbound"
            )
        result = CaptionAppearance(
            top=self.top,
            bottom=self.bottom,
            top_srgb=self.top_srgb,
            bottom_srgb=self.bottom_srgb,
            top_location=self.bright_start,
            bottom_location=self.blend_end,
            midpoint=self.midpoint,
            intent_policy=self.intent_policy,
            white_border=self.white_border,
            black_rim=self.black_rim,
            shadow_opacity=self.shadow_opacity,
            shadow_distance=self.shadow_distance,
            shadow_blur=self.shadow_blur,
        )
        result.validate()
        if (self.font_name is None) != (self.font_size is None):
            raise ReplayError(
                "Font replacement needs both a PostScript name and measured point size"
            )
        if self.font_name is not None and (
            not isinstance(self.font_name, str)
            or re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,99}", self.font_name) is None
        ):
            raise ReplayError("Invalid PostScript font name")
        if self.font_size is not None:
            replace(result, font_size=self.font_size).validate()
        if (
            isinstance(self.keyline, bool)
            or not math.isfinite(self.keyline)
            or not 0 < self.keyline < self.white_border
        ):
            raise ReplayError("Keyline must be positive and narrower than the white band")
        return result


def selected_finish_style(
    *,
    top: tuple[float, float, float],
    bottom: tuple[float, float, float],
    font_name: str = defaults.FONT_POSTSCRIPT_NAME,
    font_size: float,
    stroke_scale: float,
    shadow_scale: float,
) -> FinishStyle:
    """Apply the adopted constants with explicit renderer-unit calibration.

    Scales convert SVG stroke widths and shadow distances respectively to this
    prepared Motion document's units. They must be independently measured; this
    factory neither assumes a half-width mapping nor converts sRGB to controls.
    """
    require_selected_name(font_name)
    if any(
        isinstance(value, bool) or not math.isfinite(value) or value <= 0
        for value in (stroke_scale, shadow_scale)
    ):
        raise ReplayError("Caption reference scales must be finite and positive")
    result = FinishStyle(
        top=top,
        bottom=bottom,
        top_srgb=_hex_color(defaults.GRADIENT_TOP_HEX),
        bottom_srgb=_hex_color(defaults.GRADIENT_BOTTOM_HEX),
        bright_start=defaults.MOTION_GRADIENT_TOP_LOCATION,
        blend_end=defaults.MOTION_GRADIENT_BOTTOM_LOCATION,
        midpoint=0.5,
        keyline=defaults.INNER_BLACK_STROKE_WIDTH * stroke_scale,
        white_border=defaults.WHITE_STROKE_WIDTH * stroke_scale,
        black_rim=(defaults.OUTER_BLACK_STROKE_WIDTH - defaults.WHITE_STROKE_WIDTH) * stroke_scale,
        shadow_opacity=defaults.SHADOW_OPACITY,
        shadow_distance=defaults.SHADOW_OFFSET_Y * shadow_scale,
        shadow_blur=defaults.SHADOW_BLUR_STD_DEVIATION * shadow_scale,
        font_name=font_name,
        font_size=font_size,
        intent_policy="selected-default",
    )
    result.appearance()
    return result


def _top_color(style: ET.Element) -> tuple[float, float, float]:
    stops = _parameter(_parameter(_parameter(style, "14"), "17"), "1").findall(
        "parameter[@factoryID='4']"
    )
    if len(stops) != 2:
        raise ReplayError("Expected two original gradient stops")
    color = _parameter(stops[0], "3")
    try:
        return (
            float(_attribute(_one(_parameter(color, "1"), "curve"), "value")),
            float(_attribute(_one(_parameter(color, "2"), "curve"), "value")),
            float(_attribute(_one(_parameter(color, "3"), "curve"), "value")),
        )
    except ValueError as error:
        raise ReplayError("Invalid Motion gradient colour") from error


def _controls(node: ET.Element, path: str) -> bytes:
    """Compare controls, not each object's private generated paragraph IDs."""
    control = copy.deepcopy(_one(node, path))
    identifiers = set().union(*(_generated_ids(child) for child in control.iter()))
    _sequence_tree_ids(
        control, dict(zip(sorted(identifiers), range(len(identifiers)), strict=True))
    )
    return ET.tostring(control)


def _pairs(root: ET.Element) -> list[tuple[ET.Element, ET.Element]]:
    if root.get("version") != "5.13":
        raise ReplayError("Only the observed Motion 5.13 stages are supported")
    nodes = _one(root, "scene/layer").findall("scenenode[@factoryID='16']")
    if not nodes or len(nodes) % 2 or len(nodes) > 2 * MAX_SEQUENCE_SENTENCES:
        raise ReplayError("Expected bounded pairs of generated caption objects")
    pairs = list(zip(nodes[::2], nodes[1::2], strict=True))
    for face, rim in pairs:
        if (face.get("name"), rim.get("name")) != ("Caption Face", "Caption Outer Rim"):
            raise ReplayError("Expected ordered caption face/rim pairs")
        for path in ("timing", "parameter[@id='1']", "parameter[@id='2']"):
            if _controls(face, path) != _controls(rim, path):
                raise ReplayError("Caption geometry, text or animation is not synchronized")
    return pairs


def finish_sequence(document: str, *, style: FinishStyle) -> str:
    """Keep every stage's clocks, glyphs and position; add a synchronized face.

    Front to back: original gradient/black keyline, white band, black rim/shadow.
    Finishing is deliberately last: the historical two-object sequence generator
    must not consume this three-object result. No FCP publication is performed.
    """
    root = _parse(document)
    pairs = _pairs(root)
    layer = _one(root, "scene/layer")
    identifiers = set().union(*(_generated_ids(node) for node in root.iter()))
    if not identifiers:
        raise ReplayError("Missing generated Motion object IDs")
    maximum = max(identifiers)
    for face, rim in pairs:
        face_style, rim_style = _one(face, "style"), _one(rim, "style")
        _top_color(face_style)  # Validate the existing structure, never infer sRGB intent from it.
        appearance = style.appearance()
        foreground = copy.deepcopy(face)
        own = set().union(*(_generated_ids(node) for node in foreground.iter()))
        if not own or maximum + 1 + max(own) - min(own) >= GENERATED_ID_MAX:
            raise ReplayError("Motion object ID range exhausted")
        delta = maximum + 1 - min(own)
        _remap_tree(foreground, generated=own, delta=delta)
        maximum = max(own) + delta
        foreground.set("name", "Caption Keyline Face")
        foreground_style = _one(foreground, "style")
        if style.font_name is not None and style.font_size is not None:
            for text_style in (foreground_style, face_style, rim_style):
                _one(_parameter(text_style, "83"), "font").text = style.font_name
                _parameter(text_style, "3").set("value", str(style.font_size))
        _gradient(_parameter(foreground_style, "14"), appearance)
        _outline(foreground_style, face_style, width=style.keyline, color=(0, 0, 0))
        reference = copy.deepcopy(face_style)
        _gradient(
            _parameter(face_style, "14"), replace(appearance, top=(1, 1, 1), bottom=(1, 1, 1))
        )
        _outline(face_style, reference, width=style.white_border, color=(1, 1, 1))
        reference = copy.deepcopy(rim_style)
        _outline(rim_style, reference, width=style.white_border + style.black_rim, color=(0, 0, 0))
        _shadow(rim_style, reference, appearance=appearance)
        layer.insert(list(layer).index(face), foreground)
    _stamp_colour_intent(root, style.appearance())
    result = ET.tostring(root, encoding="unicode")
    if len(result) > MAX_XML_CHARACTERS:
        raise ReplayError("Finished Motion stages exceed the XML size limit")
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--bottom", type=float, nargs=3, required=True)
    parser.add_argument("--top", type=float, nargs=3, required=True)
    parser.add_argument("--top-srgb", type=_hex_color)
    parser.add_argument("--bottom-srgb", type=_hex_color)
    parser.add_argument(
        "--stroke-scale", type=float, help="Calibrated Motion units per SVG stroke unit"
    )
    parser.add_argument(
        "--shadow-scale", type=float, help="Calibrated Motion units per SVG shadow unit"
    )
    parser.add_argument("--font-name")
    parser.add_argument("--font-size", type=float)
    parser.add_argument("--bright-start", type=float)
    parser.add_argument("--keyline", type=float)
    parser.add_argument("--border", type=float)
    parser.add_argument("--rim", type=float)
    parser.add_argument("--shadow-opacity", type=float)
    parser.add_argument("--shadow-distance", type=float)
    parser.add_argument("--shadow-blur", type=float)
    args = parser.parse_args()
    if (args.top_srgb is None) != (args.bottom_srgb is None):
        parser.error("Explicit colour overrides need both --top-srgb and --bottom-srgb")
    if args.top_srgb is None and any(
        value is None for value in (args.font_size, args.stroke_scale, args.shadow_scale)
    ):
        parser.error("Default style needs --font-size, --stroke-scale and --shadow-scale")
    if (
        args.source.is_symlink()
        or not args.source.is_file()
        or args.source.stat().st_size > MAX_XML_CHARACTERS
    ):
        raise ReplayError("Motion source must be a bounded regular file")
    style = FinishStyle(
        bottom=(args.bottom[0], args.bottom[1], args.bottom[2]),
        top=(args.top[0], args.top[1], args.top[2]),
        top_srgb=args.top_srgb,
        bottom_srgb=args.bottom_srgb,
        font_name=args.font_name,
        font_size=args.font_size,
        bright_start=args.bright_start
        if args.bright_start is not None
        else FinishStyle.bright_start,
        keyline=args.keyline if args.keyline is not None else FinishStyle.keyline,
        white_border=args.border if args.border is not None else FinishStyle.white_border,
        black_rim=args.rim if args.rim is not None else FinishStyle.black_rim,
        shadow_opacity=args.shadow_opacity
        if args.shadow_opacity is not None
        else FinishStyle.shadow_opacity,
        shadow_distance=args.shadow_distance
        if args.shadow_distance is not None
        else FinishStyle.shadow_distance,
        shadow_blur=args.shadow_blur if args.shadow_blur is not None else FinishStyle.shadow_blur,
    )
    if args.top_srgb is None and args.bottom_srgb is None:
        if any(
            value is not None
            for value in (
                args.bright_start,
                args.keyline,
                args.border,
                args.rim,
                args.shadow_opacity,
                args.shadow_distance,
                args.shadow_blur,
            )
        ):
            parser.error(
                "Default style geometry comes from constants; use calibrated scales, not overrides"
            )
        style = selected_finish_style(
            top=(args.top[0], args.top[1], args.top[2]),
            bottom=(args.bottom[0], args.bottom[1], args.bottom[2]),
            font_name=args.font_name or defaults.FONT_POSTSCRIPT_NAME,
            font_size=args.font_size,
            stroke_scale=args.stroke_scale,
            shadow_scale=args.shadow_scale,
        )
    elif args.stroke_scale is not None or args.shadow_scale is not None:
        parser.error(
            "Reference scales require the selected default; explicit styles use document units"
        )
    result = finish_sequence(args.source.read_text(encoding="utf-8"), style=style)
    with args.output.open("x", encoding="utf-8") as stream:
        stream.write(result)


if __name__ == "__main__":
    main()
