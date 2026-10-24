"""Build a two-object caption title from local, version-matched Motion templates.

Apple templates are read at runtime, never redistributed. The output is experimental
Motion XML, not evidence of a successful render. Import caption text as plain
``<text>``: FCP text-style overrides can replace the template's gradient with a solid
face. The caller must register the new title and verify its actual rendered output.
"""

from __future__ import annotations

import argparse
import copy
import json
import math
import re
from dataclasses import dataclass
from fractions import Fraction
from itertools import pairwise
from pathlib import Path
from typing import Literal
from xml.etree import ElementTree as ET

from replay import caption_style_defaults
from replay.caption_contrast import WHITE, _hex_color, contrast_ratio, require_white_contrast
from replay.core import ReplayError

MAX_XML_CHARACTERS = 4 * 1024 * 1024
MAX_CAPTION_TEXT_UNITS = 120
MAX_POSITION_UNITS = 10_000
MAX_SEQUENCE_SECONDS = 120
MAX_SEQUENCE_SENTENCES = 32
# A Compressor caption part is one whole caption window (``caption_parts.MAX_SECONDS``): each
# Compressor job costs ~4.5 s fixed, which was 64% of caption-render with 32-sentence parts.
# Windows held at most 252 stages over 22 recordings (~38 KB of finished XML per stage).
MAX_PART_SENTENCES = 512
MAX_PART_XML_CHARACTERS = 32 * 1024 * 1024
MAX_SEQUENCE_PIXELS = 4096 * 2160
GENERATED_ID_MIN = 999_000_000
GENERATED_ID_MAX = 1_000_000_000
# Experimental Type On controls. End 321 was observed in a Motion 6.3 (ozml 5.15) save as
# Object/TypeOn/End ("終了"); Fade In 322 in the installed Line Reveal title's "Title" object
# (ozml 5.13; its "Subtitle" object uses flags 8589934608 instead). English names are labels.
TYPE_ON_GROUP: tuple[str, str, str] = ("Type On", "319", "8589938704")
TYPE_ON_CONTROLS: tuple[tuple[str, str, str, str], ...] = (
    ("End", "321", "8589934608", "1"),
    ("Fade In", "322", "8606711824", "0"),
)
TYPE_ON_CHANNEL: str = "./2/319/321"
ColourIntentPolicy = Literal["white-aaa", "selected-default"]


@dataclass(frozen=True, slots=True)
class CaptionAppearance:
    """Observed Motion reference units, not sRGB or output-pixel measurements.

    Explicit sRGB intent is checked independently of raw controls. The historical
    white-AAA policy stays strict; selected-default accepts only the adopted
    constants and records their actual, non-AAA ratios. Neither validation nor
    the annotation certifies rendered contrast or colour correspondence.
    """

    font_size: float = 66 * 1000 / 1332 * 1080 / 1920
    white_border: float = 6 * 1080 / 1920
    black_rim: float = 3 * 1080 / 1920
    border_color: tuple[float, float, float] = (1.0, 1.0, 1.0)
    top: tuple[float, float, float] = (0.012, 0.055, 0.08)
    bottom: tuple[float, float, float] = (0.02, 0.085, 0.12)
    top_location: float = 0.15
    bottom_location: float = 0.70
    midpoint: float = 0.8
    shadow_opacity: float = 0.1
    shadow_distance: float = 1080 / 1920
    shadow_blur: float = 2 * 1080 / 1920
    top_srgb: tuple[float, float, float] | None = None
    bottom_srgb: tuple[float, float, float] | None = None
    intent_policy: ColourIntentPolicy = "white-aaa"

    def validate(self) -> None:
        units = (self.font_size, self.white_border, self.black_rim)
        bounded = (
            *self.border_color,
            *self.top,
            *self.bottom,
            self.top_location,
            self.bottom_location,
            self.midpoint,
            self.shadow_opacity,
        )
        values = (*units, *bounded, self.shadow_distance, self.shadow_blur)
        if any(isinstance(value, bool) or not math.isfinite(value) for value in values):
            raise ReplayError("Caption appearance requires finite numeric values")
        if any(not 0 < value <= 1080 for value in units):
            raise ReplayError("Caption size and both outline widths must be positive")
        if any(len(color) != 3 for color in (self.top, self.bottom, self.border_color)):
            raise ReplayError("Caption RGB controls require three components")
        if any(not 0 <= v <= 1 for v in bounded):
            raise ReplayError("Invalid caption RGB, gradient position, or opacity")
        if not self.top_location < self.bottom_location or not 0 < self.midpoint < 1:
            raise ReplayError("Caption gradient stops and midpoint must be ordered")
        if not 0 <= self.shadow_distance <= 1080 or not 0 <= self.shadow_blur <= 1080:
            raise ReplayError("Invalid caption shadow geometry")
        if self.top_srgb is None or self.bottom_srgb is None:
            raise ReplayError("Explicit sRGB intent is required for both caption stops")
        # Always validate and calculate the actual ratios, even for the adopted non-AAA palette.
        contrast_ratio(WHITE, self.top_srgb)
        contrast_ratio(WHITE, self.bottom_srgb)
        if self.intent_policy == "selected-default":
            if (
                self.top_srgb != _hex_color(caption_style_defaults.GRADIENT_TOP_HEX)
                or self.bottom_srgb != _hex_color(caption_style_defaults.GRADIENT_BOTTOM_HEX)
                or self.top_location != caption_style_defaults.MOTION_GRADIENT_TOP_LOCATION
                or self.bottom_location != caption_style_defaults.MOTION_GRADIENT_BOTTOM_LOCATION
                or self.midpoint != 0.5
            ):
                raise ReplayError(
                    "Selected default must match the adopted colour and gradient constants"
                )
        elif self.intent_policy == "white-aaa":
            require_white_contrast((self.top_srgb, self.bottom_srgb))
            # An sRGB component envelope also bounds bounded RGB interpolation.
            envelope = tuple(
                max(a, b) for a, b in zip(self.top_srgb, self.bottom_srgb, strict=True)
            )
            require_white_contrast((envelope,))
        else:
            raise ReplayError("Unknown caption colour intent policy")


DEFAULT_APPEARANCE = CaptionAppearance(
    top_srgb=(36 / 255, 85 / 255, 106 / 255),
    bottom_srgb=(40 / 255, 89 / 255, 109 / 255),
)
COLOUR_INTENT_MARKER = "replay-caption-colour-v1 "


def _stamp_colour_intent(root: ET.Element, appearance: CaptionAppearance) -> None:
    """Attach intent to candidate bytes, never a render approval or colour conversion."""
    appearance.validate()
    for child in list(root):
        if not isinstance(child.tag, str) and (child.text or "").startswith(COLOUR_INTENT_MARKER):
            root.remove(child)
    payload: dict[str, object] = {
        "intent": {"space": "sRGB", "top": appearance.top_srgb, "bottom": appearance.bottom_srgb},
        "motionControls": {
            "top": appearance.top,
            "bottom": appearance.bottom,
            "topLocation": appearance.top_location,
            "bottomLocation": appearance.bottom_location,
            "midpoint": appearance.midpoint,
        },
        "requiredWhiteContrast": 7.0,
        "renderedContrastVerified": False,
    }
    if appearance.intent_policy == "selected-default":
        payload.update(
            intentPolicy=appearance.intent_policy,
            selectedStyle=caption_style_defaults.DEFAULT_CAPTION_STYLE_ID,
            requiredWhiteContrast=None,
            selectedWhiteContrast={
                "top": contrast_ratio(WHITE, _hex_color(caption_style_defaults.GRADIENT_TOP_HEX)),
                "bottom": contrast_ratio(
                    WHITE, _hex_color(caption_style_defaults.GRADIENT_BOTTOM_HEX)
                ),
            },
            renderedColourVerified=False,
        )
    root.append(ET.Comment(COLOUR_INTENT_MARKER + json.dumps(payload, sort_keys=True)))


@dataclass(frozen=True, slots=True)
class CaptionPlacement:
    """One bounded Motion sentence, with explicit local-origin and reveal clocks."""

    text: str
    start_frame: int
    end_frame: int
    samples: tuple[tuple[Fraction, Fraction], ...]
    x: float
    y: float


def _parse(text: str, *, limit: int = MAX_XML_CHARACTERS) -> ET.Element:
    if len(text) > limit:
        raise ReplayError("Motion template exceeds the XML size limit")
    # Only Motion's observed empty declaration is allowed; never load a DTD/entity.
    text = text.replace("<!DOCTYPE ozxmlscene>", "")
    if re.search(r"<!\s*(?:DOCTYPE|ENTITY)\b", text, re.IGNORECASE):
        raise ReplayError("External or custom Motion XML declarations are not allowed")
    try:
        root = ET.fromstring(text, parser=ET.XMLParser(target=ET.TreeBuilder(insert_comments=True)))
    except ET.ParseError as error:
        raise ReplayError("Invalid Motion XML") from error
    if root.tag != "ozml":
        raise ReplayError("Expected a Motion ozml document")
    return root


def _one(parent: ET.Element, path: str) -> ET.Element:
    found = parent.findall(path)
    if len(found) != 1:
        raise ReplayError(f"Unsupported Motion template structure: {path}")
    return found[0]


def _parameter(parent: ET.Element, identifier: str) -> ET.Element:
    return _one(parent, f"parameter[@id='{identifier}']")


def _attribute(node: ET.Element, name: str) -> str:
    value = node.get(name)
    if not value:
        raise ReplayError(f"Missing Motion attribute: {name}")
    return value


def _copy_control(target: ET.Element, reference: ET.Element, *, path: str) -> ET.Element:
    control = copy.deepcopy(_one(reference, path))
    identifier = control.attrib["id"]
    for old in target.findall(f"parameter[@id='{identifier}']"):
        target.remove(old)
    target.append(control)
    return control


def _rgb(group: ET.Element, color: tuple[float, float, float], *, curves: bool = False) -> None:
    for number, value in enumerate(color, 1):
        channel = _parameter(group, str(number))
        target = _one(channel, "curve") if curves else channel
        target.set("value", str(value))


def _gradient(face: ET.Element, appearance: CaptionAppearance) -> None:
    gradient = _parameter(face, "17")
    stops = _parameter(gradient, "1").findall("parameter[@factoryID='4']")
    if len(stops) != 2:
        raise ReplayError("Expected exactly two Motion gradient color stops")
    for stop, color, location in zip(
        stops,
        (appearance.top, appearance.bottom),
        (appearance.top_location, appearance.bottom_location),
        strict=True,
    ):
        _one(_parameter(stop, "1"), "curve").set("value", str(location))
        _rgb(_parameter(stop, "3"), color, curves=True)
    _one(_parameter(stops[0], "2"), "curve").set("value", str(appearance.midpoint))


def _outline(
    style: ET.Element,
    reference: ET.Element,
    *,
    width: float,
    color: tuple[float, float, float],
) -> None:
    outline = _parameter(style, "30")
    outline.set("flags", _attribute(_parameter(style, "14"), "flags"))
    control = _copy_control(outline, reference, path="parameter[@id='30']/parameter[@id='32']")
    _rgb(control, color)
    for identifier, value in (("35", 1.0), ("36", width), ("50", 0.0)):
        _copy_control(
            outline, reference, path=f"parameter[@id='30']/parameter[@id='{identifier}']"
        ).set("value", str(value))


def _shadow(style: ET.Element, reference: ET.Element, *, appearance: CaptionAppearance) -> None:
    shadow = _parameter(style, "21")
    shadow.set("flags", _attribute(_parameter(style, "14"), "flags"))
    color = _copy_control(shadow, reference, path="parameter[@id='21']/parameter[@id='23']")
    _rgb(color, (0, 0, 0))
    for identifier, value in (
        ("26", appearance.shadow_opacity),
        ("27", appearance.shadow_distance),
        ("29", math.tau * 0.75),
    ):
        _copy_control(
            shadow, reference, path=f"parameter[@id='21']/parameter[@id='{identifier}']"
        ).set("value", str(value))
    blur = _copy_control(shadow, reference, path="parameter[@id='21']/parameter[@id='75']")
    _parameter(blur, "1").set("value", str(appearance.shadow_blur))
    _parameter(blur, "2").set("value", str(appearance.shadow_blur))


def _type_on(node: ET.Element) -> None:
    """Add full-reveal Type On controls after Flatten, never merging with existing ones."""
    obj = _parameter(node, "2")
    name, identifier, flags = TYPE_ON_GROUP
    if obj.findall(f"parameter[@id='{identifier}']"):
        raise ReplayError("Motion text object already has Type On controls")
    group = ET.Element("parameter", name=name, id=identifier, flags=flags)
    for label, control_id, control_flags, value in TYPE_ON_CONTROLS:
        ET.SubElement(
            group,
            "parameter",
            name=label,
            id=control_id,
            flags=control_flags,
            default="1",
            value=value,
        )
    obj.insert(list(obj).index(_parameter(obj, "351")) + 1, group)


def _generated_ids(node: ET.Element) -> set[int]:
    return {
        int(value)
        for key, value in node.attrib.items()
        if key in {"id", "style"}
        and value.isdecimal()
        and GENERATED_ID_MIN <= int(value) < GENERATED_ID_MAX
    }


def _remap_ids(node: ET.Element, *, generated: set[int], delta: int) -> None:
    for key, value in tuple(node.attrib.items()):
        if key in {"id", "style"} and value.isdecimal() and int(value) in generated:
            replacement = int(value) + delta
            if replacement >= GENERATED_ID_MAX:
                raise ReplayError("Motion object ID range exhausted")
            node.set(key, str(replacement))


def _remap_tree(node: ET.Element, *, generated: set[int], delta: int) -> None:
    for child in node.iter():
        _remap_ids(child, generated=generated, delta=delta)


def _duplicate(root: ET.Element, *, layer: ET.Element, title: ET.Element) -> ET.Element:
    """Remap object IDs and references, never local parameter or factory IDs."""
    generated: set[int] = set()
    for node in root.iter():
        generated.update(_generated_ids(node))
    if not generated:
        raise ReplayError("Missing generated Motion object IDs")
    delta = max(generated) - min(generated) + 1
    duplicate = copy.deepcopy(title)
    _remap_tree(duplicate, generated=generated, delta=delta)
    layer.insert(list(layer).index(title), duplicate)
    return duplicate


def build_caption_template(
    basic_title_xml: str,
    control_reference_xml: str,
    *,
    appearance: CaptionAppearance = DEFAULT_APPEARANCE,
    type_on: bool = False,
) -> str:
    """Return a new template; no files, installed titles, or applications are modified.

    The control reference supplies observed scalar/colour XML from the installed
    Motion template version. Neither input nor a vendor template is bundled here.

    ``type_on`` is an opt-in, unverified rendering candidate: it adds Type On End=1 and
    Fade In=0 to both text objects and publishes each End control. End was observed in a
    Motion 6.3 (ozml 5.15) save and Fade In in an installed ozml 5.13 title, while the base
    input remains the ozml 5.13 layout; native import/export must verify the result before
    use. The default output is unchanged.
    """
    appearance.validate()
    root = _parse(basic_title_xml)
    if root.get("version") != "5.13":
        raise ReplayError("Only the observed Basic Title ozml 5.13 layout is supported")
    settings = _one(root, "scene/sceneSettings")
    if (_one(settings, "width").text, _one(settings, "height").text) != ("1920", "1080"):
        raise ReplayError("Expected the 1920x1080 Basic Title reference canvas")
    layer = _one(root, "scene/layer")
    title = _one(layer, "scenenode[@factoryID='16']")
    references = _parse(control_reference_xml).findall(".//style")
    if not references:
        raise ReplayError("Missing Motion control reference style")
    reference = references[0]
    upper = _duplicate(root, layer=layer, title=title)
    upper.set("name", "Caption Face")
    title.set("name", "Caption Outer Rim")
    for node in (upper, title):
        style = _one(node, "style")
        _one(_parameter(style, "83"), "font").text = "HiraginoSans-W6"
        _parameter(style, "3").set("value", str(appearance.font_size))
    face_style = _one(upper, "style")
    face = _parameter(face_style, "14")
    _copy_control(face, reference, path="parameter[@id='14']/parameter[@id='15']").set("value", "1")
    _gradient(face, appearance)
    _outline(face_style, reference, width=appearance.white_border, color=appearance.border_color)
    rim_style = _one(title, "style")
    rim_face = _parameter(rim_style, "14")
    _copy_control(rim_face, reference, path="parameter[@id='14']/parameter[@id='15']").set(
        "value", "0"
    )
    ink = _copy_control(rim_face, reference, path="parameter[@id='14']/parameter[@id='16']")
    _rgb(ink, (0, 0, 0))
    _outline(
        rim_style,
        reference,
        width=appearance.white_border + appearance.black_rim,
        color=(0.0, 0.0, 0.0),
    )
    _shadow(rim_style, reference, appearance=appearance)
    published = _one(root, "scene/publishSettings")
    ET.SubElement(
        published,
        "target",
        object=_attribute(upper, "id"),
        channel=f"./5/{_attribute(face_style, 'id')}/14/15",
        name="Caption Fill Mode",
    )
    if type_on:
        for node, name in ((upper, "Caption Reveal Face"), (title, "Caption Reveal Rim")):
            _type_on(node)
            ET.SubElement(
                published,
                "target",
                object=_attribute(node, "id"),
                channel=TYPE_ON_CHANNEL,
                name=name,
            )
    _stamp_colour_intent(root, appearance)
    return ET.tostring(root, encoding="unicode")


def main() -> None:
    """Prepare a new styled template; rendering and validation remain separate."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("template", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--font-size", type=float, required=True)
    parser.add_argument("--border", type=float, required=True)
    parser.add_argument("--rim", type=float, required=True)
    parser.add_argument("--fill", type=float, nargs=3, required=True, metavar=("R", "G", "B"))
    parser.add_argument(
        "--fill-srgb",
        type=_hex_color,
        required=True,
        help="desired sRGB hex colour, distinct from --fill Motion controls",
    )
    parser.add_argument("--border-color", type=float, nargs=3, default=(1.0, 1.0, 1.0))
    args = parser.parse_args()
    appearance = CaptionAppearance(
        top_srgb=args.fill_srgb,
        bottom_srgb=args.fill_srgb,
        font_size=args.font_size,
        white_border=args.border,
        black_rim=args.rim,
        top=(args.fill[0], args.fill[1], args.fill[2]),
        bottom=(args.fill[0], args.fill[1], args.fill[2]),
        border_color=(args.border_color[0], args.border_color[1], args.border_color[2]),
    )
    result = restyle_caption_template(
        args.template.read_text(encoding="utf-8"), appearance=appearance
    )
    with args.output.open("x", encoding="utf-8") as handle:
        handle.write(result)


def _project_duration(root: ET.Element) -> Fraction:
    settings = _one(root, "scene/sceneSettings")
    try:
        frames = int(_one(settings, "duration").text or "")
        rate = Fraction(_one(settings, "frameRate").text or "")
    except (ValueError, ZeroDivisionError) as error:
        raise ReplayError("Invalid Motion project clock") from error
    ntsc = _one(settings, "NTSC").text
    if frames <= 0 or not 1 <= rate <= 240 or ntsc not in {"0", "1"}:
        raise ReplayError("Unsupported Motion project clock")
    return frames / (rate * Fraction(1000, 1001) if ntsc == "1" else rate)


def _reveal_curve(samples: tuple[tuple[Fraction, Fraction], ...], *, smooth: bool) -> ET.Element:
    """Observed scalar curve with linear or zero-slope cubic handles.

    The ozml 5.13 Typewriter reference uses interpolation=1, flags=0 and CMTime
    fields (value, timescale, valid flag, epoch). The same curve was consumed by
    both Type On End controls in a Compressor-rendered two-object title.
    """
    initial = str(float(samples[0][1]))
    curve = ET.Element("curve", type="1", default=initial, value=initial)
    ET.SubElement(curve, "numberOfKeypoints").text = str(len(samples))
    ET.SubElement(curve, "min").text = "0"
    ET.SubElement(curve, "max").text = "1"
    points: list[ET.Element] = []
    for time, value in samples:
        point = ET.SubElement(curve, "keypoint", interpolation="1", flags="0")
        ET.SubElement(point, "time").text = f"{time.numerator} {time.denominator} 1 0"
        ET.SubElement(point, "value").text = str(float(value))
        points.append(point)
    for index, ((start, lower), (end, upper)) in enumerate(pairwise(samples)):
        time_delta = (end - start) / 3
        value_delta = Fraction(0) if smooth else (upper - lower) / 3
        ET.SubElement(points[index], "outputTangentTime").text = str(float(time_delta))
        ET.SubElement(points[index], "outputTangentValue").text = str(float(value_delta))
        ET.SubElement(points[index + 1], "inputTangentTime").text = str(float(-time_delta))
        ET.SubElement(points[index + 1], "inputTangentValue").text = str(float(-value_delta))
    # Motion's observed terminal key retains an outgoing handle along its last segment.
    ET.SubElement(points[-1], "outputTangentTime").text = str(float(time_delta))
    ET.SubElement(points[-1], "outputTangentValue").text = str(float(value_delta))
    return curve


def _caption_objects(root: ET.Element) -> list[ET.Element]:
    if root.get("version") != "5.13":
        raise ReplayError("Only the observed Motion ozml 5.13 caption layout is supported")
    nodes = _one(root, "scene/layer").findall("scenenode[@factoryID='16']")
    if len(nodes) != 2 or {node.get("name") for node in nodes} != {
        "Caption Face",
        "Caption Outer Rim",
    }:
        raise ReplayError("Expected the two generated Motion caption objects")
    return nodes


def restyle_caption_template(template_xml: str, *, appearance: CaptionAppearance) -> str:
    """Copy observed caption styles without changing text, timing or reveal controls.

    Sizes are template units, before optional FCP canvas normalization. Colour
    values are Motion controls, not proof of rendered sRGB or contrast. Only the
    generated two-object layout is accepted; installed files are never modified.
    """
    appearance.validate()
    root = _parse(template_xml)
    for node in _caption_objects(root):
        style = _one(node, "style")
        reference = copy.deepcopy(style)
        _parameter(style, "3").set("value", str(appearance.font_size))
        face = node.get("name") == "Caption Face"
        if face:
            _gradient(_parameter(style, "14"), appearance)
        _outline(
            style,
            reference,
            width=appearance.white_border + (0 if face else appearance.black_rim),
            color=appearance.border_color if face else (0.0, 0.0, 0.0),
        )
        if not face:
            _shadow(style, reference, appearance=appearance)
    _stamp_colour_intent(root, appearance)
    result = ET.tostring(root, encoding="unicode")
    if len(result) > MAX_XML_CHARACTERS:
        raise ReplayError("Restyled Motion document exceeds the XML size limit")
    return result


def position_caption_template(template_xml: str, *, x: float, y: float) -> str:
    """Set both default title origins using observed Motion 5.13 position controls.

    Positive X moves right and positive Y moves up in the calibrated 1920x1080
    canvas. These are local origin coordinates, not text bounding-box coordinates.
    No text alignment, baseline, canvas scaling or head position is inferred.
    Existing transforms are refused; verify clipping and other canvases separately.
    """
    if any(
        isinstance(value, bool) or not math.isfinite(value) or abs(value) > MAX_POSITION_UNITS
        for value in (x, y)
    ):
        raise ReplayError("Motion position must be finite and within 10000 units")
    root = _parse(template_xml)
    transform = ET.Element("parameter", name="Transform", id="100", flags="8589938704")
    position = ET.SubElement(transform, "parameter", name="Position", id="101", flags="8589938704")
    ET.SubElement(position, "foldFlags").text = "11"
    # Observed scalar controls on a version-5 text node in an ozml 5.13 title.
    for identifier, name, value in (("1", "X", x), ("2", "Y", y)):
        ET.SubElement(
            position,
            "parameter",
            name=name,
            id=identifier,
            flags="8606711824",
            default="0",
            value=str(value),
        )
    for node in _caption_objects(root):
        properties = _parameter(node, "1")
        if properties.findall("parameter[@id='100']"):
            raise ReplayError("Motion caption already has a transform")
        properties.insert(0, copy.deepcopy(transform))
    result = ET.tostring(root, encoding="unicode")
    if len(result) > MAX_XML_CHARACTERS:
        raise ReplayError("Positioned Motion document exceeds the XML size limit")
    return result


def replace_caption_text(template_xml: str, *, expected_text: str, text: str) -> str:
    """Copy the observed flat, single-style BMP caption bodies without normalization.

    This experimental ozml 5.13 contract supports newlines and combining marks,
    not character tables, supplementary-plane characters, or arbitrary formatting.
    Set text before generating its reveal curves. Appearance/layout stay untouched;
    callers still verify glyph coverage, placement and rendering independently.
    """
    for value in (expected_text, text):
        if not 1 <= len(value) <= MAX_CAPTION_TEXT_UNITS or any(
            character != "\n"
            and not (32 <= ord(character) <= 0xD7FF or 0xE000 <= ord(character) <= 0xFFFD)
            for character in value
        ):
            raise ReplayError("Motion caption text requires 1–120 BMP units without controls")
    root = _parse(template_xml)
    for node in _caption_objects(root):
        body = _one(_parameter(_parameter(node, "2"), "369"), "text")
        run = _one(node, "styleRun")
        style = _one(node, "style")
        end = _parameter(_parameter(_parameter(node, "2"), "319"), "321")
        if list(end) or end.get("value") != "1":
            raise ReplayError("Replace Motion text before generating its reveal curves")
        if (
            list(body)
            or len(node.findall(".//text")) != 1
            or node.findall(".//character")
            or node.findall(".//kerning")
            or node.findall(".//object")
            or node.findall(".//parameter[@name='Kerning']")
            or body.text != expected_text
            or run.get("offset") != "0"
            or run.get("length") != str(len(expected_text))
            or run.get("style") != _attribute(style, "id")
        ):
            raise ReplayError("Motion caption body or single-style run does not match")
        body.text = text
        # All supported code points are BMP: one UTF-16 unit each, not one grapheme.
        run.set("length", str(len(text)))
    result = ET.tostring(root, encoding="unicode")
    if len(result) > MAX_XML_CHARACTERS:
        raise ReplayError("Updated Motion document exceeds the XML size limit")
    return result


def _sequence_node_ids(node: ET.Element, mapping: dict[int, int]) -> None:
    for key, value in tuple(node.attrib.items()):
        if key in {"id", "style"} and value.isdecimal() and int(value) in mapping:
            node.set(key, str(mapping[int(value)]))


def _sequence_tree_ids(node: ET.Element, mapping: dict[int, int]) -> None:
    for child in node.iter():
        _sequence_node_ids(child, mapping)


def _sentence_nodes(
    template: str,
    placement: CaptionPlacement,
    *,
    expected_text: str,
    mapping: dict[int, int],
    rate: int,
    smooth: bool,
) -> list[ET.Element]:
    body = replace_caption_text(template, expected_text=expected_text, text=placement.text)
    positioned = position_caption_template(body, x=placement.x, y=placement.y)
    animated = _parse(
        animate_caption_template(
            positioned, samples=placement.samples, fade_in=smooth, smooth=smooth
        )
    )
    nodes = _caption_objects(animated)
    for node in nodes:
        _sequence_tree_ids(node, mapping)
        timing = _one(node, "timing")
        timing.set("in", f"{placement.start_frame} {rate} 1 0")
        timing.set("out", f"{placement.end_frame - 1} {rate} 1 0")
        timing.set("offset", f"{placement.start_frame} {rate} 1 0")
    return nodes


def _scale_geometry_control(style: ET.Element, path: str, factor: float) -> None:
    control = _one(style, path)
    try:
        value = float(_attribute(control, "value"))
    except ValueError as error:
        raise ReplayError("Invalid Motion caption geometry value") from error
    if list(control) or not math.isfinite(value) or not 0 <= value <= MAX_POSITION_UNITS:
        raise ReplayError("Motion caption geometry must be an unanimated bounded scalar")
    scaled = value * factor
    if scaled > MAX_POSITION_UNITS:
        raise ReplayError("Scaled Motion caption geometry exceeds the size limit")
    control.set("value", str(scaled))


def _scale_sequence_geometry(root: ET.Element, *, height: int) -> None:
    """Scale the observed flat title's glyph/outline/shadow units, not its origins."""
    try:
        reference = int(_one(root, "scene/sceneSettings/height").text or "")
    except ValueError as error:
        raise ReplayError("Invalid Motion reference canvas height") from error
    if reference <= 0:
        raise ReplayError("Invalid Motion reference canvas height")
    factor = height / reference
    for node in _caption_objects(root):
        style = _one(node, "style")
        _scale_geometry_control(style, "parameter[@id='3']", factor)
        _scale_geometry_control(style, "parameter[@id='30']/parameter[@id='36']", factor)
        if node.get("name") == "Caption Outer Rim":
            _scale_geometry_control(style, "parameter[@id='21']/parameter[@id='27']", factor)
            _scale_geometry_control(
                style, "parameter[@id='21']/parameter[@id='75']/parameter[@id='1']", factor
            )
            _scale_geometry_control(
                style, "parameter[@id='21']/parameter[@id='75']/parameter[@id='2']", factor
            )


def build_caption_sequence(
    template_xml: str,
    placements: tuple[CaptionPlacement, ...],
    *,
    expected_text: str,
    frames: int,
    width: int,
    height: int,
    rate: int = 30,
    match_fcp_scale: bool = False,
    style_height: int | None = None,
    smooth: bool = True,
    max_sentences: int = MAX_SEQUENCE_SENTENCES,
    max_characters: int = MAX_XML_CHARACTERS,
) -> str:
    """Assemble a bounded caption-only Motion candidate from the verified flat layout.

    Every sentence keeps two synchronized face/rim objects. ``smooth=False``
    disables Fade In and uses linear keys; supply ``chain_reveal_samples`` pop-in
    calibration, not smooth ramp samples, for frame-exact rhythmic changes.
    Existing styles remain unchanged by default. Opt-in ``match_fcp_scale``
    scales font size, both outlines and rim shadow geometry by output height /
    template height, matching FCP's observed title-height normalization. An
    explicit ``style_height`` keeps typography bound to the measured plan canvas
    when the output canvas is normalized. Origins remain in output units; colours,
    opacity and recognition clocks are untouched.
    This bounded scalar calibration does not certify arbitrary layouts or pixels.
    This is not a registered FCP title or a flattened substitute for its editable
    timeline. Geometry, colour and real rendering need independent verification.
    """
    if not isinstance(smooth, bool):
        raise ReplayError("Motion sequence smoothing must be a boolean")
    if not isinstance(match_fcp_scale, bool):
        raise ReplayError("Motion FCP scale matching must be a boolean")
    if style_height is not None and (
        type(style_height) is not int or not 1 <= style_height <= MAX_SEQUENCE_PIXELS
    ):
        raise ReplayError("Invalid Motion style reference height")
    if (
        type(rate) is not int
        or not 1 <= rate <= 60
        or type(frames) is not int
        or not 1 <= frames <= rate * MAX_SEQUENCE_SECONDS
        or type(width) is not int
        or type(height) is not int
        or min(width, height) <= 0
        or width * height > MAX_SEQUENCE_PIXELS
        or not 1 <= len(placements) <= max_sentences
    ):
        raise ReplayError("Invalid bounded Motion sequence dimensions or clock")
    root = _parse(template_xml)
    nodes = _caption_objects(root)
    settings = _one(root, "scene/sceneSettings")
    _project_duration(root)
    if match_fcp_scale:
        _scale_sequence_geometry(root, height=height if style_height is None else style_height)
    for name, value in (
        ("width", width),
        ("height", height),
        ("duration", frames),
        ("frameRate", rate),
        ("NTSC", 0),
    ):
        _one(settings, name).text = str(value)
    template = ET.tostring(root, encoding="unicode")
    generated: set[int] = set()
    for node in root.iter():
        generated.update(_generated_ids(node))
    if not generated:
        raise ReplayError("Missing generated Motion object IDs")
    identifiers = sorted(generated)
    maximum = max(generated)
    if maximum + (len(placements) - 1) * len(identifiers) >= GENERATED_ID_MAX:
        raise ReplayError("Motion object ID range exhausted")
    layer = _one(root, "scene/layer")
    insertion = list(layer).index(nodes[0])
    for node in nodes:
        layer.remove(node)
    parent_timing = _one(layer, "timing")
    parent_timing.set("in", "0 1 1 0")
    parent_timing.set("offset", "0 1 1 0")
    parent_timing.set("out", f"{frames - 1} {rate} 1 0")
    previous_end = 0
    for index, placement in enumerate(placements):
        # A constant pop-in needs two keys for Motion, even for a one-frame title.
        # Its identical exclusive-end key cannot reveal a late or hidden prefix.
        constant_hold = (
            not smooth
            and len(placement.samples) == 2
            and type(placement.start_frame) is int
            and type(placement.end_frame) is int
            and placement.samples[0][0] == 0
            and placement.samples[-1][0]
            == Fraction(placement.end_frame - placement.start_frame, rate)
            and placement.samples[0][1] == placement.samples[-1][1]
        )
        if (
            type(placement.start_frame) is not int
            or type(placement.end_frame) is not int
            or not previous_end <= placement.start_frame < placement.end_frame <= frames
            or not placement.samples
            or (
                not constant_hold
                and placement.samples[-1][0]
                > Fraction(placement.end_frame - placement.start_frame - 1, rate)
            )
        ):
            raise ReplayError("Invalid or overlapping Motion sentence support")
        mapping = {
            identifier: identifier
            if index == 0
            else maximum + (index - 1) * len(identifiers) + offset + 1
            for offset, identifier in enumerate(identifiers)
        }
        placed = _sentence_nodes(
            template,
            placement,
            expected_text=expected_text,
            mapping=mapping,
            rate=rate,
            smooth=smooth,
        )
        layer[insertion:insertion] = placed
        insertion += len(placed)
        previous_end = placement.end_frame
    result = ET.tostring(root, encoding="unicode")
    if len(result) > max_characters:
        raise ReplayError("Motion caption sequence exceeds the XML size limit")
    return result


def animate_caption_template(
    template_xml: str,
    *,
    samples: tuple[tuple[Fraction, Fraction], ...],
    fade_in: bool = False,
    smooth: bool = False,
) -> str:
    """Return a new experimental Motion document with synchronized Type On curves.

    Reuse the explicit title-relative (seconds, End fraction) samples used for
    FCPXML; do not pass FCP's one-hour generator clock. No timing is inferred from
    text length, and text, layout, appearance, publication and duration stay intact.
    Only this module's unanimated two-object ozml 5.13 candidate is supported.

    ``smooth`` uses zero-slope cubic handles at the supplied keys without adding
    samples or changing their times. Combine it with Fade In for gradual appearance.
    Fade In changes the fraction-to-opacity mapping: pop-in samples do NOT become
    word-aligned fades just by enabling it. Callers must independently verify their
    chosen samples, frame clocks and output. This function neither installs titles
    nor renders, certifies colours, creates alpha media, or completes delivery.
    """
    if not isinstance(fade_in, bool):
        raise ReplayError("Motion Fade In must be a boolean")
    if not isinstance(smooth, bool):
        raise ReplayError("Motion smoothing must be a boolean")
    if not 2 <= len(samples) <= 10_000:
        raise ReplayError("Motion reveal requires between 2 and 10000 samples")
    if any(not isinstance(item, Fraction) for sample in samples for item in sample):
        raise ReplayError("Motion reveal times and values must be exact fractions")
    times = [time for time, _ in samples]
    if times[0] != 0 or any(end <= start for start, end in pairwise(times)):
        raise ReplayError("Motion reveal must start at zero and have strictly increasing times")
    if any(time.numerator > 2**63 - 1 or time.denominator > 2**31 - 1 for time in times):
        raise ReplayError("Motion reveal clock exceeds CMTime limits")
    if any(not 0 <= value <= 1 for _, value in samples):
        raise ReplayError("Motion reveal values must be between zero and one")
    root = _parse(template_xml)
    nodes = _caption_objects(root)
    if times[-1] > _project_duration(root):
        raise ReplayError("Motion reveal extends beyond the project duration")
    curve = _reveal_curve(samples, smooth=smooth)
    for node in nodes:
        group = _parameter(_parameter(node, "2"), "319")
        end, fade = _parameter(group, "321"), _parameter(group, "322")
        if list(end) or list(fade) or end.get("value") != "1" or fade.get("value") != "0":
            raise ReplayError("Motion caption controls are already animated or modified")
        del end.attrib["value"]
        end.append(copy.deepcopy(curve))
        fade.set("value", str(int(fade_in)))
    result = ET.tostring(root, encoding="unicode")
    if len(result) > MAX_XML_CHARACTERS:
        raise ReplayError("Animated Motion document exceeds the XML size limit")
    return result


def add_nonfade_entrance(template_xml: str, *, rate: int) -> str:
    """Add a visible, uniform scale entrance without changing opacity.

    Caption pairs start at the shared 78% scale and reach 100% over 0.2 seconds.
    Position, opacity, text, Type On clocks and style controls remain unchanged.
    Native rendering still needs independent verification.
    """
    if type(rate) is not int or not 1 <= rate <= 60:
        raise ReplayError("Motion entrance requires a supported integer frame rate")
    root = _parse(template_xml)
    nodes = _one(root, "scene/layer").findall("scenenode[@factoryID='16']")
    if (
        not nodes
        or len(nodes) % 2
        or len(nodes) > 2 * MAX_SEQUENCE_SENTENCES
        or any(
            (face.get("name"), rim.get("name")) != ("Caption Face", "Caption Outer Rim")
            for face, rim in zip(nodes[::2], nodes[1::2], strict=True)
        )
    ):
        raise ReplayError("Motion entrance requires synchronized generated caption pairs")
    curves: list[ET.Element] = []
    for _axis in ("X", "Y"):
        curve = _reveal_curve(
            (
                (
                    Fraction(0),
                    Fraction(str(caption_style_defaults.MOTION_ENTRANCE_START_SCALE)),
                ),
                (
                    Fraction(
                        caption_style_defaults.MOTION_ENTRANCE_DURATION_NUMERATOR,
                        caption_style_defaults.MOTION_ENTRANCE_DURATION_DENOMINATOR,
                    ),
                    Fraction(1),
                ),
            ),
            smooth=False,
        )
        curve.set("default", "1")
        curve.set("value", "1")
        curve.remove(_one(curve, "min"))
        curve.remove(_one(curve, "max"))
        curves.append(curve)
    for node in nodes:
        properties = _parameter(node, "1")
        transform = _one(properties, "parameter[@id='100']")
        if transform.find("parameter[@id='105']") is not None:
            raise ReplayError("Motion caption entrance is already animated")
        scale = ET.SubElement(
            transform,
            "parameter",
            name="Scale",
            id="105",
            flags="8589938704",
        )
        ET.SubElement(scale, "foldFlags").text = "11"
        for identifier, axis, curve in zip(("1", "2"), ("X", "Y"), curves, strict=True):
            component = ET.SubElement(
                scale,
                "parameter",
                name=axis,
                id=identifier,
                flags="8589934608",
            )
            component.append(copy.deepcopy(curve))
    result = ET.tostring(root, encoding="unicode")
    if len(result) > MAX_XML_CHARACTERS:
        raise ReplayError("Motion caption entrance exceeds the XML size limit")
    return result


if __name__ == "__main__":
    main()
