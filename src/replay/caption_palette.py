"""A recording's own caption gradient colours, as an explicit editing intent.

``edits/<recording>/caption-colors.json`` (``{"version": 1, "gradientTop": "#RRGGBB",
"gradientBottom": "#RRGGBB"}``) replaces only the two colours of the default caption style; its
gradient structure (stop locations, white band, black rims, shadow, font) is kept. As in the
default, the top colour should be the base mixed toward white, so the top must be brighter than
the bottom by ``MINIMUM_LIFT`` (sRGB luma), which leaves the rendered gradient audit its margin.

The palette reaches the renders through the recording's ``captions.json``: its ``motionFinish``
gradient controls are replaced (the default's controls are its sRGB values, 0-1) and the
palette is kept beside them, so every finished Motion part is restamped with the colour intent it
actually renders (``motion_finish`` stamps the default's). Without the file nothing changes.
"""

import json
import re
from pathlib import Path

from replay.caption_contrast import WHITE, contrast_ratio
from replay.core import ReplayError
from replay.motion_caption import COLOUR_INTENT_MARKER

INTENT = "caption-colors.json"
KEYS = frozenset({"version", "gradientTop", "gradientBottom"})
HEX = re.compile(r"#[0-9A-F]{6}")
LUMA = (0.2126, 0.7152, 0.0722)
MINIMUM_LIFT = 24.0
POLICY = "recording-palette"
STAMP = re.compile(r"<!--" + re.escape(COLOUR_INTENT_MARKER) + r"(.*?)-->", re.DOTALL)

Palette = dict[str, str]


def srgb(colour: str) -> tuple[float, float, float]:
    return (int(colour[1:3], 16) / 255, int(colour[3:5], 16) / 255, int(colour[5:7], 16) / 255)


def luma(colour: str) -> float:
    return sum(weight * value * 255 for weight, value in zip(LUMA, srgb(colour), strict=True))


def validated(value: object) -> Palette:
    """The two gradient colours of a palette, refused unless strictly well formed."""
    if not isinstance(value, dict) or set(value) - {"version"} != KEYS - {"version"}:
        raise ReplayError("A caption palette has exactly gradientTop and gradientBottom")
    colours = {key: value[key] for key in ("gradientTop", "gradientBottom")}
    if not all(isinstance(item, str) and HEX.fullmatch(item) for item in colours.values()):
        raise ReplayError("Caption palette colours are upper-case #RRGGBB")
    palette = {key: str(item) for key, item in colours.items()}
    if luma(palette["gradientTop"]) - luma(palette["gradientBottom"]) < MINIMUM_LIFT:
        raise ReplayError("The palette's top colour must be brighter than its bottom colour")
    return palette


def intent(edit_root: Path) -> Palette | None:
    """The recording's caption palette, or None without ``caption-colors.json``."""
    path = edit_root / INTENT
    if not path.exists():
        return None
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict) or set(data) != KEYS or data["version"] != 1:
        raise ReplayError(f"{INTENT} must be {{version: 1, gradientTop, gradientBottom}}")
    return validated(data)


def finish_arguments(finish: list[str], palette: Palette) -> list[str]:
    """``motion_finish`` arguments with the palette's gradient controls."""
    result = list(finish)
    for flag, key in (("--top", "gradientTop"), ("--bottom", "gradientBottom")):
        if result.count(flag) != 1 or result.index(flag) + 3 >= len(result):
            raise ReplayError(f"motionFinish must hold {flag} with three values once")
        at = result.index(flag)
        result[at + 1 : at + 4] = [repr(value) for value in srgb(palette[key])]
    return result


def captions_settings(captions: dict[str, object], palette: Palette | None) -> dict[str, object]:
    """``captions.json`` content: the defaults' section, recoloured when there is a palette."""
    if palette is None:
        return {"version": 1, **captions}
    finish = captions.get("motionFinish")
    if not isinstance(finish, list) or not all(isinstance(item, str) for item in finish):
        raise ReplayError("motionFinish must be a list of arguments")
    return {
        "version": 1,
        **captions,
        "motionFinish": finish_arguments([str(item) for item in finish], palette),
        "palette": palette,
    }


def restamp(text: str, palette: Palette) -> str:
    """A finished part's colour intent rewritten to the palette it renders."""
    found = STAMP.findall(text)
    if len(found) != 1:
        raise ReplayError("A finished caption part must hold one colour intent")
    payload = json.loads(found[0])
    top, bottom = srgb(palette["gradientTop"]), srgb(palette["gradientBottom"])
    payload["intent"] = {**payload["intent"], "top": list(top), "bottom": list(bottom)}
    payload.update(
        intentPolicy=POLICY,
        palette=palette,
        selectedWhiteContrast={
            "top": contrast_ratio(WHITE, top),
            "bottom": contrast_ratio(WHITE, bottom),
        },
    )
    stamp = f"<!--{COLOUR_INTENT_MARKER}{json.dumps(payload, sort_keys=True)}-->"
    return STAMP.sub(lambda _match: stamp, text)
