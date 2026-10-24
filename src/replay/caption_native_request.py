"""Describe a Motion caption part as a native caption render request.

The request holds what ``replay.motion_reveal_stages`` and ``replay.motion_finish`` put into a
finished part: each growth chain's final text, frames, position and per-grapheme onset frames
(its ``.reveal.json`` record), and style 20 converted to output pixels with the same calibrated
scales. The native ``caption-alpha-render`` draws it as a ProRes 4444 alpha movie, so
``replay.caption_parity`` can compare it with the Motion render of the same part.

Besides the style constants, these values describe how Motion draws them. Each was measured on
Motion's own renders of synthetic captions (Compressor, 2026-10-24; the native renderer's
documentation lists the rest of the model), not chosen:

- the face is ``MOTION_FACE_GROW`` pixels bolder than CoreText's outline and the visible
  keyline ``MOTION_KEYLINE_SCALE`` of its nominal width (least squares over the set's frames;
  the white band and the rim keep their nominal widths, which the edge profile confirms);
- the rim object's shadow is cast by the glyphs alone, with a Gaussian of standard deviation
  ``MOTION_SHADOW_SIGMA_PER_BLUR`` times Motion's blur amount, its nominal offset and an
  effective opacity of ``MOTION_SHADOW_OPACITY`` (its control is ``SHADOW_OPACITY``);
- the output holds linear light encoded with a pure ``MOTION_OUTPUT_GAMMA`` power law (sRGB
  ``#2FB0EE`` renders as 41, 166, 235).

    python -m replay.caption_native_request PLAN.json REVEAL.json MOVIE.mov REQUEST.json \
        --font-size 82.9558 --stroke-scale 0.47134 --shadow-scale 0.94268
"""

from __future__ import annotations

import argparse
import json
import math
from collections.abc import Sequence
from pathlib import Path
from typing import Final

from replay import caption_style_defaults as defaults
from replay.caption_palette import Palette, validated
from replay.core import ReplayError
from replay.motion_reveal_stages import COORDINATE_SPACES, chains, stages_of
from replay.motion_stages import _integer, _placement, _reference_canvas
from replay.recordings import record

MOTION_SHADOW_SIGMA_PER_BLUR: Final[float] = 1 / math.sqrt(2)
MOTION_SHADOW_OPACITY: Final[float] = 0.7
MOTION_OUTPUT_GAMMA: Final[float] = 1.961
MOTION_FACE_GROW: Final[float] = 0.3
MOTION_KEYLINE_SCALE: Final[float] = 0.7215  # a fitted 1.36 px of the nominal 1.885 px


def style(
    *, font_size: float, stroke_scale: float, shadow_scale: float, palette: Palette | None = None
) -> dict[str, object]:
    """Style 20 in output pixels, scaled like ``motion_finish.selected_finish_style``; a
    recording ``palette`` replaces only the gradient's two colours, as in ``caption_parts``."""
    if not all(math.isfinite(value) and value > 0 for value in (font_size, stroke_scale,
                                                                 shadow_scale)):  # fmt: skip
        raise ReplayError("Caption font size and scales must be finite and positive")
    return {
        "fontName": defaults.FONT_POSTSCRIPT_NAME,
        "fontSize": font_size,
        "faceGrow": MOTION_FACE_GROW,
        "keyline": defaults.INNER_BLACK_STROKE_WIDTH * stroke_scale * MOTION_KEYLINE_SCALE,
        "whiteBorder": defaults.WHITE_STROKE_WIDTH * stroke_scale,
        "rim": defaults.OUTER_BLACK_STROKE_WIDTH * stroke_scale,
        "shadowOpacity": MOTION_SHADOW_OPACITY,
        "shadowOffsetY": defaults.SHADOW_OFFSET_Y * shadow_scale,
        "shadowSigma": defaults.SHADOW_BLUR_STD_DEVIATION * shadow_scale
        * MOTION_SHADOW_SIGMA_PER_BLUR,
        "gradientTop": defaults.GRADIENT_TOP_HEX if palette is None else palette["gradientTop"],
        "gradientBottom": defaults.GRADIENT_BOTTOM_HEX
        if palette is None
        else palette["gradientBottom"],
        "topLocation": defaults.MOTION_GRADIENT_TOP_LOCATION,
        "bottomLocation": defaults.MOTION_GRADIENT_BOTTOM_LOCATION,
        "outputGamma": MOTION_OUTPUT_GAMMA,
    }  # fmt: skip


def titles(plan: object, reveal: object) -> list[dict[str, object]]:
    """Each growth chain's title with its Motion position and recorded onset frames."""
    data = record(plan)
    rate = _integer(data["rate"])
    width, height = _integer(data["width"]), _integer(data["height"])
    if _reference_canvas(width, height) != (width, height):
        # shortcut: Motion rescales smaller plan canvases; support them when a plan needs it.
        raise ReplayError("Only plans on the Motion reference canvas are supported")
    space = data.get("coordinateSpace", "motion-origin")
    if not isinstance(space, str) or space not in COORDINATE_SPACES:
        raise ReplayError("Unsupported Motion stage coordinate space")
    entries = data["placements"]
    if not isinstance(entries, list):
        raise ReplayError("Motion placements must be an array")
    recorded = record(reveal).get("titles")
    grouped = chains(stages_of(entries))
    if not isinstance(recorded, list) or len(recorded) != len(grouped):
        raise ReplayError("The reveal record does not match the plan's titles")
    result = []
    for chain, value in zip(grouped, recorded, strict=True):
        item = record(value)
        first, last = chain[0], chain[-1]
        if (item.get("text"), item.get("startFrame"), item.get("endFrame")) != (
            last.text,
            first.start,
            last.end,
        ):
            raise ReplayError("The reveal record does not match the plan's titles")
        anchor = _placement(
            {"text": last.text, "startFrame": first.start, "endFrame": last.end,
             "x": first.x, "y": first.y},
            rate, coordinate_space=space, source_width=width, source_height=height,
            output_width=width, output_height=height,
        )  # fmt: skip
        result.append({**item, "x": float(anchor.x), "y": float(anchor.y)})
    return result


def request(
    plan: object, reveal: object, output: Path, *, font_size: float, stroke_scale: float,
    shadow_scale: float, palette: Palette | None = None,
) -> dict[str, object]:  # fmt: skip
    data = record(plan)
    return {
        "output": str(output.absolute()),
        "width": _integer(data["width"]),
        "height": _integer(data["height"]),
        "rate": _integer(data["rate"]),
        "frames": _integer(data["frames"]),
        "style": style(
            font_size=font_size,
            stroke_scale=stroke_scale,
            shadow_scale=shadow_scale,
            palette=palette,
        ),  # fmt: skip
        "titles": titles(plan, reveal),
    }


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("plan", type=Path)
    parser.add_argument("reveal", type=Path)
    parser.add_argument("movie", type=Path, help="the alpha movie the native renderer writes")
    parser.add_argument("request", type=Path)
    parser.add_argument("--font-size", type=float, required=True)
    parser.add_argument("--stroke-scale", type=float, required=True)
    parser.add_argument("--shadow-scale", type=float, required=True)
    parser.add_argument("--palette", nargs=2, metavar=("TOP", "BOTTOM"))
    args = parser.parse_args(argv)
    document = request(
        json.loads(args.plan.read_text(encoding="utf-8")),
        json.loads(args.reveal.read_text(encoding="utf-8")),
        args.movie, font_size=args.font_size, stroke_scale=args.stroke_scale,
        shadow_scale=args.shadow_scale,
        palette=None if args.palette is None else validated(
            {"gradientTop": args.palette[0], "gradientBottom": args.palette[1]}
        ),
    )  # fmt: skip
    with args.request.open("x", encoding="utf-8") as stream:
        stream.write(json.dumps(document, ensure_ascii=False, indent=1) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
