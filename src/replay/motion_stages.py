"""Prepare bounded, fully visible Motion stages without future-string anchoring.

Each stage owns its complete text and exact frame interval. No recognition times,
line breaks, geometry or contrast are inferred or approved by this adapter.
"""

from __future__ import annotations

import argparse
import json
from fractions import Fraction
from pathlib import Path

from replay.caption_style_defaults import MOTION_REFERENCE_LONG_EDGE
from replay.core import ReplayError
from replay.motion_caption import (
    MAX_SEQUENCE_PIXELS,
    MAX_SEQUENCE_SENTENCES,
    MAX_XML_CHARACTERS,
    CaptionPlacement,
    add_nonfade_entrance,
    build_caption_sequence,
)
from replay.recordings import record


def _integer(value: object) -> int:
    if type(value) is not int:
        raise ReplayError("Motion stage clocks and dimensions must be integers")
    return value


def _text(value: object) -> str:
    if not isinstance(value, str):
        raise ReplayError("Motion stage text must be a string")
    return value


def _coordinate(value: object) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ReplayError("Motion stage origins must be numbers")
    return float(value)


def _reference_canvas(width: int, height: int) -> tuple[int, int]:
    """Upscale low-resolution stage canvases to the shared reference long edge."""
    if min(width, height) <= 0 or width * height > MAX_SEQUENCE_PIXELS:
        raise ReplayError("Invalid bounded Motion stage canvas")
    long_edge = max(width, height)
    target = max(long_edge, MOTION_REFERENCE_LONG_EDGE)
    if target == long_edge:
        return width, height
    if width >= height:
        return target, max(1, (height * target + long_edge // 2) // long_edge)
    return max(1, (width * target + long_edge // 2) // long_edge), target


def _placement(
    value: object,
    rate: int,
    *,
    coordinate_space: str,
    source_width: int,
    source_height: int,
    output_width: int,
    output_height: int,
) -> CaptionPlacement:
    item = record(value)
    if set(item) != {"text", "startFrame", "endFrame", "x", "y"}:
        raise ReplayError("Invalid Motion stage fields")
    start, end = _integer(item["startFrame"]), _integer(item["endFrame"])
    x, y = _coordinate(item["x"]), _coordinate(item["y"])
    if coordinate_space == "canvas-top-left":
        x = x * output_width / source_width - output_width / 2
        y = output_height / 2 - y * output_height / source_height
    elif coordinate_space == "canvas-bottom-left":
        x = x * output_width / source_width - output_width / 2
        y = y * output_height / source_height - output_height / 2
    elif coordinate_space != "motion-origin":
        raise ReplayError("Unsupported Motion stage coordinate space")
    return CaptionPlacement(
        _text(item["text"]),
        start,
        end,
        ((Fraction(0), Fraction(1)), (Fraction(end - start, rate), Fraction(1))),
        x,
        y,
    )


def prepare(template: str, plan: object) -> str:
    """Build visible stages on the reference canvas with planned text and clocks."""
    data = record(plan)
    required = {"frames", "width", "height", "rate", "expectedText", "placements"}
    if set(data) not in (required, required | {"coordinateSpace"}):
        raise ReplayError("Invalid Motion stage plan fields")
    rate = _integer(data["rate"])
    if not 1 <= rate <= 60:
        raise ReplayError("Invalid Motion stage frame rate")
    plan_width, plan_height = _integer(data["width"]), _integer(data["height"])
    width, height = _reference_canvas(plan_width, plan_height)
    coordinate_space = data.get("coordinateSpace", "motion-origin")
    if not isinstance(coordinate_space, str) or coordinate_space not in {
        "motion-origin",
        "canvas-top-left",
        "canvas-bottom-left",
    }:
        raise ReplayError("Unsupported Motion stage coordinate space")
    entries = data["placements"]
    if not isinstance(entries, list) or not 1 <= len(entries) <= MAX_SEQUENCE_SENTENCES:
        raise ReplayError("Motion placements must be a bounded nonempty array")
    sequence = build_caption_sequence(
        template,
        tuple(
            _placement(
                entry,
                rate,
                coordinate_space=coordinate_space,
                source_width=plan_width,
                source_height=plan_height,
                output_width=width,
                output_height=height,
            )
            for entry in entries
        ),
        expected_text=_text(data["expectedText"]),
        frames=_integer(data["frames"]),
        width=width,
        height=height,
        rate=rate,
        match_fcp_scale=True,
        style_height=plan_height,
        smooth=False,
    )
    return add_nonfade_entrance(sequence, rate=rate)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("template", type=Path)
    parser.add_argument("plan", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    for path in (args.template, args.plan):
        if path.is_symlink() or not path.is_file() or path.stat().st_size > MAX_XML_CHARACTERS:
            raise ReplayError("Motion inputs must be bounded regular files")
    result = prepare(
        args.template.read_text(encoding="utf-8"),
        json.loads(args.plan.read_text(encoding="utf-8")),
    )
    with args.output.open("x", encoding="utf-8") as handle:
        handle.write(result)


if __name__ == "__main__":
    main()
