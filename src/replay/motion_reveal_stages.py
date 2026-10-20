"""Prepare Motion caption stages that reveal text with speech instead of re-entering per stage.

Paging emits one stage per recognized onset, each repeating the earlier text and adding the new
words. ``replay.motion_stages`` draws every stage as its own fully visible title with the zoom
entrance, so a growing caption zooms again at each onset. Here consecutive stages that form a
strict prefix-growth chain (contiguous frames, same origin, the earlier text a strict prefix of
the later one, line breaks included) become one title whose Type On End reveals exactly each
stage's characters on that stage's frames (``fcp_project``'s pop-in rule: ``(2k - 1) / 2N`` of the
final text's graphemes). A reset, a gap or rebalanced line breaks start a new title.

Whether a title gets an entrance animation is the shared style setting
``caption_style_defaults.MOTION_ENTRANCE``; it is off.
"""

from __future__ import annotations

import argparse
import json
from collections.abc import Sequence
from dataclasses import dataclass
from fractions import Fraction
from pathlib import Path

import regex

from replay.caption_style_defaults import MOTION_ENTRANCE
from replay.core import ReplayError
from replay.motion_caption import (
    MAX_SEQUENCE_SENTENCES,
    CaptionPlacement,
    add_nonfade_entrance,
    build_caption_sequence,
)
from replay.motion_stages import _integer, _placement, _reference_canvas, _text
from replay.recordings import record

GRAPHEME = regex.compile(r"\X")
COORDINATE_SPACES = frozenset({"motion-origin", "canvas-top-left", "canvas-bottom-left"})


@dataclass(frozen=True, slots=True)
class Stage:
    text: str
    start: int
    end: int
    x: float
    y: float


def stages_of(entries: Sequence[object]) -> list[Stage]:
    result = []
    for value in entries:
        item = record(value)
        if set(item) != {"text", "startFrame", "endFrame", "x", "y"}:
            raise ReplayError("Invalid Motion stage fields")
        x, y = item["x"], item["y"]
        if isinstance(x, bool) or isinstance(y, bool) or not isinstance(x, (int, float)):
            raise ReplayError("Motion stage origins must be numbers")
        if not isinstance(y, (int, float)):
            raise ReplayError("Motion stage origins must be numbers")
        result.append(
            Stage(_text(item["text"]), _integer(item["startFrame"]), _integer(item["endFrame"]),
                  float(x), float(y))
        )  # fmt: skip
    return result


def extends(previous: Stage, following: Stage) -> bool:
    """Whether ``following`` continues ``previous`` in one growing title."""
    return (
        following.start == previous.end
        and (following.x, following.y) == (previous.x, previous.y)
        and len(following.text) > len(previous.text)
        and following.text.startswith(previous.text)
    )


def chains(stages: Sequence[Stage]) -> list[list[Stage]]:
    grouped: list[list[Stage]] = []
    for stage in stages:
        if grouped and extends(grouped[-1][-1], stage):
            grouped[-1].append(stage)
        else:
            grouped.append([stage])
    return grouped


def visible(text: str, ends: Sequence[int]) -> Fraction:
    """Interior Type On End value that shows exactly ``text``'s graphemes of the final text."""
    if len(text) not in ends:
        raise ReplayError("A caption stage splits a composed character of its final text")
    return Fraction(2 * (list(ends).index(len(text)) + 1) - 1, 2 * len(ends))


def reveal(chain: Sequence[Stage], rate: int) -> tuple[tuple[Fraction, Fraction], ...]:
    """Linear Type On End keys holding each stage's prefix on exactly its own frames."""
    final = chain[-1].text
    ends = [match.end() for match in GRAPHEME.finditer(final)]
    first = chain[0]
    samples = [(Fraction(0), visible(first.text, ends))]
    for stage in chain[1:]:
        time = Fraction(stage.start - first.start, rate)
        hold = (time - Fraction(1, rate), samples[-1][1])
        if hold != samples[-1]:
            samples.append(hold)
        samples.append((time, visible(stage.text, ends)))
    if len(samples) == 1:  # one stage: Motion's constant pop-in needs its end key too
        samples.append((Fraction(chain[-1].end - first.start, rate), samples[0][1]))
    return tuple(samples)


def prepare(template: str, plan: object) -> str:
    """One revealing title per growth chain, on the reference canvas, with planned clocks."""
    data = record(plan)
    required = {"frames", "width", "height", "rate", "expectedText", "placements"}
    if set(data) not in (required, required | {"coordinateSpace"}):
        raise ReplayError("Invalid Motion stage plan fields")
    rate = _integer(data["rate"])
    if not 1 <= rate <= 60:
        raise ReplayError("Invalid Motion stage frame rate")
    plan_width, plan_height = _integer(data["width"]), _integer(data["height"])
    width, height = _reference_canvas(plan_width, plan_height)
    space = data.get("coordinateSpace", "motion-origin")
    if not isinstance(space, str) or space not in COORDINATE_SPACES:
        raise ReplayError("Unsupported Motion stage coordinate space")
    entries = data["placements"]
    if not isinstance(entries, list) or not entries:
        raise ReplayError("Motion placements must be a nonempty array")
    placements: list[CaptionPlacement] = []
    for chain in chains(stages_of(entries)):
        first, last = chain[0], chain[-1]
        anchor = _placement(
            {"text": last.text, "startFrame": first.start, "endFrame": last.end,
             "x": first.x, "y": first.y},
            rate, coordinate_space=space, source_width=plan_width, source_height=plan_height,
            output_width=width, output_height=height,
        )  # fmt: skip
        placements.append(
            CaptionPlacement(anchor.text, anchor.start_frame, anchor.end_frame,
                             reveal(chain, rate), anchor.x, anchor.y)
        )  # fmt: skip
    if len(placements) > MAX_SEQUENCE_SENTENCES:
        raise ReplayError("Too many Motion titles in one part")
    sequence = build_caption_sequence(
        template, tuple(placements), expected_text=_text(data["expectedText"]),
        frames=_integer(data["frames"]), width=width, height=height, rate=rate,
        match_fcp_scale=True, style_height=plan_height, smooth=False,
    )  # fmt: skip
    return add_nonfade_entrance(sequence, rate=rate) if MOTION_ENTRANCE else sequence


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("template", type=Path)
    parser.add_argument("plan", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args(argv)
    if args.output.exists():
        raise ReplayError("Refusing to overwrite the Motion document")
    text = prepare(
        args.template.read_text(encoding="utf-8"),
        json.loads(args.plan.read_text(encoding="utf-8")),
    )
    args.output.write_text(text, encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
