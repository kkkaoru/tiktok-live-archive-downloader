"""Prepare Motion caption stages that reveal text with speech instead of re-entering per stage.

Paging emits one stage per recognized onset, each repeating the earlier text and adding the new
words. ``replay.motion_stages`` draws every stage as its own fully visible title with the zoom
entrance, so a growing caption zooms again at each onset. Here consecutive stages that form a
strict prefix-growth chain (contiguous frames, same origin, the earlier text a strict prefix of
the later one, line breaks included) become one title animated by Type On End keys: each new
grapheme appears on the frame its sound starts in the speech tokens (a token of several graphemes
is spread over its own interval; shown-only graphemes such as punctuation follow the previous one;
a stage whose text and speech disagree spreads its graphemes over its speech). A token's
``anchor`` (its recognizer start, ``replay.speech_onsets``) assigns it to the stage paging built
from it; its ``start`` times it. No grapheme appears before its stage's first frame (a title shows
nothing until its first sound) and all of them by its last, so each stage still shows exactly its
text when it ends. ``fcp_project``'s pop-in rule:
``(2k - 1) / 2N`` shows k of the final text's N graphemes; a hold key one frame earlier keeps every
step a cut, with no fade. A reset, a gap or rebalanced line breaks start a new title.

Whether a title gets an entrance animation is the shared style setting
``caption_style_defaults.MOTION_ENTRANCE``; it is off.
"""

from __future__ import annotations

import argparse
import json
import math
from collections.abc import Sequence
from dataclasses import dataclass
from fractions import Fraction
from pathlib import Path

import regex

from replay.caption_style_defaults import MOTION_ENTRANCE
from replay.core import ReplayError
from replay.motion_caption import (
    MAX_PART_SENTENCES,
    MAX_PART_XML_CHARACTERS,
    CaptionPlacement,
    add_nonfade_entrance,
    build_caption_sequence,
)
from replay.motion_stages import _integer, _placement, _reference_canvas, _text
from replay.recordings import record

GRAPHEME = regex.compile(r"\X")
COORDINATE_SPACES = frozenset({"motion-origin", "canvas-top-left", "canvas-bottom-left"})
# Seconds of float slack when matching token anchors against stage frames.
EPSILON = 1e-4


@dataclass(frozen=True, slots=True)
class Stage:
    text: str
    start: int
    end: int
    x: float
    y: float


@dataclass(frozen=True, slots=True)
class Token:
    """A recognized speech token on the output clock, in seconds."""

    text: str
    start: float
    end: float
    anchor: float


def tokens_of(document: object) -> tuple[Token, ...]:
    entries = record(document).get("tokens")
    if not isinstance(entries, list):
        raise ReplayError("The speech tokens are not a list")
    result = []
    for value in entries:
        item = record(value)
        text, start, end = item.get("text"), item.get("start"), item.get("end")
        if not isinstance(text, str) or not text:
            raise ReplayError("Invalid speech token")
        if isinstance(start, bool) or not isinstance(start, (int, float)):
            raise ReplayError("Invalid speech token")
        if isinstance(end, bool) or not isinstance(end, (int, float)) or not 0 <= start <= end:
            raise ReplayError("Invalid speech token interval")
        anchor = item.get("anchor", start)
        if isinstance(anchor, bool) or not isinstance(anchor, (int, float)) or anchor < 0:
            raise ReplayError("Invalid speech token anchor")
        result.append(Token(text, float(start), float(end), float(anchor)))
    return tuple(sorted(result, key=lambda token: token.start))


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


def visible(count: int, ends: Sequence[int]) -> Fraction:
    """Interior Type On End value that shows exactly ``count`` graphemes of the final text."""
    return Fraction(2 * count - 1, 2 * len(ends))


def graphemes(text: str, ends: Sequence[int]) -> int:
    if len(text) not in ends:
        raise ReplayError("A caption stage splits a composed character of its final text")
    return list(ends).index(len(text)) + 1


def spoken_frames(
    new: Sequence[str], stage: Stage, tokens: Sequence[Token], *, rate: int, offset: int
) -> list[int]:
    """The part frame of each new grapheme: where its sound starts in the stage's speech."""
    begin = (offset + stage.start) / rate
    # Paging starts a stage on the first frame at or after its token's anchor.
    heard = [
        token for token in tokens
        if stage.start <= math.ceil((token.anchor - EPSILON) * rate) - offset < stage.end
    ]  # fmt: skip
    spoken = "".join(token.text for token in heard)
    sounds = [
        token.start + (token.end - token.start) * index / len(token.text)
        for token in heard
        for index in range(len(token.text))
    ]
    times: list[float] = []
    cursor = 0
    for grapheme in new:
        if spoken[cursor : cursor + len(grapheme)] == grapheme:
            times.append(sounds[cursor])
            cursor += len(grapheme)
        else:  # shown but not heard: punctuation or a line break
            times.append(times[-1] if times else begin)
    if heard and cursor != len(spoken):  # text and speech disagree: spread over the speech
        first, last = heard[0].start, heard[-1].end
        times = [first + (last - first) * index / len(new) for index in range(len(new))]
    frames, low = [], stage.start
    for time in times:
        low = min(max(round(time * rate) - offset, low), stage.end - 1)
        frames.append(low)
    return frames


def onset_frames(
    chain: Sequence[Stage], rate: int, tokens: Sequence[Token] | None = None, offset: int = 0
) -> list[int]:
    """The part frame on which each grapheme of the chain's final text appears.

    Without speech tokens a stage's new graphemes all appear on its first frame.
    """
    marks = [match.group() for match in GRAPHEME.finditer(chain[-1].text)]
    ends = [match.end() for match in GRAPHEME.finditer(chain[-1].text)]
    frames: list[int] = []
    for stage in chain:
        new = marks[len(frames) : graphemes(stage.text, ends)]
        frames += (
            [stage.start] * len(new)
            if tokens is None
            else spoken_frames(new, stage, tokens, rate=rate, offset=offset)
        )
    return frames


def reveal(
    chain: Sequence[Stage], rate: int, tokens: Sequence[Token] | None = None, offset: int = 0
) -> tuple[tuple[Fraction, Fraction], ...]:
    """Type On End keys showing each grapheme from its onset frame, held between onsets."""
    ends = [match.end() for match in GRAPHEME.finditer(chain[-1].text)]
    frames = onset_frames(chain, rate, tokens, offset)
    first = chain[0].start
    # A title whose first sound comes after its first frame shows nothing until then.
    samples = [(Fraction(0), Fraction(0))] if frames[0] > first else []
    for index, frame in enumerate(frames):
        if index + 1 < len(frames) and frames[index + 1] == frame:
            continue  # graphemes sharing an onset frame appear together
        time = Fraction(frame - first, rate)
        if samples and time - Fraction(1, rate) > samples[-1][0]:
            samples.append((time - Fraction(1, rate), samples[-1][1]))
        samples.append((time, visible(index + 1, ends)))
    if len(samples) == 1:  # one key: Motion's constant pop-in needs its end key too
        samples.append((Fraction(chain[-1].end - first, rate), samples[0][1]))
    return tuple(samples)


def prepare(
    template: str,
    plan: object,
    tokens: Sequence[Token] | None = None,
    *,
    offset: int = 0,
    titles: list[dict[str, object]] | None = None,
) -> str:
    """One revealing title per growth chain, on the reference canvas, with planned clocks.

    ``tokens`` time the reveal by speech; ``offset`` is the part's first frame on their clock.
    Each title's text, frames and per-grapheme onset frames are appended to ``titles``.
    """
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
                             reveal(chain, rate, tokens, offset), anchor.x, anchor.y)
        )  # fmt: skip
        if titles is not None:
            frames = onset_frames(chain, rate, tokens, offset)
            titles.append({"text": last.text, "startFrame": first.start, "endFrame": last.end,
                           "graphemeFrames": frames})  # fmt: skip
    if len(placements) > MAX_PART_SENTENCES:
        raise ReplayError("Too many Motion titles in one part")
    sequence = build_caption_sequence(
        template, tuple(placements), expected_text=_text(data["expectedText"]),
        frames=_integer(data["frames"]), width=width, height=height, rate=rate,
        match_fcp_scale=True, style_height=plan_height, smooth=False,
        max_sentences=MAX_PART_SENTENCES, max_characters=MAX_PART_XML_CHARACTERS,
    )  # fmt: skip
    return add_nonfade_entrance(sequence, rate=rate) if MOTION_ENTRANCE else sequence


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("template", type=Path)
    parser.add_argument("plan", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--tokens", type=Path, help="speech tokens timing the reveal")
    parser.add_argument("--offset-frame", type=int, default=0)
    parser.add_argument("--record", type=Path, help="write each title's grapheme onset frames")
    args = parser.parse_args(argv)
    if args.output.exists() or (args.record is not None and args.record.exists()):
        raise ReplayError("Refusing to overwrite the Motion document")
    tokens = (
        None
        if args.tokens is None
        else tokens_of(json.loads(args.tokens.read_text(encoding="utf-8")))
    )
    titles: list[dict[str, object]] = []
    text = prepare(
        args.template.read_text(encoding="utf-8"),
        json.loads(args.plan.read_text(encoding="utf-8")),
        tokens,
        offset=args.offset_frame,
        titles=titles,
    )
    if args.record is not None:
        args.record.write_text(
            json.dumps({"offsetFrame": args.offset_frame, "titles": titles}, ensure_ascii=False)
            + "\n",
            encoding="utf-8",
        )
    args.output.write_text(text, encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
