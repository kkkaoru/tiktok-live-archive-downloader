"""Measured output-clock caption candidates with explicit source-support mapping."""

from collections.abc import Callable
from dataclasses import dataclass, replace
from fractions import Fraction

from replay.caption_display import unretained_frame_prefixes
from replay.caption_linguistics import wrap_measured_lines
from replay.caption_timing import (
    CaptionCue,
    CaptionPolicy,
    SpokenToken,
    display_time,
    make_captions,
)
from replay.foreground_geometry import _check_box, _finite, place_envelope
from replay.vertical_captions import Box

MAX_TOKENS = 4096
MAX_WINDOW_SECONDS = 120
MAX_MEASUREMENTS = 65_536


@dataclass(frozen=True, slots=True)
class PositionedCue:
    cue: CaptionCue
    x: float
    y: float


def measured_plan(
    tokens: tuple[SpokenToken, ...],
    *,
    policy: CaptionPolicy,
    region: Box,
    center_x: float,
    clearance: float,
    output_support_top: Callable[[Fraction, Fraction], float],
    envelope: Callable[[str], Box],
    prepare_lines: Callable[[tuple[str, ...]], None] | None = None,
) -> tuple[PositionedCue, ...]:
    """Plan one bounded output window; do not approve a render or its semantics.

    Tokens and policy.frame_rate belong to the VERIFIED OUTPUT clock. The support
    callback must map a half-open output interval through verified cuts to all
    required source frames, rejecting unknowns; never assume source VFR is CFR.
    Envelope must supply independently bound, complete Motion bounds relative to
    its calibrated origin, including every animation state. Cache measurements in
    that provider. This function does not invent a calibration or source mapping.

    Use a conservative common available height for this window, allowing existing
    POS-aware timing to switch earlier when up to three lines do not fit. Then place each
    displayed cue at its own observed support ceiling, bottom-anchoring only its
    complete envelope. Future lines must not reserve height before they appear.
    The provider's calibrated font/baseline envelope (not glyph alpha bounds)
    preserves same-line baseline stability; its observed line advance moves older
    lines upward when another line appears. Resets need no retained Y.
    No font shrinking, token retiming, geometry fallback,
    discarded subframe word, or guessing of missing source evidence is allowed.
    """
    if (
        not 1 <= len(tokens) <= MAX_TOKENS
        or policy.frame_rate is None
        or not policy.linguistic_boundaries
        or policy.measured_layout is not None
    ):
        raise ValueError("Measured planning needs bounded tokens and an output-clock POS policy")
    start = display_time(min(token.start for token in tokens))
    end = display_time(max(token.end for token in tokens) + policy.hold_seconds)
    if not 0 < end - start <= MAX_WINDOW_SECONDS:
        raise ValueError("Measured caption window exceeds its duration budget")
    common_top = output_support_top(start, end)
    _check_box(region, relative=False)
    if (
        not all(_finite(value) for value in (common_top, center_x, clearance))
        or common_top < 0
        or clearance < 0
    ):
        raise ValueError("Invalid measured planning region constraints")
    width = 2 * min(center_x - region.left, region.right - center_x)
    height = min(region.bottom, common_top - clearance) - region.top
    cache: dict[str, Box] = {}

    def measured(text: str) -> Box:
        if text not in cache:
            if len(cache) >= MAX_MEASUREMENTS:
                raise ValueError("Caption envelope measurement budget exceeded")
            box = envelope(text)
            _check_box(box, relative=True)
            cache[text] = box
        return cache[text]

    def size(text: str) -> tuple[float, float]:
        box = measured(text)
        return 2 * max(abs(box.left), abs(box.right)), box.bottom - box.top

    def wrap(text: str) -> str:
        return wrap_measured_lines(
            text,
            measure=size,
            width=width,
            height=height,
            prepare_lines=prepare_lines,
            maximum_lines=3,
        )

    selected = replace(policy, measured_layout=wrap, maximum_lines=3)
    cues = make_captions(tokens, policy=selected)
    if not cues or unretained_frame_prefixes(tokens, policy=selected):
        raise ValueError("Measured captions contain absent or unretained output-frame prefixes")
    positioned: list[PositionedCue] = []
    for cue in cues:
        top = output_support_top(display_time(cue.start), display_time(cue.end))
        if top < common_top:
            raise ValueError("Cue support contradicts the enclosing window minimum")
        x, y = place_envelope(
            measured(cue.text),
            region=region,
            center_x=center_x,
            foreground_top=top,
            clearance=clearance,
        )
        positioned.append(PositionedCue(cue, x, y))
    return tuple(positioned)
