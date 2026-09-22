"""Display-only clipping; never alter recognition clocks or cross cut ownership."""

from math import isfinite

from .caption_projection import CutSpan
from .caption_timing import PUNCTUATION, CaptionCue, CaptionPolicy, SpokenToken, make_captions


def native_interval(
    start: float,
    end: float,
    *,
    first_frame: int,
    end_frame: int,
    frame_end_tick: bool = True,
) -> tuple[float, float] | None:
    """Localize display clocks without creating floating-point frame-start holes.

    Titles need a one-tick endpoint correction for Core Animation. Core Image
    masks already implement strict half-open membership: use frame_end_tick=False.
    Only mathematically frame-aligned endpoints are reconstructed from integers;
    native recognition clocks and genuinely subframe display bounds stay intact.
    """
    if not all(isfinite(value) for value in (start, end)) or not 0 <= start < end:
        raise ValueError("Invalid display interval")
    if not 0 <= first_frame < end_frame:
        raise ValueError("Invalid render part")
    left, right = max(start, first_frame / 30), min(end, end_frame / 30)
    if right < left - 1e-9:
        return None
    if right - left < 1e-9:
        if abs(left * 30 - round(left * 30)) > 1e-8:
            raise ValueError("Unexpected unrepresentable display interval")
        return None
    local_start, local_end = left - first_frame / 30, right - first_frame / 30
    if abs(left * 30 - round(left * 30)) <= 1e-8:
        local_start = (round(left * 30) - first_frame) / 30
    if abs(right * 30 - round(right * 30)) <= 1e-8:
        local_end = (round(right * 30) - first_frame) / 30
        if frame_end_tick:
            local_end -= 1 / 60000
    if local_start < 0 or local_end - local_start < 1 / 60000:
        raise ValueError("Display interval cannot meet the native tick budget")
    return local_start, local_end


def owned_captions(
    tokens: tuple[SpokenToken, ...],
    cuts: tuple[CutSpan, ...],
    *,
    accept_coarse_timing: bool = False,
) -> tuple[CaptionCue, ...]:
    """Apply established focus policy and clip holds to explicitly owned cuts."""
    limits: dict[float, float] = {}
    for token in tokens:
        if all(character in PUNCTUATION for character in token.text):
            continue
        owners = token.utterance.rpartition(":quiet-cuts-")
        first, separator, last = owners[2].partition("-")
        if not owners[1] or not separator or not first.isdigit() or not last.isdigit():
            raise ValueError("Missing retained-cut ownership")
        first_index, last_index = int(first), int(last)
        if not 0 <= first_index <= last_index < len(cuts):
            raise ValueError("Invalid retained-cut ownership")
        last_cut = cuts[last_index]
        limit = (last_cut.output_start_frame + last_cut.duration_frames) / 30
        if (
            token.start < cuts[first_index].output_start_frame / 30 - 1e-8
            or token.end > limit + 1e-8
        ):
            raise ValueError("Word exceeds retained-cut ownership")
        limits[token.start] = limit
    result: list[CaptionCue] = []
    # Accepted display violations let a native token keep one internal
    # punctuation break and a coarse clock; the focused width still applies.
    policy = CaptionPolicy(maximum_characters=18, maximum_lines=1)
    if accept_coarse_timing:
        policy = CaptionPolicy(maximum_characters=18, maximum_lines=2, enforce_focused_timing=False)
    for cue in make_captions(tokens, policy=policy):
        end = min(cue.end, limits[cue.start])
        if end <= cue.start:
            raise ValueError("Owned caption has no visible interval")
        result.append(CaptionCue(cue.text, cue.start, end))
    return tuple(result)
