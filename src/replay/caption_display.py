"""Display-only clipping; never alter recognition clocks or cross cut ownership."""

from math import ceil, isfinite

from .caption_projection import CutSpan
from .caption_timing import (
    PUNCTUATION,
    CaptionCue,
    CaptionPolicy,
    SpokenToken,
    _display_tokens,
    display_time,
    make_captions,
    punctuation_lines,
)

# A vertical caption line holds 18 characters; candidate captions may use two lines.
LINE_CHARACTERS = 18
CANDIDATE_CHARACTERS = 36


def unretained_frame_prefixes(
    tokens: tuple[SpokenToken, ...], *, policy: CaptionPolicy
) -> tuple[int, ...]:
    """Find unshown prefixes not retained by a later, actually displayed extension.

    Regenerate the common cues to preserve lexical ownership and exact text.
    Only contiguous, same-owner additions of the next native token may carry a
    hidden prefix. This is frame/text accounting, not fade, geometry or ASR
    approval; final smooth prefixes still require their independent two frames.
    """
    fps = policy.frame_rate
    if fps is None:
        raise ValueError("Frame-prefix accounting requires an output rate")
    entries = tuple(
        zip(make_captions(tokens, policy=policy), _display_tokens(tokens, policy), strict=True)
    )
    missing: list[int] = []
    next_supported = False
    for index in range(len(entries) - 1, -1, -1):
        cue, token = entries[index]
        visible = ceil(display_time(cue.end) * fps) > ceil(display_time(cue.start) * fps)
        supported = visible
        if not visible and next_supported:
            following, next_token = entries[index + 1]
            prefix = cue.text.replace("\n", "")
            extension = following.text.replace("\n", "")
            expected = punctuation_lines(cue.text + next_token.text).replace("\n", "")
            supported = (
                display_time(cue.end) == display_time(following.start)
                and token.utterance == next_token.utterance
                and len(extension) > len(prefix)
                and extension.startswith(prefix)
                and extension == expected
            )
        if not visible and not supported:
            missing.append(index)
        next_supported = supported
    return tuple(reversed(missing))


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
    candidate_only: bool = False,
    display_policy: CaptionPolicy | None = None,
) -> tuple[CaptionCue, ...]:
    """Clip holds to owned cuts; explicit policy cannot mix with coarse modes."""
    if display_policy is not None and (accept_coarse_timing or candidate_only):
        raise ValueError("Explicit display policy cannot be combined with coarse modes")
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
    # A technical candidate may show one native punctuation break without
    # claiming approval or fabricating finer timing. Accepted violations can
    # additionally retain coarse clocks.
    policy = (
        display_policy
        if display_policy is not None
        else CaptionPolicy(maximum_characters=18, maximum_lines=1)
    )
    if accept_coarse_timing or candidate_only:
        # Every candidate replay shares one two-line display budget, so captions are
        # split alike whether or not the recognizer left coarse phrases.
        policy = CaptionPolicy(
            maximum_characters=CANDIDATE_CHARACTERS if candidate_only else LINE_CHARACTERS,
            maximum_lines=2,
            enforce_focused_timing=not accept_coarse_timing,
        )
    coarse_candidate = candidate_only and any(
        len(token.text) > 18 or token.end - token.start > policy.maximum_context_seconds
        for token in tokens
        if not all(character in PUNCTUATION for character in token.text)
    )
    if coarse_candidate:
        # Display an entire native phrase on its real clock; never invent word clocks.
        policy = CaptionPolicy(
            maximum_characters=CANDIDATE_CHARACTERS, maximum_lines=2, enforce_focused_timing=False
        )
    for cue in make_captions(tokens, policy=policy):
        end = min(cue.end, limits[cue.start])
        if end <= cue.start:
            raise ValueError("Owned caption has no visible interval")
        display_text = cue.text
        if candidate_only and any(
            len(line) > LINE_CHARACTERS for line in display_text.splitlines()
        ):
            compact = display_text.replace("\n", "")
            if len(compact) > CANDIDATE_CHARACTERS:
                raise ValueError("Native phrase exceeds candidate display budget")
            display_text = compact[:LINE_CHARACTERS] + "\n" + compact[LINE_CHARACTERS:]
        result.append(CaptionCue(display_text, cue.start, end))
    return tuple(result)
