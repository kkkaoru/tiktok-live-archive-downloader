"""Timed, final ASR tokens; no character-proportional timing or stale carry-over."""

import re
from bisect import bisect_right
from collections.abc import Callable
from dataclasses import dataclass, replace
from fractions import Fraction
from math import ceil, isfinite

from replay.caption_scale import FOCUSED_TIMING_SECONDS, CaptionScale

PUNCTUATION = frozenset("、。！？!?.,;:…「」『』（）()\"' ")
# A token made only of these is a held note or shouting artifact, not a
# recognized word, and cannot fit the caption width.
NON_LEXICAL_MARKS = frozenset("ー〜～")
ELONGATION_VOWELS = frozenset("ぁぃぅぇぉァィゥェォ")
# A recognizer loop such as "ッハッハ…" or "FFFF…" repeats one short unit many times.
DEGENERATE_UNIT_MAXIMUM = 3
DEGENERATE_MINIMUM_REPEATS = 8
# Speech stays under about ten syllables a second; a loop crams far more units into its time.
LOOP_UNITS_PER_SECOND = 15.0
LOOP_RUN = re.compile(
    rf"(.{{1,{DEGENERATE_UNIT_MAXIMUM}}}?)\1{{{DEGENERATE_MINIMUM_REPEATS - 1},}}"
)
SENTENCE_ENDS = ("。", "！", "？", "!", "?")
# A short hiragana tail such as "よ" or "よね" closes the preceding words. It stays
# on their caption instead of being shown alone after a switch.
TRAILING_PARTICLE_MAXIMUM = 2
# The same bounded rational clock used by native FCP/Motion display events.
DISPLAY_TIMESCALE = 60_000


@dataclass(frozen=True, slots=True)
class SpokenToken:
    text: str
    start: float
    end: float
    utterance: str

    def __post_init__(self) -> None:
        if not self.text.strip() or "\0" in self.text or len(self.text) > 2048:
            raise ValueError("Invalid speech token text")
        if not isfinite(self.start) or not isfinite(self.end) or not 0 <= self.start < self.end:
            raise ValueError("Invalid speech token interval")
        if not self.utterance:
            raise ValueError("Speech token must identify its utterance")


@dataclass(frozen=True, slots=True)
class CaptionCue:
    text: str
    start: float
    end: float


@dataclass(frozen=True, slots=True)
class ScaledCaptionCue(CaptionCue):
    scale: CaptionScale


@dataclass(frozen=True, slots=True)
class CaptionPolicy:
    """Display budgets; optional recent-word retention never changes native token clocks."""

    hold_seconds: float = 0.08
    maximum_gap: float = 0.25
    maximum_context_seconds: float = FOCUSED_TIMING_SECONDS
    maximum_characters: int = 36
    maximum_lines: int = 2
    enforce_focused_timing: bool = True
    retain_recent_words: bool = False
    # Only for renderers that wrap evenly: a particle tail may overrun the width by its
    # own length when the words it closes already fill the caption.
    tail_overrun: bool = False
    # Anticipate display-budget resets before an unshown subframe prefix is lost.
    frame_rate: int | None = None
    # Smooth renderers can anticipate a reset at the second visible frame too.
    reset_lookahead_frames: int = 1
    linguistic_boundaries: bool = False
    complete_phrases: bool = False
    measured_layout: Callable[[str], str] | None = None
    indivisible_text: Callable[[str], bool] | None = None
    scoped_layout: Callable[[str, tuple[SpokenToken, ...]], str] | None = None
    scoped_indivisible: Callable[[str, tuple[SpokenToken, ...]], bool] | None = None

    def __post_init__(self) -> None:
        if (self.scoped_layout is not None or self.scoped_indivisible is not None) and (
            not callable(self.scoped_layout)
            or not callable(self.scoped_indivisible)
            or self.measured_layout is None
        ):
            raise ValueError("Scoped caption fitting requires both callbacks and measured layout")
        if self.indivisible_text is not None and (
            not callable(self.indivisible_text) or self.measured_layout is None
        ):
            raise ValueError("Indivisible caption fitting requires measured layout")
        if self.measured_layout is not None and (
            not callable(self.measured_layout) or not self.linguistic_boundaries
        ):
            raise ValueError("Measured layout requires linguistic boundaries and a callable")
        if not isinstance(self.linguistic_boundaries, bool):
            raise ValueError("Linguistic caption boundaries must be a boolean")
        if not isinstance(self.complete_phrases, bool) or (
            self.complete_phrases and not self.linguistic_boundaries
        ):
            raise ValueError("Complete phrase display requires linguistic boundaries")
        if self.linguistic_boundaries and (
            (self.maximum_lines != 2 and self.measured_layout is None) or self.tail_overrun
        ):
            raise ValueError(
                "Linguistic captions require two fixed-width lines without tail overrun"
            )
        values = (self.hold_seconds, self.maximum_gap, self.maximum_context_seconds)
        if not all(isfinite(value) for value in values):
            raise ValueError("Caption policy times must be finite")
        if not 0 <= self.hold_seconds <= self.maximum_gap <= 1:
            raise ValueError("Invalid caption hold or gap")
        if not 0.2 <= self.maximum_context_seconds <= 5:
            raise ValueError("Invalid caption context duration")
        if not 4 <= self.maximum_characters <= 80 or not 1 <= self.maximum_lines <= 3:
            raise ValueError("Invalid caption display budget")
        if self.frame_rate is not None and (
            type(self.frame_rate) is not int or not 1 <= self.frame_rate <= 120
        ):
            raise ValueError("Invalid caption frame rate")
        if type(self.reset_lookahead_frames) is not int or not (
            1 <= self.reset_lookahead_frames <= (self.frame_rate or 1)
        ):
            raise ValueError("Invalid caption reset lookahead")


def display_time(seconds: float) -> Fraction:
    """Canonical display clock only; never replace native recognition timestamps."""
    return Fraction(seconds).limit_denominator(DISPLAY_TIMESCALE)


def punctuation_lines(text: str) -> str:
    """Break after punctuation clusters, preserving decimal/thousands notation."""
    output: list[str] = []
    pending_break = False
    previous = ""
    stripped = text.strip()
    for index, character in enumerate(stripped):
        if character in "\r\n":
            pending_break = True
            continue
        if pending_break and character not in PUNCTUATION:
            if output:
                output.append("\n")
            pending_break = False
        if not output and character in "、。！？!?.,;:…」』）) ":
            continue
        if pending_break and character.isspace():
            continue
        output.append(character)
        following = stripped[index + 1 : index + 2]
        numeric = character in ",." and previous.isdecimal() and following.isdecimal()
        if character in "、。！？!?.,;:…" and not numeric:
            pending_break = True
        previous = character
    return "".join(output).strip()


def owned_tokens(
    tokens: tuple[SpokenToken, ...], *, offset: float, owner_start: float, owner_end: float
) -> tuple[SpokenToken, ...]:
    """Only a token's midpoint owner publishes it from overlapping ASR chunks."""
    if not all(isfinite(value) for value in (offset, owner_start, owner_end)):
        raise ValueError("Invalid recognition chunk times")
    if not 0 <= offset <= owner_start < owner_end:
        raise ValueError("Invalid recognition chunk ownership")
    return tuple(
        SpokenToken(token.text, token.start + offset, token.end + offset, token.utterance)
        for token in tokens
        if owner_start <= (token.start + token.end) / 2 + offset < owner_end
    )


def _repeated_unit(text: str) -> tuple[str, int] | None:
    """The short unit a text only repeats, and how many times; None for other text."""
    compact = "".join(character for character in text if character not in PUNCTUATION)
    for unit in range(1, DEGENERATE_UNIT_MAXIMUM + 1):
        if (
            compact
            and len(compact) % unit == 0
            and compact == compact[:unit] * (len(compact) // unit)
        ):
            return compact[:unit], len(compact) // unit
    return None


def _degenerate_repetition(text: str) -> bool:
    """Whole-token repetition of one short unit is a recognizer loop, not speech."""
    repeated = _repeated_unit(text)
    return repeated is not None and repeated[1] >= DEGENERATE_MINIMUM_REPEATS


def _without_loop(token: SpokenToken) -> SpokenToken:
    """Drop a loop run inside one token ("ララ…ラなん" keeps "なん"); a whole loop stays for
    the whole-token check, and repeated small vowels or marks still lengthen their word."""
    text = LOOP_RUN.sub(
        lambda match: (
            match.group(0) if set(match.group(1)) <= ELONGATION_VOWELS | NON_LEXICAL_MARKS else ""
        ),
        token.text,
    )
    dense = len(token.text) >= LOOP_UNITS_PER_SECOND * max(token.end - token.start, 1e-9)
    if not dense or not text.strip() or text == token.text:
        return token
    return replace(token, text=text)


def _loop_runs(tokens: tuple[SpokenToken, ...]) -> set[int]:
    """Indices of consecutive tokens that repeat one short unit, together a recognizer loop.

    A loop is often cut into pieces ("ララ", "ラララ…") that each stay below the whole-token
    threshold; together they repeat the unit at least ``DEGENERATE_MINIMUM_REPEATS`` times and
    faster than ``LOOP_UNITS_PER_SECOND`` (a sung "ラ, ラ, ラ" stays).
    """
    loops: set[int] = set()
    first = 0
    while first < len(tokens):
        repeated = _repeated_unit(tokens[first].text)
        last = first
        # Repeated small vowels or marks lengthen the word before them ("こぉぉぉ"): no loop.
        if repeated is not None and not set(repeated[0]) <= ELONGATION_VOWELS | NON_LEXICAL_MARKS:
            count = repeated[1]
            while last + 1 < len(tokens):
                following = _repeated_unit(tokens[last + 1].text)
                if following is None or following[0] != repeated[0]:
                    break
                count += following[1]
                last += 1
            seconds = max(tokens[last].end - tokens[first].start, 1e-9)
            if (
                last > first
                and count >= DEGENERATE_MINIMUM_REPEATS
                and count >= LOOP_UNITS_PER_SECOND * seconds
            ):
                loops.update(range(first, last + 1))
        first = last + 1
    return loops


def _lexical_tokens(
    tokens: tuple[SpokenToken, ...], *, word_mark_gap: float | None = None
) -> list[SpokenToken]:
    return lexical_token_sources(tokens, word_mark_gap=word_mark_gap)[0]


def lexical_token_sources(
    tokens: tuple[SpokenToken, ...], *, word_mark_gap: float | None = None
) -> tuple[list[SpokenToken], list[tuple[int, ...]]]:
    """Preserve source indices through lexical joins without changing source clocks."""
    result: list[SpokenToken] = []
    indices: list[tuple[int, ...]] = []
    loops = _loop_runs(tokens)
    for index, token in enumerate(tokens):
        if result and token.start < result[-1].start:
            raise ValueError("Speech tokens are not ordered")
        if index in loops:
            continue
        token = _without_loop(token)
        word_mark = word_mark_gap is not None and _word_mark_continuation(
            result, token, maximum_gap=word_mark_gap
        )
        if not word_mark and (
            all(character in NON_LEXICAL_MARKS for character in token.text)
            or _degenerate_repetition(token.text)
        ):
            # Never merge these into the previous word: a long run of marks would
            # otherwise widen that caption past the display budget.
            continue
        if all(character in PUNCTUATION for character in token.text):
            if result and _same_display_owner(
                token, result[-1], linguistic=word_mark_gap is not None
            ):
                previous = result.pop()
                result.append(
                    SpokenToken(
                        previous.text + token.text, previous.start, previous.end, previous.utterance
                    )
                )
                indices[-1] = (*indices[-1], index)
            continue
        if result and token == result[-1]:
            continue
        if result and token.start == result[-1].start:
            raise ValueError("Different lexical tokens have ambiguous identical onsets")
        result.append(token)
        indices.append((index,))
    return result, indices


def _same_display_owner(left: SpokenToken, right: SpokenToken, *, linguistic: bool) -> bool:
    if not linguistic:
        return left.utterance == right.utterance
    from replay.caption_linguistics import recognition_continuity

    return recognition_continuity(left.utterance) == recognition_continuity(right.utterance)


def _word_mark_continuation(
    previous: list[SpokenToken], token: SpokenToken, *, maximum_gap: float
) -> bool:
    vowel_run = len(set(token.text)) == 1 and token.text[0] in ELONGATION_VOWELS
    if (token.text != "ー" and not vowel_run) or not previous:
        return False
    following = token
    for last in reversed(previous):
        if (
            not _same_display_owner(last, following, linguistic=True)
            or following.start - last.end > maximum_gap
        ):
            return False
        if any(
            "ぁ" <= char <= "ヺ" and char not in ELONGATION_VOWELS | {"ー"} for char in last.text
        ):
            return True
        if not all(char in ELONGATION_VOWELS | {"ー"} for char in last.text):
            return False
        following = last
    return False


def _display_tokens(tokens: tuple[SpokenToken, ...], policy: CaptionPolicy) -> list[SpokenToken]:
    """Use identical display units in planning and frame-support accounting."""
    words = _lexical_tokens(
        tokens, word_mark_gap=policy.maximum_gap if policy.linguistic_boundaries else None
    )
    if not policy.linguistic_boundaries:
        return words
    # A protected word may span many finely timed fragments. Validate source
    # timing before aggregation so a slow intact word is not mistaken for one
    # coarse recognition token. Never invent internal times to split that word.
    for token in words:
        if (
            policy.enforce_focused_timing
            and token.end - token.start > policy.maximum_context_seconds
        ):
            raise ValueError(
                "Native token timing is too coarse for focused captions: "
                f"{token.text!r} at {token.start}..{token.end}"
            )
    from replay.caption_linguistics import atomic_tokens

    def fits(unit: SpokenToken) -> bool:
        return not _over_budget(punctuation_lines(unit.text), policy, context=(unit,)) and (
            not policy.enforce_focused_timing
            or unit.end - unit.start <= policy.maximum_context_seconds
        )

    return atomic_tokens(
        words,
        maximum_gap=policy.maximum_gap,
        complete_phrases=policy.complete_phrases,
        accept_phrase=fits,
    )


def phrase_units(
    tokens: tuple[SpokenToken, ...], *, policy: CaptionPolicy
) -> tuple[SpokenToken, ...]:
    """Expose sanitized, measured POS units for whole-phrase display renderers."""
    if not policy.linguistic_boundaries or not policy.complete_phrases:
        raise ValueError("Phrase units require complete linguistic phrases")
    return tuple(_display_tokens(tokens, policy))


def make_captions(
    tokens: tuple[SpokenToken, ...],
    *,
    policy: CaptionPolicy | None = None,
    excluded_regions: tuple[tuple[float, float], ...] = (),
) -> tuple[CaptionCue, ...]:
    """Leave uncertain evidence regions blank without changing audio or word clocks.

    Callers retain excluded recognition/supports in their private audit, not as
    visible placeholder text. Intersecting tokens are not partially guessed.
    Context and caption hold cannot carry text across an excluded region.
    """
    if len(excluded_regions) > 100_000:
        raise ValueError("Excluded caption region budget exceeded")
    previous_end = 0.0
    for start, end in excluded_regions:
        if (
            isinstance(start, bool)
            or isinstance(end, bool)
            or not isfinite(start)
            or not isfinite(end)
            or not previous_end <= start < end <= 86400
        ):
            raise ValueError("Invalid or overlapping excluded caption regions")
        previous_end = end
    if not excluded_regions:
        return _make_captions(tokens, policy=policy)
    starts = tuple(start for start, end in excluded_regions)
    ends = tuple(end for start, end in excluded_regions)
    # Validate ordering even for tokens excluded before display validation.
    if any(right.start < left.start for left, right in zip(tokens, tokens[1:], strict=False)):
        raise ValueError("Speech tokens are not ordered")
    visible: list[SpokenToken] = []
    for token in tokens:
        index = bisect_right(ends, token.start)
        if index < len(starts) and starts[index] < token.end:
            continue
        visible.append(replace(token, utterance=f"region:{index}:{token.utterance}"))
    cues = _make_captions(tuple(visible), policy=policy)
    result: list[CaptionCue] = []
    for cue in cues:
        index = bisect_right(ends, cue.start)
        end = min(cue.end, starts[index]) if index < len(starts) else cue.end
        if end > cue.start:
            result.append(replace(cue, end=end))
    return tuple(result)


def _make_captions(
    tokens: tuple[SpokenToken, ...], *, policy: CaptionPolicy | None = None
) -> tuple[CaptionCue, ...]:
    """Reveal only recognized words whose onset has occurred; clear during silence.

    Utterance identity must change across hard edits and recognition ownership
    boundaries. A long native phrase without fine timing is rejected, not split
    into fabricated word times. Exact text corrections require separate evidence.
    """
    selected = CaptionPolicy() if policy is None else policy
    words = _display_tokens(tokens, selected)
    attached = _attached_tails(words, selected)
    cues: list[CaptionCue] = []
    cue_contexts: dict[float, tuple[SpokenToken, ...]] = {}
    context: list[SpokenToken] = []
    for index, token in enumerate(words):
        if (
            len(token.text) > selected.maximum_characters
            and not (_indivisible(token.text, selected, (token,)))
        ) or (
            selected.linguistic_boundaries
            and _over_budget(punctuation_lines(token.text), selected, context=(token,))
        ):
            raise ValueError(
                "Native token exceeds display budget; finer alignment is required: "
                f"{token.text!r} at {token.start}..{token.end}"
            )
        if (
            not selected.linguistic_boundaries
            and selected.enforce_focused_timing
            and token.end - token.start > selected.maximum_context_seconds
        ):
            raise ValueError("Native token timing is too coarse for focused captions")
        if context and not attached[index] and _reset_context(context, token, selected):
            context = []
        context.append(token)
        following_text = (
            words[index + 1].text + _fitting_tail(words, attached, index + 1, selected)
            if index + 1 < len(words)
            else ""
        )
        if index + 1 < len(words) and _preempt_frame_reset(
            context, following=words[index + 1], following_text=following_text, policy=selected
        ):
            context = _rewind_frame_reset(
                context,
                cues,
                following=words[index + 1],
                following_text=following_text,
                policy=selected,
                cue_contexts=cue_contexts,
            )
        # Switch before a word whose particle tail would not fit, so the tail is
        # never left alone. The tail is only used for the decision, never shown early.
        tail = _fitting_tail(words, attached, index, selected)
        overrun = (token.text if attached[index] else "") + tail
        allowance = len(overrun) if selected.tail_overrun else 0
        while len(context) > 1 and (
            _over_budget(
                punctuation_lines("".join(item.text for item in context) + tail),
                selected,
                allowance,
                context=tuple(context + _tail(words, attached, index)),
            )
            or (
                selected.retain_recent_words
                and token.start - context[0].start >= selected.maximum_context_seconds
            )
        ):
            context = context[1:] if selected.retain_recent_words else [token]
        text = punctuation_lines("".join(item.text for item in context))
        if selected.measured_layout is None and len(text.splitlines()) > selected.maximum_lines:
            raise ValueError("Native token exceeds punctuation line budget")
        end = token.end + selected.hold_seconds
        if index + 1 < len(words):
            # Keep the words visible until their attached particle extends them.
            following = words[index + 1].start
            end = following if attached[index + 1] else min(end, following)
        if text and end > token.start:
            cues.append(CaptionCue(text, token.start, end))
            cue_contexts[token.start] = tuple(context)
    if selected.measured_layout is not None:
        return tuple(
            replace(cue, text=_measured_layout(cue.text, selected, cue_contexts[cue.start]))
            for cue in cues
        )
    return tuple(cues)


def _preempt_frame_reset(
    context: list[SpokenToken],
    *,
    following: SpokenToken,
    following_text: str,
    policy: CaptionPolicy,
) -> bool:
    """Start a new prefix early, without revealing future words or retiming speech.

    Only anticipate width/lifetime resets within the requested output-frame horizon.
    Semantic boundaries and unsupported two-token budgets
    are not bridged; callers still audit frame visibility after assembly.
    """
    if policy.frame_rate is None or policy.retain_recent_words or len(context) < 2:
        return False
    token = context[-1]
    last_frame = Fraction(
        ceil(display_time(token.start) * policy.frame_rate) + policy.reset_lookahead_frames - 1,
        policy.frame_rate,
    )
    if display_time(following.start) > last_frame:
        return False
    pair = punctuation_lines(token.text + following_text)
    if _reset_context([token], following, policy) or _over_budget(
        pair, policy, context=(token, following)
    ):
        return False
    return _reset_context(context, following, policy) or _over_budget(
        punctuation_lines("".join(item.text for item in context) + following_text),
        policy,
        context=(*context, following),
    )


def _rewind_frame_reset(
    context: list[SpokenToken],
    cues: list[CaptionCue],
    *,
    following: SpokenToken,
    following_text: str,
    policy: CaptionPolicy,
    cue_contexts: dict[float, tuple[SpokenToken, ...]],
) -> list[SpokenToken]:
    """Move a budget boundary earlier if its preceding prefix would lose support."""
    fps = policy.frame_rate
    assert fps is not None
    split = len(context) - 1
    while split:
        previous = cues[-(len(context) - split)]
        frames = ceil(display_time(previous.end) * fps) - ceil(display_time(previous.start) * fps)
        if frames >= policy.reset_lookahead_frames:
            break
        split -= 1
    candidate = context[split:]
    if _reset_context(candidate, following, policy) or _over_budget(
        punctuation_lines("".join(item.text for item in candidate) + following_text),
        policy,
        context=(*candidate, following),
    ):
        return context
    for index in range(split, len(context) - 1):
        position = len(cues) - (len(context) - 1 - index)
        cues[position] = replace(
            cues[position],
            text=punctuation_lines("".join(item.text for item in context[split : index + 1])),
        )
        cue_contexts[cues[position].start] = tuple(context[split : index + 1])
    return candidate


def _reset_context(context: list[SpokenToken], token: SpokenToken, policy: CaptionPolicy) -> bool:
    previous = context[-1]
    if policy.linguistic_boundaries:
        from replay.caption_linguistics import semantic_break

        if semantic_break(previous.text, token.text):
            return True
    return (
        token.utterance != previous.utterance
        or token.start - previous.end > min(policy.maximum_gap, policy.hold_seconds)
        or (
            not policy.retain_recent_words
            and token.start - context[0].start >= policy.maximum_context_seconds
        )
        or previous.text.rstrip().endswith(SENTENCE_ENDS)
    )


def _indivisible(text: str, policy: CaptionPolicy, context: tuple[SpokenToken, ...]) -> bool:
    if policy.scoped_indivisible is not None:
        return policy.scoped_indivisible(text, context)
    return policy.indivisible_text is not None and policy.indivisible_text(text)


def _measured_layout(
    text: str, policy: CaptionPolicy, context: tuple[SpokenToken, ...] = ()
) -> str:
    if policy.measured_layout is None:
        raise ValueError("Missing measured caption layout")
    flat = text.replace("\n", "")
    result = (
        policy.measured_layout(flat)
        if policy.scoped_layout is None
        else policy.scoped_layout(flat, context)
    )
    if result.replace("\n", "") != flat or len(result.splitlines()) > policy.maximum_lines:
        raise ValueError("Measured layout changed text or exceeded the line budget")
    return result


def _over_budget(
    text: str,
    policy: CaptionPolicy,
    allowance: int = 0,
    *,
    context: tuple[SpokenToken, ...] = (),
) -> bool:
    flat = text.replace("\n", "")
    if policy.measured_layout is not None and flat:
        from replay.caption_linguistics import CaptionLayoutError

        if len(flat) > policy.maximum_characters + allowance and not (
            _indivisible(flat, policy, context)
        ):
            return True
        try:
            _measured_layout(flat, policy, context)
        except CaptionLayoutError:
            return True
        return False
    if (
        len(flat) > policy.maximum_characters + allowance
        or len(text.splitlines()) > policy.maximum_lines
    ):
        return True
    if policy.linguistic_boundaries and flat:
        # Use the renderer's actual intact-word budget, not just the total length.
        from replay.caption_linguistics import CaptionLayoutError, wrap_two_lines

        try:
            wrap_two_lines(flat, width=policy.maximum_characters // policy.maximum_lines)
        except CaptionLayoutError:
            return True
    return False


def _particle_core(token: SpokenToken) -> str:
    return "".join(character for character in token.text if character not in PUNCTUATION)


def _continues_with_particle(
    previous: SpokenToken, token: SpokenToken, policy: CaptionPolicy
) -> bool:
    core = _particle_core(token)
    return (
        0 < len(core) <= TRAILING_PARTICLE_MAXIMUM
        and all("ぁ" <= character <= "ゖ" or character == "ー" for character in core)
        and token.utterance == previous.utterance
        and token.start - previous.end <= policy.maximum_gap
        and previous.text.rstrip()[-1:] not in PUNCTUATION
    )


def _attached_tails(words: list[SpokenToken], policy: CaptionPolicy) -> list[bool]:
    """Mark the short hiragana tail that closes each run of words.

    Within a longer run of short hiragana pieces (a fragmented word), only its
    final pieces up to the tail length are attached.
    """
    if policy.linguistic_boundaries:
        # Optional editing dependency: normal listing/acquisition never imports it.
        from replay.caption_linguistics import attached_tokens

        return attached_tokens(words, maximum_gap=policy.maximum_gap)
    attached = [False] * len(words)
    index = 1
    while index < len(words):
        end = index
        while end < len(words) and _continues_with_particle(words[end - 1], words[end], policy):
            end += 1
        first = end
        length = 0
        while (
            first > index
            and length + len(_particle_core(words[first - 1])) <= TRAILING_PARTICLE_MAXIMUM
        ):
            first -= 1
            length += len(_particle_core(words[first]))
        attached[first:end] = [True] * (end - first)
        index = max(end, index + 1)
    return attached


def _fitting_tail(
    words: list[SpokenToken], attached: list[bool], index: int, policy: CaptionPolicy
) -> str:
    """Reserve a complete phrase only if it could fit on a page by itself.

    Reserve at the phrase start, not again at each dependent fragment.
    A long dependent phrase must not repeatedly erase its already spoken prefix
    merely because its future suffix cannot fit. Current text still obeys the
    unchanged display budget, native-token boundaries and frame-retention audit.
    """
    if policy.linguistic_boundaries and attached[index]:
        return ""
    tail = "".join(item.text for item in _tail(words, attached, index))
    if policy.linguistic_boundaries and _over_budget(
        punctuation_lines(words[index].text + tail),
        policy,
        context=(words[index], *_tail(words, attached, index)),
    ):
        return ""
    return tail


def _tail(words: list[SpokenToken], attached: list[bool], index: int) -> list[SpokenToken]:
    end = index + 1
    while end < len(words) and attached[end]:
        end += 1
    return words[index + 1 : end]
