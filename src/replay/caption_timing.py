"""Timed, final ASR tokens; no character-proportional timing or stale carry-over."""

from dataclasses import dataclass
from math import isfinite

PUNCTUATION = frozenset("、。！？!?.,;:…「」『』（）()\"' ")


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
class CaptionPolicy:
    hold_seconds: float = 0.08
    maximum_gap: float = 0.25
    maximum_context_seconds: float = 2.8
    maximum_characters: int = 36
    maximum_lines: int = 2

    def __post_init__(self) -> None:
        values = (self.hold_seconds, self.maximum_gap, self.maximum_context_seconds)
        if not all(isfinite(value) for value in values):
            raise ValueError("Caption policy times must be finite")
        if not 0 <= self.hold_seconds <= self.maximum_gap <= 1:
            raise ValueError("Invalid caption hold or gap")
        if not 0.2 <= self.maximum_context_seconds <= 5:
            raise ValueError("Invalid caption context duration")
        if not 4 <= self.maximum_characters <= 80 or not 1 <= self.maximum_lines <= 3:
            raise ValueError("Invalid caption display budget")


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


def _lexical_tokens(tokens: tuple[SpokenToken, ...]) -> list[SpokenToken]:
    result: list[SpokenToken] = []
    for token in tokens:
        if result and token.start < result[-1].start:
            raise ValueError("Speech tokens are not ordered")
        if all(character in PUNCTUATION for character in token.text):
            if result and token.utterance == result[-1].utterance:
                previous = result.pop()
                result.append(
                    SpokenToken(
                        previous.text + token.text, previous.start, previous.end, previous.utterance
                    )
                )
            continue
        if result and token == result[-1]:
            continue
        if result and token.start == result[-1].start:
            raise ValueError("Different lexical tokens have ambiguous identical onsets")
        result.append(token)
    return result


def make_captions(
    tokens: tuple[SpokenToken, ...], *, policy: CaptionPolicy | None = None
) -> tuple[CaptionCue, ...]:
    """Reveal only recognized words whose onset has occurred; clear during silence.

    Utterance identity must change across hard edits and recognition ownership
    boundaries. A long native phrase without fine timing is rejected, not split
    into fabricated word times. Exact text corrections require separate evidence.
    """
    selected = CaptionPolicy() if policy is None else policy
    words = _lexical_tokens(tokens)
    cues: list[CaptionCue] = []
    context: list[SpokenToken] = []
    for index, token in enumerate(words):
        if len(token.text) > selected.maximum_characters:
            raise ValueError("Native token exceeds display budget; finer alignment is required")
        if token.end - token.start > selected.maximum_context_seconds:
            raise ValueError("Native token timing is too coarse for focused captions")
        if context and _reset_context(context, token, selected):
            context = []
        context.append(token)
        text = punctuation_lines("".join(item.text for item in context))
        if (
            len(text.replace("\n", "")) > selected.maximum_characters
            or len(text.splitlines()) > selected.maximum_lines
        ):
            context = [token]
            text = punctuation_lines(token.text)
        if len(text.splitlines()) > selected.maximum_lines:
            raise ValueError("Native token exceeds punctuation line budget")
        end = token.end + selected.hold_seconds
        if index + 1 < len(words):
            end = min(end, words[index + 1].start)
        if text and end > token.start:
            cues.append(CaptionCue(text, token.start, end))
    return tuple(cues)


def _reset_context(context: list[SpokenToken], token: SpokenToken, policy: CaptionPolicy) -> bool:
    previous = context[-1]
    return (
        token.utterance != previous.utterance
        or token.start - previous.end > min(policy.maximum_gap, policy.hold_seconds)
        or token.start - context[0].start >= policy.maximum_context_seconds
        or previous.text.rstrip().endswith(("。", "！", "？", "!", "?"))
    )
