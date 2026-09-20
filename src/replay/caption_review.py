"""Select bounded native ASR alternatives using shared, timed context anchors."""

from collections.abc import Iterator
from dataclasses import dataclass
from itertools import pairwise
from math import isfinite

from .caption_repair import ClockLabel, canonical_text
from .caption_timing import PUNCTUATION, CaptionPolicy, punctuation_lines


@dataclass(frozen=True, slots=True)
class ContextSelection:
    original_start: int
    original_end: int
    replacement: tuple[ClockLabel, ...]


def native_phrase_fallback(
    original: tuple[ClockLabel, ...],
    phrase: ClockLabel,
    *,
    first: int,
    end: int,
    source_duration: float,
    policy: CaptionPolicy | None = None,
) -> ContextSelection | None:
    """Use a supplied native phrase clock whole, never split it into word clocks.

    The caller must supply the actual provider phrase and its provenance. The
    entire corresponding word block must conserve its text and fit the unchanged
    display policy and external neighbors. Record phrase-level timing explicitly.
    """
    if (
        type(first) is not int
        or type(end) is not int
        or not 0 <= first < end <= len(original) <= 100_000
        or isinstance(source_duration, bool)
        or not isfinite(source_duration)
        or not 0 < source_duration <= 86400
    ):
        raise ValueError("Invalid native phrase selection bounds")
    if any(label.end > source_duration for label in original) or any(
        right.start < left.start for left, right in pairwise(original)
    ):
        raise ValueError("Invalid original phrase clocks")
    selected = CaptionPolicy() if policy is None else policy
    lower = original[first - 1].end if first else 0
    upper = original[end].start if end < len(original) else source_duration
    if (
        phrase.start < lower
        or phrase.end > upper
        or not 0 < phrase.end - phrase.start <= selected.maximum_context_seconds
        or len(phrase.text) > selected.maximum_characters
        or len(punctuation_lines(phrase.text).splitlines()) > selected.maximum_lines
        or canonical_text(phrase.text)
        != "".join(canonical_text(label.text) for label in original[first:end])
    ):
        return None
    return ContextSelection(first, end, (phrase,))


def refine_coarse_native_label(
    original: ClockLabel,
    alternatives: tuple[ClockLabel, ...],
    *,
    policy: CaptionPolicy | None = None,
) -> ClockLabel | None:
    """Choose a unique real fine clock inside a coarse native interval, never cap it.

    Text must match without spelling correction. The caller retains the coarse
    raw record and its audio support; only the caption source choice changes.
    """
    if len(alternatives) > 100_000:
        raise ValueError("Native refinement budget exceeded")
    selected = CaptionPolicy() if policy is None else policy
    if original.end - original.start <= selected.maximum_context_seconds:
        return None
    matches = {
        (label.text, label.start, label.end): label
        for label in alternatives
        if canonical_text(label.text) == canonical_text(original.text)
        and original.start <= label.start < label.end <= original.end
        and label.end - label.start <= selected.maximum_context_seconds
        and len(label.text) <= selected.maximum_characters
        and len(punctuation_lines(label.text).splitlines()) <= selected.maximum_lines
    }
    return next(iter(matches.values())) if len(matches) == 1 else None


def spoken_text(text: str) -> str:
    """Ignore punctuation for anchors only; never transliterate or change spelling."""
    return "".join(
        character for character in text if character not in PUNCTUATION and not character.isspace()
    )


def _runs(labels: tuple[ClockLabel, ...], start: int, end: int) -> Iterator[tuple[str, int, int]]:
    for left in range(start, end):
        text = ""
        for right in range(left + 1, min(left + 3, end) + 1):
            label = labels[right - 1]
            if label.start == label.end:
                break
            text += spoken_text(label.text)
            if len(text) >= 2:
                yield text, left, right


def _boundaries(
    original: tuple[ClockLabel, ...],
    alternative: tuple[ClockLabel, ...],
    start: int,
    end: int,
    context_start: float,
    context_end: float,
) -> set[tuple[int, int]]:
    lookup: dict[str, list[tuple[int, int]]] = {}
    for text, left, right in _runs(alternative, 0, len(alternative)):
        lookup.setdefault(text, []).append((left, right))
    boundaries: set[tuple[int, int]] = set()
    for text, left, right in _runs(original, start, end):
        if (
            original[left].start < context_start
            or max(label.end for label in original[left:right]) > context_end
        ):
            continue
        for alternative_left, alternative_right in lookup.get(text, []):
            if (
                abs(original[left].start - alternative[alternative_left].start) <= 0.5
                and abs(original[right - 1].end - alternative[alternative_right - 1].end) <= 0.5
            ):
                boundaries.update(((left, alternative_left), (right, alternative_right)))
                if len(boundaries) > 256:
                    raise ValueError("Review anchor budget exceeded")
    return boundaries


def select_native_context(
    original: tuple[ClockLabel, ...],
    alternative: tuple[ClockLabel, ...],
    *,
    target_start: int,
    target_end: int,
    context_start: float,
    context_end: float,
    source_duration: float,
    required_text: tuple[str, ...] = (),
    policy: CaptionPolicy | None = None,
) -> ContextSelection | None:
    """Choose complete native words between shared anchors, without editing clocks.

    Unlike exact clock repair, this may select different ASR text. The caller must
    retain both transcripts, record the replacement and its provider quality, and
    protect original speech evidence independently. This does not prove accuracy
    or authorize cutting audio. Missing required lexical content, invalid clocks,
    coarse display timing or missing safe seams leave the review unresolved.
    """
    if (
        any(isinstance(value, bool) for value in (context_start, context_end, source_duration))
        or not all(isfinite(value) for value in (context_start, context_end, source_duration))
        or not 0 <= context_start < context_end <= source_duration <= 86400
        or context_end - context_start > 60
        or type(target_start) is not int
        or type(target_end) is not int
        or not 0 <= target_start < target_end <= len(original) <= 100_000
        or len(alternative) > 1000
        or sum(len(label.text) for label in alternative) > 32000
        or len(required_text) > 256
        or any(len(text) > 2048 for text in required_text)
    ):
        raise ValueError("Invalid native review bounds or budget")
    for labels in (original, alternative):
        if any(right.start < left.start for left, right in pairwise(labels)):
            raise ValueError("Review onsets must be ordered")
        if any(label.end > source_duration for label in labels):
            raise ValueError("Review label exceeds source")
    if any(label.start < context_start or label.end > context_end for label in alternative):
        raise ValueError("Alternative exceeds measured context")
    if not alternative:
        return None
    if any(
        label.start < context_start or label.end > context_end
        for label in original[target_start:target_end]
    ):
        return None
    eligible = [
        index
        for index, label in enumerate(original)
        if label.start >= context_start and label.end <= context_end
    ]
    left_options = _boundaries(
        original, alternative, eligible[0], target_start, context_start, context_end
    )
    right_options = _boundaries(
        original, alternative, target_end, eligible[-1] + 1, context_start, context_end
    )
    if context_start == 0:
        left_options.add((0, 0))
    if context_end == source_duration:
        right_options.add((len(original), len(alternative)))
    selected_policy = CaptionPolicy() if policy is None else policy
    required = tuple(spoken_text(text) for text in required_text if spoken_text(text))
    best: tuple[int, int, int, int, int] | None = None
    for old_left, new_left in sorted(left_options):
        for old_right, new_right in sorted(right_options):
            if not old_left <= target_start < target_end <= old_right or not new_left < new_right:
                continue
            labels = alternative[new_left:new_right]
            lower = original[old_left - 1].end if old_left else 0
            upper = original[old_right].start if old_right < len(original) else source_duration
            if labels[0].start < lower or labels[-1].end > upper:
                continue
            if any(
                not 0 < label.end - label.start <= selected_policy.maximum_context_seconds
                or len(label.text) > selected_policy.maximum_characters
                or len(punctuation_lines(label.text).splitlines()) > selected_policy.maximum_lines
                for label in labels
            ):
                continue
            if any(
                right.start <= left.start or right.end < left.end
                for left, right in pairwise(labels)
            ):
                continue
            text = spoken_text("".join(label.text for label in labels))
            if any(word not in text for word in required):
                continue
            key = (old_right - old_left, old_left, old_right, new_left, new_right)
            if best is None or key < best:
                best = key
    if best is None:
        return None
    _, old_left, old_right, new_left, new_right = best
    return ContextSelection(old_left, old_right, alternative[new_left:new_right])
