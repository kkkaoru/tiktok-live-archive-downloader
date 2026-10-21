"""Japanese POS-aware display boundaries; recognition text and clocks stay untouched.

Rust Vibrato with pinned IPADIC supplies real morphology, not character-count
word guesses. These are conservative phrase candidates, not a dependency parse.
"""

import argparse
import hashlib
import json
from collections.abc import Callable, Sequence
from functools import lru_cache
from itertools import accumulate, combinations, islice, pairwise
from math import isfinite
from pathlib import Path

import regex
from vibrato import Vibrato

from replay.caption_timing import SpokenToken

DEPENDENT_PARTS = frozenset({"助詞", "助動詞"})
MAX_TEXT_LENGTH = 8192
MAX_LAYOUT_CANDIDATES = 65_536
DICTIONARY_PATH = Path("edits/models/vibrato/system.dic")
DICTIONARY_SHA256 = "f3ddd9cacbd6eff696e49bb9266824eed28dceaa89c55e75e4a13384cc7f9d02"
MAX_DICTIONARY_BYTES = 64 * 1024**2
KATAKANA_PAIR = regex.compile(r"[\p{Script=Katakana}ーﾞﾟ]{2}")
ALPHANUMERIC_PAIR = regex.compile(r"[\p{Script=Latin}\p{Number}]{2}")
DEPENDENT_KANA = frozenset("ぁぃぅぇぉっゃゅょゎァィゥェォッャュョヮヵヶーﾞﾟ々ゝゞヽヾ")
PROHIBITED_LINE_START = DEPENDENT_KANA | frozenset("、。！？!?.,;:…」』）)]｝")
PROHIBITED_LINE_END = frozenset("「『（([｛")
RAW_CHUNK_NAME = regex.compile(r"[0-9]+\.raw\.json")
ASSEMBLED_GROUP_OWNER = regex.compile(r"(?P<context>.*:)?(?P<window>[0-9]{12}):[0-9]+")


def recognition_continuity(utterance: str) -> str:
    """Separate processing ownership from the caller's verified display context.

    Numbered raw-ASR chunks may share context within their transcript directory
    and cut. Assembled PCM labels use window-start:group-index: group indices
    and overlapping windows identify processing ownership, not utterance changes.
    Their source/region/scene/speaker context and projected cut remain hard
    boundaries. Token consumers separately bind the original source and native
    provenance. Unknown owner formats retain their full identity; real timing
    gaps and semantic resets are still checked by callers.
    """
    owner, separator, cut = utterance.rpartition(":cut-")
    if separator and cut.isascii() and cut.isdecimal():
        assembled = ASSEMBLED_GROUP_OWNER.fullmatch(owner)
        if assembled is not None:
            context = assembled.group("context") or ""
            return f"{context}assembled:cut-{cut}"
        path = Path(owner)
        if RAW_CHUNK_NAME.fullmatch(path.name):
            return f"{path.parent}:cut-{cut}"
    return utterance


class CaptionLayoutError(ValueError):
    """Valid morphology cannot fit the requested caption line budget."""


@lru_cache(maxsize=1)
def _tokenizer() -> Vibrato:
    # Unlimited unknown grouping can swallow whole ordinary clauses before
    # POS handling ever sees them. Bound recognition inside the tokenizer;
    # display-level unknown-name and kana protections remain independent.
    return _load_tokenizer(max_grouping_len=1)


def _load_tokenizer(*, max_grouping_len: int = 0) -> Vibrato:
    if (
        not DICTIONARY_PATH.is_file()
        or not 0 < DICTIONARY_PATH.stat().st_size <= MAX_DICTIONARY_BYTES
    ):
        raise ValueError("Install the pinned Vibrato dictionary; see docs/development.md")
    data = DICTIONARY_PATH.read_bytes()
    if hashlib.sha256(data).hexdigest() != DICTIONARY_SHA256:
        raise ValueError("Vibrato dictionary does not match the pinned checksum")
    return Vibrato(data, ignore_space=False, max_grouping_len=max_grouping_len)


@lru_cache(maxsize=1)
def _bounded_tokenizer() -> Vibrato:
    """Expose dictionary anchors hidden by unlimited unknown-word grouping."""
    return _tokenizer()


def _honorific_names(text: str) -> tuple[tuple[int, int], ...]:
    """Recover a kana name's leading function-word homograph at a phrase start.

    A person suffix supplies lexical evidence. Do not cross punctuation, a verb,
    or a real token-group boundary. After an existing particle/interjection, the
    recovered name starts a new phrase instead of absorbing the preceding clause.
    """
    items = tuple(_tokenizer().tokenize(text))
    spans: list[tuple[int, int]] = []
    for index, item in enumerate(items):
        feature = item.feature().split(",")
        if len(feature) < 3 or feature[1:3] != ["接尾", "人名"]:
            continue
        first = index
        while first:
            before = items[first - 1]
            pos = before.feature().split(",")
            if (
                pos[0] != "名詞"
                or pos[1] == "接尾"
                or not regex.fullmatch(
                    r"[\p{Script=Hiragana}\p{Script=Katakana}ー]+", before.surface()
                )
            ):
                break
            first -= 1
        if not 0 < first < index:
            continue
        leading = items[first - 1]
        pos = leading.feature().split(",")
        if (
            pos[0] not in DEPENDENT_PARTS
            or not regex.fullmatch(r"[\p{Script=Hiragana}]", leading.surface())
            or (
                first > 1
                and items[first - 2].feature().split(",")[0]
                not in {"助詞", "感動詞", "記号", "連体詞"}
            )
        ):
            continue
        spans.append((leading.start(), item.end()))
    return tuple(spans)


@lru_cache(maxsize=256)
def boundaries(text: str) -> tuple[tuple[int, ...], tuple[int, ...]]:
    """Return phrase and morpheme ends as original Python string offsets."""
    if not text or len(text) > MAX_TEXT_LENGTH or "\0" in text:
        raise ValueError("Invalid Japanese caption text")
    morphemes = _tokenizer().tokenize(text)
    if "".join(item.surface() for item in morphemes) != text:
        raise ValueError("Morphological analysis changed caption text")
    phrases: list[int] = []
    grapheme_ends = {match.end() for match in regex.finditer(r"\X", text)}
    protected = set(range(1, len(text))) - grapheme_ends
    names = _honorific_names(text)
    name_starts = {start for start, _ in names}
    protected.update(point for start, end in names for point in range(start + 1, end))
    previous: tuple[str, ...] = ()
    for item in morphemes:
        pos = tuple(item.feature().split(","))
        if len(pos) < 2 or text[item.start() : item.end()] != item.surface():
            raise ValueError("Invalid Vibrato feature or character offsets")
        opening = pos[1] == "括弧開"
        dependent = (
            pos[0] in DEPENDENT_PARTS or pos[1] == "接尾" or (pos[0] == "記号" and not opening)
        )
        prefix = previous and (previous[0] == "接頭詞" or previous[1] == "括弧開")
        if item.start() and item.surface()[0] in DEPENDENT_KANA:
            protected.add(item.start())
        # Keep unknown/name fragments together, but allow dictionary-backed loanword
        # compounds (for example a modifier plus a device name) to wrap between words.
        inside_katakana = KATAKANA_PAIR.fullmatch(text[item.start() - 1 : item.start() + 1])
        uncertain_name = (
            len(pos) < 9 or pos[1] == "固有名詞" or len(previous) < 9 or previous[1] == "固有名詞"
        )
        inside_alphanumeric = ALPHANUMERIC_PAIR.fullmatch(text[item.start() - 1 : item.start() + 1])
        if (inside_katakana or inside_alphanumeric) and uncertain_name:
            protected.add(item.start())
        if (
            item.start()
            and item.start() not in protected
            and (item.start() in name_starts or (not dependent and not prefix))
        ):
            phrases.append(item.start())
        previous = pos
    phrases.append(len(text))
    return tuple(phrases), tuple(item.end() for item in morphemes if item.end() not in protected)


def wrap_two_lines(text: str, *, width: int = 18) -> str:
    """Prefer phrase ends, then intact morphemes; never split a word by count."""
    if type(width) is not int or not 1 <= width <= 80 or "\n" in text or "\r" in text:
        raise ValueError("Invalid linguistic line layout")
    phrases, words = boundaries(text)
    if len(text) <= width:
        return text
    if len(text) > 2 * width:
        raise CaptionLayoutError("Caption exceeds two-line display budget")
    for candidates in (phrases, words):
        allowed = [
            point
            for point in candidates
            if 0 < point < len(text) and max(point, len(text) - point) <= width
        ]
        if allowed:
            split = min(allowed, key=lambda point: (abs(len(text) - 2 * point), point))
            return text[:split] + "\n" + text[split:]
    raise CaptionLayoutError("No intact morpheme fits the caption line budget")


def wrap_measured_lines(
    text: str,
    *,
    measure: Callable[[str], tuple[float, float]],
    width: float,
    height: float,
    prepare_lines: Callable[[tuple[str, ...]], None] | None = None,
    maximum_lines: int = 2,
) -> str:
    """Fit up to three measured lines at phrase boundaries, without scaling text.

    The caller supplies actual renderer bounds at the chosen font size, including
    outlines, line spacing and animation clearance. Invalid measurements fail;
    an indivisible phrase that cannot fit needs an earlier timed caption switch.
    """
    if (
        type(maximum_lines) is not int
        or not 1 <= maximum_lines <= 3
        or not all(isfinite(value) and value > 0 for value in (width, height))
        or "\n" in text
        or "\r" in text
    ):
        raise ValueError("Invalid measured caption region")
    phrases, _ = boundaries(text)

    def bounds(value: str) -> tuple[float, float]:
        measured_width, measured_height = measure(value)
        if not (
            isfinite(measured_width)
            and measured_width >= 0
            and isfinite(measured_height)
            and measured_height > 0
        ):
            raise ValueError("Invalid renderer text measurement")
        return measured_width, measured_height

    first_width, first_height = bounds(text)
    if first_width <= width and first_height <= height:
        return text
    # A coarse ASR token can contain an entire polite expression. Its internal
    # morphology may support simultaneous line wrapping, but never new times.
    # Keep the auxiliary chain together instead of exposing lone particles.
    items = tuple(_tokenizer().tokenize(text))
    polite_ends = {
        left.end()
        for left, right in pairwise(items)
        if left.feature().split(",")[0] == "感動詞" and right.feature().split(",")[0] == "助動詞"
    }
    protected_ends = _rhythmic_ends(text, frozenset(set(phrases) | polite_ends))
    fitting_runs = tuple(
        match.span()
        for match in regex.finditer(
            r"[\p{Script=Katakana}ー]+|[\p{Script=Latin}\p{Number}_]+", text
        )
        if bounds(match.group())[0] <= width
    )
    points = tuple(
        sorted(
            point
            for point in protected_ends
            if 0 < point < len(text)
            and text[point] not in PROHIBITED_LINE_START
            and text[point - 1] not in PROHIBITED_LINE_END
            and not any(first < point < last for first, last in fitting_runs)
        )
    )
    for line_count in range(2, maximum_lines + 1):
        splits = tuple(islice(combinations(points, line_count - 1), MAX_LAYOUT_CANDIDATES + 1))
        if len(splits) > MAX_LAYOUT_CANDIDATES:
            raise CaptionLayoutError("Measured phrase layout candidate budget exceeded")
        layouts = [
            tuple(text[a:b] for a, b in pairwise((0, *split, len(text)))) for split in splits
        ]
        if prepare_lines is not None:
            prepare_lines(tuple(dict.fromkeys(part for layout in layouts for part in layout)))
        candidates: list[tuple[float, int, str]] = []
        for index, lines in enumerate(layouts):
            candidate = "\n".join(lines)
            measured_width, measured_height = bounds(candidate)
            if measured_width <= width and measured_height <= height:
                widths = [bounds(line)[0] for line in lines]
                candidates.append((max(widths) - min(widths), index, candidate))
        if candidates:
            return min(candidates)[2]
    raise CaptionLayoutError("No phrase layout fits measured caption region")


def wrap_split_word(
    text: str,
    *,
    measure: Callable[[str], tuple[float, float]],
    width: float,
    height: float,
    prepare_lines: Callable[[tuple[str, ...]], None] | None = None,
) -> str:
    """Two lines for a single line too wide to wrap at any phrase: the last resort before scaling.

    A long unknown katakana name or word has no phrase boundary, and its morphology is unreliable.
    The break nearest the middle is taken, preferring a morpheme end, then any grapheme end; no
    line starts with a dependent kana or closing mark, and none ends with an opening mark.
    """
    if "\n" in text or "\r" in text:
        raise ValueError("Invalid measured caption region")
    middle = len(text) / 2
    graphemes = {match.end() for match in regex.finditer(r"\X", text)}
    legal = [
        point
        for point in sorted(graphemes - {len(text)}, key=lambda point: (abs(point - middle), point))
        if text[point] not in PROHIBITED_LINE_START and text[point - 1] not in PROHIBITED_LINE_END
    ]
    morphemes = {item.end() for item in _tokenizer().tokenize(text)}
    ordered = [point for point in legal if point in morphemes]
    ordered += [point for point in legal if point not in morphemes]
    layouts = [(text[:point], text[point:]) for point in ordered]
    if prepare_lines is not None:
        prepare_lines(tuple(dict.fromkeys(line for layout in layouts for line in layout)))
    for first, second in layouts:
        candidate = f"{first}\n{second}"
        measured_width, measured_height = measure(candidate)
        if measured_width <= width and measured_height <= height:
            return candidate
    raise CaptionLayoutError("No split of the word fits measured caption region")


def attached_tokens(words: list[SpokenToken], *, maximum_gap: float) -> list[bool]:
    """Mark native-token starts inside a phrase, without merging or retiming tokens.

    Ownership changes and real gaps are hard boundaries. A morphological boundary
    inside a coarse ASR token is not assigned an invented recognition timestamp.
    """
    attached = [False] * len(words)
    start = 0
    while start < len(words):
        end = start + 1
        while (
            end < len(words)
            and words[end].utterance == words[start].utterance
            and words[end].start - words[end - 1].end <= maximum_gap
        ):
            end += 1
        group = words[start:end]
        phrases, _ = boundaries("".join(item.text for item in group))
        breaks = frozenset(phrases)
        offsets = tuple(accumulate(len(item.text) for item in group))
        attached[start + 1 : end] = [offset not in breaks for offset in offsets[:-1]]
        start = end
    return attached


@lru_cache(maxsize=256)
def _atomic_ends(text: str) -> frozenset[int]:
    """Lexical ends exclude small kana, honorific suffixes and detached prefixes."""
    _, ends = boundaries(text)
    allowed = set(ends)
    previous_prefix = False
    for item in _tokenizer().tokenize(text):
        pos = item.feature().split(",")
        if item.start() and (
            previous_prefix or pos[1] == "接尾" or item.surface()[0] in DEPENDENT_KANA
        ):
            allowed.discard(item.start())
        previous_prefix = pos[0] == "接頭詞" or pos[1] == "括弧開"
    return frozenset(allowed)


def _rhythmic_ends(text: str, ends: frozenset[int]) -> frozenset[int]:
    """Preserve repeated phonetic units before incidental POS fragments fit.

    A trailing partial repetition stays with the final complete unit. Only
    literal repetitions are used; rhythm or missing syllables are not invented.
    """
    result = set(ends)
    for match in regex.finditer(r"[\p{Script=Katakana}ーﾞﾟ]+", text):
        run = match.group()
        for size in range(2, len(run) // 2 + 1):
            unit = run[:size]
            if (
                unit[0] in DEPENDENT_KANA
                or sum(char not in DEPENDENT_KANA for char in unit) < 2
                or (unit * (len(run) // size + 1))[: len(run)] != run
            ):
                continue
            result.difference_update(point for point in ends if match.start() < point < match.end())
            result.update(match.start() + size * n for n in range(1, len(run) // size))
            result.add(match.end())
            break
    return frozenset(result)


def _overflow_ends(text: str) -> frozenset[int]:
    """Allow dictionary-backed compound edges only when a protected unit cannot fit."""
    items = tuple(_tokenizer().tokenize(text))
    ends = set(_atomic_ends(text))
    graphemes = {match.end() for match in regex.finditer(r"\X", text)}
    # Honorifics belong to the immediately preceding unknown name. A complete
    # dictionary compound before that name does not inherit this protection.
    for index, item in enumerate(items):
        features = item.feature().split(",")
        if features[1] != "接尾":
            continue
        first = index
        while first and len(items[first - 1].feature().split(",")) < 9:
            first -= 1
        if first == index or first == 0:
            continue
        previous = items[first - 1]
        preceding = previous.feature().split(",")
        name = text[items[first].start() : item.start()]
        if (
            len(previous.surface()) >= 2
            and preceding[0] == "名詞"
            and preceding[1] != "接尾"
            and sum(char not in DEPENDENT_KANA for char in name) >= 2
            and name[0] not in DEPENDENT_KANA
            and previous.end() in graphemes
        ):
            ends.add(previous.end())
    # A repeated phonetic unit is stronger evidence than incidental dictionary
    # fragments inside it. Keep each whole repeated unit (including small kana
    # and long vowels), with only a shorter intervening phonetic phrase.
    if regex.fullmatch(r"[\p{Script=Hiragana}\p{Script=Katakana}ーﾞﾟ]+", text):
        for size in range(len(text) // 2, 1, -1):
            if (
                text[:size] == text[-size:]
                and size in graphemes
                and len(text) - size in graphemes
                and sum(char not in DEPENDENT_KANA for char in text[:size]) >= 2
                and text[size] not in DEPENDENT_KANA
                and text[-size] not in DEPENDENT_KANA
                and len(text) - 2 * size <= size
            ):
                return frozenset({size, len(text) - size, len(text)})
    for item in items:
        features = item.feature().split(",")
        prefix = text[: item.start()].lstrip()
        if (
            len(prefix) >= 2
            and len(item.surface()) >= 2
            and len(features) >= 9
            and features[0] not in DEPENDENT_PARTS | {"接頭詞", "記号"}
            and features[1] != "接尾"
            and text[item.end() :].startswith(prefix)
            and prefix[0] not in DEPENDENT_KANA
            and item.end() in graphemes
        ):
            ends.add(item.end())
    for left, right in pairwise(items):
        before, after = left.feature().split(","), right.feature().split(",")
        if (
            len(before) >= 9
            and len(after) >= 9
            and before[0] != "接頭詞"
            and before[1] != "括弧開"
            and after[0] not in DEPENDENT_PARTS
            and after[1] != "接尾"
            and right.surface()[0] not in DEPENDENT_KANA
            and left.end() in graphemes
        ):
            ends.add(left.end())
    # Unknown-word grouping can swallow an entire repeated compound. Recover
    # its maximal repeated unit only when a separate dictionary-backed word
    # lies between the copies. The unknown unit itself remains indivisible.
    for size in range((len(text) - 1) // 2, 1, -1):
        if (
            size not in graphemes
            or len(text) - size not in graphemes
            or text[:size] != text[-size:]
            or text[size] in DEPENDENT_KANA
            or text[-size] in DEPENDENT_KANA
        ):
            continue
        middle = tuple(_tokenizer().tokenize(text[size:-size]))
        if middle and all(
            len(item.feature().split(",")) >= 9
            and item.feature().split(",")[0] not in DEPENDENT_PARTS | {"接頭詞", "記号"}
            and item.feature().split(",")[1] != "接尾"
            for item in middle
        ):
            ends.update((size, len(text) - size))
            break
    if ends == {len(text)}:
        for left, right in pairwise(tuple(_bounded_tokenizer().tokenize(text))):
            before, after = left.feature().split(","), right.feature().split(",")
            if (
                len(left.surface()) >= 2
                and len(right.surface()) >= 2
                and len(before) >= 9
                and len(after) >= 9
                and before[0] not in DEPENDENT_PARTS | {"接頭詞", "記号"}
                and after[0] not in DEPENDENT_PARTS | {"接頭詞", "記号"}
                and before[1] != "接尾"
                and after[1] != "接尾"
                and right.surface()[0] not in DEPENDENT_KANA
                and left.end() in graphemes
            ):
                ends.add(left.end())
    return frozenset(ends)


def atomic_tokens(
    words: list[SpokenToken],
    *,
    maximum_gap: float,
    complete_phrases: bool = False,
    accept_phrase: Callable[[SpokenToken], bool] | None = None,
) -> list[SpokenToken]:
    """Reveal an intact lexical unit at its first recognized fragment onset.

    Original recognition records remain unchanged. A coarse token is never split
    at invented times; ownership changes and real gaps remain hard boundaries.
    The complete word appears when that word starts, not at an invented character
    clock. Subsequent lexical units retain their own recognized onsets.
    """
    result: list[SpokenToken] = []
    first = 0
    while first < len(words):
        end = first + 1
        while (
            end < len(words)
            and recognition_continuity(words[end].utterance)
            == recognition_continuity(words[first].utterance)
            and words[end].start - words[end - 1].end <= maximum_gap
        ):
            end += 1
        result.extend(
            _atomic_group(
                words[first:end], complete_phrases=complete_phrases, accept_phrase=accept_phrase
            )
        )
        first = end
    return result


def _recognition_overflow(
    words: list[SpokenToken], accept: Callable[[SpokenToken], bool]
) -> list[SpokenToken] | None:
    """Estimate an unknown loanword compound break from observed timing.

    Dictionary words remain intact. Unknown spans are not claimed to be known
    words: only existing ASR edges are considered, both halves must fit measured
    bounds, and balanced speech spans avoid short trailing fragments. This is a
    fallback for mixed known/unknown katakana, not a lexical guarantee.
    """
    text = "".join(word.text for word in words)
    # ASR word records can retain surrounding whitespace. It has no lexical
    # content, but remains in the original display text and observed offsets.
    if not regex.fullmatch(r"[\p{Script=Katakana}ーﾞﾟ]+[!！?？。、]*", text.strip()):
        return None
    items = tuple(_tokenizer().tokenize(text))
    lexical = tuple(item for item in items if not item.surface().isspace())
    known = tuple(item for item in lexical if len(item.feature().split(",")) >= 9)
    if len(known) == len(lexical) or not any(len(item.surface()) >= 2 for item in known):
        return None
    graphemes = {match.end() for match in regex.finditer(r"\X", text)}
    candidates: list[tuple[float, int, SpokenToken, SpokenToken]] = []
    offset = 0
    for index, word in enumerate(words[:-1], start=1):
        offset += len(word.text)
        if (
            offset not in graphemes
            or text[offset] in DEPENDENT_KANA
            or any(item.start() < offset < item.end() for item in known)
            or min(
                sum(char not in DEPENDENT_KANA and not char.isspace() for char in text[:offset]),
                sum(
                    char not in DEPENDENT_KANA and not char.isspace()
                    for char in text[offset:].rstrip("!！?？。、")
                ),
            )
            < 2
        ):
            continue
        left = SpokenToken(
            text[:offset], words[0].start, max(w.end for w in words[:index]), words[0].utterance
        )
        right = SpokenToken(
            text[offset:],
            words[index].start,
            max(w.end for w in words[index:]),
            words[index].utterance,
        )
        if accept(left) and accept(right):
            candidates.append(
                (abs((left.end - left.start) - (right.end - right.start)), index, left, right)
            )
    if not candidates:
        return None
    chosen = min(candidates, key=lambda item: (item[0], item[1]))
    return [chosen[2], chosen[3]]


def _unit(words: Sequence[SpokenToken]) -> SpokenToken:
    return SpokenToken(
        "".join(word.text for word in words),
        words[0].start,
        max(word.end for word in words),
        recognition_continuity(words[-1].utterance),
    )


def edge_runs(
    words: Sequence[SpokenToken], accept: Callable[[SpokenToken], bool]
) -> list[SpokenToken]:
    """Split words that no lexical edge could fit at observed ASR word edges, longest run first.

    Only recognized edges are used (no invented times), never before a dependent kana, so
    "ディ" stays whole. A run that cannot shrink further is kept for the budget check to refuse.
    """
    runs: list[SpokenToken] = []
    start = 0
    while start < len(words):
        allowed = [
            end
            for end in range(start + 1, len(words) + 1)
            if end == len(words) or words[end].text.lstrip()[:1] not in DEPENDENT_KANA
        ]
        fitting = [end for end in allowed if accept(_unit(words[start:end]))]
        end = max(fitting) if fitting else allowed[0]
        runs.append(_unit(words[start:end]))
        start = end
    return runs


def _atomic_group(
    words: list[SpokenToken],
    *,
    complete_phrases: bool,
    accept_phrase: Callable[[SpokenToken], bool] | None,
    overflow: bool = False,
) -> list[SpokenToken]:
    text = "".join(word.text for word in words)
    if overflow and accept_phrase is not None:
        whole = SpokenToken(
            text, words[0].start, max(word.end for word in words), words[0].utterance
        )
        if not accept_phrase(whole):
            recovered = _recognition_overflow(words, accept_phrase)
            if recovered is not None:
                return recovered
    ends = (
        frozenset(boundaries(text)[0])
        if complete_phrases
        else _overflow_ends(text)
        if overflow
        else _atomic_ends(text)
    )
    ends = _rhythmic_ends(text, ends)
    result: list[SpokenToken] = []
    pending: list[SpokenToken] = []
    offset = 0
    for index, word in enumerate(words):
        pending.append(word)
        offset += len(word.text)
        if offset in ends or index == len(words) - 1:
            unit = SpokenToken(
                "".join(item.text for item in pending),
                pending[0].start,
                max(item.end for item in pending),
                recognition_continuity(word.utterance),
            )
            if accept_phrase is not None and not accept_phrase(unit) and not overflow:
                result.extend(
                    _atomic_group(
                        pending,
                        complete_phrases=False,
                        accept_phrase=accept_phrase,
                        overflow=not complete_phrases,
                    )
                )
            else:
                result.append(unit)
            pending = []
    return result


@lru_cache(maxsize=256)
def semantic_break(previous: str, following: str) -> bool:
    """Start a fresh page for an independent interjection or completed clause."""
    boundaries(previous)
    boundaries(following)
    before = tuple(_tokenizer().tokenize(previous))[-1].feature().split(",")
    after = _tokenizer().tokenize(following)[0].feature().split(",")
    return after[0] == "感動詞" or before[1] == "終助詞"


def main() -> None:
    """Inspect pinned morphology and protected boundaries without media processing."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("text")
    parser.add_argument("--max-grouping-length", type=int, default=1, choices=range(81))
    args = parser.parse_args()
    phrases, morphemes = boundaries(args.text)
    print(
        json.dumps(
            {
                "phrases": phrases,
                "morphemes": morphemes,
                "atomic": sorted(_atomic_ends(args.text)),
                "overflow": sorted(_overflow_ends(args.text)),
                "tokens": [
                    {
                        "text": item.surface(),
                        "start": item.start(),
                        "end": item.end(),
                        "features": item.feature().split(","),
                    }
                    for item in (
                        _load_tokenizer(max_grouping_len=args.max_grouping_length)
                        if args.max_grouping_length != 1
                        else _tokenizer()
                    ).tokenize(args.text)
                ],
            },
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()
