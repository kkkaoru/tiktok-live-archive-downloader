"""Reading-matched word-list corrections of recognition labels, with protected terms.

A glossary is configuration, not code: JSON ``{"version": 1, "replacements": [{"reading":
"どうせつ", "word": "同接"}], "protected": ["名前"]}``. Readings follow the よみ/単語 user-lexicon
convention (NFKC, katakana folded to hiragana, at least two characters). Several files merge, a
shared one first and the channel's own after it.

A replacement applies to a run of whole IPADIC morphemes (pinned Vibrato) whose reading equals the
entry's and whose surface differs and contains kanji, so a wrong homophone is corrected while a
kana spelling, a mid-word match or a different word is left alone. A run overlapping a protected
term is never edited. Recognition never sees the glossary: the audio decided the reading, and only
the spelling of that reading changes.

Edits are made on each chunk's label text (labels grouped by ``provenance``) and keep the original
span's timing: a same-length edit rewrites characters in place; any other edit merges the labels
it touches into one label from the first label's start to the last label's end.
"""

from __future__ import annotations

import hashlib
import json
import unicodedata
from collections.abc import Iterable, Iterator, Mapping, Sequence
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path

import regex
from vibrato import Vibrato

from replay import caption_linguistics as linguistics
from replay.core import ReplayError

MIN_READING = 2
KANJI = regex.compile(r"[\p{Han}々]")
KANA = regex.compile(r"^[\p{Hiragana}\p{Katakana}ー]+$")
READING_FIELD = 7  # IPADIC: pos x4, conjugation x2, base, reading, pronunciation


@dataclass(frozen=True, slots=True)
class Replacement:
    reading: str
    word: str
    # Curated misspellings that read as ``reading`` but that IPADIC reads otherwise when the word is
    # split (同節: 同+節 is どう+ふし there); matched by surface over whole morphemes.
    variants: tuple[str, ...] = ()


@dataclass(frozen=True, slots=True)
class Glossary:
    replacements: tuple[Replacement, ...] = ()
    protected: tuple[str, ...] = ()

    def guarded(self) -> tuple[str, ...]:
        """Terms no edit may touch: the protected list and every replacement's target word."""
        return (*self.protected, *(item.word for item in self.replacements))


@dataclass(frozen=True, slots=True)
class Morpheme:
    start: int
    end: int
    surface: str
    reading: str | None  # hiragana; None for an unknown word that is not kana
    proper: bool
    known: bool  # in the dictionary (an unknown kana word still has its written reading)


@dataclass(frozen=True, slots=True)
class Edit:
    chunk: str
    start: int  # character offsets in the chunk's label text
    end: int
    before: str
    after: str
    source: str
    detail: Mapping[str, object] = field(default_factory=dict)


def hiragana(text: str) -> str:
    """NFKC, katakana folded to hiragana (the user-lexicon reading form)."""
    return "".join(
        chr(ord(character) - 0x60) if "ァ" <= character <= "ヶ" else character
        for character in unicodedata.normalize("NFKC", text)
    )


def _strings(value: object, name: str) -> list[str]:
    if not isinstance(value, list) or not all(isinstance(item, str) for item in value):
        raise ReplayError(f"Glossary {name} must be a list of strings")
    return [item for item in value if isinstance(item, str)]


def _replacement(value: object) -> Replacement:
    if not isinstance(value, dict) or not {"reading", "word"} <= set(value) <= {
        "reading",
        "word",
        "variants",
    }:
        raise ReplayError("Each glossary replacement needs a reading and a word (and variants)")
    reading, word = value["reading"], value["word"]
    if not isinstance(reading, str) or not isinstance(word, str):
        raise ReplayError("Glossary readings and words must be strings")
    reading, word = hiragana(reading.strip()), word.strip()
    if len(reading) < MIN_READING or not regex.fullmatch(r"[\p{Hiragana}ー]+", reading):
        raise ReplayError(f"Glossary reading {reading!r} must be at least two kana")
    if not word or regex.search(r"\s", word):
        raise ReplayError(f"Glossary word for {reading!r} must be one nonempty word")
    variants = tuple(item.strip() for item in _strings(value.get("variants", []), "variants"))
    if any(not KANJI.search(item) or item == word for item in variants):
        raise ReplayError(f"Glossary variants of {word!r} must be other spellings with kanji")
    return Replacement(reading, word, variants)


def parse(document: object) -> Glossary:
    """One validated glossary document."""
    if not isinstance(document, dict) or document.get("version") != 1:
        raise ReplayError("Glossary must be a version 1 JSON object")
    if not set(document) <= {"version", "replacements", "protected", "note"}:
        raise ReplayError("Glossary has unknown keys")
    replacements = document.get("replacements", [])
    if not isinstance(replacements, list):
        raise ReplayError("Glossary replacements must be a list")
    protected = [term.strip() for term in _strings(document.get("protected", []), "protected")]
    if any(len(term) < MIN_READING for term in protected):
        raise ReplayError("Protected glossary terms must be at least two characters")
    return Glossary(tuple(_replacement(item) for item in replacements), tuple(protected))


def merged(glossaries: Iterable[Glossary]) -> Glossary:
    """Later files win a reading; protected terms accumulate."""
    by_reading: dict[str, Replacement] = {}
    protected: dict[str, None] = {}
    for glossary in glossaries:
        by_reading.update({item.reading: item for item in glossary.replacements})
        protected.update(dict.fromkeys(glossary.protected))
    return Glossary(tuple(by_reading.values()), tuple(protected))


def load(paths: Sequence[Path]) -> Glossary:
    return merged(parse(json.loads(path.read_text(encoding="utf-8"))) for path in paths)


def morphemes(text: str, tokenizer: Vibrato | None = None) -> list[Morpheme]:
    """IPADIC morphemes with hiragana readings; kana unknown words read as written."""
    result = []
    for token in (tokenizer or linguistics._tokenizer()).tokenize(text):
        surface, features = token.surface(), token.feature().split(",")
        known = len(features) > READING_FIELD and features[READING_FIELD] != "*"
        reading = (
            hiragana(features[READING_FIELD])
            if known
            else hiragana(surface)
            if KANA.match(surface)
            else None
        )
        proper = len(features) > 1 and features[1] == "固有名詞"
        result.append(Morpheme(token.start(), token.end(), surface, reading, proper, known))
    return result


class Trie[Value]:
    """A character trie: walking it costs the length of the walk, whatever the number of keys."""

    def __init__(self, items: Iterable[tuple[str, Value]]) -> None:
        self.children: list[dict[str, int]] = [{}]
        self.values: dict[int, Value] = {}
        for key, value in items:
            node = 0
            for character in key:
                child = self.children[node].get(character)
                if child is None:
                    child = len(self.children)
                    self.children[node][character] = child
                    self.children.append({})
                node = child
            self.values.setdefault(node, value)  # the first entry of a key wins, as in merged()

    def step(self, node: int, text: str) -> int | None:
        for character in text:
            child = self.children[node].get(character)
            if child is None:
                return None
            node = child
        return node

    def ends(self, text: str, start: int) -> Iterator[int]:
        """End offsets of every key that starts at ``start``."""
        node = 0
        for end in range(start, len(text)):
            child = self.children[node].get(text[end])
            if child is None:
                return
            node = child
            if node in self.values:
                yield end + 1


def overlaps(start: int, end: int, spans: Iterable[tuple[int, int]]) -> bool:
    return any(start < right and left < end for left, right in spans)


class Matcher:
    """A glossary compiled once: tries over readings, curated variants and guarded surfaces, so
    one pass over a text costs its length times the longest match, not the list's size."""

    def __init__(self, glossary: Glossary) -> None:
        self.glossary = glossary
        self.readings = Trie((item.reading, item) for item in glossary.replacements)
        self.variants = Trie(
            (variant, item) for item in glossary.replacements for variant in item.variants
        )
        self.guards = Trie((term, term) for term in glossary.guarded())

    def guarded(self, text: str) -> list[tuple[int, int]]:
        """Character spans of every guarded term in the text."""
        return [(start, end) for start in range(len(text)) for end in self.guards.ends(text, start)]

    def _longest(
        self, tokens: Sequence[Morpheme], index: int
    ) -> tuple[int, Replacement, str] | None:
        """The longest reading or variant match over whole morphemes from ``index``."""
        match: tuple[int, Replacement, str] | None = None
        reading: int | None = 0
        variant: int | None = 0
        for last in range(index, len(tokens)):
            token = tokens[last]
            reading = (
                None
                if reading is None or token.reading is None
                else self.readings.step(reading, token.reading)
            )
            variant = None if variant is None else self.variants.step(variant, token.surface)
            if reading is None and variant is None:
                break
            if reading is not None and reading in self.readings.values:
                match = (last, self.readings.values[reading], "reading")
            elif variant is not None and variant in self.variants.values:
                match = (last, self.variants.values[variant], "variant")
        return match

    def find(self, chunk: str, text: str, tokenizer: Vibrato | None = None) -> list[Edit]:
        """Leftmost-longest matches over whole morphemes, never touching a guarded term."""
        if not self.glossary.replacements:
            return []
        guarded = self.guarded(text)
        tokens = morphemes(text, tokenizer)
        edits, index = [], 0
        while index < len(tokens):
            match = self._longest(tokens, index)
            if match is None:
                index += 1
                continue
            last, item, rule = match
            start, end = tokens[index].start, tokens[last].end
            surface = text[start:end]
            if surface != item.word and KANJI.search(surface) and not overlaps(start, end, guarded):
                edits.append(
                    Edit(
                        chunk,
                        start,
                        end,
                        surface,
                        item.word,
                        "glossary",
                        {"reading": item.reading, "rule": rule},
                    )
                )
            index = last + 1
        return edits


@lru_cache(maxsize=4)
def _compiled(digests: tuple[tuple[str, str], ...]) -> Matcher:
    return Matcher(
        merged(
            parse(json.loads(Path(path).read_text(encoding="utf-8"))) for path, _digest in digests
        )
    )


def compiled(paths: Sequence[Path]) -> Matcher:
    """The files' matcher, built once per process and reused while their bytes are unchanged."""
    return _compiled(
        tuple((str(path), hashlib.sha256(path.read_bytes()).hexdigest()) for path in paths)
    )


def find(chunk: str, text: str, glossary: Glossary, tokenizer: Vibrato | None = None) -> list[Edit]:
    return Matcher(glossary).find(chunk, text, tokenizer)


def chunks(rows: Sequence[Mapping[str, object]]) -> dict[str, list[int]]:
    """Label indices of each recognition chunk (the ``provenance`` prefix), in label order."""
    groups: dict[str, list[int]] = {}
    for index, row in enumerate(rows):
        provenance = row.get("provenance")
        if not isinstance(provenance, str) or ":" not in provenance:
            raise ReplayError("Every label needs a chunk:index provenance")
        groups.setdefault(provenance.split(":", 1)[0], []).append(index)
    return groups


def chunk_text(rows: Sequence[Mapping[str, object]], indices: Sequence[int]) -> str:
    return "".join(str(rows[index]["text"]) for index in indices)


def _rewritten(
    rows: Sequence[Mapping[str, object]], indices: Sequence[int], edits: Sequence[Edit]
) -> list[dict[str, object]]:
    """One chunk's labels with its non-overlapping edits applied."""
    labels = [dict(rows[index]) for index in indices]
    bounds, offset = [], 0
    for label in labels:
        bounds.append((offset, offset + len(str(label["text"]))))
        offset += len(str(label["text"]))
    text = "".join(str(label["text"]) for label in labels)
    merged_into: dict[int, int] = {}
    for edit in sorted(edits, key=lambda item: item.start, reverse=True):
        if text[edit.start : edit.end] != edit.before:
            raise ReplayError("An edit does not match the label text it was planned on")
        touched = [
            at for at, (left, right) in enumerate(bounds) if edit.start < right and left < edit.end
        ]
        if len(edit.after) == len(edit.before):
            for position, character in enumerate(edit.after, edit.start):
                at = next(at for at in touched if bounds[at][0] <= position < bounds[at][1])
                relative = position - bounds[at][0]
                old = str(labels[at]["text"])
                labels[at]["text"] = old[:relative] + character + old[relative + 1 :]
            continue
        first, last = touched[0], touched[-1]
        # Edits run right to left, so text before this edit is still the planned text; a label
        # a later edit already rewrote keeps its rewritten remainder.
        head = str(labels[first]["text"])[: edit.start - bounds[first][0]]
        tail = str(labels[last]["text"])[edit.end - bounds[last][0] :]
        labels[first]["text"] = head + edit.after + tail
        labels[first]["end"] = labels[last]["end"]
        for at in touched[1:]:
            merged_into[at] = first
    return [label for at, label in enumerate(labels) if at not in merged_into]


def apply(rows: Sequence[Mapping[str, object]], edits: Sequence[Edit]) -> list[dict[str, object]]:
    """All labels with the edits applied; edits must not overlap within a chunk."""
    groups = chunks(rows)
    by_chunk: dict[str, list[Edit]] = {}
    for edit in edits:
        if edit.chunk not in groups:
            raise ReplayError("An edit names a chunk without labels")
        by_chunk.setdefault(edit.chunk, []).append(edit)
    for planned in by_chunk.values():
        ordered = sorted(planned, key=lambda item: item.start)
        if any(left.end > right.start for left, right in zip(ordered, ordered[1:], strict=False)):
            raise ReplayError("Edits overlap within a chunk")
    rewritten = {
        name: _rewritten(rows, indices, by_chunk[name])
        for name, indices in groups.items()
        if name in by_chunk
    }
    result: list[dict[str, object]] = []
    emitted: set[str] = set()
    for row in rows:
        name = str(row["provenance"]).split(":", 1)[0]
        if name not in rewritten:
            result.append(dict(row))
        elif name not in emitted:
            emitted.add(name)
            result.extend(rewritten[name])
    return result


def plan(rows: Sequence[Mapping[str, object]], terms: Glossary | Matcher) -> list[Edit]:
    """Glossary edits for every chunk of the labels."""
    matcher = terms if isinstance(terms, Matcher) else Matcher(terms)
    tokenizer = linguistics._tokenizer() if matcher.glossary.replacements else None
    return [
        edit
        for name, indices in chunks(rows).items()
        for edit in matcher.find(name, chunk_text(rows, indices), tokenizer)
    ]
