"""Part-of-speech hints for the web viewer's clip points (切り抜き).

The Worker that chooses clip points cannot run a Japanese morphological analyzer, so the
web sync adds two hints per cue to every ``data/<id>.json`` with the pinned Vibrato
dictionary: ``terms``, the cue's content nouns (base forms), which tell where the topic
changes; ``opens``, whether the cue can start a range, i.e. it does not open with a
particle, auxiliary or dependent word that continues the previous cue's sentence; and
``closes``, whether a clip can end on it, i.e. it does not end on a case, conjunctive or
binding particle (を・に・て・けど・は...) that leaves the sentence hanging.
"""

import argparse
import json
import sys
from collections.abc import Iterable, Sequence
from pathlib import Path
from typing import Protocol

from replay import caption_linguistics as linguistics
from replay.excitement import _write_atomic


class Token(Protocol):
    def surface(self) -> str: ...
    def feature(self) -> str: ...


class Tokenizer(Protocol):
    def tokenize(self, text: str) -> Iterable[Token]: ...


NOUN: str = "名詞"
CONTENT_NOUNS: frozenset[str] = frozenset({"一般", "固有名詞", "サ変接続", "形容動詞語幹"})
SKIPPED: frozenset[str] = frozenset({"記号", "フィラー"})
CONTINUING: frozenset[str] = frozenset({"助詞", "助動詞"})
DEPENDENT: frozenset[str] = frozenset({"非自立", "接尾"})
PARTICLE: str = "助詞"
HANGING: frozenset[str] = frozenset({"格助詞", "接続助詞", "係助詞"})
LEMMA_FIELD: int = 6
TERM_MIN_CHARACTERS: int = 2
CATALOG: str = "catalog.json"


def _lemma(surface: str, features: Sequence[str]) -> str:
    lemma = features[LEMMA_FIELD] if len(features) > LEMMA_FIELD else "*"
    return surface if lemma == "*" else lemma


def _is_term(lemma: str, features: Sequence[str]) -> bool:
    return (
        features[0] == NOUN
        and len(features) > 1
        and features[1] in CONTENT_NOUNS
        and len(lemma) >= TERM_MIN_CHARACTERS
        and not lemma.isdigit()
    )


def _continues(features: Sequence[str]) -> bool:
    return features[0] in CONTINUING or (len(features) > 1 and features[1] in DEPENDENT)


def _hangs(features: Sequence[str]) -> bool:
    return features[0] == PARTICLE and len(features) > 1 and features[1] in HANGING


def hints(text: str, tokenizer: Tokenizer) -> tuple[list[str], bool, bool]:
    """The cue's distinct content nouns in order, and whether a clip can start and end on it."""
    # Captions wrap lines inside words; the line breaks are layout, not text.
    tokens = [
        (t.surface(), t.feature().split(",")) for t in tokenizer.tokenize(text.replace("\n", ""))
    ]
    words = [(surface, features) for surface, features in tokens if features[0] not in SKIPPED]
    lemmas = [_lemma(surface, features) for surface, features in words]
    terms = [
        lemma
        for lemma, (_, features) in zip(lemmas, words, strict=True)
        if _is_term(lemma, features)
    ]
    spoken = [features for surface, features in words if surface.strip()]
    first = spoken[0] if spoken else None
    last = spoken[-1] if spoken else None
    return (
        list(dict.fromkeys(terms)),
        first is None or not _continues(first),
        last is None or not _hangs(last),
    )


def annotate(path: Path, tokenizer: Tokenizer) -> bool:
    """Adds the hints to one data file; False when it has no cues to annotate."""
    data = json.loads(path.read_text(encoding="utf-8"))
    cues = data.get("cues") if isinstance(data, dict) else None
    if not isinstance(cues, list):
        return False
    for cue in cues:
        cue["terms"], cue["opens"], cue["closes"] = hints(str(cue.get("text", "")), tokenizer)
    _write_atomic(path, json.dumps(data, ensure_ascii=False, separators=(",", ":")))
    return True


def annotate_directory(directory: Path, tokenizer: Tokenizer) -> int:
    """Annotates every data file of a built web catalog directory; returns how many."""
    files = sorted(p for p in directory.glob("*.json") if p.name != CATALOG)
    return sum(annotate(path, tokenizer) for path in files)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="python -m replay.clip_terms", description=__doc__)
    parser.add_argument("directory", type=Path, help="web catalog directory (data/<id>.json)")
    args = parser.parse_args(argv)
    count = annotate_directory(args.directory, linguistics._tokenizer())
    print(f"annotated {count} data files", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
