"""Correct wrong same-reading kanji in recognition labels: a glossary first, then zenz rescoring.

1. ``replay.asr_glossary``: the channel's reading-matched replacements, never on protected terms.
2. Optionally, zenz re-selection of Whisper's own close alternatives (evaluated 2026-10-24 on three
   recordings: 10 of 14 applied changes improved, 1 worse; ``edits/work/kanji-fix-20261024``):
   ``replay.asr_whisper_alternatives`` replays large-v3 on the stored chunk records; a candidate is
   kept only when its whole-segment reading (pinned IPADIC) equals the original's, it swaps kanji
   for other kanji, and it touches no protected morpheme (unknown, proper noun or katakana) and no
   glossary term. zenz scores both spellings on a phrase window with the preceding text as left
   context; the change is made when ``Δzenz + LAMBDA·Δwhisper > TAU``, at most once per segment.

The audio decided the reading in both cases; only its spelling changes, so each edit keeps the
original span's timing. Every edit and every refusal is written to a receipt beside the labels.
"""

from __future__ import annotations

import argparse
import difflib
import hashlib
import json
import os
import sys
import time
import unicodedata
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path

import regex
from vibrato import Vibrato

from replay import asr_glossary as glossary
from replay import asr_whisper_alternatives as alternatives
from replay import caption_linguistics as linguistics
from replay import zenz_lm
from replay.core import ReplayError

LAMBDA, TAU = 1.0, 2.0  # fixed before the judged evaluation; a change needs ~7x combined odds
WINDOW = 12  # characters kept on each side of a candidate edit
CONTEXT_SEGMENTS, CONTEXT_CHARACTERS = 6, 40
JAPANESE = regex.compile(r"[ぁ-ゖァ-ヺー々一-鿿]")
KANJI = regex.compile(r"[々一-鿿]")
KATAKANA_ONLY = regex.compile(r"^[ァ-ヺー]+$")
KANA_ONLY = regex.compile(r"^[ぁ-ゖァ-ヺー]+$")
PUNCTUATION = regex.compile(r"[\s、。,.!?！？・「」『』\"'ー〜~…×]")
Guard = Callable[[str], list[tuple[int, int]]]  # spans of protected terms in a text
RANK = {"none": 0, "punct": 1, "script": 2, "orthography": 3, "kanji-homophone": 4}


@dataclass(frozen=True, slots=True)
class Decision:
    segment: alternatives.Segment
    text: str
    zenz: float  # Δ log P under zenz, alternative minus original
    whisper: float  # Δ log P under whisper
    margin: float
    kind: str
    protected: bool

    @property
    def combined(self) -> float:
        return self.zenz + LAMBDA * self.whisper


def katakana(text: str) -> str:
    return "".join(chr(ord(c) + 0x60) if "ぁ" <= c <= "ゖ" else c for c in text)


def analysed(text: str, tokenizer: Vibrato, guarded: Guard) -> tuple[str, list[tuple[int, int]]]:
    """(katakana reading, protected spans): unknown kanji words read as written, so two spellings
    compare equal only where the dictionary reads both the same."""
    reading, protected = [], guarded(text)
    for morpheme in glossary.morphemes(text, tokenizer):
        surface = morpheme.surface
        if not JAPANESE.search(surface):
            reading.append(surface)
            continue
        if morpheme.known and morpheme.reading is not None:
            reading.append(katakana(morpheme.reading))
        else:
            reading.append(katakana(surface) if KANA_ONLY.match(surface) else surface)
        if not morpheme.known or morpheme.proper or KATAKANA_ONLY.match(surface):
            protected.append((morpheme.start, morpheme.end))
    return "".join(reading), protected


def _bare(text: str) -> str:
    return PUNCTUATION.sub("", unicodedata.normalize("NFKC", text))


def classify(original: str, changed: str, protected: Sequence[tuple[int, int]]) -> tuple[str, bool]:
    """The most significant kind of difference, and whether it touches a protected span."""
    worst, hit = "none", False
    matcher = difflib.SequenceMatcher(None, original, changed, autojunk=False)
    for operation, i1, i2, j1, j2 in matcher.get_opcodes():
        if operation == "equal":
            continue
        before, after = original[i1:i2], changed[j1:j2]
        if _bare(before) == _bare(after):
            kind = "punct"
        elif katakana(before) == katakana(after):
            kind = "script"
        elif KANJI.search(before) and KANJI.search(after):
            kind = "kanji-homophone"
        else:
            kind = "orthography"
        worst = max(worst, kind, key=RANK.__getitem__)
        hit = hit or glossary.overlaps(i1, max(i2, i1 + 1), protected)
    return worst, hit


def _changed(original: str, changed: str) -> tuple[int, int, int, int] | None:
    operations = [
        item
        for item in difflib.SequenceMatcher(None, original, changed, autojunk=False).get_opcodes()
        if item[0] != "equal"
    ]
    if not operations:
        return None
    return operations[0][1], operations[-1][2], operations[0][3], operations[-1][4]


def window(original: str, changed: str, context: str) -> tuple[str, str, str]:
    """(original phrase, changed phrase, left context) around the edit, spaces removed.

    shortcut: the cut is ``WINDOW`` characters, not a morpheme boundary, exactly as evaluated
    (τ was set on these scores); snap to morphemes only with a re-validated threshold.
    """
    original, changed = original.replace(" ", ""), changed.replace(" ", "")
    context = context.replace(" ", "")
    span = _changed(original, changed)
    if span is None:
        return original, changed, context[-CONTEXT_CHARACTERS:]
    i1, i2, _j1, j2 = span
    low, tail = max(0, i1 - WINDOW), min(len(original) - i2, WINDOW)
    return (
        original[low : i2 + tail],
        changed[low : j2 + tail],
        (context + original[:low])[-CONTEXT_CHARACTERS:],
    )


Score = Callable[[Sequence[tuple[str, str, str]]], list[float]]


def decide(
    segments: Sequence[alternatives.Segment],
    score: Score,
    terms: Guard,
    tokenizer: Vibrato,
) -> tuple[list[Decision], dict[str, int]]:
    """The accepted change of each segment, and how many candidates were seen.

    As evaluated, every same-reading candidate competes (spelling and punctuation variants too)
    and the segment's best one is applied only when it is an unprotected kanji-for-kanji change
    above ``TAU``: a kanji swap zenz likes less than a kana spelling is not the explanation.
    """
    jobs: list[tuple[int, str, float, float, str, bool]] = []
    stats = {"candidates": 0, "sameReading": 0, "kanjiHomophone": 0}
    contexts: list[str] = []
    for position, segment in enumerate(segments):
        contexts.append(
            "".join(item.text for item in segments[max(0, position - CONTEXT_SEGMENTS) : position])
        )
        reading, protected = analysed(segment.text, tokenizer, terms)
        best: dict[str, alternatives.Candidate] = {}
        for candidate in segment.candidates:
            stats["candidates"] += 1
            if (
                candidate.text == segment.text
                or analysed(candidate.text, tokenizer, terms)[0] != reading
            ):
                continue
            kept = best.get(candidate.text)
            if (
                kept is None
                or candidate.alternative_log_probability > kept.alternative_log_probability
            ):
                best[candidate.text] = candidate
        stats["sameReading"] += len(best)
        jobs.extend(
            (
                position,
                text,
                item.alternative_log_probability - item.original_log_probability,
                item.margin,
                *classify(segment.text, text, protected),
            )
            for text, item in best.items()
        )
    stats["kanjiHomophone"] = sum(1 for job in jobs if job[4] == "kanji-homophone")
    items = []
    for position, text, *_ in jobs:
        before, after, context = window(segments[position].text, text, contexts[position])
        phrase = analysed(before, tokenizer, terms)[0]
        items += [(phrase, context, before), (phrase, context, after)]
    scores = score(items)
    chosen: dict[int, Decision] = {}
    for at, (position, text, whisper, margin, kind, hit) in enumerate(jobs):
        decision = Decision(
            segments[position],
            text,
            scores[2 * at + 1] - scores[2 * at],
            whisper,
            margin,
            kind,
            hit,
        )
        kept_decision = chosen.get(position)
        if kept_decision is None or decision.combined > kept_decision.combined:
            chosen[position] = decision
    accepted = [
        item
        for item in chosen.values()
        if item.combined > TAU and item.kind == "kanji-homophone" and not item.protected
    ]
    return accepted, stats


def mapped(
    decisions: Sequence[Decision],
    segments: Sequence[alternatives.Segment],
    rows: Sequence[Mapping[str, object]],
) -> tuple[list[glossary.Edit], list[dict[str, object]]]:
    """Each decision as an edit of its chunk's label text, or a refusal when the labels no longer
    carry that text exactly (dropped, joined or already edited)."""
    groups = glossary.chunks(rows)
    offsets: dict[tuple[str, int], int] = {}
    spoken: dict[str, str] = {}
    for segment in segments:
        offsets[(segment.chunk, segment.index)] = len(spoken.get(segment.chunk, ""))
        spoken[segment.chunk] = spoken.get(segment.chunk, "") + segment.text
    edits, refused = [], []
    for decision in decisions:
        segment = decision.segment
        span = _changed(segment.text, decision.text)
        if span is None or segment.chunk not in groups:
            refused.append({"start": segment.start, "text": decision.text, "reason": "no-labels"})
            continue
        i1, i2, j1, j2 = span
        low = offsets[(segment.chunk, segment.index)] + i1
        high = offsets[(segment.chunk, segment.index)] + i2
        labels = glossary.chunk_text(rows, groups[segment.chunk])
        matcher = difflib.SequenceMatcher(None, spoken[segment.chunk], labels, autojunk=False)
        target = next(
            (
                block.b + low - block.a
                for block in matcher.get_matching_blocks()
                if block.a <= low and high <= block.a + block.size
            ),
            None,
        )
        if target is None:
            refused.append({"start": segment.start, "text": decision.text, "reason": "unmapped"})
            continue
        edits.append(
            glossary.Edit(
                segment.chunk,
                target,
                target + high - low,
                labels[target : target + high - low],
                decision.text[j1:j2],
                "zenz",
                {
                    "zenz": round(decision.zenz, 3),
                    "whisper": round(decision.whisper, 3),
                    "combined": round(decision.combined, 3),
                    "margin": round(decision.margin, 3),
                },
            )
        )
    return edits, refused


def _time(rows: Sequence[Mapping[str, object]], edit: glossary.Edit) -> tuple[float, float]:
    """Start and end of the labels an edit touches."""
    offset, first, last = 0, None, None
    for index in glossary.chunks(rows)[edit.chunk]:
        length = len(str(rows[index]["text"]))
        if edit.start < offset + length and offset < max(edit.end, edit.start + 1):
            first = rows[index] if first is None else first
            last = rows[index]
        offset += length
    if first is None or last is None:
        raise ReplayError("An edit lies outside its chunk's labels")
    return float(str(first["start"])), float(str(last["end"]))


def _entries(rows: Sequence[Mapping[str, object]], edits: Sequence[glossary.Edit]) -> list[object]:
    entries: list[object] = []
    for edit in edits:
        start, end = _time(rows, edit)
        entries.append(
            {
                "start": start,
                "end": end,
                "chunk": edit.chunk,
                "before": edit.before,
                "after": edit.after,
                "source": edit.source,
                **edit.detail,
            }
        )
    return entries


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write_new(path: Path, document: object) -> None:
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
        json.dump(document, stream, ensure_ascii=False, indent=1)


@dataclass(frozen=True, slots=True)
class Rescoring:
    store: Path
    mono: Path
    whisper: Path
    zenz: Path


Replayed = tuple[list[alternatives.Segment], dict[str, int]]


def replayed_on_mlx(rescoring: Rescoring) -> Replayed:
    recognizer = alternatives.MlxRecognizer(rescoring.whisper)
    return alternatives.replay(recognizer, rescoring.store, rescoring.mono)


def correct(
    labels: Path,
    output: Path,
    receipt: Path,
    glossaries: Sequence[Path],
    rescoring: Rescoring | None,
    replay: Callable[[Rescoring], Replayed] | None = None,
    scorer: Callable[[Path], Score] | None = None,
) -> dict[str, object]:
    """Write corrected labels and the receipt; returns the receipt."""
    document = json.loads(labels.read_text(encoding="utf-8"))
    if not isinstance(document, dict) or not isinstance(document.get("labels"), list):
        raise ReplayError("Labels must be a JSON object with a labels array")
    rows = [glossary_row(row) for row in document["labels"]]
    terms = glossary.compiled(glossaries)
    started = time.monotonic()
    planned = glossary.plan(rows, terms)
    corrected = glossary.apply(rows, planned)
    logged = _entries(rows, planned)
    report: dict[str, object] = {
        "version": 1,
        "inputs": {
            "labels": _sha256(labels),
            "glossaries": {str(path): _sha256(path) for path in glossaries},
        },
        "policy": {
            "lambda": LAMBDA,
            "tau": TAU,
            "window": WINDOW,
            "margin": alternatives.MARGIN,
            "contextCharacters": CONTEXT_CHARACTERS,
        },
        "glossarySeconds": round(time.monotonic() - started, 3),
        "refused": [],
    }
    if rescoring is not None:
        started = time.monotonic()
        segments, stats = (replay or replayed_on_mlx)(rescoring)
        replayed = time.monotonic()
        score = (scorer or (lambda path: zenz_lm.load(path).score))(rescoring.zenz)
        decisions, seen = decide(segments, score, terms.guarded, linguistics._tokenizer())
        edits, refused = mapped(decisions, segments, corrected)
        report["rescoring"] = {
            **stats,
            **seen,
            "accepted": len(decisions),
            "replaySeconds": round(replayed - started, 3),
            "scoreSeconds": round(time.monotonic() - replayed, 3),
            "zenz": str(rescoring.zenz),
            "whisper": str(rescoring.whisper),
        }
        logged += _entries(corrected, edits)
        corrected = glossary.apply(corrected, edits)
        report["refused"] = refused
    report["edits"] = logged
    report["labels"] = {"input": len(rows), "output": len(corrected)}
    _write_new(output, {**document, "labels": corrected})
    _write_new(receipt, report)
    return report


def glossary_row(row: object) -> dict[str, object]:
    if not isinstance(row, dict) or not isinstance(row.get("text"), str):
        raise ReplayError("Every label needs text")
    return {str(key): value for key, value in row.items()}


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="replay.asr_kanji_fix", description=__doc__)
    parser.add_argument("labels", type=Path)
    parser.add_argument("output", type=Path, help="corrected labels JSON; never overwritten")
    parser.add_argument("receipt", type=Path, help="every edit and refusal; never overwritten")
    parser.add_argument("--glossary", type=Path, action="append", default=[])
    parser.add_argument(
        "--rescore", nargs=4, type=Path, metavar=("CHUNKS", "MONO", "WHISPER", "ZENZ")
    )
    args = parser.parse_args(argv)
    rescoring = Rescoring(*args.rescore) if args.rescore else None
    try:
        report = correct(args.labels, args.output, args.receipt, args.glossary, rescoring)
    except (ReplayError, OSError, ValueError) as error:
        print(f"ASR_KANJI_FIX_ERROR {error}", file=sys.stderr)
        return 1
    edits = report["edits"]
    print(
        json.dumps(
            {"edits": len(edits) if isinstance(edits, list) else 0, "labels": report["labels"]},
            ensure_ascii=False,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
