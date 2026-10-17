"""Replace one collapsed run of assembled ASR labels with a limited re-recognition (candidate only).

When a long-form decode collapses seconds of speech into one label, a bounded re-recognition of
the same separated audio with the same pinned recognizer can restore the words. This substitutes
exactly the declared label block, whose span must equal the declared window, with the re-recognized
words that lie wholly inside that window. Every other label keeps its exact value, words that
cross the window edge are listed as excluded and never inserted, and word times are the
recognizer's own (a later CTC refinement may re-time them). Nothing here approves accuracy.
"""

import json
import os
from dataclasses import asdict, dataclass
from itertools import pairwise
from pathlib import Path

from replay.asr_clock_projection import _binding, _labels
from replay.asr_ctc_label_refine import read_json
from replay.caption_repair import ClockLabel
from replay.recordings import record
from replay.stages import file_digest

TOLERANCE = 1e-6
PROVENANCE = "clip-reasr"
MAX_FOLD = 0.1  # seconds; three frames at most, so a folded piece is never a real word
SUSTAINED = frozenset("あぁー～")  # a held vowel; a recognizer's stand-in for a tone, not words


@dataclass(frozen=True, slots=True)
class Word:
    text: str
    start: float
    end: float


@dataclass(frozen=True, slots=True)
class Plan:
    removed: tuple[ClockLabel, ...]
    inserted: tuple[ClockLabel, ...]
    excluded: tuple[Word, ...]
    folded: tuple[Word, ...] = ()


def _number(value: object) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError("Expected a finite number")
    return float(value)


def read_words(clip: dict[str, object]) -> tuple[Word, ...]:
    """Word records of a re-recognition file in the input-PCM clock."""
    if clip.get("clock") != "input-pcm-not-projected":
        raise ValueError("Re-recognition words must use the input PCM clock")
    rows = clip.get("words")
    if not isinstance(rows, list) or not rows:
        raise ValueError("Re-recognition has no words")
    words: list[Word] = []
    for raw in rows:
        row = record(raw)
        text = row.get("word")
        if not isinstance(text, str) or not text.strip():
            raise ValueError("Invalid re-recognized word")
        words.append(Word(text.strip(), _number(row.get("start")), _number(row.get("end"))))
    if any(b.start < a.start for a, b in pairwise(words)):
        raise ValueError("Re-recognized words are not ordered by onset")
    return tuple(words)


def _inside(word: Word, window: tuple[float, float]) -> bool:
    return word.start >= window[0] - TOLERANCE and word.end <= window[1] + TOLERANCE


def plan(
    labels: tuple[ClockLabel, ...],
    words: tuple[Word, ...],
    *,
    first: int,
    stop: int,
    window: tuple[float, float],
    source: str,
    fold_shorter_than: float | None = None,
    drop_sustained: bool = False,
) -> Plan:
    """The declared substitution, or ``ValueError`` naming the rule it breaks.

    With ``drop_sustained`` the block is removed and nothing is inserted, allowed only when every
    re-recognized word overlapping the window is a held vowel or none is (no words are evidenced
    there).
    """
    if not 0 <= first < stop <= len(labels) or window[0] >= window[1]:
        raise ValueError("Expected a bounded label block and a positive window")
    block = labels[first:stop]
    if abs(block[0].start - window[0]) > TOLERANCE or abs(block[-1].end - window[1]) > TOLERANCE:
        raise ValueError("The declared window is not the span of the declared label block")
    inside = [word for word in words if _inside(word, window)]
    excluded = tuple(word for word in words if not _inside(word, window) and _crosses(word, window))
    if drop_sustained:
        heard = [word for word in words if _crosses(word, window)]
        if not all(set(word.text) <= SUSTAINED for word in heard):
            raise ValueError("The window holds recognized words, not only a sustained vowel")
        return Plan(block, (), excluded)
    if not inside:
        raise ValueError("No re-recognized word lies inside the declared window")
    inside, folded = _fold_short(inside, fold_shorter_than)
    if any(word.end < word.start for word in inside):
        raise ValueError("A re-recognized word ends before it starts")
    inserted = tuple(
        ClockLabel(word.text, word.start, word.end, f"{PROVENANCE}:{source}:{index}")
        for index, word in enumerate(inside)
    )
    if any(b.start < a.end - TOLERANCE for a, b in pairwise(inserted)):
        raise ValueError("Re-recognized words overlap")
    if first > 0 and labels[first - 1].end > inserted[0].start + TOLERANCE:
        raise ValueError("Inserted words overlap the unchanged left neighbour")
    if stop < len(labels) and inserted[-1].end > labels[stop].start + TOLERANCE:
        raise ValueError("Inserted words overlap the unchanged right neighbour")
    return Plan(block, inserted, excluded, folded)


def _fold_short(words: list[Word], threshold: float | None) -> tuple[list[Word], tuple[Word, ...]]:
    """Append words shorter than ``threshold`` to the word before them (text order kept).

    A recognizer sometimes gives the pieces of one word a few milliseconds each; the folded word
    keeps the earlier start and the later end, both the recognizer's own times. A short first word
    has nothing before it and stays.
    """
    if threshold is None:
        return words, ()
    merged: list[Word] = []
    folded: list[Word] = []
    for word in words:
        if merged and word.end - word.start < threshold - TOLERANCE:
            previous = merged[-1]
            merged[-1] = Word(previous.text + word.text, previous.start, word.end)
            folded.append(word)
        else:
            merged.append(word)
    return merged, tuple(folded)


def _crosses(word: Word, window: tuple[float, float]) -> bool:
    return word.start < window[1] and word.end > window[0]


def run(config: Path, output: Path) -> int:
    """Write the candidate label set; returns its label count."""
    if output.exists() or output.is_symlink() or output.parent != output.parent.resolve():
        raise ValueError("Expected a new output under a canonical directory")
    digest = file_digest(config)
    data = read_json(config)
    if set(data) != {"version", "artifacts", "block"} or data["version"] != 1:
        raise ValueError("Unsupported window substitution configuration")
    artifacts = record(data["artifacts"])
    if set(artifacts) != {"labels", "clip"}:
        raise ValueError("Expected pinned labels and a pinned re-recognition")
    bound = {name: _binding(value) for name, value in artifacts.items()}
    for path, pin in bound.values():
        if file_digest(path) != pin:
            raise ValueError("A pinned substitution input changed")
    original = read_json(bound["labels"][0])
    block = record(data["block"])
    first, stop, start, end = (block.get(k) for k in ("firstLabel", "stopLabel", "start", "end"))
    if type(first) is not int or type(stop) is not int:
        raise ValueError("Expected exact integer block coordinates")
    fold = block.get("foldShorterThan")
    if fold is not None and (
        isinstance(fold, bool) or not isinstance(fold, (int, float)) or not 0 < fold <= MAX_FOLD
    ):
        raise ValueError("foldShorterThan must be a number of seconds up to the fold limit")
    drop = block.get("dropSustained", False)
    if type(drop) is not bool:
        raise ValueError("dropSustained must be a boolean")
    clip = read_json(bound["clip"][0])
    labels = _labels(original)
    outcome = plan(
        labels,
        read_words(clip),
        first=first,
        stop=stop,
        window=(_number(start), _number(end)),
        source=bound["clip"][1][:12],
        fold_shorter_than=None if fold is None else float(fold),
        drop_sustained=drop,
    )
    rows = [asdict(label) for label in labels]
    rows[first:stop] = [asdict(label) for label in outcome.inserted]
    document = {
        **original,
        "labels": rows,
        "parentSHA256": {
            **record(original["parentSHA256"]),
            **{str(path): pin for path, pin in bound.values()},
        },
        "windowSubstitutionSHA256": file_digest(Path(__file__)),
        "windowSubstitution": {
            "configSHA256": digest,
            "labels": {"path": str(bound["labels"][0]), "sha256": bound["labels"][1]},
            "clip": {"path": str(bound["clip"][0]), "sha256": bound["clip"][1]},
            "block": {
                "firstLabel": first,
                "stopLabel": stop,
                "start": start,
                "end": end,
                **({} if fold is None else {"foldShorterThan": fold}),
                **({"dropSustained": True} if drop else {}),
            },
            "removed": [asdict(label) for label in outcome.removed],
            "inserted": [asdict(label) for label in outcome.inserted],
            "excluded": [asdict(word) for word in outcome.excluded],
            "folded": [asdict(word) for word in outcome.folded],
            "candidateOnly": True,
        },
        "humanReviewed": False,
        "speechAccuracyVerified": False,
        "qualityApproved": False,
        "deliveryComplete": False,
    }
    with output.open("x", encoding="utf-8") as stream:
        os.fchmod(stream.fileno(), 0o600)
        json.dump(document, stream, ensure_ascii=False, allow_nan=False, indent=2)
    return len(rows)
