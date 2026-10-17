"""Read-only proof that a window substitution differs from its pinned original only as declared.

Re-derives the substitution from the pinned labels and re-recognition and checks that the
output equals it, that every other label and document field is unchanged, and that onsets stay
ordered. It proves consistency of the evidence, never speech accuracy.
"""

from dataclasses import asdict
from itertools import pairwise
from pathlib import Path

from replay.asr_clock_projection import _labels
from replay.asr_ctc_label_refine import read_json
from replay.asr_window_substitute import plan, read_words
from replay.caption_repair import ClockLabel
from replay.recordings import record
from replay.stages import file_digest

ALLOWED_CHANGES = frozenset(
    {
        "labels",
        "parentSHA256",
        "windowSubstitutionSHA256",
        "windowSubstitution",
        "humanReviewed",
        "speechAccuracyVerified",
        "qualityApproved",
        "deliveryComplete",
    }
)


def _pinned(value: object) -> Path:
    reference = record(value)
    path = Path(str(reference.get("path")))
    if file_digest(path) != reference.get("sha256"):
        raise ValueError("A pinned verification input changed")
    return path


def verify(refined: Path) -> dict[str, object]:
    """Return a proof summary or raise ``ValueError`` naming the failed property."""
    document = read_json(refined)
    meta = record(document.get("windowSubstitution"))
    original_path = _pinned(meta.get("labels"))
    clip_path = _pinned(meta.get("clip"))
    original_document = read_json(original_path)
    changed = {
        key
        for key in set(document) | set(original_document)
        if document.get(key) != original_document.get(key)
    }
    if not changed <= ALLOWED_CHANGES:
        raise ValueError(f"Undeclared document fields differ: {sorted(changed - ALLOWED_CHANGES)}")
    block = record(meta.get("block"))
    first, stop = block.get("firstLabel"), block.get("stopLabel")
    start, end = block.get("start"), block.get("end")
    if (
        type(first) is not int
        or type(stop) is not int
        or isinstance(start, bool)
        or not isinstance(start, (int, float))
        or isinstance(end, bool)
        or not isinstance(end, (int, float))
    ):
        raise ValueError("Invalid declared block")
    labels = _labels(original_document)
    fold = block.get("foldShorterThan")
    if fold is not None and (isinstance(fold, bool) or not isinstance(fold, (int, float))):
        raise ValueError("Invalid declared fold threshold")
    drop = block.get("dropSustained", False)
    if type(drop) is not bool:
        raise ValueError("Invalid declared drop flag")
    expected = plan(
        labels,
        read_words(read_json(clip_path)),
        first=first,
        stop=stop,
        window=(float(start), float(end)),
        source=str(record(meta.get("clip")).get("sha256"))[:12],
        fold_shorter_than=None if fold is None else float(fold),
        drop_sustained=drop,
    )
    if meta.get("folded", []) != [asdict(word) for word in expected.folded]:
        raise ValueError("Folded words differ from the declared substitution")
    rows = document.get("labels")
    if not isinstance(rows, list):
        raise ValueError("Refined labels are missing")
    produced = [record(row) for row in rows]
    wanted = [
        *[_row(label) for label in labels[:first]],
        *[_row(label) for label in expected.inserted],
        *[_row(label) for label in labels[stop:]],
    ]
    if produced != wanted:
        raise ValueError("Refined labels differ from the declared substitution")
    if any(_value(b["start"]) < _value(a["start"]) for a, b in pairwise(produced)):
        raise ValueError("Label onsets are not monotonic")
    return {
        "labels": len(produced),
        "removed": len(expected.removed),
        "inserted": len(expected.inserted),
        "excluded": len(expected.excluded),
    }


def _row(label: ClockLabel) -> dict[str, object]:
    return asdict(label)


def _value(value: object) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError("Expected a finite number")
    return float(value)
