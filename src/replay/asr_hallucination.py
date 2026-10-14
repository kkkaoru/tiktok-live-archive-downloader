"""Drop Whisper's silence and laughter hallucinations from word labels, by measurement.

Whisper's decoder always emits text, so on silence, music residue or laughter it falls
back on boilerplate learned from video credits, for example "ご視聴ありがとうございました".
Community guidance and the arXiv study on non-speech hallucinations (2501.11378) point to
the same layers: keep silence away from the model, and post-filter a bag of known
hallucinations. Voice-activity models also misjudge audio that is almost exactly zero, so
the first layer here is plain measured energy of the separated vocals.

Two deterministic rules apply, identical for every video:

* silent: a label whose time span (widened by the alignment tolerance) holds no frame at
  or above the speech floor cannot have been spoken, so it is removed;
* stock phrase: a run of labels spelling a known credits phrase is removed, because
  laughter is voiced and passes the energy test. The one exception is a closing phrase
  inside the last seconds of the recording that sits on measured speech energy.

Every removal is reported with its evidence; nothing is removed silently and no time is
invented for the remaining labels.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import wave
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from numpy.typing import NDArray

from replay.core import ReplayError

HOP_SECONDS = 0.02
SPEECH_FLOOR_BELOW_P90_DB = 25.0
ABSOLUTE_FLOOR_DB = -70.0
SILENT_ACTIVITY = 0.0
# Whisper word times drift by a few hundred milliseconds; real words measured 0.26-0.50 s
# from the nearest speech frame, so silence must extend well beyond that to count.
ALIGNMENT_TOLERANCE_SECONDS = 1.0
CLOSING_WINDOW_SECONDS = 90.0
CLOSING_MIN_ACTIVITY = 0.5
STOCK_PHRASES: tuple[str, ...] = (
    "ご視聴ありがとうございました",
    "ご視聴ありがとうございます",
    "最後までご視聴いただきありがとうございました",
    "ご覧いただきありがとうございました",
    "チャンネル登録よろしくお願いします",
    "チャンネル登録お願いします",
)


@dataclass(frozen=True, slots=True)
class Label:
    text: str
    start: float
    end: float


@dataclass(frozen=True, slots=True)
class Removal:
    reason: str
    text: str
    start: float
    end: float
    activity: float
    indices: tuple[int, ...]


@dataclass(frozen=True, slots=True)
class Envelope:
    db: NDArray[np.float64]
    floor_db: float
    duration: float

    def activity(self, start: float, end: float, tolerance: float = 0.0) -> float:
        """Fraction of 20 ms frames in the widened span at or above the speech floor."""
        first = max(0, int((start - tolerance) / HOP_SECONDS))
        last = min(len(self.db), int(np.ceil((end + tolerance) / HOP_SECONDS)))
        if last <= first:
            return 0.0
        return float(np.mean(self.db[first:last] >= self.floor_db))


def measure(wav: Path) -> Envelope:
    """Frame energy of a mono 16-bit PCM WAV, with a speech floor derived from the file."""
    with wave.open(str(wav), "rb") as stream:
        if stream.getnchannels() != 1 or stream.getsampwidth() != 2:
            raise ReplayError("Hallucination filter needs mono 16-bit PCM")
        rate = stream.getframerate()
        samples = np.frombuffer(stream.readframes(stream.getnframes()), dtype="<i2")
    hop = int(round(HOP_SECONDS * rate))
    frames = len(samples) // hop
    if frames < 1:
        raise ReplayError("Audio is shorter than one analysis frame")
    block = samples[: frames * hop].astype(np.float64).reshape(frames, hop) / 32768.0
    db = 20.0 * np.log10(np.sqrt((block**2).mean(axis=1)) + 1e-12)
    floor = max(float(np.percentile(db, 90)) - SPEECH_FLOOR_BELOW_P90_DB, ABSOLUTE_FLOOR_DB)
    return Envelope(db, floor, len(samples) / rate)


def _stock_runs(labels: Sequence[Label]) -> list[tuple[str, tuple[int, ...]]]:
    """Runs of labels that spell a stock phrase exactly; boundary-straddling labels are kept."""
    text = "".join(label.text for label in labels)
    offsets: list[int] = []
    position = 0
    for label in labels:
        offsets.append(position)
        position += len(label.text)
    runs: list[tuple[str, tuple[int, ...]]] = []
    taken: set[int] = set()
    for phrase in sorted(STOCK_PHRASES, key=len, reverse=True):
        found = text.find(phrase)
        while found != -1:
            inside = tuple(
                index
                for index, label in enumerate(labels)
                if label.text
                and offsets[index] >= found
                and offsets[index] + len(label.text) <= found + len(phrase)
            )
            covered = sum(len(labels[index].text) for index in inside)
            if covered == len(phrase) and taken.isdisjoint(inside):
                runs.append((phrase, inside))
                taken.update(inside)
            found = text.find(phrase, found + 1)
    return sorted(runs, key=lambda run: run[1][0])


def _closing_speech(envelope: Envelope, start: float, end: float) -> bool:
    return (
        end >= envelope.duration - CLOSING_WINDOW_SECONDS
        and envelope.activity(start, end) >= CLOSING_MIN_ACTIVITY
    )


def filter_labels(labels: Sequence[Label], envelope: Envelope) -> tuple[list[Label], list[Removal]]:
    """Return the kept labels and the evidence for every removed run."""
    removals: list[Removal] = []
    removed: set[int] = set()
    for phrase, indices in _stock_runs(labels):
        start, end = labels[indices[0]].start, labels[indices[-1]].end
        if _closing_speech(envelope, start, end):
            continue
        activity = envelope.activity(start, end)
        removals.append(Removal("stock-phrase", phrase, start, end, activity, indices))
        removed.update(indices)
    for index, label in enumerate(labels):
        if index in removed:
            continue
        activity = envelope.activity(label.start, label.end, ALIGNMENT_TOLERANCE_SECONDS)
        if activity <= SILENT_ACTIVITY:
            removals.append(
                Removal("silent", label.text, label.start, label.end, activity, (index,))
            )
            removed.add(index)
    kept = [label for index, label in enumerate(labels) if index not in removed]
    return kept, sorted(removals, key=lambda item: item.indices[0])


def _number(value: object) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ReplayError("Label times must be numbers")
    return float(value)


def _label(row: object) -> Label:
    if not isinstance(row, dict) or not isinstance(row.get("text"), str):
        raise ReplayError("Every label needs text and an ordered start and end")
    start, end = _number(row.get("start")), _number(row.get("end"))
    if end < start:
        raise ReplayError("Every label needs text and an ordered start and end")
    return Label(str(row["text"]), start, end)


def _rows(document: dict[str, object]) -> list[object]:
    rows = document.get("labels")
    if not isinstance(rows, list) or not rows:
        raise ReplayError("Candidate needs a nonempty labels array")
    return rows


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def apply(candidate: Path, vocals: Path, output: Path) -> dict[str, object]:
    """Write a new candidate without the hallucinated labels, plus a measured report."""
    document = json.loads(candidate.read_text(encoding="utf-8"))
    if not isinstance(document, dict):
        raise ReplayError("Candidate must be a JSON object")
    rows = _rows(document)
    labels = [_label(row) for row in rows]
    envelope = measure(vocals)
    kept, removals = filter_labels(labels, envelope)
    if not kept:
        raise ReplayError("Filtering would remove every label")
    drop = {index for removal in removals for index in removal.indices}
    report: dict[str, object] = {
        "policy": {
            "hopSeconds": HOP_SECONDS,
            "speechFloorDb": round(envelope.floor_db, 3),
            "silentActivity": SILENT_ACTIVITY,
            "alignmentToleranceSeconds": ALIGNMENT_TOLERANCE_SECONDS,
            "closingWindowSeconds": CLOSING_WINDOW_SECONDS,
            "stockPhrases": list(STOCK_PHRASES),
        },
        "candidateSHA256": _sha256(candidate),
        "vocalsSHA256": _sha256(vocals),
        "inputLabels": len(labels),
        "keptLabels": len(kept),
        "removed": [
            {
                "reason": item.reason,
                "text": item.text,
                "start": item.start,
                "end": item.end,
                "activity": round(item.activity, 4),
                "labelIndices": list(item.indices),
            }
            for item in removals
        ],
    }
    result = dict(document)
    result["labels"] = [row for index, row in enumerate(rows) if index not in drop]
    result["hallucinationFilter"] = report
    descriptor = os.open(output, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
        json.dump(result, stream, ensure_ascii=False, indent=1)
    return report


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="replay.asr_hallucination", description=__doc__)
    parser.add_argument("candidate", type=Path)
    parser.add_argument("vocals", type=Path, help="mono 16-bit PCM WAV the labels came from")
    parser.add_argument("output", type=Path, help="new candidate JSON; never overwritten")
    args = parser.parse_args(argv)
    try:
        report = apply(args.candidate, args.vocals, args.output)
    except (ReplayError, OSError, ValueError, wave.Error) as error:
        print(f"ASR_HALLUCINATION_ERROR {error}", file=sys.stderr)
        return 1
    print(
        json.dumps(
            {
                "inputLabels": report["inputLabels"],
                "keptLabels": report["keptLabels"],
                "removedRuns": len(report["removed"]) if isinstance(report["removed"], list) else 0,
            }
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
