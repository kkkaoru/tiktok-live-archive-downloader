"""Set aside recognized labels that lie wholly inside a measured stretch of digital silence.

A recognizer can emit text for an input that is exactly zero for seconds on end; such a label has no
acoustic basis and is not evidence of speech. This module finds runs of exactly-zero samples of at
least `minimum_seconds` in the 16-bit mono PCM the labels were recognized from, and writes a derived
label file without the labels that fit entirely inside a run. Excluded labels are kept in full next
to the span that justifies them (so a person can restore them), a label that only partly overlaps a
run stays, and the input labels are never modified.
"""

import argparse
import json
import sys
import wave
from collections.abc import Iterable, Iterator, Sequence
from dataclasses import dataclass
from fractions import Fraction
from pathlib import Path

import numpy as np

from replay.core import ReplayError

BLOCK_SAMPLES = 1 << 20
MAX_LABELS = 1_000_000


@dataclass(frozen=True, slots=True)
class Span:
    """A half-open run of exactly-zero samples, in samples at `rate`."""

    start: int
    end: int
    rate: int


def zero_runs(blocks: Iterable[np.ndarray], *, rate: int, minimum_seconds: int) -> list[Span]:
    """Runs of exactly-zero samples lasting at least `minimum_seconds`."""
    if rate < 1 or minimum_seconds < 1:
        raise ReplayError("Invalid rate or minimum silence")
    minimum = rate * minimum_seconds
    spans: list[Span] = []
    run_start: int | None = None
    offset = 0
    for block in blocks:
        previous = 0
        for index in [*np.flatnonzero(block).tolist(), len(block)]:
            if index > previous and run_start is None:
                run_start = offset + previous
            if index < len(block) and run_start is not None:
                if offset + index - run_start >= minimum:
                    spans.append(Span(run_start, offset + index, rate))
                run_start = None
            previous = index + 1
        offset += len(block)
    if run_start is not None and offset - run_start >= minimum:
        spans.append(Span(run_start, offset, rate))
    return spans


def wav_rate(path: Path) -> int:
    """Sample rate of a mono PCM16 WAV; anything else is refused."""
    with wave.open(str(path), "rb") as stream:
        if (stream.getnchannels(), stream.getsampwidth()) != (1, 2):
            raise ReplayError("Expected mono PCM16")
        return stream.getframerate()


def wav_blocks(path: Path) -> Iterator[np.ndarray]:
    """Successive blocks of a mono PCM16 WAV."""
    with wave.open(str(path), "rb") as stream:
        while data := stream.readframes(BLOCK_SAMPLES):
            yield np.frombuffer(data, dtype="<i2")


def exclude(
    labels: Sequence[dict[str, object]], spans: Sequence[Span]
) -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    """Return (kept labels, excluded records); a label is excluded only if wholly inside a span."""
    if not 0 < len(labels) <= MAX_LABELS:
        raise ReplayError("Invalid label count")
    kept: list[dict[str, object]] = []
    excluded: list[dict[str, object]] = []
    for index, label in enumerate(labels):
        start, end = Fraction(str(label["start"])), Fraction(str(label["end"]))
        home = next(
            (
                span
                for span in spans
                if Fraction(span.start, span.rate) <= start and end <= Fraction(span.end, span.rate)
            ),
            None,
        )
        if home is None:
            kept.append(dict(label))
            continue
        excluded.append(
            {
                "index": index,
                "label": dict(label),
                "span": {
                    "startSeconds": float(Fraction(home.start, home.rate)),
                    "endSeconds": float(Fraction(home.end, home.rate)),
                    "rule": "every input sample in the span is exactly zero",
                },
            }
        )
    if not kept:
        raise ReplayError("Every label lies in digital silence")
    return kept, excluded


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("labels", type=Path)
    parser.add_argument("pcm", type=Path, help="mono PCM16 WAV the labels were recognized from")
    parser.add_argument("output", type=Path)
    parser.add_argument("--minimum-seconds", type=int, required=True)
    args = parser.parse_args(argv)
    if args.output.exists() or args.output.is_symlink():
        raise ReplayError("Refusing to overwrite the output")
    document = json.loads(args.labels.read_text(encoding="utf-8"))
    rate = wav_rate(args.pcm)
    spans = zero_runs(wav_blocks(args.pcm), rate=rate, minimum_seconds=args.minimum_seconds)
    kept, excluded = exclude(document["labels"], spans)
    document = dict(document) | {
        "labels": kept,
        "excludedAsDigitalSilence": excluded,
        "digitalSilenceSpans": [
            {"startSeconds": s.start / s.rate, "endSeconds": s.end / s.rate} for s in spans
        ],
        "parentLabelsFile": str(args.labels),
        "silenceSourcePcm": str(args.pcm),
        "minimumSilenceSeconds": args.minimum_seconds,
    }
    args.output.write_text(json.dumps(document, ensure_ascii=False, indent=1), encoding="utf-8")
    print(json.dumps({"spans": len(spans), "excluded": len(excluded), "kept": len(kept)}))
    return 0


if __name__ == "__main__":
    sys.exit(main())
