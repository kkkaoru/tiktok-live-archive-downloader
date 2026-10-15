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
from decimal import ROUND_CEILING, ROUND_FLOOR, Decimal
from fractions import Fraction
from pathlib import Path

import numpy as np

from replay.core import ReplayError

BLOCK_SAMPLES = 1 << 20
MAX_LABELS = 1_000_000
MAX_AMPLITUDE = 64


@dataclass(frozen=True, slots=True)
class Span:
    """A half-open run of exactly-zero samples, in samples at `rate`."""

    start: int
    end: int
    rate: int
    amplitude: int = 0


def zero_runs(
    blocks: Iterable[np.ndarray], *, rate: int, minimum_seconds: int, maximum_amplitude: int = 0
) -> list[Span]:
    """Runs of samples no louder than `maximum_amplitude` (0 = exactly zero) of at least
    `minimum_seconds`. A positive amplitude treats decay tails and isolated dither samples
    below that magnitude as silence, so one inaudible stretch is not split into two."""
    if rate < 1 or minimum_seconds < 1 or not 0 <= maximum_amplitude <= MAX_AMPLITUDE:
        raise ReplayError("Invalid rate, minimum silence or amplitude")
    minimum = rate * minimum_seconds
    spans: list[Span] = []
    run_start: int | None = None
    offset = 0
    for block in blocks:
        previous = 0
        for index in [
            *np.flatnonzero(np.abs(block.astype(np.int32)) > maximum_amplitude).tolist(),
            len(block),
        ]:
            if index > previous and run_start is None:
                run_start = offset + previous
            if index < len(block) and run_start is not None:
                if offset + index - run_start >= minimum:
                    spans.append(Span(run_start, offset + index, rate, maximum_amplitude))
                run_start = None
            previous = index + 1
        offset += len(block)
    if run_start is not None and offset - run_start >= minimum:
        spans.append(Span(run_start, offset, rate, maximum_amplitude))
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
                    "rule": _span_rule(home),
                },
            }
        )
    if not kept:
        raise ReplayError("Every label lies in digital silence")
    return kept, excluded


def _span_rule(span: Span) -> str:
    if span.amplitude == 0:
        return "every input sample in the span is exactly zero"
    return f"every input sample in the span has magnitude at most {span.amplitude} LSB"


def _rounded(value: Fraction, mode: str) -> float:
    """A 9-digit decimal that never lies on the wrong side of `value` (ceiling or floor)."""
    exact = Decimal(value.numerator) / Decimal(value.denominator)
    return float(exact.quantize(Decimal("0.000000001"), rounding=mode))


def clamp_partial(
    labels: Sequence[dict[str, object]],
    spans: Sequence[Span],
    *,
    minimum_kept: Fraction,
) -> tuple[list[dict[str, object]], list[dict[str, object]], list[dict[str, object]]]:
    """Keep the part of a label outside digital silence when it only partly overlaps a run.

    A label that starts inside a run and ends after it starts at the run's end; one that starts
    before a run and ends inside it ends at the run's start. A label with signal on both sides of
    a run is untouched. Changed labels carry the original time, the run and the dropped length. A
    label whose kept part is shorter than `minimum_kept` has almost no acoustic support and is
    returned among the dropped records instead of as a sliver.
    """
    result: list[dict[str, object]] = []
    clamps: list[dict[str, object]] = []
    dropped_labels: list[dict[str, object]] = []
    for index, label in enumerate(labels):
        start, end = Fraction(str(label["start"])), Fraction(str(label["end"]))
        changed = dict(label)
        for span in spans:
            low, high = Fraction(span.start, span.rate), Fraction(span.end, span.rate)
            if low <= start < high < end:
                changed["start"] = _rounded(high, ROUND_CEILING)
                start = Fraction(str(changed["start"]))
                side, dropped = "start", high - Fraction(str(label["start"]))
            elif start < low < end <= high:
                changed["end"] = _rounded(low, ROUND_FLOOR)
                end = Fraction(str(changed["end"]))
                side, dropped = "end", Fraction(str(label["end"])) - low
            else:
                continue
            changed["silenceClamp"] = {
                "side": side,
                "originalStart": label["start"],
                "originalEnd": label["end"],
                "spanStartSeconds": float(low),
                "spanEndSeconds": float(high),
                "droppedSeconds": float(dropped),
                "rule": "label time inside digital silence is not delivered speech",
            }
            clamps.append({"index": index, "text": label["text"], "side": side})
        if "silenceClamp" in changed and end - start < minimum_kept:
            dropped_labels.append(
                {"index": index, "label": dict(label), "kept": float(end - start)}
            )
            continue
        result.append(changed)
    if not result:
        raise ReplayError("Every label lies in digital silence")
    return result, clamps, dropped_labels


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("labels", type=Path)
    parser.add_argument("pcm", type=Path, help="mono PCM16 WAV the labels were recognized from")
    parser.add_argument("output", type=Path)
    parser.add_argument("--minimum-seconds", type=int, required=True)
    parser.add_argument("--maximum-amplitude", type=int, default=0)
    parser.add_argument("--clamp-partial", action="store_true")
    parser.add_argument("--fps", type=int, default=30)
    args = parser.parse_args(argv)
    if args.output.exists() or args.output.is_symlink():
        raise ReplayError("Refusing to overwrite the output")
    document = json.loads(args.labels.read_text(encoding="utf-8"))
    rate = wav_rate(args.pcm)
    spans = zero_runs(
        wav_blocks(args.pcm),
        rate=rate,
        minimum_seconds=args.minimum_seconds,
        maximum_amplitude=args.maximum_amplitude,
    )
    kept, excluded = exclude(document["labels"], spans)
    clamps: list[dict[str, object]] = []
    slivers: list[dict[str, object]] = []
    if args.clamp_partial:
        kept, clamps, slivers = clamp_partial(kept, spans, minimum_kept=Fraction(1, args.fps))
    document = dict(document) | {
        "labels": kept,
        "silenceClamps": clamps,
        "droppedAsSlivers": slivers,
        "excludedAsDigitalSilence": excluded,
        "digitalSilenceSpans": [
            {"startSeconds": s.start / s.rate, "endSeconds": s.end / s.rate} for s in spans
        ],
        "parentLabelsFile": str(args.labels),
        "silenceSourcePcm": str(args.pcm),
        "minimumSilenceSeconds": args.minimum_seconds,
        "maximumSilenceAmplitude": args.maximum_amplitude,
    }
    args.output.write_text(json.dumps(document, ensure_ascii=False, indent=1), encoding="utf-8")
    summary = {"spans": len(spans), "excluded": len(excluded), "kept": len(kept)}
    print(json.dumps(summary | {"clamped": len(clamps)}))
    return 0


if __name__ == "__main__":
    sys.exit(main())
