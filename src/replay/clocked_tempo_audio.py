"""Candidate pitch-preserving tempo adjustment in exact shared video-cut slots.

Each cut uses source context before and after its owned interval. Only the central
tempo-adjusted interval is published; measured filter tail differences and source
edge zero context remain explicit. Exact output length is not speech acceptance.
"""

from __future__ import annotations

import argparse
import os
import subprocess
import wave
from collections.abc import Iterator
from dataclasses import asdict, dataclass, replace
from fractions import Fraction
from pathlib import Path
from time import monotonic

from replay.audio_clock import AudioFrameClock
from replay.caption_io import _string
from replay.caption_measured_job import _json, _local, _object, _pin, _rational
from replay.caption_native_layout import NativeCaptionLayout
from replay.clocked_cut_audio import CutSlot, plan_slots, sample_boundary
from replay.continuous_cut_job import MAX_CLOCK_BYTES, MAX_CONFIG_BYTES, MAX_PCM_BYTES
from replay.foreground_evidence import _bound_bytes
from replay.stages import file_digest
from replay.tracking_observations import _checked_output, _digest

MAX_WINDOW_BYTES = 512 * 1024**2  # one continuous kept span of about 46 min of 48 kHz stereo
PROCESS_TIMEOUT_SECONDS = 120


def tempo_filter(ratio: Fraction) -> str:
    """Factor a bounded overall rate into pitch-preserving 0.5..2 stages."""
    if not Fraction(1, 16) <= ratio <= 16:
        raise ValueError("Tempo ratio must be in 1/16..16")
    factors: list[Fraction] = []
    while ratio < Fraction(1, 2):
        factors.append(Fraction(1, 2))
        ratio *= 2
    while ratio > 2:
        factors.append(Fraction(2))
        ratio /= 2
    factors.append(ratio)
    return ",".join(f"atempo={float(factor):.17g}" for factor in factors)


def source_path(value: object, source_root: Path | None) -> Path:
    """Permit a separately authorized read-only source root, never output escape."""
    if source_root is None:
        return _local(value)
    if not source_root.is_absolute():
        raise ValueError("Source root must be absolute")
    root = _checked_output(source_root)
    path = Path(_string(value))
    if not path.is_absolute() or not root.is_dir():
        raise ValueError("Invalid source root or absolute source path")
    path = _checked_output(path)
    if not path.is_relative_to(root) or not path.is_file():
        raise ValueError("Source must be a regular file under the explicit source root")
    return path


@dataclass(frozen=True, slots=True)
class TempoPolicy:
    context_seconds: Fraction = Fraction(1, 2)
    maximum_tail_error_seconds: Fraction = Fraction(1, 10)

    def __post_init__(self) -> None:
        if not Fraction(1, 4) <= self.context_seconds <= 2:
            raise ValueError("Tempo context must be between 0.25 and 2 seconds")
        if not 0 < self.maximum_tail_error_seconds <= self.context_seconds / 2:
            raise ValueError("Tail error budget must be positive and below half the context")


@dataclass(frozen=True, slots=True)
class TempoMeasurement:
    source_start_sample: int
    source_end_sample: int
    output_start_sample: int
    output_end_sample: int
    source_context_samples: int
    synthetic_left_context_samples: int
    synthetic_right_context_samples: int
    filter_output_samples: int
    expected_filter_output_samples: int
    center_crop_start_sample: int
    center_crop_end_sample: int
    tail_error_samples: int
    process_seconds: float
    filter_applied: bool


def tempo_cut(
    reader: wave.Wave_read,
    slot: CutSlot,
    *,
    ffmpeg: Path,
    policy: TempoPolicy,
) -> tuple[bytes, TempoMeasurement]:
    """Keep cut boundaries; reject missing central audio instead of padding it."""
    if len(slot.spans) != 1 or reader.getsampwidth() != 2:
        raise ValueError("Tempo requires one monotone PCM16 source interval")
    low, high = slot.spans[0]
    target = slot.output_end - slot.output_start
    count = high - low
    ratio = Fraction(count, target) if target > 0 else Fraction(0)
    if not 0 <= low < high <= reader.getnframes():
        raise ValueError("Invalid source interval")
    filters = tempo_filter(ratio)
    rate, width = reader.getframerate(), reader.getnchannels() * 2
    context = sample_boundary(policy.context_seconds, rate) if ratio != 1 else 0
    first, last = max(0, low - context), min(reader.getnframes(), high + context)
    left, right = max(0, context - low), max(0, high + context - reader.getnframes())
    if (last - first + left + right) * width > MAX_WINDOW_BYTES:
        raise ValueError("Tempo source window exceeds byte budget")
    reader.setpos(first)
    raw = reader.readframes(last - first)
    if len(raw) != (last - first) * width:
        raise ValueError("Truncated tempo input")
    raw = bytes(left * width) + raw + bytes(right * width)
    started = monotonic()
    if ratio != 1:
        result = subprocess.run(
            [
                str(ffmpeg),
                "-hide_banner",
                "-nostdin",
                "-v",
                "error",
                "-threads",
                "1",
                "-f",
                "s16le",
                "-ar",
                str(rate),
                "-ac",
                str(reader.getnchannels()),
                "-i",
                "pipe:0",
                "-af",
                filters,
                "-c:a",
                "pcm_s16le",
                "-f",
                "s16le",
                "pipe:1",
            ],
            input=raw,
            capture_output=True,
            check=True,
            timeout=PROCESS_TIMEOUT_SECONDS,
        )
        raw = result.stdout
    elapsed = monotonic() - started
    if len(raw) % width or len(raw) > MAX_WINDOW_BYTES * 2:
        raise ValueError("Invalid or excessive tempo output bytes")
    actual = len(raw) // width
    expected = sample_boundary(Fraction(count + 2 * context, 1) / ratio, 1)
    tail_error = actual - expected
    if abs(tail_error) > sample_boundary(policy.maximum_tail_error_seconds, rate):
        raise ValueError(f"Tempo tail discrepancy exceeds explicit budget: {tail_error} samples")
    crop = sample_boundary(Fraction(context, 1) / ratio, 1)
    if crop + target > actual:
        raise ValueError("Tempo filter omitted central audio; no silent padding allowed")
    measured = TempoMeasurement(
        low,
        high,
        slot.output_start,
        slot.output_end,
        context,
        left,
        right,
        actual,
        expected,
        crop,
        crop + target,
        tail_error,
        elapsed,
        ratio != 1,
    )
    return raw[crop * width : (crop + target) * width], measured


def pieces(slot: CutSlot, low: int, high: int, width: int) -> Iterator[CutSlot]:
    """The requested part of a slot, as budget-sized slots on the slot's own source mapping."""
    if len(slot.spans) != 1 or slot.input_samples * width * 2 <= MAX_WINDOW_BYTES:
        yield slot
        return
    ((start, end),) = slot.spans
    target = slot.output_end - slot.output_start
    step = MAX_WINDOW_BYTES // (4 * width)  # leaves the context and a faster rate in budget

    def mapped(at: int) -> int:
        return start + (at - slot.output_start) * (end - start) // target

    for first in range(low, high, step):
        last = min(high, first + step)
        yield replace(
            slot, output_start=first, output_end=last, spans=((mapped(first), mapped(last)),)
        )


def render(
    source: Path,
    output: Path,
    slots: tuple[CutSlot, ...],
    *,
    ffmpeg: Path,
    policy: TempoPolicy,
    output_start: int = 0,
    output_end: int | None = None,
) -> tuple[TempoMeasurement, ...]:
    if not slots or output.exists():
        raise ValueError("Empty slots or existing tempo output")
    end = slots[-1].output_end if output_end is None else output_end
    if not 0 <= output_start < end <= slots[-1].output_end:
        raise ValueError("Invalid tempo output window")
    before = source.stat()
    measurements: list[TempoMeasurement] = []
    with wave.open(str(source), "rb") as reader, output.open("xb") as handle:
        width = reader.getnchannels() * 2
        with wave.open(handle, "wb") as writer:
            writer.setparams(reader.getparams())
            for slot in slots:
                low, high = max(output_start, slot.output_start), min(end, slot.output_end)
                if low >= high:
                    continue
                for piece in pieces(slot, low, high, width):
                    raw, measured = tempo_cut(reader, piece, ffmpeg=ffmpeg, policy=policy)
                    first, last = max(low, piece.output_start), min(high, piece.output_end)
                    writer.writeframesraw(
                        raw[
                            (first - piece.output_start) * width : (last - piece.output_start)
                            * width
                        ]
                    )
                    measurements.append(measured)
    if source.stat() != before:
        raise ValueError("Source changed during tempo rendering")
    return tuple(measurements)


def run(
    config: Path,
    output: Path,
    *,
    sha256: str,
    ffmpeg: Path,
    maximum_rate_change: float,
    policy: TempoPolicy,
    start_seconds: Fraction = Fraction(0),
    end_seconds: Fraction | None = None,
    source_root: Path | None = None,
) -> Path:
    config, output = _local(str(config.absolute())), _local(str(output.absolute()))
    scratch = _local(os.environ.get("TMPDIR"))
    if not scratch.is_dir() or not scratch.is_relative_to(Path.cwd() / "edits/work"):
        raise ValueError("Tempo subprocess scratch must stay under edits/work")
    if output.exists() or not output.is_relative_to(Path.cwd() / "edits/work"):
        raise ValueError("New tempo output must stay under edits/work")
    executable = ffmpeg.resolve(strict=True)
    if not executable.is_file() or not os.access(executable, os.X_OK):
        raise ValueError("Invalid FFmpeg executable")
    data = _object(
        _json(_bound_bytes(config, sha256, MAX_CONFIG_BYTES)),
        "version coordinateSpace source clock intervals evidence",
    )
    if (
        type(data["version"]) is not int
        or data["version"] != 1
        or data["coordinateSpace"] != "native-presentation-seconds"
    ):
        raise ValueError("Invalid native tempo config")
    bindings = {str(config): sha256, str(executable): file_digest(executable)}
    ref = _object(data["source"], "path sha256")
    source, digest = source_path(ref["path"], source_root), _digest(ref["sha256"])
    if not 0 < source.stat().st_size <= MAX_PCM_BYTES or file_digest(source) != digest:
        raise ValueError("Tempo PCM input binding or byte budget mismatch")
    bindings[str(source)] = digest
    clock_path = _pin(data["clock"], bindings)
    clock = AudioFrameClock.from_probe(
        _json(_bound_bytes(clock_path, bindings[str(clock_path)], MAX_CLOCK_BYTES))
    )
    rows, evidence = data["intervals"], data["evidence"]
    if not isinstance(rows, list) or not isinstance(evidence, list) or len(evidence) > 128:
        raise ValueError("Invalid tempo interval or evidence list")
    intervals = tuple(
        (_rational(row["start"]), _rational(row["end"]))
        for row in (_object(item, "start end") for item in rows)
    )
    for ref in evidence:
        _pin(ref, bindings)
    for name in ("clocked_tempo_audio.py", "clocked_cut_audio.py", "audio_clock.py"):
        path = Path(__file__).with_name(name).resolve()
        bindings[str(path)] = file_digest(path)
    with wave.open(str(source), "rb") as reader:
        rate, samples, channels = reader.getframerate(), reader.getnframes(), reader.getnchannels()
    slots = plan_slots(
        clock, intervals, rate=rate, samples=samples, maximum_rate_change=maximum_rate_change
    )
    first = sample_boundary(start_seconds, rate)
    last = slots[-1].output_end if end_seconds is None else sample_boundary(end_seconds, rate)
    output.mkdir(mode=0o700)
    audio = output / "candidate.wav"
    measurements = render(
        source, audio, slots, ffmpeg=executable, policy=policy, output_start=first, output_end=last
    )
    for bound_path, bound_hash in bindings.items():
        if file_digest(Path(bound_path)) != bound_hash:
            raise ValueError("Tempo input changed")
    receipt = output / "receipt.json"
    NativeCaptionLayout._save(
        receipt,
        {
            "version": 1,
            "kind": "context-atempo-exact-cut-slot-candidate",
            "inputs": bindings,
            "output": str(audio),
            "outputSHA256": file_digest(audio),
            "sampleRate": rate,
            "channels": channels,
            "outputStartSample": first,
            "outputEndSample": last,
            "outputSamples": last - first,
            "contextSeconds": str(policy.context_seconds),
            "maximumTailErrorSeconds": str(policy.maximum_tail_error_seconds),
            "maximumRateChangeAllowed": maximum_rate_change,
            "cuts": [asdict(measured) for measured in measurements],
            "pitchPreservingAlgorithm": (
                "FFmpeg atempo with source context and explicit center crop"
            ),
            "tempoInteriorTimingVerified": False,
            "speechQualityVerified": False,
            "AVSyncVerified": False,
            "sourceCorrespondenceVerified": False,
            "deliveryComplete": False,
        },
    )
    return receipt


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("config", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--sha256", required=True)
    parser.add_argument("--ffmpeg", type=Path, required=True)
    parser.add_argument("--maximum-rate-change", type=float, required=True)
    parser.add_argument("--context-seconds", type=Fraction, default=Fraction(1, 2))
    parser.add_argument("--maximum-tail-error-seconds", type=Fraction, default=Fraction(1, 10))
    parser.add_argument("--start-seconds", type=Fraction, default=Fraction(0))
    parser.add_argument("--end-seconds", type=Fraction)
    parser.add_argument("--source-root", type=Path)
    args = parser.parse_args()
    print(
        run(
            args.config,
            args.output,
            sha256=args.sha256,
            ffmpeg=args.ffmpeg,
            maximum_rate_change=args.maximum_rate_change,
            policy=TempoPolicy(args.context_seconds, args.maximum_tail_error_seconds),
            start_seconds=args.start_seconds,
            end_seconds=args.end_seconds,
            source_root=args.source_root,
        )
    )


if __name__ == "__main__":
    main()
