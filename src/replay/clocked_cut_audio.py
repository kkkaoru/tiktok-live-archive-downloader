"""Candidate-only, order-preserving PCM retiming into exact video cut sample slots.

Gap/overlap sample selection follows the supplied decoded-frame clock. Each cut's
selected PCM is linearly resampled in source order, never overlap-mixed. This
preserves cut endpoints, not the native timestamps inside a cut or speech quality.
"""

from __future__ import annotations

import argparse
import math
import wave
from bisect import bisect_right
from dataclasses import dataclass
from fractions import Fraction
from itertools import accumulate, pairwise
from pathlib import Path

import numpy as np
from numpy.typing import NDArray

from replay import tracking_observations as storage
from replay.audio_clock import AudioFrameClock
from replay.caption_measured_job import _json, _local, _object, _pin, _rational
from replay.caption_native_layout import NativeCaptionLayout
from replay.continuous_cut_job import MAX_CLOCK_BYTES, MAX_CONFIG_BYTES, MAX_PCM_BYTES
from replay.foreground_evidence import _bound_bytes
from replay.stages import file_digest

BLOCK = 65536
PCM = NDArray[np.int16]


@dataclass(frozen=True, slots=True)
class CutSlot:
    native_start: Fraction
    native_end: Fraction
    output_start: int
    output_end: int
    spans: tuple[tuple[int, int], ...]

    @property
    def input_samples(self) -> int:
        return sum(end - start for start, end in self.spans)


def sample_boundary(time: Fraction, rate: int) -> int:
    """Nearest absolute sample boundary, half-sample ties toward earlier time."""
    return math.ceil(time * rate - Fraction(1, 2))


def _source_boundary(
    clock: AudioFrameClock,
    starts: tuple[Fraction, ...],
    *,
    time: Fraction,
    rate: int,
    samples: int,
) -> int:
    """Last-started decoded frame owns a native boundary, identically on both sides.

    Gaps clamp to the preceding frame end. An overlap boundary switches to the
    next decoded frame's source position. This is a monotone policy, not proof of
    the native player's overlap policy. Continuous PCM between endpoints remains
    ordered, including earlier-frame tails that fall inside a retained cut.
    """
    index = bisect_right(starts, time) - 1
    if index < 0:
        return 0
    prior = clock.sample_ends[index - 1] if index else 0
    within = min(
        max(Fraction(0), time - starts[index]),
        Fraction(clock.samples[index], clock.sample_rate),
    )
    return min(samples, sample_boundary(Fraction(prior, clock.sample_rate) + within, rate))


def plan_slots(
    clock: AudioFrameClock,
    intervals: tuple[tuple[Fraction, Fraction], ...],
    *,
    rate: int,
    samples: int,
    maximum_rate_change: float,
) -> tuple[CutSlot, ...]:
    """Reject unknown/empty source support and caller-disallowed speed changes."""
    if (
        not 8000 <= rate <= 192000
        or samples <= 0
        or abs(Fraction(clock.sample_ends[-1] * rate, clock.sample_rate) - samples) > 1
        or not math.isfinite(maximum_rate_change)
        or not 0 <= maximum_rate_change < 1
        or not 1 <= len(intervals) <= 100000
        or any(not 0 <= start < end <= 21600 for start, end in intervals)
        or any(right[0] < left[1] for left, right in pairwise(intervals))
    ):
        raise ValueError("Invalid PCM correspondence, cuts or rate-change bound")
    starts = tuple(pts * clock.time_base for pts in clock.pts)
    cursor = Fraction(0)
    slots: list[CutSlot] = []
    for index, (start, end) in enumerate(intervals):
        low = _source_boundary(clock, starts, time=start, rate=rate, samples=samples)
        high = _source_boundary(clock, starts, time=end, rate=rate, samples=samples)
        spans = ((low, high),) if high > low else ()
        first = sample_boundary(cursor, rate)
        cursor += end - start
        slot = CutSlot(start, end, first, sample_boundary(cursor, rate), spans)
        target = slot.output_end - first
        if target <= 0 or not slot.input_samples:
            raise ValueError("Cut has no source samples or no output sample slot")
        if abs(slot.input_samples / target - 1) > maximum_rate_change:
            raise ValueError(
                f"Cut {index} [{start}, {end}) exceeds explicit maximum local rate change: "
                f"{slot.input_samples}/{target} source/output samples"
            )
        slots.append(slot)
    return tuple(slots)


def _read_virtual(reader: wave.Wave_read, slot: CutSlot, start: int, end: int) -> PCM:
    lengths = tuple(accumulate(high - low for low, high in slot.spans))
    index = bisect_right(lengths, start)
    arrays: list[PCM] = []
    cursor = start
    while cursor < end:
        prior = lengths[index - 1] if index else 0
        take = min(end, lengths[index]) - cursor
        reader.setpos(slot.spans[index][0] + cursor - prior)
        raw = reader.readframes(take)
        if len(raw) != take * reader.getnchannels() * 2:
            raise ValueError("Truncated PCM during cut resampling")
        arrays.append(np.frombuffer(raw, dtype="<i2").reshape(take, reader.getnchannels()))
        cursor += take
        index += 1
    return np.concatenate(arrays)


def resampled_block(reader: wave.Wave_read, slot: CutSlot, *, start: int, end: int) -> PCM:
    """Interpolate at sample centers; no cut-dependent accumulated rounding."""
    total = slot.output_end - slot.output_start
    if not 0 <= start < end <= total or end - start > BLOCK:
        raise ValueError("Invalid bounded cut block")
    positions = (np.arange(start, end, dtype=np.float64) + 0.5) * slot.input_samples / total - 0.5
    positions = np.clip(positions, 0, slot.input_samples - 1)
    low = np.floor(positions).astype(np.int64)
    high = np.minimum(low + 1, slot.input_samples - 1)
    first, stop = int(low[0]), int(high[-1]) + 1
    source = _read_virtual(reader, slot, first, stop).astype(np.float64)
    weight = (positions - low)[:, None]
    result = source[low - first] * (1 - weight) + source[high - first] * weight
    return np.asarray(np.rint(result), dtype=np.int16)


def render(
    source: Path,
    output: Path,
    slots: tuple[CutSlot, ...],
    *,
    output_start: int = 0,
    output_end: int | None = None,
) -> int:
    """Write a bounded output window using each full cut's unchanged time mapping."""
    if not slots:
        raise ValueError("Empty cut slots")
    end = slots[-1].output_end if output_end is None else output_end
    if not 0 <= output_start < end <= slots[-1].output_end or output.exists():
        raise ValueError("Invalid output window or existing output")
    before = source.stat()
    with wave.open(str(source), "rb") as reader, output.open("xb") as handle:
        if reader.getsampwidth() != 2 or reader.getcomptype() != "NONE":
            raise ValueError("Clocked cuts require PCM16")
        with wave.open(handle, "wb") as writer:
            writer.setparams(reader.getparams())
            for slot in slots:
                low, high = max(output_start, slot.output_start), min(end, slot.output_end)
                for first in range(low, high, BLOCK):
                    block = resampled_block(
                        reader,
                        slot,
                        start=first - slot.output_start,
                        end=min(first + BLOCK, high) - slot.output_start,
                    )
                    writer.writeframesraw(block.astype("<i2").tobytes())
    if source.stat() != before:
        raise ValueError("Source changed while writing clocked cuts")
    return end - output_start


def run(
    config: Path,
    output: Path,
    *,
    sha256: str,
    maximum_rate_change: float,
    start_seconds: Fraction = Fraction(0),
    end_seconds: Fraction | None = None,
) -> Path:
    """Consume pinned union-selection inputs as a distinct explicit retiming candidate."""
    config, output = _local(str(config.absolute())), _local(str(output.absolute()))
    if output.exists() or not output.is_relative_to(Path.cwd() / "edits/work"):
        raise ValueError("New candidate directory must remain under edits/work")
    data = _object(
        _json(_bound_bytes(config, sha256, MAX_CONFIG_BYTES)),
        "version coordinateSpace source clock intervals evidence",
    )
    if (
        type(data["version"]) is not int
        or data["version"] != 1
        or data["coordinateSpace"] != "native-presentation-seconds"
    ):
        raise ValueError("Expected native presentation selection config")
    bindings = {str(config): sha256}
    source_ref = _object(data["source"], "path sha256")
    source = _local(source_ref["path"])
    source_hash = storage._digest(source_ref["sha256"])
    if not 0 < source.stat().st_size <= MAX_PCM_BYTES or file_digest(source) != source_hash:
        raise ValueError("PCM byte budget or binding mismatch")
    bindings[str(source)] = source_hash
    clock_path = _pin(data["clock"], bindings)
    clock = AudioFrameClock.from_probe(
        _json(_bound_bytes(clock_path, bindings[str(clock_path)], MAX_CLOCK_BYTES))
    )
    raw = data["intervals"]
    evidence = data["evidence"]
    if not isinstance(raw, list) or not isinstance(evidence, list) or len(evidence) > 128:
        raise ValueError("Invalid intervals/evidence")
    intervals = tuple(
        (_rational(row["start"]), _rational(row["end"]))
        for row in (_object(value, "start end") for value in raw)
    )
    for ref in evidence:
        _pin(ref, bindings)
    for name in ("clocked_cut_audio.py", "audio_clock.py"):
        path = Path(__file__).with_name(name).resolve()
        bindings[str(path)] = file_digest(path)
    with wave.open(str(source), "rb") as reader:
        rate, samples = reader.getframerate(), reader.getnframes()
    slots = plan_slots(
        clock, intervals, rate=rate, samples=samples, maximum_rate_change=maximum_rate_change
    )
    first = sample_boundary(start_seconds, rate)
    last = slots[-1].output_end if end_seconds is None else sample_boundary(end_seconds, rate)
    output.mkdir(mode=0o700)
    movie = output / "candidate.wav"
    count = render(source, movie, slots, output_start=first, output_end=last)
    for bound_path, digest in bindings.items():
        if file_digest(Path(bound_path)) != digest:
            raise ValueError("Clocked cut input changed")
    receipt = output / "receipt.json"
    NativeCaptionLayout._save(
        receipt,
        {
            "version": 1,
            "kind": "cut-slot-monotonic-linear-resampling-candidate",
            "inputs": bindings,
            "output": str(movie),
            "outputSHA256": file_digest(movie),
            "sampleRate": rate,
            "outputStartSample": first,
            "outputEndSample": last,
            "outputSamples": count,
            "maximumRateChangeAllowed": maximum_rate_change,
            "maximumRateChangeObserved": max(
                abs(slot.input_samples / (slot.output_end - slot.output_start) - 1)
                for slot in slots
            ),
            "maximumLocalDurationDifferenceSeconds": float(
                max(
                    abs(Fraction(slot.input_samples, rate) - (slot.native_end - slot.native_start))
                    for slot in slots
                )
            ),
            "qualityLimitations": (
                "Local linear resampling changes speech speed and pitch; "
                "not phase-overlap mixing or native-interior-clock preservation"
            ),
            "gapPolicy": "stretch selected ordered PCM inside its cut; no invented silence",
            "boundaryPolicy": (
                "last-started decoded frame owns both sides of each cut boundary; "
                "gaps clamp to previous frame end; nearest shared sample rounding"
            ),
            "overlapPolicy": "retain distinct source samples in order; compress inside cut",
            "cutSlots": [
                {
                    "sourceStart": str(slot.native_start),
                    "sourceEnd": str(slot.native_end),
                    "outputStartSample": slot.output_start,
                    "outputEndSample": slot.output_end,
                    "sourceSamples": slot.input_samples,
                    "rateRatio": str(
                        Fraction(slot.input_samples, slot.output_end - slot.output_start)
                    ),
                    "durationDifferenceBeforeRetiming": str(
                        Fraction(slot.input_samples, rate) - (slot.native_end - slot.native_start)
                    ),
                }
                for slot in slots
            ],
            "nativeInteriorTimestampsPreserved": False,
            "sourceCorrespondenceVerified": False,
            "speechQualityVerified": False,
            "AVSyncVerified": False,
            "deliveryComplete": False,
        },
    )
    return receipt


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("config", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--sha256", required=True)
    parser.add_argument("--maximum-rate-change", required=True, type=float)
    parser.add_argument("--start-seconds", type=Fraction, default=Fraction(0))
    parser.add_argument("--end-seconds", type=Fraction)
    args = parser.parse_args()
    print(
        run(
            args.config,
            args.output,
            sha256=args.sha256,
            maximum_rate_change=args.maximum_rate_change,
            start_seconds=args.start_seconds,
            end_seconds=args.end_seconds,
        )
    )


if __name__ == "__main__":
    main()
