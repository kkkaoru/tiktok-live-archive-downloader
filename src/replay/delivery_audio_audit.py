"""Independent, read-only PCM selection comparison and exact clock diagnostics."""

import argparse
import math
import wave
from bisect import bisect_left, bisect_right
from fractions import Fraction
from itertools import pairwise
from pathlib import Path

from replay.audio_clock import AudioFrameClock
from replay.caption_measured_job import ROOT, _json, _local, _object, _rational
from replay.caption_native_layout import NativeCaptionLayout
from replay.foreground_evidence import _bound_bytes
from replay.recordings import record
from replay.stages import file_digest
from replay.tracking_observations import _digest

MAX_JSON_BYTES = 128 * 1024 * 1024
MAX_FILE_BYTES = 12 * 1024**3
MAX_INTERVALS = 100_000
MAX_BINDINGS = 256
MAX_PIECES = 2_000_000
BLOCK_FRAMES = 65536


def _bind(path: Path, digest: object, bindings: dict[str, str]) -> None:
    expected = _digest(digest)
    if str(path) in bindings and bindings[str(path)] != expected:
        raise ValueError("Conflicting input binding")
    if not path.is_file() or not 0 < path.stat().st_size <= MAX_FILE_BYTES:
        raise ValueError("Input exceeds file budget or is not a file")
    if file_digest(path) != expected:
        raise ValueError("Pinned input changed")
    bindings[str(path)] = expected


def _reference(value: object, bindings: dict[str, str]) -> Path:
    ref = _object(value, "path sha256")
    path = _local(ref["path"])
    _bind(path, ref["sha256"], bindings)
    return path


def _intervals(value: object) -> tuple[tuple[Fraction, Fraction], ...]:
    if not isinstance(value, list) or not 1 <= len(value) <= MAX_INTERVALS:
        raise ValueError("Expected bounded nonempty video intervals")
    result: list[tuple[Fraction, Fraction]] = []
    for item in value:
        raw = _object(item, "start end")
        start, end = _rational(raw["start"]), _rational(raw["end"])
        if not 0 <= start < end <= 21600:
            raise ValueError("Invalid video interval")
        result.append((start, end))
    if any(right[0] < left[1] for left, right in pairwise(result)):
        raise ValueError("Video intervals overlap or reverse")
    return tuple(result)


def _wav(path: Path) -> tuple[int, int, int, int]:
    with wave.open(str(path), "rb") as reader:
        rate, count = reader.getframerate(), reader.getnframes()
        channels, width = reader.getnchannels(), reader.getsampwidth()
        if not (8000 <= rate <= 192000 and count > 0 and 1 <= channels <= 64):
            raise ValueError("Unsupported WAV rate/count/channels")
        if width != 2 or reader.getcomptype() != "NONE":
            raise ValueError("Only uncompressed PCM16 is supported")
        remaining = count
        while remaining:
            size = min(remaining, BLOCK_FRAMES)
            if len(reader.readframes(size)) != size * channels * width:
                raise ValueError("Truncated WAV data")
            remaining -= size
    return rate, count, channels, width


def _clock_diagnostics(
    clock: AudioFrameClock,
    intervals: tuple[tuple[Fraction, Fraction], ...],
    *,
    rate: int,
    samples: int,
) -> tuple[list[tuple[int, int]], list[dict[str, object]], dict[str, object]]:
    """Map frame intersections independently; earlier cuts own duplicate edge samples.

    Offsets are relative to contiguous retained VIDEO time, not an AV-sync verdict.
    Distinct samples in overlapping presentation frames remain distinct. Source-order
    comparison is supported even when a contiguous per-cut output clock is not.
    """
    if abs(Fraction(clock.sample_ends[-1] * rate, clock.sample_rate) - samples) > 1:
        raise ValueError("WAV count does not match decoded clock rate projection")
    starts = tuple(start for start, _ in intervals)
    ends = tuple(end for _, end in intervals)
    pieces: list[list[tuple[Fraction, Fraction, Fraction, Fraction]]] = [[] for _ in intervals]
    spans: list[tuple[int, int]] = []
    owned = [0] * len(intervals)
    rounded = [0] * len(intervals)
    last_owner = -1
    interleaved = False
    consumed = 0
    count = 0
    for pts, size in zip(clock.pts, clock.samples, strict=True):
        first = pts * clock.time_base
        last = first + Fraction(size, clock.sample_rate)
        for index in range(bisect_right(ends, first), bisect_left(starts, last)):
            low, high = max(first, starts[index]), min(last, ends[index])
            origin = Fraction(consumed, clock.sample_rate)
            pieces[index].append(
                (low, high, (origin + low - first) * rate, (origin + high - first) * rate)
            )
            count += 1
            if count > MAX_PIECES:
                raise ValueError("Clock intersection budget exceeded")
            sample_low, sample_high = pieces[index][-1][2:]
            left, right = max(0, math.floor(sample_low)), min(samples, math.ceil(sample_high))
            rounded[index] += max(0, right - left)
            # Traverse decoded source order, not cut order: overlaps can interleave cuts.
            owned_left = max(left, spans[-1][1]) if spans else left
            added = max(0, right - owned_left)
            owned[index] += added
            if spans and left < spans[-1][1] and index < last_owner:
                # Adjacent decoded-frame projections can share one rounded sample.
                shared = max(0, min(right, spans[-1][1]) - left)
                owned[last_owner] -= shared
                owned[index] += shared
                interleaved |= shared > 0
                last_owner = index
            if added:
                interleaved |= index < last_owner
                last_owner = index
            if left < right:
                if spans and left <= spans[-1][1]:
                    spans[-1] = (spans[-1][0], max(spans[-1][1], right))
                else:
                    spans.append((left, right))
        consumed += size
    diagnostics: list[dict[str, object]] = []
    output_samples = 0
    video_time = Fraction(0)
    total_gap = total_overlap = total_exact = Fraction(0)
    for index, ((start, end), intersections) in enumerate(zip(intervals, pieces, strict=True)):
        cursor = start
        gaps: list[list[str]] = []
        exact_samples = Fraction(0)
        mapped = owned[index]
        overlap = Fraction(0)
        overlaps: list[list[str]] = []
        before = output_samples
        for low, high, sample_low, sample_high in intersections:
            if low > cursor:
                gaps.append([str(cursor), str(low)])
            if low < cursor:
                overlaps.append([str(low), str(min(cursor, high))])
                overlap += min(cursor, high) - low
            cursor = max(cursor, high)
            exact_samples += sample_high - sample_low

        if cursor < end:
            gaps.append([str(cursor), str(end)])
        output_samples += mapped
        duration = end - start
        gap = sum((Fraction(b) - Fraction(a) for a, b in gaps), Fraction(0))
        total_gap += gap
        total_overlap += overlap
        total_exact += exact_samples
        diagnostics.append(
            {
                "videoStart": str(start),
                "videoEnd": str(end),
                "videoOutputStart": str(video_time),
                "videoDuration": str(duration),
                "nativeGaps": gaps,
                "nativeGapDuration": str(gap),
                "nativeOverlaps": overlaps,
                "nativeOverlapDuration": str(overlap),
                "exactMappedSamplesBeforeRounding": str(exact_samples),
                "mappedSamples": mapped,
                "mappedDuration": str(Fraction(mapped, rate)),
                "roundingAndDeduplicationSamples": str(mapped - exact_samples),
                "roundingAndClampSamples": str(rounded[index] - exact_samples),
                "deduplicatedSamples": rounded[index] - mapped,
                "outputClockStatus": (
                    "unsupported-interleaved-cut-ownership" if interleaved else "contiguous-cut"
                ),
                "outputStartSample": None if interleaved else before,
                "outputEndSample": None if interleaved else output_samples,
                "outputClockOffsetStart": (
                    None if interleaved else str(Fraction(before, rate) - video_time)
                ),
                "outputClockOffsetEnd": (
                    None
                    if interleaved
                    else str(Fraction(output_samples, rate) - video_time - duration)
                ),
            }
        )
        video_time += duration
    adjustment = output_samples - total_exact
    accounted = total_overlap - total_gap + adjustment / rate
    actual = Fraction(sum(end - start for start, end in spans), rate) - video_time
    return (
        spans,
        diagnostics,
        {
            "nativeGapDuration": str(total_gap),
            "nativeOverlapDuration": str(total_overlap),
            "exactMappedSamplesBeforeRounding": str(total_exact),
            "roundingAndClampSamples": str(sum(rounded) - total_exact),
            "deduplicatedSamples": sum(rounded) - output_samples,
            "roundingAndDeduplicationSamples": str(adjustment),
            "accountedDurationDifference": str(accounted),
            "durationDifferenceResidual": str(actual - accounted),
        },
    )


def _compare(source: Path, selected: Path, spans: list[tuple[int, int]]) -> dict[str, object]:
    compared = mismatches = 0
    first_mismatch: int | None = None
    with wave.open(str(source), "rb") as original, wave.open(str(selected), "rb") as output:
        frame_bytes = original.getnchannels() * original.getsampwidth()
        for start, end in spans:
            original.setpos(start)
            remaining = end - start
            while remaining:
                count = min(remaining, BLOCK_FRAMES)
                left, right = original.readframes(count), output.readframes(count)
                if len(left) != count * frame_bytes or len(right) != len(left):
                    raise ValueError("Truncated PCM during comparison")
                if left != right:
                    mismatches += sum(a != b for a, b in zip(left, right, strict=True))
                    if first_mismatch is None:
                        first_mismatch = compared + next(
                            i for i, (a, b) in enumerate(zip(left, right, strict=True)) if a != b
                        )
                compared += len(left)
                remaining -= count
    return {
        "comparedBytes": compared,
        "mismatchedBytes": mismatches,
        "firstMismatchedOutputByte": first_mismatch,
        "selectedPCMBytesEqual": mismatches == 0,
    }


def audit(
    config: Path, selection: Path, output: Path, *, config_sha256: str, selection_sha256: str
) -> Path:
    """Write a new fact receipt; reject stale pins, unsupported formats and invalid receipts.

    Byte inequality and clock differences are recorded, never repaired. A successful
    call means diagnostics were saved, not that quality or synchronization passed.
    """
    config, selection, output = (
        _local(str(path.absolute())) for path in (config, selection, output)
    )
    if not output.is_relative_to(ROOT / "edits/work"):
        raise ValueError("Audit output must be under checkout edits/work")
    if output.exists():
        raise FileExistsError(output)
    implementation = {
        str(Path(__file__).resolve().with_name(name)): file_digest(
            Path(__file__).resolve().with_name(name)
        )
        for name in ("delivery_audio_audit.py", "audio_clock.py")
    }
    bindings: dict[str, str] = {}
    _bind(config, config_sha256, bindings)
    _bind(selection, selection_sha256, bindings)
    data = _object(
        _json(_bound_bytes(config, config_sha256, MAX_JSON_BYTES)),
        "version coordinateSpace source clock intervals evidence",
    )
    if (
        type(data["version"]) is not int
        or data["version"] != 1
        or data["coordinateSpace"] != "native-presentation-seconds"
    ):
        raise ValueError("Unsupported selection configuration")
    receipt = record(_json(_bound_bytes(selection, selection_sha256, MAX_JSON_BYTES)))
    if type(receipt.get("version")) is not int or receipt["version"] != 1:
        raise ValueError("Unsupported selection receipt")
    original_pins = record(receipt.get("inputs"))
    if not 1 <= len(original_pins) <= MAX_BINDINGS:
        raise ValueError("Invalid receipt input budget")
    for name, digest in original_pins.items():
        _bind(_local(name), digest, bindings)
    source = _reference(data["source"], bindings)
    clock_path = _reference(data["clock"], bindings)
    evidence = data["evidence"]
    if not isinstance(evidence, list) or len(evidence) > 128:
        raise ValueError("Invalid evidence budget")
    for ref in evidence:
        _reference(ref, bindings)
    required = set(bindings) - {str(selection)}
    if any(original_pins.get(name) != bindings[name] for name in required):
        raise ValueError("Selection receipt does not bind configuration inputs")
    selected = _local(receipt.get("output"))
    _bind(selected, receipt.get("outputSHA256"), bindings)
    clock = AudioFrameClock.from_probe(
        _json(_bound_bytes(clock_path, bindings[str(clock_path)], MAX_JSON_BYTES))
    )
    intervals = _intervals(data["intervals"])
    rate, samples, channels, width = _wav(source)
    out_rate, out_samples, out_channels, out_width = _wav(selected)
    if (rate, channels, width) != (out_rate, out_channels, out_width):
        raise ValueError("Source/selected WAV format mismatch")
    spans, diagnostics, accounting = _clock_diagnostics(
        clock, intervals, rate=rate, samples=samples
    )
    count = sum(end - start for start, end in spans)
    if not spans or count != out_samples:
        raise ValueError("Selected WAV count differs from independently mapped samples")
    receipt_spans = receipt.get("spans")
    if not isinstance(receipt_spans, list) or any(
        not isinstance(span, list)
        or len(span) != 2
        or any(type(endpoint) is not int for endpoint in span)
        for span in receipt_spans
    ):
        raise ValueError("Invalid receipt sample spans")
    expected: dict[str, object] = {
        "sourceSamples": samples,
        "selectedSamples": count,
        "sampleRate": rate,
        "channels": channels,
        "sampleWidth": width,
        "spans": [list(span) for span in spans],
        "paddingSamples": 0,
        "resamplingPerformed": False,
        "fadesApplied": False,
    }
    if any(
        type(receipt.get(key)) is not type(value) or receipt.get(key) != value
        for key, value in expected.items()
    ):
        raise ValueError("Selection receipt format/count/spans mismatch")
    duration = sum((end - start for start, end in intervals), Fraction(0))
    clocks = {
        "nativeIntervalDuration": str(duration),
        "audioDuration": str(Fraction(count, rate)),
        "durationDifference": str(Fraction(count, rate) - duration),
    }
    comparison = _compare(source, selected, spans)
    for name, digest in implementation.items():
        _bind(Path(name), digest, bindings)
    for name, digest in bindings.items():
        if file_digest(Path(name)) != digest:
            raise ValueError("Audit input changed during execution")
    NativeCaptionLayout._save(
        output,
        {
            "version": 2,
            "kind": "independent-pcm-selection-clock-audit",
            "comparisonScope": "source-order-union-of-frame-intersections",
            "mappingLimitations": [
                "Overlapping presentation frames may contain distinct decoded samples.",
                "Same-source rounded edges are counted once; earlier cuts own shared edges.",
                "Interleaved cut ownership has no supported contiguous per-cut output clock.",
                "Reassigned rounded edges conservatively invalidate per-cut output clocks.",
                "Native overlap duration counts excess frame multiplicity, not its union.",
                "Rate projection does not establish upstream waveform/clock correspondence.",
            ],
            "inputs": bindings,
            **expected,
            **clocks,
            **comparison,
            "intervals": diagnostics,
            "durationAccounting": accounting,
            "receiptClockDiscrepancies": [
                key for key, value in clocks.items() if receipt.get(key) != value
            ],
            "sourceCorrespondenceVerified": False,
            "AVSyncVerified": False,
            "musicGatePassed": False,
            "speechGatePassed": False,
            "deliveryComplete": False,
        },
    )
    return output


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("config", type=Path)
    parser.add_argument("selection", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--config-sha256", required=True)
    parser.add_argument("--selection-sha256", required=True)
    args = parser.parse_args()
    audit(
        args.config,
        args.selection,
        args.output,
        config_sha256=args.config_sha256,
        selection_sha256=args.selection_sha256,
    )


if __name__ == "__main__":
    main()
