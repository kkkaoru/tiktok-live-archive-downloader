"""Bounded same-mixture PCM clock diagnostics, never an automatic retiming map."""

import argparse
import json
import math
import wave
from dataclasses import asdict, dataclass
from itertools import pairwise
from pathlib import Path

import numpy as np
from numpy.typing import NDArray

from replay.audio_clock import AudioFrameClock
from replay.caption_measured_job import _rational
from replay.clocked_cut_audio import CutSlot, plan_slots
from replay.recordings import record
from replay.stages import file_digest

MAX_ANCHORS = 128
MAX_VALUES = 4_000_000
PCM = NDArray[np.float64]


class AlignmentRejected(ValueError):
    """The waveform does not establish a unique, energetic correspondence."""


@dataclass(frozen=True)
class AlignmentPolicy:
    window_seconds: float = 1.0
    search_seconds: float = 3.0
    minimum_correlation: float = 0.8
    minimum_peak_margin: float = 0.05
    minimum_rms: float = 0.001
    exclusion_seconds: float = 0.02

    def __post_init__(self) -> None:
        values = tuple(asdict(self).values())
        if not all(math.isfinite(value) for value in values) or not (
            0.05 <= self.window_seconds <= 5
            and 0.01 <= self.search_seconds <= 10
            and 0 < self.minimum_correlation <= 1
            and 0 < self.minimum_peak_margin <= 1
            and 0 < self.minimum_rms < 1
            and 0 < self.exclusion_seconds < self.search_seconds
        ):
            raise ValueError("Invalid alignment policy")


@dataclass(frozen=True)
class WindowMatch:
    offset_samples: int
    correlation: float
    runner_up: float
    reference_rms: float
    candidate_rms: float


@dataclass(frozen=True)
class MappedSpan:
    start: int
    end: int
    candidate_start: int

    @property
    def offset(self) -> int:
        return self.candidate_start - self.start


DEFAULT_POLICY = AlignmentPolicy()


def _correlations(reference: PCM, search: PCM) -> tuple[PCM, PCM, float]:
    length, channels = reference.shape
    centered = reference - reference.mean(axis=0)
    energy = float(np.sum(centered**2))
    extent = len(search) - length + 1
    dots = np.zeros(extent, dtype=np.float64)
    energies = np.zeros(extent, dtype=np.float64)
    fft_size = 1 << (len(search) + length - 2).bit_length()
    for channel in range(channels):
        values = search[:, channel]
        convolution = np.fft.irfft(
            np.fft.rfft(values, fft_size) * np.fft.rfft(centered[::-1, channel], fft_size),
            fft_size,
        )
        dots += convolution[length - 1 : len(search)]
        sums = np.concatenate((np.zeros(1), np.cumsum(values)))
        squares = np.concatenate((np.zeros(1), np.cumsum(values**2)))
        energies += (
            squares[length:] - squares[:-length] - (sums[length:] - sums[:-length]) ** 2 / length
        )
    energies = np.maximum(energies, 0)
    denominator = np.sqrt(energy * energies)
    scores = np.zeros(extent, dtype=np.float64)
    np.divide(dots, denominator, out=scores, where=denominator > 0)
    return np.clip(scores, -1, 1), energies, energy


def match_window(
    reference: PCM, search: PCM, *, rate: int, policy: AlignmentPolicy = DEFAULT_POLICY
) -> WindowMatch:
    """Find a sample position using per-channel zero-mean normalized correlation."""
    if (
        type(rate) is not int
        or not 8000 <= rate <= 192000
        or reference.ndim != 2
        or search.ndim != 2
        or reference.shape[1] != search.shape[1]
        or not 1 <= reference.shape[1] <= 8
        or not 2 <= len(reference) < len(search)
        or reference.size > MAX_VALUES
        or search.size > MAX_VALUES
        or not np.all(np.isfinite(reference))
        or not np.all(np.isfinite(search))
    ):
        raise ValueError("Invalid bounded PCM windows")
    scores, energies, energy = _correlations(reference, search)
    reference_rms = math.sqrt(energy / reference.size)
    if reference_rms < policy.minimum_rms:
        raise AlignmentRejected("reference-low-energy")
    eligible = energies >= policy.minimum_rms**2 * reference.size
    scores = np.where(eligible, scores, -1.0)
    peak = int(np.argmax(scores))
    correlation = float(scores[peak])
    if correlation < policy.minimum_correlation:
        raise AlignmentRejected("low-correlation")
    if peak in (0, len(scores) - 1):
        raise AlignmentRejected("search-boundary")
    exclusion = max(1, math.ceil(rate * policy.exclusion_seconds))
    alternative = scores.copy()
    alternative[max(0, peak - exclusion) : peak + exclusion + 1] = -1
    runner_up = float(np.max(alternative))
    if correlation - runner_up < policy.minimum_peak_margin:
        raise AlignmentRejected("ambiguous-match")
    return WindowMatch(
        peak,
        correlation,
        runner_up,
        reference_rms,
        math.sqrt(float(energies[peak]) / reference.size),
    )


def _format(stream: wave.Wave_read) -> tuple[int, int, int]:
    if stream.getsampwidth() != 2 or stream.getcomptype() != "NONE":
        raise ValueError("Alignment requires PCM16 WAV")
    return stream.getframerate(), stream.getnchannels(), stream.getnframes()


def _read_window(stream: wave.Wave_read, *, start: int, count: int) -> PCM:
    channels = stream.getnchannels()
    if (
        start < 0
        or count < 2
        or start + count > stream.getnframes()
        or count * channels > MAX_VALUES
    ):
        raise AlignmentRejected("unsupported-window")
    stream.setpos(start)
    data = stream.readframes(count)
    if len(data) != count * channels * 2:
        raise ValueError("Truncated PCM input")
    return np.frombuffer(data, dtype="<i2").reshape(count, channels).astype(np.float64) / 32768


def _anchor(
    reference: wave.Wave_read,
    candidate: wave.Wave_read,
    *,
    centers: tuple[float, float],
    policy: AlignmentPolicy,
) -> dict[str, object]:
    rate = reference.getframerate()
    length = round(policy.window_seconds * rate)
    radius = round(policy.search_seconds * rate)
    first = round(centers[0] * rate) - length // 2
    search_start = round(centers[1] * rate) - length // 2 - radius
    row: dict[str, object] = {
        "referenceCenterSeconds": centers[0],
        "predictedCandidateCenterSeconds": centers[1],
    }
    try:
        match = match_window(
            _read_window(reference, start=first, count=length),
            _read_window(candidate, start=search_start, count=length + 2 * radius),
            rate=rate,
            policy=policy,
        )
    except AlignmentRejected as error:
        return {**row, "status": "rejected", "reason": str(error)}
    matched = search_start + match.offset_samples
    return {
        **row,
        "status": "matched",
        "referenceStartSample": first,
        "candidateStartSample": matched,
        "candidateMinusReferenceSamples": matched - first,
        "candidateMinusReferenceSeconds": (matched - first) / rate,
        **asdict(match),
    }


def diagnose(
    reference: Path,
    candidate: Path,
    output: Path,
    *,
    anchors: tuple[tuple[float, float], ...],
    reference_sha256: str,
    candidate_sha256: str,
    policy: AlignmentPolicy = DEFAULT_POLICY,
) -> None:
    """Save observations and refusals; successful anchors are not interpolation approval."""
    root = Path.cwd().resolve()
    paths = (reference.resolve(), candidate.resolve(), output.resolve())
    if not all(path.is_relative_to(root) for path in paths) or not paths[2].is_relative_to(
        root / "edits/work"
    ):
        raise ValueError("Alignment paths must stay inside checkout; output under edits/work")
    if (
        not 1 <= len(anchors) <= MAX_ANCHORS
        or any(not all(math.isfinite(value) and value >= 0 for value in pair) for pair in anchors)
        or any(right[0] <= left[0] for left, right in pairwise(anchors))
    ):
        raise ValueError("Invalid ordered alignment anchors")
    if paths[0] == paths[1] and reference_sha256 != candidate_sha256:
        raise ValueError("Conflicting alignment input hashes")
    pins = {str(paths[0]): reference_sha256, str(paths[1]): candidate_sha256}
    if any(file_digest(Path(path)) != digest for path, digest in pins.items()):
        raise ValueError("Alignment input hash mismatch")
    with wave.open(str(paths[0]), "rb") as left, wave.open(str(paths[1]), "rb") as right:
        left_format, right_format = _format(left), _format(right)
        if left_format[:2] != right_format[:2]:
            raise ValueError("PCM sample rate/channels differ")
        observations = [_anchor(left, right, centers=pair, policy=policy) for pair in anchors]
    positions = [row["candidateStartSample"] for row in observations if row["status"] == "matched"]
    monotonic = all(
        isinstance(a, int) and isinstance(b, int) and b > a for a, b in pairwise(positions)
    )
    if any(file_digest(Path(path)) != digest for path, digest in pins.items()):
        raise ValueError("Alignment input changed during diagnosis")
    result = {
        "version": 1,
        "inputs": pins,
        "codeSHA256": file_digest(Path(__file__)),
        "policy": asdict(policy),
        "sampleRate": left_format[0],
        "channels": left_format[1],
        "anchors": observations,
        "monotonic": monotonic,
        "allAnchorsMatched": all(row["status"] == "matched" for row in observations),
        "mappingAccepted": False,
        "interpolationVerified": False,
        "rejection": None if monotonic else "non-monotonic-correspondence",
    }
    with paths[2].open("x", encoding="utf-8") as stream:
        json.dump(result, stream, indent=2)
        stream.write("\n")


def _parse_anchor(text: str) -> tuple[float, float]:
    parts = text.split(":")
    if len(parts) != 2:
        raise argparse.ArgumentTypeError("Anchor requires reference:candidate seconds")
    return float(parts[0]), float(parts[1])


def merge_support(spans: tuple[MappedSpan, ...]) -> tuple[MappedSpan, ...]:
    """Only measured, mutually consistent spans can form usable correspondence."""
    result: list[MappedSpan] = []
    for span in sorted(spans, key=lambda item: item.start):
        if not 0 <= span.start < span.end or span.candidate_start < 0:
            raise ValueError("Invalid correspondence support")
        if not result:
            result.append(span)
            continue
        previous = result[-1]
        if span.start < previous.end and span.offset != previous.offset:
            raise AlignmentRejected("inconsistent-overlapping-support")
        if span.start <= previous.end and span.offset == previous.offset:
            result[-1] = MappedSpan(
                previous.start, max(previous.end, span.end), previous.candidate_start
            )
        elif span.candidate_start < previous.end + previous.offset:
            raise AlignmentRejected("non-monotonic-support")
        else:
            result.append(span)
    return tuple(result)


def cut_support(slot: CutSlot, spans: tuple[MappedSpan, ...]) -> dict[str, object]:
    """Expose measured new-PCM pieces and exact unknown gaps, without extrapolation."""
    spans = merge_support(spans)
    mappings: list[dict[str, int]] = []
    unknown: list[list[int]] = []
    for start, end in slot.spans:
        cursor = start
        for span in spans:
            low, high = max(start, span.start), min(end, span.end)
            if low >= high:
                continue
            if cursor < low:
                unknown.append([cursor, low])
            mappings.append(
                {
                    "referenceStart": low,
                    "referenceEnd": high,
                    "candidateStart": low + span.offset,
                    "candidateEnd": high + span.offset,
                }
            )
            cursor = high
        if cursor < end:
            unknown.append([cursor, end])
    return {
        "outputStartSample": slot.output_start,
        "outputEndSample": slot.output_end,
        "nativeStart": str(slot.native_start),
        "nativeEnd": str(slot.native_end),
        "mappings": mappings,
        "unknownReferenceSpans": unknown,
        "fullyMeasured": bool(mappings) and not unknown,
    }


def _read_record(path: Path) -> dict[str, object]:
    if not 0 < path.stat().st_size <= 64 * 1024**2:
        raise ValueError("Alignment document exceeds budget")
    return record(json.loads(path.read_text(encoding="utf-8")))


def _seed_support(
    paths: tuple[Path, ...],
    *,
    pins: dict[str, str],
    rate: int,
    reference_samples: int,
    candidate_samples: int,
) -> tuple[MappedSpan, ...]:
    result: list[MappedSpan] = []
    if not 1 <= len(paths) <= 16:
        raise ValueError("Expected bounded seed reports")
    for path in paths:
        raw = _read_record(path)
        if raw.get("kind") == "selected-pcm-correspondence-diagnostic":
            result.extend(
                _resumed_support(
                    raw,
                    pins=pins,
                    rate=rate,
                    reference_samples=reference_samples,
                    candidate_samples=candidate_samples,
                )
            )
            continue
        if (
            raw.get("inputs") != pins
            or raw.get("sampleRate") != rate
            or raw.get("monotonic") is not True
        ):
            raise ValueError("Seed report input/clock mismatch")
        settings = record(raw.get("policy"))
        if set(settings) != set(asdict(DEFAULT_POLICY)) or not all(
            isinstance(value, (int, float)) and not isinstance(value, bool)
            for value in settings.values()
        ):
            raise ValueError("Invalid seed measurement policy")
        policy = AlignmentPolicy(
            **{
                key: float(value)
                for key, value in settings.items()
                if isinstance(value, (int, float)) and not isinstance(value, bool)
            }
        )
        rows = raw.get("anchors")
        if not isinstance(rows, list) or len(rows) > MAX_ANCHORS:
            raise ValueError("Invalid seed anchors")
        result.extend(
            _seed_rows(
                rows,
                policy=policy,
                rate=rate,
                reference_samples=reference_samples,
                candidate_samples=candidate_samples,
            )
        )
    if not result:
        raise ValueError("No measured seed correspondence")
    return merge_support(tuple(result))


def _resumed_support(
    raw: dict[str, object],
    *,
    pins: dict[str, str],
    rate: int,
    reference_samples: int,
    candidate_samples: int,
) -> tuple[MappedSpan, ...]:
    bindings = record(raw.get("inputs"))
    values = raw.get("measuredSupport")
    if (
        any(bindings.get(path) != digest for path, digest in pins.items())
        or raw.get("sampleRate") != rate
        or raw.get("mappingAccepted") is not False
        or raw.get("interpolationOutsideMeasuredWindows") is not False
        or not isinstance(values, list)
        or len(values) > 20000
    ):
        raise ValueError("Invalid resumed measurement report")
    result: list[MappedSpan] = []
    for value in values:
        row = record(value)
        start, end, target = row.get("start"), row.get("end"), row.get("candidate_start")
        if (
            type(start) is not int
            or type(end) is not int
            or type(target) is not int
            or not 0 <= start < end <= reference_samples
            or not 0 <= target < target + end - start <= candidate_samples
        ):
            raise ValueError("Invalid resumed support bounds")
        result.append(MappedSpan(start, end, target))
    return merge_support(tuple(result))


def _seed_rows(
    rows: list[object],
    *,
    policy: AlignmentPolicy,
    rate: int,
    reference_samples: int,
    candidate_samples: int,
) -> list[MappedSpan]:
    result: list[MappedSpan] = []
    for value in rows:
        row = record(value)
        if row.get("status") != "matched":
            continue
        start, target = row.get("referenceStartSample"), row.get("candidateStartSample")
        numeric = tuple(
            row.get(key) for key in ("correlation", "runner_up", "reference_rms", "candidate_rms")
        )
        if (
            type(start) is not int
            or type(target) is not int
            or not all(
                isinstance(number, (int, float))
                and not isinstance(number, bool)
                and math.isfinite(number)
                for number in numeric
            )
        ):
            raise ValueError("Invalid measured seed")
        correlation, runner, left_rms, right_rms = (float(str(number)) for number in numeric)
        length = round(policy.window_seconds * rate)
        if not (
            0 <= start < start + length <= reference_samples
            and 0 <= target < target + length <= candidate_samples
            and policy.minimum_correlation <= correlation <= 1
            and correlation - runner >= policy.minimum_peak_margin
            and min(left_rms, right_rms) >= policy.minimum_rms
        ):
            raise ValueError("Seed does not establish waveform support")
        result.append(MappedSpan(start, start + length, target))
    return result


def _scan_piece(
    left: wave.Wave_read,
    right: wave.Wave_read,
    *,
    start: int,
    offset: int,
    policy: AlignmentPolicy,
) -> tuple[MappedSpan, WindowMatch]:
    rate = left.getframerate()
    count = round(policy.window_seconds * rate)
    start = min(start, left.getnframes() - count)
    radius = round(policy.search_seconds * rate)
    search_start = max(0, start + offset - radius)
    search_end = min(right.getnframes(), start + offset + count + radius)
    match = match_window(
        _read_window(left, start=start, count=count),
        _read_window(right, start=search_start, count=search_end - search_start),
        rate=rate,
        policy=policy,
    )
    return MappedSpan(start, start + count, search_start + match.offset_samples), match


def scan_required_support(
    left: wave.Wave_read,
    right: wave.Wave_read,
    *,
    slots: tuple[CutSlot, ...],
    seeds: tuple[MappedSpan, ...],
    maximum_windows: int,
    policy: AlignmentPolicy = DEFAULT_POLICY,
) -> tuple[tuple[MappedSpan, ...], list[dict[str, object]]]:
    """Verify needed cut spans only; nearest seed predicts search, never acceptance."""
    if not 1 <= maximum_windows <= 20000:
        raise ValueError("Invalid alignment scan budget")
    support = merge_support(seeds)
    if not support:
        raise ValueError("No alignment seeds")
    observations: list[dict[str, object]] = []
    count = round(policy.window_seconds * left.getframerate())
    for slot in slots:
        support, measured = _scan_slot(
            left,
            right,
            slot=slot,
            support=support,
            budget=maximum_windows - len(observations),
            policy=policy,
            count=count,
        )
        observations.extend(measured)
        if len(observations) == maximum_windows:
            break
    return support, observations


def context_slots(
    slots: tuple[CutSlot, ...], *, samples: int, context_samples: int
) -> tuple[CutSlot, ...]:
    """Request real filter context separately; never manufacture source-edge support."""
    if samples <= 0 or context_samples < 0:
        raise ValueError("Invalid context sample bounds")
    result: list[CutSlot] = []
    for slot in slots:
        if len(slot.spans) != 1:
            raise ValueError("Context requires one contiguous source interval")
        low, high = slot.spans[0]
        if not 0 <= low < high <= samples:
            raise ValueError("Context source interval out of bounds")
        margin = context_samples if slot.input_samples != slot.output_end - slot.output_start else 0
        result.append(
            CutSlot(
                slot.native_start,
                slot.native_end,
                slot.output_start,
                slot.output_end,
                ((max(0, low - margin), min(samples, high + margin)),),
            )
        )
    return tuple(result)


def _scan_slot(
    left: wave.Wave_read,
    right: wave.Wave_read,
    *,
    slot: CutSlot,
    support: tuple[MappedSpan, ...],
    budget: int,
    policy: AlignmentPolicy,
    count: int,
) -> tuple[tuple[MappedSpan, ...], list[dict[str, object]]]:
    observations: list[dict[str, object]] = []
    for start, end in slot.spans:
        cursor = start
        while cursor < end and len(observations) < budget:
            covered = next((span for span in support if span.start <= cursor < span.end), None)
            if covered is not None:
                cursor = covered.end
                continue
            nearest = min(
                support, key=lambda span: min(abs(cursor - span.start), abs(cursor - span.end))
            )
            row: dict[str, object] = {"requestedReferenceStartSample": cursor}
            try:
                span, match = _scan_piece(
                    left, right, start=cursor, offset=nearest.offset, policy=policy
                )
                support = merge_support((*support, span))
                row.update(
                    {"status": "matched", "support": asdict(span), "measurement": asdict(match)}
                )
            except AlignmentRejected as error:
                row.update({"status": "rejected", "reason": str(error)})
            observations.append(row)
            cursor += count
    return support, observations


def _selection_slots(
    selection: Path,
    *,
    sha256: str,
    reference: Path,
    reference_sha256: str,
    rate: int,
    samples: int,
    maximum_rate_change: float,
) -> tuple[tuple[CutSlot, ...], dict[str, str]]:
    if file_digest(selection) != sha256:
        raise ValueError("Selection hash mismatch")
    raw = _read_record(selection)
    source, clock_ref = record(raw.get("source")), record(raw.get("clock"))
    if (
        raw.get("version") != 1
        or raw.get("coordinateSpace") != "native-presentation-seconds"
        or source != {"path": str(reference), "sha256": reference_sha256}
    ):
        raise ValueError("Selection must bind the reference PCM clock")
    clock_name, digest = clock_ref.get("path"), clock_ref.get("sha256")
    if not isinstance(clock_name, str) or not isinstance(digest, str):
        raise ValueError("Invalid clock reference")
    clock_path = Path(clock_name).resolve()
    if not clock_path.is_relative_to(Path.cwd().resolve()) or file_digest(clock_path) != digest:
        raise ValueError("Clock binding mismatch")
    clock = AudioFrameClock.from_probe(_read_record(clock_path))
    values = raw.get("intervals")
    if not isinstance(values, list):
        raise ValueError("Invalid selection intervals")
    intervals = tuple(
        (_rational(record(row).get("start")), _rational(record(row).get("end"))) for row in values
    )
    return (
        plan_slots(
            clock, intervals, rate=rate, samples=samples, maximum_rate_change=maximum_rate_change
        ),
        {str(selection): sha256, str(clock_path): digest},
    )


def map_selection(
    reference: Path,
    candidate: Path,
    output: Path,
    *,
    selection: Path,
    selection_sha256: str,
    seed_reports: tuple[Path, ...],
    reference_sha256: str,
    candidate_sha256: str,
    maximum_windows: int,
    maximum_rate_change: float,
    policy: AlignmentPolicy = DEFAULT_POLICY,
    context_seconds: float = 0,
) -> None:
    """Construct measured new-PCM pieces for selected cuts; preserve every unknown."""
    root = Path.cwd().resolve()
    reference, candidate, output, selection = (
        path.resolve() for path in (reference, candidate, output, selection)
    )
    if not all(
        path.resolve().is_relative_to(root)
        for path in (reference, candidate, selection, *seed_reports)
    ) or not output.is_relative_to(root / "edits/work"):
        raise ValueError("Mapping paths must stay inside checkout")
    pins = {str(reference): reference_sha256, str(candidate): candidate_sha256}
    if reference == candidate or any(
        file_digest(Path(path)) != digest for path, digest in pins.items()
    ):
        raise ValueError("Invalid mapping PCM bindings")
    with wave.open(str(reference), "rb") as left, wave.open(str(candidate), "rb") as right:
        reference_format, candidate_format = _format(left), _format(right)
        if reference_format[:2] != candidate_format[:2]:
            raise ValueError("Mapping PCM formats differ")
        rate, channels, samples = reference_format
        seeds = _seed_support(
            seed_reports,
            pins=pins,
            rate=rate,
            reference_samples=samples,
            candidate_samples=candidate_format[2],
        )
        slots, bindings = _selection_slots(
            selection,
            sha256=selection_sha256,
            reference=reference,
            reference_sha256=reference_sha256,
            rate=rate,
            samples=samples,
            maximum_rate_change=maximum_rate_change,
        )
        pins.update(bindings)
        pins.update({str(path.resolve()): file_digest(path) for path in seed_reports})
        if not math.isfinite(context_seconds) or not 0 <= context_seconds <= 2:
            raise ValueError("Invalid context duration")
        requested = context_slots(
            slots, samples=samples, context_samples=round(context_seconds * rate)
        )
        support, observations = scan_required_support(
            left,
            right,
            slots=requested,
            seeds=seeds,
            maximum_windows=maximum_windows,
            policy=policy,
        )
    if any(file_digest(Path(path)) != digest for path, digest in pins.items()):
        raise ValueError("Mapping inputs changed")
    cuts = [cut_support(slot, support) for slot in slots]
    result = {
        "version": 1,
        "kind": "selected-pcm-correspondence-diagnostic",
        "inputs": pins,
        "codeSHA256": file_digest(Path(__file__)),
        "sampleRate": rate,
        "channels": channels,
        "policy": asdict(policy),
        "maximumWindows": maximum_windows,
        "maximumRateChange": maximum_rate_change,
        "observations": observations,
        "measuredSupport": [asdict(span) for span in support],
        "cuts": cuts,
        "contextSeconds": context_seconds,
        "contextCuts": [cut_support(slot, support) for slot in requested],
        "fullyMeasuredCuts": sum(row["fullyMeasured"] is True for row in cuts),
        "allCutsMeasured": all(row["fullyMeasured"] is True for row in cuts),
        "mappingAccepted": False,
        "productionAudioModified": False,
        "interpolationOutsideMeasuredWindows": False,
    }
    with output.open("x", encoding="utf-8") as stream:
        json.dump(result, stream, indent=2)
        stream.write("\n")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("reference", type=Path)
    parser.add_argument("candidate", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--reference-sha256", required=True)
    parser.add_argument("--candidate-sha256", required=True)
    parser.add_argument("--anchor", action="append", type=_parse_anchor)
    parser.add_argument("--selection", type=Path)
    parser.add_argument("--selection-sha256")
    parser.add_argument("--seed-report", action="append", type=Path)
    parser.add_argument("--maximum-windows", type=int, default=128)
    parser.add_argument("--maximum-rate-change", type=float, default=0.5)
    parser.add_argument("--context-seconds", type=float, default=0)
    args = parser.parse_args()
    if args.selection is not None:
        if args.selection_sha256 is None or not args.seed_report:
            parser.error("Selection mapping requires its hash and seed reports")
        map_selection(
            args.reference,
            args.candidate,
            args.output,
            selection=args.selection,
            selection_sha256=args.selection_sha256,
            seed_reports=tuple(args.seed_report),
            reference_sha256=args.reference_sha256,
            candidate_sha256=args.candidate_sha256,
            maximum_windows=args.maximum_windows,
            maximum_rate_change=args.maximum_rate_change,
            context_seconds=args.context_seconds,
        )
        return
    if not args.anchor:
        parser.error("Expected anchors or a selection")
    diagnose(
        args.reference,
        args.candidate,
        args.output,
        anchors=tuple(args.anchor),
        reference_sha256=args.reference_sha256,
        candidate_sha256=args.candidate_sha256,
    )


if __name__ == "__main__":
    main()
