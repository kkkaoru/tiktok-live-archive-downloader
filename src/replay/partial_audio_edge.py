"""Plan explicit sub-frame audio absence without dropping recognition or stretching it.

This is an opt-in consumer contract. Existing whole-frame support and tempo
receipts cannot certify these plans. Silence represents only measured outer
audio absence; internal timestamp gaps keep the existing decoded-PCM policy.
"""

import argparse
import json
import os
import wave
from bisect import bisect_right
from dataclasses import asdict, dataclass, replace
from fractions import Fraction
from math import ceil, floor
from pathlib import Path

from replay.audio_clock import AudioFrameClock
from replay.caption_consumer_binding import _path, _read
from replay.caption_projection import CutSpan
from replay.caption_repair import ClockLabel
from replay.clocked_cut_audio import CutSlot, plan_slots, sample_boundary
from replay.clocked_tempo_audio import TempoMeasurement, TempoPolicy, tempo_cut
from replay.continuous_quiet_plan import _labels
from replay.core import ReplayError
from replay.cut_audio_support import MAX_CUTS, MAX_LABELS, _cuts, _protect_labels
from replay.input_hash_snapshot import InputHashSnapshot
from replay.recordings import record
from replay.stages import file_digest

KIND = "partial-audio-edge-supported-tempo-candidate-v1"
POLICY = "decoded-frame-order-preserving-known-gaps"


@dataclass(frozen=True, slots=True)
class PartialEdgePlan:
    cuts: tuple[CutSpan, ...]
    removed: tuple[tuple[int, int], ...]
    slots: tuple[CutSlot, ...]
    silence: tuple[tuple[int, int], ...]
    boundary_errors: tuple[Fraction, ...]
    source_start: Fraction
    source_end: Fraction
    output_samples: int
    rate: int
    source_samples: int


def _retained(
    cuts: tuple[CutSpan, ...], *, first: int, end: int
) -> tuple[tuple[CutSpan, ...], tuple[tuple[int, int], ...]]:
    result: list[CutSpan] = []
    removed: list[tuple[int, int]] = []
    source_end = old_offset = offset = 0
    for cut in cuts:
        if (
            any(
                type(value) is not int
                for value in (cut.source_start_frame, cut.source_end_frame, cut.output_start_frame)
            )
            or cut.source_start_frame < source_end
            or cut.output_start_frame != old_offset
        ):
            raise ReplayError("Expected ordered disjoint contiguous candidate cuts")
        source_end = cut.source_end_frame
        old_offset += cut.duration_frames
        low, high = max(first, cut.source_start_frame), min(end, cut.source_end_frame)
        if low >= high:
            removed.append((cut.source_start_frame, cut.source_end_frame))
            continue
        if cut.source_start_frame < low:
            removed.append((cut.source_start_frame, low))
        if high < cut.source_end_frame:
            removed.append((high, cut.source_end_frame))
        result.append(CutSpan(low, high, offset))
        offset += high - low
    if not result:
        raise ReplayError("No retained video cell intersects actual audio")
    return tuple(result), tuple(removed)


def plan(
    cuts: tuple[CutSpan, ...],
    labels: tuple[ClockLabel, ...],
    clock: AudioFrameClock,
    *,
    fps: int,
    rate: int,
    samples: int,
    maximum_rate_change: float,
) -> PartialEdgePlan:
    """Preserve all retained anchors and expose exact sample-grid rounding errors."""
    if type(fps) is not int or not 1 <= fps <= 120 or not 1 <= len(cuts) <= MAX_CUTS:
        raise ReplayError("Invalid partial-edge frame rate or cut count")
    if len(labels) > MAX_LABELS:
        raise ReplayError("Partial-edge recognition exceeds budget")
    start = clock.pts[0] * clock.time_base
    end = clock.pts[-1] * clock.time_base + Fraction(clock.samples[-1], clock.sample_rate)
    retained, removed = _retained(cuts, first=floor(start * fps), end=ceil(end * fps))
    _protect_labels(removed, labels, fps, (start, end))
    ends = tuple(Fraction(c.source_end_frame, fps) for c in retained)
    for label in labels:
        label_start, label_end = Fraction(str(label.start)), Fraction(str(label.end))
        index = bisect_right(ends, label_start)
        if (
            index < len(retained)
            and label_end >= Fraction(retained[index].source_start_frame, fps)
            and (label_start < start or label_end > end)
        ):
            raise ReplayError("Recognition claims speech outside measured audio support")
    intervals = tuple(
        (
            max(start, Fraction(c.source_start_frame, fps)),
            min(end, Fraction(c.source_end_frame, fps)),
        )
        for c in retained
    )
    # Rate guards apply to genuinely supported durations, not the absent edge.
    supported = plan_slots(
        clock, intervals, rate=rate, samples=samples, maximum_rate_change=maximum_rate_change
    )
    slots: list[CutSlot] = []
    silence: list[tuple[int, int]] = []
    errors: list[Fraction] = []
    for cut, slot in zip(retained, supported, strict=True):
        origin = Fraction(cut.output_start_frame, fps)
        native = Fraction(cut.source_start_frame, fps)
        exact_low = (origin + slot.native_start - native) * rate
        exact_high = (origin + slot.native_end - native) * rate
        low, high = (
            sample_boundary(exact_low / rate, rate),
            sample_boundary(exact_high / rate, rate),
        )
        outer_low = sample_boundary(origin, rate)
        outer_high = sample_boundary(origin + Fraction(cut.duration_frames, fps), rate)
        if low > outer_low:
            silence.append((outer_low, low))
        if high < outer_high:
            silence.append((high, outer_high))
        if high <= low or abs(slot.input_samples / (high - low) - 1) > maximum_rate_change:
            raise ReplayError("Quantized supported audio exceeds explicit rate guard")
        slots.append(replace(slot, output_start=low, output_end=high))
        errors.extend((Fraction(low) - exact_low, Fraction(high) - exact_high))
    return PartialEdgePlan(
        retained,
        removed,
        tuple(slots),
        tuple(silence),
        tuple(errors),
        start,
        end,
        sample_boundary(Fraction(sum(c.duration_frames for c in retained), fps), rate),
        rate,
        samples,
    )


def render(
    source: Path, output: Path, prepared: PartialEdgePlan, *, ffmpeg: Path, policy: TempoPolicy
) -> tuple[TempoMeasurement, ...]:
    """Render real supported PCM plus explicitly planned outer absence only.

    The caller must pin the source, clock, cuts, labels and this exact plan and
    use a partial-edge-aware receipt verifier; legacy tempo receipts are invalid.
    """
    spans = sorted((*prepared.silence, *((s.output_start, s.output_end) for s in prepared.slots)))
    cursor = 0
    for low, high in spans:
        if low != cursor or high <= low:
            raise ReplayError("Partial-edge spans overlap or leave an unexplained hole")
        cursor = high
    if not prepared.slots or cursor != prepared.output_samples:
        raise ReplayError("Partial-edge output sample extent differs")
    before = source.stat()
    measured: list[TempoMeasurement] = []
    with wave.open(str(source), "rb") as reader:
        if (reader.getframerate(), reader.getnframes(), reader.getsampwidth()) != (
            prepared.rate,
            prepared.source_samples,
            2,
        ):
            raise ReplayError("Partial-edge source PCM identity differs")
        width = reader.getnchannels() * 2
        with output.open("xb") as handle, wave.open(handle, "wb") as writer:
            writer.setparams(reader.getparams())
            cursor = 0
            for slot in prepared.slots:
                writer.writeframesraw(bytes((slot.output_start - cursor) * width))
                raw, measurement = tempo_cut(reader, slot, ffmpeg=ffmpeg, policy=policy)
                if len(raw) != (slot.output_end - slot.output_start) * width:
                    raise ReplayError("Supported tempo output has missing samples")
                writer.writeframesraw(raw)
                measured.append(measurement)
                cursor = slot.output_end
            writer.writeframesraw(bytes((prepared.output_samples - cursor) * width))
    if source.stat() != before:
        raise ReplayError("Partial-edge source changed during rendering")
    return tuple(measured)


def _payload(prepared: PartialEdgePlan) -> dict[str, object]:
    return {
        "cuts": [asdict(cut) for cut in prepared.cuts],
        "removed": [list(span) for span in prepared.removed],
        "slots": [
            {
                "sourceStart": str(slot.native_start),
                "sourceEnd": str(slot.native_end),
                "outputStart": slot.output_start,
                "outputEnd": slot.output_end,
                "sourceSpans": [list(span) for span in slot.spans],
            }
            for slot in prepared.slots
        ],
        "knownAbsentSamples": [list(span) for span in prepared.silence],
        "boundaryErrorsSamples": [str(error) for error in prepared.boundary_errors],
        "sourceStart": str(prepared.source_start),
        "sourceEnd": str(prepared.source_end),
        "outputSamples": prepared.output_samples,
        "sampleRate": prepared.rate,
        "sourceSamples": prepared.source_samples,
    }


def _load(
    config: Path,
) -> tuple[PartialEdgePlan, dict[str, Path], dict[str, object], InputHashSnapshot]:
    config = _path(str(config.absolute()))
    data = _read(config)
    if (
        set(data)
        != {
            "version",
            "kind",
            "artifacts",
            "dependencies",
            "maximumRateChange",
            "internalClockPolicy",
        }
        or data.get("version") != 1
        or data.get("kind") != KIND
        or data.get("internalClockPolicy") != POLICY
    ):
        raise ReplayError("Explicit partial-edge schema and known decoded-clock policy required")
    refs = record(data.get("artifacts"))
    if set(refs) != {"cuts", "labels", "audioClock", "source", "ffmpeg"}:
        raise ReplayError("Missing partial-edge input roles")
    paths = {name: _path(record(ref).get("path")) for name, ref in refs.items()}
    dependencies = record(data.get("dependencies"))
    required = (
        "partial_audio_edge.py",
        "clocked_cut_audio.py",
        "clocked_tempo_audio.py",
        "audio_clock.py",
        "cut_audio_support.py",
        "continuous_quiet_plan.py",
        "caption_projection.py",
    )
    if not {str(Path(__file__).with_name(name).resolve()) for name in required} <= set(
        dependencies
    ):
        raise ReplayError("Missing current partial-edge producer dependencies")
    if any(not isinstance(value, str) for value in dependencies.values()):
        raise ReplayError("Invalid dependency digest")
    bindings = {Path(name): value for name, value in dependencies.items() if isinstance(value, str)}
    for name, path in paths.items():
        digest = record(refs[name]).get("sha256")
        if not isinstance(digest, str):
            raise ReplayError("Missing input digest")
        bindings[path] = digest
    bindings[config] = file_digest(config)
    snapshot = InputHashSnapshot.capture(bindings.items())
    cuts, labels = _read(paths["cuts"]), _read(paths["labels"])
    parents = record(cuts.get("parentSHA256"))
    if (
        cuts.get("clock") != "source-presentation"
        or labels.get("clock") != "source-presentation"
        or parents.get(str(paths["audioClock"])) != bindings[paths["audioClock"]]
        or parents.get(str(paths["labels"])) != bindings[paths["labels"]]
        or parents.get(str(paths["source"])) != bindings[paths["source"]]
        or record(labels.get("parentSHA256")).get(str(paths["audioClock"]))
        != bindings[paths["audioClock"]]
    ):
        raise ReplayError("Partial-edge source labels/cuts use another clock")
    fps, bound = cuts.get("fps"), data.get("maximumRateChange")
    if type(fps) is not int or not isinstance(bound, (float, int)) or isinstance(bound, bool):
        raise ReplayError("Invalid partial-edge frame/rate contract")
    with wave.open(str(paths["source"]), "rb") as reader:
        rate, samples, channels = reader.getframerate(), reader.getnframes(), reader.getnchannels()
        if reader.getsampwidth() != 2 or channels not in (1, 2):
            raise ReplayError("Partial-edge renderer requires mono/stereo PCM16")
    original = _cuts(cuts)
    source_frames = cuts.get("sourceFrames")
    if type(source_frames) is not int or original[-1].source_end_frame > source_frames:
        raise ReplayError("Partial-edge cuts exceed source frame extent")
    prepared = plan(
        original,
        _labels(labels),
        AudioFrameClock.from_probe(_read(paths["audioClock"])),
        fps=fps,
        rate=rate,
        samples=samples,
        maximum_rate_change=float(bound),
    )
    metadata: dict[str, object] = {
        "version": 1,
        "kind": KIND,
        "inputs": {str(path): digest for path, digest in bindings.items()},
        "plan": _payload(prepared),
        "channels": channels,
        "sampleRate": rate,
        "outputSamples": prepared.output_samples,
        "outputFrames": sum(c.duration_frames for c in prepared.cuts),
        "maximumRateChangeAllowed": bound,
        "internalClockPolicy": POLICY,
        "tempoInteriorTimingVerified": False,
        "maximumBoundaryRoundingErrorSamples": str(
            max(abs(error) for error in prepared.boundary_errors)
        ),
        "recognitionSourceAnchorsChanged": False,
        "AVSyncVerified": False,
        "speechQualityVerified": False,
        "deliveryComplete": False,
    }
    return prepared, paths, metadata, snapshot


def _save(path: Path, data: dict[str, object]) -> None:
    with path.open("x", encoding="utf-8") as handle:
        os.fchmod(handle.fileno(), 0o600)
        json.dump(data, handle, ensure_ascii=False, allow_nan=False, indent=2)


def execute(config: Path, output: Path, *, prepare_only: bool = False) -> Path:
    """Publish new cuts or render with a distinct, independently rechecked receipt."""
    output = output.absolute()
    if (
        output != output.resolve()
        or not output.is_relative_to(Path.cwd() / "edits/work")
        or output.exists()
    ):
        raise ReplayError("Expected a fresh canonical repository output directory")
    prepared, paths, metadata, snapshot = _load(config)
    output.mkdir(mode=0o700)
    revised = _read(paths["cuts"])
    revised.update(
        {
            "cuts": [asdict(cut) for cut in prepared.cuts],
            "parentSHA256": {**record(revised.get("parentSHA256")), **record(metadata["inputs"])},
            "audioSupport": {
                "policy": KIND,
                "plan": _payload(prepared),
                "recognizedSpeechRemoved": False,
            },
            "deliveryComplete": False,
        }
    )
    if not prepare_only:
        scratch = Path(os.environ.get("TMPDIR", ""))
        if (
            not scratch.is_absolute()
            or scratch != scratch.resolve()
            or not scratch.is_relative_to(Path.cwd() / "edits/work")
            or not scratch.is_dir()
        ):
            raise ReplayError(
                "Tempo child scratch must be an existing canonical repository directory"
            )
        audio = output / "candidate.wav"
        measurements = render(
            paths["source"], audio, prepared, ffmpeg=paths["ffmpeg"], policy=TempoPolicy()
        )
        metadata.update(
            {
                "output": str(audio),
                "outputSHA256": file_digest(audio),
                "measurements": [asdict(item) for item in measurements],
            }
        )
    snapshot.finish()
    _save(output / "cuts.json", revised)
    _save(output / "receipt.json", metadata)
    return output / "receipt.json"


def verify(config: Path, output: Path, *, prepare_only: bool = False) -> dict[str, object]:
    """Recompute the entire clock plan; never accept a legacy tempo receipt."""
    prepared, paths, expected, snapshot = _load(config)
    receipt = _read(output / "receipt.json")
    if any(receipt.get(key) != value for key, value in expected.items()):
        raise ReplayError("Partial-edge receipt differs from current inputs or clock")
    cuts = _read(output / "cuts.json")
    if cuts.get("cuts") != _payload(prepared)["cuts"] or record(cuts.get("audioSupport")).get(
        "plan"
    ) != _payload(prepared):
        raise ReplayError("Partial-edge revised cuts differ")
    if not prepare_only:
        audio = output / "candidate.wav"
        if receipt.get("output") != str(audio) or receipt.get("outputSHA256") != file_digest(audio):
            raise ReplayError("Partial-edge audio output identity differs")
        with wave.open(str(audio), "rb") as reader:
            if (
                reader.getframerate(),
                reader.getnframes(),
                reader.getnchannels(),
                reader.getsampwidth(),
            ) != (prepared.rate, prepared.output_samples, expected["channels"], 2):
                raise ReplayError("Partial-edge output PCM clock differs")
            remaining = prepared.output_samples
            while remaining:
                count = min(remaining, 65536)
                if len(reader.readframes(count)) != count * reader.getnchannels() * 2:
                    raise ReplayError("Partial-edge full PCM decode is truncated")
                remaining -= count
            for low, high in prepared.silence:
                reader.setpos(low)
                if reader.readframes(high - low) != bytes((high - low) * reader.getnchannels() * 2):
                    raise ReplayError("Known absent edge contains unexplained audio")
        measurements = receipt.get("measurements")
        if not isinstance(measurements, list) or len(measurements) != len(prepared.slots):
            raise ReplayError("Partial-edge tempo measurements incomplete")
        for slot, item in zip(prepared.slots, measurements, strict=True):
            measured = record(item)
            if (
                measured.get("output_start_sample") != slot.output_start
                or measured.get("output_end_sample") != slot.output_end
                or measured.get("source_start_sample") != slot.spans[0][0]
                or measured.get("source_end_sample") != slot.spans[-1][1]
            ):
                raise ReplayError("Partial-edge tempo measurement clock differs")
    snapshot.finish()
    return receipt


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("prepare", "render", "verify", "verify-plan"))
    parser.add_argument("config", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args(argv)
    if args.command in ("verify", "verify-plan"):
        verify(args.config, args.output.absolute(), prepare_only=args.command == "verify-plan")
    else:
        execute(args.config, args.output, prepare_only=args.command == "prepare")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
