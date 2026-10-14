"""Compose measured new-stem cuts with exact output-clock fallback audio."""

from __future__ import annotations

import argparse
import hashlib
import json
import wave
from dataclasses import asdict, dataclass
from pathlib import Path

from replay.audio_pcm_alignment import (
    MappedSpan,
    _format,
    _read_record,
    _resumed_support,
    _selection_slots,
)
from replay.clocked_cut_audio import CutSlot, sample_boundary
from replay.clocked_tempo_audio import MAX_WINDOW_BYTES, TempoPolicy, tempo_cut
from replay.recordings import record
from replay.stages import file_digest

DEFAULT_POLICY = TempoPolicy()


@dataclass(frozen=True)
class Choice:
    slot: CutSlot
    mapped: CutSlot | None
    reason: str


@dataclass(frozen=True)
class Prepared:
    choices: tuple[Choice, ...]
    stem: Path
    fallback: Path
    rate: int
    channels: int
    pins: dict[str, str]


def choose_cut(
    slot: CutSlot,
    *,
    support: tuple[MappedSpan, ...],
    reference_samples: int,
    candidate_samples: int,
    context_samples: int,
) -> Choice:
    """Require one measured correspondence covering center and all filter context."""
    if len(slot.spans) != 1:
        return Choice(slot, None, "noncontiguous-reference")
    low, high = slot.spans[0]
    margin = context_samples if slot.input_samples != slot.output_end - slot.output_start else 0
    if low - margin < 0 or high + margin > reference_samples:
        return Choice(slot, None, "unverified-source-edge-context")
    span = next(
        (item for item in support if item.start <= low - margin and item.end >= high + margin), None
    )
    if span is None:
        return Choice(slot, None, "unmeasured-center-or-context")
    start, end = low + span.offset, high + span.offset
    if start - margin < 0 or end + margin > candidate_samples:
        return Choice(slot, None, "candidate-context-out-of-bounds")
    mapped = CutSlot(
        slot.native_start, slot.native_end, slot.output_start, slot.output_end, ((start, end),)
    )
    return Choice(slot, mapped, "measured-center-and-context")


def _pin(value: object, pins: dict[str, str]) -> Path:
    binding = record(value)
    name, digest = binding.get("path"), binding.get("sha256")
    if not isinstance(name, str) or not isinstance(digest, str):
        raise ValueError("Invalid pinned input")
    path = Path(name).resolve()
    if not path.is_relative_to(Path.cwd().resolve()) or file_digest(path) != digest:
        raise ValueError("Pinned input mismatch or path escape")
    pins[str(path)] = digest
    return path


def _verify_fallback(
    path: Path, *, fallback: Path, digest: str, slots: tuple[CutSlot, ...]
) -> None:
    receipt = _read_record(path)
    rows = receipt.get("cuts")
    if (
        receipt.get("kind") != "context-atempo-exact-cut-slot-candidate"
        or receipt.get("output") != str(fallback)
        or receipt.get("outputSHA256") != digest
        or not isinstance(rows, list)
        or len(rows) != len(slots)
    ):
        raise ValueError("Fallback receipt binding mismatch")
    for value, slot in zip(rows, slots, strict=True):
        row = record(value)
        if (
            len(slot.spans) != 1
            or row.get("source_start_sample") != slot.spans[0][0]
            or row.get("source_end_sample") != slot.spans[0][1]
            or row.get("output_start_sample") != slot.output_start
            or row.get("output_end_sample") != slot.output_end
        ):
            raise ValueError("Fallback receipt cut clock mismatch")


def prepare(config: Path, *, policy: TempoPolicy = DEFAULT_POLICY) -> Prepared:
    """Validate provenance and reconstruct cut choices without generating PCM."""
    root = Path.cwd().resolve()
    config = config.resolve()
    if not config.is_relative_to(root):
        raise ValueError("Config must stay inside checkout")
    raw = _read_record(config)
    if raw.get("version") != 1:
        raise ValueError("Unsupported mixed audio config")
    pins = {str(config): file_digest(config)}
    reference = _pin(raw.get("reference"), pins)
    mixture = _pin(raw.get("candidateMixture"), pins)
    stem = _pin(raw.get("candidateStem"), pins)
    fallback = _pin(raw.get("fallback"), pins)
    fallback_receipt = _pin(raw.get("fallbackReceipt"), pins)
    selection = _pin(raw.get("selection"), pins)
    mapping = _pin(raw.get("mapping"), pins)
    evidence = raw.get("evidence")
    if not isinstance(evidence, list) or not 1 <= len(evidence) <= 32:
        raise ValueError("Explicit producer and fallback evidence is required")
    for item in evidence:
        _pin(item, pins)
    maximum = raw.get("maximumRateChange")
    if isinstance(maximum, bool) or not isinstance(maximum, (int, float)):
        raise ValueError("Explicit rate policy is required")
    with (
        wave.open(str(reference), "rb") as ref,
        wave.open(str(mixture), "rb") as mix,
        wave.open(str(stem), "rb") as voice,
        wave.open(str(fallback), "rb") as old,
    ):
        rate, channels, reference_samples = _format(ref)
        candidate_format, fallback_format = _format(mix), _format(old)
        if (
            candidate_format != _format(voice)
            or candidate_format[:2] != (rate, channels)
            or fallback_format[:2] != (rate, channels)
        ):
            raise ValueError("Mixture, stem and fallback PCM formats differ")
        slots, bindings = _selection_slots(
            selection,
            sha256=pins[str(selection)],
            reference=reference,
            reference_sha256=pins[str(reference)],
            rate=rate,
            samples=reference_samples,
            maximum_rate_change=float(maximum),
        )
        pins.update(bindings)
        if not slots or slots[-1].output_end != fallback_format[2]:
            raise ValueError("Fallback must cover the exact selected output clock")
        _verify_fallback(
            fallback_receipt, fallback=fallback, digest=pins[str(fallback)], slots=slots
        )
        report = _read_record(mapping)
        if report.get("kind") != "selected-pcm-correspondence-diagnostic":
            raise ValueError("Expected selected PCM measurement report")
        support = _resumed_support(
            report,
            pins={
                str(reference): pins[str(reference)],
                str(mixture): pins[str(mixture)],
                **bindings,
            },
            rate=rate,
            reference_samples=reference_samples,
            candidate_samples=candidate_format[2],
        )
        choices = tuple(
            choose_cut(
                slot,
                support=support,
                reference_samples=reference_samples,
                candidate_samples=candidate_format[2],
                context_samples=sample_boundary(policy.context_seconds, rate),
            )
            for slot in slots
        )
    return Prepared(choices, stem, fallback, rate, channels, pins)


def render(
    prepared: Prepared, output: Path, *, ffmpeg: Path, policy: TempoPolicy = DEFAULT_POLICY
) -> list[dict[str, object]]:
    """Stream one candidate, copying fallback slots byte-for-byte without recutting."""
    if not prepared.choices or output.exists():
        raise ValueError("Empty selection or existing output")
    result: list[dict[str, object]] = []
    position = 0
    width = prepared.channels * 2
    with (
        wave.open(str(prepared.stem), "rb") as source,
        wave.open(str(prepared.fallback), "rb") as fallback,
        output.open("xb") as handle,
        wave.open(handle, "wb") as writer,
    ):
        writer.setnchannels(prepared.channels)
        writer.setsampwidth(2)
        writer.setframerate(prepared.rate)
        for choice in prepared.choices:
            slot = choice.slot
            if slot.output_start != position:
                raise ValueError("Noncontiguous output slots")
            count = slot.output_end - slot.output_start
            if not 0 < count * width <= MAX_WINDOW_BYTES:
                raise ValueError("Mixed cut exceeds bounded PCM budget")
            measurement: object = None
            if choice.mapped is None:
                fallback.setpos(slot.output_start)
                raw = fallback.readframes(count)
                selected = prepared.fallback
            else:
                raw, measured = tempo_cut(source, choice.mapped, ffmpeg=ffmpeg, policy=policy)
                measurement = asdict(measured)
                selected = prepared.stem
            if len(raw) != count * width:
                raise ValueError("Mixed cut is truncated")
            writer.writeframesraw(raw)
            position = slot.output_end
            result.append(
                {
                    "outputStartSample": slot.output_start,
                    "outputEndSample": position,
                    "sourcePath": str(selected),
                    "sourceSHA256": prepared.pins[str(selected)],
                    "sourceClock": "selected-output"
                    if choice.mapped is None
                    else "measured-continuous-pcm",
                    "sourceSpans": ((slot.output_start, position),)
                    if choice.mapped is None
                    else choice.mapped.spans,
                    "fallbackReadSpan": [slot.output_start, position]
                    if choice.mapped is None
                    else None,
                    "reason": choice.reason,
                    "pcmSHA256": hashlib.sha256(raw).hexdigest(),
                    "tempo": measurement,
                }
            )
    return result


def run(config: Path, output: Path, receipt: Path, *, ffmpeg: Path, check: bool = False) -> None:
    root = Path.cwd().resolve() / "edits/work"
    output, receipt = output.resolve(), receipt.resolve()
    if (
        not output.is_relative_to(root)
        or not receipt.is_relative_to(root)
        or output == receipt
        or output.exists()
        or receipt.exists()
    ):
        raise ValueError("New output and receipt must stay in repository work directory")
    prepared = prepare(config)
    result: dict[str, object] = {
        "version": 1,
        "kind": "mixed-tempo-candidate",
        "inputs": prepared.pins,
        "sampleRate": prepared.rate,
        "channels": prepared.channels,
        "outputSamples": prepared.choices[-1].slot.output_end,
        "newStemCuts": sum(choice.mapped is not None for choice in prepared.choices),
        "fallbackCuts": sum(choice.mapped is None for choice in prepared.choices),
        "policy": {"contextSeconds": 0.5, "unknown": "whole-output-cut-fallback"},
        "qualityAccepted": False,
        "productionAudioModified": False,
        "preflightOnly": check,
        "codeSHA256": file_digest(Path(__file__)),
        "tempoCodeSHA256": file_digest(Path(__file__).with_name("clocked_tempo_audio.py")),
    }
    if not check:
        result["cuts"] = render(prepared, output, ffmpeg=ffmpeg)
        if any(file_digest(Path(path)) != digest for path, digest in prepared.pins.items()):
            raise ValueError("Mixed audio inputs changed during rendering")
        with wave.open(str(output), "rb") as reader:
            if _format(reader) != (
                prepared.rate,
                prepared.channels,
                prepared.choices[-1].slot.output_end,
            ):
                raise ValueError("Mixed output header mismatch")
        result["outputSHA256"] = file_digest(output)
    else:
        result["cuts"] = [
            {
                "outputStartSample": choice.slot.output_start,
                "outputEndSample": choice.slot.output_end,
                "reason": choice.reason,
                "sourcePath": str(prepared.fallback if choice.mapped is None else prepared.stem),
                "sourceSHA256": prepared.pins[
                    str(prepared.fallback if choice.mapped is None else prepared.stem)
                ],
                "sourceClock": "selected-output"
                if choice.mapped is None
                else "measured-continuous-pcm",
            }
            for choice in prepared.choices
        ]
    with receipt.open("x", encoding="utf-8") as stream:
        json.dump(result, stream, indent=2)
        stream.write("\n")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("config", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("receipt", type=Path)
    parser.add_argument("--ffmpeg", type=Path, required=True)
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()
    run(args.config, args.output, args.receipt, ffmpeg=args.ffmpeg, check=args.check)


if __name__ == "__main__":
    main()
