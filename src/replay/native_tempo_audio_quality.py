"""Validate recorded tempo-slot correspondence before independent audio quality work.

Metadata adapters hash explicitly bound media and inspect PCM headers. The explicit
quality consumer runs the existing model gates on verified PCM copies, with resource
and writable-descriptor observations. No path grants delivery approval. Whole-cut
protection is conservative coverage, not an interior timing measurement. Original
tempo receipts remain unchanged.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib
import json
import os
import re
import wave
from dataclasses import dataclass
from fractions import Fraction
from itertools import pairwise
from math import isfinite
from pathlib import Path

import numpy as np
from numpy.typing import NDArray

from replay import aac_presentation_acceptance as aac
from replay import delivery_audio_quality as dq
from replay.audio_clock import AudioFrameClock
from replay.audio_cpu_probe import audit_descriptors
from replay.bgm_separation import SeparationBackend
from replay.caption_measured_job import _json, _local, _object, _rational
from replay.clocked_cut_audio import CutSlot, plan_slots, sample_boundary
from replay.clocked_tempo_audio import TempoPolicy
from replay.foreground_evidence import _bound_bytes
from replay.process_family_observation import SystemHost
from replay.recordings import record
from replay.stages import _fingerprint, file_digest
from replay.tracking_observations import _digest, _integer, _read
from replay.workflow import _stage

MAX_REQUEST_BYTES = 1024**2
MAX_RECORD_BYTES = 16 * 1024**2
MAX_CLOCK_BYTES = 64 * 1024**2
MAX_CUTS = 100_000
MAX_AUDIO_SECONDS = 21_600
MEASUREMENT_FIELDS = (
    "source_start_sample source_end_sample output_start_sample output_end_sample "
    "source_context_samples synthetic_left_context_samples synthetic_right_context_samples "
    "filter_output_samples expected_filter_output_samples center_crop_start_sample "
    "center_crop_end_sample tail_error_samples process_seconds filter_applied"
)


@dataclass(slots=True)
class ObservedBackend:
    """Preserve provider samples while observing each completed inference boundary."""

    backend: SeparationBackend
    directory: Path
    calls: int = 0

    @property
    def sources(self) -> tuple[str, ...]:
        return self.backend.sources

    def extract_vocals(self, chunk: NDArray[np.float32]) -> NDArray[np.float32]:
        result = self.backend.extract_vocals(chunk)
        observe_quality_resources(self.directory, f"after-inference-{self.calls:06d}")
        self.calls += 1
        return result

    def extract_instrumental(self, chunk: NDArray[np.float32]) -> NDArray[np.float32]:
        result = self.backend.extract_instrumental(chunk)
        observe_quality_resources(self.directory, f"after-instrumental-{self.calls:06d}")
        self.calls += 1
        return result


@dataclass(frozen=True, slots=True)
class TempoInputs:
    selection_sha256: str
    clock_sha256: str
    source_sha256: str
    output_sha256: str
    source_samples: int
    sample_rate: int
    channels: int
    policy: TempoPolicy
    slots: tuple[CutSlot, ...]

    @property
    def output_samples(self) -> int:
        return self.slots[-1].output_end


@dataclass(frozen=True, slots=True)
class ProtectedCut:
    cut_index: int
    source_start: int
    source_end: int
    output_start: int
    output_end: int


@dataclass(frozen=True, slots=True)
class DecodedExtent:
    sample_rate: int
    sample_count: int
    start: Fraction
    end: Fraction
    contiguous: bool
    priming_samples: int
    discarded_samples: int


def pcm_budget(
    *, rate: int, channels: int, sample_bytes: int, seconds: Fraction, files: int, header_bytes: int
) -> int:
    """Count every channel and byte; callers supply their actual container overhead."""
    values = (rate, channels, sample_bytes, files, header_bytes)
    if any(type(item) is not int for item in values) or not (
        8000 <= rate <= 192000
        and 1 <= channels <= 64
        and 1 <= sample_bytes <= 8
        and 1 <= files <= 4096
        and 0 <= header_bytes <= 1024**2
        and 0 < seconds <= MAX_AUDIO_SECONDS
    ):
        raise ValueError("Invalid bounded PCM budget")
    samples = seconds * rate
    if samples.denominator != 1:
        raise ValueError("PCM budget needs an integral sample extent")
    return files * (int(samples) * channels * sample_bytes + header_bytes)


def _fraction(value: object) -> Fraction:
    if not isinstance(value, str) or not 1 <= len(value) <= 64:
        raise ValueError("Expected a bounded exact rational")
    return Fraction(value)


def _pin(value: object) -> tuple[str, str]:
    item = _object(value, "path sha256")
    path = item["path"]
    if (
        not isinstance(path, str)
        or "\0" in path
        or not path.startswith("/")
        or str(Path(path)) != path
        or ".." in Path(path).parts
    ):
        raise ValueError("Expected an absolute original input path")
    return path, _digest(item["sha256"])


def _number(value: object) -> float:
    if type(value) not in (int, float) or not isinstance(value, (int, float)):
        raise ValueError("Expected a finite numeric measurement")
    if not isfinite(value):
        raise ValueError("Expected a finite numeric measurement")
    return float(value)


def _measurement(value: object, slot: CutSlot, inputs: TempoInputs) -> None:
    row = _object(value, MEASUREMENT_FIELDS)
    low, high = slot.spans[0]
    target = slot.output_end - slot.output_start
    ratio = Fraction(high - low, target)
    context = (
        sample_boundary(inputs.policy.context_seconds, inputs.sample_rate) if ratio != 1 else 0
    )
    crop = sample_boundary(Fraction(context) / ratio, 1)
    expected = sample_boundary(Fraction(high - low + 2 * context) / ratio, 1)
    fields = {
        "source_start_sample": low,
        "source_end_sample": high,
        "output_start_sample": slot.output_start,
        "output_end_sample": slot.output_end,
        "source_context_samples": context,
        "synthetic_left_context_samples": max(0, context - low),
        "synthetic_right_context_samples": max(0, high + context - inputs.source_samples),
        "expected_filter_output_samples": expected,
        "center_crop_start_sample": crop,
        "center_crop_end_sample": crop + target,
    }
    if any(_integer(row[key]) != expected_value for key, expected_value in fields.items()):
        raise ValueError("Original tempo measurement disagrees with its clock and crop")
    actual, tail = _integer(row["filter_output_samples"]), _integer(row["tail_error_samples"])
    if (
        actual < crop + target
        or actual - expected != tail
        or abs(tail) > sample_boundary(inputs.policy.maximum_tail_error_seconds, inputs.sample_rate)
        or type(row["filter_applied"]) is not bool
        or row["filter_applied"] != (ratio != 1)
        or _number(row["process_seconds"]) < 0
    ):
        raise ValueError("Incomplete or contradictory tempo filter measurement")
    if ratio == 1 and (actual != target or tail != 0):
        raise ValueError("A no-op tempo cut cannot change samples")


def recorded_mapping(
    selection: object,
    receipt: object,
    clock: AudioFrameClock,
    *,
    selection_path: str,
    selection_sha256: str,
    source_samples: int,
) -> TempoInputs:
    """Recompute the original producer's slots without opening source or output media."""
    config = _object(selection, "version coordinateSpace source clock intervals evidence")
    data = record(receipt)
    if (
        type(config["version"]) is not int
        or config["version"] != 1
        or config["coordinateSpace"] != "native-presentation-seconds"
        or type(data.get("version")) is not int
        or data.get("version") != 1
        or data.get("kind") != "context-atempo-exact-cut-slot-candidate"
    ):
        raise ValueError("Unsupported original tempo schema")
    source_path, source_sha = _pin(config["source"])
    clock_path, clock_sha = _pin(config["clock"])
    pins = record(data.get("inputs"))
    if (
        pins.get(selection_path) != _digest(selection_sha256)
        or pins.get(source_path) != source_sha
        or pins.get(clock_path) != clock_sha
    ):
        raise ValueError("Original selection/source/clock pin mismatch")
    rows, evidence = config["intervals"], config["evidence"]
    if not isinstance(rows, list) or not 1 <= len(rows) <= MAX_CUTS:
        raise ValueError("Invalid bounded interval list")
    if not isinstance(evidence, list) or len(evidence) > 128:
        raise ValueError("Invalid bounded original evidence")
    evidence_pins = tuple(_pin(item) for item in evidence)
    if any(pins.get(path) != digest for path, digest in evidence_pins):
        raise ValueError("Original evidence pin mismatch")
    intervals = tuple(
        (_rational(row["start"]), _rational(row["end"]))
        for row in (_object(item, "start end") for item in rows)
    )
    rate, channels = _integer(data.get("sampleRate")), _integer(data.get("channels"))
    if not 1 <= channels <= 64 or type(source_samples) is not int:
        raise ValueError("Invalid original PCM dimensions")
    slots = plan_slots(
        clock,
        intervals,
        rate=rate,
        samples=source_samples,
        maximum_rate_change=_number(data.get("maximumRateChangeAllowed")),
    )
    policy = TempoPolicy(
        _fraction(data.get("contextSeconds")), _fraction(data.get("maximumTailErrorSeconds"))
    )
    result = TempoInputs(
        _digest(selection_sha256),
        clock_sha,
        source_sha,
        _digest(data.get("outputSHA256")),
        source_samples,
        rate,
        channels,
        policy,
        slots,
    )
    if (
        _integer(data.get("outputStartSample")) != 0
        or _integer(data.get("outputEndSample")) != result.output_samples
        or _integer(data.get("outputSamples")) != result.output_samples
    ):
        raise ValueError("Receipt is not the complete selected output")
    measurements = data.get("cuts")
    if not isinstance(measurements, list) or len(measurements) != len(slots):
        raise ValueError("Missing tempo cut measurements")
    for value, slot in zip(measurements, slots, strict=True):
        _measurement(value, slot, result)
    return result


def matched_transform(reference: TempoInputs, mixture: TempoInputs) -> None:
    """Require independent inputs and the same exact transform, never direct spans."""
    if (
        reference.source_sha256 == mixture.source_sha256
        or reference.output_sha256 == mixture.output_sha256
    ):
        raise ValueError("Independent mixture and voice are required")
    if (
        reference.clock_sha256 != mixture.clock_sha256
        or reference.source_samples != mixture.source_samples
        or reference.sample_rate != mixture.sample_rate
        or reference.channels != mixture.channels
        or reference.policy != mixture.policy
        or reference.slots != mixture.slots
    ):
        raise ValueError("Mixture/reference transforms or source clocks differ")


def protect_whole_cuts(
    inputs: TempoInputs, intervals: tuple[tuple[int, int], ...]
) -> tuple[ProtectedCut, ...]:
    """Cover each touched cut in full; never interpolate atempo speech endpoints.

    This conservative mask is usable only after the source-bound ASR intervals
    have independently passed their own lineage guards. It grants no ASR approval.
    Speech crossing a removed interval is rejected, not silently clipped.
    """
    if (
        not intervals
        or len(intervals) > MAX_CUTS
        or any(
            type(low) is not int
            or type(high) is not int
            or not 0 <= low < high <= inputs.source_samples
            for low, high in intervals
        )
        or any(right[0] < left[1] for left, right in pairwise(intervals))
    ):
        raise ValueError("Invalid source-sample protected speech")
    selected: dict[int, ProtectedCut] = {}
    for low, high in intervals:
        selected.update(_cover_interval(inputs, low=low, high=high))
    return tuple(selected[index] for index in sorted(selected))


def _cover_interval(inputs: TempoInputs, *, low: int, high: int) -> dict[int, ProtectedCut]:
    selected: dict[int, ProtectedCut] = {}
    cursor = low
    for index, slot in enumerate(inputs.slots):
        start, end = slot.spans[0]
        if end <= cursor or start >= high:
            continue
        if start > cursor:
            raise ValueError("Protected speech crosses removed source samples")
        selected[index] = ProtectedCut(index, start, end, slot.output_start, slot.output_end)
        cursor = min(end, high)
    if cursor != high:
        raise ValueError("Protected speech lacks selected source support")
    return selected


def require_exact_decode(inputs: TempoInputs, decoded: DecodedExtent) -> None:
    """Require native-rate decoded extent; no resampling, inferred priming or padding."""
    if (
        type(decoded.sample_rate) is not int
        or not 8000 <= decoded.sample_rate <= 192000
        or type(decoded.sample_count) is not int
        or decoded.sample_count <= 0
        or decoded.contiguous is not True
    ):
        raise ValueError("Invalid or discontinuous decoded AAC clock")
    if (
        type(decoded.priming_samples) is not int
        or type(decoded.discarded_samples) is not int
        or decoded.priming_samples != 0
        or decoded.discarded_samples != 0
    ):
        raise ValueError("Nonzero priming/discard requires an independent supported mapping")
    duration = Fraction(inputs.output_samples, inputs.sample_rate)
    if (
        decoded.start != 0
        or decoded.end != duration
        or Fraction(decoded.sample_count, decoded.sample_rate) != duration
    ):
        raise ValueError("Decoded AAC does not cover the exact selected PCM extent")


def prepare_quality_inputs(
    reference: TempoInputs,
    mixture: TempoInputs,
    *,
    decoded: DecodedExtent,
    protected_source_samples: tuple[tuple[int, int], ...],
) -> tuple[ProtectedCut, ...]:
    """Return a usable conservative mask only when the three clocks agree.

    Callers must separately verify current media/model identities and execute the
    unchanged separation and speech/music measurement policies. This function
    neither emits a quality receipt nor changes CPU policy thresholds.
    """
    matched_transform(reference, mixture)
    require_exact_decode(reference, decoded)
    return protect_whole_cuts(reference, protected_source_samples)


def recorded_aac_extent(
    receipt: object, *, movie: tuple[str, str], inputs: TempoInputs
) -> dict[str, object]:
    """Diagnose the original full-decode summary, not current media or AAC adoption."""
    data = record(receipt)
    before, after = data.get("inputs"), data.get("inputsAfter")
    if data.get("version") != 1 or data.get("fullDecode") is not True:
        raise ValueError("Missing original full-decode receipt")
    if not isinstance(before, list) or not 1 <= len(before) <= 256 or before != after:
        raise ValueError("Original decoder input identities changed or missing")
    pins: dict[str, str] = {}
    for value in before:
        row = _object(value, "path bytes sha256 device inode mtimeNs ctimeNs")
        path, digest = _pin({"path": row["path"], "sha256": row["sha256"]})
        if path in pins or any(
            _integer(row[key]) <= 0 for key in ("bytes", "device", "inode", "mtimeNs", "ctimeNs")
        ):
            raise ValueError("Invalid original full-decode input identity")
        pins[path] = digest
    if pins.get(movie[0]) != _digest(movie[1]):
        raise ValueError("Original full decode belongs to a different movie")
    streams = record(data.get("streams")).get("streams")
    if not isinstance(streams, list):
        raise ValueError("Missing decoded audio stream")
    audio = tuple(record(item) for item in streams if record(item).get("codec_type") == "audio")
    if len(audio) != 1 or audio[0].get("codec_name") != "aac":
        raise ValueError("Expected exactly one AAC stream")
    rate = _fraction(audio[0].get("sample_rate"))
    if rate.denominator != 1 or not 8000 <= rate <= 192000:
        raise ValueError("Invalid AAC sample rate")
    if _integer(audio[0].get("channels")) != inputs.channels:
        raise ValueError("AAC channel count differs")
    clock = record(data.get("audioDecodedClock"))
    start, end, span = (_fraction(clock.get(key)) for key in ("start", "end", "span"))
    count = _integer(clock.get("sampleCount"))
    if start < 0 or count <= 0 or end - start != span or Fraction(count, rate) != span:
        raise ValueError("Contradictory decoded AAC summary")
    duration = Fraction(inputs.output_samples, inputs.sample_rate)
    return {
        "decodedRate": int(rate),
        "selectedRate": inputs.sample_rate,
        "decodedSamples": count,
        "decodedStart": str(start),
        "decodedEnd": str(end),
        "selectedDuration": str(duration),
        "endDifferenceSeconds": str(end - duration),
        "completeExtent": start == 0 and end == duration,
        "primingAndContinuityVerified": False,
        "currentMediaCryptoVerified": False,
    }


def inspect_request(request: Path, *, sha256: str) -> dict[str, object]:
    """Read only bounded, pinned metadata; never treat recorded hashes as fresh media."""
    request = _local(str(request.absolute()))
    raw = _bound_bytes(request, sha256, MAX_REQUEST_BYTES)
    config = _object(
        _json(raw), "version selection tempoReceipt clock sourceSamples finalReceipt movie"
    )
    if type(config["version"]) is not int or config["version"] != 1:
        raise ValueError("Unsupported tempo quality mapping request")
    values: dict[str, object] = {}
    observed: dict[Path, tuple[str, int]] = {}
    for key in ("selection", "tempoReceipt", "clock", "finalReceipt"):
        name, digest = _pin(config[key])
        path = _local(name)
        data = _bound_bytes(path, digest, MAX_CLOCK_BYTES if key == "clock" else MAX_RECORD_BYTES)
        if path in observed and observed[path][0] != digest:
            raise ValueError("Conflicting metadata pins")
        observed[path] = (digest, len(data))
        values[key] = _json(data)
    selection_path, selection_hash = _pin(config["selection"])
    selection_clock = _pin(record(values["selection"]).get("clock"))
    if selection_clock != _pin(config["clock"]):
        raise ValueError("Supplied clock differs from original selection")
    inputs = recorded_mapping(
        values["selection"],
        values["tempoReceipt"],
        AudioFrameClock.from_probe(values["clock"]),
        selection_path=selection_path,
        selection_sha256=selection_hash,
        source_samples=_integer(config["sourceSamples"]),
    )
    aac = recorded_aac_extent(values["finalReceipt"], movie=_pin(config["movie"]), inputs=inputs)
    for path, (digest, length) in observed.items():
        _bound_bytes(path, digest, length)
    _bound_bytes(request, sha256, MAX_REQUEST_BYTES)
    return {
        "version": 1,
        "kind": "native-tempo-quality-input-mapping",
        "request": {"path": str(request), "sha256": sha256},
        "metadataPins": {str(path): digest for path, (digest, _) in observed.items()},
        "uniqueMetadataBytes": sum(length for _, length in observed.values()) + len(raw),
        "metadataPasses": 2,
        "mediaBytesRead": 0,
        "recordedTempoSlotsVerified": True,
        "cuts": len(inputs.slots),
        "selectedSamples": inputs.output_samples,
        "selectedRate": inputs.sample_rate,
        "channels": inputs.channels,
        "sourceSamplesDeclared": inputs.source_samples,
        "sourceSamplesCurrentlyMeasured": False,
        "aac": aac,
        "measurementInputsReady": False,
        "missing": [
            "Independent mixture transformed with the same original clock/cuts/context",
            "Source-bound protected speech with conservative cut coverage "
            "or measured interior mapping",
            "Exact full decoded AAC presentation/priming/continuity and unchanged media identities",
            "Original separation model/runtime lineage and independent "
            "full-duration quality measurements",
        ],
        "currentMediaCryptoVerified": False,
        "qualityApproved": False,
        "speechQualityVerified": False,
        "AVSyncVerified": False,
        "deliveryComplete": False,
    }


def assess_audio_only(producer: object, observation: object) -> dict[str, object]:
    """Use original packet/decoded-clock guards without inventing a video track."""
    source = record(producer)
    if (
        source.get("version") != 1
        or source.get("kind") != "context-atempo-exact-cut-slot-candidate"
        or _integer(source.get("outputStartSample")) != 0
    ):
        raise ValueError("Expected complete original selected PCM producer")
    samples, rate = _integer(source.get("outputSamples")), _integer(source.get("sampleRate"))
    if not 8000 <= rate <= 192000 or samples <= 0 or source.get("outputEndSample") != samples:
        raise ValueError("Invalid original selected PCM clock")
    raw = record(observation)
    rows, streams = raw.get("packets_and_frames"), raw.get("streams")
    if not isinstance(rows, list) or not 1 <= len(rows) <= 2_000_000:
        raise ValueError("Missing bounded complete packet/frame observations")
    if not isinstance(streams, list) or len(streams) != 1:
        raise ValueError("Audio-only observation must have exactly one stream")
    stream = record(streams[0])
    if (
        stream.get("codec_type") != "audio"
        or stream.get("codec_name") != "aac"
        or stream.get("sample_rate") != str(rate)
        or stream.get("channels") != source.get("channels")
    ):
        raise ValueError("Standalone AAC changed original rate or channels")
    frames: list[dict[str, object]] = []
    packets: list[dict[str, object]] = []
    for item in rows:
        row = record(item)
        if row.get("type") == "frame":
            frames.append(row)
        elif row.get("type") == "packet":
            packets.append(row)
        else:
            raise ValueError("Unknown combined observation row")
    if not frames or not packets:
        raise ValueError("Incomplete combined AAC evidence")
    clock = AudioFrameClock.from_probe({"streams": [stream], "frames": frames})
    start, end, reasons = aac._packet_extent(stream, packets, clock)
    contiguous = all(
        (right - left) * clock.time_base == Fraction(count, rate)
        for left, right, count in zip(clock.pts, clock.pts[1:], clock.samples, strict=False)
    )
    expected = Fraction(samples, rate)
    decoded_end = clock.pts[-1] * clock.time_base + Fraction(clock.samples[-1], rate)
    return {
        "version": 1,
        "kind": "native-tempo-audio-only-presentation-observation",
        "sourceSampleRate": rate,
        "sourceSamples": samples,
        "expectedDuration": str(expected),
        "decodedSamples": clock.sample_ends[-1],
        "decodedStart": str(clock.pts[0] * clock.time_base),
        "decodedEnd": str(decoded_end),
        "decodedSamplesMinusSource": clock.sample_ends[-1] - samples,
        "packetStartAfterDecoderSkip": str(start),
        "packetEnd": str(end),
        "packetEndMinusSourceEnd": str(end - expected),
        "packetPresentationMatchesSelected": start == 0 and end == expected,
        "packetDecodedPresentationAccepted": not reasons,
        "decodedContiguous": contiguous,
        "packetReasons": reasons,
        "firstPacket": packets[0],
        "lastPacket": packets[-1],
        "videoContractSupported": False,
        "containerEditListVerified": False,
        "qualityApproved": False,
        "syncApproved": False,
        "deliveryComplete": False,
        "paddingOrTrimPerformedByVerifier": False,
    }


def inspect_audio_only(request: Path, *, sha256: str) -> dict[str, object]:
    request = _local(str(request.absolute()))
    payload = _bound_bytes(request, sha256, MAX_REQUEST_BYTES)
    config = _object(_json(payload), "version kind sourceProducer candidate observation outputRoot")
    if config["version"] != 1 or config["kind"] != "native-tempo-audio-only-request":
        raise ValueError("Unsupported standalone AAC request")
    source_name, source_digest = _pin(config["sourceProducer"])
    producer = _json(_bound_bytes(_local(source_name), source_digest, MAX_RECORD_BYTES))
    candidate, output_root = Path(str(config["candidate"])), Path(str(config["outputRoot"]))
    if (
        not output_root.is_absolute()
        or output_root.resolve(strict=True) != output_root
        or not candidate.is_relative_to(output_root)
        or candidate.resolve(strict=True) != candidate
        or any(part.is_symlink() for part in candidate.parents)
    ):
        raise ValueError("Standalone AAC candidate escapes its explicit canonical root")
    before = candidate.stat()
    digest = file_digest(candidate)
    observation = _local(config["observation"])
    raw = _read(observation, 128 * 1024**2)
    observation_digest = hashlib.sha256(raw).hexdigest()
    result = assess_audio_only(producer, _json(raw))
    _bound_bytes(_local(source_name), source_digest, MAX_RECORD_BYTES)
    _bound_bytes(observation, observation_digest, 128 * 1024**2)
    _bound_bytes(request, sha256, MAX_REQUEST_BYTES)
    if candidate.stat() != before or file_digest(candidate) != digest:
        raise ValueError("Standalone AAC changed during observation")
    result.update(
        request={"path": str(request), "sha256": sha256},
        candidate={"path": str(candidate), "sha256": digest},
        sourceProducer=config["sourceProducer"],
        observation={"path": str(observation), "sha256": observation_digest},
    )
    return result


def compare_packet_hashes(left: bytes, right: bytes, *, expected_frames: int) -> dict[str, object]:
    """Compare copy-mode framehash observations, including exact rational clocks.

    This proves equality of supplied observations only. The enclosing workflow
    must bind their actual producer commands and media identities.
    """
    if type(expected_frames) is not int or not 0 < expected_frames <= 4_000_000:
        raise ValueError("Invalid expected packet count")

    def rows(data: bytes) -> tuple[list[str], Fraction, list[str]]:
        if not 0 < len(data) <= 64 * 1024**2:
            raise ValueError("Packet observation exceeds its byte budget")
        lines = data.decode("ascii").splitlines()
        if len(lines) != expected_frames + 11:
            raise ValueError("Packet observation is incomplete or has extra records")
        header = lines[:11]
        if (
            header[:3] != ["#format: frame checksums", "#version: 2", "#hash: SHA256"]
            or re.fullmatch(r"#extradata 0,\s+[1-9][0-9]*, [a-f0-9]{64}", header[3]) is None
            or not header[4].startswith("#software: Lavf")
            or not header[5].startswith("#tb 0: ")
            or header[6] != "#media_type 0: video"
            or header[7] not in {"#codec_id 0: h264", "#codec_id 0: hevc"}
            or re.fullmatch(r"#dimensions 0: [1-9][0-9]*x[1-9][0-9]*", header[8]) is None
            or re.fullmatch(r"#sar 0: [1-9][0-9]*/[1-9][0-9]*", header[9]) is None
            or header[10] != "#stream#, dts,        pts, duration,     size, hash"
        ):
            raise ValueError("Unsupported copy-mode packet observation header")
        scale = _fraction(header[5].removeprefix("#tb 0: "))
        if not 0 < scale <= 1 or scale.denominator > 1_000_000_000:
            raise ValueError("Invalid packet time base")
        return header[:5] + header[6:], scale, lines[11:]

    a_header, a_scale, a_rows = rows(left)
    b_header, b_scale, b_rows = rows(right)
    if a_header != b_header:
        raise ValueError("Video codec, geometry or extradata changed")
    previous: Fraction | None = None
    for a, b in zip(a_rows, b_rows, strict=True):
        values: list[tuple[int, int, int, int, str]] = []
        for line in (a, b):
            fields = [item.strip() for item in line.split(",")]
            if (
                len(fields) != 6
                or fields[0] != "0"
                or any(re.fullmatch(r"-?[0-9]+", item) is None for item in fields[1:5])
                or re.fullmatch(r"[a-f0-9]{64}", fields[5]) is None
            ):
                raise ValueError("Malformed packet hash record")
            dts, pts, duration, size = map(int, fields[1:5])
            if duration <= 0 or size <= 0:
                raise ValueError("Packet duration and size must be positive")
            values.append((dts, pts, duration, size, fields[5]))
        first, second = values
        if first[3:] != second[3:] or any(
            before * a_scale != after * b_scale
            for before, after in zip(first[:3], second[:3], strict=True)
        ):
            raise ValueError("Ordered packet payload or rational clock changed")
        current = first[0] * a_scale
        if previous is not None and current <= previous:
            raise ValueError("Decode packet clock is not strictly increasing")
        previous = current
    return {
        "packets": expected_frames,
        "sourceTimeBase": str(a_scale),
        "candidateTimeBase": str(b_scale),
        "orderedPayloadsEqual": True,
        "rationalClocksEqual": True,
        "extradataAndGeometryEqual": True,
        "fullDecodeVerified": False,
        "qualityApproved": False,
        "syncApproved": False,
    }


def inspect_packet_pair(request: Path, *, sha256: str) -> dict[str, object]:
    request = _local(str(request.absolute()))
    config = _object(
        _json(_bound_bytes(request, sha256, MAX_REQUEST_BYTES)),
        "version kind sourceObservation candidateObservation expectedFrames",
    )
    if config["version"] != 1 or config["kind"] != "copy-video-packet-comparison":
        raise ValueError("Unsupported packet comparison request")
    payloads: list[bytes] = []
    for key in ("sourceObservation", "candidateObservation"):
        path, digest = _pin(config[key])
        payloads.append(_bound_bytes(_local(path), digest, 64 * 1024**2))
    result = compare_packet_hashes(*payloads, expected_frames=_integer(config["expectedFrames"]))
    _bound_bytes(request, sha256, MAX_REQUEST_BYTES)
    result.update(request={"path": str(request), "sha256": sha256}, evidence=config)
    return result


def inspect_pcm_clock(request: Path, *, sha256: str) -> dict[str, object]:
    """Measure a complete original PCM file and bind its producer's zero origin."""
    request = _local(str(request.absolute()))
    config = _object(
        _json(_bound_bytes(request, sha256, MAX_REQUEST_BYTES)), "version kind source producer"
    )
    if (
        type(config["version"]) is not int
        or config["version"] != 1
        or config["kind"] != "original-pcm-clock-request"
    ):
        raise ValueError("Unsupported PCM clock request")
    source_name, source_sha = _pin(config["source"])
    producer_name, producer_sha = _pin(config["producer"])
    source, producer = _local(source_name), _local(producer_name)
    recorded = record(_json(_bound_bytes(producer, producer_sha, MAX_RECORD_BYTES)))
    before = source.stat()
    if not 44 <= before.st_size <= 8 * 1024**3 or file_digest(source) != source_sha:
        raise ValueError("PCM source byte bound or original SHA differs")
    with wave.open(str(source), "rb") as pcm:
        rate, channels, count, width = (
            pcm.getframerate(),
            pcm.getnchannels(),
            pcm.getnframes(),
            pcm.getsampwidth(),
        )
        if not (
            pcm.getcomptype() == "NONE"
            and 8000 <= rate <= 192000
            and 1 <= channels <= 8
            and 1 <= width <= 4
            and 0 < count <= rate * MAX_AUDIO_SECONDS
        ):
            raise ValueError("Unsupported bounded uncompressed PCM header")
        pcm.setpos(count - 1)
        if len(pcm.readframes(1)) != channels * width:
            raise ValueError("PCM data ends before its declared sample extent")
    if (
        any(
            type(recorded.get(key)) is not int
            for key in (
                "version",
                "sampleRate",
                "channels",
                "outputStartSample",
                "outputEndSample",
                "outputSamples",
            )
        )
        or recorded.get("version") != 1
        or recorded.get("kind") != "context-atempo-exact-cut-slot-candidate"
        or recorded.get("output") != source_name
        or recorded.get("outputSHA256") != source_sha
        or recorded.get("sampleRate") != rate
        or recorded.get("channels") != channels
        or recorded.get("outputStartSample") != 0
        or recorded.get("outputEndSample") != count
        or recorded.get("outputSamples") != count
    ):
        raise ValueError("PCM header or zero origin differs from original producer")
    _bound_bytes(producer, producer_sha, MAX_RECORD_BYTES)
    _bound_bytes(request, sha256, MAX_REQUEST_BYTES)
    if source.stat() != before or file_digest(source) != source_sha:
        raise ValueError("PCM source changed during clock observation")
    return {
        "version": 1,
        "kind": "original-pcm-clock-observation",
        "source": config["source"],
        "sampleRate": rate,
        "sampleCount": count,
        "channels": channels,
        "sampleWidthBytes": width,
        "start": "0",
        "producer": config["producer"],
        "request": {"path": str(request), "sha256": sha256},
        "originEvidence": "Original complete zero-origin producer; PCM header carries no PTS",
        "sourceSHAFreshlyVerified": True,
        "qualityApproved": False,
        "syncApproved": False,
        "deliveryComplete": False,
    }


def inspect_transform_pair(request: Path, *, sha256: str) -> dict[str, object]:
    """Bind two original producers and recompute their identical tempo slots."""
    request = _local(str(request.absolute()))
    config = _object(
        _json(_bound_bytes(request, sha256, MAX_REQUEST_BYTES)),
        "version kind reference mixture clock sourceSamples",
    )
    if (
        type(config["version"]) is not int
        or config["version"] != 1
        or config["kind"] != "tempo-transform-pair-request"
    ):
        raise ValueError("Unsupported transform pair request")
    observed: dict[Path, tuple[str, int]] = {}

    def read_pin(value: object, limit: int) -> object:
        name, digest = _pin(value)
        path = _local(name)
        if path in observed and observed[path][0] != digest:
            raise ValueError("Conflicting transform metadata pins")
        raw = _bound_bytes(path, digest, limit)
        observed[path] = digest, len(raw)
        return _json(raw)

    clock = AudioFrameClock.from_probe(read_pin(config["clock"], MAX_CLOCK_BYTES))
    values: list[TempoInputs] = []
    for role in ("reference", "mixture"):
        pair = _object(config[role], "selection receipt")
        selection = read_pin(pair["selection"], MAX_RECORD_BYTES)
        receipt = read_pin(pair["receipt"], MAX_RECORD_BYTES)
        if _pin(record(selection).get("clock")) != _pin(config["clock"]):
            raise ValueError("Transform pair uses a different clock")
        selection_name, digest = _pin(pair["selection"])
        values.append(
            recorded_mapping(
                selection,
                receipt,
                clock,
                selection_path=selection_name,
                selection_sha256=digest,
                source_samples=_integer(config["sourceSamples"]),
            )
        )
    reference, mixture = values
    matched_transform(reference, mixture)
    for path, (digest, length) in observed.items():
        _bound_bytes(path, digest, length)
    _bound_bytes(request, sha256, MAX_REQUEST_BYTES)
    return {
        "version": 1,
        "kind": "tempo-transform-pair-observation",
        "request": {"path": str(request), "sha256": sha256},
        "metadataPins": {str(path): digest for path, (digest, _) in observed.items()},
        "sameTransformVerified": True,
        "cuts": len(reference.slots),
        "outputSamples": reference.output_samples,
        "sampleRate": reference.sample_rate,
        "channels": reference.channels,
        "referenceOutputSHA256": reference.output_sha256,
        "mixtureOutputSHA256": mixture.output_sha256,
        "mediaBytesRead": 0,
        "currentMediaCryptoVerified": False,
        "protectedSpeechVerified": False,
        "modelLineageVerified": False,
        "qualityApproved": False,
        "syncApproved": False,
        "deliveryComplete": False,
    }


def pcm16_command(ffmpeg: str, source: str, output: str) -> tuple[str, ...]:
    """Explicit quantization only: preserve rate/channels/count, never trim or pad."""
    return (
        ffmpeg,
        "-nostdin",
        "-hide_banner",
        "-loglevel",
        "error",
        "-threads",
        "1",
        "-i",
        source,
        "-map",
        "0:a:0",
        "-vn",
        "-sn",
        "-dn",
        "-c:a",
        "pcm_s16le",
        "-threads",
        "1",
        "-dither_method",
        "none",
        "-n",
        output,
    )


def _quality_pin(value: object, pins: dict[str, dq.Fingerprint]) -> Path:
    name, digest = _pin(value)
    path = Path(name)
    if path.resolve(strict=True) != path or any(part.is_symlink() for part in path.parents):
        raise ValueError("Quality input must be canonical and not linked")
    if name in pins and pins[name]["sha256"] != digest:
        raise ValueError("Conflicting quality input pins")
    return dq.reference(value, pins)


def quality_inputs(
    request: Path, *, sha256: str
) -> tuple[dict[str, object], dict[str, dq.Fingerprint], dict[str, Path], int]:
    """Validate current measurement copies against completed exact original stages."""
    pins: dict[str, dq.Fingerprint] = {}
    request = _quality_pin({"path": str(request.absolute()), "sha256": sha256}, pins)
    config = _object(
        _json(_read(request, MAX_REQUEST_BYTES)),
        "version kind pairRequest pairResult composition mediaRoot copies ffmpeg "
        "protection modelConfig modelWeights hfCache runtime",
    )
    if (
        type(config["version"]) is not int
        or config["version"] != 1
        or config["kind"] != "native-tempo-quality-request"
        or config["protection"] != "all-selected-samples-conservative"
    ):
        raise ValueError("Unsupported conservative native tempo quality request")
    pair_request = _quality_pin(config["pairRequest"], pins)
    pair_result = _quality_pin(config["pairResult"], pins)
    pair = inspect_transform_pair(pair_request, sha256=pins[str(pair_request)]["sha256"])
    if _json(_read(pair_result, MAX_RECORD_BYTES)) != pair:
        raise ValueError("Paired transform observation differs")
    composition_path = _quality_pin(config["composition"], pins)
    composition = record(_json(_read(composition_path, MAX_RECORD_BYTES)))
    assessment = record(composition.get("aacAssessment"))
    if composition.get("kind") != "video-composition-timing-observation" or any(
        assessment.get(key) is not True
        for key in (
            "originalPCMExtentAccepted",
            "packetDecodedPresentationAccepted",
            "trackEditMappingAccepted",
        )
    ):
        raise ValueError("Original AAC presentation/extent contract has not passed")
    count = _integer(pair["outputSamples"])
    if (
        pair["sampleRate"] != 44100
        or pair["channels"] != 2
        or _pin(assessment.get("source"))[1] != pair["referenceOutputSHA256"]
        or assessment.get("decodedSamples") != count
        or assessment.get("decodedStart") != "0"
        or _fraction(assessment.get("decodedEnd")) != Fraction(count, 44100)
    ):
        raise ValueError("Quality output clocks differ")
    root = Path(str(config["mediaRoot"]))
    if (
        not root.is_absolute()
        or root.resolve(strict=True) != root
        or not root.is_dir()
        or any(part.is_symlink() for part in root.parents)
    ):
        raise ValueError("Invalid explicit measurement media root")
    executable = _quality_pin(config["ffmpeg"], pins)
    copies = _object(config["copies"], "voice mixture decoded")
    paths: dict[str, Path] = {}
    expected = {
        "voice": pair["referenceOutputSHA256"],
        "mixture": pair["mixtureOutputSHA256"],
        "decoded": _pin(assessment.get("candidate"))[1],
    }
    for role, value in copies.items():
        item = _object(value, "source output workflow checkpoint stage")
        source = _quality_pin(item["source"], pins)
        output = _quality_pin(item["output"], pins)
        if (
            pins[str(source)]["sha256"] != expected[role]
            or not output.is_relative_to(root)
            or output == source
            or dq.wav_info(output) != (44100, 2, count)
        ):
            raise ValueError("Measurement copy identity/clock differs")
        graph_path = _quality_pin(item["workflow"], pins)
        checkpoint_path = _quality_pin(item["checkpoint"], pins)
        graph = record(_json(_read(graph_path, MAX_REQUEST_BYTES)))
        stages = graph.get("stages")
        if not isinstance(stages, list) or not 1 <= len(stages) <= 16:
            raise ValueError("Invalid measurement producer graph")
        matches = [_stage(row) for row in stages if record(row).get("name") == item["stage"]]
        if len(matches) != 1:
            raise ValueError("Missing unique measurement producer")
        stage = matches[0]
        checkpoint = record(_json(_read(checkpoint_path, MAX_RECORD_BYTES)))
        if (
            stage.command != pcm16_command(str(executable), str(source), str(output))
            or stage.verify_command
            or stage.adopt_existing
            or stage.outputs != (output,)
            or not {source, executable}.issubset(stage.inputs)
            or checkpoint.get("status") != "complete"
            or checkpoint.get("fingerprint") != _fingerprint(stage)
            or checkpoint.get("outputs") != {str(output): pins[str(output)]["sha256"]}
        ):
            raise ValueError("PCM16 conversion producer is not an exact completed stage")
        paths[role] = output
    if len(set(paths.values())) != 3:
        raise ValueError("Independent measurement copies required")
    runtime = config["runtime"]
    if not isinstance(runtime, list) or not 1 <= len(runtime) <= 64:
        raise ValueError("Explicit bounded current measurement runtime required")
    for value in runtime:
        _quality_pin(value, pins)
    return config, pins, paths, count


def observe_quality_resources(directory: Path, phase: str) -> None:
    """Record distinct physical/MLX memory facts; observations are not allocation caps."""
    audit_descriptors(directory, phase)
    runtime = importlib.import_module("mlx.core")
    memory: dict[str, int] = {}
    for label, method in (
        ("activeBytes", "get_active_memory"),
        ("peakBytes", "get_peak_memory"),
        ("cacheBytes", "get_cache_memory"),
    ):
        value: object = getattr(runtime, method)()
        if type(value) is not int or value < 0:
            raise ValueError("Invalid MLX memory observation")
        memory[label] = value
    host = SystemHost()
    pid = os.getpid()
    birth = host.birth(pid)
    physical = host.footprint(pid)
    if host.birth(pid) != birth:
        raise ValueError("Quality observer process identity changed")
    with (directory / f"{phase}-memory.json").open("x", encoding="utf-8") as stream:
        json.dump(
            {
                "version": 1,
                "pid": pid,
                "birth": birth,
                "physicalFootprintBytes": physical,
                "mlx": memory,
                "peakReset": False,
                "hardAllocationLimitVerified": False,
            },
            stream,
            sort_keys=True,
        )
        stream.write("\n")


def measure_quality(request: Path, output: Path, *, sha256: str) -> dict[str, object]:
    """Run unchanged full-duration gates on an explicitly conservative output mask."""
    output = _local(str(output.absolute()))
    if not output.is_relative_to(Path.cwd() / "edits/work") or output.exists():
        raise ValueError("Quality proof must be new and repository-local")
    config, pins, paths, count = quality_inputs(request, sha256=sha256)
    directory = output.with_suffix(".windows")
    directory.mkdir(mode=0o700)
    observe_quality_resources(directory, "before-load")
    backend = dq.load_model(config, pins)
    observe_quality_resources(directory, "after-load")
    measurements = dq.quality_windows(
        decoded=paths["decoded"],
        first_pass=paths["voice"],
        model_input=paths["mixture"],
        spans=((0, count),),
        speech=((0, count),),
        backend=ObservedBackend(backend, directory),
        directory=directory,
        cache_key=sha256,
    )
    for name, pin in pins.items():
        if dq.fingerprint(Path(name)) != pin:
            raise ValueError("Quality input changed during measurement")
    return {
        "version": 1,
        "kind": "native-tempo-full-audio-quality-observation",
        "request": {"path": str(request.absolute()), "sha256": sha256},
        "inputs": pins,
        "measurements": measurements,
        "protection": "all-selected-samples-conservative",
        "protectedSamples": count,
        "ASRProtectionAccuracyVerified": False,
        "originalModelRuntimeVerified": False,
        "currentModelDevice": "mps",
        "measurementQuantization": (
            "PCM16 no dither; original inputs preserved, no rate/channel/count change"
        ),
        "qualityApproved": False,
        "syncApproved": False,
        "deliveryComplete": False,
    }


def run(
    request: Path,
    output: Path,
    *,
    sha256: str,
    verify: bool = False,
    audio_only: bool = False,
    packet_pair: bool = False,
    pcm_clock: bool = False,
    transform_pair: bool = False,
    quality: bool = False,
) -> None:
    """An exit zero verifies this mapping report, never delivery or audio quality."""
    if sum((audio_only, packet_pair, pcm_clock, transform_pair, quality)) > 1:
        raise ValueError("Choose exactly one observation contract")
    if quality and verify:
        raise ValueError("Quality measurement is not re-executed as a read-only verifier")
    result = (
        measure_quality(request, output, sha256=sha256)
        if quality
        else (
            inspect_transform_pair(request, sha256=sha256)
            if transform_pair
            else (
                inspect_pcm_clock(request, sha256=sha256)
                if pcm_clock
                else (
                    inspect_packet_pair(request, sha256=sha256)
                    if packet_pair
                    else (
                        inspect_audio_only(request, sha256=sha256)
                        if audio_only
                        else inspect_request(request, sha256=sha256)
                    )
                )
            )
        )
    )
    output = _local(str(output.absolute()))
    if not output.is_relative_to(Path.cwd() / "edits/work"):
        raise ValueError("Mapping output must remain in repository work")
    if verify:
        if _json(_read(output, MAX_RECORD_BYTES)) != result:
            raise ValueError("Mapping report changed")
        return
    with output.open("x", encoding="utf-8") as stream:
        json.dump(result, stream, indent=2, sort_keys=True)
        stream.write("\n")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("request", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--sha256", required=True)
    parser.add_argument("--verify", action="store_true")
    parser.add_argument("--audio-only", action="store_true")
    parser.add_argument("--packet-pair", action="store_true")
    parser.add_argument("--pcm-clock", action="store_true")
    parser.add_argument("--transform-pair", action="store_true")
    parser.add_argument("--quality", action="store_true")
    args = parser.parse_args()
    run(
        args.request,
        args.output,
        sha256=args.sha256,
        verify=args.verify,
        audio_only=args.audio_only,
        packet_pair=args.packet_pair,
        pcm_clock=args.pcm_clock,
        transform_pair=args.transform_pair,
        quality=args.quality,
    )


if __name__ == "__main__":
    main()
