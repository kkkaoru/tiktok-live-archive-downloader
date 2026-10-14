"""Observe whole AV once and verify copied AAC without changing its clocks.

Combined decoder/frame/packet evidence is immutable and reused by later domains.
The legacy audio adapter also accepts an existing technical result. Neither path
encodes, remuxes, publishes or approves editing quality or synchronization.
"""

from __future__ import annotations

import argparse
import json
import re
from fractions import Fraction
from itertools import pairwise
from pathlib import Path
from typing import Literal

from replay import aac_presentation_normalization as normalization
from replay import compressed_part_delivery as delivery
from replay import delivery_verification as technical
from replay import whole_technical_batch as metadata
from replay.audio_clock import AudioFrameClock
from replay.caption_measured_job import _json as strict_json
from replay.core import ReplayError
from replay.delivery_clock_contract import load_clock_contract
from replay.foreground_evidence import _bound_bytes
from replay.recordings import record
from replay.stages import file_digest

MAX_PROBE_BYTES = 256 * 1024 * 1024
OBSERVATION_VERSION = 3
VideoContract = Literal["hevc", "h264"]
SourceTimestampField = Literal["pts", "best_effort_timestamp"]


def _source_timestamp_field(value: object) -> SourceTimestampField:
    if value == "pts":
        return "pts"
    if value == "best_effort_timestamp":
        return "best_effort_timestamp"
    raise ReplayError("Unsupported explicit source audio timestamp field")


def _source_clock(
    probe: dict[str, object], *, timestamp_field: SourceTimestampField, channels: object
) -> AudioFrameClock:
    """Use observed integral timestamps; never guess a missing native clock."""
    frames: list[dict[str, object]] = []
    for frame in _rows(probe, "frames"):
        point = frame.get(timestamp_field)
        if type(point) is not int:
            raise ReplayError("Missing integral selected source audio timestamp")
        if (
            "pts" in frame
            and "best_effort_timestamp" in frame
            and (
                type(frame["pts"]) is not int
                or type(frame["best_effort_timestamp"]) is not int
                or frame["pts"] != frame["best_effort_timestamp"]
            )
        ):
            raise ReplayError("Source audio timestamp formats disagree")
        frames.append({**frame, "pts": point})
    if timestamp_field != "pts":
        streams = _rows(probe, "streams")
        if (
            len(streams) != 1
            or streams[0].get("codec_type") != "audio"
            or streams[0].get("codec_name") != "aac"
            or type(channels) is not int
            or not 1 <= channels <= 8
            or type(streams[0].get("channels")) is not int
            or streams[0]["channels"] != channels
        ):
            raise ReplayError("Explicit source audio stream/channel binding differs")
    return AudioFrameClock.from_probe({**probe, "frames": frames})


def _video_contract(value: object) -> VideoContract:
    if isinstance(value, str) and value == "hevc":
        return "hevc"
    if isinstance(value, str) and value == "h264":
        return "h264"
    raise ReplayError("Unsupported explicit video contract")


def _receipt_contract(receipt: dict[str, object], contract: VideoContract) -> None:
    """Legacy unlabelled HEVC stays HEVC; an H264 result must be explicitly bound."""
    found = False
    for field in ("videoContract", "expectedVideoContract"):
        if field in receipt:
            found = True
            if _video_contract(receipt[field]) != contract:
                raise ReplayError("Technical receipt video contract differs")
    if not found and (contract != "hevc" or receipt.get("version") == OBSERVATION_VERSION):
        raise ReplayError("Technical receipt lacks an explicit video contract")


def _probe(path: Path) -> dict[str, object]:
    path = metadata._path(str(path), work=True)
    return record(strict_json(_bound_bytes(path, file_digest(path), MAX_PROBE_BYTES)))


def _rows(raw: dict[str, object], key: str) -> tuple[dict[str, object], ...]:
    rows = raw.get(key)
    if not isinstance(rows, list) or not 1 <= len(rows) <= 4_000_000:
        raise ReplayError("Whole audio requires complete bounded probe rows")
    return tuple(record(row) for row in rows)


def _payloads(raw: dict[str, object]) -> tuple[str, ...]:
    result: list[str] = []
    for row in _rows(raw, "packets"):
        value = row.get("data_hash")
        if not isinstance(value, str) or re.fullmatch(r"SHA256:[0-9a-f]{64}", value) is None:
            raise ReplayError("Every AAC packet needs an observed payload SHA256")
        result.append(value)
    return tuple(result)


def observation_command(plan: delivery.Plan) -> list[str]:
    """One software-decoder scan, with packet/decoded-frame observations together."""
    return [
        str(plan.tools["ffprobe"].path),
        "-v",
        "error",
        "-err_detect",
        "explode",
        "-protocol_whitelist",
        "file,pipe",
        "-show_error",
        "-show_frames",
        "-show_packets",
        "-show_streams",
        "-show_format",
        "-show_data_hash",
        "sha256",
        "-show_entries",
        "frame=media_type,stream_index,pts,duration,pkt_duration,nb_samples,decode_error_flags:"
        "packet=stream_index,pts,dts,duration,pos,size,data_hash:"
        "packet_side_data=side_data_type,skip_samples,discard_padding:stream:format:error",
        "-of",
        "json",
        str(plan.output),
    ]


def whole_observations(
    plan: delivery.Plan, raw: dict[str, object], *, video_contract: str = "hevc"
) -> tuple[dict[str, object], dict[str, object], dict[str, object], dict[str, object]]:
    """Require all decoded AV frames; metadata-only/packet-only cannot pass."""
    contract = _video_contract(video_contract)
    if raw.get("error") is not None:
        raise ReplayError("Whole decoder observation reported an error")
    stream_rows = list(_rows(raw, "streams"))
    streams: dict[str, object] = {"streams": stream_rows}
    expectation = load_clock_contract(plan.contract.path)
    video, audio = technical._streams(streams, expectation, video_contract=contract)
    if len(stream_rows) != 2:
        raise ReplayError("Whole observation requires exactly video and audio")
    rows = _rows(raw, "packets_and_frames")
    frames = [row for row in rows if row.get("type") == "frame"]
    packets = [row for row in rows if row.get("type") == "packet"]
    if len(frames) + len(packets) != len(rows) or any(
        row.get("media_type") not in {"audio", "video"} or row.get("decode_error_flags", 0) != 0
        for row in frames
    ):
        raise ReplayError("Unknown or corrupt decoded AV observation")
    video_frames = [
        row
        for row in frames
        if row.get("stream_index") == video.get("index") and row.get("media_type") == "video"
    ]
    audio_frames = [
        row
        for row in frames
        if row.get("stream_index") == audio.get("index") and row.get("media_type") == "audio"
    ]
    if len(video_frames) + len(audio_frames) != len(frames):
        raise ReplayError("Decoded observations name unbound streams")
    video_clock = technical._video(
        video_frames,
        scale=delivery._fraction(video.get("time_base")),
        rate=plan.rate,
        frames=plan.frames,
    )
    audio_clock = AudioFrameClock.from_probe({"streams": [audio], "frames": audio_frames})
    if any(
        (right - left) * audio_clock.time_base != Fraction(count, audio_clock.sample_rate)
        for (left, right), count in zip(
            pairwise(audio_clock.pts), audio_clock.samples, strict=False
        )
    ):
        raise ReplayError("Whole decoded audio has a gap or overlap")
    video_packets = [row for row in packets if row.get("stream_index") == video.get("index")]
    audio_packets = [row for row in packets if row.get("stream_index") == audio.get("index")]
    if len(video_packets) != plan.frames or len(video_packets) + len(audio_packets) != len(packets):
        raise ReplayError("Whole packet count/stream binding differs")
    audio_probe: dict[str, object] = {"streams": [audio], "packets": audio_packets}
    if not delivery._audio_packets(audio_probe, audio_clock):
        raise ReplayError("Whole decoded/packet priming extent differs")
    _payloads(audio_probe)
    receipt: dict[str, object] = {
        "version": OBSERVATION_VERSION,
        "videoContract": contract,
        "technicalAccepted": True,
        "fullDecode": True,
        "decodePasses": 1,
        "decoder": "ffprobe-combined-frames-packets",
        "videoClock": video_clock,
        "inputs": [metadata.Reference(plan.output, file_digest(plan.output)).json()],
        "deliveryApproved": False,
    }
    return receipt, streams, {"frames": frames}, audio_probe


def _inputs(plan: delivery.Plan) -> tuple[metadata.Reference, ...]:
    refs = {ref.path: ref for ref in delivery._references(plan)}
    refs.update(
        {path: metadata.Reference(path, file_digest(path)) for path in delivery.code_inputs()}
    )
    refs[plan.output] = metadata.Reference(plan.output, file_digest(plan.output))
    result = tuple(refs[path] for path in sorted(refs))
    if any(file_digest(ref.path) != ref.sha256 for ref in result):
        raise ReplayError("Whole observation source/tool/evidence changed")
    return result


def _producer_inputs(
    plan: delivery.Plan, evidence: Path, producer_binding: metadata.Reference | None
) -> tuple[metadata.Reference, ...]:
    """An explicit old reader archive may explain exactly one code-pin change."""
    current = _inputs(plan)
    if producer_binding is None:
        return current
    binding = delivery._read(producer_binding)
    if set(binding) != {"version", "observation", "producerReader", "currentReader"} or (
        type(binding.get("version")) is not int or binding["version"] != 1
    ):
        raise ReplayError("Invalid explicit historical reader binding")
    observation = delivery._ref(binding["observation"])
    if observation.path != evidence / "observation.json":
        raise ReplayError("Historical reader names a different observation")
    delivery._read(observation)
    producer = record(binding["producerReader"])
    reader = delivery._ref(binding["currentReader"])
    source = Path(__file__).resolve()
    if set(producer) != {"sourcePath", "historicalBytes"} or (
        producer["sourcePath"] != str(source)
        or reader.path != source
        or file_digest(reader.path) != reader.sha256
    ):
        raise ReplayError("Historical reader must bind the exact current reader")
    archive = delivery._ref(producer["historicalBytes"])
    metadata._path(str(archive.path), work=True)
    if archive.path == source or file_digest(archive.path) != archive.sha256:
        raise ReplayError("Historical reader archive bytes differ")
    if sum(ref.path == source for ref in current) != 1:
        raise ReplayError("Current input closure must bind the reader exactly once")
    return tuple(
        metadata.Reference(ref.path, archive.sha256) if ref.path == source else ref
        for ref in current
    )


def _process(plan: delivery.Plan, evidence: Path, reference: metadata.Reference) -> None:
    if reference.path != evidence / "combined.process.json":
        raise ReplayError("Whole decoder process artifact path differs")
    process = delivery._read(reference)
    if (
        process.get("returncode") != 0
        or type(process.get("returncode")) is not int
        or (
            process.get("command") != observation_command(plan)
            or metadata._path(str(evidence / "combined.stderr"), work=True).stat().st_size != 0
        )
    ):
        raise ReplayError("Saved whole decoder process failed or stderr changed")


def _saved_observation(
    plan: delivery.Plan,
    evidence: Path,
    *,
    video_contract: VideoContract = "hevc",
    producer_binding: metadata.Reference | None = None,
) -> tuple[dict[str, object], dict[str, object], dict[str, object], dict[str, object]]:
    manifest = metadata._metadata(evidence / "observation.json")
    if set(manifest) != {
        "version",
        "kind",
        "videoContract",
        "inputs",
        "raw",
        "process",
        "snapshot",
    } or (
        type(manifest["version"]) is not int
        or manifest["version"] != OBSERVATION_VERSION
        or manifest["kind"] != "single-pass-whole-observation"
        or manifest["videoContract"] != video_contract
        or manifest["inputs"]
        != [ref.json() for ref in _producer_inputs(plan, evidence, producer_binding)]
    ):
        raise ReplayError("Saved whole observation is not bound to current inputs")
    snapshot_ref = delivery._ref(manifest["snapshot"])
    if snapshot_ref.path != evidence / "input-snapshot.json":
        raise ReplayError("Whole decoder input snapshot path differs")
    expected_snapshot = _snapshot(plan, video_contract=video_contract)
    if producer_binding is not None:
        expected_snapshot["inputs"] = [
            ref.json() for ref in _producer_inputs(plan, evidence, producer_binding)
        ]
    if delivery._read(snapshot_ref) != expected_snapshot:
        raise ReplayError("Whole decoder input snapshot changed")
    _process(plan, evidence, delivery._ref(manifest["process"]))
    raw_ref = delivery._ref(manifest["raw"])
    if raw_ref.path != evidence / "combined.stdout":
        raise ReplayError("Whole observation raw artifact path differs")
    return whole_observations(plan, delivery._read_probe(raw_ref), video_contract=video_contract)


def _snapshot(plan: delivery.Plan, *, video_contract: VideoContract = "hevc") -> dict[str, object]:
    return {
        "version": OBSERVATION_VERSION,
        "videoContract": video_contract,
        "command": observation_command(plan),
        "inputs": [ref.json() for ref in _inputs(plan)],
    }


def _finish_observation(
    plan: delivery.Plan, evidence: Path, *, video_contract: VideoContract = "hevc"
) -> dict[str, object]:
    """Adopt only a complete error-free scan bound to its pre-scan input snapshot."""
    snapshot_path = evidence / "input-snapshot.json"
    snapshot = metadata._metadata(snapshot_path)
    if snapshot != _snapshot(plan, video_contract=video_contract):
        raise ReplayError("Whole observation inputs changed during the one decode")
    process_ref = metadata.Reference(
        evidence / "combined.process.json", file_digest(evidence / "combined.process.json")
    )
    _process(plan, evidence, process_ref)
    raw_ref = metadata.Reference(
        evidence / "combined.stdout", file_digest(evidence / "combined.stdout")
    )
    values = whole_observations(plan, delivery._read_probe(raw_ref), video_contract=video_contract)
    if snapshot != _snapshot(plan, video_contract=video_contract):
        raise ReplayError("Whole observation inputs changed while validating the scan")
    delivery._write(
        evidence / "observation.json",
        {
            "version": OBSERVATION_VERSION,
            "videoContract": video_contract,
            "kind": "single-pass-whole-observation",
            "inputs": snapshot["inputs"],
            "raw": raw_ref.json(),
            "process": process_ref.json(),
            "snapshot": metadata.Reference(snapshot_path, file_digest(snapshot_path)).json(),
        },
    )
    return values[0]


def observe(config: Path, evidence: Path, *, video_contract: str = "hevc") -> dict[str, object]:
    """Decode once when absent; otherwise revalidate pinned observations only."""
    contract = _video_contract(video_contract)
    plan = delivery.load(config)
    evidence = metadata._path(str(evidence), work=True)
    if evidence.exists():
        if (evidence / "observation.json").exists():
            return _saved_observation(plan, evidence, video_contract=contract)[0]
        return _finish_observation(plan, evidence, video_contract=contract)
    snapshot = _snapshot(plan, video_contract=contract)
    evidence.mkdir(mode=0o700)
    delivery._write(evidence / "input-snapshot.json", snapshot)
    technical._run(observation_command(plan), directory=evidence, name="combined", timeout=7200)
    return _finish_observation(plan, evidence, video_contract=contract)


def observed_result(
    config: Path,
    evidence: Path,
    output: Path,
    *,
    audio: bool,
    video_contract: str = "hevc",
    source_timestamp_field: str = "pts",
    producer_binding: metadata.Reference | None = None,
    normalization_binding: metadata.Reference | None = None,
) -> dict[str, object]:
    """Derived domains consume the same immutable observation without decoding."""
    contract = _video_contract(video_contract)
    timestamp_field = _source_timestamp_field(source_timestamp_field)
    if not audio and (timestamp_field != "pts" or normalization_binding is not None):
        raise ReplayError("Explicit source timestamp field requires the audio domain")
    plan = delivery.load(config)
    evidence = metadata._path(str(evidence), work=True)
    receipt, streams, frames, packets = _saved_observation(
        plan, evidence, video_contract=contract, producer_binding=producer_binding
    )
    result = (
        compare(
            plan,
            receipt,
            streams,
            frames,
            packets,
            video_contract=contract,
            source_timestamp_field=timestamp_field,
            normalization_binding=normalization_binding,
        )
        if audio
        else receipt
    )
    if producer_binding is not None:
        binding = delivery._read(producer_binding)
        result["readerProvenance"] = {
            "binding": producer_binding.json(),
            "observation": binding["observation"],
            "producerReader": binding["producerReader"],
            "currentReader": binding["currentReader"],
            "decoderReused": True,
        }
        if audio:
            result["sourceClockTimestampField"] = timestamp_field
        _producer_inputs(plan, evidence, producer_binding)
        manifest = delivery._read(delivery._ref(binding["observation"]))
        artifacts = (
            ("raw", "combined.stdout"),
            ("snapshot", "input-snapshot.json"),
            ("process", "combined.process.json"),
        )
        if any(
            delivery._ref(manifest[key]).path != evidence / filename
            or file_digest(delivery._ref(manifest[key]).path) != delivery._ref(manifest[key]).sha256
            for key, filename in artifacts
        ):
            raise ReplayError("Historical decoder artifacts changed during derived verification")
        _process(plan, evidence, delivery._ref(manifest["process"]))
    if normalization_binding is not None:
        normalization.verify(normalization_binding)
    path = metadata._path(str(output), work=True)
    if path.exists():
        if metadata._metadata(path) != result:
            raise ReplayError("Saved derived whole domain result differs")
    else:
        delivery._write(path, result)
    return result


def compare(
    plan: delivery.Plan,
    receipt: dict[str, object],
    streams: dict[str, object],
    frames: dict[str, object],
    packets: dict[str, object],
    *,
    video_contract: str = "hevc",
    source_timestamp_field: str = "pts",
    normalization_binding: metadata.Reference | None = None,
) -> dict[str, object]:
    """Evaluate all observed clocks and payloads; no review flags are manufactured."""
    contract = _video_contract(video_contract)
    timestamp_field = _source_timestamp_field(source_timestamp_field)
    _receipt_contract(receipt, contract)
    if plan.audio is None:
        raise ReplayError("Whole audio proof is required")
    proof = delivery._read(delivery._ref(plan.audio["evidence"]))
    original = delivery._read_probe(delivery._ref(proof["packetProbe"]))
    audio = [row for row in _rows(streams, "streams") if row.get("codec_type") == "audio"]
    if len(audio) != 1:
        raise ReplayError("Assembled media must contain exactly one audio track")
    clock = AudioFrameClock.from_probe(
        {
            "streams": audio,
            "frames": [row for row in _rows(frames, "frames") if row.get("media_type") == "audio"],
        }
    )
    expected = Fraction(plan.frames, plan.rate)
    source = _source_clock(
        delivery._read_probe(delivery._ref(proof["decodedProbe"])),
        timestamp_field=timestamp_field,
        channels=audio[0].get("channels"),
    )
    normalized: normalization.Normalization | None = None
    if normalization_binding is not None:
        normalized = normalization.verify(normalization_binding)
        values = record(normalized.document["inputs"])
        if (
            values["audio"] != plan.audio["movie"]
            or values["packetProbe"] != proof["packetProbe"]
            or values["decodedProbe"] != proof["decodedProbe"]
            or values["ffmpeg"] != plan.tools["ffmpeg"].json()
            or normalized.document["output"] != str(plan.output)
            or normalized.document["timestampField"] != timestamp_field
            or normalized.source != source
        ):
            raise ReplayError(
                "AAC normalization does not bind the adopted source/output/tool clock"
            )
        source = normalized.normalized
    reasons: list[str] = []
    bound = receipt.get("inputs")
    if not isinstance(bound, list) or not any(
        record(row).get("path") == str(plan.output)
        and record(row).get("sha256") == file_digest(plan.output)
        for row in bound
    ):
        raise ReplayError("Technical receipt must bind the unchanged assembled movie")
    if receipt.get("technicalAccepted") is not True or receipt.get("fullDecode") is not True:
        reasons.append("whole technical/full AV verification rejected")
    source_clock_preserved = clock == source
    if not source_clock_preserved:
        reasons.append("decoded audio PTS/sample counts differ from the single adopted whole audio")
    end = clock.pts[-1] * clock.time_base + Fraction(clock.samples[-1], clock.sample_rate)
    if (
        clock.pts[0] != 0
        or end != expected
        or (Fraction(clock.sample_ends[-1], clock.sample_rate) != expected)
    ):
        reasons.append(f"joined audio extent differs: end {end}, expected {expected}")
    payload_equal = _payloads(packets) == _payloads(original)
    if not payload_equal:
        reasons.append("ordered AAC packet payloads changed")
    internal_priming_preserved = delivery._audio_packets(packets, clock)
    if not internal_priming_preserved:
        reasons.append("joined AAC packet priming/remainder differs from decoded extent")
    integrity_preserved = (
        source_clock_preserved
        and payload_equal
        and internal_priming_preserved
        and (receipt.get("technicalAccepted") is True and receipt.get("fullDecode") is True)
    )
    candidate_mux_ready = plan.allow_clock_candidate and integrity_preserved
    result: dict[str, object] = {
        "version": OBSERVATION_VERSION,
        "videoContract": contract,
        "audioMuxAccepted": not reasons,
        "route": "compressed-technical-candidate"
        if not reasons
        else ("compressed-clock-candidate" if candidate_mux_ready else "prores-fallback"),
        "candidateMuxReady": candidate_mux_ready,
        "fallback": plan.fallback.json(),
        "reasons": reasons,
        "decodedSamples": clock.sample_ends[-1],
        "sampleRate": clock.sample_rate,
        "decodedEnd": str(end),
        "expectedEnd": str(expected),
        "orderedPayloadEqual": payload_equal,
        "sourceClockPreserved": source_clock_preserved,
        "internalPrimingPreserved": internal_priming_preserved,
        "deliveryApproved": False,
        "syncApproved": False,
        "qualityApproved": False,
    }
    if timestamp_field != "pts":
        result["sourceClockTimestampField"] = timestamp_field
    if normalized is not None and normalization_binding is not None:
        result["presentationNormalization"] = {
            "proof": normalization_binding.json(),
            "primingSamples": normalized.document["primingSamples"],
            "preRollTicks": normalized.document["preRollTicks"],
            "sourceClockPreservedUnderDeclaredMapping": source_clock_preserved,
        }
    return result


def verify(
    config: Path, evidence: Path, output: Path, *, video_contract: str = "hevc"
) -> dict[str, object]:
    """Collect only bounded packet observations of an already assembled movie."""
    contract = _video_contract(video_contract)
    plan = delivery.load(config)
    evidence = metadata._path(str(evidence), work=True)
    output = metadata._path(str(output), work=True)
    receipt = metadata._metadata(evidence / "receipt.json")
    _receipt_contract(receipt, contract)
    output.mkdir(mode=0o700)
    packet_path = technical._run(
        [
            str(plan.tools["ffprobe"].path),
            "-v",
            "error",
            "-protocol_whitelist",
            "file,pipe",
            "-select_streams",
            "a:0",
            "-show_streams",
            "-show_packets",
            "-show_data_hash",
            "sha256",
            "-show_entries",
            "stream=index,codec_name,codec_type,profile,sample_rate,channels,time_base:"
            "packet=pts,duration,data_hash:packet_side_data=side_data_type,skip_samples,discard_padding",
            "-of",
            "json",
            str(plan.output),
        ],
        directory=output,
        name="packets",
        timeout=7200,
    )
    result = compare(
        plan,
        receipt,
        _probe(evidence / "streams.stdout"),
        _probe(evidence / "frames.stdout"),
        _probe(packet_path),
        video_contract=contract,
    )
    delivery._write(output / "result.json", result)
    return result


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("config", type=Path)
    parser.add_argument("evidence", type=Path)
    parser.add_argument("output", type=Path, nargs="?")
    parser.add_argument("--video-contract", choices=("hevc", "h264"), default="hevc")
    parser.add_argument(
        "--source-timestamp-field", choices=("pts", "best_effort_timestamp"), default="pts"
    )
    parser.add_argument("--producer-binding", type=Path)
    parser.add_argument("--producer-binding-sha256")
    parser.add_argument("--normalization-binding", type=Path)
    parser.add_argument("--normalization-sha256")
    actions = parser.add_mutually_exclusive_group()
    actions.add_argument("--observe", action="store_true")
    actions.add_argument("--technical-result", action="store_true")
    actions.add_argument("--audio-result", action="store_true")
    args = parser.parse_args(argv)
    contract = _video_contract(args.video_contract)
    producer_binding: metadata.Reference | None = None
    normalization_binding: metadata.Reference | None = None
    if (args.normalization_binding is None) != (args.normalization_sha256 is None):
        raise ReplayError("AAC normalization binding requires its exact SHA256")
    if args.normalization_binding is not None:
        if not args.audio_result:
            raise ReplayError("AAC normalization requires the explicit audio-result domain")
        normalization_binding = delivery._ref(
            {"path": str(args.normalization_binding), "sha256": args.normalization_sha256}
        )
    if (args.producer_binding is None) != (args.producer_binding_sha256 is None):
        raise ReplayError("Historical reader binding requires its exact SHA256")
    if args.producer_binding is not None:
        producer_binding = delivery._ref(
            {"path": str(args.producer_binding), "sha256": args.producer_binding_sha256}
        )
    if (producer_binding is not None or args.source_timestamp_field != "pts") and not (
        args.audio_result
        or (
            args.technical_result
            and producer_binding is not None
            and args.source_timestamp_field == "pts"
        )
    ):
        raise ReplayError("Explicit reader/source clock options require a derived domain")
    if args.observe:
        result = observe(args.config, args.evidence, video_contract=contract)
        print(json.dumps(result, ensure_ascii=False))
        return 0
    if args.output is None:
        raise ReplayError("Derived verification requires an output path")
    if args.technical_result or args.audio_result:
        result = observed_result(
            args.config,
            args.evidence,
            args.output,
            audio=args.audio_result,
            video_contract=contract,
            source_timestamp_field=args.source_timestamp_field,
            producer_binding=producer_binding,
            normalization_binding=normalization_binding,
        )
        print(json.dumps(result, ensure_ascii=False))
        return (
            0
            if args.technical_result
            or result["audioMuxAccepted"] is True
            or result["candidateMuxReady"] is True
            else 1
        )
    result = verify(args.config, args.evidence, args.output, video_contract=contract)
    print(json.dumps(result, ensure_ascii=False))
    return 0 if result["audioMuxAccepted"] is True or result["candidateMuxReady"] is True else 1


if __name__ == "__main__":
    raise SystemExit(main())
