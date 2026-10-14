"""Derive AAC presentation pre-roll from pinned, observed encoder exclusion.

No samples are encoded, padded or trimmed. The proof binds one source and one
producer command; ordinary negative-pre-roll sources must not be normalized again.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from dataclasses import dataclass
from fractions import Fraction
from itertools import pairwise
from pathlib import Path

from replay import compressed_part_delivery as delivery
from replay import whole_technical_batch as metadata
from replay.audio_clock import AudioFrameClock
from replay.core import ReplayError
from replay.recordings import record
from replay.stages import file_digest
from replay.whole_technical_batch import Reference

REF_KEYS = ("audio", "video", "packetProbe", "decodedProbe", "tagProbe", "ffmpeg")
CONFIG_KEYS = frozenset(
    {"version", "output", "timestampField", "videoTag", "videoTrackTimescale", *REF_KEYS}
)


@dataclass(frozen=True)
class Normalization:
    source: AudioFrameClock
    normalized: AudioFrameClock
    document: dict[str, object]


def _rows(raw: dict[str, object], key: str) -> tuple[dict[str, object], ...]:
    values = raw.get(key)
    if not isinstance(values, list) or not 1 <= len(values) <= 1_000_000:
        raise ReplayError("AAC normalization requires bounded complete observations")
    return tuple(record(value) for value in values)


def _clock(raw: dict[str, object], field: str) -> AudioFrameClock:
    frames: list[dict[str, object]] = []
    for row in _rows(raw, "frames"):
        point = row.get(field)
        if type(point) is not int or (
            "pts" in row
            and "best_effort_timestamp" in row
            and (
                type(row["pts"]) is not int
                or type(row["best_effort_timestamp"]) is not int
                or row["pts"] != row["best_effort_timestamp"]
            )
        ):
            raise ReplayError("Missing or divergent integral source audio timestamp")
        frames.append({**row, "pts": point})
    clock = AudioFrameClock.from_probe({**raw, "frames": frames})
    if any(
        (right - left) * clock.time_base != Fraction(samples, clock.sample_rate)
        for (left, right), samples in zip(pairwise(clock.pts), clock.samples, strict=False)
    ):
        raise ReplayError("AAC source decoded clock has a gap or overlap")
    return clock


def _tag(raw: dict[str, object]) -> tuple[int, int, int]:
    value = record(record(raw.get("format")).get("tags")).get("iTunSMPB")
    if not isinstance(value, str) or len(value) > 512:
        raise ReplayError("Missing bounded original Apple AAC gapless tag")
    words = value.split()
    if not 4 <= len(words) <= 16 or any(
        re.fullmatch(r"[0-9a-fA-F]{8,16}", word) is None for word in words
    ):
        raise ReplayError("Malformed Apple AAC gapless tag")
    return int(words[1], 16), int(words[2], 16), int(words[3], 16)


def _calculate(config: Reference) -> Normalization:
    raw = delivery._read(config)
    if set(raw) != CONFIG_KEYS or type(raw["version"]) is not int or raw["version"] != 1:
        raise ReplayError("Invalid AAC presentation-normalization configuration")
    field = raw["timestampField"]
    if not isinstance(field, str) or field not in {"pts", "best_effort_timestamp"}:
        raise ReplayError("Invalid explicit source timestamp field")
    tag, scale = raw["videoTag"], raw["videoTrackTimescale"]
    if (
        not isinstance(tag, str)
        or tag not in {"avc1", "hvc1"}
        or type(scale) is not int
        or not 1 <= scale <= 1_000_000
    ):
        raise ReplayError("Invalid explicit video container contract")
    refs = {key: delivery._ref(raw[key]) for key in REF_KEYS}
    if any(file_digest(ref.path) != ref.sha256 for ref in refs.values()):
        raise ReplayError("AAC normalization input bytes changed")
    packet_raw = delivery._read_probe(refs["packetProbe"])
    decoded = delivery._read_probe(refs["decodedProbe"])
    clock = _clock(decoded, field)
    streams = _rows(packet_raw, "streams")
    decoded_streams = _rows(decoded, "streams")
    if (
        len(streams) != 1
        or len(decoded_streams) != 1
        or streams[0].get("profile") != "LC"
        or ("profile" in decoded_streams[0] and decoded_streams[0]["profile"] != "LC")
        or any(
            stream.get("codec_name") != "aac"
            or stream.get("codec_type") != "audio"
            or stream.get("channels") != 2
            or stream.get("sample_rate") != str(clock.sample_rate)
            or stream.get("time_base") != str(clock.time_base)
            for stream in (*streams, *decoded_streams)
        )
    ):
        raise ReplayError("AAC source stream/rate/timebase/channel identity differs")
    packets = _rows(packet_raw, "packets")
    priming, remainder, samples = _tag(delivery._read(refs["tagProbe"]))
    skip = delivery._skip(packets[0], "skip_samples")
    ticks = Fraction(skip, clock.sample_rate) / clock.time_base
    if (
        not 0 < skip <= 65536
        or ticks.denominator != 1
        or type(streams[0].get("initial_padding")) is not int
        or streams[0]["initial_padding"] != skip
        or priming != skip
        or packets[0].get("pts") != 0
        or packets[0].get("dts") != 0
        or clock.pts[0] != ticks.numerator
        or samples != clock.sample_ends[-1]
        or remainder != delivery._skip(packets[-1], "discard_padding")
        or not delivery._audio_packets(packet_raw, clock)
    ):
        raise ReplayError(
            "AAC leading exclusion/tag/packet/decoded clock disagree or already normalized"
        )
    if any(
        type(row.get("pts")) is not int
        or type(row.get("dts")) is not int
        or row["pts"] != row["dts"]
        or not isinstance(row.get("data_hash"), str)
        or re.fullmatch(r"SHA256:[0-9a-f]{64}", str(row["data_hash"])) is None
        or (index > 0 and delivery._skip(row, "skip_samples") != 0)
        or (index < len(packets) - 1 and delivery._skip(row, "discard_padding") != 0)
        for index, row in enumerate(packets)
    ):
        raise ReplayError("AAC packet identity or internal priming differs")
    output = metadata._path(str(raw["output"]))
    if output in {ref.path for ref in refs.values()} or output.suffix != ".mp4":
        raise ReplayError("AAC normalized output must be a distinct MP4")
    normalized = AudioFrameClock(
        tuple(point - ticks.numerator for point in clock.pts),
        clock.samples,
        clock.sample_rate,
        clock.time_base,
    )
    expression = f"setts=pts=PTS-{ticks.numerator}:dts=DTS-{ticks.numerator}"
    argv = [
        str(refs["ffmpeg"].path),
        "-nostdin",
        "-v",
        "error",
        "-n",
        "-copyts",
        "-protocol_whitelist",
        "file,pipe",
        "-i",
        str(refs["video"].path),
        "-i",
        str(refs["audio"].path),
        "-map",
        "0:v:0",
        "-map",
        "1:a:0",
        "-c",
        "copy",
        "-bsf:a:0",
        expression,
        "-avoid_negative_ts",
        "disabled",
        "-use_editlist",
        "1",
        "-tag:v",
        tag,
        "-video_track_timescale",
        str(scale),
        "-movie_timescale",
        str(clock.sample_rate),
        "-movflags",
        "+faststart",
        str(output),
    ]
    source = Path(__file__).resolve()
    document: dict[str, object] = {
        "version": 1,
        "kind": "observed-aac-leading-exclusion-to-presentation-zero",
        "config": config.json(),
        "inputs": {key: ref.json() for key, ref in refs.items()},
        "reader": Reference(source, file_digest(source)).json(),
        "output": str(output),
        "timestampField": field,
        "primingSamples": skip,
        "preRollTicks": ticks.numerator,
        "timeBase": str(clock.time_base),
        "sampleRate": clock.sample_rate,
        "decodedSamples": clock.sample_ends[-1],
        "tailDiscardSamples": remainder,
        "sourceFrameClockSha256": hashlib.sha256(
            json.dumps([clock.pts, clock.samples], separators=(",", ":")).encode()
        ).hexdigest(),
        "producerCommand": argv,
        "deliveryApproved": False,
        "syncApproved": False,
        "qualityApproved": False,
    }
    if any(file_digest(ref.path) != ref.sha256 for ref in (*refs.values(), config)):
        raise ReplayError("AAC normalization inputs changed during derivation")
    return Normalization(clock, normalized, document)


def prepare(config: Path, output: Path) -> Normalization:
    """Write an immutable measured producer proof; never run the mux command."""
    config = metadata._path(str(config), work=True)
    value = _calculate(Reference(config, file_digest(config)))
    movie = Path(str(value.document["output"]))
    if movie.exists() or movie.is_symlink():
        raise ReplayError("AAC normalization producer output already exists")
    delivery._write(metadata._path(str(output), work=True), value.document)
    return value


def verify(proof: Reference) -> Normalization:
    """Freshly rederive the complete proof, including source/tool/code byte pins."""
    metadata._path(str(proof.path), work=True)
    saved = delivery._read(proof)
    value = _calculate(delivery._ref(saved.get("config")))
    if value.document != saved or file_digest(proof.path) != proof.sha256:
        raise ReplayError("AAC normalization proof differs from current measured facts")
    return value


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="action", required=True)
    create = sub.add_parser("prepare")
    create.add_argument("config", type=Path)
    create.add_argument("proof", type=Path)
    check = sub.add_parser("verify")
    check.add_argument("proof", type=Path)
    check.add_argument("--sha256", required=True)
    args = parser.parse_args(argv)
    try:
        value = (
            prepare(args.config, args.proof)
            if args.action == "prepare"
            else verify(delivery._ref({"path": str(args.proof), "sha256": args.sha256}))
        )
    except (ReplayError, OSError, ValueError) as error:
        print(f"AAC presentation normalization failed: {error}")
        return 1
    print(json.dumps(value.document, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
