"""Bounded STTS/CTTS observations, preserving decode and presentation domains.

No decoder, encoder, media-payload read, clock adjustment or whole-file hash.
The retained full-decoder request supplies expected candidate identity/provenance.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import struct
from collections.abc import Iterator
from dataclasses import dataclass
from fractions import Fraction
from pathlib import Path
from typing import BinaryIO

from replay import aac_presentation_acceptance as aac
from replay.core import ReplayError
from replay.recordings import record
from replay.youtube_delivery_validation import Box, _boxes

MAX_SAMPLES = 1_000_000
MAX_TABLE_BYTES = 8 + 8 * MAX_SAMPLES
MAX_ROW_BYTES = 65536
CHUNK_BYTES = 16384


@dataclass(frozen=True, slots=True)
class Sample:
    decode: int
    presentation: int
    duration: int


@dataclass(frozen=True, slots=True)
class Composition:
    track: aac.Track
    samples: tuple[Sample, ...]
    table_sha256: str
    ctts_present: bool

    def range_json(self) -> dict[str, object]:
        starts = [sample.presentation for sample in self.samples]
        return {
            "trackID": self.track.identifier,
            "timescale": self.track.timescale,
            "decodeDuration": sum(sample.duration for sample in self.samples),
            "presentationStart": min(starts),
            "presentationEnd": max(s.presentation + s.duration for s in self.samples),
            "sampleCount": len(self.samples),
            "tableSHA256": self.table_sha256,
            "cttsPresent": self.ctts_present,
        }


def _table(stream: BinaryIO, box: Box, *, composition: bool) -> tuple[tuple[int, int], ...]:
    length = box.end - box.payload
    if not 8 <= length <= MAX_TABLE_BYTES:
        raise ReplayError("Sample timing table exceeds byte budget")
    stream.seek(box.payload)
    payload = stream.read(length)
    if len(payload) != length or payload[1:4] != bytes(3):
        raise ReplayError("Truncated timing table or unknown flags")
    version = payload[0]
    if version not in ((0, 1) if composition else (0,)):
        raise ReplayError("Unknown sample timing table version")
    count = int.from_bytes(payload[4:8], "big")
    if not 1 <= count <= MAX_SAMPLES or length != 8 + 8 * count:
        raise ReplayError("Sample timing table count/bytes disagree")
    entries: list[tuple[int, int]] = []
    total = 0
    for position in range(8, length, 8):
        samples = int.from_bytes(payload[position : position + 4], "big")
        value = int.from_bytes(payload[position + 4 : position + 8], "big", signed=version == 1)
        total += samples
        if samples <= 0 or total > MAX_SAMPLES or (not composition and value <= 0):
            raise ReplayError("Timing run has invalid count/duration or exceeds budget")
        entries.append((samples, value))
    return tuple(entries)


def parse_composition(stream: BinaryIO, *, size: int, container: aac.Container) -> Composition:
    videos = [track for track in container.tracks if track.handler == "vide"]
    if len(videos) != 1:
        raise ReplayError("Exactly one video track is required")
    track = videos[0]
    boxes = _boxes(stream, start=0, end=size, depth=0)
    parents = [box for box in boxes if box.kind == b"trak"]
    matches = []
    for parent in parents:
        direct = [
            b
            for b in boxes
            if parent.payload <= b.offset < parent.end and b.depth == parent.depth + 1
        ]
        if aac._identifier(aac._payload(stream, aac._one(direct, b"tkhd"))) == track.identifier:
            matches.append(parent)
    parent = aac._one(matches, b"trak")
    descendants = [b for b in boxes if parent.payload <= b.offset < parent.end]
    table_parent = aac._one(descendants, b"stbl")
    tables = [
        b
        for b in descendants
        if table_parent.payload <= b.offset < table_parent.end and b.depth == table_parent.depth + 1
    ]
    stts = aac._one(tables, b"stts")
    ctts_boxes = [b for b in tables if b.kind == b"ctts"]
    if len(ctts_boxes) > 1:
        raise ReplayError("Duplicate composition timing tables")
    durations = _table(stream, stts, composition=False)
    count = sum(run[0] for run in durations)
    offsets = _table(stream, ctts_boxes[0], composition=True) if ctts_boxes else ((count, 0),)
    if sum(run[0] for run in offsets) != count:
        raise ReplayError("STTS/CTTS sample counts disagree")
    digest = hashlib.sha256()
    for box in [stts, *ctts_boxes]:
        stream.seek(box.payload)
        digest.update(struct.pack(">QQ4s", box.offset, box.end, box.kind))
        digest.update(stream.read(box.end - box.payload))
    expanded_offsets = (value for count, value in offsets for _ in range(count))
    samples: list[Sample] = []
    decode = 0
    for run_count, duration in durations:
        for _ in range(run_count):
            offset = next(expanded_offsets)
            samples.append(Sample(decode, decode + offset, duration))
            decode += duration
    if decode != track.duration or decode > 2**63 - 1:
        raise ReplayError("STTS duration differs from observed mdhd")
    ordered = sorted(samples, key=lambda sample: sample.presentation)
    if any(
        left.presentation + left.duration != right.presentation
        for left, right in zip(ordered, ordered[1:], strict=False)
    ):
        raise ReplayError("Composition timeline has gaps, overlaps or duplicate PTS")
    return Composition(track, tuple(samples), digest.hexdigest(), bool(ctts_boxes))


class _JSONReader:
    """Incremental bounded JSON values; retained raw rows are never materialized together."""

    def __init__(self, stream: BinaryIO) -> None:
        self.stream = stream
        self.buffer = ""
        self.end = False
        self.total = 0
        self.digest = hashlib.sha256()
        self.decoder = json.JSONDecoder(
            object_pairs_hook=aac._unique, parse_constant=self._constant
        )

    @staticmethod
    def _constant(value: str) -> object:
        raise ReplayError(f"Nonfinite JSON value: {value}")

    def _more(self) -> None:
        chunk = self.stream.read(CHUNK_BYTES)
        self.total += len(chunk)
        if self.total > aac.MAX_PROBE_BYTES:
            raise ReplayError("Raw observation exceeds byte budget")
        self.digest.update(chunk)
        if not chunk:
            self.end = True
        else:
            self.buffer += chunk.decode("ascii")

    def token(self) -> str:
        self.buffer = self.buffer.lstrip()
        while not self.buffer and not self.end:
            self._more()
            self.buffer = self.buffer.lstrip()
        return self.buffer[:1]

    def expect(self, token: str) -> None:
        if self.token() != token:
            raise ReplayError("Malformed raw observation JSON structure")
        self.buffer = self.buffer[1:]

    def value(self) -> object:
        self.token()
        while True:
            try:
                value, end = self.decoder.raw_decode(self.buffer)
            except json.JSONDecodeError:
                if self.end or len(self.buffer) > MAX_ROW_BYTES:
                    raise ReplayError("Malformed or oversized raw JSON value") from None
                self._more()
                continue
            if end == len(self.buffer) and not self.end:
                self._more()
                continue
            self.buffer = self.buffer[end:]
            return value


def raw_rows(reference: aac.Reference, key: str) -> Iterator[dict[str, object]]:
    aac._canonical(reference.path)
    if not reference.path.is_relative_to(aac.ROOT / "edits/work"):
        raise ReplayError("Raw observations must stay in checkout")
    with os.fdopen(os.open(reference.path, os.O_RDONLY | os.O_NOFOLLOW), "rb") as stream:
        before = os.fstat(stream.fileno())
        if not 0 < before.st_size <= aac.MAX_PROBE_BYTES or before.st_nlink != 1:
            raise ReplayError("Raw observation exceeds file budget")
        reader = _JSONReader(stream)
        reader.expect("{")
        keys: set[str] = set()
        while reader.token() != "}":
            name = reader.value()
            if not isinstance(name, str) or name in keys:
                raise ReplayError("Duplicate or invalid raw JSON key")
            keys.add(name)
            reader.expect(":")
            if name == key:
                reader.expect("[")
                count = 0
                while reader.token() != "]":
                    count += 1
                    if count > MAX_SAMPLES * 2:
                        raise ReplayError("Raw row count exceeds budget")
                    yield record(reader.value())
                    if reader.token() == "]":
                        break
                    reader.expect(",")
                    if reader.token() == "]":
                        raise ReplayError("Trailing raw row comma")
                reader.expect("]")
            else:
                reader.value()
            if reader.token() == "}":
                break
            reader.expect(",")
            if reader.token() == "}":
                raise ReplayError("Trailing raw object comma")
        reader.expect("}")
        if reader.token() or key not in keys:
            raise ReplayError("Raw JSON has trailing data or missing rows")
        if (
            reader.digest.hexdigest() != reference.sha256
            or aac._file_facts(os.fstat(stream.fileno())) != aac._file_facts(before)
            or aac._file_facts(reference.path.lstat()) != aac._file_facts(before)
        ):
            raise ReplayError("Raw observations changed or differ from their pin")


def corroborate(composition: Composition, request: aac.ObservationRequest) -> dict[str, object]:
    receipt, raw = aac._original_provenance(request)
    streams = record(receipt["streams"])["streams"]
    if not isinstance(streams, list):
        raise ReplayError("Original video metadata is missing")
    videos = [record(item) for item in streams if record(item).get("codec_type") == "video"]
    if len(videos) != 1:
        raise ReplayError("Exactly one original video stream required")
    video = videos[0]
    video_index = aac._integer(video.get("index"))
    if Fraction(str(video.get("time_base"))) != Fraction(1, composition.track.timescale):
        raise ReplayError("Observed video/table time bases differ")
    edits = composition.track.edits
    if len(edits) > 1 or (edits and (edits[0].rate != 1 or edits[0].media_time < 0)):
        raise ReplayError("Unsupported composition edit mapping")
    shift = edits[0].media_time if edits else 0
    packet_index = 0
    for row in raw_rows(raw["packets"], "packets"):
        if aac._integer(row.get("stream_index")) != video_index:
            continue
        if packet_index >= len(composition.samples):
            raise ReplayError("Extra video packet")
        sample = composition.samples[packet_index]
        clocks = (
            aac._integer(row.get("dts"), minimum=-(2**63 - 1)),
            aac._integer(row.get("pts"), minimum=-(2**63 - 1)),
            aac._integer(row.get("duration"), minimum=1),
        )
        if clocks != (sample.decode - shift, sample.presentation - shift, sample.duration):
            raise ReplayError("Video packets differ from STTS/CTTS/edit mapping")
        packet_index += 1
    ordered = sorted(composition.samples, key=lambda sample: sample.presentation)
    frame_index = 0
    for row in raw_rows(raw["frames"], "frames"):
        if row.get("media_type") != "video":
            continue
        if frame_index >= len(ordered):
            raise ReplayError("Extra decoded video frame")
        sample = ordered[frame_index]
        if (
            aac._integer(row.get("pts"), minimum=-(2**63 - 1)) != sample.presentation - shift
            or aac._integer(row.get("duration", row.get("pkt_duration")), minimum=1)
            != sample.duration
        ):
            raise ReplayError("Decoded video PTS differ from composition range")
        frame_index += 1
    if packet_index != len(ordered) or frame_index != len(ordered):
        raise ReplayError("Video sample/packet/frame counts differ")
    return {
        "packetCount": packet_index,
        "decodedFrameCount": frame_index,
        "editAppliedOnce": True,
        "negativeDecodePrerollRetained": min(sample.decode for sample in ordered) - shift < 0,
    }


def observe(
    request: aac.ObservationRequest, *, prior_result: aac.Reference | None = None
) -> dict[str, object]:
    if aac.load_observation_request(request.reference) != request:
        raise ReplayError("Typed request differs from its immutable input")
    aac._current_candidate(request)
    with os.fdopen(os.open(request.candidate.path, os.O_RDONLY | os.O_NOFOLLOW), "rb") as stream:
        identity = aac.FileIdentity.from_stat(os.fstat(stream.fileno()))
        if identity != request.identity:
            raise ReplayError("Composition candidate FD changed")
        container = aac.parse_container_headers(stream, size=identity.size)
        composition = parse_composition(stream, size=identity.size, container=container)
        if aac.FileIdentity.from_stat(os.fstat(stream.fileno())) != identity:
            raise ReplayError("Composition candidate changed during table read")
    corroboration = corroborate(composition, request)
    aac._original_provenance(request)
    aac._current_candidate(request)
    document: dict[str, object] = {
        "version": 1,
        "kind": "video-composition-timing-observation",
        "request": request.reference.json(),
        "candidate": request.candidate.json(),
        "expectedIdentity": request.identity.json(),
        "timing": container.json(),
        "composition": composition.range_json(),
        "corroboration": corroboration,
        "rawInputs": [
            p.stdout.json() for p in request.processes if p.role in ("frames", "packets")
        ],
        "wholeSHAFreshlyVerified": False,
        "newDecode": False,
        "syncApproved": False,
        "qualityApproved": False,
        "deliveryApproved": False,
    }
    if prior_result is not None:
        prior = aac._evidence(prior_result).document()
        if prior.get("request") != request.reference.json():
            raise ReplayError("Prior AAC result binds another request")
        derived_rows = record(prior.get("normalization")).get("derived")
        if not isinstance(derived_rows, list) or len(derived_rows) != 2:
            raise ReplayError("Prior AAC derived observations missing")
        derived = tuple(aac._evidence(aac._reference(row)) for row in derived_rows)
        if (
            aac._cli_report(request, record(prior.get("observation")), (derived[0], derived[1]))
            != prior
        ):
            raise ReplayError("Prior AAC result no longer matches its original domain")
        payload = aac._json_bytes(document)
        evidence = aac.Evidence(
            aac.Reference(prior_result.path, hashlib.sha256(payload).hexdigest()), payload
        )
        document["priorAACResult"] = prior_result.json()
        document["aacAssessment"] = aac.assess(
            container,
            aac.Inputs(
                request.candidate, derived[0], derived[1], request.source, request.content, evidence
            ),
        )
    if aac.load_observation_request(request.reference) != request:
        raise ReplayError("Composition request changed during observation")
    aac._current_candidate(request)
    return document


def validate(
    document: dict[str, object], *, candidate: aac.Reference, track: aac.Track
) -> tuple[int, int]:
    row = record(document.get("composition"))
    corroboration = record(document.get("corroboration"))
    for field in ("trackID", "timescale", "decodeDuration", "sampleCount"):
        aac._integer(row.get(field), minimum=1)
    aac._identity(document.get("expectedIdentity"))
    if (
        type(document.get("version")) is not int
        or document.get("version") != 1
        or document.get("kind") != "video-composition-timing-observation"
        or document.get("candidate") != candidate.json()
        or row.get("trackID") != track.identifier
        or row.get("timescale") != track.timescale
        or row.get("decodeDuration") != track.duration
        or corroboration.get("editAppliedOnce") is not True
        or corroboration.get("packetCount") != row.get("sampleCount")
        or corroboration.get("decodedFrameCount") != row.get("sampleCount")
        or document.get("newDecode") is not False
        or document.get("wholeSHAFreshlyVerified") is not False
    ):
        raise ReplayError("Composition evidence differs from candidate/track/domain")
    start = aac._integer(row.get("presentationStart"), minimum=-(2**63 - 1))
    end = aac._integer(row.get("presentationEnd"), minimum=1)
    if end <= start:
        raise ReplayError("Composition range is empty")
    aac._integer(row.get("sampleCount"), minimum=1)
    digest = row.get("tableSHA256")
    if not isinstance(digest, str):
        raise ReplayError("Composition table pin missing")
    aac.Reference(candidate.path, digest)
    return start, end


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("observe", "verify"))
    parser.add_argument("request", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--sha256", required=True)
    parser.add_argument("--aac-result", type=Path)
    parser.add_argument("--aac-result-sha256")
    args = parser.parse_args(argv)
    try:
        request = aac.load_observation_request(aac.Reference(args.request, args.sha256))
        if (args.aac_result is None) != (args.aac_result_sha256 is None):
            raise ReplayError("Prior AAC result needs both path and SHA256")
        prior = (
            None
            if args.aac_result is None
            else aac.Reference(args.aac_result, args.aac_result_sha256)
        )
        if args.action == "observe":
            aac._output_paths(args.output)
            aac._exclusive(args.output, aac._json_bytes(observe(request, prior_result=prior)))
        else:
            aac._canonical(args.output)
            if not 0 < args.output.stat().st_size <= aac.MAX_REQUEST_BYTES:
                raise ReplayError("Composition result exceeds byte budget")
            with args.output.open("rb") as stream:
                payload = stream.read(aac.MAX_REQUEST_BYTES + 1)
            if len(payload) > aac.MAX_REQUEST_BYTES or record(json.loads(payload)) != observe(
                request, prior_result=prior
            ):
                raise ReplayError("Composition result changed or domain differs")
            aac._read(aac.Reference(args.output, hashlib.sha256(payload).hexdigest()))
    except (ReplayError, OSError, ValueError) as error:
        parser.exit(1, f"Video composition observation: {error}\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
