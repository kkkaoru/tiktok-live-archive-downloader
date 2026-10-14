"""Pure AAC presentation evidence, separate from YouTube upload recommendations.

Decoded PTS already include the demuxer's edit/skip policy. Never apply that
policy twice, infer silence, or shift/trim the media to obtain an accepted clock.
The optional header observer reads timing metadata only, not compressed media.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import stat
import struct
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import asdict, dataclass
from fractions import Fraction
from math import isfinite
from pathlib import Path
from typing import BinaryIO

from replay.audio_clock import AudioFrameClock
from replay.bgm_separation import SeparationPolicy
from replay.core import ReplayError
from replay.delivery_audio_quality import MINIMUM_CORRELATION
from replay.recordings import record
from replay.youtube_delivery_validation import Box, _boxes

MAX_TRACKS = 32
MAX_EDITS = 32
MAX_PAYLOAD_BYTES = 4096
MAX_PROBE_BYTES = 128 * 1024**2
MAX_PACKETS = 1_000_000
MAX_SKIP = 65536
TIMING_BOXES = frozenset({b"mvhd", b"tkhd", b"mdhd", b"hdlr", b"elst"})
ROOT = Path(__file__).resolve().parents[2]
MAX_REQUEST_BYTES = 1024**2
PROCESS_ROLES = ("decode", "streams", "frames", "packets")


@dataclass(frozen=True, slots=True)
class Reference:
    path: Path
    sha256: str

    def __post_init__(self) -> None:
        if (
            not self.path.is_absolute()
            or ".." in self.path.parts
            or re.fullmatch(r"[a-f0-9]{64}", self.sha256) is None
        ):
            raise ReplayError("Expected an absolute AAC evidence reference and SHA256")

    def json(self) -> dict[str, str]:
        return {"path": str(self.path), "sha256": self.sha256}


@dataclass(frozen=True, slots=True)
class Evidence:
    """Small/raw immutable bytes; the hash is checked on every pure assessment."""

    reference: Reference
    payload: bytes

    def document(self) -> dict[str, object]:
        if (
            not 0 < len(self.payload) <= MAX_PROBE_BYTES
            or hashlib.sha256(self.payload).hexdigest() != self.reference.sha256
        ):
            raise ReplayError("AAC observation bytes differ from their evidence pin")
        try:
            return record(json.loads(self.payload, object_pairs_hook=_unique))
        except (ValueError, UnicodeDecodeError) as error:
            raise ReplayError("Invalid AAC observation JSON") from error


@dataclass(frozen=True, slots=True)
class Edit:
    duration: int
    media_time: int
    rate: Fraction

    def __post_init__(self) -> None:
        if (
            type(self.duration) is not int
            or self.duration <= 0
            or type(self.media_time) is not int
            or self.media_time < -1
            or not isinstance(self.rate, Fraction)
        ):
            raise ReplayError("Invalid typed edit timing")


@dataclass(frozen=True, slots=True)
class Track:
    identifier: int
    handler: str
    timescale: int
    duration: int
    edits: tuple[Edit, ...]
    roll_metadata_present: bool

    def __post_init__(self) -> None:
        if (
            type(self.identifier) is not int
            or self.identifier <= 0
            or self.handler not in {"soun", "vide"}
            or type(self.timescale) is not int
            or not 1 <= self.timescale < 2**32
            or type(self.duration) is not int
            or self.duration <= 0
            or not isinstance(self.edits, tuple)
            or len(self.edits) > MAX_EDITS
            or any(not isinstance(edit, Edit) for edit in self.edits)
            or type(self.roll_metadata_present) is not bool
        ):
            raise ReplayError("Invalid typed track timing")


@dataclass(frozen=True, slots=True)
class Container:
    timescale: int
    duration: int
    tracks: tuple[Track, ...]
    timing_payload_sha256: str
    original_facts_sha256: str

    def __post_init__(self) -> None:
        if (
            type(self.timescale) is not int
            or not 1 <= self.timescale < 2**32
            or type(self.duration) is not int
            or self.duration <= 0
            or not isinstance(self.tracks, tuple)
            or not 1 <= len(self.tracks) <= MAX_TRACKS
            or any(not isinstance(track, Track) for track in self.tracks)
            or len({track.identifier for track in self.tracks}) != len(self.tracks)
            or re.fullmatch(r"[a-f0-9]{64}", self.timing_payload_sha256) is None
        ):
            raise ReplayError("Invalid typed container timing")
        if self.original_facts_sha256 != _facts_hash(
            self.timescale, self.duration, self.tracks, self.timing_payload_sha256
        ):
            raise ReplayError("Parsed container timing facts were changed in memory")

    def json(self) -> dict[str, object]:
        return {
            "movieTimescale": self.timescale,
            "movieDuration": self.duration,
            "timingPayloadSHA256": self.timing_payload_sha256,
            "originalFactsSHA256": self.original_facts_sha256,
            "tracks": [
                {
                    "trackID": track.identifier,
                    "handler": track.handler,
                    "mediaTimescale": track.timescale,
                    "mediaDuration": track.duration,
                    "rollMetadataPresent": track.roll_metadata_present,
                    "edits": [
                        {
                            "segmentDuration": edit.duration,
                            "mediaTime": edit.media_time,
                            "mediaRate": str(edit.rate),
                        }
                        for edit in track.edits
                    ],
                }
                for track in self.tracks
            ],
        }


@dataclass(frozen=True, slots=True)
class SourcePCM:
    reference: Reference
    sample_rate: int
    samples: int
    start: Fraction = Fraction(0)
    clock_evidence: Evidence | None = None

    def __post_init__(self) -> None:
        if (
            type(self.sample_rate) is not int
            or not 8000 <= self.sample_rate <= 192000
            or type(self.samples) is not int
            or self.samples <= 0
            or not isinstance(self.start, Fraction)
            or self.start < 0
        ):
            raise ReplayError("Original PCM requires an explicit observed sample clock")


@dataclass(frozen=True, slots=True)
class Inputs:
    candidate: Reference
    decoded: Evidence
    packets: Evidence
    source: SourcePCM | None = None
    content_comparison: Evidence | None = None
    video_composition: Evidence | None = None


@dataclass(frozen=True, slots=True)
class FileIdentity:
    device: int
    inode: int
    size: int
    modified_ns: int
    changed_ns: int

    @classmethod
    def from_stat(cls, value: os.stat_result) -> FileIdentity:
        if not stat.S_ISREG(value.st_mode) or value.st_size <= 0:
            raise ReplayError("AAC container must be a nonempty regular file")
        return cls(value.st_dev, value.st_ino, value.st_size, value.st_mtime_ns, value.st_ctime_ns)

    def json(self) -> list[int]:
        return [self.device, self.inode, self.size, self.modified_ns, self.changed_ns]


@dataclass(frozen=True, slots=True)
class ProcessEvidence:
    role: str
    process: Reference
    stdout: Reference
    stderr: Reference


@dataclass(frozen=True, slots=True)
class ObservationRequest:
    reference: Reference
    candidate: Reference
    identity: FileIdentity
    receipt: Reference
    checkpoint: Reference
    tools: tuple[Reference, Reference]
    processes: tuple[ProcessEvidence, ...]
    source: SourcePCM | None
    source_producer: Reference | None
    content: Evidence | None


def _one(boxes: list[Box], kind: bytes) -> Box:
    matches = [box for box in boxes if box.kind == kind]
    if len(matches) != 1:
        raise ReplayError(f"Expected exactly one {kind.decode('ascii')} timing box")
    return matches[0]


def _unique(pairs: list[tuple[str, object]]) -> dict[str, object]:
    if len({name for name, _ in pairs}) != len(pairs):
        raise ReplayError("Duplicate AAC observation JSON keys")
    return dict(pairs)


def _facts_hash(timescale: int, duration: int, tracks: tuple[Track, ...], payload_sha: str) -> str:
    payload = json.dumps(
        [timescale, duration, [asdict(track) for track in tracks], payload_sha],
        sort_keys=True,
        default=str,
    ).encode()
    return hashlib.sha256(payload).hexdigest()


def _payload(stream: BinaryIO, box: Box) -> bytes:
    size = box.end - box.payload
    if not 4 <= size <= MAX_PAYLOAD_BYTES:
        raise ReplayError("Timing box payload exceeds its byte budget")
    stream.seek(box.payload)
    value = stream.read(size)
    if len(value) != size:
        raise ReplayError("Truncated timing box payload")
    return value


def _clock(payload: bytes) -> tuple[int, int]:
    version = payload[0]
    offsets = {0: (12, 16, 4), 1: (20, 24, 8)}
    if version not in offsets:
        raise ReplayError("Unknown timing box version")
    scale_at, duration_at, width = offsets[version]
    if len(payload) < duration_at + width:
        raise ReplayError("Truncated movie/media timing fields")
    scale = int.from_bytes(payload[scale_at : scale_at + 4], "big")
    duration = int.from_bytes(payload[duration_at : duration_at + width], "big")
    if not scale or not duration or duration == 2 ** (width * 8) - 1:
        raise ReplayError("Unknown or empty movie/media duration")
    return scale, duration


def _identifier(payload: bytes) -> int:
    offset = {0: 12, 1: 20}.get(payload[0])
    if offset is None or len(payload) < offset + 4:
        raise ReplayError("Unknown or truncated track header")
    identifier = int.from_bytes(payload[offset : offset + 4], "big")
    if identifier <= 0:
        raise ReplayError("Track ID must be positive")
    return identifier


def _edits(payload: bytes) -> tuple[Edit, ...]:
    widths = {0: (4, 12), 1: (8, 20)}
    if payload[0] not in widths or len(payload) < 8 or payload[1:4] != b"\0\0\0":
        raise ReplayError("Unknown edit-list version/flags")
    width, entry_size = widths[payload[0]]
    count = int.from_bytes(payload[4:8], "big")
    if not 1 <= count <= MAX_EDITS or len(payload) != 8 + count * entry_size:
        raise ReplayError("Edit-list entries are missing, oversized or truncated")
    result: list[Edit] = []
    for offset in range(8, len(payload), entry_size):
        duration = int.from_bytes(payload[offset : offset + width], "big")
        media = int.from_bytes(payload[offset + width : offset + width * 2], "big", signed=True)
        rate = int.from_bytes(payload[offset + width * 2 : offset + entry_size], "big", signed=True)
        if duration <= 0 or media < -1:
            raise ReplayError("Invalid edit-list duration or media time")
        result.append(Edit(duration, media, Fraction(rate, 65536)))
    return tuple(result)


def parse_container_headers(stream: BinaryIO, *, size: int) -> Container:
    """Read bounded headers/timing fields with the existing BMFF walker; skip mdat."""
    if type(size) is not int or size <= 0:
        raise ReplayError("Container size must be positive")
    boxes = _boxes(stream, start=0, end=size, depth=0)
    moov = _one([box for box in boxes if box.depth == 0], b"moov")
    inside = [box for box in boxes if moov.payload <= box.offset < moov.end]
    direct = [box for box in inside if box.depth == moov.depth + 1]
    timescale, duration = _clock(_payload(stream, _one(direct, b"mvhd")))
    track_boxes = [box for box in direct if box.kind == b"trak"]
    if not 1 <= len(track_boxes) <= MAX_TRACKS:
        raise ReplayError("Container must have bounded timing tracks")
    tracks: list[Track] = []
    for parent in track_boxes:
        children = [box for box in inside if parent.payload <= box.offset < parent.end]
        track_direct = [box for box in children if box.depth == parent.depth + 1]
        identifier = _identifier(_payload(stream, _one(track_direct, b"tkhd")))
        mdia = _one(track_direct, b"mdia")
        media_direct = [
            box
            for box in children
            if mdia.payload <= box.offset < mdia.end and box.depth == mdia.depth + 1
        ]
        scale, length = _clock(_payload(stream, _one(media_direct, b"mdhd")))
        handler = _payload(stream, _one(media_direct, b"hdlr"))
        if len(handler) < 12 or handler[8:12] not in {b"soun", b"vide"}:
            raise ReplayError("Unknown or unsupported timing-track handler")
        edit_boxes = [box for box in children if box.kind == b"elst"]
        if len(edit_boxes) > 1:
            raise ReplayError("Duplicate edit lists in one track")
        if edit_boxes:
            edts = _one(track_direct, b"edts")
            if (
                not edts.payload <= edit_boxes[0].offset < edts.end
                or edit_boxes[0].depth != edts.depth + 1
            ):
                raise ReplayError("Edit list is outside its own track edit container")
        edits = _edits(_payload(stream, edit_boxes[0])) if edit_boxes else ()
        roll = {box.kind for box in children} >= {b"sgpd", b"sbgp"}
        tracks.append(Track(identifier, handler[8:12].decode("ascii"), scale, length, edits, roll))
    if len({track.identifier for track in tracks}) != len(tracks):
        raise ReplayError("Duplicate timing-track IDs")
    digest = hashlib.sha256()
    for box in boxes:
        if box.kind in TIMING_BOXES:
            payload = _payload(stream, box)
            digest.update(struct.pack(">QQ4s", box.offset, box.end, box.kind) + payload)
    payload_sha = digest.hexdigest()
    frozen_tracks = tuple(tracks)
    return Container(
        timescale,
        duration,
        frozen_tracks,
        payload_sha,
        _facts_hash(timescale, duration, frozen_tracks, payload_sha),
    )


def observe_headers(reference: Reference, *, expected_identity: FileIdentity) -> dict[str, object]:
    """Future admitted read only: expected whole SHA is historical, not re-hashed.

    No output creation, decoder, media payload read or app action. Caller owns
    admission/publication and must verify this observation's immutable bytes.
    """
    path = reference.path
    if path.resolve(strict=True) != path or any(
        parent.is_symlink() for parent in (path, *path.parents)
    ):
        raise ReplayError("Container observation must not traverse links")
    with os.fdopen(os.open(path, os.O_RDONLY | os.O_NOFOLLOW), "rb") as stream:
        before = FileIdentity.from_stat(os.fstat(stream.fileno()))
        if before != expected_identity or FileIdentity.from_stat(path.lstat()) != before:
            raise ReplayError("Container differs from the expected identity")
        timing = parse_container_headers(stream, size=before.size)
        after = FileIdentity.from_stat(os.fstat(stream.fileno()))
        if (
            after != before
            or FileIdentity.from_stat(path.lstat()) != before
            or path.resolve(strict=True) != path
        ):
            raise ReplayError("Container changed during its header observation")
    return {
        "version": 1,
        "kind": "aac-container-timing-observation",
        "candidate": reference.json(),
        "expectedWholeSHA256FreshlyVerified": False,
        "identityBefore": before.json(),
        "identityAfter": after.json(),
        "timing": timing.json(),
        "fullDecodePerformed": False,
        "syncApproved": False,
        "deliveryApproved": False,
    }


def _integer(value: object, *, minimum: int = 0) -> int:
    if type(value) is not int or value < minimum or value > 2**63 - 1:
        raise ReplayError("Expected a bounded integral AAC clock")
    return value


def _audio_rows(
    document: dict[str, object], key: str
) -> tuple[dict[str, object], list[dict[str, object]]]:
    raw_streams, raw_rows = document.get("streams"), document.get(key)
    if (
        not isinstance(raw_streams, list)
        or not 1 <= len(raw_streams) <= MAX_TRACKS
        or not isinstance(raw_rows, list)
        or not 1 <= len(raw_rows) <= MAX_PACKETS
    ):
        raise ReplayError("Complete bounded audio observations are required")
    streams = [record(value) for value in raw_streams]
    audio = [row for row in streams if row.get("codec_type", "audio") == "audio"]
    if len(audio) != 1:
        raise ReplayError("Exactly one observed audio stream is required")
    stream = audio[0]
    rows = [record(value) for value in raw_rows]
    if len(streams) > 1:
        index = _integer(stream.get("index"))
        rows = [row for row in rows if row.get("stream_index") == index]
    if not rows:
        raise ReplayError("Observed audio stream contains no clocks")
    return stream, rows


def _skip(row: dict[str, object], name: str) -> int:
    raw = row.get("side_data_list", [])
    if not isinstance(raw, list) or len(raw) > 16:
        raise ReplayError("AAC packet side data exceeds its budget")
    rows = [record(item) for item in raw if record(item).get("side_data_type") == "Skip Samples"]
    if len(rows) > 1:
        raise ReplayError("Ambiguous AAC skip/discard observations")
    value = _integer(rows[0].get(name, 0)) if rows else 0
    if value > MAX_SKIP:
        raise ReplayError("AAC skip/discard exceeds its bound")
    return value


def _packet_extent(
    stream: dict[str, object], rows: list[dict[str, object]], clock: AudioFrameClock
) -> tuple[Fraction, Fraction, list[str]]:
    if (
        stream.get("codec_name") != "aac"
        or stream.get("profile") != "LC"
        or stream.get("sample_rate") != str(clock.sample_rate)
        or stream.get("channels") != 2
        or stream.get("time_base") != str(clock.time_base)
    ):
        raise ReplayError("AAC packet and decoded stream contracts differ")
    points = [_integer(row.get("pts"), minimum=-(2**63)) for row in rows]
    durations = [_integer(row.get("duration"), minimum=1) for row in rows]
    reasons: list[str] = []
    if any(a + length != b for a, b, length in zip(points, points[1:], durations, strict=False)):
        reasons.append("AAC packet clock has gaps or overlaps")
    if any(_skip(row, "skip_samples") for row in rows[1:]) or any(
        _skip(row, "discard_padding") for row in rows[:-1]
    ):
        reasons.append("AAC skip/discard is present inside the program")
    start = points[0] * clock.time_base + Fraction(
        _skip(rows[0], "skip_samples"), clock.sample_rate
    )
    raw_end = (points[-1] + durations[-1]) * clock.time_base
    decoded_end = clock.pts[-1] * clock.time_base + Fraction(clock.samples[-1], clock.sample_rate)
    stream_end = (
        _integer(stream.get("start_pts")) + _integer(stream.get("duration_ts"), minimum=1)
    ) * clock.time_base
    discard = Fraction(_skip(rows[-1], "discard_padding"), clock.sample_rate)
    # Same decoder+stream corroboration as delivery_audio_quality.decode_aac.
    end = (
        decoded_end
        if discard and raw_end - decoded_end == discard and stream_end == decoded_end
        else raw_end
    )
    if start != clock.pts[0] * clock.time_base or end != decoded_end or stream_end != end:
        reasons.append("AAC packet skip/discard, stream and decoded extents differ")
    return start, end, reasons


def _edit_reasons(container: Container, clock: AudioFrameClock, inputs: Inputs) -> list[str]:
    audios = [track for track in container.tracks if track.handler == "soun"]
    videos = [track for track in container.tracks if track.handler == "vide"]
    if len(audios) != 1 or len(videos) != 1:
        return ["Exactly one audio and video timing track is required"]
    reasons: list[str] = []
    for track in container.tracks:
        if not track.edits:
            continue
        if len(track.edits) != 1 or track.edits[0].rate != 1 or track.edits[0].media_time < 0:
            reasons.append("Unsupported multiple/empty/dwell/rate-changing track edit")
            continue
        edit = track.edits[0]
        lower, upper = 0, track.duration
        if track.handler == "vide" and inputs.video_composition is not None:
            from replay.video_composition_timing import validate

            observed = inputs.video_composition.document()
            if observed.get("timing") != container.json():
                raise ReplayError("Composition observation binds different container timing")
            lower, upper = validate(observed, candidate=inputs.candidate, track=track)
        if edit.media_time < lower or Fraction(edit.media_time, track.timescale) + Fraction(
            edit.duration, container.timescale
        ) > Fraction(upper, track.timescale):
            reasons.append("Track edit exceeds observed media duration")
    audio = audios[0]
    extent = (
        Fraction(audio.edits[0].duration, container.timescale)
        if len(audio.edits) == 1
        else Fraction(audio.duration, audio.timescale)
    )
    decoded_end = clock.pts[-1] * clock.time_base + Fraction(clock.samples[-1], clock.sample_rate)
    if clock.pts[0] != 0 or extent != decoded_end:
        reasons.append("Audio track edit/presentation duration differs from decoded clock")
    return reasons


def _content(inputs: Inputs, clock: AudioFrameClock) -> bool:
    source, evidence = inputs.source, inputs.content_comparison
    if source is None or evidence is None:
        return False
    report = evidence.document()
    checks = record(report.get("checks"))
    aac = record(record(report.get("measurements")).get("aac"))
    pins = report.get("inputs")
    if not isinstance(pins, list):
        raise ReplayError("Source-content comparison lacks its input pins")
    references = [record(pin) for pin in pins]
    bound = {_content_pin(pin) for pin in references}
    return (
        type(report.get("version")) is int
        and report.get("version") == 1
        and report.get("kind") == "delivery-audio-quality"
        and report.get("phase") == "quality"
        and report.get("accepted") is True
        and all(
            checks.get(name) is True
            for name in (
                "selectedPCMReceiptBound",
                "nativeTimesPreserved",
                "aacSampleTimeMapping",
                "speechGatePassed",
                "musicGatePassed",
                "inputsStable",
            )
        )
        and (str(source.reference.path), source.reference.sha256) in bound
        and (str(inputs.candidate.path), inputs.candidate.sha256) in bound
        and aac.get("selectedSamples") == source.samples
        and type(aac.get("selectedSamples")) is int
        and aac.get("decodedSamples") == clock.sample_ends[-1]
        and type(aac.get("decodedSamples")) is int
        and aac.get("sampleRate") == clock.sample_rate
        and type(aac.get("sampleRate")) is int
        and aac.get("firstDecodedPTS") == str(source.start)
        and _quality_complete(
            record(record(report.get("measurements")).get("quality")), source.samples
        )
    )


def _speech_passed(value: object) -> bool:
    if value is None:
        return False
    row = record(value)
    numbers = [
        row.get(name) for name in ("correlation", "level_delta_db_p50", "level_delta_db_p05")
    ]
    if any(type(number) not in {int, float} for number in numbers):
        return False
    correlation, median, lower = (float(str(number)) for number in numbers)
    return (
        all(isfinite(number) for number in (correlation, median, lower))
        and MINIMUM_CORRELATION <= correlation <= 1
        and min(median, lower) >= -SeparationPolicy().maximum_speech_loss_db
    )


def _quality_complete(quality: dict[str, object], samples: int) -> bool:
    """Require whole-program measurements, not an isolated accepted boolean."""
    windows = quality.get("windows")
    if not isinstance(windows, list) or not 1 <= len(windows) <= MAX_PACKETS:
        return False
    if (
        quality.get("policy") != asdict(SeparationPolicy())
        or quality.get("minimumSpeechCorrelation") != MINIMUM_CORRELATION
        or not _speech_passed(quality.get("speech"))
    ):
        return False
    cursor = protected = 0
    for value in windows:
        row = record(value)
        end, count = (
            _integer(row.get("endSample"), minimum=1),
            _integer(row.get("protectedSamples")),
        )
        if (
            type(row.get("startSample")) is not int
            or row.get("startSample") != cursor
            or not cursor < end <= samples
            or count > end - cursor
        ):
            return False
        if record(row.get("music")).get("passed") is not True:
            return False
        if count and (
            row.get("speechGatePassed") is not True or not _speech_passed(row.get("speech"))
        ):
            return False
        if not count and (row.get("speech") is not None or row.get("speechGatePassed") is not None):
            return False
        cursor, protected = end, protected + count
    return (
        cursor == samples
        and protected > 0
        and type(quality.get("protectedSamples")) is int
        and quality.get("protectedSamples") == protected
        and quality.get("speechGatePassed") is True
        and quality.get("musicGatePassed") is True
    )


def _content_pin(pin: dict[str, object]) -> tuple[str, str]:
    path, sha = pin.get("path"), pin.get("sha256")
    if not isinstance(path, str) or not isinstance(sha, str):
        raise ReplayError("Source comparison contains invalid input references")
    reference = Reference(Path(path), sha)
    return str(reference.path), reference.sha256


def _source_clock(source: SourcePCM | None) -> bool:
    if source is None or source.clock_evidence is None:
        return False
    report = source.clock_evidence.document()
    if (
        type(report.get("version")) is not int
        or report.get("version") != 1
        or report.get("kind") != "original-pcm-clock-observation"
        or report.get("source") != source.reference.json()
        or type(report.get("sampleRate")) is not int
        or report.get("sampleRate") != source.sample_rate
        or type(report.get("sampleCount")) is not int
        or report.get("sampleCount") != source.samples
        or report.get("start") != str(source.start)
    ):
        raise ReplayError("Original PCM clock evidence differs from the supplied source")
    return True


def assess(container: Container, inputs: Inputs) -> dict[str, object]:
    """Evaluate evidence only; caller retains fresh pins/full AV/producer domains."""
    decoded_stream, frames = _audio_rows(inputs.decoded.document(), "frames")
    clock = AudioFrameClock.from_probe({"streams": [decoded_stream], "frames": frames})
    packet_stream, packets = _audio_rows(inputs.packets.document(), "packets")
    start, end, packet_reasons = _packet_extent(packet_stream, packets, clock)
    if any(
        (right - left) * clock.time_base != Fraction(count, clock.sample_rate)
        for left, right, count in zip(clock.pts, clock.pts[1:], clock.samples, strict=False)
    ):
        packet_reasons.append("Decoded AAC clock has gaps or overlaps")
    edit_reasons = _edit_reasons(container, clock, inputs)
    source = inputs.source
    extent = (
        _source_clock(source)
        and source is not None
        and start == source.start
        and end == source.start + Fraction(source.samples, source.sample_rate)
    )
    content = _content(inputs, clock) and extent
    count_delta = (
        None
        if source is None
        else str(
            Fraction(clock.sample_ends[-1])
            - Fraction(source.samples * clock.sample_rate, source.sample_rate)
        )
    )
    return {
        "version": 1,
        "kind": "aac-presentation-assessment",
        "candidate": inputs.candidate.json(),
        "evidence": [inputs.decoded.reference.json(), inputs.packets.reference.json()],
        "youtubeNoEditListsAccepted": not any(track.edits for track in container.tracks),
        "packetDecodedPresentationAccepted": not packet_reasons,
        "trackEditMappingAccepted": not edit_reasons,
        "originalPCMExtentAccepted": extent,
        "originalPCMContentAccepted": content,
        "presentationSyncAccepted": not packet_reasons and not edit_reasons and extent and content,
        "decodedStart": str(clock.pts[0] * clock.time_base),
        "decodedEnd": str(
            clock.pts[-1] * clock.time_base + Fraction(clock.samples[-1], clock.sample_rate)
        ),
        "decodedSamples": clock.sample_ends[-1],
        "decodedSamplesMinusOriginalProjection": count_delta,
        "packetReasons": packet_reasons,
        "trackReasons": edit_reasons,
        "source": None if source is None else source.reference.json(),
        "sourceClockEvidence": None
        if source is None or source.clock_evidence is None
        else source.clock_evidence.reference.json(),
        "contentComparison": None
        if inputs.content_comparison is None
        else inputs.content_comparison.reference.json(),
        "timing": container.json(),
        "primingAppliedAgain": False,
        "qualityApproved": False,
        "syncApproved": False,
        "deliveryApproved": False,
    }


def _canonical(path: Path) -> None:
    if path.resolve(strict=True) != path or any(
        item.is_symlink() for item in (path, *path.parents)
    ):
        raise ReplayError("AAC evidence must not traverse links")


def _read(reference: Reference, *, empty: bool = False, external: bool = False) -> bytes:
    """Fresh bounded observation bytes; this never reads a candidate/PCM payload."""
    _canonical(reference.path)
    if not external and not reference.path.is_relative_to(ROOT / "edits/work"):
        raise ReplayError("AAC observation metadata must remain under checkout edits/work")
    with os.fdopen(os.open(reference.path, os.O_RDONLY | os.O_NOFOLLOW), "rb") as stream:
        before = os.fstat(stream.fileno())
        if (
            not stat.S_ISREG(before.st_mode)
            or before.st_nlink != 1
            or not (0 if empty else 1) <= before.st_size <= MAX_PROBE_BYTES
        ):
            raise ReplayError("AAC evidence exceeds its byte or file budget")
        payload = stream.read(MAX_PROBE_BYTES + 1)
        if (
            _file_facts(os.fstat(stream.fileno())) != _file_facts(before)
            or _file_facts(reference.path.lstat()) != _file_facts(before)
            or len(payload) != before.st_size
            or hashlib.sha256(payload).hexdigest() != reference.sha256
        ):
            raise ReplayError("AAC evidence bytes changed or differ from their pin")
    return payload


def _file_facts(value: os.stat_result) -> tuple[int, int, int, int, int, int, int]:
    return (
        value.st_dev,
        value.st_ino,
        value.st_size,
        value.st_mtime_ns,
        value.st_ctime_ns,
        value.st_mode,
        value.st_nlink,
    )


def _reference(value: object) -> Reference:
    row = record(value)
    name, digest = row.get("path"), row.get("sha256")
    if not isinstance(name, str) or not isinstance(digest, str):
        raise ReplayError("AAC request reference requires path and SHA256")
    return Reference(Path(name), digest)


def _evidence(reference: Reference) -> Evidence:
    return Evidence(reference, _read(reference))


def _identity(value: object) -> FileIdentity:
    if not isinstance(value, list) or len(value) != 5:
        raise ReplayError("AAC request requires all five observed identity fields")
    fields = [_integer(item) for item in value]
    if fields[2] == 0:
        raise ReplayError("AAC candidate identity is empty")
    return FileIdentity(*fields)


def _source_request(value: object) -> tuple[SourcePCM | None, Reference | None]:
    if value is None:
        return None, None
    row = record(value)
    clock = row.get("clockEvidence")
    source = SourcePCM(
        _reference(row.get("reference")),
        _integer(row.get("sampleRate"), minimum=1),
        _integer(row.get("sampleCount"), minimum=1),
        Fraction(str(row.get("start"))),
        None if clock is None else _evidence(_reference(clock)),
    )
    producer = _reference(row.get("producer"))
    report = _evidence(producer).document()
    if (
        report.get("output") != str(source.reference.path)
        or report.get("outputSHA256") != source.reference.sha256
        or report.get("sampleRate") != source.sample_rate
        or report.get("outputSamples") != source.samples
        or report.get("outputStartSample") != source.start * source.sample_rate
        or report.get("outputEndSample") != (source.start * source.sample_rate + source.samples)
    ):
        raise ReplayError("Original PCM facts differ from the pinned producer")
    return source, producer


def load_observation_request(reference: Reference) -> ObservationRequest:
    evidence = _evidence(reference)
    if len(evidence.payload) > MAX_REQUEST_BYTES:
        raise ReplayError("AAC request exceeds its byte budget")
    row = evidence.document()
    if (
        type(row.get("version")) is not int
        or row["version"] != 1
        or row.get("kind") != "aac-presentation-observation-request"
    ):
        raise ReplayError("Unsupported AAC observation request")
    processes = record(row.get("processes"))
    if set(processes) != set(PROCESS_ROLES):
        raise ReplayError("AAC request needs all original process observations")
    records = tuple(
        ProcessEvidence(
            role,
            _reference(record(processes[role]).get("process")),
            _reference(record(processes[role]).get("stdout")),
            _reference(record(processes[role]).get("stderr")),
        )
        for role in PROCESS_ROLES
    )
    tools = record(row.get("tools"))
    source, producer = _source_request(row.get("source"))
    content = row.get("contentComparison")
    return ObservationRequest(
        reference,
        _reference(row.get("candidate")),
        _identity(row.get("expectedIdentity")),
        _reference(row.get("decodeReceipt")),
        _reference(row.get("completedCheckpoint")),
        (_reference(tools.get("ffmpeg")), _reference(tools.get("ffprobe"))),
        records,
        source,
        producer,
        None if content is None else _evidence(_reference(content)),
    )


def _commands(request: ObservationRequest) -> dict[str, list[str]]:
    movie = str(request.candidate.path)
    ffmpeg, ffprobe = (str(tool.path) for tool in request.tools)
    prefix = [
        ffprobe,
        "-v",
        "error",
        "-threads",
        "0",
        "-protocol_whitelist",
        "file,pipe",
        "-of",
        "json",
    ]
    return {
        "decode": [
            ffmpeg,
            "-nostdin",
            "-v",
            "error",
            "-xerror",
            "-max_error_rate",
            "0",
            "-protocol_whitelist",
            "file,pipe",
            "-i",
            movie,
            "-map",
            "0:v",
            "-map",
            "0:a",
            "-f",
            "null",
            "-",
        ],
        "streams": prefix + ["-show_streams", "-show_format", movie],
        "frames": prefix
        + [
            "-show_frames",
            "-show_entries",
            "frame=media_type,pts,duration,pkt_duration,nb_samples",
            movie,
        ],
        "packets": prefix
        + [
            "-show_packets",
            "-count_packets",
            "-show_entries",
            "packet=stream_index,pts,dts,duration:stream=index,nb_read_packets",
            movie,
        ],
    }


def _original_provenance(
    request: ObservationRequest,
) -> tuple[dict[str, object], dict[str, Reference]]:
    receipt = _evidence(request.receipt).document()
    checkpoint = _evidence(request.checkpoint).document()
    if (
        type(receipt.get("version")) is not int
        or receipt.get("version") != 1
        or receipt.get("fullDecode") is not True
        or receipt.get("technicalAccepted") is not True
        or receipt.get("inputs") != receipt.get("inputsAfter")
        or checkpoint.get("status") != "complete"
        or record(checkpoint.get("outputs")).get(str(request.receipt.path))
        != request.receipt.sha256
    ):
        raise ReplayError("Original full decoder provenance is incomplete or changed")
    inputs = receipt.get("inputs")
    if not isinstance(inputs, list):
        raise ReplayError("Original decoder input inventory is missing")
    candidates = [
        record(item) for item in inputs if record(item).get("path") == str(request.candidate.path)
    ]
    if len(candidates) != 1:
        raise ReplayError("Original decoder must bind the one actual candidate")
    bound = candidates[0]
    if (
        bound.get("sha256") != request.candidate.sha256
        or _identity(
            [
                bound.get("device"),
                bound.get("inode"),
                bound.get("bytes"),
                bound.get("mtimeNs"),
                bound.get("ctimeNs"),
            ]
        )
        != request.identity
    ):
        raise ReplayError("Original decoder candidate SHA/identity differs from request")
    tool_names = record(receipt.get("tools"))
    for name, tool in zip(("ffmpeg", "ffprobe"), request.tools, strict=True):
        _read(tool, external=True)
        if tool_names.get(name) != str(tool.path) or not os.access(tool.path, os.X_OK):
            raise ReplayError("Original process tool differs from the configured pin")
    commands = _commands(request)
    raw: dict[str, Reference] = {}
    for process in request.processes:
        expected_parent = request.receipt.path.parent
        if any(
            ref.path.parent != expected_parent
            for ref in (process.process, process.stdout, process.stderr)
        ) or (
            process.process.path.name != process.role + ".process.json"
            or process.stdout.path.name != process.role + ".stdout"
            or process.stderr.path.name != process.role + ".stderr"
        ):
            raise ReplayError("Original process observations must share the receipt namespace")
        report = _evidence(process.process).document()
        if (
            type(report.get("returncode")) is not int
            or report.get("returncode") != 0
            or report.get("command") != commands[process.role]
        ):
            raise ReplayError("Original process was unsuccessful or its actual command differs")
        if _read(process.stderr, empty=True):
            raise ReplayError("Original process reported decoder/probe errors")
        if process.role == "decode":
            if _read(process.stdout, empty=True):
                raise ReplayError("Original full decoder stdout must be empty")
        else:
            raw[process.role] = process.stdout
    if _evidence(raw["streams"]).document() != receipt.get("streams"):
        raise ReplayError("Original stream metadata differs from the decoder receipt")
    return receipt, raw


def _normalized(request: ObservationRequest) -> tuple[dict[str, object], dict[str, object]]:
    receipt, raw = _original_provenance(request)
    streams = _evidence(raw["streams"]).document().get("streams")
    if not isinstance(streams, list) or not 1 <= len(streams) <= MAX_TRACKS:
        raise ReplayError("Original observed stream inventory is invalid")
    audios = [record(item) for item in streams if record(item).get("codec_type") == "audio"]
    if len(audios) != 1:
        raise ReplayError("Legacy normalization requires exactly one actual AAC stream")
    audio = audios[0]
    index = _integer(audio.get("index"))
    documents: list[dict[str, object]] = []
    for role, selector, expected in (
        ("frames", "media_type", "audio"),
        ("packets", "stream_index", index),
    ):
        rows = _evidence(raw[role]).document().get(role)
        if not isinstance(rows, list) or not 1 <= len(rows) <= MAX_PACKETS:
            raise ReplayError("Original raw observations are missing or excessive")
        selected = [record(item) for item in rows if record(item).get(selector) == expected]
        if not selected:
            raise ReplayError("Original raw observations contain no audio rows")
        clock = record(receipt.get("audioDecodedClock" if role == "frames" else "audioPacketClock"))
        if clock.get("count") != len(selected):
            raise ReplayError("Original raw audio count differs from the complete receipt")
        documents.append({"streams": [audio], role: selected})
    return documents[0], documents[1]


def _terminal_pins(request: ObservationRequest) -> None:
    """Rehash each retained raw result once, without a second JSON object graph."""
    _original_provenance(request)
    for process in request.processes:
        if process.role in ("frames", "packets"):
            _read(process.stdout)


def _output_paths(output: Path) -> tuple[Path, Path, Path]:
    if not output.is_absolute() or not output.is_relative_to(ROOT / "edits/work"):
        raise ReplayError("AAC outputs must remain under checkout edits/work")
    _canonical(output.parent)
    return (
        output,
        output.with_name(output.name + ".decoded.json"),
        output.with_name(output.name + ".packets.json"),
    )


def _json_bytes(document: dict[str, object]) -> bytes:
    return (json.dumps(document, sort_keys=True, separators=(",", ":")) + "\n").encode()


def _exclusive(path: Path, payload: bytes) -> None:
    """Publish only in the observed real parent; interrupted files remain evidence."""
    before = path.parent.stat()
    with _directory(path.parent) as directory:
        if _file_facts(os.fstat(directory)) != _file_facts(before):
            raise ReplayError("AAC output parent changed")
        with os.fdopen(
            os.open(
                path.name,
                os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW,
                0o600,
                dir_fd=directory,
            ),
            "wb",
        ) as stream:
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
        os.fsync(directory)
        if (
            _file_facts(path.parent.stat()) != _file_facts(os.fstat(directory))
            or path.parent.resolve(strict=True) != path.parent
        ):
            raise ReplayError("AAC output parent changed during publication")


@contextmanager
def _directory(path: Path) -> Iterator[int]:
    descriptor = os.open(path, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW)
    try:
        yield descriptor
    finally:
        os.close(descriptor)


def _derived(path: Path, document: dict[str, object]) -> Evidence:
    payload = _json_bytes(document)
    return Evidence(Reference(path, hashlib.sha256(payload).hexdigest()), payload)


def _derived_pair(
    request: ObservationRequest, paths: tuple[Path, Path, Path]
) -> tuple[Evidence, Evidence]:
    """Release legacy JSON object graphs before pure assessment parses derived bytes."""
    decoded, packets = _normalized(request)
    return _derived(paths[1], decoded), _derived(paths[2], packets)


def _current_candidate(request: ObservationRequest) -> None:
    _canonical(request.candidate.path)
    if FileIdentity.from_stat(request.candidate.path.lstat()) != request.identity:
        raise ReplayError("AAC candidate changed after observation")


def _cli_report(
    request: ObservationRequest, observation: dict[str, object], derived: tuple[Evidence, Evidence]
) -> dict[str, object]:
    timing = record(observation.get("timing"))
    tracks = timing.get("tracks")
    if not isinstance(tracks, list):
        raise ReplayError("Observed timing tracks are missing")
    parsed_tracks = []
    for item in tracks:
        row = record(item)
        raw_edits = row.get("edits")
        roll = row.get("rollMetadataPresent")
        if not isinstance(raw_edits, list) or type(roll) is not bool:
            raise ReplayError("Invalid observed timing track")
        handler = row.get("handler")
        if not isinstance(handler, str):
            raise ReplayError("Observed track handler is missing")
        parsed_tracks.append(
            Track(
                _integer(row.get("trackID"), minimum=1),
                handler,
                _integer(row.get("mediaTimescale"), minimum=1),
                _integer(row.get("mediaDuration"), minimum=1),
                tuple(
                    Edit(
                        _integer(record(edit).get("segmentDuration"), minimum=1),
                        _integer(record(edit).get("mediaTime"), minimum=-1),
                        Fraction(str(record(edit).get("mediaRate"))),
                    )
                    for edit in raw_edits
                ),
                roll,
            )
        )
    payload_sha, facts_sha = timing.get("timingPayloadSHA256"), timing.get("originalFactsSHA256")
    if not isinstance(payload_sha, str) or not isinstance(facts_sha, str):
        raise ReplayError("Observed timing hashes are missing")
    container = Container(
        _integer(timing.get("movieTimescale"), minimum=1),
        _integer(timing.get("movieDuration"), minimum=1),
        tuple(parsed_tracks),
        payload_sha,
        facts_sha,
    )
    result = assess(container, Inputs(request.candidate, *derived, request.source, request.content))
    if result["decodedSamples"] != record(
        _evidence(request.receipt).document().get("audioDecodedClock")
    ).get("sampleCount"):
        raise ReplayError("Normalized decoded samples differ from the complete receipt")
    return {
        "version": 1,
        "kind": "aac-presentation-observation-result",
        "request": request.reference.json(),
        "observation": observation,
        "assessment": result,
        "normalization": {
            "method": (
                "Original media_type audio frames / actual one-AAC stream_index packets; "
                "order and all original values preserved"
            ),
            "streamIndexSynthesized": False,
            "newDecoderRun": False,
            "derived": [item.reference.json() for item in derived],
        },
        "provenance": {
            "decodeReceipt": request.receipt.json(),
            "completedCheckpoint": request.checkpoint.json(),
            "tools": [item.json() for item in request.tools],
            "processes": [
                {
                    "role": item.role,
                    "process": item.process.json(),
                    "stdout": item.stdout.json(),
                    "stderr": item.stderr.json(),
                }
                for item in request.processes
            ],
            "sourceProducer": None
            if request.source_producer is None
            else request.source_producer.json(),
        },
        "freshWholeCandidateSHA256Verified": False,
        "historicalRawStdoutSHA256CapturedByOriginalDecoder": False,
        "newFullDecode": False,
        "qualityApproved": False,
        "syncApproved": False,
        "deliveryApproved": False,
    }


def collect_presentation(reference: Reference, output: Path) -> Path:
    request = load_observation_request(reference)
    paths = _output_paths(output)
    if any(path.exists() or path.is_symlink() for path in paths):
        raise FileExistsError("AAC observation output already exists; use read-only verify")
    derived = _derived_pair(request, paths)
    observation = observe_headers(request.candidate, expected_identity=request.identity)
    report = _cli_report(request, observation, derived)
    _terminal_pins(request)
    if load_observation_request(reference) != request:
        raise ReplayError("AAC request or its dependent evidence changed during collection")
    _current_candidate(request)
    _exclusive(paths[1], derived[0].payload)
    _exclusive(paths[2], derived[1].payload)
    _exclusive(paths[0], _json_bytes(report))
    _current_candidate(request)
    return paths[0]


def verify_presentation(reference: Reference, output: Path) -> None:
    request = load_observation_request(reference)
    paths = _output_paths(output)
    derived = _derived_pair(request, paths)
    _read(derived[0].reference)
    _read(derived[1].reference)
    _current_candidate(request)
    _canonical(paths[0])
    with os.fdopen(os.open(paths[0], os.O_RDONLY | os.O_NOFOLLOW), "rb") as stream:
        before = os.fstat(stream.fileno())
        if not stat.S_ISREG(before.st_mode) or before.st_nlink != 1:
            raise ReplayError("AAC result must be a single-link regular file")
        payload = stream.read(MAX_PROBE_BYTES + 1)
        if _file_facts(os.fstat(stream.fileno())) != _file_facts(before) or _file_facts(
            paths[0].lstat()
        ) != _file_facts(before):
            raise ReplayError("AAC result changed during verification")
    report = Evidence(Reference(paths[0], hashlib.sha256(payload).hexdigest()), payload).document()
    observation = record(report.get("observation"))
    if (
        observation.get("kind") != "aac-container-timing-observation"
        or observation.get("candidate") != request.candidate.json()
        or observation.get("identityBefore") != request.identity.json()
        or observation.get("identityAfter") != request.identity.json()
        or observation.get("expectedWholeSHA256FreshlyVerified") is not False
        or observation.get("fullDecodePerformed") is not False
        or _cli_report(request, observation, derived) != report
        or load_observation_request(reference) != request
    ):
        raise ReplayError("AAC observation result changed or its domain differs")
    _read(Reference(paths[0], hashlib.sha256(payload).hexdigest()))
    _terminal_pins(request)
    _current_candidate(request)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("observe", "verify"))
    parser.add_argument("request", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--sha256", required=True)
    args = parser.parse_args(argv)
    reference = Reference(args.request, args.sha256)
    try:
        if args.action == "observe":
            collect_presentation(reference, args.output)
        else:
            verify_presentation(reference, args.output)
    except (ReplayError, OSError, ValueError) as error:
        parser.exit(1, f"AAC presentation observation: {error}\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
