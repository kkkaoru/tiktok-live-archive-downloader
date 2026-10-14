"""Inspect an SDR upload candidate without encoding or changing media clocks.

Checks a selected H.264/AAC MP4 delivery contract, not every YouTube recommendation.
Bitrates are observations; preset intent, full decode, timing and quality are separate.
Edit lists are reported as failures, never removed (AAC priming can depend on them).
"""

from __future__ import annotations

import argparse
import json
import os
import struct
import subprocess
import sys
from dataclasses import dataclass
from fractions import Fraction
from pathlib import Path
from typing import BinaryIO

from replay.core import ReplayError
from replay.recordings import record

CONTAINERS = frozenset({b"moov", b"trak", b"edts", b"mdia", b"minf", b"stbl", b"dinf"})
MP4_BRANDS = frozenset({b"isom", b"iso2", b"mp41", b"mp42", b"avc1", b"M4V "})
MAX_BOXES = 100_000
MAX_DEPTH = 12
MAX_PROBE_BYTES = 2_000_000
REPOSITORY = Path(__file__).resolve().parents[2]


@dataclass(frozen=True)
class Box:
    kind: bytes
    offset: int
    payload: int
    end: int
    depth: int


def _boxes(stream: BinaryIO, *, start: int, end: int, depth: int) -> list[Box]:
    if depth > MAX_DEPTH:
        raise ReplayError("MP4 box nesting exceeds inspection bound")
    result: list[Box] = []
    position = start
    while position < end:
        stream.seek(position)
        header = stream.read(8)
        if len(header) != 8 or end - position < 8:
            raise ReplayError("Truncated MP4 box header")
        size, kind = struct.unpack(">I4s", header)
        header_size = 8
        if size == 1:
            extended = stream.read(8)
            if len(extended) != 8:
                raise ReplayError("Truncated extended MP4 box size")
            size = int.from_bytes(extended, "big")
            header_size = 16
        if size == 0:
            size = end - position
        if size < header_size or size > end - position:
            raise ReplayError("MP4 box size escapes its parent or is invalid")
        box = Box(kind, position, position + header_size, position + size, depth)
        result.append(box)
        if kind in CONTAINERS:
            result.extend(_boxes(stream, start=box.payload, end=box.end, depth=depth + 1))
        if len(result) > MAX_BOXES:
            raise ReplayError("MP4 box count exceeds inspection bound")
        position = box.end
    return result


def inspect_container(movie: Path) -> dict[str, object]:
    """Seek across media payloads; parse bounded ISO BMFF headers, not compressed bytes."""
    with movie.open("rb") as stream:
        boxes = _boxes(stream, start=0, end=os.fstat(stream.fileno()).st_size, depth=0)
        top = [box for box in boxes if box.depth == 0]
        types = [box.kind for box in top]
        if types.count(b"ftyp") != 1 or types.count(b"moov") != 1 or b"mdat" not in types:
            raise ReplayError("Expected one MP4 ftyp/moov and at least one mdat")
        brand_box = next(box for box in top if box.kind == b"ftyp")
        length = brand_box.end - brand_box.payload
        if length < 8 or length > 4096 or length % 4:
            raise ReplayError("Invalid MP4 brand box")
        stream.seek(brand_box.payload)
        brands = stream.read(length)
        compatible = {brands[index : index + 4] for index in range(8, length, 4)}
        if brands[:4] == b"qt  " or not ({brands[:4]} | compatible) & MP4_BRANDS:
            raise ReplayError("Container brands do not identify a supported MP4")
    failures = []
    if types.index(b"moov") > types.index(b"mdat"):
        failures.append("moov follows media data: Fast Start is absent")
    if any(box.kind == b"elst" for box in boxes):
        failures.append("Edit list present: preserve priming/sync and return to producer")
    if b"moof" in types:
        failures.append("Fragmented MP4 is outside this delivery contract")
    return {
        "topLevelBoxes": [kind.decode("ascii", errors="replace") for kind in types],
        "boxCount": len(boxes),
        "editListCount": sum(box.kind == b"elst" for box in boxes),
        "fastStart": types.index(b"moov") < types.index(b"mdat"),
        "failures": failures,
    }


def _streams(probe: object, kind: str) -> list[dict[str, object]]:
    values = record(probe).get("streams")
    if not isinstance(values, list) or len(values) > 32:
        raise ReplayError("Expected bounded ffprobe stream records")
    rows = [record(value) for value in values]
    return [row for row in rows if row.get("codec_type") == kind]


def _rate(stream: dict[str, object]) -> Fraction:
    value = stream.get("avg_frame_rate")
    if not isinstance(value, str):
        raise ReplayError("Missing measured average frame rate")
    try:
        rate = Fraction(value)
    except (ValueError, ZeroDivisionError) as error:
        raise ReplayError("Invalid measured average frame rate") from error
    if rate <= 0:
        raise ReplayError("Measured average frame rate must be positive")
    return rate


def check_streams(candidate: object, master: object) -> dict[str, object]:
    """Check measured stream properties; do not treat exact VBR average as a gate."""
    videos, audios = _streams(candidate, "video"), _streams(candidate, "audio")
    masters = _streams(master, "video")
    if len(videos) != 1 or len(audios) != 1 or len(masters) != 1:
        raise ReplayError("Expected exactly one candidate video/audio and one master video")
    video, audio, reference = videos[0], audios[0], masters[0]
    for key in ("width", "height"):
        dimension = reference.get(key)
        if type(dimension) is not int or dimension <= 0:
            raise ReplayError("Master dimensions must be positive integers")
    expected_video = {
        "codec_name": "h264",
        "profile": "High",
        "pix_fmt": "yuv420p",
        "field_order": "progressive",
        "color_primaries": "bt709",
        "color_transfer": "bt709",
        "color_space": "bt709",
        "width": reference["width"],
        "height": reference["height"],
    }
    expected_audio = {
        "codec_name": "aac",
        "profile": "LC",
        "sample_rate": "48000",
        "channels": 2,
        "channel_layout": "stereo",
    }
    failures = [
        f"video.{key}: expected {value}"
        for key, value in expected_video.items()
        if video.get(key) != value
    ]
    failures.extend(
        f"audio.{key}: expected {value}"
        for key, value in expected_audio.items()
        if audio.get(key) != value
    )
    candidate_rate, master_rate = _rate(video), _rate(reference)
    if candidate_rate != master_rate:
        failures.append("Average video frame rate differs from the actual master")
    return {
        "failures": failures,
        "video": video,
        "audio": audio,
        "masterWidth": reference["width"],
        "masterHeight": reference["height"],
        "masterFrameRate": str(master_rate),
        "candidateFrameRate": str(candidate_rate),
        "bitratePolicy": (
            "Observed bit_rate only; 8 Mbps video / 384 kbps stereo are not exact VBR gates"
        ),
    }


def _identity(path: Path) -> tuple[int, int, int, int, int]:
    status = path.stat()
    return status.st_dev, status.st_ino, status.st_size, status.st_mtime_ns, status.st_ctime_ns


def _probe(movie: Path, *, executable: Path, output: Path) -> object:
    with output.open("xb") as stdout, output.with_suffix(".stderr").open("xb") as stderr:
        subprocess.run(
            [
                str(executable),
                "-v",
                "error",
                "-protocol_whitelist",
                "file,pipe",
                "-show_streams",
                "-show_format",
                "-of",
                "json",
                str(movie),
            ],
            stdin=subprocess.DEVNULL,
            stdout=stdout,
            stderr=stderr,
            check=True,
            timeout=120,
        )
    if output.stat().st_size > MAX_PROBE_BYTES:
        raise ReplayError("ffprobe metadata exceeds inspection bound")
    result: object = json.loads(output.read_text(encoding="utf-8"))
    return result


def validate(movie: Path, *, master: Path, output: Path, ffprobe: Path) -> bool:
    """Write fresh evidence, preserving failures; output must be a new local directory."""
    if not output.is_absolute() or not output.resolve().is_relative_to(REPOSITORY / "edits/work"):
        raise ReplayError("Validation evidence must be under repository edits/work")
    if output.resolve() != output:
        raise ReplayError("Validation evidence path cannot contain symlinks")
    if any(
        not path.is_absolute() or path.resolve() != path or not path.is_file()
        for path in (movie, master, ffprobe)
    ):
        raise ReplayError("Inputs and ffprobe must be canonical absolute files")
    before = (_identity(movie), _identity(master))
    output.mkdir(mode=0o700)
    container = inspect_container(movie)
    candidate = _probe(movie, executable=ffprobe, output=output / "candidate.json")
    reference = _probe(master, executable=ffprobe, output=output / "master.json")
    streams = check_streams(candidate, reference)
    if before != (_identity(movie), _identity(master)):
        raise ReplayError("Media changed during validation")
    accepted = container["failures"] == [] and streams["failures"] == []
    result = {
        "version": 1,
        "technicalPropertiesAccepted": accepted,
        "movie": str(movie),
        "master": str(master),
        "inputStatIdentity": before,
        "container": container,
        "streams": streams,
        "deliveryApproved": False,
        "notVerified": [
            "full AV decode",
            "frame/sample clocks and AAC priming sync",
            "preset VBR intent, CABAC, GOP and consecutive B-frame settings",
            "whole-media hash provenance",
            "caption/audio quality",
            "Viewer and summary",
        ],
    }
    (output / "result.json").write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    return accepted


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("movie", type=Path)
    parser.add_argument("master", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--ffprobe", type=Path, required=True)
    args = parser.parse_args()
    try:
        return (
            0
            if validate(args.movie, master=args.master, output=args.output, ffprobe=args.ffprobe)
            else 1
        )
    except (ReplayError, OSError, ValueError, subprocess.SubprocessError) as error:
        print(str(error), file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
