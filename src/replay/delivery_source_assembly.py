"""Lossless, exact-clock source assembly; never a delivery/quality approval.

Manifest v1: pinned ``index`` and ordered ``chunks`` containing pinned ``candidate``
(receipt) and ``producer`` (subtitle_candidate config). References are {path, sha256}.
Paths are absolute; run from the checkout root. Historical producer inputs are
recorded, not re-executed or treated as independent original-media verification.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import subprocess
import sys
import time
from dataclasses import dataclass
from fractions import Fraction
from pathlib import Path

from replay.core import ReplayError
from replay.input_hash_cache import cached_digest
from replay.recordings import record
from replay.subtitle_clock import VideoFrameClock

MAX_JSON = 256 * 1024 * 1024
MAX_MEDIA = 256 * 1024**3
MAX_FRAMES = 1_000_000
MAX_CHUNKS = 512
MAX_SOURCE_REFERENCE_LENGTH = 4096
PROCESS_SECONDS = 1800
JOB_SECONDS = 6 * 3600
STREAM_FIELDS = (
    "index,codec_type,codec_name,codec_tag_string,profile,level,width,height,pix_fmt,"
    "sample_aspect_ratio,field_order,color_range,color_space,color_transfer,color_primaries,"
    "time_base,extradata_hash"
)


@dataclass(frozen=True)
class Chunk:
    media: Path
    clock: VideoFrameClock
    source: tuple[str, str]


@dataclass(frozen=True)
class Probe:
    stream: dict[str, object]
    frames: tuple[tuple[int, int], ...]
    packets: tuple[tuple[int, int, str], ...]


def _path(value: object, *, local: bool = False) -> Path:
    if not isinstance(value, str) or not value or any(c in value for c in "\n\r\x00"):
        raise ReplayError("Invalid absolute path")
    path = Path(value)
    if not path.is_absolute() or path != path.resolve():
        raise ReplayError("Paths must be absolute and unlinked")
    if local and not path.is_relative_to(Path.cwd().resolve()):
        raise ReplayError("Evidence must remain checkout-local")
    return path


def _historical_source(value: object) -> str:
    """Preserve the receipt key verbatim; this identity never authorizes filesystem I/O."""
    if (
        not isinstance(value, str)
        or not 1 <= len(value) <= MAX_SOURCE_REFERENCE_LENGTH
        or not value.startswith("/")
        or any(c in value for c in "\n\r\x00")
        or any(part in {".", ".."} for part in value.split("/"))
    ):
        raise ReplayError("Invalid absolute historical source reference")
    return value


def candidate_media_path(producer: dict[str, object]) -> Path:
    """Resolve an explicit media destination; metadata and scratch remain local.

    The configured root describes the caller's authorized media scope, not a new
    permission grant. Both fields are required together, and every ancestor must
    remain canonical. Old producers retain their original adjacent media path.
    """
    output = _path(producer.get("outputDirectory"), local=True)
    explicit = "mediaOutput" in producer
    if explicit != ("allowedMediaRoot" in producer):
        raise ReplayError("Explicit media output requires its configured media root")
    if not explicit:
        return output / "candidate.mp4"
    root = _path(producer.get("allowedMediaRoot"))
    media = _path(producer.get("mediaOutput"))
    if (
        not root.is_dir()
        or not media.parent.is_dir()
        or media == root
        or not media.is_relative_to(root)
        or media.suffix != ".mp4"
    ):
        raise ReplayError("Media output must be an MP4 beneath its canonical media root")
    return media


def _sha(value: object) -> str:
    if not isinstance(value, str) or re.fullmatch("[0-9a-f]{64}", value) is None:
        raise ReplayError("Invalid SHA256 pin")
    return value


def _digest(path: Path) -> str:
    if not path.is_file() or not 0 < path.stat().st_size <= MAX_MEDIA:
        raise ReplayError("Missing or oversized input")
    return cached_digest(path, _uncached_digest)


def _uncached_digest(path: Path) -> str:
    with path.open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


def _bind(path: Path, sha: object, bindings: dict[str, str]) -> None:
    expected = _sha(sha)
    if _digest(path) != expected or bindings.get(str(path), expected) != expected:
        raise ReplayError(f"Changed input pin: {path}")
    bindings[str(path)] = expected


def _json(path: Path) -> dict[str, object]:
    if not 0 < path.stat().st_size <= MAX_JSON:
        raise ReplayError("JSON size limit exceeded")
    return record(json.loads(path.read_text(encoding="utf-8")))


def _reference(value: object, bindings: dict[str, str]) -> Path:
    raw = record(value)
    path = _path(raw.get("path"), local=True)
    _bind(path, raw.get("sha256"), bindings)
    return path


def _integer(value: object) -> int:
    if type(value) is not int:
        raise ReplayError("Expected integer clock")
    return value


def _array(value: object, maximum: int) -> list[object]:
    if not isinstance(value, list) or not 1 <= len(value) <= maximum:
        raise ReplayError("Missing or excessive array")
    return list(value)


def _chunk(row: object, reference: object, bindings: dict[str, str]) -> Chunk:
    index, refs = record(row), record(reference)
    receipt_path = _reference(refs.get("candidate"), bindings)
    producer_path = _reference(refs.get("producer"), bindings)
    receipt, producer = _json(receipt_path), _json(producer_path)
    media = _path(index.get("media"))
    if media != candidate_media_path(producer):
        raise ReplayError("Producer output does not match indexed media")
    if "mediaOutput" in producer and receipt.get("mediaOutput") != str(media):
        raise ReplayError("Receipt does not bind the explicit media destination")
    _bind(media, receipt.get("videoSHA256"), bindings)
    inputs = record(receipt.get("inputs"))
    source = (_historical_source(producer.get("source")), _sha(producer.get("sourceSHA256")))
    if (
        inputs.get(str(producer_path)) != bindings[str(producer_path)]
        or inputs.get(source[0]) != source[1]
    ):
        raise ReplayError("Receipt does not bind producer and original source")
    raw = record(producer.get("clock"))
    base = raw.get("timeBase")
    if not isinstance(base, str):
        raise ReplayError("Missing rational time base")
    clock = VideoFrameClock(
        tuple(_integer(p) for p in _array(raw.get("pts"), 50_000)),
        Fraction(base),
        _integer(raw.get("lastDuration")),
    )
    if (
        _integer(index.get("sourceFirstPTS")) != clock.pts[0]
        or _integer(index.get("sourceEndPTS")) != clock.pts[-1] + clock.last_duration
        or _integer(index.get("frames")) != len(clock.pts)
        or index.get("timeBase") != base
    ):
        raise ReplayError("Indexed clock differs from producer clock")
    return Chunk(media, clock, source)


def load_chunks(manifest: Path, *, sha256: str, bindings: dict[str, str]) -> tuple[Chunk, ...]:
    """Load pinned historical producer correspondence, without guessing timestamps."""
    _bind(_path(str(manifest), local=True), sha256, bindings)
    raw = _json(manifest)
    if _integer(raw.get("version")) != 1:
        raise ReplayError("Unsupported assembly manifest")
    index = _json(_reference(raw.get("index"), bindings))
    rows = _array(index.get("ranges"), MAX_CHUNKS)
    refs = _array(raw.get("chunks"), MAX_CHUNKS)
    if len(rows) != len(refs):
        raise ReplayError("Every chunk needs a pinned producer and receipt")
    chunks = tuple(_chunk(row, ref, bindings) for row, ref in zip(rows, refs, strict=True))
    if (
        sum(len(c.clock.pts) for c in chunks) > MAX_FRAMES
        or sum(c.media.stat().st_size for c in chunks) > MAX_MEDIA
    ):
        raise ReplayError("Total frame or media budget exceeded")
    for previous, current in zip(chunks, chunks[1:], strict=False):
        if (
            current.source != previous.source
            or current.clock.time_base != previous.clock.time_base
            or current.clock.pts[0] != previous.clock.pts[-1] + previous.clock.last_duration
        ):
            raise ReplayError("Noncontiguous or incompatible original source clocks")
    return chunks


def _write(path: Path, value: object) -> None:
    with path.open("x", encoding="utf-8") as handle:
        os.fchmod(handle.fileno(), 0o600)
        json.dump(value, handle, indent=2)
        handle.write("\n")


def _run(
    command: list[str],
    stdout: Path,
    *,
    media: Path | None = None,
    budget: int = MAX_MEDIA,
    deadline: float | None = None,
) -> None:
    """Bound child time, diagnostics and media; preserve all failed output."""
    _write(stdout.with_suffix(".command.json"), command)
    env = {**os.environ, **dict.fromkeys(("TMPDIR", "TMP", "TEMP"), str(stdout.parent))}
    error = stdout.with_suffix(".stderr")
    with stdout.open("xb") as out, error.open("xb") as err:
        os.fchmod(out.fileno(), 0o600)
        os.fchmod(err.fileno(), 0o600)
        with subprocess.Popen(
            command, stdin=subprocess.DEVNULL, stdout=out, stderr=err, env=env
        ) as child:
            process_deadline = time.monotonic() + PROCESS_SECONDS
            if deadline is not None:
                process_deadline = min(process_deadline, deadline)
            try:
                while True:
                    status = child.poll()
                    if (
                        time.monotonic() > process_deadline
                        or stdout.stat().st_size > MAX_JSON
                        or error.stat().st_size > 1024 * 1024
                        or (media is not None and media.exists() and media.stat().st_size > budget)
                    ):
                        raise ReplayError("Assembly subprocess exceeded its resource budget")
                    if status is not None:
                        break
                    time.sleep(0.05)
            finally:
                if child.poll() is None:
                    child.kill()
                    child.wait()
            if status != 0 or error.stat().st_size:
                raise ReplayError(f"Assembly subprocess failed; retained diagnostics: {error}")


def probe(
    media: Path, *, directory: Path, name: str, ffprobe: Path, deadline: float | None = None
) -> Probe:
    """Decode all presentation clocks and independently hash compressed samples."""
    common = [str(ffprobe), "-v", "error", "-protocol_whitelist", "file", "-enable_drefs", "0"]
    metadata = directory / f"{name}-streams.json"
    _run(
        [
            *common,
            "-show_entries",
            f"stream={STREAM_FIELDS}",
            "-show_data_hash",
            "sha256",
            "-of",
            "json",
            str(media),
        ],
        metadata,
        deadline=deadline,
    )
    streams = _array(_json(metadata).get("streams"), 1)
    stream = record(streams[0])
    if stream.get("codec_type") != "video" or stream.get("codec_name") not in {"h264", "hevc"}:
        raise ReplayError("Expected a single self-contained H264/HEVC video stream, no audio")
    if stream.get("index") != 0 or not isinstance(stream.get("extradata_hash"), str):
        raise ReplayError("Missing codec configuration or invalid stream index")
    frame_path = directory / f"{name}-frames.json"
    _run(
        [*common, "-show_entries", "frame=pts,duration", "-of", "json", str(media)],
        frame_path,
        deadline=deadline,
    )
    frames = tuple(
        (_integer(f.get("pts")), _integer(f.get("duration")))
        for f in map(record, _array(_json(frame_path).get("frames"), MAX_FRAMES))
    )
    packet_path = directory / f"{name}-packets.json"
    _run(
        [
            *common,
            "-show_entries",
            "packet=pts,duration,data_hash",
            "-show_data_hash",
            "sha256",
            "-of",
            "json",
            str(media),
        ],
        packet_path,
        deadline=deadline,
    )
    packets: list[tuple[int, int, str]] = []
    for packet in _array(_json(packet_path).get("packets"), MAX_FRAMES):
        item = record(packet)
        digest = item.get("data_hash")
        if not isinstance(digest, str) or not digest.startswith("SHA256:"):
            raise ReplayError("Missing compressed packet hash")
        packets.append(
            (_integer(item.get("pts")), _integer(item.get("duration")), _sha(digest[7:]))
        )
    return Probe(stream, frames, tuple(packets))


def duration_text(clock: VideoFrameClock) -> str:
    """Only exactly serializable native extents; no microsecond rounding."""
    microseconds = (
        (clock.pts[-1] + clock.last_duration - clock.pts[0]) * clock.time_base * 1_000_000
    )
    if microseconds.denominator != 1:
        raise ReplayError("Native duration has no exact ffconcat serialization")
    value = microseconds.numerator
    return f"{value // 1_000_000}.{value % 1_000_000:06d}"


def verify_chunk(chunk: Chunk, measured: Probe) -> None:
    clock = chunk.clock
    if measured.stream.get("time_base") != str(clock.time_base):
        raise ReplayError("Candidate time base differs from producer")
    following = (*clock.pts[1:], clock.pts[-1] + clock.last_duration)
    expected = tuple(
        (p - clock.pts[0], end - p) for p, end in zip(clock.pts, following, strict=True)
    )
    if measured.frames != expected:
        raise ReplayError("Candidate decoded PTS/durations differ from rebased producer")
    if tuple(sorted((p, d) for p, d, _ in measured.packets)) != expected:
        raise ReplayError("Candidate packet presentation support differs from producer")


def assemble(
    manifest: Path, *, sha256: str, output: Path, evidence: Path, ffmpeg: Path, ffprobe: Path
) -> Path:
    """Publish one new asset only after complete exact clock/payload comparison."""
    deadline = time.monotonic() + JOB_SECONDS
    output = _path(str(output))
    evidence = _path(str(evidence), local=True)
    if output.suffix.lower() != ".mov" or output.exists() or not output.parent.is_dir():
        raise ReplayError("Require a new .mov output in an existing authorized directory")
    bindings: dict[str, str] = {}
    chunks = load_chunks(manifest, sha256=sha256, bindings=bindings)
    for tool in (ffmpeg, ffprobe, Path(__file__).resolve()):
        _bind(_path(str(tool)), _digest(tool), bindings)
    # Only nonterminal extents determine the next file's timestamp offset.
    durations = tuple(duration_text(c.clock) for c in chunks[:-1])
    evidence.mkdir(mode=0o700)
    pending_directory = output.with_name(output.name + ".assembly")
    pending_directory.mkdir(mode=0o700)
    pending = pending_directory / "source.mov"
    _write(evidence / "inputs.json", bindings)
    observations = tuple(
        probe(
            c.media, directory=evidence, name=f"input-{i:04d}", ffprobe=ffprobe, deadline=deadline
        )
        for i, c in enumerate(chunks)
    )
    for chunk, measured in zip(chunks, observations, strict=True):
        verify_chunk(chunk, measured)
        if measured.stream != observations[0].stream:
            raise ReplayError("Source codec/canvas/configuration mismatch")
    concat = evidence / "sources.ffconcat"
    with concat.open("x", encoding="utf-8") as handle:
        os.fchmod(handle.fileno(), 0o600)
        handle.write("ffconcat version 1.0\n")
        for number, chunk in enumerate(chunks):
            quoted = str(chunk.media).replace("'", "'\\''")
            handle.write(f"file '{quoted}'\n")
            if number < len(durations):
                handle.write(f"duration {durations[number]}\n")
    size_budget = min(MAX_MEDIA, sum(c.media.stat().st_size for c in chunks) + 64 * 1024 * 1024)
    _run(
        [
            str(ffmpeg),
            "-nostdin",
            "-hide_banner",
            "-v",
            "error",
            "-n",
            "-protocol_whitelist",
            "file",
            "-f",
            "concat",
            "-safe",
            "0",
            "-auto_convert",
            "0",
            "-i",
            str(concat),
            "-map",
            "0:v:0",
            "-c:v",
            "copy",
            "-copyts",
            "-avoid_negative_ts",
            "disabled",
            "-video_track_timescale",
            str(chunks[0].clock.time_base.denominator),
            "-write_btrt",
            "0",
            "-movflags",
            "+faststart",
            str(pending),
        ],
        evidence / "concat.log",
        media=pending,
        budget=size_budget,
        deadline=deadline,
    )
    os.chmod(pending, 0o600)
    pending_hash = _digest(pending)
    joined = probe(pending, directory=evidence, name="output", ffprobe=ffprobe, deadline=deadline)
    first = chunks[0].clock.pts[0]
    expected_frames = tuple(
        (p + c.clock.pts[0] - first, d)
        for c, obs in zip(chunks, observations, strict=True)
        for p, d in obs.frames
    )
    expected_packets = tuple(
        (p + c.clock.pts[0] - first, d, h)
        for c, obs in zip(chunks, observations, strict=True)
        for p, d, h in obs.packets
    )
    if (
        joined.stream != observations[0].stream
        or joined.frames != expected_frames
        or joined.packets != expected_packets
    ):
        raise ReplayError("Joined stream, complete VFR clock or compressed payload mismatch")
    for path, pin in tuple(bindings.items()):
        _bind(_path(path), pin, bindings)
    output_hash = _digest(pending)
    if output_hash != pending_hash:
        raise ReplayError("Joined media changed during verification")
    probe_files = {
        str(evidence / f"{name}-{kind}.json"): _digest(evidence / f"{name}-{kind}.json")
        for name in (*[f"input-{i:04d}" for i in range(len(chunks))], "output")
        for kind in ("streams", "frames", "packets")
    }
    index = {
        "purpose": "lossless source assembly; not quality approval",
        "ranges": [
            {
                "media": str(output),
                "sha256": output_hash,
                "sourceFirstPTS": first,
                "sourceEndPTS": chunks[-1].clock.pts[-1] + chunks[-1].clock.last_duration,
                "frames": len(expected_frames),
                "timeBase": str(chunks[0].clock.time_base),
            }
        ],
    }
    _write(evidence / "single-source-index.json", index)
    os.link(pending, output)
    _write(
        evidence / "receipt.json",
        {
            "version": 1,
            "inputs": bindings,
            "output": {"path": str(output), "sha256": output_hash},
            "indexSHA256": _digest(evidence / "single-source-index.json"),
            "probeEvidence": probe_files,
            "frames": len(expected_frames),
            "exactPresentationClockVerified": True,
            "compressedPacketsIdentical": True,
            "originalSourceIndependentlyDecoded": False,
            "qualityApproved": False,
            "avSyncApproved": False,
        },
    )
    return evidence / "receipt.json"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("manifest", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("evidence", type=Path)
    parser.add_argument("--sha256", required=True)
    parser.add_argument("--ffmpeg", type=Path, required=True)
    parser.add_argument("--ffprobe", type=Path, required=True)
    args = parser.parse_args(argv)
    try:
        assemble(
            args.manifest,
            sha256=args.sha256,
            output=args.output,
            evidence=args.evidence,
            ffmpeg=args.ffmpeg,
            ffprobe=args.ffprobe,
        )
    except (ReplayError, OSError, ValueError) as error:
        print(f"SOURCE_ASSEMBLY_ERROR: {error}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
