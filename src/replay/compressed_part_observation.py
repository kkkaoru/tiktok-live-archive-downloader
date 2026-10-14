"""Capture explicit video conversion evidence and independent compact observations.

Legacy observation remains read-only. The explicit convert action invokes the
bounded technical runner once; fresh input and output identities bind its actual
process record. Neither path fabricates a native job, approves delivery, publishes
media, retires sources or establishes synchronization.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import re
import time
from dataclasses import dataclass
from fractions import Fraction
from pathlib import Path

from replay import compressed_part_delivery as delivery
from replay import delivery_verification as technical
from replay import whole_technical_batch as metadata
from replay.caption_measured_job import _json as strict_json
from replay.core import ReplayError
from replay.foreground_evidence import _bound_bytes
from replay.input_hash_cache import InputHashCache, input_hash_context
from replay.recordings import record
from replay.stages import file_digest

MAX_RAW_BYTES = 256 * 1024 * 1024
MAX_FRAMES = 4_000_000
MAX_THREADS = 256
TIMEOUT_SECONDS = 7200
WINDOW_NAMES = ("beginning", "middle", "end")
CODE_NAMES = (
    "compressed_part_observation.py",
    "compressed_part_delivery.py",
    "delivery_verification.py",
    "whole_technical_batch.py",
    "input_hash_cache.py",
    "stages.py",
    "caption_measured_job.py",
    "foreground_evidence.py",
    "recordings.py",
)


@dataclass(frozen=True)
class ConversionRequest:
    reference: metadata.Reference
    source: metadata.Reference
    native: metadata.Reference
    tool: metadata.Reference
    output: Path
    output_root: Path
    frames: int
    rate: int
    width: int
    height: int
    bitrate: int
    pixel_format: str
    timeout: int


def _list(value: object) -> list[object]:
    if not isinstance(value, list) or not 0 < len(value) <= 1024:
        raise ReplayError("Converter evidence requires a bounded nonempty list")
    return value


def load_conversion(reference: metadata.Reference) -> ConversionRequest:
    """Load an explicit conversion intent; no existing output is adopted here."""
    metadata._path(str(reference.path), work=True)
    raw = _read(reference)
    if (
        set(raw)
        != {
            "version",
            "kind",
            "source",
            "sourceNative",
            "tool",
            "output",
            "outputRoot",
            "frames",
            "rate",
            "width",
            "height",
            "bitrate",
            "pixelFormat",
            "timeoutSeconds",
        }
        or type(raw["version"]) is not int
        or raw["version"] != 1
        or raw["kind"] != "compressed-part-conversion-request"
    ):
        raise ReplayError("Converter request schema differs")
    output, root = metadata._path(raw["output"]), metadata._path(raw["outputRoot"])
    source, native, tool = (
        metadata._reference(raw["source"]),
        metadata._reference(raw["sourceNative"]),
        _tool(raw["tool"]),
    )
    if (
        not output.is_relative_to(root)
        or output == root
        or not root.is_dir()
        or not output.parent.is_dir()
        or output.suffix != ".mp4"
    ):
        raise ReplayError("Converter output requires its explicit existing output root")
    if output in {source.path, native.path, tool.path, reference.path}:
        raise ReplayError("Converter output must be distinct from every input")
    pixel = raw["pixelFormat"]
    if pixel not in ("yuv420p", "p010le") or not isinstance(pixel, str):
        raise ReplayError("Converter pixel format is unsupported")
    return ConversionRequest(
        reference,
        source,
        native,
        tool,
        output,
        root,
        _count(raw["frames"]),
        _count(raw["rate"], maximum=240),
        _count(raw["width"], maximum=16384),
        _count(raw["height"], maximum=16384),
        _count(raw["bitrate"], maximum=1_000_000_000),
        pixel,
        _count(raw["timeoutSeconds"], maximum=TIMEOUT_SECONDS),
    )


def conversion_command(request: ConversionRequest) -> list[str]:
    """One nonoverwriting, video-only CFR VideoToolbox conversion."""
    return [
        str(request.tool.path),
        "-nostdin",
        "-v",
        "error",
        "-xerror",
        "-n",
        "-protocol_whitelist",
        "file",
        "-f",
        "mov",
        "-enable_drefs",
        "0",
        "-use_absolute_path",
        "0",
        "-i",
        str(request.source.path),
        "-map",
        "0:v:0",
        "-an",
        "-sn",
        "-dn",
        "-c:v",
        "hevc_videotoolbox",
        "-b:v",
        str(request.bitrate),
        "-r",
        str(request.rate),
        "-fps_mode",
        "cfr",
        "-tag:v",
        "hvc1",
        "-pix_fmt",
        request.pixel_format,
        "-frames:v",
        str(request.frames),
        "-f",
        "mp4",
        str(request.output),
    ]


def _conversion_inputs(request: ConversionRequest) -> tuple[metadata.Reference, ...]:
    code = set(metadata.reader_code_paths())
    code.update(Path(__file__).parent / name for name in CODE_NAMES)
    code.update(delivery.code_inputs())
    refs = (request.reference, request.source, request.native, request.tool)
    by_path = {ref.path: ref for ref in refs}
    if len(by_path) != len(refs):
        raise ReplayError("Converter input roles must be distinct")
    for path in sorted(code):
        by_path[path] = metadata.Reference(path, file_digest(path))
    if request.output in by_path:
        raise ReplayError("Converter output overlaps the reader input closure")
    return tuple(by_path[path] for path in sorted(by_path))


def _conversion_snapshot(inputs: tuple[metadata.Reference, ...]) -> list[dict[str, object]]:
    snapshots = [technical._fingerprint(ref.path) for ref in inputs]
    if any(row["sha256"] != ref.sha256 for ref, row in zip(inputs, snapshots, strict=True)):
        raise ReplayError("Converter source/tool/proof/code pin changed")
    return snapshots


def _source_native(source: metadata.Reference, native: metadata.Reference) -> None:
    raw = _read(native)
    if (
        raw.get("movie") != source.json()
        or raw.get("completionVerified") is not True
        or raw.get("shareObservedIdle") is not True
        or type(raw.get("fullAVDecodeExitCode")) is not int
        or raw["fullAVDecodeExitCode"] != 0
    ):
        raise ReplayError("Converter requires the original bound native full AV proof")


def _conversion_result(request: ConversionRequest, directory: Path) -> dict[str, object]:
    initial, terminal = (
        _artifact(directory / "initial.json"),
        _artifact(directory / "terminal.json"),
    )
    before, after = _read(initial), _read(terminal)
    inputs = _conversion_inputs(request)
    current = _conversion_snapshot(inputs)
    if (
        set(before) != {"request", "command", "startedAtNs", "inputs", "outputAbsent"}
        or before["request"] != request.reference.json()
        or before["command"] != conversion_command(request)
        or before["inputs"] != current
        or before["outputAbsent"] is not True
    ):
        raise ReplayError("Converter initial identity or input closure differs")
    if set(after) != {"endedAtNs", "inputs", "output"} or after["inputs"] != current:
        raise ReplayError("Converter terminal identity or input closure differs")
    start, end = before["startedAtNs"], after["endedAtNs"]
    if (
        type(start) is not int
        or type(end) is not int
        or not 0 < start <= end
        or end - start > request.timeout * 1_000_000_000
    ):
        raise ReplayError("Converter execution timestamps are invalid or stale")
    output = technical._fingerprint(request.output)
    if output["bytes"] == 0 or output != after["output"]:
        raise ReplayError("Converter output changed after completion")
    processes = _process(directory, "convert", conversion_command(request))
    return {
        "version": 1,
        "kind": "ffmpeg-video-conversion-execution",
        "request": request.reference.json(),
        "source": request.source.json(),
        "sourceNative": request.native.json(),
        "movie": {"path": str(request.output), "sha256": output["sha256"]},
        "tool": request.tool.json(),
        "command": conversion_command(request),
        "returncode": 0,
        "startedAtNs": start,
        "endedAtNs": end,
        "inputs": [ref.json() for ref in inputs],
        "artifacts": [ref.json() for ref in (initial, terminal, *processes)],
        "sourceBeforeAfterIdentityEqual": True,
        "currentDeclaredInputsVerified": True,
        "videoOnly": True,
        "rate": request.rate,
        "frames": request.frames,
        "width": request.width,
        "height": request.height,
        "bitrate": request.bitrate,
        "pixelFormat": request.pixel_format,
        "processIdentityRecorded": False,
        "historicalNativeJobVerified": False,
        "candidateOnly": True,
        "qualityApproved": False,
        "syncApproved": False,
        "deliveryApproved": False,
    }


def convert(request: ConversionRequest, directory: Path) -> dict[str, object]:
    """Run once, retaining partial evidence; completed reentry only verifies bytes."""
    if load_conversion(request.reference) != request:
        raise ReplayError("Typed converter request differs from pinned JSON")
    directory = metadata._path(str(directory), work=True)
    if directory.exists():
        result = verify_conversion(_artifact(directory / "execution.json"))
        if result["request"] != request.reference.json():
            raise ReplayError("Completed converter namespace belongs to another request")
        return result
    if request.output.exists() or request.output.is_symlink():
        raise ReplayError("Converter output is occupied without complete execution evidence")
    parent = _directory_identity(directory.parent)
    output_parent = technical._canonical(request.output.parent)
    output_parent_identity = output_parent.stat().st_dev, output_parent.stat().st_ino
    inputs = _conversion_inputs(request)
    before = _conversion_snapshot(inputs)
    _source_native(request.source, request.native)
    directory.mkdir(mode=0o700)
    identity = _directory_identity(directory)
    metadata._write(
        directory / "initial.json",
        {
            "request": request.reference.json(),
            "command": conversion_command(request),
            "startedAtNs": time.time_ns(),
            "inputs": before,
            "outputAbsent": True,
        },
    )
    # A failed/interrupted process leaves this immutable namespace occupied.
    technical._run(
        conversion_command(request), directory=directory, name="convert", timeout=request.timeout
    )
    ended = time.time_ns()
    _guard_directory(directory, identity)
    _guard_directory(directory.parent, parent)
    metadata._path(str(request.output))
    if (output_parent.stat().st_dev, output_parent.stat().st_ino) != output_parent_identity:
        raise ReplayError("Converter output directory changed during execution")
    after = _conversion_snapshot(inputs)
    if before != after:
        raise ReplayError("Converter input identity changed during execution")
    output = technical._fingerprint(request.output)
    metadata._write(
        directory / "terminal.json", {"endedAtNs": ended, "inputs": after, "output": output}
    )
    result = _conversion_result(request, directory)
    metadata._write(directory / "execution.json", result)
    return result


def verify_conversion(reference: metadata.Reference) -> dict[str, object]:
    """Freshly verify a complete producer record; never rerun or repair a converter."""
    metadata._path(str(reference.path), work=True)
    if reference.path.name != "execution.json":
        raise ReplayError("Converter receipt must use its exclusive execution namespace")
    saved = _read(reference)
    request = load_conversion(metadata._reference(saved.get("request")))
    _source_native(request.source, request.native)
    identity = _directory_identity(reference.path.parent)
    expected = _conversion_result(request, reference.path.parent)
    if saved != expected or _artifact(reference.path) != reference:
        raise ReplayError("Immutable converter execution evidence changed")
    _guard_directory(reference.path.parent, identity)
    return expected


@dataclass(frozen=True)
class Request:
    reference: metadata.Reference
    source: metadata.Reference
    movie: metadata.Reference
    native: metadata.Reference
    ffmpeg: metadata.Reference
    ffprobe: metadata.Reference
    frames: int
    rate: int
    window_frames: int
    threads: int
    video: dict[str, object]
    thresholds: dict[str, float]
    converter: metadata.Reference | None = None

    @property
    def windows(self) -> tuple[tuple[int, int], ...]:
        size = min(self.window_frames, self.frames)
        starts = (0, (self.frames - size) // 2, self.frames - size)
        return tuple((start, start + size) for start in starts)


def _read(
    ref: metadata.Reference, *, limit: int = metadata.MAX_METADATA_BYTES
) -> dict[str, object]:
    return record(strict_json(_bound_bytes(ref.path, ref.sha256, limit)))


def _count(value: object, *, maximum: int = MAX_FRAMES) -> int:
    if type(value) is not int or not 1 <= value <= maximum:
        raise ReplayError("Observation count must be a bounded positive integer")
    return value


def _number(value: object) -> float:
    if type(value) not in (int, float) or not isinstance(value, (int, float)):
        raise ReplayError("Observation metric must be a finite number")
    result = float(value)
    if not math.isfinite(result):
        raise ReplayError("Observation metric must be a finite number")
    return result


def _tool(value: object) -> metadata.Reference:
    ref = metadata._reference(value)
    if not ref.path.is_file() or not os.access(ref.path, os.X_OK):
        raise ReplayError("Observation tool must be a canonical executable file")
    return ref


def load(reference: metadata.Reference) -> Request:
    """Parse a pinned request; no observations or child processes are started."""
    metadata._path(str(reference.path), work=True)
    raw = _read(reference)
    fields = {
        "version",
        "kind",
        "source",
        "movie",
        "sourceNative",
        "tools",
        "frames",
        "rate",
        "windowFrames",
        "threads",
        "videoConfiguration",
        "thresholds",
    }
    converted = raw.get("kind") == "converted-compressed-part-observation-request"
    if converted:
        fields.add("converterExecution")
    if (
        set(raw) != fields
        or type(raw["version"]) is not int
        or raw["version"] != 1
        or raw["kind"]
        not in (
            "compressed-part-observation-request",
            "converted-compressed-part-observation-request",
        )
    ):
        raise ReplayError("Compressed part observation request schema differs")
    tools = record(raw["tools"])
    video = record(raw["videoConfiguration"])
    thresholds = record(raw["thresholds"])
    if set(tools) != {"ffmpeg", "ffprobe"} or set(video) != set(delivery.VIDEO_FIELDS):
        raise ReplayError("Exact tools and codec configuration are required")
    if any(value is None for value in video.values()):
        raise ReplayError("Codec configuration must contain observed nonmissing facts")
    _count(video["width"])
    _count(video["height"])
    if set(thresholds) != {"ssim", "psnr"}:
        raise ReplayError("Explicit SSIM and PSNR policy is required")
    limits = {name: _number(value) for name, value in thresholds.items()}
    if not 0 <= limits["ssim"] <= 1 or limits["psnr"] < 0:
        raise ReplayError("Quality thresholds are outside their physical domain")
    source, movie = metadata._reference(raw["source"]), metadata._reference(raw["movie"])
    if source.path == movie.path:
        raise ReplayError("Original and compressed video must be distinct")
    result = Request(
        reference,
        source,
        movie,
        metadata._reference(raw["sourceNative"]),
        _tool(tools["ffmpeg"]),
        _tool(tools["ffprobe"]),
        _count(raw["frames"]),
        _count(raw["rate"], maximum=240),
        _count(raw["windowFrames"]),
        _count(raw["threads"], maximum=MAX_THREADS),
        video,
        limits,
        metadata._reference(raw["converterExecution"]) if converted else None,
    )
    if result.ffmpeg.path == result.ffprobe.path:
        raise ReplayError("Decoder tool identities must be distinct")
    return result


def _inputs(request: Request) -> tuple[metadata.Reference, ...]:
    refs = [
        request.reference,
        request.source,
        request.movie,
        request.native,
        request.ffmpeg,
        request.ffprobe,
    ]
    refs.extend(
        metadata.Reference(Path(__file__).parent / name, file_digest(Path(__file__).parent / name))
        for name in CODE_NAMES
    )
    if request.converter is not None:
        execution = verify_conversion(request.converter)
        if (
            execution["source"] != request.source.json()
            or execution["movie"] != request.movie.json()
        ):
            raise ReplayError("Converter output/source differs from observation")
        refs.append(request.converter)
        refs.extend(metadata._reference(value) for value in _list(execution["inputs"]))
        refs.extend(
            ref
            for ref in (metadata._reference(value) for value in _list(execution["artifacts"]))
            if ref.path.stat().st_size > 0
        )
    by_path: dict[Path, metadata.Reference] = {}
    for ref in refs:
        if ref.path in by_path and by_path[ref.path] != ref:
            raise ReplayError("Conflicting observation input pins")
        by_path[ref.path] = ref
    return tuple(by_path[path] for path in sorted(by_path))


def commands(request: Request) -> dict[str, list[str]]:
    """Exact read-only commands; relative stats files stay in the runner's cwd."""
    probe = str(request.ffprobe.path)
    video = [
        probe,
        "-v",
        "error",
        "-threads",
        str(request.threads),
        "-err_detect",
        "explode",
        "-protocol_whitelist",
        "file,pipe",
        "-select_streams",
        "v:0",
        "-show_error",
        "-show_frames",
        "-show_streams",
        "-show_data",
        "-show_data_hash",
        "sha256",
        "-show_entries",
        "frame=media_type,stream_index,pts,best_effort_timestamp,duration,pkt_duration,decode_error_flags:stream:error",
        "-of",
        "json",
        str(request.movie.path),
    ]
    first = [
        probe,
        "-v",
        "error",
        "-protocol_whitelist",
        "file,pipe",
        "-select_streams",
        "v:0",
        "-read_intervals",
        "%+#1",
        "-show_packets",
        "-show_data",
        "-show_data_hash",
        "sha256",
        "-show_entries",
        "packet=stream_index,pts,dts,duration,size,flags,data,data_hash",
        "-of",
        "json",
        str(request.movie.path),
    ]
    filters = ["[0:v:0]split=3[s0][s1][s2]", "[1:v:0]split=3[c0][c1][c2]"]
    mappings: list[str] = []
    for index, (start, end) in enumerate(request.windows):
        filters.extend(
            [
                f"[s{index}]trim=start_frame={start}:end_frame={end},setpts=PTS-STARTPTS,format=yuv420p10le,split=2[ss{index}][sp{index}]",
                f"[c{index}]trim=start_frame={start}:end_frame={end},setpts=PTS-STARTPTS,format=yuv420p10le,split=2[cs{index}][cp{index}]",
                f"[ss{index}][cs{index}]ssim=stats_file=quality-{index}.ssim.stats[sm{index}]",
                f"[sp{index}][cp{index}]psnr=stats_file=quality-{index}.psnr.stats[pm{index}]",
            ]
        )
        mappings.extend(["-map", f"[sm{index}]", "-map", f"[pm{index}]"])
    quality = [
        str(request.ffmpeg.path),
        "-nostdin",
        "-v",
        "error",
        "-xerror",
        "-n",
        "-protocol_whitelist",
        "file,pipe",
        "-threads",
        str(request.threads),
        "-i",
        str(request.source.path),
        "-protocol_whitelist",
        "file,pipe",
        "-threads",
        str(request.threads),
        "-i",
        str(request.movie.path),
        "-filter_complex",
        ";".join(filters),
        *mappings,
        "-an",
        "-f",
        "null",
        "-",
    ]
    result = {"video": video, "first": first, "quality": quality}
    if request.converter is not None:
        result["full-av"] = [
            str(request.ffmpeg.path),
            "-nostdin",
            "-v",
            "error",
            "-xerror",
            "-protocol_whitelist",
            "file,pipe",
            "-threads",
            str(request.threads),
            "-i",
            str(request.movie.path),
            "-map",
            "0",
            "-f",
            "null",
            "-",
        ]
    return result


def _artifact(path: Path) -> metadata.Reference:
    metadata._path(str(path), work=True)
    if path.stat().st_size > MAX_RAW_BYTES:
        raise ReplayError("Observation raw artifact exceeds its bounded budget")
    # The existing technical fingerprint admits genuine empty stderr/null stdout.
    # Stage media hashing deliberately rejects empty artifacts and is not reused
    # for those successful diagnostic streams.
    fingerprint = technical._fingerprint(path)
    return metadata.Reference(path, metadata._sha(fingerprint["sha256"]))


def _process(directory: Path, name: str, command: list[str]) -> tuple[metadata.Reference, ...]:
    process, stderr, stdout = (
        _artifact(directory / f"{name}.{suffix}") for suffix in ("process.json", "stderr", "stdout")
    )
    raw = _read(process)
    if (
        set(raw) != {"command", "returncode"}
        or raw["command"] != command
        or (
            type(raw["returncode"]) is not int
            or raw["returncode"] != 0
            or stderr.path.stat().st_size != 0
        )
    ):
        raise ReplayError("Actual observation command failed or its process evidence differs")
    return process, stderr, stdout


def _dump(value: object) -> bytes:
    if not isinstance(value, str) or not 1 <= len(value) <= 4 * 1024 * 1024:
        raise ReplayError("HEVC dump must be bounded text")
    result = bytearray()
    for line in value.strip().splitlines():
        match = re.fullmatch(r"([0-9a-fA-F]{8}): ([0-9a-fA-F ]+?)(?:  .*|$)", line.strip())
        if match is None or int(match[1], 16) != len(result):
            raise ReplayError("HEVC dump offset or format differs")
        try:
            result.extend(bytes.fromhex(match[2]))
        except ValueError as error:
            raise ReplayError("HEVC dump hex is malformed") from error
    return bytes(result)


def _hash_bytes(data: bytes, value: object) -> None:
    if value != "SHA256:" + hashlib.sha256(data).hexdigest():
        raise ReplayError("Observed codec or packet byte hash differs")


def _packet_size(value: object) -> int:
    # FFprobe serializes packet sizes as decimal strings in ordinary JSON.
    if isinstance(value, str) and re.fullmatch(r"[1-9][0-9]{0,9}", value) is not None:
        return _count(int(value), maximum=MAX_RAW_BYTES)
    return _count(value, maximum=MAX_RAW_BYTES)


def _video(request: Request, raw: dict[str, object]) -> tuple[dict[str, object], int]:
    streams, frames = raw.get("streams"), raw.get("frames")
    if (
        raw.get("error") is not None
        or not isinstance(streams, list)
        or len(streams) != 1
        or not isinstance(frames, list)
    ):
        raise ReplayError("Complete error-free video decode observations are required")
    video = record(streams[0])
    if any(video.get(key) != value for key, value in request.video.items()) or (
        video.get("codec_type") != "video"
        or video.get("codec_name") != "hevc"
        or video.get("codec_tag_string") != "hvc1"
    ):
        raise ReplayError("Observed HEVC codec configuration differs")
    base = delivery._fraction(video.get("time_base"))
    duration = video.get("duration_ts")
    if (
        base.numerator != 1
        or base <= 0
        or delivery._fraction(video.get("avg_frame_rate")) != request.rate
        or (
            type(video.get("start_pts")) is not int
            or video["start_pts"] != 0
            or not isinstance(duration, int)
            or type(duration) is not int
            or duration * base != Fraction(request.frames, request.rate)
        )
    ):
        raise ReplayError("Observed video program extent or rate differs")
    index = video.get("index")
    if not isinstance(index, int) or type(index) is not int or index < 0:
        raise ReplayError("Observed video stream index is missing")
    rows = [record(value) for value in frames]
    if any(
        row.get("media_type") != "video"
        or row.get("stream_index") != video["index"]
        or row.get("decode_error_flags", 0) != 0
        or ("best_effort_timestamp" in row and row["best_effort_timestamp"] != row.get("pts"))
        for row in rows
    ):
        raise ReplayError("Decoded frames have errors or unbound clocks/streams")
    clock = technical._video(rows, scale=base, rate=request.rate, frames=request.frames)
    extradata = _dump(video.get("extradata"))
    _hash_bytes(extradata, video.get("extradata_hash"))
    if (
        len(extradata) < 23
        or extradata[0] != 1
        or video.get("extradata_size") != len(extradata)
        or (extradata[21] & 3) + 1 != 4
    ):
        raise ReplayError("HEVC configuration lacks an observed four-byte NAL length")
    return clock, index


def _first(request: Request, raw: dict[str, object], index: int) -> int:
    packets = raw.get("packets")
    if not isinstance(packets, list) or len(packets) != 1:
        raise ReplayError("Exactly one first coded packet is required")
    packet = record(packets[0])
    data = _dump(packet.get("data"))
    _hash_bytes(data, packet.get("data_hash"))
    if (
        packet.get("stream_index") != index
        or type(packet.get("pts")) is not int
        or packet["pts"] != 0
        or _packet_size(packet.get("size")) != len(data)
    ):
        raise ReplayError("First packet source, size or presentation clock differs")
    kind = delivery._first_vcl(
        {"movie": request.movie.json(), "nalLengthBytes": 4, "dataHex": data.hex()}, request.movie
    )
    if kind not in (19, 20):
        raise ReplayError("First coded picture is not HEVC IDR")
    return kind


def _metric(ref: metadata.Reference, *, metric: str, count: int) -> tuple[float, float]:
    if ref.path.stat().st_size == 0:
        raise ReplayError("Raw metric window is incomplete")
    data = _bound_bytes(ref.path, ref.sha256, MAX_RAW_BYTES).decode("utf-8")
    lines = data.splitlines()
    if len(lines) != count:
        raise ReplayError("Raw metric window is incomplete")
    values: list[float] = []
    field = "All" if metric == "ssim" else "psnr_avg"
    for index, line in enumerate(lines, 1):
        pairs = re.findall(r"([a-zA-Z_]+):([^\s]+)", line)
        tokens = dict(pairs)
        if len(tokens) != len(pairs):
            raise ReplayError("Raw quality metric fields are duplicated")
        if tokens.get("n") != str(index) or field not in tokens:
            raise ReplayError("Raw metric indices or values are missing")
        try:
            value = _number(float(tokens[field]))
        except ValueError as error:
            raise ReplayError("Raw quality metric is malformed") from error
        if value < 0 or (metric == "ssim" and value > 1):
            raise ReplayError("Raw quality metric is outside its physical domain")
        values.append(value)
    return sum(values) / count, min(values)


def _quality(
    request: Request, directory: Path
) -> tuple[list[dict[str, object]], tuple[metadata.Reference, ...]]:
    windows: list[dict[str, object]] = []
    refs: list[metadata.Reference] = []
    for index, (start, end) in enumerate(request.windows):
        ssim, psnr = (
            _artifact(directory / f"quality-{index}.{metric}.stats") for metric in ("ssim", "psnr")
        )
        mean_ssim, min_ssim = _metric(ssim, metric="ssim", count=end - start)
        mean_psnr, min_psnr = _metric(psnr, metric="psnr", count=end - start)
        windows.append(
            {
                "name": WINDOW_NAMES[index],
                "startFrame": start,
                "endFrame": end,
                "ssim": mean_ssim,
                "psnr": mean_psnr,
                "ssimMinimum": min_ssim,
                "psnrMinimum": min_psnr,
                "thresholdsPassed": mean_ssim >= request.thresholds["ssim"]
                and mean_psnr >= request.thresholds["psnr"],
            }
        )
        refs.extend((ssim, psnr))
    return windows, tuple(refs)


def _native(request: Request) -> None:
    raw = _read(request.native)
    if (
        raw.get("movie") != request.source.json()
        or raw.get("completionVerified") is not True
        or raw.get("shareObservedIdle") is not True
        or (type(raw.get("fullAVDecodeExitCode")) is not int or raw["fullAVDecodeExitCode"] != 0)
    ):
        raise ReplayError(
            "Original production part requires bound native completion/full AV evidence"
        )


def _snapshot(inputs: tuple[metadata.Reference, ...]) -> dict[str, object]:
    return {
        "version": 1,
        "kind": "compressed-part-observation-inputs",
        "inputs": [ref.json() for ref in inputs],
    }


def _derive(
    request: Request, directory: Path, inputs: tuple[metadata.Reference, ...]
) -> dict[str, object]:
    raw_refs: list[metadata.Reference] = []
    processes: dict[str, tuple[metadata.Reference, ...]] = {}
    for name, command in commands(request).items():
        processes[name] = _process(directory, name, command)
        raw_refs.extend(processes[name])
    clock, index = _video(request, _read(processes["video"][2], limit=MAX_RAW_BYTES))
    first = _first(request, _read(processes["first"][2]), index)
    windows, metric_refs = _quality(request, directory)
    raw_refs.extend(metric_refs)
    if any(_artifact(ref.path) != ref for ref in raw_refs):
        raise ReplayError("Raw observation changed during domain validation")
    result: dict[str, object] = {
        "version": 1,
        "kind": "current-compressed-part-observation",
        "request": request.reference.json(),
        "source": request.source.json(),
        "movie": request.movie.json(),
        "sourceNative": request.native.json(),
        "commands": commands(request),
        "inputs": [ref.json() for ref in inputs],
        "artifacts": [ref.json() for ref in raw_refs],
        "videoConfiguration": request.video,
        "videoClock": clock,
        "firstVCL": first,
        "qualityWindows": windows,
        "qualityAggregation": "arithmetic-mean-per-decoded-frame",
        "qualityThresholds": request.thresholds,
        "numericalQualityAccepted": all(window["thresholdsPassed"] is True for window in windows),
        "fullVideoDecodeVerified": True,
        "actualObservationCommandsCaptured": True,
        "processIdentityRecorded": False,
        "historicalGenerationVerified": False,
        "candidateOnly": True,
        "partAudioAdopted": False,
        "sourceRetirementAllowed": False,
        "finalCompressorRequired": True,
        "qualityApproved": False,
        "syncApproved": False,
        "deliveryApproved": False,
    }
    if request.converter is not None:
        execution = verify_conversion(request.converter)
        if any(
            execution[key] != expected
            for key, expected in (
                ("source", request.source.json()),
                ("movie", request.movie.json()),
                ("sourceNative", request.native.json()),
                ("tool", request.ffmpeg.json()),
                ("frames", request.frames),
                ("rate", request.rate),
                ("width", request.video["width"]),
                ("height", request.video["height"]),
            )
        ):
            raise ReplayError("Observed converter configuration/source binding differs")
        result["converterExecution"] = request.converter.json()
        result["conversionExecutionVerified"] = True
        result["fullAVDecodeVerified"] = True
    return result


def delivery_evidence(request: Request, directory: Path, output: Path) -> dict[str, object]:
    """Bind verified converter and raw observations for the existing delivery gate."""
    if request.converter is None:
        raise ReplayError("Converter delivery evidence requires an explicit producer")
    verify(request, directory)
    output = metadata._path(str(output), work=True)
    result: dict[str, object] = {
        "version": 1,
        "kind": "converted-compressed-part-evidence",
        "source": request.source.json(),
        "movie": request.movie.json(),
        "sourceNative": request.native.json(),
        "converterExecution": request.converter.json(),
        "observation": _artifact(directory / "observation.json").json(),
    }
    metadata._write(output, result)
    return result


def _pin_inputs(cache: InputHashCache, inputs: tuple[metadata.Reference, ...]) -> None:
    cache.prime()
    if any(file_digest(ref.path) != ref.sha256 for ref in inputs):
        raise ReplayError("Current observation source/output/tool/code pin changed")


def _directory_identity(path: Path) -> tuple[int, int]:
    metadata._path(str(path), work=True)
    info = path.stat()
    if not path.is_dir():
        raise ReplayError("Observation directory is not a directory")
    return info.st_dev, info.st_ino


def _guard_directory(path: Path, identity: tuple[int, int]) -> None:
    if _directory_identity(path) != identity:
        raise ReplayError("Observation evidence directory changed during collection")


def collect(request: Request, directory: Path) -> dict[str, object]:
    """Capture current observations once; interrupted output is never blindly rerun."""
    if load(request.reference) != request:
        raise ReplayError("Typed observation request differs from its pinned JSON")
    directory = metadata._path(str(directory), work=True)
    if directory.exists():
        return verify(request, directory)
    if not directory.parent.is_dir():
        raise ReplayError("Observation evidence needs an existing work parent")
    parent_identity = _directory_identity(directory.parent)
    inputs = _inputs(request)
    cache = InputHashCache(ref.path for ref in inputs)
    with input_hash_context(cache):
        _pin_inputs(cache, inputs)
        _native(request)
        directory.mkdir(mode=0o700)
        directory_identity = _directory_identity(directory)
        _guard_directory(directory.parent, parent_identity)
        metadata._write(directory / "initial.json", _snapshot(inputs))
        for name, command in commands(request).items():
            technical._run(command, directory=directory, name=name, timeout=TIMEOUT_SECONDS)
            _guard_directory(directory, directory_identity)
            _guard_directory(directory.parent, parent_identity)
        result = _derive(request, directory, inputs)
        cache.verify()
        _guard_directory(directory, directory_identity)
        _guard_directory(directory.parent, parent_identity)
        metadata._write(directory / "terminal.json", _snapshot(inputs))
        result["initial"] = _artifact(directory / "initial.json").json()
        result["terminal"] = _artifact(directory / "terminal.json").json()
        metadata._write(directory / "observation.json", result)
        _guard_directory(directory, directory_identity)
        return result


def verify(request: Request, directory: Path) -> dict[str, object]:
    """Revalidate immutable real observations; never decode, repair or create outputs."""
    if load(request.reference) != request:
        raise ReplayError("Typed observation request differs from its pinned JSON")
    directory = metadata._path(str(directory), work=True)
    directory_identity = _directory_identity(directory)
    receipt = _artifact(directory / "observation.json")
    result = _read(receipt)
    inputs = _inputs(request)
    cache = InputHashCache(ref.path for ref in inputs)
    with input_hash_context(cache):
        _pin_inputs(cache, inputs)
        _native(request)
        for name in ("initial", "terminal"):
            ref = metadata._reference(result.get(name))
            if ref.path != directory / f"{name}.json" or _read(ref) != _snapshot(inputs):
                raise ReplayError("Observation initial/terminal input closure differs")
        expected = _derive(request, directory, inputs)
        expected["initial"], expected["terminal"] = result["initial"], result["terminal"]
        if result != expected:
            raise ReplayError("Immutable current observation evidence changed")
        cache.verify()
        if _artifact(receipt.path) != receipt:
            raise ReplayError("Observation receipt changed during verification")
        _guard_directory(directory, directory_identity)
        return result


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "action", choices=("collect", "verify", "convert", "verify-conversion", "delivery-evidence")
    )
    parser.add_argument("request", type=Path)
    parser.add_argument("evidence", type=Path)
    parser.add_argument("--sha256", required=True)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args(argv)
    try:
        reference = metadata.Reference(args.request, metadata._sha(args.sha256))
        if args.action == "convert":
            result = convert(load_conversion(reference), args.evidence)
        elif args.action == "verify-conversion":
            if metadata._path(str(args.evidence), work=True) != reference.path.parent:
                raise ReplayError("Converter verification namespace differs")
            result = verify_conversion(reference)
        elif args.action == "delivery-evidence":
            if args.output is None:
                raise ReplayError("Delivery evidence needs an exclusive output")
            result = delivery_evidence(load(reference), args.evidence, args.output)
        else:
            request = load(reference)
            result = {"collect": collect, "verify": verify}[args.action](request, args.evidence)
    except (OSError, ValueError, ReplayError) as error:
        parser.exit(1, f"COMPRESSED_PART_OBSERVATION_ERROR {error}\n")
    print(json.dumps(result, sort_keys=True, ensure_ascii=False, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
