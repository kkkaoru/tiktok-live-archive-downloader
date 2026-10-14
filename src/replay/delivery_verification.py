"""Independent, non-publishing final-media technical verification.

Receipts never approve captions, music removal, transcription, FCP or quality.
Only a newly reserved repository-local directory is written; failed runs remain.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import selectors
import stat
import subprocess
import sys
import time
from fractions import Fraction
from pathlib import Path
from typing import Literal

from replay.core import ReplayError
from replay.delivery_clock_contract import ClockExpectation, load_clock_contract
from replay.delivery_render import Config, load_config, read_cuts
from replay.recordings import record

REPOSITORY = Path(__file__).resolve().parents[2]
MAX_BYTES = 256 * 1024 * 1024
MAX_ROWS = 4_000_000
MAX_SECONDS = 86_400
ERRORS = (ReplayError, OSError, ValueError, KeyError, ZeroDivisionError)


def _canonical(path: Path) -> Path:
    absolute = path.absolute()
    if absolute.resolve() != absolute:
        raise ReplayError("Paths must be canonical and contain no symlinks")
    return absolute


def _write(path: Path, value: object) -> None:
    with path.open("x", encoding="utf-8") as handle:
        json.dump(value, handle, indent=2, ensure_ascii=False)
        handle.write("\n")


def _fingerprint(path: Path) -> dict[str, object]:
    path = _canonical(path)
    before = path.stat()
    if not stat.S_ISREG(before.st_mode) or before.st_nlink != 1:
        raise ReplayError("Input must be a regular, singly linked file")
    with path.open("rb") as handle:
        digest = hashlib.file_digest(handle, "sha256").hexdigest()
    after = path.stat()
    if (before.st_dev, before.st_ino, before.st_size, before.st_mtime_ns, before.st_ctime_ns) != (
        after.st_dev,
        after.st_ino,
        after.st_size,
        after.st_mtime_ns,
        after.st_ctime_ns,
    ):
        raise ReplayError("Input changed during hashing")
    return {
        "path": str(path),
        "bytes": before.st_size,
        "sha256": digest,
        "device": before.st_dev,
        "inode": before.st_ino,
        "mtimeNs": before.st_mtime_ns,
        "ctimeNs": before.st_ctime_ns,
    }


def _json(path: Path) -> dict[str, object]:
    if path.stat().st_size > MAX_BYTES:
        raise ReplayError("JSON byte limit exceeded")
    return record(json.loads(path.read_text(encoding="utf-8")))


def _rows(data: dict[str, object], key: str) -> list[dict[str, object]]:
    values = data.get(key)
    if not isinstance(values, list) or not 0 < len(values) <= MAX_ROWS:
        raise ReplayError(f"Missing or excessive {key}")
    return [record(value) for value in values]


def _int(value: object) -> int:
    if type(value) is int:
        return value
    if isinstance(value, str) and value.removeprefix("-").isdigit():
        return int(value)
    raise ReplayError("Expected an integer clock/count")


def _scale(value: object) -> Fraction:
    if not isinstance(value, str) or "/" not in value:
        raise ReplayError("Expected rational time base")
    result = Fraction(value)
    if result <= 0 or result.denominator > 1_000_000_000:
        raise ReplayError("Invalid time base")
    return result


def _expected(config: Config | ClockExpectation) -> int:
    if isinstance(config, ClockExpectation):
        return config.expected_frames
    rows = read_cuts(config.cuts)
    total, previous = 0, 0
    raw = _rows(_json(config.cuts), "cuts")
    for cut, value in zip(rows, raw, strict=True):
        if cut.start < previous or cut.end <= cut.start:
            raise ReplayError("Cut plan must be ordered and nonoverlapping")
        if "output_start_frame" in value and _int(value["output_start_frame"]) != total:
            raise ReplayError("Cut output clock is not contiguous")
        total += cut.end - cut.start
        previous = cut.end
    if total > MAX_ROWS or Fraction(total, config.rate) > MAX_SECONDS:
        raise ReplayError("Expected frame/duration limit exceeded")
    return total


def _run(command: list[str], *, directory: Path, name: str, timeout: int) -> Path:
    """Drain bounded pipes to retained files, including on timeout/nonzero exit."""
    output = directory / f"{name}.stdout"
    diagnostic: dict[str, object] = {"command": command, "returncode": None}
    env = dict(os.environ, TMPDIR=str(directory), TMP=str(directory), TEMP=str(directory))
    # Avoid inherited FFREPORT writing outside the evidence directory.
    env.pop("FFREPORT", None)
    try:
        with (
            output.open("xb") as out,
            (directory / f"{name}.stderr").open("xb") as err,
            subprocess.Popen(
                command,
                stdin=subprocess.DEVNULL,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                cwd=directory,
                env=env,
            ) as process,
        ):
            try:
                with selectors.DefaultSelector() as selector:
                    if process.stdout is None or process.stderr is None:
                        raise ReplayError("Missing child pipes")
                    selector.register(process.stdout, selectors.EVENT_READ)
                    selector.register(process.stderr, selectors.EVENT_READ)
                    deadline, count = time.monotonic() + timeout, 0
                    while selector.get_map():
                        if time.monotonic() >= deadline:
                            raise ReplayError(f"{name}: runtime limit exceeded")
                        for key, _ in selector.select(timeout=0.1):
                            chunk = os.read(key.fd, 65536)
                            if not chunk:
                                selector.unregister(key.fileobj)
                                continue
                            count += len(chunk)
                            if count > MAX_BYTES:
                                raise ReplayError(f"{name}: output byte limit exceeded")
                            destination = out if key.fd == process.stdout.fileno() else err
                            destination.write(chunk)
                    process.wait(timeout=max(0.001, deadline - time.monotonic()))
            finally:
                if process.poll() is None:
                    process.kill()
                diagnostic["returncode"] = process.wait()
        if diagnostic["returncode"] != 0 or (directory / f"{name}.stderr").stat().st_size:
            raise ReplayError(f"{name}: child failed; inspect retained stderr")
    except (*ERRORS, subprocess.TimeoutExpired) as exc:
        diagnostic["error"] = str(exc)
        raise ReplayError(str(exc)) from exc
    finally:
        _write(directory / f"{name}.process.json", diagnostic)
    return output


def _streams(
    data: dict[str, object],
    config: Config | ClockExpectation,
    *,
    video_contract: Literal["hevc", "h264"] = "hevc",
) -> tuple[dict[str, object], dict[str, object]]:
    if video_contract not in {"hevc", "h264"}:
        raise ReplayError("Unsupported explicit video contract")
    streams = _rows(data, "streams")
    video = [row for row in streams if row.get("codec_type") == "video"]
    audio = [row for row in streams if row.get("codec_type") == "audio"]
    if len(video) != 1 or len(audio) != 1:
        raise ReplayError("Exactly one video and one audio stream required")
    v, a = video[0], audio[0]
    if (v.get("codec_name"), v.get("codec_tag_string"), v.get("width"), v.get("height")) != (
        video_contract,
        "hvc1" if video_contract == "hevc" else "avc1",
        config.width,
        config.height,
    ):
        raise ReplayError("Video codec/tag/geometry mismatch")
    if _scale(v.get("avg_frame_rate")) != config.rate:
        raise ReplayError("Video rate mismatch")
    if not isinstance(a.get("codec_name"), str) or _int(a.get("sample_rate")) <= 0:
        raise ReplayError("Missing audio codec/sample rate")
    return v, a


def _video(
    rows: list[dict[str, object]], *, scale: Fraction, rate: int, frames: int
) -> dict[str, object]:
    if len(rows) != frames:
        raise ReplayError("Decoded frame count differs from cuts (EOF)")
    for index, row in enumerate(rows):
        duration = row.get("duration", row.get("pkt_duration"))
        if _int(row.get("pts")) * scale != Fraction(index, rate):
            raise ReplayError("Decoded PTS is not exact zero-origin CFR")
        if _int(duration) * scale != Fraction(1, rate):
            raise ReplayError("Decoded frame duration/endpoint mismatch")
    return {"frames": frames, "start": "0", "end": str(Fraction(frames, rate)), "exactCFR": True}


def _audio(
    rows: list[dict[str, object]], *, scale: Fraction, sample_rate: int | None
) -> dict[str, object]:
    starts: list[Fraction] = []
    ends: list[Fraction] = []
    samples = 0
    for row in rows:
        start = _int(row.get("pts")) * scale
        count = _int(row.get("duration") if sample_rate is None else row.get("nb_samples"))
        if count <= 0:
            raise ReplayError("Audio duration/sample count must be positive")
        duration = count * scale if sample_rate is None else Fraction(count, sample_rate)
        starts.append(start)
        ends.append(start + duration)
        samples += count
    if not starts:
        raise ReplayError("Audio has no measured clocks")
    return {
        "count": len(rows),
        "start": str(min(starts)),
        "end": str(max(ends)),
        "span": str(max(ends) - min(starts)),
        "sampleCount" if sample_rate is not None else "durationTicksSum": samples,
    }


def verify(
    config_path: Path,
    movie: Path,
    evidence: Path,
    *,
    ffmpeg: Path,
    ffprobe: Path,
    timeout: int = 7200,
    video_contract: Literal["hevc", "h264"] = "hevc",
    clock_contract: bool = False,
) -> bool:
    """Reserve evidence, then retain a receipt for success or validation failure.

    Unsafe/existing evidence destinations raise before writing anything. The
    caller must supply trusted local FFmpeg executables, not shell commands.
    """
    evidence = _canonical(evidence)
    if not evidence.is_relative_to(REPOSITORY) or evidence == REPOSITORY:
        raise ReplayError("Evidence must be inside the resolved repository")
    evidence.mkdir(mode=0o700)
    receipt: dict[str, object] = {
        "version": 1,
        "technicalAccepted": False,
        "fullDecode": False,
        "expectedVideoContract": video_contract,
        "approvals": dict.fromkeys(
            (
                "captions",
                "sourceCaptions",
                "musicRemoval",
                "transcription",
                "FCP",
                "quality",
                "sync",
            ),
            False,
        ),
    }
    bindings: dict[Path, dict[str, object]] = {}
    try:
        if not 1 <= timeout <= MAX_SECONDS:
            raise ReplayError("Invalid process timeout")
        config_path, movie = _canonical(config_path), _canonical(movie)
        bindings[config_path] = _fingerprint(config_path)
        config = load_clock_contract(config_path) if clock_contract else load_config(config_path)
        root = _canonical(config.delivery_root)
        if not root.is_dir() or not movie.is_relative_to(root):
            raise ReplayError("Candidate is outside configured delivery root")
        if isinstance(config, ClockExpectation):
            receipt["expectationKind"] = "pinned-edited-clock-contract"
            for ref in config.inputs:
                fingerprint = _fingerprint(ref.path)
                if fingerprint["sha256"] != ref.sha256 or (
                    ref.path in bindings and bindings[ref.path] != fingerprint
                ):
                    raise ReplayError("Clock contract input changed before verification")
                bindings[ref.path] = fingerprint
        else:
            bindings[config.cuts] = _fingerprint(config.cuts)
        bindings[movie] = _fingerprint(movie)
        receipt["inputs"] = list(bindings.values())
        frames = _expected(config)
        receipt["expected"] = {
            "frames": frames,
            "rate": config.rate,
            "width": config.width,
            "height": config.height,
        }
        for tool in (ffmpeg, ffprobe):
            if not tool.is_absolute() or not tool.is_file() or not os.access(tool, os.X_OK):
                raise ReplayError("Tools must be explicit absolute executable paths")
        receipt["tools"] = {"ffmpeg": str(ffmpeg.resolve()), "ffprobe": str(ffprobe.resolve())}
        _run(
            [
                str(ffmpeg),
                "-nostdin",
                "-v",
                "error",
                "-xerror",
                "-max_error_rate",
                "0",
                "-protocol_whitelist",
                "file,pipe",
                "-i",
                str(movie),
                "-map",
                "0:v",
                "-map",
                "0:a",
                "-f",
                "null",
                "-",
            ],
            directory=evidence,
            name="decode",
            timeout=timeout,
        )
        receipt["fullDecode"] = True
        prefix = [
            str(ffprobe),
            "-v",
            "error",
            "-threads",
            "0",
            "-protocol_whitelist",
            "file,pipe",
            "-of",
            "json",
        ]
        stream_path = _run(
            prefix + ["-show_streams", "-show_format", str(movie)],
            directory=evidence,
            name="streams",
            timeout=timeout,
        )
        stream_data = _json(stream_path)
        receipt["streams"] = stream_data
        video, audio = _streams(stream_data, config, video_contract=video_contract)
        frame_path = _run(
            prefix
            + [
                "-show_frames",
                "-show_entries",
                "frame=media_type,pts,duration,pkt_duration,nb_samples",
                str(movie),
            ],
            directory=evidence,
            name="frames",
            timeout=timeout,
        )
        rows = _rows(_json(frame_path), "frames")
        receipt["videoClock"] = _video(
            [row for row in rows if row.get("media_type") == "video"],
            scale=_scale(video.get("time_base")),
            rate=config.rate,
            frames=frames,
        )
        audio_clock = _audio(
            [row for row in rows if row.get("media_type") == "audio"],
            scale=_scale(audio.get("time_base")),
            sample_rate=_int(audio.get("sample_rate")),
        )
        receipt["audioDecodedClock"] = audio_clock
        receipt["audioEndMinusVideoEnd"] = str(
            Fraction(str(audio_clock["end"])) - Fraction(frames, config.rate)
        )
        receipt["audioSpanMinusVideoDuration"] = str(
            Fraction(str(audio_clock["span"])) - Fraction(frames, config.rate)
        )
        packets_path = _run(
            prefix
            + [
                "-show_packets",
                "-count_packets",
                "-show_entries",
                "packet=stream_index,pts,dts,duration:stream=index,nb_read_packets",
                str(movie),
            ],
            directory=evidence,
            name="packets",
            timeout=timeout,
        )
        packets_data = _json(packets_path)
        packets = _rows(packets_data, "packets")
        video_packets = [row for row in packets if row.get("stream_index") == video.get("index")]
        counts = _rows(packets_data, "streams")
        counted = [row for row in counts if row.get("index") == video.get("index")]
        if (
            len(counted) != 1
            or _int(counted[0].get("nb_read_packets")) != frames
            or len(video_packets) != frames
        ):
            raise ReplayError("Video packet count mismatch")
        receipt["videoPackets"] = len(video_packets)
        receipt["audioPacketClock"] = _audio(
            [row for row in packets if row.get("stream_index") == audio.get("index")],
            scale=_scale(audio.get("time_base")),
            sample_rate=None,
        )
        receipt["technicalAccepted"] = True
    except ERRORS as exc:
        receipt["error"] = str(exc)
    finally:
        receipt["inputs"] = list(bindings.values())
        try:
            after = {path: _fingerprint(path) for path in bindings}
            receipt["inputsAfter"] = list(after.values())
            if after != bindings:
                raise ReplayError("Bound input changed during verification")
        except ERRORS as exc:
            receipt["technicalAccepted"] = False
            receipt["bindingError"] = str(exc)
        _write(evidence / "receipt.json", receipt)
    return receipt["technicalAccepted"] is True


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("config", type=Path)
    parser.add_argument("movie", type=Path)
    parser.add_argument("evidence", type=Path)
    parser.add_argument("--ffmpeg", type=Path, required=True)
    parser.add_argument("--ffprobe", type=Path, required=True)
    parser.add_argument("--timeout", type=int, default=7200)
    parser.add_argument("--video-contract", choices=("hevc", "h264"), default="hevc")
    parser.add_argument(
        "--clock-contract",
        action="store_true",
        help="Read a pinned edited-clock contract instead of a legacy render configuration",
    )
    args = parser.parse_args(argv)
    try:
        return (
            0
            if verify(
                args.config,
                args.movie,
                args.evidence,
                ffmpeg=args.ffmpeg,
                ffprobe=args.ffprobe,
                timeout=args.timeout,
                video_contract=args.video_contract,
                clock_contract=args.clock_contract,
            )
            else 1
        )
    except ERRORS as exc:
        print(json.dumps({"technicalAccepted": False, "error": str(exc)}), file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
