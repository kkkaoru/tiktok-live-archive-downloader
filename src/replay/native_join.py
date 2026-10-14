"""Lossless join of verified native part renders into one QuickTime-compatible file.

Each part's packet count, codec, and mux time base are checked against the plan
before joining. The join stream-copies video (no re-encode), optionally adds one
already-encoded audio file, publishes without overwriting, and re-counts the
joined packets.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
from collections.abc import Sequence
from dataclasses import dataclass
from fractions import Fraction
from pathlib import Path
from tempfile import TemporaryDirectory

from replay.core import ReplayError
from replay.media import MP4_STREAM_COPY_OPTIONS, quicktime_video_tags
from replay.subtitle_clock import concat_durations

PROBE_TIMEOUT_SECONDS = 600
JOIN_TIMEOUT_SECONDS = 3600


@dataclass(frozen=True, slots=True)
class PartFile:
    path: Path
    frames: int


@dataclass(frozen=True, slots=True)
class VideoStream:
    codec: str
    time_base: Fraction
    packets: int
    index: int = 0


def _ffprobe(path: Path) -> object:
    try:
        probe = subprocess.run(
            [
                "ffprobe",
                "-v",
                "error",
                "-select_streams",
                "v:0",
                "-count_packets",
                "-show_entries",
                "stream=index,codec_name,time_base,nb_read_packets",
                "-of",
                "json",
                str(path),
            ],
            check=True,
            capture_output=True,
            text=True,
            timeout=PROBE_TIMEOUT_SECONDS,
        )
    except (OSError, subprocess.CalledProcessError, subprocess.TimeoutExpired) as error:
        raise ReplayError("Local ffprobe failed; cannot verify a part render.") from error
    data: object = json.loads(probe.stdout)
    return data


def probe_video(path: Path) -> VideoStream:
    """Codec, mux time base, and counted packets of the first video stream."""
    data = _ffprobe(path)
    streams = data.get("streams") if isinstance(data, dict) else None
    stream = streams[0] if isinstance(streams, list) and len(streams) == 1 else None
    if not isinstance(stream, dict):
        raise ReplayError("A part render has no single video stream")
    codec, time_base = stream.get("codec_name"), stream.get("time_base")
    packets = stream.get("nb_read_packets")
    index = stream.get("index")
    if (
        not isinstance(codec, str)
        or not isinstance(time_base, str)
        or not isinstance(packets, str)
        or type(index) is not int
        or index < 0
    ):
        raise ReplayError("Invalid local ffprobe stream data.")
    try:
        return VideoStream(codec, Fraction(time_base), int(packets), index)
    except (ValueError, ZeroDivisionError) as error:
        raise ReplayError("Invalid local ffprobe stream data.") from error


def _quoted(path: Path) -> str:
    return "'" + str(path).replace("'", "'\\''") + "'"


def concat_list(parts: Sequence[PartFile], rate: int, time_base: Fraction) -> str:
    """An ffconcat script whose offsets land exactly on each part's planned frames."""
    durations = concat_durations(
        tuple(Fraction(part.frames, rate) for part in parts), time_base=time_base
    )
    lines = ["ffconcat version 1.0"]
    for part, duration in zip(parts, durations, strict=True):
        lines.extend([f"file {_quoted(part.path.resolve())}", f"duration {duration}"])
    return "\n".join(lines) + "\n"


def verify_parts(parts: Sequence[PartFile]) -> VideoStream:
    """The common stream of all parts; every part must hold exactly its planned frames."""
    if not parts:
        raise ReplayError("No part renders to join")
    first = probe_video(parts[0].path)
    if first.time_base.numerator != 1:
        raise ReplayError("Part renders need a 1/N mux time base")
    for part in parts:
        stream = first if part is parts[0] else probe_video(part.path)
        if (stream.codec, stream.time_base) != (first.codec, first.time_base):
            raise ReplayError("Part renders must share one codec and time base")
        if stream.index != first.index:
            raise ReplayError(
                "Part renders must share one video stream index; losslessly remux mismatched parts"
            )
        if stream.packets != part.frames:
            raise ReplayError("A part render does not hold its planned frames")
    return first


def _join_command(
    script: Path, *, audio: Path | None, stream: VideoStream, tags: list[str], output: Path
) -> list[str]:
    """Stream copy that keeps the parts' mux time base, so frame times stay exact."""
    command = ["ffmpeg", "-nostdin", "-v", "error", "-n", "-f", "concat", "-safe", "0"]
    command += ["-i", str(script)]
    maps = ["-map", "0:v:0"]
    if audio is not None:
        command += ["-i", str(audio)]
        maps += ["-map", "1:a:0"]
    clock = ["-video_track_timescale", str(stream.time_base.denominator)]
    return command + maps + ["-c", "copy", *clock, *tags, *MP4_STREAM_COPY_OPTIONS, str(output)]


def join(parts: Sequence[PartFile], *, rate: int, output: Path, audio: Path | None) -> int:
    """Join verified parts into a new ``output``; return the joined frame count."""
    if output.exists():
        raise ReplayError("Refusing to overwrite the joined output")
    stream = verify_parts(parts)
    expected = sum(part.frames for part in parts)
    with TemporaryDirectory(dir=output.parent, prefix=".join-") as work:
        script = Path(work) / "parts.ffconcat"
        script.write_text(concat_list(parts, rate, stream.time_base), encoding="utf-8")
        staged = Path(work) / output.name
        command = _join_command(
            script,
            audio=audio,
            stream=stream,
            tags=quicktime_video_tags(parts[0].path),
            output=staged,
        )
        try:
            subprocess.run(command, check=True, capture_output=True, timeout=JOIN_TIMEOUT_SECONDS)
        except (OSError, subprocess.CalledProcessError, subprocess.TimeoutExpired) as error:
            raise ReplayError("Local ffmpeg could not join the part renders") from error
        if probe_video(staged).packets != expected:
            raise ReplayError("The joined output does not hold the planned frames")
        os.link(staged, output)
    return expected


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("plan", type=Path, help="render-plan.json with durationFrames per part")
    parser.add_argument("parts", type=Path, help="directory of part-NNNNNN.mp4 renders")
    parser.add_argument("output", type=Path, help="new joined .mp4")
    parser.add_argument("--rate", type=int, default=30)
    parser.add_argument("--audio", type=Path, help="already-encoded audio to stream-copy")
    args = parser.parse_args(argv)
    plan: object = json.loads(args.plan.read_text(encoding="utf-8"))
    if not isinstance(plan, list):
        raise ReplayError("Render plan is empty")
    parts: list[PartFile] = []
    for index, raw in enumerate(plan):
        frames = raw.get("durationFrames") if isinstance(raw, dict) else None
        if type(frames) is not int:
            raise ReplayError("Invalid render plan frames")
        parts.append(PartFile(args.parts / f"part-{index:06d}.mp4", frames))
    frames = join(parts, rate=args.rate, output=args.output, audio=args.audio)
    print(json.dumps({"parts": len(parts), "frames": frames}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
