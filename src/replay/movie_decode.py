"""Decode a whole movie with no reported error, splitting the video at a keyframe when it is safe.

One VideoToolbox decode of a 1080x1920 HEVC movie runs at about 450 fps on this Mac, while two
concurrent ones reach about 800 fps. A movie without reordered frames (``has_b_frames`` 0) is
decoded as two video segments that meet exactly at a keyframe near the middle, plus one audio
pass, all at once. The check is as strict as one whole pass and adds a count: every process must
exit 0 with empty stderr, and the segments' decoded frames must sum to the probe's packet count,
so a frame missing or decoded twice at the seam fails. Any other movie takes the single pass.
"""

import json
import math
import subprocess
from collections.abc import Sequence
from fractions import Fraction
from pathlib import Path

from replay.recordings import record

TIMEOUT_SECONDS = 7200
# Seconds of packets read around the middle to find the split keyframe.
WINDOW_SECONDS = 10
MICROSECONDS = 1_000_000
HWACCEL = ("-hwaccel", "videotoolbox", "-hwaccel_output_format", "videotoolbox_vld")


def _probe(movie: Path, ffprobe: str, streams: str, *arguments: str) -> dict[str, object]:
    out = subprocess.run(
        [ffprobe, "-v", "error", "-select_streams", streams, *arguments, "-of", "json", str(movie)],
        capture_output=True, text=True, check=True, timeout=1800,
    ).stdout  # fmt: skip
    return record(json.loads(out))


def split_point(movie: Path, ffprobe: str) -> str | None:
    """The middle keyframe as seconds floored to microseconds, or None when splitting is unsafe."""
    entries = "stream=time_base,has_b_frames,duration"
    streams = _probe(movie, ffprobe, "v:0", "-show_entries", entries).get("streams")
    stream = record(streams[0]) if isinstance(streams, list) and streams else {}
    audio = _probe(movie, ffprobe, "a:0", "-show_entries", "stream=index").get("streams")
    if stream.get("has_b_frames") != 0 or "duration" not in stream or not audio:
        return None
    base = Fraction(str(stream["time_base"]))
    middle = float(str(stream["duration"])) / 2
    window = f"{middle:.6f}%+{WINDOW_SECONDS}"
    listed = _probe(
        movie, ffprobe, "v:0", "-read_intervals", window, "-show_entries", "packet=pts,flags"
    )
    found = listed.get("packets")
    packets = [record(item) for item in found] if isinstance(found, list) else []
    stamps = [int(str(packet["pts"])) for packet in packets if "pts" in packet]
    if stamps != sorted(stamps):
        return None
    keys = [
        int(str(packet["pts"]))
        for packet in packets
        if "K" in str(packet.get("flags", "")) and int(str(packet.get("pts", 0))) > 0
    ]
    if not keys:
        return None
    microseconds = math.floor(keys[0] * base * MICROSECONDS)
    return f"{microseconds // MICROSECONDS}.{microseconds % MICROSECONDS:06d}"


def _start(command: Sequence[str]) -> subprocess.Popen[str]:
    return subprocess.Popen(
        list(command), stdin=subprocess.DEVNULL, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
        text=True,
    )  # fmt: skip


def _finish(child: subprocess.Popen[str]) -> tuple[bool, str]:
    """(clean, progress output): exit 0 and nothing on stderr."""
    try:
        out, err = child.communicate(timeout=TIMEOUT_SECONDS)
    except subprocess.TimeoutExpired:
        child.kill()
        child.communicate()
        return False, ""
    return child.returncode == 0 and not err.strip(), out


def decoded_frames(progress: str) -> int | None:
    """The last ``frame=N`` of an ``-progress`` report."""
    counts = [line[6:] for line in progress.splitlines() if line.startswith("frame=")]
    return int(counts[-1]) if counts and counts[-1].isdigit() else None


def single_pass(movie: Path, ffmpeg: str) -> bool:
    """The whole movie decodes (video on VideoToolbox, audio) with no reported error."""
    result = subprocess.run(
        [ffmpeg, "-v", "error", "-nostdin", "-hwaccel", "videotoolbox", "-i", str(movie),
         "-f", "null", "-"],
        capture_output=True, text=True, timeout=TIMEOUT_SECONDS,
    )  # fmt: skip
    return result.returncode == 0 and not result.stderr.strip()


def commands(movie: Path, ffmpeg: str, split: str) -> list[list[str]]:
    """Video before the split, video from the split keyframe on, and the whole audio."""
    base = [ffmpeg, "-v", "error", "-nostdin", "-nostats"]
    video = ["-map", "0:v:0", "-f", "null", "-", "-progress", "pipe:1"]
    return [
        [*base, *HWACCEL, "-t", split, "-i", str(movie), *video],
        [*base, *HWACCEL, "-ss", split, "-i", str(movie), *video],
        [*base, "-i", str(movie), "-map", "0:a:0", "-f", "null", "-"],
    ]


def decodes(movie: Path, *, ffmpeg: str, ffprobe: str, frames: int) -> bool:
    """Every packet decodes with no reported error; ``frames`` is the probe's video packet count."""
    split = split_point(movie, ffprobe)
    if split is None:
        return single_pass(movie, ffmpeg)
    children = [_start(command) for command in commands(movie, ffmpeg, split)]
    results = [_finish(child) for child in children]
    if not all(clean for clean, _ in results):
        return False
    counts = [decoded_frames(progress) for _, progress in results[:2]]
    return None not in counts and sum(count or 0 for count in counts) == frames
