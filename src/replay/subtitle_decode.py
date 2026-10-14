"""Bounded native video decode with exact presentation-clock evidence."""

import os
import subprocess
from fractions import Fraction
from pathlib import Path

from replay.subtitle_clock import VideoFrameClock

MAX_TRACE_BYTES = 16 * 1024 * 1024


def verify_native_decoded_window(
    source: Path,
    clock: VideoFrameClock,
    *,
    evidence_directory: Path,
    timeout_seconds: int,
) -> int:
    """Decode every expected frame, retaining private evidence even on failure.

    The caller binds the unchanged source and verified clock vector, and checks
    terminal extent and other streams separately. Exhaustive disjoint windows
    may run through the existing two-worker scheduler. This forces VideoToolbox
    surfaces without CPU pixel copies or a software fallback. Framecrc hashes
    AVFrame wrappers, NOT pixel planes; only decoded PTS/counts are verified here.
    It establishes neither pixel equivalence nor perceptual/mask quality.
    """
    if type(timeout_seconds) is not int or not 1 <= timeout_seconds <= 86400:
        raise ValueError("Invalid native decode deadline")
    if source.is_symlink() or not source.is_file():
        raise ValueError("Native decode requires a regular local source")
    evidence_directory.mkdir(mode=0o700)
    trace = evidence_directory / "frames.framecrc"
    errors = evidence_directory / "decode.stderr"
    seek = max(0, int(clock.pts[0] * clock.time_base) - 1)
    graph = f"settb={clock.time_base},select=gte(pts\\,{clock.pts[0]}),format=videotoolbox_vld"
    with trace.open("xb") as output, errors.open("xb") as diagnostics:
        os.fchmod(output.fileno(), 0o600)
        os.fchmod(diagnostics.fileno(), 0o600)
        subprocess.run(
            [
                "ffmpeg",
                "-v",
                "error",
                "-xerror",
                "-err_detect",
                "explode",
                "-nostdin",
                "-copyts",
                "-noaccurate_seek",
                "-ss",
                str(seek),
                "-protocol_whitelist",
                "file",
                "-hwaccel",
                "videotoolbox",
                "-hwaccel_output_format",
                "videotoolbox_vld",
                "-i",
                str(source.resolve()),
                "-map",
                "0:v:0",
                "-an",
                "-vf",
                graph,
                "-frames:v",
                str(len(clock.pts)),
                "-c:v",
                "wrapped_avframe",
                "-fps_mode",
                "passthrough",
                "-enc_time_base",
                "filter",
                "-f",
                "framecrc",
                "pipe:1",
            ],
            stdin=subprocess.DEVNULL,
            stdout=output,
            stderr=diagnostics,
            check=True,
            timeout=timeout_seconds,
        )
    if errors.stat().st_size or trace.stat().st_size > MAX_TRACE_BYTES:
        raise ValueError("Native decode diagnostics or trace budget failed")
    return _verify_trace(trace, clock)


def _verify_trace(trace: Path, clock: VideoFrameClock) -> int:
    base: Fraction | None = None
    wrapped = False
    count = 0
    with trace.open(encoding="utf-8") as stream:
        for line in stream:
            if line.startswith("#tb 0:"):
                if base is not None or count:
                    raise ValueError("Repeated decoded time base")
                base = Fraction(line.split(":", 1)[1].strip())
                if base <= 0:
                    raise ValueError("Invalid decoded time base")
            elif line.startswith("#codec_id 0:"):
                wrapped = line.strip() == "#codec_id 0: wrapped_avframe"
            elif line.strip() and not line.startswith("#"):
                fields = line.split(",")
                if (
                    base is None
                    or not wrapped
                    or len(fields) < 6
                    or fields[0].strip() != "0"
                    or count >= len(clock.pts)
                    or int(fields[2]) * base != clock.pts[count] * clock.time_base
                ):
                    raise ValueError("Decoded frame is outside the expected presentation grid")
                count += 1
    if count != len(clock.pts):
        raise ValueError("Native decode did not cover every expected frame")
    return count
