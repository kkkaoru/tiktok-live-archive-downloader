"""Streaming FFmpeg adapter for prepared CFR parts; no audio or caption changes."""

from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path
from subprocess import PIPE, Popen
from time import monotonic

import numpy as np

from replay.subtitle_inpaint import Image
from replay.subtitle_repair import RepairedFrame
from replay.subtitle_retry import RepairEvidence
from replay.subtitle_stream import SubtitleSpan, repair_stream

ENCODER_OPTIONS: tuple[tuple[str, ...], ...] = (
    ("-c:v", "libx264", "-preset", "fast", "-crf", "18"),
    ("-c:v", "h264_videotoolbox", "-b:v", "8000k"),
)


@dataclass(frozen=True, slots=True)
class VideoRepairJob:
    source: Path
    output: Path
    width: int
    height: int
    frames: int
    spans: tuple[SubtitleSpan, ...]
    fps: int = 30
    allow_padded_tail: bool = False
    hardware_encode: bool = False

    def __post_init__(self) -> None:
        if min(self.width, self.height, self.frames, self.fps) < 1:
            raise ValueError("Invalid video repair dimensions or clocks")
        if self.width * self.height > 16_000_000 or self.fps > 120:
            raise ValueError("Video repair budget exceeded")
        if self.source.resolve() == self.output.resolve():
            raise ValueError("Cannot replace source")


@dataclass(frozen=True, slots=True)
class VideoRepairReceipt:
    frames: int
    seconds: float
    softened_frames: int
    attempts: int
    quality_measured: bool = False


def spatial_evidence(source: Image, result: Image, support: Image) -> RepairEvidence:
    """Only establishes unmasked-pixel integrity, not OCR or temporal quality."""
    return RepairEvidence(bool(np.array_equal(source[support == 0], result[support == 0])))


def render_repaired_video(job: VideoRepairJob) -> VideoRepairReceipt:
    """Pipe one prepared video part through repair to an H.264 video-only file.

    Caller must provide a CFR input with exactly the expected frame count, then
    independently decode/probe output before reuse. Files from failed attempts
    are preserved, never automatically reused or overwritten.
    """
    if not job.source.is_file() or not job.output.parent.is_dir():
        raise ValueError("Missing source or output directory")
    if job.output.exists():
        raise FileExistsError(job.output)
    decode_command = [
        "ffmpeg",
        "-v",
        "error",
        "-xerror",
        "-nostdin",
        "-i",
        str(job.source),
        "-map",
        "0:v:0",
        "-an",
        "-fps_mode",
        "passthrough",
        "-pix_fmt",
        "bgr24",
        "-f",
        "rawvideo",
        "pipe:1",
    ]
    if job.allow_padded_tail:
        decode_command[-1:-1] = ["-frames:v", str(job.frames)]
    encode_command = [
        "ffmpeg",
        "-v",
        "error",
        "-xerror",
        "-nostdin",
        "-n",
        "-f",
        "rawvideo",
        "-pixel_format",
        "bgr24",
        "-video_size",
        f"{job.width}x{job.height}",
        "-framerate",
        str(job.fps),
        "-i",
        "pipe:0",
        "-an",
        *ENCODER_OPTIONS[job.hardware_encode],
        "-pix_fmt",
        "yuv420p",
        "-bf",
        "0",
        "-video_track_timescale",
        "15360",
        "-profile:v",
        "high",
        "-color_primaries",
        "bt709",
        "-color_trc",
        "bt709",
        "-colorspace",
        "bt709",
        "-movflags",
        "+faststart",
        str(job.output),
    ]
    counts = [0, 0]

    def receipt(index: int, span: SubtitleSpan, result: RepairedFrame) -> None:
        counts[0] += int(result.attempts[-1].decision.cosmetic_fallback)
        counts[1] += len(result.attempts)

    started = monotonic()
    with (
        Popen(decode_command, stdout=PIPE) as decoder,
        Popen(encode_command, stdin=PIPE) as encoder,
    ):
        if decoder.stdout is None or encoder.stdin is None:
            raise RuntimeError("Missing FFmpeg pipeline streams")
        reader, writer = decoder.stdout, encoder.stdin

        def decoded_frames() -> Iterator[Image]:
            size = job.width * job.height * 3
            while data := reader.read(size):
                if len(data) != size:
                    raise ValueError("Truncated decoded frame")
                yield np.frombuffer(data, dtype=np.uint8).reshape(job.height, job.width, 3)

        def write(frame: Image) -> None:
            writer.write(frame.tobytes())

        completed = False
        try:
            count = repair_stream(
                decoded_frames(),
                spans=job.spans,
                expected_frames=job.frames,
                measure=spatial_evidence,
                quality_available=False,
                write=write,
                receipt=receipt,
            )
            completed = True
        finally:
            writer.close()
            reader.close()
            if not completed and decoder.poll() is None:
                decoder.terminate()
        if decoder.wait(timeout=30) != 0 or encoder.wait(timeout=30) != 0:
            raise RuntimeError("FFmpeg repair pipeline failed")
    return VideoRepairReceipt(count, monotonic() - started, counts[0], counts[1])
