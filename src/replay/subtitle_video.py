"""Streaming FFmpeg repair of CFR parts or normalized windows; video only."""

import re
from collections.abc import Iterator
from contextlib import nullcontext
from dataclasses import dataclass
from fractions import Fraction
from pathlib import Path
from subprocess import PIPE, Popen
from time import monotonic

import numpy as np

from replay.subtitle_clock import VideoFrameClock
from replay.subtitle_inpaint import Image, ramp_softness, unchanged_outside_support
from replay.subtitle_repair import QualityMeasure, RepairedFrame, TextDetector
from replay.subtitle_retry import RepairEvidence
from replay.subtitle_stream import BatchDetector, EvidenceSink, SubtitleSpan, repair_stream

MAX_FRAME_SAMPLES = 50_000
MAX_CFR_FPS = 120
MAX_NORMALIZED_SECONDS = 86_400

ENCODER_OPTIONS: tuple[tuple[str, ...], ...] = (
    ("-c:v", "libx264", "-preset", "fast", "-crf", "18"),
    ("-c:v", "h264_videotoolbox", "-b:v", "8000k"),
    ("-c:v", "hevc_videotoolbox", "-b:v", "8000k", "-tag:v", "hvc1"),
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
    hardware_hevc: bool = False
    normalized_start_frame: int | None = None
    source_clock: VideoFrameClock | None = None
    hevc_quality: int | None = None
    ramp_start_frame: int = 0
    ramp_total_frames: int | None = None
    context_frames: int = 0
    context_donors_verified: bool = True
    decode_log: Path | None = None

    def __post_init__(self) -> None:
        if self.decode_log is not None and (
            self.decode_log != self.decode_log.resolve()
            or not self.decode_log.is_relative_to(Path.cwd().resolve())
            or not self.decode_log.parent.is_dir()
            or self.decode_log.exists()
        ):
            raise ValueError("Explicit decode log must be a new canonical checkout-local file")
        if type(self.context_frames) is not int or not 0 <= self.context_frames <= 8:
            raise ValueError("Invalid temporal context frame radius")
        if self.context_frames and self.source_clock is None:
            raise ValueError("Temporal repair requires source presentation clocks")
        if self.source_clock is not None and (
            not isinstance(self.source_clock, VideoFrameClock)
            or type(self.frames) is not int
            or len(self.source_clock.pts) != self.frames
            or self.normalized_start_frame is not None
            or self.allow_padded_tail
        ):
            raise ValueError("Invalid presentation-clock window or mixed timing modes")
        if self.normalized_start_frame is not None and (
            any(
                type(value) is not int
                for value in (self.normalized_start_frame, self.frames, self.fps)
            )
            or self.normalized_start_frame < 0
            or self.normalized_start_frame + self.frames > self.fps * MAX_NORMALIZED_SECONDS
            or self.allow_padded_tail
        ):
            raise ValueError("Invalid normalized video window")
        if min(self.width, self.height, self.frames, self.fps) < 1:
            raise ValueError("Invalid video repair dimensions or clocks")
        if self.width * self.height > 16_000_000 or self.fps > MAX_CFR_FPS:
            raise ValueError("Video repair budget exceeded")
        if self.hardware_hevc and not self.hardware_encode:
            raise ValueError("HEVC repair requires explicit hardware encoding")
        if self.hevc_quality is not None and (
            not self.hardware_hevc
            or type(self.hevc_quality) is not int
            or not 1 <= self.hevc_quality <= 100
        ):
            raise ValueError("Native quality requires hardware HEVC and an integer from 1 to 100")
        ramp_softness(
            frame=0,
            frames=self.frames,
            target=0,
            start_frame=self.ramp_start_frame,
            total_frames=self.ramp_total_frames,
        )
        if self.source.resolve() == self.output.resolve():
            raise ValueError("Cannot replace source")


@dataclass(frozen=True, slots=True)
class VideoRepairReceipt:
    frames: int
    seconds: float
    softened_frames: int
    attempts: int
    quality_measured: bool = False
    quality_passed: bool = False
    supported_pixels: int = 0
    unsupported_pixels: int = 0


def _frame_sample_budget(*, frames: int, fps: int) -> None:
    if type(frames) is not int or type(fps) is not int:
        raise ValueError("Frame sampling requires integer clocks")
    if not 1 <= frames <= MAX_FRAME_SAMPLES or not 1 <= fps <= MAX_CFR_FPS:
        raise ValueError("Frame sampling budget exceeded")


def frame_observation_times(*, frames: int, fps: int) -> tuple[float, ...]:
    """Sample inside each CFR frame, away from quantized seek boundaries.

    Build local clocks first; add a source offset only for source requests.
    Subtracting a large source offset can move a boundary request backward.
    Actual decoded frame coverage must still be checked independently.
    """
    _frame_sample_budget(frames=frames, fps=fps)
    return VideoFrameClock(tuple(range(frames)), Fraction(1, fps), 1).observation_times()


def observed_frame_indices(
    actual_times: tuple[float, ...], *, frames: int, fps: int
) -> tuple[int, ...]:
    """Return distinct actual CFR frames, not the number of OCR requests.

    Only use this after independently verifying the output's CFR clock. Reject
    off-grid clocks; callers compare the result with every expected frame index.
    """
    _frame_sample_budget(frames=frames, fps=fps)
    return VideoFrameClock(tuple(range(frames)), Fraction(1, fps), 1).observed_indices(actual_times)


def spatial_evidence(source: Image, result: Image, support: Image) -> RepairEvidence:
    """Only establishes unmasked-pixel integrity, not OCR or temporal quality."""
    return RepairEvidence(unchanged_outside_support(source, result, support))


def render_repaired_video(
    job: VideoRepairJob,
    *,
    measure: QualityMeasure | None = None,
    require_quality: bool = False,
    detect: TextDetector | None = None,
    detect_batch: BatchDetector | None = None,
    evidence: EvidenceSink | None = None,
) -> VideoRepairReceipt:
    """Repair a prepared video part to H.264 or explicitly selected hardware HEVC.

    HEVC uses the QuickTime hvc1 tag; unsupported hardware fails without fallback.
    Explicit hevc_quality selects native constant quality instead of the existing
    bitrate default. It is an encoder setting, not measured quality approval;
    callers must independently compare decoded quality and actual output size.

    Default input is a prepared CFR part with exactly the expected frame count.
    Explicit normalized_start_frame instead selects a bounded window on the
    job.fps grid of a verified zero-based source. FFmpeg normalizes VFR once;
    this may repeat/drop source frames, never reinterpret raw frame ordinals as
    CFR clocks. Whole-second seeking retains keyframe preroll, including held
    frames across long gaps, before normalization and exact frame trimming.
    Alternatively source_clock preserves every verified consecutive source PTS
    without fps conversion: exact timestamp selection after keyframe preroll,
    followed by explicit local presentation clocks in the encoder. This mode
    requires a caller-verified source clock vector, including its terminal extent.
    Caller verifies source provenance, zero origin, grid/extent and mask mapping,
    then independently decodes/probes output before reuse. Files from failed attempts
    are preserved, never automatically reused or overwritten. Quality receipts
    cover supplied spans only, not the completeness of subtitle detection.
    Strict mode refuses missing measurements and cosmetic fallback; callers must
    still independently validate the encoded video and source mask coverage.
    """
    if require_quality and job.context_frames:
        raise ValueError("Registered candidates require independent restoration acceptance")
    if require_quality and measure is None:
        raise ValueError("Strict repair requires an independent quality measurement")
    if not job.source.is_file() or not job.output.parent.is_dir():
        raise ValueError("Missing source or output directory")
    if job.output.exists():
        raise FileExistsError(job.output)
    input_options: tuple[str, ...] = ()
    window_options: tuple[str, ...] = ()
    if job.source_clock is not None:
        clock = job.source_clock
        seek_seconds = max(0, int(clock.pts[0] * clock.time_base) - 1)
        input_options = ("-copyts", "-noaccurate_seek", "-ss", str(seek_seconds))
        window_options = (
            "-vf",
            f"settb={clock.time_base},select=gte(pts\\,{clock.pts[0]}),trim=end_frame={job.frames},showinfo",
            "-frames:v",
            str(job.frames),
        )
    elif job.normalized_start_frame is not None:
        seek_seconds = max(0, job.normalized_start_frame // job.fps - 1)
        offset = job.normalized_start_frame - seek_seconds * job.fps
        input_options = ("-noaccurate_seek", "-ss", str(seek_seconds))
        window_options = (
            "-vf",
            f"fps={job.fps}:start_time=0:eof_action=pass,trim=start_frame={offset}",
            "-frames:v",
            str(job.frames),
        )
    elif job.allow_padded_tail:
        window_options = ("-frames:v", str(job.frames))
    decode_command = [
        "ffmpeg",
        "-v",
        "info" if job.source_clock is not None else "error",
        "-xerror",
        "-nostdin",
        *input_options,
        "-i",
        str(job.source),
        "-map",
        "0:v:0",
        "-an",
        "-fps_mode",
        "passthrough",
        "-enc_time_base",
        "filter",
        "-pix_fmt",
        "bgr24",
        "-f",
        "rawvideo",
        *window_options,
        "pipe:1",
    ]
    encoder_options = ENCODER_OPTIONS[2 if job.hardware_hevc else job.hardware_encode]
    if job.hevc_quality is not None:
        encoder_options = (
            "-c:v",
            "hevc_videotoolbox",
            "-b:v",
            "0",
            "-q:v",
            str(job.hevc_quality),
            "-tag:v",
            "hvc1",
        )
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
        *encoder_options,
        "-pix_fmt",
        "yuv420p",
        "-bf",
        "0",
        "-profile:v",
        "main" if job.hardware_hevc else "high",
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
    measured_frames = 0
    repaired_frames = 0
    supported_pixels = unsupported_pixels = 0

    def receipt(index: int, span: SubtitleSpan, result: RepairedFrame) -> None:
        nonlocal measured_frames, repaired_frames, supported_pixels, unsupported_pixels
        if result.unsupported is not None:
            supported_pixels += int(np.count_nonzero(result.support))
            unsupported_pixels += int(np.count_nonzero(result.unsupported))
            repaired_frames += 1
            return
        final = result.attempts[-1]
        measured = all(
            value is not None
            for value in (
                final.evidence.residual_fraction,
                final.evidence.unstable_fraction,
                final.evidence.readable_text,
            )
        )
        if require_quality and (not measured or final.decision.cosmetic_fallback):
            raise ValueError("Repair did not pass the measured quality gate")
        measured_frames += int(measured)
        repaired_frames += 1
        counts[0] += int(final.decision.cosmetic_fallback)
        counts[1] += len(result.attempts)

    started = monotonic()
    decode_log_path = (
        job.decode_log
        if job.decode_log is not None
        else job.output.with_suffix(job.output.suffix + ".decode.log")
    )
    with (
        (
            nullcontext(("-video_track_timescale", "15360"))
            if job.source_clock is None
            else job.source_clock.encoding_options()
        ) as clock_options,
        nullcontext(None) if job.source_clock is None else decode_log_path.open("xb") as decode_log,
        Popen(decode_command, stdout=PIPE, stderr=decode_log) as decoder,
        Popen([*encode_command[:-1], *clock_options, encode_command[-1]], stdin=PIPE) as encoder,
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
                measure=spatial_evidence if measure is None else measure,
                quality_available=measure is not None,
                detect=detect,
                detect_batch=detect_batch,
                write=write,
                receipt=receipt,
                evidence=evidence,
                ramp_start_frame=job.ramp_start_frame,
                ramp_total_frames=job.ramp_total_frames,
                source_clock=job.source_clock,
                context_frames=job.context_frames,
                context_donors_verified=job.context_donors_verified,
            )
            completed = True
        finally:
            writer.close()
            reader.close()
            if not completed and decoder.poll() is None:
                decoder.terminate()
        if decoder.wait(timeout=30) != 0 or encoder.wait(timeout=30) != 0:
            raise RuntimeError("FFmpeg repair pipeline failed")
    if job.source_clock is not None:
        observed = tuple(
            int(value)
            for value in re.findall(
                rb"\bn:\s*\d+\s+pts:\s*(-?\d+)\s",
                decode_log_path.read_bytes(),
            )
        )
        if observed != job.source_clock.pts:
            raise ValueError("Actual decoded PTS differ from the supplied source clock")
    quality_measured = repaired_frames > 0 and measured_frames == repaired_frames
    return VideoRepairReceipt(
        count,
        monotonic() - started,
        counts[0],
        counts[1],
        quality_measured=quality_measured,
        quality_passed=quality_measured and counts[0] == 0,
        supported_pixels=supported_pixels,
        unsupported_pixels=unsupported_pixels,
    )
