"""Windowed, thread-parallel scheduling of the subtitle repair render; output identical.

The decode and encode commands, per-frame repair functions, softness ramp, receipt order and
bookkeeping are those of ``replay.subtitle_video.render_repaired_video``. Only the scheduling
differs: frames are read in windows, text masks are computed for a window in detector-sized
chunks, per-frame repairs run on a thread pool, and results are written, receipted and reported
strictly in frame order. Per-frame repair (``repair_with_retries``) shares no mutable state, so
threads cannot change the pixels. A caller-supplied quality measure must therefore be thread
safe; the default spatial measure is a pure function. Temporal-context (registered) repair is
not supported here; use the original renderer for it.
"""

import re
from collections import deque
from collections.abc import Callable, Iterable, Iterator
from concurrent.futures import Future, ThreadPoolExecutor
from contextlib import nullcontext
from dataclasses import dataclass
from itertools import islice
from subprocess import PIPE, Popen
from time import monotonic

import numpy as np

from replay.subtitle_inpaint import Image, ramp_softness
from replay.subtitle_repair import QualityMeasure, RepairedFrame, TextDetector, repair_with_retries
from replay.subtitle_stream import BatchDetector, EvidenceSink, ReceiptSink, SubtitleSpan
from replay.subtitle_video import (
    ENCODER_OPTIONS,
    VideoRepairJob,
    VideoRepairReceipt,
    spatial_evidence,
)

DEFAULT_WORKERS = 4
DEFAULT_WINDOW = 32
DEFAULT_DETECT_CHUNK = 8
MAX_WORKERS = 32
MAX_WINDOW = 128
MAX_WINDOW_BYTES = 512 * 1024 * 1024
DEFAULT_SOFTNESS = 0.45


@dataclass(frozen=True, slots=True)
class Schedule:
    """Parallelism knobs; none of them can change the repaired pixels."""

    workers: int = DEFAULT_WORKERS
    window: int = DEFAULT_WINDOW
    detect_chunk: int = DEFAULT_DETECT_CHUNK

    def __post_init__(self) -> None:
        if not 1 <= self.workers <= MAX_WORKERS:
            raise ValueError("Invalid repair worker count")
        if not 1 <= self.window <= MAX_WINDOW:
            raise ValueError("Invalid repair window")
        if not 1 <= self.detect_chunk <= self.window:
            raise ValueError("Invalid detector chunk")


@dataclass(frozen=True, slots=True)
class _Pending:
    first: int
    frames: tuple[Image, ...]
    spans: tuple[SubtitleSpan | None, ...]
    glyphs: tuple[Image | None, ...]
    futures: tuple[Future[RepairedFrame] | None, ...]


@dataclass(frozen=True, slots=True)
class _Staged:
    first: int
    window: tuple[Image, ...]
    selected: tuple[SubtitleSpan | None, ...]
    rois: tuple[Image, ...]
    masks: Future[tuple[Image, ...]]


def _span_at(index: int, spans: tuple[SubtitleSpan, ...]) -> SubtitleSpan | None:
    for span in spans:
        if span.start_frame <= index < span.end_frame:
            return span
    return None


def _roi(frame: Image, span: SubtitleSpan) -> Image:
    if frame.ndim != 3 or frame.shape[2] != 3:
        raise ValueError("Expected BGR frame")
    if span.x + span.width > frame.shape[1] or span.y + span.height > frame.shape[0]:
        raise ValueError("Subtitle span exceeds decoded canvas")
    roi = frame[span.y : span.y + span.height, span.x : span.x + span.width].view()
    roi.flags.writeable = False
    return roi


def _validate_spans(spans: tuple[SubtitleSpan, ...], expected_frames: int) -> None:
    previous_end = 0
    for planned in spans:
        if planned.start_frame < previous_end or planned.end_frame > expected_frames:
            raise ValueError("Overlapping, unordered or out-of-range subtitle spans")
        previous_end = planned.end_frame


def _masks_for(
    rois: tuple[Image, ...],
    *,
    detect: TextDetector | None,
    detect_batch: BatchDetector | None,
    chunk: int,
) -> tuple[Image, ...]:
    if detect is not None:
        return tuple(detect(roi) for roi in rois)
    if detect_batch is None:
        raise ValueError("A text detector is required")
    found: list[Image] = []
    for start in range(0, len(rois), chunk):
        found.extend(detect_batch(rois[start : start + chunk]))
    return tuple(found)


def fast_repair_stream(
    frames: Iterable[Image],
    *,
    spans: tuple[SubtitleSpan, ...],
    expected_frames: int,
    measure: QualityMeasure,
    write: Callable[[Image], None],
    receipt: ReceiptSink,
    detect: TextDetector | None = None,
    detect_batch: BatchDetector | None = None,
    evidence: EvidenceSink | None = None,
    softness: float = DEFAULT_SOFTNESS,
    quality_available: bool = True,
    ramp_start_frame: int = 0,
    ramp_total_frames: int | None = None,
    schedule: Schedule | None = None,
) -> int:
    """Same contract and results as ``subtitle_stream.repair_stream`` without temporal context."""
    plan = Schedule() if schedule is None else schedule
    if detect is not None and detect_batch is not None:
        raise ValueError("Select either a single or batch text detector")
    _validate_spans(spans, expected_frames)
    ramp_softness(
        frame=0,
        frames=expected_frames,
        target=softness,
        start_frame=ramp_start_frame,
        total_frames=ramp_total_frames,
    )
    iterator = iter(frames)
    pending: deque[_Pending] = deque()
    count = 0

    def repair(frame: Image, span: SubtitleSpan, glyph: Image, index: int) -> RepairedFrame:
        return repair_with_retries(
            _roi(frame, span),
            measure=measure,
            quality_available=quality_available,
            detect=lambda _roi_view: glyph,
            softness=ramp_softness(
                frame=index,
                frames=expected_frames,
                target=softness,
                start_frame=ramp_start_frame,
                total_frames=ramp_total_frames,
            ),
        )

    def drain() -> None:
        item = pending.popleft()
        for offset, frame in enumerate(item.frames):
            output = frame.copy()
            span, glyph, future = item.spans[offset], item.glyphs[offset], item.futures[offset]
            index = item.first + offset
            result: RepairedFrame | None = None
            if span is not None and future is not None:
                result = future.result()
                output[span.y : span.y + span.height, span.x : span.x + span.width] = result.image
                receipt(index, span, result)
            write(output)
            if evidence is not None and span and result is not None and glyph is not None:
                evidence(index, span, _roi(frame, span), glyph, result)

    def submit_repairs(staged: _Staged, pool: ThreadPoolExecutor) -> None:
        masks = staged.masks.result()
        if len(masks) != len(staged.rois):
            raise ValueError("Detector mask count differs from the active ROIs")
        mask_iterator = iter(masks)
        glyphs: list[Image | None] = []
        futures: list[Future[RepairedFrame] | None] = []
        for offset, (frame, span) in enumerate(zip(staged.window, staged.selected, strict=True)):
            if span is None:
                glyphs.append(None)
                futures.append(None)
                continue
            glyph = next(mask_iterator)
            glyphs.append(glyph)
            futures.append(pool.submit(repair, frame, span, glyph, staged.first + offset))
        pending.append(
            _Pending(staged.first, staged.window, staged.selected, tuple(glyphs), tuple(futures))
        )

    # One dedicated detector thread keeps detector calls sequential while the caller reads,
    # drains and writes frames; repairs run on their own pool.
    with (
        ThreadPoolExecutor(max_workers=1) as detector_pool,
        ThreadPoolExecutor(max_workers=plan.workers) as pool,
    ):
        staged: _Staged | None = None
        while window := tuple(islice(iterator, plan.window)):
            if count + len(window) > expected_frames:
                raise ValueError("Decoder produced excess frames")
            first = window[0]
            if any(frame.shape != first.shape or frame.dtype != first.dtype for frame in window):
                raise ValueError("Frame geometry changed within the repair batch")
            if first.nbytes * plan.window > MAX_WINDOW_BYTES:
                raise ValueError("Repair window exceeds its memory budget")
            selected = tuple(_span_at(count + offset, spans) for offset in range(len(window)))
            rois = tuple(
                _roi(frame, span) for frame, span in zip(window, selected, strict=True) if span
            )
            masks = detector_pool.submit(
                _masks_for, rois, detect=detect, detect_batch=detect_batch, chunk=plan.detect_chunk
            )
            if staged is not None:
                submit_repairs(staged, pool)
            staged = _Staged(count, window, selected, rois, masks)
            count += len(window)
            while len(pending) > 1:
                drain()
        if staged is not None:
            submit_repairs(staged, pool)
        while pending:
            drain()
    if count != expected_frames:
        raise ValueError("Decoder ended before expected frame count")
    return count


def _decode_command(job: VideoRepairJob) -> list[str]:
    input_options: tuple[str, ...] = ()
    window_options: tuple[str, ...] = ()
    if job.source_clock is not None:
        clock = job.source_clock
        seek_seconds = max(0, int(clock.pts[0] * clock.time_base) - 1)
        input_options = ("-copyts", "-noaccurate_seek", "-ss", str(seek_seconds))
        window_options = (
            "-vf",
            f"settb={clock.time_base},select=gte(pts\\,{clock.pts[0]}),"
            f"trim=end_frame={job.frames},showinfo",
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
    return [
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


def _encode_command(job: VideoRepairJob) -> list[str]:
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
    return [
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


def render_repaired_video_fast(
    job: VideoRepairJob,
    *,
    measure: QualityMeasure | None = None,
    detect: TextDetector | None = None,
    detect_batch: BatchDetector | None = None,
    evidence: EvidenceSink | None = None,
    schedule: Schedule | None = None,
) -> VideoRepairReceipt:
    """Drop-in for ``render_repaired_video`` (no ``require_quality``, no temporal context)."""
    if job.context_frames:
        raise ValueError("Temporal-context repair uses the original renderer")
    if not job.source.is_file() or not job.output.parent.is_dir():
        raise ValueError("Missing source or output directory")
    if job.output.exists():
        raise FileExistsError(job.output)
    decode_command = _decode_command(job)
    encode_command = _encode_command(job)
    counts = [0, 0]
    totals = [0, 0, 0, 0]  # measured frames, repaired frames, supported px, unsupported px

    def receipt(_index: int, _span: SubtitleSpan, result: RepairedFrame) -> None:
        if result.unsupported is not None:
            totals[2] += int(np.count_nonzero(result.support))
            totals[3] += int(np.count_nonzero(result.unsupported))
            totals[1] += 1
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
        totals[0] += int(measured)
        totals[1] += 1
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
            count = fast_repair_stream(
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
                schedule=schedule,
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
    quality_measured = totals[1] > 0 and totals[0] == totals[1]
    return VideoRepairReceipt(
        count,
        monotonic() - started,
        counts[0],
        counts[1],
        quality_measured=quality_measured,
        quality_passed=quality_measured and counts[0] == 0,
        supported_pixels=totals[2],
        unsupported_pixels=totals[3],
    )
