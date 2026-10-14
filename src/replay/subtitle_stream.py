"""Frame-clocked, bounded-memory adapter for prepared subtitle ROIs."""

from bisect import bisect_right
from collections import deque
from collections.abc import Callable, Iterable, Iterator
from dataclasses import dataclass
from functools import partial
from itertools import islice

import cv2
import numpy as np

from replay.subtitle_clock import VideoFrameClock
from replay.subtitle_inpaint import Image, ramp_softness
from replay.subtitle_repair import (
    QualityMeasure,
    RepairedFrame,
    TextDetector,
    repair_registered_context,
    repair_with_retries,
)
from replay.subtitle_vision import vision_text_masks


@dataclass(frozen=True, slots=True)
class SubtitleSpan:
    start_frame: int
    end_frame: int
    x: int
    y: int
    width: int
    height: int

    def __post_init__(self) -> None:
        if self.start_frame < 0 or self.end_frame <= self.start_frame:
            raise ValueError("Invalid subtitle frame range")
        if min(self.x, self.y) < 0 or min(self.width, self.height) < 8:
            raise ValueError("Invalid subtitle region")


ReceiptSink = Callable[[int, SubtitleSpan, RepairedFrame], None]
EvidenceSink = Callable[[int, SubtitleSpan, Image, Image, RepairedFrame], None]
BatchDetector = Callable[[tuple[Image, ...]], tuple[Image, ...]]
MAX_BATCH_FRAME_BYTES = 32 * 1024 * 1024


def _frame_batches(frames: Iterable[Image]) -> Iterator[tuple[Image, ...]]:
    iterator = iter(frames)
    for first in iterator:
        size = max(1, min(8, MAX_BATCH_FRAME_BYTES // max(1, first.nbytes)))
        batch = (first, *islice(iterator, size - 1))
        if any(frame.shape != first.shape or frame.dtype != first.dtype for frame in batch):
            raise ValueError("Frame geometry changed within the repair batch")
        yield batch


def _prepared_mask(masks: Iterator[Image], _image: Image) -> Image:
    return next(masks)


def _span_at(
    index: int, *, spans: tuple[SubtitleSpan, ...], ends: tuple[int, ...]
) -> SubtitleSpan | None:
    position = bisect_right(ends, index)
    if position < len(spans) and spans[position].start_frame <= index:
        return spans[position]
    return None


def _roi(frame: Image, span: SubtitleSpan) -> Image:
    if frame.ndim != 3 or frame.shape[2] != 3:
        raise ValueError("Expected BGR frame")
    if span.x + span.width > frame.shape[1] or span.y + span.height > frame.shape[0]:
        raise ValueError("Subtitle span exceeds decoded canvas")
    roi = frame[span.y : span.y + span.height, span.x : span.x + span.width].view()
    roi.flags.writeable = False
    return roi


def _registered_stream(
    frames: Iterable[Image],
    *,
    spans: tuple[SubtitleSpan, ...],
    clock: VideoFrameClock,
    radius: int,
    donors_verified: bool,
    detect: TextDetector | None,
    detect_batch: BatchDetector | None,
    write: Callable[[Image], None],
    receipt: ReceiptSink,
    evidence: EvidenceSink | None,
) -> int:
    """Bounded lookahead retains original pixels and source PTS, never candidates."""
    # The observation layer also consumes stream plans; import after module initialization.
    from replay.subtitle_quality_observe import OriginalContext

    span = spans[0]
    ends = tuple(item.end_frame for item in spans)
    queue: deque[tuple[int, Image, OriginalContext, Image]] = deque()
    iterator = iter(frames)
    decoded = 0
    for index in range(len(clock.pts)):
        while decoded < min(len(clock.pts), index + radius + 1):
            frame = next(iterator, None)
            if frame is None:
                raise ValueError("Decoder ended before expected frame count")
            if frame.nbytes * (2 * radius + 1) > 128 * 1024 * 1024:
                raise ValueError("Temporal context buffer exceeds 128MiB")
            roi = _roi(frame, span)
            active = _span_at(decoded, spans=spans, ends=ends)
            if active is None:
                glyph = np.full(roi.shape[:2], 255, dtype=np.uint8)
            elif detect is not None:
                glyph = detect(roi)
            else:
                masks = (vision_text_masks if detect_batch is None else detect_batch)((roi,))
                if len(masks) != 1:
                    raise ValueError("Detector mask count differs from the active ROIs")
                glyph = masks[0]
            if (
                glyph.dtype != np.uint8
                or glyph.shape != roi.shape[:2]
                or not np.all((glyph == 0) | (glyph == 255))
            ):
                raise ValueError("Invalid temporal glyph support")
            # Include antialias/edge support before fitting geometry or sampling donors.
            excluded = cv2.dilate(glyph, np.ones((5, 5), np.uint8)) != 0
            context = OriginalContext(roi.copy(), excluded, clock.pts[decoded] * clock.time_base)
            queue.append((decoded, frame.copy(), context, glyph.copy()))
            decoded += 1
        while queue[0][0] < index - radius:
            queue.popleft()
        _, frame, context, glyph = next(row for row in queue if row[0] == index)
        active = _span_at(index, spans=spans, ends=ends)
        if active is None:
            write(frame.copy())
            continue
        donors = tuple(row[2] for row in queue if row[0] != index) if donors_verified else ()
        result = repair_registered_context(context, donors=donors)
        output = frame.copy()
        output[span.y : span.y + span.height, span.x : span.x + span.width] = result.image
        receipt(index, active, result)
        write(output)
        if evidence is not None:
            evidence(index, active, context.image, glyph, result)
    if next(iterator, None) is not None:
        raise ValueError("Decoder produced excess frames")
    return decoded


def repair_stream(
    frames: Iterable[Image],
    *,
    spans: tuple[SubtitleSpan, ...],
    expected_frames: int,
    measure: QualityMeasure,
    write: Callable[[Image], None],
    receipt: ReceiptSink,
    softness: float = 0.45,
    quality_available: bool = True,
    ramp_start_frame: int = 0,
    ramp_total_frames: int | None = None,
    detect: TextDetector | None = None,
    detect_batch: BatchDetector | None = None,
    evidence: EvidenceSink | None = None,
    source_clock: VideoFrameClock | None = None,
    context_frames: int = 0,
    context_donors_verified: bool = True,
) -> int:
    """Apply ordered nonoverlapping masks, preserving every input frame.

    Decoder/encoder and source fingerprints remain the caller's responsibility.
    A receipt sink must consume metadata synchronously, not retain frame arrays.
    No frames are dropped, no source array is mutated, and no old blur is used.
    """
    ramp_softness(
        frame=0,
        frames=expected_frames,
        target=softness,
        start_frame=ramp_start_frame,
        total_frames=ramp_total_frames,
    )
    previous_end = 0
    for planned in spans:
        if planned.start_frame < previous_end or planned.end_frame > expected_frames:
            raise ValueError("Overlapping, unordered or out-of-range subtitle spans")
        previous_end = planned.end_frame
    if detect is not None and detect_batch is not None:
        raise ValueError("Select either a single or batch text detector")
    if context_frames:
        if (
            source_clock is None
            or len(source_clock.pts) != expected_frames
            or type(context_frames) is not int
            or not 1 <= context_frames <= 8
            or not spans
            or any(
                (span.x, span.y, span.width, span.height)
                != (spans[0].x, spans[0].y, spans[0].width, spans[0].height)
                for span in spans
            )
        ):
            raise ValueError("Temporal repair requires a common ROI and an exact source clock")
        return _registered_stream(
            frames,
            spans=spans,
            clock=source_clock,
            radius=context_frames,
            donors_verified=context_donors_verified,
            detect=detect,
            detect_batch=detect_batch,
            write=write,
            receipt=receipt,
            evidence=evidence,
        )
    ends = tuple(span.end_frame for span in spans)
    count = 0
    for batch in _frame_batches(frames):
        if count + len(batch) > expected_frames:
            raise ValueError("Decoder produced excess frames")
        selected = tuple(
            _span_at(index, spans=spans, ends=ends) for index in range(count, count + len(batch))
        )
        rois = tuple(
            _roi(frame, span)
            for frame, span in zip(batch, selected, strict=True)
            if span is not None
        )
        masks: tuple[Image, ...] = ()
        if rois:
            masks = (
                (vision_text_masks if detect_batch is None else detect_batch)(rois)
                if detect is None
                else tuple(detect(roi) for roi in rois)
            )
        if len(masks) != len(rois):
            raise ValueError("Detector mask count differs from the active ROIs")
        mask_iterator = iter(masks)

        for frame, span in zip(batch, selected, strict=True):
            output = frame.copy()
            result: RepairedFrame | None = None
            glyph: Image | None = None
            if span is not None:
                glyph = next(mask_iterator)
                result = repair_with_retries(
                    _roi(frame, span),
                    measure=measure,
                    quality_available=quality_available,
                    detect=partial(_prepared_mask, iter((glyph,))),
                    softness=ramp_softness(
                        frame=count,
                        frames=expected_frames,
                        target=softness,
                        start_frame=ramp_start_frame,
                        total_frames=ramp_total_frames,
                    ),
                )
                output[span.y : span.y + span.height, span.x : span.x + span.width] = result.image
                receipt(count, span, result)
            write(output)
            if (
                evidence is not None
                and span is not None
                and result is not None
                and glyph is not None
            ):
                evidence(count, span, _roi(frame, span), glyph, result)
            count += 1
    if count != expected_frames:
        raise ValueError("Decoder ended before expected frame count")
    return count
