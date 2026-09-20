"""Frame-clocked, bounded-memory adapter for prepared subtitle ROIs."""

from collections.abc import Callable, Iterable
from dataclasses import dataclass

from replay.subtitle_inpaint import Image, ramp_softness
from replay.subtitle_repair import QualityMeasure, RepairedFrame, repair_with_retries


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
) -> int:
    """Apply ordered nonoverlapping masks, preserving every input frame.

    Decoder/encoder and source fingerprints remain the caller's responsibility.
    A receipt sink must consume metadata synchronously, not retain frame arrays.
    No frames are dropped, no source array is mutated, and no old blur is used.
    """
    ramp_softness(frame=0, frames=expected_frames, target=softness)
    previous_end = 0
    for span in spans:
        if span.start_frame < previous_end or span.end_frame > expected_frames:
            raise ValueError("Overlapping, unordered or out-of-range subtitle spans")
        previous_end = span.end_frame
    span_index = 0
    count = 0
    for frame_index, frame in enumerate(frames):
        if frame_index >= expected_frames:
            raise ValueError("Decoder produced excess frames")
        while span_index < len(spans) and spans[span_index].end_frame <= frame_index:
            span_index += 1
        if span_index < len(spans) and spans[span_index].start_frame <= frame_index:
            span = spans[span_index]
            if frame.ndim != 3 or frame.shape[2] != 3:
                raise ValueError("Expected BGR frame")
            if span.x + span.width > frame.shape[1] or span.y + span.height > frame.shape[0]:
                raise ValueError("Subtitle span exceeds decoded canvas")
            roi = frame[span.y : span.y + span.height, span.x : span.x + span.width]
            result = repair_with_retries(
                roi,
                measure=measure,
                quality_available=quality_available,
                softness=ramp_softness(frame=frame_index, frames=expected_frames, target=softness),
            )
            output = frame.copy()
            output[span.y : span.y + span.height, span.x : span.x + span.width] = result.image
            receipt(frame_index, span, result)
            write(output)
        else:
            write(frame.copy())
        count += 1
    if count != expected_frames:
        raise ValueError("Decoder ended before expected frame count")
    return count
