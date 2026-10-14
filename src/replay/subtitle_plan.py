"""Build sampled source masks and validate exact half-open repair membership."""

from collections import defaultdict
from dataclasses import dataclass
from itertools import chain
from math import isfinite

from replay.caption_io import FrameObservation
from replay.caption_masks import MaskPolicy, TextDetection, TextRectangle, subtitle_rectangles
from replay.subtitle_clock import VideoFrameClock
from replay.subtitle_profile import SubtitleBands, profiled_detections
from replay.subtitle_stream import SubtitleSpan


@dataclass(frozen=True, slots=True)
class ObservedMask:
    """A sampled nearest-observation cell, not exhaustive removal approval."""

    start: float
    end: float
    rectangle: TextRectangle


def number(value: object) -> float:
    if isinstance(value, bool) or not isinstance(value, (float, int)) or not isfinite(value):
        raise ValueError("Expected finite mask number")
    return float(value)


def repair_spans(
    masks: object,
    *,
    frames: int,
    width: int,
    height: int,
    fps: int = 30,
    source_clock: VideoFrameClock | None = None,
) -> tuple[SubtitleSpan, ...]:
    """Merge concurrent ROIs and coalesce identical consecutive frame regions.

    Membership uses the native comparison directly, avoiding an independent
    rounding rule. With source_clock, mask times are relative to its first PTS;
    actual presentation clocks, never average fps, determine frame membership.
    New captions must be independently checked for intersection.
    """
    if source_clock is not None and len(source_clock.pts) != frames:
        raise ValueError("Mask frame clock length mismatch")
    if not 1 <= frames <= 36_000 or min(width, height, fps) < 1:
        raise ValueError("Invalid part dimensions or frame budget")
    if not isinstance(masks, list) or len(masks) > 10_000:
        raise ValueError("Invalid mask list")
    parsed: list[tuple[float, float, tuple[int, int, int, int]]] = []
    for mask in masks:
        if not isinstance(mask, dict) or not isinstance(mask.get("region"), dict):
            raise ValueError("Invalid mask object")
        region = mask["region"]
        start, end = number(mask.get("startSeconds")), number(mask.get("endSeconds"))
        values = tuple(number(region.get(key)) for key in ("x", "y", "width", "height"))
        if any(value != int(value) for value in values):
            raise ValueError("Nonintegral pixel region")
        x, y, w, h = (int(value) for value in values)
        if start < 0 or end <= start or x + w > width or y + h > height:
            raise ValueError("Mask exceeds part bounds")
        SubtitleSpan(0, 1, x, y, w, h)
        parsed.append((start, end, (x, y, w, h)))
    spans: list[SubtitleSpan] = []
    for frame in range(frames):
        time = (
            frame / fps
            if source_clock is None
            else float((source_clock.pts[frame] - source_clock.pts[0]) * source_clock.time_base)
        )
        active = [box for start, end, box in parsed if start <= time < end]
        if not active:
            continue
        x, y = min(box[0] for box in active), min(box[1] for box in active)
        w = max(box[0] + box[2] for box in active) - x
        h = max(box[1] + box[3] for box in active) - y
        if (
            spans
            and spans[-1].end_frame == frame
            and (spans[-1].x, spans[-1].y, spans[-1].width, spans[-1].height) == (x, y, w, h)
        ):
            previous = spans.pop()
            spans.append(SubtitleSpan(previous.start_frame, frame + 1, x, y, w, h))
        else:
            spans.append(SubtitleSpan(frame, frame + 1, x, y, w, h))
    return tuple(spans)


def observed_masks(
    observations: tuple[FrameObservation, ...],
    *,
    duration: float,
    policy: MaskPolicy,
    profile: SubtitleBands,
    maximum_distance: float = 1,
) -> tuple[ObservedMask, ...]:
    """Build source-specific masks from explicitly selected, validated observations.

    Callers bind observation files and the explicitly selected source profile to
    verified evidence. Refinement never silently relearns bands from unrelated
    newly visible UI/text. Out-of-profile text remains unclassified, not absent.
    Actual decoded times determine cells, including observed-empty cells. Gaps
    beyond the distance bound stay unmasked; sparse sampling is never exhaustive.
    Geometry includes the immediately adjacent observed cells when their time
    supports touch. This guards OCR box jitter and held-frame clock differences;
    it never activates an observed-empty cell or bridges an unobserved gap.
    """
    if not 0 < number(duration) <= 86400 or not 0 < number(maximum_distance) <= 1:
        raise ValueError("Invalid sampled mask duration or uncertainty bound")
    if not observations or len(observations) > 50000:
        raise ValueError("Invalid mask observation budget")
    width, height = observations[0].width, observations[0].height
    unique: dict[float, list[TextDetection]] = defaultdict(list)
    for frame in observations:
        if (frame.width, frame.height) != (width, height):
            raise ValueError("Inconsistent observed mask canvas")
        if (
            not 0 <= number(frame.actual_time) < duration
            or not 0 <= number(frame.requested_time) < duration
        ):
            raise ValueError("Mask observation clock exceeds source")
        unique[frame.actual_time].extend(frame.detections)
    frames = tuple(
        FrameObservation(time, time, width, height, tuple(dict.fromkeys(lines)))
        for time, lines in sorted(unique.items())
    )
    frame_boxes = tuple(
        subtitle_rectangles(
            profiled_detections(frame, profile), width=width, height=height, policy=policy
        )
        for frame in frames
    )
    masks: list[ObservedMask] = []
    for index, frame in enumerate(frames):
        if not frame_boxes[index]:
            continue
        time = frame.actual_time
        neighbours = range(max(0, index - 1), min(len(frames), index + 2))
        boxes = tuple(
            chain.from_iterable(
                frame_boxes[neighbour]
                for neighbour in neighbours
                if abs(frames[neighbour].actual_time - time) <= 2 * maximum_distance
            )
        )
        start = max(
            0, time - maximum_distance, (frames[index - 1].actual_time + time) / 2 if index else 0
        )
        end = min(
            duration,
            time + maximum_distance,
            (frames[index + 1].actual_time + time) / 2 if index + 1 < len(frames) else duration,
        )
        x, y = min(box.x for box in boxes), min(box.y for box in boxes)
        right, bottom = (
            max(box.x + box.width for box in boxes),
            max(box.y + box.height for box in boxes),
        )
        rectangle = TextRectangle(x, y, right - x, bottom - y)
        if masks and masks[-1].rectangle == rectangle and start <= masks[-1].end:
            previous = masks.pop()
            start = previous.start
        masks.append(ObservedMask(start, end, rectangle))
    return tuple(masks)
