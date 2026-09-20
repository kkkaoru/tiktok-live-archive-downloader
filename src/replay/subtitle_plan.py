"""Validate native mask JSON and preserve strict half-open frame membership."""

from math import isfinite

from replay.subtitle_stream import SubtitleSpan


def number(value: object) -> float:
    if isinstance(value, bool) or not isinstance(value, (float, int)) or not isfinite(value):
        raise ValueError("Expected finite mask number")
    return float(value)


def repair_spans(
    masks: object, *, frames: int, width: int, height: int, fps: int = 30
) -> tuple[SubtitleSpan, ...]:
    """Merge concurrent ROIs and coalesce identical consecutive frame regions.

    Membership uses the native comparison directly, avoiding an independent
    rounding rule. New captions must be independently checked for intersection.
    """
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
        active = [box for start, end, box in parsed if start <= frame / fps < end]
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
