"""Admission of repair search regions from sampled OCR plus detector regions.

Only the sampled frames carry OCR. Their per-frame regions come from the existing
``frame_region`` rule, are bracketed over unsampled frames, and are united with the
temporally widened detector regions. Frames without any region stay unknown and
unmodified. The result lists the sampled frames and the fill rule; it is a candidate
search region, never a raw OCR reading, quality approval or text-absence proof.
"""

from collections.abc import Mapping, Sequence
from math import ceil, floor

from replay.caption_io import FrameObservation
from replay.caption_masks import MaskPolicy
from replay.subtitle_cleanup_plan import CaptionSearchProfile, frame_region
from replay.subtitle_clock import VideoFrameClock
from replay.subtitle_plan import repair_spans
from replay.subtitle_region_fill import Region, bracket, combine, union, widen
from replay.subtitle_sampled_observations import receipt

POLICY = "sampled-ocr-plus-detector-regions-v5"  # v5: regions stay within the caption zone
LINE_PAD = 6  # recognised boxes are tight; glyph top/bottom rows can lie a few pixels outside
MAX_REGION_PIXELS = 2_000_000
MIN_REGION_SIDE = 8


def _box(region: dict[str, int]) -> Region:
    return (region["x"], region["y"], region["x"] + region["width"], region["y"] + region["height"])


def _sample_regions(
    sampled: Mapping[int, FrameObservation],
    clock: VideoFrameClock,
    profile: CaptionSearchProfile,
    policy: MaskPolicy,
) -> dict[int, Region | None]:
    regions: dict[int, Region | None] = {}
    for index, frame in sampled.items():
        pts = clock.pts[index]
        if (
            frame.requested_time != float(pts * clock.time_base)
            or frame.actual_time != frame.requested_time
        ):
            raise ValueError("Sampled OCR frame clock differs from the actual source PTS")
        candidate = frame_region(frame, profile, policy=policy)
        regions[index] = None if candidate is None else _box(candidate[0])
    return regions


def _line_boxes(
    sampled: Mapping[int, FrameObservation], index: int, zone: Region, radius: int
) -> list[list[int]]:
    """Recognised line boxes of sampled frames within ``radius`` that touch the caption zone.

    Only horizontal boxes count: caption lines are wider than tall, while rotated or vertical
    interface text read inside the zone would stretch the region over unrelated screen content.
    Captions fade in and out, so a frame's own recognition can miss lines that its neighbours
    read clearly; the union over the temporal neighbourhood keeps those lines in the colour key.
    """
    found: list[list[int]] = []
    for source in sorted(i for i in sampled if abs(i - index) <= radius):
        for line in sampled[source].detections:
            box = line.rectangle
            left, top = floor(box.x), floor(box.y)
            right, bottom = ceil(box.x + box.width), ceil(box.y + box.height)
            touches = left < zone[2] and right > zone[0] and top < zone[3] and bottom > zone[1]
            horizontal = right - left >= bottom - top
            if (
                line.text.strip()
                and touches
                and horizontal
                and [left, top, right, bottom] not in found
            ):
                found.append([left, top, right, bottom])
    return found


def _line_reach(lines: Sequence[Sequence[int]]) -> list[Region]:
    """The horizontal lines' bounding box grown by one typical line height up and down.

    A scrolling caption line is read only where it overlaps the OCR zone; its remaining glyph
    rows lie up to one line height outside the boxes, so the search region must include them.
    Only boxes wider than tall count (caption lines are horizontal; rotated or vertical UI text
    read inside the zone must not stretch the region) and the line height is their median.
    """
    wide = [b for b in lines if b[2] - b[0] >= b[3] - b[1]]
    if not wide:
        return []
    heights = sorted(b[3] - b[1] for b in wide)
    line_height = heights[len(heights) // 2]
    left, right = min(b[0] for b in wide), max(b[2] for b in wide)
    top, bottom = min(b[1] for b in wide), max(b[3] for b in wide)
    return [(left, top - line_height, right, bottom + line_height)]


def _within_zone(region: Region, zone: Region) -> Region:
    """Clip a region vertically to the measured caption zone grown by half its height each way.

    The detector's polygons can include interface text behind the caption and stretch a region
    over hundreds of rows. The zone is the measured extent of the caption, so rows farther than
    half of it away (about one line) from it are not caption rows; a residual there would be
    found by the post-repair audit instead of being repaired blindly.
    """
    grow = (zone[3] - zone[1]) // 2
    return (region[0], max(region[1], zone[1] - grow), region[2], min(region[3], zone[3] + grow))


def _checked(region: Region, width: int, height: int) -> Region:
    clipped = (max(0, region[0]), max(0, region[1]), min(width, region[2]), min(height, region[3]))
    side = (clipped[2] - clipped[0], clipped[3] - clipped[1])
    if min(side) < MIN_REGION_SIDE or side[0] * side[1] > MAX_REGION_PIXELS:
        raise ValueError("Filled search region violates the detector budget")
    return clipped


def admission(
    *,
    sampled: Mapping[int, FrameObservation],
    clock: VideoFrameClock,
    profile: CaptionSearchProfile,
    policy: MaskPolicy,
    stride: int,
    triggers: Sequence[int],
    detected: Sequence[Region | None],
    radius: int,
) -> dict[str, object]:
    """Build candidate regions for every frame from the sampled OCR and detector regions."""
    count = len(clock.pts)
    if len(detected) != count:
        raise ValueError("Detector regions do not cover the requested source chunk")
    regions = _sample_regions(sampled, clock, profile, policy)
    filled = combine(bracket(count, regions), widen(detected, radius))
    width, height = profile.bands.width, profile.bands.height
    terminal = clock.pts[-1] + clock.last_duration
    envelope = profile.measured_envelope
    zone: Region = (envelope[0], envelope[1], envelope[0] + envelope[2], envelope[1] + envelope[3])
    masks: list[dict[str, object]] = []
    entries: list[dict[str, object]] = []
    unknown: list[int] = []
    for index, (region, pts) in enumerate(zip(filled, clock.pts, strict=True)):
        entry: dict[str, object] = {"sourcePTS": pts, "observation": index}
        if region is None:
            entry["classification"] = "unknown"
            unknown.append(pts)
        else:
            lines = _line_boxes(sampled, index, zone, radius)
            padded = [
                (a - LINE_PAD, b - LINE_PAD, c + LINE_PAD, d + LINE_PAD) for a, b, c, d in lines
            ]
            reach = union([region, *padded, *_line_reach(lines)])
            x0, y0, x1, y1 = _checked(_within_zone(reach, zone), width, height)
            box = {"x": x0, "y": y0, "width": x1 - x0, "height": y1 - y0}
            next_pts = clock.pts[index + 1] if index + 1 < count else terminal
            masks.append(
                {
                    "startSeconds": float((pts - clock.pts[0]) * clock.time_base),
                    "endSeconds": float((next_pts - clock.pts[0]) * clock.time_base),
                    "region": box,
                }
            )
            entry.update(
                classification="overlay-subtitle",
                region=box,
                sampled=index in sampled,
                lineBoxes=lines,
            )
        entries.append(entry)
    spans = repair_spans(masks, frames=count, width=width, height=height, source_clock=clock)
    if sum(span.end_frame - span.start_frame for span in spans) != count - len(unknown):
        raise ValueError("Candidate mask membership differs from the sampled admission")
    return {
        "version": 1,
        "policy": POLICY,
        "profile": {"envelope": list(profile.measured_envelope)},
        "masks": masks,
        "frames": entries,
        "unknownSourcePTS": unknown,
        "unknownFramesKeptUnmodified": True,
        "sampling": receipt(count, stride, sorted(sampled), triggers),
        "detectorRadius": radius,
        "removalQualityApproved": False,
    }
