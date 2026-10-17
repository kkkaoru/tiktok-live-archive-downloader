"""Admission of repair search regions from per-frame YOLO caption boxes, without planning OCR.

The sampled-OCR admission reads a few frames with the native OCR tool to learn where caption
lines are and brackets the rest. The pinned YOLO detector already reports the caption lines of
every frame, so this admission takes them directly. It keeps the OCR admission's schema and
rules, so the renderer, the OCR audit and the colour-residual gate run unchanged:

- a frame's line boxes are the horizontal caption boxes touching the measured caption zone in
  the frame and its ``radius`` neighbours (captions fade, neighbours keep faint lines keyed);
- the search region is the union of the padded lines and their one-line reach, clipped to the
  zone grown by half its height, and checked against the detector budget;
- frames without any caption box stay unknown and are left unmodified, but they keep the caption
  zone as their audit region and colour-gate box, so a line the detector missed is still judged.

The audit, not this admission, judges the removal; a failed audit escalates to the full-OCR plan.
"""

from collections.abc import Sequence
from math import ceil, floor

from replay.subtitle_cleanup_plan import CaptionSearchProfile
from replay.subtitle_clock import VideoFrameClock
from replay.subtitle_plan import repair_spans
from replay.subtitle_region_fill import Region, union
from replay.subtitle_sampled_admission import LINE_PAD, _checked, _line_reach, _within_zone

POLICY = "yolo-caption-boxes-v1"
KIND = "yolo-caption-boxes"

Corners = tuple[float, float, float, float]


def frame_lines(boxes: Sequence[Corners], zone: Region) -> list[list[int]]:
    """Integer horizontal caption boxes of one frame that touch the caption zone."""
    found: list[list[int]] = []
    for box in boxes:
        left, top, right, bottom = floor(box[0]), floor(box[1]), ceil(box[2]), ceil(box[3])
        touches = left < zone[2] and right > zone[0] and top < zone[3] and bottom > zone[1]
        if touches and right - left >= bottom - top and [left, top, right, bottom] not in found:
            found.append([left, top, right, bottom])
    return found


def neighbourhood_lines(
    per_frame: Sequence[list[list[int]]], index: int, radius: int
) -> list[list[int]]:
    """Distinct line boxes of the frame and its neighbours within ``radius`` frames."""
    found: list[list[int]] = []
    for lines in per_frame[max(0, index - radius) : index + radius + 1]:
        found.extend(line for line in lines if line not in found)
    return found


def admission(
    *,
    boxes: Sequence[Sequence[Corners]],
    clock: VideoFrameClock,
    profile: CaptionSearchProfile,
    radius: int,
) -> dict[str, object]:
    """Candidate regions and colour-key line boxes for every chunk frame from YOLO boxes."""
    count = len(clock.pts)
    if len(boxes) != count or radius < 0:
        raise ValueError("YOLO boxes do not cover the requested source chunk")
    width, height = profile.bands.width, profile.bands.height
    envelope = profile.measured_envelope
    zone: Region = (envelope[0], envelope[1], envelope[0] + envelope[2], envelope[1] + envelope[3])
    per_frame = [frame_lines(frame, zone) for frame in boxes]
    grow = (zone[3] - zone[1]) // 2
    audit = _checked((zone[0], zone[1] - grow, zone[2], zone[3] + grow), width, height)
    audit_box = {
        "x": audit[0],
        "y": audit[1],
        "width": audit[2] - audit[0],
        "height": audit[3] - audit[1],
    }
    terminal = clock.pts[-1] + clock.last_duration
    masks: list[dict[str, object]] = []
    entries: list[dict[str, object]] = []
    unknown: list[int] = []
    for index, pts in enumerate(clock.pts):
        entry: dict[str, object] = {"sourcePTS": pts, "observation": index}
        lines = neighbourhood_lines(per_frame, index, radius)
        if not lines:
            # Left unmodified, but still judged: the audit reads OCR text and the colour gate
            # unchanged caption pixels inside the caption zone, so a line YOLO missed fails the
            # attempt and escalates to the full-OCR plan instead of passing unseen.
            entry.update(
                classification="unknown",
                region=audit_box,
                lineBoxes=[[audit[0], audit[1], audit[2], audit[3]]],
            )
            unknown.append(pts)
            entries.append(entry)
            continue
        padded = [(a - LINE_PAD, b - LINE_PAD, c + LINE_PAD, d + LINE_PAD) for a, b, c, d in lines]
        reach = union([*padded, *_line_reach(lines)])
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
        entry.update(classification="overlay-subtitle", region=box, sampled=False, lineBoxes=lines)
        entries.append(entry)
    spans = repair_spans(masks, frames=count, width=width, height=height, source_clock=clock)
    if sum(span.end_frame - span.start_frame for span in spans) != count - len(unknown):
        raise ValueError("Candidate mask membership differs from the YOLO admission")
    return {
        "version": 1,
        "policy": POLICY,
        "profile": {"envelope": list(envelope)},
        "masks": masks,
        "frames": entries,
        "unknownSourcePTS": unknown,
        "unknownFramesKeptUnmodified": True,
        "sampling": {
            "version": 1,
            "kind": KIND,
            "rawOcr": False,
            "interpolated": False,
            "frames": count,
            "detectedFrames": count,
            "qualityApproved": False,
        },
        "detectorRadius": radius,
        "removalQualityApproved": False,
    }
