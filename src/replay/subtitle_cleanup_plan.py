"""Build measured subtitle search regions and exact, resumable source clock chunks.

Search regions are not pixel removal masks or semantic approval. The existing
PPOCR repair consumer measures glyph support inside them. Unrecognized frames
remain explicitly unknown, never silently certified as text-free.
"""

from dataclasses import asdict, dataclass
from fractions import Fraction
from itertools import pairwise
from math import ceil, floor

from replay.caption_io import FrameObservation
from replay.caption_masks import MaskPolicy, subtitle_rectangles
from replay.recordings import record
from replay.subtitle_clock import VideoFrameClock
from replay.subtitle_plan import repair_spans
from replay.subtitle_profile import SubtitleBands, learn_subtitle_bands, profiled_detections

MAX_SOURCE_FRAMES = 2_000_000
MAX_CHUNK_FRAMES = 10_000


@dataclass(frozen=True, slots=True)
class CaptionSearchProfile:
    bands: SubtitleBands
    measured_envelope: tuple[int, int, int, int]


def clock_chunks(value: object, *, maximum_frames: int) -> tuple[VideoFrameClock, ...]:
    """Use the next real PTS as every internal terminal boundary."""
    data = record(value)
    raw_pts = data.get("pts")
    base = data.get("timeBase")
    terminal = data.get("lastDuration")
    if (
        type(maximum_frames) is not int
        or not 1 <= maximum_frames <= MAX_CHUNK_FRAMES
        or not isinstance(raw_pts, list)
        or not 1 <= len(raw_pts) <= MAX_SOURCE_FRAMES
        or any(type(value) is not int or value < 0 for value in raw_pts)
        or not isinstance(base, str)
        or type(terminal) is not int
        or terminal <= 0
    ):
        raise ValueError("Invalid complete source clock or chunk budget")
    pts = tuple(int(value) for value in raw_pts)
    if any(right <= left for left, right in pairwise(pts)):
        raise ValueError("Source PTS must be globally unique and increasing")
    time_base = Fraction(base)
    result: list[VideoFrameClock] = []
    for start in range(0, len(pts), maximum_frames):
        end = min(len(pts), start + maximum_frames)
        last = pts[end] - pts[end - 1] if end < len(pts) else terminal
        result.append(VideoFrameClock(pts[start:end], time_base, last))
    return tuple(result)


def _envelope(
    rectangles: tuple[tuple[float, float, float, float], ...],
) -> tuple[int, int, int, int]:
    if not rectangles:
        raise ValueError("No measured subtitle envelope")
    x = floor(min(box[0] for box in rectangles))
    y = floor(min(box[1] for box in rectangles))
    right = ceil(max(box[0] + box[2] for box in rectangles))
    bottom = ceil(max(box[1] + box[3] for box in rectangles))
    return x, y, right - x, bottom - y


def search_profile(frames: tuple[FrameObservation, ...]) -> CaptionSearchProfile:
    """Learn an envelope from variable subtitle bands, including clipped edges."""
    bands = learn_subtitle_bands(frames)
    rectangles = tuple(
        (line.rectangle.x, line.rectangle.y, line.rectangle.width, line.rectangle.height)
        for frame in frames
        for line in profiled_detections(frame, bands)
    )
    x, y, width, height = _envelope(rectangles)
    # OCR allows subpixel numerical overflow; never extend the search off-canvas.
    right, bottom = min(bands.width, x + width), min(bands.height, y + height)
    return CaptionSearchProfile(bands, (max(0, x), max(0, y), right - x, bottom - y))


def frame_region(
    frame: FrameObservation, profile: CaptionSearchProfile, *, policy: MaskPolicy
) -> tuple[dict[str, int], list[int]] | None:
    """Retain measured profile spill when OCR misses a translated line this frame."""
    selected = profiled_detections(frame, profile.bands)
    rectangles = subtitle_rectangles(
        selected, width=frame.width, height=frame.height, policy=policy
    )
    if not selected or not rectangles:
        return None
    boxes = tuple((box.x, box.y, box.width, box.height) for box in rectangles)
    x, y, width, height = _envelope((*boxes, profile.measured_envelope))
    if width * height > 2_000_000 or min(width, height) < 8:
        raise ValueError("Measured search region exceeds detector budget")
    return (
        {"x": x, "y": y, "width": width, "height": height},
        [index for index, line in enumerate(frame.detections) if line in selected],
    )


def admission(
    frames: tuple[FrameObservation, ...],
    clock: VideoFrameClock,
    profile: CaptionSearchProfile,
    *,
    policy: MaskPolicy,
) -> dict[str, object]:
    """Create only per-frame observed candidate regions; retain unknown records."""
    if len(frames) != len(clock.pts):
        raise ValueError("OCR does not cover the requested source chunk")
    masks: list[dict[str, object]] = []
    entries: list[dict[str, object]] = []
    unknown: list[int] = []
    terminal = clock.pts[-1] + clock.last_duration
    for index, (frame, pts) in enumerate(zip(frames, clock.pts, strict=True)):
        if (
            frame.requested_time != float(pts * clock.time_base)
            or frame.actual_time != frame.requested_time
        ):
            raise ValueError("OCR frame clock differs from the actual source PTS")
        candidate = frame_region(frame, profile, policy=policy)
        entry: dict[str, object] = {"sourcePTS": pts, "observation": index}
        if candidate is None:
            entry["classification"] = "unknown"
            unknown.append(pts)
        else:
            region, lines = candidate
            next_pts = clock.pts[index + 1] if index + 1 < len(clock.pts) else terminal
            masks.append(
                {
                    "startSeconds": float((pts - clock.pts[0]) * clock.time_base),
                    "endSeconds": float((next_pts - clock.pts[0]) * clock.time_base),
                    "region": region,
                }
            )
            entry.update(classification="overlay-subtitle", region=region, lines=lines)
        entries.append(entry)
    spans = repair_spans(
        masks,
        frames=len(clock.pts),
        width=profile.bands.width,
        height=profile.bands.height,
        source_clock=clock,
    )
    if sum(span.end_frame - span.start_frame for span in spans) != len(frames) - len(unknown):
        raise ValueError("Candidate mask membership differs from measured admission")
    return {
        "version": 1,
        "policy": "measured-variable-subtitle-bands-v1",
        "profile": asdict(profile),
        "masks": masks,
        "frames": entries,
        "unknownSourcePTS": unknown,
        "unknownFramesKeptUnmodified": True,
        "semanticClassification": "heuristic-candidate-with-pinned-source-review",
        "removalQualityApproved": False,
    }
