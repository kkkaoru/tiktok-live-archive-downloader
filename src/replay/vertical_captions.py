"""Place vertical-replay captions above the character's head and style them.

A head track (face box, or the whole-figure box when no face is found) is sampled
from the source video. Captions stay horizontally centred; only their height
follows the head. Each run of growing captions keeps one height taken from the
highest head position during the run, so text never covers the head while it
grows. Between runs a caption moves up as soon as the head needs it, but moves
down only when the head clearly dropped, so a still face keeps a still caption.
The caption top never rises above the on-screen gauges and banners: fixed text that
the replay's full-frame OCR survey finds at one place in most sampled frames, in
the upper screen across the central caption column, sets this replay's top
limit. A close-up face therefore puts the caption just below them. Without any
detection the last height is kept, and before the first detection the legacy centre
line is used.

Captions use the single-speaker YouTube style: solid blue-cyan inside a thick
white border with a thin black rim. Lines break after punctuation and otherwise
split into lines of nearly equal length, at most 18 characters each.
"""

from __future__ import annotations

import json
import math
from bisect import bisect_left, bisect_right
from dataclasses import dataclass
from pathlib import Path

from replay.caption_projection import CutSpan
from replay.caption_timing import CaptionCue, punctuation_lines
from replay.core import ReplayError
from replay.recordings import record
from replay.speaker_subtitles import (
    BLACK_RIM,
    LINE_HEIGHT,
    SINGLE_FILL,
    SINGLE_FONT_SIZE,
    WHITE_BORDER,
    ass_clock,
    ass_header,
    bgr,
    bordered_layers,
    clean_text,
    wrap_text,
)

FRAME = (1080, 1920)
FRAME_RATE = 30
# libass sizes text by the font's line height, not its em box. Measured with
# Hiragino Sans on the 1080x1920 canvas, size 66 draws glyphs about as large as the
# native renderer's size 56 (42 vs 46 px for 「士」) while 18 characters plus borders
# still fit the frame width (1018 px), so every replay shows the same text size.
FONT_SIZE = 66
LINE_CHARACTERS = 18
# Border and rim keep the single-speaker YouTube proportions at this text size.
BORDER = round(WHITE_BORDER * FONT_SIZE / SINGLE_FONT_SIZE)
RIM = round(BLACK_RIM * FONT_SIZE / SINGLE_FONT_SIZE)
MAXIMUM_LINES = 2
# Placement uses the largest block so the height never depends on the text.
BLOCK_HEIGHT = math.ceil(MAXIMUM_LINES * FONT_SIZE * LINE_HEIGHT)
# Gauges and counters occupy the top of TikTok replays; caption tops stay below.
TOP_LIMIT = 380
# Fixed text recurring in this many of the surveyed frames is part of the layout.
STATIC_MINIMUM_FRAMES = 3
STATIC_OVERLAP = 0.5
# Only fixed text in the upper screen across the caption column limits the height.
UPPER_SCREEN = 700
CAPTION_COLUMN = (360, 720)
OVERLAY_PADDING = 16
BOTTOM_LIMIT = 1800
HEAD_MARGIN = 40
LEGACY_CENTER_Y = 720
# Anime-style hair extends above the detected face box.
FACE_HAIR_RATIO = 0.8
RUN_PADDING_SECONDS = 0.5
# The highest tenth of a run's head positions; a lone misdetection is ignored.
HEAD_PERCENTILE = 0.1
# Rising by up to this much keeps the caption in place; HEAD_MARGIN still separates it.
UP_TOLERANCE = 20
# The caption follows a lowered head only past this distance.
DOWN_THRESHOLD = 160
GROWTH_JOIN_SECONDS = 0.05
MAXIMUM_SAMPLES = 200_000
STYLE = "vertical"
STYLE_RECORD: dict[str, object] = {
    "font": "Hiragino Sans",
    "fontSize": FONT_SIZE,
    "fill": "#" + "".join(f"{component:02X}" for component in SINGLE_FILL),
    "whiteBorder": BORDER,
    "blackRim": RIM,
    "position": "head-tracked-height-centred",
    "lineCharacters": LINE_CHARACTERS,
    "minimumTopLimit": TOP_LIMIT,
}


@dataclass(frozen=True, slots=True)
class HeadSample:
    """Head top in output pixels at a source time; None when undetected."""

    seconds: float
    top: float | None


@dataclass(frozen=True, slots=True)
class Box:
    left: float
    top: float
    right: float
    bottom: float

    def overlap(self, other: Box) -> float:
        """Intersection area over the smaller area."""
        width = min(self.right, other.right) - max(self.left, other.left)
        height = min(self.bottom, other.bottom) - max(self.top, other.top)
        if width <= 0 or height <= 0:
            return 0.0
        smaller = min(
            (self.right - self.left) * (self.bottom - self.top),
            (other.right - other.left) * (other.bottom - other.top),
        )
        return width * height / smaller


@dataclass(frozen=True, slots=True)
class PlacedCaption:
    text: str
    start: float
    end: float
    bottom: int


def _optional(value: object, label: str) -> float | None:
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
        raise ReplayError(f"Invalid head track {label}")
    return float(value)


def head_sample(raw: object) -> HeadSample:
    """Derive the head top from a face box, else from the whole-figure box."""
    item = record(raw)
    seconds = _optional(item.get("t"), "time")
    if seconds is None or seconds < 0:
        raise ReplayError("Invalid head track time")
    face_top = _optional(item.get("faceTop"), "face top")
    face_height = _optional(item.get("faceHeight"), "face height")
    body_top = _optional(item.get("bodyTop"), "body top")
    if face_top is not None and face_height is not None:
        return HeadSample(seconds, face_top - FACE_HAIR_RATIO * face_height)
    return HeadSample(seconds, body_top)


def load_head_track(path: Path) -> tuple[HeadSample, ...]:
    data = record(json.loads(path.read_text(encoding="utf-8")))
    if data.get("frame") != list(FRAME):
        raise ReplayError("Head track was not measured on the vertical output frame")
    samples = data.get("samples")
    if not isinstance(samples, list) or not samples or len(samples) > MAXIMUM_SAMPLES:
        raise ReplayError("Head track has no bounded sample list")
    result = tuple(head_sample(item) for item in samples)
    if any(right.seconds <= left.seconds for left, right in zip(result, result[1:], strict=False)):
        raise ReplayError("Head track samples are not ordered")
    return result


def source_seconds(output: float, cuts: tuple[CutSpan, ...]) -> float:
    """Map an output time back to the source clock through the kept spans."""
    frame = output * FRAME_RATE
    for cut in cuts:
        if frame < cut.output_start_frame + cut.duration_frames:
            return (cut.source_start_frame + max(0.0, frame - cut.output_start_frame)) / FRAME_RATE
    last = cuts[-1]
    return last.source_end_frame / FRAME_RATE


def highest_head(
    samples: tuple[HeadSample, ...],
    clocks: list[float],
    start: float,
    end: float,
    *,
    conservative: bool = False,
) -> float | None:
    """A high percentile, or the highest observed head for strict placement."""
    window = samples[bisect_left(clocks, start) : bisect_right(clocks, end)]
    tops = sorted(sample.top for sample in window if sample.top is not None)
    if not tops:
        return None
    return tops[0 if conservative else int(len(tops) * HEAD_PERCENTILE)]


def growth_runs(cues: tuple[CaptionCue, ...]) -> list[list[int]]:
    """Consecutive captions that only add words to the previous one form a run."""
    runs: list[list[int]] = []
    for index, cue in enumerate(cues):
        previous = cues[index - 1] if index else None
        if (
            previous is not None
            and cue.start - previous.end <= GROWTH_JOIN_SECONDS
            and _flat(cue.text).startswith(_flat(previous.text))
        ):
            runs[-1].append(index)
        else:
            runs.append([index])
    return runs


def _flat(text: str) -> str:
    return text.replace("\n", "")


def _survey_boxes(survey: object) -> list[list[Box]]:
    """Text boxes per surveyed frame, scaled into the output canvas."""
    measurement = record(record(record(survey).get("data")).get("structuredContent")).get(
        "measurement"
    )
    if not isinstance(measurement, list):
        raise ReplayError("Full-frame survey has no measurement list")
    frames: list[list[Box]] = []
    for item in measurement:
        frame = record(item)
        width, height, lines = frame.get("width"), frame.get("height"), frame.get("lines")
        if type(width) is not int or type(height) is not int or not isinstance(lines, list):
            raise ReplayError("Full-frame survey frame is malformed")
        scale = min(FRAME[0] / width, FRAME[1] / height)
        boxes: list[Box] = []
        for line in lines:
            region = record(record(line).get("region"))
            x, y = _optional(region.get("x"), "survey x"), _optional(region.get("y"), "survey y")
            w = _optional(region.get("width"), "survey width")
            h = _optional(region.get("height"), "survey height")
            if x is None or y is None or w is None or h is None:
                raise ReplayError("Full-frame survey region is incomplete")
            boxes.append(Box(x * scale, y * scale, (x + w) * scale, (y + h) * scale))
        frames.append(boxes)
    return frames


def top_limit(survey: object) -> int:
    """Highest allowed caption top: just below fixed text over the caption column."""
    frames = _survey_boxes(survey)
    limit = TOP_LIMIT
    for index, boxes in enumerate(frames):
        others = frames[:index] + frames[index + 1 :]
        for box in boxes:
            recurring = 1 + sum(
                any(box.overlap(other) >= STATIC_OVERLAP for other in frame) for frame in others
            )
            if (
                recurring >= STATIC_MINIMUM_FRAMES
                and box.top < UPPER_SCREEN
                and box.left < CAPTION_COLUMN[1]
                and box.right > CAPTION_COLUMN[0]
            ):
                limit = max(limit, math.ceil(box.bottom) + OVERLAY_PADDING)
    return limit


def place(head_top: float, limit: int = TOP_LIMIT) -> int:
    """Caption bottom above the head, kept below the gauges and above the frame edge."""
    return min(max(round(head_top) - HEAD_MARGIN, limit + BLOCK_HEIGHT), BOTTOM_LIMIT)


def settle(target: int, previous: int | None) -> int:
    """Keep the previous height unless the head needs it higher or clearly dropped."""
    if previous is not None and -UP_TOLERANCE <= target - previous < DOWN_THRESHOLD:
        return previous
    return target


def layout(
    cues: tuple[CaptionCue, ...],
    cuts: tuple[CutSpan, ...],
    samples: tuple[HeadSample, ...],
    limit: int = TOP_LIMIT,
    *,
    require_headroom: bool = False,
    allow_top_overlap: bool = False,
    caption_scale: float = 1.0,
) -> tuple[int, ...]:
    """One bottom per growth run; optionally require observed space above the head.

    Strict placement keeps the full block and outer rim below the top limit and
    above every observed head in the run. Missing detections or insufficient space
    fail instead of silently using the centre line or clamping onto the subject.
    Sampled head geometry does not establish coverage of unobserved frames or
    calibrate a different renderer's actual text bounds. Explicit allow_top_overlap
    clamps insufficient or unknown headroom at the upper placement limit, without
    cropping or scaling the footage; overlap there is permitted, not disproved.
    """
    if (
        isinstance(caption_scale, bool)
        or not math.isfinite(caption_scale)
        or not 0 < caption_scale <= 4
    ):
        raise ReplayError("Invalid caption scale")
    block_height = math.ceil(MAXIMUM_LINES * FONT_SIZE * LINE_HEIGHT * caption_scale)
    outline = math.ceil((BORDER + RIM) * caption_scale)
    if limit + block_height + outline > BOTTOM_LIMIT:
        raise ReplayError("Scaled caption block exceeds the available canvas")
    bottoms: list[int] = []
    previous: int | None = None
    clocks = [sample.seconds for sample in samples]
    for run in growth_runs(cues):
        head = highest_head(
            samples,
            clocks,
            source_seconds(cues[run[0]].start, cuts) - RUN_PADDING_SECONDS,
            source_seconds(cues[run[-1]].end, cuts) + RUN_PADDING_SECONDS,
            conservative=require_headroom or allow_top_overlap,
        )
        if require_headroom or allow_top_overlap:
            minimum_bottom = limit + block_height + outline
            if head is None and not allow_top_overlap:
                raise ReplayError("Caption run has no observed head position")
            ceiling = (
                minimum_bottom
                if head is None
                else min(math.floor(head - HEAD_MARGIN - outline), BOTTOM_LIMIT)
            )
            if ceiling < minimum_bottom and not allow_top_overlap:
                raise ReplayError("Caption block cannot fit above the observed head")
            bottom = min(
                settle(max(ceiling, minimum_bottom), previous), max(ceiling, minimum_bottom)
            )
        elif head is not None:
            target = min(max(round(head) - HEAD_MARGIN, limit + block_height), BOTTOM_LIMIT)
            bottom = settle(target, previous)
        else:
            bottom = previous if previous is not None else LEGACY_CENTER_Y + block_height // 2
        bottoms.extend([bottom] * len(run))
        previous = bottom
    return tuple(bottoms)


def display_text(text: str) -> str:
    """Break after punctuation, then split long lines evenly within the line width."""
    flat = punctuation_lines(_flat(text))
    return wrap_text(clean_text(flat), LINE_CHARACTERS, balance=True)


def render_ass(captions: list[PlacedCaption]) -> str:
    """ASS for one render part; times are relative to the part start."""
    fill = bgr(SINGLE_FILL)
    events: list[str] = []
    for caption in captions:
        events.extend(
            bordered_layers(
                f"{ass_clock(caption.start)},{ass_clock(caption.end)}",
                STYLE,
                f"\\an2\\pos({FRAME[0] // 2},{caption.bottom})",
                display_text(caption.text).replace("\n", "\\N"),
                [(0, FRAME[1], fill)],
                width=FRAME[0],
                border=BORDER,
                rim=RIM,
            )
        )
    header = ass_header([(STYLE, FONT_SIZE)], frame=FRAME, border=BORDER)
    return header + "\n".join(events) + "\n"
