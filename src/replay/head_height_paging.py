"""Repaginate only segments exceeding observed face headroom, reusing glyph data."""

import json
from bisect import bisect_left, bisect_right
from dataclasses import dataclass
from fractions import Fraction
from functools import partial
from pathlib import Path

from replay.caption_long_layout import LongLineCaptionLayout
from replay.caption_measure_cache import CachedCaptionLayout
from replay.caption_native_inputs import segment_binding
from replay.caption_native_layout import (
    MAX_CACHE,
    NativeCaptionLayout,
    NativeCaptionStyle,
    _positive,
)
from replay.caption_native_paging import paged_cues, read_tokens
from replay.caption_scale import read_scale, scale_fields
from replay.caption_style_defaults import HEAD_CAPTION_GAP_PX
from replay.caption_timing import ScaledCaptionCue
from replay.caption_token_sources import TokenSource
from replay.core import ReplayError
from replay.fcp_caption_retarget import RetainedSpan
from replay.fcp_head_near import FaceScaleSeries
from replay.fcp_head_quality import geometry_at
from replay.foreground_evidence import _bound_bytes
from replay.frame_cell_extent import validate_terminal_cell
from replay.head_near_placement import place_near_head
from replay.motion_stages import _integer
from replay.recordings import record
from replay.stages import file_digest


@dataclass(frozen=True, slots=True)
class HeadroomPreference:
    reference_time: Fraction
    reference_head_top: float
    minimum_line_height: float


def minimum_headroom(
    faces: FaceScaleSeries,
    spans: tuple[RetainedSpan, ...],
    *,
    start: Fraction,
    end: Fraction,
    frame_duration: Fraction = Fraction(1, 30),
    allow_terminal_cell: bool = False,
    preference: HeadroomPreference | None = None,
) -> float:
    """Use only retained source intervals corresponding to this output segment."""
    heights: list[float] = []
    head_heights: list[float] = []
    ratio = 0.0
    if preference is not None:
        reference = faces.observed_reference(preference.reference_time)
        ratio = place_near_head(
            face=reference,
            reference_face=reference,
            reference_head_top=preference.reference_head_top,
            caption_height=preference.minimum_line_height,
            canvas_height=faces.tops.height,
        ).head_extension_ratio

    def collect(time: Fraction) -> None:
        stale = preference is not None and not geometry_at(faces.tops, time).observed
        if stale and preference is not None:
            # Unknown geometry uses the screen only, never an old face/reference.
            heights.append(float(faces.tops.height))
            head_heights.append(float(faces.tops.height))
        else:
            top = faces.extent(time).top
            heights.append(top)
            head_heights.append(top - ratio * faces.estimated_height(time))

    for span in spans:
        left = max(start, span.original_start)
        right = min(end, span.original_start + span.duration)
        if left >= right:
            continue
        source_left = span.source_start + left - span.original_start
        source_right = span.source_start + right - span.original_start
        validate_terminal_cell(
            start=source_left,
            end=source_right,
            physical_end=faces.tops.coverage_end,
            frame_duration=frame_duration,
            allow_terminal_cell=allow_terminal_cell,
        )
        source_right = min(source_right, faces.tops.coverage_end)
        a = bisect_right(faces.tops.positions, source_left, key=lambda p: p.time)
        b = bisect_left(faces.tops.positions, source_right, key=lambda p: p.time)
        for point in faces.tops.positions[a:b]:
            collect(point.time)
        collect(source_left)
        collect(source_right)
    if not heights:
        raise ReplayError("Caption segment has no selected source support")
    result = min(heights) - HEAD_CAPTION_GAP_PX
    if result <= 0:
        raise ReplayError("No above-face caption region exists in this segment")
    if preference is not None:
        # Prefer pages fitting above the approximate head. When even one native
        # line cannot fit there, preserve one line and let placement explicitly
        # mark its minimal canvas fit, still strictly above the measured face.
        result = min(
            result, max(preference.minimum_line_height, min(head_heights) - HEAD_CAPTION_GAP_PX)
        )
    return result


def single_line_height(part: dict[str, object], layout: CachedCaptionLayout) -> float:
    rows = part.get("captions")
    if not isinstance(rows, list) or not rows:
        raise ReplayError("Expected measured caption stages for headroom")
    lines: set[str] = set()
    for value in rows:
        text = record(value).get("text")
        if not isinstance(text, str):
            raise ReplayError("Expected caption text for native footprint")
        lines.update(text.split("\n"))
    return max(layout.measure(line)[1] for line in lines)


def repage_segment(
    *,
    captions: Path,
    segment: dict[str, object],
    part: dict[str, object],
    layout: CachedCaptionLayout,
    maximum_height: float,
    rate: int,
    token_sources: tuple[TokenSource, ...] = (),
) -> tuple[dict[str, object], bool]:
    """Keep old pages if they fit; otherwise rerun the original recognized tokens."""
    rows = part.get("captions")
    if not isinstance(rows, list) or not rows:
        raise ReplayError("Measured segment lacks its caption stages")
    texts = tuple(record(row).get("text") for row in rows)
    if any(not isinstance(text, str) for text in texts):
        raise ReplayError("Measured caption text must be a string")
    if all(
        layout.measure(str(record(row)["text"]))[1]
        * (1.0 if (scale := read_scale(record(row))) is None else scale.factor)
        <= maximum_height
        for row in rows
    ):
        return part, False
    combined = segment_binding(captions, segment, token_sources)
    raw = record(json.loads(_bound_bytes(combined, file_digest(combined), 64 * 1024 * 1024)))
    tokens, _ = read_tokens(captions, segment, record(raw.get("inputs")), token_sources)
    duration = Fraction(_integer(segment.get("frames")), rate)
    offset = Fraction(_integer(segment.get("offsetFrame")), rate)
    cues, hidden, widened = paged_cues(
        tokens, layout=layout, rate=rate, duration=duration, maximum_height=maximum_height
    )
    return {
        "captions": [
            {
                "start": cue.start + float(offset),
                "end": cue.end + float(offset),
                "text": cue.text,
                **scale_fields(cue.scale if isinstance(cue, ScaledCaptionCue) else None),
            }
            for cue in cues
        ],
        "hiddenRetainedPrefixes": hidden,
        "frameVisibilityExtensions": widened,
        "maximumHeight": maximum_height,
        "coverage": part.get("coverage"),
    }, True


def make_layout(
    *,
    reference: Path,
    executable: Path,
    output: Path,
    pinned_inputs: dict[str, str],
    cached_directory: Path,
    long_executable: Path | None = None,
) -> CachedCaptionLayout:
    """Require an explicit saved style and verify it against every cached request."""
    raw = record(json.loads(reference.read_text(encoding="utf-8")))
    style = record(raw.get("style"))
    name = style.get("font_name")
    if not isinstance(name, str):
        raise ReplayError("Expected pinned native font identity")
    factory = (
        NativeCaptionLayout
        if long_executable is None
        else partial(LongLineCaptionLayout, long_executable=long_executable)
    )
    native = factory(
        executable=executable,
        directory=output,
        style=NativeCaptionStyle(
            name,
            _positive(style.get("ass_font_size")),
            _positive(style.get("border")),
            _positive(style.get("rim")),
            _positive(style.get("motion_line_advance")),
        ),
        width=_positive(raw.get("width")),
        height=4096,
        evidence=(reference,),
        expected_point_size=_positive(raw.get("expectedPointSize")),
    )
    layout = CachedCaptionLayout(native)
    requests = tuple(
        Path(name) for name in pinned_inputs if Path(name).name.endswith(".request.json")
    )
    if not requests or len(requests) > MAX_CACHE:
        raise ReplayError("Expected bounded pinned glyph requests for head placement")
    # A fresh paging run may measure only new phrases and inherit the rest from
    # older, verified measurement directories. Reuse only its explicit pins;
    # never scan neighboring caches or infer that a whole directory is approved.
    directories = sorted(
        {request.parent for request in requests},
        key=lambda directory: (directory != cached_directory, str(directory)),
    )
    for directory in directories:
        layout.reuse(directory=directory, inputs=pinned_inputs)
    return layout
