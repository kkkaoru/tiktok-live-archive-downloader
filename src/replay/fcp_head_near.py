"""Scale-aware candidate Y from paired exact-clock face observations."""

import json
from bisect import bisect_left, bisect_right
from dataclasses import dataclass, replace
from fractions import Fraction
from itertools import pairwise
from pathlib import Path
from typing import Literal
from xml.etree import ElementTree as ET

from replay.core import ReplayError
from replay.fcp_head_quality import geometry_at
from replay.fcp_project import rational
from replay.fcp_screen_stack import ScreenStack
from replay.fcp_tracking_caption import (
    FaceObservation,
    FaceObservationSeries,
    FacePositionSeries,
    TrackingCalibration,
    TrackingPosition,
    _sample_y,
    _time,
    face_observation_series,
    merge_face_observations,
    resolve_face_observations,
)
from replay.frame_cell_extent import validate_terminal_cell
from replay.head_near_placement import (
    FaceSizeObservation,
    FaceVerticalExtent,
    caption_only_bottom,
    current_face_heights,
    place_near_head,
)
from replay.head_observations import tracking_frames
from replay.subtitle_clock import VideoFrameClock
from replay.tracking_observations import RESPONSE_BYTES, TrackingJob, _read, load_job, verify
from replay.tracking_retained import RetainedTracking, _bound_bytes


@dataclass(frozen=True, slots=True)
class FaceScaleSeries:
    tops: FacePositionSeries
    bottoms: FacePositionSeries
    sizes: tuple[FaceSizeObservation, ...] = ()
    scale_heights: tuple[TrackingPosition, ...] = ()

    def with_current_scale(self, *, reference_time: Fraction) -> "FaceScaleSeries":
        if not self.sizes:
            raise ReplayError("Scale envelope requires paired observed face widths")
        self.observed_reference(reference_time)
        index = bisect_left(self.sizes, reference_time, key=lambda item: item.time)
        heights = current_face_heights(self.sizes, reference=self.sizes[index])
        return replace(
            self,
            scale_heights=tuple(
                TrackingPosition(item.time, 0, height)
                for item, height in zip(self.sizes, heights, strict=True)
            ),
        )

    def estimated_height(self, time: Fraction) -> float:
        face = self.extent(time)
        if not self.scale_heights:
            return face.bottom - face.top
        time = min(time, self.scale_heights[-1].time)
        index = bisect_right(self.scale_heights, time, key=lambda p: p.time) - 1
        return self.scale_heights[index].y

    def __post_init__(self) -> None:
        top, bottom = self.tops, self.bottoms
        if (
            top.anchor != "top"
            or bottom.anchor != "bottom"
            or len(top.positions) < 2
            or top.coverage_end <= top.positions[-1].time
            or (top.width, top.height, top.coverage_end, top.derived_ranges)
            != (bottom.width, bottom.height, bottom.coverage_end, bottom.derived_ranges)
            or len(top.positions) != len(bottom.positions)
            or any(b.time <= a.time for a, b in pairwise(top.positions))
            or any(
                (a.time, a.x) != (b.time, b.x) or not 0 <= a.y < b.y <= top.height
                for a, b in zip(top.positions, bottom.positions, strict=True)
            )
        ):
            raise ReplayError("Face top/bottom geometry, exact clocks and hold provenance differ")
        if self.sizes and tuple(item.time for item in self.sizes) != tuple(
            p.time for p in top.positions
        ):
            raise ReplayError("Face scale sizes must share the exact observation clocks")
        if self.scale_heights and tuple(p.time for p in self.scale_heights) != tuple(
            p.time for p in top.positions
        ):
            raise ReplayError("Derived face heights must share the exact observation clocks")

    def observed_reference(self, time: Fraction) -> FaceVerticalExtent:
        index = bisect_left(self.tops.positions, time, key=lambda p: p.time)
        if (
            index == len(self.tops.positions)
            or self.tops.positions[index].time != time
            or any(gap.start <= time < gap.end for gap in self.tops.derived_ranges)
        ):
            raise ReplayError("Scale-aware title requires exact observed calibration and coverage")
        return self.extent(time)

    def extent(self, time: Fraction) -> FaceVerticalExtent:
        if not self.tops.positions[0].time <= time <= self.tops.coverage_end:
            raise ReplayError("Requested face extent exceeds frame coverage")
        time = min(time, self.tops.positions[-1].time)
        return FaceVerticalExtent(
            _sample_y(self.tops.positions, time), _sample_y(self.bottoms.positions, time)
        )


def _width_observations(
    raw: object, clock: VideoFrameClock, *, width: int, height: int
) -> FaceObservationSeries:
    frames = tracking_frames(raw, clock, width=width, height=height)
    return FaceObservationSeries(
        tuple(
            FaceObservation(
                frame.time,
                TrackingPosition(
                    frame.time,
                    frame.faces[0].box.left,
                    frame.faces[0].box.right - frame.faces[0].box.left,
                )
                if len(frame.faces) == 1
                else None,
                None if len(frame.faces) == 1 else ("missing" if not frame.faces else "ambiguous"),
            )
            for frame in frames
        ),
        (clock.pts[-1] + clock.last_duration) * clock.time_base,
        width,
        height,
        "top",
    )


def read_face_scale(
    references: list[tuple[Path, Path]],
    *,
    missing_policy: Literal["reject", "hold"],
    seed_overlap: Literal["reject", "next"],
    retained: RetainedTracking | None = None,
) -> tuple[TrackingJob, FaceScaleSeries]:
    """Verify each window once and derive both anchors from the same raw frames."""
    if not 1 <= len(references) <= 512:
        raise ReplayError("Expected bounded face window references")
    first = load_job(references[0][0]) if retained is None else retained.window(*references[0]).job
    tops: list[FaceObservationSeries] = []
    bottoms: list[FaceObservationSeries] = []
    widths: list[FaceObservationSeries] = []
    for config, evidence in references:
        window = None if retained is None else retained.window(config, evidence)
        job = load_job(config) if window is None else window.job
        if (job.source, job.source_hash) != (first.source, first.source_hash):
            raise ReplayError("Face scale windows do not share one pinned source")
        if window is None:
            verify(config, evidence)
            data = _read(evidence / "response.json", RESPONSE_BYTES)
        else:
            data = _bound_bytes(evidence / "response.json", window.response_sha256, RESPONSE_BYTES)
        raw = json.loads(data)
        widths.append(_width_observations(raw, job.clock, width=job.width, height=job.height))
        tops.append(
            face_observation_series(
                raw, job.clock, width=job.width, height=job.height, anchor="top"
            )
        )
        bottoms.append(
            face_observation_series(
                raw, job.clock, width=job.width, height=job.height, anchor="bottom"
            )
        )
    result = FaceScaleSeries(
        resolve_face_observations(
            merge_face_observations(tuple(tops), seed_overlap=seed_overlap),
            missing_policy=missing_policy,
        ),
        resolve_face_observations(
            merge_face_observations(tuple(bottoms), seed_overlap=seed_overlap),
            missing_policy=missing_policy,
        ),
    )
    width_series = resolve_face_observations(
        merge_face_observations(tuple(widths), seed_overlap=seed_overlap),
        missing_policy=missing_policy,
    )
    return first, replace(
        result,
        sizes=tuple(
            FaceSizeObservation(
                top.time,
                width.y,
                bottom.y - top.y,
                not any(gap.start <= top.time < gap.end for gap in result.tops.derived_ranges),
            )
            for top, bottom, width in zip(
                result.tops.positions, result.bottoms.positions, width_series.positions, strict=True
            )
        ),
    )


def title_samples(
    title: ET.Element,
    *,
    faces: FaceScaleSeries,
    calibration: TrackingCalibration,
    caption_height: float,
    frame_duration: Fraction = Fraction(1, 30),
    allow_terminal_cell: bool = False,
    screen_stacks: tuple[ScreenStack, ...] = (),
) -> tuple[tuple[str, str], ...]:
    """Preserve title/source clocks and centered X; never relabel holds as observed."""
    start = _time(title.get("offset"))
    end = start + _time(title.get("duration"))
    local_start = _time(title.get("start"))
    validate_terminal_cell(
        start=start,
        end=end,
        physical_end=faces.tops.coverage_end,
        frame_duration=frame_duration,
        allow_terminal_cell=allow_terminal_cell,
    )
    if (
        title.tag != "title"
        or not start < end
        or start < faces.tops.positions[0].time
        or calibration.caption_local_bottom != 0
        or calibration.tracker_y_to_pixels != 1
        or calibration.canvas_height != faces.tops.height
    ):
        raise ReplayError("Scale-aware title requires exact observed calibration and coverage")
    reference = faces.observed_reference(calibration.reference_time)
    last = max(start, min(end, faces.tops.positions[-1].time))
    left = bisect_right(faces.tops.positions, start, key=lambda p: p.time)
    right = bisect_left(faces.tops.positions, last, key=lambda p: p.time)
    times = sorted({start, *(p.time for p in faces.tops.positions[left:right]), last})
    result: list[tuple[str, str]] = []
    for time in times:
        quality = geometry_at(faces.tops, time)
        if not quality.observed:
            stack_height = max(
                (
                    stack.height
                    for stack in screen_stacks
                    if stack.start <= time < stack.end
                    or stack.start <= time == stack.end == faces.tops.coverage_end
                ),
                default=caption_height,
            )
            if stack_height < caption_height:
                raise ReplayError("Screen stack cannot be smaller than its measured caption")
            bottom = caption_only_bottom(
                caption_height=stack_height,
                canvas_height=calibration.canvas_height,
            )
            y = 50 - bottom / calibration.canvas_height * 100
            result.append((rational(local_start + time - start), f"0 {y:.12g}"))
            continue
        placement = place_near_head(
            face=faces.extent(time),
            reference_face=reference,
            reference_head_top=calibration.source_head_top,
            caption_height=caption_height,
            canvas_height=calibration.canvas_height,
            gap=calibration.gap,
            estimated_face_height=faces.estimated_height(time),
        )
        if placement.caption_bottom is None:
            raise ReplayError(f"Caption at source {time} needs measured token-clock height paging")
        y = 50 - placement.caption_bottom / calibration.canvas_height * 100
        result.append((rational(local_start + time - start), f"0 {y:.12g}"))
    return tuple(result)
