"""Bind explicitly reviewed current-image arc tracks to exact source frame cells."""

import math
import struct
from bisect import bisect_left, bisect_right
from dataclasses import dataclass
from fractions import Fraction

from replay.artifact_json import _json, _pin
from replay.core import ReplayError
from replay.recordings import record
from replay.stages import file_digest


@dataclass(frozen=True, slots=True)
class ArcCell:
    start: Fraction
    end: Fraction
    visible_top: float | None
    supported: int
    unknown: int


@dataclass(frozen=True, slots=True)
class VisibleArc:
    cells: tuple[ArcCell, ...]
    inputs: dict[str, str]

    def at(self, time: Fraction, *, ending: bool = False) -> ArcCell | None:
        search = bisect_left if ending else bisect_right
        index = search(self.cells, time, key=lambda cell: cell.start) - 1
        if index < 0 or not (
            self.cells[index].start < time <= self.cells[index].end
            if ending
            else self.cells[index].start <= time < self.cells[index].end
        ):
            return None
        return self.cells[index]

    def covers(self, start: Fraction, end: Fraction) -> bool:
        cursor = start
        for cell in self.cells:
            if cell.end <= cursor:
                continue
            if cell.start > cursor or cell.visible_top is None:
                return False
            cursor = cell.end
            if cursor >= end:
                return True
        return False


def _points(value: object, width: int, height: int) -> tuple[tuple[float, float] | None, ...]:
    if not isinstance(value, list) or not 3 <= len(value) <= 256:
        raise ReplayError("Expected bounded visible arc points")
    result: list[tuple[float, float] | None] = []
    for point in value:
        if point is None:
            result.append(None)
            continue
        if (
            not isinstance(point, list)
            or len(point) != 2
            or any(
                isinstance(number, bool)
                or not isinstance(number, (int, float))
                or not math.isfinite(number)
                for number in point
            )
        ):
            raise ReplayError("Invalid visible arc coordinates")
        x, y = float(point[0]), float(point[1])
        result.append((x, y) if 0 <= x < width and 0 <= y < height else None)
    return tuple(result)


def read_visible_arc(
    binding: dict[str, object],
    *,
    source_hash: str,
    times: tuple[Fraction, ...],
    width: int,
    height: int,
) -> VisibleArc:
    config_path, track_path = _pin(binding.get("config")), _pin(binding.get("tracks"))
    config, track = _json(config_path), _json(track_path)
    if (
        config.get("sourceMediaSHA256") != source_hash
        or track.get("configSHA256") != file_digest(config_path)
        or track.get("visibleArcReviewed") is not True
        or track.get("headContourComplete") is not False
        or track.get("placementApproved") is not False
        or track.get("timeBase") != config.get("timeBase")
        or track.get("seedPTS") != config.get("sourcePTS")
    ):
        raise ReplayError("Visible arc source or provenance differs")
    base = Fraction(str(config.get("timeBase")))
    seed = config.get("sourcePTS")
    if type(seed) is not int or seed < 0 or not 0 < base <= 1:
        raise ReplayError("Invalid visible arc clock")
    inputs = {str(path): file_digest(path) for path in (config_path, track_path)}
    review_path = _pin(config.get("visibleArcReview"))
    review = _json(review_path)
    if (
        review.get("reviewerKind") != "agent-visual-review"
        or review.get("visibleArcReviewed") is not True
    ):
        raise ReplayError("Explicit current-image visible arc review required")
    inputs[str(review_path)] = file_digest(review_path)
    donors, observations = config.get("donors"), track.get("observations")
    if (
        not isinstance(donors, list)
        or not isinstance(observations, list)
        or not 1 <= len(donors) == len(observations) <= 8
    ):
        raise ReplayError("Expected bounded matching visible arc donors")
    frames = [{"sourcePTS": seed, "sourcePNG": config.get("sourcePNG")}] + [
        record(item) for item in donors
    ]
    rows = [
        {
            "sourcePTS": seed,
            "points": config.get("points"),
            "supported": [True] * len(_points(config.get("points"), width, height)),
        }
    ] + [record(item) for item in observations]
    cells: list[ArcCell] = []
    for frame, row in zip(frames, rows, strict=True):
        pts = frame.get("sourcePTS")
        if type(pts) is not int or row.get("sourcePTS") != pts:
            raise ReplayError("Arc donor clock differs")
        time = pts * base
        index = bisect_right(times, time) - 1
        if (
            index < 0
            or index + 1 >= len(times)
            or times[index] != time
            or (cells and time <= cells[-1].start)
        ):
            raise ReplayError("Arc frame lacks exact source-clock support")
        image = _pin(frame.get("sourcePNG"))
        with image.open("rb") as handle:
            header = handle.read(24)
        if (
            len(header) != 24
            or header[:8] != b"\x89PNG\r\n\x1a\n"
            or struct.unpack(">II", header[16:24]) != (width, height)
        ):
            raise ReplayError("Visible arc image geometry differs")
        inputs[str(image)] = file_digest(image)
        points = _points(row.get("points"), width, height)
        support = row.get("supported")
        if (
            not isinstance(support, list)
            or len(support) != len(points)
            or any(type(flag) is not bool for flag in support)
        ):
            raise ReplayError("Invalid per-point arc support")
        valid = [
            point for point, flag in zip(points, support, strict=True) if flag and point is not None
        ]
        # A partial visible boundary is a candidate, never a complete head estimate.
        top = min(point[1] for point in valid) if len(valid) >= 3 else None
        cells.append(ArcCell(time, times[index + 1], top, len(valid), len(points) - len(valid)))
    return VisibleArc(tuple(cells), inputs)
