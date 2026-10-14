"""Keep current measured upper pixels distinct from face-box scale and head claims."""

from fractions import Fraction
from itertools import pairwise
from xml.etree import ElementTree as ET

from replay.caption_style_defaults import HEAD_CAPTION_GAP_PX
from replay.core import ReplayError
from replay.fcp_caption_srt import _time
from replay.fcp_project import rational
from replay.fcp_visible_arc import ArcCell, VisibleArc
from replay.foreground_observations import foreground_frames
from replay.recordings import record

MAX_UPPER_POINTS = 256


def measured_upper_cells(
    response: object,
    *,
    times: tuple[Fraction, ...],
    frame_ends: tuple[Fraction, ...],
    width: int,
    height: int,
) -> tuple[ArcCell, ...]:
    """Read actual contour pixels, preserving lateral-clipped whole-head status.

    An interior partial arc is an approximate candidate. It is never relabeled as
    a detected complete head or used beyond its independently supplied frame cell.
    Canvas-clipped, absent and unsupported contours remain unknown.
    """
    if (
        not times
        or len(times) != len(frame_ends)
        or any(a >= b for a, b in zip(times, frame_ends, strict=True))
        or any(b <= a for a, b in pairwise(times))
        or any(end > following for end, following in zip(frame_ends, times[1:], strict=False))
    ):
        raise ReplayError("Expected ordered independent source-frame cells")
    foreground_frames(response, times=times, width=width, height=height)
    if not isinstance(response, list):
        raise ReplayError("Expected foreground measurement rows")
    return tuple(
        _upper_cell(record(frame), start=start, end=end, width=width, height=height)
        for frame, start, end in zip(response, times, frame_ends, strict=True)
    )


def _upper_cell(
    frame: dict[str, object], *, start: Fraction, end: Fraction, width: int, height: int
) -> ArcCell:
    head = record(frame.get("headTop"))
    support = record(head.get("upperSupport"))
    if (
        head.get("semanticHeadVerified") is not False
        or head.get("captionPlacementApproved") is not False
        or head.get("contourScope") != "upper-component-in-face-x-range"
        or support.get("status") not in {"interiorMeasured", "canvasClipped", "unavailable"}
    ):
        raise ReplayError("Expected unapproved measured upper-component diagnostics")
    points = head.get("upperContour")
    if not isinstance(points, list) or len(points) > MAX_UPPER_POINTS:
        raise ReplayError("Expected bounded measured upper contour")
    coordinates = tuple(_point(point, width=width, height=height) for point in points)
    if any(left[0] >= right[0] for left, right in pairwise(coordinates)):
        raise ReplayError("Upper contour pixels must be distinct and ordered")
    if support["status"] != "interiorMeasured":
        return ArcCell(start, end, None, 0, len(coordinates))
    minimum = support.get("minimumY")
    covered, area = head.get("faceCoveredPixels"), head.get("facePixels")
    if (
        head.get("status") not in {"candidate", "clippedSupport"}
        or len(coordinates) < 3
        or type(minimum) is not int
        or minimum != min(point[1] for point in coordinates)
        or support.get("contourTouchesTopEdge") is not False
        or support.get("contourTouchesBottomEdge") is not False
        or support.get("contourTouchesSideEdge") is not False
        or any(not 0 < x < width - 1 or not 0 < y < height - 1 for x, y in coordinates)
        or type(covered) is not int
        or type(area) is not int
        or not 0 < covered <= area
        or covered * 2 < area
    ):
        raise ReplayError("Interior upper support contradicts its actual measured pixels")
    return ArcCell(start, end, float(minimum), len(coordinates), 0)


def _point(value: object, *, width: int, height: int) -> tuple[int, int]:
    point = record(value)
    x, y = point.get("x"), point.get("y")
    if type(x) is not int or type(y) is not int or not 0 <= x < width or not 0 <= y < height:
        raise ReplayError("Upper contour must contain actual in-canvas integer pixels")
    return x, y


def image_title_samples(
    title: ET.Element,
    *,
    arc: VisibleArc,
    caption_height: float,
    canvas_height: int,
) -> tuple[tuple[str, str], ...]:
    """Derive centered bottom keys only from covered current image frame cells.

    The partial arc remains an approximate upper support, not semantic hair.
    Existing clock, text, font, rows, entrance and later jitter policy are retained.
    Missing cells require further measurement rather than interpolation or holds.
    """
    start = _time(title.attrib, "offset")
    end = start + _time(title.attrib, "duration")
    local = _time(title.attrib, "start")
    if (
        title.tag != "title"
        or not 0 < caption_height <= canvas_height
        or not start < end
        or not arc.covers(start, end)
    ):
        raise ReplayError("Current upper support must cover the complete measured title")
    times = (start, *(cell.start for cell in arc.cells if start < cell.start < end), end)
    result: list[tuple[str, str]] = []
    for time in times:
        cell = arc.at(time, ending=time == end)
        if cell is None or cell.visible_top is None:
            raise ReplayError("Current upper support is absent at a title boundary")
        bottom = cell.visible_top - HEAD_CAPTION_GAP_PX
        if not caption_height <= bottom <= canvas_height:
            raise ReplayError("Measured title needs height paging above current upper support")
        result.append(
            (rational(local + time - start), f"0 {50 - bottom / canvas_height * 100:.12g}")
        )
    return tuple(result)
