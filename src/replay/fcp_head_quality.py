"""Keep observed face geometry separate from bounded candidate placement."""

from bisect import bisect_right
from dataclasses import dataclass
from fractions import Fraction
from xml.etree import ElementTree as ET

from replay.core import ReplayError
from replay.fcp_tracking_caption import FacePositionSeries, _time


@dataclass(frozen=True, slots=True)
class GeometryQuality:
    observed: bool
    age: Fraction
    caption_only: bool
    derived_from: Fraction | None = None
    method: str | None = None
    reason: str | None = None


def _validate(series: FacePositionSeries, time: Fraction) -> None:
    if (
        not isinstance(time, Fraction)
        or not series.positions
        or not series.positions[0].time <= time <= series.coverage_end
    ):
        raise ReplayError("Expected an exact supported face clock")


def geometry_at(series: FacePositionSeries, time: Fraction) -> GeometryQuality:
    """Every held/backfilled frame uses caption-only layout, never old coordinates."""
    _validate(series, time)
    index = bisect_right(series.derived_ranges, time, key=lambda gap: gap.start) - 1
    if index >= 0:
        gap = series.derived_ranges[index]
        if gap.start <= time < gap.end or time == gap.end == series.coverage_end:
            age = abs(time - gap.derived_from)
            return GeometryQuality(False, age, True, gap.derived_from, gap.method, gap.reason)
    return GeometryQuality(True, Fraction(0), False)


def geometry_provenance(
    series: FacePositionSeries,
    start: Fraction,
    end: Fraction,
) -> dict[str, object]:
    """Summarize an exact source interval without relabeling derived positions."""
    _validate(series, start)
    _validate(series, end)
    if start >= end:
        raise ReplayError("Expected a positive caption geometry interval")
    held = Fraction(0)
    max_age = Fraction(0)
    ranges: list[dict[str, object]] = []
    index = max(0, bisect_right(series.derived_ranges, start, key=lambda gap: gap.start) - 1)
    for gap in series.derived_ranges[index:]:
        if gap.start >= end:
            break
        left, right = max(start, gap.start), min(end, gap.end)
        if left >= right:
            continue
        age = max(abs(left - gap.derived_from), abs(right - gap.derived_from))
        held += right - left
        max_age = max(max_age, age)
        ranges.append(
            {
                "sourceStart": str(left),
                "sourceEnd": str(right),
                "derivedFromObservedTime": str(gap.derived_from),
                "maximumAge": str(age),
                "method": gap.method,
                "reason": gap.reason,
                "placementPolicy": "caption-only-screen-layout",
            }
        )
    return {
        "sourceStart": str(start),
        "sourceEnd": str(end),
        "observedDuration": str(end - start - held),
        "heldDuration": str(held),
        "maximumHeldAge": str(max_age),
        "geometry": "unknown" if ranges else "observed-face-estimate",
        "headNearVerified": False,
        "candidateOnly": True,
        "captionOnlyScreenLayout": bool(ranges),
        "faceClearanceVerified": False,
        "derivedRanges": ranges,
    }


def title_geometry_provenance(
    root: ET.Element, series: FacePositionSeries
) -> list[dict[str, object]]:
    """Bind each generated title to its source-clock geometry evidence."""
    reports: list[dict[str, object]] = []
    for index, title in enumerate(root.findall(".//asset-clip/title")):
        start = _time(title.get("offset"))
        end = start + _time(title.get("duration"))
        report = geometry_provenance(series, start, min(end, series.coverage_end))
        reports.append(
            {
                **report,
                "titleIndex": index,
                "titleName": title.get("name"),
                "titleSourceEnd": str(end),
                "terminalCellDerived": end > series.coverage_end,
            }
        )
    return reports
