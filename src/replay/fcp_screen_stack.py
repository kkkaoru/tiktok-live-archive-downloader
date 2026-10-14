"""Measured bottom anchors for unknown-head caption stacks within retained cuts."""

from collections.abc import Callable
from dataclasses import dataclass
from fractions import Fraction
from math import isfinite
from xml.etree import ElementTree as ET

from replay.core import ReplayError
from replay.fcp_tracking_caption import MissingFaceRange


@dataclass(frozen=True, slots=True)
class ScreenStack:
    start: Fraction
    end: Fraction
    height: float


def _interval(title: ET.Element) -> tuple[Fraction, Fraction]:
    start = Fraction(title.attrib["offset"].removesuffix("s"))
    end = start + Fraction(title.attrib["duration"].removesuffix("s"))
    if start < 0 or end <= start:
        raise ReplayError("Expected positive caption source interval")
    return start, end


def screen_stacks(
    root: ET.Element,
    *,
    gaps: tuple[MissingFaceRange, ...],
    measured_height: Callable[[ET.Element], float],
    canvas_height: float,
) -> dict[ET.Element, tuple[ScreenStack, ...]]:
    """Reserve actual maximum block height per missing interval and retained cut.

    XML does not retain semantic page identity. This conservative scope therefore
    never infers a page from ASR labels or joins captions across asset-clip cuts.
    Measurements already include the actual uniform fit factor exactly once.
    No face coordinate, source timing or caption text is modified.
    """
    result: dict[ET.Element, tuple[ScreenStack, ...]] = {}
    for clip in root.findall(".//asset-clip"):
        titles = clip.findall("title")
        intervals = {title: _interval(title) for title in titles}
        heights = {title: measured_height(title) for title in titles}
        if any(not isfinite(h) or not 0 < h <= canvas_height for h in heights.values()):
            raise ReplayError("Measured caption stack exceeds canvas")
        stacks = _cut_stacks(intervals, heights=heights, gaps=gaps)
        result.update(
            {
                title: tuple(s for s in stacks if start < s.end and end > s.start)
                for title, (start, end) in intervals.items()
            }
        )
    return result


def _cut_stacks(
    intervals: dict[ET.Element, tuple[Fraction, Fraction]],
    *,
    heights: dict[ET.Element, float],
    gaps: tuple[MissingFaceRange, ...],
) -> tuple[ScreenStack, ...]:
    result: list[ScreenStack] = []
    for gap in gaps:
        overlapping = [
            title
            for title, (start, end) in intervals.items()
            if start < gap.end and end > gap.start
        ]
        if overlapping:
            result.append(ScreenStack(gap.start, gap.end, max(heights[t] for t in overlapping)))
    return tuple(result)
