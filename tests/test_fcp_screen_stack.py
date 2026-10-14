from fractions import Fraction
from xml.etree import ElementTree as ET

import pytest

from replay.core import ReplayError
from replay.fcp_screen_stack import ScreenStack, screen_stacks
from replay.fcp_tracking_caption import MissingFaceRange


def test_measured_rows_stack_upward_and_reset_at_retained_cut() -> None:
    root = ET.fromstring(
        '<fcpxml><asset-clip><title offset="0s" duration="1s" h="100"/>'
        '<title offset="1s" duration="1s" h="220"/>'
        '<title offset="2s" duration="1s" h="340"/>'
        '<title offset="3s" duration="1s" h="100"/></asset-clip>'
        '<asset-clip><title offset="4s" duration="1s" h="100"/></asset-clip></fcpxml>'
    )
    before = ET.tostring(root)
    gap = MissingFaceRange(Fraction(0), Fraction(5), 150, Fraction(6), "last-known-hold", "missing")
    result = screen_stacks(
        root, gaps=(gap,), measured_height=lambda t: float(t.attrib["h"]), canvas_height=1920
    )
    assert tuple(result.values()) == (
        (ScreenStack(Fraction(0), Fraction(5), 340),),
        (ScreenStack(Fraction(0), Fraction(5), 340),),
        (ScreenStack(Fraction(0), Fraction(5), 340),),
        (ScreenStack(Fraction(0), Fraction(5), 340),),
        (ScreenStack(Fraction(0), Fraction(5), 100),),
    )
    assert ET.tostring(root) == before


def test_independent_missing_intervals_and_scaled_height_are_not_merged() -> None:
    root = ET.fromstring(
        '<fcpxml><asset-clip><title offset="0s" duration="1s" h="1000" scale="0.1"/>'
        '<title offset="1s" duration="1s" h="500" scale="1"/>'
        '<title offset="2s" duration="1s" h="300" scale="1"/></asset-clip></fcpxml>'
    )
    gaps = (
        MissingFaceRange(Fraction(0), Fraction(1), 30, Fraction(1), "last-known-hold", "missing"),
        MissingFaceRange(Fraction(2), Fraction(3), 30, Fraction(1), "last-known-hold", "missing"),
    )
    result = screen_stacks(
        root,
        gaps=gaps,
        measured_height=lambda t: float(t.attrib["h"]) * float(t.attrib["scale"]),
        canvas_height=1920,
    )
    assert tuple(result.values()) == (
        (ScreenStack(Fraction(0), Fraction(1), 100),),
        (),
        (ScreenStack(Fraction(2), Fraction(3), 300),),
    )


@pytest.mark.parametrize("height", [0, -1, 2000, float("nan"), float("inf")])
def test_invalid_measurement_rejected(height: float) -> None:
    root = ET.fromstring(
        '<fcpxml><asset-clip><title offset="0s" duration="1s"/></asset-clip></fcpxml>'
    )
    with pytest.raises(ReplayError, match="canvas"):
        screen_stacks(root, gaps=(), measured_height=lambda t: height, canvas_height=1920)


def test_invalid_interval_rejected() -> None:
    root = ET.fromstring(
        '<fcpxml><asset-clip><title offset="0s" duration="0s"/></asset-clip></fcpxml>'
    )
    with pytest.raises(ReplayError, match="interval"):
        screen_stacks(root, gaps=(), measured_height=lambda t: 100, canvas_height=1920)
