from fractions import Fraction
from xml.etree import ElementTree as ET

import pytest

from replay.core import ReplayError
from replay.fcp_visible_arc import ArcCell, VisibleArc
from replay.head_image_support import image_title_samples, measured_upper_cells


@pytest.fixture
def frame() -> dict[str, object]:
    return {
        "requested": {"value": 0, "timescale": 30},
        "actual": {"value": 0, "timescale": 30},
        "width": 100,
        "height": 200,
        "observationPresent": True,
        "instanceLabels": [1],
        "maskThreshold8Bit": 1,
        "captionPlacementApproved": False,
        "unionMask": {
            "width": 100,
            "height": 200,
            "nonzeroPixels": 3000,
            "nonzeroBounds": {"x": 0, "y": 90, "width": 100, "height": 110},
        },
        "headTop": {
            "status": "clippedSupport",
            "semanticHeadVerified": False,
            "captionPlacementApproved": False,
            "topY": None,
            "contourScope": "upper-component-in-face-x-range",
            "faceCoveredPixels": 300,
            "facePixels": 400,
            "upperContour": [{"x": 20, "y": 101}, {"x": 30, "y": 100}, {"x": 40, "y": 102}],
            "upperSupport": {
                "status": "interiorMeasured",
                "minimumY": 100,
                "contourTouchesTopEdge": False,
                "contourTouchesBottomEdge": False,
                "contourTouchesSideEdge": False,
                "componentTouchesTopEdge": False,
                "componentSideContactAtOrAboveFace": True,
            },
        },
    }


def test_lateral_clipped_component_keeps_only_actual_interior_candidate(
    frame: dict[str, object],
) -> None:
    assert measured_upper_cells(
        [frame], times=(Fraction(0),), frame_ends=(Fraction(1, 30),), width=100, height=200
    ) == (ArcCell(Fraction(0), Fraction(1, 30), 100, 3, 0),)
    assert frame["headTop"] == {
        "status": "clippedSupport",
        "semanticHeadVerified": False,
        "captionPlacementApproved": False,
        "topY": None,
        "contourScope": "upper-component-in-face-x-range",
        "faceCoveredPixels": 300,
        "facePixels": 400,
        "upperContour": [{"x": 20, "y": 101}, {"x": 30, "y": 100}, {"x": 40, "y": 102}],
        "upperSupport": {
            "status": "interiorMeasured",
            "minimumY": 100,
            "contourTouchesTopEdge": False,
            "contourTouchesBottomEdge": False,
            "contourTouchesSideEdge": False,
            "componentTouchesTopEdge": False,
            "componentSideContactAtOrAboveFace": True,
        },
    }


@pytest.mark.parametrize("status", ["canvasClipped", "unavailable"])
def test_unsupported_arc_stays_unknown(frame: dict[str, object], status: str) -> None:
    head = frame["headTop"]
    assert isinstance(head, dict)
    support = head["upperSupport"]
    assert isinstance(support, dict)
    support["status"] = status
    assert measured_upper_cells(
        [frame], times=(Fraction(0),), frame_ends=(Fraction(1, 30),), width=100, height=200
    ) == (ArcCell(Fraction(0), Fraction(1, 30), None, 0, 3),)


@pytest.mark.parametrize(
    "patch",
    [
        {"semanticHeadVerified": True},
        {"captionPlacementApproved": True},
        {"contourScope": "rectangle"},
        {"upperSupport": {"status": "invented"}},
        {"upperContour": None},
        {"upperContour": [{"x": 10, "y": 100}] * 3},
        {"upperContour": [{"x": -1, "y": 100}]},
        {"upperContour": [{"x": True, "y": 100}]},
        {"faceCoveredPixels": 1},
        {"facePixels": 0},
        {"faceCoveredPixels": True},
        {"status": "ambiguousComponents"},
        {"upperContour": [{"x": 0, "y": 100}, {"x": 20, "y": 100}, {"x": 30, "y": 100}]},
        {
            "upperSupport": {
                "status": "interiorMeasured",
                "minimumY": 99,
                "contourTouchesTopEdge": False,
                "contourTouchesSideEdge": False,
            }
        },
    ],
)
def test_invalid_pixel_or_semantic_claims_rejected(
    frame: dict[str, object], patch: dict[str, object]
) -> None:
    head = frame["headTop"]
    assert isinstance(head, dict)
    frame["headTop"] = head | patch
    with pytest.raises((ReplayError, ValueError)):
        measured_upper_cells(
            [frame], times=(Fraction(0),), frame_ends=(Fraction(1, 30),), width=100, height=200
        )


@pytest.mark.parametrize(
    "times,ends",
    [
        ((), ()),
        ((Fraction(0),), ()),
        ((Fraction(0),), (Fraction(0),)),
        ((Fraction(1), Fraction(0)), (Fraction(2), Fraction(1))),
        ((Fraction(0), Fraction(1)), (Fraction(2), Fraction(3))),
    ],
)
def test_frame_cells_cannot_extend_into_unknown_clock(
    times: tuple[Fraction, ...], ends: tuple[Fraction, ...]
) -> None:
    with pytest.raises(ReplayError, match="source-frame cells"):
        measured_upper_cells([], times=times, frame_ends=ends, width=100, height=200)


def test_body_clipping_and_face_height_noise_do_not_change_same_upper_pixels(
    frame: dict[str, object],
) -> None:
    head = frame["headTop"]
    assert isinstance(head, dict)
    frame["headTop"] = head | {"faceCoveredPixels": 3000, "facePixels": 4000}
    assert (
        measured_upper_cells(
            [frame], times=(Fraction(0),), frame_ends=(Fraction(1, 30),), width=100, height=200
        )[0].visible_top
        == 100
    )


def test_bottom_alignment_changes_only_with_current_image_motion() -> None:
    arc = VisibleArc(
        (
            ArcCell(Fraction(0), Fraction(1, 30), 100, 3, 0),
            ArcCell(Fraction(1, 30), Fraction(2, 30), 120, 3, 0),
        ),
        {},
    )
    title = ET.fromstring('<title offset="0s" start="3600s" duration="1/15s"/>')
    assert image_title_samples(title, arc=arc, caption_height=50, canvas_height=200) == (
        ("3600s", "0 12"),
        ("108001/30s", "0 2"),
        ("54001/15s", "0 2"),
    )
    assert image_title_samples(title, arc=arc, caption_height=70, canvas_height=200) == (
        ("3600s", "0 12"),
        ("108001/30s", "0 2"),
        ("54001/15s", "0 2"),
    )
    assert title.attrib == {"offset": "0s", "start": "3600s", "duration": "1/15s"}


@pytest.mark.parametrize(
    "arc",
    [
        VisibleArc((), {}),
        VisibleArc((ArcCell(Fraction(0), Fraction(1, 30), 100, 3, 0),), {}),
        VisibleArc((ArcCell(Fraction(0), Fraction(1, 15), None, 0, 3),), {}),
        VisibleArc(
            (
                ArcCell(Fraction(0), Fraction(1, 30), 100, 3, 0),
                ArcCell(Fraction(1, 20), Fraction(1, 15), 100, 3, 0),
            ),
            {},
        ),
    ],
)
def test_no_title_hold_or_interpolation_across_missing_images(arc: VisibleArc) -> None:
    title = ET.fromstring('<title offset="0s" start="3600s" duration="1/15s"/>')
    with pytest.raises(ReplayError, match="complete measured title"):
        image_title_samples(title, arc=arc, caption_height=50, canvas_height=200)


def test_insufficient_space_needs_real_height_paging() -> None:
    title = ET.fromstring('<title offset="0s" start="3600s" duration="1/15s"/>')
    arc = VisibleArc((ArcCell(Fraction(0), Fraction(1, 15), 100, 3, 0),), {})
    with pytest.raises(ReplayError, match="height paging"):
        image_title_samples(title, arc=arc, caption_height=80, canvas_height=200)
