import json
import sys
from dataclasses import replace
from fractions import Fraction
from pathlib import Path
from xml.etree import ElementTree as ET

import pytest
from test_fcp_head_near import paired as paired

from replay import head_geometry_audit as audit
from replay.core import ReplayError
from replay.fcp_head_near import FaceScaleSeries
from replay.fcp_tracking_caption import MissingFaceRange, TrackingCalibration
from replay.head_near_placement import FaceSizeObservation
from replay.stages import file_digest
from replay.subtitle_clock import VideoFrameClock
from replay.tracking_observations import TrackingJob


@pytest.fixture
def root() -> ET.Element:
    return ET.fromstring(
        '<fcpxml><resources><format id="r1" frameDuration="1/30s"/></resources>'
        '<library><event><project><sequence format="r1"><spine>'
        '<asset-clip offset="0s" start="0s" duration="2s">'
        '<title name="one" offset="0s" start="3600s" duration="2s">'
        "<text>甲\n乙</text><text>甲\n乙</text><text>甲\n乙</text><adjust-transform>"
        '<param name="position"><keyframeAnimation>'
        '<keyframe time="3600s" value="0 20"/><keyframe time="3602s" value="0 40"/>'
        "</keyframeAnimation></param></adjust-transform></title></asset-clip>"
        "</spine></sequence></project></event></library></fcpxml>"
    )


@pytest.fixture
def scaled(paired: FaceScaleSeries) -> FaceScaleSeries:
    return replace(
        paired,
        sizes=(
            FaceSizeObservation(Fraction(0), 200, 400),
            FaceSizeObservation(Fraction(1), 200, 100),
        ),
    ).with_current_scale(reference_time=Fraction(0))


def test_clock_face_estimate_and_xml_position_remain_distinct(
    root: ET.Element, scaled: FaceScaleSeries
) -> None:
    result = audit.sample_geometry(
        root,
        output_time=Fraction(1, 2),
        faces=scaled,
        calibration=TrackingCalibration(Fraction(0), 500, 1, 0, 24, 1920, (0, 900, 200, 400)),
    )
    assert result == {
        "outputTime": "1/2",
        "sourceTime": "1/2",
        "title": "one",
        "text": "甲\n乙",
        "lineCount": 2,
        "observedFace": True,
        "heldAge": "0",
        "faceTop": 650,
        "faceBottom": 900,
        "faceHeightAtSample": 250,
        "scaleObservationTime": "0",
        "scaleObservationWidth": 200,
        "scaleObservationHeight": 400,
        "estimatedScaleHeight": 400,
        "estimatedHeadTop": 250,
        "estimatedUnfittedBottom": 226,
        "serializedY": 25,
        "serializedBottom": 480,
        "headNearVerified": False,
        "nativePaintVerified": False,
    }


@pytest.mark.parametrize("time", [Fraction(-1), Fraction(2)])
def test_uncovered_output_clock_rejected(
    root: ET.Element, scaled: FaceScaleSeries, time: Fraction
) -> None:
    with pytest.raises(ReplayError, match="source clip"):
        audit.sample_geometry(
            root,
            output_time=time,
            faces=scaled,
            calibration=TrackingCalibration(Fraction(0), 500, 1, 0, 24, 1920, (0, 0, 1, 1)),
        )


@pytest.mark.parametrize("value", ["1 20", "0 inf", "0 51", "0", "x y"])
def test_invalid_centered_position_rejected(value: str) -> None:
    title = ET.fromstring(f'<title><adjust-transform position="{value}"/></title>')
    with pytest.raises(ReplayError):
        audit.caption_y(title, Fraction(0))


def test_static_position_and_ambiguous_channel(root: ET.Element) -> None:
    title = ET.fromstring('<title><adjust-transform position="0 12"/></title>')
    assert audit.caption_y(title, Fraction(0)) == 12
    actual = root.find(".//title")
    assert actual is not None
    transform = actual.find("adjust-transform")
    assert transform is not None
    transform.set("position", "0 12")
    with pytest.raises(ReplayError, match="representation"):
        audit.caption_y(actual, Fraction(0))


@pytest.mark.parametrize("change", ["empty", "duplicate", "outside", "order"])
def test_invalid_key_coverage(root: ET.Element, change: str) -> None:
    title = root.find(".//title")
    assert title is not None
    keys = title.findall(".//keyframe")
    if change == "empty":
        keys[0].set("value", "0")
    elif change == "duplicate":
        keys[1].set("time", "3600s")
    elif change == "order":
        keys[1].set("time", "3599s")
    else:
        keys[0].set("time", "3601s")
    with pytest.raises(ReplayError):
        audit.caption_y(title, Fraction(0))


def test_missing_caption_is_rejected(root: ET.Element, scaled: FaceScaleSeries) -> None:
    title = root.find(".//title")
    assert title is not None
    title.set("offset", "1s")
    with pytest.raises(ReplayError, match="one caption"):
        audit.sample_geometry(
            root,
            output_time=Fraction(1, 2),
            faces=scaled,
            calibration=TrackingCalibration(Fraction(0), 500, 1, 0, 24, 1920, (0, 0, 1, 1)),
        )


def test_three_layer_text_is_required(root: ET.Element, scaled: FaceScaleSeries) -> None:
    text = root.find(".//text")
    assert text is not None
    text.text = "丙"
    with pytest.raises(ReplayError, match="three-layer"):
        audit.sample_geometry(
            root,
            output_time=Fraction(1, 2),
            faces=scaled,
            calibration=TrackingCalibration(Fraction(0), 500, 1, 0, 24, 1920, (0, 0, 1, 1)),
        )


def test_cli_readonly_pinned_report(
    root: ET.Element, scaled: FaceScaleSeries, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    xml = tmp_path / "source.fcpxml"
    xml.write_bytes(ET.tostring(root))
    index = tmp_path / "index.json"
    index.write_text("{}", encoding="utf-8")
    refresh = tmp_path / "refresh.json"
    refresh.write_text(
        json.dumps(
            {
                "faceIndex": {"path": str(index), "sha256": file_digest(index)},
                "referenceTime": "0",
                "referenceHeadTop": 500,
                "referenceROI": [0, 900, 200, 400],
            }
        ),
        encoding="utf-8",
    )
    config = tmp_path / "config.json"
    config.write_text(
        json.dumps(
            {
                "version": 1,
                "frames": [15],
                "refresh": {"path": str(refresh), "sha256": file_digest(refresh)},
                "xml": {"path": str(xml), "sha256": file_digest(xml)},
            }
        ),
        encoding="utf-8",
    )
    monkeypatch.setattr(audit, "face_references", lambda _path: [])
    evidence = tmp_path / "evidence"
    evidence.mkdir()
    (evidence / "response.json").write_text("{}", encoding="utf-8")
    (evidence / "provenance.json").write_text("{}", encoding="utf-8")
    job = TrackingJob(
        refresh, "a", xml, xml, "a", "a", VideoFrameClock((0, 1), Fraction(1), 1), 1080, 1920, 0.5
    )
    monkeypatch.setattr(audit, "face_references", lambda _path: [(refresh, evidence)])
    monkeypatch.setattr(audit, "load_job", lambda _path: job)
    monkeypatch.setattr(audit, "read_face_scale", lambda _paths, **_kwargs: (job, scaled))
    output = tmp_path / "result.json"
    monkeypatch.setattr(sys, "argv", ["audit", str(config), str(output)])
    audit.main()
    assert json.loads(output.read_text())["samples"][0]["serializedBottom"] == 480
    assert file_digest(xml) == "d03b3842217d26a120c2a7cf61d424bd7db37e9c3d0e9f8768c6749b32d2e7bd"


@pytest.mark.parametrize("frames", [[], [True], [1, 1], [-1], "1"])
def test_bad_sample_configuration(tmp_path: Path, frames: object) -> None:
    config = tmp_path / "config.json"
    config.write_text(json.dumps({"version": 1, "frames": frames}), encoding="utf-8")
    with pytest.raises(ReplayError, match="samples"):
        audit.audit(config, tmp_path / "result.json")


def test_bounded_audit_rejects_unpinned_automatic_reference(
    root: ET.Element, tmp_path: Path
) -> None:
    with pytest.raises(ReplayError, match="explicit observed reference"):
        audit._bounded_samples(root, refresh={}, index=tmp_path, times=(Fraction(0),), pins={})


def test_window_selection_is_half_open_and_does_not_span_unobserved_gap(tmp_path: Path) -> None:
    first = TrackingJob(
        tmp_path,
        "a",
        tmp_path,
        tmp_path,
        "a",
        "a",
        VideoFrameClock((0, 1), Fraction(1), 1),
        100,
        200,
        0.5,
    )
    second = replace(first, clock=VideoFrameClock((3, 4), Fraction(1), 1))
    assert audit._window_index((first, second), Fraction(1)) == 0
    assert audit._window_index((first, second), Fraction(3)) == 1
    with pytest.raises(ReplayError, match="window coverage"):
        audit._window_index((first, second), Fraction(2))


def test_unknown_face_not_reported_as_observed(root: ET.Element, scaled: FaceScaleSeries) -> None:
    gap = MissingFaceRange(
        Fraction(0), Fraction(1), 30, Fraction(1), "leading-observation-backfill", "missing"
    )
    faces = replace(
        scaled,
        tops=replace(scaled.tops, derived_ranges=(gap,)),
        bottoms=replace(scaled.bottoms, derived_ranges=(gap,)),
    )
    result = audit.sample_geometry(
        root,
        output_time=Fraction(1, 2),
        faces=faces,
        calibration=TrackingCalibration(Fraction(0), 500, 1, 0, 24, 1920, (0, 0, 1, 1)),
        reference_face=scaled.extent(Fraction(0)),
    )
    assert result["observedFace"] is False
    assert result["estimatedHeadTop"] is None
    assert result["estimatedUnfittedBottom"] is None
