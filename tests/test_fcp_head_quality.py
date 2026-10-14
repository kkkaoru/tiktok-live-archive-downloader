from dataclasses import replace
from fractions import Fraction as F
from xml.etree import ElementTree as ET

import pytest

from replay.core import ReplayError
from replay.fcp_head_quality import geometry_at, geometry_provenance, title_geometry_provenance
from replay.fcp_tracking_caption import FacePositionSeries, MissingFaceRange, TrackingPosition


@pytest.fixture
def series() -> FacePositionSeries:
    return FacePositionSeries(
        (TrackingPosition(F(0), 0, 400), TrackingPosition(F(10), 0, 400)),
        F(11),
        1080,
        1920,
        derived_ranges=(
            MissingFaceRange(F(0), F(1), 30, F(1), "leading-observation-backfill", "missing"),
            MissingFaceRange(F(2), F(8), 180, F(1), "last-known-hold", "ambiguous"),
        ),
    )


def test_observed_held_and_future_seed_remain_distinct(series: FacePositionSeries) -> None:
    assert geometry_at(series, F(1)).observed
    seed = geometry_at(series, F(0))
    assert not seed.observed and seed.age == 1 and seed.caption_only
    held = geometry_at(series, F(3))
    assert not held.observed and held.age == 2 and held.caption_only
    stale = geometry_at(series, F(4))
    assert stale.caption_only and stale.reason == "ambiguous" and stale.derived_from == 1
    assert geometry_at(series, F(8)).observed


def test_interval_reports_exact_intersections_without_inventing_observations(
    series: FacePositionSeries,
) -> None:
    result = geometry_provenance(series, F(1, 2), F(9))
    assert result["heldDuration"] == "13/2"
    assert result["observedDuration"] == "2"
    assert result["maximumHeldAge"] == "7"
    assert result["geometry"] == "unknown"
    assert result["captionOnlyScreenLayout"] is True
    assert result["headNearVerified"] is False
    assert geometry_provenance(series, F(8), F(9))["geometry"] == ("observed-face-estimate")
    assert geometry_provenance(series, F(9), F(10))["heldDuration"] == "0"


def test_terminal_hold_is_not_an_observed_endpoint(series: FacePositionSeries) -> None:
    tail = replace(
        series,
        derived_ranges=(MissingFaceRange(F(2), F(11), 270, F(1), "last-known-hold", "missing"),),
    )
    assert not geometry_at(tail, F(11)).observed


@pytest.mark.parametrize("time", [F(-1), F(12)])
def test_invalid_clock_rejected(series: FacePositionSeries, time: F) -> None:
    with pytest.raises(ReplayError):
        geometry_at(series, time)


def test_invalid_interval_rejected(series: FacePositionSeries) -> None:
    with pytest.raises(ReplayError):
        geometry_provenance(series, F(2), F(2))


def test_title_report_uses_source_offset_and_retains_terminal_derivation(
    series: FacePositionSeries,
) -> None:
    root = ET.fromstring(
        '<fcpxml><asset-clip><title name="held" offset="4s" start="3600s" duration="1s"/>'
        '<title offset="10s" start="3600s" duration="31/30s"/></asset-clip></fcpxml>'
    )
    reports = title_geometry_provenance(root, series)
    assert reports[0]["titleName"] == "held"
    assert reports[0]["sourceStart"] == "4"
    assert reports[0]["heldDuration"] == "1"
    assert reports[0]["captionOnlyScreenLayout"] is True
    assert reports[1]["sourceEnd"] == "11"
    assert reports[1]["titleSourceEnd"] == "331/30"
    assert reports[1]["terminalCellDerived"] is True
