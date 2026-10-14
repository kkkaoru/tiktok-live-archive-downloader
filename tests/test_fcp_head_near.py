from dataclasses import replace
from fractions import Fraction
from pathlib import Path
from xml.etree import ElementTree as ET

import pytest

from replay import fcp_head_near as head
from replay.core import ReplayError
from replay.fcp_caption_retarget import RetainedSpan
from replay.fcp_screen_stack import ScreenStack
from replay.fcp_tracking_caption import (
    FaceObservation,
    FaceObservationSeries,
    FacePositionSeries,
    MissingFaceRange,
    TrackingCalibration,
    TrackingPosition,
)
from replay.head_height_paging import minimum_headroom
from replay.head_near_placement import FaceSizeObservation
from replay.subtitle_clock import VideoFrameClock
from replay.tracking_observations import TrackingJob
from replay.tracking_retained import RetainedTracking, RetainedWindow


def test_derived_envelope_keeps_raw_geometry_and_uses_observed_width(
    paired: head.FaceScaleSeries,
) -> None:
    source = replace(
        paired,
        sizes=(
            FaceSizeObservation(Fraction(0), 200, 400),
            FaceSizeObservation(Fraction(1), 200, 100),
        ),
    )
    result = source.with_current_scale(reference_time=Fraction(0))
    assert result.estimated_height(Fraction(1)) == 400
    assert result.extent(Fraction(1)).bottom == 500
    assert result.observed_reference(Fraction(1)).bottom == 500
    assert source.scale_heights == ()
    assert result.tops == paired.tops


def test_scale_envelope_requires_paired_widths(paired: head.FaceScaleSeries) -> None:
    with pytest.raises(ReplayError, match="widths"):
        paired.with_current_scale(reference_time=Fraction(0))


def test_current_frame_cell_never_blends_future_scene_scale(
    paired: head.FaceScaleSeries,
) -> None:
    source = replace(
        paired,
        sizes=(
            FaceSizeObservation(Fraction(0), 200, 400),
            FaceSizeObservation(Fraction(1), 100, 100),
        ),
    )
    result = source.with_current_scale(reference_time=Fraction(0))
    assert result.estimated_height(Fraction(1, 4)) == 400
    assert result.estimated_height(Fraction(1, 2)) == 400
    assert result.estimated_height(Fraction(1)) == 200


def test_missing_uses_caption_only_layout_without_old_reference_position(
    paired: head.FaceScaleSeries,
) -> None:
    gap = MissingFaceRange(Fraction(1), Fraction(6), 150, Fraction(0), "last-known-hold", "missing")
    faces = head.FaceScaleSeries(
        replace(
            paired.tops,
            positions=(
                TrackingPosition(Fraction(0), 0, 900),
                TrackingPosition(Fraction(3), 0, 400),
                TrackingPosition(Fraction(5), 0, 400),
            ),
            coverage_end=Fraction(6),
            derived_ranges=(gap,),
        ),
        replace(
            paired.bottoms,
            positions=(
                TrackingPosition(Fraction(0), 0, 1300),
                TrackingPosition(Fraction(3), 0, 1000),
                TrackingPosition(Fraction(5), 0, 1000),
            ),
            coverage_end=Fraction(6),
            derived_ranges=(gap,),
        ),
    )
    result = head.title_samples(
        ET.fromstring('<title offset="3s" start="3600s" duration="1s"/>'),
        faces=faces,
        caption_height=200,
        calibration=TrackingCalibration(Fraction(0), 500, 1, 0, 24, 1920, (0, 0, 1, 1)),
    )
    assert result == (("3600s", "0 38.3333333333"), ("3601s", "0 38.3333333333"))
    assert faces.tops.derived_ranges == (gap,)


def test_scale_clock_mismatch_is_rejected(paired: head.FaceScaleSeries) -> None:
    with pytest.raises(ReplayError, match="sizes"):
        replace(paired, sizes=(FaceSizeObservation(Fraction(3), 100, 100),))
    with pytest.raises(ReplayError, match="Derived"):
        replace(paired, scale_heights=(TrackingPosition(Fraction(3), 0, 100),))


def test_adjacent_missing_stack_boundary_uses_current_interval(
    paired: head.FaceScaleSeries,
) -> None:
    gap = MissingFaceRange(Fraction(0), Fraction(1), 30, Fraction(1), "last-known-hold", "missing")
    faces = head.FaceScaleSeries(
        replace(paired.tops, derived_ranges=(gap,)),
        replace(paired.bottoms, derived_ranges=(gap,)),
    )
    points = head.title_samples(
        ET.fromstring('<title offset="1/2s" start="3600s" duration="1/4s"/>'),
        faces=faces,
        calibration=TrackingCalibration(Fraction(1), 400, 1, 0, 24, 1920, (0, 400, 100, 100)),
        caption_height=100,
        screen_stacks=(
            ScreenStack(Fraction(0), Fraction(1, 2), 340),
            ScreenStack(Fraction(1, 2), Fraction(1), 100),
        ),
    )
    assert points == (("3600s", "0 43.5416666667"), ("14401/4s", "0 43.5416666667"))


@pytest.mark.parametrize("height", [100, 220, 340, 100])
def test_missing_row_growth_preserves_bottom_and_recovery(
    paired: head.FaceScaleSeries, height: float
) -> None:
    gap = MissingFaceRange(Fraction(0), Fraction(1), 30, Fraction(1), "last-known-hold", "missing")
    faces = head.FaceScaleSeries(
        replace(paired.tops, derived_ranges=(gap,)),
        replace(paired.bottoms, derived_ranges=(gap,)),
    )
    title = ET.fromstring('<title offset="0s" start="3600s" duration="2s"/>')
    calibration = TrackingCalibration(Fraction(1), 400, 1, 0, 24, 1920, (0, 400, 100, 100))
    points = head.title_samples(
        title,
        faces=faces,
        calibration=calibration,
        caption_height=height,
        screen_stacks=(ScreenStack(Fraction(0), Fraction(1), 340),),
    )
    assert points[0] == ("3600s", "0 31.0416666667")
    assert points[1] == ("3601s", "0 30.4166666667")
    assert faces.tops.derived_ranges == (gap,)
    with pytest.raises(ReplayError, match="smaller"):
        head.title_samples(
            title,
            faces=faces,
            calibration=calibration,
            caption_height=height,
            screen_stacks=(ScreenStack(Fraction(0), Fraction(1), 50),),
        )


@pytest.mark.parametrize("reference_head", [100, 500])
def test_missing_caption_layout_ignores_reference_then_returns_to_current_head(
    paired: head.FaceScaleSeries, reference_head: int
) -> None:
    gap = MissingFaceRange(
        Fraction(0), Fraction(1), 30, Fraction(1), "leading-observation-backfill", "missing"
    )
    faces = head.FaceScaleSeries(
        replace(paired.tops, derived_ranges=(gap,)),
        replace(paired.bottoms, derived_ranges=(gap,)),
    )
    # Calibration itself must be observed; place it at the recovered frame.
    calibration = TrackingCalibration(
        Fraction(1), min(reference_head, 400), 1, 0, 24, 1920, (0, 400, 100, 100)
    )
    points = head.title_samples(
        ET.fromstring('<title offset="0s" start="3600s" duration="2s"/>'),
        faces=faces,
        calibration=calibration,
        caption_height=100,
    )
    assert points[0] == ("3600s", "0 43.5416666667")
    assert points[1][0] == "3601s"
    assert float(points[1][1].split()[1]) > 30
    assert faces.tops.derived_ranges == (gap,)


def test_width_reader_preserves_missing_and_observed_frames() -> None:
    raw = [
        {
            "presentationValue": 0,
            "presentationTimescale": 30,
            "width": 1080,
            "height": 1920,
            "method": "detection",
            "detectionAttempted": True,
            "faces": [
                {"region": {"x": 100, "y": 200, "width": 300, "height": 400}, "confidence": 0.9}
            ],
        },
        {
            "presentationValue": 1,
            "presentationTimescale": 30,
            "width": 1080,
            "height": 1920,
            "method": "missing",
            "detectionAttempted": True,
            "faces": [],
        },
    ]
    result = head._width_observations(
        raw, VideoFrameClock((0, 1), Fraction(1, 30), 1), width=1080, height=1920
    )
    assert result.samples[0].position == TrackingPosition(Fraction(0), 100, 300)
    assert result.samples[1] == FaceObservation(Fraction(1, 30), None, "missing")


@pytest.fixture
def paired() -> head.FaceScaleSeries:
    return head.FaceScaleSeries(
        FacePositionSeries(
            (TrackingPosition(Fraction(0), 0, 900), TrackingPosition(Fraction(1), 0, 400)),
            Fraction(2),
            1080,
            1920,
        ),
        FacePositionSeries(
            (TrackingPosition(Fraction(0), 0, 1300), TrackingPosition(Fraction(1), 0, 500)),
            Fraction(2),
            1080,
            1920,
            "bottom",
        ),
    )


def test_scale_changes_centered_y_and_preserves_title_clock(paired: head.FaceScaleSeries) -> None:
    title = ET.fromstring('<title offset="0s" start="3600s" duration="2s"/>')
    result = head.title_samples(
        title,
        faces=paired,
        calibration=TrackingCalibration(Fraction(0), 500, 1, 0, 24, 1920, (0, 0, 1, 1)),
        caption_height=200,
    )
    assert result == (("3600s", "0 25.2083333333"), ("3601s", "0 35.625"))
    assert paired.extent(Fraction(3, 2)).top == 400
    assert (
        minimum_headroom(
            paired,
            (RetainedSpan(Fraction(0), Fraction(0), Fraction(0), Fraction(2)),),
            start=Fraction(0),
            end=Fraction(2),
        )
        == 376
    )


@pytest.mark.parametrize("change", ["anchor", "clock", "geometry", "hold", "order"])
def test_pairing_rejects_incompatible_observations(
    paired: head.FaceScaleSeries, change: str
) -> None:
    bottom = paired.bottoms
    if change == "anchor":
        bottom = replace(bottom, anchor="top")
    elif change == "clock":
        bottom = replace(bottom, coverage_end=Fraction(3))
    elif change == "geometry":
        bottom = replace(
            bottom, positions=(TrackingPosition(Fraction(0), 0, 800), *bottom.positions[1:])
        )
    elif change == "hold":
        bottom = replace(
            bottom,
            derived_ranges=(
                MissingFaceRange(
                    Fraction(0),
                    Fraction(1),
                    1,
                    Fraction(1),
                    "leading-observation-backfill",
                    "missing",
                ),
            ),
        )
    else:
        bottom = replace(bottom, positions=tuple(reversed(bottom.positions)))
    with pytest.raises(ReplayError, match="differ"):
        head.FaceScaleSeries(paired.tops, bottom)


def test_unknown_reference_and_insufficient_space_do_not_become_success(
    paired: head.FaceScaleSeries,
) -> None:
    title = ET.fromstring('<title offset="0s" start="3600s" duration="2s"/>')
    calibration = TrackingCalibration(Fraction(0), 500, 1, 0, 24, 1920, (0, 0, 1, 1))
    with pytest.raises(ReplayError, match="paging"):
        head.title_samples(title, faces=paired, calibration=calibration, caption_height=500)
    with pytest.raises(ReplayError, match="exact observed"):
        head.title_samples(
            title,
            faces=paired,
            calibration=replace(calibration, reference_time=Fraction(1, 2)),
            caption_height=200,
        )
    with pytest.raises(ReplayError, match="coverage"):
        paired.extent(Fraction(3))
    with pytest.raises(ReplayError, match="no selected"):
        minimum_headroom(paired, (), start=Fraction(0), end=Fraction(1))


def test_reader_requires_window_references() -> None:
    with pytest.raises(ReplayError, match="bounded"):
        head.read_face_scale([], missing_policy="hold", seed_overlap="next")


def test_held_reference_is_not_observed_calibration(paired: head.FaceScaleSeries) -> None:
    missing = (
        MissingFaceRange(
            Fraction(0), Fraction(1), 1, Fraction(1), "leading-observation-backfill", "missing"
        ),
    )
    held = head.FaceScaleSeries(
        replace(paired.tops, derived_ranges=missing),
        replace(paired.bottoms, derived_ranges=missing),
    )
    with pytest.raises(ReplayError, match="exact observed"):
        held.observed_reference(Fraction(0))
    with pytest.raises(ReplayError, match="exact observed"):
        paired.observed_reference(Fraction(3))


def test_last_display_cell_does_not_extend_observations(paired: head.FaceScaleSeries) -> None:
    limited = head.FaceScaleSeries(
        replace(paired.tops, coverage_end=Fraction(197, 100)),
        replace(paired.bottoms, coverage_end=Fraction(197, 100)),
    )
    title = ET.fromstring('<title offset="0s" start="3600s" duration="2s"/>')
    calibration = TrackingCalibration(Fraction(0), 500, 1, 0, 24, 1920, (0, 0, 1, 1))
    with pytest.raises(ReplayError, match="physical coverage"):
        head.title_samples(title, faces=limited, calibration=calibration, caption_height=200)
    assert head.title_samples(
        title, faces=limited, calibration=calibration, caption_height=200, allow_terminal_cell=True
    ) == (("3600s", "0 25.2083333333"), ("3601s", "0 35.625"))
    assert limited.tops.coverage_end == Fraction(197, 100)
    assert limited.tops.derived_ranges == ()
    assert (
        minimum_headroom(
            limited,
            (RetainedSpan(Fraction(0), Fraction(0), Fraction(0), Fraction(2)),),
            start=Fraction(0),
            end=Fraction(2),
            allow_terminal_cell=True,
        )
        == 376
    )


@pytest.mark.parametrize("historical", [False, True])
def test_reader_pairs_one_verified_payload(
    paired: head.FaceScaleSeries, monkeypatch: pytest.MonkeyPatch, historical: bool
) -> None:
    job = TrackingJob(
        Path("/config"),
        "a" * 64,
        Path("/source"),
        Path("/native"),
        "b" * 64,
        "c" * 64,
        VideoFrameClock((0, 1), Fraction(1), 1),
        1080,
        1920,
        1,
    )
    monkeypatch.setattr(head, "load_job", lambda path: job)
    monkeypatch.setattr(head, "verify", lambda config, evidence: None)
    monkeypatch.setattr(head, "_read", lambda path, limit: b"{}")
    monkeypatch.setattr(head, "_bound_bytes", lambda path, digest, limit: b"{}")
    monkeypatch.setattr(
        head,
        "_width_observations",
        lambda raw, clock, *, width, height: FaceObservationSeries(
            tuple(
                FaceObservation(p.time, TrackingPosition(p.time, p.x, 100))
                for p in paired.tops.positions
            ),
            Fraction(2),
            1080,
            1920,
            "top",
        ),
    )
    monkeypatch.setattr(
        head,
        "face_observation_series",
        lambda raw, clock, *, width, height, anchor: FaceObservationSeries(
            tuple(
                FaceObservation(p.time, p)
                for p in (paired.tops if anchor == "top" else paired.bottoms).positions
            ),
            Fraction(2),
            1080,
            1920,
            anchor,
        ),
    )
    actual_job, result = head.read_face_scale(
        [(Path("/config"), Path("/evidence"))],
        missing_policy="hold",
        seed_overlap="next",
        retained=RetainedTracking(
            (RetainedWindow(job, Path("/evidence"), "a" * 64, "b" * 64, 2, {}),), {}
        )
        if historical
        else None,
    )
    assert actual_job.source == Path("/source")
    assert result.tops.positions[1].y == 400
    assert result.bottoms.positions[1].y == 500
