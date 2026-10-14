import json
import sys
from dataclasses import replace
from fractions import Fraction
from pathlib import Path
from xml.etree import ElementTree as ET

import pytest
from test_fcp_head_near import paired as paired

from replay import head_image_seed_plan as seeds
from replay import tracking_observations as storage
from replay.core import ReplayError
from replay.fcp_head_near import FaceScaleSeries
from replay.fcp_tracking_caption import MissingFaceRange, TrackingPosition
from replay.fcp_visible_arc import ArcCell, VisibleArc
from replay.head_approx_placement import ApproxPlan, CurrentFaceCell, MeasurementRequest
from replay.head_image_collection import ImageJob
from replay.head_image_plan import ImageFrameCell, caption_spans
from replay.head_image_retained import RetainedUpperPixels
from replay.head_near_placement import FaceSizeObservation
from replay.recordings import record
from replay.stages import file_digest
from replay.subtitle_clock import VideoFrameClock
from replay.tracking_retained import RetainedTracking


@pytest.fixture
def faces(paired: FaceScaleSeries) -> FaceScaleSeries:
    return replace(
        paired,
        sizes=(
            FaceSizeObservation(Fraction(0), 200, 400),
            FaceSizeObservation(Fraction(1), 100, 100),
        ),
    )


@pytest.fixture
def stable(faces: FaceScaleSeries) -> FaceScaleSeries:
    return replace(
        faces,
        tops=replace(
            faces.tops,
            positions=(
                TrackingPosition(Fraction(0), 0, 900),
                TrackingPosition(Fraction(1), 0, 904),
            ),
        ),
        bottoms=replace(
            faces.bottoms,
            positions=(
                TrackingPosition(Fraction(0), 0, 1300),
                TrackingPosition(Fraction(1), 0, 1304),
            ),
        ),
        sizes=(
            FaceSizeObservation(Fraction(0), 200, 400),
            FaceSizeObservation(Fraction(1), 200, 400),
        ),
    )


@pytest.fixture
def job(tmp_path: Path) -> storage.TrackingJob:
    source = tmp_path / "source.mov"
    native = tmp_path / "native"
    config = tmp_path / "tracking.json"
    source.write_bytes(b"synthetic source")
    native.write_bytes(b"synthetic native")
    native.chmod(0o700)
    config.write_text("{}", encoding="utf-8")
    return storage.TrackingJob(
        config,
        file_digest(config),
        source,
        native,
        file_digest(source),
        file_digest(native),
        VideoFrameClock((0, 1), Fraction(1), 1),
        1080,
        1920,
        1,
    )


@pytest.fixture
def config(
    job: storage.TrackingJob,
    faces: FaceScaleSeries,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> Path:
    xml = tmp_path / "timeline.fcpxml"
    xml.write_text(
        "<fcpxml><library><event><project><sequence><spine>"
        '<asset-clip start="0s" duration="2s"><title offset="0s" duration="2s"/>'
        "</asset-clip></spine></sequence></project></event></library></fcpxml>",
        encoding="utf-8",
    )
    index = tmp_path / "index.json"
    index.write_text("{}", encoding="utf-8")
    sandbox = tmp_path / "sandbox.sb"
    sandbox.write_text("synthetic sandbox", encoding="utf-8")
    evidence = tmp_path / "tracking"
    evidence.mkdir()
    (evidence / "request.json").write_text("{}", encoding="utf-8")
    (evidence / "response.json").write_text("{}", encoding="utf-8")
    (evidence / "provenance.json").write_text("{}", encoding="utf-8")
    path = tmp_path / "seed-config.json"
    path.write_text(
        json.dumps(
            {
                "version": 1,
                "xml": {"path": str(xml), "sha256": file_digest(xml)},
                "faceIndex": {"path": str(index), "sha256": file_digest(index)},
                "native": {"path": str(job.native), "sha256": file_digest(job.native)},
                "sandbox": {"path": str(sandbox), "sha256": file_digest(sandbox)},
                "zoomResetTimes": [],
            }
        ),
        encoding="utf-8",
    )
    monkeypatch.setattr(seeds, "WORK_ROOT", tmp_path)
    monkeypatch.setattr(
        seeds, "selected_references", lambda _index, _scopes: [(job.config, evidence)]
    )
    monkeypatch.setattr(storage, "load_job", lambda _config: job)
    monkeypatch.setattr(seeds, "read_face_scale", lambda *_args, **_kwargs: (job, faces))
    return path


def test_current_shape_and_y_changes_request_real_pixels(faces: FaceScaleSeries) -> None:
    result = seeds.build_requests(
        faces,
        scopes=(seeds.SourceScope(Fraction(0), Fraction(2), ((Fraction(0), Fraction(2)),)),),
        pixels=VisibleArc((), {}),
    )
    assert result.cells == (
        ImageFrameCell(Fraction(0), Fraction(1), (0, 900, 200, 400)),
        ImageFrameCell(Fraction(1), Fraction(2), (0, 400, 100, 100)),
    )
    assert result.cost() == {
        "newRequestedCells": 2,
        "missingFaceRequests": 0,
        "captionFreeRequests": 0,
        "reusedPixelSeeds": 0,
        "unsupportedScopedClockCells": 0,
        "unsupportedCaptionClockCells": 0,
        "chunksAt128Cells": 1,
        "nativeBatchesAt8Cells": 1,
        "nativeSecondsEstimate": "1/3",
        "throughputBasisCellsPerSecond": 6,
        "workflowElapsedEstimate": None,
        "nativeMeasurementsStarted": False,
        "candidateOnly": True,
        "headNearVerified": False,
    }


def test_title_gap_defers_measurement_to_an_actual_required_frame(stable: FaceScaleSeries) -> None:
    result = seeds.build_requests(
        stable,
        scopes=(seeds.SourceScope(Fraction(0), Fraction(2), ((Fraction(5, 4), Fraction(3, 2)),)),),
        pixels=VisibleArc((), {}),
    )
    assert result.cells == (ImageFrameCell(Fraction(1), Fraction(2), (0, 904, 200, 400)),)
    assert result.missing == () and result.unused_requests == 0


def test_caption_free_changes_need_no_pixel_request(faces: FaceScaleSeries) -> None:
    result = seeds.build_requests(
        faces,
        scopes=(seeds.SourceScope(Fraction(0), Fraction(2), ((Fraction(0), Fraction(1, 2)),)),),
        pixels=VisibleArc((), {}),
    )
    assert result.cells == (ImageFrameCell(Fraction(0), Fraction(1), (0, 900, 200, 400)),)
    assert result.unused_requests == 1


def test_missing_does_not_gain_observed_roi_or_head_approval(faces: FaceScaleSeries) -> None:
    missing = MissingFaceRange(
        Fraction(0), Fraction(1), 1, Fraction(0), "last-known-hold", "missing"
    )
    source = replace(
        faces,
        tops=replace(faces.tops, derived_ranges=(missing,)),
        bottoms=replace(faces.bottoms, derived_ranges=(missing,)),
        sizes=(FaceSizeObservation(Fraction(0), 200, 400, False), faces.sizes[1]),
    )
    result = seeds.build_requests(
        source,
        scopes=(seeds.SourceScope(Fraction(0), Fraction(2), ((Fraction(0), Fraction(2)),)),),
        pixels=VisibleArc((), {}),
    )
    assert result.cells == (ImageFrameCell(Fraction(1), Fraction(2), (0, 400, 100, 100)),)
    assert result.missing[0].encoded() == {
        "sourceFrameStart": "0",
        "sourceFrameEnd": "1",
        "unsupportedStart": "0",
        "unsupportedEnd": "1",
        "face": None,
        "reason": "missing-face",
        "kind": "current-face-observation",
    }
    assert result.cost()["headNearVerified"] is False


def test_cut_and_zoom_reset_require_new_seed(stable: FaceScaleSeries) -> None:
    pixels = VisibleArc((ArcCell(Fraction(0), Fraction(1), 700, 3, 0),), {})
    scopes = (
        seeds.SourceScope(Fraction(0), Fraction(1), ((Fraction(0), Fraction(1)),)),
        seeds.SourceScope(Fraction(1), Fraction(2), ((Fraction(1), Fraction(2)),)),
    )
    result = seeds.build_requests(stable, scopes=scopes, pixels=pixels)
    assert result.cells == (ImageFrameCell(Fraction(1), Fraction(2), (0, 904, 200, 400)),)
    assert result.reused_seeds == 1
    zoom = seeds.build_requests(
        stable,
        scopes=(seeds.SourceScope(Fraction(0), Fraction(2), ((Fraction(0), Fraction(2)),)),),
        pixels=pixels,
        resets=(Fraction(1),),
    )
    assert zoom.cells == (ImageFrameCell(Fraction(1), Fraction(2), (0, 904, 200, 400)),)


def test_supported_pixels_reuse_preserves_exact_seed_and_avoids_new_measurement(
    stable: FaceScaleSeries,
) -> None:
    result = seeds.build_requests(
        stable,
        scopes=(seeds.SourceScope(Fraction(0), Fraction(2), ((Fraction(0), Fraction(2)),)),),
        pixels=VisibleArc((ArcCell(Fraction(0), Fraction(1), 700, 3, 0),), {}),
    )
    assert result.cells == () and result.reused_seeds == 1
    assert result.cost()["nativeSecondsEstimate"] == "0"


@pytest.fixture
def long_clock(faces: FaceScaleSeries) -> FaceScaleSeries:
    return replace(
        faces,
        tops=replace(
            faces.tops,
            positions=(
                TrackingPosition(Fraction(0), 0, 900),
                TrackingPosition(Fraction(1), 0, 902),
                TrackingPosition(Fraction(4), 0, 901),
                TrackingPosition(Fraction(5), 0, 904),
            ),
            coverage_end=Fraction(6),
        ),
        bottoms=replace(
            faces.bottoms,
            positions=(
                TrackingPosition(Fraction(0), 0, 1300),
                TrackingPosition(Fraction(1), 0, 1302),
                TrackingPosition(Fraction(4), 0, 1301),
                TrackingPosition(Fraction(5), 0, 1304),
            ),
            coverage_end=Fraction(6),
        ),
        sizes=(
            FaceSizeObservation(Fraction(0), 200, 400),
            FaceSizeObservation(Fraction(1), 200, 400),
            FaceSizeObservation(Fraction(4), 200, 400),
            FaceSizeObservation(Fraction(5), 200, 400),
        ),
    )


def test_long_actual_cell_preserves_observed_flag_and_requires_fresh_seed_after_gap(
    long_clock: FaceScaleSeries,
) -> None:
    result = seeds.build_requests(
        long_clock,
        scopes=(seeds.SourceScope(Fraction(0), Fraction(6), ((Fraction(0), Fraction(6)),)),),
        pixels=VisibleArc((ArcCell(Fraction(0), Fraction(1), 700, 3, 0),), {}),
    )
    assert result.cells == (ImageFrameCell(Fraction(4), Fraction(5), (0, 901, 200, 400)),)
    assert result.missing == ()
    assert result.unsupported[0].encoded() == {
        "sourceFrameStart": "1",
        "sourceFrameEnd": "4",
        "scopeStart": "0",
        "scopeEnd": "6",
        "currentFaceObserved": True,
        "titleOverlaps": [{"start": "1", "end": "4"}],
        "kind": "source-clock-support",
        "reason": "source-cell-exceeds-native-duration",
        "nativeCellBudgetSeconds": "1",
        "displayPersistenceVerified": False,
        "headNearVerified": False,
    }
    assert result.cost()["unsupportedCaptionClockCells"] == 1


def test_long_caption_free_cell_does_not_create_native_or_missing_request(
    long_clock: FaceScaleSeries,
) -> None:
    result = seeds.build_requests(
        long_clock,
        scopes=(
            seeds.SourceScope(
                Fraction(0), Fraction(6), ((Fraction(0), Fraction(1)), (Fraction(4), Fraction(6)))
            ),
        ),
        pixels=VisibleArc((), {}),
    )
    assert result.cells == (
        ImageFrameCell(Fraction(0), Fraction(1), (0, 900, 200, 400)),
        ImageFrameCell(Fraction(4), Fraction(5), (0, 901, 200, 400)),
    )
    assert result.unsupported[0].title_overlaps == ()
    assert result.cost()["unsupportedCaptionClockCells"] == 0
    assert result.missing == ()


def test_title_only_inside_long_frame_remains_unsupported_without_fabricated_pts(
    long_clock: FaceScaleSeries,
) -> None:
    result = seeds.build_requests(
        long_clock,
        scopes=(seeds.SourceScope(Fraction(2), Fraction(3), ((Fraction(2), Fraction(3)),)),),
        pixels=VisibleArc((), {}),
    )
    assert result.cells == () and result.missing == ()
    assert result.unsupported[0].source == CurrentFaceCell(
        Fraction(1), Fraction(4), (0, 902, 200, 400)
    )
    assert result.unsupported[0].title_overlaps == ((Fraction(2), Fraction(3)),)


def test_missing_face_long_clock_is_distinct_from_observed_long_clock(
    long_clock: FaceScaleSeries,
) -> None:
    gap = MissingFaceRange(Fraction(1), Fraction(4), 1, Fraction(0), "last-known-hold", "missing")
    source = replace(
        long_clock,
        tops=replace(long_clock.tops, derived_ranges=(gap,)),
        bottoms=replace(long_clock.bottoms, derived_ranges=(gap,)),
        sizes=(
            long_clock.sizes[0],
            replace(long_clock.sizes[1], observed=False),
            *long_clock.sizes[2:],
        ),
    )
    result = seeds.build_requests(
        source,
        scopes=(seeds.SourceScope(Fraction(1), Fraction(4), ((Fraction(1), Fraction(4)),)),),
        pixels=VisibleArc((), {}),
    )
    assert result.cells == ()
    assert result.unsupported[0].encoded()["currentFaceObserved"] is False


@pytest.mark.parametrize("duration", [Fraction(1), Fraction(1001, 1000), Fraction(6)])
def test_long_cell_threshold_uses_original_fraction_without_guard_widening(
    faces: FaceScaleSeries, duration: Fraction
) -> None:
    source = replace(
        faces,
        tops=replace(
            faces.tops,
            positions=(faces.tops.positions[0], TrackingPosition(duration, 0, 900)),
            coverage_end=duration + 1,
        ),
        bottoms=replace(
            faces.bottoms,
            positions=(faces.bottoms.positions[0], TrackingPosition(duration, 0, 1300)),
            coverage_end=duration + 1,
        ),
        sizes=(faces.sizes[0], FaceSizeObservation(duration, 200, 400)),
    )
    result = seeds.build_requests(
        source,
        scopes=(seeds.SourceScope(Fraction(0), duration, ((Fraction(0), duration),)),),
        pixels=VisibleArc((), {}),
    )
    assert len(result.unsupported) == (0 if duration == 1 else 1)
    assert len(result.cells) == (1 if duration == 1 else 0)


@pytest.mark.parametrize("resets", [(Fraction(1), Fraction(1)), (Fraction(1, 2),)])
def test_zoom_requires_distinct_actual_clock_starts(
    faces: FaceScaleSeries, resets: tuple[Fraction, ...]
) -> None:
    with pytest.raises(ReplayError, match="reset"):
        seeds.build_requests(
            faces,
            scopes=(seeds.SourceScope(Fraction(0), Fraction(2), ((Fraction(0), Fraction(2)),)),),
            pixels=VisibleArc((), {}),
            resets=resets,
        )


def test_full_canvas_and_fractional_cut_keep_original_cells(faces: FaceScaleSeries) -> None:
    source = replace(
        faces,
        tops=replace(
            faces.tops,
            positions=(
                TrackingPosition(Fraction(0), 0, 900),
                TrackingPosition(Fraction(1), 0, 904),
            ),
        ),
        bottoms=replace(
            faces.bottoms,
            positions=(
                TrackingPosition(Fraction(0), 0, 1300),
                TrackingPosition(Fraction(1), 0, 1304),
            ),
        ),
        sizes=(
            FaceSizeObservation(Fraction(0), 1080, 400),
            FaceSizeObservation(Fraction(1), 1080, 400),
        ),
    )
    result = seeds.build_requests(
        source,
        scopes=(
            seeds.SourceScope(Fraction(1, 2), Fraction(3, 2), ((Fraction(1, 2), Fraction(3, 2)),)),
        ),
        pixels=VisibleArc((), {}),
    )
    assert result.cells == (ImageFrameCell(Fraction(0), Fraction(1), (0, 900, 1080, 400)),)


@pytest.mark.parametrize("time", [Fraction(-1), Fraction(1)])
def test_target_cannot_invent_a_frame(time: Fraction) -> None:
    with pytest.raises(ReplayError, match="actual current"):
        seeds._target((CurrentFaceCell(Fraction(0), Fraction(1), (0, 900, 200, 400)),), time)


def test_target_cannot_use_missing_face() -> None:
    with pytest.raises(ReplayError, match="missing"):
        seeds._target((CurrentFaceCell(Fraction(0), Fraction(1), None),), Fraction(0))


@pytest.mark.parametrize("fault", ["duplicate", "overlap"])
def test_conflicting_or_overlapping_requests_fail(
    faces: FaceScaleSeries, monkeypatch: pytest.MonkeyPatch, fault: str
) -> None:
    scope = seeds.SourceScope(Fraction(0), Fraction(2), ((Fraction(0), Fraction(2)),))
    request = MeasurementRequest(
        Fraction(0), Fraction(1), Fraction(0), Fraction(1), (0, 900, 200, 400), "scope-start"
    )
    monkeypatch.setattr(seeds, "scoped_plan", lambda *_args, **_kwargs: ApproxPlan((), (request,)))
    calls = iter(
        (
            ImageFrameCell(Fraction(0), Fraction(1), (0, 900, 200, 400)),
            ImageFrameCell(
                Fraction(0 if fault == "duplicate" else 1, 2), Fraction(1), (0, 900, 201, 400)
            ),
        )
    )
    monkeypatch.setattr(seeds, "_target", lambda *_args: next(calls))
    with pytest.raises(ReplayError, match="conflicting|overlap"):
        seeds.build_requests(faces, scopes=(scope, scope), pixels=VisibleArc((), {}))


@pytest.mark.parametrize("fault", ["empty", "titles", "cells"])
def test_full_planner_budgets_remain_finite(
    faces: FaceScaleSeries, monkeypatch: pytest.MonkeyPatch, fault: str
) -> None:
    if fault == "titles":
        monkeypatch.setattr(seeds, "MAX_TITLES", 0)
    if fault == "cells":
        monkeypatch.setattr(seeds, "MAX_CELLS", 1)
    scope = seeds.SourceScope(Fraction(0), Fraction(2), ((Fraction(0), Fraction(2)),))
    with pytest.raises(ReplayError, match="budget"):
        seeds.build_requests(
            faces, scopes=() if fault == "empty" else (scope,), pixels=VisibleArc((), {})
        )


def test_repeated_source_scopes_require_partition_before_excessive_work(
    faces: FaceScaleSeries, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(seeds, "MAX_SCOPED_CELLS", 3)
    scope = seeds.SourceScope(Fraction(0), Fraction(2), ((Fraction(0), Fraction(2)),))
    with pytest.raises(ReplayError, match="work budget"):
        seeds.build_requests(faces, scopes=(scope, scope), pixels=VisibleArc((), {}))


@pytest.mark.parametrize(
    "titles",
    [
        (),
        ((Fraction(1), Fraction(1)),),
        ((Fraction(-1), Fraction(1)),),
        ((Fraction(0), Fraction(1)), (Fraction(1), Fraction(2))),
    ],
)
def test_scope_requires_positive_ordered_source_ranges(
    titles: tuple[tuple[Fraction, Fraction], ...],
) -> None:
    with pytest.raises(ReplayError, match="scope"):
        seeds.SourceScope(Fraction(0), Fraction(2), titles)


def test_full_plan_merges_title_ranges_without_changing_dense_guard() -> None:
    xml = ET.fromstring(
        "<fcpxml><library><event><project><sequence><spine>"
        '<asset-clip start="0s" duration="2s">'
        '<title offset="1s" duration="1s"/><title offset="0s" duration="1s"/>'
        '<title offset="1/2s" duration="1s"/></asset-clip>'
        '<asset-clip start="2s" duration="1s"/>'
        "</spine></sequence></project></event></library></fcpxml>"
    )
    assert seeds.source_scopes(xml) == (
        seeds.SourceScope(Fraction(0), Fraction(2), ((Fraction(0), Fraction(2)),)),
    )
    clip = xml.find(".//asset-clip")
    assert clip is not None
    clip.extend(ET.fromstring('<title offset="0s" duration="1s"/>') for _ in range(513))
    assert len(seeds.source_scopes(xml)) == 1
    with pytest.raises(ReplayError, match="bounded caption-covered"):
        caption_spans(xml)


@pytest.mark.parametrize("fault", ["empty", "timeMap", "zero", "outside", "budget"])
def test_unsupported_full_timeline_fails(fault: str, monkeypatch: pytest.MonkeyPatch) -> None:
    xml = ET.fromstring(
        '<fcpxml><library><event><project><sequence><spine><asset-clip start="0s" duration="2s">'
        '<title offset="0s" duration="1s"/></asset-clip></spine></sequence>'
        "</project></event></library></fcpxml>"
    )
    clip = xml.find(".//asset-clip")
    assert clip is not None
    title = clip.find("title")
    assert title is not None
    if fault == "empty":
        clip.remove(title)
    if fault == "timeMap":
        ET.SubElement(clip, "timeMap")
    if fault == "zero":
        title.set("duration", "0s")
    if fault == "outside":
        title.set("offset", "3s")
    if fault == "budget":
        monkeypatch.setattr(seeds, "MAX_TITLES", 0)
    with pytest.raises(ReplayError):
        seeds.source_scopes(xml)


def test_windows_select_whole_cut_coverage_not_only_title(
    job: storage.TrackingJob, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    later = replace(job, config=tmp_path / "later", clock=VideoFrameClock((2, 3), Fraction(1), 1))
    monkeypatch.setattr(
        seeds, "face_references", lambda _index: [(job.config, tmp_path), (later.config, tmp_path)]
    )
    monkeypatch.setattr(storage, "load_job", lambda config: job if config == job.config else later)
    assert seeds.selected_references(
        tmp_path / "index",
        (seeds.SourceScope(Fraction(0), Fraction(2), ((Fraction(3, 2), Fraction(2)),)),),
    ) == [(tmp_path / "tracking.json", tmp_path)]


@pytest.mark.parametrize("fault", ["none", "duplicate", "budget"])
def test_selected_window_membership_is_bounded_and_unique(
    job: storage.TrackingJob, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, fault: str
) -> None:
    monkeypatch.setattr(
        seeds,
        "face_references",
        lambda _index: [] if fault == "none" else [(job.config, tmp_path)] * 2,
    )
    monkeypatch.setattr(storage, "load_job", lambda _config: job)
    if fault == "budget":
        monkeypatch.setattr(seeds, "MAX_WINDOWS", 1)
    with pytest.raises(ReplayError, match="windows"):
        seeds.selected_references(
            tmp_path / "index",
            (seeds.SourceScope(Fraction(0), Fraction(2), ((Fraction(0), Fraction(2)),)),),
        )


def test_plan_publishes_immutable_shards_and_all_protected_inputs(
    config: Path, tmp_path: Path
) -> None:
    result = seeds.plan(config, tmp_path / "requests")
    assert result["protocol"] == "current-upper-seed-requests-v1"
    assert result["completeRequestedSeedsMeasured"] is False
    assert result["denseFrameCoverage"] is False
    assert result["headNearVerified"] is False
    page = record(
        json.loads((tmp_path / "requests/requests-0000.json").read_text(encoding="utf-8"))
    )
    cells = page["cells"]
    assert isinstance(cells, list)
    assert page["version"] == 1 and len(cells) == 2
    inputs = record(result["inputs"])
    assert {"foreground_observations.py", "tracking_retained.py", "head_image_seed_plan.py"} <= {
        Path(name).name for name in inputs
    }
    with pytest.raises(ReplayError, match="absent"):
        seeds.plan(config, tmp_path / "requests")


def test_optional_retained_inputs_have_separate_current_crypto_boundary(
    config: Path, job: storage.TrackingJob, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    descriptor = tmp_path / "descriptor.json"
    descriptor.write_text("{}", encoding="utf-8")
    data = record(json.loads(config.read_text(encoding="utf-8")))
    data["retainedTracking"] = {"path": str(descriptor), "sha256": file_digest(descriptor)}
    data["retainedImage"] = {"synthetic": True}
    config.write_text(json.dumps(data), encoding="utf-8")
    monkeypatch.setattr(
        seeds,
        "verify_catalog",
        lambda *_args, **_kwargs: RetainedTracking((), {str(descriptor): file_digest(descriptor)}),
    )
    image = RetainedUpperPixels(
        ImageJob(config, job.source, job.native, tmp_path / "sandbox.sb", 1080, 1920, (), {}),
        VisibleArc((ArcCell(Fraction(0), Fraction(1), 700, 3, 0),), {}),
        (),
        {str(descriptor): file_digest(descriptor)},
        {},
        "a" * 64,
    )
    monkeypatch.setattr(seeds, "read_image_binding", lambda *_args, **_kwargs: image)
    result = seeds.plan(config, tmp_path / "retained")
    assert record(result["cost"])["reusedPixelSeeds"] == 1
    assert record(result["cost"])["newRequestedCells"] == 1


@pytest.mark.parametrize("fault", ["field", "version", "resettype", "resetbudget", "pin"])
def test_config_rejects_unbound_or_oversized_input(
    config: Path, monkeypatch: pytest.MonkeyPatch, fault: str
) -> None:
    data = record(json.loads(config.read_text(encoding="utf-8")))
    if fault == "field":
        data["guessROI"] = True
    if fault == "version":
        data["version"] = True
    if fault == "resettype":
        data["zoomResetTimes"] = "0"
    if fault == "resetbudget":
        data["zoomResetTimes"] = ["0", "1"]
        monkeypatch.setattr(seeds, "MAX_TITLES", 1)
    if fault == "pin":
        record(data["native"])["sha256"] = "a" * 64
    config.write_text(json.dumps(data), encoding="utf-8")
    with pytest.raises((ReplayError, ValueError)):
        seeds._config(config)


def test_input_change_after_planning_prevents_complete_manifest(
    config: Path, job: storage.TrackingJob, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    original = seeds.build_requests

    def changed(*args: FaceScaleSeries, **kwargs: object) -> seeds.SeedPlan:
        job.source.write_bytes(b"changed source")
        return seeds.SeedPlan((), (), 0, 0)

    monkeypatch.setattr(seeds, "build_requests", changed)
    with pytest.raises(ReplayError, match="changed"):
        seeds.plan(config, tmp_path / "failed")
    assert not (tmp_path / "failed/requests.json").exists()
    monkeypatch.setattr(seeds, "build_requests", original)


def test_changed_source_before_planning_cannot_be_adopted(
    config: Path, job: storage.TrackingJob, tmp_path: Path
) -> None:
    job.source.write_bytes(b"wrong bytes")
    with pytest.raises(ReplayError, match="differs"):
        seeds.plan(config, tmp_path / "rejected")
    assert not (tmp_path / "rejected").exists()


def test_page_budget_failure_leaves_no_complete_receipt(
    config: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(seeds, "PAGE_CELLS", 1)
    monkeypatch.setattr(seeds, "MAX_PAGES", 1)
    with pytest.raises(ReplayError, match="pages"):
        seeds.plan(config, tmp_path / "bounded")
    assert (tmp_path / "bounded/requests-0000.json").exists()
    assert not (tmp_path / "bounded/requests.json").exists()


def test_output_rejects_root_parent_and_symlink(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(seeds, "WORK_ROOT", tmp_path)
    with pytest.raises(ReplayError, match="child"):
        seeds._output(tmp_path)
    with pytest.raises(ReplayError, match="child"):
        seeds._output(tmp_path.parent / "escape")
    (tmp_path / "linked").symlink_to(tmp_path.parent, target_is_directory=True)
    with pytest.raises(ValueError, match="Linked"):
        seeds._output(tmp_path / "linked/output")


def test_cli_only_prints_cost_and_never_invokes_native(
    config: Path,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    monkeypatch.setattr(sys, "argv", ["seed-plan", str(config), str(tmp_path / "cli-output")])
    seeds.main()
    assert json.loads(capsys.readouterr().out)["newRequestedCells"] == 2
