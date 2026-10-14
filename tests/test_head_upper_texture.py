import json
import sys
from collections.abc import Iterator
from dataclasses import replace
from fractions import Fraction
from pathlib import Path
from unittest.mock import Mock

import cv2
import numpy as np
import pytest

from replay import head_contour_track as tracker
from replay import head_image_retained as retained
from replay import head_upper_texture as upper
from replay.core import ReplayError
from replay.fcp_visible_arc import ArcCell, VisibleArc
from replay.head_image_collection import ImageJob
from replay.head_image_plan import ImageFrameCell
from replay.head_image_retained import RetainedUpperBatch, RetainedUpperPixels
from replay.recordings import record
from replay.stages import _uncached_file_digest
from replay.subtitle_clock import VideoFrameClock
from replay.subtitle_inpaint import Image


def native_frame(tick: int) -> dict[str, object]:
    return {
        "requested": {"value": tick, "timescale": 30},
        "actual": {"value": tick, "timescale": 30},
        "width": 160,
        "height": 160,
        "observationPresent": True,
        "instanceLabels": [1],
        "maskThreshold8Bit": 1,
        "captionPlacementApproved": False,
        "unionMask": {
            "width": 160,
            "height": 160,
            "nonzeroPixels": 10000,
            "nonzeroBounds": {"x": 24, "y": 16, "width": 112, "height": 144},
        },
        "headTop": {
            "status": "clippedSupport",
            "semanticHeadVerified": False,
            "captionPlacementApproved": False,
            "topY": None,
            "contourScope": "upper-component-in-face-x-range",
            "faceCoveredPixels": 2000,
            "facePixels": 2240,
            "upperContour": [
                {"x": 40, "y": 20 + tick * 2},
                {"x": 80, "y": 16 + tick * 2},
                {"x": 120, "y": 20 + tick * 2},
            ],
            "interiorRegions": [
                {"x": 24, "y": 32, "width": 112, "height": 64},
                {"x": 24, "y": 112, "width": 112, "height": 16},
                {"x": 0, "y": 32, "width": 16, "height": 16},
            ],
            "upperSupport": {
                "status": "interiorMeasured",
                "minimumY": 16 + tick * 2,
                "contourTouchesTopEdge": False,
                "contourTouchesBottomEdge": False,
                "contourTouchesSideEdge": False,
            },
        },
    }


@pytest.fixture
def evidence(tmp_path: Path) -> RetainedUpperPixels:
    source = tmp_path / "source.mp4"
    source.write_bytes(b"synthetic media input")
    cells = tuple(
        ImageFrameCell(Fraction(i, 30), Fraction(i + 1, 30), (24, 110, 112, 20)) for i in range(3)
    )
    response = tmp_path / "response.json"
    response.write_text(json.dumps([native_frame(i) for i in range(3)]), encoding="utf-8")
    inputs = {str(source): _uncached_file_digest(source)}
    job = ImageJob(tmp_path / "config.json", source, source, source, 160, 160, cells, inputs)
    arc = VisibleArc(
        tuple(ArcCell(c.start, c.end, 16 + i * 2, 3, 0) for i, c in enumerate(cells)), inputs
    )
    return RetainedUpperPixels(
        job,
        arc,
        (RetainedUpperBatch(cells, response, _uncached_file_digest(response)),),
        inputs,
        {},
        "a" * 64,
    )


@pytest.fixture
def image() -> Image:
    raw = np.random.default_rng(52).integers(0, 255, (160, 160, 3), dtype=np.uint8)
    return np.asarray(cv2.GaussianBlur(raw, (3, 3), 0), np.uint8)


@pytest.fixture
def seed(evidence: RetainedUpperPixels) -> upper.ApproxTextureSeed:
    value = upper.read_seed(evidence)
    assert value is not None
    return value


def frames(image: Image) -> tuple[tuple[int, Image], ...]:
    return tuple(
        (
            i,
            np.asarray(
                cv2.warpAffine(image, np.array([[1, 0, 0], [0, 1, i * 2]], np.float32), (160, 160)),
                np.uint8,
            ),
        )
        for i in range(3)
    )


def test_partial_seed_keeps_status_and_excludes_body_and_ui(seed: upper.ApproxTextureSeed) -> None:
    assert seed.native_status == "clippedSupport"
    assert seed.contour.interior_regions == ((24, 32, 112, 64),)
    np.testing.assert_array_equal(
        seed.contour.points, np.array([[40, 20], [80, 16], [120, 20]], np.float32)
    )


@pytest.mark.parametrize("case", ["missing", "unavailable", "no-head-tiles", "lk-border"])
def test_unavailable_seed_is_not_fabricated(
    evidence: RetainedUpperPixels, monkeypatch: pytest.MonkeyPatch, case: str
) -> None:
    frame = native_frame(0)
    head = record(frame["headTop"])
    if case == "missing":
        cells = (replace(evidence.job.cells[0], face=None), *evidence.job.cells[1:])
        evidence = replace(evidence, job=replace(evidence.job, cells=cells))
    elif case == "unavailable":
        head["upperSupport"] = {"status": "unavailable"}
    elif case == "no-head-tiles":
        head["interiorRegions"] = []
    else:
        head["upperContour"] = [{"x": 4, "y": 20}, {"x": 80, "y": 16}, {"x": 120, "y": 20}]
    frame["headTop"] = head
    monkeypatch.setattr(
        retained,
        "read_response",
        Mock(return_value=[frame, native_frame(1), native_frame(2)]),
    )
    assert upper.read_seed(evidence) is None


@pytest.mark.parametrize(
    "tiles",
    [
        [{"x": 24, "y": 32, "width": 112}],
        [{"x": 24, "y": 32, "width": 999, "height": 16}],
        [{"x": 24, "y": 32, "width": 0, "height": 16}],
        [{"x": 24, "y": 32, "width": 16, "height": 16}] * 257,
    ],
)
def test_invalid_native_tile_evidence(
    evidence: RetainedUpperPixels, monkeypatch: pytest.MonkeyPatch, tiles: list[dict[str, int]]
) -> None:
    frame = native_frame(0)
    head = record(frame["headTop"])
    head["interiorRegions"] = tiles
    frame["headTop"] = head
    monkeypatch.setattr(
        retained,
        "read_response",
        Mock(return_value=[frame, native_frame(1), native_frame(2)]),
    )
    with pytest.raises(ValueError):
        upper.read_seed(evidence)


def test_raw_seed_missing_or_changed(evidence: RetainedUpperPixels) -> None:
    with pytest.raises(ValueError, match="absent"):
        upper.read_seed(replace(evidence, batches=()))
    evidence.batches[0].response.write_text("[]", encoding="utf-8")
    with pytest.raises(ValueError, match="changed"):
        upper.read_seed(evidence)


@pytest.mark.parametrize(
    "cells,base",
    [
        ((), Fraction(1, 30)),
        ((ImageFrameCell(Fraction(0), Fraction(1), None),), Fraction(0)),
        ((ImageFrameCell(Fraction(1, 30), Fraction(2, 30), None),), Fraction(1, 25)),
        ((ImageFrameCell(Fraction(1), Fraction(0), None),), Fraction(1)),
        (
            (
                ImageFrameCell(Fraction(0), Fraction(1), None),
                ImageFrameCell(Fraction(1, 2), Fraction(2), None),
            ),
            Fraction(1, 2),
        ),
    ],
)
def test_invalid_native_cell_clocks(cells: tuple[ImageFrameCell, ...], base: Fraction) -> None:
    with pytest.raises(ValueError):
        upper.source_clock(cells, base)


def test_texture_positions_follow_pixels_not_bbox_height(
    evidence: RetainedUpperPixels, seed: upper.ApproxTextureSeed, image: Image
) -> None:
    cells = (
        evidence.job.cells[0],
        replace(evidence.job.cells[1], face=(24, 90, 112, 50)),
        replace(evidence.job.cells[2], face=(24, 70, 112, 70)),
    )
    result = upper.track_upper(seed, cells, Fraction(1, 30), frames(image))
    assert result.request is None
    assert len(result.observations) == 3
    assert result.observations[-1].candidate_top == pytest.approx(20, abs=0.5)
    assert result.observations[-1].provenance == "texture-derived-foreground-contour"


@pytest.mark.parametrize("reason", ["source-cut", "zoom-reset"])
def test_true_boundary_requires_new_current_measurement(
    evidence: RetainedUpperPixels, seed: upper.ApproxTextureSeed, image: Image, reason: str
) -> None:
    result = upper.track_upper(
        seed,
        evidence.job.cells,
        Fraction(1, 30),
        frames(image),
        resets=((Fraction(1, 30), reason),),
    )
    assert len(result.observations) == 1
    assert result.request == upper.MeasurementRequest(
        Fraction(1, 30), Fraction(2, 30), reason, True
    )
    assert result.request.encoded()["faceObserved"] is True


@pytest.mark.parametrize(
    "reason", ["missing-current-face", "noncontiguous-source-clock", "unsupported-frame-extent"]
)
def test_unknown_and_long_extent_preserve_original_cells(
    evidence: RetainedUpperPixels, seed: upper.ApproxTextureSeed, image: Image, reason: str
) -> None:
    second = evidence.job.cells[1]
    if reason == "missing-current-face":
        second = replace(second, face=None)
    elif reason == "noncontiguous-source-clock":
        second = replace(second, start=Fraction(1, 15), end=Fraction(1, 10))
    else:
        second = replace(second, end=Fraction(2))
    result = upper.track_upper(
        seed, (evidence.job.cells[0], second), Fraction(1, 30), frames(image)
    )
    assert len(result.observations) == 1
    assert result.request == upper.MeasurementRequest(
        second.start, second.end, reason, second.face is not None
    )


def test_initial_missing_preserves_unknown_without_decode(
    evidence: RetainedUpperPixels, seed: upper.ApproxTextureSeed
) -> None:
    cell = replace(evidence.job.cells[0], face=None)
    result = upper.track_upper(replace(seed, cell=cell), (cell,), Fraction(1, 30), ())
    assert result.observations == ()
    assert result.request == upper.MeasurementRequest(
        cell.start, cell.end, "missing-current-face", False
    )


@pytest.mark.parametrize(
    "resets",
    [
        ((Fraction(1, 60), "source-cut"),),
        ((Fraction(1, 30), "invented"),),
        ((Fraction(2, 30), "source-cut"), (Fraction(1, 30), "zoom-reset")),
    ],
)
def test_reset_contract_is_exact(
    evidence: RetainedUpperPixels,
    seed: upper.ApproxTextureSeed,
    resets: tuple[tuple[Fraction, str], ...],
) -> None:
    with pytest.raises(ValueError, match="resets"):
        upper.track_upper(seed, evidence.job.cells, Fraction(1, 30), (), resets=resets)


def test_seed_and_decoder_clock_must_match(
    evidence: RetainedUpperPixels, seed: upper.ApproxTextureSeed, image: Image
) -> None:
    with pytest.raises(ValueError, match="seed"):
        upper.track_upper(
            replace(seed, cell=evidence.job.cells[1]), evidence.job.cells, Fraction(1, 30), ()
        )
    with pytest.raises(ValueError, match="clock"):
        upper.track_upper(seed, evidence.job.cells, Fraction(1, 30), ((1, image),))
    with pytest.raises(ValueError, match="incomplete"):
        upper.track_upper(seed, evidence.job.cells, Fraction(1, 30), ((0, image),))


def test_occlusion_never_holds_last_supported_position(
    evidence: RetainedUpperPixels, seed: upper.ApproxTextureSeed, image: Image
) -> None:
    result = upper.track_upper(
        seed, evidence.job.cells, Fraction(1, 30), ((0, image), (1, np.zeros_like(image)))
    )
    assert result.request is not None
    assert result.request.start == Fraction(1, 30)
    assert result.observations[-1].candidate_top is None
    assert result.observations[-1].points == ()


def test_code_closure_pins_decoder_and_partial_guard() -> None:
    paths = upper._code_paths()
    assert upper.ROOT / "src/replay/subtitle_quality_observe.py" in paths
    assert upper.ROOT / "src/replay/head_texture_track.py" in paths
    assert upper.ROOT / "src/replay/head_image_support.py" in paths
    assert len(paths) == len(set(paths))


@pytest.fixture
def run_config(
    evidence: RetainedUpperPixels, image: Image, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> Path:
    work = tmp_path / "edits/work"
    work.mkdir(parents=True)
    (tmp_path / "uv.lock").write_text("synthetic lock", encoding="utf-8")
    ffmpeg = tmp_path / "ffmpeg"
    ffmpeg.write_bytes(b"synthetic decoder executable")
    config = work / "config.json"
    config.write_text(
        json.dumps(
            {
                "version": 1,
                "retainedImage": {},
                "sourceSHA256": evidence.inputs[str(evidence.job.source)],
                "width": 160,
                "height": 160,
                "timeBase": "1/30",
                "ffmpeg": {"path": str(ffmpeg), "sha256": _uncached_file_digest(ffmpeg)},
                "resets": [],
            }
        ),
        encoding="utf-8",
    )
    monkeypatch.setattr(upper, "ROOT", tmp_path)
    monkeypatch.setattr(upper, "_code_paths", Mock(return_value=()))
    monkeypatch.setattr(retained, "read_image_binding", Mock(return_value=evidence))
    monkeypatch.setattr(tracker, "_decode", Mock(return_value=iter(frames(image))))
    return config


def test_bounded_producer_records_support_and_false_approvals(run_config: Path) -> None:
    path = upper.run(run_config, run_config.parent / "result")
    raw = record(json.loads(path.read_text(encoding="utf-8")))
    assert raw["probeAccepted"] is True
    assert raw["completeTrack"] is True
    assert raw["requestedFrames"] == 3
    assert raw["semanticHeadVerified"] is False
    assert raw["qualityApproved"] is False
    assert raw["headNearVerified"] is False
    assert raw["newSegmentationCalls"] == 0
    assert raw["seedNativeStatus"] == "clippedSupport"
    assert raw["measurementRequest"] is None
    upper.verify(run_config, path.parent)
    assert path.stat().st_mode & 0o777 == 0o600
    assert path.parent.stat().st_mode & 0o777 == 0o700
    with pytest.raises(ValueError, match="work paths"):
        upper.run(run_config, path.parent)


def test_missing_seed_producer_emits_measurement_request(
    run_config: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(upper, "read_seed", Mock(return_value=None))
    raw = record(
        json.loads(upper.run(run_config, run_config.parent / "missing").read_text(encoding="utf-8"))
    )
    assert raw["rows"] == []
    assert raw["probeAccepted"] is False
    assert raw["maximumAbsoluteErrorPixels"] is None
    assert record(raw["measurementRequest"])["reason"] == "measured-upper-texture-unavailable"
    upper.verify(run_config, run_config.parent / "missing")


@pytest.mark.parametrize(
    "patch", [{"version": True}, {"extra": 0}, {"timeBase": []}, {"sourceSHA256": "bad"}]
)
def test_invalid_run_configuration(run_config: Path, patch: dict[str, object]) -> None:
    raw = record(json.loads(run_config.read_text(encoding="utf-8")))
    run_config.write_text(json.dumps(raw | patch), encoding="utf-8")
    with pytest.raises(ValueError):
        upper.run(run_config, run_config.parent / "invalid")


def test_retained_cell_extent_cannot_change_in_comparison(
    run_config: Path, evidence: RetainedUpperPixels, monkeypatch: pytest.MonkeyPatch
) -> None:
    arc = replace(evidence.arc.cells[0], end=Fraction(1, 60))
    changed = replace(evidence, arc=VisibleArc((arc, *evidence.arc.cells[1:]), evidence.inputs))
    monkeypatch.setattr(retained, "read_image_binding", Mock(return_value=changed))
    with pytest.raises(ValueError, match="comparison cell clock"):
        upper.run(run_config, run_config.parent / "changed-clock")


def test_error_budget_is_not_acceptance_by_tracking_alone(
    run_config: Path, evidence: RetainedUpperPixels, monkeypatch: pytest.MonkeyPatch
) -> None:
    arc = replace(evidence.arc.cells[2], visible_top=100)
    changed = replace(evidence, arc=VisibleArc((*evidence.arc.cells[:2], arc), evidence.inputs))
    monkeypatch.setattr(retained, "read_image_binding", Mock(return_value=changed))
    raw = record(
        json.loads(upper.run(run_config, run_config.parent / "error").read_text(encoding="utf-8"))
    )
    assert raw["completeTrack"] is True
    assert raw["probeAccepted"] is False
    assert raw["errorBudgetPixels"] == 6
    assert record(raw["measurementRequest"])["reason"] == "measured-upper-disagreement"
    upper.verify(run_config, run_config.parent / "error")


def test_input_change_during_decode_is_rejected(
    run_config: Path, evidence: RetainedUpperPixels, image: Image, monkeypatch: pytest.MonkeyPatch
) -> None:
    def changed_decode(
        source: Path, clock: VideoFrameClock, contour: tracker.ContourSeed, ffmpeg: Path
    ) -> Iterator[tuple[int, Image]]:
        evidence.job.source.write_bytes(b"changed synthetic source")
        yield from frames(image)

    monkeypatch.setattr(tracker, "_decode", changed_decode)
    with pytest.raises(ValueError, match="bytes changed"):
        upper.run(run_config, run_config.parent / "changed")


def test_work_symlink_is_rejected(run_config: Path) -> None:
    link = run_config.parent / "linked"
    link.symlink_to(run_config.parent, target_is_directory=True)
    with pytest.raises(ValueError, match="Linked"):
        upper.run(run_config, link / "result")


def test_main_dispatch(
    run_config: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    output = run_config.parent / "main-result"
    monkeypatch.setattr(sys, "argv", ["head_upper_texture", str(run_config), str(output)])
    upper.main()
    assert str(output / "upper-texture.json") in capsys.readouterr().out


def test_partial_reader_keeps_complete_contour_guard(evidence: RetainedUpperPixels) -> None:
    frame = native_frame(0)
    with pytest.raises(ValueError, match="Unapproved"):
        tracker.contour_seed([frame], clock=VideoFrameClock((0,), Fraction(1, 30), 1))
    assert upper.read_seed(evidence) is not None


def test_unsupported_native_pixels_do_not_become_a_seed(
    evidence: RetainedUpperPixels, monkeypatch: pytest.MonkeyPatch
) -> None:
    frame = native_frame(0)
    head = record(frame["headTop"])
    head["semanticHeadVerified"] = True
    frame["headTop"] = head
    monkeypatch.setattr(
        retained,
        "read_response",
        Mock(return_value=[frame, native_frame(1), native_frame(2)]),
    )
    with pytest.raises(ReplayError):
        upper.read_seed(evidence)


@pytest.mark.parametrize(
    "patch",
    [
        {"version": True},
        {"protocol": "other"},
        {"requestedFrames": 99},
        {"completeTrack": False},
        {"probeAccepted": False},
        {"maximumAbsoluteErrorPixels": 9},
        {"errorBudgetPixels": 8},
        {"qualityApproved": True},
        {"newSegmentationCalls": 1},
    ],
)
def test_diagnostic_verifier_rejects_changed_claims(
    run_config: Path, patch: dict[str, object]
) -> None:
    path = upper.run(run_config, run_config.parent / "diagnostic")
    raw = record(json.loads(path.read_text(encoding="utf-8")))
    path.write_text(json.dumps(raw | patch), encoding="utf-8")
    with pytest.raises(ValueError):
        upper.verify(run_config, path.parent)


@pytest.mark.parametrize(
    "patch",
    [
        {"source_pts": 2},
        {"candidate_top": True},
        {"candidate_top": 999},
        {"points": [[40, float("nan")]]},
        {"points": [[40]]},
        {"points": [[40, 16]] * 257},
    ],
)
def test_diagnostic_verifier_rejects_fabricated_clock_and_pixels(
    run_config: Path, patch: dict[str, object]
) -> None:
    path = upper.run(run_config, run_config.parent / "diagnostic")
    raw = record(json.loads(path.read_text(encoding="utf-8")))
    entries = raw["rows"]
    assert isinstance(entries, list)
    first = record(entries[0])
    first["observation"] = record(first["observation"]) | patch
    entries[0] = first
    path.write_text(json.dumps(raw), encoding="utf-8")
    with pytest.raises(ValueError):
        upper.verify(run_config, path.parent)


def test_diagnostic_verifier_rejects_missing_dependency_roles(
    run_config: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    path = upper.run(run_config, run_config.parent / "diagnostic")
    monkeypatch.setattr(
        upper, "_code_paths", Mock(return_value=(run_config.parent / "unbound-helper.py",))
    )
    with pytest.raises(ValueError, match="closure"):
        upper.verify(run_config, path.parent)


def test_diagnostic_verifier_rejects_changed_configuration(run_config: Path) -> None:
    path = upper.run(run_config, run_config.parent / "diagnostic")
    run_config.write_text(run_config.read_text(encoding="utf-8") + " ", encoding="utf-8")
    with pytest.raises(ValueError, match="configuration pin"):
        upper.verify(run_config, path.parent)


def test_main_readonly_verification(
    run_config: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    path = upper.run(run_config, run_config.parent / "diagnostic")
    monkeypatch.setattr(
        sys, "argv", ["head_upper_texture", str(run_config), str(path.parent), "--verify"]
    )
    upper.main()
    assert "head quality remain unverified" in capsys.readouterr().out


@pytest.mark.parametrize("scale", [0.98, 1.02])
def test_image_supported_small_zoom_is_derived_not_bbox_scale(
    evidence: RetainedUpperPixels, seed: upper.ApproxTextureSeed, image: Image, scale: float
) -> None:
    matrix = cv2.getRotationMatrix2D((80, 80), 0, scale)
    donor = np.asarray(cv2.warpAffine(image, matrix, (160, 160)), np.uint8)
    result = upper.track_upper(
        seed, evidence.job.cells[:2], Fraction(1, 30), ((0, image), (1, donor))
    )
    assert result.request is None
    expected = float(np.min((seed.contour.points @ matrix[:, :2].T + matrix[:, 2])[:, 1]))
    assert result.observations[-1].candidate_top == pytest.approx(expected, abs=0.5)
