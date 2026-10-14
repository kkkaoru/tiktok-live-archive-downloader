import json
from dataclasses import dataclass, replace
from fractions import Fraction
from pathlib import Path

import numpy as np
import pytest

from replay import head_local_support_probe as probe
from replay import head_partial_pixel_track as pixels
from replay.code_dependency_inventory import Source
from replay.delivery_mask_quality import rows
from replay.head_contour_track import ContourSeed
from replay.head_seed import ContourTrackMetrics, Coordinates
from replay.head_texture_track import Region, TextureFit, TextureMotion, fit_texture_model
from replay.recordings import record
from replay.stages import _uncached_file_digest
from replay.subtitle_clock import VideoFrameClock


@dataclass(frozen=True, slots=True)
class MetadataCase:
    config: Path
    output: Path
    report: Path
    producer: Path


@pytest.fixture
def regions() -> tuple[Region, ...]:
    return ((200, 600, 600, 300), (200, 1000, 600, 300))


@pytest.fixture
def controls() -> Coordinates:
    return np.array(((400, 340), (500, 341), (600, 340)), np.float32)


@pytest.fixture
def fits() -> tuple[TextureFit, TextureFit]:
    first = np.column_stack(
        (np.tile((250, 450, 650, 750), 6), np.repeat((650, 750, 850, 1050, 1150, 1250), 4))
    ).astype(np.float32)
    second = first + np.array((10, 5), np.float32)
    flags = np.ones(24, np.bool_)
    errors = np.zeros(24, np.float32)
    one = ContourTrackMetrics(first + (1, 2), flags, flags, flags, flags, errors, errors)
    two = replace(one, points=second + (1, 2))
    return fit_texture_model(first, one), fit_texture_model(second, two)


def test_rigid_current_support_has_independent_local_models_without_adoption(
    fits: tuple[TextureFit, TextureFit], controls: Coordinates, regions: tuple[Region, ...]
) -> None:
    result = probe.replay_local_models(
        fits=fits,
        controls=(controls, controls.copy()),
        original_regions=regions,
        current_regions=regions,
        source_pts=31,
        width=1080,
        height=1920,
        budget=8,
    )
    assert result.reason is None
    assert result.cut_y == 899
    assert result.tile_owners == (0, 1)
    assert result.pair is not None
    assert result.pair["accepted"] is True
    assert result.pair["estimatedUpperY"] == 342
    assert result.local_models[0]["selected"] == 24
    assert result.local_models[1]["selected"] == 24
    assert result.audit()["YAdopted"] is False
    assert result.audit()["continuousControlChainVerified"] is False


def test_predeclared_local_motion_differs_from_failed_global_affine(
    fits: tuple[TextureFit, TextureFit], controls: Coordinates, regions: tuple[Region, ...]
) -> None:
    first, second = fits
    center = np.array((500, 340), np.float32)
    one_scales = np.where(first.source[:, 1] < 900, 0.98, 1.02).reshape(-1, 1)
    two_scales = np.where(second.source[:, 1] < 900, 0.98, 1.02).reshape(-1, 1)
    one = fit_texture_model(
        first.source,
        replace(
            first.metrics,
            points=np.asarray((first.source - center) * one_scales + center, np.float32),
        ),
    )
    two = fit_texture_model(
        second.source,
        replace(
            second.metrics,
            points=np.asarray((second.source - center) * two_scales + center, np.float32),
        ),
    )
    assert one.reason == "motion_support_insufficient"
    assert two.reason == "motion_support_insufficient"
    result = probe.replay_local_models(
        fits=(one, two),
        controls=(controls, controls.copy()),
        original_regions=regions,
        current_regions=regions,
        source_pts=31,
        width=1080,
        height=1920,
        budget=8,
    )
    assert result.reason is None
    assert result.pair is not None
    assert result.pair["maximumControlsDisagreementPx"] == pytest.approx(4, abs=0.001)
    assert result.local_models[0]["legacyReason"] is None
    assert result.local_models[1]["legacyReason"] is None


def test_static_competitor_cannot_agree_with_moving_region(
    fits: tuple[TextureFit, TextureFit], controls: Coordinates, regions: tuple[Region, ...]
) -> None:
    first, second = fits
    one_delta = np.column_stack((np.zeros(24), np.where(first.source[:, 1] < 900, 0, 20)))
    two_delta = np.column_stack((np.zeros(24), np.where(second.source[:, 1] < 900, 0, 20)))
    one = fit_texture_model(
        first.source,
        replace(first.metrics, points=np.asarray(first.source + one_delta, np.float32)),
    )
    two = fit_texture_model(
        second.source,
        replace(second.metrics, points=np.asarray(second.source + two_delta, np.float32)),
    )
    result = probe.replay_local_models(
        fits=(one, two),
        controls=(controls, controls.copy()),
        original_regions=regions,
        current_regions=regions,
        source_pts=31,
        width=1080,
        height=1920,
        budget=8,
    )
    assert result.reason == "competing_corroborated_target"
    assert result.pair is not None
    assert result.pair["maximumControlsDisagreementPx"] == 20


def test_failed_current_points_keep_original_regional_denominator(
    fits: tuple[TextureFit, TextureFit], controls: Coordinates, regions: tuple[Region, ...]
) -> None:
    first, second = fits
    valid = np.ones(24, np.bool_)
    valid[0:4] = False
    first = replace(first, metrics=replace(first.metrics, supported=valid))
    result = probe.replay_local_models(
        fits=(first, second),
        controls=(controls, controls.copy()),
        original_regions=regions,
        current_regions=regions,
        source_pts=31,
        width=1080,
        height=1920,
        budget=8,
    )
    assert result.local_models[0]["selected"] == 24
    assert result.local_models[0]["independentlySupported"] == 20
    indices = result.local_models[0]["originalIndices"]
    assert isinstance(indices, tuple)
    assert len(indices) == 24
    assert result.feature_owners[0] == (
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        1,
        1,
        1,
        1,
        1,
        1,
        1,
        1,
        1,
        1,
        1,
        1,
    )


@pytest.mark.parametrize(
    "fault", ["hud", "negative", "nonfinite", "overlap", "dropped", "geometry", "budget"]
)
def test_regional_replay_rejects_unsupported_association_and_changed_bounds(
    fits: tuple[TextureFit, TextureFit],
    controls: Coordinates,
    regions: tuple[Region, ...],
    fault: str,
) -> None:
    first, second = fits
    modified = first.source.copy()
    if fault == "hud":
        modified[0] = (100, 350)
    elif fault == "negative":
        modified[0] = (-1, 650)
    elif fault == "nonfinite":
        modified[0] = (np.nan, 650)
    first = replace(first, source=modified)
    supplied = (first, first) if fault == "overlap" else (first, second)
    current = regions[:1] if fault == "dropped" else regions
    with pytest.raises(ValueError):
        probe.replay_local_models(
            fits=supplied,
            controls=(controls, controls.copy()),
            original_regions=regions,
            current_regions=current,
            source_pts=31,
            width=4 if fault == "geometry" else 1080,
            height=1920,
            budget=9 if fault == "budget" else 8,
        )


def test_one_measured_region_without_twelve_points_is_not_completed(
    fits: tuple[TextureFit, TextureFit], controls: Coordinates, regions: tuple[Region, ...]
) -> None:
    first, second = fits
    owner0 = np.zeros(24, np.int64)
    owner0[:5] = 1
    local, indices = probe._local_fit(
        (first, second), (tuple(owner0), tuple(np.zeros(24, np.int64))), 1
    )
    assert local is None
    assert indices == ((0, 0), (0, 1), (0, 2), (0, 3), (0, 4))
    reduced = ((200, 600, 600, 600), (200, 1200, 600, 100))
    result = probe.replay_local_models(
        fits=fits,
        controls=(controls, controls.copy()),
        original_regions=reduced,
        current_regions=reduced,
        source_pts=31,
        width=1080,
        height=1920,
        budget=8,
    )
    assert result.reason == "local_texture_insufficient"
    assert result.pair is None
    assert result.local_models[1]["selected"] == 8


@pytest.mark.parametrize("value", [True, "1", None, float("nan"), float("inf")])
def test_numeric_metadata_must_be_finite_and_not_boolean(value: object) -> None:
    with pytest.raises(ValueError, match="Finite numeric"):
        probe._number(value)


@pytest.mark.parametrize(
    "value",
    [
        [],
        [[1, 2, 3]],
        [[0, 0, 0, 10]],
        [[0, 0, 20, 20], [10, 10, 20, 20]],
        [[95, 95, 10, 10]],
        [[0, 0, 1, 1]] * 257,
    ],
)
def test_measured_tile_bounds_and_overlap_are_rejected(value: object) -> None:
    with pytest.raises(ValueError):
        probe._regions(value, width=100, height=100)


@pytest.mark.parametrize(
    "value", [(), ((0, 0, 10, 1),), ((0, 20, 10, 20),), ((0, 120, 10, 10),), ((-1, 20, 10, 10),)]
)
def test_partition_requires_real_bounded_two_sided_support(value: tuple[Region, ...]) -> None:
    with pytest.raises(ValueError):
        probe.measured_partition(value, height=100)


@pytest.fixture
def metadata_case(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    fits: tuple[TextureFit, TextureFit],
    controls: Coordinates,
    regions: tuple[Region, ...],
) -> MetadataCase:
    work = tmp_path / "edits" / "work"
    work.mkdir(parents=True)
    monkeypatch.setattr(probe, "ROOT", tmp_path)
    code = work / "reader.py"
    code.write_text("# independent reader\n", encoding="utf-8")
    identity = code.stat()
    captured = Source(
        code,
        _uncached_file_digest(code),
        (
            identity.st_dev,
            identity.st_ino,
            identity.st_size,
            identity.st_mtime_ns,
            identity.st_ctime_ns,
        ),
    )
    monkeypatch.setattr(probe, "capture", lambda modules: (captured,))
    seed = pixels.PartialPixelSeed(
        ContourSeed(Fraction(1), 1080, 1920, controls), regions, regions, "clippedSupport"
    )

    def detector(value: object, *, clock: VideoFrameClock) -> tuple[pixels.Face, tuple[bool, ...]]:
        return (200.0, 600.0, 600.0, 700.0), (True, False)

    def native(
        value: object, *, clock: VideoFrameClock, width: int, height: int, face: pixels.Face
    ) -> pixels.PartialPixelSeed:
        return seed

    def observations(value: object) -> tuple[dict[str, object], ...]:
        return tuple(record(v) for v in rows(record(value)["observations"]))

    monkeypatch.setattr(pixels, "detector_seed", detector)
    monkeypatch.setattr(pixels, "read_seed", native)
    monkeypatch.setattr(pixels, "spatial_observations", observations)
    one, two = fits
    one_motion = TextureMotion(
        None, one.metrics.points, one.inliers, one.residuals, one.metrics, one.reason, one
    )
    two_motion = TextureMotion(
        None, two.metrics.points, two.inliers, two.residuals, two.metrics, two.reason, two
    )
    one_decision = pixels.approximate_model(
        one, controls, maximum_target_y_uncertainty_px=8, width=1080, height=1920
    )
    two_decision = pixels.approximate_model(
        two, controls, maximum_target_y_uncertainty_px=8, width=1080, height=1920
    )
    models = [
        {
            **pixels._model_audit(one.source, controls, one_motion),
            "fitMatrix": one.matrix,
            "sourceControls": controls.tolist(),
            "approximateHeadContract": one_decision.audit,
        },
        {
            **pixels._model_audit(two.source, controls, two_motion),
            "fitMatrix": two.matrix,
            "sourceControls": controls.tolist(),
            "approximateHeadContract": two_decision.audit,
        },
    ]
    detector_path, native_path = work / "detector.json", work / "native.json"
    detector_path.write_text("{}", encoding="utf-8")
    native_path.write_text("{}", encoding="utf-8")
    producer: dict[str, object] = {
        "source": {"path": "/unopened-original.mp4", "sha256": "a" * 64},
        "width": 1080,
        "height": 1920,
        "clock": {"pts": [30, 31], "timeBase": "1/30", "lastDuration": 1},
        "detectorResponse": {
            "path": str(detector_path),
            "sha256": _uncached_file_digest(detector_path),
        },
        "nativeResponse": {"path": str(native_path), "sha256": _uncached_file_digest(native_path)},
        "approximateYPolicy": {"maximumTargetYModelUncertaintyPx": 8},
    }
    producer_path = work / "producer.json"
    producer_path.write_text(json.dumps(producer), encoding="utf-8")
    report: dict[str, object] = {
        "inputs": {
            str(producer_path): _uncached_file_digest(producer_path),
            str(detector_path): _uncached_file_digest(detector_path),
            str(native_path): _uncached_file_digest(native_path),
            "/unopened-original.mp4": "a" * 64,
        },
        "spatialModelRecords": [],
        "seedDetectorRegion": [200, 600, 600, 700],
        "seedAnchorRegions": regions,
        "observations": [
            {
                "source_pts": 30,
                "reason": None,
                "feature_refresh": {
                    "status": "current-pixel-features-selected",
                    "modelPairDecision": {"accepted": True},
                    "derivedSupportRegions": regions,
                },
            },
            {
                "source_pts": 31,
                "reason": "global_failure",
                "face_observed": False,
                "models": models,
            },
        ],
    }
    report_path = work / "report.json"
    report_path.write_text(json.dumps(report), encoding="utf-8")
    config: dict[str, object] = {
        "version": 1,
        "protocol": probe.PROTOCOL,
        "sourcePTS": 31,
        "producerConfig": {
            "path": str(producer_path),
            "sha256": _uncached_file_digest(producer_path),
        },
        "report": {"path": str(report_path), "sha256": _uncached_file_digest(report_path)},
        "contract": {
            "partitionPolicy": probe.PARTITION_POLICY,
            "maximumTargetYModelUncertaintyPx": 8,
        },
    }
    config_path = work / "config.json"
    config_path.write_text(json.dumps(config), encoding="utf-8")
    return MetadataCase(config_path, work / "output.json", report_path, producer_path)


def test_metadata_cli_keeps_missing_raw_face_and_does_not_open_source(
    metadata_case: MetadataCase,
) -> None:
    result = probe.run(metadata_case.config, metadata_case.output)
    assert result["faceObserved"] is False
    assert result["sourceMediaRead"] is False
    assert result["opticalFlowExecuted"] is False
    assert result["nativeExecuted"] is False
    assert record(result["replay"])["reason"] is None
    assert record(result["replay"])["headNearVerified"] is False
    assert metadata_case.output.stat().st_mode & 0o777 == 0o600
    assert (
        json.loads(metadata_case.output.read_text(encoding="utf-8"))["newFaceObservation"] is False
    )


@pytest.mark.parametrize(
    "fault,match",
    [
        ("version", "Unsupported"),
        ("contract", "Explicit unchanged"),
        ("clock", "non-seed"),
        ("sha", "SHA256"),
        ("conflicting-pin", "binding differs"),
        ("source-pin", "source metadata binding"),
        ("predecessor", "fully supported predecessor"),
        ("raw-face", "source/face presentation"),
        ("detector", "accepted seed"),
        ("tiles", "ownership differs"),
        ("refresh", "fully supported"),
        ("models", "two original"),
        ("budget", "budget cannot change"),
        ("time-base", "presentation clock"),
    ],
)
def test_bound_metadata_rejects_changed_original_inputs(
    metadata_case: MetadataCase, fault: str, match: str
) -> None:
    config = record(json.loads(metadata_case.config.read_text(encoding="utf-8")))
    report = record(json.loads(metadata_case.report.read_text(encoding="utf-8")))
    producer = record(json.loads(metadata_case.producer.read_text(encoding="utf-8")))
    current = record(rows(report["observations"])[1])
    previous = record(rows(report["observations"])[0])
    if fault == "version":
        config["version"] = True
    elif fault == "contract":
        record(config["contract"])["partitionPolicy"] = "fit-selected-crop"
    elif fault == "clock":
        config["sourcePTS"] = 30
    elif fault == "sha":
        record(config["report"])["sha256"] = "invalid"
    elif fault == "conflicting-pin":
        record(report["inputs"])[str(metadata_case.producer)] = "0" * 64
    elif fault == "source-pin":
        record(report["inputs"])["/unopened-original.mp4"] = "0" * 64
    elif fault == "predecessor":
        previous["reason"] = "scene-reset"
    elif fault == "raw-face":
        current["face_observed"] = 0
    elif fault == "detector":
        report["seedDetectorRegion"] = [1, 2, 3, 4]
    elif fault == "tiles":
        report["seedAnchorRegions"] = [[201, 600, 600, 300], [200, 1000, 600, 300]]
    elif fault == "refresh":
        record(previous["feature_refresh"])["status"] = "held"
    elif fault == "models":
        current["models"] = []
    elif fault == "budget":
        record(config["contract"])["maximumTargetYModelUncertaintyPx"] = 7
    elif fault == "time-base":
        record(producer["clock"])["timeBase"] = 30
    metadata_case.producer.write_text(json.dumps(producer), encoding="utf-8")
    if fault != "conflicting-pin":
        record(report["inputs"])[str(metadata_case.producer)] = _uncached_file_digest(
            metadata_case.producer
        )
    record(config["producerConfig"])["sha256"] = _uncached_file_digest(metadata_case.producer)
    metadata_case.report.write_text(json.dumps(report), encoding="utf-8")
    if fault != "sha":
        record(config["report"])["sha256"] = _uncached_file_digest(metadata_case.report)
    metadata_case.config.write_text(json.dumps(config), encoding="utf-8")
    with pytest.raises(ValueError, match=match):
        probe.run(metadata_case.config, metadata_case.output)
    assert not metadata_case.output.exists()


def test_replay_requires_supported_original_native_seed(
    metadata_case: MetadataCase, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(pixels, "read_seed", lambda *args, **kwargs: None)
    with pytest.raises(ValueError, match="interior seed is unsupported"):
        probe.run(metadata_case.config, metadata_case.output)


@pytest.mark.parametrize("fault", ["absent-parent", "occupied", "escaping-link", "linked-config"])
def test_exclusive_output_and_canonical_work_paths(metadata_case: MetadataCase, fault: str) -> None:
    output = metadata_case.output
    config = metadata_case.config
    if fault == "absent-parent":
        output = output.parent / "absent" / "output.json"
    elif fault == "occupied":
        output.write_text("kept", encoding="utf-8")
    elif fault == "escaping-link":
        output.symlink_to(output.parent.parent.parent / "escape")
    elif fault == "linked-config":
        config = config.parent / "link.json"
        config.symlink_to(metadata_case.config)
    with pytest.raises(ValueError):
        probe.run(config, output)


def test_reader_membership_change_rejects_output(
    metadata_case: MetadataCase, monkeypatch: pytest.MonkeyPatch
) -> None:
    one = Source(metadata_case.config, "0" * 64, (1, 2, 3, 4, 5))
    two = Source(metadata_case.config, "1" * 64, (1, 2, 3, 4, 5))
    calls = iter(((one,), (two,)))
    monkeypatch.setattr(probe, "capture", lambda modules: next(calls))
    with pytest.raises(ValueError, match="Reader code membership"):
        probe.run(metadata_case.config, metadata_case.output)
    assert not metadata_case.output.exists()


def test_metadata_byte_budget_is_not_relaxed(
    metadata_case: MetadataCase, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(probe, "MAX_METADATA_BYTES", 1)
    with pytest.raises(ValueError, match="byte budget"):
        probe.run(metadata_case.config, metadata_case.output)


def test_output_byte_budget_is_not_relaxed(
    metadata_case: MetadataCase, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(probe, "MAX_JSON_BYTES", 20_000)
    original = probe.replay_local_models

    def replay(**kwargs: object) -> probe.LocalReplay:
        fitted = original(
            fits=typed_fits,
            controls=typed_controls,
            original_regions=typed_regions,
            current_regions=typed_regions,
            source_pts=31,
            width=1080,
            height=1920,
            budget=8,
        )
        return replace(fitted, local_models=({"oversized": "x" * 20_000},))

    parsed = record(json.loads(metadata_case.report.read_text(encoding="utf-8")))
    models = rows(record(rows(parsed["observations"])[1])["models"])
    first = pixels._validated_model(
        models[0], maximum_target_y_uncertainty_px=8, width=1080, height=1920
    )
    second = pixels._validated_model(
        models[1], maximum_target_y_uncertainty_px=8, width=1080, height=1920
    )
    typed_fits = first[0], second[0]
    typed_controls = first[1], second[1]
    typed_regions = ((200, 600, 600, 300), (200, 1000, 600, 300))
    monkeypatch.setattr(probe, "replay_local_models", replay)
    with pytest.raises(ValueError, match="output exceeds"):
        probe.run(metadata_case.config, metadata_case.output)


def test_cli_is_thin_and_preserves_reason(
    metadata_case: MetadataCase, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.setattr(
        "sys.argv", ["head-local-support", str(metadata_case.config), str(metadata_case.output)]
    )
    assert probe.main() == 0
    assert capsys.readouterr().out == '{"sourcePTS": 31, "reason": null}\n'


def test_cli_reports_invalid_configuration(
    metadata_case: MetadataCase, monkeypatch: pytest.MonkeyPatch
) -> None:
    metadata_case.config.write_text("{}", encoding="utf-8")
    monkeypatch.setattr(
        "sys.argv", ["head-local-support", str(metadata_case.config), str(metadata_case.output)]
    )
    with pytest.raises(SystemExit) as error:
        probe.main()
    assert error.value.code == 2
