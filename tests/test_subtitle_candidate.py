import hashlib
import json
import runpy
import shutil
import sys
from pathlib import Path
from unittest.mock import Mock

import cv2
import numpy as np
import pytest
from pytest import MonkeyPatch

from replay import (
    disk_budget,
    subtitle_candidate,
    subtitle_detector,
    subtitle_line_candidate,
    subtitle_prepared,
)
from replay.core import ReplayError
from replay.local_asr import fingerprint
from replay.subtitle_inpaint import Image
from replay.subtitle_quality_diagnostic import _png
from replay.subtitle_stream import BatchDetector, EvidenceSink, repair_stream
from replay.subtitle_video import VideoRepairJob, VideoRepairReceipt, spatial_evidence


@pytest.fixture
def config(tmp_path: Path, monkeypatch: MonkeyPatch) -> Path:
    monkeypatch.chdir(tmp_path)
    source, native = tmp_path / "source.mp4", tmp_path / "native"
    source.write_bytes(b"synthetic source")
    native.write_bytes(b"synthetic executable")
    native.chmod(0o700)
    data = {
        "source": str(source),
        "outputDirectory": str(tmp_path / "output"),
        "workDirectory": str(tmp_path / "tmp"),
        "nativeExecutable": str(native),
        "sourceSHA256": fingerprint(source),
        "nativeSHA256": fingerprint(native),
        "width": 8,
        "height": 8,
        "visionTextRegions": True,
        "clock": {"pts": [30, 31], "timeBase": "1/30", "lastDuration": 1},
        "masks": [
            {
                "startSeconds": 0,
                "endSeconds": 1,
                "region": {"x": 0, "y": 0, "width": 8, "height": 8},
            }
        ],
    }
    ocr = tmp_path / "ocr.json"
    ocr.write_text(
        json.dumps(
            [
                {
                    "width": 8,
                    "height": 8,
                    "requestedTimeSeconds": time,
                    "actualTimeSeconds": time,
                    "lines": [
                        {
                            "text": "Synthetic caption",
                            "confidence": 0.8,
                            "region": {"x": 1, "y": 1, "width": 6, "height": 6},
                        }
                    ],
                }
                for time in (1, 31 / 30)
            ]
        ),
        encoding="utf-8",
    )
    codec = tmp_path / "codec.json"
    codec.write_text('{"codec_name":"synthetic"}', encoding="utf-8")
    review = tmp_path / "classification.json"
    review.write_text('{"evidence":"synthetic overlay subtitle fixture"}', encoding="utf-8")
    provenance = tmp_path / "ocr-provenance.json"
    provenance.write_text(
        json.dumps(
            {
                "sourceSHA256": fingerprint(source),
                "measurementTool": "video_text_recognize",
                "fullFrame": True,
                "nativeExecutable": str(native),
                "nativeSHA256": fingerprint(native),
            }
        ),
        encoding="utf-8",
    )
    eligibility = tmp_path / "eligibility.json"
    eligibility.write_text(
        json.dumps(
            {
                "version": 1,
                "kind": "source-caption-eligibility",
                "sourceSHA256": fingerprint(source),
                "canvas": [8, 8],
                "timeBase": "1/30",
                "decoder": {"path": str(native), "sha256": fingerprint(native)},
                "codec": {"path": str(codec), "sha256": fingerprint(codec)},
                "ocr": {"path": str(ocr), "sha256": fingerprint(ocr)},
                "ocrProvenance": {"path": str(provenance), "sha256": fingerprint(provenance)},
                "classificationEvidence": {"path": str(review), "sha256": fingerprint(review)},
                "frames": [
                    {
                        "sourcePTS": pts,
                        "classification": "overlay-subtitle",
                        "observation": index,
                        "lines": [0],
                        "region": {"x": 0, "y": 0, "width": 8, "height": 8},
                    }
                    for index, pts in enumerate((30, 31))
                ],
            }
        ),
        encoding="utf-8",
    )
    data["sourceEligibility"] = {"path": str(eligibility), "sha256": fingerprint(eligibility)}
    path = tmp_path / "config.json"
    path.write_text(json.dumps(data), encoding="utf-8")

    def masks(
        images: tuple[Image, ...], *, executable: Path, work_directory: Path, text_regions: bool
    ) -> tuple[Image, ...]:
        assert len(images) == 2
        assert executable.name == "native"
        assert work_directory.name == "tmp"
        assert text_regions is True
        return (np.full((8, 8), 255, dtype=np.uint8), np.zeros((8, 8), dtype=np.uint8))

    def render(
        job: VideoRepairJob, *, detect_batch: BatchDetector, evidence: EvidenceSink | None = None
    ) -> VideoRepairReceipt:
        assert job.hardware_encode and job.hardware_hevc
        assert job.source_clock is not None
        assert job.source_clock.pts == (30, 31)
        repair_stream(
            (np.zeros((8, 8, 3), dtype=np.uint8), np.zeros((8, 8, 3), dtype=np.uint8)),
            spans=job.spans,
            expected_frames=2,
            measure=spatial_evidence,
            quality_available=False,
            detect_batch=detect_batch,
            write=lambda image: None,
            receipt=lambda *args: None,
            evidence=evidence,
            source_clock=job.source_clock,
            context_frames=job.context_frames,
            context_donors_verified=job.context_donors_verified,
        )
        job.output.write_bytes(b"synthetic video")
        return VideoRepairReceipt(2, 0.1, 0, 1)

    monkeypatch.setattr(subtitle_candidate, "vision_text_masks", masks)
    monkeypatch.setattr(subtitle_candidate, "render_repaired_video", render)
    return path


@pytest.mark.parametrize("fault", [None, "clock", "partial", "pixels", "mixed", "mask", "unknown"])
def test_reviewed_temporal_candidate(config: Path, fault: str | None) -> None:
    data = json.loads(config.read_text(encoding="utf-8"))
    mask = config.parent / "glyph.png"
    assert cv2.imwrite(str(mask), np.full((8, 8), 255, np.uint8))
    review = config.parent / "review.json"
    review.write_text('{"review":"synthetic complete glyph support"}', encoding="utf-8")
    support = config.parent / "support.json"
    source_hash = hashlib.sha256(bytes(192)).hexdigest()
    support.write_text(
        json.dumps(
            {
                "sourceSHA256": data["sourceSHA256"],
                "timeBase": "1/30",
                "glyphsComplete": None if fault == "partial" else fault != "unknown",
                "reviewEvidence": {"path": str(review), "sha256": fingerprint(review)},
                "frames": [
                    {
                        "sourcePTS": pts if fault != "clock" else 0,
                        "sourceROISHA256": source_hash if fault != "pixels" else "wrong",
                        "mask": {"path": str(mask), "sha256": fingerprint(mask)},
                    }
                    for pts in (30, 31)
                ],
            }
        ),
        encoding="utf-8",
    )
    data["contextFrames"] = 1
    data["registeredGlyphSupport"] = {"path": str(support), "sha256": fingerprint(support)}
    data["evidenceFrames"] = [0, 1]
    if fault == "mixed":
        data["prepareVisionContext"] = True
    if fault == "mask":
        assert cv2.imwrite(str(mask), np.ones((8, 8), np.uint8))
    config.write_text(json.dumps(data), encoding="utf-8")
    if fault not in (None, "unknown"):
        with pytest.raises(ValueError):
            subtitle_candidate.render_candidate(config)
        return
    result = json.loads(subtitle_candidate.render_candidate(config).read_text(encoding="utf-8"))
    assert result["detectionBackend"] == "reviewed-glyph-support"
    assert result["unsupportedPixelsRetained"] is True
    assert result["qualityApproval"] is False
    assert result["losslessEvidence"]["frames"][0]["unsupportedPixels"] == 64
    assert result["losslessEvidence"]["frames"][0]["repairPixels"] == 0


def test_candidate_keeps_empty_support_unknown(config: Path) -> None:
    result = subtitle_candidate.render_candidate(config)
    data = json.loads(result.read_text(encoding="utf-8"))
    assert data["detectedSupportPixels"] == [64, 0]
    assert data["activeSourceFrames"] == [0, 1]
    assert data["emptyDetections"] == 1
    assert data["emptyDetectionIsNotAbsence"] is True
    assert data["eligibilityIsNotRepairQuality"] is True
    assert data["sourceEligibility"]["sha256"] == fingerprint(config.parent / "eligibility.json")
    assert data["qualityApproval"] is False
    assert data["deliveryComplete"] is False
    assert data["sourceStart"] == "1"
    assert result.stat().st_mode & 0o777 == 0o600
    with pytest.raises(FileExistsError):
        subtitle_candidate.render_candidate(config)


def test_explicit_media_keeps_receipt_and_log_local(config: Path) -> None:
    media_root = config.parent.parent / (config.parent.name + "-media")
    media_root.mkdir()
    media = media_root / "repaired.mp4"
    data = json.loads(config.read_text(encoding="utf-8"))
    data.update(mediaOutput=str(media), allowedMediaRoot=str(media_root))
    config.write_text(json.dumps(data), encoding="utf-8")
    receipt = subtitle_candidate.render_candidate(config)
    result = json.loads(receipt.read_text(encoding="utf-8"))
    assert receipt == config.parent / "output/candidate.json"
    assert media.read_bytes() == b"synthetic video"
    assert result["mediaOutput"] == str(media)
    assert result["qualityApproval"] is False
    assert not (config.parent / "output/candidate.mp4").exists()
    with pytest.raises(FileExistsError):
        subtitle_candidate.render_candidate(config)


def test_explicit_media_capacity_is_checked_on_its_actual_volume(
    config: Path, monkeypatch: MonkeyPatch
) -> None:
    media_root = config.parent.parent / (config.parent.name + "-media")
    media_root.mkdir()
    data = json.loads(config.read_text(encoding="utf-8"))
    data.update(mediaOutput=str(media_root / "candidate.mp4"), allowedMediaRoot=str(media_root))
    config.write_text(json.dumps(data), encoding="utf-8")
    monkeypatch.setattr(
        disk_budget,
        "measure",
        lambda path: disk_budget.Headroom(path, 100 * 1024**3, 30 * 1024**3, 30 * 1024**3, 0),
    )
    with pytest.raises(ValueError, match="media-volume capacity"):
        subtitle_candidate.render_candidate(config)
    assert not (config.parent / "output").exists()


@pytest.mark.parametrize(
    "field,value",
    [
        ("width", True),
        ("source", "relative"),
        ("visionTextRegions", None),
        ("sourceSHA256", "wrong"),
        ("clock", {"pts": "invalid", "timeBase": "1/30"}),
    ],
)
def test_invalid_config(config: Path, field: str, value: object) -> None:
    data = json.loads(config.read_text(encoding="utf-8"))
    data[field] = value
    config.write_text(json.dumps(data), encoding="utf-8")
    with pytest.raises(ValueError):
        subtitle_candidate.render_candidate(config)
    assert not (config.parent / "output").exists()


def test_prepared_detector_is_explicit(config: Path, monkeypatch: MonkeyPatch) -> None:
    data = json.loads(config.read_text())
    data["prepareVisionContext"] = True
    config.write_text(json.dumps(data))
    calls: list[int] = []

    def prepared(
        images: tuple[Image, ...], *, executable: Path, work_directory: Path
    ) -> subtitle_prepared.PreparedMasks:
        calls.append(len(images))
        return subtitle_prepared.PreparedMasks(
            tuple(np.full(image.shape[:2], 255, dtype=np.uint8) for image in images), (1, 1), (0, 1)
        )

    monkeypatch.setattr(subtitle_prepared, "prepared_text_masks", prepared)
    result = json.loads(subtitle_candidate.render_candidate(config).read_text())
    assert calls == [2]
    assert result["prepareVisionContext"] is True
    assert result["detectedSupportPixels"] == [64, 64]
    assert result["observedContextLines"] == [1, 1]
    assert result["clippedContextLines"] == [0, 1]
    assert result["qualityApproval"] is False


def test_invalid_prepared_policy(config: Path) -> None:
    data = json.loads(config.read_text())
    data.update(prepareVisionContext=True, visionTextRegions=False)
    config.write_text(json.dumps(data))
    with pytest.raises(ValueError, match="Prepared observation"):
        subtitle_candidate.render_candidate(config)


def test_learned_detector_requires_explicit_pinned_selection(
    config: Path, monkeypatch: MonkeyPatch
) -> None:
    model = config.with_name("model.onnx")
    model.write_bytes(b"synthetic model")
    digest = fingerprint(model)
    data = json.loads(config.read_text())
    data["learnedTextDetector"] = {"backend": "ppocrv3-db", "model": str(model), "sha256": digest}
    config.write_text(json.dumps(data))
    detector = Mock()
    detector.grid_sizes = [(736, 736), (736, 736)]
    detector.square_context_regions = [0, 0]
    detector.observed_regions = [0, 1]
    detector.intersected_regions = [0, 0]
    detector.detect.return_value = (
        np.zeros((8, 8), dtype=np.uint8),
        np.full((8, 8), 255, dtype=np.uint8),
    )
    factory = Mock(return_value=detector)
    monkeypatch.setattr(subtitle_detector, "LearnedTextDetector", factory)
    result = json.loads(subtitle_candidate.render_candidate(config).read_text())
    factory.assert_called_once_with(model, sha256=digest)
    assert result["detectionBackend"] == "ppocrv3-db"
    assert result["visionTextRegions"] is None
    assert result["inputs"][str(model)] == digest
    assert result["detectedSupportPixels"] == [0, 64]
    assert result["qualityApproval"] is False


@pytest.mark.parametrize(
    "changes",
    [
        {"prepareVisionContext": True},
        {"visionTextRegions": False},
        {"learnedTextDetector": {"backend": "unknown", "sha256": "0" * 64}},
        {"learnedTextDetector": {"backend": "ppocrv3-db", "sha256": False}},
    ],
)
def test_incompatible_detector_selection_fails(config: Path, changes: dict[str, object]) -> None:
    data = json.loads(config.read_text())
    data["learnedTextDetector"] = {"backend": "ppocrv3-db", "sha256": "0" * 64}
    data.update(changes)
    config.write_text(json.dumps(data))
    with pytest.raises(ValueError, match="learned detector"):
        subtitle_candidate.render_candidate(config)


def test_linked_config(config: Path) -> None:
    link = config.with_name("linked.json")
    link.symlink_to(config)
    with pytest.raises(ValueError, match="regular"):
        subtitle_candidate.render_candidate(link)


def test_incomplete_detection_keeps_failed_candidate(
    config: Path, monkeypatch: MonkeyPatch
) -> None:
    def render(
        job: VideoRepairJob, *, detect_batch: BatchDetector, evidence: EvidenceSink | None = None
    ) -> VideoRepairReceipt:
        job.output.write_bytes(b"partial evidence")
        return VideoRepairReceipt(2, 0.1, 0, 1)

    monkeypatch.setattr(subtitle_candidate, "render_repaired_video", render)
    with pytest.raises(ValueError, match="accounting"):
        subtitle_candidate.render_candidate(config)
    assert (config.parent / "output/candidate.mp4").read_bytes() == b"partial evidence"
    assert not (config.parent / "output/candidate.json").exists()


def test_changed_input_is_not_approved(config: Path, monkeypatch: MonkeyPatch) -> None:
    def render(
        job: VideoRepairJob, *, detect_batch: BatchDetector, evidence: EvidenceSink | None = None
    ) -> VideoRepairReceipt:
        detect_batch((np.zeros((8, 8, 3), dtype=np.uint8), np.zeros((8, 8, 3), dtype=np.uint8)))
        job.output.write_bytes(b"synthetic video")
        job.source.write_bytes(b"changed")
        return VideoRepairReceipt(2, 0.1, 0, 1)

    monkeypatch.setattr(subtitle_candidate, "render_repaired_video", render)
    with pytest.raises(ValueError, match="during rendering"):
        subtitle_candidate.render_candidate(config)


def test_opt_in_capture_binds_writer_and_inputs(config: Path, monkeypatch: MonkeyPatch) -> None:
    data = json.loads(config.read_text())
    data["evidenceFrames"] = [0, 1]
    parent = config.parent / "parent.json"
    parent.write_text("{}", encoding="utf-8")
    data["probeParent"] = {"path": str(parent), "sha256": fingerprint(parent)}
    config.write_text(json.dumps(data))
    monkeypatch.setattr(shutil, "which", lambda name: str(config.parent / "native"))
    result = json.loads(subtitle_candidate.render_candidate(config).read_text())
    assert result["losslessEvidence"]["frames"][0]["sourcePTS"] == 30
    assert result["losslessEvidence"]["frames"][1]["glyphPixels"] == 0
    assert result["losslessEvidence"]["qualityApproval"] is False
    assert result["videoSHA256"] == fingerprint(config.parent / "output/candidate.mp4")
    assert result["inputs"][str(config)] == fingerprint(config)
    assert result["evidenceRuntime"]["numpy"] == np.__version__


@pytest.mark.parametrize("escape", [False, True])
def test_probe_provenance_cannot_be_forged(config: Path, escape: bool) -> None:
    data = json.loads(config.read_text())
    data["probeParent"] = {
        "path": str(config.parent.parent / "outside.json" if escape else config),
        "sha256": "0" * 64,
    }
    config.write_text(json.dumps(data))
    with pytest.raises(ValueError, match="provenance"):
        subtitle_candidate.render_candidate(config)


def test_capture_requires_encoder_identity(config: Path, monkeypatch: MonkeyPatch) -> None:
    data = json.loads(config.read_text())
    data["evidenceFrames"] = [0]
    config.write_text(json.dumps(data))
    monkeypatch.setattr(shutil, "which", lambda name: None)
    with pytest.raises(ValueError, match="decoder/encoder"):
        subtitle_candidate.render_candidate(config)


def test_cli(config: Path) -> None:
    assert subtitle_candidate.main([str(config)]) == 0


def test_module_help(monkeypatch: MonkeyPatch) -> None:
    monkeypatch.setattr(sys, "argv", ["subtitle_candidate", "--help"])
    monkeypatch.delitem(sys.modules, "replay.subtitle_candidate")
    with pytest.raises(SystemExit) as error:
        runpy.run_module("replay.subtitle_candidate", run_name="__main__")
    assert error.value.code == 0


def test_unqualified_roi_fails_before_source_hash_or_render(
    config: Path, monkeypatch: MonkeyPatch
) -> None:
    data = json.loads(config.read_text())
    del data["sourceEligibility"]
    config.write_text(json.dumps(data))
    source_hash = Mock(side_effect=AssertionError("Must not hash source"))
    monkeypatch.setattr(subtitle_candidate, "fingerprint", source_hash)
    with pytest.raises(ValueError):
        subtitle_candidate.render_candidate(config)
    source_hash.assert_not_called()
    assert not (config.parent / "output").exists()


@pytest.mark.parametrize(
    "field,value",
    [
        ("classification", "garment-pattern"),
        ("classification", "logo"),
        ("classification", "unknown"),
        ("classification", "absent"),
        ("classification", None),
        ("sourcePTS", 29),
        ("sourcePTS", True),
        ("observation", 1),
        ("observation", -1),
        ("observation", 2),
        ("lines", []),
        ("lines", [True]),
        ("lines", [0, 0]),
        ("lines", [1]),
        ("lines", [-1]),
        ("region", {"x": 1, "y": 0, "width": 7, "height": 8}),
    ],
)
def test_semantic_selection_is_not_db_support(config: Path, field: str, value: object) -> None:
    eligibility = config.parent / "eligibility.json"
    selection = json.loads(eligibility.read_text())
    selection["frames"][0][field] = value
    eligibility.write_text(json.dumps(selection))
    data = json.loads(config.read_text())
    data["sourceEligibility"]["sha256"] = fingerprint(eligibility)
    config.write_text(json.dumps(data))
    with pytest.raises(ValueError):
        subtitle_candidate.render_candidate(config)
    assert not (config.parent / "output").exists()


@pytest.mark.parametrize(
    "field,value",
    [
        ("sourceSHA256", "wrong"),
        ("version", True),
        ("version", 2),
        ("kind", "text-region-detector"),
        ("timeBase", "1/60"),
        ("canvas", [16, 8]),
        ("frames", []),
        ("frames", None),
        ("decoder", None),
        ("codec", {"path": "relative", "sha256": "wrong"}),
        ("classificationEvidence", {"path": 42, "sha256": None}),
        ("ocr", {"path": "/outside.json", "sha256": "wrong"}),
    ],
)
def test_eligibility_identity_is_required(config: Path, field: str, value: object) -> None:
    eligibility = config.parent / "eligibility.json"
    selection = json.loads(eligibility.read_text())
    selection[field] = value
    eligibility.write_text(json.dumps(selection))
    data = json.loads(config.read_text())
    data["sourceEligibility"]["sha256"] = fingerprint(eligibility)
    config.write_text(json.dumps(data))
    with pytest.raises(ValueError):
        subtitle_candidate.render_candidate(config)
    assert not (config.parent / "output").exists()


@pytest.mark.parametrize(
    "name", ["codec.json", "native", "ocr.json", "classification.json", "ocr-provenance.json"]
)
def test_stale_semantic_inputs_fail(config: Path, name: str) -> None:
    (config.parent / name).write_bytes(b"changed")
    with pytest.raises(ValueError, match="fingerprint"):
        subtitle_candidate.render_candidate(config)
    assert not (config.parent / "output").exists()


@pytest.mark.parametrize(
    "change",
    ["empty", "blank", "outside", "shifted", "requested-shifted", "canvas", "duplicate"],
)
def test_actual_ocr_is_bound_to_frame_and_region(config: Path, change: str) -> None:
    eligibility = config.parent / "eligibility.json"
    selection = json.loads(eligibility.read_text())
    ocr = config.parent / "ocr.json"
    observations = json.loads(ocr.read_text())
    if change == "empty":
        observations[0]["lines"] = []
    elif change == "blank":
        observations[0]["lines"][0]["text"] = " "
    elif change == "outside":
        observations[0]["lines"][0]["region"]["x"] = 3
    elif change == "shifted":
        observations[0]["actualTimeSeconds"] = 1.001
    elif change == "requested-shifted":
        observations[0]["requestedTimeSeconds"] = 1.001
    elif change == "canvas":
        observations[0]["width"] = 16
    else:
        selection["frames"][1]["sourcePTS"] = 30
    ocr.write_text(json.dumps(observations))
    selection["ocr"]["sha256"] = fingerprint(ocr)
    eligibility.write_text(json.dumps(selection))
    data = json.loads(config.read_text())
    data["sourceEligibility"]["sha256"] = fingerprint(eligibility)
    config.write_text(json.dumps(data))
    with pytest.raises(ValueError):
        subtitle_candidate.render_candidate(config)
    assert not (config.parent / "output").exists()


@pytest.mark.parametrize(
    "field,value",
    [
        ("sourceSHA256", "wrong"),
        ("measurementTool", "image_text_rectangles_batch"),
        ("fullFrame", False),
        ("nativeExecutable", "relative"),
        ("nativeExecutable", None),
        ("nativeExecutable", "/outside"),
        ("nativeSHA256", "wrong"),
    ],
)
def test_ocr_provenance_is_not_arbitrary_text(config: Path, field: str, value: object) -> None:
    provenance = config.parent / "ocr-provenance.json"
    producer = json.loads(provenance.read_text())
    producer[field] = value
    provenance.write_text(json.dumps(producer))
    eligibility = config.parent / "eligibility.json"
    selection = json.loads(eligibility.read_text())
    selection["ocrProvenance"]["sha256"] = fingerprint(provenance)
    eligibility.write_text(json.dumps(selection))
    data = json.loads(config.read_text())
    data["sourceEligibility"]["sha256"] = fingerprint(eligibility)
    config.write_text(json.dumps(data))
    with pytest.raises(ValueError):
        subtitle_candidate.render_candidate(config)
    assert not (config.parent / "output").exists()


@pytest.mark.parametrize("executable", [True, False])
def test_installed_decoder_is_read_only_not_evidence_storage(
    config: Path, executable: bool
) -> None:
    decoder = config.parent.parent / "installed-decoder"
    decoder.write_bytes(b"synthetic installed decoder")
    decoder.chmod(0o700 if executable else 0o600)
    eligibility = config.parent / "eligibility.json"
    selection = json.loads(eligibility.read_text())
    selection["decoder"] = {"path": str(decoder), "sha256": fingerprint(decoder)}
    eligibility.write_text(json.dumps(selection))
    data = json.loads(config.read_text())
    data["sourceEligibility"]["sha256"] = fingerprint(eligibility)
    config.write_text(json.dumps(data))
    if executable:
        result = json.loads(subtitle_candidate.render_candidate(config).read_text())
        assert result["inputs"][str(decoder)] == fingerprint(decoder)
    else:
        with pytest.raises(ValueError, match="installed executable"):
            subtitle_candidate.render_candidate(config)
        assert not (config.parent / "output").exists()


def test_linked_eligibility_is_not_followed(config: Path) -> None:
    eligibility = config.parent / "eligibility.json"
    link = config.parent / "eligibility-link.json"
    link.symlink_to(eligibility)
    data = json.loads(config.read_text())
    data["sourceEligibility"]["path"] = str(link)
    config.write_text(json.dumps(data))
    with pytest.raises(ValueError, match="local evidence"):
        subtitle_candidate.render_candidate(config)
    assert not (config.parent / "output").exists()


def test_changed_classification_during_render_is_not_published(
    config: Path, monkeypatch: MonkeyPatch
) -> None:
    def render(
        job: VideoRepairJob, *, detect_batch: BatchDetector, evidence: EvidenceSink | None = None
    ) -> VideoRepairReceipt:
        detect_batch((np.zeros((8, 8, 3), dtype=np.uint8), np.zeros((8, 8, 3), dtype=np.uint8)))
        job.output.write_bytes(b"synthetic video")
        (config.parent / "classification.json").write_text("changed")
        return VideoRepairReceipt(2, 0.1, 0, 1)

    monkeypatch.setattr(subtitle_candidate, "render_repaired_video", render)
    with pytest.raises(ValueError, match="during rendering"):
        subtitle_candidate.render_candidate(config)
    assert not (config.parent / "output/candidate.json").exists()


@pytest.fixture
def saved_config(config: Path, monkeypatch: MonkeyPatch) -> tuple[Path, Path, Path]:
    monkeypatch.chdir(Path(__file__).resolve().parents[1])
    data = json.loads(config.read_text())
    data["clock"]["pts"] = [30]
    config.write_text(json.dumps(data))
    source = np.zeros((8, 8, 3), dtype=np.uint8)
    source[3:5, 3:5] = 255
    ink = np.zeros((8, 8), dtype=np.uint8)
    ink[3:5, 3:5] = 255
    allowed = np.zeros((8, 8), dtype=np.uint8)
    allowed[1:7, 1:7] = 255
    assert cv2.imwrite(str(config.parent / "saved-source.png"), source)
    assert cv2.imwrite(str(config.parent / "ink.png"), ink)
    assert cv2.imwrite(str(config.parent / "allowed.png"), allowed)
    review = config.parent / "pixel-review.json"
    review.write_text(
        json.dumps(
            {
                "version": 1,
                "kind": "reviewed-line-ink-core",
                "configSHA256": fingerprint(config),
                "sourcePTS": 30,
                "inkClassified": True,
                "repairSupportReviewed": True,
                "fullCaptionSupport": False,
                "sourcePNG": {
                    "path": str(config.parent / "saved-source.png"),
                    "sha256": fingerprint(config.parent / "saved-source.png"),
                },
                "lines": [
                    {
                        "line": 0,
                        "ink": {
                            "path": str(config.parent / "ink.png"),
                            "sha256": fingerprint(config.parent / "ink.png"),
                        },
                        "allowed": {
                            "path": str(config.parent / "allowed.png"),
                            "sha256": fingerprint(config.parent / "allowed.png"),
                        },
                    }
                ],
            }
        ),
        encoding="utf-8",
    )
    return config, review, config.parent / "saved-candidate"


@pytest.mark.parametrize("tamper", [False, True])
def test_saved_candidate_explicit_localization_review(
    saved_config: tuple[Path, Path, Path], tamper: bool
) -> None:
    config, review, directory = saved_config
    evidence = config.parent / "segmentation.json"
    evidence.write_text('{"observation": "synthetic pixel review"}', encoding="utf-8")
    data = json.loads(review.read_text(encoding="utf-8"))
    data.update(
        version=2,
        kind="reviewed-line-ink-support",
        reviewerType="agent",
        ocrIsLocalizationOnly=True,
        segmentationEvidence={"path": str(evidence), "sha256": fingerprint(evidence)},
    )
    review.write_text(json.dumps(data), encoding="utf-8")
    if tamper:
        evidence.write_text("{}", encoding="utf-8")
        with pytest.raises(ReplayError, match="Changed input"):
            subtitle_line_candidate.render_saved_candidate(config, review, directory)
    else:
        receipt = subtitle_line_candidate.render_saved_candidate(config, review, directory)
        result = json.loads(receipt.read_text(encoding="utf-8"))
        assert result["ocrIsLocalizationOnly"] is True
        assert result["qualityApproved"] is False
        assert result["changedPixelsOutsidePermission"] == 0
        assert str(evidence) in result["inputs"]


def test_saved_candidate_is_partial_single_glyph_not_quality(
    saved_config: tuple[Path, Path, Path],
) -> None:
    config, review, directory = saved_config
    receipt = subtitle_line_candidate.render_saved_candidate(config, review, directory)
    data = json.loads(receipt.read_text())
    assert data["method"] == "GLYPH"
    assert data["softness"] == 0
    assert data["inkPixels"] == 4
    assert data["repairPixels"] == 36
    assert data["changedPixelsOutsidePermission"] == 0
    assert data["sourceMediaReverified"] is False
    assert data["sourcePNGReverified"] is True
    assert data["fullCaptionSupport"] is False
    assert data["qualityApproved"] is False
    assert data["temporalMeasured"] is False
    assert data["deliveryComplete"] is False
    assert data["images"]["candidate"]["roundtripIdentical"] is True
    with pytest.raises(FileExistsError):
        subtitle_line_candidate.render_saved_candidate(config, review, directory)


@pytest.mark.parametrize(
    "field,value",
    [
        ("version", 2),
        ("kind", "db-regions"),
        ("inkClassified", False),
        ("repairSupportReviewed", False),
        ("fullCaptionSupport", True),
        ("configSHA256", "wrong"),
        ("sourcePTS", 31),
        ("lines", []),
    ],
)
def test_saved_candidate_refuses_missing_review(
    saved_config: tuple[Path, Path, Path], field: str, value: object
) -> None:
    config, review, directory = saved_config
    data = json.loads(review.read_text())
    data[field] = value
    review.write_text(json.dumps(data))
    with pytest.raises(ValueError):
        subtitle_line_candidate.render_saved_candidate(config, review, directory)
    assert not directory.exists()


@pytest.mark.parametrize("kind", ["clock", "source", "spans"])
def test_saved_candidate_refuses_non_single_bound_frame(
    saved_config: tuple[Path, Path, Path], kind: str
) -> None:
    config, review, directory = saved_config
    data = json.loads(config.read_text())
    if kind == "clock":
        data["clock"]["pts"] = [30, 31]
    elif kind == "source":
        data["sourceSHA256"] = None
    else:
        data["masks"] = []
    config.write_text(json.dumps(data))
    reviewed = json.loads(review.read_text())
    reviewed["configSHA256"] = fingerprint(config)
    review.write_text(json.dumps(reviewed))
    with pytest.raises(ValueError):
        subtitle_line_candidate.render_saved_candidate(config, review, directory)


@pytest.mark.parametrize("kind", ["unknown", "empty", "bad-shape", "bad-png", "stale"])
def test_saved_candidate_refuses_unknown_or_changed_masks(
    saved_config: tuple[Path, Path, Path], kind: str
) -> None:
    config, review, directory = saved_config
    mask = config.parent / "allowed.png"
    pixels = np.zeros((8, 8), dtype=np.uint8)
    if kind == "unknown":
        pixels[0:8, 0:8] = 255
    elif kind == "bad-shape":
        pixels = np.zeros((9, 9), dtype=np.uint8)
    if kind in ("bad-png", "stale"):
        mask.write_bytes(b"not a PNG")
    else:
        assert cv2.imwrite(str(mask), pixels)
    data = json.loads(review.read_text())
    if kind != "stale":
        data["lines"][0]["allowed"]["sha256"] = fingerprint(mask)
    review.write_text(json.dumps(data))
    with pytest.raises((ValueError, ReplayError)):
        subtitle_line_candidate.render_saved_candidate(config, review, directory)
    assert not directory.exists()


def test_saved_candidate_rechecks_inputs(
    saved_config: tuple[Path, Path, Path], monkeypatch: MonkeyPatch
) -> None:
    config, review, directory = saved_config

    def changed(path: Path, pixels: Image) -> dict[str, object]:
        review.write_text("changed")
        return _png(path, pixels)

    monkeypatch.setattr(subtitle_line_candidate, "_png", changed)
    with pytest.raises(ValueError, match="inputs changed"):
        subtitle_line_candidate.render_saved_candidate(config, review, directory)
    assert not (directory / "candidate.json").exists()


def test_saved_candidate_cli(saved_config: tuple[Path, Path, Path]) -> None:
    config, review, directory = saved_config
    assert subtitle_line_candidate.main([str(config), str(review), str(directory)]) == 0


def test_saved_module_help(monkeypatch: MonkeyPatch) -> None:
    monkeypatch.setattr(sys, "argv", ["subtitle_line_candidate", "--help"])
    monkeypatch.delitem(sys.modules, "replay.subtitle_line_candidate")
    with pytest.raises(SystemExit) as error:
        runpy.run_module("replay.subtitle_line_candidate", run_name="__main__")
    assert error.value.code == 0
