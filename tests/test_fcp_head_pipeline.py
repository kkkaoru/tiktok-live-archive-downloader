import json
import sys
from dataclasses import replace
from fractions import Fraction
from pathlib import Path
from unittest.mock import Mock
from xml.etree import ElementTree as ET

import pytest
from test_caption_native_layout import layout as layout
from test_caption_native_reflow import exact as exact
from test_fcp_caption_retarget import template as template
from test_fcp_head_near import paired as paired

from replay import fcp_head_pipeline as pipeline
from replay import reflow_producer_evidence
from replay.caption_measure_cache import CachedCaptionLayout
from replay.caption_native_layout import NativeCaptionLayout
from replay.core import ReplayError
from replay.delivery_captions import Cut
from replay.fcp_caption_retarget import Cue, RetainedSpan
from replay.fcp_caption_visibility import VisibilityBinding
from replay.fcp_head_near import FaceScaleSeries
from replay.fcp_project import CaptionTemplate, MediaInfo
from replay.head_near_placement import FaceSizeObservation
from replay.stages import file_digest
from replay.subtitle_clock import VideoFrameClock
from replay.tracking_observations import TrackingJob
from replay.tracking_retained import READER_FILES, RetainedTracking


@pytest.mark.parametrize("saved_matches", [True, False])
@pytest.mark.parametrize("automatic", [True, False])
@pytest.mark.parametrize("historical", [True, False])
@pytest.mark.parametrize(
    "failure", ["none", "collect", "immediate", "unknown", "mutated", "joint-mutated"]
)
@pytest.mark.parametrize("visibility", [False, True])
def test_pipeline_connects_scale_cache_paging_and_semantic_xml(
    tmp_path: Path,
    exact: NativeCaptionLayout,
    paired: FaceScaleSeries,
    template: CaptionTemplate,
    monkeypatch: pytest.MonkeyPatch,
    saved_matches: bool,
    automatic: bool,
    historical: bool,
    failure: str,
    visibility: bool,
) -> None:
    paired = replace(
        paired,
        sizes=tuple(
            FaceSizeObservation(top.time, bottom.y - top.y, bottom.y - top.y)
            for top, bottom in zip(paired.tops.positions, paired.bottoms.positions, strict=True)
        ),
    )
    config = tmp_path / "config.json"
    paths = {
        name: tmp_path / name
        for name in (
            "faceIndex",
            "cuts",
            "reflow",
            "producerEvidence",
            "segmentIndex",
            "nativeExecutable",
            "longExecutable",
            "measurementReference",
            "template",
            "voice",
        )
    }
    documents: dict[Path, dict[str, object]] = {
        config: {
            "version": 1,
            **{key: str(value) for key, value in paths.items()},
            "referenceTime": "0",
            "referenceHeadTop": 500,
            "referenceROI": [0, 900, 100, 400],
            "terminalFrameCell": True,
            "resumeParts": {"0": str(tmp_path / "prepared.json")},
        },
        paths["segmentIndex"]: {
            "rate": 30,
            "segments": [{"segment": 0, "frames": 60, "offsetFrame": 0, "jobs": [0]}],
        },
        paths["reflow"]: {
            "inputs": {
                str(paths["segmentIndex"]): "a" * 64,
                str(tmp_path / "part-0000.json"): "a" * 64,
            }
        },
        tmp_path / "part-0000.json": {
            "inputs": {},
            "captions": [{"start": 0, "end": 2, "text": "ab"}],
            "coverage": {"start": 0, "end": 2},
        },
    }
    documents[tmp_path / "prepared.json"] = (
        documents[tmp_path / "part-0000.json"] if saved_matches else {"captions": []}
    )
    if automatic:
        documents[config].pop("referenceTime")
        documents[config].pop("referenceHeadTop")
        documents[config].pop("referenceROI")
        documents[config]["headReferencePolicy"] = "automatic-face-relative"
    if not historical:
        documents[config].pop("producerEvidence")
    job = TrackingJob(
        tmp_path / "job",
        "a" * 64,
        tmp_path / "source",
        tmp_path / "native",
        "b" * 64,
        "c" * 64,
        VideoFrameClock((0, 1), Fraction(1), 1),
        1080,
        1920,
        1,
    )
    monkeypatch.setattr(pipeline, "_json", lambda path: documents[path])
    monkeypatch.setattr(pipeline, "caption_reflow_json", lambda path: documents[path])
    monkeypatch.setattr(pipeline, "_pin", lambda value: Path(value))
    monkeypatch.setattr(pipeline, "file_digest", lambda path: "a" * 64)
    monkeypatch.setattr(pipeline, "face_references", lambda path: [])
    monkeypatch.setattr(pipeline, "read_face_scale", lambda refs, **kwargs: (job, paired))
    monkeypatch.setattr(pipeline, "read_cuts", lambda path: (Cut(0, 60, 0),))
    if visibility:
        documents[config]["candidateVisibility"] = {"explicit": True}
        visibility_binding = VisibilityBinding(
            ((1.0, 1.5),),
            {str(tmp_path / "visibility-proof.json"): "a" * 64},
            {"candidateOnly": True, "speechAccuracyVerified": False},
        )
        monkeypatch.setattr(pipeline, "read_binding", lambda value, **kwargs: visibility_binding)
    else:
        monkeypatch.setattr(
            pipeline,
            "read_binding",
            Mock(side_effect=AssertionError("Ordinary path must not hide")),
        )

    def verify_reflow(
        path: Path,
        spans: tuple[RetainedSpan, ...],
        *,
        source: Path,
        source_sha256: str,
        producer_evidence: Path | None,
    ) -> tuple[Cue, ...]:
        assert producer_evidence == (paths["producerEvidence"] if historical else None)
        return ()

    historical_calls: list[Path] = []

    def historical_read(path: Path, *, artifact: Path, inputs: dict[str, object]) -> dict[str, str]:
        historical_calls.append(path)
        return {}

    monkeypatch.setattr(pipeline, "read_reflow", verify_reflow)
    monkeypatch.setattr(reflow_producer_evidence, "read", historical_read)
    cached = CachedCaptionLayout(exact)
    monkeypatch.setattr(cached, "verify_inputs", lambda: {})

    def layout(**kwargs: object) -> CachedCaptionLayout:
        assert kwargs["long_executable"] == paths["longExecutable"]
        return cached

    monkeypatch.setattr(pipeline, "make_layout", layout)
    monkeypatch.setattr(
        pipeline, "probe_media", lambda path: MediaInfo(Fraction(2), True, True, audio_channels=2)
    )
    monkeypatch.setattr(pipeline, "caption_template", lambda path, **kwargs: template)
    if failure != "none":
        documents[config]["collectLayoutFailures"] = failure != "immediate"
        documents[config].pop("resumeParts")
        documents[paths["segmentIndex"]]["segments"] = [
            {"segment": number, "frames": 30, "offsetFrame": number * 30, "jobs": [number]}
            for number in range(2)
        ]
        documents[tmp_path / "part-0001.json"] = documents[tmp_path / "part-0000.json"]
        inputs = documents[paths["reflow"]]["inputs"]
        assert isinstance(inputs, dict)
        inputs[str(tmp_path / "part-0001.json")] = "a" * 64
        calls: list[int] = []

        def repage(**kwargs: object) -> tuple[dict[str, object], bool]:
            segment = kwargs["segment"]
            assert isinstance(segment, dict)
            number = segment["segment"]
            assert isinstance(number, int)
            calls.append(number)
            if number == 0:
                message = (
                    "Invalid input hash"
                    if failure == "unknown"
                    else ("Native token exceeds display budget; finer alignment is required")
                )
                raise ValueError(message)
            if failure == "mutated":
                monkeypatch.setattr(pipeline, "file_digest", lambda path: "b" * 64)
            elif failure == "joint-mutated":
                monkeypatch.setattr(
                    pipeline,
                    "file_digest",
                    lambda path: "b" * 64 if path.name == "caption_joint_fit.py" else "a" * 64,
                )
            return documents[tmp_path / "part-0001.json"], False

        monkeypatch.setattr(pipeline, "repage_segment", repage)
        expected_error = {
            "collect": "Head layout failures collected",
            "immediate": "Native token exceeds display budget",
            "unknown": "Invalid input hash",
            "mutated": "Head placement inputs changed",
            "joint-mutated": "Head placement inputs changed",
        }[failure]
        with pytest.raises((ReplayError, ValueError), match=expected_error):
            pipeline.prepare(config, tmp_path / "output")
        assert calls == ([0, 1] if failure in {"collect", "mutated", "joint-mutated"} else [0])
        output = tmp_path / "output"
        assert not (output / "semantic.fcpxml").exists()
        assert not (output / "receipt.json").exists()
        assert not (output / "clean-transform-config.json").exists()
        if failure == "collect":
            summary = json.loads((output / "layout-failures.json").read_text())
            assert summary["complete"] is False
            assert summary["readyForRender"] is False
            assert summary["deliveryComplete"] is False
            assert summary["failures"][0]["segment"] == 0
            assert summary["successfulSegments"][0]["segment"] == 1
            assert (output / "part-0001.json").exists()
            assert not (output / "part-0000.json").exists()
        else:
            assert not (output / "layout-failures.json").exists()
        return
    destination = pipeline.prepare(config, tmp_path / "output")
    assert historical_calls == ([paths["producerEvidence"]] if historical else [])
    xml = ET.parse(destination)
    key = xml.find(".//title/adjust-transform/param/keyframeAnimation/keyframe")
    assert key is not None
    if automatic:
        assert key.get("value") == "0 30.4166666667"
    else:
        assert key.get("value") == "0 25.2083333333"
    receipt = json.loads((destination.parent / "receipt.json").read_text(encoding="utf-8"))
    assert any(path.endswith("/caption_y_stability.py") for path in receipt["inputs"])
    assert any(path.endswith("/fcp_caption_stability.py") for path in receipt["inputs"])
    joint_source = str(Path(pipeline.__file__).with_name("caption_joint_fit.py").resolve())
    assert receipt["inputs"][joint_source] == "a" * 64
    assert receipt["readyForRender"] is False
    assert receipt["faceScale"]["derivedSamples"] == 2
    assert receipt["faceScale"]["rawGeometryUnchanged"] is True
    assert receipt["faceScale"]["missingObservationsRecovered"] is False
    assert receipt["titleGeometry"][0]["headNearVerified"] is False
    assert receipt["requiredNextStage"] == "verified-clean-source-substitution"
    assert receipt["segments"][0]["segment"] == 0
    if visibility:
        assert receipt["candidateVisibility"]["visibleTitleIntervalsVerified"] is True
        assert receipt["candidateVisibility"]["completeCaptionsVerified"] is False
        title = xml.find(".//title")
        assert title is not None
        assert title.get("duration") == "1s"
    else:
        assert "candidateVisibility" not in receipt
    if automatic:
        assert receipt["segments"][0]["maximumHeight"] == 251
        assert (
            receipt["headReference"]["referenceROIMeaning"] == "full-width-measured-vertical-band"
        )
        assert receipt["headReference"]["semanticHeadDetected"] is False
    else:
        assert receipt["segments"][0]["maximumHeight"] == 276
    assert receipt["segments"][0]["repaged"] is False
    if saved_matches:
        assert receipt["segments"][0]["reusedPreparedPart"].endswith("prepared.json")
    else:
        assert receipt["segments"][0]["reusedPreparedPart"] is None
    assert receipt["terminalFrameCell"]["observedCoverageExtended"] is False
    with pytest.raises(ReplayError, match="new canonical"):
        pipeline.prepare(config, tmp_path / "output")
    monkeypatch.setattr(pipeline, "prepare", lambda config, output: Path("/candidate.fcpxml"))
    monkeypatch.setattr(sys, "argv", ["pipeline", str(config), str(tmp_path / "unused")])
    pipeline.main()


@pytest.mark.parametrize("part", [{}, {"captions": [{"text": 1}]}])
def test_missing_text_bounds_fail(exact: NativeCaptionLayout, part: dict[str, object]) -> None:
    with pytest.raises(ReplayError):
        pipeline._text_bounds(part, CachedCaptionLayout(exact))


def test_failure_collection_requires_boolean(tmp_path: Path) -> None:
    config = tmp_path / "config.json"
    config.write_text('{"version":1,"collectLayoutFailures":1}', encoding="utf-8")
    with pytest.raises(ReplayError, match="explicit boolean"):
        pipeline.prepare(config, tmp_path / "output")


def test_reflow_byte_failure_precedes_face_verification(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config = tmp_path / "config.json"
    config.write_text('{"version":1}', encoding="utf-8")
    monkeypatch.setattr(pipeline, "_pin", lambda value: tmp_path / "artifact")
    face = Mock(side_effect=AssertionError("Face verification must not start"))
    monkeypatch.setattr(pipeline, "read_face_scale", face)
    monkeypatch.setattr(
        pipeline,
        "caption_reflow_json",
        Mock(side_effect=ValueError("Tracking artifact exceeds its byte budget")),
    )
    with pytest.raises(ValueError, match="byte budget"):
        pipeline.prepare(config, tmp_path / "output")
    face.assert_not_called()
    assert not (tmp_path / "output").exists()


def test_explicit_tracking_catalog_is_verified_saved_and_passed_to_face_consumer(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config = tmp_path / "config.json"
    config.write_text('{"version":1,"retainedTracking":{}}', encoding="utf-8")
    descriptor = tmp_path / "descriptor.json"
    descriptor.write_text("{}", encoding="utf-8")
    monkeypatch.setattr(pipeline, "_pin", lambda value: descriptor)
    monkeypatch.setattr(pipeline, "caption_reflow_json", lambda path: {})
    references = [(tmp_path / "window.json", tmp_path / "observed")]
    monkeypatch.setattr(pipeline, "face_references", lambda path: references)
    retained = RetainedTracking(
        (),
        {str(descriptor): file_digest(descriptor)}
        | {
            str(Path(pipeline.__file__).with_name(name)): file_digest(
                Path(pipeline.__file__).with_name(name)
            )
            for name in READER_FILES
        },
    )
    verify_catalog = Mock(return_value=retained)
    monkeypatch.setattr(pipeline, "verify_catalog", verify_catalog)
    face = Mock(side_effect=ValueError("bounded consumer test stop"))
    monkeypatch.setattr(pipeline, "read_face_scale", face)
    with pytest.raises(ValueError, match="bounded consumer test stop"):
        pipeline.prepare(config, tmp_path / "output")
    verify_catalog.assert_called_once_with(
        descriptor, sha256=file_digest(descriptor), references=references
    )
    face.assert_called_once_with(
        references, missing_policy="hold", seed_overlap="next", retained=retained
    )
    receipt = json.loads((tmp_path / "output/retained-tracking-receipt.json").read_text())
    assert receipt["mediaBytesVerified"] is True
    assert receipt["captionPlacementApproved"] is False


def test_clean_config_binds_new_semantic_hash(tmp_path: Path) -> None:
    semantic = tmp_path / "semantic.fcpxml"
    semantic.write_text("<fcpxml/>", encoding="utf-8")
    evidence = tmp_path / "evidence.json"
    evidence.write_text("{}", encoding="utf-8")
    pin: dict[str, object] = {"path": str(evidence), "sha256": file_digest(evidence)}
    output = tmp_path / "clean.json"
    pipeline._clean_config(
        {"assemblyManifest": pin, "assemblyReceipt": pin, "cleanIndex": pin, "voiceReceipt": pin},
        semantic=semantic,
        voice=pin,
        output=output,
    )
    saved = json.loads(output.read_text(encoding="utf-8"))
    assert saved["removeSourceMasks"] is True
    assert saved["version"] == 1
    assert (
        saved["semanticXML"]["sha256"]
        == "979b046eafb48ab6dee711d09591a6bdef92f0b1da8780d5ecab43d760608a0a"
    )
