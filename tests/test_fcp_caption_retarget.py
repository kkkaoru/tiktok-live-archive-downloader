import json
import sys
from dataclasses import replace
from fractions import Fraction
from pathlib import Path
from xml.etree import ElementTree as ET

import pytest

from replay import caption_native_paging as paging
from replay import fcp_caption_retarget as retarget
from replay import reflow_producer_evidence
from replay.core import ReplayError
from replay.delivery_captions import Cut as SourceCut
from replay.fcp_measured import CaptionAnchor
from replay.fcp_project import CaptionTemplate, MediaInfo
from replay.fcp_tracking_caption import (
    FaceObservation,
    FaceObservationSeries,
    FacePositionSeries,
    MissingFaceRange,
    TrackingCalibration,
    TrackingPosition,
)
from replay.stages import file_digest
from replay.subtitle_clock import VideoFrameClock
from replay.tracking_observations import TrackingJob


@pytest.fixture
def faces() -> FacePositionSeries:
    return FacePositionSeries(
        (
            TrackingPosition(Fraction(2), 300, 700),
            TrackingPosition(Fraction(3), 310, 710),
            TrackingPosition(Fraction(4), 320, 720),
        ),
        Fraction(5),
        1080,
        1920,
    )


@pytest.fixture
def template() -> CaptionTemplate:
    return CaptionTemplate(
        "Bottom",
        "~/Titles.localized/Test/Test.moti",
        "file:///test.moti",
        text_objects=3,
        anchor=CaptionAnchor.BOTTOM,
    )


@pytest.fixture
def calibration() -> TrackingCalibration:
    return TrackingCalibration(Fraction(2), 500, 1, 0, 24, 1920, (300, 700, 100, 100))


def test_cut_projection_keeps_text_clock_and_source_gap() -> None:
    spans = retarget.selected_spans(
        (SourceCut(60, 90, 0), SourceCut(120, 150, 30)),
        rate=30,
        source_window=(Fraction(5, 2), Fraction(9, 2)),
    )
    assert spans == (
        retarget.RetainedSpan(Fraction(5, 2), Fraction(1, 2), Fraction(0), Fraction(1, 2)),
        retarget.RetainedSpan(Fraction(4), Fraction(1), Fraction(1, 2), Fraction(1, 2)),
    )
    captions = retarget.project_cues((retarget.Cue(Fraction(0), Fraction(2), "a\nb"),), spans)
    assert [(c.start, c.end, c.lines) for c in captions] == [
        (Fraction(0), Fraction(1, 2), ("a", "b")),
        (Fraction(1, 2), Fraction(1), ("a", "b")),
    ]


@pytest.mark.parametrize(
    "rate,window", [(0, None), (30, (Fraction(-1), Fraction(1))), (30, (Fraction(8), Fraction(9)))]
)
def test_bad_or_empty_selection(rate: int, window: tuple[Fraction, Fraction] | None) -> None:
    with pytest.raises(ReplayError):
        retarget.selected_spans((SourceCut(60, 90, 0),), rate=rate, source_window=window)


def test_no_recognized_stages_is_not_success() -> None:
    with pytest.raises(ReplayError, match="no recognized"):
        retarget.project_cues(
            (retarget.Cue(Fraction(8), Fraction(9), "later"),),
            (retarget.RetainedSpan(Fraction(2), Fraction(0), Fraction(0), Fraction(1)),),
        )


def test_output_window_projects_through_source_gaps_once() -> None:
    spans = retarget.selected_spans(
        (SourceCut(60, 90, 0), SourceCut(120, 180, 30)),
        rate=30,
        source_window=None,
        output_window=(Fraction(1, 2), Fraction(5, 2)),
    )
    assert spans == (
        retarget.RetainedSpan(Fraction(5, 2), Fraction(1, 2), Fraction(0), Fraction(1, 2)),
        retarget.RetainedSpan(Fraction(4), Fraction(1), Fraction(1, 2), Fraction(3, 2)),
    )
    with pytest.raises(ReplayError, match="one valid"):
        retarget.selected_spans(
            (SourceCut(60, 90, 0),),
            rate=30,
            source_window=(Fraction(2), Fraction(3)),
            output_window=(Fraction(0), Fraction(1)),
        )


def test_native_title_keys_follow_source_clock_across_cut(
    faces: FacePositionSeries, template: CaptionTemplate, calibration: TrackingCalibration
) -> None:
    spans = retarget.selected_spans(
        (SourceCut(60, 90, 0), SourceCut(120, 150, 30)), rate=30, source_window=None
    )
    xml = retarget.build(
        (retarget.Cue(Fraction(0), Fraction(2), "a\nb\nc"),),
        spans,
        source=Path("/source.mp4"),
        media=MediaInfo(Fraction(5), True, True, audio_channels=2),
        template=template,
        faces=faces,
        calibration=calibration,
        rate=30,
        name="actual stages",
    )
    root = ET.fromstring(xml)
    assert [t.get("offset") for t in root.findall(".//title")] == ["2s", "4s"]
    assert [t.findtext("text") for t in root.findall(".//title")] == ["a\nb\nc", "a\nb\nc"]
    assert [
        k.get("time")
        for k in root.findall(".//title/adjust-transform/param/keyframeAnimation/keyframe")
    ] == ["3600s", "3601s", "3600s"]
    transform = root.find(".//title/adjust-transform")
    assert transform is not None
    assert transform.get("position") is None
    first_clip = root.find(".//spine/asset-clip")
    assert first_clip is not None
    assert [child.tag for child in first_clip] == ["title", "audio-channel-source"]


@pytest.mark.parametrize("change", ["canvas", "anchor", "coverage"])
def test_retarget_refuses_unobserved_geometry(
    change: str,
    faces: FacePositionSeries,
    template: CaptionTemplate,
    calibration: TrackingCalibration,
) -> None:
    if change == "canvas":
        faces = replace(faces, width=1920)
    if change == "anchor":
        template = replace(template, anchor=CaptionAnchor.TOP)
    start = Fraction(1) if change == "coverage" else Fraction(2)
    with pytest.raises(ReplayError):
        retarget.build(
            (retarget.Cue(Fraction(0), Fraction(1), "a"),),
            (retarget.RetainedSpan(start, Fraction(0), Fraction(0), Fraction(1)),),
            source=Path("/source.mp4"),
            media=MediaInfo(Fraction(5), True, False),
            template=template,
            faces=faces,
            calibration=calibration,
            rate=30,
            name="bad",
        )


@pytest.fixture
def caption_directory(tmp_path: Path) -> Path:
    directory = tmp_path / "captions"
    directory.mkdir()
    (directory / "segments.json").write_text(
        json.dumps(
            {
                "rate": 30,
                "segments": [
                    {"segment": 0, "offsetFrame": 0, "frames": 30},
                    {"segment": 1, "offsetFrame": 30, "frames": 30},
                ],
            }
        ),
        encoding="utf-8",
    )
    candidate = {
        "version": 1,
        "candidateOnly": True,
        "canvas": [1080, 1920],
        "coordinateSystem": "top-left canvas; calibrated text origin",
        "inputs": {"/source.mp4": "source-sha"},
        "policy": {"frame_rate": 30, "hold_seconds": 0.08},
        "candidates": [{"cue": {"start": 0, "end": 1, "text": "speech"}, "x": 540, "y": 700}],
    }
    (directory / "combined-0000.json").write_text(json.dumps(candidate), encoding="utf-8")
    (directory / "combined-0001.json").write_text(json.dumps(candidate), encoding="utf-8")
    return directory


def test_read_segments_adds_offsets_without_using_old_y(caption_directory: Path) -> None:
    spans = (retarget.RetainedSpan(Fraction(2), Fraction(0), Fraction(0), Fraction(2)),)
    assert retarget.read_cues(
        caption_directory, spans, rate=30, source=Path("/source.mp4"), source_sha256="source-sha"
    ) == (
        retarget.Cue(Fraction(0), Fraction(1), "speech"),
        retarget.Cue(Fraction(1), Fraction(2), "speech"),
    )


@pytest.mark.parametrize("change", ["rate", "source", "frames", "empty"])
def test_bad_caption_binding(caption_directory: Path, change: str) -> None:
    index = caption_directory / "segments.json"
    text = index.read_text(encoding="utf-8")
    if change == "rate":
        index.write_text(text.replace('"rate": 30', '"rate": 25'), encoding="utf-8")
    if change == "frames":
        index.write_text(text.replace('"frames": 30', '"frames": 0'), encoding="utf-8")
    source = Path("/wrong.mp4") if change == "source" else Path("/source.mp4")
    start = Fraction(5) if change == "empty" else Fraction(0)
    with pytest.raises(ReplayError):
        retarget.read_cues(
            caption_directory,
            (retarget.RetainedSpan(Fraction(2), start, Fraction(0), Fraction(1)),),
            rate=30,
            source=source,
            source_sha256="source-sha",
        )


def test_cli_emits_new_candidate_only(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    caption_directory: Path,
    faces: FacePositionSeries,
    template: CaptionTemplate,
) -> None:
    monkeypatch.setattr(retarget, "ROOT", tmp_path)
    source = tmp_path / "source.mp4"
    source.write_bytes(b"synthetic")
    job = TrackingJob(
        tmp_path / "config.json",
        "config-sha",
        Path("/source.mp4"),
        tmp_path / "native",
        "source-sha",
        "native-sha",
        VideoFrameClock((2, 3, 4), Fraction(1), 1),
        1080,
        1920,
        1,
    )
    monkeypatch.setattr(retarget, "read_verified_face_positions", lambda config, evidence: faces)
    monkeypatch.setattr(retarget, "load_job", lambda config: job)
    monkeypatch.setattr(retarget, "probe_media", lambda path: MediaInfo(Fraction(5), True, False))
    monkeypatch.setattr(retarget, "caption_template", lambda path, anchor: template)
    cuts = tmp_path / "cuts.json"
    cuts.write_text(
        json.dumps(
            {"cuts": [{"source_start_frame": 60, "source_end_frame": 90, "output_start_frame": 0}]}
        ),
        encoding="utf-8",
    )
    output = tmp_path / "actual.fcpxml"
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "retarget",
            str(caption_directory),
            str(cuts),
            "unused.moti",
            str(output),
            "--face-config",
            "unused.json",
            "--face-evidence",
            "unused",
            "--reference-time",
            "2",
            "--head-top",
            "500",
            "--roi",
            "300",
            "700",
            "100",
            "100",
            "--source-window",
            "2",
            "3",
            "--name",
            "actual",
        ],
    )
    retarget.main()
    assert output.is_file()
    assert len(file_digest(output)) == 64
    with pytest.raises(FileExistsError):
        retarget.main()


def test_consecutive_native_windows_preserve_frames(faces: FacePositionSeries) -> None:
    later = FacePositionSeries(
        (TrackingPosition(Fraction(5), 300, 700), TrackingPosition(Fraction(6), 300, 700)),
        Fraction(7),
        1080,
        1920,
    )
    merged = retarget.merge_faces((faces, later))
    assert [point.time for point in merged.positions] == [
        Fraction(2),
        Fraction(3),
        Fraction(4),
        Fraction(5),
        Fraction(6),
    ]
    assert merged.coverage_end == 7


@pytest.mark.parametrize("mode", ["gap", "overlap", "canvas", "empty"])
def test_face_window_gaps_are_not_interpolated(faces: FacePositionSeries, mode: str) -> None:
    later = faces
    if mode == "gap":
        later = replace(
            faces,
            positions=(
                TrackingPosition(Fraction(6), 300, 700),
                TrackingPosition(Fraction(7), 300, 700),
            ),
        )
    if mode == "canvas":
        later = replace(faces, width=1920)
    with pytest.raises(ReplayError):
        retarget.merge_faces(() if mode == "empty" else (faces, later))


def test_bounded_face_references() -> None:
    with pytest.raises(ReplayError, match="bounded"):
        retarget.read_face_windows([])


def test_explicit_next_seed_uses_new_detection(faces: FacePositionSeries) -> None:
    later = FacePositionSeries(
        (TrackingPosition(Fraction(4), 333, 777), TrackingPosition(Fraction(5), 333, 778)),
        Fraction(6),
        1080,
        1920,
    )
    with pytest.raises(ReplayError):
        retarget.merge_faces((faces, later))
    merged = retarget.merge_faces((faces, later), seed_overlap="next")
    assert [point.time for point in merged.positions] == [2, 3, 4, 5]
    assert merged.positions[2].y == 777
    assert merged.coverage_end == 6
    with pytest.raises(ReplayError):
        retarget.merge_faces((faces, faces), seed_overlap="next")


def test_processed_voice_remains_output_clock_without_fixed_blur(
    faces: FacePositionSeries, template: CaptionTemplate, calibration: TrackingCalibration
) -> None:
    spans = retarget.selected_spans(
        (SourceCut(60, 90, 0), SourceCut(120, 150, 30)),
        rate=30,
        source_window=(Fraction(21, 10), Fraction(49, 10)),
    )
    xml = retarget.build(
        (retarget.Cue(Fraction(0), Fraction(2), "words"),),
        spans,
        source=Path("/source.mp4"),
        media=MediaInfo(Fraction(5), True, True, audio_channels=2),
        template=template,
        faces=faces,
        calibration=calibration,
        rate=30,
        name="processed",
        voice=(Path("/voice.wav"), MediaInfo(Fraction(2), False, True, audio_channels=2)),
    )
    root = ET.fromstring(xml)
    assert root.find(".//filter-video") is None
    assert "Gaussian" not in xml.decode("utf-8")
    voices = [c for c in root.findall(".//asset-clip") if c.get("name") == "Voice"]
    assert [v.get("duration") for v in voices] == ["9/10s", "9/10s"]
    assert [v.get("start") for v in voices] == ["1/10s", "1s"]


@pytest.fixture
def reflow_document() -> dict[str, object]:
    return {
        "version": 1,
        "candidateOnly": True,
        "inputs": {"/source.mp4": "source-sha"},
        "coverage": {"start": 0, "end": 2},
        "captions": [{"start": 0, "end": 2, "text": "ありがとう\n次の語\n終わり"}],
        "layout": {
            "version": 1,
            "scope": "horizontal-phrase-layout-only",
            "pointSize": 83,
            "paragraphWidth": 972,
            "maximumLines": 3,
            "widthMeasurement": "native-point-size-checked-glyph-bounds-including-border-rim",
            "boundaries": "pinned-vibrato-ipadic-phrase-boundaries",
            "recognitionClocksChanged": False,
            "originsRequireRetargeting": True,
        },
    }


def test_reflow_replaces_geometry_without_changing_recognition_clock(
    tmp_path: Path, reflow_document: dict[str, object]
) -> None:
    evidence = tmp_path / "measure.json"
    evidence.write_text("{}", encoding="utf-8")
    reflow_document["inputs"] = {"/source.mp4": "source-sha", str(evidence): file_digest(evidence)}
    path = tmp_path / "reflow.json"
    path.write_text(json.dumps(reflow_document), encoding="utf-8")
    spans = (retarget.RetainedSpan(Fraction(7), Fraction(1), Fraction(0), Fraction(1)),)
    cues = retarget.read_reflow(path, spans, source=Path("/source.mp4"), source_sha256="source-sha")
    assert cues == (retarget.Cue(Fraction(0), Fraction(2), "ありがとう\n次の語\n終わり"),)
    evidence.write_text("changed", encoding="utf-8")
    with pytest.raises(ReplayError, match="hash changed"):
        retarget.read_reflow(path, spans, source=Path("/source.mp4"), source_sha256="source-sha")


def test_empty_recognition_span_preserves_full_output_clock(
    tmp_path: Path, reflow_document: dict[str, object]
) -> None:
    reflow_document["coverage"] = {"start": 0, "end": 1 / 3}
    reflow_document["captions"] = []
    reflow_document["emptyRecognitionSpans"] = [
        {
            "start": 0,
            "end": 1 / 3,
            "recognizedTokens": 0,
            "measuredGlyphs": 0,
            "classification": "no-recognized-caption-tokens",
        }
    ]
    path = tmp_path / "empty.json"
    path.write_text(json.dumps(reflow_document), encoding="utf-8")
    spans = (retarget.RetainedSpan(Fraction(7), Fraction(0), Fraction(0), Fraction(1, 3)),)
    assert (
        retarget.read_reflow(path, spans, source=Path("/source.mp4"), source_sha256="source-sha")
        == ()
    )


@pytest.mark.parametrize(
    "value",
    [
        [],
        [
            {
                "start": 0,
                "end": 1,
                "recognizedTokens": 1,
                "measuredGlyphs": 0,
                "classification": "no-recognized-caption-tokens",
            }
        ],
        [
            {
                "start": 0,
                "end": 1,
                "recognizedTokens": 0,
                "measuredGlyphs": 0,
                "classification": "no-recognized-caption-tokens",
            }
        ],
    ],
)
def test_empty_reflow_rejects_missing_invalid_or_partial_support(
    tmp_path: Path, reflow_document: dict[str, object], value: object
) -> None:
    reflow_document["captions"] = []
    reflow_document["emptyRecognitionSpans"] = value
    path = tmp_path / "empty.json"
    path.write_text(json.dumps(reflow_document), encoding="utf-8")
    spans = (retarget.RetainedSpan(Fraction(0), Fraction(0), Fraction(0), Fraction(2)),)
    with pytest.raises(ReplayError):
        retarget.read_reflow(path, spans, source=Path("/source.mp4"), source_sha256="source-sha")


def test_empty_span_cannot_overlap_actual_caption(
    tmp_path: Path, reflow_document: dict[str, object]
) -> None:
    reflow_document["emptyRecognitionSpans"] = [
        {
            "start": 0,
            "end": 1,
            "recognizedTokens": 0,
            "measuredGlyphs": 0,
            "classification": "no-recognized-caption-tokens",
        }
    ]
    path = tmp_path / "overlap.json"
    path.write_text(json.dumps(reflow_document), encoding="utf-8")
    spans = (retarget.RetainedSpan(Fraction(0), Fraction(0), Fraction(0), Fraction(2)),)
    with pytest.raises(ReplayError, match="overlaps"):
        retarget.read_reflow(path, spans, source=Path("/source.mp4"), source_sha256="source-sha")


def test_explicit_producer_evidence_preserves_strict_media_and_measurements(
    tmp_path: Path, reflow_document: dict[str, object]
) -> None:
    code = Path(retarget.__file__).with_name("caption_native_paging.py").resolve()
    saved = tmp_path / "producer.bytes"
    saved.write_bytes(b"historical producer")
    measurement = tmp_path / "measurement.json"
    measurement.write_bytes(b"measured geometry")
    reflow_document["inputs"] = {
        "/source.mp4": "source-sha",
        str(code): file_digest(saved),
        str(measurement): file_digest(measurement),
    }
    artifact = tmp_path / "reflow.json"
    artifact.write_text(json.dumps(reflow_document), encoding="utf-8")
    evidence = tmp_path / "producers.json"
    reflow_producer_evidence.prepare(
        artifact=artifact, source=code, evidence=saved, output=evidence
    )
    spans = (retarget.RetainedSpan(Fraction(0), Fraction(0), Fraction(0), Fraction(2)),)
    with pytest.raises(ReplayError, match="hash changed"):
        retarget.read_reflow(
            artifact, spans, source=Path("/source.mp4"), source_sha256="source-sha"
        )
    assert (
        len(
            retarget.read_reflow(
                artifact,
                spans,
                source=Path("/source.mp4"),
                source_sha256="source-sha",
                producer_evidence=evidence,
            )
        )
        == 1
    )
    with pytest.raises(ReplayError, match="source differ"):
        retarget.read_reflow(
            artifact,
            spans,
            source=Path("/source.mp4"),
            source_sha256="replacement-source",
            producer_evidence=evidence,
        )
    measurement.write_bytes(b"replacement measurement")
    with pytest.raises(ReplayError, match="hash changed"):
        retarget.read_reflow(
            artifact,
            spans,
            source=Path("/source.mp4"),
            source_sha256="source-sha",
            producer_evidence=evidence,
        )


@pytest.mark.parametrize(
    "key,value",
    [
        ("layout", {}),
        ("inputs", {}),
        ("coverage", {"start": 0, "end": 0.5}),
        ("coverage", {"start": False, "end": 2}),
        ("captions", []),
        ("captions", [{"start": 0, "end": 2, "text": "a\nb\nc\nd"}]),
        ("captions", [{"start": 0, "end": 2, "text": "a"}, {"start": 1, "end": 2, "text": "b"}]),
    ],
)
def test_reflow_rejects_stale_partial_or_invalid_contract(
    tmp_path: Path, reflow_document: dict[str, object], key: str, value: object
) -> None:
    reflow_document[key] = value
    path = tmp_path / "reflow.json"
    path.write_text(json.dumps(reflow_document), encoding="utf-8")
    with pytest.raises(ReplayError):
        retarget.read_reflow(
            path,
            (retarget.RetainedSpan(Fraction(7), Fraction(1), Fraction(0), Fraction(1)),),
            source=Path("/source.mp4"),
            source_sha256="source-sha",
        )


@pytest.mark.parametrize("missing", ["--voice", "--reflow"])
def test_full_cli_cannot_silently_reuse_old_layout_or_omit_stages(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, missing: str
) -> None:
    monkeypatch.setattr(retarget, "ROOT", tmp_path)
    arguments = [
        "retarget",
        "captions",
        "cuts",
        "title",
        str(tmp_path / "full.fcpxml"),
        "--face-config",
        "face",
        "--face-evidence",
        "evidence",
        "--reference-time",
        "1",
        "--head-top",
        "500",
        "--roi",
        "1",
        "1",
        "1",
        "1",
        "--name",
        "full",
    ]
    for key, values in (
        ("--voice", ["voice.wav"]),
        ("--reflow", ["reflow.json"]),
    ):
        if key != missing:
            arguments.extend([key, *values])
    monkeypatch.setattr(sys, "argv", arguments)
    with pytest.raises(ReplayError, match="Full-length candidates require measured reflow"):
        retarget.main()


def test_cli_rejects_removed_source_caption_mask(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "retarget",
            "captions",
            "cuts",
            "title",
            "output.fcpxml",
            "--reference-time",
            "1",
            "--head-top",
            "500",
            "--roi",
            "1",
            "1",
            "1",
            "1",
            "--name",
            "removed mask",
            "--source-caption-mask",
            "1",
            "1",
            "100",
            "100",
            "37.5",
        ],
    )
    with pytest.raises(SystemExit) as error:
        retarget.main()
    assert error.value.code == 2
    assert "unrecognized arguments: --source-caption-mask" in capsys.readouterr().err


def test_full_paging_cli_check_generation_and_unknown_receipt(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    reflow_document: dict[str, object],
    template: CaptionTemplate,
    capsys: pytest.CaptureFixture[str],
) -> None:
    monkeypatch.setattr(retarget, "ROOT", tmp_path)
    config, evidence = tmp_path / "config.json", tmp_path / "evidence"
    config.write_text("{}", encoding="utf-8")
    evidence.mkdir()
    for name in ("response.json", "provenance.json"):
        (evidence / name).write_text("{}", encoding="utf-8")
    index = tmp_path / "windows.json"
    index.write_text(
        json.dumps({"windows": [{"index": 0, "config": str(config), "evidence": str(evidence)}]}),
        encoding="utf-8",
    )
    job = TrackingJob(
        config,
        "config-sha",
        Path("/source.mp4"),
        tmp_path / "native",
        "source-sha",
        "native-sha",
        VideoFrameClock((2, 3, 4), Fraction(1), 1),
        1080,
        1920,
        1,
    )
    raw = FaceObservationSeries(
        (
            FaceObservation(Fraction(2), TrackingPosition(Fraction(2), 300, 700)),
            FaceObservation(Fraction(5, 2), None, "missing"),
            FaceObservation(Fraction(3), TrackingPosition(Fraction(3), 300, 710)),
            FaceObservation(Fraction(4), TrackingPosition(Fraction(4), 300, 720)),
        ),
        Fraction(5),
        1080,
        1920,
    )
    monkeypatch.setattr(retarget, "load_job", lambda path: job)
    monkeypatch.setattr(
        retarget,
        "read_verified_face_observations",
        lambda config, evidence, anchor: replace(raw, anchor=anchor),
    )
    monkeypatch.setattr(retarget, "caption_template", lambda path, anchor: template)
    monkeypatch.setattr(
        retarget,
        "probe_media",
        lambda path: (
            MediaInfo(Fraction(2), False, True, audio_channels=2)
            if path.name == "voice.wav"
            else MediaInfo(Fraction(5), True, True, audio_channels=2)
        ),
    )
    reflow_document["captions"] = paging.assemble(
        (
            {"captions": [{"start": 0, "end": 31 / 30, "text": "first\nphrase\nthree"}]},
            {"captions": [{"start": 1, "end": 2, "text": "second phrase"}]},
        ),
        rate=30,
    )
    reflow = tmp_path / "reflow.json"
    reflow.write_text(json.dumps(reflow_document), encoding="utf-8")
    cuts = tmp_path / "cuts.json"
    cuts.write_text(
        json.dumps(
            {
                "cuts": [
                    {"source_start_frame": 60, "source_end_frame": 90, "output_start_frame": 0},
                    {"source_start_frame": 120, "source_end_frame": 150, "output_start_frame": 30},
                ]
            }
        ),
        encoding="utf-8",
    )
    output = tmp_path / "synthetic-full.fcpxml"
    arguments = [
        "retarget",
        "unused-captions",
        str(cuts),
        "title",
        str(output),
        "--face-index",
        str(index),
        "--face-anchor",
        "bottom",
        "--missing-face-policy",
        "hold",
        "--reference-time",
        "2",
        "--head-top",
        "500",
        "--roi",
        "300",
        "700",
        "100",
        "100",
        "--name",
        "synthetic full",
        "--voice",
        "voice.wav",
        "--reflow",
        str(reflow),
    ]
    monkeypatch.setattr(sys, "argv", [*arguments, "--check"])
    retarget.main()
    assert not output.exists()
    assert not output.with_suffix(".tracking.json").exists()
    monkeypatch.setattr(sys, "argv", arguments)
    retarget.main()
    root = ET.parse(output).getroot()
    titles = root.findall(".//asset-clip/title")
    assert [title.findtext("text") for title in titles] == ["first\nphrase\nthree", "second phrase"]
    assert [title.get("offset") for title in titles] == ["2s", "4s"]
    assert [title.get("duration") for title in titles] == ["1s", "1s"]
    assert all(title.get("start") == "3600s" for title in titles)
    receipt = json.loads(output.with_suffix(".tracking.json").read_text(encoding="utf-8"))
    assert receipt["derivedFrames"] == 1
    assert receipt["observedFrames"] == 3
    assert receipt["formalTrackingAccepted"] is False
    reports = [json.loads(line) for line in capsys.readouterr().out.splitlines()]
    assert [report["sourceCaptionRepairRequired"] for report in reports] == [True, True]
    assert receipt["unknownSourceRanges"][0]["outputRanges"] == [{"start": "1/2", "end": "1"}]
    assert receipt["fcpxmlSHA256"] == file_digest(output)


def test_derived_calibration_reference_is_not_an_observation(
    faces: FacePositionSeries,
    template: CaptionTemplate,
    calibration: TrackingCalibration,
) -> None:
    faces = replace(
        faces,
        derived_ranges=(
            MissingFaceRange(
                Fraction(2), Fraction(3), 1, Fraction(4), "leading-observation-backfill", "missing"
            ),
        ),
    )
    with pytest.raises(ReplayError, match="reference must be an observed"):
        retarget.build(
            (retarget.Cue(Fraction(0), Fraction(1), "text"),),
            (retarget.RetainedSpan(Fraction(2), Fraction(0), Fraction(0), Fraction(1)),),
            source=Path("/source.mp4"),
            media=MediaInfo(Fraction(5), True, False),
            template=template,
            faces=faces,
            calibration=calibration,
            rate=30,
            name="invalid reference",
        )
    with pytest.raises(ReplayError, match="Merge raw"):
        retarget.merge_faces((faces,))


@pytest.mark.parametrize(
    "rows",
    [
        [],
        [{"index": 2, "config": "x", "evidence": "y"}],
        [{"index": 0, "config": False, "evidence": "y"}],
    ],
)
def test_face_index_rejects_ambiguous_window_order(tmp_path: Path, rows: object) -> None:
    path = tmp_path / "index.json"
    path.write_text(json.dumps({"windows": rows}), encoding="utf-8")
    with pytest.raises(ReplayError):
        retarget.face_references(path)
