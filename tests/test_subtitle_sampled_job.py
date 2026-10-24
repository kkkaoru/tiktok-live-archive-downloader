"""The one-chunk job wires real collaborators around synthetic media and a fake native tool."""

import json
import subprocess
import sys
from fractions import Fraction
from pathlib import Path

import numpy as np
import pytest
import torch
from numpy.typing import NDArray

from replay import subtitle_detector_mps as mps
from replay import subtitle_sampled_job as job
from replay import subtitle_sampled_run as runner
from replay import subtitle_yolo_regions as yolo_regions
from replay import yolo_boxes
from replay.local_asr import fingerprint
from replay.recordings import record
from replay.subtitle_repair_audit import Residual
from replay.subtitle_stream import SubtitleSpan
from replay.subtitle_video import VideoRepairJob

WIDTH = 320
HEIGHT = 240
FRAMES = 40
START = 5 * 90000  # the chunk starts five seconds into the source
PTS = [START + offset for offset in range(0, FRAMES * 3000, 3000)]
LINE = {"x": 60, "y": 90, "width": 200, "height": 30}

FAKE_NATIVE = """#!{python}
import json, sys
from pathlib import Path

request = json.loads(Path(sys.argv[3]).read_text())
repaired = request["path"].endswith("candidate.mp4")
confidence = {low} if repaired else 0.9
if repaired and any(s["value"] / s["timescale"] > 2 for s in request["samples"]):
    print("Invalid request: Frame sample lies outside source duration", file=sys.stderr)
    sys.exit(2)
frames = [
    {{
        "requestedTimeSeconds": s["value"] / s["timescale"],
        "actualTimeSeconds": s["value"] / s["timescale"],
        "width": request["width"],
        "height": request["height"],
        "lines": [{{"text": "こんにちは世界です", "confidence": confidence, "region": {line}}}],
    }}
    for s in request["samples"]
]
directory = Path(request["checkpointDirectory"])
for number in range(0, len(frames), 8):
    (directory / f"block-{{number // 8}}.json").write_text(json.dumps(frames[number:number + 8]))
"""


def _bar(batch: NDArray[np.float32]) -> NDArray[np.float32]:
    count, _, height, width = batch.shape
    maps = np.zeros((count, 1, height, width), dtype=np.float32)
    maps[:, :, int(height * 0.40) : int(height * 0.50), width // 8 : width // 2] = 0.95
    return maps


def _configuration(tmp_path: Path, *, residual: float) -> dict[str, object]:
    source = tmp_path / "source.mp4"
    subprocess.run(
        ["ffmpeg", "-nostdin", "-v", "error", "-f", "lavfi", "-i",
         f"color=c=black:s={WIDTH}x{HEIGHT}:r=30:d=8", "-pix_fmt", "yuv420p", "-y", str(source)],
        check=True,
        timeout=60,
    )  # fmt: skip
    profile = tmp_path / "profile"
    profile.mkdir()
    observations = [
        {
            "requestedTimeSeconds": 0.0,
            "actualTimeSeconds": 0.0,
            "width": WIDTH,
            "height": HEIGHT,
            "lines": [{"text": text, "confidence": 0.9, "region": LINE}],
        }
        # Captions change their wording; one text with one-letter variants would be a fixed label.
        for text in (
            "こんにちは世界です",
            "今日はいい天気ですね",
            "配信を始めます",
            "ありがとうございます",
            "それでは歌います",
            "コメント読みます",
            "また明日会いましょう",
            "楽しかったです",
            "少し休憩します",
            "おかえりなさい",
            "準備ができました",
            "最後まで見てね",
        )
    ]
    (profile / "0000.result.json").write_text(json.dumps(observations[:8]), encoding="utf-8")
    (profile / "0001.result.json").write_text(json.dumps(observations[8:]), encoding="utf-8")
    native = tmp_path / "native"
    native.write_text(
        FAKE_NATIVE.format(python=sys.executable, low=residual, line=json.dumps(LINE)),
        encoding="utf-8",
    )
    native.chmod(0o700)
    return {
        "source": str(source),
        "native": str(native),
        "clock": {"pts": PTS, "timeBase": "1/90000", "lastDuration": 3000},
        "width": WIDTH,
        "height": HEIGHT,
        "checkoutRoot": str(tmp_path),
        "output": str(tmp_path / "chunk"),
        "profileDirectory": str(profile),
    }


def _fake_render(video_job: VideoRepairJob, _detector: job.Detects) -> None:
    video_job.output.write_bytes(b"video")


def test_sampled_attempt_runs_plan_render_and_audit(tmp_path: Path) -> None:
    config = _configuration(tmp_path, residual=0.3)
    tools, pins = job.real_collaborators(config, network=_bar, render=_fake_render)
    receipt = runner.run(tmp_path / "chunk", tools, stride=13, pins=pins)
    assert receipt["accepted"] == "attempt-sampled"
    assert receipt["escalated"] is False
    admission = json.loads(
        (tmp_path / "chunk" / "attempt-sampled" / "admission.json").read_text(encoding="utf-8")
    )
    assert admission["sampling"]["sampledFrames"][0] == 0
    assert pins["native"] == fingerprint(tmp_path / "native")


def test_the_video_may_live_in_a_separate_media_root(tmp_path: Path) -> None:
    config = _configuration(tmp_path, residual=0.3)
    config["mediaOutput"] = str(tmp_path / "media")
    config["output"] = str(tmp_path / "state")
    tools, _ = job.real_collaborators(config, network=_bar, render=_fake_render)
    receipt = runner.run(tmp_path / "state", tools, stride=13)
    assert receipt["accepted"] == "attempt-sampled"
    assert (tmp_path / "media" / "attempt-sampled" / "candidate.mp4").read_bytes() == b"video"
    assert (tmp_path / "state" / "attempt-sampled" / "audit" / "child-000").is_dir()


def test_a_confident_residual_escalates_and_fails_without_acceptance(tmp_path: Path) -> None:
    config = _configuration(tmp_path, residual=1.0)
    tools, _ = job.real_collaborators(config, network=_bar, render=_fake_render)
    with pytest.raises(ValueError, match="nothing accepted"):
        runner.run(tmp_path / "chunk", tools, stride=13)
    receipt = json.loads((tmp_path / "chunk" / "chunk-receipt.json").read_text(encoding="utf-8"))
    assert receipt["escalated"] is True
    assert receipt["accepted"] is None


def test_missing_profile_observations_are_rejected(tmp_path: Path) -> None:
    config = _configuration(tmp_path, residual=0.3)
    empty = tmp_path / "empty"
    empty.mkdir()
    config["profileDirectory"] = str(empty)
    with pytest.raises(ValueError, match="profile"):
        job.real_collaborators(config, network=_bar, render=_fake_render)


def test_invalid_clock_is_rejected(tmp_path: Path) -> None:
    config = _configuration(tmp_path, residual=0.3)
    config["clock"] = {"pts": ["x"], "timeBase": "1/90000", "lastDuration": 3000}
    with pytest.raises(ValueError, match="clock"):
        job.real_collaborators(config, network=_bar, render=_fake_render)
    config["clock"] = {"pts": PTS, "timeBase": 1, "lastDuration": 3000}
    with pytest.raises(ValueError, match="clock"):
        job.real_collaborators(config, network=_bar, render=_fake_render)


def test_main_loads_the_pinned_network_and_runs(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    config = _configuration(tmp_path, residual=0.3)
    config.update(output=str(tmp_path / "out"), stride=13)

    class Head(torch.nn.Module):
        def forward(self, x: torch.Tensor) -> torch.Tensor:
            return x[:, :1] * 0.0

    model = tmp_path / "head.pt2"
    torch.export.save(torch.export.export(Head(), (torch.zeros(1, 3, 4, 6),)), str(model))
    config.update(detectorProgram=str(model), detectorProgramSHA256=fingerprint(model))
    path = tmp_path / "config.json"
    path.write_text(json.dumps(config), encoding="utf-8")
    monkeypatch.setattr(
        job, "render_repaired_video_fast", lambda video_job, **_: video_job.output.write_bytes(b"v")
    )
    monkeypatch.setattr(mps, "lazy_network", lambda *_args, **_kw: _bar)
    assert job.main([str(path)]) == 0
    assert json.loads(capsys.readouterr().out) == {
        "accepted": "attempt-sampled",
        "escalated": False,
    }


def test_main_rejects_an_oversized_configuration(tmp_path: Path) -> None:
    path = tmp_path / "config.json"
    path.write_bytes(b"x" * (job.MAX_CONFIG_BYTES + 1))
    with pytest.raises(ValueError, match="byte budget"):
        job.main([str(path)])


def test_fraction_clock_round_trip() -> None:
    clock = job._clock({"pts": PTS[:3], "timeBase": "1/90000", "lastDuration": 3000})
    assert clock.time_base == Fraction(1, 90000)


def _detecting_render(video_job: VideoRepairJob, detector: job.Detects) -> None:
    for span in video_job.spans:
        for _ in range(span.start_frame, span.end_frame):
            detector.detect([np.zeros((span.height, span.width, 3), dtype=np.uint8)])
    video_job.output.write_bytes(b"video")


def _partial_render(video_job: VideoRepairJob, detector: job.Detects) -> None:
    span = video_job.spans[0]
    detector.detect([np.zeros((span.height, span.width, 3), dtype=np.uint8)])
    video_job.output.write_bytes(b"video")


def test_colour_key_wraps_the_detector_and_is_pinned(tmp_path: Path) -> None:
    config = _configuration(tmp_path, residual=0.3)
    config["colourKey"] = True
    tools, pins = job.real_collaborators(config, network=_bar, render=_detecting_render)
    receipt = runner.run(tmp_path / "chunk", tools, stride=13, pins=pins)
    pinned = receipt["pins"]
    assert isinstance(pinned, dict)
    assert (receipt["accepted"], pinned["colourKey"]) == ("attempt-sampled", "pink-key-v1")


def test_colour_key_requires_every_planned_frame_to_be_consumed(tmp_path: Path) -> None:
    config = _configuration(tmp_path, residual=0.3)
    config["colourKey"] = True
    tools, _ = job.real_collaborators(config, network=_bar, render=_partial_render)
    with pytest.raises(ValueError, match="not all consumed"):
        runner.run(tmp_path / "chunk", tools, stride=13)


def test_key_frames_skip_unknown_frames_and_read_span_origins() -> None:
    admitted: dict[str, object] = {
        "frames": [
            {"classification": "unknown"},
            {"classification": "overlay-subtitle", "lineBoxes": [[1, 2, 30, 40]]},
        ]
    }
    spans = (SubtitleSpan(1, 2, 7, 9, 64, 32),)
    frames = job.key_frames(admitted, spans)
    assert [(f.origin, f.boxes) for f in frames] == [((7, 9), ((1, 2, 30, 40),))]


def test_key_frames_key_an_unread_detected_region() -> None:
    region = {"x": 3, "y": 4, "width": 10, "height": 20}
    admitted: dict[str, object] = {
        "frames": [{"classification": "overlay-subtitle", "lineBoxes": [], "region": region}]
    }
    frames = job.key_frames(admitted, (SubtitleSpan(0, 1, 0, 0, 64, 32),))
    assert [f.boxes for f in frames] == [((3, 4, 13, 24),)]


def test_key_frames_reject_missing_boxes_spans_and_entries() -> None:
    span = (SubtitleSpan(0, 1, 0, 0, 64, 32),)
    with pytest.raises(ValueError, match="line boxes"):
        job.key_frames({"frames": [{"classification": "overlay-subtitle"}]}, span)
    with pytest.raises(ValueError, match="no repair span"):
        job.key_frames({"frames": [{"classification": "overlay-subtitle", "lineBoxes": []}]}, ())
    with pytest.raises(ValueError, match="per-frame entries"):
        job.key_frames({"frames": 3}, span)


def test_colour_gate_collaborator_counts_unchanged_caption_colour(tmp_path: Path) -> None:
    config = _configuration(tmp_path, residual=0.3)
    config["colourGate"] = True
    tools, _ = job.real_collaborators(config, network=_bar, render=_fake_render)
    assert tools.colour is not None


def test_entry_boxes_reject_non_list_boxes() -> None:
    assert job._entry_boxes({"lineBoxes": [[1, 2, 3, 4]]}) == [(1, 2, 3, 4)]
    assert job._entry_boxes({}) == []
    with pytest.raises(ValueError, match="must be a list"):
        job._entry_boxes({"lineBoxes": 4})


def test_caption_coloured_collaborator_classifies_residuals_by_source_pink(tmp_path: Path) -> None:
    config = _configuration(tmp_path, residual=0.3)
    config["colourGate"] = True
    tools, _ = job.real_collaborators(config, network=_bar, render=_fake_render)
    assert tools.caption_coloured is not None
    residual = Residual(0, "UI", 1.0, (10.0, 10.0, 30.0, 10.0))
    assert tools.caption_coloured([residual]) == [False]


def test_caption_coloured_collaborator_needs_every_residual_box(tmp_path: Path) -> None:
    config = _configuration(tmp_path, residual=0.3)
    config["colourGate"] = True
    tools, _ = job.real_collaborators(config, network=_bar, render=_fake_render)
    assert tools.caption_coloured is not None
    with pytest.raises(ValueError, match="lack their boxes"):
        tools.caption_coloured([Residual(0, "UI", 1.0)])


def test_without_the_colour_gate_there_is_no_interface_classifier(tmp_path: Path) -> None:
    tools, _ = job.real_collaborators(
        _configuration(tmp_path, residual=0.3), network=_bar, render=_fake_render
    )
    assert tools.caption_coloured is None


def _caption_detector(
    images: list[NDArray[np.uint8]],
) -> list[list[tuple[float, float, float, float]]]:
    """A caption box over the line's rows in the 640-wide detector frame."""
    return [[(120.0, 180.0, 520.0, 240.0)] for _ in images]


def test_a_yolo_caption_detector_supplies_the_search_regions(tmp_path: Path) -> None:
    config = _configuration(tmp_path, residual=0.3)

    def unused_network(batch: NDArray[np.float32]) -> NDArray[np.float32]:
        raise AssertionError("the PP-OCR network must not plan regions when YOLO is given")

    tools, _ = job.real_collaborators(
        config, network=unused_network, render=_fake_render, caption_detector=_caption_detector
    )
    receipt = runner.run(tmp_path / "chunk", tools, stride=13, pins={})
    assert receipt["accepted"] == "attempt-sampled"


def test_main_pins_the_yolo_region_detector(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config = _configuration(tmp_path, residual=0.3)
    config.update(output=str(tmp_path / "out"), stride=13, detectorProgram="d.pt2",
                  detectorProgramSHA256="0" * 64)  # fmt: skip
    config["regionDetector"] = {
        "kind": "yolo", "weights": "w.pt", "sha256": "1" * 64, "configDirectory": "c",
        "work": "w", "imgsz": 640,
    }  # fmt: skip
    path = tmp_path / "config.json"
    path.write_text(json.dumps(config), encoding="utf-8")
    monkeypatch.setattr(mps, "lazy_network", lambda *_args, **_kw: _bar)
    monkeypatch.setattr(yolo_regions, "load_detector", lambda settings: _caption_detector)
    monkeypatch.setattr(
        job, "render_repaired_video_fast", lambda video_job, **_: video_job.output.write_bytes(b"v")
    )
    assert job.main([str(path)]) == 0
    receipt = json.loads((tmp_path / "out" / "chunk-receipt.json").read_text(encoding="utf-8"))
    assert receipt["pins"]["regionDetector"] == "yolo:" + "1" * 64 + ":640"


def test_yolo_masks_replace_the_learned_detector_in_the_real_render(tmp_path: Path) -> None:
    config = _configuration(tmp_path, residual=0.3)

    def unused_network(batch: NDArray[np.float32]) -> NDArray[np.float32]:
        raise AssertionError("the PP-OCR network must not run when YOLO masks are used")

    tools, _ = job.real_collaborators(
        config, network=unused_network, caption_detector=_caption_detector, caption_masks=True
    )
    receipt = runner.run(tmp_path / "chunk", tools, stride=13, pins={})
    assert receipt["accepted"] == "attempt-sampled"
    assert (tmp_path / "chunk" / "attempt-sampled" / "candidate.mp4").stat().st_size > 0


def test_yolo_masks_need_the_yolo_detector(tmp_path: Path) -> None:
    config = _configuration(tmp_path, residual=0.3)
    with pytest.raises(ValueError, match="need the YOLO caption detector"):
        job.real_collaborators(config, network=_bar, caption_masks=True)


def test_box_frames_follow_the_active_frames_and_their_span_origins() -> None:
    admitted: dict[str, object] = {
        "frames": [{"classification": "unknown"}, {"classification": "overlay-subtitle"}]
    }
    spans = [SubtitleSpan(1, 2, 7, 9, 16, 16)]
    boxes = [[(0.0, 0.0, 1.0, 1.0)], [(5.0, 6.0, 7.0, 8.0)]]
    assert job.box_frames(admitted, spans, boxes) == [
        yolo_regions.PlannedBoxes((7, 9), ((5.0, 6.0, 7.0, 8.0),))
    ]
    with pytest.raises(ValueError, match="do not match the admitted frames"):
        job.box_frames(admitted, spans, boxes[:1])


def test_a_yolo_admission_plans_without_ocr_and_records_its_kind(tmp_path: Path) -> None:
    config = _configuration(tmp_path, residual=0.3)
    tools, _ = job.real_collaborators(
        config,
        network=_bar,
        caption_detector=_caption_detector,
        caption_masks=True,
        yolo_admission=True,
    )
    receipt = runner.run(tmp_path / "chunk", tools, stride=13, pins={})
    attempts = receipt["attempts"]
    assert isinstance(attempts, list)
    attempt = attempts[0]
    assert receipt["accepted"] == "attempt-sampled"
    assert attempt["sampling"]["kind"] == "yolo-caption-boxes"
    assert list(attempt["stageSeconds"]) == ["yoloBoxes"]
    assert not (tmp_path / "chunk" / "attempt-sampled" / "ocr").exists()


def test_the_band_audit_reads_tiled_bands_and_is_pinned(tmp_path: Path) -> None:
    config = _configuration(tmp_path, residual=0.3)
    config["auditTile"] = 2
    tools, pins = job.real_collaborators(
        config, network=_bar, caption_detector=_caption_detector, caption_masks=True,
        yolo_admission=True,
    )  # fmt: skip
    assert pins["auditBand"] == "tile2"
    # The fake OCR reads confident text in any image that is not the repaired candidate itself,
    # so the band audit flags residual text and both attempts fail.
    with pytest.raises(ValueError, match="Audit failed after escalation"):
        runner.run(tmp_path / "chunk", tools, stride=13, pins=pins)
    band = json.loads(
        (tmp_path / "chunk" / "attempt-sampled" / "audit" / "band.json").read_text(encoding="utf-8")
    )
    assert band["tile"] == 2 and band["indices"][0] == 0


def test_an_invalid_audit_tile_is_refused(tmp_path: Path) -> None:
    config = _configuration(tmp_path, residual=0.3)
    config["auditTile"] = 9
    with pytest.raises(ValueError, match="auditTile"):
        job.real_collaborators(config, network=_bar)


def test_residual_boxes_join_caption_frames_only() -> None:
    region = {"x": 0, "y": 0, "width": 10, "height": 10}
    admitted: dict[str, object] = {
        "frames": [
            {"classification": "overlay-subtitle", "lineBoxes": [[1, 2, 3, 4]], "region": region},
            {"classification": "unknown"},
        ],
        "masks": [{"startSeconds": 0, "endSeconds": 1, "region": dict(region)}],
    }
    merged = job.with_residual(
        admitted,
        [[(0.0, 0.0, 9.0, 9.0)], []],
        {0: [(5, 6, 7, 8)], 1: [(5, 6, 7, 8)], 4: [(1, 1, 2, 2)]},
    )
    assert merged == [[(0.0, 0.0, 9.0, 9.0), (5.0, 6.0, 7.0, 8.0)], []]
    frames = admitted["frames"]
    assert isinstance(frames, list)
    assert frames[0]["lineBoxes"] == [(1, 2, 3, 4), [5, 6, 7, 8]]
    assert "lineBoxes" not in frames[1]
    assert frames[0]["region"] == {"x": 0, "y": 0, "width": 10, "height": 10}
    masks = admitted["masks"]
    assert isinstance(masks, list)
    assert masks[0]["region"] == {"x": 0, "y": 0, "width": 10, "height": 10}
    above: dict[str, object] = {
        "frames": [{"classification": "overlay-subtitle", "lineBoxes": [], "region": dict(region)}],
        "masks": [{"region": dict(region)}],
    }
    job.with_residual(above, [[]], {0: [(-5, -8, 4, 2), (6, 20, 30, 24)]})
    above_frames, above_masks = above["frames"], above["masks"]
    assert isinstance(above_frames, list)
    assert isinstance(above_masks, list)
    assert above_frames[0]["region"] == {"x": -5, "y": -18, "width": 35, "height": 46}
    assert above_masks[0]["region"] == {"x": -5, "y": -18, "width": 35, "height": 46}
    with pytest.raises(ValueError, match="per-frame entries"):
        job.with_residual({"frames": 3}, [], {0: [(1, 1, 2, 2)]})


def test_stored_yolo_boxes_supply_the_regions_without_decoding(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config = _configuration(tmp_path, residual=0.3)
    pts = record(config["clock"])["pts"]
    assert isinstance(pts, list)
    scale = int(str(config["width"])) / 640

    def no_decode(*_args: object, **_kwargs: object) -> object:
        raise AssertionError("stored boxes must not decode the chunk again")

    monkeypatch.setattr(yolo_regions, "chunk_frames", no_decode)
    box = (120.0 * scale, 180.0 * scale, 520.0 * scale, 240.0 * scale)
    tools, _ = job.real_collaborators(
        config, network=_bar, render=_fake_render, stored_boxes=[[box]] * len(pts)
    )
    receipt = runner.run(tmp_path / "chunk", tools, stride=13, pins={})
    assert receipt["accepted"] == "attempt-sampled"


def test_main_uses_stored_boxes_and_skips_the_detector_model(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    config = _configuration(tmp_path, residual=0.3)
    config.update(output=str(tmp_path / "out"), stride=13, detectorProgram="d.pt2",
                  detectorProgramSHA256="0" * 64, yoloBoxes=str(tmp_path / "boxes"),
                  sourceSHA256="5" * 64)  # fmt: skip
    config["regionDetector"] = {
        "kind": "yolo", "weights": "w.pt", "sha256": "1" * 64, "configDirectory": "c",
        "work": "w", "imgsz": 640,
    }  # fmt: skip
    path = tmp_path / "config.json"
    path.write_text(json.dumps(config), encoding="utf-8")
    asked: list[tuple[Path, str, str, int]] = []

    def load(directory: Path, pts: object, **keywords: object) -> list[list[object]]:
        asked.append((directory, str(keywords["source_sha256"]), str(keywords["weights_sha256"]),
                      int(str(keywords["imgsz"]))))  # fmt: skip
        assert isinstance(pts, tuple)
        return [[] for _ in pts]

    def no_model(_settings: object) -> object:
        raise AssertionError("the detector model must not load when boxes are stored")

    monkeypatch.setattr(mps, "lazy_network", lambda *_args, **_kw: _bar)
    monkeypatch.setattr(yolo_boxes, "load", load)
    monkeypatch.setattr(yolo_regions, "load_detector", no_model)
    monkeypatch.setattr(
        job, "render_repaired_video_fast", lambda video_job, **_: video_job.output.write_bytes(b"v")
    )
    assert job.main([str(path)]) == 0
    assert asked == [(tmp_path / "boxes", "5" * 64, "1" * 64, 640)]
    assert capsys.readouterr().out.splitlines()[0] == "YOLO_BOXES stored"
