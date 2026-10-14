import hashlib
import json
import subprocess
from collections.abc import Sequence
from dataclasses import replace
from fractions import Fraction
from pathlib import Path

import pytest

from replay import delivery_render as render
from replay import edit_completion, native_edit_parts, source_caption_binding, workflow
from replay.core import ReplayError
from replay.fcp_part_assembly import _reference


def write(path: Path, value: object) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value), encoding="utf-8")
    return path


def configured(tmp_path: Path) -> Path:
    """A synthetic recording with one caption segment of two Motion parts."""
    captions, work, storage = tmp_path / "captions", tmp_path / "work", tmp_path / "storage"
    write(
        captions / "segments.json",
        {"segments": [{"segment": 0, "offsetFrame": 0, "frames": 120, "jobs": [0]}]},
    )
    write(captions / "combined-0000.json", {"candidates": []})
    plans = work / "motion" / "segment-0000" / "plans"
    write(
        plans / "manifest.json",
        {
            "parts": [
                {"path": str(plans / "part-0000.json"), "offsetFrame": 3, "frames": 60},
                {"path": str(plans / "part-0001.json"), "offsetFrame": 63, "frames": 50},
            ]
        },
    )
    cuts = write(
        tmp_path / "cuts.json",
        {
            "cuts": [
                {"source_start_frame": 30, "source_end_frame": 90, "output_start_frame": 0},
                {"source_start_frame": 110, "source_end_frame": 170, "output_start_frame": 60},
            ]
        },
    )
    chunks = write(
        tmp_path / "chunks.json",
        {
            "ranges": [
                {
                    "media": str(tmp_path / "a.mp4"),
                    "sourceFirstPTS": 0,
                    "sourceEndPTS": 450000,
                    "timeBase": "1/90000",
                },
                {
                    "media": str(tmp_path / "b.mp4"),
                    "sourceFirstPTS": 450000,
                    "sourceEndPTS": 900000,
                    "timeBase": "1/90000",
                },
            ]
        },
    )
    font = tmp_path / "synthetic-font.bin"
    font.write_bytes(b"synthetic font; not native resolution evidence")
    voice, clock = tmp_path / "voice.wav", tmp_path / "clock.json"
    voice.write_bytes(b"RIFF")
    clock.write_text("{}", encoding="utf-8")
    (tmp_path / "final.setting").write_text(
        '<setting><encoder name="H264iPod"><file-extension>m4v</file-extension>'
        "</encoder><filter-set/></setting>",
        encoding="utf-8",
    )
    return write(
        tmp_path / "render-config.json",
        {
            "version": 1,
            "recording": "replay-1",
            "captions": str(captions),
            "work": str(work),
            "storage": str(storage),
            "rate": 30,
            "width": 1080,
            "height": 1920,
            "cuts": str(cuts),
            "sourceChunks": str(chunks),
            "motion": {
                "template": str(tmp_path / "template.moti"),
                "expectedText": "Title",
                "finish": ["--top", "0", "0", "0"],
                "resolvedAssets": {
                    "fonts": [
                        {"path": str(font), "sha256": hashlib.sha256(font.read_bytes()).hexdigest()}
                    ],
                    "resources": [],
                },
            },
            "alphaPreset": str(tmp_path / "alpha.setting"),
            "encodePreset": str(tmp_path / "hevc.setting"),
            "native": str(tmp_path / "native"),
            "encoding": {"codec": "hevc"},
            "audio": {"source": str(voice), "clock": str(clock)},
            "source": str(tmp_path / "2024-01-01_09-00-00_JST_replay-1.mp4"),
            "deliveryRoot": str(tmp_path / "deliveries" / "replay-1"),
            "finalPreset": str(tmp_path / "final.setting"),
            "calibration": str(tmp_path / "calibration.json"),
            "viewerRoot": str(tmp_path / "edits" / "replay-1"),
        },
    )


def stages(path: Path) -> list[dict[str, object]]:
    value = json.loads(path.read_text(encoding="utf-8"))["stages"]
    assert isinstance(value, list)
    return value


@pytest.fixture
def cleanup_config(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    path = configured(tmp_path)
    clean = tmp_path / "clean.mov"
    clean.write_bytes(b"synthetic cleaned video")
    (tmp_path / "native").write_bytes(b"synthetic native executable")
    chunks = write(
        tmp_path / "clean-index.json",
        {
            "ranges": [
                {
                    "media": str(clean),
                    "sourceFirstPTS": 0,
                    "sourceEndPTS": 900000,
                    "timeBase": "1/90000",
                }
            ]
        },
    )
    cleanup = write(tmp_path / "cleanup.json", {"version": 1, "cleanIndex": _reference(chunks)})
    raw = json.loads(path.read_text(encoding="utf-8"))
    raw["sourceChunks"] = str(chunks)
    raw["sourceCleanup"] = _reference(cleanup)
    write(path, raw)
    original = Path(raw["source"])
    monkeypatch.setattr(
        source_caption_binding, "clean_source", lambda value: (original, clean, Fraction(10))
    )
    return path


def test_load_config_rejects_unknown_fields(tmp_path: Path) -> None:
    path = write(tmp_path / "config.json", {"version": 1})
    with pytest.raises(ReplayError, match="unknown, missing"):
        render.load_config(path)


@pytest.mark.parametrize("operation", ["compose", "render", "publish"])
def test_production_refuses_legacy_raw_configuration(tmp_path: Path, operation: str) -> None:
    config = render.load_config(configured(tmp_path))
    with pytest.raises(ReplayError, match="requires pinned sourceCleanup"):
        if operation == "compose":
            render.compose(config, workers=1)
        elif operation == "render":
            render.render(config, workers=1)
        else:
            render.publish(config)
    assert not config.composition.exists()
    assert not config.delivery_root.exists()


def test_load_config_rejects_stale_cleanup_pin(cleanup_config: Path) -> None:
    write(cleanup_config.parent / "cleanup.json", {"changed": True})
    with pytest.raises(ReplayError, match="Assembly input hash changed"):
        render.load_config(cleanup_config)


def test_cleanup_is_repinned_after_loading(cleanup_config: Path) -> None:
    config = render.load_config(cleanup_config)
    write(cleanup_config.parent / "cleanup.json", {"changed": True})
    with pytest.raises(ReplayError, match="Assembly input hash changed"):
        render.compose(config, workers=1)
    assert not config.composition.exists()


@pytest.mark.parametrize("field", ["source", "sourceChunks"])
def test_cleanup_rejects_unrelated_source_or_index(cleanup_config: Path, field: str) -> None:
    raw = json.loads(cleanup_config.read_text(encoding="utf-8"))
    raw[field] = str(cleanup_config.parent / "unrelated.mov")
    write(cleanup_config, raw)
    config = render.load_config(cleanup_config)
    with pytest.raises(ReplayError, match="does not select its verified cleanup"):
        render.compose(config, workers=1)
    assert not config.composition.exists()


def test_cleanup_path_cannot_change_after_loading(cleanup_config: Path) -> None:
    config = render.load_config(cleanup_config)
    other = write(
        cleanup_config.parent / "other-cleanup.json",
        json.loads((cleanup_config.parent / "cleanup.json").read_text(encoding="utf-8")),
    )
    raw = json.loads(cleanup_config.read_text(encoding="utf-8"))
    raw["sourceCleanup"] = _reference(other)
    write(cleanup_config, raw)
    with pytest.raises(ReplayError, match="changed after loading"):
        render.render(config, workers=1)


def test_cleanup_rejects_mutated_loaded_configuration(cleanup_config: Path) -> None:
    config = replace(render.load_config(cleanup_config), source=Path("/unrelated.mov"))
    with pytest.raises(ReplayError, match="differ from verified cleanup assembly"):
        render.compose(config, workers=1)


def test_reused_native_request_cannot_reintroduce_raw_source(cleanup_config: Path) -> None:
    config = render.load_config(cleanup_config)
    part = config.composition / "parts-0000" / "part-0000" / "request.json"
    config.source.write_bytes(b"synthetic original")
    write(part, {"recipe": {"video": {}, "clips": [{"sourcePath": str(config.source)}]}})
    write(
        config.composition / "parts-0000" / "manifest.json",
        {
            "parts": [{"path": str(part), "frames": 120}],
        },
    )
    with pytest.raises(ReplayError, match="unverified source"):
        render.render(config, workers=1)


def test_workflow_bindings_follow_worker_partition(tmp_path: Path) -> None:
    plans = render._workflows(
        tmp_path,
        "render",
        [{"name": "first"}, {"name": "second"}],
        2,
        bindings={"first": Path("/first-binding.json"), "second": Path("/second-binding.json")},
    )
    assert [
        json.loads(path.read_text(encoding="utf-8"))["sourceCaptionProduction"] for path in plans
    ] == [
        {"bindings": {"first": "/first-binding.json"}},
        {"bindings": {"second": "/second-binding.json"}},
    ]


def test_load_config_requires_absolute_paths(tmp_path: Path) -> None:
    path = configured(tmp_path)
    data = json.loads(path.read_text(encoding="utf-8"))
    data["cuts"] = "relative.json"
    write(path, data)
    with pytest.raises(ReplayError, match="absolute"):
        render.load_config(path)


def test_load_config_requires_finish_strings(tmp_path: Path) -> None:
    path = configured(tmp_path)
    data = json.loads(path.read_text(encoding="utf-8"))
    data["motion"]["finish"] = [1]
    write(path, data)
    with pytest.raises(ReplayError, match="finish arguments"):
        render.load_config(path)


def test_motion_export_uses_segment_frames(tmp_path: Path) -> None:
    config = render.load_config(configured(tmp_path))
    [workflow] = render.motion_export(config, workers=2)
    command = stages(workflow)[0]["command"]
    assert isinstance(command, list)
    assert command[command.index("--frames") + 1] == "120"
    assert command[-1] == "Title"


def test_motion_export_requires_combined_segment(tmp_path: Path) -> None:
    config = render.load_config(configured(tmp_path))
    (config.captions / "combined-0000.json").unlink()
    with pytest.raises(ReplayError, match="not been combined"):
        render.motion_export(config, workers=1)


def test_motion_parts_use_global_offsets(tmp_path: Path) -> None:
    config = render.load_config(configured(tmp_path))
    parts = render.motion_parts(config)
    assert [(part.segment, part.part, part.offset, part.frames) for part in parts] == [
        (0, 0, 3, 60),
        (0, 1, 63, 50),
    ]


def test_motion_prepare_pairs_stages_and_finish(tmp_path: Path) -> None:
    config = render.load_config(configured(tmp_path))
    [workflow] = render.motion_prepare(config, workers=1)
    assert [stage["name"] for stage in stages(workflow)] == [
        "stages-0000-0000",
        "finish-0000-0000",
        "stages-0000-0001",
        "finish-0000-0001",
    ]


def test_alpha_jobs_name_each_part(tmp_path: Path) -> None:
    config = render.load_config(configured(tmp_path))
    jobs = json.loads(render.alpha_jobs(config).read_text(encoding="utf-8"))["jobs"]
    assert [job["name"] for job in jobs] == ["alpha-0000-0000", "alpha-0000-0001"]
    assert jobs[1]["output"] == str(config.storage / "alpha" / "segment-0000-part-0001.mov")


def test_clips_split_a_cut_at_an_aligned_chunk_boundary(tmp_path: Path) -> None:
    chunks = render.read_chunks(configured(tmp_path).parent / "chunks.json")
    pieces = render.clips([render.CutRow(140, 160)], chunks, 30)
    assert pieces == [
        render.Clip(tmp_path / "a.mp4", Fraction(14, 3), 10),
        render.Clip(tmp_path / "b.mp4", Fraction(0), 10),
    ]


@pytest.mark.parametrize("boundary", [Fraction(5005, 1000), Fraction(4995, 1000)])
def test_clips_refuse_shift_or_overrun_at_vfr_seam(boundary: Fraction) -> None:
    chunks = [
        render.Chunk(Path("/a.mp4"), Fraction(0), boundary),
        render.Chunk(Path("/b.mp4"), boundary, Fraction(10)),
    ]
    with pytest.raises(ReplayError, match="verified source/candidate frame-clock mapping"):
        render.clips([render.CutRow(140, 160)], chunks, 30)


def test_clips_refuse_small_native_tail_overrun() -> None:
    chunks = [render.Chunk(Path("/candidate.mp4"), Fraction(45012, 125), Fraction(60016, 125))]
    # 195 output frames would extend 2/375 seconds beyond the source extent.
    assert Fraction(14404, 30) - chunks[0].end == Fraction(2, 375)
    with pytest.raises(ReplayError, match="native chunk boundary"):
        render.clips([render.CutRow(14209, 14404)], chunks, 30)


def test_clips_preserve_interior_vfr_selection_without_rounding() -> None:
    chunks = [render.Chunk(Path("/candidate.mp4"), Fraction(45012, 125), Fraction(60016, 125))]
    assert render.clips([render.CutRow(14209, 14403)], chunks, 30) == [
        render.Clip(Path("/candidate.mp4"), Fraction(425765, 3750), 194)
    ]


def test_clips_reject_cut_outside_chunks(tmp_path: Path) -> None:
    chunks = render.read_chunks(configured(tmp_path).parent / "chunks.json")
    with pytest.raises(ReplayError, match="outside the source chunk"):
        render.clips([render.CutRow(400, 410)], chunks, 30)


def test_read_chunks_requires_contiguity(tmp_path: Path) -> None:
    path = write(
        tmp_path / "chunks.json",
        {
            "ranges": [
                {"media": "/a", "sourceFirstPTS": 0, "sourceEndPTS": 10, "timeBase": "1/90000"},
                {"media": "/b", "sourceFirstPTS": 11, "sourceEndPTS": 20, "timeBase": "1/90000"},
            ]
        },
    )
    with pytest.raises(ReplayError, match="contiguous"):
        render.read_chunks(path)


def test_compose_writes_requests_with_clipped_overlays(cleanup_config: Path) -> None:
    config = render.load_config(cleanup_config)
    [workflow] = render.compose(config, workers=1)
    assert [stage["name"] for stage in stages(workflow)] == ["partition-0000"]
    request = json.loads(
        (config.work / "compose" / "request-0000.json").read_text(encoding="utf-8")
    )
    assert [clip["selection"]["startSeconds"] for clip in request["recipe"]["clips"]] == [
        1.0,
        3.6666666666666665,
    ]
    assert [layer["offsetSeconds"] for layer in request["recipe"]["additionalVideo"]] == [0.1, 2.1]


def test_compose_partitions_route_media_separately_from_requests(cleanup_config: Path) -> None:
    config = render.load_config(cleanup_config)
    render.compose(config, workers=1)
    request = config.work / "compose" / "request-0000.json"
    original = request.read_bytes()
    local = config.work / "compose" / "parts-0000"
    native_edit_parts.export(request, local, sha256=render._digest(request))
    part = json.loads((local / "part-0000" / "request.json").read_text(encoding="utf-8"))
    destination = Path(part["outputDirectory"])
    assert destination.parent.parent == config.storage / "compose" / "request-0000"
    assert destination.name == "part-0000"
    assert request.read_bytes() == original
    assert not config.storage.exists()


def test_compose_refuses_changed_index_before_writing_requests(cleanup_config: Path) -> None:
    config = render.load_config(cleanup_config)
    write(
        config.chunks,
        {
            "ranges": [
                {"media": "/a", "sourceFirstPTS": 0, "sourceEndPTS": 5005, "timeBase": "1/1000"},
                {
                    "media": "/b",
                    "sourceFirstPTS": 5005,
                    "sourceEndPTS": 10000,
                    "timeBase": "1/1000",
                },
            ]
        },
    )
    with pytest.raises(ReplayError, match="Assembly input hash changed"):
        render.compose(config, workers=1)
    assert not (config.work / "compose").exists()


def test_render_and_join_follow_partition_manifests(tmp_path: Path, cleanup_config: Path) -> None:
    config = render.load_config(cleanup_config)
    part = config.work / "compose" / "parts-0000" / "part-0000" / "request.json"
    movie = tmp_path / "movie.mp4"
    write(
        part,
        {
            "outputDirectory": str(tmp_path),
            "outputName": "movie.mp4",
            "recipe": {
                "video": {},
                "clips": [{"sourcePath": str(tmp_path / "clean.mov")}],
            },
        },
    )
    write(
        config.work / "compose" / "parts-0000" / "manifest.json",
        {"parts": [{"path": str(part), "frames": 120}]},
    )
    [render_workflow] = render.render(config, workers=3)
    assert stages(render_workflow)[0]["outputs"] == [
        str(part.with_name("render.invocation.json")),
        str(part.with_name("render.response.json")),
        str(part.with_name("render.binding.json")),
    ]
    inputs = stages(render_workflow)[0]["inputs"]
    assert isinstance(inputs, list)
    assert str(tmp_path / "clean.mov") in inputs
    cleanup_binding = part.with_name("source-caption.binding.json")
    assert str(cleanup_binding) in inputs
    generated = json.loads(render_workflow.read_text(encoding="utf-8"))
    assert generated["sourceCaptionProduction"] == {
        "bindings": {"render-000000": str(cleanup_binding)}
    }
    assert len(workflow.parse_stages(generated)) == 1
    command = stages(render_workflow)[0]["command"]
    assert isinstance(command, list)
    source_caption_binding.validate_binding(
        cleanup_binding,
        consumer_inputs=tuple(Path(str(value)) for value in inputs),
        consumer_command=tuple(str(value) for value in command),
    )
    movie.write_bytes(b"movie")
    write(part.with_name("render.response.json"), {"outputPath": str(movie), "frameCount": 120})
    [join] = render.join(config)
    join_inputs = stages(join)[0]["inputs"]
    assert isinstance(join_inputs, list)
    assert join_inputs[:2] == [str(config.work / "join-plan.json"), str(movie)]
    assert json.loads((config.work / "join-plan.json").read_text(encoding="utf-8")) == [
        {"durationFrames": 120, "sourcePath": str(movie)}
    ]
    assert stages(join)[0]["name"] == "join"


def test_join_rejects_frame_mismatch(tmp_path: Path) -> None:
    config = render.load_config(configured(tmp_path))
    part = config.work / "compose" / "parts-0000" / "part-0000" / "request.json"
    write(part, {})
    write(
        config.work / "compose" / "parts-0000" / "manifest.json",
        {"parts": [{"path": str(part), "frames": 120}]},
    )
    write(part.with_name("render.response.json"), {"outputPath": "/movie.mp4", "frameCount": 1})
    with pytest.raises(ReplayError, match="frame count"):
        render.join(config)


def test_render_requires_partitions(cleanup_config: Path) -> None:
    config = render.load_config(cleanup_config)
    with pytest.raises(ReplayError, match="No partitioned"):
        render.render(config, workers=1)


def test_encode_and_audio_jobs(tmp_path: Path) -> None:
    config = render.load_config(configured(tmp_path))
    encode = json.loads(render.encode_job(config).read_text(encoding="utf-8"))["jobs"][0]
    assert encode["output"] == str(config.storage / "encoded-video.mov")
    [audio] = render.audio(config)
    assert stages(audio)[0]["name"] == "select-audio"
    selection = json.loads((config.work / "audio-selection.json").read_text(encoding="utf-8"))
    assert selection["intervals"][1] == {
        "start": {"value": 110, "timescale": 30},
        "end": {"value": 170, "timescale": 30},
    }


def test_segments_must_exist(tmp_path: Path) -> None:
    config = render.load_config(configured(tmp_path))
    write(config.captions / "segments.json", {"segments": []})
    with pytest.raises(ReplayError, match="empty"):
        render.segments(config)


def test_cli_runs_steps_and_reports_errors(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    path = configured(tmp_path)
    assert render.main(["alpha-jobs", str(path)]) == 0
    assert render.main(["alpha-jobs", str(path)]) == 1
    assert "DELIVERY_RENDER_ERROR" in capsys.readouterr().err


def test_final_mux_and_job(tmp_path: Path) -> None:
    config = render.load_config(configured(tmp_path))
    [mux] = render.final_mux(config)
    assert stages(mux)[0]["outputs"] == [str(config.storage / "av-master.mov")]
    job = json.loads(render.final_job(config).read_text(encoding="utf-8"))["jobs"][0]
    assert job == {
        "name": "final",
        "source": str(config.storage / "av-master.mov"),
        "preset": str(tmp_path / "final.setting"),
        "output": str(config.storage / "final" / "2024-01-01_09-00-00_JST_replay-1-review.m4v"),
    }


def test_delivery_stem_uses_dated_original(tmp_path: Path) -> None:
    config = render.load_config(configured(tmp_path))
    assert render.delivery_stem(config) == "2024-01-01_09-00-00_JST_replay-1-review"


def test_delivery_stem_rejects_undated_original(tmp_path: Path) -> None:
    path = configured(tmp_path)
    data = json.loads(path.read_text(encoding="utf-8"))
    data["source"] = str(tmp_path / "replay-1.mp4")
    write(path, data)
    with pytest.raises(ReplayError, match="JST start"):
        render.delivery_stem(render.load_config(path))


def test_caption_cues_and_srt_use_global_clock(tmp_path: Path) -> None:
    path = configured(tmp_path)
    data = json.loads(path.read_text(encoding="utf-8"))
    captions = Path(data["captions"])
    write(
        captions / "segments.json",
        {"segments": [{"segment": 0, "offsetFrame": 90, "frames": 120, "jobs": [0]}]},
    )
    write(
        captions / "combined-0000.json",
        {"candidates": [{"cue": {"text": "あ\nい", "start": 0.5, "end": 1.25}}]},
    )
    cues = render.caption_cues(render.load_config(path))
    assert cues == [(Fraction(7, 2), Fraction(17, 4), "あ\nい")]
    assert render.render_srt(cues) == "1\n00:00:03,500 --> 00:00:04,250\nあ\nい\n"


def test_caption_cues_reject_missing_clock(tmp_path: Path) -> None:
    path = configured(tmp_path)
    captions = Path(json.loads(path.read_text(encoding="utf-8"))["captions"])
    write(captions / "combined-0000.json", {"candidates": [{"cue": {"text": "a"}}]})
    with pytest.raises(ReplayError, match="text or clock"):
        render.caption_cues(render.load_config(path))


def test_caption_cues_reject_missing_end(tmp_path: Path) -> None:
    path = configured(tmp_path)
    captions = Path(json.loads(path.read_text(encoding="utf-8"))["captions"])
    write(captions / "combined-0000.json", {"candidates": [{"cue": {"text": "a", "start": 0}}]})
    with pytest.raises(ReplayError, match="its end"):
        render.caption_cues(render.load_config(path))


@pytest.mark.parametrize("stages_pass", [True, False])
def test_publish_copies_encode_and_writes_srt(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, stages_pass: bool, cleanup_config: Path
) -> None:
    monkeypatch.setattr(
        edit_completion,
        "assess_delivery",
        lambda _movie, *, root: {} if stages_pass else {"bgm_removal": "unverified"},
    )
    config = render.load_config(cleanup_config)
    write(
        config.captions / "combined-0000.json",
        {"candidates": [{"cue": {"text": "あ", "start": 0, "end": 1}}]},
    )
    (config.storage / "final").mkdir(parents=True)
    subprocess.run(
        [
            "ffmpeg",
            "-v",
            "error",
            "-f",
            "lavfi",
            "-i",
            "testsrc=size=320x240:rate=30:duration=4",
            "-f",
            "lavfi",
            "-i",
            "sine=frequency=440:duration=4",
            "-c:a",
            "aac",
            "-c:v",
            "hevc_videotoolbox",
            "-allow_sw",
            "1",
            "-tag:v",
            "hvc1",
            "-f",
            "mp4",
            str(render.configured_movie(config)),
        ],
        check=True,
    )
    if not stages_pass:
        with pytest.raises(ReplayError, match="Required edit stages remain unverified"):
            render.publish(config)
        candidate_dir = config.delivery_root / "revisions" / "native-v1" / "delivery"
        assert len(list(candidate_dir.glob("*.mp4"))) == 1
        assert not (config.viewer_root / "revisions").exists()
        monkeypatch.setattr(edit_completion, "assess_delivery", lambda _movie, *, root: {})
        links = render.register(config)
        assert len(links) == 2
        assert links[0].resolve() == next(candidate_dir.glob("*.mp4"))
        return
    movie, srt, probe, report, linked_movie, linked_srt = render.publish(config)
    verification = json.loads(report.read_text(encoding="utf-8"))
    assert verification["videoFrames"] == 120
    assert verification["expectedFrames"] == 120
    assert verification["framesMatchCutPlan"] is True
    assert verification["audioPackets"] > 0
    assert verification["fullDecodePassed"] is True
    assert linked_movie.resolve() == movie
    assert linked_srt.resolve() == srt
    assert movie.name == "2024-01-01_09-00-00_JST_replay-1-review.mp4"
    assert srt.read_text(encoding="utf-8") == "1\n00:00:00,000 --> 00:00:01,000\nあ\n"
    assert json.loads(probe.read_text(encoding="utf-8")) == {
        "format": {"filename": str(tmp_path / "2024-01-01_09-00-00_JST_replay-1.mp4")}
    }
    with pytest.raises(ReplayError, match="overwrite"):
        render.publish(config)


@pytest.mark.parametrize(
    "frames,audio_packets,decode_code,decode_error",
    [(119, 10, 0, ""), (120, 0, 0, ""), (120, 10, 1, ""), (120, 10, 0, "decode error")],
)
def test_failed_media_is_not_linked(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    frames: int,
    audio_packets: int,
    decode_code: int,
    decode_error: str,
    cleanup_config: Path,
) -> None:
    config = render.load_config(cleanup_config)
    write(
        config.captions / "combined-0000.json",
        {"candidates": [{"cue": {"text": "あ", "start": 0, "end": 1}}]},
    )
    encoded = render.configured_movie(config)
    encoded.parent.mkdir(parents=True)
    encoded.write_bytes(b"candidate")

    def run(arguments: Sequence[str], **_kwargs: object) -> subprocess.CompletedProcess[str]:
        if "-movflags" in arguments:
            Path(arguments[-1]).write_bytes(b"copied candidate")
        return subprocess.CompletedProcess(arguments, decode_code, "", decode_error)

    monkeypatch.setattr(subprocess, "run", run)
    monkeypatch.setattr(
        render, "_packets", lambda _movie, stream: frames if stream == "v:0" else audio_packets
    )
    with pytest.raises(ReplayError, match="verification failed; candidate retained"):
        render.publish(config)
    delivery = config.delivery_root / "revisions" / "native-v1" / "delivery"
    assert len(list(delivery.glob("*.verification.json"))) == 1
    assert len(list(delivery.glob("*.mp4"))) == 1
    assert not (config.viewer_root / "revisions" / "native-v1" / "delivery").exists()


def test_composition_override_is_local_to_compose_and_render(
    tmp_path: Path, cleanup_config: Path
) -> None:
    path = cleanup_config
    original = render.load_config(path)
    assert original.composition == original.work / "compose"
    override = tmp_path / "retry-compose"
    write(
        path,
        {**json.loads(path.read_text(encoding="utf-8")), "compositionDirectory": str(override)},
    )
    config = render.load_config(path)
    write(config.work / "motion" / "segment-0000" / "plans" / "manifest.json", {"parts": []})
    [workflow] = render.compose(config, workers=1)
    assert workflow.parent == override
    assert config.work == original.work
    assert config.storage == original.storage
    partition = stages(workflow)[0]
    command = partition["command"]
    assert isinstance(command, list)
    native_edit_parts.export(Path(str(command[3])), Path(str(command[4])), sha256=str(command[6]))
    [render_workflow] = render.render(config, workers=1)
    assert render_workflow.parent == override
    assert len(render.rendered_parts(config)) == 2
    write(override / "resume.json", {})
    with pytest.raises(ReplayError, match="repartitioned"):
        render.compose(config, workers=1)
    with pytest.raises(ReplayError, match="sealed prepared"):
        render.render(config, workers=1)
