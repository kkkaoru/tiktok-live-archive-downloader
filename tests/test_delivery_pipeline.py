import hashlib
import json
import subprocess
from collections.abc import Sequence
from fractions import Fraction
from pathlib import Path

import pytest

from replay import delivery_pipeline as pipeline
from replay import (
    delivery_render,
    disk_budget,
    mechanical_join,
    mechanical_lane,
    mechanical_plan_provenance,
    native_join,
    source_caption_binding,
)
from replay.core import ReplayError
from replay.fcp_part_assembly import _reference
from replay.stages import StageError
from replay.workflow import parse_stages


@pytest.fixture(autouse=True)
def ample_disk(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(disk_budget, "require_headroom", lambda paths, need=0: [])


def write(path: Path, value: object) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value), encoding="utf-8")
    return path


@pytest.fixture(autouse=True)
def synthetic_cleanup_producer(monkeypatch: pytest.MonkeyPatch) -> None:
    """Only media production is mocked; config selection and pins remain real."""
    monkeypatch.setattr(
        source_caption_binding,
        "clean_source",
        lambda config: (
            Path(str(config["testOriginal"])),
            Path(str(config["testClean"])),
            Fraction(2),
        ),
    )


def configured(tmp_path: Path) -> Path:
    work = tmp_path / "work"
    original = tmp_path / "2024-01-01_09-00-00_JST_replay-1.mp4"
    clean = tmp_path / "clean.mov"
    clean.write_bytes(b"synthetic clean media")
    chunks = write(tmp_path / "chunks.json", {})
    cleanup = write(
        tmp_path / "cleanup.json",
        {
            "cleanIndex": _reference(chunks),
            "testOriginal": str(original),
            "testClean": str(clean),
        },
    )
    return write(
        tmp_path / "render-config.json",
        {
            "version": 1,
            "recording": "replay-1",
            "captions": str(tmp_path / "captions"),
            "work": str(work),
            "storage": str(tmp_path / "storage"),
            "rate": 30,
            "width": 1080,
            "height": 1920,
            "cuts": str(tmp_path / "cuts.json"),
            "sourceChunks": str(chunks),
            "sourceCleanup": _reference(cleanup),
            "motion": {"template": str(tmp_path / "t.moti"), "expectedText": "Title", "finish": []},
            "alphaPreset": str(tmp_path / "a.setting"),
            "encodePreset": str(tmp_path / "e.setting"),
            "native": str(tmp_path / "native"),
            "encoding": {},
            "audio": {"source": str(tmp_path / "v.wav"), "clock": str(tmp_path / "c.json")},
            "source": str(original),
            "deliveryRoot": str(tmp_path / "deliveries" / "replay-1"),
            "finalPreset": str(tmp_path / "f.setting"),
            "calibration": str(tmp_path / "calibration.json"),
            "viewerRoot": str(tmp_path / "edits" / "replay-1"),
        },
    )


def test_commands_keep_state_beside_each_file(tmp_path: Path) -> None:
    step = pipeline.Step("render", "compose/render-*-*.json")
    [command] = pipeline.commands(step, [tmp_path / "render-0-0000.json"])
    assert command[1:] == [
        "-m",
        "replay.workflow",
        str(tmp_path / "render-0-0000.json"),
        "--state",
        str(tmp_path / "render-0-0000-state"),
    ]


def test_compressor_steps_use_the_batch_runner(tmp_path: Path) -> None:
    step = pipeline.Step("final-job", "final-job.json", compressor=True)
    [command] = pipeline.commands(step, [tmp_path / "final-job.json"])
    assert command[2] == "replay.compressor_batch"


def test_run_reuses_generated_files_and_stops_on_failure(tmp_path: Path) -> None:
    config = configured(tmp_path)
    write(tmp_path / "work" / "join-workflow.json", {"version": 1, "stages": []})
    launched: list[list[str]] = []

    def launch(batch: Sequence[Sequence[str]]) -> list[int]:
        launched.extend(list(command) for command in batch)
        return [1]

    with pytest.raises(ReplayError, match="Step join failed twice"):
        pipeline.run(config, launch=launch, start="join")
    assert launched[0][3] == str(tmp_path / "work" / "join-workflow.json")


def test_run_reports_generation_failure(tmp_path: Path) -> None:
    config = configured(tmp_path)
    with pytest.raises(ReplayError, match="Could not generate step audio"):
        pipeline.run(config, launch=lambda batch: [0 for _ in batch], start="audio")


def test_combine_generation_requires_measurements(tmp_path: Path) -> None:
    config = configured(tmp_path)
    write(tmp_path / "captions" / "segments.json", {"rate": 30, "segments": []})
    with pytest.raises(ReplayError, match="Invalid segment index"):
        pipeline.run(config, launch=lambda batch: [0 for _ in batch])


def test_cli_reports_errors(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    assert pipeline.main([str(tmp_path / "missing.json")]) == 1
    assert "DELIVERY_PIPELINE_ERROR" in capsys.readouterr().err


def test_default_launcher_waits_for_processes() -> None:
    assert pipeline._launch([["/usr/bin/true"], ["/usr/bin/false"]]) == [0, 1]


def test_full_run_executes_every_step_then_publishes(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config = configured(tmp_path)
    work = tmp_path / "work"
    write(tmp_path / "captions" / "combine-workflow.json", {})
    write(work / "motion" / "export-0-0000.json", {})
    write(work / "motion" / "prepare-0-0000.json", {})
    write(work / "alpha-jobs.json", {})
    write(work / "compose" / "partition-0-0000.json", {})
    write(work / "compose" / "render-0-0000.json", {})
    write(work / "compose" / "render-1-0000.json", {})
    write(work / "join-workflow.json", {})
    write(work / "audio-workflow.json", {})
    write(work / "final-mux-workflow.json", {})
    write(work / "final-job.json", {})
    monkeypatch.setattr(delivery_render, "publish", lambda _config: [tmp_path / "movie.mp4"])
    monkeypatch.setattr(pipeline, "_production_preflight", lambda _config, _files: None)
    sizes: list[int] = []

    def launch(batch: Sequence[Sequence[str]]) -> list[int]:
        sizes.append(len(batch))
        return [0 for _ in batch]

    assert pipeline.run(config, launch=launch) == [
        "combine",
        "motion-export",
        "motion-prepare",
        "alpha-jobs",
        "compose",
        "render",
        "join",
        "audio",
        "final-mux",
        "final-job",
        "publish",
    ]
    assert sizes == [1, 1, 1, 1, 1, 2, 1, 1, 1, 1]


def test_cli_success(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(pipeline, "run", lambda _path, start: ["publish"])
    assert pipeline.main([str(tmp_path / "config.json"), "--start", "publish"]) == 0


def test_override_never_discovers_old_bad_workflows(tmp_path: Path) -> None:
    path = configured(tmp_path)
    old = delivery_render.load_config(path)
    write(old.composition / "render-0-0000.json", {})
    write(old.composition / "partition-0-0000.json", {})
    retry = tmp_path / "retry"
    new_render = write(retry / "render-2-0000.json", {})
    new_partition = write(retry / "partition-1-0000.json", {})
    write(
        path, {**json.loads(path.read_text(encoding="utf-8")), "compositionDirectory": str(retry)}
    )
    config = delivery_render.load_config(path)
    assert pipeline._files(config, pipeline.STEPS[5]) == [new_render]
    assert pipeline._files(config, pipeline.STEPS[4]) == [new_partition]
    alpha = write(old.work / "alpha-jobs.json", {})
    assert pipeline._files(config, pipeline.STEPS[3]) == [alpha]


def test_resume_preflight_stops_changed_completed_media_before_join(tmp_path: Path) -> None:
    path = configured(tmp_path)
    retry = tmp_path / "retry"
    write(
        path, {**json.loads(path.read_text(encoding="utf-8")), "compositionDirectory": str(retry)}
    )
    movie = tmp_path / "completed.mp4"
    movie.write_bytes(b"changed movie")
    write(retry / "resume.json", {"version": 1, "inputs": {str(movie): "0" * 64}})
    with pytest.raises(ReplayError, match="Changed input pin"):
        pipeline.run(path, launch=lambda _batch: pytest.fail("Must not launch"), start="join")


@pytest.mark.parametrize("reuse", [False, True])
def test_production_generation_and_reuse_validate_clean_consumer(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, reuse: bool
) -> None:
    config = configured(tmp_path)
    clean = tmp_path / "clean.mov"
    clean.write_bytes(b"synthetic cleaned source")
    original = tmp_path / "raw.mov"
    index = write(tmp_path / "index.json", {})
    cleanup = write(tmp_path / "cleanup.json", {"cleanIndex": _reference(index)})
    write(
        config,
        {
            **json.loads(config.read_text()),
            "source": str(original),
            "sourceChunks": str(index),
            "sourceCleanup": _reference(cleanup),
        },
    )
    monkeypatch.setattr(
        source_caption_binding, "clean_source", lambda _: (original, clean, Fraction(2))
    )
    monkeypatch.setattr(mechanical_plan_provenance, "validate_stages", lambda _: None)
    request = write(
        tmp_path / "request.json", {"recipe": {"clips": [{"sourcePath": str(clean)}], "video": {}}}
    )
    binding = source_caption_binding.create_binding(
        source_caption_binding.require_cleanup(config),
        consumer=request,
        kind="native-request",
        output=tmp_path / "binding.json",
    )
    plan = tmp_path / "work/compose/render-0-0000.json"
    document = {
        "version": 1,
        "sourceCaptionProduction": {"bindings": {"render": str(binding)}},
        "stages": [
            {
                "name": "render",
                "command": ["native", "edit-media", str(request)],
                "inputs": list(map(str, (binding, clean, request))),
                "outputs": [str(tmp_path / "result.json")],
                "timeout_seconds": 1,
            }
        ],
    }
    if reuse:
        write(plan, document)
    monkeypatch.setattr(pipeline, "_generate", lambda *_: write(plan, document))
    monkeypatch.setattr(pipeline, "STEPS", (pipeline.Step("render", "compose/render-*-*.json"),))
    monkeypatch.setattr(delivery_render, "publish", lambda _: [])
    codes = iter([1, 0] if reuse else [0])
    assert pipeline.run(config, launch=lambda _: [next(codes)]) == ["render", "publish"]
    saved_binding = json.loads(binding.read_text())
    write(binding, {**saved_binding, "cleanIndex": _reference(cleanup)})
    with pytest.raises(ReplayError, match="different cleanup source"):
        pipeline.run(config, launch=lambda _: pytest.fail("Other recording must not launch"))
    write(binding, saved_binding)
    write(request, {"recipe": {"clips": [{"sourcePath": str(original)}], "video": {}}})
    with pytest.raises(ReplayError):
        pipeline.run(config, launch=lambda _: pytest.fail("Raw source must not launch"))


def test_existing_unbound_workflow_cannot_launch(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config = configured(tmp_path)
    plan = write(tmp_path / "work/compose/render-0-0000.json", {"version": 1, "stages": []})
    monkeypatch.setattr(
        source_caption_binding,
        "require_cleanup",
        lambda _: source_caption_binding.Cleanup(config, config, config, Fraction(1)),
    )
    with pytest.raises((ReplayError, StageError)):
        pipeline._production_preflight(config, [plan])
    write(plan, {"version": 1, "stages": [], "sourceCaptionProduction": {"bindings": {}}})
    with pytest.raises((ReplayError, StageError)):
        pipeline._production_preflight(config, [plan])


def test_failed_cleanup_cannot_launch_reused_workflow(tmp_path: Path) -> None:
    config = configured(tmp_path)
    data = json.loads(config.read_text())
    del data["sourceCleanup"]
    write(config, data)
    write(tmp_path / "work/compose/render-0-0000.json", {"version": 1, "stages": []})
    with pytest.raises(ReplayError):
        pipeline.run(config, start="render", launch=lambda _: pytest.fail("Missing cleanup"))


@pytest.mark.parametrize("start", pipeline.STEP_NAMES)
def test_every_start_requires_cleanup_before_launch_or_publication(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, start: str
) -> None:
    config = configured(tmp_path)
    data = json.loads(config.read_text())
    del data["sourceCleanup"]
    write(config, data)
    monkeypatch.setattr(delivery_render, "publish", lambda _: pytest.fail("Must not publish"))
    with pytest.raises(ReplayError):
        pipeline.run(config, start=start, launch=lambda _: pytest.fail("Must not launch"))


def test_oversize_join_uses_the_manifest_workflow(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(mechanical_plan_provenance, "join_needs_manifest", lambda *_args: True)
    path = configured(tmp_path)
    config = delivery_render.load_config(path)
    parts = [(tmp_path / f"request-{index}.json", 30) for index in range(254)]
    seen: list[int] = []

    def fake_generate(
        rendered: Sequence[tuple[Path, int]], configured_config: delivery_render.Config
    ) -> Path:
        seen.append(len(rendered))
        return tmp_path / "join-workflow.json"

    monkeypatch.setattr(delivery_render, "rendered_parts", lambda _config: parts)
    monkeypatch.setattr(mechanical_join, "generate_join", fake_generate)
    monkeypatch.setattr(delivery_render, "main", lambda _arguments: pytest.fail("classic join"))
    pipeline._generate(path, config, pipeline.STEPS[6])
    assert seen == [254]


def test_small_join_keeps_the_classic_audited_stage(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(mechanical_plan_provenance, "join_needs_manifest", lambda *_args: False)
    path = configured(tmp_path)
    config = delivery_render.load_config(path)
    parts = [(tmp_path / "request.json", 30)]
    calls: list[list[str]] = []

    def fake_main(arguments: list[str]) -> int:
        calls.append(arguments)
        return 0

    monkeypatch.setattr(delivery_render, "rendered_parts", lambda _config: parts)
    monkeypatch.setattr(delivery_render, "main", fake_main)
    monkeypatch.setattr(
        mechanical_join, "generate_join", lambda *_args: pytest.fail("manifest join")
    )
    pipeline._generate(path, config, pipeline.STEPS[6])
    assert calls == [["join", str(path), "--workers", "1"]]


def test_compose_generation_records_clamps_beside_the_plan(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    path = configured(tmp_path)
    config = delivery_render.load_config(path)
    config.work.mkdir(parents=True, exist_ok=True)

    def fake_main(arguments: list[str]) -> int:
        delivery_render._clip_json(delivery_render.Clip(Path("/clean.mov"), Fraction(0), 90), 30)
        return 0

    monkeypatch.setattr(mechanical_lane, "media_limit", lambda _media: Fraction(2985, 1000))
    monkeypatch.setattr(delivery_render, "main", fake_main)
    pipeline._generate(path, config, pipeline.STEPS[4])
    written = json.loads((config.work / "clamps.json").read_text(encoding="utf-8"))
    assert written["clamps"] == [
        {"media": "/clean.mov", "startSeconds": "0", "frames": 90, "trimmedSeconds": "3/200"}
    ]


def test_compose_exception_restores_runtime_and_does_not_publish_clamps(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    path = configured(tmp_path)
    config = delivery_render.load_config(path)
    config.work.mkdir()
    original = (
        delivery_render.clips,
        delivery_render._clip_json,
        delivery_render.compose,
        delivery_render._workflows,
    )

    def fail_compose(arguments: list[str]) -> int:
        row = delivery_render._clip_json(
            delivery_render.Clip(Path("/clean.mov"), Fraction(0), 90), 30
        )
        assert row["selection"] == {"startSeconds": 0.0, "durationSeconds": 2.985, "rate": 1}
        raise StageError("Synthetic compose failure")

    monkeypatch.setattr(mechanical_lane, "media_limit", lambda _media: Fraction(2985, 1000))
    monkeypatch.setattr(delivery_render, "main", fail_compose)
    with pytest.raises(StageError, match="Synthetic compose failure"):
        pipeline._generate(path, config, pipeline.STEPS[4])
    assert (
        delivery_render.clips,
        delivery_render._clip_json,
        delivery_render.compose,
        delivery_render._workflows,
    ) == original
    assert not (config.work / "clamps.json").exists()
    assert delivery_render._clip_json(
        delivery_render.Clip(Path("/clean.mov"), Fraction(0), 90), 30
    )["selection"] == {"startSeconds": 0.0, "durationSeconds": 3.0, "rate": 1}


@pytest.mark.parametrize(
    ("names", "error", "message"),
    [
        (("first",), ReplayError, "requires every cleanup binding"),
        (("first", "second", "extra"), StageError, "Unknown source-caption consumer stage"),
    ],
)
def test_compose_incomplete_or_extra_bindings_refuse_before_launch(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    names: tuple[str, ...],
    error: type[Exception],
    message: str,
) -> None:
    path = configured(tmp_path)
    plan = {
        "version": 1,
        "sourceCaptionProduction": {
            "bindings": dict.fromkeys(names, str(tmp_path / "binding.json"))
        },
        "stages": [
            {
                "name": "first",
                "command": ["native", "edit-media", str(tmp_path / "first.json")],
                "inputs": [str(tmp_path / "first.json"), str(tmp_path / "binding.json")],
                "outputs": [str(tmp_path / "first-output.json")],
                "timeout_seconds": 1,
            },
            {
                "name": "second",
                "command": ["native", "edit-media", str(tmp_path / "second.json")],
                "inputs": [str(tmp_path / "second.json"), str(tmp_path / "binding.json")],
                "outputs": [str(tmp_path / "second-output.json")],
                "timeout_seconds": 1,
            },
        ],
    }
    write(tmp_path / "work/compose/partition-0-0000.json", plan)
    monkeypatch.setattr(
        pipeline, "STEPS", (pipeline.Step("compose", "compose/partition-*-*.json"),)
    )
    monkeypatch.setattr(delivery_render, "publish", lambda _: pytest.fail("Must not publish"))
    with pytest.raises(error, match=message):
        pipeline.run(path, launch=lambda _: pytest.fail("Must not launch incomplete bindings"))


def test_reused_unbound_compose_plan_refuses_before_launch(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    path = configured(tmp_path)
    request = write(
        tmp_path / "request.json",
        {"recipe": {"clips": [{"sourcePath": str(tmp_path / "clean.mov")}], "video": {}}},
    )
    binding = source_caption_binding.create_binding(
        source_caption_binding.require_cleanup(path),
        consumer=request,
        kind="native-request",
        output=tmp_path / "binding.json",
    )
    write(
        tmp_path / "work/compose/partition-0-0000.json",
        {
            "version": 1,
            "sourceCaptionProduction": {"bindings": {"partition": str(binding)}},
            "stages": [
                {
                    "name": "partition",
                    "command": ["native", "edit-media", str(request)],
                    "inputs": [str(request), str(binding), str(tmp_path / "clean.mov")],
                    "outputs": [str(tmp_path / "parts/manifest.json")],
                    "timeout_seconds": 1,
                }
            ],
        },
    )
    monkeypatch.setattr(
        pipeline, "STEPS", (pipeline.Step("compose", "compose/partition-*-*.json"),)
    )
    with pytest.raises(ReplayError, match="legacy mechanical plan must be regenerated"):
        pipeline.run(path, launch=lambda _: pytest.fail("Must not launch unbound legacy request"))


def test_render_generation_restores_wrapper_when_original_writer_fails(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    path = configured(tmp_path)
    original = delivery_render._workflows

    def fail(arguments: list[str]) -> int:
        assert delivery_render._workflows is not original
        raise StageError("Synthetic render generation failure")

    monkeypatch.setattr(delivery_render, "main", fail)
    with pytest.raises(StageError, match="Synthetic render generation failure"):
        pipeline._generate(path, delivery_render.load_config(path), pipeline.STEPS[5])
    assert delivery_render._workflows is original


def test_manifest_join_pins_actual_response_movies_in_partition_order(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    path = configured(tmp_path)
    config = delivery_render.load_config(path)
    first = write(tmp_path / "first/request.json", {})
    second = write(tmp_path / "second/request.json", {})
    first_movie, second_movie = tmp_path / "first.mp4", tmp_path / "second.mp4"
    first_movie.write_bytes(b"first native movie")
    second_movie.write_bytes(b"second native movie")
    write(
        second.with_name("render.response.json"),
        {"outputPath": str(second_movie), "frameCount": 60},
    )
    write(
        first.with_name("render.response.json"), {"outputPath": str(first_movie), "frameCount": 30}
    )
    write(
        config.composition / "parts-0001/manifest.json",
        {"parts": [{"path": str(second), "frames": 60}]},
    )
    write(
        config.composition / "parts-0000/manifest.json",
        {"parts": [{"path": str(first), "frames": 30}]},
    )
    monkeypatch.setattr(mechanical_join, "WORKFLOW_ARRAY_LIMIT", 4)
    monkeypatch.setattr(delivery_render, "main", lambda _: pytest.fail("Must use bounded manifest"))
    pipeline._generate(path, config, pipeline.STEPS[6])
    manifest = config.work / "join-manifest.json"
    assert mechanical_join.parse_manifest(json.loads(manifest.read_text())) == (
        mechanical_join.ManifestPart(
            first_movie, 30, hashlib.sha256(b"first native movie").hexdigest()
        ),
        mechanical_join.ManifestPart(
            second_movie, 60, hashlib.sha256(b"second native movie").hexdigest()
        ),
    )
    [stage] = parse_stages(json.loads((config.work / "join-workflow.json").read_text()))
    assert stage.inputs[:3] == (manifest, config.cuts, config.chunks)
    verified_tools = mechanical_plan_provenance.validate_join_stage(stage)
    assert {pin.path for pin in verified_tools}.issubset(stage.inputs)
    assert stage.command[1:4] == ("-m", "replay.mechanical_join", "join-manifest")
    assert stage.outputs == (config.storage / "joined-video.mp4",)


@pytest.mark.parametrize("response", [None, {"frameCount": 29, "outputPath": "/unverified.mp4"}])
def test_manifest_join_missing_or_mismatched_response_never_launches(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, response: object
) -> None:
    path = configured(tmp_path)
    config = delivery_render.load_config(path)
    request = write(tmp_path / "part/request.json", {})
    write(
        config.composition / "parts-0000/manifest.json",
        {"parts": [{"path": str(request), "frames": 30}]},
    )
    if response is not None:
        write(request.with_name("render.response.json"), response)
    monkeypatch.setattr(mechanical_join, "WORKFLOW_ARRAY_LIMIT", 3)
    monkeypatch.setattr(pipeline, "STEPS", (pipeline.Step("join", "join-workflow.json"),))
    monkeypatch.setattr(
        delivery_render, "main", lambda _: pytest.fail("Must not fall back to classic join")
    )
    monkeypatch.setattr(delivery_render, "publish", lambda _: pytest.fail("Must not publish"))
    with pytest.raises((FileNotFoundError, ReplayError)):
        pipeline.run(path, launch=lambda _: pytest.fail("Must not launch unverified parts"))
    assert not (config.work / "join-manifest.json").exists()
    assert not (config.work / "join-workflow.json").exists()


def test_manifest_claim_cannot_replace_the_join_packet_count_check(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    path = configured(tmp_path)
    config = delivery_render.load_config(path)
    request = write(tmp_path / "part/request.json", {})
    movie = tmp_path / "part/movie.mp4"
    movie.write_bytes(b"synthetic bytes with a forged count claim")
    write(request.with_name("render.response.json"), {"outputPath": str(movie), "frameCount": 30})
    write(
        config.composition / "parts-0000/manifest.json",
        {"parts": [{"path": str(request), "frames": 30}]},
    )
    monkeypatch.setattr(mechanical_join, "WORKFLOW_ARRAY_LIMIT", 3)
    pipeline._generate(path, config, pipeline.STEPS[6])
    monkeypatch.setattr(
        native_join,
        "probe_video",
        lambda _: native_join.VideoStream("hevc", Fraction(1, 15360), 29),
    )
    monkeypatch.setattr(
        subprocess,
        "run",
        lambda *_args, **_kwargs: pytest.fail("Must not mux missing frames"),
    )
    with pytest.raises(ReplayError, match="planned frames"):
        mechanical_join.join_manifest(
            config.work / "join-manifest.json", config.storage / "joined-video.mp4", 30
        )
    assert not (config.storage / "joined-video.mp4").exists()


def test_every_step_checks_the_free_space_floor(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config = configured(tmp_path)

    def refuse(paths: object, need: int = 0) -> list[disk_budget.Headroom]:
        raise ReplayError("Free space is below the floor")

    monkeypatch.setattr(disk_budget, "require_headroom", refuse)
    with pytest.raises(ReplayError, match="below the floor"):
        pipeline.run(config, launch=lambda batch: [0 for _ in batch], start="join")
