"""Pipeline steps call the producers in order, resume interrupted work and reset on force."""

import json
import signal
import subprocess
import sys
import threading
from collections.abc import Sequence
from pathlib import Path

import pytest

from replay import recording_steps as steps
from replay.code_dependency_inventory import capture
from replay.core import ReplayError
from replay.stages import execution_lock


def configure(tmp_path: Path) -> Path:
    source = tmp_path / "source.mp4"
    source.write_bytes(b"video")
    references = {name: str(tmp_path / f"{name}.json") for name in sorted(steps.REFERENCES)}
    (tmp_path / "separationTemplate.json").write_text('{"device": "cpu", "budget": {}}', "utf-8")
    (tmp_path / "separationConfig.json").write_text('{"codeInputs": []}', "utf-8")
    data = {
        "version": 1,
        "recording": "123",
        "editRoot": str(tmp_path / "edit"),
        "source": str(source),
        "work": str(tmp_path / "replay-123"),
        "steps": str(tmp_path / "steps"),
        "python": "/py",
        "providerPython": "/provider",
        "native": "/native",
        "ffmpeg": "/ffmpeg",
        "ffprobe": "/ffprobe",
        "references": references,
        "yolo": {"weights": "/w.pt", "sha256": "a" * 64, "configDirectory": "/cfg", "step": "2"},
    }
    (tmp_path / "steps").mkdir()
    path = tmp_path / "pipeline.json"
    path.write_text(json.dumps(data), encoding="utf-8")
    return path


def plan_file(path: Path, output: Path) -> None:
    """A minimal valid sub-workflow with one stage producing ``output``."""
    stage = {
        "name": "only",
        "command": ["true"],
        "inputs": [str(path.parent / "in.txt")],
        "outputs": [str(output)],
        "timeout_seconds": 10,
    }
    path.write_text(json.dumps({"version": 1, "stages": [stage]}), encoding="utf-8")


class Producers:
    """A fake runner that records commands and creates the files real producers would."""

    def __init__(self, tmp_path: Path) -> None:
        self.tmp = tmp_path
        self.commands: list[list[str]] = []

    def work(self) -> Path:
        return self.tmp / "replay-123"

    def __call__(self, command: Sequence[str], log: Path) -> None:
        self.commands.append(list(command))
        work = self.work()
        name = command[command.index("-m") + 1]
        if name == "replay.replay_preparation_plan":
            work.mkdir()
            plan_file(work / "workflow.json", work / "clock-out.txt")
        elif name == "replay.workflow" and command[4] == str(work / "workflow.json"):
            (work / "video-clock.json").write_text('{"frames": [{}, {}, {}]}', "utf-8")
            (work / "audio-clock.json").write_text("{}", encoding="utf-8")
            (work / "preparation.json").write_text("{}", encoding="utf-8")
        elif name == "replay.cleanup_bootstrap_request":
            (work / "cleanup-preparation-v1").mkdir()
            (work / "cleanup-preparation-v1/BOOTSTRAP-REQUEST-v1.json").write_text("{}", "utf-8")
        elif name == "replay.replay_cleanup_plan":
            (work / "cleanup-bootstrap-v1").mkdir()
            plan_file(work / "cleanup-bootstrap-v1/workflow.json", work / "boot-out.txt")
        elif name == "replay.workflow" and "cleanup-bootstrap-v1" in command[4]:
            (work / "cleanup-bootstrap-v1/settings.json").write_text("{}", "utf-8")
            (work / "cleanup-bootstrap-v1/physical-clock.json").write_text("{}", "utf-8")
        elif name == "replay.yolo_track":
            Path(command[5]).write_text("{}", encoding="utf-8")
        elif name == "replay.separation_plan_refresh":
            Path(command[6]).mkdir()
            (Path(command[6]) / "PIPELINE-PLAN-v1.json").write_text("{}", "utf-8")
        elif name == "replay.separation_window_pipeline":
            (work / "separation-v1/pipeline").mkdir(exist_ok=True)
            (work / "separation-v1/pipeline/vocals.wav").write_bytes(b"v")
        elif name == "replay.metal_asr_config":
            model = {"config": {"path": str(work / "model/config.json")}}
            config = {"model": model, "ffmpeg": {"path": "/bin/ffmpeg"}}
            Path(command[6]).write_text(json.dumps(config), encoding="utf-8")
        elif name == "replay.vocal_metal_asr_pipeline":
            plan_file(Path(command[7]) / "workflow.json", Path(command[7]) / "asr-out.txt")
        elif name == "replay.workflow":
            if command[-1] == "assemble":
                (work / "asr-v1/prepared/labels.json").write_text("{}", encoding="utf-8")
            (work / "asr-v1/prepared/mono.wav").write_bytes(b"m")
        elif name == "replay.asr_context_streams" and command[4] == "mono":
            Path(command[-2]).write_bytes(b"m")
            Path(command[-1]).write_text("{}", encoding="utf-8")
        elif name == "replay.asr_context_streams":
            Path(command[-2]).write_text("{}", encoding="utf-8")
        elif name == "replay.asr_silence_exclude":
            Path(command[6]).write_text("{}", encoding="utf-8")


def upstream(tmp_path: Path, *names: str) -> None:
    """Receipts of upstream steps (their content is irrelevant to the step under test)."""
    for name in names:
        (tmp_path / "steps" / f"{name}.json").write_text("{}", encoding="utf-8")


def modules(producers: Producers) -> list[str]:
    return [command[command.index("-m") + 1] for command in producers.commands]


def test_clocks_prepare_run_both_clock_stages_and_hash_the_source(tmp_path: Path) -> None:
    producers = Producers(tmp_path)
    receipt = steps.execute("clocks", configure(tmp_path), run=producers)
    assert modules(producers) == ["replay.replay_preparation_plan", "replay.workflow"]
    assert producers.commands[1][-4:] == [
        "--target",
        "source-video-clock",
        "--target",
        "source-audio-clock",
    ]
    assert (tmp_path / "replay-123/source.sha256").read_text(encoding="utf-8") == (
        f"{steps.sha256(tmp_path / 'source.mp4')}  {tmp_path / 'source.mp4'}\n"
    )
    data = json.loads(receipt.read_text(encoding="utf-8"))
    assert data["step"] == "clocks"
    assert data["qualityApproved"] is False
    assert data["inputs"] == {
        str(tmp_path / "pipeline.json"): steps.sha256(tmp_path / "pipeline.json")
    }
    assert sorted(data["outputs"]) == ["audioClock", "preparation", "videoClock"]


def test_a_partial_preparation_is_removed_and_prepared_again(tmp_path: Path) -> None:
    (tmp_path / "replay-123").mkdir()
    (tmp_path / "replay-123/half.txt").write_text("x", encoding="utf-8")
    producers = Producers(tmp_path)
    steps.execute("clocks", configure(tmp_path), run=producers)
    assert modules(producers) == ["replay.replay_preparation_plan", "replay.workflow"]
    assert (tmp_path / "replay-123/half.txt").exists() is False


def test_a_finished_preparation_is_reused(tmp_path: Path) -> None:
    path = configure(tmp_path)
    steps.execute("clocks", path, run=Producers(tmp_path))
    (tmp_path / "steps/clocks.json").unlink()
    producers = Producers(tmp_path)
    steps.execute("clocks", path, run=producers)
    assert modules(producers) == ["replay.workflow"]


def test_a_finished_step_is_not_run_again(tmp_path: Path) -> None:
    path = configure(tmp_path)
    steps.execute("clocks", path, run=Producers(tmp_path))
    with pytest.raises(ReplayError, match="receipt exists"):
        steps.execute("clocks", path, run=Producers(tmp_path))


def test_finished_compares_the_receipt_with_its_current_inputs(tmp_path: Path) -> None:
    path = configure(tmp_path)
    pipeline = steps.load(path)
    assert steps.finished(pipeline, "separation") is False
    steps.execute("clocks", path, run=Producers(tmp_path))
    steps.execute("separation", path, run=Producers(tmp_path))
    assert steps.finished(pipeline, "separation") is True
    (tmp_path / "steps/clocks.json").write_text("{}", encoding="utf-8")
    with pytest.raises(ReplayError, match="predates its inputs"):
        steps.finished(pipeline, "separation")


def test_an_interrupted_sub_stage_is_cleared_and_rerun(tmp_path: Path) -> None:
    plan, state = tmp_path / "plan.json", tmp_path / "state"
    output = tmp_path / "out.txt"
    plan_file(plan, output)
    state.mkdir()
    (state / "only.json").write_text('{"status": "running"}', encoding="utf-8")
    output.write_text("half", encoding="utf-8")
    assert steps.clear_interrupted(plan, state) == ["only"]
    assert output.exists() is False
    assert (state / "only.json").exists() is False


def test_an_interrupted_stage_leaves_no_empty_output_directory(tmp_path: Path) -> None:
    plan, state = tmp_path / "plan.json", tmp_path / "state"
    output = tmp_path / "tempo/candidate.wav"
    plan_file(plan, output)
    state.mkdir()
    (state / "only.json").write_text('{"status": "running"}', encoding="utf-8")
    output.parent.mkdir()
    assert steps.clear_interrupted(plan, state) == ["only"]
    assert output.parent.exists() is False


def test_an_interrupted_stage_directory_with_partial_files_is_removed(tmp_path: Path) -> None:
    plan, state = tmp_path / "plan.json", tmp_path / "state"
    output = tmp_path / "quiet/cuts.json"
    plan_file(plan, output)
    state.mkdir()
    (state / "only.json").write_text('{"status": "running"}', encoding="utf-8")
    output.parent.mkdir()
    (output.parent / "peaks.txt").write_text("partial", encoding="utf-8")
    assert steps.clear_interrupted(plan, state) == ["only"]
    assert output.parent.exists() is False


def test_partial_files_without_a_receipt_are_removed(tmp_path: Path) -> None:
    plan, state = tmp_path / "plan.json", tmp_path / "state"
    output = tmp_path / "quiet/cuts.json"
    plan_file(plan, output)
    state.mkdir()
    output.parent.mkdir()
    (output.parent / "peaks.txt").write_text("partial", encoding="utf-8")
    assert steps.clear_interrupted(plan, state) == ["only"]
    assert output.parent.exists() is False


def test_a_consumer_of_the_stage_output_does_not_share_its_directory(tmp_path: Path) -> None:
    plan, state = tmp_path / "plan.json", tmp_path / "state"
    quiet = tmp_path / "quiet"
    stages = [
        {"name": "cuts", "command": ["true"], "inputs": [str(tmp_path / "in.txt")],
         "outputs": [str(quiet / "cuts.json")], "timeout_seconds": 10},
        {"name": "use", "command": ["true"], "inputs": [str(quiet / "cuts.json")],
         "outputs": [str(tmp_path / "used.json")], "timeout_seconds": 10},
    ]  # fmt: skip
    plan.write_text(json.dumps({"version": 1, "stages": stages}), encoding="utf-8")
    state.mkdir()
    quiet.mkdir()
    (quiet / "peaks.txt").write_text("partial", encoding="utf-8")
    assert steps.clear_interrupted(plan, state) == ["cuts"]
    assert quiet.exists() is False


def test_complete_outputs_without_a_receipt_are_kept(tmp_path: Path) -> None:
    plan, state = tmp_path / "plan.json", tmp_path / "state"
    output = tmp_path / "quiet/cuts.json"
    plan_file(plan, output)
    state.mkdir()
    output.parent.mkdir()
    output.write_text("{}", encoding="utf-8")
    assert steps.clear_interrupted(plan, state) == []
    assert output.exists() is True


def test_a_directory_another_stage_uses_is_kept(tmp_path: Path) -> None:
    plan, state = tmp_path / "plan.json", tmp_path / "state"
    shared = tmp_path / "prepared"
    stages = [
        {"name": "first", "command": ["true"], "inputs": [str(tmp_path / "in.txt")],
         "outputs": [str(shared / "a.json")], "timeout_seconds": 10},
        {"name": "second", "command": ["true"], "inputs": [str(shared / "a.json")],
         "outputs": [str(shared / "b.json")], "timeout_seconds": 10},
    ]  # fmt: skip
    plan.write_text(json.dumps({"version": 1, "stages": stages}), encoding="utf-8")
    state.mkdir()
    (state / "second.json").write_text('{"status": "running"}', encoding="utf-8")
    shared.mkdir()
    (shared / "a.json").write_text("{}", encoding="utf-8")
    (shared / "b.json").write_text("half", encoding="utf-8")
    assert steps.clear_interrupted(plan, state) == ["second"]
    assert (shared / "a.json").exists() is True
    assert (shared / "b.json").exists() is False


def test_completed_sub_stages_are_kept(tmp_path: Path) -> None:
    plan, state = tmp_path / "plan.json", tmp_path / "state"
    plan_file(plan, tmp_path / "out.txt")
    state.mkdir()
    (state / "only.json").write_text('{"status": "complete"}', encoding="utf-8")
    assert steps.clear_interrupted(plan, state) == []
    assert steps.clear_interrupted(tmp_path / "missing.json", state) == []
    assert (state / "only.json").exists() is True


def test_a_sub_stage_still_running_elsewhere_is_refused(tmp_path: Path) -> None:
    plan, state = tmp_path / "plan.json", tmp_path / "state"
    plan_file(plan, tmp_path / "out.txt")
    state.mkdir()
    (state / "only.json").write_text('{"status": "running"}', encoding="utf-8")
    with execution_lock(state, "only"), pytest.raises(ReplayError, match="still running"):
        steps.clear_interrupted(plan, state)


def test_the_bootstrap_stops_at_the_profile_settings(tmp_path: Path) -> None:
    path = configure(tmp_path)
    steps.execute("clocks", path, run=Producers(tmp_path))
    producers = Producers(tmp_path)
    steps.execute("cleanup-bootstrap", path, run=producers)
    assert modules(producers) == [
        "replay.cleanup_bootstrap_request",
        "replay.replay_cleanup_plan",
        "replay.workflow",
    ]
    assert producers.commands[2][-2:] == ["--target", "profile-settings"]


class CaptionFreeProducers(Producers):
    """The bootstrap's profile verifier refuses a band-free source; the survey succeeds."""

    def __call__(self, command: Sequence[str], log: Path) -> None:
        name = command[command.index("-m") + 1]
        if name == "replay.replay_cleanup_plan" and "caption-absence-v1" in command[-1]:
            self.commands.append(list(command))
            Path(command[-1]).mkdir()
            plan_file(Path(command[-1]) / "workflow.json", self.work() / "survey-out.txt")
            return
        if name == "replay.workflow" and "cleanup-bootstrap-v1" in command[4]:
            self.commands.append(list(command))
            raise ReplayError("Producer failed (1)")
        if name == "replay.workflow":
            self.commands.append(list(command))
            return
        super().__call__(command, log)


def test_a_band_free_source_is_surveyed_and_composed_unchanged(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    path = configure(tmp_path)
    steps.execute("clocks", path, run=Producers(tmp_path))
    monkeypatch.setattr("replay.caption_absence.no_band", lambda _request, _output: True)
    monkeypatch.setattr(
        "replay.caption_absence.summary", lambda _request, _output: {"reason": "none"}
    )
    producers = CaptionFreeProducers(tmp_path)
    receipt = steps.execute("cleanup-bootstrap", path, run=producers)
    survey = tmp_path / "replay-123/caption-absence-v1"
    assert producers.commands[-1][-2:] == ["--target", "profile-ocr"]
    assert json.loads((survey / "request.json").read_text("utf-8")) == {"profileSamples": 240}
    assert json.loads(receipt.read_text("utf-8"))["outputs"] == {
        "captionFree": steps.pin(survey / "summary.json")
    }
    cleaned = steps.execute("cleanup", path, run=Producers(tmp_path))
    assert json.loads(cleaned.read_text("utf-8"))["outputs"] == {
        "captionFree": steps.pin(survey / "summary.json"),
        "cleanSource": str(tmp_path / "source.mp4"),
    }
    assert steps.step_outputs(steps.load(path), "cleanup-bootstrap")[-1] == survey


def test_a_failed_survey_with_a_band_still_fails(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    path = configure(tmp_path)
    steps.execute("clocks", path, run=Producers(tmp_path))
    verdicts = iter((True, False))
    monkeypatch.setattr("replay.caption_absence.no_band", lambda _request, _output: next(verdicts))

    class FailingSurvey(CaptionFreeProducers):
        def __call__(self, command: Sequence[str], log: Path) -> None:
            super().__call__(command, log)
            if command[-1] == "profile-ocr":
                raise ReplayError("Producer failed (1)")

    with pytest.raises(ReplayError, match="Producer failed"):
        steps.execute("cleanup-bootstrap", path, run=FailingSurvey(tmp_path))
    assert (tmp_path / "steps/cleanup-bootstrap.json").exists() is False


def test_a_bootstrap_failing_for_another_reason_is_not_surveyed(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    path = configure(tmp_path)
    steps.execute("clocks", path, run=Producers(tmp_path))
    monkeypatch.setattr("replay.caption_absence.no_band", lambda _request, _output: False)
    producers = CaptionFreeProducers(tmp_path)
    with pytest.raises(ReplayError, match="Producer failed"):
        steps.execute("cleanup-bootstrap", path, run=producers)
    assert (tmp_path / "replay-123/caption-absence-v1").exists() is False


def test_yolo_track_covers_every_clock_frame(tmp_path: Path) -> None:
    path = configure(tmp_path)
    steps.execute("clocks", path, run=Producers(tmp_path))
    producers = Producers(tmp_path)
    steps.execute("yolo-track", path, run=producers)
    command = producers.commands[0]
    assert command[command.index("--end") + 1] == "3"
    assert command[command.index("--step") + 1] == "2"
    assert command[command.index("--device") + 1] == "mps"
    assert command[command.index("--decoder") + 1] == "videotoolbox"


def test_separation_uses_a_gpu_copy_of_the_reference_template(tmp_path: Path) -> None:
    path = configure(tmp_path)
    steps.execute("clocks", path, run=Producers(tmp_path))
    producers = Producers(tmp_path)
    steps.execute("separation", path, run=producers)
    assert modules(producers) == [
        "replay.separation_plan_refresh",
        "replay.separation_window_pipeline",
    ]
    assert producers.commands[1][0] == "/usr/bin/env"
    assert "/provider" in producers.commands[1]
    assert "HF_HUB_OFFLINE=1" in producers.commands[1]
    assert producers.commands[1][-1].endswith("PIPELINE-PLAN-v1.json")
    template = json.loads((tmp_path / "replay-123/separation-template-gpu.json").read_text("utf-8"))
    config = json.loads((tmp_path / "replay-123/separation-config.json").read_text("utf-8"))
    assert (template["device"], template["budget"]) == ("mps", {})
    assert template["codeInputs"] == [
        {"path": str(item.path), "sha256": item.sha256} for item in capture(steps.PLAN_CLOSURE)
    ]
    assert config["codeInputs"] == [
        {"path": str(item.path), "sha256": item.sha256}
        for item in capture(steps.PREPARATION_CLOSURE)
    ]
    assert producers.commands[0][4:6] == [
        str(tmp_path / "replay-123/separation-config.json"),
        str(tmp_path / "replay-123/separation-template-gpu.json"),
    ]


def test_an_interrupted_separation_resumes(tmp_path: Path) -> None:
    path = configure(tmp_path)
    steps.execute("clocks", path, run=Producers(tmp_path))
    namespace = tmp_path / "replay-123/separation-v1"
    (namespace / "pipeline").mkdir(parents=True)
    (namespace / "PIPELINE-PLAN-v1.json").write_text("{}", encoding="utf-8")
    producers = Producers(tmp_path)
    steps.execute("separation", path, run=producers)
    assert modules(producers) == ["replay.separation_window_pipeline"]
    assert producers.commands[0][-1] == "--resume"


def test_a_plan_pinning_an_older_closure_is_rebuilt_before_it_runs(tmp_path: Path) -> None:
    path = configure(tmp_path)
    steps.execute("clocks", path, run=Producers(tmp_path))
    namespace = tmp_path / "replay-123/separation-v1"
    namespace.mkdir()
    (namespace / "PIPELINE-PLAN-v1.json").write_text('{"codeInputs": []}', encoding="utf-8")
    producers = Producers(tmp_path)
    steps.execute("separation", path, run=producers)
    assert modules(producers) == [
        "replay.separation_plan_refresh",
        "replay.separation_window_pipeline",
    ]


def test_a_plan_pinning_the_current_closure_is_kept(tmp_path: Path) -> None:
    plan = tmp_path / "plan.json"
    current = [
        {"path": str(item.path), "sha256": item.sha256} for item in capture(steps.PLAN_CLOSURE)
    ]
    plan.write_text(json.dumps({"codeInputs": current}), encoding="utf-8")
    assert steps.current_pins(plan) is True


def test_asr_prepares_runs_and_excludes_silence(tmp_path: Path) -> None:
    path = configure(tmp_path)
    steps.execute("clocks", path, run=Producers(tmp_path))
    upstream(tmp_path, "separation")
    producers = Producers(tmp_path)
    receipt = steps.execute("asr", path, run=producers)
    assert modules(producers) == [
        "replay.metal_asr_config",
        "replay.asr_context_streams",
        "replay.asr_context_streams",
        "replay.asr_silence_exclude",
    ]
    prepared = tmp_path / "replay-123/asr-v1/prepared"
    assert producers.commands[1][-5:] == [
        "mono", str(tmp_path / "replay-123/separation-v1/pipeline/vocals.wav"), "/bin/ffmpeg",
        str(prepared / "mono.wav"), str(prepared / "mono-receipt.json"),
    ]  # fmt: skip
    assert producers.commands[2][-6:] == [
        "publish", str(prepared / "mono.wav"), str(tmp_path / "replay-123/model"),
        str(prepared / "context-streams"), str(prepared / "labels.json"),
        str(prepared / "asr/inputs.json"),
    ]  # fmt: skip
    assert producers.commands[3][-5:] == [
        "--minimum-seconds",
        "5",
        "--maximum-amplitude",
        "8",
        "--clamp-partial",
    ]
    assert sorted(json.loads(receipt.read_text("utf-8"))["outputs"]) == ["labels", "mono"]


def write_cleanup(tmp_path: Path) -> Path:
    clean = tmp_path / "clean"
    (clean / "consumer").mkdir(parents=True)
    (clean / "consumer/source-cleanup.json").write_text("{}", encoding="utf-8")
    description = tmp_path / "edits/work/clean-replay-123-p1.description.json"
    description.parent.mkdir(parents=True, exist_ok=True)
    description.write_text(
        json.dumps({"work": str(clean), "mediaRoot": str(tmp_path / "media"), "recording": "r"}),
        encoding="utf-8",
    )
    return description


def test_cleanup_redoes_only_unfinished_chunks_and_reuses_the_assembly(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    path = configure(tmp_path)
    write_cleanup(tmp_path)
    states = tmp_path / "clean/state"
    (states / "chunk-00000").mkdir(parents=True)
    (states / "chunk-00000/chunk-receipt.json").write_text("{}", encoding="utf-8")
    (states / "chunk-00001").mkdir()
    (tmp_path / "media/r/sampled/chunk-00001").mkdir(parents=True)
    (tmp_path / "clean/logs").mkdir()
    (tmp_path / "clean/logs/chunk-00001.log").write_text("cut", encoding="utf-8")
    (tmp_path / "clean/logs/chunk-00002.log").write_text("cut early", encoding="utf-8")
    (tmp_path / "clean/logs/chunk-00000.log").write_text("done", encoding="utf-8")
    (states / "chunk-00003").mkdir()
    (states / "chunk-00003/chunk-receipt.json").write_text('{"accepted": null}', encoding="utf-8")
    (tmp_path / "media/r/sampled/chunk-00003").mkdir()
    monkeypatch.chdir(tmp_path)
    upstream(tmp_path, "cleanup-bootstrap")
    producers = Producers(tmp_path)
    receipt = steps.execute("cleanup", path, run=producers)
    assert [command[-1] for command in producers.commands] == ["configs", "run"]
    assert (states / "chunk-00000").exists() is True
    assert (states / "chunk-00001").exists() is False
    assert (tmp_path / "media/r/sampled/chunk-00001").exists() is False
    assert (tmp_path / "clean/logs/chunk-00001.log").exists() is False
    assert len(list((tmp_path / "clean/logs").glob("chunk-00001.interrupted-*.log"))) == 1
    assert len(list((tmp_path / "clean/logs").glob("chunk-00002.interrupted-*.log"))) == 1
    assert (tmp_path / "clean/logs/chunk-00000.log").exists() is True
    assert (states / "chunk-00003").exists() is False
    assert len(list((tmp_path / "clean/rejected").glob("chunk-00003-*/chunk-receipt.json"))) == 1
    assert (tmp_path / "media/r/sampled/chunk-00003").exists() is False
    outputs = json.loads(receipt.read_text("utf-8"))["outputs"]
    assert outputs["cleanSource"] == str(tmp_path / "media/r/assembly/source.mov")


def test_cleanup_describes_and_assembles_a_new_recording(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    path = configure(tmp_path)
    monkeypatch.chdir(tmp_path)
    commands: list[list[str]] = []

    def run(command: Sequence[str], log: Path) -> None:
        commands.append(list(command))
        if "describe" in command:
            write_cleanup(tmp_path)
            (tmp_path / "clean/consumer/source-cleanup.json").unlink()
            (tmp_path / "media/r/assembly").mkdir(parents=True)
        if command[-1] == "assemble":
            (tmp_path / "clean/consumer").mkdir()
            (tmp_path / "clean/consumer/source-cleanup.json").write_text("{}", encoding="utf-8")

    upstream(tmp_path, "cleanup-bootstrap")
    steps.execute("cleanup", path, run=run)
    assert commands[0][-5:] == [
        "describe",
        "--defaults",
        str(tmp_path / "cleanupDefaults.json"),
        "--revision",
        "p1",
    ]
    assert [command[-1] for command in commands[1:]] == ["configs", "run", "assemble"]


def test_a_caption_free_cleanup_retires_only_separation_intermediates(tmp_path: Path) -> None:
    path = configure(tmp_path)
    (tmp_path / "replay-123/separation-v1").mkdir(parents=True)
    (tmp_path / "replay-123/separation-v1/model-input.wav").write_bytes(b"123")
    (tmp_path / "steps/asr.json").write_text("{}", encoding="utf-8")
    (tmp_path / "steps/cleanup.json").write_text(
        json.dumps({"outputs": {"captionFree": {}, "cleanSource": "/s.mp4"}}), encoding="utf-8"
    )
    receipt = steps.execute("retire-intermediates", path, run=Producers(tmp_path))
    assert json.loads(receipt.read_text("utf-8"))["outputs"] == {
        "removed": [
            {"path": str(tmp_path / "replay-123/separation-v1/model-input.wav"), "bytes": 3}
        ],
        "bytes": 3,
    }


def test_intermediates_retire_only_after_their_consumers_passed(tmp_path: Path) -> None:
    path = configure(tmp_path)
    with pytest.raises(ReplayError, match="The asr step has not passed"):
        steps.execute("retire-intermediates", path, run=Producers(tmp_path))
    chunks = tmp_path / "replay-123/separation-v1/pipeline/chunks"
    chunks.mkdir(parents=True)
    (chunks / "c.npy").write_bytes(b"12345")
    (tmp_path / "replay-123/separation-v1/model-input.wav").write_bytes(b"123")
    media = tmp_path / "media/sampled"
    media.mkdir(parents=True)
    (media / "chunk.mp4").write_bytes(b"1")
    (tmp_path / "steps/asr.json").write_text("{}", encoding="utf-8")
    (tmp_path / "steps/cleanup.json").write_text(
        json.dumps({"outputs": {"chunkMedia": str(media)}}), encoding="utf-8"
    )
    receipt = steps.execute("retire-intermediates", path, run=Producers(tmp_path))
    outputs = json.loads(receipt.read_text("utf-8"))["outputs"]
    assert outputs == {
        "removed": [
            {"path": str(chunks), "bytes": 5},
            {"path": str(tmp_path / "replay-123/separation-v1/model-input.wav"), "bytes": 3},
            {"path": str(media), "bytes": 1},
        ],
        "bytes": 9,
    }


def test_reset_removes_a_step_outputs_and_receipt(tmp_path: Path) -> None:
    path = configure(tmp_path)
    steps.execute("clocks", path, run=Producers(tmp_path))
    steps.execute("separation", path, run=Producers(tmp_path))
    pipeline = steps.load(path)
    removed = steps.reset(pipeline, "separation")
    assert [item["path"] for item in removed] == [
        str(tmp_path / "replay-123/separation-v1"),
        str(tmp_path / "replay-123/separation-config.json"),
        str(tmp_path / "replay-123/separation-template-gpu.json"),
        str(tmp_path / "steps/separation.json"),
    ]
    assert steps.reset(pipeline, "retire-intermediates") == []


def test_reset_of_cleanup_removes_its_work_media_and_description(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    pipeline = steps.load(configure(tmp_path))
    monkeypatch.chdir(tmp_path)
    assert steps.reset(pipeline, "cleanup") == []
    description = write_cleanup(tmp_path)
    (tmp_path / "media/r").mkdir(parents=True)
    paths = [item["path"] for item in steps.reset(pipeline, "cleanup")]
    assert paths == [str(tmp_path / "clean"), str(tmp_path / "media/r"), str(description)]


def test_remove_reports_sizes_and_ignores_missing_paths(tmp_path: Path) -> None:
    (tmp_path / "f.bin").write_bytes(b"abc")
    assert steps.remove(tmp_path / "f.bin") == 3
    assert steps.remove(tmp_path / "missing") == 0


def test_the_producer_appends_output_and_stops_on_failure(tmp_path: Path) -> None:
    producer = steps.Producer()
    log = tmp_path / "logs/step.log"
    producer([sys.executable, "-c", "print('hello')"], log)
    assert log.read_text(encoding="utf-8").endswith("hello\n")
    with pytest.raises(ReplayError, match="Producer failed \\(4\\)"):
        producer([sys.executable, "-c", "raise SystemExit(4)"], log)
    producer.stop()


def test_the_producer_stops_its_running_child(tmp_path: Path) -> None:
    producer = steps.Producer()
    failures: list[str] = []

    def target() -> None:
        try:
            producer([sys.executable, "-c", "import time; time.sleep(60)"], tmp_path / "s.log")
        except ReplayError as error:
            failures.append(str(error))

    thread = threading.Thread(target=target)
    thread.start()
    threading.Event().wait(0.5)
    producer.stop()
    thread.join(timeout=20)
    assert failures == [f"Producer failed (-15); inspect {tmp_path / 's.log'}"]


def test_a_producer_ignoring_sigterm_is_killed(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(steps, "STOP_GRACE_SECONDS", 0.5)
    producer = steps.Producer()
    failures: list[str] = []
    stubborn = "import signal,time; signal.signal(signal.SIGTERM, signal.SIG_IGN); time.sleep(60)"

    def target() -> None:
        try:
            producer([sys.executable, "-c", stubborn], tmp_path / "k.log")
        except ReplayError as error:
            failures.append(str(error))

    thread = threading.Thread(target=target)
    thread.start()
    threading.Event().wait(0.8)
    producer.stop()
    thread.join(timeout=20)
    assert failures == [f"Producer failed (-9); inspect {tmp_path / 'k.log'}"]


@pytest.mark.parametrize(
    "change",
    [
        {"version": 2},
        {"extra": 1},
        {"references": {}},
        {"yolo": {}},
        {"python": "relative/python"},
    ],
)
def test_invalid_configurations_are_refused(tmp_path: Path, change: dict[str, object]) -> None:
    path = configure(tmp_path)
    data = {**json.loads(path.read_text(encoding="utf-8")), **change}
    path.write_text(json.dumps(data), encoding="utf-8")
    with pytest.raises(ReplayError):
        steps.load(path)


def test_a_clock_without_frames_is_refused(tmp_path: Path) -> None:
    clock = tmp_path / "clock.json"
    clock.write_text('{"frames": []}', encoding="utf-8")
    with pytest.raises(ReplayError, match="no frames"):
        steps.frame_count(clock)


def test_a_text_value_must_be_nonempty(tmp_path: Path) -> None:
    pipeline = steps.load(configure(tmp_path))
    data = {**pipeline.data, "native": ""}
    with pytest.raises(ReplayError, match="Expected text"):
        steps.Pipeline(pipeline.path, data).text("native")


def test_the_cli_prints_the_receipt(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    path = configure(tmp_path)
    monkeypatch.setitem(steps.STEPS, "clocks", lambda pipeline, run: {"called": True})
    assert steps.main(["clocks", str(path)]) == 0
    assert capsys.readouterr().out.strip() == str(tmp_path / "steps/clocks.json")


def test_sigterm_stops_a_step(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    path = configure(tmp_path)

    def interrupted(pipeline: steps.Pipeline, run: steps.Runner) -> dict[str, object]:
        handler = signal.getsignal(signal.SIGTERM)
        if callable(handler):
            handler(signal.SIGTERM, None)
        return {}

    monkeypatch.setitem(steps.STEPS, "clocks", interrupted)
    with pytest.raises(ReplayError, match="Step stopped"):
        steps.main(["clocks", str(path)])
    assert (tmp_path / "steps/clocks.json").exists() is False


def test_the_cli_runs_as_a_module(tmp_path: Path) -> None:
    result = subprocess.run(
        [sys.executable, "-m", "replay.recording_steps", "--help"],
        capture_output=True, text=True, check=True,
    )  # fmt: skip
    assert "retire-intermediates" in result.stdout


def test_asr_with_windows_assembles_the_windowed_recognition(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(steps, "ASR_METHOD", "windows")
    path = configure(tmp_path)
    steps.execute("clocks", path, run=Producers(tmp_path))
    upstream(tmp_path, "separation")
    producers = Producers(tmp_path)
    steps.execute("asr", path, run=producers)
    assert "replay.asr_context_streams" not in modules(producers)
    assert producers.commands[2][-2:] == ["--target", "assemble"]
