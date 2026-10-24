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
        elif name == "replay.workflow_parallel" and command[4] == str(work / "workflow.json"):
            (work / "video-clock.json").write_text('{"frames": [{}, {}, {}]}', "utf-8")
            (work / "audio-clock.json").write_text("{}", encoding="utf-8")
            (work / "preparation.json").write_text("{}", encoding="utf-8")
            pinned = {"path": str(self.tmp / "source.mp4"), "sha256": "ab" * 32}
            (work / "video-clock.producer.json").write_text(json.dumps({"source": pinned}), "utf-8")
        elif name == "replay.workflow" and command[4] == str(work / "workflow.json"):
            (work / "audio-clock.json").write_text("{}", encoding="utf-8")
            (work / "audio-clock.producer.json").write_text("{}", encoding="utf-8")
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
        elif name == "replay.asr_hallucination":
            Path(command[-1]).write_text("{}", encoding="utf-8")


def upstream(tmp_path: Path, *names: str) -> None:
    """Receipts of upstream steps (their content is irrelevant to the step under test)."""
    for name in names:
        (tmp_path / "steps" / f"{name}.json").write_text("{}", encoding="utf-8")


def modules(producers: Producers) -> list[str]:
    return [command[command.index("-m") + 1] for command in producers.commands]


def clocked(path: Path, tmp_path: Path) -> None:
    """The audio clock, then both clocks: what every step after them waits for."""
    steps.execute("audio-clock", path, run=Producers(tmp_path))
    steps.execute("clocks", path, run=Producers(tmp_path))


def test_the_audio_clock_prepares_and_runs_only_the_audio_clock_stage(tmp_path: Path) -> None:
    producers = Producers(tmp_path)
    receipt = steps.execute("audio-clock", configure(tmp_path), run=producers)
    assert modules(producers) == ["replay.replay_preparation_plan", "replay.workflow"]
    assert producers.commands[1][-2:] == ["--target", "source-audio-clock"]
    data = json.loads(receipt.read_text(encoding="utf-8"))
    assert data["inputs"] == {
        str(tmp_path / "pipeline.json"): steps.sha256(tmp_path / "pipeline.json")
    }
    assert sorted(data["outputs"]) == ["audioClock", "audioClockProducer", "preparation"]


def test_the_audio_clock_of_a_recording_whose_clocks_passed_runs_nothing(tmp_path: Path) -> None:
    path = configure(tmp_path)
    clocked(path, tmp_path)
    (tmp_path / "steps/audio-clock.json").unlink()
    producers = Producers(tmp_path)
    receipt = steps.execute("audio-clock", path, run=producers)
    assert producers.commands == []
    assert json.loads(receipt.read_text(encoding="utf-8"))["outputs"]["audioClock"] == {
        "path": str(tmp_path / "replay-123/audio-clock.json"),
        "sha256": "44136fa355b3678a1146ad16f7e8649e94fb4fc21fe77e8310c060f61caaff8a",
    }


def test_receipts_bound_before_the_audio_clock_step_are_still_finished(tmp_path: Path) -> None:
    path = configure(tmp_path)
    pipeline = steps.load(path)
    clocked(path, tmp_path)
    steps.execute("separation", path, run=Producers(tmp_path))
    clocks = json.loads((tmp_path / "steps/clocks.json").read_text(encoding="utf-8"))
    alone = {str(tmp_path / "pipeline.json"): steps.sha256(tmp_path / "pipeline.json")}
    (tmp_path / "steps/clocks.json").write_text(json.dumps({**clocks, "inputs": alone}), "utf-8")
    assert steps.finished(pipeline, "clocks") is True
    separation = json.loads((tmp_path / "steps/separation.json").read_text(encoding="utf-8"))
    legacy = {
        **alone,
        str(tmp_path / "steps/clocks.json"): steps.sha256(tmp_path / "steps/clocks.json"),
    }
    (tmp_path / "steps/separation.json").write_text(
        json.dumps({**separation, "inputs": legacy}), "utf-8"
    )
    assert steps.finished(pipeline, "separation") is True
    (tmp_path / "steps/clocks.json").write_text("{}", encoding="utf-8")
    with pytest.raises(ReplayError, match="separation receipt predates"):
        steps.finished(pipeline, "separation")


def test_reset_of_the_clocks_keeps_the_audio_clock_that_separation_used(tmp_path: Path) -> None:
    path = configure(tmp_path)
    clocked(path, tmp_path)
    (tmp_path / "replay-123/state").mkdir()
    (tmp_path / "replay-123/state/source-video-clock.json").write_text("{}", encoding="utf-8")
    paths = [item["path"] for item in steps.reset(steps.load(path), "clocks")]
    assert paths == [
        str(tmp_path / "replay-123/video-clock.json"),
        str(tmp_path / "replay-123/video-clock.producer.json"),
        str(tmp_path / "replay-123/state/source-video-clock.json"),
        str(tmp_path / "replay-123/source.sha256"),
        str(tmp_path / "steps/clocks.json"),
    ]
    assert (tmp_path / "replay-123/audio-clock.json").exists() is True


def test_clocks_run_both_clock_stages_after_the_audio_clock_and_hash_the_source(
    tmp_path: Path,
) -> None:
    path = configure(tmp_path)
    steps.execute("audio-clock", path, run=Producers(tmp_path))
    producers = Producers(tmp_path)
    receipt = steps.execute("clocks", path, run=producers)
    assert modules(producers) == ["replay.workflow_parallel"]
    assert producers.commands[0][-4:] == [
        "--target",
        "source-video-clock",
        "--target",
        "source-audio-clock",
    ]
    assert (tmp_path / "replay-123/source.sha256").read_text(encoding="utf-8") == (
        f"{'ab' * 32}  {tmp_path / 'source.mp4'}\n"
    )
    data = json.loads(receipt.read_text(encoding="utf-8"))
    assert data["step"] == "clocks"
    assert data["qualityApproved"] is False
    assert data["inputs"] == {
        str(tmp_path / "pipeline.json"): steps.sha256(tmp_path / "pipeline.json"),
        str(tmp_path / "steps/audio-clock.json"): steps.sha256(tmp_path / "steps/audio-clock.json"),
    }
    assert sorted(data["outputs"]) == ["audioClock", "preparation", "videoClock"]


def test_clocks_refuse_evidence_of_another_source(tmp_path: Path) -> None:
    class Elsewhere(Producers):
        def __call__(self, command: Sequence[str], log: Path) -> None:
            super().__call__(command, log)
            evidence = self.work() / "video-clock.producer.json"
            if evidence.exists():
                evidence.write_text('{"source": {"path": "/other.mp4"}}', encoding="utf-8")

    path = configure(tmp_path)
    steps.execute("audio-clock", path, run=Producers(tmp_path))
    with pytest.raises(ReplayError, match="another source"):
        steps.execute("clocks", path, run=Elsewhere(tmp_path))
    assert not (tmp_path / "replay-123/source.sha256").exists()


def test_a_partial_preparation_is_removed_and_prepared_again(tmp_path: Path) -> None:
    (tmp_path / "replay-123").mkdir()
    (tmp_path / "replay-123/half.txt").write_text("x", encoding="utf-8")
    producers = Producers(tmp_path)
    steps.execute("audio-clock", configure(tmp_path), run=producers)
    assert modules(producers) == ["replay.replay_preparation_plan", "replay.workflow"]
    assert (tmp_path / "replay-123/half.txt").exists() is False


def test_a_finished_preparation_is_reused(tmp_path: Path) -> None:
    path = configure(tmp_path)
    clocked(path, tmp_path)
    (tmp_path / "steps/clocks.json").unlink()
    producers = Producers(tmp_path)
    steps.execute("clocks", path, run=producers)
    assert modules(producers) == ["replay.workflow_parallel"]


def test_a_finished_step_is_not_run_again(tmp_path: Path) -> None:
    path = configure(tmp_path)
    steps.execute("audio-clock", path, run=Producers(tmp_path))
    with pytest.raises(ReplayError, match="receipt exists"):
        steps.execute("audio-clock", path, run=Producers(tmp_path))


def test_finished_compares_the_receipt_with_its_current_inputs(tmp_path: Path) -> None:
    path = configure(tmp_path)
    pipeline = steps.load(path)
    assert steps.finished(pipeline, "separation") is False
    steps.execute("audio-clock", path, run=Producers(tmp_path))
    steps.execute("separation", path, run=Producers(tmp_path))
    assert steps.finished(pipeline, "separation") is True
    (tmp_path / "steps/audio-clock.json").write_text("{}", encoding="utf-8")
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
    clocked(path, tmp_path)
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
    clocked(path, tmp_path)
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


def test_a_band_free_source_with_comment_boxes_is_cleaned_by_its_intent(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    path = configure(tmp_path)
    clocked(path, tmp_path)
    monkeypatch.setattr("replay.caption_absence.no_band", lambda _request, _output: True)
    monkeypatch.setattr(
        "replay.caption_absence.summary", lambda _request, _output: {"reason": "none"}
    )
    steps.execute("cleanup-bootstrap", path, run=CaptionFreeProducers(tmp_path))
    (tmp_path / "edit").mkdir(exist_ok=True)
    (tmp_path / "edit/cleanup.json").write_text(
        '{"version": 1, "remove": "comment-boxes"}', encoding="utf-8"
    )
    cleaned = tmp_path / "deliveries/source-comment-clean/edit/source.mp4"
    monkeypatch.setattr(
        "replay.source_comment_clean.location", lambda _root, source: cleaned.parent / source.name
    )
    commands: list[list[str]] = []

    def run(command: Sequence[str], _log: Path) -> None:
        commands.append(list(command[2:]))
        cleaned.parent.mkdir(parents=True)
        (cleaned.parent / "comment-clean.json").write_text("{}", encoding="utf-8")

    receipt = steps.execute("cleanup", path, run=run)
    survey = tmp_path / "replay-123/caption-absence-v1"
    assert commands == [
        ["-m", "replay.source_comment_clean", str(tmp_path / "edit"), str(tmp_path / "source.mp4"),
         "--defaults", str(tmp_path / "cleanupDefaults.json"),
         "--ffmpeg", "/ffmpeg", "--ffprobe", "/ffprobe"]
    ]  # fmt: skip
    assert json.loads(receipt.read_text("utf-8"))["outputs"] == {
        "captionFree": steps.pin(survey / "summary.json"),
        "commentClean": steps.pin(cleaned.parent / "comment-clean.json"),
        "cleanSource": str(cleaned),
    }


def test_comment_box_removal_refuses_a_source_with_a_caption_band(tmp_path: Path) -> None:
    path = configure(tmp_path)
    (tmp_path / "edit").mkdir(exist_ok=True)
    (tmp_path / "edit/cleanup.json").write_text(
        '{"version": 1, "remove": "comment-boxes"}', encoding="utf-8"
    )
    clocked(path, tmp_path)
    steps.execute("cleanup-bootstrap", path, run=Producers(tmp_path))
    with pytest.raises(ReplayError, match="without a caption band"):
        steps.execute("cleanup", path, run=Producers(tmp_path))


def test_a_failed_survey_with_a_band_still_fails(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    path = configure(tmp_path)
    clocked(path, tmp_path)
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
    clocked(path, tmp_path)
    monkeypatch.setattr("replay.caption_absence.no_band", lambda _request, _output: False)
    producers = CaptionFreeProducers(tmp_path)
    with pytest.raises(ReplayError, match="Producer failed"):
        steps.execute("cleanup-bootstrap", path, run=producers)
    assert (tmp_path / "replay-123/caption-absence-v1").exists() is False


def test_yolo_track_covers_every_clock_frame(tmp_path: Path) -> None:
    path = configure(tmp_path)
    clocked(path, tmp_path)
    (tmp_path / "cleanupDefaults.json").write_text('{"chunkFrames": 900}', "utf-8")
    producers = Producers(tmp_path)
    steps.execute("yolo-track", path, run=producers)
    command = producers.commands[0]
    assert command[command.index("--batch") :][:4] == ["--batch", "32", "--chunk-frames", "900"]
    assert command[command.index("--end") + 1] == "3"
    assert command[command.index("--step") + 1] == "2"
    assert command[command.index("--device") + 1] == "mps"
    assert command[command.index("--decoder") + 1] == "videotoolbox"


def test_separation_uses_a_gpu_copy_of_the_reference_template(tmp_path: Path) -> None:
    path = configure(tmp_path)
    clocked(path, tmp_path)
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
    clocked(path, tmp_path)
    namespace = tmp_path / "replay-123/separation-v1"
    (namespace / "pipeline").mkdir(parents=True)
    (namespace / "PIPELINE-PLAN-v1.json").write_text("{}", encoding="utf-8")
    producers = Producers(tmp_path)
    steps.execute("separation", path, run=producers)
    assert modules(producers) == ["replay.separation_window_pipeline"]
    assert producers.commands[0][-1] == "--resume"


def test_a_plan_pinning_an_older_closure_is_rebuilt_before_it_runs(tmp_path: Path) -> None:
    path = configure(tmp_path)
    clocked(path, tmp_path)
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
    clocked(path, tmp_path)
    upstream(tmp_path, "separation")
    producers = Producers(tmp_path)
    receipt = steps.execute("asr", path, run=producers)
    assert modules(producers) == [
        "replay.metal_asr_config",
        "replay.asr_context_streams",
        "replay.asr_context_streams",
        "replay.asr_silence_exclude",
        "replay.asr_hallucination",
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
    assert producers.commands[4][-3:] == [
        str(prepared / "labels-silence-v1.json"), str(prepared / "mono.wav"),
        str(prepared / "labels-hallucination-v1.json"),
    ]  # fmt: skip


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


def test_a_chunk_decoding_other_pts_than_the_packet_clock_refuses_it(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    path = configure(tmp_path)
    write_cleanup(tmp_path)
    (tmp_path / "replay-123").mkdir()
    clock = tmp_path / "replay-123/video-clock.json"
    clock.write_text('{"clockProducer": {"method": "packet-v1"}}', encoding="utf-8")
    monkeypatch.chdir(tmp_path)
    upstream(tmp_path, "cleanup-bootstrap")

    def run(command: Sequence[str], log: Path) -> None:
        if command[-1] == "run":
            (tmp_path / "clean/logs").mkdir()
            (tmp_path / "clean/logs/chunk-00000.log").write_text(
                "ValueError: Actual decoded PTS differ from the supplied source clock", "utf-8"
            )
            raise ReplayError("Producer failed (1)")

    with pytest.raises(ReplayError, match="DECODED_CLOCK_MISMATCH cleanup chunk-00000"):
        steps.execute("cleanup", path, run=run)
    assert json.loads((tmp_path / "replay-123/packet-clock-refused.json").read_text("utf-8")) == {
        "version": 1,
        "consumer": "cleanup chunk-00000",
        "reason": "Actual decoded PTS differ from the supplied source clock",
    }


def test_a_failed_chunk_with_a_decoded_clock_fails_as_it_is(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    path = configure(tmp_path)
    write_cleanup(tmp_path)
    (tmp_path / "replay-123").mkdir()
    (tmp_path / "replay-123/video-clock.json").write_text('{"frames": []}', encoding="utf-8")
    monkeypatch.chdir(tmp_path)
    upstream(tmp_path, "cleanup-bootstrap")

    def run(command: Sequence[str], log: Path) -> None:
        if command[-1] == "run":
            raise ReplayError("Producer failed (1)")

    with pytest.raises(ReplayError, match=r"^Producer failed \(1\)$"):
        steps.execute("cleanup", path, run=run)


def test_only_the_current_failed_chunk_logs_classify_a_refusal(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    path = configure(tmp_path)
    write_cleanup(tmp_path)
    (tmp_path / "replay-123").mkdir()
    clock = tmp_path / "replay-123/video-clock.json"
    clock.write_text('{"clockProducer": {"method": "packet-v1"}}', encoding="utf-8")
    monkeypatch.chdir(tmp_path)
    upstream(tmp_path, "cleanup-bootstrap")
    mismatch = "Actual decoded PTS differ from the supplied source clock"

    def run(command: Sequence[str], log: Path) -> None:
        if command[-1] == "run":
            (tmp_path / "clean/logs").mkdir()
            (tmp_path / "clean/state/chunk-00000").mkdir(parents=True)
            (tmp_path / "clean/state/chunk-00000/chunk-receipt.json").write_text("{}", "utf-8")
            (tmp_path / "clean/logs/chunk-00000.log").write_text(mismatch, "utf-8")
            (tmp_path / "clean/logs/chunk-00001.interrupted-1.log").write_text(mismatch, "utf-8")
            (tmp_path / "clean/logs/chunk-00002.log").write_text("other failure", "utf-8")
            raise ReplayError("Producer failed (1)")

    with pytest.raises(ReplayError, match=r"^Producer failed \(1\)$"):
        steps.execute("cleanup", path, run=run)
    assert (tmp_path / "replay-123/packet-clock-refused.json").exists() is False


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
    clocked(path, tmp_path)
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
    monkeypatch.setitem(steps.STEPS, "audio-clock", lambda pipeline, run: {"called": True})
    assert steps.main(["audio-clock", str(path)]) == 0
    assert capsys.readouterr().out.strip() == str(tmp_path / "steps/audio-clock.json")


def test_sigterm_stops_a_step(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    path = configure(tmp_path)

    def interrupted(pipeline: steps.Pipeline, run: steps.Runner) -> dict[str, object]:
        handler = signal.getsignal(signal.SIGTERM)
        if callable(handler):
            handler(signal.SIGTERM, None)
        return {}

    monkeypatch.setitem(steps.STEPS, "audio-clock", interrupted)
    with pytest.raises(ReplayError, match="Step stopped"):
        steps.main(["audio-clock", str(path)])
    assert (tmp_path / "steps/audio-clock.json").exists() is False


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
    clocked(path, tmp_path)
    upstream(tmp_path, "separation")
    producers = Producers(tmp_path)
    steps.execute("asr", path, run=producers)
    assert "replay.asr_context_streams" not in modules(producers)
    assert producers.commands[2][-2:] == ["--target", "assemble"]


def test_a_recording_is_named_by_its_replay_or_youtube_id() -> None:
    assert steps.recording_name("7698733815817341702") == "replay-7698733815817341702"
    assert steps.recording_name("youtube-Eg9-gl3Z954") == "youtube-Eg9-gl3Z954"
    with pytest.raises(ReplayError, match="TikTok replay ID or youtube-<video id>"):
        steps.recording_name("youtube-../x")


def test_a_failed_workflow_keeps_its_failed_stages_log_tails(tmp_path: Path) -> None:
    state = tmp_path / "state"
    (state / "native-pos-paging-ab12cd_3").mkdir(parents=True)
    (state / "native-pos-paging-ab12cd_3/run.log").write_text(
        "".join(f"line {n}\n" for n in range(50)) + "ReplayError: geometry\n", "utf-8"
    )
    (state / "native-pos-paging.json").write_text('{"status": "running"}', "utf-8")
    (state / "cut-tempo-audio-zz99zz99").mkdir()
    (state / "cut-tempo-audio-zz99zz99/run.log").write_text("fine\n", "utf-8")
    (state / "cut-tempo-audio.json").write_text('{"status": "complete"}', "utf-8")
    (state / "broken.json").write_text("[", "utf-8")
    (state / "orphan.json").write_text('{"status": "running"}', "utf-8")
    log = tmp_path / "logs/paging.log"
    with pytest.raises(ReplayError, match="Producer failed \\(3\\)"):
        steps.Producer()([sys.executable, "-c", "raise SystemExit(3)", "--state", str(state)], log)
    lines = log.read_text(encoding="utf-8").splitlines()
    assert lines[1] == (
        f"--- failed stage native-pos-paging: last lines of "
        f"{state}/native-pos-paging-ab12cd_3/run.log ---"
    )
    assert lines[2:4] == ["line 11", "line 12"]
    assert lines[-1] == "ReplayError: geometry"
    assert len(lines) == 42


def test_cleanup_defaults_must_name_the_chunk_size(tmp_path: Path) -> None:
    pipeline = steps.load(configure(tmp_path))
    (tmp_path / "cleanupDefaults.json").write_text("{}", "utf-8")
    with pytest.raises(ReplayError, match="name no chunkFrames"):
        steps.cleanup_frames(pipeline)


def test_a_step_receipt_records_the_code_snapshot_it_ran_from(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    path = configure(tmp_path)
    monkeypatch.setenv(steps.SNAPSHOT, "4f" * 32)
    receipt = steps.execute("audio-clock", path, run=Producers(tmp_path))
    assert json.loads(receipt.read_text("utf-8"))["codeSnapshot"] == "4f" * 32


def test_a_forced_asr_keeps_the_keyed_chunk_results_and_the_legacy_manifest(
    tmp_path: Path,
) -> None:
    path = configure(tmp_path)
    prepared = tmp_path / "replay-123/asr-v1/prepared"
    (prepared / "context-streams/chunks").mkdir(parents=True)
    (prepared / "asr").mkdir()
    (prepared / "context-streams/chunks/000000000000.json").write_text("{}", encoding="utf-8")
    (prepared / "asr/inputs.json").write_text("{}", encoding="utf-8")
    (prepared / "context-streams/labels.json").write_text("{}", encoding="utf-8")
    (prepared / "mono.wav").write_bytes(b"m")
    (tmp_path / "replay-123/asr-v1/CONFIG-v1.json").write_text("{}", encoding="utf-8")
    paths = [item["path"] for item in steps.reset(steps.load(path), "asr")]
    assert paths == [
        str(tmp_path / "replay-123/asr-v1/CONFIG-v1.json"),
        str(prepared / "mono.wav"),
        str(prepared / "context-streams/labels.json"),
    ]
    assert (prepared / "context-streams/chunks/000000000000.json").exists() is True
    assert (prepared / "asr/inputs.json").exists() is True


def test_the_windowed_asr_reset_removes_its_whole_namespace(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(steps, "ASR_METHOD", "windows")
    assert steps.asr_outputs(tmp_path) == (tmp_path / "asr-v1",)


def test_asr_moves_a_published_manifest_into_its_chunk_store(tmp_path: Path) -> None:
    path = configure(tmp_path)
    clocked(path, tmp_path)
    upstream(tmp_path, "separation")
    prepared = tmp_path / "replay-123/asr-v1/prepared"
    (prepared / "context-streams/chunks").mkdir(parents=True)
    (prepared / "asr").mkdir()
    (prepared / "asr/inputs.json").write_text('{"inputs": {}}', encoding="utf-8")
    steps.execute("asr", path, run=Producers(tmp_path))
    moved = prepared / "context-streams/chunks/previous-inputs.manifest"
    assert moved.read_text(encoding="utf-8") == '{"inputs": {}}'
