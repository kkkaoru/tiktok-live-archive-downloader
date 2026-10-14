"""Synthetic mixed-backend prerequisites and real isolated ASR/assembly contracts."""

from __future__ import annotations

import json
import os
import subprocess
import wave
from dataclasses import dataclass
from pathlib import Path

import pytest

from replay import asr_assembly, workflow
from replay import completion_barrier as barrier
from replay import device_vocal_metal_asr_plan as factory
from replay import isolated_metal_asr as metal
from replay import measured_job_admission as admission
from replay import replay_preparation_plan as initial
from replay import replay_transcription_plan as base
from replay.audio_windows import AudioWindow
from replay.core import ReplayError
from replay.native_invocation import durable, pin, rows
from replay.recordings import record
from replay.stages import Stage, StageError, _fingerprint


def write(path: Path, value: object) -> None:
    path.write_text(json.dumps(value), encoding="utf-8")


def data(path: Path) -> dict[str, object]:
    return record(json.loads(path.read_bytes()))


def ref(path: Path) -> dict[str, object]:
    return {"path": str(path), "sha256": pin(path)["sha256"]}


def pcm(path: Path, *, frames: int, channels: int = 2, rate: int = 44100) -> None:
    with wave.open(str(path), "wb") as audio:
        audio.setparams((channels, 2, rate, 0, "NONE", "not compressed"))
        audio.writeframes(bytes(frames * channels * 2))


def complete(stage: Stage, state: Path) -> Path:
    state.mkdir(exist_ok=True)
    output = state / f"{stage.name}.json"
    write(
        output,
        {
            "status": "complete",
            "fingerprint": _fingerprint(stage),
            "outputs": {str(path): pin(path)["sha256"] for path in stage.outputs},
        },
    )
    return output


@dataclass(frozen=True)
class Fixture:
    root: Path
    config: Path
    work: Path
    device: Path
    receipt: Path
    vocals: Path
    source: Path
    model: Path
    code: Path
    count: int


@dataclass(frozen=True)
class PrerequisiteFixture:
    request: Path
    workflow: Path
    capture: Path
    status: Path
    state: Path
    output: Path
    code: Path


class PrerequisiteHost:
    def __init__(self, *, live: bool = False) -> None:
        self.elapsed = 0.0
        self.live = live

    def now_ns(self) -> int:
        return 42

    def monotonic(self) -> float:
        return self.elapsed

    def sleep(self, seconds: float) -> None:
        self.elapsed += seconds

    def groups(self) -> tuple[tuple[int, int], ...]:
        return ((4, 4),) if self.live else ()

    def release(self, item: barrier.Pin) -> dict[str, object]:
        admission.load_completion_capture(item.path, item.sha256)
        return {"resourceReleased": True, "stageQualityApproved": False}


@pytest.fixture
def fixture(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, request: pytest.FixtureRequest
) -> Fixture:
    seconds = request.param if hasattr(request, "param") else 1
    assert isinstance(seconds, int)
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(factory, "ROOT", tmp_path)
    monkeypatch.setattr(base, "ROOT", tmp_path)
    monkeypatch.setattr(initial, "ROOT", tmp_path)
    (tmp_path / ".python-version").write_text("3.12.13\n", encoding="utf-8")
    (tmp_path / "pyproject.toml").write_bytes(b"[project]\n")
    (tmp_path / "uv.lock").write_bytes(b"version=1\n")
    root = tmp_path / "edits/work"
    root.mkdir(parents=True)
    source = root / "source.mp4"
    source.write_bytes(b"synthetic source with genuine clock roles")
    code = tmp_path / "src/replay/code.py"
    code.parent.mkdir(parents=True)
    code.write_bytes(b"# synthetic frozen code\n")
    monkeypatch.setattr(factory, "_code", lambda: {str(code): str(pin(code)["sha256"])})
    monkeypatch.setattr(metal, "required_code", lambda: (code,))
    model = tmp_path / "edits/models/large-v3"
    model.mkdir(parents=True)
    (model / "config.json").write_bytes(b"synthetic config")
    (model / "weights.npz").write_bytes(b"synthetic nonTurbo weights")
    monkeypatch.setattr(base, "DEFAULT_CONFIG_SHA256", pin(model / "config.json")["sha256"])
    monkeypatch.setattr(base, "DEFAULT_WEIGHTS_SHA256", pin(model / "weights.npz")["sha256"])
    monkeypatch.setattr(metal, "DEFAULT_CONFIG_SHA256", pin(model / "config.json")["sha256"])
    monkeypatch.setattr(metal, "DEFAULT_WEIGHTS_SHA256", pin(model / "weights.npz")["sha256"])
    python = root / "venv/bin/python"
    python.parent.mkdir(parents=True)
    python.write_bytes(b"synthetic pinned Python executable")
    python.chmod(0o700)
    (python.parent.parent / "pyvenv.cfg").write_text("version = 3.12.13\n", encoding="utf-8")
    ffmpeg = root / "ffmpeg"
    ffmpeg.write_bytes(b"synthetic decoder")
    ffmpeg.chmod(0o700)
    count = seconds * 44100 + 1
    stereo = root / "model-input.wav"
    pcm(stereo, frames=count)
    video, audio = root / "video.json", root / "audio.json"
    write(
        video, {"streams": [{"time_base": "1/90000"}], "frames": [{"pts": 90000, "duration": 3000}]}
    )
    write(
        audio,
        {
            "streams": [{"sample_rate": "44100", "time_base": "1/44100"}],
            "frames": [
                {"pts": 44100 + offset, "nb_samples": min(44100, count - offset)}
                for offset in range(0, count, 44100)
            ],
        },
    )
    clock_graph = root / "clocks.json"
    clock_definition = {
        "version": 1,
        "stages": [
            {
                "name": "audio",
                "command": ["synthetic", "audio"],
                "verify_command": ["synthetic", "verify-audio"],
                "inputs": [str(source)],
                "outputs": [str(audio)],
                "timeout_seconds": 30,
            },
            {
                "name": "video",
                "command": ["synthetic", "video"],
                "verify_command": ["synthetic", "verify-video"],
                "inputs": [str(source)],
                "outputs": [str(video)],
                "timeout_seconds": 30,
            },
        ],
    }
    write(clock_graph, clock_definition)
    clock_stages = workflow.parse_stages(clock_definition)
    audio_receipt = complete(clock_stages[0], root / "clock-state")
    video_receipt = complete(clock_stages[1], root / "clock-state")
    old_work = root / "cpu-original"
    old_work.mkdir()
    home = model / "mel"
    hub = home / "hub"
    hub.mkdir(parents=True)
    write(
        old_work / "input.json", {"environment": {"HF_HOME": str(home), "HF_HUB_CACHE": str(hub)}}
    )
    old_plan = root / "original-plan.json"
    write(old_plan, {"outputDirectory": str(old_work)})
    device = root / "device-config.json"
    device_work = root / "device-whole"
    device_work.mkdir()
    write(
        device,
        {
            "kind": "explicit-device-separation-continuation-v1",
            "outputDirectory": str(device_work),
            "originalRecording": ref(source),
            "originalPlan": ref(old_plan),
        },
    )
    vocals = device_work / "vocals.wav"
    pcm(vocals, frames=count)
    write(
        device_work / "result.json",
        {
            "protocol": "explicit-device-separation-continuation-v1",
            "source": str(stereo),
            "modelInput": str(stereo),
            "vocals": str(vocals),
            "preparedInput": True,
            "device": "mixed",
            "cpuExactGateChanged": False,
            "loudnessNormalization": False,
            "timestampCorrection": False,
            "deliveryComplete": False,
        },
    )
    device_graph = root / "device-workflow.json"
    device_definition = {
        "version": 1,
        "stages": [
            {
                "name": "device-whole",
                "command": [
                    str(python),
                    "-B",
                    "-m",
                    "replay.separation_device_pipeline",
                    "assemble",
                    str(device),
                ],
                "verify_command": [
                    str(python),
                    "-B",
                    "-m",
                    "replay.separation_device_pipeline",
                    "verify",
                    str(device),
                ],
                "inputs": [
                    str(source),
                    str(stereo),
                    str(device),
                    str(old_plan),
                    str(old_work / "input.json"),
                ],
                "outputs": [str(vocals), str(device_work / "result.json")],
                "timeout_seconds": 30,
            }
        ],
    }
    write(device_graph, device_definition)
    receipt = complete(workflow.parse_stages(device_definition)[0], root / "device-state")
    config = root / "continuation.json"
    write(
        config,
        {
            "version": 1,
            "deviceWhole": {
                "configuration": ref(device),
                "workflow": ref(device_graph),
                "receiptPath": str(receipt),
                "stageName": "device-whole",
            },
            "transcriptionTemplate": {
                "version": 1,
                "model": str(model),
                "fps": 30,
                "timeouts": {"mono": 30, "recognize": 30, "assemble": 30, "child": 30, "quiet": 30},
                "predecessor": {
                    "roles": {
                        "source": ref(source),
                        "audioClock": ref(audio),
                        "videoClock": ref(video),
                        "modelInput": ref(stereo),
                    },
                    "stages": {
                        "audio": {
                            "workflow": ref(clock_graph),
                            "receipt": ref(audio_receipt),
                            "stageName": "audio",
                        },
                        "video": {
                            "workflow": ref(clock_graph),
                            "receipt": ref(video_receipt),
                            "stageName": "video",
                        },
                    },
                },
                "tools": {"python": ref(python), "ffmpeg": ref(ffmpeg), "env": ref(ffmpeg)},
            },
            "metalCodeInputs": {str(code): pin(code)["sha256"]},
            "timeouts": {"whole": 30, "mono": 30, "window": 30, "recognize": 30, "assemble": 30},
        },
    )
    output = root / "transcription"
    output.mkdir()
    return Fixture(root, config, output, device, receipt, vocals, source, model, code, count)


@pytest.fixture
def prerequisite_fixture(fixture: Fixture, monkeypatch: pytest.MonkeyPatch) -> PrerequisiteFixture:
    monkeypatch.setattr(admission, "ROOT", factory.ROOT)
    state = fixture.root / "macro-state"
    state.mkdir()
    first, second = fixture.root / "first.json", fixture.root / "second.json"
    write(first, {"retained": "first"})
    write(second, {"retained": "second"})
    macro = fixture.root / "macro.json"
    definition = {
        "version": 1,
        "stages": [
            {
                "name": "first",
                "command": ["synthetic-first"],
                "verify_command": ["synthetic-verify"],
                "inputs": [str(fixture.code)],
                "outputs": [str(first)],
                "timeout_seconds": 30,
            },
            {
                "name": "second",
                "command": ["synthetic-ordinary-without-domain-verifier"],
                "inputs": [str(first), str(fixture.code)],
                "outputs": [str(second)],
                "timeout_seconds": 30,
            },
        ],
    }
    write(macro, definition)
    stages = workflow.parse_stages(definition)
    running = state / "first.json"
    write(running, {"status": "running", "fingerprint": _fingerprint(stages[0])})
    observed = ref(running)
    captured = fixture.root / "capture.json"
    write(
        captured,
        {
            "version": 1,
            "kind": "owned-family-completion-capture",
            "observedAtNs": 42,
            "parent": {
                "pid": 4,
                "birth": "123:456",
                "executable": str(fixture.root / "venv/bin/python"),
                "commandSHA256": "0" * 64,
            },
            "processGroups": [4],
            "workflow": ref(macro),
            "stage": "first",
            "checkpoint": str(running),
            "statusPath": str(fixture.root / "macro.status"),
            "initialCheckpointSHA256": observed["sha256"],
            "initialCheckpoint": data(running),
        },
    )
    complete(stages[0], state)
    complete(stages[1], state)
    status = fixture.root / "macro.status"
    status.write_bytes(b"0\n")
    output = fixture.root / "resource-proof.json"
    request = fixture.root / "prerequisite.json"
    write(
        request,
        {
            "version": 1,
            "kind": "resource-only-prerequisite-completion-v1",
            "workflow": ref(macro),
            "capture": ref(captured),
            "stateDirectory": str(state),
            "terminalStatus": str(status),
            "codeInputs": [ref(fixture.code)],
            "waitSeconds": 2,
            "pollSeconds": 1,
            "output": str(output),
        },
    )
    return PrerequisiteFixture(request, macro, captured, status, state, output, fixture.code)


class FakeRuntime:
    def __init__(self, *, fail: bool = False) -> None:
        self.calls = 0
        self.fail = fail

    def execute(
        self, request: metal.Request, window: AudioWindow, work: Path, identities: dict[str, object]
    ) -> None:
        self.calls += 1
        if self.fail:
            raise ReplayError("synthetic uncertain inference")
        assert identities[str(request.pcm)]
        assert window.sample_count > 0
        durable(
            work / "raw.json",
            {
                "language": "ja",
                "text": "声",
                "segments": [
                    {
                        "start": 0.01,
                        "end": 0.08,
                        "text": "声",
                        "words": [{"word": "声", "start": 0.01, "end": 0.08, "probability": 1.0}],
                    }
                ],
            },
        )
        (work / "worker.log").write_bytes(b"")
        durable(
            work / "resources.json",
            {"exitCode": 0, "familyGone": True, "samples": [{"physicalBytes": 42}]},
        )
        durable(
            work / "mlx-memory.json",
            {"device": "metal", "activeBytes": 1, "peakBytes": 2, "cacheBytes": 3},
        )
        durable(work / "before-core-import-descriptors.json", {"externalWritableRegular": []})
        durable(work / "after-core-import-descriptors.json", {"externalWritableRegular": []})
        durable(work / "after-whisper-import-descriptors.json", {"externalWritableRegular": []})
        durable(work / "after-inference-eval-descriptors.json", {"externalWritableRegular": []})
        durable(work / "child.json", {"synthetic": True})
        durable(work / "launch.json", {"synthetic": True})


def execute(command: tuple[str, ...], log: Path, timeout: int, environment: dict[str, str]) -> None:
    assert timeout == 30
    assert environment["HF_HUB_OFFLINE"] == "1"
    assert environment["TRANSFORMERS_OFFLINE"] == "1"
    assert Path(environment["TMPDIR"]).is_relative_to(Path.cwd() / "edits/work")
    log.write_text("synthetic bounded command\n", encoding="utf-8")
    if "replay.separation_device_pipeline" in command:
        assert command[-2] == "verify"
    else:
        assert command[1:6] == ("-nostdin", "-hide_banner", "-v", "error", "-n")
        with wave.open(command[7], "rb") as source:
            frames = round(source.getnframes() * 16000 / 44100)
        pcm(Path(command[-1]), frames=frames, channels=1, rate=16000)


def prefix(fixture: Fixture) -> Path:
    factory.prepare(fixture.config, fixture.work)
    operation = fixture.work / "operation.json"
    factory.run(operation, "whole", execute=execute)
    factory.run(operation, "mono", execute=execute)
    factory.run(operation, "request")
    return operation


def test_plan_derives_real_pins_and_declares_all_consumers(fixture: Fixture) -> None:
    graph = factory.prepare(fixture.config, fixture.work)
    stages = workflow.load_stages(graph)
    assert tuple(stage.name for stage in stages) == (
        "whole",
        "mono",
        "request",
        "recognize",
        "assemble",
    )
    assert workflow.stage_graph(stages)["assemble"] == ("whole", "mono", "request", "recognize")
    assert stages[0].command[3] == "replay.phase_timer"
    assert fixture.root / "PHASES.jsonl" not in stages[0].inputs
    assert not (fixture.work / "mono.wav").exists()
    job = factory._job(fixture.work / "operation.json")
    assert fixture.model / "weights.npz" in {item.path for item in job.inputs}
    assert fixture.model / "config.json" in stages[0].inputs
    assert job.ready.source == fixture.source
    assert job.ready.vocals == fixture.vocals
    assert job.whole_command[3] == "replay.separation_device_pipeline"
    assert factory.verify_plan(fixture.work / "operation.json") == graph


def test_full_decoded_clock_over_small_metadata_budget_reaches_whole(
    fixture: Fixture,
) -> None:
    video = fixture.root / "video.json"
    write(
        video,
        {
            "streams": [{"time_base": "1/90000"}],
            "frames": [
                {
                    "pts": 90000 + index * 3000,
                    "duration": 3000,
                    "best_effort_timestamp": 90000 + index * 3000,
                    "pkt_dts": 90000 + index * 3000,
                }
                for index in range(159148)
            ],
        },
    )
    assert video.stat().st_size > 8388608
    clock_stage = workflow.load_stages(fixture.root / "clocks.json")[1]
    receipt = complete(clock_stage, fixture.root / "clock-state")
    config = data(fixture.config)
    template = record(config["transcriptionTemplate"])
    predecessor = record(template["predecessor"])
    roles = record(predecessor["roles"])
    roles["videoClock"] = ref(video)
    parents = record(predecessor["stages"])
    parent = record(parents["video"])
    parent["receipt"] = ref(receipt)
    parents["video"] = parent
    template["predecessor"] = {"roles": roles, "stages": parents}
    config["transcriptionTemplate"] = template
    write(fixture.config, config)
    factory.prepare(fixture.config, fixture.work)
    operation = fixture.work / "operation.json"
    factory.run(operation, "whole", execute=execute)
    factory.run(operation, "verify-whole")
    assert data(fixture.work / "whole.json")["videoClock"] == ref(video)
    with pytest.raises(ReplayError, match="bounded"):
        factory._metadata(video)


def test_full_labels_use_typed_budget_and_preserve_original_ancestry(
    fixture: Fixture, monkeypatch: pytest.MonkeyPatch
) -> None:
    operation = prefix(fixture)
    factory.run(operation, "recognize", runtime=FakeRuntime())
    original = asr_assembly.assemble

    def large_labels(source: Path, output: Path) -> int:
        count = original(source, output)
        with output.open("ab") as stream:
            stream.write(b" " * 8388608)
        return count

    monkeypatch.setattr(asr_assembly, "assemble", large_labels)
    factory.run(operation, "assemble")
    factory.run(operation, "verify-assemble")
    assert (fixture.work / "labels.json").stat().st_size > 8388608
    assert data(fixture.work / "assemble.json")["sourceProjectionApplications"] == 0
    assert data(fixture.work / "labels.json")["speechAccuracyVerified"] is False


@pytest.mark.parametrize("kind", ["empty", "oversize", "symlink", "hardlink", "directory"])
def test_artifact_stream_refuses_invalid_identity_and_size(tmp_path: Path, kind: str) -> None:
    path = tmp_path / "artifact.json"
    path.write_bytes(b"{}")
    if kind == "empty":
        path.write_bytes(b"")
    elif kind == "oversize":
        with path.open("ab") as stream:
            stream.truncate(134217729)
    elif kind == "symlink":
        alias = tmp_path / "alias.json"
        alias.symlink_to(path)
        path = alias
    elif kind == "hardlink":
        os.link(path, tmp_path / "linked.json")
    else:
        path = tmp_path
    with pytest.raises(ReplayError):
        factory._metadata(path, budget=134217728)


@pytest.mark.parametrize("change", ["grow", "replace", "truncate"])
def test_artifact_stream_rejects_mutation_during_read(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, change: str
) -> None:
    path = tmp_path / "artifact.json"
    path.write_bytes(b"{}\n")
    original = os.read
    changed = False

    def mutating_read(descriptor: int, count: int) -> bytes:
        nonlocal changed
        chunk = original(descriptor, count)
        if chunk and not changed:
            changed = True
            if change == "grow":
                with path.open("ab") as stream:
                    stream.write(b" ")
            elif change == "replace":
                path.unlink()
                path.write_bytes(b"{}\n")
            else:
                path.write_bytes(b"")
        return chunk

    monkeypatch.setattr(os, "read", mutating_read)
    with pytest.raises(ReplayError, match="grew|changed"):
        factory._metadata(path, budget=3)


@pytest.mark.parametrize("budget", [0, 134217729])
def test_artifact_stream_requires_finite_typed_budget(tmp_path: Path, budget: int) -> None:
    path = tmp_path / "artifact.json"
    path.write_bytes(b"{}")
    with pytest.raises(ReplayError, match="byte budget"):
        factory._metadata(path, budget=budget)


@pytest.mark.parametrize("document", [b'{"v":1,"v":2}', b'{"v":NaN}'])
def test_typed_json_budget_keeps_duplicate_and_nonfinite_rejection(
    tmp_path: Path, document: bytes
) -> None:
    path = tmp_path / "artifact.json"
    path.write_bytes(document)
    with pytest.raises(ValueError):
        factory._read(path, budget=134217728)


def test_parent_stage_verifier_rejects_altered_child_commands(fixture: Fixture) -> None:
    graph = factory.prepare(fixture.config, fixture.work)
    saved = data(graph)
    record(rows(saved["stages"])[0])["command"] = ["unbound-substitute"]
    write(graph, saved)
    with pytest.raises(ReplayError, match="child workflow"):
        factory.verify_plan(fixture.work / "operation.json")


def test_cpu_only_backend_and_boolean_operation_version_cannot_be_adopted(fixture: Fixture) -> None:
    factory.prepare(fixture.config, fixture.work)
    operation = fixture.work / "operation.json"
    saved = data(operation)
    saved["version"] = True
    write(operation, saved)
    with pytest.raises(ReplayError, match="Operation/code"):
        factory.run(operation, "whole", execute=execute)
    saved["version"] = 1
    write(operation, saved)
    result = data(fixture.vocals.parent / "result.json")
    result["device"] = "cpu"
    write(fixture.vocals.parent / "result.json", result)
    config = data(fixture.config)
    parent = record(config["deviceWhole"])
    graph = Path(str(record(parent["workflow"])["path"]))
    complete(workflow.load_stages(graph)[0], fixture.receipt.parent)
    fresh = fixture.root / "fresh"
    fresh.mkdir()
    with pytest.raises(ReplayError, match="explicit device"):
        factory.prepare(fixture.config, fresh)


@pytest.mark.parametrize("fixture", [1, 3, 65], indirect=True)
def test_real_metal_collection_assembly_and_single_input_pcm_clock(fixture: Fixture) -> None:
    operation = prefix(fixture)
    runtime = FakeRuntime()
    factory.run(operation, "recognize", runtime=runtime)
    factory.run(operation, "assemble")
    factory.run(operation, "verify-assemble")
    labels = data(fixture.work / "labels.json")
    assert labels["clock"] == "input-pcm-not-projected"
    assert labels["speechAccuracyVerified"] is False
    assert labels["humanReviewed"] is False
    assert labels["deliveryComplete"] is False
    assert record(rows(labels["labels"])[0]) == {
        "text": "声",
        "start": 0.01,
        "end": 0.08,
        "provenance": "000000000000:0",
    }
    assert data(fixture.work / "assemble.json")["sourceProjectionApplications"] == 0
    assert data(fixture.work / "whole.json")["backend"] == "mixed"
    assert runtime.calls >= 1
    with pytest.raises(ReplayError, match="Occupied"):
        factory.run(operation, "assemble")


def test_draft_never_adopts_future_null_outputs(fixture: Fixture) -> None:
    fixture.receipt.unlink()
    draft = factory.draft(fixture.config, fixture.root / "draft.json")
    assert data(draft)["canDispatch"] is False
    assert data(draft)["wholeBarrierCompletionVerified"] is False
    assert "sha256" not in str(data(draft)["futureOutputs"])
    with pytest.raises((OSError, ReplayError)):
        factory.prepare(fixture.config, fixture.work)
    assert tuple(fixture.work.iterdir()) == ()


def test_failed_or_changed_genuine_whole_is_not_adopted(fixture: Fixture) -> None:
    value = data(fixture.receipt)
    value["status"] = "failed"
    write(fixture.receipt, value)
    with pytest.raises((ReplayError, StageError), match="barrier|schema"):
        factory.prepare(fixture.config, fixture.work)
    value["status"] = "complete"
    write(fixture.receipt, value)
    fixture.vocals.write_bytes(fixture.vocals.read_bytes() + b"changed")
    with pytest.raises(ReplayError, match="Predecessor"):
        factory.prepare(fixture.config, fixture.work)


@pytest.mark.parametrize("target", ["source", "model", "code", "config", "operation", "resolved"])
def test_actual_inputs_cannot_change_between_barrier_and_child(
    fixture: Fixture, target: str
) -> None:
    factory.prepare(fixture.config, fixture.work)
    path = {
        "source": fixture.source,
        "model": fixture.model / "weights.npz",
        "code": fixture.code,
        "config": fixture.config,
        "operation": fixture.work / "operation.json",
        "resolved": fixture.work / "transcription-config.json",
    }[target]
    path.write_bytes(path.read_bytes() + b"changed")
    with pytest.raises((ReplayError, StageError, ValueError)):
        factory.run(fixture.work / "operation.json", "whole", execute=execute)


def test_real_completed_windows_resume_but_uncertain_window_never_reissues(
    fixture: Fixture,
) -> None:
    operation = prefix(fixture)
    runtime = FakeRuntime()
    metal.collect(fixture.work / "metal-request.json", fixture.work / "asr", runtime=runtime)
    raw_pin = pin(fixture.work / "asr/000000000000.raw.json")
    factory.run(operation, "recognize", runtime=FakeRuntime(fail=True))
    assert pin(fixture.work / "asr/000000000000.raw.json") == raw_pin
    factory.run(operation, "verify-recognize")
    assert runtime.calls == 1


def test_uncertain_first_window_stays_retained(fixture: Fixture) -> None:
    operation = prefix(fixture)
    with pytest.raises(ReplayError, match="uncertain inference"):
        factory.run(operation, "recognize", runtime=FakeRuntime(fail=True))
    assert (fixture.work / "asr/000000000000.metal-runtime/failed.json").is_file()
    with pytest.raises(ReplayError, match="Uncertain/failed"):
        factory.run(operation, "recognize", runtime=FakeRuntime())


@pytest.mark.parametrize("target", ["labels", "parent", "parser", "clock", "text"])
def test_assembly_preserves_original_words_clocks_and_full_parent_membership(
    fixture: Fixture, target: str
) -> None:
    operation = prefix(fixture)
    factory.run(operation, "recognize", runtime=FakeRuntime())
    factory.run(operation, "assemble")
    labels = data(fixture.work / "labels.json")
    if target == "parent":
        labels["parentSHA256"] = {}
    elif target == "parser":
        labels["parserSHA256"] = "0" * 64
    elif target == "clock":
        labels["clock"] = "source-media"
    elif target == "text":
        labels["labels"] = []
    else:
        labels["speechAccuracyVerified"] = True
    write(fixture.work / "labels.json", labels)
    with pytest.raises(ReplayError):
        factory.run(operation, "verify-assemble")


def test_shortened_mono_and_changed_readonly_phase_evidence_rejected(fixture: Fixture) -> None:
    operation = prefix(fixture)
    pcm(fixture.work / "mono.wav", frames=1598, channels=1, rate=16000)
    with pytest.raises(ReplayError, match="duration"):
        factory.run(operation, "verify-mono")
    pcm(fixture.work / "mono.wav", frames=1600, channels=1, rate=16000)
    write(fixture.work / "whole.json", {"backend": "cpu"})
    with pytest.raises(ReplayError, match="evidence"):
        factory.run(operation, "verify-mono")


def test_cli_requires_explicit_new_destination_and_fails_closed(fixture: Fixture) -> None:
    assert factory.main(["prepare", str(fixture.config)]) == 1
    assert factory.main(["draft", str(fixture.config), str(fixture.root / "draft.json")]) == 0
    assert factory.main(["prepare", str(fixture.config), str(fixture.work)]) == 0
    assert factory.main(["verify-plan", str(fixture.work / "operation.json")]) == 0
    assert factory.main(["verify-whole", str(fixture.work / "operation.json")]) == 1


def test_subprocess_path_has_exact_timeout_and_offline_environment(
    fixture: Fixture, monkeypatch: pytest.MonkeyPatch
) -> None:
    factory.prepare(fixture.config, fixture.work)

    def command(
        argv: tuple[str, ...],
        *,
        stdin: int,
        stdout: object,
        stderr: int,
        check: bool,
        timeout: int,
        env: dict[str, str],
    ) -> subprocess.CompletedProcess[bytes]:
        assert argv[3] == "replay.separation_device_pipeline"
        assert stdin == subprocess.DEVNULL and stderr == subprocess.STDOUT
        assert check is True and timeout == 30
        assert env["HF_HUB_OFFLINE"] == "1"
        return subprocess.CompletedProcess(argv, 0, b"")

    monkeypatch.setattr(subprocess, "run", command)
    factory.run(fixture.work / "operation.json", "whole")
    assert (fixture.work / "whole.log").is_file()


def test_resource_wait_allows_ordinary_macro_without_claiming_domain_quality(
    prerequisite_fixture: PrerequisiteFixture,
) -> None:
    proof = factory.wait_prerequisite(prerequisite_fixture.request, host=PrerequisiteHost())
    assert data(proof)["resourceReleased"] is True
    assert data(proof)["qualityApproved"] is False
    assert data(proof)["domainApproved"] is False
    assert len(rows(data(proof)["ordinaryCheckpoints"])) == 2
    assert (
        factory.verify_prerequisite(prerequisite_fixture.request, host=PrerequisiteHost()) == proof
    )
    with pytest.raises(ReplayError, match="Occupied"):
        factory.wait_prerequisite(prerequisite_fixture.request, host=PrerequisiteHost())


@pytest.mark.parametrize("condition", ["absent-status", "live-actor"])
def test_resource_wait_is_finite_and_never_dispatches_macro(
    prerequisite_fixture: PrerequisiteFixture,
    condition: str,
) -> None:
    if condition == "absent-status":
        prerequisite_fixture.status.unlink()
    host = PrerequisiteHost(live=condition == "live-actor")
    with pytest.raises(ReplayError, match="finite deadline"):
        factory.wait_prerequisite(prerequisite_fixture.request, host=host)
    assert host.elapsed == 2.0
    assert not prerequisite_fixture.output.exists()


@pytest.mark.parametrize(
    "condition", ["failed", "malformed", "running", "wrong-output", "prefix-tamper"]
)
def test_zero_status_cannot_hide_stale_or_changed_original_macro(
    prerequisite_fixture: PrerequisiteFixture,
    condition: str,
) -> None:
    if condition in {"failed", "malformed"}:
        prerequisite_fixture.status.write_bytes(b"1\n" if condition == "failed" else b"0 fake")
    else:
        path = prerequisite_fixture.state / (
            "first.json" if condition == "prefix-tamper" else "second.json"
        )
        saved = data(path)
        if condition == "running":
            saved["status"] = "running"
        elif condition == "wrong-output":
            saved["outputs"] = {}
        else:
            saved["fingerprint"] = "f" * 64
        write(path, saved)
    with pytest.raises(ReplayError):
        factory.wait_prerequisite(prerequisite_fixture.request, host=PrerequisiteHost())
    assert not prerequisite_fixture.output.exists()


@pytest.mark.parametrize(
    "condition", ["code", "capture", "missing-code", "repeated-code", "wrong-state", "deadline"]
)
def test_resource_wait_has_exact_capture_and_finite_code_closure(
    prerequisite_fixture: PrerequisiteFixture,
    condition: str,
) -> None:
    if condition in {"code", "capture"}:
        path = prerequisite_fixture.code if condition == "code" else prerequisite_fixture.capture
        path.write_bytes(path.read_bytes() + b"changed")
    else:
        saved = data(prerequisite_fixture.request)
        if condition == "missing-code":
            saved["codeInputs"] = []
        elif condition == "repeated-code":
            saved["codeInputs"] = [ref(prerequisite_fixture.code), ref(prerequisite_fixture.code)]
        elif condition == "wrong-state":
            saved["stateDirectory"] = str(prerequisite_fixture.request.parent)
        else:
            saved["waitSeconds"] = 28801
        write(prerequisite_fixture.request, saved)
    with pytest.raises(ReplayError):
        factory.wait_prerequisite(prerequisite_fixture.request, host=PrerequisiteHost())


def test_published_resource_proof_cannot_be_changed_or_reused_with_live_actor(
    prerequisite_fixture: PrerequisiteFixture,
) -> None:
    factory.wait_prerequisite(prerequisite_fixture.request, host=PrerequisiteHost())
    with pytest.raises(ReplayError, match="actor is live"):
        factory.verify_prerequisite(prerequisite_fixture.request, host=PrerequisiteHost(live=True))
    saved = data(prerequisite_fixture.output)
    saved["qualityApproved"] = True
    write(prerequisite_fixture.output, saved)
    with pytest.raises(ReplayError, match="proof changed"):
        factory.verify_prerequisite(prerequisite_fixture.request, host=PrerequisiteHost())


def test_prerequisite_cli_checks_only_metadata_before_real_wait(
    prerequisite_fixture: PrerequisiteFixture,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(barrier, "SystemHost", PrerequisiteHost)
    assert factory.main(["check-prerequisite", str(prerequisite_fixture.request)]) == 0
    assert not prerequisite_fixture.output.exists()
    assert factory.main(["wait-prerequisite", str(prerequisite_fixture.request)]) == 0
    assert factory.main(["verify-prerequisite", str(prerequisite_fixture.request)]) == 0
