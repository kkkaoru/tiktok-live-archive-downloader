"""Synthetic whole clocks and real Metal collection/assembly APIs; no MLX inference."""

import json
import subprocess
import wave
from dataclasses import dataclass
from pathlib import Path

import pytest

from replay import isolated_metal_asr as metal
from replay import vocal_metal_asr_pipeline as pipeline
from replay import workflow
from replay.audio_windows import AudioWindow
from replay.core import ReplayError
from replay.native_invocation import durable, pin
from replay.sequence_journal import decode


@dataclass(frozen=True)
class Fixture:
    root: Path
    config: Path
    output: Path
    operation: Path
    graph: Path
    separation: Path
    source: Path
    count: int


def write(path: Path, value: object) -> None:
    path.write_text(json.dumps(value), encoding="utf-8")


def ref(path: Path) -> dict[str, object]:
    value = pin(path)
    return {"path": str(path), "sha256": value["sha256"]}


def pcm(path: Path, *, frames: int, rate: int, channels: int) -> None:
    with wave.open(str(path), "wb") as audio:
        audio.setparams((channels, 2, rate, 0, "NONE", "not compressed"))
        audio.writeframes(bytes(frames * channels * 2))


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
        assert window.sample_count > 0
        assert identities[str(request.pcm)]
        durable(work / "raw.json", {"text": "", "segments": [], "language": "ja"})
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
    assert "MLX_DISABLE_COMPILE" not in environment
    log.write_text("synthetic subprocess verification\n", encoding="utf-8")
    if "replay.separation_window_pipeline" in command:
        assert command[1:4] == ("-B", "-m", "replay.separation_window_pipeline")
        assert command[-1] == "--verify"
        assert environment["TMPDIR"].endswith("separation/tmp")
    else:
        assert command[1:6] == ("-nostdin", "-hide_banner", "-v", "error", "-n")
        with wave.open(command[7], "rb") as source:
            frames = round(source.getnframes() * 16000 / 44100)
        pcm(Path(command[-1]), frames=frames, rate=16000, channels=1)


@pytest.fixture
def fixture(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, request: pytest.FixtureRequest
) -> Fixture:
    seconds = request.param if hasattr(request, "param") else 90
    assert isinstance(seconds, int)
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(pipeline, "ROOT", tmp_path)
    root = tmp_path / "edits/work"
    root.mkdir(parents=True)
    source, separation = root / "source.wav", root / "separation"
    separation.mkdir()
    count = seconds * 44100 + 1
    pcm(source, frames=count, rate=44100, channels=2)
    pcm(separation / "vocals.wav", frames=count, rate=44100, channels=2)
    model = tmp_path / "edits/models/model"
    model.mkdir(parents=True)
    config, weights = model / "config.json", model / "weights.npz"
    config.write_bytes(b"config")
    weights.write_bytes(b"weights")
    monkeypatch.setattr(pipeline, "DEFAULT_CONFIG_SHA256", pin(config)["sha256"])
    monkeypatch.setattr(pipeline, "DEFAULT_WEIGHTS_SHA256", pin(weights)["sha256"])
    monkeypatch.setattr(metal, "DEFAULT_CONFIG_SHA256", pin(config)["sha256"])
    monkeypatch.setattr(metal, "DEFAULT_WEIGHTS_SHA256", pin(weights)["sha256"])
    code = tmp_path / "src/producer.py"
    code.parent.mkdir()
    code.write_text("# synthetic pinned code\n", encoding="utf-8")
    monkeypatch.setattr(pipeline, "_code", lambda: {str(code): pin(code)["sha256"]})
    monkeypatch.setattr(metal, "required_code", lambda: (code,))
    python = root / "mel/bin/python"
    python.parent.mkdir(parents=True)
    python.write_bytes(b"synthetic interpreter")
    python.chmod(0o700)
    (python.parent.parent / "pyvenv.cfg").write_text("version = 3.12.13\n", encoding="utf-8")
    decoder = root / "ffmpeg"
    decoder.write_bytes(b"synthetic decoder")
    decoder.chmod(0o700)
    home, hub = model / "mel", model / "mel/hub"
    hub.mkdir(parents=True)
    (separation / "tmp").mkdir()
    (separation / "cache").mkdir()
    write(
        separation / "input.json",
        {
            "environment": {
                "TMPDIR": str(separation / "tmp"),
                "TMP": str(separation / "tmp"),
                "TEMP": str(separation / "tmp"),
                "XDG_CACHE_HOME": str(separation / "cache"),
                "HF_HOME": str(home),
                "HF_HUB_CACHE": str(hub),
            }
        },
    )
    write(
        separation / "result.json",
        {
            "source": str(source),
            "vocals": str(separation / "vocals.wav"),
            "inputSampleCount": count,
            "outputSampleCount": count,
            "sampleRate": 44100,
            "timestampCorrection": False,
            "deliveryComplete": False,
            "qualityApproved": False,
        },
    )
    plan = root / "original-plan.json"
    write(
        plan,
        {
            "kind": "separation-window-pipeline",
            "sampleRate": 44100,
            "source": ref(source),
            "outputDirectory": str(separation),
            "qualityApproved": False,
            "productionBackendAdopted": False,
            "codeInputs": [ref(code)],
            "providerInputs": [ref(decoder)],
            "modelInputs": [ref(config), ref(weights)],
        },
    )
    path = root / "config.json"
    write(
        path,
        {
            "version": 1,
            "separationPlan": ref(plan),
            "separationPython": ref(python),
            "separationInvocation": str(python),
            "ffmpeg": ref(decoder),
            "model": {"config": ref(config), "weights": ref(weights)},
            "separationCache": {"HF_HOME": str(home), "HF_HUB_CACHE": str(hub)},
            "metalCodeInputs": {str(code): pin(code)["sha256"]},
            "timeouts": {"whole": 30, "mono": 30, "window": 30, "collection": 30},
            "trialWindows": 1,
        },
    )
    output = root / "pipeline"
    output.mkdir()
    graph = pipeline.prepare(path, output)
    return Fixture(root, path, output, output / "operation.json", graph, separation, source, count)


def test_prepare_is_metadata_only_and_full_target_excludes_trial(fixture: Fixture) -> None:
    graph = workflow.load_stages(fixture.graph)
    mapping = workflow.stage_graph(graph)
    assert mapping["assemble"] == ("request", "recognize")
    assert mapping["recognize"] == ("mono", "request")
    assert not (fixture.output / "mono.wav").exists()
    assert not (fixture.output / "asr").exists()
    assert not (fixture.output / "whole.json").exists()
    whole = next(stage for stage in graph if stage.name == "whole")
    assert fixture.root / "ffmpeg" in whole.inputs
    assert fixture.root.parent / "models/model/weights.npz" in whole.inputs


@pytest.mark.parametrize("fixture", [1, 90, 120], indirect=True)
def test_original_whole_mono_genuine_metal_and_normal_assembly(fixture: Fixture) -> None:
    pipeline.run(fixture.operation, phase="whole", execute=execute)
    pipeline.run(fixture.operation, phase="verify-whole")
    pipeline.run(fixture.operation, phase="mono", execute=execute)
    pipeline.run(fixture.operation, phase="verify-mono")
    pipeline.run(fixture.operation, phase="request")
    pipeline.run(fixture.operation, phase="verify-request")
    runtime = FakeRuntime()
    pipeline.run(fixture.operation, phase="recognize", runtime=runtime)
    pipeline.run(fixture.operation, phase="verify-recognize")
    before = pin(fixture.output / "asr/complete.json")
    pipeline.run(fixture.operation, phase="assemble")
    pipeline.run(fixture.operation, phase="verify-assemble")
    assert pin(fixture.output / "asr/complete.json") == before
    assert (
        json.loads((fixture.output / "assembly.json").read_text(encoding="utf-8"))[
            "normalAssemblyConnected"
        ]
        is True
    )
    assert json.loads((fixture.output / "labels.json").read_text(encoding="utf-8"))["labels"] == []
    assert not (fixture.output / "policy").exists()
    assert not (fixture.output / "started.json").exists()
    assert (
        json.loads((fixture.output / "mono.json").read_text(encoding="utf-8"))["clock"]
        == "input-pcm-not-projected"
    )


def test_partial_trial_full_resume_reuses_actual_window(fixture: Fixture) -> None:
    pipeline.run(fixture.operation, phase="whole", execute=execute)
    pipeline.run(fixture.operation, phase="mono", execute=execute)
    pipeline.run(fixture.operation, phase="request")
    runtime = FakeRuntime()
    pipeline.run(fixture.operation, phase="trial", runtime=runtime)
    pipeline.run(fixture.operation, phase="verify-trial")
    assert runtime.calls == 1
    first = pin(fixture.output / "asr/000000000000.raw.json")
    assert not (fixture.output / "asr/complete.json").exists()
    with pytest.raises(FileNotFoundError):
        pipeline.run(fixture.operation, phase="assemble")
    pipeline.run(fixture.operation, phase="recognize", runtime=runtime)
    assert runtime.calls == 2
    assert pin(fixture.output / "asr/000000000000.raw.json") == first
    pipeline.run(fixture.operation, phase="verify-trial")
    pipeline.run(fixture.operation, phase="verify-recognize")
    pipeline.run(fixture.operation, phase="assemble")


def test_failed_window_is_preserved_and_not_resubmitted(fixture: Fixture) -> None:
    pipeline.run(fixture.operation, phase="whole", execute=execute)
    pipeline.run(fixture.operation, phase="mono", execute=execute)
    pipeline.run(fixture.operation, phase="request")
    runtime = FakeRuntime(fail=True)
    with pytest.raises(ReplayError, match="uncertain"):
        pipeline.run(fixture.operation, phase="recognize", runtime=runtime)
    with pytest.raises(ReplayError, match="no retry"):
        pipeline.run(fixture.operation, phase="recognize", runtime=runtime)
    assert runtime.calls == 1
    assert not (fixture.output / "recognition.json").exists()


@pytest.mark.parametrize(
    "target",
    [
        "source.wav",
        "separation/vocals.wav",
        "separation/result.json",
        "original-plan.json",
        "config.json",
    ],
)
def test_changed_whole_or_configuration_is_not_adopted(fixture: Fixture, target: str) -> None:
    pipeline.run(fixture.operation, phase="whole", execute=execute)
    (fixture.root / target).write_bytes(b"changed")
    with pytest.raises((ReplayError, ValueError, wave.Error, EOFError)):
        pipeline.run(fixture.operation, phase="verify-whole")
    assert not (fixture.output / "mono.wav").exists()


def test_uncertain_mono_and_wrong_sample_duration_are_rejected(fixture: Fixture) -> None:
    pipeline.run(fixture.operation, phase="whole", execute=execute)
    pcm(fixture.output / "mono.wav", frames=42, rate=16000, channels=1)
    with pytest.raises(ReplayError, match="Occupied/uncertain"):
        pipeline.run(fixture.operation, phase="mono", execute=execute)
    with pytest.raises(ReplayError, match="sample duration"):
        pipeline._mono(pipeline._load(fixture.operation))


@pytest.mark.parametrize(
    "target",
    [
        "asr/000000000000.raw.json",
        "asr/000000000000.clock.json",
        "asr/000000000000.metal-runtime/resources.json",
        "asr/inputs.json",
        "trial.json",
    ],
)
def test_trial_raw_and_native_evidence_tamper_rejected(fixture: Fixture, target: str) -> None:
    pipeline.run(fixture.operation, phase="whole", execute=execute)
    pipeline.run(fixture.operation, phase="mono", execute=execute)
    pipeline.run(fixture.operation, phase="request")
    pipeline.run(fixture.operation, phase="trial", runtime=FakeRuntime())
    (fixture.output / target).write_text("{}", encoding="utf-8")
    with pytest.raises((ReplayError, KeyError)):
        pipeline.run(fixture.operation, phase="verify-trial")


def test_mono_and_metal_request_tamper_rejected(fixture: Fixture) -> None:
    pipeline.run(fixture.operation, phase="whole", execute=execute)
    pipeline.run(fixture.operation, phase="mono", execute=execute)
    pipeline.run(fixture.operation, phase="request")
    (fixture.output / "metal-request.json").write_text("{}", encoding="utf-8")
    with pytest.raises(ReplayError, match="Dynamic Metal"):
        pipeline.run(fixture.operation, phase="verify-request")


def test_original_environment_is_preserved_and_escape_rejected(fixture: Fixture) -> None:
    context = decode((fixture.separation / "input.json").read_bytes())
    context["environment"] = {"TMPDIR": "/tmp"}
    write(fixture.separation / "input.json", context)
    with pytest.raises(ReplayError, match="original separation environment"):
        pipeline.run(fixture.operation, phase="whole", execute=execute)
    assert not (fixture.output / "whole-verifier.log").exists()


def test_cli_phase_and_preparation_errors(
    fixture: Fixture, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.setattr(pipeline, "run", lambda operation, *, phase: fixture.output / "result.json")
    assert pipeline.main(["recognize", str(fixture.operation)]) == 0
    assert "result.json" in capsys.readouterr().out
    with pytest.raises(ReplayError, match="empty purpose"):
        pipeline.main(["prepare", str(fixture.config)])
    with pytest.raises(ReplayError, match="immutable configured"):
        pipeline.main(["recognize", str(fixture.operation), "--output", str(fixture.output)])


def test_executor_uses_owned_environment_and_finite_deadline(
    fixture: Fixture, monkeypatch: pytest.MonkeyPatch
) -> None:
    observed: list[object] = []

    def subprocess_run(command: tuple[str, ...], **kwargs: object) -> None:
        observed.append(kwargs["timeout"])
        observed.append(kwargs["env"])

    monkeypatch.setattr(subprocess, "run", subprocess_run)
    pipeline._execute(
        ("fake",), fixture.output / "executor.log", 12, {"TMPDIR": str(fixture.output / "tmp")}
    )
    assert observed[0] == 12
    assert (fixture.output / "executor.log").stat().st_mode & 0o777 == 0o600


def test_original_pin_roles_and_unknown_phases_are_rejected(fixture: Fixture) -> None:
    with pytest.raises(ReplayError, match="Unknown verification"):
        pipeline.run(fixture.operation, phase="verify-invalid")
    with pytest.raises(ReplayError, match="Unknown vocal"):
        pipeline.run(fixture.operation, phase="invalid")
    plan = decode((fixture.root / "original-plan.json").read_bytes())
    plan["providerInputs"] = plan["codeInputs"]
    write(fixture.root / "original-plan.json", plan)
    with pytest.raises(ReplayError, match="Invalid original pin role"):
        pipeline._original_inputs(fixture.root / "original-plan.json")


def test_truncated_continuous_pcm_is_rejected(fixture: Fixture) -> None:
    with (fixture.separation / "vocals.wav").open("r+b") as stream:
        stream.truncate(44 + 4)
    with pytest.raises(ReplayError, match="Truncated PCM"):
        pipeline.run(fixture.operation, phase="whole", execute=execute)
    assert not (fixture.output / "whole.json").exists()
