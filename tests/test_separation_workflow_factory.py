"""Synthetic factory/Stage integration; no model or private replay execution."""

from __future__ import annotations

import hashlib
import json
import stat
import subprocess
import wave
from dataclasses import dataclass
from pathlib import Path

import pytest

from replay import separation_pipeline_prepare as bridge
from replay import separation_workflow_factory as factory
from replay import workflow
from replay.bgm_separation import MODEL_NAME, MODEL_REVISION
from replay.code_dependency_inventory import Source
from replay.core import ReplayError
from replay.native_invocation import durable, pin, read
from replay.recordings import record
from replay.stages import StageError, file_digest, run_stage


@dataclass(frozen=True)
class Fixture:
    request: Path
    config: Path
    template: Path
    directory: Path
    source: Path


@pytest.fixture
def files(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Fixture:
    monkeypatch.chdir(tmp_path)
    work = tmp_path / "edits/work"
    directory = work / "prepared"
    directory.mkdir(parents=True)
    runtime = work / "runtime"
    (runtime / "bin").mkdir(parents=True)
    python = runtime / "bin/python"
    python.write_bytes(b"synthetic Python\n")
    cfg = runtime / "pyvenv.cfg"
    cfg.write_bytes(b"synthetic Python configuration\n")
    code, source, ffmpeg = work / "code.py", work / "source.bin", work / "ffmpeg"
    code.write_bytes(b"synthetic pinned code\n")
    source.write_bytes(b"synthetic media\n")
    ffmpeg.write_bytes(b"synthetic ffmpeg\n")
    code_hash = file_digest(code)
    info = code.stat()
    monkeypatch.setattr(bridge, "code_inputs", lambda: {code: code_hash})
    monkeypatch.setattr(factory, "code_inputs", lambda: {str(code): code_hash})
    monkeypatch.setattr(
        factory,
        "capture",
        lambda modules: (
            Source(
                code,
                code_hash,
                (info.st_dev, info.st_ino, info.st_size, info.st_mtime_ns, info.st_ctime_ns),
            ),
        ),
    )
    preparation = work / "preparation.json"
    durable(preparation, {"version": 1})
    clock = work / "audio-clock.json"
    durable(
        clock,
        {
            "streams": [{"sample_rate": "48000", "time_base": "1/48000"}],
            "frames": [{"pts": 0, "nb_samples": 48000}],
        },
    )
    producer = work / "audio-clock.producer.json"
    durable(
        producer,
        {
            "version": 1,
            "protocol": "replay-preparation-v1",
            "kind": "audio",
            "sharedProducerCompleted": True,
            "candidateOnly": True,
            "deliveryComplete": False,
            "preparationSHA256": file_digest(preparation),
            "source": {"path": str(source), "sha256": file_digest(source)},
            "outputs": {str(clock): file_digest(clock)},
        },
    )
    config = work / "config.json"
    durable(
        config,
        {
            "version": 1,
            "kind": "separation-pcm-preparation",
            "source": {"path": str(source), "sha256": file_digest(source)},
            "mediaRoot": str(work),
            "preparation": {"path": str(preparation), "sha256": file_digest(preparation)},
            "audioClock": {"path": str(clock), "sha256": file_digest(clock)},
            "audioClockProducer": {"path": str(producer), "sha256": file_digest(producer)},
            "ffmpeg": {"path": str(ffmpeg), "sha256": file_digest(ffmpeg)},
            "directory": str(directory),
            "gainDB": -0.1,
            "codeInputs": [{"path": str(code), "sha256": code_hash}],
            "completedExtraction": None,
        },
    )
    provider = work / "provider.so"
    snapshot = work / ("models--" + MODEL_NAME.replace("/", "--")) / "snapshots" / MODEL_REVISION
    snapshot.mkdir(parents=True)
    model, model_config = snapshot / "model.safetensors", snapshot / "config.json"
    provider.write_bytes(b"synthetic provider\n")
    model.write_bytes(b"synthetic model\n")
    model_config.write_bytes(b"{}\n")
    monkeypatch.setattr(factory, "MODEL_SHA256", file_digest(model))
    monkeypatch.setattr(factory, "MODEL_CONFIG_SHA256", file_digest(model_config))
    monkeypatch.setattr(factory, "_required_provider_paths", lambda: {str(provider)})
    template = work / "template.json"
    durable(
        template,
        {
            "version": 1,
            "kind": "separation-window-pipeline",
            "source": {"path": str(directory / "model-input.wav"), "sha256": None},
            "outputDirectory": str(directory / "pipeline"),
            "budget": {
                "window": {
                    "workers": 2,
                    "max_samples": 26460000,
                    "timeout_seconds": 3600,
                    "max_worker_rss_bytes": 5905580032,
                    "max_saved_bytes": 1073741824,
                },
                "max_samples": 44101,
                "max_saved_bytes": 1073741824,
                "max_parent_rss_bytes": 2147483648,
                "timeout_seconds": 21600,
            },
            "codeInputs": [{"path": str(code), "sha256": code_hash}],
            "providerInputs": [{"path": str(provider), "sha256": file_digest(provider)}],
            "modelInputs": [
                {"path": str(model), "sha256": file_digest(model)},
                {"path": str(model_config), "sha256": file_digest(model_config)},
            ],
            "chunkSamples": 26460000,
            "overlapSamples": 220500,
            "sampleRate": 44100,
            "qualityApproved": False,
            "productionBackendAdopted": False,
        },
    )
    request = work / "request.json"
    durable(
        request,
        {
            "version": 1,
            "kind": "separation-workflow-request",
            "config": {"path": str(config), "sha256": file_digest(config)},
            "template": {"path": str(template), "sha256": file_digest(template)},
            "mode": "extract",
            "python": str(python),
            "runtimeInputs": [
                {"path": str(python), "sha256": file_digest(python)},
                {"path": str(cfg), "sha256": file_digest(cfg)},
                {"path": "/usr/bin/env", "sha256": "a" * 64},
            ],
            "environment": {
                "TMPDIR": str(work),
                "TMP": str(work),
                "TEMP": str(work),
                "XDG_CACHE_HOME": str(work),
                "HF_HOME": str(work),
                "HF_HUB_CACHE": str(work),
            },
            "phases": str(directory / "phases.jsonl"),
            "planOutput": str(directory / "plan.json"),
        },
    )
    monkeypatch.setattr(
        bridge, "expected_pcm_digest", lambda config: hashlib.sha256(bytes(44100 * 4)).hexdigest()
    )

    def ffmpeg_run(command: list[str], **options: object) -> subprocess.CompletedProcess[bytes]:
        assert command[-3:-1] == ["-c:a", "pcm_s16le"]
        with wave.open(str(directory / "model-input.wav"), "wb") as audio:
            audio.setnchannels(2)
            audio.setsampwidth(2)
            audio.setframerate(44100)
            audio.writeframes(bytes(44100 * 4))
        return subprocess.CompletedProcess(command, 0)

    monkeypatch.setattr(subprocess, "run", ffmpeg_run)
    return Fixture(request, config, template, directory, source)


def test_extract_graph_and_private_completion_with_empty_stderr(files: Fixture) -> None:
    output = files.directory / "workflow.json"
    factory.prepare(files.request, output)
    assert stat.S_IMODE(output.stat().st_mode) == 0o600
    stages = workflow.load_stages(output)
    assert workflow.stage_graph(stages) == {
        "extract-and-verify-model-pcm": (),
        "prepare-two-worker-plan": ("extract-and-verify-model-pcm",),
        "two-worker-original-window-pipeline": (
            "extract-and-verify-model-pcm",
            "prepare-two-worker-plan",
        ),
    }
    assert (files.directory / "extract.stderr") not in stages[0].outputs
    assert (files.directory / "extract.stderr") not in stages[1].inputs
    state = files.directory / "state"
    state.mkdir()

    def execute(command: tuple[str, ...], log: Path, timeout: int) -> None:
        bridge.extract(files.config) if "extract" in command else bridge.verify(files.config)
        log.write_text("synthetic domain succeeded\n", encoding="utf-8")

    assert run_stage(stages[0], state_directory=state, execute=execute) == "completed"
    assert (files.directory / "extract.stderr").stat().st_size == 0
    assert read(state / "extract-and-verify-model-pcm.json")["status"] == "complete"
    before = pin(output)
    assert factory.prepare(files.request, output, verify_only=True) == output
    assert pin(output) == before
    assert workflow.main([str(output), "--state", str(state), "--check"]) == 0


def test_completed_extraction_adoption_preserves_failed_checkpoint(files: Fixture) -> None:
    bridge.extract(files.config)
    request = read(files.request)
    request["mode"] = "adopt-existing"
    files.request.write_text(json.dumps(request), encoding="utf-8")
    failed = files.directory / "old-state.json"
    failed.write_text('{"status":"running","fingerprint":"old-failure"}\n', encoding="utf-8")
    old = failed.read_bytes()
    output = files.directory / "adoption.json"
    factory.prepare(files.request, output)
    stage = workflow.load_stages(output)[0]
    assert stage.adopt_existing is True
    assert "extract" not in stage.command
    assert "verify" in stage.command
    state = files.directory / "new-state"
    state.mkdir()

    def verify_only(command: tuple[str, ...], log: Path, timeout: int) -> None:
        assert "extract" not in command
        bridge.verify(files.config)
        log.write_text("verified without extraction\n", encoding="utf-8")

    assert run_stage(stage, state_directory=state, execute=verify_only) == "adopted"
    assert failed.read_bytes() == old
    assert read(state / "verify-and-adopt-model-pcm.json")["status"] == "complete"
    assert bridge.verify(files.config)["inputSampleCount"] == 44100


def test_changed_source_fails_actual_adoption_without_new_extract(files: Fixture) -> None:
    bridge.extract(files.config)
    request = read(files.request)
    request["mode"] = "adopt-existing"
    files.request.write_text(json.dumps(request), encoding="utf-8")
    output = files.directory / "adoption.json"
    factory.prepare(files.request, output)
    stage = workflow.load_stages(output)[0]
    before = pin(files.directory / "model-input.wav")
    files.source.write_bytes(b"changed synthetic source\n")
    state = files.directory / "state"
    state.mkdir()

    def verify_only(command: tuple[str, ...], log: Path, timeout: int) -> None:
        bridge.verify(files.config)

    with pytest.raises(StageError, match="SHA256"):
        run_stage(stage, state_directory=state, execute=verify_only)
    assert pin(files.directory / "model-input.wav") == before
    assert read(state / "verify-and-adopt-model-pcm.json")["status"] == "running"


@pytest.mark.parametrize("leaf", ["extraction-attempt.json", "extract.stderr", "model-input.wav"])
def test_uncertain_extraction_refuses_generation(files: Fixture, leaf: str) -> None:
    (files.directory / leaf).write_bytes(b"uncertain\n")
    with pytest.raises(ReplayError, match="already attempted"):
        factory.prepare(files.request, files.directory / "workflow.json")
    assert not (files.directory / "workflow.json").exists()


def test_empty_pcm_is_not_adoptable(files: Fixture) -> None:
    bridge.extract(files.config)
    (files.directory / "model-input.wav").write_bytes(b"")
    request = read(files.request)
    request["mode"] = "adopt-existing"
    files.request.write_text(json.dumps(request), encoding="utf-8")
    with pytest.raises(ReplayError, match="Incomplete/empty"):
        factory.prepare(files.request, files.directory / "workflow.json")


def test_zero_sample_pcm_is_rejected(files: Fixture) -> None:
    bridge.extract(files.config)
    with wave.open(str(files.directory / "model-input.wav"), "wb") as audio:
        audio.setnchannels(2)
        audio.setsampwidth(2)
        audio.setframerate(44100)
    request = read(files.request)
    request["mode"] = "adopt-existing"
    files.request.write_text(json.dumps(request), encoding="utf-8")
    with pytest.raises(ReplayError, match="PCM clock"):
        factory.prepare(files.request, files.directory / "workflow.json")


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("version", True),
        ("kind", "other"),
        ("mode", "retry"),
        ("python", None),
        ("runtimeInputs", []),
        ("environment", {}),
    ],
)
def test_invalid_request(files: Fixture, field: str, value: object) -> None:
    data = read(files.request)
    data[field] = value
    files.request.write_text(json.dumps(data), encoding="utf-8")
    with pytest.raises(ReplayError):
        factory.prepare(files.request, files.directory / "workflow.json")


@pytest.mark.parametrize(
    ("field", "value"),
    [("qualityApproved", True), ("sampleRate", 48000), ("codeInputs", []), ("chunkSamples", 1)],
)
def test_changed_template_contract(files: Fixture, field: str, value: object) -> None:
    data = read(files.template)
    data[field] = value
    files.template.write_text(json.dumps(data), encoding="utf-8")
    request = read(files.request)
    record(request["template"])["sha256"] = file_digest(files.template)
    files.request.write_text(json.dumps(request), encoding="utf-8")
    with pytest.raises(ReplayError):
        factory.prepare(files.request, files.directory / "workflow.json")


def test_metadata_hash_mismatch(files: Fixture) -> None:
    files.template.write_text("{}\n", encoding="utf-8")
    with pytest.raises(ReplayError, match="SHA256 changed"):
        factory.prepare(files.request, files.directory / "workflow.json")


def test_permissions_finalized_before_hashing(files: Fixture) -> None:
    files.config.chmod(0o644)
    with pytest.raises(ReplayError, match="finalized"):
        factory.prepare(files.request, files.directory / "workflow.json")
    assert stat.S_IMODE(files.config.stat().st_mode) == 0o644


@pytest.mark.parametrize("field", ["planOutput", "phases"])
def test_future_occupied_output(files: Fixture, field: str) -> None:
    Path(str(read(files.request)[field])).write_bytes(b"occupied\n")
    with pytest.raises(ReplayError, match="occupied"):
        factory.prepare(files.request, files.directory / "workflow.json")


def test_changed_graph_readonly_rejects_without_repair(files: Fixture) -> None:
    output = files.directory / "workflow.json"
    factory.prepare(files.request, output)
    output.write_text("{}\n", encoding="utf-8")
    before = output.read_bytes()
    with pytest.raises(ReplayError, match="workflow changed"):
        factory.prepare(files.request, output, verify_only=True)
    assert output.read_bytes() == before


def test_no_graph_overwrite_and_cli(files: Fixture) -> None:
    output = files.directory / "workflow.json"
    assert factory.main([str(files.request), str(output)]) == 0
    assert factory.main([str(files.request), str(output), "--verify"]) == 0
    before = pin(output)
    assert factory.main([str(files.request), str(output)]) == 1
    assert pin(output) == before


def test_factory_does_not_decode_or_hash_model(
    files: Fixture, monkeypatch: pytest.MonkeyPatch
) -> None:
    def forbid(*arguments: object, **options: object) -> None:
        raise AssertionError("Factory must not process media")

    monkeypatch.setattr(subprocess, "run", forbid)
    monkeypatch.setattr(bridge, "expected_pcm_digest", forbid)
    output = files.directory / "workflow.json"
    factory.prepare(files.request, output)
    assert workflow.stage_graph(workflow.load_stages(output))[
        "two-worker-original-window-pipeline"
    ] == ("extract-and-verify-model-pcm", "prepare-two-worker-plan")


def test_modeled_provider_is_required(files: Fixture, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(factory, "_required_provider_paths", lambda: {"/missing/provider.so"})
    with pytest.raises(ReplayError, match="provider code/library"):
        factory.prepare(files.request, files.directory / "workflow.json")
    assert not (files.directory / "workflow.json").exists()


def test_model_cache_mapping_is_pinned(files: Fixture, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(factory, "MODEL_SHA256", "0" * 64)
    with pytest.raises(ReplayError, match="model/cache pins changed"):
        factory.prepare(files.request, files.directory / "workflow.json")


def test_empty_diagnostic_cannot_be_a_required_runtime_input(files: Fixture) -> None:
    diagnostic = files.directory / "conversion.log"
    diagnostic.write_bytes(b"")
    data = read(files.request)
    runtime = data["runtimeInputs"]
    assert isinstance(runtime, list)
    data["runtimeInputs"] = runtime + [{"path": str(diagnostic), "sha256": "0" * 64}]
    files.request.write_text(json.dumps(data), encoding="utf-8")
    with pytest.raises(ReplayError, match="Empty factory input"):
        factory.prepare(files.request, files.directory / "workflow.json")
    assert not (files.directory / "workflow.json").exists()


def test_receipt_identity_change_refuses_adoption(files: Fixture) -> None:
    bridge.extract(files.config)
    request = read(files.request)
    request["mode"] = "adopt-existing"
    files.request.write_text(json.dumps(request), encoding="utf-8")
    pcm = files.directory / "model-input.wav"
    pcm.write_bytes(pcm.read_bytes())
    with pytest.raises(ReplayError, match="complete prepared PCM receipt"):
        factory.prepare(files.request, files.directory / "workflow.json")


def test_phase_log_cannot_mutate_receipt(files: Fixture) -> None:
    request = read(files.request)
    request["phases"] = str(files.directory / "pcm-preparation.json")
    files.request.write_text(json.dumps(request), encoding="utf-8")
    with pytest.raises(ReplayError, match="Phase log must not mutate"):
        factory.prepare(files.request, files.directory / "workflow.json")


def test_generation_input_change_is_rejected(
    files: Fixture, monkeypatch: pytest.MonkeyPatch
) -> None:
    code = files.config.parent / "code.py"
    info = code.stat()
    source = Source(
        code,
        file_digest(code),
        (info.st_dev, info.st_ino, info.st_size, info.st_mtime_ns, info.st_ctime_ns),
    )

    def mutate(modules: tuple[str, ...]) -> tuple[Source, ...]:
        files.config.write_text("{}\n", encoding="utf-8")
        return (source,)

    monkeypatch.setattr(factory, "capture", mutate)
    with pytest.raises(ReplayError, match="inputs changed during generation"):
        factory.prepare(files.request, files.directory / "workflow.json")
    assert not (files.directory / "workflow.json").exists()


def _set_template(files: Fixture, field: str, value: object) -> None:
    data = read(files.template)
    data[field] = value
    files.template.write_text(json.dumps(data), encoding="utf-8")
    request = read(files.request)
    record(request["template"])["sha256"] = file_digest(files.template)
    files.request.write_text(json.dumps(request), encoding="utf-8")


def test_a_gpu_template_is_accepted(files: Fixture) -> None:
    _set_template(files, "device", "mps")
    assert factory.prepare(files.request, files.directory / "workflow.json").is_file()


def test_an_unknown_template_device_is_refused(files: Fixture) -> None:
    _set_template(files, "device", "cuda")
    with pytest.raises(ReplayError, match="Unsupported original pipeline template"):
        factory.prepare(files.request, files.directory / "workflow.json")
