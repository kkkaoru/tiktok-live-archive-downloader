"""Synthetic PCM/source-clock bridge tests; no real model or media invocation."""

from __future__ import annotations

import hashlib
import json
import subprocess
import wave
from pathlib import Path

import pytest

from replay import separation_pipeline_prepare as bridge
from replay import separation_window_pipeline as pipeline
from replay.bgm_separation import MODEL_NAME, MODEL_REVISION
from replay.core import ReplayError
from replay.native_invocation import durable, pin, read
from replay.recordings import record
from replay.stages import StageError, file_digest


@pytest.fixture
def prepared(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    monkeypatch.chdir(tmp_path)
    work = tmp_path / "edits/work"
    work.mkdir(parents=True)
    media = tmp_path / "media"
    media.mkdir()
    directory = work / "prepared"
    directory.mkdir()
    code, source, ffmpeg = work / "code.py", media / "source.bin", work / "ffmpeg"
    code.write_bytes(b"synthetic code\n")
    source.write_bytes(b"synthetic source\n")
    ffmpeg.write_bytes(b"synthetic ffmpeg\n")
    preparation = work / "preparation.json"
    durable(preparation, {"version": 1})
    clock = work / "clock.json"
    durable(
        clock,
        {
            "streams": [{"sample_rate": "48000", "time_base": "1/48000"}],
            "frames": [{"pts": 12000, "nb_samples": 48000}],
        },
    )
    producer = work / "clock.producer.json"
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
    (directory / "status").write_text("0\n", encoding="utf-8")
    (directory / "phases.jsonl").write_text(
        '{"phase":"extract","start":1,"end":2,"seconds":1,"exit":0}\n', encoding="utf-8"
    )
    config = work / "config.json"
    durable(
        config,
        {
            "version": 1,
            "kind": "separation-pcm-preparation",
            "source": {"path": str(source), "sha256": file_digest(source)},
            "mediaRoot": str(media),
            "preparation": {"path": str(preparation), "sha256": file_digest(preparation)},
            "audioClock": {"path": str(clock), "sha256": file_digest(clock)},
            "audioClockProducer": {"path": str(producer), "sha256": file_digest(producer)},
            "ffmpeg": {"path": str(ffmpeg), "sha256": file_digest(ffmpeg)},
            "directory": str(directory),
            "gainDB": -0.1,
            "codeInputs": [{"path": str(code), "sha256": file_digest(code)}],
            "completedExtraction": {
                "status": {
                    "path": str(directory / "status"),
                    "sha256": file_digest(directory / "status"),
                },
                "phases": {
                    "path": str(directory / "phases.jsonl"),
                    "sha256": file_digest(directory / "phases.jsonl"),
                },
                "phase": "extract",
            },
        },
    )
    monkeypatch.setattr(bridge, "code_inputs", lambda: {code: file_digest(code)})
    monkeypatch.setattr(
        bridge, "expected_pcm_digest", lambda config: hashlib.sha256(bytes(44100 * 4)).hexdigest()
    )
    with wave.open(str(directory / "model-input.wav"), "wb") as audio:
        audio.setnchannels(2)
        audio.setsampwidth(2)
        audio.setframerate(44100)
        audio.writeframes(bytes(44100 * 4))
    return config


def test_seal_verify_and_readonly(prepared: Path) -> None:
    receipt = bridge.seal(prepared)
    before = pin(receipt)
    result = bridge.verify(prepared)
    assert result["inputSampleCount"] == 44100
    assert result["outputSampleCount"] == 44100
    assert result["sourceDecodedSamples"] == 48000
    assert result["sourceProjectionApplications"] == 0
    assert result["qualityApproved"] is False
    assert result["productionBackendAdopted"] is False
    assert pin(receipt) == before
    assert bridge.main(["verify", str(prepared)]) == 0
    with pytest.raises(ReplayError, match="occupied"):
        bridge.seal(prepared)


@pytest.mark.parametrize("samples", [44099, 44101])
def test_single_output_sample_tolerance(
    prepared: Path, samples: int, monkeypatch: pytest.MonkeyPatch
) -> None:
    target = Path(str(read(prepared)["directory"])) / "model-input.wav"
    with wave.open(str(target), "wb") as audio:
        audio.setnchannels(2)
        audio.setsampwidth(2)
        audio.setframerate(44100)
        audio.writeframes(bytes(samples * 4))
    monkeypatch.setattr(
        bridge, "expected_pcm_digest", lambda config: hashlib.sha256(bytes(samples * 4)).hexdigest()
    )
    bridge.seal(prepared)
    assert bridge.verify(prepared)["sampleRate"] == 44100


@pytest.mark.parametrize("samples", [44098, 44102])
def test_clock_mismatch(prepared: Path, samples: int) -> None:
    target = Path(str(read(prepared)["directory"])) / "model-input.wav"
    with wave.open(str(target), "wb") as audio:
        audio.setnchannels(2)
        audio.setsampwidth(2)
        audio.setframerate(44100)
        audio.writeframes(bytes(samples * 4))
    with pytest.raises(ReplayError, match="one output sample"):
        bridge.seal(prepared)
    assert not (target.parent / "pcm-preparation.json").exists()


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("version", True),
        ("kind", "unknown"),
        ("gainDB", -0.2),
        ("gainDB", 0),
        ("codeInputs", []),
        ("codeInputs", None),
        ("completedExtraction", {"phase": "extract"}),
    ],
)
def test_config_rejects_changed_policy(prepared: Path, field: str, value: object) -> None:
    data = read(prepared)
    data[field] = value
    prepared.write_text(json.dumps(data), encoding="utf-8")
    with pytest.raises(ReplayError):
        bridge.load_config(prepared)


@pytest.mark.parametrize("role", ["source", "audioClock", "ffmpeg"])
def test_hash_mismatch(prepared: Path, role: str) -> None:
    data = read(prepared)
    ref = record(data[role])
    ref["sha256"] = "0" * 64
    prepared.write_text(json.dumps(data), encoding="utf-8")
    with pytest.raises(StageError, match="SHA256"):
        bridge.seal(prepared)


@pytest.mark.parametrize("value", ["bad", "A" * 64, 42])
def test_reference_digest_rejected(prepared: Path, value: object) -> None:
    data = read(prepared)
    record(data["source"])["sha256"] = value
    prepared.write_text(json.dumps(data), encoding="utf-8")
    with pytest.raises(ReplayError, match="reference"):
        bridge.load_config(prepared)


def test_source_scope_and_config_scope(prepared: Path) -> None:
    data = read(prepared)
    data["mediaRoot"] = str(prepared.parent)
    prepared.write_text(json.dumps(data), encoding="utf-8")
    with pytest.raises(ReplayError, match="media root"):
        bridge.load_config(prepared)
    outside = prepared.parent.parent.parent / "outside.json"
    outside.write_text(json.dumps(data), encoding="utf-8")
    with pytest.raises(ReplayError, match="local"):
        bridge.load_config(outside)


def test_ast_mismatch_and_duplicate(prepared: Path) -> None:
    data = read(prepared)
    values = data["codeInputs"]
    assert isinstance(values, list)
    data["codeInputs"] = values + values
    prepared.write_text(json.dumps(data), encoding="utf-8")
    with pytest.raises(ReplayError, match="AST"):
        bridge.load_config(prepared)


def test_bad_completed_status(prepared: Path) -> None:
    data = read(prepared)
    completion = record(data["completedExtraction"])
    status = Path(str(record(completion["status"])["path"]))
    status.write_text("1\n", encoding="utf-8")
    record(completion["status"])["sha256"] = file_digest(status)
    prepared.write_text(json.dumps(data), encoding="utf-8")
    with pytest.raises(ReplayError, match="terminal zero"):
        bridge.seal(prepared)


@pytest.mark.parametrize(
    "text",
    [
        '{"phase":"extract","exit":1}\n',
        '{"phase":"other","exit":0}\n',
        '{"phase":"extract","exit":0}\n{"phase":"extract","exit":0}\n',
    ],
)
def test_bad_phase_evidence(prepared: Path, text: str) -> None:
    data = read(prepared)
    completion = record(data["completedExtraction"])
    phases = Path(str(record(completion["phases"])["path"]))
    phases.write_text(text, encoding="utf-8")
    record(completion["phases"])["sha256"] = file_digest(phases)
    prepared.write_text(json.dumps(data), encoding="utf-8")
    with pytest.raises(ReplayError, match="exactly one"):
        bridge.seal(prepared)


def test_producer_source_binding(prepared: Path) -> None:
    data = read(prepared)
    producer_ref = record(data["audioClockProducer"])
    producer = Path(str(producer_ref["path"]))
    evidence = read(producer)
    evidence["source"] = {"path": "other", "sha256": "0" * 64}
    producer.write_text(json.dumps(evidence), encoding="utf-8")
    producer_ref["sha256"] = file_digest(producer)
    prepared.write_text(json.dumps(data), encoding="utf-8")
    with pytest.raises(ReplayError, match="producer"):
        bridge.seal(prepared)


def test_truncated_pcm(prepared: Path) -> None:
    target = Path(str(read(prepared)["directory"])) / "model-input.wav"
    content = target.read_bytes()
    target.write_bytes(content[:-4])
    with pytest.raises(ReplayError, match="truncated"):
        bridge.seal(prepared)


def test_wrong_audio_same_sample_count(prepared: Path) -> None:
    target = Path(str(read(prepared)["directory"])) / "model-input.wav"
    with wave.open(str(target), "wb") as audio:
        audio.setnchannels(2)
        audio.setsampwidth(2)
        audio.setframerate(44100)
        audio.writeframes(b"\x01\x00" * (44100 * 2))
    with pytest.raises(ReplayError, match="payload differs"):
        bridge.seal(prepared)
    assert not (target.parent / "pcm-preparation.json").exists()


def test_hash_muxer_command(prepared: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    config = bridge.load_config(prepared)
    monkeypatch.undo()

    def run(command: list[str], **options: object) -> subprocess.CompletedProcess[bytes]:
        assert command[-5:] == ["-f", "hash", "-hash", "sha256", "-"]
        assert command[11:13] == ["-af", "asetpts=N/SR/TB,aresample=44100,volume=-0.100000dB"]
        assert options["capture_output"] is True
        return subprocess.CompletedProcess(command, 0, b"SHA256=" + b"a" * 64 + b"\n", b"")

    monkeypatch.setattr(subprocess, "run", run)
    assert bridge.expected_pcm_digest(config) == "a" * 64


@pytest.mark.parametrize(
    ("stdout", "stderr"),
    [
        (b"unknown", b""),
        (b"SHA256=bad", b""),
        (b"a" * 257, b""),
        (b"SHA256=" + b"a" * 64, b"e" * 8193),
    ],
)
def test_hash_muxer_rejects_invalid_response(
    prepared: Path, monkeypatch: pytest.MonkeyPatch, stdout: bytes, stderr: bytes
) -> None:
    config = bridge.load_config(prepared)
    monkeypatch.undo()

    def run(command: list[str], **options: object) -> subprocess.CompletedProcess[bytes]:
        return subprocess.CompletedProcess(command, 0, stdout, stderr)

    monkeypatch.setattr(subprocess, "run", run)
    with pytest.raises(ReplayError):
        bridge.expected_pcm_digest(config)


@pytest.mark.parametrize(
    ("rate", "channels", "width"), [(48000, 2, 2), (44100, 1, 2), (44100, 2, 3)]
)
def test_wrong_pcm_format(prepared: Path, rate: int, channels: int, width: int) -> None:
    target = Path(str(read(prepared)["directory"])) / "model-input.wav"
    with wave.open(str(target), "wb") as audio:
        audio.setnchannels(channels)
        audio.setsampwidth(width)
        audio.setframerate(rate)
        audio.writeframes(bytes(44100 * channels * width))
    with pytest.raises(ReplayError, match="PCM16/44100"):
        bridge.seal(prepared)


def test_changed_receipt(prepared: Path) -> None:
    receipt = bridge.seal(prepared)
    data = read(receipt)
    data["qualityApproved"] = True
    receipt.write_text(json.dumps(data), encoding="utf-8")
    with pytest.raises(ReplayError, match="changed"):
        bridge.verify(prepared)


def test_extract_one_call_and_no_retry(prepared: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    config = read(prepared)
    config["completedExtraction"] = None
    prepared.write_text(json.dumps(config), encoding="utf-8")
    directory = Path(str(config["directory"]))
    pcm = directory / "model-input.wav"
    content = pcm.read_bytes()
    pcm.unlink()

    def run(command: list[str], **options: object) -> subprocess.CompletedProcess[bytes]:
        assert command[-3:-1] == ["-c:a", "pcm_s16le"]
        assert options["check"] is True
        pcm.write_bytes(content)
        return subprocess.CompletedProcess(command, 0)

    monkeypatch.setattr(subprocess, "run", run)
    monkeypatch.setattr(bridge, "input_peak_db", lambda source, *, ffmpeg: -6.0)
    assert bridge.main(["extract", str(prepared)]) == 0
    assert bridge.verify(prepared)["inputSampleCount"] == 44100
    with pytest.raises(ReplayError, match="attempted"):
        bridge.extract(prepared)


def test_extract_failure_is_preserved(prepared: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    config = read(prepared)
    config["completedExtraction"] = None
    prepared.write_text(json.dumps(config), encoding="utf-8")
    directory = Path(str(config["directory"]))
    (directory / "model-input.wav").unlink()

    def fail(command: list[str], **options: object) -> subprocess.CompletedProcess[bytes]:
        raise subprocess.CalledProcessError(1, command)

    monkeypatch.setattr(subprocess, "run", fail)
    monkeypatch.setattr(bridge, "input_peak_db", lambda source, *, ffmpeg: -6.0)
    assert bridge.main(["extract", str(prepared)]) == 1
    assert (directory / "extraction-attempt.json").is_file()
    assert not (directory / "pcm-preparation.json").exists()
    with pytest.raises(ReplayError, match="attempted"):
        bridge.extract(prepared)


def test_missing_shared_completion_and_cli_error(prepared: Path) -> None:
    data = read(prepared)
    data["completedExtraction"] = None
    prepared.write_text(json.dumps(data), encoding="utf-8")
    assert bridge.main(["adopt", str(prepared)]) == 1
    assert not (Path(str(data["directory"])) / "pcm-preparation.json").exists()


@pytest.fixture
def template(prepared: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    config = read(prepared)
    directory = Path(str(config["directory"]))
    code = (
        Path(str(record(config["codeInputs"][0])["path"]))
        if isinstance(config["codeInputs"], list)
        else prepared
    )
    provider = prepared.parent / "provider.so"
    provider.write_bytes(b"synthetic provider")
    cache = prepared.parent / "cache"
    cache.mkdir()
    monkeypatch.setenv("HF_HUB_CACHE", str(cache))
    monkeypatch.setenv("HF_HOME", str(cache))
    monkeypatch.setenv("HF_HUB_OFFLINE", "1")
    monkeypatch.setenv("TRANSFORMERS_OFFLINE", "1")
    monkeypatch.setenv("TMPDIR", str(cache))
    monkeypatch.setenv("TMP", str(cache))
    monkeypatch.setenv("TEMP", str(cache))
    monkeypatch.setenv("XDG_CACHE_HOME", str(cache))
    snapshot = cache / ("models--" + MODEL_NAME.replace("/", "--")) / "snapshots" / MODEL_REVISION
    snapshot.mkdir(parents=True)
    weights, model_config = snapshot / "model.safetensors", snapshot / "config.json"
    weights.write_bytes(b"synthetic weights")
    model_config.write_bytes(b"{}")
    monkeypatch.setattr(pipeline, "MODEL_SHA256", file_digest(weights))
    monkeypatch.setattr(pipeline, "MODEL_CONFIG_SHA256", file_digest(model_config))
    monkeypatch.setattr(pipeline, "code_inputs", lambda: {str(code): file_digest(code)})
    monkeypatch.setattr(pipeline, "_required_provider_paths", lambda: {str(provider)})
    path = prepared.parent / "template.json"
    durable(
        path,
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
                "max_samples": 44100,
                "max_saved_bytes": 1073741824,
                "max_parent_rss_bytes": 2147483648,
                "timeout_seconds": 21600,
            },
            "codeInputs": [{"path": str(code), "sha256": file_digest(code)}],
            "providerInputs": [{"path": str(provider), "sha256": file_digest(provider)}],
            "modelInputs": [
                {"path": str(weights), "sha256": file_digest(weights)},
                {"path": str(model_config), "sha256": file_digest(model_config)},
            ],
            "chunkSamples": 26460000,
            "overlapSamples": 220500,
            "sampleRate": 44100,
            "qualityApproved": False,
            "productionBackendAdopted": False,
        },
    )
    bridge.seal(prepared)
    return path


def test_prepare_and_verify_plan(prepared: Path, template: Path) -> None:
    output = Path(str(read(prepared)["directory"])) / "plan.json"
    assert bridge.main(["prepare-plan", str(prepared), str(template), str(output)]) == 0
    before = pin(output)
    assert bridge.main(["prepare-plan", str(prepared), str(template), str(output), "--verify"]) == 0
    assert read(output)["qualityApproved"] is False
    assert pin(output) == before
    with pytest.raises(ReplayError, match="occupied"):
        bridge.prepare_plan(prepared, template, output)


def test_reject_wrong_plan_source(prepared: Path, template: Path) -> None:
    data = read(template)
    record(data["source"])["sha256"] = "0" * 64
    template.write_text(json.dumps(data), encoding="utf-8")
    output = Path(str(read(prepared)["directory"])) / "plan.json"
    with pytest.raises(ReplayError, match="unresolved"):
        bridge.prepare_plan(prepared, template, output)
    assert not output.exists()


def test_reject_plan_scope_and_changed_plan(prepared: Path, template: Path) -> None:
    with pytest.raises(ReplayError, match="distinct"):
        bridge.prepare_plan(prepared, template, prepared)
    output = Path(str(read(prepared)["directory"])) / "plan.json"
    bridge.prepare_plan(prepared, template, output)
    output.write_text("{}", encoding="utf-8")
    with pytest.raises(ReplayError, match="changed"):
        bridge.prepare_plan(prepared, template, output, verify_only=True)


def extraction_ready(prepared: Path) -> tuple[Path, bytes]:
    config = read(prepared)
    config["completedExtraction"] = None
    prepared.write_text(json.dumps(config), encoding="utf-8")
    pcm = Path(str(config["directory"])) / "model-input.wav"
    content = pcm.read_bytes()
    pcm.unlink()
    return pcm, content


def test_an_over_range_source_is_extracted_with_a_measured_static_gain(
    prepared: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    pcm, content = extraction_ready(prepared)
    commands: list[list[str]] = []

    def run(command: list[str], **options: object) -> subprocess.CompletedProcess[bytes]:
        commands.append(command)
        pcm.write_bytes(content)
        return subprocess.CompletedProcess(command, 0)

    monkeypatch.setattr(subprocess, "run", run)
    monkeypatch.setattr(bridge, "input_peak_db", lambda source, *, ffmpeg: 2.9)
    bridge.extract(prepared)
    gain = read(pcm.with_name("input-gain.json"))
    assert (gain["peakDB"], gain["gainDB"]) == (2.9, -3.0)
    assert "asetpts=N/SR/TB,aresample=44100,volume=-3.000000dB" in commands[0]
    receipt = bridge.verify(prepared)
    assert (receipt["gainDB"], receipt["gainPolicy"]) == (
        -3.0, "measured decoded float peak; one static gain keeps it below full scale",
    )  # fmt: skip
    assert receipt["expectedExtractionCommand"] == commands[0]


def test_a_source_within_full_scale_keeps_the_fixed_gain(
    prepared: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    pcm, content = extraction_ready(prepared)

    def run(command: list[str], **options: object) -> subprocess.CompletedProcess[bytes]:
        pcm.write_bytes(content)
        return subprocess.CompletedProcess(command, 0)

    monkeypatch.setattr(subprocess, "run", run)
    monkeypatch.setattr(bridge, "input_peak_db", lambda source, *, ffmpeg: -6.0)
    bridge.extract(prepared)
    assert read(pcm.with_name("input-gain.json"))["gainDB"] == -0.1


def test_a_changed_gain_record_is_refused(prepared: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    pcm, content = extraction_ready(prepared)

    def run(command: list[str], **options: object) -> subprocess.CompletedProcess[bytes]:
        pcm.write_bytes(content)
        return subprocess.CompletedProcess(command, 0)

    monkeypatch.setattr(subprocess, "run", run)
    monkeypatch.setattr(bridge, "input_peak_db", lambda source, *, ffmpeg: 2.9)
    bridge.extract(prepared)
    record_path = pcm.with_name("input-gain.json")
    data = read(record_path)
    record_path.unlink()
    durable(record_path, {**data, "gainDB": 0.5})
    with pytest.raises(ReplayError, match="Invalid measured extraction gain"):
        bridge.verify(prepared)


def test_the_input_peak_is_read_from_a_float_measurement(monkeypatch: pytest.MonkeyPatch) -> None:
    seen: list[list[str]] = []

    def run(command: list[str], **options: object) -> subprocess.CompletedProcess[str]:
        seen.append(command)
        return subprocess.CompletedProcess(command, 0, "", "Overall\nPeak level dB: 2.890329\n")

    monkeypatch.setattr(subprocess, "run", run)
    assert bridge.input_peak_db(Path("/s.mp4"), ffmpeg="ff") == 2.890329
    assert "aformat=sample_fmts=flt" in seen[0][seen[0].index("-af") + 1]

    def silent(command: list[str], **options: object) -> subprocess.CompletedProcess[str]:
        return subprocess.CompletedProcess(command, 0, "", "Peak level dB: -inf\n")

    monkeypatch.setattr(subprocess, "run", silent)
    with pytest.raises(ReplayError, match="no measurable peak"):
        bridge.input_peak_db(Path("/s.mp4"), ffmpeg="ff")
