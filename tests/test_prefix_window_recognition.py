"""Independent prefix candidates retain original clocks and truthful adoption gaps."""

import json
import subprocess
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pytest
from numpy.typing import NDArray

from replay import isolated_cpu_asr as isolated
from replay import prefix_window_recognition as bridge
from replay.asr_runtime import ASRDevice
from replay.core import ReplayError
from replay.native_invocation import pin
from replay.stages import execution_lock


@dataclass(frozen=True)
class Fixture:
    root: Path
    request: Path
    output: Path
    pcm: Path
    weights: Path
    window: Path
    prefix: Path


def write(path: Path, value: object) -> None:
    path.write_text(json.dumps(value), encoding="utf-8")


def ref(path: Path) -> dict[str, object]:
    value = pin(path)
    return {"path": str(path), "sha256": value["sha256"]}


class FakeRuntime:
    def __init__(self, device: ASRDevice) -> None:
        assert device == ASRDevice.CPU

    def predict(self, samples: NDArray[np.float32], model: Path, *, evidence: Path) -> object:
        assert samples.dtype == np.float32
        assert samples.shape == (960000,)
        assert model.name == "model"
        evidence.mkdir()
        return {"segments": [{"start": 1.0, "end": 2.0, "text": "認識"}], "text": "認識"}


@pytest.fixture
def fixture(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Fixture:
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(bridge, "ROOT", tmp_path)
    root = tmp_path / "edits/work"
    root.mkdir(parents=True)
    scratch = root / "scratch"
    scratch.mkdir()
    monkeypatch.setenv("TMPDIR", str(scratch))
    monkeypatch.setenv("TMP", str(scratch))
    monkeypatch.setenv("TEMP", str(scratch))
    monkeypatch.setenv("XDG_CACHE_HOME", str(scratch))
    monkeypatch.setenv("NUMBA_CACHE_DIR", str(scratch))
    monkeypatch.setenv("HF_HUB_OFFLINE", "1")
    monkeypatch.setenv("HF_DATASETS_OFFLINE", "1")
    model = tmp_path / "edits/models/model"
    model.mkdir(parents=True)
    config, weights = model / "config.json", model / "weights.npz"
    config.write_bytes(b"config")
    weights.write_bytes(b"weights")
    monkeypatch.setattr(bridge, "DEFAULT_CONFIG_SHA256", pin(config)["sha256"])
    monkeypatch.setattr(bridge, "DEFAULT_WEIGHTS_SHA256", pin(weights)["sha256"])
    interpreter = root / "mel-environment/bin/python"
    interpreter.parent.mkdir(parents=True)
    interpreter.write_bytes(b"synthetic-original-provider-interpreter")
    interpreter.chmod(0o700)
    pcm = root / "000000480000.pcm"
    pcm.write_bytes(bytes(1920000))
    prefix_request = root / "prefix-request.json"
    write(prefix_request, {"version": 1, "originalProvider": "mel"})
    model_contract = {
        "id": bridge.MODEL,
        "configSHA256": pin(config)["sha256"],
        "weightsSHA256": pin(weights)["sha256"],
    }
    window = root / "000000480000.window.json"
    write(
        window,
        {
            "kind": "verified-separation-prefix-window",
            "request": pin(prefix_request),
            "startSample": 480000,
            "endSample": 1440000,
            "sampleRate": 16000,
            "channels": 1,
            "sampleWidth": 2,
            "clock": "input-pcm-not-projected",
            "pcm": ref(pcm),
            "model": model_contract,
        },
    )
    prefix = root / "prefix-receipt.json"
    write(
        prefix,
        {
            "kind": "separation-prefix-asr-windows",
            "chunkDomainVerified": True,
            "bindings": {"request": pin(prefix_request)},
            "windows": [{"receipt": pin(window)}],
        },
    )
    monkeypatch.setattr(bridge, "runtime_inputs", lambda: {"synthetic-source": "a" * 64})
    monkeypatch.setattr(bridge, "require_headroom", lambda *args, **kwargs: [])
    monkeypatch.setattr(isolated, "WhisperRuntime", FakeRuntime)
    request, output = root / "recognition-request.json", root / "recognition"
    write(
        request,
        {
            "version": 1,
            "prefixRequest": ref(prefix_request),
            "prefixReceipt": ref(prefix),
            "window": ref(window),
            "model": {
                "id": bridge.MODEL,
                "directory": str(model),
                "config": ref(config),
                "weights": ref(weights),
            },
            "verifierPython": ref(interpreter),
            "verifierInvocation": str(interpreter),
            "outputDirectory": str(output),
            "scratchDirectory": str(scratch),
            "runtime": bridge.runtime_inputs(),
            "timeoutSeconds": 30,
            "deadlineSeconds": 60,
            "ramReservationBytes": 64 * 1024**2,
            "maxSavedBytes": 64 * 1024**2,
        },
    )
    return Fixture(root, request, output, pcm, weights, window, prefix)


def execute(command: tuple[str, ...], log: Path, timeout: int) -> None:
    assert timeout == 30
    log.write_text("synthetic subprocess completion\n", encoding="utf-8")
    if command[-2] == "--worker":
        isolated.worker(Path(command[-1]))
    else:
        assert command[0].endswith("mel-environment/bin/python")
        assert command[1:5] == ("-B", "-m", "replay.separation_prefix_asr", "verify")


def test_genuine_isolated_worker_clock_and_idempotent_resume(fixture: Fixture) -> None:
    calls: list[str] = []

    def tracked(command: tuple[str, ...], log: Path, timeout: int) -> None:
        calls.append(command[-2])
        execute(command, log, timeout)

    result = bridge.run(fixture.request, execute=tracked)
    assert calls == ["verify", "--worker"]
    before = pin(result)
    assert bridge.run(fixture.request, execute=tracked) == result
    assert bridge.verify(fixture.request) == result
    assert pin(result) == before
    assert calls == ["verify", "--worker"]
    assert json.loads((fixture.output / "clock.json").read_text()) == {
        "startSample": 480000,
        "endSample": 1440000,
        "sampleRate": 16000,
        "humanReviewed": False,
        "emptyRecognitionIsNotSilence": True,
        "rawSHA256": pin(fixture.output / "raw.json")["sha256"],
    }
    record = json.loads(result.read_text())
    assert record["recognitionExecuted"] is True
    assert record["consumerConnected"] is False
    assert record["finalMonoWindowEqual"] is False
    assert not (fixture.output / "inputs.json").exists()
    assert not (fixture.output / "policy").exists()


def test_failure_retains_attempt_and_does_not_duplicate_worker(fixture: Fixture) -> None:
    calls: list[str] = []

    def failed(command: tuple[str, ...], log: Path, timeout: int) -> None:
        calls.append(command[-2])
        if command[-2] == "--worker":
            raise subprocess.TimeoutExpired(command, timeout)
        execute(command, log, timeout)

    with pytest.raises(subprocess.TimeoutExpired):
        bridge.run(fixture.request, execute=failed)
    assert (fixture.output / "attempt.json").exists()
    assert (fixture.output / "cpu-runtime/request.json").exists()
    assert not (fixture.output / "complete.json").exists()
    with pytest.raises(ReplayError, match="Incomplete attempt"):
        bridge.run(fixture.request, execute=failed)
    assert calls == ["verify", "--worker"]


def test_prefix_verifier_failure_precedes_inference(fixture: Fixture) -> None:
    def failed(command: tuple[str, ...], log: Path, timeout: int) -> None:
        assert command[-2] == "verify"
        raise ReplayError("Original provider not installed")

    with pytest.raises(ReplayError, match="Original provider"):
        bridge.run(fixture.request, execute=failed)
    assert not (fixture.output / "cpu-runtime").exists()
    assert not (fixture.output / "complete.json").exists()


@pytest.mark.parametrize("target", ["pcm", "weights", "prefix", "window"])
def test_changed_window_model_or_prefix_is_not_reused(fixture: Fixture, target: str) -> None:
    bridge.run(fixture.request, execute=execute)
    path = {
        "pcm": fixture.pcm,
        "weights": fixture.weights,
        "prefix": fixture.prefix,
        "window": fixture.window,
    }[target]
    path.write_bytes(bytes(path.stat().st_size))
    with pytest.raises(ReplayError, match="changed"):
        bridge.run(fixture.request, execute=execute)


@pytest.mark.parametrize(
    "name",
    [
        "raw.json",
        "clock.json",
        "cpu-runtime/raw.json",
        "cpu-runtime/result.json",
        "cpu-runtime/samples.npy",
    ],
)
def test_candidate_tampering_is_rejected(fixture: Fixture, name: str) -> None:
    bridge.run(fixture.request, execute=execute)
    (fixture.output / name).write_text("{}", encoding="utf-8")
    with pytest.raises(ReplayError):
        bridge.verify(fixture.request)


def test_preflight_reads_no_pcm_or_model_bytes(
    fixture: Fixture, monkeypatch: pytest.MonkeyPatch
) -> None:
    original = pin

    def guarded(path: Path) -> dict[str, object]:
        assert path not in (fixture.pcm, fixture.weights)
        return original(path)

    monkeypatch.setattr(bridge, "pin", guarded)
    result = bridge.preflight(fixture.request)
    assert result["fullWindows"] == 1
    assert result["modelBytesVerified"] is False
    assert result["recognitionExecuted"] is False
    assert not fixture.output.exists()


def test_unconnected_adoption_protocol_preserves_candidate(fixture: Fixture) -> None:
    complete = bridge.run(fixture.request, execute=execute)
    before = pin(complete)
    output = bridge.adoption_plan(fixture.request, fixture.root / "adoption.json")
    result = json.loads(output.read_text())
    assert result["requiredFinalExecutor"] == "replay.isolated_cpu_asr.run"
    assert result["requiredComparison"] == "replay.separation_prefix_asr.compare_final"
    assert result["normalFactorySeedImportImplemented"] is False
    assert result["consumerConnected"] is False
    assert result["adopted"] is False
    assert pin(complete) == before


@pytest.mark.parametrize(
    "key,value",
    [
        ("timeoutSeconds", 0),
        ("deadlineSeconds", 1),
        ("maxSavedBytes", 1),
        ("ramReservationBytes", 1),
        ("version", True),
        ("runtime", {}),
    ],
)
def test_request_bounds(fixture: Fixture, key: str, value: object) -> None:
    raw = json.loads(fixture.request.read_text())
    raw[key] = value
    write(fixture.request, raw)
    with pytest.raises(ReplayError):
        bridge.preflight(fixture.request)


@pytest.mark.parametrize("key", ["TMPDIR", "HF_HUB_OFFLINE"])
def test_scratch_and_offline_required_before_side_effects(
    fixture: Fixture, monkeypatch: pytest.MonkeyPatch, key: str
) -> None:
    monkeypatch.delenv(key)
    with pytest.raises(ReplayError):
        bridge.run(fixture.request, execute=execute)
    assert not fixture.output.exists()


def test_lost_deadline_stops_before_worker(
    fixture: Fixture, monkeypatch: pytest.MonkeyPatch
) -> None:
    values = iter((0.0, 61.0))
    monkeypatch.setattr(time, "monotonic", lambda: next(values))
    with pytest.raises(ReplayError, match="deadline"):
        bridge.run(fixture.request, execute=execute)
    assert not (fixture.output / "cpu-runtime").exists()


def test_budget_failure_retains_output_without_completion(
    fixture: Fixture, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(bridge, "_saved_bytes", lambda output: 128 * 1024**2)
    with pytest.raises(ReplayError, match="resource budget"):
        bridge.run(fixture.request, execute=execute)
    assert (fixture.output / "clock.json").exists()
    assert not (fixture.output / "complete.json").exists()


def test_output_lease_prevents_duplicate(fixture: Fixture) -> None:
    fixture.output.mkdir()
    with execution_lock(fixture.output, ".recognition") as acquired:
        assert acquired is True
        with pytest.raises(ReplayError, match="already owned"):
            bridge.run(fixture.request, execute=execute)


def test_cli_check_verify_and_protocol(
    fixture: Fixture, capsys: pytest.CaptureFixture[str]
) -> None:
    assert bridge.main(["check", str(fixture.request)]) == 0
    assert json.loads(capsys.readouterr().out)["fullWindows"] == 1
    bridge.run(fixture.request, execute=execute)
    assert bridge.main(["verify", str(fixture.request)]) == 0
    assert (
        bridge.main(
            ["adoption-plan", str(fixture.request), "--output", str(fixture.root / "protocol.json")]
        )
        == 0
    )
    with pytest.raises(SystemExit, match="1"):
        bridge.main(["adoption-plan", str(fixture.request)])


def test_real_static_capture_without_inference() -> None:
    result = bridge.runtime_inputs()
    assert str(Path(bridge.__file__).resolve()) in result
    assert str(Path(isolated.__file__).resolve()) in result
