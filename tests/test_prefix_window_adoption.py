"""A real public bootstrap and resume consume equal synthetic prefix candidates."""

import json
import shutil
import wave
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pytest
from numpy.typing import NDArray

from replay import asr_assembly
from replay import isolated_cpu_asr as isolated
from replay import prefix_window_adoption as adoption
from replay import prefix_window_recognition as bridge
from replay.asr_runtime import ASRDevice
from replay.core import ReplayError
from replay.native_invocation import pin
from replay.stages import execution_lock


@dataclass(frozen=True)
class Fixture:
    root: Path
    request: Path
    source: Path
    model: Path
    collection: Path
    candidate: Path
    candidate_request: Path
    comparison: Path
    bootstrap: Path
    receipt: Path
    start: int


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
        return {
            "segments": [
                {
                    "start": 30.0,
                    "end": 31.0,
                    "text": "認識",
                    "words": [{"start": 30.0, "end": 31.0, "word": "認識", "probability": 0.9}],
                }
            ],
            "text": "認識",
            "language": "ja",
        }


def execute(command: tuple[str, ...], log: Path, timeout: int) -> None:
    assert timeout == 30
    log.write_text("synthetic subprocess completion\n", encoding="utf-8")
    if command[-2] == "--worker":
        isolated.worker(Path(command[-1]))
    else:
        assert command[0].endswith("mel-environment/bin/python")
        assert command[1:5] == ("-B", "-m", "replay.separation_prefix_asr", "verify")


@pytest.fixture
def fixture(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, request: pytest.FixtureRequest
) -> Fixture:
    start = request.param if hasattr(request, "param") else 480000
    assert isinstance(start, int)
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(bridge, "ROOT", tmp_path)
    monkeypatch.setattr(adoption, "ROOT", tmp_path)
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
    code = tmp_path / "src/replay"
    code.mkdir(parents=True)
    shutil.copyfile(isolated.__file__, code / "isolated_cpu_asr.py")
    shutil.copyfile(Path(isolated.__file__).with_name("batch_asr.py"), code / "batch_asr.py")
    shutil.copyfile(Path(isolated.__file__).with_name("asr_runtime.py"), code / "asr_runtime.py")
    monkeypatch.setattr(isolated, "__file__", str(code / "isolated_cpu_asr.py"))
    model = tmp_path / "edits/models/model"
    model.mkdir(parents=True)
    config, weights = model / "config.json", model / "weights.npz"
    config.write_bytes(b"config")
    weights.write_bytes(b"weights")
    monkeypatch.setattr(bridge, "DEFAULT_CONFIG_SHA256", pin(config)["sha256"])
    monkeypatch.setattr(bridge, "DEFAULT_WEIGHTS_SHA256", pin(weights)["sha256"])
    monkeypatch.setattr(adoption, "DEFAULT_CONFIG_SHA256", pin(config)["sha256"])
    monkeypatch.setattr(adoption, "DEFAULT_WEIGHTS_SHA256", pin(weights)["sha256"])
    monkeypatch.setattr(isolated, "WhisperRuntime", FakeRuntime)
    interpreter = root / "mel-environment/bin/python"
    interpreter.parent.mkdir(parents=True)
    interpreter.write_bytes(b"synthetic-original-provider-interpreter")
    interpreter.chmod(0o700)
    source = root / "final-mono.wav"
    with wave.open(str(source), "wb") as audio:
        audio.setparams((1, 2, 16000, 0, "NONE", "not compressed"))
        audio.writeframes(bytes(16000 * 2 * 120))
    pcm = root / "prefix.pcm"
    pcm.write_bytes(bytes(1920000))
    prefix_request = root / "prefix-request.json"
    write(prefix_request, {"version": 1, "syntheticOriginalProvider": "mel"})
    window = root / "window.json"
    write(
        window,
        {
            "kind": "verified-separation-prefix-window",
            "request": pin(prefix_request),
            "startSample": start,
            "endSample": start + 960000,
            "sampleRate": 16000,
            "channels": 1,
            "sampleWidth": 2,
            "clock": "input-pcm-not-projected",
            "pcm": ref(pcm),
            "model": {
                "id": bridge.MODEL,
                "configSHA256": pin(config)["sha256"],
                "weightsSHA256": pin(weights)["sha256"],
            },
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
    monkeypatch.setattr(bridge, "runtime_inputs", lambda: {"synthetic-prefix-source": "a" * 64})
    monkeypatch.setattr(bridge, "require_headroom", lambda *args, **kwargs: [])
    candidate_request, candidate = root / "candidate-request.json", root / "candidate"
    write(
        candidate_request,
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
            "outputDirectory": str(candidate),
            "scratchDirectory": str(scratch),
            "runtime": bridge.runtime_inputs(),
            "timeoutSeconds": 30,
            "deadlineSeconds": 60,
            "ramReservationBytes": 64 * 1024**2,
            "maxSavedBytes": 64 * 1024**2,
        },
    )
    complete = bridge.run(candidate_request, execute=execute)
    candidate_status = root / "candidate.status"
    candidate_status.write_text("0\n", encoding="utf-8")
    collection, bootstrap = root / "collection", root / "bootstrap-policy"
    isolated.run(
        source,
        collection,
        bootstrap,
        limit=1,
        timeout=30,
        model=model,
        config_sha256=str(pin(config)["sha256"]),
        weights_sha256=str(pin(weights)["sha256"]),
        execute=execute,
    )
    bootstrap_status = root / "bootstrap.status"
    bootstrap_status.write_text("0\n", encoding="utf-8")
    final_plan = root / "final-plan.json"
    write(final_plan, {"version": 1, "syntheticFinalMonoProducer": "verified"})
    comparison = root / "comparison.json"
    write(
        comparison,
        {
            "version": 1,
            "kind": "final-mono-prefix-window-comparison",
            "prefix": pin(prefix),
            "finalPlan": pin(final_plan),
            "finalMono": pin(source),
            "windows": [{"window": pin(window), "finalMonoWindowEqual": True}],
            "fullMonoProducerVerified": True,
            "finalMonoWindowEqual": True,
            "recognitionExecuted": False,
            "adopted": False,
            "qualityApproved": False,
            "deliveryComplete": False,
        },
    )
    monkeypatch.setattr(adoption, "runtime_inputs", lambda: {"synthetic-adoption-source": "b" * 64})
    path, receipt = root / "adoption-request.json", root / "adoption-receipt.json"
    write(
        path,
        {
            "version": 1,
            "candidates": [
                {
                    "request": ref(candidate_request),
                    "complete": ref(complete),
                    "jobStatus": ref(candidate_status),
                }
            ],
            "comparison": ref(comparison),
            "bootstrap": {
                "manifest": ref(collection / "inputs.json"),
                "started": ref(bootstrap / "started.json"),
                "finished": ref(bootstrap / "result.json"),
                "jobStatus": ref(bootstrap_status),
            },
            "output": str(receipt),
            "runtime": adoption.runtime_inputs(),
        },
    )
    return Fixture(
        root,
        path,
        source,
        model,
        collection,
        candidate,
        candidate_request,
        comparison,
        bootstrap,
        receipt,
        start,
    )


def repin_request(fixture: Fixture) -> None:
    data = json.loads(fixture.request.read_text(encoding="utf-8"))
    data["comparison"] = ref(fixture.comparison)
    data["bootstrap"]["manifest"] = ref(fixture.collection / "inputs.json")
    data["bootstrap"]["started"] = ref(fixture.bootstrap / "started.json")
    data["bootstrap"]["finished"] = ref(fixture.bootstrap / "result.json")
    data["bootstrap"]["jobStatus"] = ref(fixture.root / "bootstrap.status")
    data["candidates"][0]["jobStatus"] = ref(fixture.root / "candidate.status")
    write(fixture.request, data)


def test_public_bootstrap_import_resume_and_normal_assembly(fixture: Fixture) -> None:
    before_manifest = pin(fixture.collection / "inputs.json")
    before_started = pin(fixture.bootstrap / "started.json")
    assert adoption.adopt(fixture.request) == fixture.receipt
    assert not (fixture.collection / "complete.json").exists()
    assert (fixture.collection / "000000480000.raw.json").read_bytes() == (
        fixture.candidate / "raw.json"
    ).read_bytes()
    assert (fixture.collection / "000000480000.clock.json").read_bytes() == (
        fixture.candidate / "clock.json"
    ).read_bytes()
    assert not (fixture.collection / "000000480000.cpu-runtime").exists()
    calls: list[str] = []

    def tracked(command: tuple[str, ...], log: Path, timeout: int) -> None:
        calls.append(command[-2])
        execute(command, log, timeout)

    result = isolated.run(
        fixture.source,
        fixture.collection,
        fixture.root / "resume-policy",
        timeout=30,
        model=fixture.model,
        config_sha256=str(pin(fixture.model / "config.json")["sha256"]),
        weights_sha256=str(pin(fixture.model / "weights.npz")["sha256"]),
        execute=tracked,
    )
    assert calls == ["--worker"]
    assert json.loads(result.read_text(encoding="utf-8"))["newWindows"] == 1
    assert json.loads(result.read_text(encoding="utf-8"))["verifiedWindows"] == 3
    assert pin(fixture.collection / "inputs.json") == before_manifest
    assert pin(fixture.bootstrap / "started.json") == before_started
    assert (
        json.loads((fixture.root / "resume-policy/started.json").read_text(encoding="utf-8"))[
            "limitNewWindows"
        ]
        is None
    )
    assert asr_assembly.assemble(fixture.collection, fixture.root / "labels.json") == 3
    assert json.loads((fixture.root / "labels.json").read_text(encoding="utf-8"))["labels"] == [
        {"text": "認識", "start": 30.0, "end": 31.0, "provenance": "000000000000:0"},
        {"text": "認識", "start": 60.0, "end": 61.0, "provenance": "000000480000:0"},
        {"text": "認識", "start": 90.0, "end": 91.0, "provenance": "000000960000:0"},
    ]
    record = json.loads(fixture.receipt.read_text(encoding="utf-8"))
    assert record["fullASRCompletionCreated"] is False
    assert record["normalFactoryConnected"] is False
    assert record["requiredLimitNewWindows"] is None
    assert adoption.verify(fixture.request) == fixture.receipt


@pytest.mark.parametrize("fixture", [0], indirect=True)
def test_genuine_first_window_is_preserved_and_duplicate_explicit(fixture: Fixture) -> None:
    original = pin(fixture.collection / "000000000000.raw.json")
    adoption.adopt(fixture.request)
    assert pin(fixture.collection / "000000000000.raw.json") == original
    data = json.loads(fixture.receipt.read_text(encoding="utf-8"))
    assert data["windows"][0]["adopted"] is False
    assert data["windows"][0]["duplicateFirstWindow"] is True
    assert adoption.verify(fixture.request) == fixture.receipt
    assert not (fixture.collection / "complete.json").exists()
    with pytest.raises(ReplayError, match="Occupied"):
        adoption.adopt(fixture.request)


@pytest.mark.parametrize("status", ["candidate.status", "bootstrap.status"])
def test_failed_or_uncertain_jobs_rejected_before_import(fixture: Fixture, status: str) -> None:
    (fixture.root / status).write_text("1\n", encoding="utf-8")
    repin_request(fixture)
    with pytest.raises(ReplayError, match="Failed or uncertain"):
        adoption.adopt(fixture.request)
    assert not (fixture.collection / "000000480000.raw.json").exists()


@pytest.mark.parametrize("field", ["fullMonoProducerVerified", "finalMonoWindowEqual"])
def test_missing_final_equality_never_adopts(fixture: Fixture, field: str) -> None:
    data = json.loads(fixture.comparison.read_text(encoding="utf-8"))
    data[field] = False
    write(fixture.comparison, data)
    repin_request(fixture)
    with pytest.raises(ReplayError, match="compare_final"):
        adoption.adopt(fixture.request)


@pytest.mark.parametrize(
    "target", ["final-plan.json", "candidate/raw.json", "collection/inputs.json", "comparison.json"]
)
def test_pinned_inputs_or_candidate_bytes_cannot_change(fixture: Fixture, target: str) -> None:
    (fixture.root / target).write_text("{}", encoding="utf-8")
    with pytest.raises(ReplayError):
        adoption.adopt(fixture.request)
    assert not fixture.receipt.exists()


@pytest.mark.parametrize("suffix", ["raw.json", "clock.json", "cpu-runtime"])
def test_existing_or_dangling_attempt_rejects_before_any_copy(
    fixture: Fixture, suffix: str
) -> None:
    target = fixture.collection / f"000000480000.{suffix}"
    target.symlink_to(fixture.root / "missing")
    with pytest.raises(ReplayError, match="existing/uncertain"):
        adoption.adopt(fixture.request)
    assert not (fixture.collection / "000000480000.raw.json").is_file()


def test_collection_lease_does_not_mutate_or_duplicate(fixture: Fixture) -> None:
    with execution_lock(fixture.collection, ".run") as acquired:
        assert acquired is True
        with pytest.raises(ReplayError, match="still owned"):
            adoption.adopt(fixture.request)
    assert not fixture.receipt.exists()


def test_altered_final_pcm_is_rejected_despite_rebound_container(fixture: Fixture) -> None:
    with wave.open(str(fixture.source), "wb") as audio:
        audio.setparams((1, 2, 16000, 0, "NONE", "not compressed"))
        audio.writeframes(b"\x01\x00" * (16000 * 120))
    comparison = json.loads(fixture.comparison.read_text(encoding="utf-8"))
    comparison["finalMono"] = pin(fixture.source)
    write(fixture.comparison, comparison)
    manifest = json.loads((fixture.collection / "inputs.json").read_text(encoding="utf-8"))
    manifest["inputs"][str(fixture.source)] = pin(fixture.source)["sha256"]
    write(fixture.collection / "inputs.json", manifest)
    started = json.loads((fixture.bootstrap / "started.json").read_text(encoding="utf-8"))
    started["inputs"] = manifest["inputs"]
    started["manifestSHA256"] = pin(fixture.collection / "inputs.json")["sha256"]
    write(fixture.bootstrap / "started.json", started)
    repin_request(fixture)
    with pytest.raises(ReplayError, match="PCM interval differs"):
        adoption.adopt(fixture.request)


def test_failed_copy_retains_partial_and_cannot_repeat(
    fixture: Fixture, monkeypatch: pytest.MonkeyPatch
) -> None:
    real_copy = adoption._copy

    def failed(source: Path, target: Path) -> dict[str, object]:
        if target.name.endswith("clock.json"):
            raise OSError("synthetic interrupted write")
        return real_copy(source, target)

    monkeypatch.setattr(adoption, "_copy", failed)
    with pytest.raises(OSError, match="interrupted"):
        adoption.adopt(fixture.request)
    assert (fixture.collection / "000000480000.raw.json").is_file()
    assert not fixture.receipt.exists()
    with pytest.raises(ReplayError, match="existing/uncertain"):
        adoption.adopt(fixture.request)


def test_cli_success_and_failure(fixture: Fixture, capsys: pytest.CaptureFixture[str]) -> None:
    assert adoption.main(["import", str(fixture.request)]) == 0
    assert '"fullASRCompletionCreated": false' in capsys.readouterr().out
    assert adoption.main(["verify", str(fixture.request)]) == 0
    with pytest.raises(SystemExit) as error:
        adoption.main(["import", str(fixture.request)])
    assert error.value.code == 1


@pytest.mark.parametrize(
    "target",
    [
        "adoption-receipt.json",
        "collection/000000480000.raw.json",
        "collection/000000480000.clock.json",
    ],
)
def test_imported_provenance_and_windows_remain_verified(fixture: Fixture, target: str) -> None:
    adoption.adopt(fixture.request)
    (fixture.root / target).write_text("{}", encoding="utf-8")
    with pytest.raises(ReplayError):
        adoption.verify(fixture.request)


def test_verification_rejects_live_normal_owner(fixture: Fixture) -> None:
    adoption.adopt(fixture.request)
    with execution_lock(fixture.collection, ".run") as acquired:
        assert acquired is True
        with pytest.raises(ReplayError, match="still owned"):
            adoption.verify(fixture.request)


def test_rebound_output_receipt_cannot_change_native_clock(fixture: Fixture) -> None:
    adoption.adopt(fixture.request)
    target = fixture.collection / "000000480000.clock.json"
    changed = json.loads(target.read_text(encoding="utf-8"))
    changed["startSample"] = 0
    write(target, changed)
    receipt = json.loads(fixture.receipt.read_text(encoding="utf-8"))
    receipt["windows"][0]["clock"] = pin(target)
    write(fixture.receipt, receipt)
    with pytest.raises(ReplayError, match="Imported window changed"):
        adoption.verify(fixture.request)


def test_changed_prefix_cannot_be_relabelled_current(fixture: Fixture) -> None:
    (fixture.root / "prefix-receipt.json").write_text("{}", encoding="utf-8")
    with pytest.raises(ReplayError, match="changed"):
        adoption.adopt(fixture.request)
    assert not fixture.receipt.exists()


def test_duplicate_windows_rejected_before_mutation(fixture: Fixture) -> None:
    data = json.loads(fixture.request.read_text(encoding="utf-8"))
    data["candidates"].append(data["candidates"][0])
    write(fixture.request, data)
    with pytest.raises(ReplayError, match="Duplicate candidate"):
        adoption.adopt(fixture.request)
    assert not (fixture.collection / "000000480000.raw.json").exists()


def test_wrong_bootstrap_limit_never_becomes_normal_policy(fixture: Fixture) -> None:
    data = json.loads((fixture.bootstrap / "started.json").read_text(encoding="utf-8"))
    data["limitNewWindows"] = None
    write(fixture.bootstrap / "started.json", data)
    repin_request(fixture)
    with pytest.raises(ReplayError, match="limit=1"):
        adoption.adopt(fixture.request)


@pytest.mark.parametrize("field", ["runtime", "candidates"])
def test_request_requires_current_code_and_bounded_candidates(fixture: Fixture, field: str) -> None:
    data = json.loads(fixture.request.read_text(encoding="utf-8"))
    data[field] = {} if field == "runtime" else []
    write(fixture.request, data)
    with pytest.raises(ReplayError):
        adoption.adopt(fixture.request)
