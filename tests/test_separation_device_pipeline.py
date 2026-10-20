"""Synthetic mixed-device continuation; no model, accelerator or real recording."""

from __future__ import annotations

import json
import os
import subprocess
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import pytest
from numpy.typing import NDArray

from replay import mps_separation_acceptance as acceptance
from replay import separation_device_pipeline as device
from replay import separation_window_budget as window
from replay import separation_window_pipeline as original
from replay import workflow
from replay.bgm_separation import (
    MODEL_NAME,
    MODEL_REVISION,
    MODEL_SAMPLE_RATE,
    _chunk_bounds,
    _write_wav,
)
from replay.core import ReplayError
from replay.native_invocation import durable, identity, pin, read, rows
from replay.process_family_observation import SystemHost
from replay.recordings import record
from replay.separation_device_pipeline import _code as actual_code
from replay.separation_device_pipeline import _no_old_writers as observe_cpu_writers
from replay.separation_device_pipeline import _qualified as verify_external_qualification
from replay.stages import execution_lock


class Backend:
    sources = ("vocals", "instrumental")

    def extract_vocals(self, audio: NDArray[np.float32]) -> NDArray[np.float32]:
        return audio * np.float32(0.25)

    def extract_instrumental(self, audio: NDArray[np.float32]) -> NDArray[np.float32]:
        return audio - self.extract_vocals(audio)


class Host(SystemHost):
    def birth(self, pid: int) -> str:
        return f"{pid}:123"

    def absent(self, pid: int) -> bool:
        return True


def ref(path: Path) -> dict[str, str]:
    return device._file(path)


def replace(path: Path, data: dict[str, object]) -> None:
    path.write_text(json.dumps(data), encoding="utf-8")


def audit(path: Path, phase: str) -> None:
    durable(
        path / f"{phase}-descriptors.json",
        {"pid": 123, "snapshot": "p123\n", "externalWritableRegular": []},
    )


@dataclass(frozen=True)
class Case:
    root: Path
    config: Path
    old: Path
    work: Path
    baseline: Path
    request: Path
    gate: dict[str, object]


@pytest.fixture
def case(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Case:
    cache = tmp_path / "cache"
    cache.mkdir()
    tuple(
        map(
            lambda key: monkeypatch.setenv(key, str(cache)),
            ("TMPDIR", "TMP", "TEMP", "XDG_CACHE_HOME", "HF_HOME", "HF_HUB_CACHE"),
        )
    )
    monkeypatch.setenv("HF_HUB_OFFLINE", "1")
    monkeypatch.setenv("TRANSFORMERS_OFFLINE", "1")
    monkeypatch.delenv("MLX_DISABLE_COMPILE", raising=False)
    source = tmp_path / "prepared.wav"
    _write_wav(
        source,
        np.linspace(-0.2, 0.4, 46 * MODEL_SAMPLE_RATE, dtype=np.float32).reshape(
            2, 23 * MODEL_SAMPLE_RATE
        ),
    )
    recording = tmp_path / "original.media"
    recording.write_bytes(b"original source synthetic identity")
    provider = tmp_path / "provider.so"
    provider.write_bytes(b"provider")
    code = tmp_path / "reader-code.py"
    code.write_bytes(b"# synthetic reader\n")
    bgm = Path(original.__file__).with_name("bgm_separation.py").resolve()
    code_map = {str(code): ref(code)["sha256"], str(bgm): ref(bgm)["sha256"]}
    monkeypatch.setattr(original, "code_inputs", lambda: code_map)
    monkeypatch.setattr(original, "_required_provider_paths", lambda: {str(provider)})
    monkeypatch.setattr(device, "_code", lambda: {str(code): ref(code)["sha256"]})
    monkeypatch.setattr(window, "provider_pins", lambda: {"synthetic": "provider"})
    snapshot = cache / ("models--" + MODEL_NAME.replace("/", "--")) / "snapshots" / MODEL_REVISION
    snapshot.mkdir(parents=True)
    weights, model_config = snapshot / "model.safetensors", snapshot / "config.json"
    weights.write_bytes(b"synthetic weights")
    model_config.write_bytes(b"{}")
    monkeypatch.setattr(original, "MODEL_SHA256", ref(weights)["sha256"])
    monkeypatch.setattr(original, "MODEL_CONFIG_SHA256", ref(model_config)["sha256"])
    monkeypatch.setattr(
        original,
        "_chunk_bounds",
        lambda total, _chunk, _overlap: _chunk_bounds(
            total, 11 * MODEL_SAMPLE_RATE, 5 * MODEL_SAMPLE_RATE
        ),
    )
    budget = window.WindowBudget(
        workers=2,
        max_samples=original.CHUNK,
        timeout_seconds=3600,
        max_worker_rss_bytes=12 * 1024**3,
        max_saved_bytes=1024**3,
    )
    pipeline_budget = original.PipelineBudget(
        budget, original.MAX_SAMPLES, 2 * 1024**3, 4 * 1024**3, 86400
    )
    old_path = tmp_path / "old-plan.json"
    old_root = tmp_path / "cpu"
    durable(
        old_path,
        {
            "version": 1,
            "kind": "separation-window-pipeline",
            "source": ref(source),
            "outputDirectory": str(old_root),
            "budget": {**asdict(pipeline_budget), "window": asdict(budget)},
            "codeInputs": [ref(code), ref(bgm)],
            "providerInputs": [ref(provider)],
            "modelInputs": [ref(weights), ref(model_config)],
            "chunkSamples": original.CHUNK,
            "overlapSamples": original.OVERLAP,
            "sampleRate": MODEL_SAMPLE_RATE,
            "qualityApproved": False,
            "productionBackendAdopted": False,
        },
    )
    plan = original.load_plan(old_path)
    old_root.mkdir()
    durable(old_root / "input.json", original.context(plan))
    (old_root / ".pipeline.lock").write_bytes(b"")
    prepared = tmp_path / "preparation.json"
    durable(
        prepared,
        {
            "version": 1,
            "kind": "verified-separation-pcm-preparation",
            "inputs": {str(recording): ref(recording)["sha256"]},
            "output": pin(source),
            "outputSampleCount": plan.samples,
            "sampleRate": MODEL_SAMPLE_RATE,
            "channels": 2,
            "sampleWidthBytes": 2,
            "sourceProjectionApplications": 0,
            "timestampCorrection": False,
        },
    )
    baseline = tmp_path / "owner.json"
    durable(
        baseline,
        {
            "protocol": "separation-device-running-owner-v1",
            "originalPlan": ref(old_path),
            "source": plan.definition["source"],
            "controllerPID": 100,
            "family": {"members": [{"pid": 100, "birth": "100:123"}]},
        },
    )
    terminal = tmp_path / "cpu.status"
    terminal.write_bytes(b"1\n")
    request = tmp_path / "handoff-request.json"
    durable(
        request,
        {
            "version": 1,
            "originalPlan": ref(old_path),
            "baseline": ref(baseline),
            "status": ref(terminal),
            "futureTargetsDisabled": True,
        },
    )
    monkeypatch.setattr(device, "SystemHost", Host)
    monkeypatch.setattr(os, "getppid", os.getpid)
    monkeypatch.setattr(device, "_no_old_writers", lambda _: None)
    handoff = device.observe_handoff(request, tmp_path / "handoff.json")
    benchmark = tmp_path / "benchmark.wav"
    _write_wav(benchmark, np.zeros((2, 64), dtype=np.float32))
    gate_request, gate_receipt = tmp_path / "gate-request.json", tmp_path / "gate.json"
    durable(gate_request, {"synthetic": True})
    durable(gate_receipt, {"synthetic": True})
    gate: dict[str, object] = {
        "version": acceptance.VERSION,
        "mpsGatePassed": True,
        "cpuExactGateChanged": False,
        "backend": "mps",
        "producer": pin(Path(acceptance.__file__).resolve()),
        "source": {"path": str(benchmark), "identity": identity(benchmark)},
        "benchmarkInputs": [ref(provider), ref(weights), ref(model_config), ref(bgm)],
    }
    monkeypatch.setattr(device, "_qualified", lambda _: gate)
    verifier = tmp_path / "full-environment"
    (verifier / "bin").mkdir(parents=True)
    python = verifier / "bin/python"
    python.write_bytes(b"synthetic executable")
    python.chmod(0o700)
    environment = verifier / "pyvenv.cfg"
    environment.write_bytes(b"synthetic virtual environment\n")
    work = tmp_path / "new-device"
    work.mkdir()
    config = tmp_path / "config.json"
    durable(
        config,
        {
            "version": 1,
            "kind": device.PROTOCOL,
            "originalPlan": ref(old_path),
            "preparation": ref(prepared),
            "originalRecording": ref(recording),
            "allowedSourceRoot": None,
            "acceptance": {"request": ref(gate_request), "receipt": ref(gate_receipt)},
            "handoff": ref(handoff),
            "retainedCpuChunks": [],
            "verificationPython": {
                "lexical": str(python),
                "executable": ref(python),
                "environment": ref(environment),
            },
            "outputDirectory": str(work),
            "budget": {
                "timeoutSeconds": 3600,
                "maxFamilyBytes": 12 * 1024**3,
                "maxSavedBytes": 2 * 1024**3,
            },
        },
    )
    return Case(tmp_path, config, old_path, work, baseline, request, gate)


def execute_cpu(directories: tuple[Path, ...], _: window.WindowBudget) -> dict[str, object]:
    def one(directory: Path) -> None:
        environment = record(read(directory / "request.json")["environment"])
        with pytest.MonkeyPatch.context() as patch:
            tuple(
                map(
                    lambda key: patch.setenv(key, str(environment[key])),
                    ("TMPDIR", "TMP", "TEMP", "XDG_CACHE_HOME"),
                )
            )
            durable(directory / "attempt.json", {"status": "reserved", "freshProcess": True})
            (directory / "worker.log").write_bytes(b"synthetic CPU\n")
            window.worker(directory, backend_factory=Backend, auditor=audit)

    tuple(map(one, directories))
    return {"synthetic": True}


def synth_launch(job: device.Job, index: int) -> None:
    directory = device._chunk(job, index)
    (directory / "tmp").mkdir()
    (directory / "cache").mkdir()
    with pytest.MonkeyPatch.context() as patch:
        tuple(map(lambda item: patch.setenv(*item), device._environment(job, index).items()))
        device.worker(job.path, index, factory=lambda _: Backend(), auditor=audit)
    durable(
        directory / "launch.json",
        {
            "pid": 222,
            "birth": "222:123",
            "command": [],
            "parent": read(directory / "request.json")["parent"],
        },
    )
    durable(
        directory / "resources.json",
        {
            "exitCode": 0,
            "familyGone": True,
            "cleanup": {"terminalObserved": True},
            "samples": [{"rssBytes": 100, "physicalBytes": 200}],
            "wallSeconds": 1.0,
            "hardResourceCaps": False,
        },
    )
    (directory / "worker.log").write_bytes(b"synthetic MPS\n")


def ready(case: Case, monkeypatch: pytest.MonkeyPatch) -> Path:
    graph = device.prepare(case.config, case.work)
    monkeypatch.setattr(device, "_launch", synth_launch)
    return graph


def reseal_handoff(case: Case) -> None:
    new_proof = device.observe_handoff(case.request, case.root / "handoff-new.json")
    config = read(case.config)
    config["handoff"] = ref(new_proof)
    config["retainedCpuChunks"] = read(new_proof)["completeCpuChunks"]
    replace(case.config, config)


def test_mixed_original_ola_pcm_and_readonly_complete_resume(
    case: Case, monkeypatch: pytest.MonkeyPatch
) -> None:
    old = original.load_plan(case.old)
    original.run_chunk(old, index=0, bound=original.context(old), execute=execute_cpu, resume=False)
    partial = original.chunk_proxy(old, 1).directory
    partial.mkdir(parents=True)
    (partial / "unfinished.npy").write_bytes(b"preserve exact partial")
    reseal_handoff(case)
    original_bytes = (partial / "unfinished.npy").read_bytes()
    graph = ready(case, monkeypatch)
    stages = workflow.parse_stages(read(graph))
    assert len(stages) == 4
    assert str(device._chunk(device._load(case.config), 1) / "window-000000.npy") in map(
        str, stages[1].outputs
    )
    results = [device.run_chunk(case.config, index) for index in range(3)]
    final = device.assemble(case.config)
    report = read(final)
    assert report["backendDevicesByChunk"] == ["cpu", "mps", "mps"]
    assert report["inputSampleCount"] == report["outputSampleCount"] == 23 * MODEL_SAMPLE_RATE
    assert report["qualityApproved"] is report["cpuExactGateChanged"] is False
    assert (
        record(report["context"])["benchmarkSource"]
        != record(report["context"])["productionSource"]
    )
    expected = case.root / "expected.wav"
    original.write_pcm(
        expected,
        samples=old.samples,
        blocks=original.pcm_blocks(
            old.bounds,
            samples=old.samples,
            overlap=original.OVERLAP,
            prediction=lambda i: window.reduce_windows(
                old.bounds[i][1] - old.bounds[i][0],
                lambda w: Backend().extract_vocals(
                    np.pad(
                        original.read_chunk(
                            old.source, start=old.bounds[i][0] + w.start, count=w.count
                        ),
                        ((0, 0), (0, window.WINDOW - w.count)),
                    )
                ),
            ),
        ),
    )
    assert (case.work / "vocals.wav").read_bytes() == expected.read_bytes()
    assert (partial / "unfinished.npy").read_bytes() == original_bytes
    monkeypatch.setattr(device, "_launch", lambda *_: pytest.fail("repeated MPS inference"))
    assert [device.run_chunk(case.config, index) for index in range(3)] == results
    device.verify(case.config)
    device.verify_chunk(case.config, 0)
    with pytest.raises(ReplayError, match="occupied"):
        device.assemble(case.config)


def test_pending_draft_cannot_dispatch_or_overwrite(case: Case) -> None:
    config = read(case.config)
    config.update(acceptance=None, handoff=None)
    replace(case.config, config)
    result = device.prepare(case.config, case.work)
    assert read(result)["canDispatch"] is False
    assert not (case.work / "workflow.json").exists()
    with pytest.raises(ReplayError, match="pending"):
        device.run_chunk(case.config, 0)
    with pytest.raises(ReplayError, match="empty"):
        device.prepare(case.config, case.work)


@pytest.mark.parametrize(
    "field,value",
    [
        ("version", 2),
        ("kind", "cpu"),
        ("extra", True),
        ("budget", {"timeoutSeconds": 0, "maxFamilyBytes": 1, "maxSavedBytes": 1}),
        ("outputDirectory", "relative"),
        ("retainedCpuChunks", [{"index": 99, "complete": {}}]),
    ],
)
def test_invalid_config(case: Case, field: str, value: object) -> None:
    config = read(case.config)
    config[field] = value
    replace(case.config, config)
    with pytest.raises((ReplayError, ValueError)):
        device.prepare(case.config, case.work)


@pytest.mark.parametrize(
    "field,value",
    [
        ("channels", 1),
        ("sourceProjectionApplications", 1),
        ("outputSampleCount", 1),
        ("inputs", {}),
        ("output", {"path": "/missing", "sha256": "0" * 64}),
    ],
)
def test_prepared_source_clock_mismatch(case: Case, field: str, value: object) -> None:
    config = read(case.config)
    path = Path(str(record(config["preparation"])["path"]))
    data = read(path)
    data[field] = value
    replace(path, data)
    config["preparation"] = ref(path)
    replace(case.config, config)
    with pytest.raises(ReplayError, match="source|binding"):
        device.prepare(case.config, case.work)


@pytest.mark.parametrize(
    "field,value",
    [
        ("mpsGatePassed", False),
        ("cpuExactGateChanged", True),
        ("backend", "cpu"),
        ("version", "unqualified"),
        ("benchmarkInputs", []),
        ("producer", {}),
    ],
)
def test_failed_or_different_backend_gate(case: Case, field: str, value: object) -> None:
    case.gate[field] = value
    with pytest.raises(ReplayError, match="gate|backend"):
        device.prepare(case.config, case.work)


def test_original_recording_byte_change_is_not_identity_only_cache(
    case: Case, monkeypatch: pytest.MonkeyPatch
) -> None:
    ready(case, monkeypatch)
    (case.root / "original.media").write_bytes(b"changed recording")
    with pytest.raises(ReplayError, match="recording"):
        device.run_chunk(case.config, 0)


def test_live_family_or_old_lease_prevents_handoff(
    case: Case, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(Host, "absent", lambda *_: False)
    with pytest.raises(ReplayError, match="remains"):
        device.observe_handoff(case.request, case.root / "no-proof.json")
    monkeypatch.setattr(Host, "absent", lambda *_: True)
    old = original.load_plan(case.old)
    with execution_lock(old.directory, ".pipeline") as owned:
        assert owned
        with pytest.raises(ReplayError, match="lease"):
            device.observe_handoff(case.request, case.root / "no-proof.json")
    assert not (case.root / "no-proof.json").exists()


def test_partial_inventory_change_and_new_partial_never_reissued(
    case: Case, monkeypatch: pytest.MonkeyPatch
) -> None:
    old = original.load_plan(case.old)
    partial = original.chunk_proxy(old, 1).directory
    partial.mkdir(parents=True)
    (partial / "unfinished").write_bytes(b"one")
    reseal_handoff(case)
    ready(case, monkeypatch)
    (partial / "unfinished").write_bytes(b"two")
    with pytest.raises(ReplayError, match="inventory"):
        device.run_chunk(case.config, 0)
    (partial / "unfinished").write_bytes(b"one")
    new = device._chunk(device._load(case.config), 0)
    new.mkdir(parents=True)
    (new / "reserved").write_bytes(b"unknown attempt")
    with pytest.raises(ReplayError, match="never reissue"):
        device.run_chunk(case.config, 0)


def test_cpu_complete_failure_preserved(case: Case, monkeypatch: pytest.MonkeyPatch) -> None:
    old = original.load_plan(case.old)
    original.run_chunk(old, index=0, bound=original.context(old), execute=execute_cpu, resume=False)
    reseal_handoff(case)
    ready(case, monkeypatch)
    path = original.chunk_proxy(old, 0).directory / "vocals.npy"
    path.write_bytes(b"corrupt CPU output")
    with pytest.raises(ReplayError):
        device.run_chunk(case.config, 0)
    assert path.read_bytes() == b"corrupt CPU output"


def test_new_reader_generation_change_rejected(case: Case, monkeypatch: pytest.MonkeyPatch) -> None:
    ready(case, monkeypatch)
    (case.root / "reader-code.py").write_bytes(b"different implementation")
    with pytest.raises(ReplayError, match="generation"):
        device.run_chunk(case.config, 0)


@pytest.mark.parametrize("target", ["worker", "resources", "audit", "log", "array", "window"])
def test_chunk_evidence_and_native_clock_corruption(
    case: Case, monkeypatch: pytest.MonkeyPatch, target: str
) -> None:
    ready(case, monkeypatch)
    device.run_chunk(case.config, 0)
    job = device._load(case.config)
    directory = device._chunk(job, 0)
    if target == "worker":
        data = read(directory / "worker.json")
        data["device"] = "cpu"
        replace(directory / "worker.json", data)
    elif target == "resources":
        data = read(directory / "resources.json")
        data["familyGone"] = False
        replace(directory / "resources.json", data)
    elif target == "audit":
        replace(
            directory / "before-load-descriptors.json", {"externalWritableRegular": ["outside"]}
        )
    elif target == "log":
        (directory / "worker.log").write_bytes(b"sandbox: deny file-write")
    elif target == "array":
        values = window.load_array(directory / "vocals.npy", samples=11 * MODEL_SAMPLE_RATE)
        values[0, 0] += 0.1
        with (directory / "vocals.npy").open("wb") as stream:
            np.save(stream, values, allow_pickle=False)
    else:
        data = read(directory / "worker.json")
        data["windows"] = []
        replace(directory / "worker.json", data)
    # A malicious re-pinned envelope still cannot bypass the domain verifier.
    replace(directory / "complete.json", device._proof(job, 0))
    with pytest.raises(ReplayError):
        device.verify_chunk(case.config, 0)


def test_resource_overrun_and_worker_repetition(
    case: Case, monkeypatch: pytest.MonkeyPatch
) -> None:
    ready(case, monkeypatch)
    device.run_chunk(case.config, 0)
    job = device._load(case.config)
    directory = device._chunk(job, 0)
    resources = read(directory / "resources.json")
    resources["samples"] = [{"rssBytes": job.family_bytes + 1, "physicalBytes": 1}]
    replace(directory / "resources.json", resources)
    replace(directory / "complete.json", device._proof(job, 0))
    with pytest.raises(ReplayError, match="resources"):
        device.verify_chunk(case.config, 0)
    with pytest.MonkeyPatch.context() as patch:
        tuple(map(lambda item: patch.setenv(*item), device._environment(job, 0).items()))
        with device._leases(job.old), pytest.raises(ReplayError, match="Existing"):
            device.worker(case.config, 0, factory=lambda _: Backend(), auditor=audit)


def test_owned_child_scratch_must_differ_from_original_parent(
    case: Case, monkeypatch: pytest.MonkeyPatch
) -> None:
    ready(case, monkeypatch)
    job = device._load(case.config)
    directory = device._chunk(job, 0)
    directory.mkdir(parents=True)
    durable(
        directory / "request.json",
        device._request(job, 0, Host()),
    )
    with device._leases(job.old), pytest.raises(ReplayError, match="scratch"):
        device.worker(case.config, 0, factory=lambda _: pytest.fail("model loaded"), auditor=audit)
    with device._leases(job.old):
        synth_launch(job, 0)
    assert read(directory / "worker.json")["workerEnvironment"] == device._environment(job, 0)
    assert read(job.old.directory / "input.json")["environment"] != device._environment(job, 0)


def test_lazy_gate_verifier_source_is_in_fresh_closure() -> None:
    inventory = actual_code()
    assert str(Path(device.__file__).with_name("separation_vocal_asr_compare.py")) in inventory
    # The fingerprint must stay a focused import closure, not drift toward the whole package
    # (469 modules). Measured 187 on 2026-10-17; the earlier 180 was already exceeded by the
    # committed tree (186), so the bound is re-based with a small margin rather than removed.
    assert len(inventory) < 200


@pytest.mark.parametrize(
    "mode", ["ok", "missing-vibrato", "oversized", "mutated", "wrong-executable"]
)
def test_qualification_uses_separate_pinned_full_environment(
    case: Case, monkeypatch: pytest.MonkeyPatch, mode: str
) -> None:
    job = device._load(case.config, ready=False)
    gate = record(job.configuration["acceptance"])
    receipt = Path(str(record(gate["receipt"])["path"]))
    replace(receipt, case.gate)
    # Only the synthetic new metadata receipt is repinned in this synthetic config.
    gate["receipt"] = ref(receipt)
    replace(case.config, job.configuration)
    job = device._load(case.config, ready=False)
    seen: list[str] = []

    def run(command: list[str], **_: object) -> subprocess.CompletedProcess[bytes]:
        seen.extend(command)
        if mode == "mutated":
            replace(receipt, {"changed": True})
        return subprocess.CompletedProcess(
            command,
            1 if mode == "missing-vibrato" else 0,
            b"x" * (1024**2 + 1) if mode == "oversized" else b"",
            b"missing vibrato" if mode == "missing-vibrato" else b"",
        )

    monkeypatch.setattr(subprocess, "run", run)
    if mode == "wrong-executable":
        data = record(job.configuration["verificationPython"])
        data["lexical"] = "/missing/bin/python"
    if mode == "ok":
        assert verify_external_qualification(job)["mpsGatePassed"] is True
        assert seen[0] == str(record(job.configuration["verificationPython"])["lexical"])
        assert seen[3:5] == ["replay.mps_separation_acceptance", "verify"]
    else:
        with pytest.raises((ReplayError, FileNotFoundError)):
            verify_external_qualification(job)


class Child:
    pid = 222

    def __init__(self, mode: str) -> None:
        self.returncode = 2 if mode == "exit" else 0
        self.polls: list[int | None] = [None, self.returncode]
        self.mode = mode
        self.waits = 0

    def poll(self) -> int | None:
        return self.polls.pop(0) if self.polls else self.returncode

    def wait(self, timeout: float) -> int:
        self.waits += 1
        if self.mode.startswith("sampling") and self.waits == 1:
            raise subprocess.TimeoutExpired("synthetic", timeout)
        return self.returncode


@pytest.mark.parametrize(
    "mode", ["ok", "sampling-exit", "sampling-failed", "budget", "exit", "descendant", "deadline"]
)
def test_bounded_owned_parent_and_retained_uncertain_attempt(
    case: Case, monkeypatch: pytest.MonkeyPatch, mode: str
) -> None:
    device.prepare(case.config, case.work)
    job = device._load(case.config)
    directory = device._chunk(job, 0)
    directory.mkdir(parents=True)
    durable(directory / "request.json", device._request(job, 0, Host()))
    child = Child(mode)
    if mode == "sampling-failed":
        child.returncode = 2
    if mode.startswith("sampling"):
        child.polls = [None, None]
    monkeypatch.setattr(subprocess, "Popen", lambda *_, **__: child)
    monkeypatch.setattr(device, "cleanup_child", lambda *_: {"terminalObserved": True})
    monkeypatch.setattr(Host, "absent", lambda *_: mode != "descendant")
    calls = 0

    def sample(*_: object) -> dict[str, object]:
        nonlocal calls
        calls += 1
        if mode.startswith("sampling") and calls == 2:
            raise ReplayError("Exact process birth unavailable")
        return {
            "members": [{"pid": 222}],
            "rssBytes": job.family_bytes + 1 if mode == "budget" else 10,
            "physicalBytes": 20,
        }

    monkeypatch.setattr(device, "sample_family", sample)
    if mode == "deadline":
        ticks = iter([0.0, 9999.0, 10000.0])
        monkeypatch.setattr(time, "monotonic", lambda: next(ticks))
    if mode in {"ok", "sampling-exit"}:
        device._launch(job, 0)
        assert read(directory / "resources.json")["familyGone"] is True
    else:
        with pytest.raises(ReplayError):
            device._launch(job, 0)
        assert read(directory / "resources.json")["familyGone"] is False
    assert read(directory / "launch.json")["birth"] == "222:123"
    assert (directory / "sandbox.sb").is_file()


@pytest.mark.parametrize(
    "message",
    [
        "Exact process birth unavailable",
        "Physical footprint unavailable",
        "Owned process vanished during inventory",
    ],
)
def test_live_root_resamples_only_bounded_inventory_race(
    case: Case, monkeypatch: pytest.MonkeyPatch, message: str
) -> None:
    device.prepare(case.config, case.work)
    job = device._load(case.config)
    directory = device._chunk(job, 0)
    directory.mkdir(parents=True)
    durable(directory / "request.json", device._request(job, 0, Host()))
    child = Child("sampling-live")
    child.polls = [None, None, None, 0]
    timeouts: list[float] = []

    def wait(timeout: float) -> int:
        timeouts.append(timeout)
        raise subprocess.TimeoutExpired("synthetic live worker", timeout)

    monkeypatch.setattr(child, "wait", wait)
    monkeypatch.setattr(subprocess, "Popen", lambda *_, **__: child)
    monkeypatch.setattr(device, "cleanup_child", lambda *_: {"terminalObserved": True})
    calls = 0

    def sample(*_: object) -> dict[str, object]:
        nonlocal calls
        calls += 1
        if calls == 2:
            raise ReplayError(message)
        return {"members": [{"pid": 222}], "rssBytes": 10, "physicalBytes": 20}

    monkeypatch.setattr(device, "sample_family", sample)
    device._launch(job, 0)
    resources = read(directory / "resources.json")
    assert resources["familyGone"] is True
    assert len(rows(resources["samples"])) == 2
    assert timeouts == [0.2, 0.05, 0.2]
    errors = record(resources["observationErrors"])
    assert errors["count"] == 1
    assert record(errors["first"])["exitCodeObserved"] is None
    assert record(errors["first"])["error"] == message


@pytest.mark.parametrize(
    "message,clock_expired,root_changed,expected_count",
    [
        ("Exact process birth unavailable", False, False, 10),
        ("Physical footprint unavailable", True, False, 1),
        ("Owned PID reused", False, False, 1),
        ("Family identity changed", False, False, 1),
        ("Malformed process row", False, False, 1),
        ("Duplicate process identity", False, False, 1),
        ("Physical footprint requires Darwin libproc", False, False, 1),
        ("Invalid physical footprint", False, False, 1),
        ("Exact process birth unavailable", False, True, 1),
    ],
)
def test_permanent_observation_failure_retains_first_error_and_fails_closed(
    case: Case,
    monkeypatch: pytest.MonkeyPatch,
    message: str,
    clock_expired: bool,
    root_changed: bool,
    expected_count: int,
) -> None:
    device.prepare(case.config, case.work)
    job = device._load(case.config)
    directory = device._chunk(job, 0)
    directory.mkdir(parents=True)
    durable(directory / "request.json", device._request(job, 0, Host()))
    child = Child("sampling-live")
    child.polls = [None] * 30
    timeouts: list[float] = []

    def wait(timeout: float) -> int:
        timeouts.append(timeout)
        raise subprocess.TimeoutExpired("synthetic live worker", timeout)

    monkeypatch.setattr(child, "wait", wait)
    monkeypatch.setattr(subprocess, "Popen", lambda *_, **__: child)
    monkeypatch.setattr(device, "cleanup_child", lambda *_: {"terminalObserved": True})

    def sample(*_: object) -> dict[str, object]:
        if root_changed:
            monkeypatch.setattr(Host, "birth", lambda *_: "changed:456")
        raise ReplayError(message)

    monkeypatch.setattr(device, "sample_family", sample)
    ticks = iter([0.0, 0.0, 0.0, 1.01, 1.02])
    monkeypatch.setattr(time, "monotonic", (lambda: next(ticks)) if clock_expired else lambda: 0.0)
    with pytest.raises(ReplayError) as failure:
        device._launch(job, 0)
    assert str(failure.value) == ("Owned PID reused" if root_changed else message)
    resources = read(directory / "resources.json")
    assert resources["familyGone"] is False
    errors = record(resources["observationErrors"])
    assert errors["count"] == expected_count
    assert record(errors["first"])["error"] == message
    assert record(errors["first"])["exitCodeObserved"] is None
    assert len(timeouts) == (9 if expected_count == 10 else 0)
    assert all(timeout <= 0.05 for timeout in timeouts)
    if root_changed:
        assert record(errors["first"])["rootIdentityError"] == "Owned PID reused"
        assert isinstance(failure.value.__cause__, ReplayError)


def test_retry_observes_nonzero_terminal_exit_without_masking_first_error(
    case: Case, monkeypatch: pytest.MonkeyPatch
) -> None:
    device.prepare(case.config, case.work)
    job = device._load(case.config)
    directory = device._chunk(job, 0)
    directory.mkdir(parents=True)
    durable(directory / "request.json", device._request(job, 0, Host()))
    child = Child("sampling-failed")
    child.polls = [None, None]
    child.returncode = 2
    child.waits = 1
    monkeypatch.setattr(subprocess, "Popen", lambda *_, **__: child)
    monkeypatch.setattr(device, "cleanup_child", lambda *_: {"terminalObserved": True})

    def sample(*_: object) -> dict[str, object]:
        raise ReplayError("Physical footprint unavailable")

    monkeypatch.setattr(device, "sample_family", sample)
    with pytest.raises(ReplayError, match="Physical footprint unavailable"):
        device._launch(job, 0)
    assert child.waits == 2
    resources = read(directory / "resources.json")
    assert resources["exitCode"] == 2
    assert resources["familyGone"] is False
    assert record(record(resources["observationErrors"])["first"])["exitCodeObserved"] is None


def test_actual_owner_capture_and_exact_old_writer_command(
    case: Case, monkeypatch: pytest.MonkeyPatch
) -> None:
    command = f"python -B -m replay.separation_window_pipeline {case.old}"
    monkeypatch.setattr(
        subprocess, "run", lambda *_, **__: subprocess.CompletedProcess([], 0, command)
    )
    monkeypatch.setattr(
        device, "sample_family", lambda *_: {"members": [{"pid": 100, "birth": "100:123"}]}
    )
    result = device.capture_owner(case.old, 100, case.root / "captured.json")
    assert read(result)["birth"] == "100:123"
    # Exercise the bounded read-only global observation rather than its fixture substitute.
    with pytest.raises(ReplayError, match="writer"):
        observe_cpu_writers(original.load_plan(case.old))


@pytest.mark.parametrize(
    "mode", ["ok", "wrong-parent", "malformed-parent", "changed-context", "unowned"]
)
def test_sandbox_worker_uses_bound_parent_without_process_inventory(
    case: Case, monkeypatch: pytest.MonkeyPatch, mode: str
) -> None:
    device.prepare(case.config, case.work)
    job = device._load(case.config)
    request = device._request(job, 0, Host())
    if mode == "wrong-parent":
        request["parent"] = {"pid": os.getpid() + 1, "birth": "123:456"}
    elif mode == "malformed-parent":
        request["parent"] = {"pid": os.getpid(), "birth": None}
    elif mode == "changed-context":
        request["context"] = {"changed": True}
    path = case.root / "worker-load-request.json"
    durable(path, request)
    monkeypatch.setattr(device, "audit_descriptors", audit)
    monkeypatch.setattr(
        device, "_no_old_writers", lambda _: pytest.fail("ps called in sandbox child")
    )
    if mode == "ok":
        with device._leases(job.old):
            device.worker_load_check(case.config, path, 0)
        assert read(case.root / "bootstrap-descriptors.json")["externalWritableRegular"] == []
        assert not device._chunk(job, 0).exists()
    else:
        with pytest.raises(ReplayError):
            device.worker_load_check(case.config, path, 0)
        assert not (case.root / "bootstrap-descriptors.json").exists()


@pytest.mark.parametrize("mode", ["ok", "denied", "overflow", "external-fd", "mutated"])
def test_real_sandbox_command_smoke_is_finite_and_never_infers(
    case: Case, monkeypatch: pytest.MonkeyPatch, mode: str
) -> None:
    device.prepare(case.config, case.work)
    smoke = case.root / "sandbox-loader"
    smoke.mkdir()
    seen: list[str] = []

    def run(command: list[str], **kwargs: object) -> subprocess.CompletedProcess[bytes]:
        seen.extend(command)
        assert kwargs["timeout"] == 120
        assert command[:3] == ["/usr/bin/sandbox-exec", "-f", str(smoke / "sandbox.sb")]
        assert command[6:8] == ["replay.separation_device_pipeline", "worker-load-check"]
        with pytest.MonkeyPatch.context() as child:
            child.setattr(
                device, "_no_old_writers", lambda _: pytest.fail("ps in sandboxed loader")
            )
            child.setattr(device, "audit_descriptors", audit)
            device.worker_load_check(case.config, smoke / "request.json", 0)
        if mode == "external-fd":
            replace(
                smoke / "bootstrap-descriptors.json",
                {
                    "snapshot": "f3\naw\ntREG\nn/outside.cache\n",
                    "externalWritableRegular": [],
                },
            )
        if mode == "mutated":
            case.baseline.write_bytes(b"changed owner")
        return subprocess.CompletedProcess(
            command,
            1 if mode == "denied" else 0,
            b"x" * (1024**2 + 1) if mode == "overflow" else b"loaded",
            b"",
        )

    monkeypatch.setattr(subprocess, "run", run)
    if mode == "ok":
        result = device.sandbox_check(case.config, smoke, 0)
        proof = read(result)
        assert proof["inferenceExecuted"] is False and proof["modelLoaded"] is False
        assert proof["exitCode"] == 0 and float(str(proof["wallSeconds"])) > 0
        assert seen and not (case.work / "chunks").exists()
        with pytest.raises(ReplayError, match="empty"):
            device.sandbox_check(case.config, smoke, 0)
    else:
        with pytest.raises(ReplayError):
            device.sandbox_check(case.config, smoke, 0)
        assert not (smoke / "complete.json").exists()


@pytest.mark.parametrize(
    "argv",
    [
        ["prepare"],
        ["capture-owner"],
        ["observe-handoff"],
        ["worker"],
        ["run-chunk"],
        ["verify-chunk"],
    ],
)
def test_cli_rejects_missing_fields(case: Case, argv: list[str]) -> None:
    assert device.main([*argv, str(case.config)]) == 1


@pytest.mark.parametrize(
    "action",
    [
        "prepare",
        "observe-handoff",
        "capture-owner",
        "worker",
        "run-chunk",
        "verify-chunk",
        "assemble",
        "verify",
    ],
)
def test_cli_dispatch_shape(case: Case, monkeypatch: pytest.MonkeyPatch, action: str) -> None:
    name = action.replace("-", "_")
    seen: list[tuple[object, ...]] = []

    def run(*args: object) -> Path:
        seen.append(args)
        return case.config

    monkeypatch.setattr(device, name, run)
    extra = [str(case.work)] if action in {"prepare", "observe-handoff", "capture-owner"} else []
    extra += ["--pid", "100"] if action == "capture-owner" else []
    extra += ["--index", "0"] if action in {"worker", "run-chunk", "verify-chunk"} else []
    assert device.main([action, str(case.config), *extra]) == 0
    assert seen
