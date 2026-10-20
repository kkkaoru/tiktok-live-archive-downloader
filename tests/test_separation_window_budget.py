"""Synthetic independent windows, complete-only reuse, and owned child guards."""

import importlib.metadata
import json
import subprocess
import time
from contextlib import AbstractContextManager, nullcontext
from dataclasses import asdict
from pathlib import Path

import numpy as np
import pytest
from numpy.typing import NDArray

from replay import separation_window_budget as window
from replay.audio_cpu_probe import sha256
from replay.bgm_separation import (
    MODEL_NAME,
    MODEL_REVISION,
    SeparationConfig,
    _MlxBackend,
    _write_wav,
    separate_vocals,
)
from replay.code_dependency_inventory import Source
from replay.core import ReplayError
from replay.native_invocation import durable, identity, pin, read
from replay.recordings import record


class FakeMX:
    def stream(self, device: str) -> AbstractContextManager[None]:
        assert device == "cpu"
        return nullcontext()

    def array(self, value: object) -> NDArray[np.float32]:
        return np.asarray(value, dtype=np.float32)

    def eval(self, value: object) -> None:
        assert isinstance(value, np.ndarray)


class OriginalSynthetic(_MlxBackend):
    def __init__(self) -> None:
        self._mx = FakeMX()
        self._device = "cpu"
        self._model = self.forward

    def forward(self, value: object) -> NDArray[np.float32]:
        audio = np.asarray(value, dtype=np.float32)
        return audio * np.float32(0.25) + audio.mean(axis=-1, keepdims=True, dtype=np.float32)


class FakeChild:
    pid = 999

    def __init__(self, code: int | None = 0) -> None:
        self.code = code
        self.terminated = False
        self.killed = False
        self.wait_calls = 0

    def poll(self) -> int | None:
        return self.code

    def terminate(self) -> None:
        self.terminated = True
        self.code = 0

    def kill(self) -> None:
        self.killed = True
        self.code = 0

    def wait(self, timeout: int) -> int:
        self.wait_calls += 1
        return 0


class FakeDistribution:
    version = "test-version"

    def __init__(self, directory: Path, files: list[importlib.metadata.PackagePath] | None) -> None:
        self.directory = directory
        self.files = files

    def locate_file(self, path: importlib.metadata.PackagePath) -> Path:
        return self.directory / path


@pytest.fixture
def local(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> Path:
    monkeypatch.chdir(tmp_path)
    work = tmp_path / "edits/work"
    work.mkdir(parents=True)
    cache = work / "parent-cache"
    cache.mkdir()
    monkeypatch.setenv("TMPDIR", str(cache))
    monkeypatch.setenv("TMP", str(cache))
    monkeypatch.setenv("TEMP", str(cache))
    monkeypatch.setenv("XDG_CACHE_HOME", str(cache))
    monkeypatch.setenv("HF_HOME", str(cache))
    monkeypatch.setenv("HF_HUB_CACHE", str(cache))
    monkeypatch.setenv("HF_HUB_OFFLINE", "1")
    monkeypatch.setenv("TRANSFORMERS_OFFLINE", "1")
    monkeypatch.setattr(window, "provider_pins", lambda: {"fakeProvider": "test-only"})
    return work


@pytest.fixture
def budget() -> window.WindowBudget:
    return window.WindowBudget(
        workers=2,
        max_samples=2646000,
        timeout_seconds=600,
        max_worker_rss_bytes=12884901888,
        max_saved_bytes=100000000,
    )


@pytest.fixture
def execute(monkeypatch: pytest.MonkeyPatch) -> window.Execute:
    def one(directory: Path) -> None:
        environment = read(directory / "request.json")["environment"]
        assert isinstance(environment, dict)
        with monkeypatch.context() as patch:
            patch.setenv("TMPDIR", environment["TMPDIR"])
            patch.setenv("TMP", environment["TMP"])
            patch.setenv("TEMP", environment["TEMP"])
            patch.setenv("XDG_CACHE_HOME", environment["XDG_CACHE_HOME"])
            window.worker(
                directory, backend_factory=OriginalSynthetic, auditor=lambda path, phase: None
            )

    def run(
        directories: tuple[Path, ...], selected_budget: window.WindowBudget
    ) -> dict[str, object]:
        tuple(map(one, reversed(directories)))
        return {"fake": True}

    return run


@pytest.mark.parametrize("samples", [1, 176399, 176400, 176401, 352800, 352801, 529207, 705600])
def test_exact_original_stereo_and_tail(
    samples: int, local: Path, budget: window.WindowBudget, execute: window.Execute
) -> None:
    audio = np.stack((np.linspace(-0.75, 0.5, samples), np.linspace(0.25, -0.5, samples))).astype(
        np.float32
    )
    actual = window.run_windows(audio, local / "run", budget=budget, execute=execute)
    np.testing.assert_array_equal(actual, OriginalSynthetic().extract_vocals(audio))
    assert actual.shape == (2, samples)
    assert actual.dtype == np.dtype(np.float32)


def test_original_backend_injection_preserves_saved_pcm(
    local: Path, budget: window.WindowBudget, execute: window.Execute
) -> None:
    source = local / "source.wav"
    _write_wav(
        source,
        np.stack((np.linspace(-0.5, 0.5, 705600), np.linspace(0.3, -0.3, 705600))).astype(
            np.float32
        ),
    )
    adapter = window.WindowBackend(local / "adapter", budget=budget, execute=execute)
    separate_vocals(
        source, local / "original.wav", SeparationConfig(device="cpu"), OriginalSynthetic()
    )
    separate_vocals(source, local / "parallel.wav", SeparationConfig(device="cpu"), adapter)
    assert (local / "original.wav").read_bytes() == (local / "parallel.wav").read_bytes()
    assert adapter.sources == ("vocals", "instrumental")
    assert adapter.chunk_index == 1


def test_complete_workers_reused_without_execution(
    local: Path, budget: window.WindowBudget, execute: window.Execute
) -> None:
    audio = np.ones((2, 352801), dtype=np.float32)
    window.run_windows(audio, local / "run", budget=budget, execute=execute)
    request = (local / "run/worker-00/request.json").read_bytes()
    receipt = (local / "run/result.json").read_bytes()
    result = window.run_windows(
        audio,
        local / "run",
        budget=budget,
        execute=lambda paths, selected: pytest.fail("unexpected inference"),
    )
    np.testing.assert_array_equal(result, OriginalSynthetic().extract_vocals(audio))
    assert (local / "run/worker-00/request.json").read_bytes() == request
    assert (local / "run/result.json").read_bytes() == receipt


def test_changed_complete_payload_refused(
    local: Path, budget: window.WindowBudget, execute: window.Execute
) -> None:
    window.run_windows(
        np.zeros((2, 2), dtype=np.float32), local / "run", budget=budget, execute=execute
    )
    with pytest.raises(ReplayError, match="input changed"):
        window.run_windows(
            np.ones((2, 2), dtype=np.float32), local / "run", budget=budget, execute=execute
        )


def test_changed_output_refused(
    local: Path, budget: window.WindowBudget, execute: window.Execute
) -> None:
    audio = np.zeros((2, 2), dtype=np.float32)
    window.run_windows(audio, local / "run", budget=budget, execute=execute)
    (local / "run/worker-00/window-000000.npy").write_bytes(b"altered")
    with pytest.raises(ReplayError, match="completion changed"):
        window.run_windows(audio, local / "run", budget=budget, execute=execute)


@pytest.mark.parametrize(
    "change",
    [
        {"workers": 3},
        {"workers": True},
        {"max_samples": 158760001},
        {"timeout_seconds": 3601},
        {"max_saved_bytes": 0},
    ],
)
def test_invalid_budget(change: dict[str, int]) -> None:
    values = {
        "workers": 2,
        "max_samples": 2646000,
        "timeout_seconds": 600,
        "max_worker_rss_bytes": 12884901888,
        "max_saved_bytes": 100000000,
    }
    with pytest.raises(ReplayError):
        window.WindowBudget(**{**values, **change})


@pytest.mark.parametrize("samples", [0, -1, True, 158760001])
def test_invalid_window_clock(samples: int) -> None:
    with pytest.raises(ReplayError):
        window.windows(samples)


def test_exact_window_plan_and_storage() -> None:
    assert window.windows(352801) == (window.Window(0, 0, 352800), window.Window(1, 176400, 176401))
    assert window.estimated_saved_bytes(352801) == 25256712


@pytest.mark.parametrize(
    "audio",
    [
        np.zeros((1, 2), dtype=np.float32),
        np.zeros((2, 0), dtype=np.float32),
        np.zeros((2, 2), dtype=np.float64),
        np.asarray([[float("nan")], [0]], dtype=np.float32),
    ],
)
def test_invalid_input(
    audio: NDArray[np.float32], local: Path, budget: window.WindowBudget, execute: window.Execute
) -> None:
    with pytest.raises(ReplayError, match="bounded stereo"):
        window.run_windows(audio, local / "run", budget=budget, execute=execute)


def test_disk_budget_before_dispatch(local: Path, execute: window.Execute) -> None:
    budget = window.WindowBudget(
        workers=2, max_samples=2, timeout_seconds=1, max_worker_rss_bytes=1, max_saved_bytes=1
    )
    with pytest.raises(ReplayError, match="insufficient"):
        window.run_windows(
            np.zeros((2, 2), dtype=np.float32), local / "run", budget=budget, execute=execute
        )
    assert not (local / "run").exists()


def test_nonlocal_and_linked_output(local: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    linked = local / "linked"
    linked.symlink_to(local / "parent-cache")
    with pytest.raises(ReplayError, match="linked"):
        window.local_directory(linked)
    with pytest.raises(ReplayError, match="local"):
        window.local_directory(local.parent / "outside")


@pytest.mark.parametrize(
    "prediction",
    [
        np.zeros((2, 352800), dtype=np.float64),
        np.zeros((1, 352800), dtype=np.float32),
        np.full((2, 352800), np.nan, dtype=np.float32),
    ],
)
def test_invalid_prediction(prediction: NDArray[np.float32]) -> None:
    with pytest.raises(ReplayError, match="prediction"):
        window.reduce_windows(2, lambda selected: prediction)


def test_ordered_reduction() -> None:
    observed: list[int] = []

    def prediction(selected: window.Window) -> NDArray[np.float32]:
        observed.append(selected.index)
        return np.zeros((2, 352800), dtype=np.float32)

    window.reduce_windows(705600, prediction)
    assert observed == [0, 1, 2]


def test_worker_invalid_plan() -> None:
    with pytest.raises(ReplayError, match="clock changed"):
        window.selected_windows({"samples": 2, "windows": [{"index": 0, "start": 1, "count": 2}]})
    with pytest.raises(ReplayError, match="Unknown"):
        window.selected_windows({"samples": 2, "windows": [{"index": 1}]})
    with pytest.raises(ReplayError, match="Duplicate"):
        window.selected_windows({"samples": 2, "windows": []})
    with pytest.raises(ReplayError, match="sample count"):
        window.selected_windows({"samples": True})


def test_worker_model_pin_changed(
    local: Path, budget: window.WindowBudget, execute: window.Execute
) -> None:
    audio = np.zeros((2, 2), dtype=np.float32)
    window.run_windows(audio, local / "run", budget=budget, execute=execute)
    request = read(local / "run/worker-00/request.json")
    request["modelRevision"] = "changed"
    (local / "run/worker-00/request.json").unlink()
    durable(local / "run/worker-00/request.json", request)
    with pytest.raises(ReplayError, match="pins changed"):
        window.verify_worker(local / "run/worker-00")


def test_instrumental_complement(
    local: Path, budget: window.WindowBudget, execute: window.Execute
) -> None:
    audio = np.zeros((2, 2), dtype=np.float32)
    backend = window.WindowBackend(local / "backend", budget=budget, execute=execute)
    np.testing.assert_array_equal(
        backend.extract_instrumental(audio), np.zeros((2, 2), dtype=np.float32)
    )


def test_array_link_or_wrong_shape(local: Path) -> None:
    payload = local / "payload.npy"
    window.save_array(payload, np.zeros((2, 2), dtype=np.float32))
    with pytest.raises(ReplayError, match="Invalid array"):
        window.load_array(payload, samples=3)
    linked = local / "linked.npy"
    linked.symlink_to(payload)
    with pytest.raises(ReplayError, match="linked"):
        window.load_array(linked, samples=2)


def test_owned_child_success(
    local: Path, budget: window.WindowBudget, monkeypatch: pytest.MonkeyPatch
) -> None:
    directory = local / "worker"
    directory.mkdir()
    durable(directory / "request.json", {"environment": {"TMPDIR": str(local)}})
    observed: list[object] = []

    def spawn(command: object, **kwargs: object) -> FakeChild:
        observed.append(kwargs.get("stdin"))
        observed.append(kwargs.get("env"))
        return FakeChild()

    monkeypatch.setattr(subprocess, "Popen", spawn)
    monkeypatch.setattr(window, "child_rss", lambda pid: 100)
    monkeypatch.setattr(window, "verify_worker", lambda path: {"kernelPeakRSSBytes": 100})
    result = window.execute_children((directory,), budget)
    assert result["sampledWorkerSumPeakRSSBytes"] == 100
    assert result["hardCap"] is False
    assert observed[0] == subprocess.DEVNULL
    assert isinstance(observed[1], dict)


@pytest.mark.parametrize("mode", ["rss", "failed", "deadline", "disk"])
def test_owned_child_guard_stops_only_owned(
    local: Path, budget: window.WindowBudget, monkeypatch: pytest.MonkeyPatch, mode: str
) -> None:
    directory = local / "worker"
    directory.mkdir()
    durable(directory / "request.json", {"environment": {"TMPDIR": str(local)}})
    child = FakeChild(1 if mode == "failed" else None)
    monkeypatch.setattr(subprocess, "Popen", lambda *args, **kwargs: child)
    monkeypatch.setattr(window, "child_rss", lambda pid: 12884901889 if mode == "rss" else 100)
    monkeypatch.setattr(window, "saved_bytes", lambda path: 100000001 if mode == "disk" else 0)
    clock = iter([0, 601] if mode == "deadline" else [0, 0])
    monkeypatch.setattr(time, "monotonic", lambda: next(clock))
    with pytest.raises(ReplayError):
        window.execute_children((directory,), budget)
    assert child.terminated is (mode != "failed")


def test_stop_owned_child_kills_after_timeout(monkeypatch: pytest.MonkeyPatch) -> None:
    child = FakeChild(None)
    calls = iter([subprocess.TimeoutExpired("owned", 5), 0])

    def wait(timeout: int) -> int:
        result = next(calls)
        if isinstance(result, subprocess.TimeoutExpired):
            raise result
        assert isinstance(result, int)
        return result

    monkeypatch.setattr(child, "wait", wait)
    window.stop_child(child)
    assert child.killed is True


def test_uncertain_worker_not_restarted(local: Path, budget: window.WindowBudget) -> None:
    directory = local / "worker"
    directory.mkdir()
    durable(directory / "attempt.json", {"status": "reserved"})
    with pytest.raises(ReplayError, match="Uncertain"):
        window.execute_children((directory,), budget)


def test_child_rss_parsing(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        subprocess,
        "run",
        lambda *args, **kwargs: subprocess.CompletedProcess([], 0, "123\n", ""),
    )
    assert window.child_rss(1) == 125952
    monkeypatch.setattr(
        subprocess, "run", lambda *args, **kwargs: subprocess.CompletedProcess([], 1, "", "")
    )
    assert window.child_rss(1) == 0
    monkeypatch.setattr(
        subprocess,
        "run",
        lambda *args, **kwargs: subprocess.CompletedProcess([], 0, "bad", ""),
    )
    with pytest.raises(ReplayError, match="Invalid"):
        window.child_rss(1)


def test_cli_routes_and_failure(local: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    called: list[Path] = []
    monkeypatch.setattr(window, "worker", lambda path: called.append(path))
    assert window.main(["--worker", str(local)]) == 0
    assert called == [local]
    assert window.main([]) == 1


def test_probe_synthetic(
    local: Path,
    budget: window.WindowBudget,
    execute: window.Execute,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source = local / "source.wav"
    _write_wav(source, np.zeros((2, 352801), dtype=np.float32))
    original = window.run_windows

    def run(
        audio: NDArray[np.float32], path: Path, *, budget: window.WindowBudget
    ) -> NDArray[np.float32]:
        return original(audio, path, budget=budget, execute=execute)

    monkeypatch.setattr(window, "run_windows", run)
    window.probe(source, local / "probe", samples=352801, budget=budget)
    result = read(local / "probe/comparison.json")
    assert result["sampleCount"] == 352801
    assert result["qualityApproved"] is False
    assert (local / "probe/serial.wav").read_bytes() == (local / "probe/parallel.wav").read_bytes()


def test_provider_metadata_pins(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    metadata = tmp_path / "METADATA"
    metadata.write_text("test package\n", encoding="utf-8")
    package = FakeDistribution(tmp_path, [importlib.metadata.PackagePath("METADATA")])
    monkeypatch.setattr(importlib.metadata, "distribution", lambda name: package)
    result = window.provider_pins()
    assert result["mlxVersion"] == "test-version"
    assert (
        result["mlxMetadataSHA256"]
        == "d400a577a056c0530d6acfc0e0a2f937be15323bffe02dfc81a8ca66b332a907"
    )
    assert result["pythonLexical"]


@pytest.mark.parametrize("files", [None, []])
def test_provider_metadata_missing(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    files: list[importlib.metadata.PackagePath] | None,
) -> None:
    monkeypatch.setattr(
        importlib.metadata, "distribution", lambda name: FakeDistribution(tmp_path, files)
    )
    with pytest.raises(ReplayError, match="metadata"):
        window.provider_pins()


def test_provider_package_missing(monkeypatch: pytest.MonkeyPatch) -> None:
    def missing(name: str) -> importlib.metadata.Distribution:
        raise importlib.metadata.PackageNotFoundError(name)

    monkeypatch.setattr(importlib.metadata, "distribution", missing)
    with pytest.raises(ReplayError, match="not installed"):
        window.provider_pins()


def test_parent_complete_quality_marker_changed(
    local: Path, budget: window.WindowBudget, execute: window.Execute
) -> None:
    audio = np.zeros((2, 2), dtype=np.float32)
    window.run_windows(audio, local / "run", budget=budget, execute=execute)
    result = read(local / "run/result.json")
    result["qualityApproved"] = True
    (local / "run/result.json").unlink()
    durable(local / "run/result.json", result)
    with pytest.raises(ReplayError, match="completion changed"):
        window.run_windows(audio, local / "run", budget=budget, execute=execute)


def test_completed_worker_never_repeats_model_load(
    local: Path, budget: window.WindowBudget, execute: window.Execute
) -> None:
    window.run_windows(
        np.zeros((2, 2), dtype=np.float32), local / "run", budget=budget, execute=execute
    )
    with pytest.raises(ReplayError, match="never repeat inference"):
        window.worker(
            local / "run/worker-00",
            backend_factory=lambda: pytest.fail("model loaded again"),
            auditor=lambda path, phase: None,
        )


def test_worker_payload_cannot_escape_owner(
    local: Path, budget: window.WindowBudget, execute: window.Execute
) -> None:
    window.run_windows(
        np.zeros((2, 2), dtype=np.float32), local / "run", budget=budget, execute=execute
    )
    path = local / "run/worker-00/request.json"
    request = read(path)
    request["payload"] = str(local / "elsewhere.npy")
    path.unlink()
    durable(path, request)
    with pytest.raises(ReplayError, match="escapes"):
        window.verify_worker(local / "run/worker-00")


def test_kernel_peak_guard_after_completion(
    local: Path, budget: window.WindowBudget, monkeypatch: pytest.MonkeyPatch
) -> None:
    directory = local / "worker"
    directory.mkdir()
    durable(directory / "request.json", {"environment": {"TMPDIR": str(local)}})
    monkeypatch.setattr(subprocess, "Popen", lambda *args, **kwargs: FakeChild())
    monkeypatch.setattr(window, "child_rss", lambda pid: 100)
    monkeypatch.setattr(window, "verify_worker", lambda path: {"kernelPeakRSSBytes": 12884901889})
    with pytest.raises(ReplayError, match="Kernel"):
        window.execute_children((directory,), budget)


def test_cli_probe_configuration(local: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    called: list[int] = []

    def probe(source: Path, directory: Path, *, samples: int, budget: window.WindowBudget) -> None:
        called.append(samples)

    monkeypatch.setattr(window, "probe", probe)
    assert (
        window.main(
            [
                "--source",
                str(local / "source.wav"),
                "--directory",
                str(local / "new"),
                "--samples",
                "529200",
                "--max-worker-rss-bytes",
                "6442450944",
                "--max-saved-bytes",
                "67108864",
            ]
        )
        == 0
    )
    assert called == [529200]


@pytest.fixture
def fixed_plan(
    local: Path,
    budget: window.WindowBudget,
    execute: window.Execute,
    monkeypatch: pytest.MonkeyPatch,
) -> tuple[Path, Path]:
    source = local / "source.wav"
    _write_wav(source, np.linspace(-0.1, 0.3, 1058400, dtype=np.float32).reshape(2, 529200))
    code, provider, library = local / "code.py", local / "provider.py", local / "core.so"
    code.write_text("# synthetic code\n", encoding="utf-8")
    provider.write_text("# synthetic provider\n", encoding="utf-8")
    library.write_bytes(b"synthetic library")
    snapshot = (
        local
        / "parent-cache"
        / ("models--" + MODEL_NAME.replace("/", "--"))
        / "snapshots"
        / MODEL_REVISION
    )
    snapshot.mkdir(parents=True)
    weights, config = snapshot / "model.safetensors", snapshot / "config.json"
    weights.write_bytes(b"synthetic weights")
    config.write_bytes(b"{}")
    monkeypatch.setattr(window, "MODEL_SHA256", sha256(weights))
    monkeypatch.setattr(window, "MODEL_CONFIG_SHA256", sha256(config))
    monkeypatch.setattr(window, "_code_inputs", lambda: {str(code): sha256(code)})
    monkeypatch.setattr(window, "_required_provider_paths", lambda: {str(provider), str(library)})
    original = window.run_windows

    def complete(directories: tuple[Path, ...], selected: window.WindowBudget) -> dict[str, object]:
        measurement = execute(directories, selected)

        def receipts(directory: Path) -> None:
            durable(directory / "attempt.json", {"status": "reserved", "freshProcess": True})
            (directory / "worker.log").write_text("synthetic child\n", encoding="utf-8")

            def audit(phase: str) -> None:
                durable(
                    directory / f"{phase}-descriptors.json",
                    {"pid": 123, "snapshot": "p123\n", "externalWritableRegular": []},
                )

            tuple(map(audit, ("before-load", "after-load", "after-inference")))

        tuple(map(receipts, directories))
        return measurement

    def run(
        audio: NDArray[np.float32], directory: Path, *, budget: window.WindowBudget
    ) -> NDArray[np.float32]:
        return original(audio, directory, budget=budget, execute=complete)

    monkeypatch.setattr(window, "run_windows", run)
    destination = local / "planned-probe"
    path = local / "plan.json"
    selected = window.WindowBudget(**{**asdict(budget), "max_samples": 529200})
    durable(
        path,
        {
            "version": 1,
            "kind": "separation-window-probe",
            "source": {"path": str(source), "identity": identity(source)},
            "sourceContract": "interval-only-with-stable-source-identity",
            "sampleCount": 529200,
            "budget": asdict(selected),
            "outputDirectory": str(destination),
            "codeInputs": [{"path": str(code), "sha256": sha256(code)}],
            "providerInputs": [
                {"path": str(provider), "sha256": sha256(provider)},
                {"path": str(library), "sha256": sha256(library)},
            ],
            "modelInputs": [
                {"path": str(weights), "sha256": sha256(weights)},
                {"path": str(config), "sha256": sha256(config)},
            ],
            "qualityApproved": False,
            "productionBackendAdopted": False,
        },
    )
    return path, destination


def replace_json(path: Path, data: dict[str, object]) -> None:
    path.unlink()
    durable(path, data)


def repin_outputs(plan_path: Path, directory: Path) -> None:
    plan = window.load_plan(plan_path, directory)
    receipt = read(directory / "result.json")
    receipt["outputs"] = [pin(path) for path in window._probe_outputs(plan)]
    replace_json(directory / "result.json", receipt)


def test_fixed_plan_and_pure_complete_verifier(
    fixed_plan: tuple[Path, Path],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    plan, directory = fixed_plan
    window.probe_plan(plan, directory)
    before = {
        str(path): (identity(path), path.read_bytes())
        for path in directory.rglob("*")
        if path.is_file()
    }
    monkeypatch.setattr(window, "durable", lambda *args: pytest.fail("readonly wrote"))
    monkeypatch.setattr(
        window, "run_windows", lambda *args, **kwargs: pytest.fail("readonly inferred")
    )
    window.verify_probe(plan, directory)
    assert before == {
        str(path): (identity(path), path.read_bytes())
        for path in directory.rglob("*")
        if path.is_file()
    }
    assert (
        record(read(directory / "comparison.json")["comparison"])["pcm16DifferingChannelSamples"]
        == 0
    )
    with pytest.raises(ReplayError, match="already attempted"):
        window.probe_plan(plan, directory)


@pytest.mark.parametrize(
    "name",
    [
        "serial/result.json",
        "parallel/worker-01/result.json",
        "parallel/worker-00/window-000000.npy",
        "serial/worker-00/after-load-descriptors.json",
        "serial/worker-00/attempt.json",
    ],
)
def test_readonly_missing_completion_never_repaired(
    fixed_plan: tuple[Path, Path], name: str
) -> None:
    plan, directory = fixed_plan
    window.probe_plan(plan, directory)
    missing = directory / name
    missing.unlink()
    with pytest.raises((ReplayError, OSError)):
        window.verify_probe(plan, directory)
    assert not missing.exists()


@pytest.mark.parametrize(
    "field,value",
    [
        ("sourceContract", "whole-source-hashed"),
        ("version", True),
        ("sampleCount", True),
        ("sampleCount", 2646001),
        ("qualityApproved", True),
        ("outputDirectory", "/outside"),
        ("codeInputs", []),
        ("unknown", 1),
    ],
)
def test_fixed_plan_rejects_invalid_fields(
    fixed_plan: tuple[Path, Path], field: str, value: object
) -> None:
    path, directory = fixed_plan
    data = read(path)
    data[field] = value
    replace_json(path, data)
    with pytest.raises(ReplayError):
        window.load_plan(path, directory)


def test_source_identity_and_model_hash_changes_stop_before_inference(
    fixed_plan: tuple[Path, Path],
) -> None:
    path, directory = fixed_plan
    data = read(path)
    source = Path(str(record(data["source"])["path"]))
    source.write_bytes(b"changed")
    with pytest.raises(ReplayError, match="source identity"):
        window.probe_plan(path, directory)
    assert not directory.exists()


def test_full_ast_and_native_provider_pins_required(
    fixed_plan: tuple[Path, Path], monkeypatch: pytest.MonkeyPatch
) -> None:
    path, directory = fixed_plan
    monkeypatch.setattr(window, "_code_inputs", lambda: {"missing": "0" * 64})
    with pytest.raises(ReplayError, match="AST"):
        window.probe_plan(path, directory)
    assert not directory.exists()


@pytest.mark.parametrize("kind", ["provider", "hash", "overlap"])
def test_probe_ref_guard(fixed_plan: tuple[Path, Path], kind: str) -> None:
    path, directory = fixed_plan
    data = read(path)
    code = data["codeInputs"]
    providers = data["providerInputs"]
    assert isinstance(code, list) and isinstance(providers, list)
    if kind == "provider":
        data["providerInputs"] = providers[:1]
    elif kind == "hash":
        Path(str(providers[0]["path"])).write_bytes(b"changed")
    else:
        data["providerInputs"] = providers + code
    replace_json(path, data)
    with pytest.raises(ReplayError):
        window.probe_plan(path, directory)
    assert not directory.exists()


@pytest.mark.parametrize(
    "name,field,value",
    [
        ("comparison.json", "comparison", {}),
        ("comparison.json", "serialSeconds", float("nan")),
        ("parallel/result.json", "workers", 1),
        ("parallel/worker-00/request.json", "budget", {}),
        ("serial/worker-00/attempt.json", "freshProcess", False),
        ("serial/worker-00/after-load-descriptors.json", "snapshot", "f1\naw\ntREG\nn/outside\n"),
    ],
)
def test_readonly_domain_checks_even_with_rebound_output_receipt(
    fixed_plan: tuple[Path, Path], name: str, field: str, value: object
) -> None:
    plan, directory = fixed_plan
    window.probe_plan(plan, directory)
    path = directory / name
    data = read(path)
    data[field] = value
    replace_json(path, data)
    repin_outputs(plan, directory)
    with pytest.raises(ReplayError):
        window.verify_probe(plan, directory)


def test_saved_pcm_content_is_independently_verified(fixed_plan: tuple[Path, Path]) -> None:
    plan, directory = fixed_plan
    window.probe_plan(plan, directory)
    (directory / "parallel.wav").unlink()
    _write_wav(directory / "parallel.wav", np.zeros((2, 529200), dtype=np.float32))
    repin_outputs(plan, directory)
    with pytest.raises(ReplayError, match="PCM"):
        window.verify_probe(plan, directory)


def test_probe_plan_cli_routes(
    fixed_plan: tuple[Path, Path], monkeypatch: pytest.MonkeyPatch
) -> None:
    plan, directory = fixed_plan
    calls: list[str] = []
    monkeypatch.setattr(window, "probe_plan", lambda path, output: calls.append("producer"))
    monkeypatch.setattr(window, "verify_probe", lambda path, output: calls.append("readonly"))
    assert window.main(["--plan", str(plan), "--directory", str(directory)]) == 0
    assert window.main(["--plan", str(plan), "--directory", str(directory), "--verify"]) == 0
    assert (
        window.main(["--plan", str(plan), "--source", "extra", "--directory", str(directory)]) == 1
    )
    assert window.main(["--verify"]) == 1
    assert calls == ["producer", "readonly"]


def test_installed_provider_library_inventory(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    package = FakeDistribution(tmp_path, [importlib.metadata.PackagePath("mlx/core.cpython.so")])
    monkeypatch.setattr(importlib.metadata, "distribution", lambda name: package)
    expected = {
        str(tmp_path / "mlx/core.cpython.so"),
        str(tmp_path / "mlx_audio/sts/models/mel_roformer/model.py"),
        str(tmp_path / "mlx_audio/sts/models/mel_roformer/config.py"),
    }
    assert window._required_provider_paths() == expected
    package.files = []
    with pytest.raises(ReplayError, match="core library"):
        window._required_provider_paths()


def test_reference_validation(local: Path) -> None:
    with pytest.raises(ReplayError):
        window._references([{"path": str(local / "a"), "sha256": "Z" * 64}])
    with pytest.raises(ReplayError):
        window._references([{"path": str(local / "a"), "sha256": "a" * 64}] * 2)


def test_static_source_inventory_adapter(local: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    source = Source(local / "code.py", "a" * 64, (1, 2, 3, 4, 5))
    monkeypatch.setattr(window, "capture", lambda modules: (source,))
    assert window._code_inputs() == {str(source.path): source.sha256}


def test_provider_library_package_missing(monkeypatch: pytest.MonkeyPatch) -> None:
    def missing(name: str) -> importlib.metadata.Distribution:
        raise importlib.metadata.PackageNotFoundError(name)

    monkeypatch.setattr(importlib.metadata, "distribution", missing)
    with pytest.raises(ReplayError, match="not installed"):
        window._required_provider_paths()


def test_fixed_model_hash_changed_before_inference(fixed_plan: tuple[Path, Path]) -> None:
    path, directory = fixed_plan
    inputs = read(path)["modelInputs"]
    assert isinstance(inputs, list)
    weights = record(inputs[0])
    Path(str(weights["path"])).write_bytes(b"changed")
    with pytest.raises(ReplayError, match="input changed"):
        window.probe_plan(path, directory)
    assert not directory.exists()


def test_successful_silent_child_has_bindable_log(
    local: Path, budget: window.WindowBudget, monkeypatch: pytest.MonkeyPatch
) -> None:
    directory = local / "silent-worker"
    directory.mkdir()
    durable(directory / "request.json", {"environment": {"TMPDIR": str(local)}})
    monkeypatch.setattr(subprocess, "Popen", lambda *args, **kwargs: FakeChild())
    monkeypatch.setattr(window, "child_rss", lambda pid: 1)
    monkeypatch.setattr(window, "verify_worker", lambda path: {"kernelPeakRSSBytes": 1})
    window.execute_children((directory,), budget)
    log = directory / "worker.log"
    assert log.read_bytes() == b"WINDOW_WORKER reserved\n"
    assert pin(log)["bytes"] == 23


def test_a_cpu_worker_runs_directly(tmp_path: Path) -> None:
    (tmp_path / "request.json").write_text(json.dumps({"device": "cpu"}), encoding="utf-8")
    command = window.worker_command(tmp_path)
    assert command[1:] == ["-B", "-m", "replay.separation_window_budget", "--worker", str(tmp_path)]
    assert not (tmp_path / "write-sandbox.sb").exists()


def test_an_accelerator_worker_runs_in_a_checkout_only_write_sandbox(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.chdir(tmp_path)
    (tmp_path / "request.json").write_text(json.dumps({"device": "mps"}), encoding="utf-8")
    command = window.worker_command(tmp_path)
    profile = tmp_path / "write-sandbox.sb"
    assert command[:3] == ["/usr/bin/sandbox-exec", "-f", str(profile)]
    assert command[-2:] == ["--worker", str(tmp_path)]
    assert f'(allow file-write* (subpath "{tmp_path.resolve()}")' in profile.read_text(
        encoding="utf-8"
    )
