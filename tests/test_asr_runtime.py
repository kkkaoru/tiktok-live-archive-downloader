import importlib
import sys
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path
from types import ModuleType

import numpy as np
import pytest
from pytest import MonkeyPatch

from replay import asr_runtime
from replay.core import ReplayError


@pytest.fixture
def fake_runtime(tmp_path: Path, monkeypatch: MonkeyPatch) -> list[str]:
    events: list[str] = []
    monkeypatch.delitem(sys.modules, "mlx_whisper", raising=False)
    monkeypatch.setenv("TMPDIR", str(tmp_path))
    monkeypatch.setenv("TMP", str(tmp_path))
    monkeypatch.setenv("TEMP", str(tmp_path))
    monkeypatch.setenv("XDG_CACHE_HOME", str(tmp_path))
    monkeypatch.setenv("NUMBA_CACHE_DIR", str(tmp_path))
    active = False
    mx = ModuleType("mlx.core")
    whisper = ModuleType("mlx_whisper")

    @contextmanager
    def stream(device: object) -> Iterator[None]:
        nonlocal active
        assert device == "synthetic-cpu"
        assert not active
        active = True
        events.append("enter-cpu")
        try:
            yield
        finally:
            active = False
            events.append("exit-cpu")

    def synchronize(device: object) -> None:
        assert active
        assert device == "synthetic-cpu"
        events.append("synchronize")

    def transcribe(samples: object, **kwargs: object) -> object:
        assert active
        assert isinstance(samples, np.ndarray)
        assert samples.dtype == np.float32
        assert kwargs["temperature"] == 0.0
        assert kwargs["language"] == "ja"
        assert kwargs["word_timestamps"] is True
        assert kwargs["condition_on_previous_text"] is False
        assert "fp16" not in kwargs
        events.append("load-infer-eval")
        return {"segments": []}

    def imported(name: str) -> ModuleType:
        events.append(name)
        if name == "mlx.core":
            return mx
        assert name == "mlx_whisper"
        assert active
        return whisper

    def audit(directory: Path, phase: str) -> None:
        assert directory.is_dir()
        events.append(phase)

    monkeypatch.setattr(mx, "cpu", "synthetic-cpu", raising=False)
    monkeypatch.setattr(mx, "stream", stream, raising=False)
    monkeypatch.setattr(mx, "synchronize", synchronize, raising=False)
    monkeypatch.setattr(whisper, "transcribe", transcribe, raising=False)
    monkeypatch.setattr(importlib, "import_module", imported)
    monkeypatch.setattr(asr_runtime, "audit_descriptors", audit)
    return events


def test_cpu_context_import_load_inference_eval_and_batch_reuse(
    tmp_path: Path, fake_runtime: list[str]
) -> None:
    runtime = asr_runtime.WhisperRuntime(asr_runtime.ASRDevice.CPU)
    assert runtime.predict(np.zeros(16, np.float32), tmp_path, evidence=tmp_path / "one") == {
        "segments": []
    }
    assert fake_runtime == [
        "before-core-import",
        "mlx.core",
        "after-core-import",
        "enter-cpu",
        "mlx_whisper",
        "after-whisper-import",
        "load-infer-eval",
        "synchronize",
        "after-inference-eval",
        "exit-cpu",
    ]
    fake_runtime.clear()
    runtime.predict(np.ones(32, np.float32), tmp_path, evidence=tmp_path / "two")
    assert fake_runtime == [
        "before-core-import",
        "mlx.core",
        "after-core-import",
        "enter-cpu",
        "after-whisper-import",
        "load-infer-eval",
        "synchronize",
        "after-inference-eval",
        "exit-cpu",
    ]


def test_cpu_rejects_preimported_provider(
    tmp_path: Path, fake_runtime: list[str], monkeypatch: MonkeyPatch
) -> None:
    monkeypatch.setitem(sys.modules, "mlx_whisper", ModuleType("mlx_whisper"))
    with pytest.raises(ReplayError, match="fresh process"):
        asr_runtime.WhisperRuntime(asr_runtime.ASRDevice.CPU).predict(
            np.zeros(1, np.float32), tmp_path, evidence=tmp_path / "new"
        )
    assert fake_runtime == []
    assert not (tmp_path / "new").exists()


@pytest.mark.parametrize("name", ["TMPDIR", "TMP", "TEMP", "XDG_CACHE_HOME", "NUMBA_CACHE_DIR"])
def test_cpu_rejects_missing_scratch_before_import(
    tmp_path: Path, fake_runtime: list[str], monkeypatch: MonkeyPatch, name: str
) -> None:
    monkeypatch.delenv(name)
    with pytest.raises(ReplayError, match="Missing absolute path"):
        asr_runtime.WhisperRuntime(asr_runtime.ASRDevice.CPU).predict(
            np.zeros(1, np.float32), tmp_path, evidence=tmp_path / "new"
        )
    assert fake_runtime == []


def test_cpu_rejects_external_scratch(
    tmp_path: Path, fake_runtime: list[str], monkeypatch: MonkeyPatch
) -> None:
    monkeypatch.setenv("NUMBA_CACHE_DIR", "/")
    with pytest.raises(ReplayError, match="must stay local"):
        asr_runtime.WhisperRuntime(asr_runtime.ASRDevice.CPU).predict(
            np.zeros(1, np.float32), tmp_path, evidence=tmp_path / "new"
        )
    assert fake_runtime == []


def test_cpu_rejects_linked_evidence(tmp_path: Path, fake_runtime: list[str]) -> None:
    (tmp_path / "link").symlink_to(tmp_path, target_is_directory=True)
    with pytest.raises(ReplayError, match="Noncanonical"):
        asr_runtime.WhisperRuntime(asr_runtime.ASRDevice.CPU).predict(
            np.zeros(1, np.float32), tmp_path, evidence=tmp_path / "link" / "new"
        )
    assert fake_runtime == []


def test_cpu_paths_checked_before_writes(tmp_path: Path) -> None:
    asr_runtime.validate_cpu_paths(asr_runtime.ASRDevice.AUTO, Path("/not-created"))
    with pytest.raises(ReplayError, match="inside checkout"):
        asr_runtime.validate_cpu_paths(asr_runtime.ASRDevice.CPU, Path("/not-created"))
    (tmp_path / "linked").symlink_to(tmp_path, target_is_directory=True)
    with pytest.raises(ReplayError, match="Noncanonical"):
        asr_runtime.validate_cpu_paths(asr_runtime.ASRDevice.CPU, tmp_path / "linked" / "out")


def test_cpu_rejects_external_evidence(tmp_path: Path, fake_runtime: list[str]) -> None:
    with pytest.raises(ReplayError, match="evidence must stay local"):
        asr_runtime.WhisperRuntime(asr_runtime.ASRDevice.CPU).predict(
            np.zeros(1, np.float32), tmp_path, evidence=Path("/not-created-asr-evidence")
        )
    assert fake_runtime == []


@pytest.mark.parametrize("mode", ["missing-api", "bad-context", "operator"])
def test_cpu_failure_never_falls_back(
    tmp_path: Path, fake_runtime: list[str], monkeypatch: MonkeyPatch, mode: str
) -> None:
    original = importlib.import_module

    def failing(*args: object, **kwargs: object) -> object:
        raise RuntimeError("unsupported CPU operator")

    def imported(name: str) -> ModuleType:
        module = original(name)
        if mode == "missing-api" and name == "mlx.core":
            monkeypatch.setattr(module, "synchronize", None)
        if mode == "bad-context" and name == "mlx.core":
            monkeypatch.setattr(module, "stream", lambda device: None)
        if mode == "operator" and name == "mlx_whisper":
            monkeypatch.setattr(module, "transcribe", failing)
        return module

    monkeypatch.setattr(importlib, "import_module", imported)
    with pytest.raises((ReplayError, RuntimeError), match="MLX|unsupported CPU"):
        asr_runtime.WhisperRuntime(asr_runtime.ASRDevice.CPU).predict(
            np.zeros(1, np.float32), tmp_path, evidence=tmp_path / "failure"
        )
    assert (tmp_path / "failure").is_dir()
    assert "synchronize" not in fake_runtime
    if mode == "operator":
        assert fake_runtime[-1] == "exit-cpu"


def test_descriptor_violation_stops_before_core_import(
    tmp_path: Path, fake_runtime: list[str], monkeypatch: MonkeyPatch
) -> None:
    def reject(directory: Path, phase: str) -> None:
        raise ReplayError("external writable file")

    monkeypatch.setattr(asr_runtime, "audit_descriptors", reject)
    with pytest.raises(ReplayError, match="external writable"):
        asr_runtime.WhisperRuntime(asr_runtime.ASRDevice.CPU).predict(
            np.zeros(1, np.float32), tmp_path, evidence=tmp_path / "failure"
        )
    assert fake_runtime == []
