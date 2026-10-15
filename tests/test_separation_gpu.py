"""GPU separation: per-window gate, retries, receipt and refusal to overwrite."""

import json
from pathlib import Path

import numpy as np
import pytest
from numpy.typing import NDArray

from replay import separation_gpu as gpu
from replay.bgm_separation import MODEL_SAMPLE_RATE, _write_wav
from replay.core import ReplayError
from replay.separation_window_budget import WINDOW


class Halving:
    """A deterministic stand-in for the model: vocals are half of the input."""

    sources = ("vocals", "instrumental")

    def extract_vocals(self, chunk: NDArray[np.float32]) -> NDArray[np.float32]:
        return chunk * np.float32(0.5)

    def extract_instrumental(self, chunk: NDArray[np.float32]) -> NDArray[np.float32]:
        return chunk - self.extract_vocals(chunk)


class SilentOnce(Halving):
    """Returns silence for the first call only, as a transient accelerator failure would."""

    def __init__(self) -> None:
        self.calls = 0

    def extract_vocals(self, chunk: NDArray[np.float32]) -> NDArray[np.float32]:
        self.calls += 1
        if self.calls == 1:
            return np.zeros_like(chunk)
        return super().extract_vocals(chunk)


class AlwaysSilent(Halving):
    def extract_vocals(self, chunk: NDArray[np.float32]) -> NDArray[np.float32]:
        return np.zeros_like(chunk)


class NotFinite(Halving):
    def extract_vocals(self, chunk: NDArray[np.float32]) -> NDArray[np.float32]:
        return np.full_like(chunk, np.nan)


def audible(samples: int) -> NDArray[np.float32]:
    return np.stack((np.linspace(-0.5, 0.5, samples), np.linspace(0.25, -0.25, samples))).astype(
        np.float32
    )


def test_extract_vocals_matches_the_unwrapped_backend() -> None:
    chunk = audible(WINDOW + WINDOW // 2 + 10)
    gated = gpu.GatedBackend(Halving())
    np.testing.assert_array_equal(gated.extract_vocals(chunk), chunk * np.float32(0.5))
    assert gated.windows == 3
    assert gated.retries == 0


def test_transient_silence_is_retried_and_counted() -> None:
    chunk = audible(WINDOW)
    gated = gpu.GatedBackend(SilentOnce())
    np.testing.assert_array_equal(gated.extract_vocals(chunk), chunk * np.float32(0.5))
    assert gated.windows == 1
    assert gated.retries == 1


def test_persistent_silence_for_audible_input_is_refused() -> None:
    gated = gpu.GatedBackend(AlwaysSilent())
    with pytest.raises(ReplayError, match="no usable voice for window 0"):
        gated.extract_vocals(audible(WINDOW))
    assert gated.retries == 3


def test_silent_input_may_yield_silent_output() -> None:
    chunk = np.zeros((2, WINDOW), dtype=np.float32)
    gated = gpu.GatedBackend(AlwaysSilent())
    np.testing.assert_array_equal(gated.extract_vocals(chunk), chunk)
    assert gated.retries == 0


def test_non_finite_output_is_refused() -> None:
    with pytest.raises(ReplayError, match="no usable voice"):
        gpu.GatedBackend(NotFinite()).extract_vocals(audible(WINDOW))


def test_instrumental_is_the_remainder() -> None:
    chunk = audible(WINDOW)
    np.testing.assert_array_equal(
        gpu.GatedBackend(Halving()).extract_instrumental(chunk), chunk * np.float32(0.5)
    )


def test_sources_come_from_the_wrapped_backend() -> None:
    assert gpu.GatedBackend(Halving()).sources == ("vocals", "instrumental")


def test_separate_writes_vocals_and_receipt(tmp_path: Path) -> None:
    source = tmp_path / "input.wav"
    _write_wav(source, audible(MODEL_SAMPLE_RATE * 3))
    document = gpu.separate(source, tmp_path / "vocals.wav", tmp_path / "receipt.json", Halving())
    receipt = json.loads((tmp_path / "receipt.json").read_text(encoding="utf-8"))
    assert receipt["device"] == "mps"
    assert receipt["qualityApproved"] is False
    assert receipt["windows"] == 1
    assert receipt["retries"] == 0
    assert receipt["audioSeconds"] == 3.0
    assert receipt["sourceSHA256"] == gpu.sha256(source)
    assert receipt["vocalsSHA256"] == gpu.sha256(tmp_path / "vocals.wav")
    assert document["windows"] == 1


def test_separate_refuses_to_overwrite(tmp_path: Path) -> None:
    source = tmp_path / "input.wav"
    _write_wav(source, audible(MODEL_SAMPLE_RATE))
    (tmp_path / "vocals.wav").write_bytes(b"keep")
    with pytest.raises(ReplayError, match="Refusing to overwrite"):
        gpu.separate(source, tmp_path / "vocals.wav", tmp_path / "receipt.json", Halving())
    assert (tmp_path / "vocals.wav").read_bytes() == b"keep"
    (tmp_path / "vocals.wav").unlink()
    (tmp_path / "receipt.json").write_text("{}", encoding="utf-8")
    with pytest.raises(ReplayError, match="Refusing to overwrite"):
        gpu.separate(source, tmp_path / "vocals.wav", tmp_path / "receipt.json", Halving())


def test_main_runs_the_real_backend_factory(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    source = tmp_path / "input.wav"
    _write_wav(source, audible(MODEL_SAMPLE_RATE))
    seen: list[str] = []

    def factory(device: str) -> Halving:
        seen.append(device)
        return Halving()

    monkeypatch.setattr(gpu, "_MlxBackend", factory)
    code = gpu.main([str(source), str(tmp_path / "v.wav"), str(tmp_path / "r.json")])
    assert code == 0
    assert seen == ["mps"]
    assert (tmp_path / "r.json").is_file()
