"""Synthetic bounded reads and compile wrappers; no real MLX or models."""

import resource
import sys
import wave
from contextlib import AbstractContextManager, nullcontext
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
from numpy.typing import NDArray

from replay import separation_forward_probe as probe
from replay.bgm_separation import _MlxBackend, _write_wav
from replay.core import ReplayError
from replay.native_invocation import read


class FakeMX:
    cpu = "cpu"

    def stream(self, device: str) -> AbstractContextManager[None]:
        assert device == "cpu"
        return nullcontext()

    def array(self, value: object) -> NDArray[np.float32]:
        return np.asarray(value, dtype=np.float32)

    def eval(self, value: object) -> None:
        assert isinstance(value, np.ndarray)

    def compile(self, forward: probe.Forward) -> probe.Forward:
        return forward


class FakeBackend(_MlxBackend):
    def __init__(self) -> None:
        self._mx = FakeMX()
        self._device = "cpu"
        self._model = self.forward

    def forward(self, value: object) -> NDArray[np.float32]:
        return np.asarray(value, dtype=np.float32) * np.float32(0.5)


@pytest.fixture
def local(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> Path:
    monkeypatch.chdir(tmp_path)
    work = tmp_path / "edits/work"
    work.mkdir(parents=True)
    cache = work / "cache"
    cache.mkdir()
    monkeypatch.setenv("TMPDIR", str(cache))
    monkeypatch.setenv("TMP", str(cache))
    monkeypatch.setenv("TEMP", str(cache))
    monkeypatch.setenv("XDG_CACHE_HOME", str(cache))
    monkeypatch.setenv("HF_HOME", str(cache))
    monkeypatch.setenv("HF_HUB_CACHE", str(cache))
    monkeypatch.setenv("HF_HUB_OFFLINE", "1")
    monkeypatch.setenv("TRANSFORMERS_OFFLINE", "1")
    return work


@pytest.fixture
def pcm(local: Path) -> Path:
    source = local / "source.wav"
    audio = np.asarray(
        [[0, 0.25, -0.5, 0.75, -1, 0.5], [0.5, -0.25, 0.5, 0, 1, -0.75]], dtype=np.float32
    )
    _write_wav(source, audio)
    return source


@pytest.mark.parametrize("samples", [1, 176399, 176400, 352800, 352801, 529207])
def test_compiled_wrapper_preserves_stereo_tail(samples: int) -> None:
    audio = np.stack((np.linspace(-0.75, 0.75, samples), np.linspace(0.5, -0.25, samples))).astype(
        np.float32
    )
    original = FakeBackend()
    model = original._model
    compiled = probe.CompiledBackend(original)
    np.testing.assert_array_equal(compiled.extract_vocals(audio), original.extract_vocals(audio))
    assert original._model is model
    assert compiled.extract_vocals(audio).shape == (2, samples)


def test_compiled_wrapper_requires_cpu() -> None:
    original = FakeBackend()
    original._device = "gpu"
    with pytest.raises(ReplayError, match="CPU"):
        probe.CompiledBackend(original)


@pytest.mark.parametrize("attribute", ["_model", "compile"])
def test_compiled_wrapper_rejects_missing_callable(
    attribute: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    original = FakeBackend()
    target = original if attribute == "_model" else original._mx
    monkeypatch.setattr(target, attribute, None)
    with pytest.raises(ReplayError, match="callable"):
        probe.CompiledBackend(original)


def test_compiler_rejects_noncallable_result(monkeypatch: pytest.MonkeyPatch) -> None:
    original = FakeBackend()
    monkeypatch.setattr(original._mx, "compile", lambda forward: None)
    with pytest.raises(ReplayError, match="return a callable"):
        probe.CompiledBackend(original)


def test_read_exact_interval(pcm: Path) -> None:
    interval = probe.read_interval(pcm, start_sample=1, samples=3)
    np.testing.assert_array_equal(
        interval.audio,
        np.asarray([[8191, -16383, 24575], [-8191, 16383, 0]], dtype=np.float32) / 32768.0,
    )
    assert interval.source_samples == 6
    assert len(interval.sha256) == 64


def test_reader_reads_only_requested_frames(pcm: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    calls: list[int] = []
    real_read = wave.Wave_read.readframes

    def bounded_read(reader: wave.Wave_read, count: int) -> bytes:
        calls.append(count)
        return real_read(reader, count)

    monkeypatch.setattr(wave.Wave_read, "readframes", bounded_read)
    probe.read_interval(pcm, start_sample=2, samples=2)
    assert calls == [2]


@pytest.mark.parametrize(
    ("start", "samples"), [(-1, 1), (False, 1), (0, 0), (0, True), (0, 2646001), (5, 2)]
)
def test_invalid_interval_rejected(pcm: Path, start: int, samples: int) -> None:
    with pytest.raises(ReplayError):
        probe.read_interval(pcm, start_sample=start, samples=samples)


def test_link_rejected(pcm: Path, local: Path) -> None:
    linked = local / "linked.wav"
    linked.symlink_to(pcm)
    with pytest.raises(ReplayError, match="linked"):
        probe.read_interval(linked, start_sample=0, samples=1)


def test_nonlocal_input_rejected(pcm: Path, monkeypatch: pytest.MonkeyPatch, local: Path) -> None:
    other = local / "other"
    other.mkdir()
    monkeypatch.chdir(other)
    with pytest.raises(ReplayError, match="inside checkout"):
        probe.read_interval(pcm, start_sample=0, samples=1)


def test_wrong_format(local: Path) -> None:
    source = local / "mono.wav"
    _write_wav(source, np.zeros((1, 4), dtype=np.float32))
    with pytest.raises(ReplayError, match="stereo"):
        probe.read_interval(source, start_sample=0, samples=1)


def test_truncated_pcm(pcm: Path) -> None:
    with pcm.open("r+b") as stream:
        stream.truncate(48)
    with pytest.raises(ReplayError, match="Truncated"):
        probe.read_interval(pcm, start_sample=0, samples=6)


def test_identity_change_during_read(pcm: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    changed = iter([{"inode": 1}, {"inode": 2}])
    monkeypatch.setattr(probe, "identity", lambda path: next(changed))
    with pytest.raises(ReplayError, match="changed while reading"):
        probe.read_interval(pcm, start_sample=0, samples=2)


def test_comparison_equal_and_silence() -> None:
    result = probe.compare_audio(
        np.zeros((2, 3), dtype=np.float32), np.zeros((2, 3), dtype=np.float32)
    )
    assert result == {
        "sampleCount": 3,
        "floatMaxAbsoluteDifference": 0.0,
        "floatRMSDifference": 0.0,
        "floatCorrelation": None,
        "pcm16MaxAbsoluteDifference": 0.0,
        "pcm16RMSDifference": 0.0,
        "pcm16Correlation": None,
        "pcm16DifferingChannelSamples": 0,
        "qualityApproved": False,
    }


def test_comparison_measures_unapproved_difference() -> None:
    original = np.asarray([[0.25, 0.5], [-0.5, -0.25]], dtype=np.float32)
    result = probe.compare_audio(original, original * np.float32(0.5))
    assert result["floatMaxAbsoluteDifference"] == 0.25
    assert result["floatCorrelation"] == pytest.approx(1.0)
    assert result["pcm16MaxAbsoluteDifference"] == 8192.0
    assert result["pcm16DifferingChannelSamples"] == 4
    assert result["qualityApproved"] is False


@pytest.mark.parametrize("kind", ["shape", "nan", "empty"])
def test_comparison_invalid(kind: str) -> None:
    baseline = np.ones((2, 3), dtype=np.float32)
    candidate = {
        "shape": np.ones((1, 3)),
        "nan": np.full((2, 3), np.nan),
        "empty": np.ones((2, 0)),
    }[kind].astype(np.float32)
    with pytest.raises(ReplayError, match="Invalid comparison"):
        probe.compare_audio(baseline, candidate)


def test_run_preserves_original_and_reports(pcm: Path, local: Path) -> None:
    phases: list[str] = []
    result = probe.run(
        pcm,
        local / "probe",
        start_sample=1,
        samples=4,
        backend_factory=FakeBackend,
        auditor=lambda directory, phase: phases.append(phase),
    )
    saved = read(result)
    assert saved["sourceAndCodeUnchanged"] is True
    assert saved["sampleCount"] == 4
    assert saved["qualityApproved"] is False
    assert saved["productionBackendAdopted"] is False
    assert read(local / "probe/input.json")["wholeSourceHashed"] is False
    assert phases == [
        "before-load",
        "after-load",
        "after-baseline",
        "after-compile-wrapper",
        "after-compiled-first",
        "after-compiled-warm",
        "after-save",
    ]
    assert (local / "probe/baseline.wav").read_bytes() == (
        local / "probe/compiled-warm.wav"
    ).read_bytes()
    assert result.stat().st_mode & 0o777 == 0o600


def test_run_rejects_occupied_output(pcm: Path, local: Path) -> None:
    occupied = local / "occupied"
    occupied.mkdir()
    with pytest.raises(FileExistsError):
        probe.run(pcm, occupied, start_sample=0, samples=1, backend_factory=FakeBackend)


def test_run_rejects_nonlocal_output(pcm: Path, local: Path) -> None:
    with pytest.raises(ReplayError, match="output must stay local"):
        probe.run(
            pcm,
            local.parent.parent / "outside",
            start_sample=0,
            samples=1,
            backend_factory=FakeBackend,
        )


def test_run_rejects_code_change(pcm: Path, local: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    pins = iter([{"source": "before"}, {"source": "after"}])
    monkeypatch.setattr(probe, "_code_pins", lambda: next(pins))
    with pytest.raises(ReplayError, match="code changed"):
        probe.run(
            pcm,
            local / "changing",
            start_sample=0,
            samples=1,
            backend_factory=FakeBackend,
            auditor=lambda directory, phase: None,
        )
    assert not (local / "changing/result.json").exists()


def test_run_rejects_external_scratch(
    pcm: Path, local: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("TMPDIR", "/")
    with pytest.raises(ReplayError, match="TMPDIR"):
        probe.run(pcm, local / "probe", start_sample=0, samples=1, backend_factory=FakeBackend)


def test_run_requires_offline(pcm: Path, local: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("HF_HUB_OFFLINE", "0")
    with pytest.raises(ReplayError, match="offline"):
        probe.run(pcm, local / "probe", start_sample=0, samples=1, backend_factory=FakeBackend)


def test_run_rejects_disabled_compile(
    pcm: Path, local: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("MLX_DISABLE_COMPILE", "1")
    with pytest.raises(ReplayError, match="disable mx.compile"):
        probe.run(pcm, local / "probe", start_sample=0, samples=1, backend_factory=FakeBackend)


def test_loader_explicit_cpu(monkeypatch: pytest.MonkeyPatch) -> None:
    devices: list[str] = []

    def factory(device: str) -> FakeBackend:
        devices.append(device)
        return FakeBackend()

    monkeypatch.setattr(probe, "_MlxBackend", factory)
    assert isinstance(probe.load_cpu_backend(), FakeBackend)
    assert devices == ["cpu"]


@pytest.mark.parametrize(("platform", "expected"), [("darwin", 123), ("linux", 125952)])
def test_kernel_peak_rss_units(
    platform: str, expected: int, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(sys, "platform", platform)
    monkeypatch.setattr(resource, "getrusage", lambda who: SimpleNamespace(ru_maxrss=123))
    assert probe._kernel_peak_rss() == expected


def test_main_success(pcm: Path, local: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(probe, "run", lambda *args, **kwargs: local / "result.json")
    assert probe.main([str(pcm), str(local / "out"), "--samples", "1"]) == 0


def test_main_failure(
    pcm: Path, local: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    def fail(*args: object, **kwargs: object) -> Path:
        raise ReplayError("synthetic refusal")

    monkeypatch.setattr(probe, "run", fail)
    assert probe.main([str(pcm), str(local / "out"), "--samples", "1"]) == 1
    assert "synthetic refusal" in capsys.readouterr().err
