"""Synthetic regression tests for Demucs-based instrumental BGM removal."""

from __future__ import annotations

import hashlib
import importlib
import wave
from contextlib import nullcontext
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import mlx.core as mx
import numpy as np
import pytest
from numpy.typing import NDArray

from replay import bgm_separation as bgm
from replay.bgm_separation import (
    SeparationConfig,
    SeparationPolicy,
    adapt_vocal_mix,
    apply_wiener_vocal_mask,
    instrumental_level,
    policy_from_mapping,
    resolve_device,
    select_music_strength,
    separate_vocals,
    subtract_music,
    verify_music_removal,
)
from replay.core import ReplayError

SAMPLERATE = 44100


class _FakeTensor:
    """Minimal stand-in for the torch surface the backend touches."""

    def __init__(self, data: NDArray[np.float32]) -> None:
        self._data = np.asarray(data, dtype=np.float32)

    def unsqueeze(self, dim: int) -> _FakeTensor:
        return _FakeTensor(np.expand_dims(self._data, dim))

    def to(self, device: object) -> _FakeTensor:
        return self

    def cpu(self) -> _FakeTensor:
        return self

    def numpy(self) -> NDArray[np.float32]:
        return self._data

    def __getitem__(self, index: int) -> _FakeTensor:
        return _FakeTensor(self._data[index])


class _FakeModel:
    def __init__(self, sources: tuple[str, ...]) -> None:
        self.sources = sources

    def to(self, device: object) -> _FakeModel:
        return self

    def eval(self) -> None:
        return None


class _ScaleBackend:
    """Deterministic vocals stand-in: the tests never load torch."""

    def __init__(self, gain: float = 0.9) -> None:
        self._gain = gain

    @property
    def sources(self) -> tuple[str, ...]:
        return ("drums", "bass", "other", "vocals")

    def extract_vocals(self, chunk: NDArray[np.float32]) -> NDArray[np.float32]:
        return (chunk * self._gain).astype(np.float32)

    def extract_instrumental(self, chunk: NDArray[np.float32]) -> NDArray[np.float32]:
        return np.zeros_like(chunk)


def _stub_imports(
    monkeypatch: pytest.MonkeyPatch,
    vocals_gain: float | None = None,
    sources: tuple[str, ...] = ("drums", "bass", "other", "vocals"),
    short_by: int = 0,
) -> None:
    """Route the lazy torch/Demucs seam at stub backends without torch."""

    def _echo_apply(model: object, batch: _FakeTensor, **kwargs: object) -> _FakeTensor:
        data = batch.numpy()
        zeros = np.zeros_like(data)
        stems = np.stack([zeros[0], zeros[0], zeros[0], data[0]])
        if short_by:
            stems = stems[:, :, :-short_by]
        return _FakeTensor(np.expand_dims(stems, 0))

    torch_stub = SimpleNamespace(
        from_numpy=lambda values: _FakeTensor(np.asarray(values, dtype=np.float32)),
        stack=lambda tensors: _FakeTensor(np.stack([tensor.numpy() for tensor in tensors])),
        no_grad=nullcontext,
        float32=np.float32,
        backends=SimpleNamespace(mps=SimpleNamespace(is_available=lambda: True)),
        mps=SimpleNamespace(synchronize=lambda: None),
    )
    demucs_pretrained = SimpleNamespace(
        get_model=lambda name: _FakeModel(sources),
    )
    demucs_apply = SimpleNamespace(apply_model=_echo_apply)

    def _fake_import(name: str) -> Any:
        if name == "torch":
            return torch_stub
        if name == "demucs.pretrained":
            return demucs_pretrained
        if name == "demucs.apply":
            return demucs_apply
        raise AssertionError(f"unexpected import {name}")

    monkeypatch.setattr(bgm, "_import", _fake_import)
    if vocals_gain is not None:
        monkeypatch.setattr(bgm, "_load_backend", lambda name, device: _ScaleBackend(vocals_gain))


def _write_wav(
    path: Path,
    seconds: float,
    rate: int = SAMPLERATE,
    channels: int = 2,
    width: int = 2,
    freq: float = 440.0,
) -> None:
    frames = int(seconds * rate)
    tone = 0.5 * np.sin(2.0 * np.pi * freq * np.arange(frames) / rate)
    data = np.stack([tone] * channels)
    raw = (data.T * 32767.0).astype(np.int16 if width == 2 else np.int8).tobytes()
    with wave.open(str(path), "wb") as handle:
        handle.setnchannels(channels)
        handle.setsampwidth(width)
        handle.setframerate(rate)
        handle.writeframes(raw)


def test_config_rejects_bad_values() -> None:
    assert SeparationConfig().model_name == "mlx-community/mel-roformer-mlx"
    with pytest.raises(ValueError, match="model name"):
        SeparationConfig(model_name="")
    with pytest.raises(ValueError, match="device"):
        SeparationConfig(device="cuda")
    with pytest.raises(ValueError, match="chunking"):
        SeparationConfig(chunk_seconds=10.0)
    with pytest.raises(ValueError, match="chunking"):
        SeparationConfig(chunk_seconds=60.0, overlap_seconds=30.0)
    with pytest.raises(ValueError, match="chunking"):
        SeparationConfig(chunk_seconds=True)
    with pytest.raises(ValueError, match="44100"):
        SeparationConfig(sample_rate=48000)
    with pytest.raises(ValueError, match="44100"):
        SeparationConfig(sample_rate=True)


def test_policy_from_mapping() -> None:
    policy = policy_from_mapping(
        {"minimum_instrumental_drop_db": 12.0, "quiet_instrumental_db": -55}
    )
    assert policy.minimum_instrumental_drop_db == 12.0
    assert policy.quiet_instrumental_db == -55.0
    with pytest.raises(ValueError, match="Unknown"):
        policy_from_mapping({"per_video_bonus": 1.0})
    with pytest.raises(ValueError, match="numbers"):
        policy_from_mapping({"maximum_speech_loss_db": "small"})
    with pytest.raises(ValueError, match="numbers"):
        policy_from_mapping({"maximum_speech_loss_db": True})
    with pytest.raises(ValueError, match="policy"):
        SeparationPolicy(minimum_instrumental_drop_db=0.0)
    with pytest.raises(ValueError, match="policy"):
        SeparationPolicy(maximum_speech_loss_db=99.0)
    with pytest.raises(ValueError, match="policy"):
        SeparationPolicy(quiet_instrumental_db=5.0)


def test_resolve_device_cpu_needs_no_backend() -> None:
    assert resolve_device("cpu", "mdx_extra") == "cpu"
    with pytest.raises(ValueError, match="device"):
        resolve_device("cuda")


def test_resolve_device_auto_without_torch_points_at_extra(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def unavailable(name: str) -> object:
        raise ImportError(name)

    monkeypatch.setattr(importlib, "import_module", unavailable)
    with pytest.raises(ReplayError, match="separation"):
        resolve_device("auto", "mdx_extra")


def test_resolve_device_mps_paths(monkeypatch: pytest.MonkeyPatch) -> None:
    _stub_imports(monkeypatch)
    assert resolve_device("auto", "mdx_extra") == "mps"
    assert resolve_device("mps", "mdx_extra") == "mps"


def test_import_missing_points_at_extra() -> None:
    with pytest.raises(ReplayError, match="separation"):
        bgm._import("replay.definitely.missing.backend")


def test_chunk_bounds_cover_every_sample() -> None:
    assert bgm._chunk_bounds(1000, 400, 100) == [(0, 400), (300, 700), (600, 1000)]
    assert bgm._chunk_bounds(700, 400, 100) == [(0, 400), (300, 700)]
    assert bgm._chunk_bounds(200, 400, 100) == [(0, 200)]
    with pytest.raises(ValueError, match="overlap"):
        bgm._chunk_bounds(1000, 400, 200)


def test_chunk_bounds_enforce_budget(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(bgm, "MAXIMUM_CHUNKS", 2)
    with pytest.raises(ValueError, match="budget"):
        bgm._chunk_bounds(1000, 100, 10)


def test_read_rejects_non_conforming_wav(tmp_path: Path) -> None:
    missing = tmp_path / "missing.wav"
    with pytest.raises(ReplayError, match="Cannot read"):
        bgm._read_wav(missing)
    text = tmp_path / "text.wav"
    text.write_text("not audio")
    with pytest.raises(ReplayError, match="Cannot read"):
        bgm._read_wav(text)
    narrow = tmp_path / "narrow.wav"
    _write_wav(narrow, 0.5, rate=22050)
    with pytest.raises(ReplayError, match="16-bit mono/stereo at 44100"):
        bgm._read_wav(narrow)
    eight = tmp_path / "eight.wav"
    _write_wav(eight, 0.5, width=1)
    with pytest.raises(ReplayError, match="16-bit mono/stereo at 44100"):
        bgm._read_wav(eight)
    surround = tmp_path / "surround.wav"
    _write_wav(surround, 0.5, channels=3)
    with pytest.raises(ReplayError, match="16-bit mono/stereo at 44100"):
        bgm._read_wav(surround)
    empty = tmp_path / "empty.wav"
    _write_wav(empty, 0.0)
    with pytest.raises(ReplayError, match="no audio frames"):
        bgm._read_wav(empty)


def test_write_refuses_overwrite_and_missing_parent(tmp_path: Path) -> None:
    target = tmp_path / "out.wav"
    target.write_bytes(b"existing")
    with pytest.raises(ReplayError, match="already exists"):
        bgm._write_wav(target, np.zeros((1, 8), dtype=np.float32))
    nested = tmp_path / "no-dir" / "out.wav"
    with pytest.raises(ReplayError, match="directory is missing"):
        bgm._write_wav(nested, np.zeros((1, 8), dtype=np.float32))


def test_separate_stereo_end_to_end(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    _stub_imports(monkeypatch, vocals_gain=0.9)
    source = tmp_path / "in.wav"
    destination = tmp_path / "out.wav"
    _write_wav(source, 7.0)
    config = SeparationConfig(
        model_name="mdx_extra", chunk_seconds=30.0, overlap_seconds=1.0, device="cpu"
    )
    result = separate_vocals(source, destination, config)
    assert result.audio_seconds == pytest.approx(7.0)
    assert result.wall_seconds > 0.0
    assert result.realtime_factor > 0.0
    cleaned, channels = bgm._read_wav(destination)
    assert channels == 2
    assert cleaned.shape == (2, 7 * SAMPLERATE)
    expected = 0.9 * 0.5 * np.sin(2.0 * np.pi * 440.0 * np.arange(7 * SAMPLERATE) / SAMPLERATE)
    assert np.abs(cleaned[0] - expected).max() < 1e-3


def test_separate_mono_stays_mono(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    _stub_imports(monkeypatch, vocals_gain=1.0)
    source = tmp_path / "mono.wav"
    destination = tmp_path / "mono-clean.wav"
    _write_wav(source, 3.0, channels=1)
    config = SeparationConfig(
        model_name="mdx_extra", chunk_seconds=30.0, overlap_seconds=1.0, device="cpu"
    )
    separate_vocals(source, destination, config)
    cleaned, channels = bgm._read_wav(destination)
    assert channels == 1
    assert cleaned.shape == (1, 3 * SAMPLERATE)


def test_separate_merges_runt_tail(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    _stub_imports(monkeypatch, vocals_gain=1.0)
    source = tmp_path / "tail.wav"
    destination = tmp_path / "tail-clean.wav"
    _write_wav(source, 61.5)
    config = SeparationConfig(
        model_name="mdx_extra", chunk_seconds=30.0, overlap_seconds=1.0, device="cpu"
    )
    result = separate_vocals(source, destination, config)
    assert result.audio_seconds == pytest.approx(61.5)
    cleaned, _ = bgm._read_wav(destination)
    assert cleaned.shape[1] == int(61.5 * SAMPLERATE)


def test_separate_rejects_backend_without_vocals(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    class _NoVocals(_ScaleBackend):
        @property
        def sources(self) -> tuple[str, ...]:
            return ("drums",)

    monkeypatch.setattr(bgm, "_load_backend", lambda name, device: _NoVocals())
    source = tmp_path / "in.wav"
    _write_wav(source, 1.0)
    config = SeparationConfig(
        model_name="mdx_extra", chunk_seconds=30.0, overlap_seconds=1.0, device="cpu"
    )
    with pytest.raises(ReplayError, match="no vocals stem"):
        separate_vocals(source, tmp_path / "out.wav", config)


def test_separate_rejects_misshapen_chunk(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    class _Short(_ScaleBackend):
        def extract_vocals(self, chunk: NDArray[np.float32]) -> NDArray[np.float32]:
            return np.zeros((2, chunk.shape[1] - 4), dtype=np.float32)

    monkeypatch.setattr(bgm, "_load_backend", lambda name, device: _Short())
    source = tmp_path / "in.wav"
    _write_wav(source, 1.0)
    config = SeparationConfig(
        model_name="mdx_extra", chunk_seconds=30.0, overlap_seconds=1.0, device="cpu"
    )
    with pytest.raises(ReplayError, match="misshapen"):
        separate_vocals(source, tmp_path / "out.wav", config)


@pytest.mark.parametrize("device", ["mps", "cpu"])
def test_mlx_backend_requires_pinned_offline_weights_and_explicit_stream(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, device: str
) -> None:
    weights = tmp_path / "model.safetensors"
    weights.write_bytes(b"synthetic weights")
    (tmp_path / "config.json").write_text("{}", encoding="utf-8")
    monkeypatch.setattr(bgm, "MODEL_SHA256", hashlib.sha256(b"synthetic weights").hexdigest())
    monkeypatch.setattr(bgm, "MODEL_CONFIG_SHA256", hashlib.sha256(b"{}").hexdigest())
    calls: list[object] = []
    placement: list[str] = []

    class Placement:
        def __init__(self, selected: str) -> None:
            self.selected = selected

        def __enter__(self) -> None:
            placement.append(self.selected)

        def __exit__(self, *args: object) -> None:
            placement.pop()

    class Model:
        @classmethod
        def from_pretrained(cls, path: Path) -> Model:
            assert path == tmp_path
            assert placement == [device]
            return cls()

        def sanitize(self, weights_dict: dict[str, object]) -> dict[str, object]:
            return weights_dict

        def load_weights(self, values: list[tuple[str, object]], strict: bool) -> None:
            calls.append((values, strict))

        def eval(self) -> None:
            calls.append("eval")

        def __call__(self, audio: NDArray[np.float32]) -> NDArray[np.float32]:
            assert placement == [device]
            return audio * np.float32(0.8)

    mx = SimpleNamespace(
        cpu="cpu",
        gpu="mps",
        stream=Placement,
        metal=SimpleNamespace(is_available=lambda: True),
        array=lambda audio: audio,
        load=lambda path: {"weight": 1},
        eval=lambda audio: None,
    )
    hub = SimpleNamespace(try_to_load_from_cache=lambda *args, **kwargs: str(weights))

    def fake_import(name: str) -> object:
        return {
            "mlx.core": mx,
            "huggingface_hub": hub,
            "mlx_audio.sts.models.mel_roformer": SimpleNamespace(MelRoFormer=Model),
        }[name]

    monkeypatch.setattr(bgm, "_import", fake_import)
    assert resolve_device("auto") == "mps"
    assert resolve_device("cpu") == "cpu"
    backend = bgm._load_backend(bgm.MODEL_NAME, device)
    assert placement == []
    assert backend.sources == ("vocals", "instrumental")
    assert calls == [([("weight", 1)], True), "eval"]
    audio = np.ones((2, 9 * SAMPLERATE), dtype=np.float32)
    assert np.allclose(backend.extract_vocals(audio), 0.8, atol=1e-6)
    assert np.allclose(backend.extract_instrumental(audio), 0.2, atol=1e-6)
    assert placement == []
    (tmp_path / "config.json").write_text("wrong", encoding="utf-8")
    with pytest.raises(ReplayError, match="config.*checksum"):
        bgm._load_backend(bgm.MODEL_NAME, device)
    weights.write_bytes(b"wrong weights")
    with pytest.raises(ReplayError, match="checksum"):
        bgm._load_backend(bgm.MODEL_NAME, device)
    assert placement == []
    with pytest.raises(ReplayError, match="explicit mps or cpu"):
        bgm._load_backend(bgm.MODEL_NAME, "auto")
    mx.metal.is_available = lambda: False
    with pytest.raises(ReplayError, match="Apple Metal"):
        resolve_device("auto")


def test_demucs_backend_returns_input_as_vocals(monkeypatch: pytest.MonkeyPatch) -> None:
    _stub_imports(monkeypatch)
    backend = bgm._load_backend("htdemucs", "cpu")
    assert backend.sources == ("drums", "bass", "other", "vocals")
    chunk = np.stack([np.linspace(-0.5, 0.5, SAMPLERATE, dtype=np.float32)] * 2).astype(np.float32)
    vocals = backend.extract_vocals(chunk)
    assert np.array_equal(vocals, chunk)
    assert np.array_equal(backend.extract_instrumental(chunk), np.zeros_like(chunk))


def test_a_demucs_backend_on_mps_registers_the_graph_release_at_exit(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _stub_imports(monkeypatch)
    registered: list[str] = []
    monkeypatch.setattr(bgm, "release_mps_graphs_at_exit", lambda: registered.append("mps"))
    bgm._load_backend("htdemucs", "mps")
    bgm._load_backend("htdemucs", "cpu")
    assert registered == ["mps"]


def test_demucs_backend_requires_vocals_stem(monkeypatch: pytest.MonkeyPatch) -> None:
    _stub_imports(monkeypatch, sources=("drums", "bass"))
    with pytest.raises(ReplayError, match="no vocals stem"):
        bgm._load_backend("htdemucs", "cpu")


def test_demucs_backend_rejects_short_chunk(monkeypatch: pytest.MonkeyPatch) -> None:
    _stub_imports(monkeypatch, short_by=8)
    backend = bgm._load_backend("htdemucs", "cpu")
    chunk = np.zeros((2, SAMPLERATE), dtype=np.float32)
    with pytest.raises(ReplayError, match="unexpected shape"):
        backend.extract_vocals(chunk)


def test_wiener_mask_preserves_voice_phase_and_removes_separated_music(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(bgm, "SPECTRAL_BLOCK_FRAMES", 64)
    samples = 2 * SAMPLERATE
    time_axis = np.arange(samples, dtype=np.float32) / SAMPLERATE
    voice = np.float32(0.35) * np.sin(
        2.0 * np.pi * (20 * SAMPLERATE / bgm.SPECTRAL_FRAME_SAMPLES) * time_axis
    )
    music = np.float32(0.25) * np.sin(
        2.0 * np.pi * (80 * SAMPLERATE / bgm.SPECTRAL_FRAME_SAMPLES) * time_axis
    )
    speech = np.stack((voice, voice)).astype(np.float32)
    instrumental = np.stack((music, music)).astype(np.float32)
    mixture = speech + instrumental

    cleaned = apply_wiener_vocal_mask(mixture, speech)
    residual = cleaned - speech
    quality = bgm.measure_speech_quality(speech, cleaned, np.ones(samples, dtype=np.bool_))

    assert quality.level_delta_db_p05 > -0.1
    assert quality.correlation > 0.999
    assert bgm.signal_level_dbfs(instrumental) - bgm.signal_level_dbfs(residual) > 10.0


def test_mix_level_gate_confuses_removed_music_with_lost_speech() -> None:
    time_axis = np.arange(SAMPLERATE, dtype=np.float32) / SAMPLERATE
    speech = np.float32(0.2) * np.sin(2.0 * np.pi * 440.0 * time_axis)
    music = np.float32(0.2) * np.sin(2.0 * np.pi * 880.0 * time_axis)
    voice = np.stack((speech, speech)).astype(np.float32)
    mixture = voice + np.stack((music, music)).astype(np.float32)
    speech_mask = np.ones(SAMPLERATE, dtype=np.bool_)

    whole_mix = bgm.measure_speech_quality(mixture, voice, speech_mask)
    voice_only = bgm.measure_speech_quality(voice, voice, speech_mask)

    assert whole_mix.level_delta_db_p50 < -2.0
    assert voice_only.level_delta_db_p50 == pytest.approx(0.0)
    assert voice_only.level_delta_db_p05 == pytest.approx(0.0)


def test_wiener_mask_reconstructs_voice_only_input_and_rejects_bad_shapes() -> None:
    audio = np.zeros((2, 4096), dtype=np.float32)
    assert np.array_equal(apply_wiener_vocal_mask(audio, audio), audio)
    with pytest.raises(ValueError, match="matching"):
        apply_wiener_vocal_mask(audio, audio[:, :-1])


def test_squared_wiener_mask_suppresses_imperfect_vocal_stem_music() -> None:
    samples = 2 * SAMPLERATE
    time_axis = np.arange(samples, dtype=np.float32) / SAMPLERATE
    voice = 0.25 * np.sin(2.0 * np.pi * 440.0 * time_axis)
    music = 0.25 * np.sin(2.0 * np.pi * 4_000.0 * time_axis)
    mixture = np.stack((voice + music, voice + music)).astype(np.float32)
    vocals = np.stack((voice + 0.5 * music, voice + 0.5 * music)).astype(np.float32)

    filtered = apply_wiener_vocal_mask(mixture, vocals)
    music_projection = np.dot(filtered[0], music) / np.dot(music, music)
    voice_projection = np.dot(filtered[0], voice) / np.dot(voice, voice)

    assert music_projection < 0.3
    assert voice_projection > 0.9


def test_music_stem_subtraction_chooses_strength_only_when_voice_survives() -> None:
    time = np.arange(SAMPLERATE, dtype=np.float32) / SAMPLERATE
    music = np.sin(2 * np.pi * 440 * time) * np.float32(0.2)
    speech = np.sin(2 * np.pi * 173 * time) * np.float32(0.2)
    original = np.stack((np.concatenate((music, speech)),) * 2).astype(np.float32)
    estimated = np.stack((np.concatenate((music, np.zeros_like(music))),) * 2).astype(np.float32)
    music_only = np.zeros(2 * SAMPLERATE, dtype=np.bool_)
    music_only[:SAMPLERATE] = True
    quiet_speech = ~music_only
    assert np.array_equal(subtract_music(original, estimated, 0.0), original)
    selected = select_music_strength(original, estimated, music_only, quiet_speech)
    assert selected.passed is True
    assert selected.strength == 1.0
    assert selected.music_drop_db > 100.0
    assert np.max(
        np.abs(subtract_music(original, estimated, 1.0)[:, SAMPLERATE:])
    ) == pytest.approx(0.2, abs=0.002)
    contaminated = estimated.copy()
    contaminated[:, SAMPLERATE:] += np.stack((speech, speech)) * np.float32(0.8)
    rejected = select_music_strength(original, contaminated, music_only, quiet_speech)
    assert rejected.passed is False
    assert rejected.strength == 0.0


def test_music_stem_subtraction_requires_valid_controls_and_strength() -> None:
    audio = np.ones((2, 16), dtype=np.float32)
    music_only = np.zeros(16, dtype=np.bool_)
    quiet_speech = np.ones(16, dtype=np.bool_)
    with pytest.raises(ValueError, match="aligned"):
        subtract_music(audio, audio[:, :-1], 0.5)
    with pytest.raises(ValueError, match="strength"):
        subtract_music(audio, audio, 1.1)
    with pytest.raises(ValueError, match="strength"):
        subtract_music(audio, audio, True)
    with pytest.raises(ValueError, match="controls"):
        select_music_strength(audio, audio, music_only, quiet_speech)
    music_only[:8] = True
    quiet_speech[:8] = False
    with pytest.raises(ValueError, match="floor"):
        select_music_strength(np.zeros_like(audio), audio, music_only, quiet_speech)


def test_crossfade_weights_preserve_coherent_signal_level() -> None:
    weights = bgm._crossfade_weights(64)
    assert weights.shape == (2, 64)
    assert weights[0, 0] == pytest.approx(1.0)
    assert weights[1, 0] == pytest.approx(0.0)
    assert weights[:, 32].tolist() == pytest.approx([0.5, 0.5])
    assert np.abs(weights.sum(axis=0) - 1.0).max() < 1e-6
    assert np.all(np.diff(weights[1]) >= 0)


def test_identical_vocal_estimates_do_not_boost_or_clip_at_chunk_seams(tmp_path: Path) -> None:
    source = tmp_path / "input.wav"
    output = tmp_path / "vocals.wav"
    bgm._write_wav(source, np.full((2, 31 * SAMPLERATE), 0.75, dtype=np.float32))
    separate_vocals(
        source,
        output,
        SeparationConfig(chunk_seconds=30, overlap_seconds=2),
        _backend=_ScaleBackend(gain=1.0),
    )
    decoded, channels = bgm._read_wav(output)
    assert channels == 2
    assert decoded.shape == (2, 1367100)
    # Two PCM16 writes quantize the constant to 24574 / 32768 everywhere.
    assert np.unique(decoded).tolist() == [0.74993896484375]


def test_verify_music_removal() -> None:
    passed = verify_music_removal(-50.0, -65.0, -20.0, -20.2)
    assert (passed.passed, passed.removed) == (True, True)
    assert passed.instrumental_drop_db == pytest.approx(15.0)
    quiet = verify_music_removal(-70.0, -71.0, -20.0, -20.0)
    assert (quiet.passed, quiet.removed) == (True, False)
    weak = verify_music_removal(-50.0, -55.0, -20.0, -20.0)
    assert (weak.passed, weak.removed) == (False, False)
    assert "below the minimum" in weak.reason
    damaged = verify_music_removal(-50.0, -70.0, -20.0, -22.0)
    assert damaged.passed is False
    assert "speech loss" in damaged.reason
    with pytest.raises(ValueError, match="numbers"):
        verify_music_removal(True, -65.0, -20.0, -20.0)


def test_adaptive_vocal_mix_keeps_reference_when_candidate_is_quieter() -> None:
    frames = np.arange(SAMPLERATE + 5, dtype=np.float32) / SAMPLERATE
    speech = np.stack([0.5 * np.sin(2.0 * np.pi * 440.0 * frames)]).astype(np.float32)
    separated = speech * 0.1
    result = adapt_vocal_mix(speech, separated, np.ones(len(frames), dtype=np.bool_))
    assert result.mean_strength == 0.0
    assert result.protected_fraction == 1.0
    assert result.speech_level_delta_db == pytest.approx(0.0, abs=0.02)
    assert result.lower_tail_speech_delta_db >= -0.01
    assert result.speech_correlation == pytest.approx(1.0)


def test_adaptive_vocal_mix_uses_separator_when_candidate_is_louder() -> None:
    frames = np.arange(SAMPLERATE, dtype=np.float32) / SAMPLERATE
    speech = np.stack([0.5 * np.sin(2.0 * np.pi * 440.0 * frames)] * 2).astype(np.float32)
    separated = speech * 1.1
    result = adapt_vocal_mix(speech, separated, np.ones(SAMPLERATE, dtype=np.bool_))
    assert result.mean_strength == 1.0
    assert np.allclose(result.audio, separated, atol=1e-7)
    assert result.speech_level_delta_db > 0.0


def test_measure_speech_quality_reuses_full_file_mask_metrics() -> None:
    samples = SAMPLERATE + 5
    frames = np.arange(samples, dtype=np.float32) / SAMPLERATE
    original = np.stack([np.sin(2.0 * np.pi * 440.0 * frames)]).astype(np.float32)
    processed = original * np.float32(0.8)
    mask = np.ones(samples, dtype=np.bool_)
    quality = bgm.measure_speech_quality(original, processed, mask)
    assert quality.level_delta_db_p50 == pytest.approx(-1.94, abs=0.03)
    assert quality.level_delta_db_p05 == pytest.approx(-1.94, abs=0.03)
    assert quality.correlation == pytest.approx(1.0)
    assert bgm.measure_speech_quality(original, processed, np.zeros(samples, dtype=np.bool_)) == (
        bgm.SpeechQuality(0.0, 0.0, 1.0)
    )
    with pytest.raises(ValueError, match="matching"):
        bgm.measure_speech_quality(original, processed, mask[:-1])
    processed[0, 0] = np.nan
    with pytest.raises(ValueError, match="finite"):
        bgm.measure_speech_quality(original, processed, mask)


def test_adaptive_vocal_mix_without_transcript_is_noop() -> None:
    audio = np.ones((2, SAMPLERATE), dtype=np.float32)
    result = adapt_vocal_mix(audio, np.zeros_like(audio), np.zeros(SAMPLERATE, dtype=np.bool_))
    assert np.array_equal(result.audio, audio)
    assert result.mean_strength == 0.0
    assert result.speech_correlation == 1.0


def test_adaptive_vocal_mix_rejects_bad_shapes_and_nonfinite_audio() -> None:
    audio = np.ones((2, SAMPLERATE), dtype=np.float32)
    with pytest.raises(ValueError, match="matching"):
        adapt_vocal_mix(audio, audio[:1], np.ones(SAMPLERATE, dtype=np.bool_))
    audio[0, 0] = np.nan
    with pytest.raises(ValueError, match="finite"):
        adapt_vocal_mix(audio, np.ones_like(audio), np.ones(SAMPLERATE, dtype=np.bool_))


def test_signal_level_measures_full_scale_sine() -> None:
    audio = np.ones((1, SAMPLERATE), dtype=np.float32)
    assert bgm.signal_level_dbfs(audio) == pytest.approx(0.0)
    with pytest.raises(ValueError, match="non-empty"):
        bgm.signal_level_dbfs(np.empty((1, 0), dtype=np.float32))
    audio[0, 0] = np.inf
    with pytest.raises(ValueError, match="finite"):
        bgm.signal_level_dbfs(audio)


def test_instrumental_level_measures_full_scale_sine() -> None:
    frames = np.arange(SAMPLERATE) / SAMPLERATE
    sine = np.sin(2.0 * np.pi * 440.0 * frames)
    level = instrumental_level([sine.tolist(), sine.tolist()])
    assert level == pytest.approx(-3.01, abs=0.02)
    with pytest.raises(ValueError, match="at least one sample"):
        instrumental_level([[], []])
    with pytest.raises(ValueError, match="numbers"):
        instrumental_level([[True]])


def test_gather_merge_matches_the_scatter_add_merge() -> None:
    freq_indices = [np.array([0, 1, 2]), np.array([2, 3]), np.array([3, 4, 5])]
    counts = np.array([1, 1, 2, 2, 1, 1, 1, 1], dtype=np.float32)
    split = SimpleNamespace(
        filterbank=SimpleNamespace(freq_indices=freq_indices, num_bands_per_freq=counts)
    )
    generator = np.random.default_rng(7)
    masks = [
        mx.array(generator.standard_normal((1, 4, len(indices) * 2)).astype(np.float32))
        for indices in freq_indices
    ]
    expected = mx.zeros((1, 8, 4, 2))
    for indices, mask in zip(freq_indices, masks, strict=True):
        reshaped = mask.reshape(1, 4, len(indices), 2).transpose(0, 2, 1, 3)
        for position, index in enumerate(indices.tolist()):
            expected = expected.at[:, index, :, :].add(reshaped[:, position, :, :])
    expected = expected / mx.array(counts).reshape(1, -1, 1, 1)
    merged = bgm._gather_merge(mx, split)(masks, 8)
    assert np.array_equal(np.asarray(merged), np.asarray(expected))
    assert np.asarray(merged)[0, 6:].tolist() == [[[0.0, 0.0]] * 4] * 2
