"""Instrumental BGM separation with per-delivery speech-preserving adaptation.

A per-bin Wiener mask estimated from pinned MLX Mel-RoFormer vocals is applied to the original
phase. The result is adapted against the source vocal stem, so removing BGM
cannot be mistaken for speech loss and restored as music. This is one generic
signal-derived rule, not a per-recording preset.

Heavy ML dependencies stay behind optional uv extras. NumPy is a development
dependency; MLX and legacy Demucs are loaded lazily so the default locked
environment never imports them.
"""

from __future__ import annotations

import hashlib
import importlib
import time
import wave
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol

import numpy as np
from numpy.typing import NDArray

from replay.core import ReplayError
from replay.mps_graph_cache import release_mps_graphs_at_exit

MODEL_NAME = "mlx-community/mel-roformer-mlx"
MODEL_REVISION = "639de4848eb79a9e29cc8cba3c01170ddf165eef"
MODEL_SHA256 = "86eee1472d825bf978bc6b6bb104bb71cb7950c4fea9dd192a8c55ed2db28b27"
MODEL_CONFIG_SHA256 = "6acce689a874fe92c3e474d42a98352205ff46e88c8b686686ab0b57a01e72c2"
MODEL_SAMPLE_RATE = 44100
VOCALS_SOURCE = "vocals"
INSTRUMENTAL_SOURCES = ("drums", "bass", "other")

MAXIMUM_CHUNKS = 10_000
ADAPTIVE_FRAME_SAMPLES = MODEL_SAMPLE_RATE // 100
ADAPTIVE_SMOOTHING_FRAMES = 33
PROCESSING_BLOCK_FRAMES = 6_000
ADAPTIVE_OUTPUT_HEADROOM_DB = 1.5
SPECTRAL_FRAME_SAMPLES = 2_048
SPECTRAL_HOP_SAMPLES = 512
SPECTRAL_BLOCK_FRAMES = 4_096


def _import(name: str) -> Any:
    """Import an optional backend dependency without loading it on normal CLI runs."""
    try:
        return importlib.import_module(name)
    except ImportError as error:
        extra = "roformer" if name.startswith(("mlx", "huggingface")) else "separation"
        raise ReplayError(f"Install separation support: uv sync --extra {extra}") from error


class SeparationBackend(Protocol):
    """Vocals extraction behind a seam the tests can stub without torch."""

    @property
    def sources(self) -> tuple[str, ...]: ...

    def extract_vocals(self, chunk: NDArray[np.float32]) -> NDArray[np.float32]: ...

    def extract_instrumental(self, chunk: NDArray[np.float32]) -> NDArray[np.float32]: ...


class _DemucsBackend:
    """Demucs vocals extraction; torch I/O only, math stays in NumPy."""

    def __init__(self, model_name: str, device: str) -> None:
        torch = _import("torch")
        pretrained = _import("demucs.pretrained")
        apply = _import("demucs.apply")
        model = pretrained.get_model(model_name)
        if device == "mps":
            release_mps_graphs_at_exit()
        model.to(device)
        model.eval()
        names = tuple(str(name) for name in model.sources)
        if VOCALS_SOURCE not in names:
            raise ReplayError(f"Model {model_name} has no vocals stem")
        self._model: Any = model
        self._torch: Any = torch
        self._apply: Any = apply
        self._device = device
        self._sources = names

    @property
    def sources(self) -> tuple[str, ...]:
        return self._sources

    def _extract_sources(self, chunk: NDArray[np.float32]) -> NDArray[np.float32]:
        torch = self._torch
        batch = torch.stack(
            [torch.from_numpy(np.ascontiguousarray(channel)) for channel in chunk]
        ).unsqueeze(0)
        batch = batch.to(self._device)
        with torch.no_grad():
            stems = self._apply.apply_model(self._model, batch, shifts=0, split=True, overlap=0.25)
        result = np.asarray(stems[0].cpu().numpy(), dtype=np.float32)
        if result.shape != (len(self._sources), *chunk.shape):
            raise ReplayError("Model returned stems with an unexpected shape")
        return result

    def extract_vocals(self, chunk: NDArray[np.float32]) -> NDArray[np.float32]:
        vocals = self._extract_sources(chunk)[self._sources.index(VOCALS_SOURCE)]
        return np.asarray(vocals, dtype=np.float32)

    def extract_instrumental(self, chunk: NDArray[np.float32]) -> NDArray[np.float32]:
        indices = [index for index, name in enumerate(self._sources) if name != VOCALS_SOURCE]
        if not indices:
            return np.zeros_like(chunk)
        return np.asarray(
            np.sum(self._extract_sources(chunk)[indices], axis=0, dtype=np.float32),
            dtype=np.float32,
        )


class _MlxBackend:
    """Offline, pinned Mel-RoFormer with 8-second, half-overlapped inference."""

    def __init__(self, device: str) -> None:
        if device not in ("mps", "cpu"):
            raise ReplayError("Pinned MLX separation requires explicit mps or cpu")
        self._mx = _import("mlx.core")
        self._device = self._mx.cpu if device == "cpu" else self._mx.gpu
        # Include module-level arrays and lazy weight operations in the selected stream.
        with self._mx.stream(self._device):
            self._initialize()

    def _initialize(self) -> None:
        mx = self._mx
        hub = _import("huggingface_hub")
        module = _import("mlx_audio.sts.models.mel_roformer")
        cached = hub.try_to_load_from_cache(
            MODEL_NAME, "model.safetensors", revision=MODEL_REVISION
        )
        if not isinstance(cached, str):
            raise ReplayError("Pinned MLX weights are not in the local Hugging Face cache")
        weights = Path(cached)
        folder = weights.parent
        if not (folder / "config.json").is_file():
            raise ReplayError("Pinned MLX config is not in the local Hugging Face cache")
        with weights.open("rb") as stream:
            if hashlib.file_digest(stream, "sha256").hexdigest() != MODEL_SHA256:
                raise ReplayError("Pinned MLX model weights do not match the verified checksum")
        with (folder / "config.json").open("rb") as stream:
            if hashlib.file_digest(stream, "sha256").hexdigest() != MODEL_CONFIG_SHA256:
                raise ReplayError("Pinned MLX config does not match the verified checksum")
        model = module.MelRoFormer.from_pretrained(folder)
        model.load_weights(list(model.sanitize(dict(mx.load(str(weights)))).items()), strict=True)
        model.eval()
        self._model = model

    @property
    def sources(self) -> tuple[str, ...]:
        return (VOCALS_SOURCE, "instrumental")

    def extract_vocals(self, chunk: NDArray[np.float32]) -> NDArray[np.float32]:
        size = 8 * MODEL_SAMPLE_RATE
        hop = size // 2
        total = chunk.shape[1]
        output = np.zeros((2, total), dtype=np.float64)
        coverage = np.zeros(total, dtype=np.float64)
        ramp = np.sin(np.linspace(0.0, np.pi / 2.0, hop, endpoint=False)) ** 2
        for start in range(0, total, hop):
            end = min(start + size, total)
            count = end - start
            if start and count <= hop:
                break
            padded = np.zeros((2, size), dtype=np.float32)
            padded[:, :count] = chunk[:, start:end]
            with self._mx.stream(self._device):
                predicted = self._model(self._mx.array(padded[None, :, :]))[0]
                self._mx.eval(predicted)
                voice = np.asarray(predicted, dtype=np.float32)[:, :count]
            if voice.shape != (2, count) or not np.isfinite(voice).all():
                raise ReplayError("Pinned MLX model returned invalid audio")
            blend = np.ones(count, dtype=np.float64)
            if start:
                blend[: min(count, hop)] = ramp[: min(count, hop)]
            if end < total:
                blend[-hop:] *= 1.0 - ramp
            output[:, start:end] += voice * blend
            coverage[start:end] += blend
        if np.any(coverage <= 0):
            raise ReplayError("MLX overlap-add left uncovered audio")
        return np.asarray(output / coverage[None, :], dtype=np.float32)

    def extract_instrumental(self, chunk: NDArray[np.float32]) -> NDArray[np.float32]:
        return chunk - self.extract_vocals(chunk)


def _load_backend(model_name: str, device: str) -> SeparationBackend:
    if model_name == MODEL_NAME:
        return _MlxBackend(device)
    return _DemucsBackend(model_name, device)


@dataclass(frozen=True)
class SeparationConfig:
    """Generic guardrails; identical for every video, never per-ID tuned."""

    model_name: str = MODEL_NAME
    chunk_seconds: float = 600.0
    overlap_seconds: float = 5.0
    device: str = "auto"
    sample_rate: int = MODEL_SAMPLE_RATE

    def __post_init__(self) -> None:
        if not isinstance(self.model_name, str) or not self.model_name:
            raise ValueError("Separation model name must be a non-empty string")
        values = (self.chunk_seconds, self.overlap_seconds)
        if (
            any(isinstance(value, bool) for value in values)
            or not all(isinstance(value, (int, float)) for value in values)
            or not 30.0 <= self.chunk_seconds <= 3600.0
            or not 1.0 <= self.overlap_seconds <= 60.0
            or self.overlap_seconds * 2.0 >= self.chunk_seconds
        ):
            raise ValueError("Invalid separation chunking")
        if self.device not in ("auto", "mps", "cpu"):
            raise ValueError("Separation device must be auto, mps, or cpu")
        if (
            isinstance(self.sample_rate, bool)
            or not isinstance(self.sample_rate, int)
            or self.sample_rate != MODEL_SAMPLE_RATE
        ):
            raise ValueError("Separation input must be 44100 Hz; resample it first")


@dataclass(frozen=True)
class SeparationPolicy:
    """Music-specific verification gates; identical for every video."""

    minimum_instrumental_drop_db: float = 10.0
    maximum_speech_loss_db: float = 1.5
    quiet_instrumental_db: float = -60.0

    def __post_init__(self) -> None:
        values = (
            self.minimum_instrumental_drop_db,
            self.maximum_speech_loss_db,
            self.quiet_instrumental_db,
        )
        if (
            any(isinstance(value, bool) for value in values)
            or not all(isinstance(value, (int, float)) for value in values)
            or not 0.0 < self.minimum_instrumental_drop_db <= 40.0
            or not 0.0 <= self.maximum_speech_loss_db <= 6.0
            or not -90.0 <= self.quiet_instrumental_db <= 0.0
        ):
            raise ValueError("Invalid separation policy")


@dataclass(frozen=True)
class SpeechQuality:
    """Full-file voice change measured between separated vocal stems."""

    level_delta_db_p50: float
    level_delta_db_p05: float
    correlation: float


@dataclass(frozen=True)
class AdaptiveMixResult:
    """Per-recording blend and speech-preservation measurements."""

    audio: NDArray[np.float32]
    mean_strength: float
    protected_fraction: float
    speech_level_delta_db: float
    lower_tail_speech_delta_db: float
    speech_correlation: float


@dataclass(frozen=True)
class MusicStrengthSelection:
    """Strength measured against independent music-only and quiet-speech controls."""

    strength: float
    music_drop_db: float
    speech: SpeechQuality
    passed: bool


@dataclass(frozen=True)
class SeparationResult:
    """Throughput proof for one separated file."""

    audio_seconds: float
    wall_seconds: float
    realtime_factor: float


@dataclass(frozen=True)
class MusicRemovalVerification:
    """Measured proof that the instrumental bed dropped and speech survived."""

    passed: bool
    instrumental_drop_db: float
    speech_loss_db: float
    removed: bool
    reason: str


def resolve_device(requested: str, model_name: str = MODEL_NAME) -> str:
    """Honor explicit CPU; auto/mps never silently fall back from Metal."""
    if requested not in ("auto", "mps", "cpu"):
        raise ValueError("Separation device must be auto, mps, or cpu")
    if model_name == MODEL_NAME:
        if requested == "cpu":
            return "cpu"
        if not bool(_import("mlx.core").metal.is_available()):
            raise ReplayError("Pinned MLX separation requires Apple Metal")
        return "mps"
    if requested == "cpu":
        return "cpu"
    torch = _import("torch")
    available = bool(torch.backends.mps.is_available())
    if requested == "mps" and not available:
        raise ReplayError("MPS is not available on this machine")
    return "mps" if available else "cpu"


def _chunk_bounds(total: int, chunk: int, overlap: int) -> list[tuple[int, int]]:
    """Window bounds covering every sample; a runt tail merges backwards."""
    if overlap * 2 >= chunk:
        raise ValueError("Separation overlap must be below half the chunk")
    step = chunk - overlap
    starts = list(range(0, total, step))
    if len(starts) > 1 and total - starts[-1] <= overlap:
        starts.pop()
    bounds = [(start, min(start + chunk, total)) for start in starts]
    if len(bounds) > MAXIMUM_CHUNKS:
        raise ValueError("Separation chunk count exceeds budget")
    return bounds


def _read_wav(path: Path) -> tuple[NDArray[np.float32], int]:
    """Read 16-bit PCM mono/stereo; anything else fails loudly."""
    try:
        with wave.open(str(path), "rb") as handle:
            channels = handle.getnchannels()
            width = handle.getsampwidth()
            rate = handle.getframerate()
            frames = handle.getnframes()
            raw = handle.readframes(frames)
    except (wave.Error, OSError) as error:
        raise ReplayError(f"Cannot read WAV input: {path}") from error
    if channels not in (1, 2) or width != 2 or rate != MODEL_SAMPLE_RATE:
        raise ReplayError("Separation input must be 16-bit mono/stereo at 44100 Hz")
    if frames <= 0:
        raise ReplayError("Separation input has no audio frames")
    samples = np.frombuffer(raw, dtype=np.int16).reshape(-1, channels).T.astype(np.float32)
    samples /= 32768.0
    return samples, channels


def _write_wav(path: Path, audio: NDArray[np.float32]) -> None:
    """Write 16-bit PCM in bounded blocks; never overwrite an existing output."""
    if path.exists():
        raise ReplayError(f"Separation output already exists: {path}")
    if not path.parent.exists():
        raise ReplayError(f"Separation output directory is missing: {path.parent}")
    if audio.ndim != 2 or audio.shape[0] not in (1, 2):
        raise ReplayError("Separation output has an invalid audio shape")
    block_samples = PROCESSING_BLOCK_FRAMES * ADAPTIVE_FRAME_SAMPLES
    try:
        with wave.open(str(path), "wb") as handle:
            handle.setnchannels(audio.shape[0])
            handle.setsampwidth(2)
            handle.setframerate(MODEL_SAMPLE_RATE)
            for start in range(0, audio.shape[1], block_samples):
                end = min(start + block_samples, audio.shape[1])
                encoded = (
                    (np.clip(audio[:, start:end].T, -1.0, 1.0) * 32767.0).astype(np.int16).tobytes()
                )
                handle.writeframesraw(encoded)
    except (wave.Error, OSError) as error:
        raise ReplayError(f"Cannot write WAV output: {path}") from error


def _crossfade_weights(length: int) -> NDArray[np.float32]:
    """Complementary weights: overlapping estimates describe the same waveform."""
    positions = np.arange(length, dtype=np.float32) / float(length)
    fresh = np.square(np.sin(0.5 * np.pi * positions)).astype(np.float32)
    return np.stack((1.0 - fresh, fresh))


def _spectrogram(audio: NDArray[np.float32], window: NDArray[np.float32]) -> NDArray[np.complex64]:
    half_frame = SPECTRAL_FRAME_SAMPLES // 2
    padded = np.pad(audio, ((0, 0), (half_frame, half_frame)), mode="reflect")
    frames = np.lib.stride_tricks.sliding_window_view(padded, SPECTRAL_FRAME_SAMPLES, axis=1)[
        :, ::SPECTRAL_HOP_SAMPLES, :
    ]
    return np.fft.rfft(frames * window, axis=-1).astype(np.complex64)


def _synthesize(
    spectrum: NDArray[np.complex64],
    sample_count: int,
    window: NDArray[np.float32],
) -> NDArray[np.float32]:
    frames = np.fft.irfft(spectrum, n=SPECTRAL_FRAME_SAMPLES, axis=-1).astype(np.float32)
    frames *= window
    frame_count = frames.shape[1]
    output_length = (frame_count - 1) * SPECTRAL_HOP_SAMPLES + SPECTRAL_FRAME_SAMPLES
    output = np.zeros((frames.shape[0], output_length), dtype=np.float64)
    weight = np.zeros(output_length, dtype=np.float64)
    window_power = window.astype(np.float64) ** 2
    for index in range(frame_count):
        start = index * SPECTRAL_HOP_SAMPLES
        end = start + SPECTRAL_FRAME_SAMPLES
        output[:, start:end] += frames[:, index].astype(np.float64)
        weight[start:end] += window_power
    first = SPECTRAL_FRAME_SAMPLES // 2
    denominator = weight[first : first + sample_count]
    if denominator.shape[0] != sample_count or np.any(denominator <= 0.0):
        raise ReplayError("Spectral synthesis left uncovered audio samples")
    return (output[:, first : first + sample_count] / denominator).astype(np.float32)


def apply_wiener_vocal_mask(
    original: NDArray[np.float32],
    vocals: NDArray[np.float32],
) -> NDArray[np.float32]:
    """Apply a per-bin Wiener mask to the original phase in bounded blocks."""
    if (
        original.ndim != 2
        or original.shape[0] not in (1, 2)
        or original.shape != vocals.shape
        or original.shape[1] == 0
    ):
        raise ValueError("Audio arrays must have matching non-empty mono/stereo shapes")
    if not np.isfinite(original).all() or not np.isfinite(vocals).all():
        raise ValueError("Audio arrays must contain only finite samples")

    window = np.hanning(SPECTRAL_FRAME_SAMPLES).astype(np.float32)
    output = np.empty_like(original)
    block_samples = SPECTRAL_BLOCK_FRAMES * SPECTRAL_HOP_SAMPLES
    for block_start in range(0, original.shape[1], block_samples):
        block_end = min(block_start + block_samples, original.shape[1])
        start = max(0, block_start - SPECTRAL_FRAME_SAMPLES)
        start -= start % SPECTRAL_HOP_SAMPLES
        end = min(original.shape[1], block_end + SPECTRAL_FRAME_SAMPLES)
        bounds = slice(start, end)
        mixture_spectrum = _spectrogram(original[:, bounds], window)
        vocal_spectrum = _spectrogram(vocals[:, bounds], window)
        instrumental_spectrum = mixture_spectrum - vocal_spectrum
        vocal_power = np.abs(vocal_spectrum) ** 2
        instrumental_power = np.abs(instrumental_spectrum) ** 2
        gain = vocal_power / np.maximum(vocal_power + instrumental_power, 1e-12)
        filtered = _synthesize(mixture_spectrum * np.square(gain), end - start, window)
        crop_start = block_start - start
        output[:, block_start:block_end] = filtered[
            :, crop_start : crop_start + block_end - block_start
        ]
    return output


def subtract_music(
    original: NDArray[np.float32],
    music: NDArray[np.float32],
    strength: float,
) -> NDArray[np.float32]:
    """Keep the original speech/effects; subtract only an aligned music estimate."""
    if (
        original.ndim != 2
        or original.shape[0] not in (1, 2)
        or original.shape != music.shape
        or original.shape[1] == 0
        or not np.isfinite(original).all()
        or not np.isfinite(music).all()
        or isinstance(strength, bool)
        or not isinstance(strength, (int, float))
        or not np.isfinite(strength)
        or not 0.0 <= strength <= 1.0
    ):
        raise ValueError("Music subtraction needs finite aligned audio and a strength in [0, 1]")
    return original - np.float32(strength) * music


def select_music_strength(
    original: NDArray[np.float32],
    music: NDArray[np.float32],
    music_only: NDArray[np.bool_],
    quiet_speech: NDArray[np.bool_],
    policy: SeparationPolicy | None = None,
) -> MusicStrengthSelection:
    """Choose a shared strength from this input's independent control intervals.

    SRT and a music estimate alone cannot prove true voice preservation. The
    caller must supply independently selected controls and verify real audio.
    """
    active = policy if policy is not None else SeparationPolicy()
    subtract_music(original, music, 0.0)
    if (
        music_only.shape != (original.shape[1],)
        or quiet_speech.shape != music_only.shape
        or music_only.dtype != np.bool_
        or quiet_speech.dtype != np.bool_
        or not music_only.any()
        or not quiet_speech.any()
        or np.any(music_only & quiet_speech)
    ):
        raise ValueError("Non-overlapping music-only and quiet-speech controls are required")
    before_db = signal_level_dbfs(original[:, music_only])
    if before_db <= active.quiet_instrumental_db:
        raise ValueError("Music-only control is below the measurable instrumental floor")
    selected = MusicStrengthSelection(0.0, 0.0, SpeechQuality(0.0, 0.0, 1.0), False)
    for strength in (0.25, 0.5, 0.75, 1.0):
        candidate = subtract_music(original, music, strength)
        speech = measure_speech_quality(original, candidate, quiet_speech)
        drop = before_db - signal_level_dbfs(candidate[:, music_only])
        if (
            speech.level_delta_db_p50 < -active.maximum_speech_loss_db
            or speech.level_delta_db_p05 < -active.maximum_speech_loss_db
            or speech.correlation < 0.995
            or drop < active.minimum_instrumental_drop_db
        ):
            continue
        if drop > selected.music_drop_db:
            selected = MusicStrengthSelection(strength, drop, speech, True)
    return selected


def adapt_vocal_mix(
    original: NDArray[np.float32],
    vocals: NDArray[np.float32],
    speech_mask: NDArray[np.bool_],
    policy: SeparationPolicy | None = None,
) -> AdaptiveMixResult:
    """Choose each 10ms blend weight from separated-voice evidence.

    ``original`` is the source vocal estimate, not the music-bearing mix. Any
    transcript-covered frame is protected against more than the configured
    RMS loss. With no speech mask the safe result is that voice reference.
    """
    active = policy if policy is not None else SeparationPolicy()
    if (
        original.ndim != 2
        or original.shape[0] not in (1, 2)
        or original.shape != vocals.shape
        or original.shape[1] == 0
        or speech_mask.ndim != 1
        or speech_mask.shape[0] != original.shape[1]
        or speech_mask.dtype != np.bool_
    ):
        raise ValueError("Audio arrays and speech mask must have matching mono/stereo shapes")
    if not np.isfinite(original).all() or not np.isfinite(vocals).all():
        raise ValueError("Audio arrays must contain only finite samples")
    protected_fraction = float(np.mean(speech_mask))
    if not speech_mask.any():
        return AdaptiveMixResult(original.copy(), 0.0, 0.0, 0.0, 0.0, 1.0)

    sample_count = original.shape[1]
    frame_count = (sample_count + ADAPTIVE_FRAME_SAMPLES - 1) // ADAPTIVE_FRAME_SAMPLES
    padding = frame_count * ADAPTIVE_FRAME_SAMPLES - sample_count
    source_padded = np.pad(original, ((0, 0), (0, padding))) if padding else original
    vocals_padded = np.pad(vocals, ((0, 0), (0, padding))) if padding else vocals
    mask_padded = np.pad(speech_mask, (0, padding)) if padding else speech_mask
    source_frames = source_padded.reshape(original.shape[0], frame_count, ADAPTIVE_FRAME_SAMPLES)
    vocal_frames = vocals_padded.reshape(original.shape[0], frame_count, ADAPTIVE_FRAME_SAMPLES)
    source_energy = np.einsum("cft,cft->f", source_frames, source_frames, dtype=np.float64)
    vocal_energy = np.einsum("cft,cft->f", vocal_frames, vocal_frames, dtype=np.float64)
    cross_energy = np.einsum("cft,cft->f", source_frames, vocal_frames, dtype=np.float64)
    divisor = float(original.shape[0] * ADAPTIVE_FRAME_SAMPLES)
    source_energy /= divisor
    vocal_energy /= divisor
    cross_energy /= divisor
    protected_frames = mask_padded.reshape(frame_count, ADAPTIVE_FRAME_SAMPLES).any(axis=1)
    adaptive_loss_limit = max(0.0, active.maximum_speech_loss_db - ADAPTIVE_OUTPUT_HEADROOM_DB)
    target_energy = source_energy * 10.0 ** (-adaptive_loss_limit / 10.0)
    needs_protection = protected_frames & (vocal_energy < target_energy)
    strengths = np.ones(frame_count, dtype=np.float64)
    lower = np.zeros(frame_count, dtype=np.float64)
    upper = np.ones(frame_count, dtype=np.float64)
    difference_energy = source_energy + vocal_energy - 2.0 * cross_energy
    for _ in range(24):
        middle = (lower + upper) / 2.0
        energy = (
            source_energy
            + 2.0 * middle * (cross_energy - source_energy)
            + middle**2 * difference_energy
        )
        safe = energy >= target_energy
        lower = np.where(needs_protection & safe, middle, lower)
        upper = np.where(needs_protection & ~safe, middle, upper)
    strengths[needs_protection] = lower[needs_protection]

    half_window = ADAPTIVE_SMOOTHING_FRAMES // 2
    kernel = np.hanning(ADAPTIVE_SMOOTHING_FRAMES)
    kernel /= kernel.sum()
    smoothed = np.convolve(
        np.pad(strengths, (half_window, half_window), mode="edge"),
        kernel,
        mode="valid",
    )
    strengths = np.minimum(smoothed, strengths).astype(np.float32)
    output = np.empty_like(original)
    block_samples = PROCESSING_BLOCK_FRAMES * ADAPTIVE_FRAME_SAMPLES
    for start in range(0, sample_count, block_samples):
        end = min(start + block_samples, sample_count)
        first_frame = start // ADAPTIVE_FRAME_SAMPLES
        last_frame = (end + ADAPTIVE_FRAME_SAMPLES - 1) // ADAPTIVE_FRAME_SAMPLES
        frame_weights = np.repeat(strengths[first_frame:last_frame], ADAPTIVE_FRAME_SAMPLES)
        weights = frame_weights[: end - start]
        output[:, start:end] = (
            original[:, start:end] + (vocals[:, start:end] - original[:, start:end]) * weights
        )

    quality = measure_speech_quality(original, output, speech_mask)
    return AdaptiveMixResult(
        audio=output,
        mean_strength=float(np.mean(strengths)),
        protected_fraction=protected_fraction,
        speech_level_delta_db=quality.level_delta_db_p50,
        lower_tail_speech_delta_db=quality.level_delta_db_p05,
        speech_correlation=quality.correlation,
    )


def measure_speech_quality(
    original: NDArray[np.float32],
    processed: NDArray[np.float32],
    speech_mask: NDArray[np.bool_],
) -> SpeechQuality:
    """Measure full-file level change and correlation on transcript speech."""
    if (
        original.ndim != 2
        or original.shape[0] not in (1, 2)
        or original.shape != processed.shape
        or original.shape[1] == 0
        or speech_mask.ndim != 1
        or speech_mask.shape[0] != original.shape[1]
        or speech_mask.dtype != np.bool_
    ):
        raise ValueError("Audio arrays and speech mask must have matching mono/stereo shapes")
    if not np.isfinite(original).all() or not np.isfinite(processed).all():
        raise ValueError("Audio arrays must contain only finite samples")
    if not speech_mask.any():
        return SpeechQuality(0.0, 0.0, 1.0)

    sample_count = original.shape[1]
    frame_count = (sample_count + ADAPTIVE_FRAME_SAMPLES - 1) // ADAPTIVE_FRAME_SAMPLES
    padding = frame_count * ADAPTIVE_FRAME_SAMPLES - sample_count
    source_padded = np.pad(original, ((0, 0), (0, padding))) if padding else original
    processed_padded = np.pad(processed, ((0, 0), (0, padding))) if padding else processed
    mask_padded = np.pad(speech_mask, (0, padding)) if padding else speech_mask
    shape = (original.shape[0], frame_count, ADAPTIVE_FRAME_SAMPLES)
    source_frames = source_padded.reshape(shape)
    processed_frames = processed_padded.reshape(shape)
    source_energy = np.einsum("cft,cft->f", source_frames, source_frames, dtype=np.float64)
    processed_energy = np.einsum("cft,cft->f", processed_frames, processed_frames, dtype=np.float64)
    protected = mask_padded.reshape(frame_count, ADAPTIVE_FRAME_SAMPLES).any(axis=1)
    delta = 10.0 * np.log10(
        (processed_energy[protected] + 1e-18) / (source_energy[protected] + 1e-18)
    )

    sums = np.zeros(5, dtype=np.float64)
    block_samples = PROCESSING_BLOCK_FRAMES * ADAPTIVE_FRAME_SAMPLES
    for start in range(0, sample_count, block_samples):
        end = min(start + block_samples, sample_count)
        selected = speech_mask[start:end]
        if not selected.any():
            continue
        source_samples = original[:, start:end][:, selected].astype(np.float64).ravel()
        processed_samples = processed[:, start:end][:, selected].astype(np.float64).ravel()
        sums += (
            source_samples.sum(),
            processed_samples.sum(),
            np.dot(source_samples, source_samples),
            np.dot(processed_samples, processed_samples),
            np.dot(source_samples, processed_samples),
        )
    sample_total = float(speech_mask.sum() * original.shape[0])
    source_sum, processed_sum, source_square, processed_square, cross = sums
    source_variance = source_square - source_sum**2 / sample_total
    processed_variance = processed_square - processed_sum**2 / sample_total
    covariance = cross - source_sum * processed_sum / sample_total
    correlation = (
        float(covariance / np.sqrt(source_variance * processed_variance))
        if source_variance > 0.0 and processed_variance > 0.0
        else 1.0
    )
    return SpeechQuality(
        level_delta_db_p50=float(np.percentile(delta, 50)),
        level_delta_db_p05=float(np.percentile(delta, 5)),
        correlation=correlation,
    )


def separate_vocals(
    input_path: str | Path,
    output_path: str | Path,
    config: SeparationConfig | None = None,
    _backend: SeparationBackend | None = None,
) -> SeparationResult:
    """Keep the vocals stem of one WAV file with chunked overlap-add inference.

    Long files stream through fixed-size windows, so peak RAM stays near one
    chunk instead of growing with duration. The output never overwrites.
    """
    active = config if config is not None else SeparationConfig()
    source = Path(input_path)
    destination = Path(output_path)
    audio, in_channels = _read_wav(source)
    total = int(audio.shape[1])
    chunk = int(active.chunk_seconds * MODEL_SAMPLE_RATE)
    overlap = int(active.overlap_seconds * MODEL_SAMPLE_RATE)
    bounds = _chunk_bounds(total, chunk, overlap)
    device = resolve_device(active.device, active.model_name) if _backend is None else active.device
    backend = _backend if _backend is not None else _load_backend(active.model_name, device)
    if VOCALS_SOURCE not in backend.sources:
        raise ReplayError("Separation backend has no vocals stem")

    stereo = audio if in_channels == 2 else np.stack((audio[0], audio[0]))
    weights = _crossfade_weights(overlap)
    output = np.zeros((2, total), dtype=np.float32)
    started = time.perf_counter()
    for index, (start, end) in enumerate(bounds):
        vocals = np.asarray(backend.extract_vocals(stereo[:, start:end]), dtype=np.float32)
        if vocals.shape != (2, end - start):
            raise ReplayError("Separation backend returned a misshapen chunk")
        if index == 0:
            output[:, start:end] = vocals
        else:
            head = (
                output[:, start : start + overlap] * weights[0] + vocals[:, :overlap] * weights[1]
            )
            output[:, start : start + overlap] = head
            output[:, start + overlap : end] = vocals[:, overlap:]
    wall = time.perf_counter() - started
    cleaned = output if in_channels == 2 else output[0:1]
    _write_wav(destination, np.ascontiguousarray(cleaned))
    audio_seconds = total / MODEL_SAMPLE_RATE
    return SeparationResult(
        audio_seconds=audio_seconds,
        wall_seconds=wall,
        realtime_factor=audio_seconds / wall if wall > 0.0 else float("inf"),
    )


def verify_music_removal(
    before_instrumental_db: float,
    after_instrumental_db: float,
    before_speech_db: float,
    after_speech_db: float,
    policy: SeparationPolicy | None = None,
) -> MusicRemovalVerification:
    """Require measured instrumental drop with negligible speech loss.

    A delivery whose instrumental bed already sits at or below the quiet
    floor passes as nothing-to-remove: there is no music to take out, and
    penalizing that case would reward per-ID threshold shopping.
    """
    active = policy if policy is not None else SeparationPolicy()
    for value in (
        before_instrumental_db,
        after_instrumental_db,
        before_speech_db,
        after_speech_db,
    ):
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise ValueError("Music removal verification levels must be numbers")
    drop = before_instrumental_db - after_instrumental_db
    loss = before_speech_db - after_speech_db
    if loss > active.maximum_speech_loss_db:
        return MusicRemovalVerification(
            passed=False,
            instrumental_drop_db=drop,
            speech_loss_db=loss,
            removed=False,
            reason="Measured speech loss exceeds the maximum",
        )
    if before_instrumental_db <= active.quiet_instrumental_db:
        return MusicRemovalVerification(
            passed=True,
            instrumental_drop_db=drop,
            speech_loss_db=loss,
            removed=False,
            reason="No instrumental bed to remove",
        )
    if drop < active.minimum_instrumental_drop_db:
        return MusicRemovalVerification(
            passed=False,
            instrumental_drop_db=drop,
            speech_loss_db=loss,
            removed=False,
            reason="Measured instrumental drop is below the minimum",
        )
    return MusicRemovalVerification(
        passed=True,
        instrumental_drop_db=drop,
        speech_loss_db=loss,
        removed=True,
        reason="Instrumental bed removed with speech preserved",
    )


def policy_from_mapping(values: Mapping[str, object]) -> SeparationPolicy:
    """Build a policy from caller overrides; unknown keys are rejected."""
    names = (
        "minimum_instrumental_drop_db",
        "maximum_speech_loss_db",
        "quiet_instrumental_db",
    )
    unknown = set(values) - set(names)
    if unknown:
        raise ValueError("Unknown separation policy keys")
    selected: dict[str, float] = {}
    for name in names:
        if name in values:
            value = values[name]
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                raise ValueError("Separation policy overrides must be numbers")
            selected[name] = float(value)
    return SeparationPolicy(**selected)


def signal_level_dbfs(audio: NDArray[np.float32]) -> float:
    """Measure mean-square level of a channel-first audio window."""
    if audio.ndim != 2 or audio.shape[0] not in (1, 2) or audio.shape[1] == 0:
        raise ValueError("Audio level needs non-empty mono/stereo samples")
    if not np.isfinite(audio).all():
        raise ValueError("Audio level samples must be finite")
    energy = float(np.mean(audio * audio, dtype=np.float64))
    return 10.0 * float(np.log10(energy + 1e-18))


def instrumental_level(channels: Sequence[Sequence[float]]) -> float:
    """Mean-square level in dBFS of instrumental (non-vocals) stems."""
    energy = 0.0
    count = 0
    for channel in channels:
        for sample in channel:
            if isinstance(sample, bool) or not isinstance(sample, (int, float)):
                raise ValueError("Instrumental samples must be numbers")
            energy += float(sample) * float(sample)
            count += 1
    if count == 0:
        raise ValueError("Instrumental level needs at least one sample")
    return 10.0 * float(np.log10(energy / count))
