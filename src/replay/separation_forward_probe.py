"""Bounded CPU compile comparison; never a production or quality approval.

Only the requested prepared PCM interval is read. A compiled wrapper shares the
original pinned model, while the unchanged backend owns windowing and overlap
add. No existing backend instance, global device or production module is patched.
"""

from __future__ import annotations

import argparse
import hashlib
import os
import resource
import subprocess
import sys
import time
import wave
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from numpy.typing import NDArray

from replay.audio_cpu_probe import audit_descriptors, sha256
from replay.bgm_separation import (
    MODEL_CONFIG_SHA256,
    MODEL_REVISION,
    MODEL_SAMPLE_RATE,
    MODEL_SHA256,
    _MlxBackend,
    _write_wav,
)
from replay.core import ReplayError
from replay.native_invocation import canonical, durable, identity, require

Forward = Callable[[object], object]
Auditor = Callable[[Path, str], None]
BackendFactory = Callable[[], _MlxBackend]
MAX_SAMPLES = 60 * MODEL_SAMPLE_RATE
MAX_SOURCE_SAMPLES = 21600 * MODEL_SAMPLE_RATE


@dataclass(frozen=True)
class PCMInterval:
    audio: NDArray[np.float32]
    sha256: str
    source_samples: int
    source_identity: dict[str, int]


class CompiledBackend(_MlxBackend):
    """Share pinned weights; inherit the original overlap-add implementation."""

    def __init__(self, original: _MlxBackend) -> None:
        self._mx = original._mx
        self._device = original._device
        require(self._device == self._mx.cpu, "Compile probe requires CPU backend")
        model: object = original._model
        compiler: object = getattr(self._mx, "compile", None)
        if not callable(model) or not callable(compiler):
            raise ReplayError("Missing callable model or mx.compile")

        def forward(value: object) -> object:
            result: object = model(value)
            return result

        with self._mx.stream(self._device):
            compiled: object = compiler(forward)
        if not callable(compiled):
            raise ReplayError("mx.compile did not return a callable")
        self._model = compiled


def load_cpu_backend() -> _MlxBackend:
    return _MlxBackend("cpu")


def read_interval(source: Path, *, start_sample: int, samples: int) -> PCMInterval:
    """Read at most 60 seconds twice across the run, never the whole source."""
    require(type(start_sample) is int and start_sample >= 0, "Invalid start sample")
    require(type(samples) is int and 0 < samples <= MAX_SAMPLES, "Sample budget is 60 seconds")
    source = canonical(str(source))
    require(source.is_relative_to(Path.cwd()), "Prepared PCM must stay inside checkout")
    before = identity(source)
    descriptor = os.open(source, os.O_RDONLY | os.O_NOFOLLOW)
    with os.fdopen(descriptor, "rb") as stream, wave.open(stream, "rb") as audio:
        require(
            (audio.getnchannels(), audio.getsampwidth(), audio.getframerate(), audio.getcomptype())
            == (2, 2, MODEL_SAMPLE_RATE, "NONE"),
            "Expected stereo PCM16/44100 prepared input",
        )
        total = audio.getnframes()
        require(0 < total <= MAX_SOURCE_SAMPLES, "Prepared PCM exceeds six hours")
        require(start_sample + samples <= total, "Requested interval exceeds source")
        audio.setpos(start_sample)
        raw = audio.readframes(samples)
    require(len(raw) == samples * 4, "Truncated prepared PCM interval")
    require(identity(source) == before, "Prepared PCM changed while reading interval")
    decoded = np.frombuffer(raw, dtype="<i2").reshape(samples, 2).T.astype(np.float32)
    decoded /= 32768.0
    return PCMInterval(decoded, hashlib.sha256(raw).hexdigest(), total, before)


def _correlation(left: NDArray[np.float64], right: NDArray[np.float64]) -> float | None:
    first = left.ravel() - float(np.mean(left))
    second = right.ravel() - float(np.mean(right))
    denominator = float(np.sqrt(np.sum(first * first) * np.sum(second * second)))
    if denominator == 0:
        return None
    return float(np.sum(first * second) / denominator)


def compare_audio(
    baseline: NDArray[np.float32], candidate: NDArray[np.float32]
) -> dict[str, object]:
    """Report numerical differences without inventing a quality tolerance."""
    require(
        baseline.ndim == 2
        and baseline.shape[0] == 2
        and baseline.shape[1] > 0
        and candidate.shape == baseline.shape
        and bool(np.isfinite(baseline).all())
        and bool(np.isfinite(candidate).all()),
        "Invalid comparison audio",
    )
    left = baseline.astype(np.float64)
    right = candidate.astype(np.float64)
    difference = right - left
    # Match the unchanged backend's saved PCM16 clipping/truncation semantics.
    pcm_left = (np.clip(baseline, -1.0, 1.0) * 32767.0).astype(np.int16).astype(np.float64)
    pcm_right = (np.clip(candidate, -1.0, 1.0) * 32767.0).astype(np.int16).astype(np.float64)
    pcm_difference = pcm_right - pcm_left
    return {
        "sampleCount": int(baseline.shape[1]),
        "floatMaxAbsoluteDifference": float(np.max(np.abs(difference))),
        "floatRMSDifference": float(np.sqrt(np.mean(difference * difference))),
        "floatCorrelation": _correlation(left, right),
        "pcm16MaxAbsoluteDifference": float(np.max(np.abs(pcm_difference))),
        "pcm16RMSDifference": float(np.sqrt(np.mean(pcm_difference * pcm_difference))),
        "pcm16Correlation": _correlation(pcm_left, pcm_right),
        "pcm16DifferingChannelSamples": int(np.count_nonzero(pcm_difference)),
        "qualityApproved": False,
    }


def _local_environment() -> dict[str, str]:
    root = Path.cwd()
    checked: dict[str, str] = {}
    for name in ("TMPDIR", "TMP", "TEMP", "XDG_CACHE_HOME", "HF_HOME", "HF_HUB_CACHE"):
        directory = canonical(os.environ.get(name))
        require(directory.is_relative_to(root) and directory.is_dir(), f"{name} must stay local")
        checked[name] = str(directory)
    require(
        os.environ.get("HF_HUB_OFFLINE") == "1" and os.environ.get("TRANSFORMERS_OFFLINE") == "1",
        "Probe requires explicit offline model loading",
    )
    require("MLX_DISABLE_COMPILE" not in os.environ, "Compile comparison cannot disable mx.compile")
    return checked


def _code_pins() -> dict[str, str]:
    return {
        str(path): sha256(path)
        for path in (
            Path(__file__).resolve(),
            Path(__file__).with_name("bgm_separation.py").resolve(),
            Path(__file__).with_name("audio_cpu_probe.py").resolve(),
        )
    }


def _kernel_peak_rss() -> int:
    peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return peak if sys.platform == "darwin" else peak * 1024


def run(
    source: Path,
    directory: Path,
    *,
    start_sample: int,
    samples: int,
    backend_factory: BackendFactory = load_cpu_backend,
    auditor: Auditor = audit_descriptors,
) -> Path:
    """Run baseline, compiled first call and compiled warm call in one CPU process."""
    started = time.monotonic()
    environment = _local_environment()
    interval = read_interval(source, start_sample=start_sample, samples=samples)
    directory = canonical(str(directory))
    require(directory.is_relative_to(Path.cwd() / "edits/work"), "Probe output must stay local")
    directory.mkdir(mode=0o700)
    pins = _code_pins()
    durable(
        directory / "input.json",
        {
            "source": str(source),
            "sourceIdentity": interval.source_identity,
            "sourceSamples": interval.source_samples,
            "startSample": start_sample,
            "sampleCount": samples,
            "sampleRate": MODEL_SAMPLE_RATE,
            "intervalPCMSHA256": interval.sha256,
            "wholeSourceHashed": False,
            "modelRevision": MODEL_REVISION,
            "weightsSHA256": MODEL_SHA256,
            "configSHA256": MODEL_CONFIG_SHA256,
            "device": "cpu",
            "windowSamples": 8 * MODEL_SAMPLE_RATE,
            "hopSamples": 4 * MODEL_SAMPLE_RATE,
            "codePins": pins,
            "environment": environment,
            "threadEnvironment": {
                name: os.environ.get(name)
                for name in (
                    "OMP_NUM_THREADS",
                    "VECLIB_MAXIMUM_THREADS",
                    "OPENBLAS_NUM_THREADS",
                    "MKL_NUM_THREADS",
                )
            },
            "deliveryApproved": False,
        },
    )
    auditor(directory, "before-load")
    phase_started = time.monotonic()
    backend = backend_factory()
    load_seconds = time.monotonic() - phase_started
    phase_rss = {"afterLoad": _kernel_peak_rss()}
    auditor(directory, "after-load")
    phase_started = time.monotonic()
    baseline = backend.extract_vocals(interval.audio)
    baseline_seconds = time.monotonic() - phase_started
    phase_rss["afterBaseline"] = _kernel_peak_rss()
    auditor(directory, "after-baseline")
    phase_started = time.monotonic()
    compiled = CompiledBackend(backend)
    compile_seconds = time.monotonic() - phase_started
    phase_rss["afterCompileWrapper"] = _kernel_peak_rss()
    auditor(directory, "after-compile-wrapper")
    phase_started = time.monotonic()
    first = compiled.extract_vocals(interval.audio)
    first_seconds = time.monotonic() - phase_started
    phase_rss["afterCompiledFirst"] = _kernel_peak_rss()
    auditor(directory, "after-compiled-first")
    phase_started = time.monotonic()
    warm = compiled.extract_vocals(interval.audio)
    warm_seconds = time.monotonic() - phase_started
    phase_rss["afterCompiledWarm"] = _kernel_peak_rss()
    auditor(directory, "after-compiled-warm")
    require(
        baseline.shape == first.shape == warm.shape == (2, samples), "Output sample count changed"
    )
    first_comparison = compare_audio(baseline, first)
    warm_comparison = compare_audio(baseline, warm)
    repeated_comparison = compare_audio(first, warm)
    after = read_interval(source, start_sample=start_sample, samples=samples)
    require(
        after.sha256 == interval.sha256
        and after.source_identity == interval.source_identity
        and _code_pins() == pins,
        "Probe source interval or code changed",
    )
    output_pins: dict[str, str] = {}
    for name, audio in (("baseline", baseline), ("compiled-first", first), ("compiled-warm", warm)):
        destination = directory / f"{name}.wav"
        _write_wav(destination, audio)
        output_pins[str(destination)] = sha256(destination)
    auditor(directory, "after-save")
    result = directory / "result.json"
    durable(
        result,
        {
            "version": 1,
            "device": "cpu",
            "sampleCount": samples,
            "audioSeconds": samples / MODEL_SAMPLE_RATE,
            "phaseSeconds": {
                "modelLoad": load_seconds,
                "baseline": baseline_seconds,
                "compileWrapper": compile_seconds,
                "compiledFirst": first_seconds,
                "compiledWarm": warm_seconds,
                "total": time.monotonic() - started,
            },
            "kernelPeakRSSBytes": _kernel_peak_rss(),
            "phaseKernelPeakRSSBytes": phase_rss,
            "kernelPeakRSSIsCumulative": True,
            "compiledFirstComparison": first_comparison,
            "compiledWarmComparison": warm_comparison,
            "compiledRepeatComparison": repeated_comparison,
            "outputs": output_pins,
            "sourceAndCodeUnchanged": True,
            "descriptorSnapshotsAreWriteTrace": False,
            "globalDeviceChanged": False,
            "baselineMode": "original-forward-without-outer-compile-wrapper",
            "productionBackendAdopted": False,
            "qualityApproved": False,
            "deliveryApproved": False,
        },
    )
    return result


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path)
    parser.add_argument("directory", type=Path)
    parser.add_argument("--start-sample", type=int, default=0)
    parser.add_argument("--samples", type=int, required=True)
    args = parser.parse_args(argv)
    try:
        print(
            run(args.source, args.directory, start_sample=args.start_sample, samples=args.samples)
        )
    except (
        ReplayError,
        OSError,
        ValueError,
        RuntimeError,
        wave.Error,
        subprocess.SubprocessError,
    ) as error:
        print(f"SEPARATION_FORWARD_PROBE_ERROR {error}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
