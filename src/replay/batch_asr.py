"""Resumable raw ASR windows, not speech approval or an assembled transcript.

One process reuses MLX Whisper's model cache. Hash large model files at the run
boundaries, not for every window. Overlap clocks remain explicit for later assembly.
"""

import argparse
import fcntl
import json
import wave
from pathlib import Path

import numpy as np
from numpy.typing import NDArray

from replay.asr_runtime import ASRDevice, WhisperRuntime, validate_cpu_paths
from replay.audio_windows import AudioWindow, plan_audio_windows
from replay.core import ReplayError
from replay.local_asr import (
    DEFAULT_CONFIG_SHA256,
    DEFAULT_MODEL,
    DEFAULT_WEIGHTS_SHA256,
    fingerprint,
)
from replay.recordings import record

RATE = 16000
MAX_SECONDS = 21600
MAX_JSON_BYTES = 8 * 1024 * 1024


def _read(path: Path) -> dict[str, object]:
    if path.is_symlink() or not path.is_file() or path.stat().st_size > MAX_JSON_BYTES:
        raise ReplayError("Expected bounded regular ASR evidence")
    return record(json.loads(path.read_text(encoding="utf-8")))


def _write(path: Path, value: object) -> None:
    with path.open("x", encoding="utf-8") as stream:
        json.dump(value, stream, ensure_ascii=False, allow_nan=False, indent=2)


def _predict(samples: NDArray[np.float32], model: Path) -> object:
    return WhisperRuntime().predict(samples, model, evidence=Path("."))


def _window(
    audio: wave.Wave_read,
    window: AudioWindow,
    output: Path,
    model: Path,
    runtime: WhisperRuntime | None = None,
) -> str:
    raw = output / f"{window.start_sample:012d}.raw.json"
    receipt = output / f"{window.start_sample:012d}.clock.json"
    clock = {
        "startSample": window.start_sample,
        "endSample": window.end_sample,
        "sampleRate": RATE,
        "humanReviewed": False,
        "emptyRecognitionIsNotSilence": True,
    }
    if raw.exists() or receipt.exists() or raw.is_symlink() or receipt.is_symlink():
        saved = _read(receipt)
        if saved != clock | {"rawSHA256": fingerprint(raw)}:
            raise ReplayError("Completed ASR window changed")
        return fingerprint(raw)
    audio.setpos(window.start_sample)
    payload = audio.readframes(window.sample_count)
    if len(payload) != window.sample_count * 2:
        raise ReplayError("Truncated recognition PCM")
    samples = np.frombuffer(payload, dtype="<i2").astype(np.float32) / np.float32(32768)
    result = (
        _predict(samples, model)
        if runtime is None
        else runtime.predict(
            samples, model, evidence=output / f"{window.start_sample:012d}.cpu-runtime"
        )
    )
    _write(raw, result)
    # Preserve raw provider clocks, including uncertain words and empty windows.
    # This is a raw production receipt, never caption/timing/silence approval.
    if not isinstance(record(result).get("segments"), list):
        raise ReplayError("Recognizer did not return a segment array")
    digest = fingerprint(raw)
    _write(receipt, clock | {"rawSHA256": digest})
    return digest


def transcribe_windows(
    source: Path,
    output: Path,
    *,
    model: Path = DEFAULT_MODEL,
    weights_sha256: str = DEFAULT_WEIGHTS_SHA256,
    config_sha256: str = DEFAULT_CONFIG_SHA256,
    device: ASRDevice = ASRDevice.AUTO,
) -> Path:
    """Capture every 60s/30s-stride window, reusing only unchanged raw receipts.

    Input is already clocked mono PCM16/16kHz, at most six hours. Neither this
    function nor its completion receipt asserts word accuracy, merges overlaps,
    projects original media clocks, or interprets an empty window as silence.
    Partial writes fail closed and remain available for explicit inspection.
    """
    device = ASRDevice(device)
    validate_cpu_paths(device, source, model, output)
    runtime = WhisperRuntime(device) if device == ASRDevice.CPU else None
    if model.is_symlink() or not model.is_dir() or output.is_symlink():
        raise ReplayError("Expected local model and regular output storage")
    weights = [p for p in (model / "weights.npz", model / "weights.safetensors") if p.exists()]
    if len(weights) != 1:
        raise ReplayError("Expected exactly one local weights file")
    paths = (
        source,
        model / "config.json",
        weights[0],
        Path(__file__),
        Path(__file__).with_name("asr_runtime.py"),
    )
    before = {str(p.resolve()): fingerprint(p) for p in paths}
    if (
        before[str(paths[1].resolve())] != config_sha256
        or before[str(paths[2].resolve())] != weights_sha256
    ):
        raise ReplayError("Local recognition model fingerprint mismatch")
    with wave.open(str(source), "rb") as audio:
        if (
            audio.getnchannels(),
            audio.getsampwidth(),
            audio.getframerate(),
            audio.getcomptype(),
        ) != (1, 2, RATE, "NONE") or not 0 < audio.getnframes() <= RATE * MAX_SECONDS:
            raise ReplayError("Expected bounded mono PCM16/16kHz WAV")
        windows = plan_audio_windows(
            audio.getnframes(), window_samples=60 * RATE, stride_samples=30 * RATE
        )
        definition = {
            "version": 1,
            "inputs": before,
            "samples": audio.getnframes(),
            "sampleRate": RATE,
            "windowSamples": 60 * RATE,
            "strideSamples": 30 * RATE,
            "clock": "input-pcm-not-projected",
        }
        if device == ASRDevice.CPU:
            definition["device"] = "cpu"
            definition["dtypeContract"] = "provider-default-fp16-unchanged"
        existed = output.exists()
        output.mkdir(mode=0o700, exist_ok=True)
        lock_path = output / ".run.lock"
        if lock_path.is_symlink():
            raise ReplayError("Linked ASR lock is not allowed")
        with lock_path.open("a", encoding="utf-8") as lock:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            manifest = output / "inputs.json"
            if manifest.exists() or manifest.is_symlink():
                if _read(manifest) != definition:
                    raise ReplayError("ASR run inputs changed")
            else:
                if existed:
                    raise ReplayError("Existing ASR output has no input receipt")
                _write(manifest, definition)
            complete = output / "complete.json"
            completed = complete.exists() or complete.is_symlink()
            hashes = []
            for index, window in enumerate(windows):
                if completed and not all(
                    (output / f"{window.start_sample:012d}.{suffix}.json").is_file()
                    for suffix in ("raw", "clock")
                ):
                    raise ReplayError("Completed ASR window is missing")
                hashes.append(_window(audio, window, output, model, runtime))
                print(f"ASR_WINDOW {index + 1}/{len(windows)}", flush=True)
            if {str(p.resolve()): fingerprint(p) for p in paths} != before:
                raise ReplayError("ASR inputs changed during inference")
            receipt = {"rawSHA256": hashes, "windows": len(windows), "deliveryComplete": False}
            if complete.exists() or complete.is_symlink():
                if _read(complete) != receipt:
                    raise ReplayError("ASR completion changed")
            else:
                _write(complete, receipt)
            return complete


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--device", type=ASRDevice, choices=list(ASRDevice), default=ASRDevice.AUTO)
    args = parser.parse_args(argv)
    transcribe_windows(args.source, args.output, device=args.device)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
