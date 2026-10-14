"""Bounded, fingerprint-bound local word recognition for any prepared recording.

Produces unreviewed evidence, not an edited delivery. Recognition clocks describe
continuous input PCM; project them through source frame clocks separately.
"""

import argparse
import json
import wave
from hashlib import file_digest
from pathlib import Path

import numpy as np
from numpy.typing import NDArray

from replay.asr_runtime import ASRDevice, WhisperRuntime, validate_cpu_paths
from replay.caption_io import mlx_groups
from replay.core import ReplayError

MAX_SAMPLES = 60 * 16000
# User-selected full large-v3 MLX model; never substitute turbo or Core ML assets.
DEFAULT_MODEL = Path("edits/models/mlx-whisper-large-v3/49e6aa286ad60c14352c404340ded53710378a11")
DEFAULT_WEIGHTS_SHA256 = "05ff791ce3630fae47e7c51004e9666204d786246ec07cac6110af768099b40d"
DEFAULT_CONFIG_SHA256 = "34982ce6ae286095000f82ae9583b3431639e8b092bf60c961f203745e6500e3"


def fingerprint(path: Path) -> str:
    if path.is_symlink() or not path.is_file():
        raise ReplayError("Expected a regular local input file")
    with path.open("rb") as stream:
        return file_digest(stream, "sha256").hexdigest()


def pcm_audio(path: Path) -> NDArray[np.float32]:
    """Read exact mono PCM16/16kHz samples without a decoder or timeline adjustment."""
    with wave.open(str(path), "rb") as stream:
        count = stream.getnframes()
        if (
            stream.getnchannels() != 1
            or stream.getsampwidth() != 2
            or stream.getframerate() != 16000
            or stream.getcomptype() != "NONE"
            or not 0 < count <= MAX_SAMPLES
        ):
            raise ReplayError("Expected at most 60 seconds of mono PCM16/16kHz WAV")
        payload = stream.readframes(count)
    if len(payload) != count * 2:
        raise ReplayError("Truncated recognition PCM")
    return np.frombuffer(payload, dtype="<i2").astype(np.float32) / np.float32(32768)


def transcribe_file(
    source: Path,
    *,
    model: Path,
    weights_sha256: str,
    config_sha256: str,
    output: Path,
    device: ASRDevice = ASRDevice.AUTO,
) -> Path:
    """Create a new evidence directory; never download a model or overwrite a run.

    Failed recognition/validation retains inputs and any raw prediction. Empty ASR
    is not evidence of silence. The caller supplies reviewed local model hashes.
    """
    runtime = WhisperRuntime(device)
    validate_cpu_paths(runtime.device, source, model, output)
    if not model.is_dir() or model.is_symlink():
        raise ReplayError("Expected an explicit local model directory")
    weights = [
        path for path in (model / "weights.npz", model / "weights.safetensors") if path.exists()
    ]
    if len(weights) != 1:
        raise ReplayError("Expected exactly one local model weights file")
    paths = (source, model / "config.json", weights[0])
    before = tuple(fingerprint(path) for path in paths)
    if before[1:] != (config_sha256, weights_sha256):
        raise ReplayError("Local recognition model fingerprint mismatch")
    samples = pcm_audio(source)
    output.mkdir(mode=0o700, parents=False)
    provenance = {
        "version": 1,
        "source": str(source.resolve()),
        "model": str(model.resolve()),
        "source_sha256": before[0],
        "config_sha256": config_sha256,
        "weights_sha256": weights_sha256,
        "sample_rate": 16000,
        "samples": len(samples),
        "clock": "continuous-input-pcm",
        "human_reviewed": False,
    }
    if device == ASRDevice.CPU:
        provenance["device"] = "cpu"
        provenance["dtype_contract"] = "provider-default-fp16-unchanged"
        provenance["runtime_sha256"] = fingerprint(Path(__file__).with_name("asr_runtime.py"))
    with (output / "inputs.json").open("x", encoding="utf-8") as stream:
        json.dump(provenance, stream, indent=2)
    result = runtime.predict(samples, model, evidence=output / "cpu-runtime")
    raw = output / "raw-transcript.json"
    with raw.open("x", encoding="utf-8") as stream:
        json.dump(result, stream, ensure_ascii=False, allow_nan=False)
    if tuple(fingerprint(path) for path in paths) != before:
        raise ReplayError("Recognition input changed during inference")
    groups = mlx_groups(result, duration=len(samples) / 16000)
    if not groups:
        raise ReplayError("Empty recognition does not prove absent speech")
    with (output / "validation.json").open("x", encoding="utf-8") as stream:
        json.dump(
            {
                "word_groups": len(groups),
                "unresolved_points": sum(group.start == group.end for group in groups),
                "human_reviewed": False,
                "source_clock_projected": False,
            },
            stream,
        )
    return raw


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument(
        "--model",
        type=Path,
        default=DEFAULT_MODEL,
        help="Local model directory (default: pinned mlx-community/whisper-large-v3-mlx)",
    )
    parser.add_argument("--weights-sha256", default=DEFAULT_WEIGHTS_SHA256)
    parser.add_argument("--config-sha256", default=DEFAULT_CONFIG_SHA256)
    parser.add_argument("--device", type=ASRDevice, choices=list(ASRDevice), default=ASRDevice.AUTO)
    args = parser.parse_args(argv)
    transcribe_file(
        args.source,
        model=args.model,
        weights_sha256=args.weights_sha256,
        config_sha256=args.config_sha256,
        output=args.output,
        device=args.device,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
