"""Bounded same-model CPU evidence on prepared PCM; never delivery approval.

Use a workflow timeout and external monitoring as well. Descriptor snapshots do
not trace writes or exclude transient files between observations.
"""

from __future__ import annotations

import argparse
import hashlib
import os
import resource
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

from replay.bgm_separation import (
    MODEL_CONFIG_SHA256,
    MODEL_REVISION,
    MODEL_SAMPLE_RATE,
    MODEL_SHA256,
    _import,
    _load_backend,
    _read_wav,
    _write_wav,
)
from replay.core import ReplayError
from replay.native_invocation import canonical, durable, read, require


def sha256(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def external_writable_files(snapshot: str, root: Path) -> list[str]:
    """Inspect lsof -Ffatn records, excluding pipes/devices, not regular caches."""
    outside: list[str] = []
    access = kind = ""
    for line in snapshot.splitlines():
        if line.startswith("f"):
            access = kind = ""
        elif line.startswith("a"):
            access = line[1:]
        elif line.startswith("t"):
            kind = line[1:]
        elif line.startswith("n") and kind == "REG" and access in ("w", "u"):
            path = Path(line[1:])
            if not path.is_absolute() or not path.resolve().is_relative_to(root):
                outside.append(line[1:])
    return outside


def audit_descriptors(directory: Path, phase: str) -> None:
    result = subprocess.run(
        ["/usr/sbin/lsof", "-a", "-p", str(os.getpid()), "-Ffatn"],
        capture_output=True,
        text=True,
        check=True,
        timeout=10,
    )
    outside = external_writable_files(result.stdout, Path.cwd().resolve())
    durable(
        directory / f"{phase}-descriptors.json",
        {"pid": os.getpid(), "snapshot": result.stdout, "externalWritableRegular": outside},
    )
    require(not outside, "Observed writable regular file outside checkout; no inference retry")


def probe(source: Path, directory: Path, *, samples: int, input_sha256: str) -> None:
    source, directory = canonical(str(source)), canonical(str(directory))
    require(directory.is_relative_to(Path.cwd() / "edits/work"), "Probe output must stay local")
    require(0 < samples <= 60 * MODEL_SAMPLE_RATE, "Probe sample budget is 60 seconds")
    require(sha256(source) == input_sha256, "Prepared input hash changed")
    audio, channels = _read_wav(source)
    require(channels == 2 and samples <= audio.shape[1], "Probe needs supported stereo samples")
    directory.mkdir(mode=0o700)
    started = time.monotonic()
    print(f"CPU_PROBE pid={os.getpid()} samples={samples}", flush=True)
    durable(
        directory / "input.json",
        {
            "source": str(source),
            "inputSHA256": input_sha256,
            "startSample": 0,
            "sampleCount": samples,
            "sourceSampleCount": int(audio.shape[1]),
            "modelRevision": MODEL_REVISION,
            "weightsSHA256": MODEL_SHA256,
            "configSHA256": MODEL_CONFIG_SHA256,
            "windowSamples": 8 * MODEL_SAMPLE_RATE,
            "hopSamples": 4 * MODEL_SAMPLE_RATE,
            "device": "cpu",
            "implementationSHA256": sha256(Path(__file__)),
            "backendSHA256": sha256(Path(__file__).with_name("bgm_separation.py")),
            "deliveryApproved": False,
        },
    )
    audit_descriptors(directory, "before-import")
    _import("mlx.core")
    audit_descriptors(directory, "after-import")
    backend = _load_backend("mlx-community/mel-roformer-mlx", "cpu")
    audit_descriptors(directory, "after-load")
    inference_started = time.monotonic()
    output = backend.extract_vocals(audio[:, :samples])
    inference_seconds = time.monotonic() - inference_started
    audit_descriptors(directory, "after-inference")
    require(output.shape == (2, samples) and bool(np.isfinite(output).all()), "Invalid CPU output")
    require(sha256(source) == input_sha256, "Prepared input changed during probe")
    destination = directory / "vocals.wav"
    _write_wav(destination, output)
    elapsed = time.monotonic() - started
    peak_rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    durable(
        directory / "result.json",
        {
            "elapsedSeconds": elapsed,
            "inferenceSeconds": inference_seconds,
            "audioSeconds": samples / MODEL_SAMPLE_RATE,
            "elapsedPerAudioSecond": elapsed / (samples / MODEL_SAMPLE_RATE),
            "inferencePerAudioSecond": inference_seconds / (samples / MODEL_SAMPLE_RATE),
            "kernelPeakRSSBytes": peak_rss if sys.platform == "darwin" else peak_rss * 1024,
            "outputSHA256": sha256(destination),
            "savedBytesBeforeReport": sum(path.stat().st_size for path in directory.iterdir()),
            "descriptorSnapshotsAreWriteTrace": False,
            "deliveryApproved": False,
        },
    )


def verify_probe(source: Path, directory: Path, *, samples: int, input_sha256: str) -> None:
    source, directory = canonical(str(source)), canonical(str(directory))
    inputs, result = read(directory / "input.json"), read(directory / "result.json")
    require(
        inputs.get("inputSHA256") == input_sha256 == sha256(source)
        and inputs.get("source") == str(source)
        and inputs.get("sampleCount") == samples
        and inputs.get("device") == "cpu"
        and inputs.get("modelRevision") == MODEL_REVISION
        and inputs.get("weightsSHA256") == MODEL_SHA256
        and inputs.get("configSHA256") == MODEL_CONFIG_SHA256
        and result.get("deliveryApproved") is False,
        "CPU probe provenance changed",
    )
    output = directory / "vocals.wav"
    require(result.get("outputSHA256") == sha256(output), "CPU probe output changed")
    audio, channels = _read_wav(output)
    require(channels == 2 and audio.shape == (2, samples), "CPU probe output clock changed")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path)
    parser.add_argument("directory", type=Path)
    parser.add_argument("--samples", required=True, type=int)
    parser.add_argument("--input-sha256", required=True)
    parser.add_argument("--verify", action="store_true")
    args = parser.parse_args(argv)
    try:
        action = verify_probe if args.verify else probe
        action(args.source, args.directory, samples=args.samples, input_sha256=args.input_sha256)
    except (ReplayError, OSError, ValueError, RuntimeError, subprocess.SubprocessError) as error:
        print(f"CPU_PROBE_ERROR {error}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
