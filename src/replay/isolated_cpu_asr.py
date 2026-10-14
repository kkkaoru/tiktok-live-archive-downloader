"""Fresh CPU process per missing ASR window, preserving the batch clock contract."""

import argparse
import subprocess
import sys
import wave
from collections.abc import Callable
from pathlib import Path

import numpy as np
from numpy.typing import NDArray

from replay.asr_runtime import ASRDevice, WhisperRuntime, validate_cpu_paths
from replay.audio_windows import plan_audio_windows
from replay.batch_asr import MAX_SECONDS, RATE, _read, _window, _write
from replay.core import ReplayError
from replay.local_asr import (
    DEFAULT_CONFIG_SHA256,
    DEFAULT_MODEL,
    DEFAULT_WEIGHTS_SHA256,
    fingerprint,
)
from replay.native_invocation import canonical, require
from replay.stages import execution_lock

Executor = Callable[[tuple[str, ...], Path, int], None]


def execute_child(command: tuple[str, ...], log: Path, timeout: int) -> None:
    """Keep descendants in the controller's workflow-owned process group."""
    with (
        log.open("xb") as stream,
        subprocess.Popen(
            command, stdin=subprocess.DEVNULL, stdout=stream, stderr=subprocess.STDOUT
        ) as child,
    ):
        try:
            code = child.wait(timeout=timeout)
        finally:
            if child.poll() is None:
                child.kill()
                child.wait()
        require(code == 0, f"Isolated ASR child failed with return code {code}")


class IsolatedRuntime(WhisperRuntime):
    """The parent never imports MLX; each child uses the existing CPU runtime."""

    def __init__(self, execute: Executor, timeout: int) -> None:
        super().__init__(ASRDevice.CPU)
        self.execute = execute
        self.timeout = timeout

    def predict(self, samples: NDArray[np.float32], model: Path, *, evidence: Path) -> object:
        validate_cpu_paths(ASRDevice.CPU, evidence, model)
        evidence.mkdir(mode=0o700)
        payload = evidence / "samples.npy"
        with payload.open("xb") as stream:
            np.save(stream, samples, allow_pickle=False)
        request = {
            "samplesSHA256": fingerprint(payload),
            "model": str(model.resolve()),
            "producerSHA256": fingerprint(Path(__file__)),
        }
        _write(evidence / "request.json", request)
        self.execute(
            (
                sys.executable,
                "-B",
                "-m",
                "replay.isolated_cpu_asr",
                "--worker",
                str(evidence.resolve()),
            ),
            evidence / "worker.log",
            self.timeout,
        )
        require(
            _read(evidence / "request.json") == request
            and fingerprint(payload) == request["samplesSHA256"],
            "Isolated sample payload changed",
        )
        result = _read(evidence / "result.json")
        require(
            result
            == {
                "rawSHA256": fingerprint(evidence / "raw.json"),
                "samplesSHA256": request["samplesSHA256"],
                "producerSHA256": request["producerSHA256"],
                "device": "cpu",
                "deliveryComplete": False,
            },
            "Isolated raw output changed",
        )
        return _read(evidence / "raw.json")


def worker(evidence: Path) -> None:
    """Validate the IPC payload; model arguments and dtype remain unchanged."""
    evidence = canonical(str(evidence.absolute()))
    require(evidence.is_relative_to(Path.cwd() / "edits/work"), "Worker evidence must stay local")
    request = _read(evidence / "request.json")
    payload = evidence / "samples.npy"
    require(payload.stat().st_size <= RATE * 60 * 4 + 4096, "Oversized isolated PCM")
    require(
        fingerprint(payload) == request.get("samplesSHA256")
        and fingerprint(Path(__file__)) == request.get("producerSHA256"),
        "Isolated worker inputs changed",
    )
    model = canonical(request.get("model"))
    validate_cpu_paths(ASRDevice.CPU, model)
    with payload.open("rb") as stream:
        loaded: object = np.load(stream, allow_pickle=False)
    if not isinstance(loaded, np.ndarray):
        raise ReplayError("Expected numeric PCM array")
    require(
        loaded.dtype == np.dtype(np.float32) and loaded.ndim == 1 and 0 < loaded.size <= RATE * 60,
        "Expected bounded float32 PCM",
    )
    samples: NDArray[np.float32] = loaded
    require(
        bool(np.isfinite(samples).all()) and bool((np.abs(samples) <= 1).all()),
        "Invalid normalized PCM",
    )
    result = WhisperRuntime(ASRDevice.CPU).predict(samples, model, evidence=evidence / "runtime")
    _write(evidence / "raw.json", result)
    require(
        fingerprint(payload) == request["samplesSHA256"],
        "Isolated payload changed during inference",
    )
    _write(
        evidence / "result.json",
        {
            "rawSHA256": fingerprint(evidence / "raw.json"),
            "samplesSHA256": request["samplesSHA256"],
            "producerSHA256": request["producerSHA256"],
            "device": "cpu",
            "deliveryComplete": False,
        },
    )


def run(
    source: Path,
    output: Path,
    policy: Path,
    *,
    limit: int | None = None,
    timeout: int = 300,
    model: Path = DEFAULT_MODEL,
    weights_sha256: str = DEFAULT_WEIGHTS_SHA256,
    config_sha256: str = DEFAULT_CONFIG_SHA256,
    execute: Executor = execute_child,
) -> Path:
    """Reuse immutable batch results; stop bounded diagnostics without completion."""
    require(limit is None or type(limit) is int and 1 <= limit <= 256, "Invalid new-window limit")
    require(type(timeout) is int and 1 <= timeout <= 3600, "Invalid child deadline")
    validate_cpu_paths(ASRDevice.CPU, source, output, policy, model)
    root = Path.cwd().resolve()
    policy = canonical(str(policy.absolute()))
    require(
        policy.is_relative_to(root / "edits/work")
        and not policy.is_relative_to(output.resolve())
        and not output.resolve().is_relative_to(policy),
        "Independent local policy evidence required",
    )
    weights = [p for p in (model / "weights.npz", model / "weights.safetensors") if p.exists()]
    require(len(weights) == 1, "Expected one pinned model weight file")
    paths = (
        source,
        model / "config.json",
        weights[0],
        Path(__file__).with_name("batch_asr.py"),
        Path(__file__).with_name("asr_runtime.py"),
    )
    before = {str(p.resolve()): fingerprint(p) for p in paths}
    require(
        before[str(paths[1].resolve())] == config_sha256
        and before[str(paths[2].resolve())] == weights_sha256,
        "Recognition model fingerprint mismatch",
    )
    own_pin = fingerprint(Path(__file__))
    with wave.open(str(source), "rb") as audio:
        require(
            (audio.getnchannels(), audio.getsampwidth(), audio.getframerate(), audio.getcomptype())
            == (1, 2, RATE, "NONE")
            and 0 < audio.getnframes() <= RATE * MAX_SECONDS,
            "Expected bounded mono PCM16/16kHz",
        )
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
            "device": "cpu",
            "dtypeContract": "provider-default-fp16-unchanged",
        }
        existed = output.exists()
        output.mkdir(mode=0o700, exist_ok=True)
        with execution_lock(output, ".run") as acquired:
            require(acquired, "ASR output already has an owner")
            manifest = output / "inputs.json"
            if manifest.exists():
                require(_read(manifest) == definition, "ASR run inputs changed")
            else:
                require(not existed, "Existing ASR output lacks input receipt")
                _write(manifest, definition)
            policy.mkdir(mode=0o700)
            _write(
                policy / "started.json",
                {
                    "version": 1,
                    "producerSHA256": own_pin,
                    "inputs": before,
                    "manifestSHA256": fingerprint(manifest),
                    "processPolicy": "fresh-cpu-child-per-missing-window",
                    "cachePolicy": "unchanged-provider-default",
                    "limitNewWindows": limit,
                    "childTimeoutSeconds": timeout,
                },
            )
            hashes: list[str] = []
            created = 0
            complete = output / "complete.json"
            for index, window in enumerate(windows):
                present = all(
                    (output / f"{window.start_sample:012d}.{suffix}.json").is_file()
                    for suffix in ("raw", "clock")
                )
                require(not complete.exists() or present, "Completed ASR window is missing")
                if not present and limit is not None and created >= limit:
                    break
                hashes.append(
                    _window(audio, window, output, model, IsolatedRuntime(execute, timeout))
                )
                created += not present
                print(f"ASR_WINDOW {index + 1}/{len(windows)}", flush=True)
            require(
                {str(p.resolve()): fingerprint(p) for p in paths} == before
                and fingerprint(Path(__file__)) == own_pin,
                "Isolated ASR inputs changed",
            )
            if len(hashes) == len(windows):
                receipt = {"rawSHA256": hashes, "windows": len(windows), "deliveryComplete": False}
                if complete.exists():
                    require(_read(complete) == receipt, "ASR completion changed")
                else:
                    _write(complete, receipt)
            result_path = policy / "result.json"
            _write(
                result_path,
                {
                    "version": 1,
                    "producerSHA256": own_pin,
                    "newWindows": created,
                    "verifiedWindows": len(hashes),
                    "totalWindows": len(windows),
                    "recognitionComplete": len(hashes) == len(windows),
                    "deliveryComplete": False,
                },
            )
            return result_path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path, nargs="?")
    parser.add_argument("output", type=Path, nargs="?")
    parser.add_argument("--policy", type=Path)
    parser.add_argument("--worker", type=Path)
    parser.add_argument("--limit-new-windows", type=int)
    parser.add_argument("--child-timeout", type=int, default=300)
    args = parser.parse_args()
    if args.worker is not None:
        worker(args.worker)
    else:
        require(
            args.source is not None and args.output is not None and args.policy is not None,
            "Controller requires source, output and policy",
        )
        print(
            run(
                args.source,
                args.output,
                args.policy,
                limit=args.limit_new_windows,
                timeout=args.child_timeout,
            )
        )


if __name__ == "__main__":
    main()
