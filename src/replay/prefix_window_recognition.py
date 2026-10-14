"""Recognize one verified prefix window without fabricating a whole ASR collection.

The original separation environment verifies the prefix in a separate process.
The current CPU interpreter then uses the existing isolated Whisper worker. A
completed candidate is resumable, but final PCM comparison and genuine full-run
adoption remain separate operations. RAM is a reservation, not an MLX RSS limit.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import subprocess
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from replay.code_dependency_inventory import capture
from replay.core import ReplayError
from replay.disk_budget import require_headroom
from replay.isolated_cpu_asr import Executor, IsolatedRuntime, execute_child
from replay.local_asr import DEFAULT_CONFIG_SHA256, DEFAULT_WEIGHTS_SHA256
from replay.native_invocation import canonical, durable, identity, pin, require
from replay.recordings import record
from replay.stages import execution_lock

ROOT = Path(__file__).resolve().parents[2]
RATE = 16000
WINDOW = RATE * 60
STRIDE = RATE * 30
MAX_JSON = 8 * 1024**2
MAX_SAVED = 128 * 1024**2
BUFFER_BYTES = WINDOW * 6 + MAX_JSON * 3
MODEL = "mlx-community/whisper-large-v3-mlx"
SCRATCH_NAMES = ("TMPDIR", "TMP", "TEMP", "XDG_CACHE_HOME", "NUMBA_CACHE_DIR")


@dataclass(frozen=True)
class Ref:
    path: Path
    sha256: str

    def data(self) -> dict[str, str]:
        return {"path": str(self.path), "sha256": self.sha256}


@dataclass(frozen=True)
class Request:
    path: Path
    definition: dict[str, object]
    prefix_request: Ref
    prefix_receipt: Ref
    window: Ref
    pcm: Ref
    start: int
    model: Path
    config: Ref
    weights: Ref
    verifier_python: Ref
    verifier_invocation: Path
    output: Path
    scratch: Path
    timeout: int
    deadline: int
    ram: int
    saved: int


def _pairs(items: list[tuple[str, object]]) -> dict[str, object]:
    result: dict[str, object] = {}
    for key, value in items:
        require(key not in result, "Duplicate metadata key")
        result[key] = value
    return result


def _read(path: Path) -> dict[str, object]:
    before = identity(path)
    require(0 < before["bytes"] <= MAX_JSON, "Metadata byte budget exceeded")
    data: object = json.loads(path.read_bytes(), object_pairs_hook=_pairs)
    require(identity(path) == before, "Metadata changed during reading")
    return record(data)


def _local(value: object) -> Path:
    path = canonical(value)
    require(path.is_relative_to(ROOT / "edits/work"), "Expected checkout work path")
    return path


def _ref(value: object, *, tool: bool = False) -> Ref:
    raw = record(value)
    sha = raw.get("sha256")
    require(set(raw) == {"path", "sha256"}, "Exact reference fields required")
    require(isinstance(sha, str) and re.fullmatch("[0-9a-f]{64}", sha) is not None, "Invalid SHA")
    path = canonical(raw["path"])
    require(tool or path.is_relative_to(ROOT), "Input escapes checkout")
    return Ref(path, str(sha))


def _check(ref: Ref) -> dict[str, object]:
    value = pin(ref.path)
    require(value["sha256"] == ref.sha256, "Pinned input changed")
    return value


def _integer(value: object, *, minimum: int, maximum: int) -> int:
    if type(value) is not int or not minimum <= value <= maximum:
        raise ReplayError("Invalid integer resource budget")
    return value


def runtime_inputs() -> dict[str, str]:
    """Capture static Python dependencies without importing MLX or loading a model."""
    return {
        str(item.path): item.sha256
        for item in capture(("prefix_window_recognition", "separation_prefix_asr"))
    }


def load_request(path: Path) -> Request:
    """Validate metadata and bounds; do not read PCM or model weights."""
    path = _local(str(path))
    raw = _read(path)
    require(
        set(raw)
        == {
            "version",
            "prefixRequest",
            "prefixReceipt",
            "window",
            "model",
            "verifierPython",
            "verifierInvocation",
            "outputDirectory",
            "scratchDirectory",
            "runtime",
            "timeoutSeconds",
            "deadlineSeconds",
            "ramReservationBytes",
            "maxSavedBytes",
        }
        and type(raw["version"]) is int
        and raw["version"] == 1,
        "Unsupported prefix recognition request",
    )
    prefix_request, prefix_receipt, window = (
        _ref(raw[key]) for key in ("prefixRequest", "prefixReceipt", "window")
    )
    _check(prefix_request)
    _check(prefix_receipt)
    _check(window)
    prefix = _read(prefix_receipt.path)
    item = _read(window.path)
    windows = prefix.get("windows")
    if not isinstance(windows, list):
        raise ReplayError("Missing verified prefix windows")
    require(
        prefix.get("kind") == "separation-prefix-asr-windows"
        and prefix.get("chunkDomainVerified") is True
        and record(prefix.get("bindings")).get("request") == pin(prefix_request.path)
        and any(record(row).get("receipt") == pin(window.path) for row in windows),
        "Window is not bound to the original verified prefix",
    )
    start = _integer(item.get("startSample"), minimum=0, maximum=RATE * 21600)
    require(
        item.get("kind") == "verified-separation-prefix-window"
        and item.get("request") == pin(prefix_request.path)
        and start % STRIDE == 0
        and item.get("endSample") == start + WINDOW
        and item.get("sampleRate") == RATE
        and item.get("channels") == 1
        and item.get("sampleWidth") == 2
        and item.get("clock") == "input-pcm-not-projected",
        "Expected one immutable full prefix window",
    )
    pcm = _ref(item.get("pcm"))
    require(identity(pcm.path)["bytes"] == WINDOW * 2, "Incorrect full-window PCM byte count")
    model_row = record(raw["model"])
    require(
        set(model_row) == {"id", "directory", "config", "weights"}, "Exact model fields required"
    )
    model = canonical(model_row["directory"])
    config, weights = _ref(model_row["config"]), _ref(model_row["weights"])
    require(
        model.is_relative_to(ROOT / "edits/models")
        and model.is_dir()
        and model_row["id"] == MODEL
        and config.path == model / "config.json"
        and weights.path in (model / "weights.npz", model / "weights.safetensors")
        and config.sha256 == DEFAULT_CONFIG_SHA256
        and weights.sha256 == DEFAULT_WEIGHTS_SHA256
        and item.get("model")
        == {"id": MODEL, "configSHA256": config.sha256, "weightsSHA256": weights.sha256},
        "Expected the pinned offline large-v3 model",
    )
    identity(config.path)
    weight_size = identity(weights.path)["bytes"]
    verifier_python = _ref(raw["verifierPython"], tool=True)
    invocation_value = raw["verifierInvocation"]
    require(isinstance(invocation_value, str), "Missing verifier Python invocation")
    invocation = Path(str(invocation_value))
    require(
        invocation.is_absolute()
        and invocation.is_relative_to(ROOT)
        and ".." not in invocation.parts
        and invocation.parent.resolve(strict=True) == invocation.parent
        and invocation.parent.name == "bin"
        and invocation.name in ("python", "python3", "python3.12")
        and invocation.resolve(strict=True) == verifier_python.path
        and os.access(invocation, os.X_OK),
        "Verifier must use its exact configured Python alias",
    )
    _check(verifier_python)
    output, scratch = (_local(raw[key]) for key in ("outputDirectory", "scratchDirectory"))
    protected = (
        path,
        prefix_request.path,
        prefix_receipt.path,
        window.path,
        pcm.path,
        model,
        scratch,
    )
    require(
        output.parent.is_dir()
        and scratch.is_dir()
        and not any(
            p == output or p.is_relative_to(output) or output.is_relative_to(p) for p in protected
        ),
        "Output overlaps protected inputs or scratch",
    )
    require(raw["runtime"] == runtime_inputs(), "Recognition runtime changed")
    timeout = _integer(raw["timeoutSeconds"], minimum=1, maximum=3600)
    deadline = _integer(raw["deadlineSeconds"], minimum=timeout * 2, maximum=7200)
    saved = _integer(raw["maxSavedBytes"], minimum=BUFFER_BYTES, maximum=MAX_SAVED)
    ram = _integer(
        raw["ramReservationBytes"], minimum=weight_size + BUFFER_BYTES, maximum=128 * 1024**3
    )
    return Request(
        path,
        raw,
        prefix_request,
        prefix_receipt,
        window,
        pcm,
        start,
        model,
        config,
        weights,
        verifier_python,
        invocation,
        output,
        scratch,
        timeout,
        deadline,
        ram,
        saved,
    )


def preflight(path: Path) -> dict[str, object]:
    """Return explicit reservations; this does not verify PCM or execute inference."""
    request = load_request(path)
    return {
        "version": 1,
        "fullWindows": 1,
        "ramReservationBytes": request.ram,
        "estimatedAdditionalSavedBytes": BUFFER_BYTES,
        "residentMemoryHardLimited": False,
        "modelBytesVerified": False,
        "recognitionExecuted": False,
        "adopted": False,
    }


def _bindings(request: Request) -> dict[str, object]:
    require(
        _read(request.path) == request.definition
        and runtime_inputs() == request.definition["runtime"],
        "Request/runtime changed at boundary",
    )
    return {
        "request": pin(request.path),
        "prefixRequest": _check(request.prefix_request),
        "prefixReceipt": _check(request.prefix_receipt),
        "window": _check(request.window),
        "pcm": _check(request.pcm),
        "config": _check(request.config),
        "weights": _check(request.weights),
        "verifierPython": _check(request.verifier_python),
        "runtime": request.definition["runtime"],
    }


def _environment(request: Request) -> None:
    for name in SCRATCH_NAMES:
        path = canonical(os.environ.get(name))
        require(
            path.is_relative_to(request.scratch) and path.is_dir(),
            "Child scratch escapes reservation",
        )
    require(
        os.environ.get("HF_HUB_OFFLINE") == "1" and os.environ.get("HF_DATASETS_OFFLINE") == "1",
        "Offline inference environment required",
    )


def _clock(request: Request, raw: Path) -> dict[str, object]:
    return {
        "startSample": request.start,
        "endSample": request.start + WINDOW,
        "sampleRate": RATE,
        "humanReviewed": False,
        "emptyRecognitionIsNotSilence": True,
        "rawSHA256": pin(raw)["sha256"],
    }


def _completed(request: Request, bindings: dict[str, object]) -> dict[str, object]:
    raw, clock = request.output / "raw.json", request.output / "clock.json"
    prediction = _read(raw)
    require(isinstance(prediction.get("segments"), list), "Missing raw recognition segments")
    require(_read(clock) == _clock(request, raw), "Candidate clock/raw changed")
    runtime = request.output / "cpu-runtime"
    result = _read(runtime / "result.json")
    require(
        result.get("rawSHA256") == pin(runtime / "raw.json")["sha256"]
        and _read(runtime / "raw.json") == prediction
        and result.get("device") == "cpu"
        and result.get("deliveryComplete") is False,
        "Isolated inference evidence changed",
    )
    return {
        "version": 1,
        "kind": "recognized-separation-prefix-window",
        "bindings": bindings,
        "raw": pin(raw),
        "clock": pin(clock),
        "runtimeRequest": pin(runtime / "request.json"),
        "runtimeResult": pin(runtime / "result.json"),
        "runtimeSamples": pin(runtime / "samples.npy"),
        "prefixVerifierLog": pin(request.output / "prefix-verifier.log"),
        "modelBytesVerified": True,
        "recognitionExecuted": True,
        "device": "cpu",
        "dtypeContract": "provider-default-fp16-unchanged",
        "residentMemoryHardLimited": False,
        "finalMonoWindowEqual": False,
        "adopted": False,
        "consumerConnected": False,
        "qualityApproved": False,
        "deliveryComplete": False,
    }


def _saved_bytes(output: Path) -> int:
    total, count = 0, 0
    for parent, directories, files in output.walk():
        require(
            not any((parent / name).is_symlink() for name in directories),
            "Linked runtime directory",
        )
        for name in files:
            path = parent / name
            total += identity(path)["bytes"]
            count += 1
            require(count <= 512, "Runtime file count budget exceeded")
    return total


def run(path: Path, *, execute: Executor = execute_child) -> Path:
    """Run one finite candidate; never retry an incomplete owned attempt automatically."""
    request = load_request(path)
    _environment(request)
    require_headroom((request.output.parent,), need=request.saved)
    request.output.mkdir(mode=0o700, exist_ok=True)
    with execution_lock(request.output, ".recognition") as acquired:
        require(acquired, "Prefix recognition already owned")
        before = _bindings(request)
        complete = request.output / "complete.json"
        if complete.exists():
            require(_read(complete) == _completed(request, before), "Completed candidate changed")
            return complete
        require(
            not (request.output / "attempt.json").exists(),
            "Incomplete attempt: reconcile before retry",
        )
        started = time.monotonic()
        durable(
            request.output / "attempt.json",
            {"version": 1, "bindings": before, "deadlineSeconds": request.deadline},
        )
        command = (
            str(request.verifier_invocation),
            "-B",
            "-m",
            "replay.separation_prefix_asr",
            "verify",
            str(request.prefix_request.path),
        )
        execute(command, request.output / "prefix-verifier.log", request.timeout)
        require(_bindings(request) == before, "Inputs changed during prefix verification")
        remaining = int(request.deadline - (time.monotonic() - started))
        require(remaining > 0, "Recognition deadline exhausted")
        payload = request.pcm.path.read_bytes()
        require(
            len(payload) == WINDOW * 2
            and hashlib.sha256(payload).hexdigest() == request.pcm.sha256,
            "Stable prefix PCM changed",
        )
        samples = np.frombuffer(payload, dtype="<i2").astype(np.float32) / np.float32(32768)
        prediction = IsolatedRuntime(execute, min(request.timeout, remaining)).predict(
            samples, request.model, evidence=request.output / "cpu-runtime"
        )
        encoded = json.dumps(prediction, ensure_ascii=False, allow_nan=False).encode()
        require(
            len(encoded) <= MAX_JSON and isinstance(record(prediction).get("segments"), list),
            "Invalid bounded recognition result",
        )
        durable(request.output / "raw.json", prediction)
        durable(request.output / "clock.json", _clock(request, request.output / "raw.json"))
        require(_bindings(request) == before, "Inputs changed during recognition")
        require(
            time.monotonic() - started <= request.deadline
            and _saved_bytes(request.output) <= request.saved,
            "Recognition resource budget exceeded",
        )
        durable(complete, _completed(request, before))
        return complete


def verify(path: Path) -> Path:
    """Freshly verify a candidate, without inference or invented full-run receipts."""
    request = load_request(path)
    with execution_lock(request.output, ".recognition") as acquired:
        require(acquired, "Prefix recognition already owned")
        before = _bindings(request)
        result = request.output / "complete.json"
        require(_read(result) == _completed(request, before), "Completed candidate changed")
        require(_bindings(request) == before, "Verification inputs changed")
        return result


def adoption_plan(path: Path, output: Path) -> Path:
    """Publish an unconnected protocol; actual final PCM comparison is still required."""
    candidate = verify(path)
    request = load_request(path)
    output = _local(str(output))
    require(not output.is_relative_to(request.output), "Adoption plan must be independent")
    durable(
        output,
        {
            "version": 1,
            "kind": "prefix-window-adoption-protocol",
            "candidate": pin(candidate),
            "window": request.window.data(),
            "prefixRequest": request.prefix_request.data(),
            "requiredComparison": "replay.separation_prefix_asr.compare_final",
            "comparisonPythonInvocation": str(request.verifier_invocation),
            "requiredFinalExecutor": "replay.isolated_cpu_asr.run",
            "normalFactorySeedImportImplemented": False,
            "finalMonoWindowEqual": False,
            "fullMonoProducerVerified": False,
            "adopted": False,
            "consumerConnected": False,
            "qualityApproved": False,
            "deliveryComplete": False,
        },
    )
    return output


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("phase", choices=("check", "run", "verify", "adoption-plan"))
    parser.add_argument("request", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args(argv)
    try:
        if args.phase == "check":
            require(args.output is None, "Unexpected output argument")
            result: object = preflight(args.request)
        elif args.phase == "adoption-plan":
            require(args.output is not None, "Independent adoption output required")
            result = {"evidence": str(adoption_plan(args.request, args.output))}
        else:
            require(args.output is None, "Unexpected output argument")
            result = {"evidence": str({"run": run, "verify": verify}[args.phase](args.request))}
    except (ReplayError, OSError, ValueError, subprocess.SubprocessError) as error:
        parser.exit(1, f"Prefix recognition failed: {error}\n")
    print(json.dumps(result))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
