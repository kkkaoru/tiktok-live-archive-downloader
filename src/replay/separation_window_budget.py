"""Independent CPU windows with ordered overlap-add; an unadopted candidate.

Children initialize the unchanged pinned backend independently, without fork,
shared MLX objects, compilation, or thread environment changes. A complete padded
8-second window invokes the original backend exactly once. Only the parent blends
results, in the original ascending order. RSS/disk guards are sampled, not quotas.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import math
import os
import resource
import subprocess
import sys
import time
from collections.abc import Callable
from contextlib import ExitStack
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Protocol

import numpy as np
from numpy.typing import NDArray

from replay.audio_cpu_probe import audit_descriptors, external_writable_files, sha256
from replay.bgm_separation import (
    MODEL_CONFIG_SHA256,
    MODEL_NAME,
    MODEL_REVISION,
    MODEL_SAMPLE_RATE,
    MODEL_SHA256,
    VOCALS_SOURCE,
    SeparationBackend,
    _MlxBackend,
    _write_wav,
)
from replay.code_dependency_inventory import capture
from replay.core import ReplayError
from replay.native_invocation import canonical, durable, identity, pin, read, require, rows
from replay.process_family_observation import sandbox_profile
from replay.recordings import record
from replay.separation_forward_probe import _local_environment, compare_audio, read_interval
from replay.stages import execution_lock
from replay.stateful_guard import saved_bytes

WINDOW = 8 * MODEL_SAMPLE_RATE
HOP = WINDOW // 2
MAX_CHUNK_SAMPLES = 3600 * MODEL_SAMPLE_RATE
MAX_WORKERS = 2
DEVICES = ("cpu", "mps")
SANDBOX_EXEC = "/usr/bin/sandbox-exec"
POLL_SECONDS = 0.2
METADATA_ALLOWANCE_BYTES = 16 * 1024 * 1024
ARRAY_HEADER_ALLOWANCE_BYTES = 4096
BackendFactory = Callable[[], SeparationBackend]
Auditor = Callable[[Path, str], None]


@dataclass(frozen=True)
class Window:
    index: int
    start: int
    count: int


@dataclass(frozen=True, kw_only=True)
class WindowBudget:
    workers: int
    max_samples: int
    timeout_seconds: int
    max_worker_rss_bytes: int
    max_saved_bytes: int

    def __post_init__(self) -> None:
        values = asdict(self)
        require(
            all(type(value) is int and value > 0 for value in values.values()), "Invalid budget"
        )
        require(
            self.workers <= MAX_WORKERS
            and self.max_samples <= MAX_CHUNK_SAMPLES
            and self.timeout_seconds <= 3600,
            "Unsupported worker/sample/deadline budget",
        )


class Child(Protocol):
    @property
    def pid(self) -> int: ...

    def poll(self) -> int | None: ...

    def terminate(self) -> None: ...

    def kill(self) -> None: ...

    def wait(self, timeout: int) -> int: ...


Execute = Callable[[tuple[Path, ...], WindowBudget], dict[str, object]]


@dataclass(frozen=True)
class ProbePlan:
    path: Path
    source: Path
    directory: Path
    samples: int
    budget: WindowBudget
    definition: dict[str, object]


def _integer(value: object) -> int:
    require(type(value) is int and int(str(value)) > 0, "Expected positive integer")
    return int(str(value))


def _references(value: object) -> dict[str, str]:
    result: dict[str, str] = {}
    for item in rows(value):
        row = record(item)
        require(set(row) == {"path", "sha256"}, "Expected exact file reference")
        path = canonical(row.get("path"))
        digest = row.get("sha256")
        require(
            path.is_relative_to(Path.cwd().resolve())
            and isinstance(digest, str)
            and len(digest) == 64
            and all(character in "0123456789abcdef" for character in digest)
            and str(path) not in result,
            "Invalid or duplicate probe reference",
        )
        result[str(path)] = str(digest)
    require(bool(result), "Missing probe references")
    return result


def _code_inputs() -> dict[str, str]:
    return {
        str(source.path): source.sha256
        for source in capture(("separation_window_budget", "code_dependency_inventory"))
    }


def _required_provider_paths() -> set[str]:
    try:
        package = importlib.metadata.distribution("mlx-audio")
        libraries = tuple(importlib.metadata.distribution(name) for name in ("mlx", "mlx-metal"))
    except importlib.metadata.PackageNotFoundError as error:
        raise ReplayError("Required CPU provider is not installed") from error
    required = {
        str(Path(str(package.locate_file(relative))).resolve())
        for relative in (
            "mlx_audio/sts/models/mel_roformer/model.py",
            "mlx_audio/sts/models/mel_roformer/config.py",
        )
    }
    matches: set[str] = set()
    for mlx in libraries:
        matches.update(
            str(Path(str(mlx.locate_file(item))).resolve())
            for item in mlx.files or ()
            if str(item).startswith("mlx/core.") and item.suffix in (".so", ".dylib")
        )
    require(len(matches) == 1, "Expected one installed MLX core library")
    required.update(matches)
    return required


def load_plan(path: Path, directory: Path) -> ProbePlan:
    path = canonical(str(path))
    data = read(path)
    require(
        set(data)
        == {
            "version",
            "kind",
            "source",
            "sourceContract",
            "sampleCount",
            "budget",
            "outputDirectory",
            "codeInputs",
            "providerInputs",
            "modelInputs",
            "qualityApproved",
            "productionBackendAdopted",
        }
        and type(data.get("version")) is int
        and data.get("version") == 1
        and data.get("kind") == "separation-window-probe"
        and data.get("sourceContract") == "interval-only-with-stable-source-identity"
        and data.get("qualityApproved") is False
        and data.get("productionBackendAdopted") is False,
        "Unsupported probe plan",
    )
    source_row = record(data.get("source"))
    require(set(source_row) == {"path", "identity"}, "Expected interval source identity")
    source = canonical(source_row.get("path"))
    require(
        source.is_relative_to(Path.cwd().resolve())
        and identity(source) == source_row.get("identity"),
        "Prepared source identity changed",
    )
    samples = _integer(data.get("sampleCount"))
    require(samples <= 60 * MODEL_SAMPLE_RATE, "Probe interval budget is 60 seconds")
    row = record(data.get("budget"))
    require(
        set(row)
        == {"workers", "max_samples", "timeout_seconds", "max_worker_rss_bytes", "max_saved_bytes"},
        "Invalid plan budget fields",
    )
    budget = WindowBudget(
        workers=_integer(row.get("workers")),
        max_samples=_integer(row.get("max_samples")),
        timeout_seconds=_integer(row.get("timeout_seconds")),
        max_worker_rss_bytes=_integer(row.get("max_worker_rss_bytes")),
        max_saved_bytes=_integer(row.get("max_saved_bytes")),
    )
    directory = local_directory(directory)
    require(
        budget.workers == 2
        and budget.max_samples == samples
        and canonical(data.get("outputDirectory")) == directory,
        "Probe plan clock/output changed",
    )
    _references(data.get("codeInputs"))
    _references(data.get("providerInputs"))
    model = _references(data.get("modelInputs"))
    environment = _local_environment()
    snapshot = (
        Path(environment["HF_HUB_CACHE"])
        / ("models--" + MODEL_NAME.replace("/", "--"))
        / "snapshots"
        / MODEL_REVISION
    )
    require(
        model
        == {
            str(snapshot / "model.safetensors"): MODEL_SHA256,
            str(snapshot / "config.json"): MODEL_CONFIG_SHA256,
        },
        "Probe plan model/cache changed",
    )
    return ProbePlan(path, source, directory, samples, budget, data)


def _plan_context(plan: ProbePlan) -> dict[str, object]:
    code = _references(plan.definition["codeInputs"])
    providers = _references(plan.definition["providerInputs"])
    model = _references(plan.definition["modelInputs"])
    require(code == _code_inputs(), "Full AST source closure changed")
    require(_required_provider_paths() <= set(providers), "Missing installed provider code/library")
    combined = {**code, **providers, **model}
    require(len(combined) == len(code) + len(providers) + len(model), "Overlapping probe pin roles")
    pinned: list[dict[str, object]] = []
    for name, expected in sorted(combined.items()):
        actual = pin(Path(name))
        require(actual["sha256"] == expected, "Declared probe input changed")
        pinned.append(actual)
    require(read(plan.path) == plan.definition, "Probe plan changed")
    return {"plan": pin(plan.path), "inputs": pinned, "environment": _local_environment()}


def windows(samples: int) -> tuple[Window, ...]:
    require(type(samples) is int and 0 < samples <= MAX_CHUNK_SAMPLES, "Invalid sample count")
    result: list[Window] = []
    for start in range(0, samples, HOP):
        count = min(WINDOW, samples - start)
        if start and count <= HOP:
            break
        result.append(Window(len(result), start, count))
    return tuple(result)


def estimated_saved_bytes(samples: int) -> int:
    """Retained float32 input and full window outputs plus explicit metadata room."""
    return (
        samples * 2 * 4
        + ARRAY_HEADER_ALLOWANCE_BYTES
        + len(windows(samples)) * (WINDOW * 2 * 4 + ARRAY_HEADER_ALLOWANCE_BYTES)
        + METADATA_ALLOWANCE_BYTES
    )


def code_pins() -> dict[str, str]:
    paths = (
        Path(__file__),
        Path(__file__).with_name("bgm_separation.py"),
        Path(__file__).with_name("separation_forward_probe.py"),
        Path(__file__).with_name("audio_cpu_probe.py"),
        Path(__file__).with_name("stateful_guard.py"),
    )
    return {str(path.resolve()): sha256(path) for path in paths}


def provider_pins() -> dict[str, str]:
    """Bind the interpreter and installed package metadata without importing MLX.

    The supervising original workflow must additionally pin provider library/code
    files. Metadata/version agreement alone is not a full installed-library digest.
    """
    result = {
        "pythonLexical": sys.executable,
        "pythonSHA256": sha256(Path(sys.executable).resolve()),
        "pythonVersion": sys.version,
        "numpyVersion": np.__version__,
    }
    for name in ("mlx", "mlx-audio", "numpy"):
        try:
            package = importlib.metadata.distribution(name)
        except importlib.metadata.PackageNotFoundError as error:
            raise ReplayError("Required CPU provider is not installed") from error
        files = package.files
        require(files is not None, "Provider metadata inventory missing")
        matches = tuple(path for path in files or () if path.name == "METADATA")
        require(len(matches) == 1, "Expected one provider metadata record")
        metadata = Path(str(package.locate_file(matches[0]))).resolve()
        result[f"{name}Version"] = package.version
        result[f"{name}MetadataSHA256"] = sha256(metadata)
    return result


def local_directory(directory: Path) -> Path:
    directory = canonical(str(directory))
    require(directory.is_relative_to(Path.cwd().resolve() / "edits/work"), "Output must stay local")
    require(not directory.exists() or directory.is_dir(), "Expected output directory")
    return directory


def save_array(path: Path, audio: NDArray[np.float32]) -> None:
    with path.open("xb") as stream:
        os.fchmod(stream.fileno(), 0o600)
        np.save(stream, audio, allow_pickle=False)
        stream.flush()
        os.fsync(stream.fileno())


def load_array(path: Path, *, samples: int) -> NDArray[np.float32]:
    before = identity(path)
    require(before["bytes"] <= samples * 8 + ARRAY_HEADER_ALLOWANCE_BYTES, "Oversized array")
    descriptor = os.open(path, os.O_RDONLY | os.O_NOFOLLOW)
    with os.fdopen(descriptor, "rb") as stream:
        opened = os.fstat(stream.fileno())
        require(
            (opened.st_dev, opened.st_ino) == (path.lstat().st_dev, before["inode"]),
            "Array replaced before read",
        )
        loaded: object = np.load(stream, allow_pickle=False)
    require(identity(path) == before, "Array identity changed")
    if not isinstance(loaded, np.ndarray):
        raise ReplayError("Expected numeric array")
    require(loaded.dtype == np.dtype(np.float32) and loaded.shape == (2, samples), "Invalid array")
    return loaded


def reduce_windows(
    samples: int, predictions: Callable[[Window], NDArray[np.float32]]
) -> NDArray[np.float32]:
    """Keep every float64 addition and blend in the original backend order."""
    output = np.zeros((2, samples), dtype=np.float64)
    coverage = np.zeros(samples, dtype=np.float64)
    ramp = np.sin(np.linspace(0.0, np.pi / 2.0, HOP, endpoint=False)) ** 2
    for window in windows(samples):
        predicted = predictions(window)
        require(
            predicted.dtype == np.dtype(np.float32)
            and predicted.shape == (2, WINDOW)
            and bool(np.isfinite(predicted).all()),
            "Invalid independent prediction",
        )
        voice = predicted[:, : window.count]
        end = window.start + window.count
        blend = np.ones(window.count, dtype=np.float64)
        if window.start:
            blend[: min(window.count, HOP)] = ramp[: min(window.count, HOP)]
        if end < samples:
            blend[-HOP:] *= 1.0 - ramp
        output[:, window.start : end] += voice * blend
        coverage[window.start : end] += blend
    require(not bool(np.any(coverage <= 0)), "Uncovered independent audio")
    return np.asarray(output / coverage[None, :], dtype=np.float32)


def request_definition(
    payload: Path,
    *,
    samples: int,
    selected: tuple[Window, ...],
    budget: WindowBudget,
    directory: Path,
    device: str = "cpu",
) -> dict[str, object]:
    require(device in DEVICES, "Unsupported separation device")
    environment = _local_environment()
    scratch = directory / "tmp"
    cache = directory / "cache"
    scratch.mkdir(mode=0o700, exist_ok=True)
    cache.mkdir(mode=0o700, exist_ok=True)
    require(scratch == scratch.resolve() and cache == cache.resolve(), "Linked worker scratch")
    environment.update(
        {
            "TMPDIR": str(scratch),
            "TMP": str(scratch),
            "TEMP": str(scratch),
            "XDG_CACHE_HOME": str(cache),
        }
    )
    return {
        "version": 1,
        "payload": str(payload),
        "payloadSHA256": sha256(payload),
        "payloadIdentity": identity(payload),
        "samples": samples,
        "windows": [asdict(window) for window in selected],
        "budget": asdict(budget),
        "codePins": code_pins(),
        "providerPins": provider_pins(),
        "modelRevision": MODEL_REVISION,
        "weightsSHA256": MODEL_SHA256,
        "configSHA256": MODEL_CONFIG_SHA256,
        "environment": environment,
        "device": device,
        "qualityApproved": False,
    }


def selected_windows(request: dict[str, object]) -> tuple[Window, ...]:
    samples = request.get("samples")
    require(type(samples) is int, "Missing sample count")
    planned = {window.index: window for window in windows(int(str(samples)))}
    selected: list[Window] = []
    for value in rows(request.get("windows")):
        row = record(value)
        index = row.get("index")
        require(type(index) is int and index in planned, "Unknown window")
        window = planned[int(str(index))]
        require(row == asdict(window), "Window clock changed")
        selected.append(window)
    require(
        bool(selected)
        and [window.index for window in selected] == sorted({w.index for w in selected}),
        "Duplicate or unordered windows",
    )
    return tuple(selected)


def verify_worker(directory: Path) -> dict[str, object]:
    request = read(directory / "request.json")
    payload = canonical(request.get("payload"))
    require(payload == directory.parent / "chunk.npy", "Worker payload escapes its owner")
    require(
        request.get("codePins") == code_pins()
        and request.get("providerPins") == provider_pins()
        and request.get("device") in DEVICES
        and request.get("modelRevision") == MODEL_REVISION
        and request.get("weightsSHA256") == MODEL_SHA256
        and request.get("configSHA256") == MODEL_CONFIG_SHA256
        and identity(payload) == request.get("payloadIdentity")
        and sha256(payload) == request.get("payloadSHA256"),
        "Worker input/code/model pins changed",
    )
    outputs = {
        str(w.index): sha256(directory / f"window-{w.index:06d}.npy")
        for w in selected_windows(request)
    }
    result = read(directory / "result.json")
    require(
        result.get("requestSHA256") == sha256(directory / "request.json")
        and result.get("outputs") == outputs
        and result.get("qualityApproved") is False,
        "Worker completion changed",
    )
    return result


def worker(
    directory: Path,
    *,
    backend_factory: BackendFactory | None = None,
    auditor: Auditor = audit_descriptors,
) -> None:
    """A fresh Python process owns its model, arrays, scratch and only its outputs."""
    directory = local_directory(directory)
    request = read(directory / "request.json")
    payload = canonical(request.get("payload"))
    samples = request.get("samples")
    require(type(samples) is int, "Missing sample count")
    selected = selected_windows(request)
    require(payload == directory.parent / "chunk.npy", "Worker payload escapes its owner")
    destinations = (
        directory / "result.json",
        *(directory / f"window-{w.index:06d}.npy" for w in selected),
    )
    require(
        not any(path.exists() or path.is_symlink() for path in destinations),
        "Worker already has output; never repeat inference",
    )
    require(
        request.get("codePins") == code_pins()
        and request.get("providerPins") == provider_pins()
        and request.get("environment") == _local_environment()
        and request.get("device") in DEVICES
        and request.get("modelRevision") == MODEL_REVISION
        and request.get("weightsSHA256") == MODEL_SHA256
        and request.get("configSHA256") == MODEL_CONFIG_SHA256
        and identity(payload) == request.get("payloadIdentity")
        and sha256(payload) == request.get("payloadSHA256"),
        "Worker initialization pins changed",
    )
    source = load_array(payload, samples=int(str(samples)))
    auditor(directory, "before-load")
    loaded = time.monotonic()
    device = str(request["device"])
    backend = backend_factory() if backend_factory is not None else _MlxBackend(device)
    load_seconds = time.monotonic() - loaded
    auditor(directory, "after-load")
    timings: dict[str, float] = {}
    for window in selected:
        padded = np.zeros((2, WINDOW), dtype=np.float32)
        padded[:, : window.count] = source[:, window.start : window.start + window.count]
        require(bool(np.isfinite(padded).all()), "Nonfinite window input")
        started = time.monotonic()
        predicted = backend.extract_vocals(padded)
        require(
            predicted.shape == (2, WINDOW) and bool(np.isfinite(predicted).all()), "Invalid voice"
        )
        require(
            device == "cpu" or not bool(np.any(padded)) or bool(np.any(predicted)),
            "Silent accelerator output for non-silent input",
        )
        timings[str(window.index)] = time.monotonic() - started
        save_array(directory / f"window-{window.index:06d}.npy", predicted)
    auditor(directory, "after-inference")
    require(
        identity(payload) == request.get("payloadIdentity")
        and sha256(payload) == request.get("payloadSHA256")
        and code_pins() == request.get("codePins")
        and provider_pins() == request.get("providerPins"),
        "Worker inputs changed during inference",
    )
    peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    outputs = {str(w.index): sha256(directory / f"window-{w.index:06d}.npy") for w in selected}
    durable(
        directory / "result.json",
        {
            "requestSHA256": sha256(directory / "request.json"),
            "outputs": outputs,
            "modelLoadSeconds": load_seconds,
            "windowSeconds": timings,
            "kernelPeakRSSBytes": peak if sys.platform == "darwin" else peak * 1024,
            "descriptorSnapshotsAreWriteTrace": False,
            "qualityApproved": False,
        },
    )


def child_rss(pid: int) -> int:
    result = subprocess.run(
        ["/bin/ps", "-o", "rss=", "-p", str(pid)], capture_output=True, text=True, timeout=2
    )
    require(result.returncode in (0, 1), "Cannot sample owned child RSS")
    text = result.stdout.strip()
    require(not text or text.isdecimal(), "Invalid owned child RSS")
    return int(text) * 1024 if text else 0


def stop_child(child: Child) -> None:
    if child.poll() is None:
        child.terminate()
        try:
            child.wait(timeout=5)
        except subprocess.TimeoutExpired:
            child.kill()
            child.wait(timeout=5)


def worker_command(directory: Path) -> list[str]:
    """The worker process command; accelerator workers run in a checkout-only write sandbox.

    On first GPU use macOS writes the Metal shader cache under the per-user cache directory,
    which the descriptor audit rightly refuses. As in ``separation_device_pipeline``, a non-CPU
    worker therefore runs under ``sandbox-exec`` denying every write outside the checkout, so the
    audit stays unchanged and nothing is written outside the repository.
    """
    command = [sys.executable, "-B", "-m", "replay.separation_window_budget", "--worker"]
    command.append(str(directory))
    if read(directory / "request.json").get("device", "cpu") == "cpu":
        return command
    profile = directory / "write-sandbox.sb"
    with profile.open("x", encoding="utf-8") as stream:
        stream.write(sandbox_profile(Path.cwd().resolve()))
    return [SANDBOX_EXEC, "-f", str(profile), *command]


def execute_children(directories: tuple[Path, ...], budget: WindowBudget) -> dict[str, object]:
    require(0 < len(directories) <= budget.workers, "Too many worker children")
    active: list[Child] = []
    started = time.monotonic()
    peak = 0
    try:
        with ExitStack() as stack:
            for directory in directories:
                require(
                    not (directory / "attempt.json").exists(), "Uncertain worker attempt; reconcile"
                )
                durable(directory / "attempt.json", {"status": "reserved", "freshProcess": True})
                log = stack.enter_context((directory / "worker.log").open("xb"))
                os.fchmod(log.fileno(), 0o600)
                log.write(b"WINDOW_WORKER reserved\n")
                log.flush()
                environment = os.environ.copy()
                selected_environment = record(read(directory / "request.json").get("environment"))
                require(
                    all(isinstance(value, str) for value in selected_environment.values()),
                    "Invalid worker environment",
                )
                environment.update({key: str(value) for key, value in selected_environment.items()})
                child = subprocess.Popen(
                    worker_command(directory),
                    stdin=subprocess.DEVNULL,
                    stdout=log,
                    stderr=subprocess.STDOUT,
                    env=environment,
                )
                active.append(child)
            while True:
                rss = [child_rss(child.pid) for child in active]
                peak = max(peak, sum(rss))
                require(
                    all(value <= budget.max_worker_rss_bytes for value in rss)
                    and time.monotonic() - started <= budget.timeout_seconds
                    and saved_bytes(directories[0].parent) <= budget.max_saved_bytes,
                    "Sampled RSS/disk/deadline guard exceeded",
                )
                codes = [child.poll() for child in active]
                require(all(code is None or code == 0 for code in codes), "Worker child failed")
                if all(code is not None for code in codes):
                    break
                time.sleep(POLL_SECONDS)
    finally:
        for owned_child in active:
            stop_child(owned_child)
    results = [verify_worker(directory) for directory in directories]
    require(
        all(
            type(row.get("kernelPeakRSSBytes")) is int
            and int(str(row["kernelPeakRSSBytes"])) <= budget.max_worker_rss_bytes
            for row in results
        ),
        "Kernel worker peak exceeded budget",
    )
    return {
        "sampledWorkerSumPeakRSSBytes": peak,
        "elapsedSeconds": time.monotonic() - started,
        "hardCap": False,
    }


def run_windows(
    audio: NDArray[np.float32],
    directory: Path,
    *,
    budget: WindowBudget,
    execute: Execute = execute_children,
    device: str = "cpu",
) -> NDArray[np.float32]:
    require(device in DEVICES, "Unsupported separation device")
    directory = local_directory(directory)
    require(
        audio.dtype == np.dtype(np.float32)
        and audio.ndim == 2
        and audio.shape[0] == 2
        and 0 < audio.shape[1] <= budget.max_samples
        and bool(np.isfinite(audio).all()),
        "Expected bounded stereo float32 input",
    )
    samples = int(audio.shape[1])
    require(
        estimated_saved_bytes(samples) <= budget.max_saved_bytes,
        "Saved byte budget is insufficient",
    )
    directory.mkdir(mode=0o700, exist_ok=True)
    payload = directory / "chunk.npy"
    selected = windows(samples)
    definition = {
        "sampleCount": samples,
        "audioSHA256": hashlib.sha256(audio.tobytes(order="C")).hexdigest(),
        "budget": asdict(budget),
        "codePins": code_pins(),
        "providerPins": provider_pins(),
    }
    if device != "cpu":
        definition["device"] = device
    with execution_lock(directory, ".parent") as acquired:
        require(acquired, "Parallel window owner already active")
        if (directory / "input.json").exists():
            require(read(directory / "input.json") == definition, "Parallel run input changed")
        else:
            require(not payload.exists(), "Unowned partial payload")
            durable(directory / "input.json", definition)
            save_array(payload, audio)
        require(
            np.array_equal(load_array(payload, samples=samples), audio), "Parallel input changed"
        )
        count = min(budget.workers, len(selected))
        directories = tuple(directory / f"worker-{index:02d}" for index in range(count))
        for index, child_directory in enumerate(directories):
            child_directory.mkdir(mode=0o700, exist_ok=True)
            request = request_definition(
                payload,
                samples=samples,
                selected=selected[index::count],
                budget=budget,
                directory=child_directory,
                device=device,
            )
            path = child_directory / "request.json"
            if path.exists():
                require(read(path) == request, "Worker request changed")
            else:
                durable(path, request)
        missing = tuple(path for path in directories if not (path / "result.json").exists())
        measurement = execute(missing, budget) if missing else {"reusedCompleteWorkers": True}
        for child_directory in directories:
            verify_worker(child_directory)

        def prediction(window: Window) -> NDArray[np.float32]:
            return load_array(
                directories[window.index % count] / f"window-{window.index:06d}.npy", samples=WINDOW
            )

        result = reduce_windows(samples, prediction)
        require(
            code_pins() == definition["codePins"] and provider_pins() == definition["providerPins"],
            "Parent source/provider changed",
        )
        if not (directory / "result.json").exists():
            durable(
                directory / "result.json",
                {
                    "inputSHA256": sha256(directory / "input.json"),
                    "workers": count,
                    "measurement": measurement,
                    "qualityApproved": False,
                },
            )
        else:
            require(
                read(directory / "result.json").get("inputSHA256")
                == sha256(directory / "input.json")
                and read(directory / "result.json").get("workers") == count
                and read(directory / "result.json").get("qualityApproved") is False,
                "Parent completion changed",
            )
        require(saved_bytes(directory) <= budget.max_saved_bytes, "Final saved bytes exceeded")
        return result


class WindowBackend:
    """Callable SeparationBackend seam; identical chunk order enables complete-only reuse."""

    def __init__(
        self, directory: Path, *, budget: WindowBudget, execute: Execute = execute_children
    ) -> None:
        self.directory = local_directory(directory)
        self.directory.mkdir(mode=0o700, exist_ok=True)
        self.budget = budget
        self.execute = execute
        self.chunk_index = 0

    @property
    def sources(self) -> tuple[str, ...]:
        return (VOCALS_SOURCE, "instrumental")

    def extract_vocals(self, chunk: NDArray[np.float32]) -> NDArray[np.float32]:
        destination = self.directory / f"chunk-{self.chunk_index:06d}"
        output = run_windows(chunk, destination, budget=self.budget, execute=self.execute)
        self.chunk_index += 1
        return output

    def extract_instrumental(self, chunk: NDArray[np.float32]) -> NDArray[np.float32]:
        return chunk - self.extract_vocals(chunk)


def probe(
    source: Path,
    directory: Path,
    *,
    samples: int,
    budget: WindowBudget,
    context: dict[str, object] | None = None,
) -> None:
    require(samples <= 60 * MODEL_SAMPLE_RATE, "Probe interval budget is 60 seconds")
    interval = read_interval(source, start_sample=0, samples=samples)
    directory = local_directory(directory)
    directory.mkdir(mode=0o700)
    if context is not None:
        durable(
            directory / "input.json",
            {
                "context": context,
                "intervalSHA256": interval.sha256,
                "sourceIdentity": interval.source_identity,
                "sourceSamples": interval.source_samples,
                "sampleCount": samples,
                "budget": asdict(budget),
            },
        )
    single = WindowBudget(
        workers=1,
        max_samples=budget.max_samples,
        timeout_seconds=budget.timeout_seconds,
        max_worker_rss_bytes=budget.max_worker_rss_bytes,
        max_saved_bytes=budget.max_saved_bytes,
    )
    started = time.monotonic()
    baseline = run_windows(interval.audio, directory / "serial", budget=single)
    serial_seconds = time.monotonic() - started
    started = time.monotonic()
    parallel = run_windows(interval.audio, directory / "parallel", budget=budget)
    parallel_seconds = time.monotonic() - started
    after = read_interval(source, start_sample=0, samples=samples)
    require(
        after.sha256 == interval.sha256 and after.source_identity == interval.source_identity,
        "Probe source changed",
    )
    _write_wav(directory / "serial.wav", baseline)
    _write_wav(directory / "parallel.wav", parallel)
    durable(
        directory / "comparison.json",
        {
            "serialSeconds": serial_seconds,
            "parallelSeconds": parallel_seconds,
            "sampleCount": samples,
            "comparison": compare_audio(baseline, parallel),
            "productionBackendAdopted": False,
            "qualityApproved": False,
        },
    )


def _serial_budget(budget: WindowBudget) -> WindowBudget:
    return WindowBudget(
        workers=1,
        max_samples=budget.max_samples,
        timeout_seconds=budget.timeout_seconds,
        max_worker_rss_bytes=budget.max_worker_rss_bytes,
        max_saved_bytes=budget.max_saved_bytes,
    )


def _probe_outputs(plan: ProbePlan) -> tuple[Path, ...]:
    paths = [
        plan.directory / name
        for name in ("input.json", "comparison.json", "serial.wav", "parallel.wav")
    ]
    for name, budget in (("serial", _serial_budget(plan.budget)), ("parallel", plan.budget)):
        pool = plan.directory / name
        paths.extend(pool / item for item in ("input.json", "result.json", "chunk.npy"))
        count = min(budget.workers, len(windows(plan.samples)))
        for index in range(count):
            child = pool / f"worker-{index:02d}"
            paths.extend(
                child / item
                for item in (
                    "request.json",
                    "result.json",
                    "attempt.json",
                    "worker.log",
                    "before-load-descriptors.json",
                    "after-load-descriptors.json",
                    "after-inference-descriptors.json",
                )
            )
            paths.extend(
                child / f"window-{item.index:06d}.npy"
                for item in windows(plan.samples)[index::count]
            )
    return tuple(sorted(paths))


def _nonnegative_number(value: object) -> bool:
    return (
        type(value) in (int, float) and math.isfinite(float(str(value))) and float(str(value)) >= 0
    )


def _verify_pool(
    plan: ProbePlan, name: str, audio: NDArray[np.float32], device: str = "cpu"
) -> NDArray[np.float32]:
    pool = plan.directory / name
    budget = _serial_budget(plan.budget) if name == "serial" else plan.budget
    expected = {
        "sampleCount": plan.samples,
        "audioSHA256": hashlib.sha256(audio.tobytes(order="C")).hexdigest(),
        "budget": asdict(budget),
        "codePins": code_pins(),
        "providerPins": provider_pins(),
    }
    if device != "cpu":
        expected["device"] = device
    require(read(pool / "input.json") == expected, "Pool input binding changed")
    require(
        np.array_equal(load_array(pool / "chunk.npy", samples=plan.samples), audio),
        "Pool interval changed",
    )
    count = min(budget.workers, len(windows(plan.samples)))
    completed = read(pool / "result.json")
    require(
        completed.get("inputSHA256") == sha256(pool / "input.json")
        and completed.get("workers") == count
        and completed.get("qualityApproved") is False
        and saved_bytes(pool) <= budget.max_saved_bytes,
        "Pool completion/budget changed",
    )
    for index in range(count):
        child = pool / f"worker-{index:02d}"
        request = read(child / "request.json")
        require(
            selected_windows(request) == windows(plan.samples)[index::count]
            and request.get("samples") == plan.samples
            and request.get("budget") == asdict(budget)
            and request.get("device") == device,
            "Worker allocation changed",
        )
        environment = _local_environment()
        scratch, cache = child / "tmp", child / "cache"
        require(
            scratch.is_dir()
            and cache.is_dir()
            and scratch == scratch.resolve()
            and cache == cache.resolve(),
            "Worker scratch/cache missing or linked",
        )
        environment.update(
            {
                "TMPDIR": str(scratch),
                "TMP": str(scratch),
                "TEMP": str(scratch),
                "XDG_CACHE_HOME": str(cache),
            }
        )
        require(request.get("environment") == environment, "Worker scratch environment changed")
        result = verify_worker(child)
        timings = record(result.get("windowSeconds"))
        require(
            type(result.get("kernelPeakRSSBytes")) is int
            and 0 < int(str(result["kernelPeakRSSBytes"])) <= budget.max_worker_rss_bytes
            and _nonnegative_number(result.get("modelLoadSeconds"))
            and set(timings) == {str(item.index) for item in windows(plan.samples)[index::count]}
            and all(_nonnegative_number(value) for value in timings.values())
            and result.get("descriptorSnapshotsAreWriteTrace") is False
            and read(child / "attempt.json") == {"status": "reserved", "freshProcess": True},
            "Worker measurement/attempt changed",
        )
        for phase in ("before-load", "after-load", "after-inference"):
            audit = read(child / f"{phase}-descriptors.json")
            require(
                type(audit.get("pid")) is int
                and int(str(audit["pid"])) > 0
                and isinstance(audit.get("snapshot"), str)
                and audit.get("externalWritableRegular") == [],
                "Worker descriptor audit missing/unsafe",
            )
            require(
                not external_writable_files(str(audit["snapshot"]), Path.cwd().resolve()),
                "Worker observed external writable file",
            )

    def prediction(item: Window) -> NDArray[np.float32]:
        return load_array(
            pool / f"worker-{item.index % count:02d}" / f"window-{item.index:06d}.npy",
            samples=WINDOW,
        )

    return reduce_windows(plan.samples, prediction)


def verify_probe(plan_path: Path, directory: Path) -> None:
    """Read complete records and recompute reductions; never infer, repair or lock."""
    plan = load_plan(plan_path, directory)
    context = _plan_context(plan)
    receipt_pin = pin(plan.directory / "result.json")
    receipt = read(plan.directory / "result.json")
    before = [pin(path) for path in _probe_outputs(plan)]
    require(receipt.get("outputs") == before, "Probe output receipt changed")
    interval = read_interval(plan.source, start_sample=0, samples=plan.samples)
    expected = {
        "context": context,
        "intervalSHA256": interval.sha256,
        "sourceIdentity": interval.source_identity,
        "sourceSamples": interval.source_samples,
        "sampleCount": plan.samples,
        "budget": asdict(plan.budget),
    }
    require(read(plan.directory / "input.json") == expected, "Probe provenance changed")
    baseline = _verify_pool(plan, "serial", interval.audio)
    parallel = _verify_pool(plan, "parallel", interval.audio)
    comparison = read(plan.directory / "comparison.json")
    require(
        comparison.get("sampleCount") == plan.samples
        and comparison.get("comparison") == compare_audio(baseline, parallel)
        and _nonnegative_number(comparison.get("serialSeconds"))
        and _nonnegative_number(comparison.get("parallelSeconds"))
        and comparison.get("productionBackendAdopted") is False
        and comparison.get("qualityApproved") is False
        and receipt.get("planSHA256") == sha256(plan.path)
        and receipt.get("productionBackendAdopted") is False
        and receipt.get("qualityApproved") is False,
        "Probe comparison/approval changed",
    )
    for name, predicted in (("serial", baseline), ("parallel", parallel)):
        saved = read_interval(plan.directory / f"{name}.wav", start_sample=0, samples=plan.samples)
        pcm = (np.clip(predicted, -1.0, 1.0) * 32767.0).astype(np.int16).astype(
            np.float32
        ) / 32768.0
        require(
            saved.source_samples == plan.samples and np.array_equal(saved.audio, pcm),
            "Saved PCM clock/content changed",
        )
    require(
        _plan_context(plan) == context
        and [pin(path) for path in _probe_outputs(plan)] == before
        and read(plan.directory / "result.json") == receipt
        and pin(plan.directory / "result.json") == receipt_pin
        and identity(plan.source) == interval.source_identity,
        "Probe inputs/outputs changed during readonly verification",
    )


def probe_plan(plan_path: Path, directory: Path) -> None:
    plan = load_plan(plan_path, directory)
    context = _plan_context(plan)
    require(
        not plan.directory.exists() and not plan.directory.is_symlink(),
        "Probe already attempted; inspect retained outputs",
    )
    probe(plan.source, plan.directory, samples=plan.samples, budget=plan.budget, context=context)
    require(_plan_context(plan) == context, "Probe declared inputs changed during inference")
    durable(
        plan.directory / "result.json",
        {
            "planSHA256": sha256(plan.path),
            "outputs": [pin(path) for path in _probe_outputs(plan)],
            "productionBackendAdopted": False,
            "qualityApproved": False,
        },
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--worker", type=Path)
    parser.add_argument("--source", type=Path)
    parser.add_argument("--directory", type=Path)
    parser.add_argument("--samples", type=int)
    parser.add_argument("--plan", type=Path)
    parser.add_argument("--verify", action="store_true")
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--timeout", type=int, default=600)
    parser.add_argument("--max-worker-rss-bytes", type=int)
    parser.add_argument("--max-saved-bytes", type=int)
    args = parser.parse_args(argv)
    try:
        if args.worker is not None:
            require(args.plan is None and not args.verify, "Worker cannot adopt a probe plan")
            worker(args.worker)
        elif args.plan is not None:
            require(
                args.directory is not None and args.source is None and args.samples is None,
                "Plan owns probe paths/clock",
            )
            action = verify_probe if args.verify else probe_plan
            action(args.plan, args.directory)
        else:
            require(not args.verify, "Readonly verification requires a fixed plan")
            require(
                args.source is not None
                and args.directory is not None
                and type(args.samples) is int
                and type(args.max_worker_rss_bytes) is int
                and type(args.max_saved_bytes) is int,
                "Probe paths/sample/resource budgets required",
            )
            budget = WindowBudget(
                workers=args.workers,
                max_samples=args.samples,
                timeout_seconds=args.timeout,
                max_worker_rss_bytes=args.max_worker_rss_bytes,
                max_saved_bytes=args.max_saved_bytes,
            )
            probe(args.source, args.directory, samples=args.samples, budget=budget)
    except (ReplayError, OSError, ValueError, RuntimeError, subprocess.SubprocessError) as error:
        print(f"WINDOW_BUDGET_ERROR {error}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
