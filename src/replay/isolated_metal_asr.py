"""Serial, fresh-process Metal ASR windows on the original PCM sample clock.

This is a raw-recognition producer, not quality approval or a resource admission
service. The caller admits one GPU family. Completed windows resume unchanged;
an interrupted child intent is never automatically submitted again. CPU adoption
requires a separate verified quiescent handoff and is intentionally not inferred.
"""

from __future__ import annotations

import argparse
import fcntl
import hashlib
import importlib
import json
import os
import signal
import subprocess
import sys
import sysconfig
import time
import wave
from collections.abc import Sequence
from contextlib import AbstractContextManager, suppress
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

import numpy as np

from replay.audio_cpu_probe import audit_descriptors
from replay.audio_windows import AudioWindow, plan_audio_windows
from replay.core import ReplayError
from replay.fcp_recovery_runtime import SystemHost
from replay.isolated_metal_asr_probe import planned_environment, sandbox_profile
from replay.local_asr import DEFAULT_CONFIG_SHA256, DEFAULT_WEIGHTS_SHA256
from replay.measured_job_admission import _descendants, parse_processes
from replay.native_invocation import canonical, durable, identity, pin, require
from replay.recordings import record
from replay.sequence_journal import bounded_read, unique_pairs

RATE = 16000
MAX_SAMPLES = 21600 * RATE
PHASES = ("before-core-import", "after-core-import", "after-whisper-import", "after-inference-eval")
DENIALS = ("deny file-write", "sandbox: deny", "operation not permitted")


def _read(path: Path) -> dict[str, object]:
    return record(json.loads(bounded_read(path), object_pairs_hook=unique_pairs))


def _integer(value: object, low: int, high: int) -> int:
    if type(value) is not int or not low <= value <= high:
        raise ReplayError("Invalid bounded integer")
    return value


def _local(value: object, root: Path) -> Path:
    path = canonical(value)
    require(path.is_relative_to(root), "Input/output escapes checkout")
    return path


def _digest(value: object) -> str:
    require(isinstance(value, str) and len(value) == 64, "Missing SHA256")
    result = str(value)
    require(all(char in "0123456789abcdef" for char in result), "Malformed SHA256")
    return result


@dataclass(frozen=True)
class Request:
    path: Path
    sha256: str
    pcm: Path
    weights: Path
    config: Path
    samples: int
    timeout: int
    pins: dict[Path, str]


def required_code() -> tuple[Path, ...]:
    """Minimum direct execution/provider bindings; callers may bind more sources."""
    source = Path(__file__).resolve().parent
    provider = (
        Path(sys.prefix)
        / f"lib/python{sys.version_info.major}.{sys.version_info.minor}"
        / "site-packages/mlx_whisper"
    )
    suffix: object = sysconfig.get_config_var("EXT_SUFFIX")
    require(isinstance(suffix, str) and bool(suffix), "Missing native extension ABI")
    mlx = provider.parent / "mlx"
    return (
        tuple(
            source / name
            for name in (
                "isolated_metal_asr.py",
                "isolated_metal_asr_probe.py",
                "audio_cpu_probe.py",
                "audio_windows.py",
                "native_invocation.py",
                "sequence_journal.py",
                "measured_job_admission.py",
                "fcp_recovery_runtime.py",
                "fcp_native_queue.py",
            )
        )
        + tuple(
            provider / name
            for name in (
                "__init__.py",
                "transcribe.py",
                "decoding.py",
                "load_models.py",
                "audio.py",
                "timing.py",
                "tokenizer.py",
                "whisper.py",
            )
        )
        + (mlx / f"core{suffix}", mlx / "lib/libmlx.dylib", mlx / "lib/mlx.metallib")
    )


def load_request(path: Path, *, verify: bool = True) -> Request:
    root = Path.cwd().resolve()
    _local(str(path), root)
    value = _read(path)
    require(value.get("version") == 1 and value.get("device") == "metal", "Wrong ASR backend")
    require(
        value.get("fp16") is True and value.get("cpuAdoption") is None, "Unsupported dtype/adoption"
    )
    inputs = record(value.get("inputs"))
    paths: list[Path] = []
    pins: dict[Path, str] = {}
    for role in ("pcm", "weights", "config"):
        ref = record(inputs.get(role))
        actual = _local(ref.get("path"), root)
        require(identity(actual)["bytes"] > 0, "Empty input")
        require(actual not in pins, "Duplicate input role")
        pins[actual] = _digest(ref.get("sha256"))
        paths.append(actual)
    require(paths[1].parent == paths[2].parent, "Model directory mismatch")
    require(
        pins[paths[1]] == DEFAULT_WEIGHTS_SHA256 and pins[paths[2]] == DEFAULT_CONFIG_SHA256,
        "Only pinned large-v3 model is supported",
    )
    code = record(value.get("codeInputs"))
    require(1 <= len(code) <= 1024, "Invalid code closure")
    require(
        {str(item) for item in required_code()} <= code.keys(), "Missing producer/provider pins"
    )
    for name, digest in code.items():
        actual = _local(name, root)
        require(actual not in pins, "Code/input collision")
        pins[actual] = _digest(digest)
    request = Request(
        path,
        str(pin(path)["sha256"]),
        paths[0],
        paths[1],
        paths[2],
        _integer(value.get("samples"), 1, MAX_SAMPLES),
        _integer(value.get("timeoutSeconds"), 1, 3600),
        pins,
    )
    if verify:
        verify_inputs(request)
    return request


def verify_inputs(request: Request) -> dict[str, object]:
    identities: dict[str, object] = {}
    for path, digest in request.pins.items():
        observed = pin(path)
        require(observed["sha256"] == digest, "ASR input/code changed")
        identities[str(path)] = observed["stat"]
    require(pin(request.path)["sha256"] == request.sha256, "Request changed")
    with wave.open(str(request.pcm), "rb") as audio:
        require(
            (audio.getnchannels(), audio.getsampwidth(), audio.getframerate(), audio.getcomptype())
            == (1, 2, RATE, "NONE")
            and audio.getnframes() == request.samples,
            "PCM clock mismatch",
        )
    return identities


def _identities(expected: dict[str, object]) -> None:
    for name, observed in expected.items():
        require(identity(Path(name)) == observed, "Pinned file identity changed")


def _clock(window: AudioWindow, digest: str) -> dict[str, object]:
    return {
        "startSample": window.start_sample,
        "endSample": window.end_sample,
        "sampleRate": RATE,
        "humanReviewed": False,
        "emptyRecognitionIsNotSilence": True,
        "rawSHA256": digest,
    }


def _same(path: Path, data: dict[str, object]) -> None:
    if path.exists() or path.is_symlink():
        require(_read(path) == data, "Immutable ASR evidence differs")
    else:
        durable(path, data)


class Runtime(Protocol):
    def execute(
        self, request: Request, window: AudioWindow, work: Path, identities: dict[str, object]
    ) -> None: ...


def _evidence(work: Path) -> dict[str, object]:
    raw = _read(work / "raw.json")
    require(isinstance(raw.get("segments"), list), "Missing raw native segments")
    log = bounded_read(work / "worker.log").decode("utf-8", errors="replace")
    require(not any(word in log.lower() for word in DENIALS), "Sandbox denial")
    paths = [
        work / "raw.json",
        work / "worker.log",
        work / "resources.json",
        work / "mlx-memory.json",
    ]
    for phase in PHASES:
        path = work / f"{phase}-descriptors.json"
        require(_read(path).get("externalWritableRegular") == [], "External writable descriptor")
        paths.append(path)
    resources = _read(work / "resources.json")
    require(
        resources.get("exitCode") == 0 and resources.get("familyGone") is True,
        "Incomplete child family",
    )
    require(
        isinstance(resources.get("samples"), list) and bool(resources["samples"]),
        "Missing family measurements",
    )
    paths.extend(work / name for name in ("intent.json", "child.json", "launch.json"))
    memory = _read(work / "mlx-memory.json")
    for key in ("activeBytes", "peakBytes", "cacheBytes"):
        _integer(memory.get(key), 0, 2**63 - 1)
    require(memory.get("device") == "metal", "Wrong memory backend")
    # Empty logs are normal; retain their SHA without pretending they are nonempty media.

    return {str(path): hashlib.sha256(bounded_read(path)).hexdigest() for path in paths}


def collect(
    request_path: Path,
    output: Path,
    *,
    runtime: Runtime | None = None,
    verify_only: bool = False,
    maximum_new_windows: int | None = None,
) -> Path:
    if maximum_new_windows is not None:
        _integer(maximum_new_windows, 1, 720)
        require(not verify_only, "Verification cannot truncate the collection")
    request = load_request(request_path, verify=False)
    identities = verify_inputs(request)
    root = Path.cwd().resolve()
    require(
        _local(str(output), root).is_relative_to(root / "edits/work"), "Output must be work-local"
    )
    if verify_only:
        require(output.is_dir(), "Missing collection")
    else:
        output.mkdir(mode=0o700, parents=True, exist_ok=True)
    flags = os.O_RDWR | os.O_NOFOLLOW | (0 if verify_only else os.O_CREAT)
    lock = os.open(output / ".run.lock", flags, 0o600)
    try:
        require(identity(output / ".run.lock")["inode"] == os.fstat(lock).st_ino, "Lock changed")
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        manifest = {
            "version": 1,
            "inputs": {
                str(request.path): request.sha256,
                **{
                    str(path): request.pins[path]
                    for path in (request.pcm, request.weights, request.config)
                },
            },
            "samples": request.samples,
            "sampleRate": RATE,
            "windowSamples": 60 * RATE,
            "strideSamples": 30 * RATE,
            "clock": "input-pcm-not-projected",
            "device": "metal",
            "dtypeContract": "provider-default-fp16-unchanged",
            "processPolicy": "fresh-child-per-window",
            "providerOptions": {
                "language": "ja",
                "verbose": False,
                "condition_on_previous_text": False,
                "word_timestamps": True,
                "temperature": 0.0,
                "fp16": True,
            },
            "compileCachePolicy": "provider-default; checkout-write-only-child-sandbox",
            "compilerServiceConfinementVerified": False,
        }
        if verify_only:
            require(_read(output / "inputs.json") == manifest, "Manifest mismatch")
        else:
            _same(output / "inputs.json", manifest)
        windows = plan_audio_windows(
            request.samples, window_samples=60 * RATE, stride_samples=30 * RATE
        )
        hashes: list[str] = []
        provenance: list[dict[str, object]] = []
        new_windows = 0
        for window in windows:
            work = output / f"{window.start_sample:012d}.metal-runtime"
            terminal = work / "complete.json"
            if not terminal.exists():
                if maximum_new_windows is not None and new_windows >= maximum_new_windows:
                    verify_inputs(request)
                    partial = (
                        output
                        / f"partial-through-{windows[len(hashes) - 1].start_sample:012d}.json"
                    )
                    _same(
                        partial,
                        {
                            "version": 1,
                            "collectionComplete": False,
                            "device": "metal",
                            "requestSHA256": request.sha256,
                            "rawSHA256": hashes,
                            "runtimeReceipts": provenance,
                            "deliveryComplete": False,
                        },
                    )
                    return partial
                require(
                    not verify_only and not work.exists(),
                    "Uncertain/failed window: reconcile; no retry",
                )
                work.mkdir(mode=0o700)
                durable(
                    work / "intent.json",
                    {
                        "requestSHA256": request.sha256,
                        "startSample": window.start_sample,
                        "sampleCount": window.sample_count,
                        "device": "metal",
                        "inputIdentities": identities,
                    },
                )
                try:
                    (runtime or MetalRuntime()).execute(request, window, work, identities)
                    _identities(identities)
                    durable(
                        terminal,
                        {
                            "version": 1,
                            "device": "metal",
                            "requestSHA256": request.sha256,
                            "startSample": window.start_sample,
                            "sampleCount": window.sample_count,
                            "evidence": _evidence(work),
                            "deliveryComplete": False,
                        },
                    )
                finally:
                    if not terminal.exists():
                        durable(
                            work / "failed.json",
                            {
                                "version": 1,
                                "device": "metal",
                                "effect": "uncertain",
                                "automaticRetryAllowed": False,
                                "requestSHA256": request.sha256,
                            },
                        )
                new_windows += 1
            saved = _read(terminal)
            require(
                saved
                == {
                    "version": 1,
                    "device": "metal",
                    "requestSHA256": request.sha256,
                    "startSample": window.start_sample,
                    "sampleCount": window.sample_count,
                    "evidence": _evidence(work),
                    "deliveryComplete": False,
                },
                "Window provenance changed",
            )
            raw = _read(work / "raw.json")
            raw_path = output / f"{window.start_sample:012d}.raw.json"
            clock_path = output / f"{window.start_sample:012d}.clock.json"
            if verify_only:
                require(_read(raw_path) == raw, "Raw differs")
            else:
                _same(raw_path, raw)
            digest = str(pin(raw_path)["sha256"])
            if verify_only:
                require(_read(clock_path) == _clock(window, digest), "Clock changed")
            else:
                _same(clock_path, _clock(window, digest))
            hashes.append(digest)
            provenance.append(pin(terminal))
        verify_inputs(request)
        complete = {
            "rawSHA256": hashes,
            "windows": len(windows),
            "deliveryComplete": False,
            "device": "metal",
            "runtimeReceipts": provenance,
        }
        if verify_only:
            require(_read(output / "complete.json") == complete, "Completion mismatch")
        else:
            _same(output / "complete.json", complete)
        return output / "complete.json"
    finally:
        os.close(lock)


def sample_family(pid: int, birth: str, host: SystemHost) -> dict[str, object]:
    require(host.birth(pid) == birth, "Child PID reused")
    result = subprocess.run(
        ["/bin/ps", "-axo", "pid=,ppid=,rss="],
        check=True,
        capture_output=True,
        text=True,
        timeout=2,
    )
    processes = parse_processes(result.stdout)
    members = _descendants(pid, processes)
    births: dict[int, str] = {}
    physical = 0
    exited: list[int] = []
    for member in members:
        try:
            observed = host.birth(member)
            amount = host.footprint(member)
            require(host.birth(member) == observed, "Family changed")
        except ReplayError:
            # A vanished descendant is an observed race, not a zero-byte sample.
            if member == pid or not host.absent(member):
                raise
            exited.append(member)
            continue
        births[member] = observed
        physical += amount
    return {
        "observedAtNs": time.time_ns(),
        "members": [{"pid": member, "birth": value} for member, value in births.items()],
        "exitedDuringObservation": exited,
        "rssBytes": sum(row.rss_bytes for row in processes if row.pid in births),
        "physicalBytes": physical,
    }


def _cleanup_child(
    child: subprocess.Popen[bytes], birth: str | None, host: SystemHost
) -> dict[str, object]:
    """Never hide the initiating failure; never signal an unverified/reaped PID."""
    result: dict[str, object] = {"signalSent": False, "terminalObserved": False}
    try:
        if child.poll() is None:
            require(birth is not None and host.birth(child.pid) == birth, "Cleanup owner changed")
            require(
                os.getpgid(child.pid) == child.pid and os.getsid(child.pid) == child.pid,
                "Cleanup process group/session changed",
            )
            # No intervening wait/poll reaps this owned child before killpg.
            os.killpg(child.pid, signal.SIGKILL)
            result["signalSent"] = True
        child.wait(timeout=5)
        result["terminalObserved"] = True
    except (OSError, ReplayError, subprocess.TimeoutExpired) as error:
        result["errorType"] = type(error).__name__
        result["error"] = str(error)
    return result


class MetalRuntime:
    """One owned process group; sampled family observations are not absolute peaks."""

    def execute(
        self, request: Request, window: AudioWindow, work: Path, identities: dict[str, object]
    ) -> None:
        root = Path.cwd().resolve()
        env = os.environ | planned_environment(work, root)
        for name in ("scratch", "cache", "numba", "huggingface"):
            (work / name).mkdir(mode=0o700)
        profile = work / "sandbox.sb"
        with profile.open("x", encoding="utf-8") as stream:
            stream.write(sandbox_profile(root))
        child_request = work / "child.json"
        durable(
            child_request,
            {
                "request": str(request.path),
                "requestSHA256": request.sha256,
                "startSample": window.start_sample,
                "sampleCount": window.sample_count,
                "identities": identities,
            },
        )
        command = [
            "/usr/bin/sandbox-exec",
            "-f",
            str(profile),
            sys.executable,
            "-B",
            "-m",
            "replay.isolated_metal_asr",
            str(child_request),
            str(work),
            "--worker",
        ]
        samples: list[dict[str, object]] = []
        started = time.monotonic()
        family_gone = False
        birth: str | None = None
        host = SystemHost()
        sampling_errors: list[dict[str, object]] = []
        with (work / "worker.log").open("xb") as log:
            child = subprocess.Popen(
                command,
                stdout=log,
                stderr=subprocess.STDOUT,
                cwd=root,
                env=env,
                start_new_session=True,
            )
            try:
                birth = host.birth(child.pid)
                durable(
                    work / "launch.json",
                    {
                        "pid": child.pid,
                        "birth": birth,
                        "command": command,
                        "startedAtNs": time.time_ns(),
                        "timeoutSeconds": request.timeout,
                    },
                )
                while child.poll() is None:
                    require(time.monotonic() - started < request.timeout, "Metal window timed out")
                    require(
                        (work / "worker.log").stat().st_size <= 8 * 1024**2, "Worker log overflow"
                    )
                    try:
                        samples.append(sample_family(child.pid, birth, host))
                    except ReplayError as error:
                        sampling_errors.append(
                            {"observedAtNs": time.time_ns(), "error": str(error)}
                        )
                        # A sampling race is reconciled only with this Popen's real
                        # terminal success and previously measured family evidence.
                        terminal = child.poll()
                        if terminal is None:
                            # Kernel identity can disappear just before waitpid
                            # publishes the genuine exit status to this parent.
                            try:
                                terminal = child.wait(timeout=1)
                            except subprocess.TimeoutExpired:
                                terminal = None
                        sampling_errors[-1]["terminalReconciliationExitCode"] = terminal
                        if terminal != 0:
                            raise
                        break
                    with suppress(subprocess.TimeoutExpired):
                        child.wait(timeout=1)
                require(child.returncode == 0 and bool(samples), "Metal child failed or unobserved")
                last_members: list[object] = []
                for sample in samples:
                    members = sample["members"]
                    if not isinstance(members, list):
                        raise ReplayError("Missing family members")
                    last_members.extend(members)
                family_gone = all(
                    host.absent(int(str(record(member)["pid"]))) for member in last_members
                )
                require(family_gone, "Owned child descendants remain")
            finally:
                cleanup = _cleanup_child(child, birth, host)
                durable(
                    work / "resources.json",
                    {
                        "version": 1,
                        "exitCode": child.returncode,
                        "familyGone": family_gone,
                        "wallSeconds": time.monotonic() - started,
                        "samples": samples,
                        "samplingErrors": sampling_errors,
                        "cleanup": cleanup,
                        "peakRSSBytesObserved": max(
                            (int(str(row["rssBytes"])) for row in samples), default=0
                        ),
                        "peakPhysicalBytesObserved": max(
                            (int(str(row["physicalBytes"])) for row in samples), default=0
                        ),
                        "samplingIntervalSeconds": 1,
                        "absoluteFamilyPeakVerified": False,
                        "compilerServiceWritesVerified": False,
                        "transientWritesVerified": False,
                    },
                )


def worker(child_path: Path, output: Path) -> None:
    child = _read(child_path)
    request = load_request(canonical(child.get("request")), verify=False)
    require(request.sha256 == child.get("requestSHA256"), "Child request changed")
    identities = record(child.get("identities"))
    require(set(identities) == {str(path) for path in request.pins}, "Incomplete child identities")
    _identities(identities)
    window = AudioWindow(
        _integer(child.get("startSample"), 0, request.samples - 1),
        _integer(child.get("sampleCount"), 1, 60 * RATE),
    )
    require(
        window
        in plan_audio_windows(request.samples, window_samples=60 * RATE, stride_samples=30 * RATE),
        "Noncanonical ASR window",
    )
    expected = planned_environment(output, Path.cwd().resolve())
    require(
        all(os.environ.get(key) == value for key, value in expected.items()),
        "Unconfined environment",
    )
    require(
        all(
            Path(expected[key]).is_dir()
            for key in ("TMPDIR", "XDG_CACHE_HOME", "NUMBA_CACHE_DIR", "HF_HOME")
        ),
        "Missing scratch",
    )
    with wave.open(str(request.pcm), "rb") as audio:
        audio.setpos(window.start_sample)
        payload = audio.readframes(window.sample_count)
    require(len(payload) == 2 * window.sample_count, "Truncated PCM window")
    samples = np.frombuffer(payload, dtype="<i2").astype(np.float32) / np.float32(32768)
    audit_descriptors(output, PHASES[0])
    mx = importlib.import_module("mlx.core")
    audit_descriptors(output, PHASES[1])
    mx.reset_peak_memory()
    context: object = mx.stream(mx.gpu)
    if not isinstance(context, AbstractContextManager):
        raise ReplayError("Missing Metal stream")
    with context:
        provider = importlib.import_module("mlx_whisper")
        audit_descriptors(output, PHASES[2])
        raw: object = provider.transcribe(
            samples,
            path_or_hf_repo=str(request.weights.parent),
            language="ja",
            verbose=False,
            condition_on_previous_text=False,
            word_timestamps=True,
            temperature=0.0,
            fp16=True,
        )
        mx.synchronize(mx.gpu)
        memory = {
            "version": 1,
            "device": "metal",
            "activeBytes": mx.get_active_memory(),
            "peakBytes": mx.get_peak_memory(),
            "cacheBytes": mx.get_cache_memory(),
            "peakReset": "before-provider-import",
            "physicalFootprint": False,
        }
        audit_descriptors(output, PHASES[3])
    _identities(identities)
    durable(output / "mlx-memory.json", memory)
    durable(output / "raw.json", record(raw))


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("request", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--maximum-new-windows", type=int)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--verify", action="store_true")
    mode.add_argument("--worker", action="store_true")
    args = parser.parse_args(argv)
    if args.worker:
        require(args.maximum_new_windows is None, "Worker cannot alter collection window limit")
        worker(args.request, args.output)
    else:
        collect(
            args.request,
            args.output,
            verify_only=args.verify,
            maximum_new_windows=args.maximum_new_windows,
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
