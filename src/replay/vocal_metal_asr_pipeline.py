"""Finite original-clock whole-vocals to normal Metal recognition workflows.

Planning reads only metadata. The original separation interpreter verifies its
whole result; mono decoding uses the existing continuous command. The genuine
Metal producer owns its raw windows and completion. No CPU policy is fabricated.
"""

from __future__ import annotations

import argparse
import hashlib
import os
import subprocess
import sys
import wave
from collections.abc import Callable
from dataclasses import asdict, dataclass
from fractions import Fraction
from pathlib import Path

from replay import asr_assembly, workflow
from replay import isolated_metal_asr as metal
from replay.audio_windows import plan_audio_windows
from replay.code_dependency_inventory import capture
from replay.core import ReplayError
from replay.local_asr import DEFAULT_CONFIG_SHA256, DEFAULT_WEIGHTS_SHA256
from replay.native_invocation import canonical, identity, pin, require
from replay.recordings import record
from replay.replay_transcription_plan import mono_command
from replay.sequence_journal import bounded_read, decode, encode, publish
from replay.stages import execution_lock

ROOT = Path(__file__).resolve().parents[2]
PROTOCOL = "vocal-metal-asr-pipeline-v1"
RATE = 16000
SOURCE_RATE = 44100
MAX_SECONDS = 21600
Executor = Callable[[tuple[str, ...], Path, int, dict[str, str]], None]


@dataclass(frozen=True)
class Ref:
    path: Path
    sha256: str

    def data(self) -> dict[str, str]:
        return {"path": str(self.path), "sha256": self.sha256}


@dataclass(frozen=True)
class Job:
    operation: Path
    config: Ref
    root: Path
    plan: Ref
    source: Path
    separation: Path
    python: Ref
    invocation: Path
    ffmpeg: Ref
    model_config: Ref
    weights: Ref
    home: Path
    hub: Path
    code: dict[str, str]
    metal_code: dict[str, str]
    timeouts: dict[str, int]
    trial: int


def _path(value: object, *, model: bool = False) -> Path:
    result = canonical(value)
    require(
        result.is_relative_to(ROOT / ("edits/models" if model else "edits/work")),
        "Expected local purpose path",
    )
    return result


def _read(path: Path) -> dict[str, object]:
    return decode(bounded_read(path))


def _sha(value: object) -> str:
    require(
        isinstance(value, str) and len(value) == 64 and all(c in "0123456789abcdef" for c in value),
        "Invalid SHA256",
    )
    return str(value)


def _ref(value: object, *, model: bool = False, executable: bool = False) -> Ref:
    raw = record(value)
    require(set(raw) == {"path", "sha256"}, "Exact input reference required")
    path = canonical(raw["path"]) if executable else _path(raw["path"], model=model)
    require(
        not executable or path.is_file() and os.access(path, os.X_OK),
        "Expected installed executable",
    )
    return Ref(path, _sha(raw["sha256"]))


def _metadata_pin(path: Path) -> dict[str, str]:
    return {"path": str(path), "sha256": hashlib.sha256(bounded_read(path)).hexdigest()}


def _code() -> dict[str, str]:
    return {str(item.path): item.sha256 for item in capture(("vocal_metal_asr_pipeline",))}


def _positive(value: object, high: int) -> int:
    require(type(value) is int and 1 <= value <= high, "Invalid finite resource bound")
    if not isinstance(value, int):
        raise ReplayError("Missing integer")
    return value


def _original_inputs(plan: Path) -> tuple[str, ...]:
    """Declare original whole-verifier inputs without reading model/media bytes."""
    data = _read(plan)
    result: dict[str, str] = {}
    for name in ("codeInputs", "providerInputs", "modelInputs"):
        rows = data.get(name)
        require(isinstance(rows, list) and 1 <= len(rows) <= 1024, "Missing original input pins")
        if not isinstance(rows, list):
            raise ReplayError("Missing original input list")
        for item in rows:
            raw = record(item)
            require(set(raw) == {"path", "sha256"}, "Exact original input reference required")
            path, digest = canonical(raw["path"]), _sha(raw["sha256"])
            require(
                path.is_relative_to(ROOT) and str(path) not in result, "Invalid original pin role"
            )
            identity(path)
            result[str(path)] = digest
    require(len(result) <= 1024, "Original verifier input budget exceeded")
    return tuple(result)


def _definition(config: Path) -> tuple[dict[str, object], Ref, Path, Path]:
    raw = _read(config)
    require(
        set(raw)
        == {
            "version",
            "separationPlan",
            "separationPython",
            "separationInvocation",
            "ffmpeg",
            "model",
            "separationCache",
            "metalCodeInputs",
            "timeouts",
            "trialWindows",
        }
        and type(raw["version"]) is int
        and raw["version"] == 1,
        "Invalid vocal Metal pipeline configuration",
    )
    plan = _ref(raw["separationPlan"])
    require(_metadata_pin(plan.path) == plan.data(), "Separation plan changed")
    original = _read(plan.path)
    require(
        original.get("kind") == "separation-window-pipeline"
        and original.get("sampleRate") == SOURCE_RATE
        and original.get("qualityApproved") is False
        and original.get("productionBackendAdopted") is False,
        "Expected original separation pipeline",
    )
    source = _path(record(original.get("source"))["path"])
    _sha(record(original["source"])["sha256"])
    directory = _path(original.get("outputDirectory"))
    _original_inputs(plan.path)
    return raw, plan, source, directory


def _load(operation: Path) -> Job:
    op = _read(_path(str(operation)))
    require(
        set(op) == {"version", "protocol", "config", "workRoot", "readerCode"}
        and type(op["version"]) is int
        and op["version"] == 1
        and op["protocol"] == PROTOCOL
        and op["readerCode"] == _code(),
        "Operation or reader code changed",
    )
    config = _ref(op["config"])
    require(_metadata_pin(config.path) == config.data(), "Pipeline configuration changed")
    raw, plan, source, separation = _definition(config.path)
    root = _path(op["workRoot"])
    require(
        not root.is_relative_to(separation) and not separation.is_relative_to(root),
        "Independent recognition namespace required",
    )
    python, ffmpeg = (
        _ref(raw["separationPython"], executable=True),
        _ref(raw["ffmpeg"], executable=True),
    )
    invocation = Path(str(raw["separationInvocation"]))
    require(
        invocation.is_absolute()
        and invocation.is_relative_to(ROOT)
        and invocation.parent.name == "bin"
        and invocation.resolve(strict=True) == python.path
        and (invocation.parent.parent / "pyvenv.cfg").is_file(),
        "Expected original repository interpreter alias",
    )
    model = record(raw["model"])
    require(set(model) == {"config", "weights"}, "Exact large-v3 model inputs required")
    model_config, weights = _ref(model["config"], model=True), _ref(model["weights"], model=True)
    require(
        model_config.path.name == "config.json"
        and weights.path.name in {"weights.npz", "weights.safetensors"}
        and weights.path.parent == model_config.path.parent
        and model_config.sha256 == DEFAULT_CONFIG_SHA256
        and weights.sha256 == DEFAULT_WEIGHTS_SHA256,
        "Only original pinned large-v3 model is supported",
    )
    cache = record(raw["separationCache"])
    require(set(cache) == {"HF_HOME", "HF_HUB_CACHE"}, "Original local separation cache required")
    home, hub = _path(cache["HF_HOME"], model=True), _path(cache["HF_HUB_CACHE"], model=True)
    require(home.is_dir() and hub.is_dir(), "Missing original offline cache")
    code = {
        str(canonical(name)): _sha(value) for name, value in record(raw["metalCodeInputs"]).items()
    }
    require(
        1 <= len(code) <= 1024
        and {str(item) for item in metal.required_code()} <= code.keys()
        and all(Path(name).is_relative_to(ROOT) for name in code),
        "Missing local producer/provider pins",
    )
    timeouts = record(raw["timeouts"])
    require(
        set(timeouts) == {"whole", "mono", "window", "collection"}, "Exact phase deadlines required"
    )
    return Job(
        operation,
        config,
        root,
        plan,
        source,
        separation,
        python,
        invocation,
        ffmpeg,
        model_config,
        weights,
        home,
        hub,
        {name: _sha(sha) for name, sha in record(op["readerCode"]).items()},
        code,
        {
            name: _positive(value, 3600 if name == "window" else 86400)
            for name, value in timeouts.items()
        },
        _positive(raw["trialWindows"], 720),
    )


def _environment(job: Job, *, original: bool = False) -> dict[str, str]:
    result = os.environ.copy()
    result.pop("MLX_DISABLE_COMPILE", None)
    result.update(
        {
            "TMPDIR": str(job.root / "tmp"),
            "TMP": str(job.root / "tmp"),
            "TEMP": str(job.root / "tmp"),
            "XDG_CACHE_HOME": str(job.root / "cache"),
            "HF_HOME": str(job.home),
            "HF_HUB_CACHE": str(job.hub),
            "HF_HUB_OFFLINE": "1",
            "TRANSFORMERS_OFFLINE": "1",
            "HF_DATASETS_OFFLINE": "1",
            "PYTHONDONTWRITEBYTECODE": "1",
            "PYTHONPATH": str(ROOT / "src"),
        }
    )
    if original:
        saved = record(_read(job.separation / "input.json").get("environment"))
        require(
            set(saved) == {"TMPDIR", "TMP", "TEMP", "XDG_CACHE_HOME", "HF_HOME", "HF_HUB_CACHE"},
            "Missing original separation environment",
        )
        paths = {name: canonical(value) for name, value in saved.items()}
        require(
            all(path.is_relative_to(ROOT) and path.is_dir() for path in paths.values())
            and paths["HF_HOME"] == job.home
            and paths["HF_HUB_CACHE"] == job.hub,
            "Original separation environment changed",
        )
        result.update({name: str(path) for name, path in paths.items()})
    return result


def _execute(
    command: tuple[str, ...], log: Path, timeout: int, environment: dict[str, str]
) -> None:
    with log.open("xb") as stream:
        os.fchmod(stream.fileno(), 0o600)
        subprocess.run(
            command,
            stdin=subprocess.DEVNULL,
            stdout=stream,
            stderr=subprocess.STDOUT,
            timeout=timeout,
            check=True,
            env=environment,
            cwd=ROOT,
        )


def _pcm(path: Path, *, rate: int, channels: int) -> int:
    before = identity(path)
    with wave.open(str(path), "rb") as audio:
        require(
            (audio.getframerate(), audio.getnchannels(), audio.getsampwidth(), audio.getcomptype())
            == (rate, channels, 2, "NONE"),
            "Wrong continuous PCM contract",
        )
        count = audio.getnframes()
        require(0 < count <= MAX_SECONDS * rate, "PCM outside six-hour budget")
        decoded = 0
        while block := audio.readframes(65536):
            decoded += len(block)
        require(decoded == count * channels * 2, "Truncated PCM")
    require(identity(path) == before, "PCM changed during clock verification")
    return count


def _whole(job: Job) -> dict[str, object]:
    original, result = _read(job.plan.path), _read(job.separation / "result.json")
    vocals = job.separation / "vocals.wav"
    samples = _pcm(vocals, rate=SOURCE_RATE, channels=2)
    require(
        result.get("inputSampleCount") == result.get("outputSampleCount") == samples
        and result.get("sampleRate") == SOURCE_RATE
        and result.get("source") == str(job.source)
        and result.get("vocals") == str(vocals)
        and result.get("timestampCorrection") is False
        and result.get("deliveryComplete") is False
        and result.get("qualityApproved") is False,
        "Original whole separation clock/receipt differs",
    )
    require(
        pin(job.source)["sha256"] == record(original["source"])["sha256"]
        and _pcm(job.source, rate=SOURCE_RATE, channels=2) == samples,
        "Whole source binding or clock changed",
    )
    return {
        "version": 1,
        "protocol": PROTOCOL,
        "operation": _metadata_pin(job.operation),
        "plan": _metadata_pin(job.plan.path),
        "result": pin(job.separation / "result.json"),
        "originalContext": _metadata_pin(job.separation / "input.json"),
        "source": pin(job.source),
        "vocals": pin(vocals),
        "samples": samples,
        "sampleRate": SOURCE_RATE,
        "originalWholeVerifier": [
            str(job.invocation),
            "-B",
            "-m",
            "replay.separation_window_pipeline",
            str(job.plan.path),
            "--verify",
        ],
        "originalWholeDomainVerified": True,
        "qualityApproved": False,
        "deliveryComplete": False,
    }


def _verify_whole(job: Job) -> None:
    require(_read(job.root / "whole.json") == _whole(job), "Whole verification checkpoint changed")


def _mono(job: Job) -> dict[str, object]:
    _verify_whole(job)
    vocals, mono = job.separation / "vocals.wav", job.root / "mono.wav"
    samples, count = _pcm(vocals, rate=SOURCE_RATE, channels=2), _pcm(mono, rate=RATE, channels=1)
    require(
        abs(Fraction(samples, SOURCE_RATE) - Fraction(count, RATE)) <= Fraction(1, RATE),
        "Mono changed the continuous sample duration",
    )
    require(pin(job.ffmpeg.path)["sha256"] == job.ffmpeg.sha256, "Mono decoder changed")
    return {
        "version": 1,
        "protocol": PROTOCOL,
        "whole": _metadata_pin(job.root / "whole.json"),
        "vocals": pin(vocals),
        "mono": pin(mono),
        "inputSamples": samples,
        "inputSampleRate": SOURCE_RATE,
        "samples": count,
        "sampleRate": RATE,
        "command": list(mono_command(vocals, mono, job.ffmpeg.path)),
        "decoder": job.ffmpeg.data(),
        "clock": "input-pcm-not-projected",
        "qualityApproved": False,
        "deliveryComplete": False,
    }


def _verify_mono(job: Job) -> None:
    require(_read(job.root / "mono.json") == _mono(job), "Mono producer or sample clock changed")


def _metal_request(job: Job) -> dict[str, object]:
    _verify_mono(job)
    return {
        "version": 1,
        "device": "metal",
        "fp16": True,
        "cpuAdoption": None,
        "inputs": {
            "pcm": pin(job.root / "mono.wav"),
            "config": job.model_config.data(),
            "weights": job.weights.data(),
        },
        "samples": _pcm(job.root / "mono.wav", rate=RATE, channels=1),
        "timeoutSeconds": job.timeouts["window"],
        "codeInputs": job.metal_code,
    }


def _verify_request(job: Job) -> None:
    require(
        _read(job.root / "metal-request.json") == _metal_request(job),
        "Dynamic Metal request changed",
    )
    metal.load_request(job.root / "metal-request.json", verify=False)


def _prefix(job: Job, result: Path) -> None:
    """Reuse pinned read-only Metal window contracts for a bounded partial trial."""
    request = metal.load_request(job.root / "metal-request.json")
    collection = job.root / "asr"
    with execution_lock(collection, ".run") as acquired:
        require(acquired, "Recognition collection still owned")
        manifest = _read(collection / "inputs.json")
        require(
            manifest.get("inputs")
            == {
                str(request.path): request.sha256,
                **{
                    str(path): request.pins[path]
                    for path in (request.pcm, request.weights, request.config)
                },
            }
            and manifest.get("samples") == request.samples
            and manifest.get("sampleRate") == RATE
            and manifest.get("windowSamples") == 60 * RATE
            and manifest.get("strideSamples") == 30 * RATE
            and manifest.get("clock") == "input-pcm-not-projected"
            and manifest.get("device") == "metal",
            "Trial collection manifest changed",
        )
        saved = _read(result)
        hashes, receipts = saved.get("rawSHA256"), saved.get("runtimeReceipts")
        require(
            isinstance(hashes, list)
            and isinstance(receipts, list)
            and 1 <= len(hashes) == len(receipts),
            "Missing genuine Metal trial windows",
        )
        if not isinstance(hashes, list) or not isinstance(receipts, list):
            raise ReplayError("Missing trial lists")
        windows = plan_audio_windows(
            request.samples, window_samples=60 * RATE, stride_samples=30 * RATE
        )
        require(
            len(hashes) <= len(windows)
            and saved.get("device") == "metal"
            and saved.get("deliveryComplete") is False,
            "Invalid trial completion",
        )
        if result.name != "complete.json":
            require(
                saved.get("collectionComplete") is False
                and saved.get("requestSHA256") == request.sha256
                and result.name
                == f"partial-through-{windows[len(hashes) - 1].start_sample:012d}.json",
                "Trial belongs to another request",
            )
        for index, window in enumerate(windows[: len(hashes)]):
            work = collection / f"{window.start_sample:012d}.metal-runtime"
            terminal = work / "complete.json"
            require(
                _read(terminal)
                == {
                    "version": 1,
                    "device": "metal",
                    "requestSHA256": request.sha256,
                    "startSample": window.start_sample,
                    "sampleCount": window.sample_count,
                    "evidence": metal._evidence(work),
                    "deliveryComplete": False,
                }
                and pin(terminal) == receipts[index],
                "Trial runtime provenance changed",
            )
            raw, clock = (
                collection / f"{window.start_sample:012d}.raw.json",
                collection / f"{window.start_sample:012d}.clock.json",
            )
            require(
                pin(raw)["sha256"] == hashes[index]
                and _read(raw) == _read(work / "raw.json")
                and _read(clock) == metal._clock(window, str(hashes[index])),
                "Trial raw clock changed",
            )
        metal.verify_inputs(request)


def _recognized(job: Job, *, trial: bool) -> dict[str, object]:
    _verify_request(job)
    if trial:
        saved = _read(job.root / "trial.json")
        result = _path(record(saved["result"])["path"])
        require(result.parent == job.root / "asr", "Trial result escapes collection")
        _prefix(job, result)
    else:
        result = metal.collect(job.root / "metal-request.json", job.root / "asr", verify_only=True)
    return {
        "version": 1,
        "protocol": PROTOCOL,
        "request": _metadata_pin(job.root / "metal-request.json"),
        "result": pin(result),
        "trial": trial,
        "fullRecognitionVerified": not trial,
        "clock": "input-pcm-not-projected",
        "qualityApproved": False,
        "deliveryComplete": False,
    }


def _assembled(job: Job) -> dict[str, object]:
    require(
        _read(job.root / "recognition.json") == _recognized(job, trial=False),
        "Full Metal recognition changed",
    )
    labels = _read(job.root / "labels.json")
    collection = job.root / "asr"
    manifest = _read(collection / "inputs.json")
    windows = asr_assembly._windows(manifest)
    native: list[dict[str, object]] = []
    for index, window in enumerate(windows):
        left = (windows[index - 1].end_sample + window.start_sample) / (2 * RATE) if index else 0.0
        right = (
            (window.end_sample + windows[index + 1].start_sample) / (2 * RATE)
            if index + 1 < len(windows)
            else window.end_sample / RATE
        )
        native.extend(
            asdict(item)
            for item in asr_assembly._labels(
                _read(collection / f"{window.start_sample:012d}.raw.json"),
                window=window,
                owner_start=left,
                owner_end=right,
            )
        )
    require(
        labels.get("labels") == native
        and labels.get("clock") == "input-pcm-not-projected"
        and labels.get("speechAccuracyVerified") is False
        and labels.get("humanReviewed") is False
        and labels.get("deliveryComplete") is False
        and labels.get("assemblySHA256") == pin(Path(asr_assembly.__file__).resolve())["sha256"],
        "Normal assembled labels changed",
    )
    for name, digest in record(labels.get("parentSHA256")).items():
        require(pin(canonical(name))["sha256"] == digest, "Assembly parent changed")
    return {
        "version": 1,
        "protocol": PROTOCOL,
        "recognition": _metadata_pin(job.root / "recognition.json"),
        "labels": pin(job.root / "labels.json"),
        "groups": len(native),
        "clock": "input-pcm-not-projected",
        "normalAssemblyConnected": True,
        "qualityApproved": False,
        "deliveryComplete": False,
    }


def run(
    operation: Path,
    *,
    phase: str,
    execute: Executor = _execute,
    runtime: metal.Runtime | None = None,
) -> Path:
    """Run one declared phase or its read-only verifier; never stop another backend."""
    job = _load(operation)
    if phase.startswith("verify-"):
        name = phase.removeprefix("verify-")
        require(
            name in {"whole", "mono", "request", "trial", "recognize", "assemble"},
            "Unknown verification phase",
        )
        output = (
            job.root
            / {
                "whole": "whole.json",
                "mono": "mono.json",
                "request": "metal-request.json",
                "trial": "trial.json",
                "recognize": "recognition.json",
                "assemble": "assembly.json",
            }[name]
        )
        if name == "whole":
            _verify_whole(job)
        elif name == "mono":
            _verify_mono(job)
        elif name == "request":
            _verify_request(job)
        else:
            expected = (
                _assembled(job) if name == "assemble" else _recognized(job, trial=name == "trial")
            )
            require(_read(output) == expected, "Phase checkpoint changed")
        return output
    if phase == "whole":
        require(
            pin(job.python.path)["sha256"] == job.python.sha256,
            "Original separation interpreter changed",
        )
        execute(
            (
                str(job.invocation),
                "-B",
                "-m",
                "replay.separation_window_pipeline",
                str(job.plan.path),
                "--verify",
            ),
            job.root / "whole-verifier.log",
            job.timeouts["whole"],
            _environment(job, original=True),
        )
        output, data = job.root / "whole.json", _whole(job)
    elif phase == "mono":
        _verify_whole(job)
        require(
            not (job.root / "mono.wav").exists()
            and not (job.root / "mono.wav").is_symlink()
            and pin(job.ffmpeg.path)["sha256"] == job.ffmpeg.sha256,
            "Occupied/uncertain mono or changed decoder",
        )
        execute(
            mono_command(job.separation / "vocals.wav", job.root / "mono.wav", job.ffmpeg.path),
            job.root / "mono.log",
            job.timeouts["mono"],
            _environment(job),
        )
        output, data = job.root / "mono.json", _mono(job)
    elif phase == "request":
        output, data = job.root / "metal-request.json", _metal_request(job)
    elif phase in {"trial", "recognize"}:
        _verify_request(job)
        result = metal.collect(
            job.root / "metal-request.json",
            job.root / "asr",
            runtime=runtime,
            maximum_new_windows=job.trial if phase == "trial" else None,
        )
        output = job.root / ("trial.json" if phase == "trial" else "recognition.json")
        data = {
            "version": 1,
            "protocol": PROTOCOL,
            "request": _metadata_pin(job.root / "metal-request.json"),
            "result": pin(result),
            "trial": phase == "trial",
            "fullRecognitionVerified": phase != "trial",
            "clock": "input-pcm-not-projected",
            "qualityApproved": False,
            "deliveryComplete": False,
        }
    elif phase == "assemble":
        require(
            _read(job.root / "recognition.json") == _recognized(job, trial=False),
            "Full recognition prerequisite missing",
        )
        asr_assembly.assemble(job.root / "asr", job.root / "labels.json")
        output, data = job.root / "assembly.json", _assembled(job)
    else:
        raise ReplayError("Unknown vocal Metal phase")
    publish(output, encode(data))
    _load(operation)
    return output


def _command(phase: str, operation: Path) -> list[str]:
    return [sys.executable, "-B", "-m", "replay.vocal_metal_asr_pipeline", phase, str(operation)]


def prepare(config: Path, output: Path) -> Path:
    """Prepare a standard parent graph; full target excludes the optional trial."""
    config, output = _path(str(config)), _path(str(output))
    require(output.is_dir() and not tuple(output.iterdir()), "Existing empty purpose root required")
    _definition(config)
    operation = output / "operation.json"
    publish(
        operation,
        encode(
            {
                "version": 1,
                "protocol": PROTOCOL,
                "config": _metadata_pin(config),
                "workRoot": str(output),
                "readerCode": _code(),
            }
        ),
    )
    job = _load(operation)
    (output / "tmp").mkdir(mode=0o700)
    (output / "cache").mkdir(mode=0o700)
    common = list(
        dict.fromkeys(
            [
                str(operation),
                str(config),
                str(job.plan.path),
                *job.code,
                *job.metal_code,
                str(job.python.path),
                str(job.ffmpeg.path),
                str(job.invocation.parent.parent / "pyvenv.cfg"),
            ]
        )
    )
    whole, mono, request = (
        str(output / "whole.json"),
        str(output / "mono.json"),
        str(output / "metal-request.json"),
    )
    phases = (
        (
            "whole",
            list(
                dict.fromkeys(
                    common
                    + [
                        str(job.source),
                        str(job.separation / "input.json"),
                        str(job.separation / "result.json"),
                        str(job.separation / "vocals.wav"),
                        *_original_inputs(job.plan.path),
                    ]
                )
            ),
            [whole],
            job.timeouts["whole"],
        ),
        ("mono", common + [whole], [str(output / "mono.wav"), mono], job.timeouts["mono"]),
        ("request", common + [mono, str(output / "mono.wav")], [request], job.timeouts["mono"]),
        (
            "trial",
            common
            + [
                mono,
                request,
                str(output / "mono.wav"),
                str(job.model_config.path),
                str(job.weights.path),
            ],
            [str(output / "trial.json")],
            job.timeouts["collection"],
        ),
        (
            "recognize",
            common
            + [
                mono,
                request,
                str(output / "mono.wav"),
                str(job.model_config.path),
                str(job.weights.path),
            ],
            [str(output / "asr/complete.json"), str(output / "recognition.json")],
            job.timeouts["collection"],
        ),
        (
            "assemble",
            common + [request, str(output / "recognition.json"), str(output / "asr/complete.json")],
            [str(output / "labels.json"), str(output / "assembly.json")],
            job.timeouts["mono"],
        ),
    )
    graph = {
        "version": 1,
        "stages": [
            {
                "name": phase,
                "command": _command(phase, operation),
                "inputs": inputs,
                "outputs": outputs,
                "timeout_seconds": seconds,
                "verify_command": _command("verify-" + phase, operation),
            }
            for phase, inputs, outputs, seconds in phases
        ],
    }
    workflow.stage_graph(workflow.parse_stages(graph))
    path = output / "workflow.json"
    publish(path, encode(graph))
    return path


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "phase",
        choices=(
            "prepare",
            "whole",
            "mono",
            "request",
            "trial",
            "recognize",
            "assemble",
            "verify-whole",
            "verify-mono",
            "verify-request",
            "verify-trial",
            "verify-recognize",
            "verify-assemble",
        ),
    )
    parser.add_argument("input", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args(argv)
    if args.phase == "prepare":
        require(args.output is not None, "Preparation needs an empty purpose root")
        if args.output is None:
            raise ReplayError("Missing preparation output")
        print(prepare(args.input, args.output))
    else:
        require(args.output is None, "Runtime uses its immutable configured output")
        print(run(args.input, phase=args.phase))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
