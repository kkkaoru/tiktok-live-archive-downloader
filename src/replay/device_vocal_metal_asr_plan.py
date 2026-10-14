"""Continue a genuine mixed-device whole through the existing Metal ASR producer.

The configuration declares a future whole Stage path, never a guessed output
digest. Preparation resolves it only after genuine completion. CPU verification
is not substituted for the explicit mixed-device verifier.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import stat
import subprocess
import sys
from collections.abc import Callable, Iterator
from dataclasses import asdict, dataclass
from fractions import Fraction
from pathlib import Path

from replay import asr_assembly, caption_io
from replay import completion_barrier as barrier
from replay import isolated_metal_asr as metal
from replay import measured_job_admission as admission
from replay import replay_transcription_plan as base
from replay.code_dependency_inventory import capture
from replay.core import ReplayError
from replay.input_hash_snapshot import InputHashSnapshot
from replay.native_invocation import canonical, pin, require, rows
from replay.recordings import record
from replay.replay_preparation_plan import Pin, _pin, _timeout
from replay.sequence_journal import (
    MAX_BYTES,
    bounded_read,
    encode,
    invalid_constant,
    publish,
    unique_pairs,
)
from replay.sequence_journal import identity as stat_identity
from replay.stages import Stage, StageError, _read_receipt
from replay.workflow import MAX_PLAN_BYTES, parse_stages, stage_graph

ROOT = Path(__file__).resolve().parents[2]
PROTOCOL = "device-vocal-metal-asr-plan-v1"
PREREQUISITE_PROTOCOL = "resource-only-prerequisite-completion-v1"
MAX_PREREQUISITE_WAIT_SECONDS = 8 * 3600
HASH_CHUNK_BYTES = 65536
PHASES = ("whole", "mono", "request", "recognize", "assemble")
Executor = Callable[[tuple[str, ...], Path, int, dict[str, str]], None]


@dataclass(frozen=True)
class Job:
    root: Path
    operation: Path
    settings: base.Settings
    ready: base.Ready
    inputs: tuple[Pin, ...]
    device_config: Pin
    whole_command: tuple[str, ...]
    metal_code: dict[str, str]
    deadlines: dict[str, int]


@dataclass(frozen=True)
class Prerequisite:
    config: Pin
    workflow: Pin
    capture: Pin
    state: Path
    status: Path
    stages: tuple[Stage, ...]
    code: tuple[Pin, ...]
    reader_code: tuple[Pin, ...]
    wait_seconds: int
    poll_seconds: int
    output: Path


def _local(value: object) -> Path:
    path = canonical(value)
    require(path.is_relative_to(ROOT / "edits/work"), "Expected checkout purpose path")
    return path


def _chunks(path: Path, *, budget: int) -> Iterator[bytes]:
    """Read bounded bytes on one unchanged FD; clocks and labels have typed limits."""
    canonical(str(path))
    require(0 < budget <= base.MAX_JSON, "Invalid artifact byte budget")
    descriptor = os.open(path, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK)
    try:
        before = os.fstat(descriptor)
        require(
            stat.S_ISREG(before.st_mode) and before.st_nlink == 1 and 0 < before.st_size <= budget,
            "Expected nonempty bounded regular single-link artifact",
        )
        require(
            stat_identity(path.lstat()) == stat_identity(before), "Artifact changed before read"
        )
        total = 0
        while chunk := os.read(descriptor, min(HASH_CHUNK_BYTES, budget + 1 - total)):
            total += len(chunk)
            require(total <= budget, "Artifact grew beyond its byte budget")
            yield chunk
        require(
            total == before.st_size
            and stat_identity(os.fstat(descriptor)) == stat_identity(before)
            and stat_identity(path.lstat()) == stat_identity(before)
            and path == path.resolve(),
            "Artifact changed during read",
        )
    finally:
        os.close(descriptor)


def _read(path: Path, *, budget: int = MAX_BYTES) -> dict[str, object]:
    return record(
        json.loads(
            b"".join(_chunks(path, budget=budget)),
            object_pairs_hook=unique_pairs,
            parse_constant=invalid_constant,
        )
    )


def _ref(value: object) -> Pin:
    item = _pin(value)
    require(pin(item.path)["sha256"] == item.sha256, "Declared metadata input changed")
    return item


def _metadata(path: Path, *, budget: int = MAX_BYTES) -> dict[str, str]:
    digest = hashlib.sha256()
    for chunk in _chunks(path, budget=budget):
        digest.update(chunk)
    return {"path": str(path), "sha256": digest.hexdigest()}


def _code() -> dict[str, str]:
    return {
        str(item.path): item.sha256
        for item in capture(("device_vocal_metal_asr_plan", "phase_timer"))
    }


def _config(path: Path) -> dict[str, object]:
    raw = _read(_local(str(path)))
    require(
        set(raw)
        == {"version", "deviceWhole", "transcriptionTemplate", "metalCodeInputs", "timeouts"}
        and type(raw["version"]) is int
        and raw["version"] == 1,
        "Invalid device vocal configuration",
    )
    return raw


def _whole_stage(raw: dict[str, object]) -> tuple[Pin, Pin, Path, Stage]:
    whole = record(raw["deviceWhole"])
    require(
        set(whole) == {"configuration", "workflow", "receiptPath", "stageName"},
        "Explicit future whole Stage required",
    )
    config, workflow = _ref(whole["configuration"]), _ref(whole["workflow"])
    receipt = _local(whole["receiptPath"])
    require(
        whole["stageName"] == "device-whole" and receipt.name == "device-whole.json",
        "Expected explicit device whole Stage",
    )
    stages = [item for item in parse_stages(_read(workflow.path)) if item.name == "device-whole"]
    require(len(stages) == 1, "Unique whole Stage required")
    stage = stages[0]
    require(
        len(stage.verify_command) == 6
        and stage.verify_command[1:5] == ("-B", "-m", "replay.separation_device_pipeline", "verify")
        and stage.verify_command[-1] == str(config.path),
        "Only the actual explicit device verifier is accepted",
    )
    device = _read(config.path)
    require(
        device.get("kind") == "explicit-device-separation-continuation-v1",
        "Wrong separation backend contract",
    )
    directory = _local(device["outputDirectory"])
    outputs = (directory / "vocals.wav", directory / "result.json")
    require(stage.outputs == outputs, "Whole output namespace differs")
    return config, workflow, receipt, stage


def _resolve(raw: dict[str, object]) -> tuple[dict[str, object], Pin, tuple[str, ...]]:
    config, workflow, receipt, stage = _whole_stage(raw)
    saved = record(_read_receipt(receipt))
    require(saved.get("status") == "complete", "Device whole barrier not complete")
    device = _read(config.path)
    template = record(raw["transcriptionTemplate"])
    predecessor = record(template.get("predecessor"))
    roles, parents = record(predecessor.get("roles")), record(predecessor.get("stages"))
    require(
        set(roles) == {"source", "audioClock", "videoClock", "modelInput"}
        and set(parents) == {"audio", "video"},
        "Exact completed clock/source roles required",
    )
    require(roles["source"] == device.get("originalRecording"), "Original source binding differs")
    declared = record(saved.get("outputs"))
    for name, output in zip(("vocals", "separationResult"), stage.outputs, strict=True):
        require(str(output) in declared, "Whole Stage output pin missing")
        roles[name] = {"path": str(output), "sha256": declared[str(output)]}
    parents["separation"] = {
        "workflow": asdict(workflow) | {"path": str(workflow.path)},
        "receipt": _metadata(receipt),
        "stageName": "device-whole",
    }
    template["predecessor"] = {"roles": roles, "stages": parents}
    return template, config, stage.verify_command


def _job(
    operation: Path, *, prepared: bool = True, definition: dict[str, object] | None = None
) -> Job:
    op = _read(_local(str(operation))) if definition is None else definition
    require(
        set(op)
        == {
            "version",
            "protocol",
            "configuration",
            "workRoot",
            "baseConfig",
            "readerCode",
            "inputs",
        }
        and type(op.get("version")) is int
        and op.get("version") == 1
        and op.get("protocol") == PROTOCOL
        and op.get("readerCode") == _code(),
        "Operation/code changed",
    )
    config = _ref(op["configuration"])
    raw = _config(config.path)
    resolved, device, command = _resolve(raw)
    generated = _ref(op["baseConfig"])
    require(_read(generated.path) == resolved, "Resolved predecessor roles changed")
    settings = base._settings(generated.path)
    ready, inputs = base._explicit(settings)
    result = _read(settings.roles["separationResult"].path)
    require(
        result.get("protocol") == "explicit-device-separation-continuation-v1"
        and result.get("device") in {"mps", "mixed"}
        and result.get("cpuExactGateChanged") is False,
        "Whole is not an explicit device result",
    )
    code = record(raw["metalCodeInputs"])
    require(
        1 <= len(code) <= 1024 and all(isinstance(v, str) for v in code.values()),
        "Missing bounded Metal producer pins",
    )
    metal_code = {str(canonical(k)): str(v) for k, v in code.items()}
    require(
        {str(p) for p in metal.required_code()} <= metal_code.keys(), "Metal closure incomplete"
    )
    for name, digest in metal_code.items():
        _pin({"path": name, "sha256": digest})
        require(Path(name).is_relative_to(ROOT), "Metal code must stay in checkout")
    deadlines = record(raw["timeouts"])
    require(
        set(deadlines) == {"whole", "mono", "window", "recognize", "assemble"},
        "Exact finite phase deadlines required",
    )
    limits = {name: _timeout(value) for name, value in deadlines.items()}
    require(limits["window"] <= 3600, "Metal window deadline too large")
    pins = {p.path: p.sha256 for p in inputs}
    for value in (config, generated, device, *base._models(settings), *settings.runtime):
        require(
            value.path not in pins or pins[value.path] == value.sha256,
            "Conflicting declared input pin",
        )
        pins[value.path] = value.sha256
    for name in (".python-version", "pyproject.toml", "uv.lock"):
        path = ROOT / name
        pins[path] = str(pin(path)["sha256"])
    pins.update({Path(k): v for k, v in _code().items()})
    pins.update({Path(k): v for k, v in metal_code.items()})
    actual = [{"path": str(p), "sha256": s} for p, s in pins.items()]
    require(len(actual) <= 4096, "Transitive input budget exceeded")
    if prepared:
        require(op["inputs"] == actual, "Declared transitive inputs changed")
    return Job(
        _local(op["workRoot"]),
        operation,
        settings,
        ready,
        tuple(Pin(p, s) for p, s in pins.items()),
        device,
        command,
        metal_code,
        limits,
    )


def _environment(job: Job) -> dict[str, str]:
    env = os.environ.copy()
    env.update({name: str(job.root / "tmp") for name in ("TMPDIR", "TMP", "TEMP")})
    env.update(
        XDG_CACHE_HOME=str(job.root / "cache"),
        PYTHONDONTWRITEBYTECODE="1",
        PYTHONPATH=str(ROOT / "src"),
        HF_HUB_OFFLINE="1",
        TRANSFORMERS_OFFLINE="1",
    )
    device = _read(job.device_config.path)
    original = _read(_pin(device["originalPlan"]).path)
    context = _read(_local(original["outputDirectory"]) / "input.json")
    original_env = record(context["environment"])
    for name in ("HF_HOME", "HF_HUB_CACHE"):
        value = canonical(original_env[name])
        require(
            value.is_relative_to(ROOT / "edits/models") and value.is_dir(), "Offline cache missing"
        )
        env[name] = str(value)
    return env


def _data(job: Job, phase: str) -> dict[str, object]:
    if phase == "whole":
        samples = base._pcm(job.ready.vocals, 2, 44100)
        return {
            "result": pin(job.settings.roles["separationResult"].path),
            "vocals": pin(job.ready.vocals),
            "samples": samples,
            "sampleRate": 44100,
            "audioClock": asdict(job.settings.roles["audioClock"]) | {"path": str(job.ready.audio)},
            "videoClock": _metadata(job.ready.video, budget=base.MAX_JSON),
            "verifier": list(job.whole_command),
            "backend": _read(job.settings.roles["separationResult"].path)["device"],
        }
    _verify(job, "whole")
    mono = job.root / "mono.wav"
    if phase == "mono":
        samples, count = base._pcm(job.ready.vocals, 2, 44100), base._pcm(mono, 1, 16000)
        require(
            abs(Fraction(samples, 44100) - Fraction(count, 16000)) <= Fraction(1, 16000),
            "Mono changed duration",
        )
        return {
            "vocals": pin(job.ready.vocals),
            "mono": pin(mono),
            "samples": count,
            "sampleRate": 16000,
            "inputSamples": samples,
        }
    _verify(job, "mono")
    request = job.root / "metal-request.json"
    if phase == "request":
        weights, config = base._models(job.settings)
        return {
            "version": 1,
            "device": "metal",
            "fp16": True,
            "cpuAdoption": None,
            "inputs": {
                "pcm": pin(mono),
                "config": {"path": str(config.path), "sha256": config.sha256},
                "weights": {"path": str(weights.path), "sha256": weights.sha256},
            },
            "samples": base._pcm(mono, 1, 16000),
            "timeoutSeconds": job.deadlines["window"],
            "codeInputs": job.metal_code,
        }
    _verify(job, "request")
    complete = metal.collect(request, job.root / "asr", verify_only=True)
    if phase == "recognize":
        return {
            "request": _metadata(request),
            "complete": pin(complete),
            "device": "metal",
            "fullRecognitionVerified": True,
            "clock": "input-pcm-not-projected",
        }
    _verify(job, "recognize")
    labels = _read(job.root / "labels.json", budget=base.MAX_JSON)
    collection = job.root / "asr"
    windows = asr_assembly._windows(_read(collection / "inputs.json"))
    native: list[dict[str, object]] = []
    for i, w in enumerate(windows):
        left = (windows[i - 1].end_sample + w.start_sample) / 32000 if i else 0.0
        right = (
            (w.end_sample + windows[i + 1].start_sample) / 32000
            if i + 1 < len(windows)
            else w.end_sample / 16000
        )
        native.extend(
            asdict(item)
            for item in asr_assembly._labels(
                _read(collection / f"{w.start_sample:012d}.raw.json"),
                window=w,
                owner_start=left,
                owner_end=right,
            )
        )
    require(
        labels.get("labels") == native
        and labels.get("clock") == "input-pcm-not-projected"
        and labels.get("speechAccuracyVerified") is False
        and labels.get("humanReviewed") is False
        and labels.get("deliveryComplete") is False,
        "Original-clock labels changed",
    )
    ancestry = {
        str(collection / name): pin(collection / name)["sha256"]
        for name in ("inputs.json", "complete.json")
    }
    for window in windows:
        ancestry.update(
            {
                str(collection / f"{window.start_sample:012d}.{kind}.json"): pin(
                    collection / f"{window.start_sample:012d}.{kind}.json"
                )["sha256"]
                for kind in ("raw", "clock")
            }
        )
    require(
        labels.get("parentSHA256") == ancestry
        and labels.get("assemblySHA256") == pin(Path(asr_assembly.__file__))["sha256"]
        and labels.get("parserSHA256") == pin(Path(caption_io.__file__))["sha256"],
        "ASR ancestry or parser changed",
    )
    return {
        "labels": pin(job.root / "labels.json"),
        "clock": "input-pcm-not-projected",
        "groups": len(native),
        "sourceProjectionApplications": 0,
    }


def _checkpoint(job: Job, phase: str) -> Path:
    return job.root / ("metal-request.json" if phase == "request" else phase + ".json")


def _verify(job: Job, phase: str) -> None:
    expected = _data(job, phase)
    require(_read(_checkpoint(job, phase)) == expected, "Phase evidence changed")
    if phase == "request":
        metal.load_request(_checkpoint(job, phase))


def run(
    operation: Path,
    phase: str,
    *,
    execute: Executor | None = None,
    runtime: metal.Runtime | None = None,
) -> Path:
    job = _job(operation)
    verify = phase.startswith("verify-")
    phase = phase.removeprefix("verify-")
    require(phase in PHASES, "Unknown phase")
    snapshot = InputHashSnapshot.capture((p.path, p.sha256) for p in job.inputs)
    if verify:
        _verify(job, phase)
    else:
        output = _checkpoint(job, phase)
        require(
            not output.exists() and not output.is_symlink(), "Occupied phase; inspect before resume"
        )
        command: tuple[str, ...] | None = None
        if phase == "whole":
            command = job.whole_command
        elif phase == "mono":
            command = base.mono_command(
                job.ready.vocals, job.root / "mono.wav", job.ready.tools["ffmpeg"].path
            )
        if command is not None:
            if execute is not None:
                execute(
                    command, job.root / (phase + ".log"), job.deadlines[phase], _environment(job)
                )
            else:
                with (job.root / (phase + ".log")).open("xb") as log:
                    subprocess.run(
                        command,
                        stdin=subprocess.DEVNULL,
                        stdout=log,
                        stderr=subprocess.STDOUT,
                        check=True,
                        timeout=job.deadlines[phase],
                        env=_environment(job),
                    )
        elif phase == "recognize":
            _verify(job, "request")
            metal.collect(job.root / "metal-request.json", job.root / "asr", runtime=runtime)
        elif phase == "assemble":
            _verify(job, "recognize")
            asr_assembly.assemble(job.root / "asr", job.root / "labels.json")
        publish(output, encode(_data(job, phase)))
    snapshot.finish()
    return _checkpoint(job, phase)


def prepare(config: Path, output: Path) -> Path:
    config, output = _local(str(config)), _local(str(output))
    raw = _config(config)
    resolved, _, _ = _resolve(raw)
    require(output.is_dir() and not tuple(output.iterdir()), "New empty purpose root required")
    generated, operation = output / "transcription-config.json", output / "operation.json"
    publish(generated, encode(resolved))
    initial: dict[str, object] = {
        "version": 1,
        "protocol": PROTOCOL,
        "configuration": _metadata(config),
        "workRoot": str(output),
        "baseConfig": _metadata(generated),
        "readerCode": _code(),
        "inputs": [],
    }
    job = _job(operation, prepared=False, definition=initial)
    initial["inputs"] = [{"path": str(p.path), "sha256": p.sha256} for p in job.inputs]
    publish(operation, encode(initial))
    for name in ("tmp", "cache"):
        (output / name).mkdir(mode=0o700)
    path = output / "workflow.json"
    publish(path, encode(_graph(job)))
    return path


def _graph(job: Job) -> dict[str, object]:
    output, operation = job.root, job.operation
    selected = set(_code()) | set(job.metal_code)
    selected.update(str(ROOT / name) for name in (".python-version", "pyproject.toml", "uv.lock"))
    selected.update(str(p.path) for p in job.settings.roles.values())
    selected.update(str(t.pin.path) for t in job.ready.tools.values())
    selected.update(str(p.path) for p in base._models(job.settings))
    selected.update(
        str(p.path)
        for p in (
            job.device_config,
            _pin(_read(operation)["configuration"]),
            _pin(_read(operation)["baseConfig"]),
        )
    )
    for role in job.settings.stages.values():
        parent = record(role)
        selected.update(str(_pin(parent[k]).path) for k in ("workflow", "receipt"))
    common = [str(p.path) for p in job.inputs if str(p.path) in selected] + [str(operation)]
    stages: list[dict[str, object]] = []
    previous: list[str] = []
    for phase in PHASES:
        command = [
            str(job.ready.tools["python"].path),
            "-B",
            "-m",
            "replay.device_vocal_metal_asr_plan",
        ]
        outputs = [str(_checkpoint(job, phase))]
        if phase == "mono":
            outputs.append(str(output / "mono.wav"))
        if phase == "recognize":
            outputs.append(str(output / "asr/complete.json"))
        if phase == "assemble":
            outputs.append(str(output / "labels.json"))
        stages.append(
            {
                "name": phase,
                "command": [
                    str(job.ready.tools["python"].path),
                    "-B",
                    "-m",
                    "replay.phase_timer",
                    str(output / "PHASES.jsonl"),
                    phase,
                    "--",
                    *command,
                    phase,
                    str(operation),
                ],
                "verify_command": [*command, "verify-" + phase, str(operation)],
                "inputs": list(dict.fromkeys(common + previous)),
                "outputs": outputs,
                "timeout_seconds": job.deadlines.get(phase, job.deadlines["mono"]),
            }
        )
        previous += outputs
    graph = {"version": 1, "stages": stages}
    stage_graph(parse_stages(graph))
    require(len(encode(graph)) <= MAX_PLAN_BYTES, "Ordinary workflow exceeds its byte budget")
    return graph


def verify_plan(operation: Path) -> Path:
    """Verify exact child commands and resolved prerequisites for a parent Stage."""
    job = _job(operation)
    snapshot = InputHashSnapshot.capture((p.path, p.sha256) for p in job.inputs)
    path = job.root / "workflow.json"
    require(_read(path) == _graph(job), "Generated child workflow changed")
    snapshot.finish()
    return path


def _prerequisite(config: Path) -> Prerequisite:
    definition = _ref(_metadata(_local(str(config))))
    raw = _read(definition.path)
    require(
        set(raw)
        == {
            "version",
            "kind",
            "workflow",
            "capture",
            "stateDirectory",
            "terminalStatus",
            "codeInputs",
            "waitSeconds",
            "pollSeconds",
            "output",
        }
        and type(raw["version"]) is int
        and raw["version"] == 1
        and raw["kind"] == PREREQUISITE_PROTOCOL,
        "Invalid resource-only prerequisite",
    )
    workflow, captured = _ref(raw["workflow"]), _ref(raw["capture"])
    stages = parse_stages(_read(workflow.path))
    stage_graph(stages)
    state, status, output = (_local(raw[k]) for k in ("stateDirectory", "terminalStatus", "output"))
    require(
        state.is_dir() and status.parent.is_dir() and output.parent.is_dir(),
        "Missing prerequisite namespace",
    )
    require(
        output not in {definition.path, workflow.path, captured.path, status}
        and not output.is_relative_to(state),
        "Prerequisite output overlaps original evidence",
    )
    _parent, capture_record = admission.load_completion_capture(captured.path, captured.sha256)
    require(
        capture_record.workflow == workflow.path
        and capture_record.workflow_sha256 == workflow.sha256
        and capture_record.checkpoint.parent == state
        and capture_record.status == status,
        "Captured original prefix does not bind this workflow/state/status",
    )
    entries = rows(raw["codeInputs"])
    require(1 <= len(entries) <= barrier.MAX_CODE_INPUTS, "Bounded producer code pins required")
    expected = {
        path
        for stage in stages
        for path in stage.inputs
        if path.suffix in barrier.CODE_SUFFIXES or path.name == "METADATA"
    }
    code: list[Pin] = []
    for entry in entries:
        item = _pin(entry)
        require(
            item.path.is_relative_to(ROOT)
            and item.path in expected
            and item.path.stat().st_size <= barrier.MAX_CODE_BYTES,
            "Producer code pin is outside the bounded declared closure",
        )
        code.append(_ref(entry))
    require(
        {item.path for item in code} == expected
        and len(code) == len(expected)
        and sum(item.path.stat().st_size for item in code) <= barrier.MAX_CODE_TOTAL,
        "Producer code closure is incomplete or repeated",
    )
    wait, poll = _timeout(raw["waitSeconds"]), _timeout(raw["pollSeconds"])
    require(
        wait <= MAX_PREREQUISITE_WAIT_SECONDS
        and poll <= 60
        and (wait + poll - 1) // poll + 1 <= barrier.MAX_POLLS,
        "Prerequisite polling exceeds its finite budget",
    )
    reader = tuple(Pin(Path(name), digest) for name, digest in _code().items())
    return Prerequisite(
        definition,
        workflow,
        captured,
        state,
        status,
        stages,
        tuple(code),
        reader,
        wait,
        poll,
        output,
    )


def _ordinary_checkpoints(request: Prerequisite) -> list[dict[str, object]]:
    checkpoints: list[dict[str, object]] = []
    for stage in request.stages:
        path = _local(str(request.state / f"{stage.name}.json"))
        saved = _read(path)
        outputs = record(saved.get("outputs"))
        require(
            set(saved) == {"status", "fingerprint", "outputs"}
            and saved["status"] == "complete"
            and set(outputs) == {str(path) for path in stage.outputs},
            "Zero macro status lacks all ordinary completed checkpoints",
        )
        admission._hash(saved["fingerprint"])
        tuple(admission._hash(digest) for digest in outputs.values())
        checkpoints.append(
            {
                "stage": stage.name,
                **_metadata(path),
                "fingerprint": saved["fingerprint"],
                "outputs": outputs,
            }
        )
    return checkpoints


def _prerequisite_observation(
    request: Prerequisite, host: barrier.Host
) -> dict[str, object] | None:
    require(
        _prerequisite(request.config.path) == request, "Prerequisite inputs changed while waiting"
    )
    if not request.status.exists():
        return None
    status_bytes = bounded_read(request.status)
    require(status_bytes in (b"0", b"0\n"), "Failed or malformed prerequisite macro status")
    checkpoints = _ordinary_checkpoints(request)
    _parent, captured = admission.load_completion_capture(
        request.capture.path, request.capture.sha256
    )
    require(
        bool(admission.terminal_evidence(captured)),
        "Captured prefix lacks original terminal evidence",
    )
    if any(group in captured.process_groups for _pid, group in host.groups()):
        return None
    released = host.release(barrier.Pin(request.capture.path, request.capture.sha256))
    require(
        released.get("resourceReleased") is True and released.get("stageQualityApproved") is False,
        "Prerequisite ownership release is incomplete or claims quality approval",
    )
    require(
        _prerequisite(request.config.path) == request
        and _ordinary_checkpoints(request) == checkpoints
        and bounded_read(request.status) == status_bytes,
        "Prerequisite terminal metadata changed during release observation",
    )
    return {
        "version": 1,
        "kind": PREREQUISITE_PROTOCOL,
        "configuration": asdict(request.config) | {"path": str(request.config.path)},
        "workflow": asdict(request.workflow) | {"path": str(request.workflow.path)},
        "capture": asdict(request.capture) | {"path": str(request.capture.path)},
        "readerCode": {str(item.path): item.sha256 for item in request.reader_code},
        "terminalStatus": _metadata(request.status),
        "ordinaryCheckpoints": checkpoints,
        "actorRelease": released,
        "resourceReleased": True,
        "qualityApproved": False,
        "domainApproved": False,
        "limitations": "GPU ownership dependency only; no ASR or media/domain adoption",
    }


def wait_prerequisite(config: Path, *, host: barrier.Host | None = None) -> Path:
    """Wait for an already captured ordinary macro; never run or retry it."""
    request = _prerequisite(config)
    require(not request.output.exists(), "Occupied prerequisite output requires reconciliation")
    current = barrier.SystemHost() if host is None else host
    deadline = current.monotonic() + request.wait_seconds
    for _attempt in range(
        (request.wait_seconds + request.poll_seconds - 1) // request.poll_seconds + 1
    ):
        result = _prerequisite_observation(request, current)
        if result is not None:
            publish(request.output, encode(result))
            return request.output
        remaining = deadline - current.monotonic()
        require(
            remaining > 0, "Prerequisite macro did not release ownership within its finite deadline"
        )
        current.sleep(min(request.poll_seconds, remaining))
    raise ReplayError("Prerequisite macro exhausted its finite observation budget")


def verify_prerequisite(config: Path, *, host: barrier.Host | None = None) -> Path:
    """Recheck resource release and retained ordinary checkpoints without media reads."""
    request = _prerequisite(config)
    current = barrier.SystemHost() if host is None else host
    observed = _prerequisite_observation(request, current)
    require(
        observed is not None and _read(request.output) == observed,
        "Prerequisite release proof changed or actor is live",
    )
    return request.output


def draft(config: Path, output: Path) -> Path:
    """Publish declared future roles without adopting or hashing absent outputs."""
    config, output = _local(str(config)), _local(str(output))
    raw = _config(config)
    device, workflow, receipt, stage = _whole_stage(raw)
    publish(
        output,
        encode(
            {
                "version": 1,
                "protocol": PROTOCOL,
                "configuration": _metadata(config),
                "wholeConfiguration": asdict(device) | {"path": str(device.path)},
                "wholeWorkflow": asdict(workflow) | {"path": str(workflow.path)},
                "wholeReceiptPath": str(receipt),
                "futureOutputs": [str(path) for path in stage.outputs],
                "canDispatch": False,
                "wholeBarrierCompletionVerified": False,
                "qualityApproved": False,
                "deliveryComplete": False,
            }
        ),
    )
    return output


class Arguments(argparse.Namespace):
    action: str
    config: Path
    output: Path | None


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "action",
        choices=(
            "prepare",
            "draft",
            "verify-plan",
            "check-prerequisite",
            "wait-prerequisite",
            "verify-prerequisite",
            *PHASES,
            *("verify-" + p for p in PHASES),
        ),
    )
    parser.add_argument("config", type=Path)
    parser.add_argument("output", nargs="?", type=Path)
    args = parser.parse_args(argv, namespace=Arguments())
    try:
        if args.action in {"prepare", "draft"}:
            if args.output is None:
                raise ReplayError("New work root required")
            result = (prepare if args.action == "prepare" else draft)(args.config, args.output)
        elif args.action == "verify-plan":
            result = verify_plan(args.config)
        elif args.action == "check-prerequisite":
            result = _prerequisite(args.config).config.path
        elif args.action in {"wait-prerequisite", "verify-prerequisite"}:
            result = (
                wait_prerequisite if args.action == "wait-prerequisite" else verify_prerequisite
            )(args.config)
        else:
            result = run(args.config, args.action)
        print(result)
        return 0
    except (OSError, ValueError, ReplayError, StageError, subprocess.SubprocessError) as error:
        print(str(error), file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
