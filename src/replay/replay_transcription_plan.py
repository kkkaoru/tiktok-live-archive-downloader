"""Continue a completed preparation through vocal ASR and source-clock cuts.

Plans never launch processing. Genuine predecessor stage receipts are required;
direct clock CLI results without those receipts are not silently adopted. A
stopped ASR can use an explicit fresh resume plan, preserving its prior state,
verified mono PCM and immutable windows. Source projection happens exactly once.
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
import wave
from dataclasses import dataclass
from fractions import Fraction
from pathlib import Path
from typing import Literal

from replay import asr_assembly, asr_clock_projection, continuous_quiet_plan, isolated_cpu_asr
from replay import replay_preparation_plan as predecessor
from replay.audio_clock import AudioFrameClock
from replay.audio_windows import plan_audio_windows
from replay.code_dependency_inventory import capture
from replay.core import ReplayError
from replay.delivery_audio import verify_material
from replay.delivery_media import parse_clock
from replay.input_hash_snapshot import InputHashSnapshot
from replay.local_asr import DEFAULT_CONFIG_SHA256, DEFAULT_WEIGHTS_SHA256
from replay.recordings import record
from replay.replay_preparation_plan import Pin, _local, _path, _pin, _read, _require, _write
from replay.stages import StageError, _fingerprint, execution_lock, file_digest
from replay.workflow import parse_stages, stage_graph

ROOT = Path(__file__).resolve().parents[2]
PROTOCOL = "replay-transcription-plan-v1"
RATE = 16000
MAX_INPUTS = 230
MAX_JSON = 128 * 1024 * 1024
Kind = Literal["mono", "recognize", "assemble", "lineage", "projection", "quiet"]


@dataclass(frozen=True, slots=True)
class Settings:
    preparation: Pin | None
    state: Path | None
    model: Path
    fps: int
    timeouts: dict[str, int]
    runtime: tuple[Pin, ...]
    roles: dict[str, Pin]
    stages: dict[str, object]
    tools: dict[str, predecessor.Tool]


@dataclass(frozen=True, slots=True)
class Ready:
    source: Path
    audio: Path
    video: Path
    stereo: Path
    vocals: Path
    audio_receipt: Path
    tools: dict[str, predecessor.Tool]


@dataclass(frozen=True, slots=True)
class Plan:
    config: Path
    work: Path
    prepared: Ready
    settings: Settings
    pins: tuple[Pin, ...]
    mono_root: Path
    collection: Path
    resume_from: Pin | None = None


def _settings(path: Path) -> Settings:
    _local(_path(str(path)), ROOT / "edits" / "work")
    data = record(_read(path))
    common = {"version", "model", "fps", "timeouts"}
    legacy = set(data) - {"runtimeInputs"} == common | {"preparation", "predecessorState"}
    explicit = set(data) - {"runtimeInputs"} == common | {"predecessor", "tools"}
    _require(
        (legacy or explicit) and type(data["version"]) is int and data["version"] == 1,
        "Invalid transcription settings",
    )
    preparation = _pin(data["preparation"]) if legacy else None
    state = _path(data["predecessorState"]) if legacy else None
    roles: dict[str, Pin] = {}
    stages: dict[str, object] = {}
    tools: dict[str, predecessor.Tool] = {}
    if preparation is not None and state is not None:
        _local(preparation.path, ROOT / "edits" / "work")
        _local(state, ROOT / "edits" / "work")
        _require(state.is_dir(), "Completed predecessor state required")
    else:
        before = record(data["predecessor"])
        _require(set(before) == {"roles", "stages"}, "Explicit predecessor roles/stages required")
        raw_roles = record(before["roles"])
        _require(
            set(raw_roles)
            == {"source", "audioClock", "videoClock", "modelInput", "vocals", "separationResult"},
            "Missing explicit predecessor artifact role",
        )
        roles = {name: _pin(value) for name, value in raw_roles.items()}
        for name, item in roles.items():
            if name != "source":
                _local(item.path, ROOT / "edits" / "work")
        stages = record(before["stages"])
        _require(
            set(stages) == {"audio", "video", "separation"}, "Missing genuine predecessor stage"
        )
        raw_tools = record(data["tools"])
        _require(set(raw_tools) == {"python", "ffmpeg", "env"}, "Missing pinned transcription tool")
        tools = {name: predecessor._tool(value) for name, value in raw_tools.items()}
        _require(
            all(os.access(tool.path, os.X_OK) for tool in tools.values()),
            "Tools must be executable",
        )
        predecessor._python_config(tools["python"])
    model = _path(data["model"])
    _local(model, ROOT / "edits" / "models")
    _require(model.is_dir(), "Installed model required; no model download")
    fps = data["fps"]
    if type(fps) is not int or not 1 <= fps <= 120:
        raise ReplayError("Expected an explicit supported output frame rate")
    timeouts = record(data["timeouts"])
    _require(
        set(timeouts) == {"mono", "recognize", "assemble", "child", "quiet"},
        "Missing transcription phase deadlines",
    )
    checked = {name: predecessor._timeout(value) for name, value in timeouts.items()}
    _require(checked["child"] <= 3600, "ASR child deadline exceeds backend bound")
    values = data.get("runtimeInputs", [])
    if not isinstance(values, list) or len(values) > 32:
        raise ReplayError("Expected bounded explicit runtime pins")
    runtime = tuple(_pin(value) for value in values)
    _require(len({item.path for item in runtime}) == len(runtime), "Duplicate runtime pin")
    for item in runtime:
        _local(item.path, ROOT)
    return Settings(preparation, state, model, fps, checked, runtime, roles, stages, tools)


def _code_paths() -> tuple[Path, ...]:
    return tuple(
        item.path
        for item in capture(
            ("replay_transcription_plan", "phase_timer", "workflow"),
            package=ROOT / "src" / "replay",
        )
    )


def _models(settings: Settings) -> tuple[Pin, Pin]:
    weights = tuple(
        path
        for path in (settings.model / "weights.npz", settings.model / "weights.safetensors")
        if path.exists()
    )
    _require(len(weights) == 1, "Exactly one installed nonTurbo large-v3 weight file required")
    return Pin(_path(str(weights[0])), DEFAULT_WEIGHTS_SHA256), Pin(
        _path(str(settings.model / "config.json")), DEFAULT_CONFIG_SHA256
    )


def _stage_receipt(workflow: Path, state: Path, name: str) -> Path:
    stages = parse_stages(_read(workflow))
    matches = tuple(stage for stage in stages if stage.name == name)
    _require(len(matches) == 1, "Required predecessor stage missing")
    stage = matches[0]
    path = state / f"{name}.json"
    _require(path.is_file(), "Missing genuine workflow receipt; direct CLI output is insufficient")
    _require(
        record(_read(path))
        == {
            "status": "complete",
            "fingerprint": _fingerprint(stage),
            "outputs": {str(output): file_digest(output) for output in stage.outputs},
        },
        "Predecessor stage is incomplete or its inputs/outputs changed",
    )
    return path


def _explicit(settings: Settings) -> tuple[Ready, tuple[Pin, ...]]:
    pins = dict((item.path, item.sha256) for item in settings.roles.values())
    receipts: dict[str, Path] = {}
    expected = {
        "audio": {"audioClock"},
        "video": {"videoClock"},
        "separation": {"vocals", "separationResult"},
    }
    for role, value in settings.stages.items():
        data = record(value)
        _require(
            set(data) == {"workflow", "receipt", "stageName"},
            "Pinned original workflow/receipt/name required",
        )
        workflow, receipt = _pin(data["workflow"]), _pin(data["receipt"])
        _local(workflow.path, ROOT / "edits" / "work")
        _local(receipt.path, ROOT / "edits" / "work")
        name = data["stageName"]
        _require(
            isinstance(name, str) and receipt.path.name == f"{name}.json",
            "Original stage identity differs",
        )
        _require(
            file_digest(workflow.path) == workflow.sha256
            and file_digest(receipt.path) == receipt.sha256,
            "Original workflow/receipt pin changed",
        )
        if not isinstance(name, str):
            raise ReplayError("Invalid predecessor stage name")
        _stage_receipt(workflow.path, receipt.path.parent, name)
        stage = next(stage for stage in parse_stages(_read(workflow.path)) if stage.name == name)
        _require(
            bool(stage.verify_command) and settings.roles["source"].path in stage.inputs,
            "Predecessor requires source-bound domain verification",
        )
        _require(
            all(settings.roles[key].path in stage.outputs for key in expected[role]),
            "Stage does not own the declared completed role",
        )
        pins[workflow.path], pins[receipt.path] = workflow.sha256, receipt.sha256
        for path in (*stage.inputs, *stage.outputs):
            pins.setdefault(_path(str(path)), file_digest(path))
        receipts[role] = receipt.path
    paths = {name: item.path for name, item in settings.roles.items()}
    result = record(_read(paths["separationResult"]))
    _require(
        paths["vocals"] == paths["separationResult"].parent / "vocals.wav"
        and result.get("modelInput") == str(paths["modelInput"])
        and result.get("loudnessNormalization") is False,
        "Separated input/output role or gain policy differs",
    )
    separation_source = (
        paths["modelInput"] if result.get("preparedInput") is True else paths["source"]
    )
    verify_material(separation_source, paths["separationResult"].parent)
    clock = AudioFrameClock.from_probe(_read(paths["audioClock"], budget=MAX_JSON))
    parse_clock(_read(paths["videoClock"], budget=MAX_JSON))
    count = _pcm(paths["modelInput"], 2, 44100)
    _require(
        abs(Fraction(count, 44100) - Fraction(clock.sample_ends[-1], clock.sample_rate))
        <= Fraction(1, 44100),
        "Separation model input differs from decoded source duration",
    )
    for tool in settings.tools.values():
        _require(
            tool.pin.path not in pins or pins[tool.pin.path] == tool.pin.sha256,
            "Conflicting predecessor/tool pins",
        )
        pins[tool.pin.path] = tool.pin.sha256
    python_config = predecessor._python_config(settings.tools["python"])
    pins[python_config] = file_digest(python_config)
    return Ready(
        paths["source"],
        paths["audioClock"],
        paths["videoClock"],
        paths["modelInput"],
        paths["vocals"],
        receipts["audio"],
        settings.tools,
    ), tuple(Pin(path, digest) for path, digest in pins.items())


def _basis(config: Path) -> tuple[Settings, Ready, tuple[Pin, ...]]:
    settings = _settings(config)
    if settings.preparation is None:
        ready, basis = _explicit(settings)
        pins = {item.path: item.sha256 for item in basis}
        paths = (
            config,
            *_code_paths(),
            ROOT / "pyproject.toml",
            ROOT / "uv.lock",
            ROOT / ".python-version",
        )
    else:
        ready, basis, paths = _legacy(config, settings)
        pins = {item.path: item.sha256 for item in basis}
    for item in (*_models(settings), *settings.runtime):
        _require(item.path not in pins or pins[item.path] == item.sha256, "Conflicting role pins")
        pins[item.path] = item.sha256
    for path in paths:
        pins.setdefault(path, file_digest(path))
    _require(len(pins) <= MAX_INPUTS, "Transcription closure exceeds direct input budget")
    return settings, ready, tuple(Pin(path, digest) for path, digest in sorted(pins.items()))


def _legacy(config: Path, settings: Settings) -> tuple[Ready, tuple[Pin, ...], tuple[Path, ...]]:
    preparation, state = settings.preparation, settings.state
    if preparation is None or state is None:
        raise ReplayError("Missing original preparation roles")
    _require(
        file_digest(preparation.path) == preparation.sha256,
        "Predecessor plan pin changed",
    )
    prepared = predecessor._load(preparation.path)
    _require(state == prepared.work / "state", "Use the original predecessor state")
    receipts = tuple(
        _stage_receipt(prepared.work / "workflow.json", state, name)
        for name in ("source-video-clock", "source-audio-clock", "source-voice-separation")
    )
    for kind in ("video", "audio", "separation"):
        predecessor._verify_producer(prepared, kind)
    predecessor._domain(prepared, "separation")
    paths = (
        config,
        preparation.path,
        *receipts,
        *predecessor._outputs(prepared, "video"),
        *predecessor._outputs(prepared, "audio"),
        *predecessor._outputs(prepared, "separation"),
        *(predecessor._producer_path(prepared, kind) for kind in ("video", "audio", "separation")),
        *_code_paths(),
    )
    ready = Ready(
        prepared.source,
        prepared.work / "audio-clock.json",
        prepared.work / "video-clock.json",
        prepared.work / "separation/model-input.wav",
        prepared.work / "separation/vocals.wav",
        state / "source-audio-clock.json",
        prepared.settings.tools,
    )
    return ready, prepared.pins, paths


def _manifest(plan: Plan) -> dict[str, object]:
    return {
        "version": 1,
        "protocol": PROTOCOL,
        "config": str(plan.config),
        "workRoot": str(plan.work),
        "monoRoot": str(plan.mono_root),
        "collection": str(plan.collection),
        "inputs": [item.data() for item in plan.pins],
        "resumeFrom": plan.resume_from.data() if plan.resume_from else None,
        "candidateOnly": True,
        "speechAccuracyVerified": False,
        "deliveryComplete": False,
    }


def _proof(plan: Plan, kind: Kind) -> Path:
    return (plan.mono_root if kind == "mono" else plan.work) / f"{kind}.producer.json"


def _outputs(plan: Plan, kind: Kind) -> tuple[Path, ...]:
    if kind == "mono":
        return (plan.mono_root / "mono.wav",)
    if kind == "recognize":
        policy = plan.work / "asr-policy"
        return (
            (plan.collection / "complete.json", policy / "started.json", policy / "result.json")
            if plan.resume_from
            else (
                plan.collection / "inputs.json",
                plan.collection / "complete.json",
                policy / "started.json",
                policy / "result.json",
            )
        )
    if kind == "assemble":
        return (plan.work / "labels.json",)
    if kind == "lineage":
        return (plan.work / "lineage.json",)
    if kind == "projection":
        return (plan.work / "projection-config.json", plan.work / "source-labels.json")
    return (
        plan.work / "quiet-config.json",
        plan.work / "quiet" / "cuts.json",
        plan.work / "quiet" / "peaks.txt",
    )


def _command(plan: Plan, *args: str) -> list[str]:
    return [
        str(plan.prepared.tools["python"].path),
        "-B",
        "-m",
        "replay.replay_transcription_plan",
        *args,
    ]


def _environment(plan: Plan) -> list[str]:
    return [
        str(plan.prepared.tools["env"].path),
        *(f"{name}={plan.work / 'scratch'}" for name in ("TMPDIR", "TMP", "TEMP")),
        f"XDG_CACHE_HOME={plan.work / 'cache'}",
        f"NUMBA_CACHE_DIR={plan.work / 'numba'}",
        "HF_HUB_OFFLINE=1",
        "HF_DATASETS_OFFLINE=1",
        "PYTHONDONTWRITEBYTECODE=1",
        f"PYTHONPATH={ROOT / 'src'}",
    ]


def _workflow(plan: Plan, *, successor: bool = False) -> dict[str, object]:
    manifest = plan.work / "transcription.json"
    common = [str(item.path) for item in plan.pins] + [str(manifest)]
    stages: list[dict[str, object]] = []
    for kind in (
        ("lineage", "projection", "quiet") if successor else ("mono", "recognize", "assemble")
    ):
        inputs = common.copy()
        if kind in ("recognize", "assemble", "lineage", "projection", "quiet"):
            inputs.extend((str(plan.mono_root / "mono.wav"), str(_proof(plan, "mono"))))
        if kind == "assemble":
            inputs.extend(
                (
                    str(_proof(plan, "recognize")),
                    str(plan.collection / "inputs.json"),
                    str(plan.collection / "complete.json"),
                )
            )
        if kind in ("lineage", "projection", "quiet"):
            inputs.extend((str(plan.work / "labels.json"), str(_proof(plan, "assemble"))))
        if kind in ("projection", "quiet"):
            inputs.extend((str(plan.work / "lineage.json"), str(_proof(plan, "lineage"))))
        if kind == "projection":
            inputs.append(str(plan.mono_root / "state" / "vocal-mono-pcm.json"))
        if kind == "quiet":
            inputs.extend(
                (
                    str(plan.work / "projection-config.json"),
                    str(plan.work / "source-labels.json"),
                    str(_proof(plan, "projection")),
                )
            )
        command = _command(plan, "produce", str(manifest), kind)
        duration = plan.settings.timeouts.get(kind, plan.settings.timeouts["assemble"])
        stage: dict[str, object] = {
            "name": {
                "mono": "vocal-mono-pcm",
                "recognize": "vocal-asr",
                "assemble": "vocal-asr-assembly",
            }.get(kind, f"source-{kind}"),
            "command": [
                *_environment(plan),
                str(plan.prepared.tools["python"].path),
                "-B",
                "-m",
                "replay.phase_timer",
                str(plan.work / "phases.jsonl"),
                kind,
                "--",
                *command,
            ],
            "inputs": list(dict.fromkeys(inputs)),
            "outputs": [*(str(path) for path in _outputs(plan, kind)), str(_proof(plan, kind))],
            "timeout_seconds": duration,
            "verify_command": [*_environment(plan), *_command(plan, "verify", str(manifest), kind)],
        }
        if kind == "mono" and plan.resume_from:
            stage["inputs"] = [
                name
                for name in inputs
                if name not in {str(plan.mono_root / "mono.wav"), str(_proof(plan, "mono"))}
            ]
            stage["adopt_existing"] = True
        stages.append(stage)
    result: dict[str, object] = {"version": 1, "stages": stages}
    stage_graph(parse_stages(result))
    return result


def _save(plan: Plan) -> Path:
    _local(_path(str(plan.work)), ROOT / "edits" / "work")
    _require(
        plan.work.parent.is_dir() and not plan.work.exists(), "Use an absent canonical work root"
    )
    snapshot = InputHashSnapshot.capture((item.path, item.sha256) for item in plan.pins)
    workflow = _workflow(plan)
    snapshot.finish()
    plan.work.mkdir(mode=0o700)
    for name in ("scratch", "cache", "numba"):
        (plan.work / name).mkdir(mode=0o700)
    _write(plan.work / "transcription.json", _manifest(plan))
    _write(plan.work / "workflow.json", workflow)
    return plan.work / "workflow.json"


def prepare(config: Path, work: Path) -> Path:
    """Bind a genuine completed predecessor; create a new three-stage workflow."""
    settings, prepared, pins = _basis(config)
    return _save(Plan(config, work, prepared, settings, pins, work, work / "asr"))


def _load(path: Path, *, depth: int = 0) -> Plan:
    _require(depth <= 8, "Resume ancestry exceeds its bounded depth")
    data = record(_read(path))
    _require(
        set(data)
        == {
            "version",
            "protocol",
            "config",
            "workRoot",
            "monoRoot",
            "collection",
            "inputs",
            "resumeFrom",
            "candidateOnly",
            "speechAccuracyVerified",
            "deliveryComplete",
        }
        and type(data["version"]) is int
        and data["version"] == 1
        and data["protocol"] == PROTOCOL
        and data["candidateOnly"] is True
        and data["speechAccuracyVerified"] is False
        and data["deliveryComplete"] is False,
        "Invalid transcription manifest",
    )
    config, work = _path(data["config"]), _path(data["workRoot"])
    _local(work, ROOT / "edits" / "work")
    _require(path == work / "transcription.json", "Manifest moved from its original work root")
    settings, prepared, expected = _basis(config)
    raw = data["inputs"]
    if not isinstance(raw, list) or not 1 <= len(raw) <= MAX_INPUTS:
        raise ReplayError("Expected bounded transcription input pins")
    pins = tuple(_pin(item) for item in raw)
    resume = None if data["resumeFrom"] is None else _pin(data["resumeFrom"])
    mono_root, collection = _path(data["monoRoot"]), _path(data["collection"])
    if resume is None:
        _require(
            mono_root == work and collection == work / "asr" and pins == expected,
            "Transcription input membership or hashes changed",
        )
    else:
        _require(file_digest(resume.path) == resume.sha256, "Prior plan changed")
        old = _load(resume.path, depth=depth + 1)
        extra = _resume_pins(old, resume, running=False)
        _require(
            mono_root == old.mono_root
            and collection == old.collection
            and pins == _merge(expected, extra),
            "Resume bindings changed",
        )
    plan = Plan(config, work, prepared, settings, pins, mono_root, collection, resume)
    _require(_read(work / "workflow.json") == _workflow(plan), "Pinned workflow changed")
    return plan


def _merge(base: tuple[Pin, ...], extra: tuple[Pin, ...]) -> tuple[Pin, ...]:
    result = {item.path: item.sha256 for item in base}
    for item in extra:
        _require(
            item.path not in result or result[item.path] == item.sha256, "Conflicting resume pins"
        )
        result[item.path] = item.sha256
    _require(len(result) <= MAX_INPUTS, "Resume closure exceeds direct input budget")
    return tuple(Pin(path, digest) for path, digest in sorted(result.items()))


def _pcm(path: Path, channels: int, rate: int) -> int:
    _require(path.is_file() and _path(str(path)) == path, "Canonical PCM required")
    with wave.open(str(path), "rb") as stream:
        _require(
            (
                stream.getnchannels(),
                stream.getsampwidth(),
                stream.getframerate(),
                stream.getcomptype(),
            )
            == (channels, 2, rate, "NONE")
            and 0 < stream.getnframes() <= 21600 * rate,
            "Unexpected bounded PCM format",
        )
        samples = stream.getnframes()
        count = 0
        while block := stream.readframes(65536):
            count += len(block)
        _require(count == samples * channels * 2, "Truncated PCM data")
    return samples


def mono_command(source: Path, output: Path, ffmpeg: Path) -> tuple[str, ...]:
    """Decode samples continuously; no timestamp stretching, padding or trimming."""
    return (
        str(ffmpeg),
        "-nostdin",
        "-hide_banner",
        "-v",
        "error",
        "-n",
        "-i",
        str(source),
        "-map",
        "0:a:0",
        "-vn",
        "-ac",
        "1",
        "-ar",
        str(RATE),
        "-c:a",
        "pcm_s16le",
        str(output),
    )


def _mono_domain(plan: Plan) -> None:
    vocals = plan.prepared.vocals
    count = _pcm(vocals, 2, 44100)
    samples = _pcm(plan.mono_root / "mono.wav", 1, RATE)
    clock = AudioFrameClock.from_probe(_read(plan.prepared.audio, budget=MAX_JSON))
    _require(
        abs(Fraction(samples, RATE) - Fraction(count, 44100)) <= Fraction(1, RATE)
        and abs(Fraction(samples, RATE) - Fraction(clock.sample_ends[-1], clock.sample_rate))
        <= Fraction(1, RATE),
        "Mono PCM changed the continuous decoded duration",
    )


def _definition(plan: Plan) -> dict[str, object]:
    paths = (
        plan.mono_root / "mono.wav",
        plan.settings.model / "config.json",
        _models(plan.settings)[0].path,
        ROOT / "src" / "replay" / "batch_asr.py",
        ROOT / "src" / "replay" / "asr_runtime.py",
    )
    return {
        "version": 1,
        "inputs": {str(path): file_digest(path) for path in paths},
        "samples": _pcm(paths[0], 1, RATE),
        "sampleRate": RATE,
        "windowSamples": 60 * RATE,
        "strideSamples": 30 * RATE,
        "clock": "input-pcm-not-projected",
        "device": "cpu",
        "dtypeContract": "provider-default-fp16-unchanged",
    }


def _collection(plan: Plan) -> dict[str, str]:
    manifest, complete = plan.collection / "inputs.json", plan.collection / "complete.json"
    definition = _definition(plan)
    _require(_read(manifest) == definition, "ASR does not bind the exact mono PCM/model/runtime")
    windows = plan_audio_windows(
        _pcm(plan.mono_root / "mono.wav", 1, RATE),
        window_samples=60 * RATE,
        stride_samples=30 * RATE,
    )
    bindings = {str(manifest): file_digest(manifest), str(complete): file_digest(complete)}
    hashes: list[str] = []
    for window in windows:
        raw = plan.collection / f"{window.start_sample:012d}.raw.json"
        clock = plan.collection / f"{window.start_sample:012d}.clock.json"
        digest = file_digest(raw)
        _require(
            _read(clock)
            == {
                "startSample": window.start_sample,
                "endSample": window.end_sample,
                "sampleRate": RATE,
                "humanReviewed": False,
                "emptyRecognitionIsNotSilence": True,
                "rawSHA256": digest,
            },
            "ASR raw window clock or hash changed",
        )
        hashes.append(digest)
        bindings[str(raw)], bindings[str(clock)] = digest, file_digest(clock)
    _require(
        type(record(_read(complete)).get("windows")) is int
        and _read(complete)
        == {"rawSHA256": hashes, "windows": len(windows), "deliveryComplete": False},
        "ASR collection is not wholly complete",
    )
    return bindings


def _domain(plan: Plan, kind: Kind) -> dict[str, str]:
    if kind == "mono":
        _mono_domain(plan)
        return {}
    if kind == "recognize":
        bindings = _collection(plan)
        result = record(_read(plan.work / "asr-policy" / "result.json"))
        started = record(_read(plan.work / "asr-policy" / "started.json"))
        count = record(_read(plan.collection / "complete.json"))["windows"]
        _require(
            result.get("version") == 1
            and type(result.get("version")) is int
            and type(result.get("verifiedWindows")) is int
            and type(result.get("totalWindows")) is int
            and started.get("version") == 1
            and type(started.get("version")) is int
            and started.get("producerSHA256")
            == file_digest(ROOT / "src/replay/isolated_cpu_asr.py")
            and started.get("processPolicy") == "fresh-cpu-child-per-missing-window"
            and started.get("cachePolicy") == "unchanged-provider-default"
            and result.get("recognitionComplete") is True
            and result.get("deliveryComplete") is False
            and result.get("verifiedWindows") == count
            and result.get("totalWindows") == count
            and result.get("producerSHA256") == file_digest(ROOT / "src/replay/isolated_cpu_asr.py")
            and started.get("inputs") == _definition(plan)["inputs"]
            and started.get("manifestSHA256") == bindings[str(plan.collection / "inputs.json")]
            and started.get("limitNewWindows") is None
            and started.get("childTimeoutSeconds") == plan.settings.timeouts["child"],
            "Full CPU ASR completion policy changed",
        )
        return bindings
    if kind == "assemble":
        bindings = _collection(plan)
        data = record(_read(plan.work / "labels.json", budget=MAX_JSON))
        _require(
            data.get("clock") == "input-pcm-not-projected"
            and data.get("parentSHA256") == bindings
            and data.get("assemblySHA256") == file_digest(ROOT / "src/replay/asr_assembly.py")
            and data.get("speechAccuracyVerified") is False
            and data.get("deliveryComplete") is False,
            "Assembly lost the original ASR evidence or clock",
        )
        return bindings
    if kind == "lineage":
        _require(
            _read(plan.work / "lineage.json") == _prepared_template(plan, "lineage"),
            "Lineage differs from verified transforms",
        )
    elif kind == "projection":
        _require(
            _read(plan.work / "projection-config.json") == _prepared_template(plan, "projection"),
            "Projection roles changed",
        )
        data = record(_read(plan.work / "source-labels.json", budget=MAX_JSON))
        _require(
            data.get("clock") == "source-presentation"
            and data.get("configSHA256") == file_digest(plan.work / "projection-config.json")
            and data.get("deliveryComplete") is False,
            "Source projection evidence changed",
        )
    else:
        _require(
            _read(plan.work / "quiet-config.json") == _prepared_template(plan, "quiet"),
            "Quiet planning roles changed",
        )
        data = record(_read(plan.work / "quiet" / "cuts.json", budget=MAX_JSON))
        _require(
            data.get("clock") == "source-presentation"
            and data.get("fps") == plan.settings.fps
            and data.get("configSHA256") == file_digest(plan.work / "quiet-config.json")
            and data.get("nonSpeechRemovalVerified") is False
            and data.get("deliveryComplete") is False,
            "Quiet candidate evidence changed",
        )
    return {}


def _producer_data(plan: Plan, kind: Kind) -> dict[str, object]:
    evidence = _domain(plan, kind)
    sample_counts = (
        {
            "stereoInputSamples": _pcm(plan.prepared.vocals, 2, 44100),
            "stereoInputRate": 44100,
            "monoOutputSamples": _pcm(plan.mono_root / "mono.wav", 1, RATE),
            "monoOutputRate": RATE,
            "durationPreserving": True,
        }
        if kind == "mono"
        else {}
    )
    return {
        "version": 1,
        "protocol": PROTOCOL,
        "kind": kind,
        "planSHA256": file_digest(
            (plan.mono_root if kind == "mono" else plan.work) / "transcription.json"
        ),
        "outputs": {str(path): file_digest(path) for path in _outputs(plan, kind)},
        "collectionEvidence": evidence,
        "sampleCounts": sample_counts,
        "candidateOnly": True,
        "deliveryComplete": False,
    }


def verify(path: Path, kind: Kind) -> None:
    """Read-only full domain/provenance verification; partial outputs never pass."""
    plan = _load(path)
    snapshot = InputHashSnapshot.capture((item.path, item.sha256) for item in plan.pins)
    _require(
        _read(_proof(plan, kind), budget=MAX_JSON) == _producer_data(plan, kind),
        "Missing or changed completed producer evidence",
    )
    snapshot.finish()


def produce(path: Path, kind: Kind) -> None:
    """Produce exclusive results and seal them only after domain and endpoint checks."""
    plan = _load(path)
    _require(
        not any(
            output.exists() or output.is_symlink()
            for output in (*_outputs(plan, kind), _proof(plan, kind))
        ),
        "Occupied producer outputs; inspect the retained attempt",
    )
    snapshot = InputHashSnapshot.capture((item.path, item.sha256) for item in plan.pins)
    if kind == "mono":
        subprocess.run(
            mono_command(
                plan.prepared.vocals,
                plan.mono_root / "mono.wav",
                plan.prepared.tools["ffmpeg"].path,
            ),
            stdin=subprocess.DEVNULL,
            check=True,
            timeout=plan.settings.timeouts["mono"],
        )
    elif kind == "recognize":
        verify(path, "mono")
        isolated_cpu_asr.run(
            plan.mono_root / "mono.wav",
            plan.collection,
            plan.work / "asr-policy",
            timeout=plan.settings.timeouts["child"],
            model=plan.settings.model,
            weights_sha256=DEFAULT_WEIGHTS_SHA256,
            config_sha256=DEFAULT_CONFIG_SHA256,
        )
    elif kind == "assemble":
        verify(path, "recognize")
        asr_assembly.assemble(plan.collection, plan.work / "labels.json")
    else:
        verify(path, "assemble")
        template = plan.work / f"{kind}-template.json"
        target = _outputs(plan, kind)[0]
        continuous_quiet_plan.prepare_config(template, target)
        if kind == "projection":
            asr_clock_projection.run(target, plan.work / "source-labels.json")
        elif kind == "quiet":
            continuous_quiet_plan.run(
                target, plan.work / "quiet", ffmpeg=str(plan.prepared.tools["ffmpeg"].path)
            )
    proof = _producer_data(plan, kind)
    snapshot.finish()
    _write(_proof(plan, kind), proof)


def _resume_pins(old: Plan, parent: Pin, *, running: bool) -> tuple[Pin, ...]:
    verify(old.work / "transcription.json", "mono")
    mono_receipt = _stage_receipt(
        old.mono_root / "workflow.json", old.mono_root / "state", "vocal-mono-pcm"
    )
    _require(
        _read(old.collection / "inputs.json") == _definition(old),
        "Partial ASR input definition changed",
    )
    receipt = old.work / "state" / "vocal-asr.json"
    data = record(_read(receipt))
    if running:
        stage = next(
            stage
            for stage in parse_stages(_read(old.work / "workflow.json"))
            if stage.name == "vocal-asr"
        )
        _require(
            data == {"status": "running", "fingerprint": _fingerprint(stage)},
            "Only the inspected unfinished ASR stage may resume",
        )
    paths = (
        parent.path,
        old.work / "workflow.json",
        receipt,
        mono_receipt,
        old.mono_root / "mono.wav",
        _proof(old, "mono"),
        old.collection / "inputs.json",
    )
    return tuple(Pin(path, file_digest(path)) for path in paths)


def prepare_resume(path: Path, work: Path) -> Path:
    """Plan a fresh ASR attempt while old workflow/stage ownership is absent.

    No signal, state rewrite, raw-window deletion or guessed output pin occurs.
    The ASR backend independently locks the collection and verifies every reused
    window; a changed or missing completed collection is never a retry candidate.
    """
    old = _load(path)
    with execution_lock(old.work / "state", ".workflow") as workflow_owned:
        _require(workflow_owned, "Prior workflow still owns its state")
        with execution_lock(old.work / "state", "vocal-asr") as stage_owned:
            _require(stage_owned, "Prior ASR stage still owns its state")
            _require(
                not (old.collection / "complete.json").exists(),
                "Reconcile completed ASR instead of resuming",
            )
            with execution_lock(old.collection, ".run") as collection_owned:
                _require(collection_owned, "Prior ASR collection still owns its windows")
                parent = Pin(path, file_digest(path))
                extra = _resume_pins(old, parent, running=True)
                settings, prepared, pins = _basis(old.config)
                return _save(
                    Plan(
                        old.config,
                        work,
                        prepared,
                        settings,
                        _merge(pins, extra),
                        old.mono_root,
                        old.collection,
                        parent,
                    )
                )


def _template(plan: Plan, kind: Literal["lineage", "projection", "quiet"]) -> dict[str, object]:
    audio = plan.prepared.audio
    stereo = plan.prepared.stereo
    mono = plan.mono_root / "mono.wav"
    transforms = ["concatenate-decoded-frames", "duration-preserving-resample"]
    if kind == "lineage":
        return {
            "version": 1,
            "prepareKind": kind,
            "clock": "continuous-decoded-pcm",
            "transforms": transforms,
            "artifactPaths": {
                "stereoPCM": str(stereo),
                "monoPCM": str(mono),
                "audioClock": str(audio),
            },
        }
    if kind == "projection":
        return {
            "version": 1,
            "prepareKind": kind,
            "inputClock": "continuous-decoded-pcm",
            "transforms": transforms,
            "artifactPaths": {
                "labels": str(plan.work / "labels.json"),
                "pcm": str(mono),
                "audioClock": str(audio),
                "asrInputs": str(plan.collection / "inputs.json"),
                "clockReceipt": str(plan.prepared.audio_receipt),
                "pcmReceipt": str(plan.mono_root / "state" / "vocal-mono-pcm.json"),
                "lineageEvidence": str(plan.work / "lineage.json"),
            },
        }
    return {
        "version": 1,
        "prepareKind": kind,
        "pcmClock": "continuous-decoded-pcm",
        "fps": plan.settings.fps,
        "artifactPaths": {
            "labels": str(plan.work / "source-labels.json"),
            "stereoPCM": str(stereo),
            "audioClock": str(audio),
            "videoClock": str(plan.prepared.video),
            "projectionConfig": str(plan.work / "projection-config.json"),
            "lineageEvidence": str(plan.work / "lineage.json"),
        },
    }


def _prepared_template(
    plan: Plan, kind: Literal["lineage", "projection", "quiet"]
) -> dict[str, object]:
    template = plan.work / f"{kind}-template.json"
    data = _template(plan, kind)
    _require(_read(template) == data, "Pinned successor template changed")
    fields = record(data.pop("artifactPaths"))
    data.pop("prepareKind")
    bindings = {
        name: {"path": str(_path(value)), "sha256": file_digest(_path(value))}
        for name, value in fields.items()
    }
    if kind == "lineage":
        data.update(bindings)
    else:
        data["artifacts"] = bindings
    data["templateSHA256"] = file_digest(template)
    return data


def prepare_successor(path: Path) -> Path:
    """Connect completed recognition to actual clocks and data-only prepareKind templates."""
    plan = _load(path)
    verify(path, "assemble")
    _stage_receipt(plan.work / "workflow.json", plan.work / "state", "vocal-asr-assembly")
    _require(
        not (plan.work / "successor-workflow.json").exists(), "Successor plan is already occupied"
    )
    for kind in ("lineage", "projection", "quiet"):
        _write(plan.work / f"{kind}-template.json", _template(plan, kind))
    workflow = _workflow(plan, successor=True)
    stages = record(workflow)["stages"]
    if not isinstance(stages, list):
        raise ReplayError("Invalid successor stages")
    for value in stages:
        stage = record(value)
        name = str(stage["name"]).removeprefix("source-")
        inputs = stage["inputs"]
        if not isinstance(inputs, list):
            raise ReplayError("Invalid successor inputs")
        inputs.append(str(plan.work / f"{name}-template.json"))
    stage_graph(parse_stages(workflow))
    _write(plan.work / "successor-workflow.json", workflow)
    return plan.work / "successor-workflow.json"


class Arguments(argparse.Namespace):
    action: str
    config: Path
    work: Path
    plan: Path
    kind: Kind


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="action", required=True)
    prepare_command = commands.add_parser("prepare")
    prepare_command.add_argument("config", type=Path)
    prepare_command.add_argument("work", type=Path)
    resume = commands.add_parser("prepare-resume")
    resume.add_argument("plan", type=Path)
    resume.add_argument("work", type=Path)
    successor = commands.add_parser("prepare-successor")
    successor.add_argument("plan", type=Path)
    for action in ("produce", "verify"):
        command = commands.add_parser(action)
        command.add_argument("plan", type=Path)
        command.add_argument(
            "kind", choices=("mono", "recognize", "assemble", "lineage", "projection", "quiet")
        )
    args = Arguments()
    parser.parse_args(argv, namespace=args)
    try:
        if args.action == "prepare":
            print(prepare(args.config, args.work))
        elif args.action == "prepare-resume":
            print(prepare_resume(args.plan, args.work))
        elif args.action == "prepare-successor":
            print(prepare_successor(args.plan))
        elif args.action == "produce":
            produce(args.plan, args.kind)
        else:
            verify(args.plan, args.kind)
    except (
        ReplayError,
        StageError,
        OSError,
        ValueError,
        subprocess.SubprocessError,
        wave.Error,
    ) as error:
        print(f"TRANSCRIPTION_PLAN_ERROR {error}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
