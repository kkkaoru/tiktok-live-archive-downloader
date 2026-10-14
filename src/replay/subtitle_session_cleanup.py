"""Explicit session OCR consumer and local repair; old producer evidence is immutable."""

from __future__ import annotations

import argparse
import hashlib
import os
import re
import shutil
import stat
import subprocess
import sys
import tempfile
import time
from collections.abc import Iterator
from contextlib import ExitStack, contextmanager
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np

from replay import code_dependency_inventory, subtitle_detector, subtitle_eligibility
from replay import code_generation_compat as compat
from replay import subtitle_cleanup_assembly as legacy_assembly
from replay import subtitle_cleanup_workflow as legacy
from replay import subtitle_session_observations as session
from replay.caption_io import FrameObservation, native_ocr_frames
from replay.caption_masks import MaskPolicy
from replay.cleanup_lane_successor import _quiescent, _unlocked
from replay.input_hash_cache import InputHashCache, input_hash_context
from replay.recordings import record
from replay.sequence_journal import bounded_read, decode, encode, publish
from replay.stages import Stage, execution_lock, file_digest
from replay.subtitle_cleanup_plan import admission, clock_chunks, search_profile
from replay.subtitle_clock import VideoFrameClock
from replay.subtitle_evidence import RepairCapture
from replay.subtitle_inpaint import Image
from replay.subtitle_plan import repair_spans
from replay.subtitle_video import VideoRepairJob, render_repaired_video
from replay.workflow import load_stages, parse_stages, stage_graph

PROTOCOL = "source-caption-session-cleanup-v1"

VERIFY_MODES = frozenset({"verify-ocr", "verify-admission", "verify-repair"})
MAX_CHUNKS = 512
MAX_DIRECT_INPUTS = 256
ROOT_FIELDS = frozenset(
    {"version", "originalPlan", "chunkIndex", "originalWorker", "handoff", "ocrProvider"}
)
TERMINAL_KIND = "terminal-target-cleanup-handoff-v1"
OWNER_KIND = "terminal-cleanup-owner-observation-v1"
PS_COMMAND = ("/bin/ps", "-ww", "-ax", "-o", "pid=,ppid=,lstart=,command=")
MAX_OWNER_SCAN_BYTES = 1024 * 1024


@dataclass(frozen=True)
class Job:
    config: Path
    original_plan: Path
    chunk: Path
    settings: dict[str, object]
    clock: VideoFrameClock
    provider: session.SessionPlan
    provider_path: Path
    original_state: Path
    original_stages: tuple[Stage, ...]
    handoff: dict[str, object]
    pins: dict[str, str]


@dataclass(frozen=True)
class Prepared:
    operation: Path
    workflow: Path
    state: Path


def _path(value: object, *, local: bool = True) -> Path:
    if not isinstance(value, str) or not value or any(char in value for char in "\n\r\x00"):
        raise ValueError("Expected canonical cleanup path")
    path = Path(value)
    root = Path.cwd().resolve()
    if (
        not path.is_absolute()
        or str(path) != value
        or path != path.resolve()
        or (local and not path.is_relative_to(root / "edits/work"))
        or any(path.is_relative_to(root / name) for name in ("private", "downloads"))
    ):
        raise ValueError("Cleanup metadata/output must be canonical repository work")
    return path


def _sha(value: object) -> str:
    if not isinstance(value, str) or re.fullmatch(r"[0-9a-f]{64}", value) is None:
        raise ValueError("Expected explicit SHA256")
    return value


def _pin(path: Path) -> dict[str, str]:
    return {"path": str(path), "sha256": file_digest(path)}


def _read(path: Path, pins: dict[str, str]) -> dict[str, object]:
    raw = bounded_read(_path(str(path)))
    data = decode(raw)
    digest = hashlib.sha256(raw).hexdigest()
    if pins.get(str(path), digest) != digest:
        raise ValueError("Session cleanup metadata changed")
    pins[str(path)] = digest
    return data


def _reference(value: object, pins: dict[str, str]) -> tuple[Path, dict[str, object]]:
    path = _pin_reference(value, pins)
    return path, _read(path, pins)


def _pin_reference(value: object, pins: dict[str, str]) -> Path:
    ref = record(value)
    if set(ref) != {"path", "sha256"}:
        raise ValueError("Expected exact metadata reference")
    path = _path(ref["path"])
    digest = hashlib.sha256(bounded_read(path)).hexdigest()
    if digest != _sha(ref["sha256"]) or pins.get(str(path), digest) != digest:
        raise ValueError("Session cleanup reference pin changed")
    pins[str(path)] = digest
    return path


def _asset(value: object, pins: dict[str, str], *, executable: bool = False) -> Path:
    """Plan without hashing media/models; ordinary runtime boundaries verify actual bytes."""
    ref = record(value)
    if set(ref) != {"path", "sha256"}:
        raise ValueError("Expected explicit asset reference")
    path = _path(ref["path"], local=False)
    info = path.lstat()
    if (
        not stat.S_ISREG(info.st_mode)
        or info.st_size <= 0
        or (executable and not os.access(path, os.X_OK))
    ):
        raise ValueError("Missing regular asset or executable")
    digest = _sha(ref["sha256"])
    if pins.get(str(path), digest) != digest:
        raise ValueError("Conflicting asset pins")
    pins[str(path)] = digest
    return path


def _old_stages(workflow: Path, chunk: Path, settings: dict[str, object]) -> tuple[Stage, ...]:
    stages = {stage.name: stage for stage in load_stages(workflow)}
    expected = parse_stages({"version": 1, "stages": legacy._stages(chunk, settings, None)})
    selected: list[Stage] = []
    for item in expected:
        actual = stages.get(item.name)
        if actual is None or (
            actual.command != item.command
            or actual.outputs != item.outputs
            or actual.verify_command != item.verify_command
            or actual.timeout_seconds != item.timeout_seconds
            or actual.adopt_existing
            or actual.inputs[: len(item.inputs)] != item.inputs
            or len(actual.inputs) > len(item.inputs) + (1 if item.name.endswith("-ocr") else 0)
        ):
            raise ValueError("Original Stage is not the unchanged cleanup producer")
        selected.append(actual)
    return tuple(selected)


def _profile(settings: dict[str, object], pins: dict[str, str]) -> tuple[FrameObservation, ...]:
    inputs, _ = _reference(settings.get("profileInputs"), pins)
    complete, receipt = _reference(settings.get("profileComplete"), pins)
    frames, manifest = legacy.collector_frames(inputs, complete, settings)
    batches = manifest.get("batches")
    if not isinstance(batches, list):
        raise ValueError("Missing original profile batches")
    for item in batches:
        _pin_reference(record(item).get("ocr"), pins)
    if receipt.get("observations") != len(frames):
        raise ValueError("Original profile observation count differs")
    return frames


def load_config(config: Path, *, require_observation: bool = True) -> Job:
    """Validate explicit physical membership and original authorities, without inference."""
    pins: dict[str, str] = {}
    raw = _read(config, pins)
    if set(raw) != ROOT_FIELDS or type(raw.get("version")) is not int or raw["version"] != 1:
        raise ValueError("Invalid session cleanup config")
    original, plan = _reference(raw["originalPlan"], pins)
    settings_path, settings = _reference(plan.get("settings"), pins)
    _, source_clock = _reference(settings.get("sourceClock"), pins)
    clocks = clock_chunks(source_clock, maximum_frames=_positive(settings.get("chunkFrames")))
    index = raw["chunkIndex"]
    if (
        type(index) is not int
        or not 0 <= index < len(clocks)
        or len(clocks) > MAX_CHUNKS
        or clocks[0].pts[0] != 0
        or plan.get("version") != 1
        or settings.get("version") != 1
        or plan.get("wholeSourcePlanned") is not True
        or plan.get("chunks") != len(clocks)
        or plan.get("totalSourceFrames") != sum(len(clock.pts) for clock in clocks)
        or plan.get("plannedFrames") != plan["totalSourceFrames"]
    ):
        raise ValueError("Original plan/selected chunk must cover the real physical source clock")
    chunk = original.parent / f"chunk-{index:05d}" / "chunk.json"
    chunk_data = _read(chunk, pins)
    clock = clocks[index]
    if (
        chunk_data.get("version") != 1
        or chunk_data.get("settings") != _pin(settings_path)
        or chunk_data.get("clock") != _clock_data(clock)
        or chunk_data.get("ocrDirectory") != str(chunk.parent / "ocr")
    ):
        raise ValueError("Original chunk clock/settings identity differs")
    source = _asset(settings.get("source"), pins)
    _asset(settings.get("native"), pins, executable=True)
    _asset(settings.get("decoder"), pins, executable=True)
    _asset(settings.get("model"), pins)
    _reference(settings.get("classificationEvidence"), pins)
    _profile(settings, pins)
    provider_path, _ = _reference(raw["ocrProvider"], pins)
    provider = session.load_plan(provider_path)
    if (
        provider.source != source
        or provider.source_sha != record(settings["source"])["sha256"]
        or provider.samples != tuple(pts * clock.time_base for pts in clock.pts)
        or provider.workers != 2
    ):
        raise ValueError("Session provider source/actual rational clock/worker contract differs")
    seeds = _profile(settings, pins)
    if (provider.width, provider.height) != (seeds[0].width, seeds[0].height):
        raise ValueError("Session provider canvas differs from original profile")
    _asset({"path": str(provider.native), "sha256": provider.native_sha}, pins, executable=True)
    worker = record(raw["originalWorker"])
    if set(worker) != {"workflow", "state"}:
        raise ValueError("Expected original worker workflow and Stage state")
    workflow, _ = _reference(worker["workflow"], pins)
    state = _path(worker["state"])
    if not state.is_dir():
        raise ValueError("Original Stage state must exist")
    handoff_path, handoff = _reference(raw["handoff"], pins)
    original_stages = _old_stages(workflow, chunk, settings)
    if handoff.get("kind") == TERMINAL_KIND:
        _terminal_bindings(
            handoff,
            workflow=workflow,
            state=state,
            future=original_stages,
            pins=pins,
            require_observation=require_observation,
        )
    else:
        evidence = handoff.get("evidence")
        if not isinstance(evidence, list):
            raise ValueError("Missing original owner handoff evidence")
        for ref in evidence:
            _reference(ref, pins)
    pins[str(handoff_path)] = file_digest(handoff_path)
    return Job(
        config,
        original,
        chunk,
        settings,
        clock,
        provider,
        provider_path,
        state,
        original_stages,
        handoff,
        pins,
    )


def _terminal_bindings(
    handoff: dict[str, object],
    *,
    workflow: Path,
    state: Path,
    future: tuple[Stage, ...],
    pins: dict[str, str],
    require_observation: bool,
) -> None:
    expected = {
        "version",
        "kind",
        "observedAtNs",
        "maximumAgeSeconds",
        "oldWorkersDisabled",
        "workflow",
        "state",
        "terminalJob",
        "completedStages",
        "currentObservation",
        "historicalActorIdentityRecorded",
    }
    if (
        set(handoff) != expected
        or type(handoff.get("version")) is not int
        or handoff["version"] != 1
        or handoff.get("kind") != TERMINAL_KIND
        or handoff.get("oldWorkersDisabled") is not True
        or handoff.get("historicalActorIdentityRecorded") is not False
        or handoff.get("state") != str(state)
    ):
        raise ValueError("Invalid explicit terminal-target handoff")
    path, _ = _reference(handoff["workflow"], pins)
    if path != workflow:
        raise ValueError("Terminal handoff original workflow differs")
    job = record(handoff["terminalJob"])
    if set(job) != {"root", "name", "status", "log", "targets"}:
        raise ValueError("Expected exact retained terminal job correspondence")
    root = _path(job["root"])
    name = job["name"]
    if not isinstance(name, str) or re.fullmatch(r"[a-z0-9][a-z0-9._-]{0,63}", name) is None:
        raise ValueError("Invalid terminal job name")
    status = _pin_reference(job["status"], pins)
    log = _pin_reference(job["log"], pins)
    if (
        status != root / f"{name}.status"
        or log != root / f"{name}.log"
        or bounded_read(status) != b"0\n"
    ):
        raise ValueError("Terminal job must have exact retained successful status/log paths")
    stages = {stage.name: stage for stage in load_stages(workflow)}
    graph = stage_graph(tuple(stages.values()))
    targets = list_rows(job["targets"])
    if (
        not targets
        or len(targets) > MAX_CHUNKS
        or any(not isinstance(target, str) or target not in stages for target in targets)
    ):
        raise ValueError("Terminal targets must be exact original workflow members")
    required: set[str] = set()
    pending = [str(target) for target in targets]
    while pending:
        target = pending.pop()
        if target not in required:
            required.add(target)
            pending.extend(graph[target])
    if required.intersection(stage.name for stage in future):
        raise ValueError("Terminal predecessor coverage cannot claim an unstarted future chunk")
    completed: set[str] = set()
    text = bounded_read(log).decode("utf-8")
    for value in list_rows(handoff["completedStages"]):
        entry = record(value)
        stage_name = entry.get("name")
        if (
            set(entry) != {"name", "receipt", "fingerprint"}
            or not isinstance(stage_name, str)
            or stage_name not in required
            or stage_name in completed
        ):
            raise ValueError("Terminal Stage membership is incomplete or duplicated")
        receipt_path, receipt = _reference(entry["receipt"], pins)
        stage = stages[stage_name]
        outputs = record(receipt.get("outputs"))
        if (
            receipt_path != state / f"{stage_name}.json"
            or receipt.get("status") != "complete"
            or _sha(receipt.get("fingerprint")) != _sha(entry["fingerprint"])
            or set(outputs) != {str(path) for path in stage.outputs}
            or any(not re.fullmatch(r"[0-9a-f]{64}", str(value)) for value in outputs.values())
            or re.search(
                rf"^{re.escape(stage_name)}: (completed|reused|adopted)$", text, re.MULTILINE
            )
            is None
        ):
            raise ValueError("Terminal Stage receipt/fingerprint/output/log correspondence differs")
        completed.add(stage_name)
    if completed != required:
        raise ValueError("Terminal handoff needs every actual selected predecessor Stage")
    if require_observation:
        _, observation = _reference(handoff["currentObservation"], pins)
        if (
            observation.get("kind") != OWNER_KIND
            or observation.get("historicalActorIdentityRecorded") is not False
            or observation.get("state") != str(state)
            or observation.get("workflow") != str(workflow)
        ):
            raise ValueError("Wrong retained current owner observation")
    elif handoff["currentObservation"] is not None:
        raise ValueError("Owner observation sealing requires an explicit unsealed draft")


def _scan_command(command: tuple[str, ...], *, allow_absent_server: bool = False) -> bytes:
    result = subprocess.run(
        list(command),
        stdin=subprocess.DEVNULL,
        capture_output=True,
        timeout=10,
        check=False,
    )
    if max(len(result.stdout), len(result.stderr)) > MAX_OWNER_SCAN_BYTES:
        raise ValueError("Owner observation exceeded bounded output")
    absent = (
        allow_absent_server
        and result.returncode == 1
        and not result.stdout
        and (
            result.stderr.startswith(b"no server running on")
            or result.stderr.startswith(b"error connecting to")
            or result.stderr.startswith(b"no sessions")
        )
    )
    if result.returncode != 0 and not absent:
        raise ValueError("Read-only owner observation failed")
    return result.stdout


def _owner_scan(job: Job) -> dict[str, object]:
    terminal = record(job.handoff["terminalJob"])
    worker = record(_read(job.config, {})["originalWorker"])
    workflow = _path(record(worker["workflow"])["path"])
    socket = _path(terminal["root"]) / "tmux.sock"
    selectors = (
        str(workflow),
        str(job.original_state),
        str(job.original_plan.parent),
        str(_path(terminal["root"]) / f"{terminal['name']}.log"),
    )
    ps = _scan_command(PS_COMMAND).decode("utf-8")
    if any(any(selector in line for selector in selectors) for line in ps.splitlines()):
        raise ValueError("Original owner or native child command is currently present")
    pane_command: tuple[str, ...] = ()
    if socket.exists() or socket.is_symlink():
        if socket.is_symlink() or not stat.S_ISSOCK(socket.lstat().st_mode):
            raise ValueError("Original owner socket is not canonical")
        tmux = shutil.which("tmux")
        if tmux is None:
            raise ValueError("Original owner pane observer is unavailable")
        pane_command = (
            str(Path(tmux).resolve()),
            "-f",
            "/dev/null",
            "-S",
            str(socket),
            "list-panes",
            "-a",
            "-F",
            "#{pane_start_command}",
        )
        panes = _scan_command(pane_command, allow_absent_server=True).decode("utf-8")
        if any(any(selector in line for selector in selectors) for line in panes.splitlines()):
            raise ValueError("Original owner pane command is currently present")
    return {
        "kind": OWNER_KIND,
        "workflow": str(workflow),
        "state": str(job.original_state),
        "selectors": list(selectors),
        "processCommand": list(PS_COMMAND),
        "paneCommand": list(pane_command),
        "matchedOwnerProcesses": [],
        "matchedOwnerPanes": [],
        "historicalActorIdentityRecorded": False,
    }


def _terminal_quiescent(job: Job, *, fresh: bool) -> None:
    observed, age = job.handoff.get("observedAtNs"), job.handoff.get("maximumAgeSeconds")
    if (
        type(observed) is not int
        or type(age) is not int
        or not 1 <= age <= 3600
        or (fresh and not 0 <= time.time_ns() - observed <= age * 1000000000)
    ):
        raise ValueError("Terminal owner handoff observation expired or invalid")
    _, observation = _reference(job.handoff["currentObservation"], {})
    if observation != {"version": 1, "observedAtNs": observed, **_owner_scan(job)}:
        raise ValueError("Terminal owner observation no longer matches actual processes/panes")


def observe_terminal(config: Path, output: Path) -> Path:
    """Seal current owner absence under real leases; never infer a historical PID/birth."""
    output = _path(str(output))
    if not output.is_dir() or tuple(output.iterdir()):
        raise ValueError("Expected existing empty owner observation root")
    job = load_config(config, require_observation=False)
    if job.handoff.get("kind") != TERMINAL_KIND:
        raise ValueError("Owner observation requires the explicit terminal-target draft")
    with ExitStack() as stack:
        stack.enter_context(_unlocked(job.original_state / ".workflow.lock"))
        for stage in job.original_stages:
            if not stack.enter_context(execution_lock(job.original_state, stage.name)):
                raise ValueError("Original future Stage is still owned")
        _unstarted(job)
        observed = time.time_ns()
        observation = {"version": 1, "observedAtNs": observed, **_owner_scan(job)}
        publish(output / "owner-observation.json", encode(observation))
        handoff = {
            **job.handoff,
            "observedAtNs": observed,
            "currentObservation": _pin(output / "owner-observation.json"),
        }
        publish(output / "handoff.json", encode(handoff))
        request = _read(config, {})
        request["handoff"] = _pin(output / "handoff.json")
        publish(output / "config-ready.json", encode(request))
        _unstarted(job)
    return output / "config-ready.json"


def _positive(value: object) -> int:
    if type(value) is not int or value < 1:
        raise ValueError("Expected positive frame budget")
    return value


def _clock_data(clock: VideoFrameClock) -> dict[str, object]:
    return {
        "pts": list(clock.pts),
        "timeBase": str(clock.time_base),
        "lastDuration": clock.last_duration,
    }


def _unstarted(job: Job) -> None:
    for stage in job.original_stages:
        receipt = job.original_state / f"{stage.name}.json"
        if receipt.exists() or receipt.is_symlink():
            raise ValueError(
                "Original chunk has running/partial/complete Stage evidence; preserve its provider"
            )
    for name in (
        "ocr",
        "ocr-batches.json",
        "admission.json",
        "eligibility.json",
        "candidate-config.json",
        "repair",
        "scratch",
    ):
        path = job.chunk.parent / name
        if path.exists() or path.is_symlink():
            raise ValueError("Original chunk is already started or has uncertain partial outputs")


@contextmanager
def _leases(job: Job, *, fresh: bool) -> Iterator[None]:
    terminal = job.handoff.get("kind") == TERMINAL_KIND
    if terminal:
        _terminal_quiescent(job, fresh=fresh)
    else:
        _quiescent(job.handoff, (job.original_state,), fresh=fresh)
    with ExitStack() as stack:
        if terminal:
            stack.enter_context(_unlocked(job.original_state / ".workflow.lock"))
        for stage in job.original_stages:
            if not stack.enter_context(execution_lock(job.original_state, stage.name)):
                raise ValueError("Original chunk Stage is still owned")
        _unstarted(job)
        yield
        _unstarted(job)
        if terminal:
            _terminal_quiescent(job, fresh=False)


def _code() -> dict[str, str]:
    return {
        str(item.path): item.sha256
        for item in code_dependency_inventory.capture(("subtitle_session_cleanup",))
    }


def _command(mode: str, operation: Path) -> list[str]:
    return [sys.executable, "-B", "-m", "replay.subtitle_session_cleanup", mode, str(operation)]


SUPERSEDED_BINDING_NAME = "superseded-consumer-binding.json"


def _reservation_identity(document: dict[str, object]) -> tuple[object, ...]:
    """Attempt identity beyond per-generation artifact paths: plan, chunk and worker pins."""

    def digest(value: object) -> object:
        return record(value).get("sha256")

    worker = record(document.get("originalWorker"))
    return (
        document.get("version"),
        digest(document.get("originalPlan")),
        document.get("chunkIndex"),
        worker.get("state"),
        digest(worker.get("workflow")),
        digest(document.get("ocrProvider")),
    )


def interrupted_reservation(job: Job, config: Path) -> tuple[Path, bytes] | None:
    """Read-only: prove an existing reservation belongs to a verifiably incomplete attempt.

    The attempt's own evidence never leaves its bound work root; only a binding that never
    produced completion.json may be succeeded by a fresh generation.
    """
    marker = job.chunk.parent / "session-consumer-binding.json"
    if not marker.exists() and not marker.is_symlink():
        return None
    if marker.is_symlink():
        raise ValueError("Session reservation must be a regular file")
    evidence = bounded_read(marker)
    pins: dict[str, str] = {}
    binding = _read(marker, pins)
    if set(binding) != {"version", "operation", "workRoot"} or binding.get("version") != 1:
        raise ValueError("Unknown session reservation schema")
    operation_path, operation = _reference(binding["operation"], pins)
    work_root = _path(binding["workRoot"])
    if operation.get("workRoot") != str(work_root):
        raise ValueError("Reservation disagrees with its consumer operation")
    _, prior_config = _reference(operation.get("config"), pins)
    current = record(decode(bounded_read(_path(str(config)))))
    if _reservation_identity(prior_config) != _reservation_identity(current):
        raise ValueError("Reservation belongs to a different cleanup chunk attempt")
    completion = work_root / "completion.json"
    if completion.exists() or completion.is_symlink():
        raise ValueError("Bound session attempt completed; do not supersede it")
    return work_root / SUPERSEDED_BINDING_NAME, evidence


def _replace_verified(path: Path, prior: bytes, data: bytes) -> None:
    """Atomic replacement only after the exact superseded bytes are verified in place."""
    descriptor, temporary = tempfile.mkstemp(prefix=".reservation-", dir=path.parent)
    replaced = False
    try:
        with os.fdopen(descriptor, "wb") as stream:
            stream.write(data)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
        replaced = True
        directory = os.open(path.parent, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW)
        try:
            os.fsync(directory)
        finally:
            os.close(directory)
    finally:
        if not replaced:
            os.unlink(temporary)


def _rebind_interrupted_reservation(
    archive: Path, evidence: bytes, marker: Path, data: bytes
) -> None:
    if archive.exists() or archive.is_symlink():
        if bounded_read(archive) != evidence:
            raise ValueError("Superseded binding evidence differs from the bound reservation")
    else:
        publish(archive, evidence)
    _replace_verified(marker, evidence, data)


def prepare(config: Path, output: Path) -> Prepared:
    """Reserve one genuinely untouched chunk; graph generation does not collect or repair."""
    output = _path(str(output))
    if not output.is_dir() or tuple(output.iterdir()):
        raise ValueError("Expected existing empty distinct purpose work root")
    job = load_config(config)
    code = _code()
    operation, workflow = output / "operation.json", output / "workflow.json"
    marker = job.chunk.parent / "session-consumer-binding.json"
    with _leases(job, fresh=True):
        superseded = None
        if marker.exists() or marker.is_symlink():
            superseded = interrupted_reservation(job, config)
        common = list(dict.fromkeys([str(operation), str(marker), *job.pins, *code]))
        if len(common) > MAX_DIRECT_INPUTS:
            raise ValueError("Session consumer direct inputs exceed workflow budget")
        definitions: list[dict[str, object]] = []
        phases = (
            (
                "collect",
                "verify-ocr",
                ["session/inputs.json", "session/complete.json", "ocr-verification.json"],
            ),
            (
                "admission",
                "verify-admission",
                ["admission.json", "eligibility.json", "candidate-config.json"],
            ),
            (
                "repair",
                "verify-repair",
                ["repair/candidate.mp4", "repair/candidate.json", "assembly-row.json"],
            ),
        )
        previous: list[str] = []
        for mode, verifier, outputs in phases:
            paths = [str(output / name) for name in outputs]
            definitions.append(
                {
                    "name": job.chunk.parent.name + "-session-" + mode,
                    "command": _command(mode, operation),
                    "inputs": [*common, *previous],
                    "outputs": paths,
                    "verify_command": _command(verifier, operation),
                    "timeout_seconds": 3600,
                }
            )
            previous = paths
        stage_graph(parse_stages({"version": 1, "stages": definitions}))
        for name, digest in {**job.pins, **code}.items():
            if (
                name
                not in {
                    str(job.provider.source),
                    str(job.provider.native),
                    *(_asset_names(job.settings)),
                }
                and file_digest(Path(name)) != digest
            ):
                raise ValueError("Metadata/code changed during session preparation")
        publish(
            operation,
            encode(
                {
                    "version": 1,
                    "protocol": PROTOCOL,
                    "config": _pin(config),
                    "workRoot": str(output),
                    "inputs": job.pins,
                    "readerCode": code,
                    "consumerConnected": False,
                    "qualityApproved": False,
                }
            ),
        )
        reservation = encode({"version": 1, "operation": _pin(operation), "workRoot": str(output)})
        if superseded is None:
            publish(marker, reservation)
        else:
            _rebind_interrupted_reservation(superseded[0], superseded[1], marker, reservation)
        publish(workflow, encode({"version": 1, "stages": definitions}))
    return Prepared(operation, workflow, output / "state")


def _asset_names(settings: dict[str, object]) -> tuple[str, ...]:
    return tuple(
        str(record(settings[key])["path"]) for key in ("source", "native", "decoder", "model")
    )


@contextmanager
def _owned(
    operation: Path, *, allow_historical: bool = False
) -> Iterator[tuple[Job, Path, dict[str, str]]]:
    data = _read(operation, {})
    code = _code()
    reader = record(data.get("readerCode"))
    if (
        data.get("version") != 1
        or data.get("protocol") != PROTOCOL
        or (
            reader != code and not (allow_historical and compat.historical_generation(reader, code))
        )
    ):
        raise ValueError("Session cleanup operation/code changed")
    config, _ = _reference(data.get("config"), {})
    job = load_config(config)
    output = _path(data.get("workRoot"))
    if data.get("inputs") != job.pins:
        raise ValueError("Operation no longer binds its actual inputs")
    marker = _read(job.chunk.parent / "session-consumer-binding.json", {})
    if marker != {"version": 1, "operation": _pin(operation), "workRoot": str(output)}:
        raise ValueError("Wrong or changed chunk session reservation")
    marker_path = job.chunk.parent / "session-consumer-binding.json"
    pins = {
        **job.pins,
        **_code(),
        str(operation): file_digest(operation),
        str(marker_path): file_digest(marker_path),
    }
    cache = InputHashCache(tuple(Path(name) for name in pins), allow_uncached_overflow=True)
    cache.prime()
    with _leases(job, fresh=False), input_hash_context(cache):
        for name, digest in pins.items():
            if file_digest(Path(name)) != digest:
                raise ValueError("Session cleanup actual input fingerprint differs")
        yield job, output, pins
        cache.verify()


def _frames(
    job: Job, output: Path
) -> tuple[tuple[FrameObservation, ...], dict[str, object], dict[str, str]]:
    count = session.collect(job.provider_path, output / "session", verify=True)
    pins: dict[str, str] = {}
    producer_path, producer = _reference(_pin(output / "session/inputs.json"), pins)
    complete_path, complete = _reference(_pin(output / "session/complete.json"), pins)
    if (
        producer.get("measurementTool") != session.TOOL
        or producer.get("protocolVersion") != session.PROTOCOL
        or producer.get("source") != str(job.provider.source)
        or producer.get("sourceSHA256") != job.provider.source_sha
        or producer.get("nativeExecutable") != str(job.provider.native)
        or producer.get("nativeSHA256") != job.provider.native_sha
        or producer.get("fullFrame") is not True
        or count != len(job.clock.pts)
    ):
        raise ValueError("Authentic session tool/source/provider coverage differs")
    frames: list[FrameObservation] = []
    blocks: list[dict[str, object]] = []
    boundaries = record(complete.get("boundaries"))
    for name, digest in sorted(boundaries.items()):
        boundary_path, boundary = _reference({"path": name, "sha256": digest}, pins)
        for raw_name, raw_sha in record(boundary.get("raw")).items():
            _pin_reference({"path": raw_name, "sha256": raw_sha}, pins)
        directory = boundary_path.parent / "native"
        block_count = boundary.get("count")
        if type(block_count) is not int or block_count < 0:
            raise ValueError("Invalid verified session boundary count")
        for index in range((block_count + 7) // 8):
            raw = directory / f"block-{index}.json"
            # Raw measurement arrays use the existing bounded parser; they are never relabeled.
            observed = native_ocr_frames(session._read(raw))
            frames.extend(observed)
            blocks.append(
                {
                    "ocr": _pin(raw),
                    "clocks": _pin(directory / f"block-{index}-clocks.json"),
                    "boundary": _pin(boundary_path),
                }
            )
    expected = tuple(float(pts * job.clock.time_base) for pts in job.clock.pts)
    if (
        tuple(frame.requested_time for frame in frames) != expected
        or tuple(frame.actual_time for frame in frames) != expected
    ):
        raise ValueError("Session raw clocks differ from actual physical source PTS")
    evidence = {
        "version": 1,
        "protocol": PROTOCOL,
        "measurementTool": session.TOOL,
        "source": job.settings["source"],
        "sessionPlan": _pin(job.provider_path),
        "sessionInputs": _pin(producer_path),
        "sessionComplete": _pin(complete_path),
        "rawBlocks": blocks,
        "observations": count,
        "candidateOnly": True,
        "consumerConnected": False,
        "removalQualityApproved": False,
    }
    return tuple(frames), evidence, pins


def _admitted(
    job: Job, output: Path
) -> tuple[dict[str, object], dict[str, object], dict[str, object], dict[str, str]]:
    frames, proof, pins = _frames(job, output)
    stored = _read(output / "ocr-verification.json", pins)
    if stored != proof:
        raise ValueError("Session consumer OCR verification changed")
    seeds = _profile(job.settings, pins)
    plan = admission(frames, job.clock, search_profile(seeds), policy=MaskPolicy())
    eligibility = {
        "version": 1,
        "protocol": PROTOCOL,
        "sourceSHA256": job.provider.source_sha,
        "canvas": [job.provider.width, job.provider.height],
        "timeBase": str(job.clock.time_base),
        "decoder": job.settings["decoder"],
        "codec": job.settings["sourceClock"],
        "classificationEvidence": job.settings["classificationEvidence"],
        "ocrVerification": _pin(output / "ocr-verification.json"),
        "frames": plan["frames"],
    }
    native = record(job.settings["native"])
    candidate = {
        "source": str(job.provider.source),
        "sourceSHA256": job.provider.source_sha,
        "nativeExecutable": native["path"],
        "nativeSHA256": native["sha256"],
        "outputDirectory": str(output / "repair"),
        "clock": _clock_data(job.clock),
        "width": job.provider.width,
        "height": job.provider.height,
        "masks": plan["masks"],
        "model": job.settings["model"],
        "eligibility": str(output / "eligibility.json"),
        "protocol": PROTOCOL,
        "ocrProvider": _pin(job.provider_path),
        "localRepairOnly": True,
        "fixedBlur": False,
        "hardwareEncode": False,
    }
    return plan, eligibility, candidate, pins


def _documents(output: Path, values: tuple[dict[str, object], ...], *, verify: bool) -> None:
    for name, value in zip(
        ("admission.json", "eligibility.json", "candidate-config.json"), values, strict=True
    ):
        path = output / name
        if verify:
            if _read(path, {}) != decode(encode(value)):
                raise ValueError("Measured admission/eligibility/candidate contract changed")
        else:
            publish(path, encode(value))


def _capacity(job: Job, output: Path) -> None:
    usage = shutil.disk_usage(output)
    reserve = len(job.clock.pts) * job.provider.width * job.provider.height * 3
    if usage.free < max(30 * 1024**3, usage.total // 20) + reserve:
        raise ValueError("Insufficient capacity for one local repair chunk")


def _repair_runtime(job: Job) -> dict[str, object]:
    """Bind the immutable shared renderer's actual PATH and temporary-file destination."""
    resolved = shutil.which("ffmpeg")
    decoder = record(job.settings["decoder"])
    if resolved is None or str(Path(resolved).resolve()) != decoder["path"]:
        raise ValueError("Actual FFmpeg decoder differs from original pinned settings")
    paths = tuple(os.environ.get(name) for name in ("TMPDIR", "TMP", "TEMP"))
    if any(path is None for path in paths) or len(set(paths)) != 1:
        raise ValueError("Explicit identical checkout-local scratch environment required")
    scratch = _path(paths[0])
    if not scratch.is_dir() or Path(tempfile.gettempdir()).resolve() != scratch:
        raise ValueError("Actual temporary-file scratch escapes the declared checkout directory")
    return {"decoder": decoder, "scratchDirectory": str(scratch)}


def _render(job: Job, output: Path, pins: dict[str, str]) -> None:
    _capacity(job, output)
    runtime = _repair_runtime(job)
    plan, eligibility, candidate, observations = _admitted(job, output)
    _documents(output, (plan, eligibility, candidate), verify=True)
    pins.update(observations)
    frames, _, _ = _frames(job, output)
    spans = repair_spans(
        plan["masks"],
        frames=len(job.clock.pts),
        width=job.provider.width,
        height=job.provider.height,
        source_clock=job.clock,
    )
    entries = {
        int(str(record(row)["sourcePTS"])): record(row) for row in list_rows(eligibility["frames"])
    }
    for span in spans:
        subtitle_eligibility._validate_span(
            span,
            entries=entries,
            observations=frames,
            clock=job.clock,
            width=job.provider.width,
            height=job.provider.height,
        )
    for name in (
        "candidate-config.json",
        "admission.json",
        "eligibility.json",
        "ocr-verification.json",
    ):
        pins[str(output / name)] = file_digest(output / name)
    model = record(job.settings["model"])
    detector = subtitle_detector.LearnedTextDetector(
        Path(str(model["path"])), sha256=_sha(model["sha256"])
    )
    repair = output / "repair"
    repair.mkdir(mode=0o700)
    active = tuple(index for span in spans for index in range(span.start_frame, span.end_frame))
    capture = (
        RepairCapture(repair / "lossless-evidence", sorted({active[0], active[-1]}), job.clock)
        if active
        else None
    )
    counts: list[int] = []

    def detect(images: tuple[Image, ...]) -> tuple[Image, ...]:
        masks = detector.detect(images)
        counts.extend(int(np.count_nonzero(mask)) for mask in masks)
        return masks

    video = repair / "candidate.mp4"
    receipt = render_repaired_video(
        VideoRepairJob(
            job.provider.source,
            video,
            job.provider.width,
            job.provider.height,
            len(job.clock.pts),
            spans,
            source_clock=job.clock,
        ),
        detect_batch=detect,
        evidence=capture,
    )
    if len(counts) != len(active):
        raise ValueError("Incomplete local glyph detection accounting")
    for name, digest in pins.items():
        if file_digest(Path(name)) != digest:
            raise ValueError("Local repair inputs changed")
    result: dict[str, object] = {
        "protocol": PROTOCOL,
        "inputs": pins,
        "videoSHA256": file_digest(video),
        "render": asdict(receipt),
        "sourceStart": str(job.clock.pts[0] * job.clock.time_base),
        "detectionBackend": "ppocrv3-db",
        "detectedSupportPixels": counts,
        "activeSourceFrames": list(active),
        "unknownSourcePTS": plan["unknownSourcePTS"],
        "inactiveFramesAreUnreviewedNotApproved": True,
        "qualityApproval": False,
        "fullDecodeVerified": False,
        "deliveryComplete": False,
        "consumerConnected": False,
        "runtime": runtime,
    }
    if capture is not None:
        result["losslessEvidence"] = capture.manifest()
    publish(repair / "candidate.json", encode(result))
    publish(output / "assembly-row.json", encode(_row(job, output)))


def list_rows(value: object) -> list[object]:
    if not isinstance(value, list):
        raise ValueError("Expected bounded frame list")
    return list(value)


def _row(job: Job, output: Path) -> dict[str, object]:
    return {
        "media": str(output / "repair/candidate.mp4"),
        "sourceFirstPTS": job.clock.pts[0],
        "sourceEndPTS": job.clock.pts[-1] + job.clock.last_duration,
        "frames": len(job.clock.pts),
        "timeBase": str(job.clock.time_base),
        "producer": _pin(output / "candidate-config.json"),
        "candidate": _pin(output / "repair/candidate.json"),
        "originalChunk": _pin(job.chunk),
        "actualSourceAssemblyVerified": False,
        "consumerConnected": False,
    }


def _session_assembly_row(
    operation: Path,
    *,
    original: Path,
    clock: VideoFrameClock,
    pins: dict[str, str],
    allow_historical: bool = False,
) -> tuple[dict[str, object], dict[str, object], tuple[int, int]]:
    data = _read(operation, pins)
    config, _ = _reference(data.get("config"), pins)
    job = load_config(config)
    assets = {str(job.provider.source), str(job.provider.native), *_asset_names(job.settings)}
    pins.update({name: digest for name, digest in job.pins.items() if name not in assets})
    output = _path(data.get("workRoot"))
    code = _code()
    reader = record(data.get("readerCode"))
    if (
        data.get("version") != 1
        or data.get("protocol") != PROTOCOL
        or (
            reader != code and not (allow_historical and compat.historical_generation(reader, code))
        )
        or data.get("inputs") != job.pins
        or job.original_plan != original
        or job.clock != clock
    ):
        raise ValueError("Session assembly selection differs from its original physical chunk")
    marker = _read(job.chunk.parent / "session-consumer-binding.json", pins)
    if marker != {"version": 1, "operation": _pin(operation), "workRoot": str(output)}:
        raise ValueError("Session assembly requires its exact original namespace reservation")
    producer_path, receipt_path = output / "candidate-config.json", output / "repair/candidate.json"
    producer, receipt = _read(producer_path, pins), _read(receipt_path, pins)
    native = record(job.settings["native"])
    bound = record(receipt.get("inputs"))
    if (
        producer.get("protocol") != PROTOCOL
        or producer.get("clock") != _clock_data(clock)
        or producer.get("source") != str(job.provider.source)
        or producer.get("sourceSHA256") != job.provider.source_sha
        or producer.get("nativeExecutable") != native["path"]
        or producer.get("nativeSHA256") != native["sha256"]
        or producer.get("outputDirectory") != str(output / "repair")
        or producer.get("localRepairOnly") is not True
        or producer.get("fixedBlur") is not False
        or producer.get("ocrProvider") != _pin(job.provider_path)
        or receipt.get("protocol") != PROTOCOL
        or receipt.get("sourceStart") != str(clock.pts[0] * clock.time_base)
        or record(receipt.get("render")).get("frames") != len(clock.pts)
        or bound.get(str(producer_path)) != pins[str(producer_path)]
        or bound.get(str(job.provider.source)) != job.provider.source_sha
        or bound.get(str(native["path"])) != native["sha256"]
    ):
        raise ValueError("Session local-repair producer/result source/native/clock binding differs")
    _asset({"path": str(output / "repair/candidate.mp4"), "sha256": receipt.get("videoSHA256")}, {})
    row = _read(output / "assembly-row.json", pins)
    if row != _row(job, output):
        raise ValueError("Session local-repair assembly row differs")
    refs: dict[str, object] = {
        "producer": {"path": str(producer_path), "sha256": pins[str(producer_path)]},
        "candidate": {"path": str(receipt_path), "sha256": pins[str(receipt_path)]},
    }
    return row, refs, (job.provider.width, job.provider.height)


def prepare_assembly(config: Path, output: Path, *, allow_historical: bool = False) -> Path:
    """Prepare one ordered legacy/session manifest; actual assembly remains mandatory.

    This bounded metadata operation does not hash media or assert Stage completion.
    The normal producer workflows/domain verifiers must finish before the manifest
    is used by delivery_source_assembly, which verifies every chunk's actual bytes,
    all decoded PTS and compressed packets. Existing producer evidence is unchanged.
    """
    output = _path(str(output))
    if not output.is_dir() or tuple(output.iterdir()):
        raise ValueError("Expected existing empty distinct assembly preparation root")
    pins: dict[str, str] = {}
    raw = _read(config, pins)
    if (
        set(raw) != {"version", "originalPlan", "sessions"}
        or type(raw.get("version")) is not int
        or raw["version"] != 1
    ):
        raise ValueError("Invalid mixed cleanup assembly config")
    original, plan = _reference(raw["originalPlan"], pins)
    settings_path, settings = _reference(plan.get("settings"), pins)
    clock_path, source_clock = _reference(settings.get("sourceClock"), pins)
    clocks = clock_chunks(source_clock, maximum_frames=_positive(settings.get("chunkFrames")))
    if (
        type(plan.get("version")) is not int
        or plan["version"] != 1
        or type(settings.get("version")) is not int
        or settings["version"] != 1
        or plan.get("wholeSourcePlanned") is not True
        or plan.get("chunks") != len(clocks)
        or len(clocks) > MAX_CHUNKS
        or plan.get("plannedFrames") != sum(len(clock.pts) for clock in clocks)
        or plan.get("totalSourceFrames") != plan["plannedFrames"]
        or clocks[0].pts[0] != 0
    ):
        raise ValueError("Mixed assembly requires the complete actual physical source plan")
    source = record(settings.get("source"))
    native = record(settings.get("native"))
    source_identity = (str(_path(source.get("path"), local=False)), _sha(source.get("sha256")))
    native_identity = (str(_path(native.get("path"), local=False)), _sha(native.get("sha256")))
    selected: dict[int, Path] = {}
    selections = list_rows(raw["sessions"])
    if len(selections) > len(clocks):
        raise ValueError("Too many session assembly selections")
    for value in selections:
        selection = record(value)
        index = selection.get("chunkIndex")
        if (
            set(selection) != {"chunkIndex", "operation"}
            or type(index) is not int
            or not 0 <= index < len(clocks)
            or index in selected
        ):
            raise ValueError("Session assembly membership must be unique physical chunk indices")
        selected[index] = _pin_reference(selection["operation"], pins)
    rows: list[dict[str, object]] = []
    refs: list[dict[str, object]] = []
    canvases: set[tuple[int, int]] = set()
    for index, clock in enumerate(clocks):
        if index in selected:
            row, ref, canvas = _session_assembly_row(
                selected[index],
                original=original,
                clock=clock,
                pins=pins,
                allow_historical=allow_historical,
            )
        else:
            row, ref, canvas = legacy_assembly._chunk(
                original.parent / f"chunk-{index:05d}",
                clock,
                settings_path,
                source_identity,
                native_identity,
                pins,
            )
        rows.append(row)
        refs.append(ref)
        canvases.add(canvas)
    if len(canvases) != 1:
        raise ValueError("Mixed cleanup assembly geometry differs")
    code = _code()
    for name, digest in pins.items():
        if hashlib.sha256(bounded_read(Path(name))).hexdigest() != digest:
            raise ValueError("Mixed assembly metadata changed during preparation")
    if code != _code():
        raise ValueError("Mixed assembly preparation code changed")
    index_path, manifest_path = output / "index.json", output / "manifest.json"
    publish(
        index_path,
        encode(
            {"purpose": "mixed cleanup source-clock candidates; quality unapproved", "ranges": rows}
        ),
    )
    publish(manifest_path, encode({"version": 1, "index": _pin(index_path), "chunks": refs}))
    publish(
        output / "preparation.json",
        encode(
            {
                "version": 1,
                "protocol": PROTOCOL,
                "inputs": pins,
                "currentPreparationCode": code,
                "sourceIdentity": settings["source"],
                "sourceClock": {"path": str(clock_path), "sha256": pins[str(clock_path)]},
                "physicalSourceFrames": sum(len(clock.pts) for clock in clocks),
                "physicalSourceTerminalPTS": clocks[-1].pts[-1] + clocks[-1].last_duration,
                "sessionChunkIndices": sorted(selected),
                "manifest": _pin(manifest_path),
                "exactPreparedSourceClockVerified": True,
                "stageFingerprintRecomputed": False,
                "mediaBytesVerified": False,
                "consumerConnected": False,
                "qualityApproved": False,
                "deliveryComplete": False,
                "requiredNextStage": (
                    "finished producer workflows/domain verification; "
                    "delivery_source_assembly all-PTS/packets/full AV verification"
                ),
            }
        ),
    )
    return manifest_path


def run(operation: Path, *, mode: str) -> None:
    """One ordinary phase or read-only verifier under original namespace leases."""
    allowed = {"collect", "verify-ocr", "admission", "verify-admission", "repair", "verify-repair"}
    if mode not in allowed:
        raise ValueError("Unknown session cleanup phase")
    historical = mode in VERIFY_MODES
    with _owned(operation, allow_historical=historical) as (job, output, pins):
        if mode == "collect":
            session.collect(job.provider_path, output / "session")
            _, proof, _ = _frames(job, output)
            publish(output / "ocr-verification.json", encode(proof))
        elif mode == "verify-ocr":
            _, proof, _ = _frames(job, output)
            if _read(output / "ocr-verification.json", {}) != proof:
                raise ValueError("Session OCR proof changed")
        elif mode in {"admission", "verify-admission"}:
            plan, eligibility, candidate, _ = _admitted(job, output)
            _documents(output, (plan, eligibility, candidate), verify=mode == "verify-admission")
        elif mode == "repair":
            _render(job, output, pins)
        else:
            result = _read(output / "repair/candidate.json", {})
            if (
                result.get("protocol") != PROTOCOL
                or record(result.get("render")).get("frames") != len(job.clock.pts)
                or result.get("sourceStart") != str(job.clock.pts[0] * job.clock.time_base)
            ):
                raise ValueError("Local repair receipt clock/frame contract differs")
            if result.get("videoSHA256") != file_digest(output / "repair/candidate.mp4"):
                raise ValueError("Local repaired media changed")
            for name, digest in record(result.get("inputs")).items():
                current_digest = file_digest(_path(name, local=False))
                historical = compat.accepted_input_digest(name, digest, current_digest)
                if current_digest != _sha(digest) and not historical:
                    raise ValueError("Local repair receipt input changed")
            plan, eligibility, candidate, _ = _admitted(job, output)
            _documents(output, (plan, eligibility, candidate), verify=True)
            if _read(output / "assembly-row.json", {}) != _row(job, output):
                raise ValueError("Local assembly row changed")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "mode",
        choices=(
            "prepare",
            "prepare-assembly",
            "observe-terminal",
            "collect",
            "verify-ocr",
            "admission",
            "verify-admission",
            "repair",
            "verify-repair",
        ),
    )
    parser.add_argument("input", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args(argv)
    if args.mode in {"prepare", "prepare-assembly", "observe-terminal"}:
        if args.output is None:
            raise ValueError("Preparation requires a distinct existing --output work directory")
        if args.mode == "prepare-assembly":
            print(prepare_assembly(args.input, args.output))
        elif args.mode == "observe-terminal":
            print(observe_terminal(args.input, args.output))
        else:
            prepared = prepare(args.input, args.output)
            print(prepared.workflow)
    else:
        if args.output is not None:
            raise ValueError("Runtime uses only its pinned operation output root")
        run(args.input, mode=args.mode)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
