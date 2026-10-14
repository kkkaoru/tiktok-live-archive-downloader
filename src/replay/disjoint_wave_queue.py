"""Run disjoint, pinned workflows serially through the normal admission gate.

No producer backend or admission arithmetic lives here. A failed/uncertain wave
stops the queue, retaining its original stage state and exclusive admission lease.
Only actual completed-peer captures release reservations; no future PID is guessed.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import signal
import subprocess
import sys
import time
from collections.abc import Iterator
from concurrent.futures import ThreadPoolExecutor, as_completed
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from types import FrameType

from replay import completion_barrier, disk_budget, workflow
from replay import measured_job_admission as admission
from replay.core import ReplayError
from replay.native_invocation import canonical, durable, pin, require, rows
from replay.process_family_observation import SystemHost as ProcessHost
from replay.recordings import record
from replay.stages import StageError, execution_lock

ROOT = Path(__file__).resolve().parents[2]
WORK = ROOT / "edits/work"
MAX_JSON_BYTES = 1024 * 1024
MAX_WAVES = 128
MAX_SOURCES = 512


@dataclass(frozen=True)
class Reference:
    path: Path
    sha256: str


@dataclass(frozen=True)
class Wave:
    name: str
    request: admission.Request
    workflow: Reference
    state: Path
    peers: tuple[Reference, ...]
    sandbox_profile: Reference | None = None
    python: admission.Executable | None = None


@dataclass(frozen=True)
class Mutex:
    directory: Path
    name: str
    wait_seconds: int
    poll_seconds: int


@dataclass(frozen=True)
class Plan:
    reference: Reference
    waves: tuple[Wave, ...]
    sources: tuple[Reference, ...]
    mutex: Mutex
    minimum_boot_free_bytes: int
    barriers: tuple[Reference, ...] = ()


@dataclass(frozen=True)
class GroupLane:
    workflow: Reference
    state: Path


@dataclass(frozen=True)
class Group:
    reference: Reference
    lanes: tuple[GroupLane, ...]
    workers: int
    output: Path
    sources: tuple[Reference, ...]


def _integer(value: object, *, minimum: int, maximum: int = 2**63 - 1) -> int:
    if type(value) is not int or not minimum <= value <= maximum:
        raise ReplayError("Invalid bounded queue integer")
    return value


def _name(value: object) -> str:
    if not isinstance(value, str) or admission.SAFE_NAME.fullmatch(value) is None:
        raise ReplayError("Invalid queue name")
    return value


def _lock_name(value: object) -> str:
    if (
        not isinstance(value, str)
        or re.fullmatch(r"[a-z0-9_.-]{1,64}", value) is None
        or value in {".", ".."}
    ):
        raise ReplayError("Invalid cooperative mutex leaf name")
    return value


def _path(value: object, *, work: bool = True) -> Path:
    path = canonical(value)
    require(path.is_relative_to(WORK if work else ROOT), "Queue path escapes declared scope")
    return path


def _reference(value: object, *, work: bool = True) -> Reference:
    raw = record(value)
    require(set(raw) == {"path", "sha256"}, "Invalid queue reference")
    digest = raw["sha256"]
    require(
        isinstance(digest, str)
        and len(digest) == 64
        and all(char in "0123456789abcdef" for char in digest),
        "Invalid queue digest",
    )
    if not isinstance(digest, str):
        raise ReplayError("Expected queue digest")
    result = Reference(_path(raw["path"], work=work), digest)
    require(pin(result.path)["sha256"] == digest, "Pinned queue input changed")
    return result


def _unique(pairs: list[tuple[str, object]]) -> dict[str, object]:
    result: dict[str, object] = {}
    for key, value in pairs:
        require(key not in result, "Duplicate queue JSON key")
        result[key] = value
    return result


def _read(path: Path) -> dict[str, object]:
    _path(str(path))
    require(0 < path.stat().st_size <= MAX_JSON_BYTES, "Invalid queue metadata size")
    with path.open("rb") as stream:
        data = stream.read(MAX_JSON_BYTES + 1)
    require(len(data) <= MAX_JSON_BYTES, "Queue metadata exceeded size bound")
    return record(json.loads(data, object_pairs_hook=_unique))


def _wave(value: object) -> Wave:
    raw = record(value)
    required = {"name", "admission", "workflow", "stateDirectory", "completedPeers"}
    require(
        set(raw) in (required, required | {"sandboxProfile", "pythonExecutable"}),
        "Invalid wave schema",
    )
    request_ref = _reference(raw["admission"])
    request = admission.load(request_ref.path, request_ref.sha256)
    manifest = _reference(raw["workflow"])
    state = _path(raw["stateDirectory"])
    peers = tuple(_reference(item) for item in rows(raw["completedPeers"]))
    require(len(peers) <= admission.MAX_FAMILIES, "Excessive completed peer captures")
    expected = {
        (family.completion.capture, family.completion.sha256)
        for family in request.families
        if family.completion is not None
    }
    require(
        {(item.path, item.sha256) for item in peers} == expected and len(peers) == len(expected),
        "Completed peer bindings differ from admission families",
    )
    sandbox = _reference(raw["sandboxProfile"]) if "sandboxProfile" in raw else None
    python = (
        admission._executable(raw["pythonExecutable"], fresh=True)
        if "pythonExecutable" in raw
        else None
    )
    prefix: tuple[str, ...] = (str(request.executable.invocation),)
    if sandbox is not None and python is not None:
        require(
            request.executable.path == Path("/usr/bin/sandbox-exec")
            and request.executable.invocation == request.executable.path,
            "Only the pinned canonical sandbox-exec wrapper is supported",
        )
        prefix = (*prefix, "-f", str(sandbox.path), str(python.invocation))
    command = (
        *prefix,
        "-B",
        "-m",
        "replay.workflow",
        str(manifest.path),
        "--state",
        str(state),
    )
    require(request.command == command, "Queue admission must run the exact normal workflow")
    require(
        workflow.main([str(manifest.path), "--state", str(state), "--check"]) == 0,
        "Wave dependency preflight failed",
    )
    stages = workflow.load_stages(manifest.path)
    require(all(stage.verify_command for stage in stages), "Wave needs every domain verifier")
    return Wave(_name(raw["name"]), request, manifest, state, peers, sandbox, python)


def load(path: Path, sha256: str) -> Plan:
    """Validate pinned source generation and disjoint normal workflow ownership."""
    reference = _reference({"path": str(path), "sha256": sha256})
    raw = _read(reference.path)
    required = {"version", "kind", "waves", "sources", "mutex", "minimumBootFreeBytes"}
    require(
        set(raw) in (required, required | {"barriers"})
        and type(raw["version"]) is int
        and raw["version"] == 1
        and raw["kind"] == "disjoint-wave-queue",
        "Invalid queue schema/version",
    )
    entries = rows(raw["waves"])
    require(0 < len(entries) <= MAX_WAVES, "Invalid wave count")
    waves = tuple(_wave(item) for item in entries)
    require(len({wave.name for wave in waves}) == len(waves), "Duplicate wave name")
    require(
        len({(wave.request.work, wave.request.name) for wave in waves}) == len(waves),
        "Admission leases must be distinct",
    )
    states = tuple(wave.state for wave in waves)
    outputs = tuple(
        path.resolve()
        for wave in waves
        for stage in workflow.load_stages(wave.workflow.path)
        for path in stage.outputs
    )
    _disjoint(states, "Workflow states overlap")
    _disjoint(outputs, "Wave producer outputs overlap")
    sources = tuple(_reference(item, work=False) for item in rows(raw["sources"]))
    require(0 < len(sources) <= MAX_SOURCES, "Invalid fixed source generation")
    require(len({source.path for source in sources}) == len(sources), "Duplicate source pin")
    mutex = record(raw["mutex"])
    require(
        set(mutex) == {"directory", "name", "waitSeconds", "pollSeconds"},
        "Invalid shared priority mutex schema",
    )
    directory = _path(mutex["directory"])
    require(directory.is_dir(), "Shared mutex directory must exist")
    barriers = tuple(_reference(item) for item in rows(raw.get("barriers", [])))
    require(len(barriers) <= 16, "Excessive predecessor barriers")
    require(len({item.path for item in barriers}) == len(barriers), "Duplicate predecessor barrier")
    for barrier in barriers:
        completion_barrier.load(barrier.path, barrier.sha256)
    result = Plan(
        reference,
        waves,
        sources,
        Mutex(
            directory,
            _lock_name(mutex["name"]),
            _integer(mutex["waitSeconds"], minimum=0, maximum=3600),
            _integer(mutex["pollSeconds"], minimum=1, maximum=2),
        ),
        _integer(raw["minimumBootFreeBytes"], minimum=1),
        barriers,
    )
    require(pin(reference.path)["sha256"] == reference.sha256, "Queue changed during load")
    return result


def _disjoint(paths: tuple[Path, ...], message: str) -> None:
    for index, path in enumerate(paths):
        require(
            all(
                not path.is_relative_to(other) and not other.is_relative_to(path)
                for other in paths[:index]
            ),
            message,
        )


def _pins(plan: Plan) -> list[dict[str, object]]:
    references = (
        plan.reference,
        *plan.barriers,
        *plan.sources,
        *(wave.workflow for wave in plan.waves),
        *(Reference(wave.request.path, wave.request.sha256) for wave in plan.waves),
        *(peer for wave in plan.waves for peer in wave.peers),
        *(wave.sandbox_profile for wave in plan.waves if wave.sandbox_profile is not None),
    )
    result: list[dict[str, object]] = []
    for reference in references:
        observed = pin(reference.path)
        require(observed["sha256"] == reference.sha256, "Queue source generation changed")
        result.append(observed)
    result.append(pin(Path(__file__).resolve()))
    for wave in plan.waves:
        if wave.python is not None:
            executable = wave.python
            checked_tool = admission._executable(
                {
                    "path": str(executable.path),
                    "sha256": executable.sha256,
                    "invocationPath": str(executable.invocation),
                    "linkTarget": executable.link_target,
                },
                fresh=True,
            )
            require(checked_tool == executable, "Pinned wrapper interpreter changed")
            result.append(pin(executable.path))
    return result


def load_group(path: Path, sha256: str) -> Group:
    """Bind pre-split lanes; this does not launch or adopt any workflow."""
    reference = _reference({"path": str(path), "sha256": sha256})
    raw = _read(reference.path)
    require(
        set(raw) == {"version", "kind", "lanes", "maximumWorkers", "outputDirectory", "sources"}
        and type(raw["version"]) is int
        and raw["version"] == 1
        and raw["kind"] == "parallel-workflow-group",
        "Invalid parallel workflow group schema",
    )
    entries = rows(raw["lanes"])
    require(0 < len(entries) <= 16, "Invalid parallel lane count")
    lanes: list[GroupLane] = []
    outputs: list[Path] = []
    total_stages = 0
    for entry in entries:
        item = record(entry)
        require(set(item) == {"workflow", "stateDirectory"}, "Invalid group lane schema")
        manifest = _reference(item["workflow"])
        state = _path(item["stateDirectory"])
        require(
            workflow.main([str(manifest.path), "--state", str(state), "--check"]) == 0,
            "Parallel lane dependency preflight failed",
        )
        stages = workflow.load_stages(manifest.path)
        require(all(stage.verify_command for stage in stages), "Every lane needs domain verifiers")
        total_stages += len(stages)
        outputs.extend(path.resolve() for stage in stages for path in stage.outputs)
        lanes.append(GroupLane(manifest, state))
    require(total_stages <= workflow.MAX_STAGES, "Parallel group exceeds bounded stage count")
    output = _path(raw["outputDirectory"])
    require(output.parent.is_dir(), "Group output parent must exist")
    _disjoint((output, *(lane.state for lane in lanes)), "Group journal/state overlap")
    _disjoint(tuple(outputs), "Parallel producer outputs overlap")
    sources = tuple(_reference(item, work=False) for item in rows(raw["sources"]))
    require(0 < len(sources) <= MAX_SOURCES, "Invalid group source generation")
    require(len({item.path for item in sources}) == len(sources), "Duplicate group source")
    return Group(
        reference,
        tuple(lanes),
        _integer(raw["maximumWorkers"], minimum=1, maximum=len(lanes)),
        output,
        sources,
    )


def _group_pins(group: Group) -> list[dict[str, object]]:
    result: list[dict[str, object]] = []
    for reference in (group.reference, *group.sources, *(lane.workflow for lane in group.lanes)):
        observed = pin(reference.path)
        require(observed["sha256"] == reference.sha256, "Parallel group generation changed")
        result.append(observed)
    result.append(pin(Path(__file__).resolve()))
    return result


def _lane_run(group: Group, index: int) -> dict[str, object]:
    lane = group.lanes[index]
    started = time.time_ns()
    error_kind: str | None = None
    try:
        workflow.run_plan(lane.workflow.path, state_directory=lane.state)
    except (StageError, ReplayError, OSError, ValueError, subprocess.SubprocessError) as error:
        error_kind = type(error).__name__
    result: dict[str, object] = {
        "index": index,
        "workflow": {"path": str(lane.workflow.path), "sha256": lane.workflow.sha256},
        "stateDirectory": str(lane.state),
        "startedAtNs": started,
        "endedAtNs": time.time_ns(),
        "status": "complete" if error_kind is None else "failed",
        "errorKind": error_kind,
    }
    if error_kind is None:
        result["checkpoints"] = [
            pin(lane.state / f"{stage.name}.json")
            for stage in workflow.load_stages(lane.workflow.path)
        ]
    durable(group.output / f"lane-{index:04d}.json", result)
    return result


def run_group(group: Group) -> int:
    """Wait for every owned lane; workflow owns subprocesses, deadlines and reuse.

    run_plan has no signal/environment mutation. Its states/atomic writes are
    disjoint and input_hash_cache uses a ContextVar, not a shared active cache.
    We do not call workflow.main from threads or activate a cross-thread cache.
    """
    require(not group.output.exists(), "Parallel group output already occupied")
    source_pins = _group_pins(group)
    process = ProcessHost()
    producer_pid = os.getpid()
    producer_birth = process.birth(producer_pid)
    interpreter_pin = pin(Path(sys.executable).resolve(strict=True))
    group.output.mkdir(mode=0o700)
    durable(
        group.output / "start.json",
        {
            "startedAtNs": time.time_ns(),
            "pid": producer_pid,
            "birth": producer_birth,
            "interpreter": interpreter_pin,
            "pins": source_pins,
        },
    )
    results: dict[int, dict[str, object]] = {}
    with ThreadPoolExecutor(max_workers=group.workers) as executor:
        futures = {
            executor.submit(_lane_run, group, index): index for index in range(len(group.lanes))
        }
        for future in as_completed(futures):
            index = futures[future]
            results[index] = future.result()
    require(_group_pins(group) == source_pins, "Parallel group generation changed during execution")
    require(process.birth(producer_pid) == producer_birth, "Group producer identity changed")
    require(
        pin(Path(sys.executable).resolve(strict=True)) == interpreter_pin,
        "Group interpreter changed during execution",
    )
    completed = all(result["status"] == "complete" for result in results.values())
    durable(
        group.output / "result.json",
        {
            "version": 1,
            "kind": "parallel-workflow-group-result",
            "group": {"path": str(group.reference.path), "sha256": group.reference.sha256},
            "pins": source_pins,
            "lanes": [results[index] for index in range(len(group.lanes))],
            "allOwnedLanesWaited": True,
            "completed": completed,
            "endedAtNs": time.time_ns(),
        },
    )
    return 0 if completed else 1


def verify_group(group: Group) -> None:
    """Verify immutable normal checkpoints after all original domain gates passed."""
    result = _read(group.output / "result.json")
    require(
        type(result.get("version")) is int
        and result.get("version") == 1
        and result.get("kind") == "parallel-workflow-group-result"
        and result.get("completed") is True
        and result.get("allOwnedLanesWaited") is True
        and result.get("group")
        == {"path": str(group.reference.path), "sha256": group.reference.sha256}
        and result.get("pins") == _group_pins(group),
        "Parallel group lacks verified complete immutable evidence",
    )
    lanes = rows(result.get("lanes"))
    require(len(lanes) == len(group.lanes), "Incomplete group lane results")
    for index, lane in enumerate(group.lanes):
        saved = record(lanes[index])
        require(
            type(saved.get("index")) is int
            and saved.get("index") == index
            and saved.get("workflow")
            == {"path": str(lane.workflow.path), "sha256": lane.workflow.sha256}
            and saved.get("stateDirectory") == str(lane.state)
            and saved.get("status") == "complete"
            and saved.get("errorKind") is None,
            "Parallel lane lacks complete normal outcome",
        )
        checkpoints = [
            pin(lane.state / f"{stage.name}.json")
            for stage in workflow.load_stages(lane.workflow.path)
        ]
        require(saved.get("checkpoints") == checkpoints, "Parallel lane normal checkpoints changed")
        require(
            all(_read(Path(str(item["path"]))).get("status") == "complete" for item in checkpoints),
            "Parallel lane normal checkpoint is incomplete",
        )


def _cancel_group(_signum: int, _frame: FrameType | None) -> None:
    raise StageError("Parallel group cancelled; owned lanes are waited and partial results remain")


def _peers(wave: Wave) -> list[dict[str, object]]:
    return [admission.verify_completion_capture(peer.path, peer.sha256) for peer in wave.peers]


def _release(wave: Wave) -> dict[str, object]:
    end = _read(wave.request.work / f"{wave.request.name}.end.json")
    require(
        end.get("kind") == "measured-admission-end"
        and end.get("requestSHA256") == wave.request.sha256
        and end.get("reservationReleased") is True
        and end.get("completed") is True
        and end.get("dispatchAttempted") is True
        and end.get("uncertainLaunch") is False
        and type(end.get("exitCode")) is int
        and end.get("exitCode") == 0
        and end.get("errorKind") is None,
        "Prior wave lacks actual successful normal admission release",
    )
    _integer(end.get("endedAtNs"), minimum=1)
    return end


@contextmanager
def _mutex(mutex: Mutex) -> Iterator[bool]:
    deadline = time.monotonic() + mutex.wait_seconds
    while True:
        with execution_lock(mutex.directory, mutex.name) as acquired:
            if acquired:
                yield True
                return
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            yield False
            return
        time.sleep(min(mutex.poll_seconds, remaining))


def run(plan: Plan, output: Path) -> int:
    """Dispatch serially; admission and workflow own all resource/domain decisions."""
    output = _path(str(output))
    require(output.parent.is_dir() and not output.exists(), "Queue output already occupied")
    _disjoint(
        (output, plan.mutex.directory, *(wave.state for wave in plan.waves)),
        "Queue journal overlaps mutex/workflow state",
    )
    source_pins = _pins(plan)
    output.mkdir(mode=0o700)
    durable(output / "queue-start.json", {"startedAtNs": time.time_ns(), "pins": source_pins})
    for index, barrier in enumerate(plan.barriers):
        barrier_request = completion_barrier.load(barrier.path, barrier.sha256)
        proof = completion_barrier.wait(barrier_request)
        completion_barrier.verify(barrier_request)
        require(_pins(plan) == source_pins, "Queue generation changed while awaiting predecessor")
        durable(
            output / f"predecessor-{index:04d}-release.json",
            {"observedAtNs": time.time_ns(), "request": pin(barrier.path), "proof": pin(proof)},
        )
    previous: dict[str, object] | None = None
    previous_pin: dict[str, object] | None = None
    for index, wave in enumerate(plan.waves):
        require(_pins(plan) == source_pins, "Queue generation changed before wave")
        peer_releases = _peers(wave)
        if index:
            prior = plan.waves[index - 1]
            require(
                pin(prior.request.work / f"{prior.request.name}.end.json") == previous_pin,
                "Prior normal release evidence changed",
            )
            previous = _release(prior)
        with _mutex(plan.mutex) as acquired:
            if not acquired:
                durable(
                    output / f"wave-{index:04d}-held.json",
                    {"reason": "priority-mutex", "exitCode": 75},
                )
                return 75
            require(_pins(plan) == source_pins, "Queue generation changed while waiting")
            free = disk_budget.measure(ROOT).free
            require(free >= plan.minimum_boot_free_bytes, "Boot free-space start gate failed")
            request = admission.load(wave.request.path, wave.request.sha256)
            require(request == wave.request, "Wave admission request changed")
            started = time.time_ns()
            durable(
                output / f"wave-{index:04d}-start.json",
                {
                    "name": wave.name,
                    "startedAtNs": started,
                    "bootFreeBytes": free,
                    "peerReleases": peer_releases,
                },
            )
            code = admission.run(request)
        require(_pins(plan) == source_pins, "Queue generation changed during wave")
        evidence: dict[str, object] = {
            "name": wave.name,
            "exitCode": code,
            "endedAtNs": time.time_ns(),
        }
        if code == 0:
            release = _release(wave)
            evidence["release"] = release
            previous_pin = pin(wave.request.work / f"{wave.request.name}.end.json")
            evidence["releasePin"] = previous_pin
            intent_path = wave.request.work / f"{wave.request.name}.launch-intent.json"
            intent = _read(intent_path)
            launched = _integer(intent.get("startedAtNs"), minimum=1)
            evidence["launchIntentPin"] = pin(intent_path)
            evidence["dispatchAtNs"] = launched
            evidence["gapSincePreviousReleaseNs"] = (
                None if previous is None else launched - _integer(previous["endedAtNs"], minimum=1)
            )
        durable(output / f"wave-{index:04d}-end.json", evidence)
        if code:
            return code
    durable(
        output / "queue-complete.json",
        {"endedAtNs": time.time_ns(), "waves": len(plan.waves), "pins": source_pins},
    )
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("check", "run", "group", "verify-group"))
    parser.add_argument("plan", type=Path)
    parser.add_argument("--sha256", required=True)
    parser.add_argument("--output", type=Path)
    arguments = parser.parse_args(argv)
    try:
        if arguments.mode in {"group", "verify-group"}:
            group = load_group(arguments.plan, arguments.sha256)
            if arguments.mode == "verify-group":
                verify_group(group)
                return 0
            previous = signal.signal(signal.SIGTERM, _cancel_group)
            try:
                return run_group(group)
            finally:
                signal.signal(signal.SIGTERM, previous)
        plan = load(arguments.plan, arguments.sha256)
        if arguments.mode == "check":
            _pins(plan)
            print(json.dumps({"waves": len(plan.waves), "sourceGenerationFixed": True}))
            return 0
        require(arguments.output is not None, "Queue run requires exclusive output")
        return run(plan, arguments.output)
    except (ReplayError, StageError, OSError, ValueError) as error:
        print(str(error), file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
