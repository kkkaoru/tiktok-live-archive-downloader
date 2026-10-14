"""Plan disjoint session cleanup drafts and time ordinary consumer stages.

This metadata-only factory never observes owners, acquires a cleanup lease or
starts OCR/repair. Each draft still requires the consumer's fresh owner observer
and normal exclusive preparation. Existing original/session work is preserved.
"""

from __future__ import annotations

import argparse
import hashlib
import re
import sys
from dataclasses import dataclass
from pathlib import Path

from replay import code_dependency_inventory
from replay import subtitle_session_cleanup as consumer
from replay.core import ReplayError
from replay.recordings import record
from replay.sequence_journal import bounded_read, decode, encode, publish
from replay.subtitle_cleanup_plan import clock_chunks
from replay.workflow import load_stages, parse_stages, stage_graph

PROTOCOL = "source-caption-session-drafts-v1"
MAX_CHUNKS = 512
MODES = (("collect", "verify-ocr"), ("admission", "verify-admission"), ("repair", "verify-repair"))
PHASE_OUTPUTS = {
    "collect": ("session/inputs.json", "session/complete.json", "ocr-verification.json"),
    "admission": ("admission.json", "eligibility.json", "candidate-config.json"),
    "repair": ("repair/candidate.mp4", "repair/candidate.json", "assembly-row.json"),
}
UNSTARTED_NAMES = (
    "ocr",
    "ocr-batches.json",
    "admission.json",
    "eligibility.json",
    "candidate-config.json",
    "repair",
    "scratch",
    "session-consumer-binding.json",
)


@dataclass(frozen=True)
class Worker:
    reference: dict[str, object]
    handoff: dict[str, object]
    workflow: Path
    state: Path
    names: frozenset[str]


def _path(value: object) -> Path:
    if not isinstance(value, str) or any(char in value for char in "\n\r\x00"):
        raise ValueError("Expected canonical repository work path")
    path = Path(value)
    if (
        not path.is_absolute()
        or str(path) != value
        or path != path.resolve()
        or not path.is_relative_to(Path.cwd().resolve() / "edits/work")
    ):
        raise ValueError("Planner metadata must remain in canonical repository work")
    return path


def _pin(path: Path) -> dict[str, str]:
    return {"path": str(path), "sha256": hashlib.sha256(bounded_read(path)).hexdigest()}


def _read(reference: object, pins: dict[str, str]) -> tuple[Path, dict[str, object]]:
    ref = record(reference)
    if set(ref) != {"path", "sha256"}:
        raise ValueError("Expected exact metadata pin")
    path = _path(ref["path"])
    data = bounded_read(path)
    digest = hashlib.sha256(data).hexdigest()
    if re.fullmatch(r"[0-9a-f]{64}", str(ref["sha256"])) is None or ref["sha256"] != digest:
        raise ValueError("Pinned planner metadata changed")
    if pins.get(str(path), digest) != digest:
        raise ValueError("Conflicting metadata pin")
    pins[str(path)] = digest
    return path, decode(data)


def _empty(output: Path) -> None:
    _path(str(output))
    if not output.is_dir() or tuple(output.iterdir()):
        raise ValueError("Expected existing empty distinct planner work root")


def _list(value: object) -> list[object]:
    if not isinstance(value, list) or not 1 <= len(value) <= MAX_CHUNKS:
        raise ValueError("Expected bounded nonempty planner list")
    return list(value)


def _workers(value: object, pins: dict[str, str]) -> tuple[Worker, ...]:
    result: list[Worker] = []
    for reference in _list(value):
        _, handoff = _read(reference, pins)
        if (
            handoff.get("kind") != consumer.TERMINAL_KIND
            or handoff.get("currentObservation") is not None
            or handoff.get("observedAtNs") is not None
            or handoff.get("oldWorkersDisabled") is not True
            or handoff.get("historicalActorIdentityRecorded") is not False
        ):
            raise ValueError("Planner needs an explicit unsealed terminal-target draft")
        workflow, _ = _read(handoff.get("workflow"), pins)
        state = _path(handoff.get("state"))
        names = frozenset(stage.name for stage in load_stages(workflow))
        if any(item.workflow == workflow or item.state == state for item in result):
            raise ValueError("Original workers must be unique")
        result.append(Worker(record(reference), handoff, workflow, state, names))
    return tuple(result)


def _unstarted(job: consumer.Job) -> None:
    receipts = tuple(job.original_state / f"{stage.name}.json" for stage in job.original_stages)
    paths = tuple(
        job.chunk.parent / name
        for name in UNSTARTED_NAMES
        if name != "session-consumer-binding.json"
    )
    if any(path.exists() or path.is_symlink() for path in (*receipts, *paths)):
        raise ValueError("Requested chunk is already reserved, started or partially produced")
    marker = job.chunk.parent / "session-consumer-binding.json"
    if marker.exists() or marker.is_symlink():
        # Read-only: a surviving marker must be a verifiably incomplete interrupted attempt.
        consumer.interrupted_reservation(job, job.config)


def validate(config: Path) -> consumer.Job:
    """Validate a draft's real domains and untouched target, without owner observation."""
    job = consumer.load_config(_path(str(config)), require_observation=False)
    _unstarted(job)
    return job


def draft(config: Path, output: Path) -> Path:
    """Generate exact-clock drafts only for explicitly requested untouched chunks."""
    _empty(output)
    pins: dict[str, str] = {}
    _, settings = _read(_pin(_path(str(config))), pins)
    if (
        set(settings) != {"version", "originalPlan", "providerTemplate", "handoffs", "chunkIndices"}
        or settings.get("version") != 1
    ):
        raise ValueError("Invalid session draft request")
    original, plan = _read(settings["originalPlan"], pins)
    _, original_settings = _read(plan.get("settings"), pins)
    _, source_clock = _read(original_settings.get("sourceClock"), pins)
    frame_budget = original_settings.get("chunkFrames")
    if type(frame_budget) is not int:
        raise ValueError("Missing original chunk frame budget")
    clocks = clock_chunks(source_clock, maximum_frames=frame_budget)
    if plan.get("chunks") != len(clocks) or plan.get("wholeSourcePlanned") is not True:
        raise ValueError("Original plan does not cover the complete physical source clock")
    _, provider = _read(settings["providerTemplate"], pins)
    workers = _workers(settings["handoffs"], pins)
    indices = _list(settings["chunkIndices"])
    if any(type(index) is not int or not 0 <= index < len(clocks) for index in indices) or len(
        set(indices)
    ) != len(indices):
        raise ValueError("Requested chunks must be distinct real original members")
    rows: list[dict[str, object]] = []
    for raw_index in indices:
        if not isinstance(raw_index, int):
            raise ValueError("Expected integer chunk index")
        name = f"chunk-{raw_index:05d}"
        names = frozenset(f"{name}-{mode}" for mode in ("ocr", "admission", "repair"))
        selected = tuple(worker for worker in workers if names.issubset(worker.names))
        if len(selected) != 1:
            raise ValueError("Requested chunk needs exactly one original worker membership")
        worker, clock = selected[0], clocks[raw_index]
        child = output / name
        child.mkdir()
        provider_path, handoff_path, config_path = (
            child / name for name in ("provider.json", "handoff-draft.json", "config-draft.json")
        )
        samples = tuple(point * clock.time_base for point in clock.pts)
        publish(
            provider_path,
            encode(
                {
                    **provider,
                    "samples": [
                        {"value": point.numerator, "timescale": point.denominator}
                        for point in samples
                    ],
                }
            ),
        )
        publish(handoff_path, encode(worker.handoff))
        publish(
            config_path,
            encode(
                {
                    "version": 1,
                    "originalPlan": settings["originalPlan"],
                    "chunkIndex": raw_index,
                    "originalWorker": {
                        "workflow": _pin(worker.workflow),
                        "state": str(worker.state),
                    },
                    "handoff": _pin(handoff_path),
                    "ocrProvider": _pin(provider_path),
                }
            ),
        )
        job = validate(config_path)
        if job.clock != clock or job.original_plan != original:
            raise ValueError("Consumer draft does not preserve the selected physical clock")
        rows.append(
            {
                "chunkIndex": raw_index,
                "config": _pin(config_path),
                "frames": len(clock.pts),
                "firstPTS": clock.pts[0],
                "endPTS": clock.pts[-1] + clock.last_duration,
                "timeBase": str(clock.time_base),
                "ownerObserved": False,
                "reserved": False,
            }
        )
    for name, digest in pins.items():
        if _pin(Path(name))["sha256"] != digest:
            raise ValueError("Planner metadata changed before terminal capture")
    result = output / "draft-plan.json"
    publish(
        result,
        encode(
            {
                "version": 1,
                "protocol": PROTOCOL,
                "inputs": pins,
                "chunks": rows,
                "currentOwnerObservationRequired": True,
                "oldWorkersMustRemainDisabled": True,
                "mediaBytesVerified": False,
                "consumerConnected": False,
                "qualityApproved": False,
            }
        ),
    )
    return result


def timed_workflow(config: Path, output: Path) -> Path:
    """Wrap only a fresh ordinary session graph; validators remain read-only on resume."""
    _empty(output)
    pins: dict[str, str] = {}
    _, settings = _read(_pin(_path(str(config))), pins)
    if (
        set(settings) != {"version", "consumerWorkflow", "consumerState", "timingRoot"}
        or settings.get("version") != 1
    ):
        raise ValueError("Invalid timing graph request")
    original, _ = _read(settings["consumerWorkflow"], pins)
    state, timing = _path(settings["consumerState"]), _path(settings["timingRoot"])
    _empty(timing)
    if state.exists() and (not state.is_dir() or tuple(state.iterdir())):
        raise ValueError("Timing must be configured before any ordinary Stage state exists")
    stages = load_stages(original)
    if len(stages) != len(MODES):
        raise ValueError("Expected the three ordinary session consumer stages")
    operation: Path | None = None
    definitions: list[dict[str, object]] = []
    code = {
        str(item.path): item.sha256
        for item in code_dependency_inventory.capture(
            ("phase_timer", "subtitle_session_cleanup_plan")
        )
    }
    binding = output / "timing-binding.json"
    for stage, (mode, verifier) in zip(stages, MODES, strict=True):
        command = stage.command
        if (
            len(command) != 6
            or command[1:5] != ("-B", "-m", "replay.subtitle_session_cleanup", mode)
            or stage.verify_command != (*command[:4], verifier, command[5])
            or stage.adopt_existing
            or not stage.name.endswith("-session-" + mode)
            or any(path.exists() or path.is_symlink() for path in stage.outputs)
        ):
            raise ValueError("Timing requires fresh unchanged ordinary consumer commands/outputs")
        selected = _path(command[5])
        if operation is not None and selected != operation:
            raise ValueError("Timed phases must bind the same consumer operation")
        operation = selected
        _, operation_data = _read(_pin(operation), pins)
        work = _path(operation_data.get("workRoot"))
        if (
            operation_data.get("protocol") != consumer.PROTOCOL
            or selected != work / "operation.json"
            or selected not in stage.inputs
            or stage.outputs != tuple(work / name for name in PHASE_OUTPUTS[mode])
        ):
            raise ValueError("Timing requires exact ordinary operation/output bindings")
        log = timing / f"{stage.name}.jsonl"
        inputs = list(
            dict.fromkeys([*(str(path) for path in stage.inputs), str(binding), *pins, *code])
        )
        definitions.append(
            {
                "name": stage.name,
                "command": [
                    command[0],
                    "-B",
                    "-m",
                    "replay.phase_timer",
                    str(log),
                    stage.name,
                    "--",
                    *command,
                ],
                "inputs": inputs,
                "outputs": [*(str(path) for path in stage.outputs), str(log)],
                "verify_command": list(stage.verify_command),
                "timeout_seconds": stage.timeout_seconds,
            }
        )
    stage_graph(parse_stages({"version": 1, "stages": definitions}))
    for name, digest in {**pins, **code}.items():
        if hashlib.sha256(bounded_read(Path(name))).hexdigest() != digest:
            raise ValueError("Timing metadata/code changed before terminal capture")
    publish(
        binding,
        encode(
            {
                "version": 1,
                "inputs": pins,
                "readerCode": code,
                "ordinaryState": str(state),
                "timingRoot": str(timing),
                "freshDispatchRequired": True,
                "verificationDoesNotAppendTiming": True,
            }
        ),
    )
    result = output / "workflow.json"
    publish(result, encode({"version": 1, "stages": definitions}))
    return result


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("draft", "validate", "timed-workflow"))
    parser.add_argument("config", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args(argv)
    try:
        if args.mode == "validate":
            job = validate(args.config)
            print(
                f"Validated untouched chunk with {len(job.clock.pts)} actual source samples; "
                "owner not observed"
            )
        else:
            if args.output is None:
                raise ValueError("Draft/timing preparation requires an output work root")
            action = {"draft": draft, "timed-workflow": timed_workflow}[args.mode]
            print(action(args.config, args.output))
    except (OSError, ValueError, ReplayError) as error:
        print(f"SESSION_CLEANUP_PLAN_ERROR {error}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
