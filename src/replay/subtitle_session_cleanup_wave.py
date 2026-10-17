"""Prepare ordinary hierarchical cleanup workflows; never dispatch a wave itself.

Each lane is an existing replay.workflow graph. Runtime commands use the frozen
session observer/preparer and the ordinary child workflow. Uncertain outputs are
retained, not removed, reset or silently reissued. Root supplies resource limits.
"""

from __future__ import annotations

import argparse
import hashlib
import subprocess
import sys
from pathlib import Path

from replay import cleanup_lane_successor as proof
from replay import code_dependency_inventory, workflow
from replay import code_generation_compat as compat
from replay import subtitle_session_cleanup as consumer
from replay import subtitle_session_cleanup_plan as drafts
from replay.core import ReplayError
from replay.recordings import record
from replay.sequence_journal import bounded_read, decode, encode, publish
from replay.stages import _fingerprint, file_digest

PROTOCOL = "source-caption-session-wave-v1"

VERIFY_MODES = frozenset({"verify-observe", "verify-prepare-child", "verify-run-child"})
MAX_CHUNKS = 128
MAX_LANES = 10
MAX_TIMEOUT = 43200
OWNER_FILES = ("config-ready.json", "owner-observation.json", "handoff.json")
PREPARED_FILES = (
    "consumer/operation.json",
    "consumer/workflow.json",
    "timing-config.json",
    "timed/timing-binding.json",
    "timed/workflow.json",
    "prepared.json",
)
RESULT_FILES = (
    "session/inputs.json",
    "session/complete.json",
    "ocr-verification.json",
    "admission.json",
    "eligibility.json",
    "candidate-config.json",
    "repair/candidate.mp4",
    "repair/candidate.json",
    "assembly-row.json",
)
MODES = ("collect", "admission", "repair")


def _path(value: object) -> Path:
    if not isinstance(value, str) or any(char in value for char in "\n\r\x00"):
        raise ValueError("Expected canonical wave work path")
    path = Path(value)
    if (
        not path.is_absolute()
        or str(path) != value
        or path != path.resolve()
        or not path.is_relative_to(Path.cwd().resolve() / "edits/work")
    ):
        raise ValueError("Wave metadata/output must remain in canonical repository work")
    return path


def _pin(path: Path) -> dict[str, str]:
    return {"path": str(path), "sha256": hashlib.sha256(bounded_read(path)).hexdigest()}


def _read(path: Path) -> dict[str, object]:
    return decode(bounded_read(_path(str(path))))


def _reference(value: object) -> Path:
    ref = record(value)
    if set(ref) != {"path", "sha256"}:
        raise ValueError("Expected exact wave metadata reference")
    path = _path(ref["path"])
    if _pin(path)["sha256"] != ref["sha256"]:
        raise ValueError("Wave metadata pin changed")
    return path


def _empty(path: Path) -> None:
    _path(str(path))
    if not path.is_dir() or tuple(path.iterdir()):
        raise ValueError("Expected existing empty distinct wave root")


def _integer(value: object, maximum: int) -> int:
    if type(value) is not int or not 1 <= value <= maximum:
        raise ValueError("Expected bounded positive wave integer")
    return value


def _code() -> dict[str, str]:
    return {
        str(item.path): item.sha256
        for item in code_dependency_inventory.capture(("subtitle_session_cleanup_wave",))
    }


def _verify_pins(value: object) -> None:
    for name, sha in record(value).items():
        if hashlib.sha256(bounded_read(Path(name))).hexdigest() != sha:
            raise ValueError("Wave metadata/code changed")


def _operation(path: Path, *, allow_historical: bool = False) -> tuple[dict[str, object], Path]:
    data = _read(path)
    if (
        set(data)
        != {
            "version",
            "protocol",
            "draft",
            "workRoot",
            "originalMarker",
            "metadataPins",
            "assetPins",
            "readerCode",
            "timeoutSeconds",
            "chunkName",
        }
        or type(data.get("version")) is not int
        or data.get("version") != 1
        or data.get("protocol") != PROTOCOL
    ):
        raise ValueError("Invalid or changed wave operation")
    code = _code()
    reader = record(data.get("readerCode"))
    if reader != code and not (allow_historical and compat.historical_generation(reader, code)):
        raise ValueError("Invalid or changed wave operation")
    job = consumer.load_config(_reference(data["draft"]), require_observation=False)
    assets = {str(job.provider.source), str(job.provider.native)}
    assets.update(
        str(record(job.settings[key])["path"]) for key in ("source", "native", "decoder", "model")
    )
    if (
        data["metadataPins"] != {name: sha for name, sha in job.pins.items() if name not in assets}
        or data["assetPins"] != {name: sha for name, sha in job.pins.items() if name in assets}
        or data["chunkName"] != job.chunk.parent.name
        or data["originalMarker"] != str(job.chunk.parent / "session-consumer-binding.json")
    ):
        raise ValueError("Wave operation must retain exact original domain/asset bindings")
    _verify_pins(data["metadataPins"])
    if reader == code:
        _verify_pins(reader)
    else:
        compat.verify_recorded_pins(reader, code)
    return data, _path(data["workRoot"])


def _command(mode: str, operation: Path) -> list[str]:
    return [
        sys.executable,
        "-B",
        "-m",
        "replay.subtitle_session_cleanup_wave",
        mode,
        str(operation),
    ]


def _definitions(
    operation: Path, data: dict[str, object], *, previous: tuple[str, ...], recovery: bool
) -> list[dict[str, object]]:
    work = _path(data["workRoot"])
    chunk = str(data["chunkName"])
    common = list(
        dict.fromkeys([str(operation), *record(data["metadataPins"]), *record(data["readerCode"])])
    )
    owner = [str(work / "owner" / name) for name in OWNER_FILES]
    marker = str(_path(data["originalMarker"]))
    ready = [str(work / name) for name in PREPARED_FILES]
    # A recovery chunk supersedes its old reservation from inside prepare-child, so the
    # changing marker must not be a stage input or output; its prior bytes stay archived in
    # the interrupted attempt's own work root and the fresh prepared.json pins the new one.
    prepare_inputs = [*common, *owner]
    prepare_outputs = [*ready, *([] if recovery else [marker])]
    final = [str(work / "consumer" / name) for name in RESULT_FILES]
    final += [str(work / "consumer/state" / f"{chunk}-session-{mode}.json") for mode in MODES]
    final += [str(work / "timing" / f"{chunk}-session-{mode}.jsonl") for mode in MODES]
    final.append(str(work / "completion.json"))
    phases = (
        ("observe", [*common, *previous], owner),
        ("prepare-child", prepare_inputs, prepare_outputs),
        ("run-child", [*common, *owner, *ready, marker, *record(data["assetPins"])], final),
    )
    return [
        {
            "name": f"{chunk}-wave-{mode}",
            "command": _command(mode, operation),
            "inputs": inputs,
            "outputs": outputs,
            "verify_command": _command("verify-" + mode, operation),
            "timeout_seconds": data["timeoutSeconds"],
        }
        for mode, inputs, outputs in phases
    ]


def prepare(config: Path, output: Path) -> Path:
    """Plan disjoint bounded lanes without observation, reservation or inference."""
    _empty(output)
    request = _read(config)
    if (
        set(request) != {"version", "drafts", "maximumWorkers", "timeoutSeconds"}
        or type(request.get("version")) is not int
        or request.get("version") != 1
    ):
        raise ValueError("Invalid wave preparation request")
    raw = request["drafts"]
    if not isinstance(raw, list) or not 1 <= len(raw) <= MAX_CHUNKS:
        raise ValueError("Expected bounded explicit wave draft references")
    lanes = min(_integer(request["maximumWorkers"], MAX_LANES), len(raw))
    timeout = _integer(request["timeoutSeconds"], MAX_TIMEOUT)
    code = _code()
    groups: list[list[dict[str, object]]] = [[] for index in range(lanes)]
    predecessors: list[tuple[str, ...]] = [() for index in range(lanes)]
    membership: set[tuple[Path, Path]] = set()
    rows: list[dict[str, object]] = []
    for index, reference in enumerate(raw):
        config_path = _reference(reference)
        job = drafts.validate(config_path)
        member = job.original_plan, job.chunk
        if member in membership:
            raise ValueError("Wave chunk membership must be unique")
        membership.add(member)
        chunk_name = job.chunk.parent.name
        child = output / f"selection-{index:05d}"
        child.mkdir()
        for name in ("owner", "consumer", "timed", "timing"):
            (child / name).mkdir()
        assets = {str(job.provider.source), str(job.provider.native)}
        assets.update(
            str(record(job.settings[key])["path"])
            for key in ("source", "native", "decoder", "model")
        )
        data: dict[str, object] = {
            "version": 1,
            "protocol": PROTOCOL,
            "draft": reference,
            "workRoot": str(child),
            "originalMarker": str(job.chunk.parent / "session-consumer-binding.json"),
            "metadataPins": {name: sha for name, sha in job.pins.items() if name not in assets},
            "assetPins": {name: sha for name, sha in job.pins.items() if name in assets},
            "readerCode": code,
            "timeoutSeconds": timeout,
            "chunkName": chunk_name,
        }
        operation = child / "wave-operation.json"
        publish(operation, encode(data))
        lane = index % lanes
        marker = job.chunk.parent / "session-consumer-binding.json"
        recovery = marker.exists() or marker.is_symlink()
        definitions = _definitions(operation, data, previous=predecessors[lane], recovery=recovery)
        groups[lane].extend(definitions)
        final_outputs = definitions[-1]["outputs"]
        if not isinstance(final_outputs, list):
            raise ValueError("Missing wave output mapping")
        predecessors[lane] = tuple(str(item) for item in final_outputs)
        rows.append(
            {
                "draft": reference,
                "operation": _pin(operation),
                "lane": lane,
                "chunkName": chunk_name,
                "frames": len(job.clock.pts),
                "ownerObserved": False,
            }
        )
    lane_refs: list[dict[str, object]] = []
    for index, definitions in enumerate(groups):
        value = {"version": 1, "stages": definitions}
        workflow.stage_graph(workflow.parse_stages(value))
        serialized = encode(value)
        if len(serialized) > workflow.MAX_PLAN_BYTES:
            raise ValueError("Parent workflow exceeds normal reader byte budget; use smaller waves")
        path = output / f"lane-{index:02d}.json"
        publish(path, serialized)
        lane_refs.append({"workflow": _pin(path), "state": str(output / f"state-{index:02d}")})
    _verify_pins(code)
    result = output / "wave.json"
    publish(
        result,
        encode(
            {
                "version": 1,
                "protocol": PROTOCOL,
                "request": _pin(config),
                "lanes": lane_refs,
                "chunks": rows,
                "maximumSimultaneousLaneControllers": lanes,
                "resourceAdmissionRequired": True,
                "dispatched": False,
                "consumerConnected": False,
            }
        ),
    )
    return result


def _observed(data: dict[str, object], work: Path) -> consumer.Job:
    job = consumer.load_config(work / "owner/config-ready.json")
    original = record(_read(_reference(data["draft"]))["originalPlan"])
    if (
        str(job.original_plan) != original["path"]
        or job.chunk.parent.name != data["chunkName"]
        or job.handoff.get("currentObservation") != _pin(work / "owner/owner-observation.json")
        or job.handoff.get("historicalActorIdentityRecorded") is not False
    ):
        raise ValueError("Wave observer/result correspondence differs")
    return job


def _prepared(data: dict[str, object], work: Path) -> None:
    job = _observed(data, work)
    saved = _read(work / "prepared.json")
    expected = {
        "version": 1,
        "protocol": PROTOCOL,
        "ownerConfig": _pin(job.config),
        "outputs": {str(work / name): _pin(work / name)["sha256"] for name in PREPARED_FILES[:-1]},
        "marker": _pin(_path(data["originalMarker"])),
        "normalDependencyPreflightPassed": True,
    }
    if saved != expected:
        raise ValueError("Wave prepared checkpoint/output correspondence differs")
    operation = _read(work / "consumer/operation.json")
    marker = _read(_path(data["originalMarker"]))
    if (
        operation.get("config") != _pin(job.config)
        or operation.get("inputs") != job.pins
        or marker
        != {
            "version": 1,
            "operation": _pin(work / "consumer/operation.json"),
            "workRoot": str(work / "consumer"),
        }
    ):
        raise ValueError("Wave ordinary consumer/namespace reservation differs")


def _completed(data: dict[str, object], work: Path) -> dict[str, object]:
    _prepared(data, work)
    stages = workflow.load_stages(work / "timed/workflow.json")
    state = work / "consumer/state"
    for stage in stages:
        saved = _read(state / f"{stage.name}.json")
        outputs = {str(path): file_digest(path) for path in stage.outputs}
        expected = {
            "status": "complete",
            "fingerprint": _fingerprint(stage),
            "outputs": outputs,
        }
        if saved != expected:
            historical_sets = compat.historical_stage_input_sets(
                {str(path): file_digest(path) for path in stage.inputs}
            )
            historical_match = any(
                saved
                == {
                    "status": "complete",
                    "fingerprint": proof._historical_fingerprint(stage, historical),
                    "outputs": outputs,
                }
                for historical in historical_sets
            )
            if not historical_match:
                raise ValueError(
                    "Wave child Stage is incomplete or its immutable inputs/outputs changed"
                )
    for mode in ("verify-ocr", "verify-admission", "verify-repair"):
        consumer.run(work / "consumer/operation.json", mode=mode)
    return {
        "version": 1,
        "protocol": PROTOCOL,
        "childWorkflow": _pin(work / "timed/workflow.json"),
        "stageReceipts": [_pin(state / f"{stage.name}.json") for stage in stages],
        "assemblyRow": _pin(work / "consumer/assembly-row.json"),
        "consumerConnected": False,
        "qualityApproved": False,
        "fullDecodeVerified": False,
    }


def run(operation: Path, *, mode: str) -> None:
    """One normal parent Stage command/verifier; never poll or reset a child."""
    historical = mode in VERIFY_MODES
    data, work = _operation(operation, allow_historical=historical)
    if mode == "observe":
        consumer.observe_terminal(_reference(data["draft"]), work / "owner")
    elif mode == "verify-observe":
        _observed(data, work)
    elif mode == "prepare-child":
        job = _observed(data, work)
        prepared = consumer.prepare(job.config, work / "consumer")
        timing = work / "timing-config.json"
        publish(
            timing,
            encode(
                {
                    "version": 1,
                    "consumerWorkflow": _pin(prepared.workflow),
                    "consumerState": str(prepared.state),
                    "timingRoot": str(work / "timing"),
                }
            ),
        )
        graph = drafts.timed_workflow(timing, work / "timed")
        if workflow.main([str(graph), "--state", str(prepared.state), "--check"]) != 0:
            raise ValueError("Ordinary child workflow dependency preflight failed")
        publish(
            work / "prepared.json",
            encode(
                {
                    "version": 1,
                    "protocol": PROTOCOL,
                    "ownerConfig": _pin(job.config),
                    "outputs": {
                        str(work / name): _pin(work / name)["sha256"]
                        for name in PREPARED_FILES[:-1]
                    },
                    "marker": _pin(_path(data["originalMarker"])),
                    "normalDependencyPreflightPassed": True,
                }
            ),
        )
    elif mode == "verify-prepare-child":
        _prepared(data, work)
    elif mode == "run-child":
        _prepared(data, work)
        subprocess.run(
            [
                sys.executable,
                "-B",
                "-m",
                "replay.workflow",
                str(work / "timed/workflow.json"),
                "--state",
                str(work / "consumer/state"),
            ],
            stdin=subprocess.DEVNULL,
            check=True,
            timeout=_integer(data["timeoutSeconds"], MAX_TIMEOUT),
        )
        publish(work / "completion.json", encode(_completed(data, work)))
    elif mode == "verify-run-child":
        if _read(work / "completion.json") != _completed(data, work):
            raise ValueError("Wave completed checkpoint differs")
    else:
        raise ValueError("Unsupported wave runtime mode")
    _operation(operation, allow_historical=historical)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "mode",
        choices=(
            "prepare",
            "observe",
            "verify-observe",
            "prepare-child",
            "verify-prepare-child",
            "run-child",
            "verify-run-child",
        ),
    )
    parser.add_argument("input", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args(argv)
    try:
        if args.mode == "prepare":
            if args.output is None:
                raise ValueError("Wave preparation requires an empty --output root")
            print(prepare(args.input, args.output))
        else:
            if args.output is not None:
                raise ValueError("Runtime uses only its pinned wave operation")
            run(args.input, mode=args.mode)
    except (OSError, ValueError, ReplayError, subprocess.SubprocessError) as error:
        print(f"SESSION_CLEANUP_WAVE_ERROR {error}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
