"""Finalize a genuine recovery generation through ordinary source assembly.

Preparation reads bounded metadata only. Runtime never performs OCR, restarts a
worker, rewrites historical receipts, or treats an occupied partial as complete.
"""

from __future__ import annotations

import argparse
import shutil
import sys
from collections.abc import Iterator
from contextlib import ExitStack, contextmanager
from pathlib import Path

from replay import cleanup_finalization_workflow as common
from replay import cleanup_generation_recovery as recovery
from replay import cleanup_lane_successor as original
from replay import code_dependency_inventory, workflow
from replay import delivery_source_assembly as media
from replay import subtitle_cleanup_assembly as assembly
from replay.core import ReplayError
from replay.recordings import record
from replay.sequence_journal import bounded_read, encode, publish
from replay.stages import StageError
from replay.subtitle_cleanup_plan import clock_chunks
from replay.subtitle_eligibility import _reference as tool_reference

PROTOCOL = "recovery-source-finalization-v1"
FILES = ("index.json", "manifest.json", "lineage.json")
MODES = ("finalize", "prepare-child", "run-child")


def _code() -> dict[str, str]:
    return {
        str(item.path): item.sha256
        for item in code_dependency_inventory.capture(("recovery_source_finalization",))
    }


def _generation(path: Path) -> dict[str, object]:
    """Read metadata; full original recovery validation belongs to runtime."""
    data = common._read(path)
    if (
        type(data.get("version")) is not int
        or data.get("version") != 1
        or data.get("kind") != "cleanup-recovery-generation"
    ):
        raise ValueError("Expected original recovery generation")
    workers = data.get("workers")
    rows = original._list(data.get("chunks"))
    graphs, states = original._strings(data.get("workflows")), original._strings(data.get("states"))
    lanes = original._list(data.get("workflowLanes"))
    if (
        type(workers) is not int
        or not 1 <= workers <= 8
        or not 1 <= len(rows) <= assembly.MAX_CHUNKS
        or not 1 <= len(graphs) <= 64
        or len(graphs) != len(states)
        or len(graphs) != len(lanes)
        or len(set(graphs + states)) != 2 * len(graphs)
        or any(type(lane) is not int or not 0 <= lane < workers for lane in lanes)
    ):
        raise ValueError("Invalid recovery lanes or segment coverage")
    settings = common._read(common._path(data["settings"]))
    source_clock = common._ref(settings["sourceClock"])
    clocks = clock_chunks(
        common._read(source_clock), maximum_frames=assembly._positive(settings["chunkFrames"])
    )
    prior = common._read(common._path(data["originalPlan"]))
    frames = sum(len(clock.pts) for clock in clocks)
    if (
        not clocks
        or clocks[0].pts[0] != 0
        or len(clocks) != len(rows)
        or prior.get("wholeSourcePlanned") is not True
        or prior.get("chunks") != len(rows)
        or prior.get("plannedFrames") != frames
        or prior.get("totalSourceFrames") != frames
        or common._ref(prior["settings"]) != common._path(data["settings"])
    ):
        raise ValueError("Recovery must cover the entire original physical clock")
    for index, (value, clock) in enumerate(zip(rows, clocks, strict=True)):
        row = record(value)
        selected = common._read(common._path(row["chunk"]))
        if (
            type(row.get("index")) is not int
            or row.get("index") != index
            or type(row.get("lane")) is not int
            or row.get("lane") != index % workers
            or selected.get("settings") != common._pin(common._path(data["settings"]))
            or selected.get("clock")
            != {
                "pts": list(clock.pts),
                "timeBase": str(clock.time_base),
                "lastDuration": clock.last_duration,
            }
        ):
            raise ValueError("Recovery chunk order, clock or settings differs")
    return data


def _request(path: Path, operation: Path, root: Path, code: dict[str, str]) -> common.Job:
    raw = common._read(path)
    if (
        set(raw) - {"ownerObservation"}
        != {"version", "generation", "controllers", "ffmpeg", "ffprobe", "timeoutSeconds"}
        or type(raw.get("version")) is not int
        or raw.get("version") != 1
    ):
        raise ValueError("Invalid recovery finalization request")
    if "ownerObservation" in raw:
        common._ref(raw["ownerObservation"])
    timeout = raw["timeoutSeconds"]
    if type(timeout) is not int or not 1 <= timeout <= common.MAX_SECONDS:
        raise ValueError("Expected bounded original assembly timeout")
    plan = common._ref(raw["generation"])
    data = _generation(plan)
    graphs, states = original._strings(data["workflows"]), original._strings(data["states"])
    lane_numbers = original._list(data["workflowLanes"])
    controls = original._list(raw["controllers"])
    if len(controls) != data["workers"]:
        raise ValueError("Every original recovery parent is required")
    lanes: list[tuple[Path, Path, Path]] = []
    seen: set[int] = set()
    for lane, value in enumerate(controls):
        item = record(value)
        if set(item) != {"workflow", "state", "status"}:
            raise ValueError("Expected exact original parent binding")
        graph, state, status = (
            common._ref(item["workflow"]),
            common._path(item["state"]),
            common._path(item["status"]),
        )
        stages = workflow.load_stages(graph)
        for stage in stages:
            command = stage.command
            if (
                len(command) != 7
                or command[:4] != (sys.executable, "-B", "-m", "replay.workflow")
                or command[5] != "--state"
            ):
                raise ValueError("Recovery parent must use its ordinary child workflow")
            child = str(Path(command[4]))
            if child not in graphs:
                raise ValueError("Parent contains an undeclared recovery segment")
            index = graphs.index(child)
            children = workflow.load_stages(Path(child))
            produced = {output for child_stage in children for output in child_stage.outputs}
            required = {
                source for child_stage in children for source in child_stage.inputs
            } - produced
            if (
                index in seen
                or lane_numbers[index] != lane
                or command[6] != states[index]
                or stage.verify_command
                != (
                    sys.executable,
                    "-B",
                    "-m",
                    "replay.cleanup_generation_recovery",
                    "verify-segment",
                    str(plan),
                    "--index",
                    str(index),
                )
                or set(stage.outputs) != produced
                or set(stage.inputs) != required | {Path(child)}
            ):
                raise ValueError("Original recovery parent/segment identity differs")
            seen.add(index)
        lanes.append((graph, state, status))
    if seen != set(range(len(graphs))) or len({path for lane in lanes for path in lane}) != 3 * len(
        lanes
    ):
        raise ValueError("Recovery parent coverage is incomplete or repeated")
    tools = record(raw["ffmpeg"]), record(raw["ffprobe"])
    for tool in tools:
        if set(tool) != {"path", "sha256"}:
            raise ValueError("Expected exact installed assembly tool reference")
        media._path(tool["path"])
        assembly._sha(tool["sha256"])
    return common.Job(operation, root, plan, tuple(lanes), code, tools, timeout)


def _load(operation: Path) -> common.Job:
    raw = common._read(operation)
    if (
        set(raw) != {"version", "protocol", "request", "workRoot", "readerCode"}
        or type(raw.get("version")) is not int
        or raw.get("version") != 1
        or raw.get("protocol") != PROTOCOL
        or raw["readerCode"] != _code()
    ):
        raise ValueError("Recovery finalization operation or current reader changed")
    code = {name: assembly._sha(value) for name, value in record(raw["readerCode"]).items()}
    return _request(common._ref(raw["request"]), operation, common._path(raw["workRoot"]), code)


@contextmanager
def _ready(job: common.Job) -> Iterator[dict[str, object]]:
    with ExitStack() as stack:
        metadata = _generation(job.plan)
        states = [state for _, state, _ in job.lanes] + [
            common._path(name) for name in original._strings(metadata["states"])
        ]
        for state in states:
            stack.enter_context(original._unlocked(state / ".workflow.lock"))
        for graph, state, status in job.lanes:
            if bounded_read(status).strip() != b"0":
                raise ValueError("Every original recovery parent must be terminal zero")
            for stage in workflow.load_stages(graph):
                recovery.retained_complete(stage, state, metadata["readerCode"])
        data = recovery._load(job.plan, retained=True)
        for tool in job.tools:
            tool_reference(tool, {}, executable=True)
        yield data
        if recovery._load(job.plan, retained=True) != data:
            raise ValueError("Recovery generation changed during finalization")
        for graph, state, status in job.lanes:
            if bounded_read(status).strip() != b"0":
                raise ValueError("Original recovery terminal outcome changed")
            for stage in workflow.load_stages(graph):
                recovery.retained_complete(stage, state, metadata["readerCode"])
        for tool in job.tools:
            tool_reference(tool, {}, executable=True)
        _load(job.operation)


def _owner_observation(job: common.Job) -> Path | None:
    operation = common._read(job.operation)
    request = common._read(common._ref(operation["request"]))
    return common._ref(request["ownerObservation"]) if "ownerObservation" in request else None


def _finalized(job: common.Job, data: dict[str, object]) -> dict[str, object]:
    """Reconstruct real recovery output using original domain functions only."""
    pins: dict[str, str] = {}
    settings_path = common._path(data["settings"])
    settings = assembly._read(settings_path, pins)
    clock_path, clock = assembly._ref(settings["sourceClock"], pins)
    clocks = clock_chunks(clock, maximum_frames=assembly._positive(settings["chunkFrames"]))
    rows: list[dict[str, object]] = []
    refs: list[dict[str, object]] = []
    canvases: set[tuple[int, int]] = set()
    for index, (value, physical) in enumerate(
        zip(original._list(data["chunks"]), clocks, strict=True)
    ):
        owner = _owner_observation(job)
        if owner is None:
            recovery.phase(job.plan, index, verify=True)
        else:
            recovery.phase(job.plan, index, verify=True, owner_observation=owner)
        chunk = common._path(record(value)["chunk"])
        row, ref, canvas = assembly._chunk(
            chunk.parent,
            physical,
            settings_path,
            assembly._identity(settings["source"]),
            assembly._identity(settings["native"]),
            pins,
        )
        rows.append(row)
        refs.append(ref)
        canvases.add(canvas)
    final = job.root / "finalized"
    if (
        len(canvases) != 1
        or common._read(final / "index.json")
        != {"ranges": rows, "purpose": "explicit recovery selected results"}
        or common._read(final / "manifest.json")
        != {"version": 1, "index": common._pin(final / "index.json"), "chunks": refs}
        or common._read(final / "lineage.json")
        != {
            "version": 1,
            "generation": common._pin(job.plan),
            "manifest": common._pin(final / "manifest.json"),
            "selectedResults": data["chunks"],
            "requiredNextStage": "delivery_source_assembly full packet/decoded PTS verification",
            "qualityApproved": False,
            "deliveryComplete": False,
        }
    ):
        raise ValueError("Recovery final index, geometry, manifest or lineage differs")
    media.load_chunks(
        final / "manifest.json", sha256=common._pin(final / "manifest.json")["sha256"], bindings={}
    )
    return {
        "version": 1,
        "protocol": PROTOCOL,
        "operation": common._pin(job.operation),
        "sourceClock": common._pin(clock_path),
        "physicalSourceFrames": sum(len(item.pts) for item in clocks),
        "outputs": {str(final / name): common._pin(final / name)["sha256"] for name in FILES},
    }


def _command(mode: str, operation: Path) -> list[str]:
    return [sys.executable, "-B", "-m", "replay.recovery_source_finalization", mode, str(operation)]


def _child(job: common.Job) -> dict[str, object]:
    data = common._child(job)
    stage = record(original._list(data["stages"])[0])
    removed = {str(job.root / "finalized" / name) for name in common.FINAL_FILES}
    stage["inputs"] = (
        [name for name in original._strings(stage["inputs"]) if name not in removed]
        + [str(job.root / "finalized" / name) for name in FILES]
        + [str(job.root / "finalized.json")]
    )
    stage["verify_command"] = _command("verify-assembly", job.operation)
    data["stages"] = [stage]
    workflow.stage_graph(workflow.parse_stages(data))
    return data


def _prepared(job: common.Job) -> dict[str, object]:
    graph = job.root / "child/workflow.json"
    if common._read(graph) != _child(job):
        raise ValueError("Prepared child differs from generated recovery manifest")
    return {
        "version": 1,
        "protocol": PROTOCOL,
        "operation": common._pin(job.operation),
        "workflow": common._pin(graph),
        "finalized": common._pin(job.root / "finalized.json"),
    }


def _assembled(job: common.Job) -> dict[str, object]:
    result = common._assembled(job)
    result["protocol"] = PROTOCOL
    return result


def _completed(job: common.Job) -> dict[str, object]:
    if common._read(job.root / "prepared.json") != _prepared(job):
        raise ValueError("Prepared child checkpoint changed")
    original._complete(
        workflow.load_stages(job.root / "child/workflow.json")[0], job.root / "child-state"
    )
    result = _assembled(job)
    result["childStageReceipt"] = common._pin(job.root / "child-state/assemble-clean-source.json")
    return result


def prepare(config: Path, output: Path) -> Path:
    """Create a normal graph with future producer outputs; no source/media read."""
    common._path(str(output))
    original._absent(output)
    code = _code()
    operation = output / "operation.json"
    job = _request(config, operation, output, code)
    data = _generation(job.plan)
    settings = common._read(common._path(data["settings"]))
    inputs = [
        str(operation),
        str(config),
        str(job.plan),
        *code,
        *original._strings(data["workflows"]),
        *(str(path) for graph, _, status in job.lanes for path in (graph, status)),
        str(data["settings"]),
        str(data["originalPlan"]),
        *[
            str(record(settings[field])["path"])
            for field in ("source", "native", "decoder", "model", "sourceClock")
        ],
        *[str(tool["path"]) for tool in job.tools],
    ]
    request = common._read(config)
    if "ownerObservation" in request:
        inputs.append(str(common._ref(request["ownerObservation"])))
    final = [*[str(output / "finalized" / name) for name in FILES], str(output / "finalized.json")]
    child = [str(output / "child/workflow.json"), str(output / "prepared.json")]
    result = [
        str(output / name)
        for name in (
            "clean-source.mov",
            "evidence/receipt.json",
            "evidence/single-source-index.json",
            "completion.json",
            "child-state/assemble-clean-source.json",
        )
    ]
    graph = {
        "version": 1,
        "stages": [
            {
                "name": mode,
                "command": _command(mode, operation),
                "inputs": list(dict.fromkeys(required)),
                "outputs": produced,
                "timeout_seconds": job.timeout,
                "verify_command": _command("verify-" + mode, operation),
            }
            for mode, required, produced in (
                (MODES[0], inputs, final),
                (MODES[1], inputs + final, child),
                (MODES[2], inputs + final + child, result),
            )
        ],
    }
    workflow.stage_graph(workflow.parse_stages(graph))
    output.mkdir()
    (output / "child").mkdir()
    publish(
        operation,
        encode(
            {
                "version": 1,
                "protocol": PROTOCOL,
                "request": common._pin(config),
                "workRoot": str(output),
                "readerCode": code,
            }
        ),
    )
    publish(output / "workflow.json", encode(graph))
    return output / "workflow.json"


def run(operation: Path, mode: str) -> None:
    job = _load(operation)
    with _ready(job) as data:
        if mode == "finalize":
            recovery.assemble(
                job.plan,
                job.root / "finalized",
                retained=True,
                owner_observation=_owner_observation(job),
            )
            publish(job.root / "finalized.json", encode(_finalized(job, data)))
            return
        if common._read(job.root / "finalized.json") != _finalized(job, data):
            raise ValueError("Sealed recovery manifest changed")
        if mode == "verify-finalize":
            return
        if mode == "prepare-child":
            publish(job.root / "child/workflow.json", encode(_child(job)))
            if (
                workflow.main(
                    [
                        str(job.root / "child/workflow.json"),
                        "--state",
                        str(job.root / "child-state"),
                        "--check",
                    ]
                )
                != 0
            ):
                raise ValueError("Normal assembly child preflight failed")
            publish(job.root / "prepared.json", encode(_prepared(job)))
        elif mode == "verify-prepare-child":
            if common._read(job.root / "prepared.json") != _prepared(job):
                raise ValueError("Prepared child changed")
        elif mode == "verify-assembly":
            _assembled(job)
        elif mode in {"run-child", "verify-run-child"}:
            if mode == "run-child" and not (job.root / "completion.json").exists():
                if common._read(job.root / "prepared.json") != _prepared(job):
                    raise ValueError("Prepared child changed")
                stage = workflow.load_stages(job.root / "child/workflow.json")[0]
                if any(path.exists() or path.is_symlink() for path in stage.outputs):
                    original._complete(stage, job.root / "child-state")
                chunks = media.load_chunks(
                    job.root / "finalized/manifest.json",
                    sha256=common._pin(job.root / "finalized/manifest.json")["sha256"],
                    bindings={},
                )
                usage = shutil.disk_usage(job.root)
                if (
                    usage.free
                    < max(30 * 1024**3, usage.total // 20)
                    + sum(chunk.media.stat().st_size for chunk in chunks)
                    + 64 * 1024**2
                ):
                    raise ValueError("Insufficient headroom for exact-clock source assembly")
                if (
                    workflow.main(
                        [
                            str(job.root / "child/workflow.json"),
                            "--state",
                            str(job.root / "child-state"),
                        ]
                    )
                    != 0
                ):
                    raise ValueError("Normal child stopped; preserve partial outputs")
                publish(job.root / "completion.json", encode(_completed(job)))
            if common._read(job.root / "completion.json") != _completed(job):
                raise ValueError("Completed child changed")
        else:
            raise ValueError("Unknown recovery finalization mode")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "mode",
        choices=("prepare", *MODES, *["verify-" + mode for mode in MODES], "verify-assembly"),
    )
    parser.add_argument("input", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args(argv)
    try:
        if args.mode == "prepare":
            if args.output is None:
                raise ValueError("Preparation requires output")
            print(prepare(args.input, args.output))
        else:
            if args.output is not None:
                raise ValueError("Runtime uses pinned operation only")
            run(args.input, args.mode)
        return 0
    except (OSError, ValueError, KeyError, ReplayError, StageError) as error:
        print(f"RECOVERY_FINALIZATION_ERROR {error}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
