"""Finalize completed cleanup generations without leasing unconsumed old receipts.

Selected successor inputs are intersected with its immutable recorded inputs.
Absent pre-handoff repair receipts are not invented or required; actual producing
Stages, recovery proofs and unchanged original domain artifacts remain mandatory.
"""

import argparse
import shutil
import sys
from collections.abc import Iterator
from contextlib import ExitStack, contextmanager
from pathlib import Path

from replay import cleanup_finalization_workflow as common
from replay import cleanup_generation_recovery as recovery
from replay import cleanup_lane_successor as old
from replay import code_dependency_inventory, workflow
from replay import delivery_source_assembly as media
from replay import recovery_source_finalization as original
from replay import subtitle_admission_verification as admission
from replay import subtitle_cleanup_assembly as assembly
from replay.core import ReplayError
from replay.input_hash_cache import InputHashCache, input_hash_context
from replay.recordings import record
from replay.sequence_journal import bounded_read, encode, publish
from replay.stages import StageError, execution_lock
from replay.subtitle_cleanup_plan import clock_chunks
from replay.subtitle_eligibility import _reference as tool_reference

PROTOCOL = "retained-cleanup-finalization-v1"
MODES = original.MODES


def _code() -> dict[str, str]:
    return {
        str(row.path): row.sha256
        for row in code_dependency_inventory.capture(("retained_cleanup_finalization",))
    }


def selected_recorded_inputs(data: dict[str, object], index: int) -> tuple[Path, ...]:
    """Only original inputs actually consumed by this immutable selection."""
    selected = old._selected_inputs(data, index)
    return tuple(
        old._path(name, work=False) for name in old._object(data["inputs"]) if name in selected
    )


@contextmanager
def _chunk_owned(job: common.Job, data: dict[str, object], index: int) -> Iterator[None]:
    row = record(old._list(data["chunks"])[index])
    legacy_path = old._path(data["successor"])
    legacy = recovery._retained_successor(legacy_path, index)
    prior = record(old._list(legacy["chunks"])[index])
    states = tuple(old._path(name) for name in old._strings(data["oldStates"]))
    handoff = old._read(old._path(data["handoff"]), {})
    owner = original._owner_observation(job)
    with ExitStack() as stack:
        recovery._owner_guard(handoff, states, fresh=False, retained=True, owner_observation=owner)
        if not stack.enter_context(execution_lock(job.plan.parent, f"recovery-lane-{row['lane']}")):
            raise ValueError("Recovery lane already active")
        for name in old._strings(row["oldStageNames"]):
            if not stack.enter_context(execution_lock(old._path(row["oldState"]), name)):
                raise ValueError("Completed producer is active")
        for state_name in old._strings(legacy["oldStates"]):
            stack.enter_context(old._unlocked(old._path(state_name) / ".workflow.lock"))
        for value in old._list(prior["oldStages"]):
            stage = record(value)
            if not stack.enter_context(
                execution_lock(old._path(stage["state"]), str(stage["name"]))
            ):
                raise ValueError("Original producer is active")
        pins = {
            str(path): old._pin(path)["sha256"] for path in selected_recorded_inputs(legacy, index)
        }
        yield
        if any(old._pin(Path(name))["sha256"] != digest for name, digest in pins.items()):
            raise ValueError("Consumed successor input changed")
        if recovery._retained_successor(legacy_path, index) != legacy:
            raise ValueError("Successor selection changed")
        recovery._owner_guard(handoff, states, fresh=False, retained=True, owner_observation=owner)


def verify_chunk(job: common.Job, data: dict[str, object], index: int) -> None:
    row = record(old._list(data["chunks"])[index])
    with _chunk_owned(job, data, index):
        chunk = old._path(row["chunk"])
        if row["mode"] == "retained-completed":
            legacy = old._read(old._path(data["successor"]), {})
            prior = record(old._list(legacy["chunks"])[index])
            lane = int(str(prior["lane"]))
            names = old._strings(prior["successorStages"])
            stages = [
                stage
                for stage in workflow.load_stages(Path(old._strings(legacy["workflows"])[lane]))
                if stage.name in names
            ]
            if len(stages) != len(names):
                raise ValueError("Missing actual producer Stage")
            for stage in stages:
                recovery.retained_complete(
                    stage, Path(old._strings(legacy["states"])[lane]), legacy["readerCode"]
                )
        elif row["mode"] == "adopted-existing":
            _, proof = old._ref(row["adoptedProof"], {})
            if proof != admission.verify_admission(chunk, verify_repair=True):
                raise ValueError("Adopted original proof differs")
        repair = recovery._repair_evidence(chunk)
        if row["mode"] != "pending" and repair != row["repairEvidence"]:
            raise ValueError("Retained original repair differs")
        expected = {
            "version": 1,
            "generation": old._pin(job.plan),
            "index": index,
            "mode": row["mode"],
            "originalChunk": row["originalChunk"],
            "adoptedProof": row["adoptedProof"],
            "verification": admission.verify_admission(chunk, verify_repair=True),
            "repairEvidence": repair,
            "qualityApproved": False,
        }
        if old._read(old._path(row["proof"]), {}) != expected:
            raise ValueError("Original recovery proof differs")


def _load(operation: Path) -> common.Job:
    raw = common._read(operation)
    if (
        set(raw) != {"version", "protocol", "request", "workRoot", "readerCode"}
        or raw["version"] != 1
        or raw["protocol"] != PROTOCOL
        or raw["readerCode"] != _code()
    ):
        raise ValueError("Retained finalization operation changed")
    return original._request(
        common._ref(raw["request"]), operation, common._path(raw["workRoot"]), _code()
    )


@contextmanager
def _ready(job: common.Job) -> Iterator[dict[str, object]]:
    metadata = original._generation(job.plan)
    with ExitStack() as stack:
        for state in (
            [state for _, state, _ in job.lanes]
            + [old._path(name) for name in old._strings(metadata["states"])]
            + [old._path(name) for name in old._strings(metadata.get("oldStates", []))]
        ):
            stack.enter_context(old._unlocked(state / ".workflow.lock"))
        paths = [
            job.operation,
            job.plan,
            *[Path(name) for name in old._object(metadata.get("inputs", {}))],
            *[Path(name) for name in job.code],
        ]
        settings = common._read(common._path(metadata["settings"]))
        paths.extend(
            Path(str(record(settings[key])["path"]))
            for key in ("source", "native", "decoder", "model")
        )
        paths.extend(
            old._path(record(row)["chunk"]).parent / "repair/candidate.mp4"
            for row in old._list(metadata["chunks"])
        )
        cache = InputHashCache(tuple(dict.fromkeys(paths)), allow_uncached_overflow=True)
        cache.prime()
        with input_hash_context(cache):
            data = recovery._load(job.plan, retained=True)

            def check() -> None:
                for graph, state, status in job.lanes:
                    if bounded_read(status).strip() != b"0":
                        raise ValueError("Original parent is not terminal zero")
                    for stage in workflow.load_stages(graph):
                        recovery.retained_complete(stage, state, data["readerCode"])
                for graph_name, state_name in zip(
                    old._strings(data["workflows"]), old._strings(data["states"]), strict=True
                ):
                    for stage in workflow.load_stages(Path(graph_name)):
                        recovery.retained_complete(stage, Path(state_name), data["readerCode"])
                for tool in job.tools:
                    tool_reference(tool, {}, executable=True)

            check()
            yield data
            check()
            if recovery._load(job.plan, retained=True) != data:
                raise ValueError("Recovery changed during finalization")
            cache.verify()
        _load(job.operation)


def _seal(job: common.Job, data: dict[str, object], *, create: bool) -> dict[str, object]:
    settings_path = common._path(data["settings"])
    pins: dict[str, str] = {}
    settings = assembly._read(settings_path, pins)
    clock_path, clock = assembly._ref(settings["sourceClock"], pins)
    clocks = clock_chunks(clock, maximum_frames=assembly._positive(settings["chunkFrames"]))
    rows: list[dict[str, object]] = []
    refs: list[dict[str, object]] = []
    canvases: set[tuple[int, int]] = set()
    for index, (raw, physical) in enumerate(zip(old._list(data["chunks"]), clocks, strict=True)):
        verify_chunk(job, data, index)
        row, ref, canvas = assembly._chunk(
            old._path(record(raw)["chunk"]).parent,
            physical,
            settings_path,
            assembly._identity(settings["source"]),
            assembly._identity(settings["native"]),
            pins,
        )
        rows.append(row)
        refs.append(ref)
        canvases.add(canvas)
    if len(canvases) != 1:
        raise ValueError("Original chunk canvases differ")
    final = job.root / "finalized"
    if create:
        old._absent(final)
        final.mkdir()
    values: dict[str, object] = {
        "index.json": {"ranges": rows, "purpose": "explicit recovery selected results"}
    }
    if create:
        publish(final / "index.json", encode(values["index.json"]))
    values["manifest.json"] = {
        "version": 1,
        "index": common._pin(final / "index.json"),
        "chunks": refs,
    }
    if create:
        publish(final / "manifest.json", encode(values["manifest.json"]))
    values["lineage.json"] = {
        "version": 1,
        "generation": common._pin(job.plan),
        "manifest": common._pin(final / "manifest.json"),
        "selectedResults": data["chunks"],
        "requiredNextStage": "delivery_source_assembly full packet/decoded PTS verification",
        "qualityApproved": False,
        "deliveryComplete": False,
    }
    if create:
        publish(final / "lineage.json", encode(values["lineage.json"]))
    if any(common._read(final / name) != value for name, value in values.items()):
        raise ValueError("Sealed original manifest changed")
    media.load_chunks(
        final / "manifest.json", sha256=common._pin(final / "manifest.json")["sha256"], bindings={}
    )
    return {
        "version": 1,
        "protocol": PROTOCOL,
        "operation": common._pin(job.operation),
        "sourceClock": common._pin(clock_path),
        "physicalSourceFrames": sum(len(x.pts) for x in clocks),
        "outputs": {
            str(final / name): common._pin(final / name)["sha256"] for name in original.FILES
        },
    }


def _command(mode: str, operation: Path) -> list[str]:
    return [
        sys.executable,
        "-B",
        "-m",
        "replay.retained_cleanup_finalization",
        mode,
        str(operation),
    ]


def _child(job: common.Job) -> dict[str, object]:
    child = original._child(job)
    stage = record(old._list(child["stages"])[0])
    stage["verify_command"] = _command("verify-assembly", job.operation)
    return {"version": 1, "stages": [stage]}


def _prepared(job: common.Job) -> dict[str, object]:
    graph = job.root / "child/workflow.json"
    if common._read(graph) != _child(job):
        raise ValueError("Normal assembly child changed")
    return {
        "version": 1,
        "protocol": PROTOCOL,
        "operation": common._pin(job.operation),
        "workflow": common._pin(graph),
        "finalized": common._pin(job.root / "finalized.json"),
    }


def _completed(job: common.Job) -> dict[str, object]:
    if common._read(job.root / "prepared.json") != _prepared(job):
        raise ValueError("Prepared checkpoint changed")
    old._complete(
        workflow.load_stages(job.root / "child/workflow.json")[0], job.root / "child-state"
    )
    result = common._assembled(job)
    result["protocol"] = PROTOCOL
    result["childStageReceipt"] = common._pin(job.root / "child-state/assemble-clean-source.json")
    return result


def run(operation: Path, mode: str) -> None:
    job = _load(operation)
    with _ready(job) as data:
        if mode == "finalize":
            publish(job.root / "finalized.json", encode(_seal(job, data, create=True)))
            return
        if common._read(job.root / "finalized.json") != _seal(job, data, create=False):
            raise ValueError("Finalized proof changed")
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
                raise ValueError("Normal assembly preflight failed")
            publish(job.root / "prepared.json", encode(_prepared(job)))
            return
        if common._read(job.root / "prepared.json") != _prepared(job):
            raise ValueError("Prepared checkpoint changed")
        if mode == "verify-prepare-child":
            return
        if mode == "verify-assembly":
            common._assembled(job)
            return
        if mode not in {"run-child", "verify-run-child"}:
            raise ValueError("Unknown retained finalization mode")
        if mode == "run-child" and not (job.root / "completion.json").exists():
            chunks = media.load_chunks(
                job.root / "finalized/manifest.json",
                sha256=common._pin(job.root / "finalized/manifest.json")["sha256"],
                bindings={},
            )
            usage = shutil.disk_usage(job.root)
            if (
                usage.free
                < max(30 * 1024**3, usage.total // 20)
                + sum(x.media.stat().st_size for x in chunks)
                + 64 * 1024**2
            ):
                raise ValueError("Insufficient exact-clock assembly headroom")
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
                raise ValueError("Normal assembly stopped; retain partial outputs")
            publish(job.root / "completion.json", encode(_completed(job)))
        if common._read(job.root / "completion.json") != _completed(job):
            raise ValueError("Completed assembly checkpoint changed")


def prepare(config: Path, output: Path) -> Path:
    old._absent(output)
    common._path(str(output))
    operation = output / "operation.json"
    code = _code()
    job = original._request(config, operation, output, code)
    raw = original._generation(job.plan)
    inputs = [
        str(operation),
        str(config),
        str(job.plan),
        *code,
        *old._strings(raw["workflows"]),
        *[str(path) for graph, _, status in job.lanes for path in (graph, status)],
    ]
    request = common._read(config)
    if "ownerObservation" in request:
        inputs.append(str(common._ref(request["ownerObservation"])))
    final = [str(output / "finalized" / name) for name in original.FILES] + [
        str(output / "finalized.json")
    ]
    child = [str(output / "child/workflow.json"), str(output / "prepared.json")]
    completed = [
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
                "verify_command": _command("verify-" + mode, operation),
                "inputs": required,
                "outputs": produced,
                "timeout_seconds": job.timeout,
            }
            for mode, required, produced in zip(
                MODES,
                (inputs, inputs + final, inputs + final + child),
                (final, child, completed),
                strict=True,
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
                raise ValueError("Runtime requires pinned operation")
            run(args.input, args.mode)
        return 0
    except (OSError, ValueError, KeyError, ReplayError, StageError) as error:
        print(f"RETAINED_FINALIZATION_ERROR {error}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
