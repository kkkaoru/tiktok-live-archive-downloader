"""Finalize genuine original/session chunks with ordinary exact-clock assembly.

Preparation is metadata-only. Runtime seals dynamically produced consumer pins,
never invents future completion, reruns OCR or rewrites original evidence. Root
owns resource admission; workers must be terminal before this finite workflow.
"""

from __future__ import annotations

import argparse
import shutil
import sys
from collections.abc import Iterator
from contextlib import ExitStack, contextmanager
from dataclasses import dataclass
from pathlib import Path

from replay import cleanup_finalization_workflow as common
from replay import cleanup_lane_successor as proof
from replay import code_dependency_inventory, workflow
from replay import code_generation_compat as compat
from replay import delivery_source_assembly as media
from replay import subtitle_cleanup_assembly as legacy
from replay import subtitle_cleanup_workflow as original_domains
from replay import subtitle_session_cleanup as consumer
from replay import subtitle_session_cleanup_wave as wave
from replay.core import ReplayError
from replay.fcp_transformed_source import clean_source
from replay.input_hash_cache import InputHashCache, input_hash_context
from replay.recordings import record
from replay.sequence_journal import bounded_read, encode, publish
from replay.stages import Stage, StageError, file_digest
from replay.subtitle_cleanup_plan import clock_chunks
from replay.subtitle_clock import VideoFrameClock
from replay.subtitle_eligibility import _reference as tool_reference

PROTOCOL = "session-cleanup-finalization-v1"
MAX_CONTROLLERS = 64
MAX_SECONDS = 43200
FINAL_FILES = ("index.json", "manifest.json", "preparation.json")
MODES = ("finalize", "prepare-child", "run-child")


@dataclass(frozen=True)
class Controller:
    graph: Path
    state: Path
    status: Path


@dataclass(frozen=True)
class Selection:
    index: int
    operation: Path
    graph: Path
    state: Path


@dataclass(frozen=True)
class Plan:
    request: Path
    original: Path
    originals: tuple[tuple[Path, Path], ...]
    sessions: tuple[Selection, ...]
    controllers: tuple[Controller, ...]
    clocks: tuple[VideoFrameClock, ...]
    settings: Path
    tools: tuple[dict[str, object], dict[str, object]]
    timeout: int
    pins: dict[str, str]


@dataclass(frozen=True)
class Job:
    operation: Path
    root: Path
    plan: Plan
    code: dict[str, str]


def _code() -> dict[str, str]:
    return {
        str(item.path): item.sha256
        for item in code_dependency_inventory.capture(("subtitle_session_cleanup_finalization",))
    }


def _rows(value: object, maximum: int) -> list[object]:
    if not isinstance(value, list) or not 1 <= len(value) <= maximum:
        raise ValueError("Expected bounded nonempty finalization rows")
    return list(value)


def _read_ref(value: object, pins: dict[str, str]) -> Path:
    path = common._ref(value)
    pins[str(path)] = common._pin(path)["sha256"]
    return path


def _controller(value: object, pins: dict[str, str]) -> Controller:
    row = record(value)
    if set(row) != {"workflow", "state", "status"}:
        raise ValueError("Expected exact controller terminal binding")
    return Controller(
        _read_ref(row["workflow"], pins), common._path(row["state"]), common._path(row["status"])
    )


def _historical_inputs(stage: Stage) -> list[dict[str, str]]:
    """Known-generation input digests for evidence recorded before the current code."""
    return compat.historical_stage_input_sets(
        {str(path): file_digest(path) for path in stage.inputs}
    )


def _selections(
    value: object, *, original: Path, clocks: tuple[VideoFrameClock, ...], pins: dict[str, str]
) -> tuple[Selection, ...]:
    selected: dict[int, Selection] = {}
    for reference in _rows(value, legacy.MAX_CHUNKS):
        path = _read_ref(reference, pins)
        data = common._read(path)
        if data.get("protocol") != wave.PROTOCOL or data.get("version") != 1:
            raise ValueError("Expected original ordinary session wave")
        lanes = _rows(data.get("lanes"), wave.MAX_LANES)
        lane_refs: list[tuple[Path, Path]] = []
        for raw in lanes:
            row = record(raw)
            if set(row) != {"workflow", "state"}:
                raise ValueError("Invalid wave lane binding")
            lane_refs.append((_read_ref(row["workflow"], pins), common._path(row["state"])))
        for raw in _rows(data.get("chunks"), wave.MAX_CHUNKS):
            row = record(raw)
            operation = _read_ref(row.get("operation"), pins)
            bound, _ = wave._operation(operation, allow_historical=True)
            job = consumer.load_config(wave._reference(bound["draft"]), require_observation=False)
            index = common._read(job.config).get("chunkIndex")
            if type(index) is not int:
                raise ValueError("Expected exact physical session chunk index")
            lane = row.get("lane")
            if (
                type(lane) is not int
                or not 0 <= lane < len(lane_refs)
                or job.original_plan != original
                or index in selected
                or not 0 <= index < len(clocks)
                or job.clock != clocks[index]
                or row.get("chunkName") != job.chunk.parent.name
                or row.get("frames") != len(job.clock.pts)
            ):
                raise ValueError("Session wave membership/source/physical clock differs")
            graph, state = lane_refs[lane]
            selected[index] = Selection(index, operation, graph, state)
    return tuple(selected[index] for index in sorted(selected))


def _coverage(plan: Plan) -> None:
    known = {item.operation: item for item in plan.sessions}
    covered: list[Path] = []
    graphs = {item.graph for item in plan.sessions}
    for controller in plan.controllers:
        for stage in workflow.load_stages(controller.graph):
            if controller.graph in graphs:
                if stage.command[1:5] != (
                    "-B",
                    "-m",
                    "replay.subtitle_session_cleanup_wave",
                    "run-child",
                ):
                    continue
                operation = Path(stage.command[-1])
                if operation not in known or stage.verify_command != tuple(
                    wave._command("verify-run-child", operation)
                ):
                    raise ValueError("Controller has an unknown session producer")
            else:
                command = stage.command
                if (
                    len(command) != 9
                    or command[1:4] != ("-B", "-m", "replay.workflow")
                    or command[5] != "--state"
                    or command[7] != "--target"
                ):
                    raise ValueError("Continuation must invoke the ordinary child workflow target")
                graph, state, target = Path(command[4]), Path(command[6]), command[8]
                child = {item.name: item for item in workflow.load_stages(graph)}
                if graph not in graphs or target not in child:
                    raise ValueError("Unknown continuation child target")
                final = child[target]
                operation = Path(final.command[-1])
                selected = known.get(operation)
                if (
                    selected is None
                    or selected.graph != graph
                    or selected.state != state
                    or target != f"chunk-{selected.index:05d}-wave-run-child"
                ):
                    raise ValueError("Continuation target differs from its physical session chunk")
                phases = tuple(
                    item
                    for item in child.values()
                    if item.name.startswith(f"chunk-{selected.index:05d}-wave-")
                )
                outputs = {path for item in phases for path in item.outputs}
                outputs.update(state / f"{item.name}.json" for item in phases)
                if (
                    set(stage.outputs) != outputs
                    or stage.verify_command != final.verify_command
                    or not {graph, operation}.issubset(stage.inputs)
                ):
                    raise ValueError(
                        "Continuation must retain all ordinary child outputs and verifier"
                    )
            covered.append(operation)
    if len(covered) != len(known) or set(covered) != set(known):
        raise ValueError("Controllers must cover each session chunk exactly once")


def _request(path: Path) -> Plan:
    raw = common._read(path)
    if (
        set(raw)
        != {
            "version",
            "originalPlan",
            "originalWorkers",
            "waves",
            "controllers",
            "ffmpeg",
            "ffprobe",
            "timeoutSeconds",
        }
        or type(raw.get("version")) is not int
        or raw["version"] != 1
    ):
        raise ValueError("Invalid mixed session finalization request")
    timeout = raw["timeoutSeconds"]
    if type(timeout) is not int or not 1 <= timeout <= MAX_SECONDS:
        raise ValueError("Expected bounded finalization deadline")
    pins = {str(path): common._pin(path)["sha256"]}
    original = _read_ref(raw["originalPlan"], pins)
    data = common._read(original)
    settings = _read_ref(data.get("settings"), pins)
    settings_data = common._read(settings)
    clock = _read_ref(settings_data.get("sourceClock"), pins)
    clocks = clock_chunks(
        common._read(clock), maximum_frames=legacy._positive(settings_data.get("chunkFrames"))
    )
    frames = sum(len(item.pts) for item in clocks)
    if (
        not clocks
        or len(clocks) > legacy.MAX_CHUNKS
        or clocks[0].pts[0] != 0
        or data.get("wholeSourcePlanned") is not True
        or data.get("chunks") != len(clocks)
        or data.get("plannedFrames") != frames
        or data.get("totalSourceFrames") != frames
    ):
        raise ValueError("Finalization needs the full original physical source plan")
    originals: list[tuple[Path, Path]] = []
    for value in _rows(raw["originalWorkers"], MAX_CONTROLLERS):
        row = record(value)
        if set(row) != {"workflow", "state"}:
            raise ValueError("Expected original worker graph/state binding")
        originals.append((_read_ref(row["workflow"], pins), common._path(row["state"])))
    controllers = tuple(
        _controller(item, pins) for item in _rows(raw["controllers"], MAX_CONTROLLERS)
    )
    if len(
        {item for control in controllers for item in (control.graph, control.state, control.status)}
    ) != 3 * len(controllers) or len(set(originals)) != len(originals):
        raise ValueError("Controller/original worker namespaces must be unique")
    sessions = _selections(raw["waves"], original=original, clocks=clocks, pins=pins)
    tools = record(raw["ffmpeg"]), record(raw["ffprobe"])
    for tool in tools:
        if set(tool) != {"path", "sha256"}:
            raise ValueError("Expected exact installed assembly tool pin")
        media._path(tool["path"])
        legacy._sha(tool["sha256"])
    plan = Plan(
        path,
        original,
        tuple(originals),
        sessions,
        controllers,
        clocks,
        settings,
        tools,
        timeout,
        pins,
    )
    _coverage(plan)
    return plan


def _load(operation: Path) -> Job:
    raw = common._read(operation)
    if (
        set(raw) != {"version", "protocol", "request", "workRoot", "readerCode"}
        or type(raw.get("version")) is not int
        or raw["version"] != 1
        or raw["protocol"] != PROTOCOL
        or raw["readerCode"] != _code()
    ):
        raise ValueError("Finalization operation or reader code changed")
    return Job(
        operation, common._path(raw["workRoot"]), _request(common._ref(raw["request"])), _code()
    )


def _original_stages(plan: Plan) -> tuple[tuple[Stage, Path], ...]:
    selected = {item.index for item in plan.sessions}
    settings = common._read(plan.settings)
    result: list[tuple[Stage, Path]] = []
    for index in range(len(plan.clocks)):
        if index in selected:
            continue
        chunk = plan.original.parent / f"chunk-{index:05d}/chunk.json"
        matches = tuple(
            (graph, state)
            for graph, state in plan.originals
            if f"chunk-{index:05d}-ocr" in {stage.name for stage in workflow.load_stages(graph)}
        )
        if len(matches) != 1:
            raise ValueError("Each retained original chunk needs one genuine original worker")
        graph, state = matches[0]
        result.extend((stage, state) for stage in consumer._old_stages(graph, chunk, settings))
    return tuple(result)


@contextmanager
def _ready(job: Job) -> Iterator[None]:
    with ExitStack() as stack:
        locks = {item.state for item in job.plan.controllers}
        locks.update(state for _, state in job.plan.originals)
        locks.update(item.state for item in job.plan.sessions)
        for state in sorted(locks):
            stack.enter_context(proof._unlocked(state / ".workflow.lock"))
        for controller in job.plan.controllers:
            if bounded_read(controller.status).strip() != b"0":
                raise ValueError("All session controllers must be genuine terminal zero")
        names = set(job.code) | set(job.plan.pins)
        original_stages = _original_stages(job.plan)
        controller_stages = tuple(
            (stage, item.state)
            for item in job.plan.controllers
            for stage in workflow.load_stages(item.graph)
        )
        evidence_stages = list(original_stages + controller_stages)
        names.update(str(item.status) for item in job.plan.controllers)
        names.update(str(path) for stage, _ in original_stages for path in stage.inputs)
        for selection in job.plan.sessions:
            _, work = wave._operation(selection.operation, allow_historical=True)
            current = consumer.load_config(work / "owner/config-ready.json")
            # Raw OCR metadata stays freshly checked by its original collector.
            # Cache only the bounded shared assets, not every other chunk's raw
            # files; neither the cache budget nor its 4096-input guard expands.
            names.update(consumer._asset_names(current.settings))
            names.update((str(current.provider.source), str(current.provider.native)))
            names.add(str(work / "consumer/repair/candidate.mp4"))
            evidence_stages.extend(
                (stage, work / "consumer/state")
                for stage in workflow.load_stages(work / "timed/workflow.json")
            )
            stack.enter_context(proof._unlocked(work / "consumer/state/.workflow.lock"))
        names.update(str(state / f"{stage.name}.json") for stage, state in evidence_stages)
        cache = InputHashCache(
            tuple(Path(name) for name in sorted(names)), allow_uncached_overflow=True
        )
        cache.prime()
        with input_hash_context(cache):
            for stage, state in evidence_stages:
                proof._complete(stage, state, historical_input_sets=_historical_inputs(stage))
            for tool in job.plan.tools:
                tool_reference(tool, {}, executable=True)
            selected = {item.index for item in job.plan.sessions}
            for index in range(len(job.plan.clocks)):
                if index not in selected:
                    chunk = job.plan.original.parent / f"chunk-{index:05d}/chunk.json"
                    original_domains.check_ocr(chunk)
                    original_domains.check_repair(chunk)
            initial = _mixed(job)
            yield
            if any(bounded_read(item.status).strip() != b"0" for item in job.plan.controllers):
                raise ValueError("Controller outcome changed during finalization")
            for stage, state in evidence_stages:
                proof._complete(stage, state, historical_input_sets=_historical_inputs(stage))
            if _mixed(job) != initial:
                raise ValueError("Consumer lineage changed during finalization")
            for index in range(len(job.plan.clocks)):
                if index not in selected:
                    chunk = job.plan.original.parent / f"chunk-{index:05d}/chunk.json"
                    original_domains.check_ocr(chunk)
                    original_domains.check_repair(chunk)
            cache.verify()
            _load(job.operation)


def _mixed(job: Job) -> dict[str, object]:
    sessions: list[dict[str, object]] = []
    for selection in job.plan.sessions:
        data, work = wave._operation(selection.operation, allow_historical=True)
        wave._prepared(data, work)
        for stage in workflow.load_stages(work / "timed/workflow.json"):
            proof._complete(
                stage, work / "consumer/state", historical_input_sets=_historical_inputs(stage)
            )
        current = consumer.load_config(work / "owner/config-ready.json")
        admitted = consumer._admitted(current, work / "consumer")
        consumer._documents(work / "consumer", admitted[:3], verify=True)
        receipt = common._read(work / "consumer/repair/candidate.json")
        for name, sha in record(receipt.get("inputs")).items():
            current_digest = file_digest(consumer._path(name, local=False))
            historical = compat.accepted_input_digest(name, sha, current_digest)
            if current_digest != consumer._sha(sha) and not historical:
                raise ValueError("Session repair input changed")
        operation = work / "consumer/operation.json"
        sessions.append({"chunkIndex": selection.index, "operation": common._pin(operation)})
    return {"version": 1, "originalPlan": common._pin(job.plan.original), "sessions": sessions}


def _manifest(job: Job) -> dict[str, object]:
    """Use original mixed/legacy typed domains, never call the preparation writer."""
    mixed = _mixed(job)
    pins: dict[str, str] = {}
    if consumer._read(job.root / "mixed-config.json", pins) != mixed:
        raise ValueError("Dynamic consumer pins or mixed lineage changed")
    selected: dict[int, Path] = {}
    consumer._reference(mixed["originalPlan"], pins)
    for value in _rows(mixed["sessions"], legacy.MAX_CHUNKS):
        row = record(value)
        index = row["chunkIndex"]
        if type(index) is not int:
            raise ValueError("Invalid sealed physical chunk index")
        selected[index] = consumer._pin_reference(row["operation"], pins)
    settings, raw = legacy._ref(common._pin(job.plan.settings), pins)
    clock, _ = legacy._ref(raw["sourceClock"], pins)
    source, native = legacy._identity(raw["source"]), legacy._identity(raw["native"])
    rows: list[dict[str, object]] = []
    refs: list[dict[str, object]] = []
    canvases: set[tuple[int, int]] = set()
    for index, frame_clock in enumerate(job.plan.clocks):
        if index in selected:
            row, ref, canvas = consumer._session_assembly_row(
                selected[index],
                original=job.plan.original,
                clock=frame_clock,
                pins=pins,
                allow_historical=True,
            )
        else:
            row, ref, canvas = legacy._chunk(
                job.plan.original.parent / f"chunk-{index:05d}",
                frame_clock,
                settings,
                source,
                native,
                pins,
            )
        rows.append(row)
        refs.append(ref)
        canvases.add(canvas)
    if len(canvases) != 1:
        raise ValueError("Mixed cleanup geometry differs")
    final = job.root / "finalized"
    if common._read(final / "index.json") != {
        "purpose": "mixed cleanup source-clock candidates; quality unapproved",
        "ranges": rows,
    } or common._read(final / "manifest.json") != {
        "version": 1,
        "index": common._pin(final / "index.json"),
        "chunks": refs,
    }:
        raise ValueError("Mixed assembly manifest/index differs from real chunk domains")
    saved = common._read(final / "preparation.json")
    if saved != {
        "version": 1,
        "protocol": consumer.PROTOCOL,
        "inputs": pins,
        "currentPreparationCode": consumer._code(),
        "sourceIdentity": raw["source"],
        "sourceClock": common._pin(clock),
        "physicalSourceFrames": sum(len(item.pts) for item in job.plan.clocks),
        "physicalSourceTerminalPTS": job.plan.clocks[-1].pts[-1]
        + job.plan.clocks[-1].last_duration,
        "sessionChunkIndices": sorted(selected),
        "manifest": common._pin(final / "manifest.json"),
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
    }:
        raise ValueError("Mixed assembly preparation clock/provenance differs")
    for name, sha in record(saved.get("inputs")).items():
        if common._pin(common._path(name))["sha256"] != sha:
            raise ValueError("Mixed preparation metadata changed")
    return {
        "version": 1,
        "protocol": PROTOCOL,
        "operation": common._pin(job.operation),
        "outputs": {str(final / name): common._pin(final / name)["sha256"] for name in FINAL_FILES},
    }


def _command(mode: str, operation: Path) -> list[str]:
    return [
        sys.executable,
        "-B",
        "-m",
        "replay.subtitle_session_cleanup_finalization",
        mode,
        str(operation),
    ]


def _child(job: Job) -> dict[str, object]:
    manifest = job.root / "finalized/manifest.json"
    command = [
        sys.executable,
        "-B",
        "-m",
        "replay.delivery_source_assembly",
        str(manifest),
        str(job.root / "clean-source.mov"),
        str(job.root / "evidence"),
        "--sha256",
        common._pin(manifest)["sha256"],
        "--ffmpeg",
        str(job.plan.tools[0]["path"]),
        "--ffprobe",
        str(job.plan.tools[1]["path"]),
    ]
    return {
        "version": 1,
        "stages": [
            {
                "name": "assemble-clean-source",
                "command": command,
                "inputs": list(
                    dict.fromkeys(
                        [
                            str(job.operation),
                            str(manifest),
                            str(job.root / "finalized/index.json"),
                            *job.code,
                            *(str(tool["path"]) for tool in job.plan.tools),
                        ]
                    )
                ),
                "outputs": [
                    str(job.root / name)
                    for name in (
                        "clean-source.mov",
                        "evidence/receipt.json",
                        "evidence/single-source-index.json",
                    )
                ],
                "timeout_seconds": job.plan.timeout,
                "verify_command": _command("verify-assembly", job.operation),
            }
        ],
    }


def _prepared(job: Job) -> dict[str, object]:
    graph = job.root / "child/workflow.json"
    if common._read(graph) != _child(job):
        raise ValueError("Assembly child graph differs from generated manifest")
    return {
        "version": 1,
        "protocol": PROTOCOL,
        "workflow": common._pin(graph),
        "finalized": common._pin(job.root / "finalized.json"),
    }


def _assembled(job: Job) -> dict[str, object]:
    clean_source(
        {
            "assemblyManifest": common._pin(job.root / "finalized/manifest.json"),
            "assemblyReceipt": common._pin(job.root / "evidence/receipt.json"),
            "cleanIndex": common._pin(job.root / "evidence/single-source-index.json"),
        }
    )
    receipt = common._read(job.root / "evidence/receipt.json")
    if record(receipt["output"])["path"] != str(job.root / "clean-source.mov"):
        raise ValueError("Assembly belongs to a different namespace")
    return {
        "version": 1,
        "protocol": PROTOCOL,
        "output": receipt["output"],
        "assemblyReceipt": common._pin(job.root / "evidence/receipt.json"),
        "qualityApproved": False,
        "avSyncApproved": False,
    }


def _completed(job: Job) -> dict[str, object]:
    if common._read(job.root / "prepared.json") != _prepared(job):
        raise ValueError("Prepared child checkpoint changed")
    stage = workflow.load_stages(job.root / "child/workflow.json")[0]
    proof._complete(stage, job.root / "child-state")
    result = _assembled(job)
    result["childStageReceipt"] = common._pin(job.root / "child-state/assemble-clean-source.json")
    return result


def prepare(config: Path, output: Path) -> Path:
    """Create a normal bounded graph without guessing future output SHA or readiness."""
    common._path(str(output))
    if not output.is_dir() or tuple(output.iterdir()):
        raise ValueError("Expected existing empty finalization work root")
    plan, code = _request(config), _code()
    operation = output / "operation.json"
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
    (output / "finalized").mkdir()
    (output / "child").mkdir()
    raw = common._read(config)
    wave_paths = [str(common._ref(value)) for value in _rows(raw["waves"], legacy.MAX_CHUNKS)]
    inputs = list(
        dict.fromkeys(
            [
                str(operation),
                str(config),
                str(plan.original),
                *wave_paths,
                *code,
                *(str(item.graph) for item in plan.controllers),
                *(str(item.status) for item in plan.controllers),
            ]
        )
    )
    final = [
        str(output / "mixed-config.json"),
        *[str(output / "finalized" / name) for name in FINAL_FILES],
        str(output / "finalized.json"),
    ]
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
    phases = (
        (MODES[0], inputs, final),
        (MODES[1], inputs + final, child),
        (MODES[2], inputs + final + child, result),
    )
    data = {
        "version": 1,
        "stages": [
            {
                "name": mode,
                "command": _command(mode, operation),
                "inputs": required,
                "outputs": produced,
                "timeout_seconds": plan.timeout,
                "verify_command": _command("verify-" + mode, operation),
            }
            for mode, required, produced in phases
        ],
    }
    workflow.stage_graph(workflow.parse_stages(data))
    graph = output / "workflow.json"
    publish(graph, encode(data))
    return graph


def _continue(job: Job, mode: str) -> None:
    if mode == "prepare-child":
        graph = job.root / "child/workflow.json"
        publish(graph, encode(_child(job)))
        if workflow.main([str(graph), "--state", str(job.root / "child-state"), "--check"]) != 0:
            raise ValueError("Normal assembly child preflight failed")
        publish(job.root / "prepared.json", encode(_prepared(job)))
    elif mode == "verify-prepare-child":
        if common._read(job.root / "prepared.json") != _prepared(job):
            raise ValueError("Prepared child changed")
    elif mode == "verify-assembly":
        _assembled(job)
    elif mode == "run-child":
        if common._read(job.root / "prepared.json") != _prepared(job):
            raise ValueError("Prepared child changed")
        if (job.root / "completion.json").exists():
            if common._read(job.root / "completion.json") != _completed(job):
                raise ValueError("Completed child changed; no reissue")
            return
        stage = workflow.load_stages(job.root / "child/workflow.json")[0]
        if any(path.exists() or path.is_symlink() for path in stage.outputs):
            proof._complete(stage, job.root / "child-state")
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
                [str(job.root / "child/workflow.json"), "--state", str(job.root / "child-state")]
            )
            != 0
        ):
            raise ValueError("Normal assembly child stopped; preserve all partial outputs")
        publish(job.root / "completion.json", encode(_completed(job)))
    elif mode == "verify-run-child":
        if common._read(job.root / "completion.json") != _completed(job):
            raise ValueError("Completed child changed")
    else:
        raise ValueError("Unknown finalization mode")


def run(operation: Path, *, mode: str) -> None:
    """Finite producer/readonly verifier; never poll a lane or rerun a chunk."""
    job = _load(operation)
    with _ready(job):
        if mode == "finalize":
            publish(job.root / "mixed-config.json", encode(_mixed(job)))
            consumer.prepare_assembly(
                job.root / "mixed-config.json",
                job.root / "finalized",
                allow_historical=True,
            )
            publish(job.root / "finalized.json", encode(_manifest(job)))
        elif mode == "verify-finalize":
            if common._read(job.root / "finalized.json") != _manifest(job):
                raise ValueError("Finalized checkpoint changed")
        else:
            if common._read(job.root / "finalized.json") != _manifest(job):
                raise ValueError("Finalized checkpoint changed")
            _continue(job, mode)


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
                raise ValueError("Preparation requires --output")
            print(prepare(args.input, args.output))
        elif args.output is not None:
            raise ValueError("Runtime uses only its pinned operation")
        else:
            run(args.input, mode=args.mode)
    except (OSError, ValueError, ReplayError, StageError) as error:
        print(f"SESSION_CLEANUP_FINALIZATION_ERROR {error}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
