"""Normal Stage receipts connect contrasting recovery generations without OCR."""

import fcntl
import shutil
import sys
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

import pytest
from test_cleanup_finalization_workflow import Usage, chunk, write

from replay import cleanup_finalization_workflow as common
from replay import cleanup_generation_recovery as recovery
from replay import cleanup_lane_successor as old
from replay import delivery_source_assembly as media
from replay import recovery_source_finalization as flow
from replay import workflow
from replay.core import ReplayError
from replay.recordings import record
from replay.stages import Stage, StageError, file_digest, run_stage
from replay.subtitle_cleanup_plan import clock_chunks


@dataclass(frozen=True)
class Case:
    request: Path
    output: Path
    plan: Path
    settings: Path
    source: Path
    model: Path
    code: Path
    graphs: tuple[Path, ...]
    states: tuple[Path, ...]
    statuses: tuple[Path, ...]
    segments: tuple[Path, ...]
    frames: int

    @property
    def operation(self) -> Path:
        return self.output / "operation.json"


def complete(stage: Stage, state: Path) -> None:
    def execute(command: tuple[str, ...], log: Path, timeout: int) -> None:
        if command == stage.command:
            if command[:4] == (sys.executable, "-B", "-m", "replay.workflow"):
                for child in workflow.load_stages(Path(command[4])):
                    complete(child, Path(command[6]))
            else:
                for output in stage.outputs:
                    output.parent.mkdir(parents=True, exist_ok=True)
                    output.write_bytes(b"real synthetic Stage output")
        log.write_text("synthetic local producer; no OCR", encoding="utf-8")

    run_stage(stage, state_directory=state, execute=execute)


def rewrite_request(case: Case, field: str, value: object) -> None:
    raw = common._read(case.request)
    raw[field] = value
    write(case.request, raw)


def repin_plan(case: Case) -> None:
    rewrite_request(case, "generation", common._pin(case.plan))


@pytest.fixture(params=[((0, 3, 9, 14, 20), 3, 1), ((0, 1, 5, 6, 9, 13, 14), 3, 2)])
def case(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, request: pytest.FixtureRequest) -> Case:
    monkeypatch.chdir(tmp_path)
    work = tmp_path / "edits/work"
    work.mkdir(parents=True)
    parameters = request.param
    if not isinstance(parameters, tuple):
        raise ValueError("Missing synthetic topology")
    points, chunk_frames, workers = parameters
    source, model, tool, code = (
        work / name for name in ("source.mp4", "model.bin", "tool", "reader.py")
    )
    source.write_bytes(b"synthetic physical source")
    model.write_bytes(b"pinned synthetic model")
    tool.write_bytes(b"#!/bin/sh\nexit 0\n")
    tool.chmod(0o700)
    code.write_bytes(b"pinned current reader")
    clock = work / "clock.json"
    write(clock, {"pts": list(points), "timeBase": "1/100", "lastDuration": 2})
    settings = work / "settings.json"
    write(
        settings,
        {
            "version": 1,
            "source": common._pin(source),
            "native": common._pin(tool),
            "decoder": common._pin(tool),
            "model": common._pin(model),
            "sourceClock": common._pin(clock),
            "chunkFrames": chunk_frames,
        },
    )
    clocks = clock_chunks(common._read(clock), maximum_frames=chunk_frames)
    original = work / "original-plan.json"
    write(
        original,
        {
            "version": 1,
            "settings": common._pin(settings),
            "wholeSourcePlanned": True,
            "chunks": len(clocks),
            "plannedFrames": len(points),
            "totalSourceFrames": len(points),
        },
    )
    rows: list[dict[str, object]] = []
    for index, physical in enumerate(clocks):
        directory = work / f"selected-{index}"
        chunk(
            directory,
            settings,
            {
                "pts": list(physical.pts),
                "timeBase": "1/100",
                "lastDuration": physical.last_duration,
            },
        )
        # Keep a retained canonical chunk and a separate pending-generation chunk.
        selected = directory / ("chunk-successor.json" if index == 0 else "chunk.json")
        if index == 0:
            selected.write_bytes((directory / "chunk.json").read_bytes())
        rows.append(
            {
                "index": index,
                "lane": index % workers,
                "chunk": str(selected),
                "mode": "retained-completed" if index == 0 else "pending",
            }
        )
    plan = work / "generation.json"
    segments: list[Path] = []
    segment_states: list[Path] = []
    parents: list[Path] = []
    states: list[Path] = []
    statuses: list[Path] = []
    controls: list[dict[str, object]] = []
    lane_numbers: list[int] = []
    for lane in range(workers):
        parent_stages: list[dict[str, object]] = []
        for index in range(lane, len(clocks), workers):
            segment, state, result = (
                work / f"segment-{index}.json",
                work / f"segment-state-{index}",
                work / f"result-{index}.json",
            )
            write(
                segment,
                {
                    "version": 1,
                    "stages": [
                        {
                            "name": "selected-result",
                            "command": [str(tool)],
                            "inputs": [str(source), str(model)],
                            "outputs": [str(result)],
                            "timeout_seconds": 60,
                            "verify_command": [str(tool), "verify"],
                        }
                    ],
                },
            )
            segment_index = len(segments)
            segments.append(segment)
            segment_states.append(state)
            lane_numbers.append(lane)
            parent_stages.append(
                {
                    "name": f"segment-{index}",
                    "command": [
                        sys.executable,
                        "-B",
                        "-m",
                        "replay.workflow",
                        str(segment),
                        "--state",
                        str(state),
                    ],
                    "inputs": [str(segment), str(source), str(model)],
                    "outputs": [str(result)],
                    "timeout_seconds": 60,
                    "verify_command": [
                        sys.executable,
                        "-B",
                        "-m",
                        "replay.cleanup_generation_recovery",
                        "verify-segment",
                        str(plan),
                        "--index",
                        str(segment_index),
                    ],
                }
            )
        graph, state, status = (
            work / f"parent-{lane}.json",
            work / f"parent-state-{lane}",
            work / f"parent-{lane}.status",
        )
        write(graph, {"version": 1, "stages": parent_stages})
        for stage in workflow.load_stages(graph):
            complete(stage, state)
        status.write_text("0\n", encoding="utf-8")
        parents.append(graph)
        states.append(state)
        statuses.append(status)
        controls.append(
            {"workflow": common._pin(graph), "state": str(state), "status": str(status)}
        )
    data: dict[str, object] = {
        "version": 1,
        "kind": "cleanup-recovery-generation",
        "readerCode": {},
        "workers": workers,
        "chunks": rows,
        "workflows": [str(path) for path in segments],
        "states": [str(path) for path in segment_states],
        "workflowLanes": lane_numbers,
        "settings": str(settings),
        "originalPlan": str(original),
    }
    write(plan, data)
    bound = {
        str(path): file_digest(path) for path in (source, model, settings, clock, original, code)
    }
    monkeypatch.setattr(flow, "_code", lambda: {str(code): file_digest(code)})

    def load(path: Path, *, retained: bool = False) -> dict[str, object]:
        assert retained, "Finalization must explicitly read the retained generation"
        for name, sha in bound.items():
            if file_digest(Path(name)) != sha:
                raise ValueError("Original recovery input changed")
        return common._read(path)

    def verify(path: Path, index: int, *, verify: bool = False) -> None:
        assert path == plan and verify is True
        selected = Path(str(record(rows[index])["chunk"]))
        # Original collector/repair proof validation is delegated, never inferred.
        assert (
            common._read(selected)["clock"] == common._read(selected.parent / "chunk.json")["clock"]
        )

    monkeypatch.setattr(recovery, "_load", load)
    monkeypatch.setattr(recovery, "phase", verify)
    config = work / "request.json"
    write(
        config,
        {
            "version": 1,
            "generation": common._pin(plan),
            "controllers": controls,
            "ffmpeg": common._pin(tool),
            "ffprobe": common._pin(tool),
            "timeoutSeconds": 60,
        },
    )
    return Case(
        config,
        work / "finalization",
        plan,
        settings,
        source,
        model,
        code,
        tuple(parents),
        tuple(states),
        tuple(statuses),
        tuple(segments),
        len(points),
    )


def finalize(case: Case) -> common.Job:
    flow.prepare(case.request, case.output)
    flow.run(case.operation, "finalize")
    flow.run(case.operation, "verify-finalize")
    return flow._load(case.operation)


def materialize(job: common.Job) -> None:
    bindings: dict[str, str] = {}
    chunks = media.load_chunks(
        job.root / "finalized/manifest.json",
        sha256=file_digest(job.root / "finalized/manifest.json"),
        bindings=bindings,
    )
    frames = sum(len(chunk.clock.pts) for chunk in chunks)
    movie = job.root / "clean-source.mov"
    movie.write_bytes(b"synthetic assembled packets")
    evidence = job.root / "evidence"
    evidence.mkdir()
    probe = evidence / "probe.json"
    probe.write_bytes(b"real retained synthetic probe")
    index = evidence / "single-source-index.json"
    write(
        index,
        {
            "ranges": [
                {
                    "media": str(movie),
                    "sha256": file_digest(movie),
                    "sourceFirstPTS": chunks[0].clock.pts[0],
                    "sourceEndPTS": chunks[-1].clock.pts[-1] + chunks[-1].clock.last_duration,
                    "frames": frames,
                    "timeBase": str(chunks[0].clock.time_base),
                }
            ]
        },
    )
    write(
        evidence / "receipt.json",
        {
            "inputs": bindings,
            "output": common._pin(movie),
            "indexSHA256": file_digest(index),
            "probeEvidence": {str(probe): file_digest(probe)},
            "frames": frames,
            "exactPresentationClockVerified": True,
            "compressedPacketsIdentical": True,
        },
    )


def test_metadata_only_prepare_has_normal_structural_preflight(case: Case) -> None:
    graph = flow.prepare(case.request, case.output)
    assert workflow.main([str(graph), "--state", str(case.output / "parent-state"), "--check"]) == 0
    assert [stage.name for stage in workflow.load_stages(graph)] == list(flow.MODES)
    assert all(stage.verify_command for stage in workflow.load_stages(graph))
    assert not (case.output / "finalized").exists()
    assert not (case.output / "clean-source.mov").exists()
    with pytest.raises(ValueError, match="Occupied"):
        flow.prepare(case.request, case.output)


def test_full_original_recovery_assembler_and_readonly_child(
    case: Case, monkeypatch: pytest.MonkeyPatch
) -> None:
    # The real assembler publishes selection lineage using genuine Stage records.
    job = finalize(case)
    assert common._read(job.root / "finalized.json")["physicalSourceFrames"] == case.frames
    lineage = common._read(job.root / "finalized/lineage.json")
    assert lineage["qualityApproved"] is False and lineage["deliveryComplete"] is False
    before = [path.read_bytes() for path in case.states[0].glob("*.json")]
    flow.run(case.operation, "prepare-child")
    flow.run(case.operation, "verify-prepare-child")
    monkeypatch.setattr(shutil, "disk_usage", lambda path: Usage(10**12, 0, 10**12))
    calls: list[str] = []

    def execute(arguments: Sequence[str] | None = None) -> int:
        calls.append("run")
        stage = workflow.load_stages(job.root / "child/workflow.json")[0]

        def command(argv: tuple[str, ...], log: Path, timeout: int) -> None:
            if argv == stage.command:
                materialize(job)
            else:
                flow.run(case.operation, "verify-assembly")
            log.write_bytes(b"real local test execution")

        run_stage(stage, state_directory=job.root / "child-state", execute=command)
        return 0

    monkeypatch.setattr(workflow, "main", execute)
    flow.run(case.operation, "run-child")
    result_bytes = (job.root / "completion.json").read_bytes()
    flow.run(case.operation, "verify-run-child")
    flow.run(case.operation, "run-child")
    assert calls == ["run"]
    assert (job.root / "completion.json").read_bytes() == result_bytes
    assert [path.read_bytes() for path in case.states[0].glob("*.json")] == before
    result = common._read(job.root / "completion.json")
    assert result["protocol"] == flow.PROTOCOL
    assert result["qualityApproved"] is False and result["avSyncApproved"] is False


@pytest.mark.parametrize("mode", ["finalize", "verify-finalize", "prepare-child", "run-child"])
def test_nonzero_parent_refuses_every_phase(case: Case, mode: str) -> None:
    flow.prepare(case.request, case.output)
    case.statuses[0].write_bytes(b"1\n")
    with pytest.raises(ValueError, match="terminal zero"):
        flow.run(case.operation, mode)
    assert not (case.output / "finalized").exists()


@pytest.mark.parametrize("segment", [False, True])
def test_exclusive_parent_or_segment_lock_refuses(case: Case, segment: bool) -> None:
    flow.prepare(case.request, case.output)
    state = Path(old._strings(common._read(case.plan)["states"])[0]) if segment else case.states[0]
    with (state / ".workflow.lock").open("a") as stream:
        fcntl.flock(stream, fcntl.LOCK_EX | fcntl.LOCK_NB)
        with pytest.raises(ValueError, match="active"):
            flow.run(case.operation, "finalize")


def test_completed_parent_requires_unchanged_genuine_output(case: Case) -> None:
    flow.prepare(case.request, case.output)
    workflow.load_stages(case.graphs[0])[0].outputs[0].write_bytes(b"changed")
    with pytest.raises(ValueError, match="complete Stage"):
        flow.run(case.operation, "finalize")


@pytest.mark.parametrize("role", ["source", "model", "code"])
def test_pinned_source_model_reader_changes_refuse(case: Case, role: str) -> None:
    flow.prepare(case.request, case.output)
    {"source": case.source, "model": case.model, "code": case.code}[role].write_bytes(
        b"changed original"
    )
    with pytest.raises(ValueError, match="changed|complete Stage"):
        flow.run(case.operation, "finalize")


@pytest.mark.parametrize("file", list(flow.FILES) + ["finalized.json"])
def test_finalized_tamper_refuses_readonly(case: Case, file: str) -> None:
    finalize(case)
    path = case.output / file if file == "finalized.json" else case.output / "finalized" / file
    write(path, {"tamper": True})
    with pytest.raises(ValueError, match="differs|changed"):
        flow.run(case.operation, "verify-finalize")


def test_partial_finalization_is_preserved(case: Case) -> None:
    flow.prepare(case.request, case.output)
    original = case.output / "finalized/index.json"
    write(original, {"unapproved": True})
    with pytest.raises(ValueError, match="Occupied"):
        flow.run(case.operation, "finalize")
    assert common._read(original) == {"unapproved": True}


@pytest.mark.parametrize("mutation", ["outcome", "generation", "reader"])
def test_terminal_guard_rechecks_operation(
    case: Case, monkeypatch: pytest.MonkeyPatch, mutation: str
) -> None:
    flow.prepare(case.request, case.output)
    assemble = recovery.assemble

    def changed(
        plan: Path, output: Path, *, retained: bool = False, owner_observation: Path | None = None
    ) -> Path:
        result = assemble(plan, output, retained=retained, owner_observation=owner_observation)
        if mutation == "outcome":
            case.statuses[0].write_bytes(b"1\n")
        elif mutation == "generation":
            raw = common._read(case.plan)
            raw["changedDuringOperation"] = True
            write(case.plan, raw)
        else:
            case.code.write_bytes(b"changed during operation")
        return result

    monkeypatch.setattr(recovery, "assemble", changed)
    with pytest.raises(ValueError, match="changed|differs"):
        flow.run(case.operation, "finalize")


@pytest.mark.parametrize(
    "field,value",
    [
        ("version", True),
        ("unknown", 1),
        ("timeoutSeconds", 0),
        ("timeoutSeconds", 43201),
        ("controllers", []),
        ("generation", {"path": "/not-a-file", "sha256": "a" * 64, "unknown": True}),
        ("ffmpeg", {"path": "relative", "sha256": "a" * 64}),
        ("ffprobe", {"path": "/invalid", "sha256": "bad"}),
    ],
)
def test_bad_request_rejected_before_output(case: Case, field: str, value: object) -> None:
    rewrite_request(case, field, value)
    with pytest.raises((ValueError, StageError, ReplayError)):
        flow.prepare(case.request, case.output)
    assert not case.output.exists()


@pytest.mark.parametrize(
    "mutation",
    ["version", "kind", "workers", "duplicate", "lane", "coverage", "clock", "index", "settings"],
)
def test_bad_generation_selection_rejected(case: Case, mutation: str) -> None:
    data = common._read(case.plan)
    if mutation in {"version", "kind", "workers"}:
        data[mutation] = {"version": True, "kind": "not-recovery", "workers": 9}[mutation]
    elif mutation == "duplicate":
        data["states"] = data["workflows"]
    elif mutation == "lane":
        data["workflowLanes"] = [-1] * len(case.segments)
    elif mutation == "coverage":
        data["chunks"] = []
    else:
        rows = old._list(data["chunks"])
        row = record(rows[0])
        if mutation == "index":
            row["index"] = True
        else:
            selected = Path(str(row["chunk"]))
            contents = common._read(selected)
            contents[mutation] = {}
            write(selected, contents)
    write(case.plan, data)
    repin_plan(case)
    with pytest.raises(ValueError):
        flow.prepare(case.request, case.output)


@pytest.mark.parametrize(
    "mutation",
    ["command", "undeclared", "state", "verify", "inputs", "outputs", "extra-controller-field"],
)
def test_parent_identity_not_replaceable(case: Case, mutation: str) -> None:
    raw = common._read(case.request)
    controls = old._list(raw["controllers"])
    row = record(controls[0])
    if mutation == "extra-controller-field":
        row["extra"] = True
    else:
        graph = common._read(case.graphs[0])
        stage = record(old._list(graph["stages"])[0])
        command = old._strings(stage["command"])
        if mutation == "command":
            command[3] = "replay.unsupported"
        elif mutation == "undeclared":
            command[4] = str(case.request)
        elif mutation == "state":
            command[6] = str(case.states[0])
        elif mutation == "verify":
            stage["verify_command"] = ["false"]
        elif mutation == "inputs":
            stage["inputs"] = [str(case.graphs[0])]
        else:
            stage["outputs"] = [str(case.output / "unknown")]
        stage["command"] = command
        write(case.graphs[0], graph)
        row["workflow"] = common._pin(case.graphs[0])
    write(case.request, raw)
    with pytest.raises(ValueError):
        flow.prepare(case.request, case.output)


def test_incomplete_segment_coverage_refused(case: Case) -> None:
    raw = common._read(case.request)
    controls = old._list(raw["controllers"])
    graph = common._read(case.graphs[0])
    graph["stages"] = old._list(graph["stages"])[1:]
    write(case.graphs[0], graph)
    record(controls[0])["workflow"] = common._pin(case.graphs[0])
    write(case.request, raw)
    with pytest.raises((ValueError, StageError)):
        flow.prepare(case.request, case.output)


def test_child_preflight_failure_is_preserved(case: Case, monkeypatch: pytest.MonkeyPatch) -> None:
    finalize(case)
    monkeypatch.setattr(workflow, "main", lambda args: 1)
    with pytest.raises(ValueError, match="preflight"):
        flow.run(case.operation, "prepare-child")
    assert (case.output / "child/workflow.json").is_file()
    assert not (case.output / "prepared.json").exists()
    with pytest.raises(FileExistsError):
        flow.run(case.operation, "prepare-child")


@pytest.mark.parametrize("checkpoint", ["graph", "prepared"])
def test_prepared_child_tamper_not_adopted(case: Case, checkpoint: str) -> None:
    finalize(case)
    flow.run(case.operation, "prepare-child")
    path = case.output / ("child/workflow.json" if checkpoint == "graph" else "prepared.json")
    write(path, {"tamper": True})
    with pytest.raises(ValueError, match="differs|changed"):
        flow.run(case.operation, "verify-prepare-child")


def test_partial_media_never_restarts_child(case: Case) -> None:
    finalize(case)
    flow.run(case.operation, "prepare-child")
    partial = case.output / "clean-source.mov"
    partial.write_bytes(b"unapproved partial")
    with pytest.raises(FileNotFoundError):
        flow.run(case.operation, "run-child")
    assert partial.read_bytes() == b"unapproved partial"


@pytest.mark.parametrize("failure", ["capacity", "child"])
def test_capacity_and_child_failure_no_completion(
    case: Case, monkeypatch: pytest.MonkeyPatch, failure: str
) -> None:
    finalize(case)
    flow.run(case.operation, "prepare-child")
    monkeypatch.setattr(
        shutil, "disk_usage", lambda path: Usage(10**12, 0, 0 if failure == "capacity" else 10**12)
    )
    monkeypatch.setattr(workflow, "main", lambda args: 1)
    with pytest.raises(ValueError, match="headroom|preserve partial"):
        flow.run(case.operation, "run-child")
    assert not (case.output / "completion.json").exists()


def test_cli_exact_args_and_unknown_mode(case: Case) -> None:
    assert flow.main(["prepare", str(case.request), "--output", str(case.output)]) == 0
    assert flow.main(["prepare", str(case.request)]) == 1
    assert flow.main(["finalize", str(case.operation), "--output", str(case.output)]) == 1
    assert flow.main(["finalize", str(case.operation)]) == 0
    with pytest.raises(ValueError, match="Unknown"):
        flow.run(case.operation, "invalid")


def test_scanner_and_linked_input_rejection(
    tmp_path: Path, case: Case, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.undo()
    code = flow._code()
    assert str(Path(flow.__file__).resolve()) in code
    assert any(name.endswith("cleanup_generation_recovery.py") for name in code)
    assert any(name.endswith("delivery_source_assembly.py") for name in code)
    assert any(name.endswith("stages.py") for name in code)
    monkeypatch.chdir(case.request.parents[2])
    link = case.request.parent / "linked.json"
    link.symlink_to(case.request)
    with pytest.raises(ValueError, match="canonical"):
        flow.prepare(link, case.output)


def test_explicit_owner_observation_is_pinned_in_successor(case: Case) -> None:
    owner = case.output.parent / "fresh-owner.json"
    write(owner, {"currentOwnerEvidence": True})
    request = common._read(case.request)
    request["ownerObservation"] = common._pin(owner)
    write(case.request, request)
    graph = flow.prepare(case.request, case.output)
    job = flow._load(case.operation)
    assert flow._owner_observation(job) == owner
    assert all(owner in stage.inputs for stage in workflow.load_stages(graph))
    owner.write_bytes(b"changed")
    with pytest.raises(ValueError):
        flow._load(case.operation)


def test_finalization_propagates_retained_owner_to_every_chunk(
    case: Case, monkeypatch: pytest.MonkeyPatch
) -> None:
    owner = case.output.parent / "retained-owner.json"
    write(owner, {"currentOwnerEvidence": True})
    request = common._read(case.request)
    request["ownerObservation"] = common._pin(owner)
    write(case.request, request)
    verify_chunk = recovery.phase
    calls: list[int] = []

    def verify(
        plan: Path, index: int, *, verify: bool = False, owner_observation: Path | None = None
    ) -> None:
        assert verify and owner_observation == owner
        calls.append(index)
        verify_chunk(plan, index, verify=verify)

    monkeypatch.setattr(recovery, "phase", verify)
    finalize(case)
    chunks = old._list(common._read(case.plan)["chunks"])
    assert calls == list(range(len(chunks))) * 3
