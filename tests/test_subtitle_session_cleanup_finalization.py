"""Mixed producer lineage and ordinary assembly with small genuine Stage records."""

import fcntl
import json
import shutil
import sys
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import NamedTuple

import pytest

from replay import cleanup_finalization_workflow as common
from replay import cleanup_lane_successor as proof
from replay import code_generation_compat as compat
from replay import delivery_source_assembly as media
from replay import subtitle_cleanup_assembly as legacy
from replay import subtitle_cleanup_workflow as original_domains
from replay import subtitle_session_cleanup as consumer
from replay import subtitle_session_cleanup_finalization as flow
from replay import subtitle_session_cleanup_wave as wave
from replay import workflow
from replay.core import ReplayError
from replay.recordings import record
from replay.stages import Stage, file_digest, run_stage
from replay.subtitle_cleanup_plan import clock_chunks
from replay.subtitle_clock import VideoFrameClock
from replay.subtitle_session_observations import SessionPlan


class Usage(NamedTuple):
    total: int
    used: int
    free: int


@dataclass(frozen=True)
class Case:
    request: Path
    output: Path
    original: Path
    source: Path
    code: Path
    wave: Path
    wave_operation: Path
    work: Path
    graph: Path
    state: Path
    status: Path
    original_graph: Path
    original_state: Path

    @property
    def operation(self) -> Path:
        return self.output / "operation.json"


def write(path: Path, data: object) -> dict[str, str]:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data), encoding="utf-8")
    return common._pin(path)


def complete(stage: Stage, state: Path) -> None:
    def execute(command: tuple[str, ...], log: Path, timeout: int) -> None:
        if command == stage.command:
            for path in stage.outputs:
                path.parent.mkdir(parents=True, exist_ok=True)
            tuple(path.write_bytes(b"genuine synthetic producer") for path in stage.outputs)
        log.write_text("synthetic local execution", encoding="utf-8")

    run_stage(stage, state_directory=state, execute=execute)


def graph(path: Path, *, name: str, command: list[str], source: Path, output: Path) -> Path:
    write(
        path,
        {
            "version": 1,
            "stages": [
                {
                    "name": name,
                    "command": command,
                    "inputs": [str(source)],
                    "outputs": [str(output)],
                    "timeout_seconds": 60,
                    "verify_command": command,
                }
            ],
        },
    )
    return path


def chunk(directory: Path, settings: Path, clock: VideoFrameClock) -> None:
    data = common._read(settings)
    source, native = record(data["source"]), record(data["native"])
    points = {
        "pts": list(clock.pts),
        "timeBase": str(clock.time_base),
        "lastDuration": clock.last_duration,
    }
    write(
        directory / "chunk.json",
        {
            "version": 1,
            "settings": common._pin(settings),
            "clock": points,
            "ocrDirectory": str(directory / "ocr"),
        },
    )
    movie = directory / "repair/candidate.mp4"
    movie.parent.mkdir(parents=True)
    movie.write_bytes(b"synthetic exact-clock packets")
    producer = directory / "candidate-config.json"
    write(
        producer,
        {
            "source": source["path"],
            "sourceSHA256": source["sha256"],
            "nativeExecutable": native["path"],
            "nativeSHA256": native["sha256"],
            "width": 32,
            "height": 64,
            "clock": points,
            "outputDirectory": str(movie.parent),
        },
    )
    write(
        movie.with_suffix(".json"),
        {
            "inputs": {
                str(producer): file_digest(producer),
                str(source["path"]): source["sha256"],
                str(native["path"]): native["sha256"],
            },
            "render": {"frames": len(clock.pts)},
            "sourceStart": str(clock.pts[0] * clock.time_base),
            "videoSHA256": file_digest(movie),
        },
    )


@pytest.fixture(params=[(0, 1, 2, 3, 4), (0, 3, 9, 14, 20)])
def case(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, request: pytest.FixtureRequest) -> Case:
    monkeypatch.chdir(tmp_path)
    work = tmp_path / "edits/work"
    work.mkdir(parents=True)
    original = work / "original"
    original.mkdir()
    source, tool, code = work / "source", work / "tool", work / "reader.py"
    source.write_bytes(b"synthetic original source")
    tool.write_bytes(b"#!/bin/sh\nexit 0\n")
    tool.chmod(0o700)
    code.write_bytes(b"synthetic immutable reader")
    points = request.param
    if not isinstance(points, tuple):
        raise ValueError("Missing fixture clock")
    clock = original / "clock.json"
    write(clock, {"pts": list(points), "timeBase": "1/100", "lastDuration": 2})
    settings = original / "settings.json"
    write(
        settings,
        {
            "version": 1,
            "source": common._pin(source),
            "native": common._pin(tool),
            "decoder": common._pin(tool),
            "model": common._pin(tool),
            "sourceClock": common._pin(clock),
            "chunkFrames": 3,
        },
    )
    original_plan = original / "plan.json"
    write(
        original_plan,
        {
            "version": 1,
            "settings": common._pin(settings),
            "wholeSourcePlanned": True,
            "chunks": 2,
            "plannedFrames": 5,
            "totalSourceFrames": 5,
        },
    )
    clocks = clock_chunks(common._read(clock), maximum_frames=3)
    chunk(original / "chunk-00000", settings, clocks[0])
    chunk(original / "chunk-00001", settings, clocks[1])
    original_state = work / "original-state"
    original_graph = graph(
        work / "original-worker.json",
        name="chunk-00000-ocr",
        command=[str(tool)],
        source=source,
        output=original / "chunk-00000/ocr-proof.json",
    )
    complete(workflow.load_stages(original_graph)[0], original_state)
    selection = work / "selection"
    selection.mkdir()
    draft = selection / "draft.json"
    owner = selection / "owner/config-ready.json"
    write(draft, {"chunkIndex": 1})
    write(owner, {"chunkIndex": 1})
    operation = selection / "wave-operation.json"
    write(operation, {"draft": common._pin(draft), "workRoot": str(selection)})
    consumer_operation = selection / "consumer/operation.json"
    write(consumer_operation, {"config": common._pin(owner), "workRoot": str(selection)})
    write(
        selection / "consumer/repair/candidate.json", {"inputs": {str(source): file_digest(source)}}
    )
    (selection / "consumer/repair/candidate.mp4").write_bytes(b"current repair")
    child = graph(
        selection / "timed/workflow.json",
        name="repair",
        command=[str(tool)],
        source=source,
        output=selection / "consumer/child-proof.json",
    )
    complete(workflow.load_stages(child)[0], selection / "consumer/state")
    command = wave._command("run-child", operation)
    lane = graph(
        work / "lane.json",
        name="chunk-00001-wave-run-child",
        command=command,
        source=source,
        output=selection / "completion.json",
    )
    lane_data = common._read(lane)
    stages = lane_data["stages"]
    if not isinstance(stages, list):
        raise ValueError("Missing fixture stage")
    record(stages[0])["verify_command"] = wave._command("verify-run-child", operation)
    write(lane, lane_data)
    state = work / "lane-state"
    complete(workflow.load_stages(lane)[0], state)
    status = work / "lane.status"
    status.write_text("0\n", encoding="utf-8")
    wave_path = work / "wave.json"
    write(
        wave_path,
        {
            "version": 1,
            "protocol": wave.PROTOCOL,
            "lanes": [{"workflow": common._pin(lane), "state": str(state)}],
            "chunks": [
                {
                    "operation": common._pin(operation),
                    "lane": 0,
                    "chunkName": "chunk-00001",
                    "frames": 2,
                }
            ],
        },
    )

    def load(path: Path, *, require_observation: bool = True) -> consumer.Job:
        data = common._read(path)
        index = data.get("chunkIndex")
        if type(index) is not int:
            raise ValueError("Missing physical chunk index")
        provider = SessionPlan(
            source,
            tool,
            file_digest(source),
            file_digest(tool),
            tuple(point * clocks[index].time_base for point in clocks[index].pts),
            32,
            64,
        )
        return consumer.Job(
            path,
            original_plan,
            original / f"chunk-{index:05d}/chunk.json",
            common._read(settings),
            clocks[index],
            provider,
            draft,
            original_state,
            (),
            {},
            {str(source): file_digest(source)},
        )

    def session_row(
        path: Path,
        *,
        original: Path,
        clock: VideoFrameClock,
        pins: dict[str, str],
        allow_historical: bool = False,
    ) -> tuple[dict[str, object], dict[str, object], tuple[int, int]]:
        data = consumer._read(path, pins)
        consumer._reference(data["config"], pins)
        return legacy._chunk(
            original.parent / "chunk-00001",
            clock,
            settings,
            legacy._identity(common._read(settings)["source"]),
            legacy._identity(common._read(settings)["native"]),
            pins,
        )

    monkeypatch.setattr(flow, "_code", lambda: {str(code): file_digest(code)})
    monkeypatch.setattr(consumer, "_code", lambda: {str(code): file_digest(code)})
    monkeypatch.setattr(
        wave, "_operation", lambda path, *, allow_historical=False: (common._read(path), selection)
    )
    monkeypatch.setattr(wave, "_prepared", lambda data, root: None)
    monkeypatch.setattr(consumer, "load_config", load)
    monkeypatch.setattr(
        consumer,
        "_old_stages",
        lambda path, chunk, settings: tuple(workflow.load_stages(path)),
    )
    monkeypatch.setattr(consumer, "_admitted", lambda job, root: ({}, {}, {}, {}))
    monkeypatch.setattr(consumer, "_documents", lambda root, docs, *, verify: None)
    monkeypatch.setattr(consumer, "_session_assembly_row", session_row)
    monkeypatch.setattr(original_domains, "check_ocr", lambda path: None)
    monkeypatch.setattr(original_domains, "_settings", common._read)
    config = work / "request.json"
    write(
        config,
        {
            "version": 1,
            "originalPlan": common._pin(original_plan),
            "originalWorkers": [
                {"workflow": common._pin(original_graph), "state": str(original_state)}
            ],
            "waves": [common._pin(wave_path)],
            "controllers": [
                {"workflow": common._pin(lane), "state": str(state), "status": str(status)}
            ],
            "ffmpeg": common._pin(tool),
            "ffprobe": common._pin(tool),
            "timeoutSeconds": 60,
        },
    )
    output = work / "finalization"
    output.mkdir()
    return Case(
        config,
        output,
        original_plan,
        source,
        code,
        wave_path,
        operation,
        selection,
        lane,
        state,
        status,
        original_graph,
        original_state,
    )


def finalize(case: Case) -> flow.Job:
    flow.prepare(case.request, case.output)
    flow.run(case.operation, mode="finalize")
    flow.run(case.operation, mode="verify-finalize")
    return flow._load(case.operation)


def materialize(job: flow.Job) -> None:
    bindings: dict[str, str] = {}
    chunks = media.load_chunks(
        job.root / "finalized/manifest.json",
        sha256=file_digest(job.root / "finalized/manifest.json"),
        bindings=bindings,
    )
    movie = job.root / "clean-source.mov"
    movie.write_bytes(b"synthetic assembled packets")
    evidence = job.root / "evidence"
    evidence.mkdir()
    probe = evidence / "probe.json"
    probe.write_bytes(b"retained probe")
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
                    "frames": 5,
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
            "frames": 5,
            "exactPresentationClockVerified": True,
            "compressedPacketsIdentical": True,
        },
    )


def test_prepare_and_complete_readonly_reuse(case: Case, monkeypatch: pytest.MonkeyPatch) -> None:
    job = finalize(case)
    graph = case.output / "workflow.json"
    assert workflow.main([str(graph), "--state", str(case.output / "parent-state"), "--check"]) == 0
    flow.run(case.operation, mode="prepare-child")
    flow.run(case.operation, mode="verify-prepare-child")
    monkeypatch.setattr(shutil, "disk_usage", lambda path: Usage(10**12, 0, 10**12))

    def run(arguments: Sequence[str] | None = None) -> int:
        stage = workflow.load_stages(job.root / "child/workflow.json")[0]

        def execute(command: tuple[str, ...], log: Path, timeout: int) -> None:
            if command == stage.command:
                materialize(job)
            log.write_text("real tiny local producer", encoding="utf-8")

        run_stage(stage, state_directory=job.root / "child-state", execute=execute)
        return 0

    monkeypatch.setattr(workflow, "main", run)
    flow.run(case.operation, mode="run-child")
    before = (case.output / "completion.json").read_bytes()
    flow.run(case.operation, mode="verify-run-child")
    flow.run(case.operation, mode="verify-assembly")
    flow.run(case.operation, mode="run-child")
    assert (case.output / "completion.json").read_bytes() == before
    assert common._read(case.output / "completion.json")["qualityApproved"] is False
    assert common._read(case.output / "finalized/preparation.json")["physicalSourceFrames"] == 5


@pytest.mark.parametrize(
    "key,value",
    [
        ("version", True),
        ("timeoutSeconds", 0),
        ("timeoutSeconds", 43201),
        ("waves", []),
        ("originalWorkers", []),
        ("controllers", []),
        ("unknown", 1),
        ("ffprobe", {"path": "/outside", "sha256": "bad"}),
    ],
)
def test_invalid_requests(case: Case, key: str, value: object) -> None:
    data = common._read(case.request)
    data[key] = value
    write(case.request, data)
    with pytest.raises((ValueError, OSError)):
        flow.prepare(case.request, case.output)


@pytest.mark.parametrize(
    "key,value", [("plannedFrames", 4), ("chunks", 3), ("wholeSourcePlanned", False)]
)
def test_incomplete_source_plan(case: Case, key: str, value: object) -> None:
    data = common._read(case.original)
    data[key] = value
    write(case.original, data)
    request = common._read(case.request)
    request["originalPlan"] = common._pin(case.original)
    write(case.request, request)
    with pytest.raises(ValueError, match="full original"):
        flow.prepare(case.request, case.output)


@pytest.mark.parametrize("path", ["status", "source", "code"])
def test_changed_or_nonterminal_input(case: Case, path: str) -> None:
    flow.prepare(case.request, case.output)
    target = {"status": case.status, "source": case.source, "code": case.code}[path]
    target.write_bytes(b"changed")
    with pytest.raises((ValueError, OSError)):
        flow.run(case.operation, mode="finalize")
    assert not (case.output / "finalized.json").exists()


def test_active_owner_and_partial_consumer_refused(case: Case) -> None:
    flow.prepare(case.request, case.output)
    lock = case.state / ".workflow.lock"
    with lock.open("wb") as handle:
        fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
        with pytest.raises(ValueError, match="active"):
            flow.run(case.operation, mode="finalize")
    (case.work / "consumer/state/repair.json").unlink()
    with pytest.raises(FileNotFoundError):
        flow.run(case.operation, mode="finalize")
    assert not (case.output / "finalized.json").exists()


@pytest.mark.parametrize("name", ["index.json", "manifest.json", "preparation.json"])
def test_sealed_outputs_tamper(case: Case, name: str) -> None:
    finalize(case)
    path = case.output / "finalized" / name
    data = common._read(path)
    data["forged"] = True
    write(path, data)
    with pytest.raises(ValueError):
        flow.run(case.operation, mode="verify-finalize")


def test_exclusive_outputs_and_cli(case: Case) -> None:
    assert flow.main(["prepare", str(case.request)]) == 1
    assert flow.main(["prepare", str(case.request), "--output", str(case.output)]) == 0
    assert flow.main(["prepare", str(case.request), "--output", str(case.output)]) == 1
    assert flow.main(["finalize", str(case.operation), "--output", str(case.output)]) == 1
    assert flow.main(["finalize", str(case.operation)]) == 0
    assert flow.main(["verify-finalize", str(case.operation)]) == 0
    with pytest.raises(FileExistsError):
        flow.run(case.operation, mode="finalize")


def test_resource_failure_and_uncertain_output(case: Case, monkeypatch: pytest.MonkeyPatch) -> None:
    finalize(case)
    flow.run(case.operation, mode="prepare-child")
    monkeypatch.setattr(shutil, "disk_usage", lambda path: Usage(10**12, 0, 1))
    with pytest.raises(ValueError, match="headroom"):
        flow.run(case.operation, mode="run-child")
    (case.output / "clean-source.mov").write_bytes(b"unproven partial")
    with pytest.raises(FileNotFoundError):
        flow.run(case.operation, mode="run-child")
    assert not (case.output / "completion.json").exists()


def test_terminal_mutation(case: Case) -> None:
    flow.prepare(case.request, case.output)
    job = flow._load(case.operation)
    with pytest.raises((ValueError, ReplayError), match="changed"), flow._ready(job):
        case.source.write_bytes(b"mutated inside scope")


def test_noncanonical_metadata_refused(case: Case) -> None:
    link = case.request.parent / "linked.json"
    link.symlink_to(case.request)
    with pytest.raises(ValueError):
        flow.prepare(link, case.output)


def test_changed_dynamic_consumer_pin(case: Case) -> None:
    finalize(case)
    write(case.work / "consumer/operation.json", {"config": "forged"})
    with pytest.raises(ValueError):
        flow.run(case.operation, mode="verify-finalize")


def continuation(case: Case) -> Path:
    inner = workflow.load_stages(case.graph)[0]
    outer = case.request.parent / "continuation.json"
    state = case.request.parent / "continuation-state"
    write(
        outer,
        {
            "version": 1,
            "stages": [
                {
                    "name": "ordinary-child",
                    "command": [
                        sys.executable,
                        "-B",
                        "-m",
                        "replay.workflow",
                        str(case.graph),
                        "--state",
                        str(case.state),
                        "--target",
                        inner.name,
                    ],
                    "inputs": [str(case.graph), str(case.wave_operation)],
                    "outputs": [
                        *(str(path) for path in inner.outputs),
                        str(case.state / f"{inner.name}.json"),
                    ],
                    "verify_command": list(inner.verify_command),
                    "timeout_seconds": 60,
                    "adopt_existing": True,
                }
            ],
        },
    )

    def verify(command: tuple[str, ...], log: Path, timeout: int) -> None:
        proof._complete(inner, case.state)
        log.write_text("genuine readonly child verification", encoding="utf-8")

    run_stage(workflow.load_stages(outer)[0], state_directory=state, execute=verify)
    data = common._read(case.request)
    data["controllers"] = [
        {"workflow": common._pin(outer), "state": str(state), "status": str(case.status)}
    ]
    write(case.request, data)
    return outer


def test_genuine_hierarchical_controller(case: Case) -> None:
    continuation(case)
    job = finalize(case)
    assert len(job.plan.sessions) == 1
    assert common._read(case.output / "finalized/preparation.json")["sessionChunkIndices"] == [1]


@pytest.mark.parametrize("mutation", ["target", "outputs", "command", "verifier"])
def test_continuation_wrong_target_or_undeclared_proof(case: Case, mutation: str) -> None:
    outer = continuation(case)
    data = common._read(outer)
    stages = data["stages"]
    if not isinstance(stages, list):
        raise ValueError("Missing synthetic controller")
    stage = record(stages[0])
    if mutation == "target":
        command = stage["command"]
        if not isinstance(command, list):
            raise ValueError("Missing command")
        command[-1] = "unknown"
    elif mutation == "outputs":
        stage["outputs"] = [str(case.work / "completion.json")]
    elif mutation == "verifier":
        stage["verify_command"] = [str(case.code)]
    else:
        stage["command"] = [str(case.code)]
    write(outer, data)
    request = common._read(case.request)
    controllers = request["controllers"]
    if not isinstance(controllers, list):
        raise ValueError("Missing controller")
    record(controllers[0])["workflow"] = common._pin(outer)
    write(case.request, request)
    with pytest.raises(ValueError):
        flow.prepare(case.request, case.output)


@pytest.mark.parametrize("mutation", ["version", "frames", "lanes", "duplicate"])
def test_wave_membership_fail_closed(case: Case, mutation: str) -> None:
    data = common._read(case.wave)
    if mutation == "version":
        data["version"] = 2
    elif mutation == "lanes":
        data["lanes"] = [{"unknown": 1}]
    else:
        rows = data["chunks"]
        if not isinstance(rows, list):
            raise ValueError("Missing chunks")
        if mutation == "frames":
            record(rows[0])["frames"] = 9
        else:
            data["chunks"] = [rows[0], rows[0]]
    write(case.wave, data)
    request = common._read(case.request)
    request["waves"] = [common._pin(case.wave)]
    write(case.request, request)
    with pytest.raises(ValueError):
        flow.prepare(case.request, case.output)


def test_child_failure_preserves_checkpoint(case: Case, monkeypatch: pytest.MonkeyPatch) -> None:
    finalize(case)
    flow.run(case.operation, mode="prepare-child")
    monkeypatch.setattr(shutil, "disk_usage", lambda path: Usage(10**12, 0, 10**12))
    monkeypatch.setattr(workflow, "main", lambda arguments: 1)
    with pytest.raises(ValueError, match="stopped"):
        flow.run(case.operation, mode="run-child")
    assert (case.output / "prepared.json").is_file()
    assert not (case.output / "completion.json").exists()
    with pytest.raises(ValueError, match="Unknown"):
        flow.run(case.operation, mode="unknown")


def test_status_changes_inside_scope(case: Case) -> None:
    flow.prepare(case.request, case.output)
    with (
        pytest.raises(ValueError, match="outcome changed"),
        flow._ready(flow._load(case.operation)),
    ):
        case.status.write_text("1\n", encoding="utf-8")


def test_incomplete_original_evidence(case: Case) -> None:
    flow.prepare(case.request, case.output)
    data = common._read(case.original_graph)
    rows = data["stages"]
    if not isinstance(rows, list):
        raise ValueError("Missing original stage")
    record(rows[0])["name"] = "different-chunk"
    write(case.original_graph, data)
    request = common._read(case.request)
    request["originalWorkers"] = [
        {"workflow": common._pin(case.original_graph), "state": str(case.original_state)}
    ]
    write(case.request, request)
    with pytest.raises(ValueError):
        flow.run(case.operation, mode="finalize")


def test_genuine_receipt_mutation_during_phase(case: Case) -> None:
    flow.prepare(case.request, case.output)
    receipt = case.state / "chunk-00001-wave-run-child.json"
    with pytest.raises(ValueError, match="complete Stage"), flow._ready(flow._load(case.operation)):
        write(receipt, {"status": "running"})


def test_controller_output_mutation_during_phase(case: Case) -> None:
    flow.prepare(case.request, case.output)
    with pytest.raises(ValueError, match="complete Stage"), flow._ready(flow._load(case.operation)):
        (case.work / "completion.json").write_bytes(b"changed producer footer")


def test_historical_inputs_map_only_for_recovery_stages(tmp_path: Path) -> None:
    code = tmp_path / "subtitle_session_cleanup_wave.py"
    code.write_bytes(b"current wave code")
    data = tmp_path / "data.txt"
    data.write_bytes(b"data")
    plain = Stage(
        name="plain",
        command=("true",),
        inputs=(data,),
        outputs=(tmp_path / "plain-out.txt",),
        timeout_seconds=1,
        verify_command=("true",),
    )
    assert flow._historical_inputs(plain) == []
    out = tmp_path / "old-out.txt"
    out.write_bytes(b"result")
    old = Stage(
        name="old",
        command=("true",),
        inputs=(code, data),
        outputs=(out,),
        timeout_seconds=1,
        verify_command=("true",),
    )
    assert flow._historical_inputs(old) == [
        {
            str(code): compat.PRE_RECOVERY_DIGESTS["subtitle_session_cleanup_wave.py"],
            str(data): file_digest(data),
        },
        {
            str(code): compat.RECOVERY_WAVE_DIGESTS["subtitle_session_cleanup_wave.py"],
            str(data): file_digest(data),
        },
    ]
