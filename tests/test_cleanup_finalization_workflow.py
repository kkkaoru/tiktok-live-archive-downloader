"""Finite integration with genuine tiny Stage records and heterogeneous clocks."""

import fcntl
import json
import shutil
import sys
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import NamedTuple

import pytest

from replay import cleanup_finalization_workflow as flow
from replay import cleanup_lane_successor as successor
from replay import delivery_source_assembly as media
from replay import subtitle_cleanup_assembly as assembly
from replay import subtitle_cleanup_workflow as cleanup
from replay import workflow
from replay.core import ReplayError
from replay.recordings import record
from replay.stages import StageError, _fingerprint, file_digest, run_stage
from replay.subtitle_cleanup_plan import clock_chunks
from replay.workflow import load_stages


class Usage(NamedTuple):
    total: int
    used: int
    free: int


def write(path: Path, value: object) -> dict[str, str]:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value), encoding="utf-8")
    return flow._pin(path)


@dataclass(frozen=True)
class Case:
    request: Path
    output: Path
    plan: Path
    source: Path
    settings: Path
    status: Path
    state: Path
    code: Path

    @property
    def operation(self) -> Path:
        return self.output / "operation.json"


def chunk(directory: Path, settings: Path, clock: dict[str, object]) -> None:
    data = flow._read(settings)
    source, native = record(data["source"]), record(data["native"])
    write(
        directory / "chunk.json",
        {
            "version": 1,
            "settings": flow._pin(settings),
            "clock": clock,
            "ocrDirectory": str(directory / "ocr"),
        },
    )
    movie = directory / "repair/candidate.mp4"
    movie.parent.mkdir()
    movie.write_bytes(b"synthetic compressed packet identity")
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
            "clock": clock,
            "outputDirectory": str(movie.parent),
        },
    )
    frames = clock_chunks(clock, maximum_frames=20)[0]
    write(
        movie.with_suffix(".json"),
        {
            "inputs": {
                str(producer): file_digest(producer),
                str(source["path"]): source["sha256"],
                str(native["path"]): native["sha256"],
            },
            "render": {"frames": len(frames.pts)},
            "sourceStart": str(frames.pts[0] * frames.time_base),
            "videoSHA256": file_digest(movie),
        },
    )


@pytest.fixture(params=[(0, 3, 9, 14, 20), (0, 1, 2, 3, 4)])
def case(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, request: pytest.FixtureRequest) -> Case:
    monkeypatch.chdir(tmp_path)
    work = tmp_path / "edits/work"
    work.mkdir(parents=True)
    original = work / "original"
    original.mkdir()
    source, tool = original / "source.mp4", original / "tool"
    source.write_bytes(b"original synthetic source")
    tool.write_bytes(b"#!/bin/sh\nexit 0\n")
    tool.chmod(0o700)
    points = request.param
    if not isinstance(points, tuple):
        raise ValueError("Synthetic clock missing")
    clock = original / "clock.json"
    write(clock, {"pts": list(points), "timeBase": "1/100", "lastDuration": 2})
    settings = original / "settings.json"
    write(
        settings,
        {
            "version": 1,
            "source": flow._pin(source),
            "native": flow._pin(tool),
            "sourceClock": flow._pin(clock),
            "chunkFrames": 3,
        },
    )
    original_plan = original / "plan.json"
    write(
        original_plan,
        {
            "version": 1,
            "settings": flow._pin(settings),
            "wholeSourcePlanned": True,
            "chunks": 2,
            "plannedFrames": 5,
            "totalSourceFrames": 5,
        },
    )
    clocks = clock_chunks(flow._read(clock), maximum_frames=3)
    chunk(
        original / "chunk-00000",
        settings,
        {"pts": list(clocks[0].pts), "timeBase": "1/100", "lastDuration": clocks[0].last_duration},
    )
    chunk(
        original / "chunk-00001",
        settings,
        {"pts": list(clocks[1].pts), "timeBase": "1/100", "lastDuration": clocks[1].last_duration},
    )
    state = work / "state"
    state.mkdir()
    completed = work / "lane-output.json"
    graph = work / "lane.json"
    write(
        graph,
        {
            "version": 1,
            "stages": [
                {
                    "name": "real-local-producer",
                    "command": [
                        sys.executable,
                        "-c",
                        "from pathlib import Path; import sys; "
                        "Path(sys.argv[1]).write_text('complete')",
                        str(completed),
                    ],
                    "inputs": [str(source)],
                    "outputs": [str(completed)],
                    "timeout_seconds": 10,
                    "verify_command": [str(tool)],
                }
            ],
        },
    )
    run_stage(load_stages(graph)[0], state_directory=state)
    status = work / "lane.status"
    status.write_text("0\n", encoding="utf-8")
    plan = work / "successor.json"
    write(
        plan,
        {
            "version": 1,
            "settings": str(settings),
            "originalPlan": str(original_plan),
            "workers": 1,
            "chunks": [{"index": 0}, {"index": 1}],
            "workflows": [str(graph)],
            "states": [str(state)],
        },
    )
    code = work / "reader.py"
    code.write_text("immutable synthetic reader\n", encoding="utf-8")
    monkeypatch.setattr(assembly, "__file__", str(code))
    monkeypatch.setattr(assembly, "CODE_FILES", ("reader.py",))
    monkeypatch.setattr(flow, "_code", lambda: {str(code): file_digest(code)})
    monkeypatch.setattr(successor, "_load", lambda path: flow._read(path))
    monkeypatch.setattr(cleanup, "_settings", lambda path: flow._read(path))

    def assemble(plan_path: Path, output: Path) -> Path:
        data = flow._read(plan_path)
        manifest = assembly.prepare(Path(str(data["originalPlan"])), output)
        write(
            output / "successor-generation.json",
            {
                "version": 1,
                "successor": flow._pin(plan_path),
                "manifest": flow._pin(manifest),
                "maximumActiveChunks": data["workers"],
                "chunks": data["chunks"],
                "allPhysicalPTSAccountedByCanonicalAssembly": True,
                "qualityApproved": False,
                "mediaBytesVerified": False,
            },
        )
        return manifest

    monkeypatch.setattr(successor, "assemble", assemble)
    config = work / "request.json"
    write(
        config,
        {
            "version": 1,
            "successor": flow._pin(plan),
            "lanes": [{"workflow": flow._pin(graph), "state": str(state), "status": str(status)}],
            "ffmpeg": flow._pin(tool),
            "ffprobe": flow._pin(tool),
            "timeoutSeconds": 60,
        },
    )
    output = work / "finalization"
    output.mkdir()
    return Case(config, output, plan, source, settings, status, state, code)


def finalize(case: Case) -> flow.Job:
    flow.prepare(case.request, case.output)
    flow.run(case.operation, mode="finalize")
    flow.run(case.operation, mode="verify-finalize")
    return flow._load(case.operation)


def materialize(job: flow.Job) -> None:
    root = job.root
    chunks = media.load_chunks(
        root / "finalized/manifest.json",
        sha256=file_digest(root / "finalized/manifest.json"),
        bindings={},
    )
    bindings: dict[str, str] = {}
    media.load_chunks(
        root / "finalized/manifest.json",
        sha256=file_digest(root / "finalized/manifest.json"),
        bindings=bindings,
    )
    movie = root / "clean-source.mov"
    movie.write_bytes(b"synthetic assembled packets")
    evidence = root / "evidence"
    evidence.mkdir()
    probe = evidence / "probe.json"
    probe.write_text("synthetic original probe evidence", encoding="utf-8")
    first, last = chunks[0].clock, chunks[-1].clock
    index = evidence / "single-source-index.json"
    write(
        index,
        {
            "ranges": [
                {
                    "media": str(movie),
                    "sha256": file_digest(movie),
                    "sourceFirstPTS": first.pts[0],
                    "sourceEndPTS": last.pts[-1] + last.last_duration,
                    "frames": 5,
                    "timeBase": str(first.time_base),
                }
            ]
        },
    )
    write(
        evidence / "receipt.json",
        {
            "inputs": bindings,
            "output": {"path": str(movie), "sha256": file_digest(movie)},
            "indexSHA256": file_digest(index),
            "probeEvidence": {str(probe): file_digest(probe)},
            "frames": 5,
            "exactPresentationClockVerified": True,
            "compressedPacketsIdentical": True,
        },
    )
    state = root / "child-state"
    state.mkdir()
    stage = load_stages(root / "child/workflow.json")[0]
    write(
        state / "assemble-clean-source.json",
        {
            "status": "complete",
            "fingerprint": _fingerprint(stage),
            "outputs": {str(path): file_digest(path) for path in stage.outputs},
        },
    )


def test_metadata_prepare_and_exact_standard_graph(case: Case) -> None:
    graph = flow.prepare(case.request, case.output)
    assert workflow.main([str(graph), "--state", str(case.output / "state"), "--check"]) == 0
    assert tuple(stage.name for stage in load_stages(graph)) == (
        "finalize",
        "prepare-child",
        "run-child",
    )
    assert not (case.output / "clean-source.mov").exists()
    assert not (case.output / "finalized/manifest.json").exists()


def test_serial_connection_and_readonly_reverification(
    case: Case, monkeypatch: pytest.MonkeyPatch
) -> None:
    job = finalize(case)
    flow.run(case.operation, mode="prepare-child")
    flow.run(case.operation, mode="verify-prepare-child")
    monkeypatch.setattr(shutil, "disk_usage", lambda path: Usage(10**12, 0, 10**12))

    def run(arguments: Sequence[str] | None = None) -> int:
        assert arguments is not None
        assert arguments[1] == "--state"
        materialize(job)
        return 0

    monkeypatch.setattr(workflow, "main", run)
    flow.run(case.operation, mode="run-child")
    before = (case.output / "completion.json").read_bytes()
    flow.run(case.operation, mode="verify-run-child")
    flow.run(case.operation, mode="verify-assembly")
    flow.run(case.operation, mode="run-child")
    assert (case.output / "completion.json").read_bytes() == before
    result = flow._read(case.output / "completion.json")
    assert result["qualityApproved"] is False
    assert result["avSyncApproved"] is False
    assert flow._read(case.output / "finalized/preparation.json")["physicalSourceFrames"] == 5


@pytest.mark.parametrize("mode", ["finalize", "verify-finalize", "prepare-child", "run-child"])
def test_active_or_failed_lane_never_starts_finalization(case: Case, mode: str) -> None:
    flow.prepare(case.request, case.output)
    case.status.write_text("1\n", encoding="utf-8")
    with pytest.raises(ValueError, match="terminal zero"):
        flow.run(case.operation, mode=mode)
    assert not (case.output / "finalized/index.json").exists()


def test_live_exclusive_lane_lock_blocks_even_zero_status(case: Case) -> None:
    flow.prepare(case.request, case.output)
    with (case.state / ".workflow.lock").open("a") as stream:
        fcntl.flock(stream, fcntl.LOCK_EX | fcntl.LOCK_NB)
        with pytest.raises(ValueError, match="active"):
            flow.run(case.operation, mode="finalize")


def test_changed_original_stage_output_rejected(case: Case) -> None:
    flow.prepare(case.request, case.output)
    (case.request.parent / "lane-output.json").write_text("tampered", encoding="utf-8")
    with pytest.raises(ValueError, match="complete Stage"):
        flow.run(case.operation, mode="finalize")


def test_changed_source_is_not_adopted(case: Case) -> None:
    flow.prepare(case.request, case.output)
    case.source.write_bytes(b"new source")
    with pytest.raises(ValueError, match="complete Stage"):
        flow.run(case.operation, mode="finalize")


def test_reader_change_rejected_before_effects(case: Case) -> None:
    flow.prepare(case.request, case.output)
    case.code.write_text("different reader", encoding="utf-8")
    with pytest.raises(ValueError, match="reader code"):
        flow.run(case.operation, mode="finalize")


def test_partial_finalization_preserved_without_rerun(case: Case) -> None:
    flow.prepare(case.request, case.output)
    partial = case.output / "finalized/index.json"
    partial.write_text("original partial", encoding="utf-8")
    with pytest.raises(ValueError, match="absent"):
        flow.run(case.operation, mode="finalize")
    assert partial.read_text(encoding="utf-8") == "original partial"


@pytest.mark.parametrize(
    "filename", ["index.json", "manifest.json", "preparation.json", "successor-generation.json"]
)
def test_finalized_metadata_tamper_rejected(case: Case, filename: str) -> None:
    finalize(case)
    write(case.output / "finalized" / filename, {"tamper": True})
    with pytest.raises(ValueError, match="differs"):
        flow.run(case.operation, mode="verify-finalize")


def test_child_graph_and_completion_are_pinned(case: Case) -> None:
    finalize(case)
    flow.run(case.operation, mode="prepare-child")
    write(case.output / "child/workflow.json", {"version": 1, "stages": []})
    with pytest.raises(ValueError, match="graph differs"):
        flow.run(case.operation, mode="verify-prepare-child")


def test_headroom_denial_has_no_child_effect(case: Case, monkeypatch: pytest.MonkeyPatch) -> None:
    finalize(case)
    flow.run(case.operation, mode="prepare-child")
    monkeypatch.setattr(shutil, "disk_usage", lambda path: Usage(10**12, 10**12, 0))
    with pytest.raises(ValueError, match="headroom"):
        flow.run(case.operation, mode="run-child")
    assert not (case.output / "evidence").exists()


def test_outcome_changes_inside_operation_rejected(
    case: Case, monkeypatch: pytest.MonkeyPatch
) -> None:
    flow.prepare(case.request, case.output)
    original = successor.assemble

    def changed(plan: Path, output: Path) -> Path:
        result = original(plan, output)
        case.status.write_text("1", encoding="utf-8")
        return result

    monkeypatch.setattr(successor, "assemble", changed)
    with pytest.raises(ValueError, match="outcome changed"):
        flow.run(case.operation, mode="finalize")


@pytest.mark.parametrize(
    "field,value",
    [
        ("version", True),
        ("timeoutSeconds", 0),
        ("timeoutSeconds", 43201),
        ("lanes", []),
        ("unknown", 1),
    ],
)
def test_malformed_request_rejected(case: Case, field: str, value: object) -> None:
    data = flow._read(case.request)
    data[field] = value
    write(case.request, data)
    with pytest.raises(ValueError):
        flow.prepare(case.request, case.output)


def test_unknown_reference_and_mismatched_lane_rejected(case: Case) -> None:
    data = flow._read(case.request)
    data["lanes"] = [
        {"workflow": flow._pin(case.plan), "state": str(case.state), "status": str(case.status)}
    ]
    write(case.request, data)
    with pytest.raises(ValueError, match="bindings differ"):
        flow.prepare(case.request, case.output)


def test_linked_paths_rejected(case: Case) -> None:
    linked = case.request.parent / "linked.json"
    linked.symlink_to(case.request)
    with pytest.raises(ValueError, match="canonical"):
        flow.prepare(linked, case.output)


def test_cli_preflight_and_argument_failures(case: Case) -> None:
    assert flow.main(["prepare", str(case.request), "--output", str(case.output)]) == 0
    assert flow.main(["prepare", str(case.request)]) == 1
    assert flow.main(["finalize", str(case.operation), "--output", str(case.output)]) == 1
    assert flow.main(["prepare", str(case.request), "--output", str(case.output)]) == 1


def test_unknown_mode_rejected_without_mutation(case: Case) -> None:
    flow.prepare(case.request, case.output)
    with pytest.raises(ValueError, match="Unknown"):
        flow.run(case.operation, mode="unsupported")


def test_graph_parser_rejects_unbounded_code_inputs(
    case: Case, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(
        flow, "_code", lambda: {str(case.code) + str(index): "a" * 64 for index in range(260)}
    )
    with pytest.raises(StageError, match="bounded"):
        flow.prepare(case.request, case.output)


def test_partial_child_is_preserved_and_never_dispatched_again(case: Case) -> None:
    finalize(case)
    flow.run(case.operation, mode="prepare-child")
    partial = case.output / "clean-source.mov"
    partial.write_bytes(b"unapproved partial")
    with pytest.raises(FileNotFoundError):
        flow.run(case.operation, mode="run-child")
    assert partial.read_bytes() == b"unapproved partial"
    assert not (case.output / "completion.json").exists()


def test_original_source_guard_is_independent_of_stage_receipt(case: Case) -> None:
    flow.prepare(case.request, case.output)
    case.source.write_bytes(b"different original")
    graph = case.request.parent / "lane.json"
    stage = load_stages(graph)[0]
    receipt = flow._read(case.state / "real-local-producer.json")
    receipt["fingerprint"] = _fingerprint(stage)
    write(case.state / "real-local-producer.json", receipt)
    with pytest.raises(ValueError, match="source bytes changed"):
        flow.run(case.operation, mode="finalize")


def test_source_mutation_before_terminal_refuses_success(
    case: Case, monkeypatch: pytest.MonkeyPatch
) -> None:
    flow.prepare(case.request, case.output)
    original = successor.assemble

    def changed(plan: Path, output: Path) -> Path:
        result = original(plan, output)
        case.source.write_bytes(b"changed during operation")
        return result

    monkeypatch.setattr(successor, "assemble", changed)
    with pytest.raises(ValueError, match="changed during"):
        flow.run(case.operation, mode="finalize")


def test_incomplete_normal_child_has_no_completion_claim(
    case: Case, monkeypatch: pytest.MonkeyPatch
) -> None:
    finalize(case)
    flow.run(case.operation, mode="prepare-child")
    monkeypatch.setattr(shutil, "disk_usage", lambda path: Usage(10**12, 0, 10**12))
    monkeypatch.setattr(workflow, "main", lambda args: 1)
    with pytest.raises(ValueError, match="did not complete"):
        flow.run(case.operation, mode="run-child")
    assert not (case.output / "completion.json").exists()


@pytest.mark.parametrize(
    "field,value",
    [
        ("ffmpeg", {"path": "relative", "sha256": "a" * 64}),
        ("ffprobe", {"path": "/bad", "sha256": "invalid"}),
        ("successor", {"path": "/bad", "sha256": "a" * 64, "extra": 1}),
    ],
)
def test_malformed_references_are_not_repaired(case: Case, field: str, value: object) -> None:
    data = flow._read(case.request)
    data[field] = value
    write(case.request, data)
    with pytest.raises((ValueError, ReplayError)):
        flow.prepare(case.request, case.output)
