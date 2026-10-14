"""Hierarchical ordinary workflows with synthetic providers and tiny local assets."""

import json
import subprocess
import sys
from dataclasses import dataclass
from fractions import Fraction
from pathlib import Path

import pytest

from replay import code_generation_compat as compat
from replay import subtitle_session_cleanup as consumer
from replay import subtitle_session_cleanup_plan as drafts
from replay import subtitle_session_cleanup_wave as flow
from replay import workflow
from replay.stages import file_digest, run_stage
from replay.subtitle_clock import VideoFrameClock
from replay.subtitle_session_observations import SessionPlan
from replay.workflow import load_stages


@dataclass(frozen=True)
class Case:
    config: Path
    output: Path
    source: Path
    draft_one: Path
    draft_two: Path


def document(path: Path, value: object) -> dict[str, str]:
    path.write_text(json.dumps(value), encoding="utf-8")
    return {"path": str(path), "sha256": file_digest(path)}


@pytest.fixture
def case(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Case:
    monkeypatch.chdir(tmp_path)
    work = tmp_path / "edits/work"
    work.mkdir(parents=True)
    original = work / "original"
    original.mkdir()
    original_ref = document(
        original / "plan.json", {"version": 1, "kind": "synthetic physical source"}
    )
    source, native = work / "source", work / "native"
    source.write_bytes(b"synthetic source")
    native.write_bytes(b"synthetic provider")
    source_ref = {"path": str(source), "sha256": file_digest(source)}
    native_ref = {"path": str(native), "sha256": file_digest(native)}
    state = original / "state"
    state.mkdir()
    (original / "chunk-00001").mkdir()
    (original / "chunk-00002").mkdir()
    document(original / "chunk-00001/chunk.json", {"index": 1})
    document(original / "chunk-00002/chunk.json", {"index": 2})
    first, second = work / "draft-one.json", work / "draft-two.json"
    first_ref = document(first, {"chunkIndex": 1, "originalPlan": original_ref})
    second_ref = document(second, {"chunkIndex": 2, "originalPlan": original_ref})
    config = work / "request.json"
    document(
        config,
        {
            "version": 1,
            "drafts": [first_ref, second_ref],
            "maximumWorkers": 2,
            "timeoutSeconds": 60,
        },
    )
    output = work / "wave"
    output.mkdir()

    def load(config_path: Path, *, require_observation: bool = True) -> consumer.Job:
        data = flow._read(config_path)
        index = data["chunkIndex"]
        if not isinstance(index, int):
            raise ValueError("Synthetic index missing")
        settings: dict[str, object] = {
            "source": source_ref,
            "native": native_ref,
            "decoder": native_ref,
            "model": native_ref,
        }
        pins = {
            str(config_path): file_digest(config_path),
            str(original / "plan.json"): file_digest(original / "plan.json"),
            str(source): file_digest(source),
            str(native): file_digest(native),
        }
        handoff: dict[str, object] = {}
        if require_observation:
            handoff = {
                "currentObservation": flow._pin(config_path.parent / "owner-observation.json"),
                "historicalActorIdentityRecorded": False,
            }
        clock = VideoFrameClock((index * 5, index * 5 + 2), Fraction(1, 100), 3)
        provider = SessionPlan(
            source,
            native,
            file_digest(source),
            file_digest(native),
            tuple(point * clock.time_base for point in clock.pts),
            100,
            120,
            2,
        )
        return consumer.Job(
            config_path,
            original / "plan.json",
            original / f"chunk-{index:05d}/chunk.json",
            settings,
            clock,
            provider,
            first,
            state,
            (),
            handoff,
            pins,
        )

    def validate(config_path: Path) -> consumer.Job:
        job = load(config_path, require_observation=False)
        marker = job.chunk.parent / "session-consumer-binding.json"
        if marker.exists():
            binding = json.loads(marker.read_text(encoding="utf-8"))
            if (Path(binding["workRoot"]) / "completion.json").exists():
                raise ValueError("Synthetic target reserved")
        return job

    def observe(config_path: Path, output: Path) -> Path:
        flow._empty(output)
        document(
            output / "owner-observation.json",
            {"kind": "synthetic fresh ownership", "historicalActorIdentityRecorded": False},
        )
        document(output / "handoff.json", {"historicalActorIdentityRecorded": False})
        document(output / "config-ready.json", flow._read(config_path))
        return output / "config-ready.json"

    def prepare(config_path: Path, output: Path) -> consumer.Prepared:
        flow._empty(output)
        job = load(config_path)
        operation = output / "operation.json"
        document(
            operation,
            {
                "protocol": consumer.PROTOCOL,
                "workRoot": str(output),
                "config": flow._pin(config_path),
                "inputs": job.pins,
            },
        )
        document(
            job.chunk.parent / "session-consumer-binding.json",
            {"version": 1, "operation": flow._pin(operation), "workRoot": str(output)},
        )
        entries = [
            {
                "name": job.chunk.parent.name + "-session-" + mode,
                "command": [
                    sys.executable,
                    "-B",
                    "-m",
                    "replay.subtitle_session_cleanup",
                    mode,
                    str(operation),
                ],
                "inputs": [str(operation), str(source), str(native)],
                "outputs": [str(output / name) for name in drafts.PHASE_OUTPUTS[mode]],
                "verify_command": [
                    sys.executable,
                    "-B",
                    "-m",
                    "replay.subtitle_session_cleanup",
                    verifier,
                    str(operation),
                ],
                "timeout_seconds": 60,
            }
            for mode, verifier in drafts.MODES
        ]
        document(output / "workflow.json", {"version": 1, "stages": entries})
        return consumer.Prepared(operation, output / "workflow.json", output / "state")

    def readonly(operation: Path, *, mode: str) -> None:
        if mode not in {"verify-ocr", "verify-admission", "verify-repair"}:
            raise ValueError("Synthetic readonly mode missing")

    def child(
        command: list[str], *, stdin: int, check: bool, timeout: int
    ) -> subprocess.CompletedProcess[bytes]:
        child_graph, child_state = Path(command[4]), Path(command[6])
        child_state.mkdir()
        stages = load_stages(child_graph)

        def execute(arguments: tuple[str, ...], log: Path, limit: int) -> None:
            if "replay.phase_timer" in arguments:
                stage = next(item for item in stages if item.command == arguments)
                for path in stage.outputs:
                    path.parent.mkdir(parents=True, exist_ok=True)
                    path.write_bytes(b"synthetic input-bound outcome")

        run_stage(stages[0], state_directory=child_state, execute=execute)
        run_stage(stages[1], state_directory=child_state, execute=execute)
        run_stage(stages[2], state_directory=child_state, execute=execute)
        return subprocess.CompletedProcess(command, 0)

    monkeypatch.setattr(consumer, "load_config", load)
    monkeypatch.setattr(drafts, "validate", validate)
    monkeypatch.setattr(consumer, "observe_terminal", observe)
    monkeypatch.setattr(consumer, "prepare", prepare)
    monkeypatch.setattr(consumer, "run", readonly)
    monkeypatch.setattr(subprocess, "run", child)
    return Case(config, output, source, first, second)


def operation(case: Case) -> Path:
    flow.prepare(case.config, case.output)
    return case.output / "selection-00000/wave-operation.json"


def test_historical_generation_verifies_but_never_executes(case: Case) -> None:
    original = operation(case)
    data = json.loads(original.read_text(encoding="utf-8"))
    reader = dict(data["readerCode"])
    for path in list(reader):
        name = Path(path).name
        if name in compat.RECOVERY_FILES:
            reader[path] = compat.PRE_RECOVERY_DIGESTS[name]
    data["readerCode"] = reader
    historical = original.with_name("wave-operation-historical.json")
    historical.write_text(json.dumps(data), encoding="utf-8")
    with pytest.raises(ValueError, match="Invalid or changed"):
        flow._operation(historical)
    bound, work = flow._operation(historical, allow_historical=True)
    assert work == historical.parent
    assert bound["chunkName"] == "chunk-00001"
    with pytest.raises(ValueError, match="Invalid or changed"):
        flow.run(historical, mode="observe")
    drifted = dict(reader)
    drifted["/repo/src/replay/unrelated.py"] = "e" * 64
    data["readerCode"] = drifted
    historical.write_text(json.dumps(data), encoding="utf-8")
    with pytest.raises(ValueError, match="Invalid or changed"):
        flow._operation(historical, allow_historical=True)


def test_explicit_lanes_are_data_only_disjoint_and_bounded(case: Case) -> None:
    result = flow.prepare(case.config, case.output)
    assert flow._read(result)["maximumSimultaneousLaneControllers"] == 2
    assert flow._read(result)["dispatched"] is False
    assert len(load_stages(case.output / "lane-00.json")) == 3
    assert len(load_stages(case.output / "lane-01.json")) == 3
    assert not (case.output / "selection-00000/owner/config-ready.json").exists()
    assert not (case.output / "selection-00000/consumer/operation.json").exists()
    assert not (case.output / "selection-00000/completion.json").exists()


def test_recovery_draft_pins_superseded_marker_and_completed_refuses(case: Case) -> None:
    old_root = case.source.parent / "original/chunk-00002/interrupted-root"
    old_root.mkdir(parents=True)
    operation = old_root / "interrupted-operation.json"
    document(
        operation,
        {
            "version": 1,
            "protocol": consumer.PROTOCOL,
            "config": flow._pin(case.draft_two),
            "workRoot": str(old_root),
            "inputs": {},
            "readerCode": {},
        },
    )
    document(
        case.source.parent / "original/chunk-00002/session-consumer-binding.json",
        {
            "version": 1,
            "operation": {"path": str(operation), "sha256": file_digest(operation)},
            "workRoot": str(old_root),
        },
    )
    result = flow.prepare(case.config, case.output)
    assert flow._read(result) is not None
    stages = {stage.name: stage for stage in load_stages(case.output / "lane-01.json")}
    marker = str(case.source.parent / "original/chunk-00002/session-consumer-binding.json")
    prepare = stages["chunk-00002-wave-prepare-child"]
    assert all(str(path) != marker for path in prepare.inputs)
    assert all(str(path) != marker for path in prepare.outputs)
    assert marker in {str(path) for path in stages["chunk-00002-wave-run-child"].inputs}
    (old_root / "completion.json").write_text("{}", encoding="utf-8")
    second = case.output.with_name("wave-completed-refusal")
    second.mkdir()
    with pytest.raises(ValueError, match="reserved"):
        flow.prepare(case.config, second)


def test_same_lane_next_chunk_depends_on_previous_completion(case: Case) -> None:
    data = flow._read(case.config)
    data["maximumWorkers"] = 1
    document(case.config, data)
    flow.prepare(case.config, case.output)
    stages = load_stages(case.output / "lane-00.json")
    assert len(stages) == 6
    assert case.output / "selection-00000/completion.json" in stages[3].inputs


@pytest.mark.parametrize(
    "field,value",
    [
        ("maximumWorkers", 0),
        ("maximumWorkers", 11),
        ("maximumWorkers", True),
        ("timeoutSeconds", 0),
        ("timeoutSeconds", 43201),
        ("drafts", []),
        ("drafts", None),
        ("version", 2),
        ("version", True),
    ],
)
def test_invalid_requests_rejected(case: Case, field: str, value: object) -> None:
    data = flow._read(case.config)
    data[field] = value
    document(case.config, data)
    with pytest.raises(ValueError):
        flow.prepare(case.config, case.output)


def test_duplicate_chunk_rejected(case: Case) -> None:
    data = flow._read(case.config)
    data["drafts"] = [flow._pin(case.draft_one), flow._pin(case.draft_one)]
    document(case.config, data)
    with pytest.raises(ValueError, match="unique"):
        flow.prepare(case.config, case.output)


def test_wrong_draft_pin_rejected(case: Case) -> None:
    case.draft_one.write_bytes(b"{}")
    with pytest.raises(ValueError, match="pin changed"):
        flow.prepare(case.config, case.output)


def test_full_normal_parent_child_chain_and_readonly_reuse(case: Case) -> None:
    flow.prepare(case.config, case.output)
    stages = load_stages(case.output / "lane-00.json")
    state = case.output / "state-00"
    state.mkdir()

    def execute(command: tuple[str, ...], log: Path, timeout: int) -> None:
        flow.run(Path(command[5]), mode=command[4])

    assert run_stage(stages[0], state_directory=state, execute=execute) == "completed"
    assert run_stage(stages[1], state_directory=state, execute=execute) == "completed"
    assert run_stage(stages[2], state_directory=state, execute=execute) == "completed"
    assert run_stage(stages[0], state_directory=state, execute=execute) == "reused"
    assert run_stage(stages[1], state_directory=state, execute=execute) == "reused"
    assert run_stage(stages[2], state_directory=state, execute=execute) == "reused"
    complete = flow._read(case.output / "selection-00000/completion.json")
    assert complete["qualityApproved"] is False
    assert complete["consumerConnected"] is False
    assert complete["fullDecodeVerified"] is False


def test_occupied_observation_retained_and_not_restarted(case: Case) -> None:
    op = operation(case)
    flow.run(op, mode="observe")
    with pytest.raises(ValueError, match="empty"):
        flow.run(op, mode="observe")
    assert (
        flow._read(case.output / "selection-00000/owner/handoff.json")[
            "historicalActorIdentityRecorded"
        ]
        is False
    )


def test_partial_preparation_refuses_reset(case: Case) -> None:
    op = operation(case)
    flow.run(op, mode="observe")
    (case.output / "selection-00000/consumer/partial.json").write_bytes(b"retained")
    with pytest.raises(ValueError, match="empty"):
        flow.run(op, mode="prepare-child")
    assert not (case.output / "selection-00000/prepared.json").exists()


def test_changed_prepared_checkpoint_refused(case: Case) -> None:
    op = operation(case)
    flow.run(op, mode="observe")
    flow.run(op, mode="prepare-child")
    (case.output / "selection-00000/timed/workflow.json").write_bytes(b"{}")
    with pytest.raises(ValueError, match="correspondence differs"):
        flow.run(op, mode="verify-prepare-child")


def test_child_failure_preserves_uncertain_outputs(
    case: Case, monkeypatch: pytest.MonkeyPatch
) -> None:
    op = operation(case)
    flow.run(op, mode="observe")
    flow.run(op, mode="prepare-child")

    def fail(
        command: list[str], *, stdin: int, check: bool, timeout: int
    ) -> subprocess.CompletedProcess[bytes]:
        raise subprocess.CalledProcessError(1, command)

    monkeypatch.setattr(subprocess, "run", fail)
    with pytest.raises(subprocess.CalledProcessError):
        flow.run(op, mode="run-child")
    assert (case.output / "selection-00000/prepared.json").exists()
    assert not (case.output / "selection-00000/completion.json").exists()


def test_wrong_operation_source_or_code_refused(case: Case) -> None:
    op = operation(case)
    data = flow._read(op)
    data["assetPins"] = {}
    document(op, data)
    with pytest.raises(ValueError, match="original domain"):
        flow.run(op, mode="observe")


def test_metadata_changed_before_observation(case: Case) -> None:
    op = operation(case)
    case.draft_one.write_bytes(b"{}")
    with pytest.raises(ValueError, match="pin changed"):
        flow.run(op, mode="observe")


def test_arbitrary_metadata_map_rejected_before_reading(case: Case) -> None:
    op = operation(case)
    data = flow._read(op)
    data["metadataPins"] = {"/private/never-read": "0" * 64}
    document(op, data)
    with pytest.raises(ValueError, match="original domain"):
        flow.run(op, mode="observe")


def test_parent_graph_byte_budget_required(case: Case, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(workflow, "MAX_PLAN_BYTES", 1)
    with pytest.raises(ValueError, match="reader byte budget"):
        flow.prepare(case.config, case.output)
    assert not (case.output / "wave.json").exists()


@pytest.mark.parametrize("value", ["relative", "/tmp/no", 1, "\n", ""])
def test_outside_work_path_rejected(value: object) -> None:
    with pytest.raises(ValueError):
        flow._path(value)


def test_unknown_reference_fields_rejected(case: Case) -> None:
    with pytest.raises(ValueError, match="exact wave"):
        flow._reference({"path": str(case.config), "sha256": "0" * 64, "extra": True})


def test_cli_prepare_and_output_errors(case: Case) -> None:
    assert flow.main(["prepare", str(case.config), "--output", str(case.output)]) == 0
    assert flow.main(["prepare", str(case.config)]) == 1
    assert flow.main(["observe", str(case.config), "--output", str(case.output)]) == 1
