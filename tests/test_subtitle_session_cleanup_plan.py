"""Metadata-only rolling drafts and ordinary Stage timing, without OCR/models."""

import json
import sys
from dataclasses import dataclass
from pathlib import Path

import pytest

from replay import subtitle_session_cleanup as consumer
from replay import subtitle_session_cleanup_plan as plan
from replay.sequence_journal import decode
from replay.stages import file_digest, run_stage
from replay.subtitle_cleanup_plan import clock_chunks
from replay.subtitle_session_observations import SessionPlan
from replay.workflow import load_stages


@dataclass(frozen=True)
class Case:
    config: Path
    output: Path
    original: Path
    worker: Path
    state: Path
    clock: Path
    handoff: Path


def document(path: Path, value: object) -> dict[str, str]:
    path.write_text(json.dumps(value), encoding="utf-8")
    return {"path": str(path), "sha256": file_digest(path)}


def read(path: Path) -> dict[str, object]:
    return decode(path.read_bytes())


@pytest.fixture
def case(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Case:
    work = tmp_path / "edits/work"
    work.mkdir(parents=True)
    monkeypatch.chdir(tmp_path)
    clock = work / "clock.json"
    clock_ref = document(
        clock,
        {"pts": [0, 7, 19, 26, 39, 46, 58, 70, 81, 93], "timeBase": "2/1001", "lastDuration": 8},
    )
    settings_ref = document(work / "settings.json", {"sourceClock": clock_ref, "chunkFrames": 4})
    original = work / "original"
    original.mkdir()
    original_ref = document(
        original / "plan.json", {"settings": settings_ref, "chunks": 3, "wholeSourcePlanned": True}
    )
    state = original / "state"
    state.mkdir()
    source, native = work / "source", work / "native"
    source.write_bytes(b"synthetic source")
    native.write_bytes(b"synthetic native")
    entries: list[dict[str, object]] = []
    for index in range(3):
        chunk = original / f"chunk-{index:05d}"
        chunk.mkdir()
        document(chunk / "chunk.json", {"version": 1, "index": index})
        for mode in ("ocr", "admission", "repair"):
            entries.append(
                {
                    "name": f"chunk-{index:05d}-{mode}",
                    "command": ["/usr/bin/true"],
                    "inputs": [str(chunk / "chunk.json")],
                    "outputs": [str(chunk / (mode + "-output"))],
                    "timeout_seconds": 10,
                }
            )
    worker = original / "worker.json"
    worker_ref = document(worker, {"version": 1, "stages": entries})
    handoff = work / "handoff.json"
    handoff_ref = document(
        handoff,
        {
            "kind": consumer.TERMINAL_KIND,
            "workflow": worker_ref,
            "state": str(state),
            "currentObservation": None,
            "observedAtNs": None,
            "oldWorkersDisabled": True,
            "historicalActorIdentityRecorded": False,
        },
    )
    provider_ref = document(
        work / "provider.json",
        {
            "version": 1,
            "protocol": "source-caption-ocr-session-v1",
            "source": str(source),
            "sourceSHA256": file_digest(source),
            "nativeExecutable": str(native),
            "nativeSHA256": file_digest(native),
            "width": 100,
            "height": 120,
            "workers": 2,
            "samples": [{"value": 0, "timescale": 1}],
        },
    )
    config = work / "config.json"
    document(
        config,
        {
            "version": 1,
            "originalPlan": original_ref,
            "providerTemplate": provider_ref,
            "handoffs": [handoff_ref],
            "chunkIndices": [1, 2],
        },
    )
    output = work / "drafts"
    output.mkdir()

    def domain(config_path: Path, *, require_observation: bool = True) -> consumer.Job:
        if require_observation:
            raise ValueError("Synthetic draft must not claim owner observation")
        data = read(config_path)
        index = data["chunkIndex"]
        if not isinstance(index, int):
            raise ValueError("Expected synthetic index")
        chunks = clock_chunks(read(clock), maximum_frames=4)
        selected = tuple(
            stage for stage in load_stages(worker) if stage.name.startswith(f"chunk-{index:05d}-")
        )
        provider = SessionPlan(
            source,
            native,
            file_digest(source),
            file_digest(native),
            tuple(point * chunks[index].time_base for point in chunks[index].pts),
            100,
            120,
            2,
        )
        return consumer.Job(
            config_path,
            original / "plan.json",
            original / f"chunk-{index:05d}/chunk.json",
            {},
            chunks[index],
            provider,
            work / "provider.json",
            state,
            selected,
            {},
            {},
        )

    monkeypatch.setattr(consumer, "load_config", domain)
    return Case(config, output, original, worker, state, clock, handoff)


def test_disjoint_drafts_preserve_variable_actual_pts_and_final_short_chunk(case: Case) -> None:
    result = plan.draft(case.config, case.output)
    data = read(result)
    assert data["mediaBytesVerified"] is False
    assert data["currentOwnerObservationRequired"] is True
    assert read(case.output / "chunk-00001/provider.json")["samples"] == [
        {"value": 6, "timescale": 77},
        {"value": 92, "timescale": 1001},
        {"value": 116, "timescale": 1001},
        {"value": 20, "timescale": 143},
    ]
    assert read(case.output / "chunk-00002/provider.json")["samples"] == [
        {"value": 162, "timescale": 1001},
        {"value": 186, "timescale": 1001},
    ]
    assert not (case.original / "chunk-00001/session-consumer-binding.json").exists()
    assert not (case.output / "chunk-00001/consumer").exists()
    assert (
        read(case.output / "chunk-00001/handoff-draft.json")["historicalActorIdentityRecorded"]
        is False
    )


@pytest.mark.parametrize("indices", [[1, 1], [-1], [3], [True], ["1"], [], [[1]]])
def test_invalid_indices_rejected(case: Case, indices: object) -> None:
    data = read(case.config)
    data["chunkIndices"] = indices
    document(case.config, data)
    with pytest.raises(ValueError):
        plan.draft(case.config, case.output)


@pytest.mark.parametrize("name", ["ocr", "admission.json", "repair", "scratch"])
def test_existing_partial_provider_rejected(case: Case, name: str) -> None:
    (case.original / "chunk-00001" / name).write_bytes(b"retained partial")
    with pytest.raises(ValueError, match="already reserved"):
        plan.draft(case.config, case.output)
    assert not (case.output / "draft-plan.json").exists()


def test_interrupted_reservation_allows_replanning_completion_refuses(case: Case) -> None:
    first = plan.draft(case.config, case.output)
    assert first.is_file()
    draft_config = case.output / "chunk-00001/config-draft.json"
    assert draft_config.is_file()
    old_root = case.original / "chunk-00001/interrupted-root"
    old_root.mkdir()
    interrupted = case.original / "chunk-00001/interrupted-operation.json"
    document(
        interrupted,
        {
            "version": 1,
            "protocol": consumer.PROTOCOL,
            "config": {"path": str(draft_config), "sha256": file_digest(draft_config)},
            "workRoot": str(old_root),
            "inputs": {},
            "readerCode": {},
        },
    )
    document(
        case.original / "chunk-00001/session-consumer-binding.json",
        {
            "version": 1,
            "operation": {"path": str(interrupted), "sha256": file_digest(interrupted)},
            "workRoot": str(old_root),
        },
    )
    second = case.output.with_name("drafts-recovered")
    second.mkdir()
    assert plan.draft(case.config, second) is not None
    (old_root / "completion.json").write_text("{}", encoding="utf-8")
    third = case.output.with_name("drafts-completed-refusal")
    third.mkdir()
    with pytest.raises(ValueError, match="supersede"):
        plan.draft(case.config, third)


def test_original_running_or_complete_receipt_not_reassigned(case: Case) -> None:
    document(case.state / "chunk-00001-ocr.json", {"status": "complete"})
    with pytest.raises(ValueError, match="already reserved"):
        plan.draft(case.config, case.output)


@pytest.mark.parametrize(
    "field,value",
    [
        ("observedAtNs", 123),
        ("currentObservation", {}),
        ("oldWorkersDisabled", False),
        ("historicalActorIdentityRecorded", True),
        ("kind", "actor-route"),
    ],
)
def test_no_stale_observation_or_fabricated_actor(case: Case, field: str, value: object) -> None:
    handoff = read(case.handoff)
    handoff[field] = value
    ref = document(case.handoff, handoff)
    config = read(case.config)
    config["handoffs"] = [ref]
    document(case.config, config)
    with pytest.raises(ValueError, match="unsealed terminal"):
        plan.draft(case.config, case.output)


def test_duplicate_worker_rejected(case: Case) -> None:
    config = read(case.config)
    config["handoffs"] = [plan._pin(case.handoff), plan._pin(case.handoff)]
    document(case.config, config)
    with pytest.raises(ValueError, match="unique"):
        plan.draft(case.config, case.output)


def test_wrong_membership_rejected(case: Case) -> None:
    worker = read(case.worker)
    worker["stages"] = [
        {
            "name": "other",
            "command": ["/usr/bin/true"],
            "inputs": [str(case.clock)],
            "outputs": [str(case.output / "other")],
            "timeout_seconds": 10,
        }
    ]
    worker_ref = document(case.worker, worker)
    handoff = read(case.handoff)
    handoff["workflow"] = worker_ref
    handoff_ref = document(case.handoff, handoff)
    config = read(case.config)
    config["handoffs"] = [handoff_ref]
    document(case.config, config)
    with pytest.raises(ValueError, match="exactly one"):
        plan.draft(case.config, case.output)


def test_changed_metadata_pin_rejected(case: Case) -> None:
    case.clock.write_bytes(b"{}")
    with pytest.raises(ValueError, match="metadata changed"):
        plan.draft(case.config, case.output)


def test_changed_metadata_at_terminal_rejected(case: Case, monkeypatch: pytest.MonkeyPatch) -> None:
    original_validate = plan.validate

    def change(config: Path) -> consumer.Job:
        job = original_validate(config)
        case.handoff.write_bytes(b"{}")
        return job

    monkeypatch.setattr(plan, "validate", change)
    with pytest.raises(ValueError, match="terminal capture"):
        plan.draft(case.config, case.output)


def test_output_must_be_new_and_canonical(case: Case) -> None:
    (case.output / "retained").write_bytes(b"keep")
    with pytest.raises(ValueError, match="empty"):
        plan.draft(case.config, case.output)
    assert (case.output / "retained").read_bytes() == b"keep"


@pytest.mark.parametrize("value", ["relative", "", "/tmp/no", 1, "invalid\npath"])
def test_paths_reject_outside_checkout(value: object) -> None:
    with pytest.raises(ValueError):
        plan._path(value)


def test_ref_unknown_fields_and_conflicting_pin(case: Case) -> None:
    with pytest.raises(ValueError, match="exact metadata"):
        plan._read({"path": str(case.clock), "sha256": "0" * 64, "extra": True}, {})
    with pytest.raises(ValueError, match="Conflicting"):
        plan._read(plan._pin(case.clock), {str(case.clock): "0" * 64})


@dataclass(frozen=True)
class Timed:
    config: Path
    original: Path
    operation: Path
    output: Path
    timing: Path
    state: Path


@pytest.fixture
def timed(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Timed:
    monkeypatch.chdir(tmp_path)
    work = tmp_path / "edits/work"
    work.mkdir(parents=True)
    consumer_root = work / "consumer"
    consumer_root.mkdir()
    operation = consumer_root / "operation.json"
    document(operation, {"protocol": consumer.PROTOCOL, "workRoot": str(consumer_root)})
    definitions = [
        {
            "name": "chunk-00009-session-" + mode,
            "command": [
                sys.executable,
                "-B",
                "-m",
                "replay.subtitle_session_cleanup",
                mode,
                str(operation),
            ],
            "inputs": [str(operation)],
            "outputs": [str(consumer_root / name) for name in plan.PHASE_OUTPUTS[mode]],
            "verify_command": [
                sys.executable,
                "-B",
                "-m",
                "replay.subtitle_session_cleanup",
                verify,
                str(operation),
            ],
            "timeout_seconds": 300,
        }
        for mode, verify in plan.MODES
    ]
    original = consumer_root / "workflow.json"
    original_ref = document(original, {"version": 1, "stages": definitions})
    timing, output, state = work / "timing", work / "timed", consumer_root / "state"
    timing.mkdir()
    output.mkdir()
    config = work / "timing-config.json"
    document(
        config,
        {
            "version": 1,
            "consumerWorkflow": original_ref,
            "consumerState": str(state),
            "timingRoot": str(timing),
        },
    )
    return Timed(config, original, operation, output, timing, state)


def test_timing_preserves_command_verifier_and_producer_outputs(timed: Timed) -> None:
    original = load_stages(timed.original)
    graph = plan.timed_workflow(timed.config, timed.output)
    current = load_stages(graph)
    assert current[0].command[2:4] == ("-m", "replay.phase_timer")
    assert current[0].command[7:] == original[0].command
    assert current[0].outputs[:3] == original[0].outputs
    assert current[0].verify_command == original[0].verify_command
    assert len(current[0].outputs) == 4
    assert str(timed.output / "timing-binding.json") in tuple(
        str(path) for path in current[0].inputs
    )
    assert not (timed.timing / "chunk-00009-session-collect.jsonl").exists()


def test_normal_stage_reuse_verifier_never_appends_timing(timed: Timed) -> None:
    graph = plan.timed_workflow(timed.config, timed.output)
    stage = load_stages(graph)[0]
    timed.state.mkdir()

    def execute(command: tuple[str, ...], log: Path, timeout: int) -> None:
        if command == stage.command:
            stage.outputs[0].parent.mkdir(parents=True, exist_ok=True)
            stage.outputs[0].write_bytes(b"synthetic inputs")
            stage.outputs[1].write_bytes(b"synthetic complete")
            stage.outputs[2].write_bytes(b"synthetic verification")
            stage.outputs[3].write_bytes(b'{"phase":"collect","exit":0,"seconds":1}\n')
        else:
            assert command == stage.verify_command

    assert run_stage(stage, state_directory=timed.state, execute=execute) == "completed"
    assert run_stage(stage, state_directory=timed.state, execute=execute) == "reused"
    assert stage.outputs[3].read_bytes() == b'{"phase":"collect","exit":0,"seconds":1}\n'


def test_old_state_cannot_be_rebound_to_timed_graph(timed: Timed) -> None:
    timed.state.mkdir()
    document(timed.state / "chunk-00009-session-collect.json", {"status": "running"})
    with pytest.raises(ValueError, match="before any ordinary"):
        plan.timed_workflow(timed.config, timed.output)


def test_partial_old_output_rejected(timed: Timed) -> None:
    (timed.operation.parent / "ocr-verification.json").write_bytes(b"partial")
    with pytest.raises(ValueError, match="fresh unchanged"):
        plan.timed_workflow(timed.config, timed.output)


def test_wrong_command_refused(timed: Timed) -> None:
    graph = read(timed.original)
    stages = graph["stages"]
    if not isinstance(stages, list) or not isinstance(stages[0], dict):
        raise ValueError("Synthetic stages missing")
    stages[0]["command"] = ["/usr/bin/true"]
    original_ref = document(timed.original, graph)
    config = read(timed.config)
    config["consumerWorkflow"] = original_ref
    document(timed.config, config)
    with pytest.raises(ValueError, match="fresh unchanged"):
        plan.timed_workflow(timed.config, timed.output)


def test_wrong_operation_domain_rejected(timed: Timed) -> None:
    document(timed.operation, {"protocol": "old-forged", "workRoot": str(timed.operation.parent)})
    with pytest.raises(ValueError, match="exact ordinary"):
        plan.timed_workflow(timed.config, timed.output)


def test_cli_validation_and_error_return(case: Case, capsys: pytest.CaptureFixture[str]) -> None:
    result = plan.draft(case.config, case.output)
    assert result.exists()
    assert plan.main(["validate", str(case.output / "chunk-00001/config-draft.json")]) == 0
    assert "owner not observed" in capsys.readouterr().out
    assert plan.main(["draft", str(case.config)]) == 1
    assert "requires an output" in capsys.readouterr().err


def test_cli_draft(case: Case) -> None:
    assert plan.main(["draft", str(case.config), "--output", str(case.output)]) == 0


def test_cli_timed(timed: Timed) -> None:
    assert plan.main(["timed-workflow", str(timed.config), "--output", str(timed.output)]) == 0
