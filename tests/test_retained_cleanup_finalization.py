"""Synthetic exact-clock finalization and consumed-input lease regression."""

from collections.abc import Iterator, Sequence
from contextlib import contextmanager
from pathlib import Path

import pytest
from test_cleanup_generation_recovery import (
    recovery_request as recovery_request,
)
from test_cleanup_generation_recovery import (
    render_fixture,
)
from test_recovery_source_finalization import Case, materialize
from test_recovery_source_finalization import case as case
from test_subtitle_cleanup_workflow import chunk as chunk
from test_subtitle_cleanup_workflow import settings as settings

from replay import cleanup_finalization_workflow as common
from replay import cleanup_generation_recovery as recovery
from replay import cleanup_lane_successor as old
from replay import recovery_source_finalization as existing
from replay import retained_cleanup_finalization as adapter
from replay import subtitle_candidate, workflow
from replay.stages import file_digest, run_stage


@pytest.fixture
def ready_case(case: Case, monkeypatch: pytest.MonkeyPatch) -> Case:
    monkeypatch.setattr(adapter, "_code", lambda: {str(case.code): file_digest(case.code)})
    monkeypatch.setattr(
        adapter,
        "verify_chunk",
        lambda job, data, index: recovery.phase(job.plan, index, verify=True),
    )
    return case


def test_full_finalization_child_and_reuse(
    ready_case: Case, monkeypatch: pytest.MonkeyPatch
) -> None:
    c = ready_case
    graph = adapter.prepare(c.request, c.output)
    assert workflow.main([str(graph), "--state", str(c.output / "parent-state"), "--check"]) == 0
    adapter.run(c.operation, "finalize")
    adapter.run(c.operation, "verify-finalize")
    adapter.run(c.operation, "prepare-child")
    adapter.run(c.operation, "verify-prepare-child")
    job = adapter._load(c.operation)
    calls: list[str] = []

    def execute(args: Sequence[str] | None = None) -> int:
        calls.append("assembly")
        stage = workflow.load_stages(job.root / "child/workflow.json")[0]

        def command(argv: tuple[str, ...], log: Path, timeout: int) -> None:
            if argv == stage.command:
                materialize(job)
            else:
                common._assembled(job)
            log.write_bytes(b"genuine synthetic execution")

        run_stage(stage, state_directory=job.root / "child-state", execute=command)
        return 0

    monkeypatch.setattr(workflow, "main", execute)
    adapter.run(c.operation, "run-child")
    adapter.run(c.operation, "verify-assembly")
    before = (job.root / "completion.json").read_bytes()
    adapter.run(c.operation, "verify-run-child")
    adapter.run(c.operation, "run-child")
    assert calls == ["assembly"]
    assert (job.root / "completion.json").read_bytes() == before
    assert common._read(job.root / "completion.json")["qualityApproved"] is False
    assert adapter.main(["verify-finalize", str(c.operation)]) == 0


def test_nonzero_and_changed_pins_refused(ready_case: Case) -> None:
    c = ready_case
    adapter.prepare(c.request, c.output)
    c.statuses[0].write_bytes(b"1\n")
    with pytest.raises(ValueError, match="terminal zero"):
        adapter.run(c.operation, "finalize")
    c.code.write_bytes(b"changed code")
    with pytest.raises(ValueError, match="operation changed"):
        adapter._load(c.operation)
    assert adapter.main(["prepare", str(c.request)]) == 1
    assert adapter.main(["finalize", str(c.operation), "--output", str(c.output)]) == 1


def test_selected_intersection_never_requires_unconsumed_receipt(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    present = tmp_path / "actual.json"
    absent = tmp_path / "never-produced-repair.json"
    present.write_bytes(b"actual recorded input")
    monkeypatch.setattr(old, "_selected_inputs", lambda data, index: {str(present), str(absent)})
    monkeypatch.setattr(old, "_path", lambda value, **kwargs: Path(str(value)))
    assert adapter.selected_recorded_inputs(
        {"inputs": {str(present): file_digest(present)}}, 25
    ) == (present,)
    with pytest.raises(FileNotFoundError):
        for path in adapter.selected_recorded_inputs({"inputs": {str(absent): "0" * 64}}, 25):
            file_digest(path)


def test_pending_original_domain_and_proof(
    recovery_request: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(subtitle_candidate, "render_candidate", render_fixture)
    plan = recovery.prepare(recovery_request, recovery_request.parent / "original")
    recovery.phase(plan, 0)

    @contextmanager
    def owned(job: common.Job, data: dict[str, object], index: int) -> Iterator[None]:
        yield

    monkeypatch.setattr(adapter, "_chunk_owned", owned)
    job = common.Job(plan, plan.parent, plan, (), {}, ({}, {}), 1)
    data = recovery._load(plan)
    adapter.verify_chunk(job, data, 0)
    (plan.parent / "verified-00000.json").write_bytes(b"{}")
    with pytest.raises(ValueError, match="recovery proof differs"):
        adapter.verify_chunk(job, data, 0)


def test_owned_consumed_inputs_and_kernel_leases(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from replay.sequence_journal import encode
    from replay.stages import execution_lock

    actual = tmp_path / "actual.json"
    actual.write_bytes(b"genuine recorded synthetic input")
    missing = tmp_path / "unconsumed-repair.json"
    prior_state, producing_state = tmp_path / "prior-state", tmp_path / "producer-state"
    legacy_path, handoff, plan = (
        tmp_path / "legacy.json",
        tmp_path / "handoff.json",
        tmp_path / "generation.json",
    )
    legacy = {
        "oldStates": [str(prior_state)],
        "inputs": {str(actual): file_digest(actual)},
        "chunks": [{"oldStages": [{"state": str(prior_state), "name": "old-repair"}]}],
    }
    legacy_path.write_bytes(encode(legacy))
    handoff.write_bytes(b"{}")
    plan.write_bytes(b"{}")
    data: dict[str, object] = {
        "successor": str(legacy_path),
        "oldStates": [],
        "handoff": str(handoff),
        "chunks": [
            {"lane": 0, "oldStageNames": ["actual-repair"], "oldState": str(producing_state)}
        ],
    }
    job = common.Job(plan, tmp_path, plan, (), {}, ({}, {}), 1)
    monkeypatch.setattr(old, "_path", lambda value, **kwargs: Path(str(value)))
    monkeypatch.setattr(old, "_selected_inputs", lambda *args: {str(actual), str(missing)})
    monkeypatch.setattr(recovery, "_retained_successor", lambda *args: legacy)
    monkeypatch.setattr(recovery, "_owner_guard", lambda *args, **kwargs: None)
    monkeypatch.setattr(existing, "_owner_observation", lambda job: None)
    with adapter._chunk_owned(job, data, 0):
        assert not missing.exists()
    with (
        pytest.raises(ValueError, match="Consumed successor input changed"),
        adapter._chunk_owned(job, data, 0),
    ):
        actual.write_bytes(b"changed")
    with (
        execution_lock(prior_state, "old-repair"),
        pytest.raises(ValueError, match="Original producer is active"),
        adapter._chunk_owned(job, data, 0),
    ):
        pass
    with (
        execution_lock(producing_state, "actual-repair"),
        pytest.raises(ValueError, match="Completed producer is active"),
        adapter._chunk_owned(job, data, 0),
    ):
        pass
    with (
        execution_lock(plan.parent, "recovery-lane-0"),
        pytest.raises(ValueError, match="Recovery lane already active"),
        adapter._chunk_owned(job, data, 0),
    ):
        pass


def test_retained_and_adopted_require_original_role_proof(
    recovery_request: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from replay.sequence_journal import encode
    from replay.stages import Stage

    monkeypatch.setattr(subtitle_candidate, "render_candidate", render_fixture)
    plan = recovery.prepare(recovery_request, recovery_request.parent / "role-proof")
    recovery.phase(plan, 0)
    data = recovery._load(plan)

    @contextmanager
    def owned(job: common.Job, data: dict[str, object], index: int) -> Iterator[None]:
        yield

    monkeypatch.setattr(adapter, "_chunk_owned", owned)
    row = old._object(old._list(data["chunks"])[0])
    row["mode"] = "retained-completed"
    job = common.Job(plan, plan.parent, plan, (), {}, ({}, {}), 1)
    with pytest.raises(FileNotFoundError):
        adapter.verify_chunk(job, data, 0)
    legacy = old._read(Path(str(data["successor"])), {})
    legacy["readerCode"] = {}
    Path(str(data["successor"])).write_bytes(encode(legacy))
    stage = Stage(
        name="old",
        command=("synthetic-producer",),
        inputs=(plan,),
        outputs=(plan.parent / "verified-00000.json",),
        timeout_seconds=1,
        verify_command=("synthetic-domain",),
    )
    monkeypatch.setattr(workflow, "load_stages", lambda path: [stage])
    monkeypatch.setattr(recovery, "retained_complete", lambda *args: None)
    row["repairEvidence"] = recovery._repair_evidence(Path(str(row["chunk"])))
    with pytest.raises(ValueError, match="Original recovery proof differs"):
        adapter.verify_chunk(job, data, 0)
    monkeypatch.setattr(workflow, "load_stages", lambda path: [])
    with pytest.raises(ValueError, match="Missing actual producer"):
        adapter.verify_chunk(job, data, 0)
    row["mode"] = "adopted-existing"
    proof = plan.parent / "adopted-proof.json"
    proof.write_bytes(b"{}")
    row["adoptedProof"] = old._pin(proof)
    with pytest.raises(ValueError, match="Adopted original proof differs"):
        adapter.verify_chunk(job, data, 0)
