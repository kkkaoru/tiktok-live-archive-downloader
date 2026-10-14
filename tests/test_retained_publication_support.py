"""Exact retained child definitions and genuine readonly adoption."""

import json
import os
from pathlib import Path

import pytest
from test_recovery_source_finalization import Case, materialize
from test_recovery_source_finalization import case as case
from test_retained_cleanup_finalization import ready_case as ready_case

from replay import retained_cleanup_finalization as retained
from replay import retained_publication_support as support
from replay import workflow
from replay.stages import _fingerprint, run_stage


@pytest.fixture
def published_case(ready_case: Case) -> Case:
    c = ready_case
    retained.prepare(c.request, c.output)
    retained.run(c.operation, "finalize")
    retained.run(c.operation, "prepare-child")
    job = retained._load(c.operation)
    materialize(job)
    state = c.output / "child-state"
    state.mkdir()
    stage = workflow.load_stages(c.output / "child/workflow.json")[0]
    (state / "assemble-clean-source.json").write_text(
        json.dumps({"status": "running", "fingerprint": _fingerprint(stage)}), encoding="utf-8"
    )
    return c


def test_pending_context_preserves_stale_checkpoint(published_case: Case) -> None:
    c = published_case
    checkpoint = c.output / "child-state/assemble-clean-source.json"
    before = checkpoint.read_bytes()
    alias = c.output / "pending.mov"
    os.link(c.output / "clean-source.mov", alias)
    with support.pending_child_ready(c.operation, checkpoint) as job:
        assert support.pending_job(c.operation, checkpoint) == job
        assert support.child_document(job) == json.loads(
            (c.output / "child/workflow.json").read_text()
        )
        alias.unlink()
    assert checkpoint.read_bytes() == before
    assert not (c.output / "completion.json").exists()


@pytest.mark.parametrize("target", ["graph", "prepared", "fingerprint", "finalized"])
def test_pending_provenance_tamper(published_case: Case, target: str) -> None:
    c = published_case
    path = {
        "graph": c.output / "child/workflow.json",
        "prepared": c.output / "prepared.json",
        "fingerprint": c.output / "child-state/assemble-clean-source.json",
        "finalized": c.output / "finalized.json",
    }[target]
    path.write_text("{}", encoding="utf-8")
    with (
        pytest.raises(ValueError, match="changed"),
        support.pending_child_ready(
            c.operation, c.output / "child-state/assemble-clean-source.json"
        ),
    ):
        pytest.fail("tampered publication was accepted")


def test_other_child_rejected(published_case: Case) -> None:
    job = retained._load(published_case.operation)
    with pytest.raises(ValueError, match="another child"):
        support.check_pending_child(job, published_case.output / "unrelated.json")


def test_genuine_adopted_receipt_and_tampered_graph(published_case: Case) -> None:
    c = published_case
    job = retained._load(c.operation)
    graph = c.output / "adopted.json"
    graph.write_text(json.dumps(support.adopted_document(job, ())), encoding="utf-8")
    stage = workflow.load_stages(graph)[0]
    state = c.output / "fresh-child-state"
    before = (c.output / "child-state/assemble-clean-source.json").read_bytes()

    def verify(command: tuple[str, ...], log: Path, timeout: int) -> None:
        assert command == stage.verify_command
        retained.run(c.operation, "verify-assembly")
        log.write_text("genuine synthetic readonly domain verification", encoding="utf-8")

    assert run_stage(stage, state_directory=state, execute=verify) == "adopted"
    result = support.completed(job, graph, state, ())
    assert result["protocol"] == "retained-cleanup-finalization-v1"
    assert (c.output / "child-state/assemble-clean-source.json").read_bytes() == before
    graph.write_text("{}", encoding="utf-8")
    with pytest.raises(ValueError, match="definition changed"):
        support.completed(job, graph, state, ())


def test_prepare_and_cli(published_case: Case, monkeypatch: pytest.MonkeyPatch) -> None:
    c = published_case
    target = c.output / "future-adoption.json"
    monkeypatch.setattr(
        "sys.argv", ["support", "prepare-adoption", str(c.operation), "--output", str(target)]
    )
    support.main()
    stage = workflow.load_stages(target)[0]
    assert stage.adopt_existing is True
    assert stage.command == stage.verify_command
    assert "replay.code_inventory_memo_run" in stage.command
    with pytest.raises(FileExistsError):
        support.prepare_adoption(c.operation, target)
    with pytest.raises(ValueError, match="replace original"):
        support.prepare_adoption(c.operation, c.output / "child/workflow.json")
    with pytest.raises(ValueError, match="canonical checkout"):
        support.prepare_adoption(c.operation, Path("relative.json"))
