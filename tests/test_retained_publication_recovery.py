"""A fresh normal child proof seals retained media without rewriting old states."""

import json
from pathlib import Path

import pytest
from test_recovery_source_finalization import Case
from test_recovery_source_finalization import case as case
from test_retained_cleanup_finalization import ready_case as ready_case
from test_retained_publication_support import published_case as published_case

from replay import publication_completion_seal as seal
from replay import retained_cleanup_finalization as retained
from replay import retained_publication_support as support
from replay import verified_hardlink_publication as retirement
from replay import workflow
from replay.core import ReplayError
from replay.fcp_part_assembly import _reference
from replay.stages import file_digest, run_stage


@pytest.fixture
def retained_seal(published_case: Case, monkeypatch: pytest.MonkeyPatch) -> Path:
    c = published_case
    monkeypatch.setattr(seal, "ROOT", Path.cwd())
    job = retained._load(c.operation)
    repair = c.output / "publication-repair"
    repair.mkdir()
    graph = repair / "child-workflow.json"
    graph.write_text(
        json.dumps(support.adopted_document(job, seal.retained_child_inputs())), encoding="utf-8"
    )
    stage = workflow.load_stages(graph)[0]
    state = repair / "child-state"

    def verify(command: tuple[str, ...], log: Path, timeout: int) -> None:
        assert command == stage.verify_command
        retained.run(c.operation, "verify-assembly")
        log.write_text("normal synthetic readonly assembly verifier", encoding="utf-8")

    run_stage(stage, state_directory=state, execute=verify)
    retired = repair / "retired.json"
    retired.write_text("{}", encoding="utf-8")
    monkeypatch.setattr(retirement, "execute", lambda request, *, sha256, verify=False: retired)
    request = repair / "SEAL.json"
    request.write_text(
        json.dumps(
            {
                "version": 2,
                "kind": "publication-completion-seal",
                "operation": _reference(c.operation),
                "retirementRequest": _reference(retired),
                "completionOutput": str(c.output / "completion.json"),
                "proofOutput": str(repair / "sealed.json"),
                "leases": [str(c.output / "child-state/.workflow.lock")],
                "childWorkflow": _reference(graph),
                "childState": str(state),
            }
        ),
        encoding="utf-8",
    )
    return request


def test_retained_real_ready_and_fresh_child_receipt(retained_seal: Path) -> None:
    data = json.loads(retained_seal.read_text())
    old = Path(data["completionOutput"]).parent / "child-state/assemble-clean-source.json"
    before = old.read_bytes()
    result = seal.execute(retained_seal, sha256=file_digest(retained_seal))
    completion = json.loads(Path(data["completionOutput"]).read_text())
    assert completion["protocol"] == "retained-cleanup-finalization-v1"
    assert (
        completion["childStageReceipt"]["path"]
        == data["childState"] + "/assemble-clean-source.json"
    )
    assert old.read_bytes() == before
    assert seal.execute(retained_seal, sha256=file_digest(retained_seal), verify=True) == result


def test_retained_stale_original_state_cannot_be_adopted(retained_seal: Path) -> None:
    data = json.loads(retained_seal.read_text())
    data["childState"] = str(Path(data["completionOutput"]).parent / "child-state")
    retained_seal.write_text(json.dumps(data), encoding="utf-8")
    with pytest.raises(ReplayError, match="fresh child state"):
        seal.execute(retained_seal, sha256=file_digest(retained_seal))


def test_retained_new_child_tamper_fails(retained_seal: Path) -> None:
    data = json.loads(retained_seal.read_text())
    checkpoint = Path(data["childState"]) / "assemble-clean-source.json"
    checkpoint.write_text('{"status":"complete","fingerprint":"changed","outputs":{}}')
    with pytest.raises(ValueError, match="Genuine unchanged complete"):
        seal.execute(retained_seal, sha256=file_digest(retained_seal))
