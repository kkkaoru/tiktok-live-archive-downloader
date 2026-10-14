"""Historical stage receipts remain authentic after source bytes change."""

import sys
from dataclasses import dataclass, replace
from pathlib import Path

import pytest
from test_native_retired_hardlink_verification import Case, pin, write
from test_native_retired_hardlink_verification import case as case

from replay import cleanup_finalization_workflow as common
from replay import native_retired_hardlink_verification as aliases
from replay import publication_proof_completion_seal as proof
from replay import recovery_source_finalization as original
from replay import workflow
from replay.core import ReplayError
from replay.recordings import record
from replay.stages import Stage, _fingerprint, file_digest


@dataclass(frozen=True)
class HistoryCase:
    alias_case: Case
    operation: Path
    graph: Path
    child_graph: Path
    adoption_checkpoint: Path
    child_checkpoint: Path
    source: Path

    def load(self) -> proof.PublicationHistory:
        return proof.load_publication_history(
            operation=self.operation,
            custody_seal=self.alias_case.seal,
            publication_workflow=self.graph,
            adoption_checkpoint=self.adoption_checkpoint,
            child_workflow=self.child_graph,
            child_checkpoint=self.child_checkpoint,
        )


def stage_json(stage: Stage) -> dict[str, object]:
    return {
        "name": stage.name,
        "inputs": [str(path) for path in stage.inputs],
        "outputs": [str(path) for path in stage.outputs],
        "command": list(stage.command),
        "verify_command": list(stage.verify_command),
        "timeout_seconds": stage.timeout_seconds,
        "adopt_existing": stage.adopt_existing,
    }


@pytest.fixture
def history_case(case: Case, monkeypatch: pytest.MonkeyPatch) -> HistoryCase:
    c = case
    monkeypatch.setattr(proof, "ROOT", c.root)
    monkeypatch.setattr(aliases, "load_stages", workflow.load_stages)
    source_dir = c.root / "src/replay"
    source_dir.mkdir(parents=True)
    source = source_dir / "historical.py"
    source.write_bytes(b"def old():\n return 1\n")
    generation = write(c.root / "generation.json", {"version": 1})
    tool = c.root / "tool"
    tool.write_bytes(b"synthetic installed tool")
    request = write(
        c.root / "operation-request.json",
        {"generation": pin(generation), "ffmpeg": pin(tool), "ffprobe": pin(tool)},
    )
    operation = write(
        c.root / "operation.json",
        {
            "version": 1,
            "protocol": "retained-cleanup-finalization-v1",
            "request": pin(request),
            "workRoot": str(c.root),
            "readerCode": {str(source): file_digest(source)},
        },
    )
    receipt = write(c.root / "assembly-receipt.json", {"exactPresentationClockVerified": True})
    index = write(c.root / "assembly-index.json", {"frames": 7})
    child_state = c.root / "child-state"
    parent_state = c.root / "parent-state"
    child_state.mkdir()
    parent_state.mkdir()
    child_checkpoint = child_state / "assemble-clean-source.json"
    command = (
        sys.executable,
        "-B",
        "-m",
        "replay.code_inventory_memo_run",
        "run",
        "replay.retained_cleanup_finalization",
        "verify-assembly",
        str(operation),
    )
    child = Stage(
        name="assemble-clean-source",
        command=command,
        verify_command=command,
        inputs=(operation, request, generation, tool, source),
        outputs=(c.keeper, receipt, index),
        timeout_seconds=60,
        adopt_existing=True,
    )
    child_graph = write(c.root / "child.json", {"version": 1, "stages": [stage_json(child)]})
    write(
        child_checkpoint,
        {
            "status": "complete",
            "fingerprint": _fingerprint(child),
            "outputs": {
                str(c.keeper): file_digest(c.keeper),
                str(receipt): file_digest(receipt),
                str(index): file_digest(index),
            },
        },
    )
    seal_request = common._read(c.request)
    retirement_path = Path(str(record(seal_request["retirementRequest"])["path"]))
    retirement = common._read(retirement_path)
    retirement.update(
        {
            "publicationOperation": pin(operation),
            "assembly": {
                "assemblyReceipt": pin(receipt),
                "cleanIndex": pin(index),
            },
        }
    )
    write(retirement_path, retirement)
    retired = common._read(c.proof)
    retired["request"] = pin(retirement_path)
    disposition_path = Path(str(record(retired["disposition"])["path"]))
    disposition = common._read(disposition_path)
    disposition["request"] = pin(retirement_path)
    write(disposition_path, disposition)
    retired["disposition"] = pin(disposition_path)
    write(c.proof, retired)
    parent_command = (
        sys.executable,
        "-B",
        "-m",
        "replay.workflow",
        str(child_graph),
        "--state",
        str(child_state),
    )
    adoption = Stage(
        name="adopt-verified-existing-source",
        command=parent_command,
        verify_command=parent_command,
        inputs=(operation, request, source, child_graph, c.proof),
        outputs=(child_checkpoint,),
        timeout_seconds=60,
    )
    suffix = (str(retirement_path), "--sha256", file_digest(retirement_path))
    retirement_stage = Stage(
        name="retire-alias",
        command=(sys.executable, "-m", "replay.verified_hardlink_publication", "run", *suffix),
        verify_command=(
            sys.executable,
            "-m",
            "replay.verified_hardlink_publication",
            "verify",
            *suffix,
        ),
        inputs=(retirement_path, operation, request, source, child_graph, receipt, index),
        outputs=(c.proof, disposition_path, Path(str(record(retired["retirementLog"])["path"]))),
        timeout_seconds=60,
    )
    graph = write(
        c.root / "workflow.json",
        {"version": 1, "stages": [stage_json(retirement_stage), stage_json(adoption)]},
    )
    write(
        c.checkpoint,
        {
            "status": "complete",
            "fingerprint": _fingerprint(retirement_stage),
            "outputs": {str(path): file_digest(path) for path in retirement_stage.outputs},
        },
    )
    adoption_checkpoint = write(
        parent_state / "adopt-verified-existing-source.json",
        {
            "status": "complete",
            "fingerprint": _fingerprint(adoption),
            "outputs": {str(child_checkpoint): file_digest(child_checkpoint)},
        },
    )
    seal_request.update(
        {
            "workflow": pin(graph),
            "completedStage": pin(c.checkpoint),
            "retirementRequest": pin(retirement_path),
            "retirementProof": pin(c.proof),
        }
    )
    write(c.request, seal_request)
    aliases.seal(c.request, c.seal)

    def construct(path: Path, operation_path: Path, root: Path, code: dict[str, str]) -> common.Job:
        # Original constructor's full generation validation has separate genuine
        # synthetic tests; this seam keeps these tests about historical custody.
        return common.Job(
            operation_path,
            root,
            generation,
            (),
            code,
            (record(common._read(path)["ffmpeg"]), record(common._read(path)["ffprobe"])),
            60,
        )

    monkeypatch.setattr(original, "_request", construct)
    return HistoryCase(
        c, operation, graph, child_graph, adoption_checkpoint, child_checkpoint, source
    )


def test_history_uses_genuine_custody_with_changed_source(history_case: HistoryCase) -> None:
    before = file_digest(history_case.source)
    history_case.source.write_bytes(b"def old():\n return 2\n")
    result = history_case.load()
    assert result.job.code[str(history_case.source)] == before
    assert result.child_stage.adopt_existing is True
    assert result.alias.exists() is False
    assert result.keeper.stat().st_nlink == 1
    assert result.child_outputs[str(result.keeper)] == file_digest(result.keeper)


@pytest.mark.parametrize("checkpoint", ["child", "adoption"])
@pytest.mark.parametrize(
    "field,value", [("status", "running"), ("fingerprint", "0" * 64), ("outputs", {})]
)
def test_history_rejects_changed_completion(
    history_case: HistoryCase, checkpoint: str, field: str, value: object
) -> None:
    path = (
        history_case.child_checkpoint if checkpoint == "child" else history_case.adoption_checkpoint
    )
    data = common._read(path)
    data[field] = value
    write(path, data)
    with pytest.raises(ReplayError, match="Historical complete"):
        history_case.load()


def test_history_rejects_changed_operation(history_case: HistoryCase) -> None:
    history_case.operation.write_bytes(b"{}")
    with pytest.raises(ReplayError):
        history_case.load()


def test_history_rejects_alias_reappearance(history_case: HistoryCase) -> None:
    history_case.alias_case.alias.symlink_to(history_case.alias_case.root / "missing")
    with pytest.raises((ReplayError, ValueError), match="reappeared|canonical"):
        history_case.load()


def test_historical_complete_requires_exact_input_table(tmp_path: Path) -> None:
    path = tmp_path / "input"
    path.write_bytes(b"fixture")
    stage = Stage(
        name="s",
        command=("run",),
        verify_command=("verify",),
        inputs=(path,),
        outputs=(tmp_path / "output",),
        timeout_seconds=1,
    )
    with pytest.raises(ReplayError, match="input table"):
        proof._history_complete(
            stage,
            checkpoint={"status": "complete", "fingerprint": "x", "outputs": {}},
            pins={},
            outputs={str(tmp_path / "output"): "0" * 64},
        )


def test_history_merge_conflicting_pins_rejected(tmp_path: Path) -> None:
    with pytest.raises(ReplayError, match="Conflicting"):
        proof._history_merge({str(tmp_path): "0" * 64}, tmp_path, "1" * 64)


def test_history_reference_extra_field_rejected() -> None:
    with pytest.raises(ReplayError, match="exact historical reference"):
        proof._history_reference({"path": "/x", "sha256": "0" * 64, "extra": True})


def test_history_read_requires_previously_authenticated_metadata(tmp_path: Path) -> None:
    with pytest.raises(ReplayError, match="authenticated binding"):
        proof._history_read(tmp_path / "unbound.json", {})


def test_history_read_rejects_hash_bound_invalid_json(tmp_path: Path) -> None:
    path = tmp_path / "invalid.json"
    path.write_bytes(b"not JSON")
    with pytest.raises(ReplayError, match="metadata is invalid"):
        proof._history_read(path, {str(path): file_digest(path)})


def test_history_complete_requires_exact_outputs_and_postcommand(tmp_path: Path) -> None:
    stage = Stage(
        name="s",
        command=("run",),
        inputs=(tmp_path / "input",),
        outputs=(tmp_path / "output",),
        timeout_seconds=1,
    )
    with pytest.raises(ReplayError, match="post-command output"):
        proof._history_complete(stage, checkpoint={}, pins={}, outputs={})


def test_history_rejects_other_operation_path(history_case: HistoryCase) -> None:
    other = replace(history_case, operation=history_case.alias_case.root / "different.json")
    with pytest.raises(ReplayError, match="operation differs"):
        other.load()


def test_history_rejects_missing_custody_owner(history_case: HistoryCase) -> None:
    raw = common._read(history_case.alias_case.seal)
    raw["owner"] = None
    write(history_case.alias_case.seal, raw)
    with pytest.raises(ReplayError, match="owner is missing"):
        history_case.load()
