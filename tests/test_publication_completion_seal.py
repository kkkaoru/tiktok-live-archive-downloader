"""Publication repair seals genuine completed child evidence, never old success."""

import json
from contextlib import nullcontext
from pathlib import Path

import pytest
from test_subtitle_session_cleanup_finalization import Case, finalize, materialize
from test_subtitle_session_cleanup_finalization import case as case

from replay import publication_completion_seal as seal
from replay import subtitle_session_cleanup_finalization as finalization
from replay import verified_hardlink_publication as retirement
from replay import workflow
from replay.core import ReplayError
from replay.fcp_part_assembly import _reference
from replay.stages import file_digest, run_stage


@pytest.fixture
def seal_request(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    monkeypatch.setattr(seal, "ROOT", tmp_path)
    work = tmp_path / "edits/work/assembly"
    work.mkdir(parents=True)
    work.joinpath("finalized").mkdir()
    work.joinpath("evidence").mkdir()
    work.joinpath("child").mkdir()
    work.joinpath("child-state").mkdir()
    original_media = work / "candidate.mp4"
    original_media.write_bytes(b"genuine candidate fixture")
    candidate = work / "candidate.json"
    candidate.write_text("{}")
    producer = work / "producer.json"
    producer.write_text("{}")
    index = work / "index.json"
    index.write_text(json.dumps({"ranges": [{"media": str(original_media)}]}))
    work.joinpath("finalized/manifest.json").write_text(
        json.dumps(
            {
                "index": _reference(index),
                "chunks": [{"candidate": _reference(candidate), "producer": _reference(producer)}],
            }
        )
    )
    work.joinpath("evidence/receipt.json").write_text(
        json.dumps({"probeEvidence": {str(index): file_digest(index)}})
    )
    work.joinpath("evidence/single-source-index.json").write_text("{}")
    work.joinpath("clean-source.mov").write_bytes(b"genuine completed media")
    work.joinpath("prepared.json").write_text("{}")
    work.joinpath("finalized.json").write_text("{}")
    work.joinpath("child/workflow.json").write_text("{}")
    work.joinpath("child-state/assemble-clean-source.json").write_text('{"status":"complete"}')
    operation = work / "operation.json"
    operation.write_text("{}")
    retirement_request = work / "retirement-request.json"
    retirement_request.write_text("{}")
    retirement_proof = work / "retired.json"
    retirement_proof.write_text("{}")
    lease = work / ".workflow.lock"
    lease.write_bytes(b"")
    request = work / "seal-request.json"
    request.write_text(
        json.dumps(
            {
                "version": 1,
                "kind": "publication-completion-seal",
                "operation": _reference(operation),
                "retirementRequest": _reference(retirement_request),
                "completionOutput": str(work / "completion.json"),
                "proofOutput": str(work / "sealed.json"),
                "leases": [str(lease)],
            }
        )
    )
    plan = finalization.Plan(request, request, (), (), (), (), request, ({}, {}), 300, {})
    job = finalization.Job(operation, work, plan, {})
    expected: dict[str, object] = {
        "version": 1,
        "protocol": "session-cleanup-finalization-v1",
        "output": _reference(work / "clean-source.mov"),
        "qualityApproved": False,
    }
    monkeypatch.setattr(finalization, "_load", lambda operation: job)
    monkeypatch.setattr(finalization, "_ready", lambda job: nullcontext())
    monkeypatch.setattr(finalization, "_completed", lambda job: expected)

    def normal_verify(job: finalization.Job, mode: str) -> None:
        assert mode == "verify-run-child"
        assert json.loads((job.root / "completion.json").read_text()) == expected

    monkeypatch.setattr(finalization, "_continue", normal_verify)
    monkeypatch.setattr(
        retirement, "execute", lambda request, *, sha256, verify=False: retirement_proof
    )
    return request


def test_original_schema_sealed_and_verified(seal_request: Path) -> None:
    digest = file_digest(seal_request)
    proof = seal.execute(seal_request, sha256=digest)
    assert json.loads(proof.read_text())["mediaReassembled"] is False
    assert json.loads(proof.read_text())["originalOuterStatusRewritten"] is False
    assert seal.execute(seal_request, sha256=digest, verify=True) == proof
    with pytest.raises(ReplayError, match="already exists"):
        seal.execute(seal_request, sha256=digest)


@pytest.mark.parametrize("field,value", [("version", 2), ("leases", []), ("kind", "force-success")])
def test_invalid_seal_schema(seal_request: Path, field: str, value: object) -> None:
    data = json.loads(seal_request.read_text())
    data[field] = value
    seal_request.write_text(json.dumps(data))
    with pytest.raises(ReplayError):
        seal.execute(seal_request, sha256=file_digest(seal_request))


@pytest.mark.parametrize("name", ["completion.json", "sealed.json"])
def test_output_tamper(seal_request: Path, name: str) -> None:
    digest = file_digest(seal_request)
    seal.execute(seal_request, sha256=digest)
    seal_request.with_name(name).write_text("{}")
    with pytest.raises(ReplayError):
        seal.execute(seal_request, sha256=digest, verify=True)


def test_domain_check_failure(seal_request: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    def reject(job: finalization.Job) -> dict[str, object]:
        raise ReplayError("genuine child proof rejected")

    monkeypatch.setattr(finalization, "_completed", reject)
    with pytest.raises(ReplayError, match="child proof rejected"):
        seal.execute(seal_request, sha256=file_digest(seal_request))
    assert not seal_request.with_name("completion.json").exists()


def test_wrong_completion_namespace(seal_request: Path) -> None:
    data = json.loads(seal_request.read_text())
    data["completionOutput"] = str(seal_request.with_name("other.json"))
    seal_request.write_text(json.dumps(data))
    with pytest.raises(ReplayError, match="original completion namespace"):
        seal.execute(seal_request, sha256=file_digest(seal_request))


def test_path_overlap_and_scope(seal_request: Path) -> None:
    with pytest.raises(ReplayError, match="checkout-local"):
        seal._local(str(seal_request.parents[3] / "escape"))
    data = json.loads(seal_request.read_text())
    data["proofOutput"] = data["completionOutput"]
    seal_request.write_text(json.dumps(data))
    with pytest.raises(ReplayError, match="overlap"):
        seal.execute(seal_request, sha256=file_digest(seal_request))


def test_cli(seal_request: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        "sys.argv", ["seal", "run", str(seal_request), "--sha256", file_digest(seal_request)]
    )
    seal.main()


def test_real_ready_owns_hash_context_without_nesting(
    case: Case, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(seal, "ROOT", Path.cwd())
    job = finalize(case)
    finalization.run(case.operation, mode="prepare-child")
    stage = workflow.load_stages(job.root / "child/workflow.json")[0]

    def produce(command: tuple[str, ...], log: Path, timeout: int) -> None:
        if command == stage.command:
            materialize(job)
        log.write_text("synthetic exact-clock completed assembly", encoding="utf-8")

    run_stage(stage, state_directory=job.root / "child-state", execute=produce)
    retired = case.output / "retired.json"
    retired.write_text("{}", encoding="utf-8")
    request = case.output / "seal-real-ready.json"
    request.write_text(
        json.dumps(
            {
                "version": 1,
                "kind": "publication-completion-seal",
                "operation": _reference(case.operation),
                "retirementRequest": _reference(retired),
                "completionOutput": str(case.output / "completion.json"),
                "proofOutput": str(case.output / "sealed.json"),
                "leases": [str(case.output / "child-state/.workflow.lock")],
            }
        ),
        encoding="utf-8",
    )
    monkeypatch.setattr(retirement, "execute", lambda request, *, sha256, verify=False: retired)
    digest = file_digest(request)
    proof = seal.execute(request, sha256=digest)
    assert proof.name == "sealed.json"
    assert finalization._completed(job) == json.loads((case.output / "completion.json").read_text())
    assert seal.execute(request, sha256=digest, verify=True) == proof
