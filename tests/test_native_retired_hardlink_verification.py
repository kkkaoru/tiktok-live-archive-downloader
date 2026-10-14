import json
import os
import sys
from dataclasses import dataclass
from pathlib import Path

import pytest

from replay import native_retired_hardlink_verification as aliases
from replay.core import ReplayError
from replay.stages import Stage, _fingerprint, file_digest


def write(path: Path, value: object) -> Path:
    path.write_text(json.dumps(value), encoding="utf-8")
    return path


def pin(path: Path) -> dict[str, str]:
    return {"path": str(path), "sha256": file_digest(path)}


@dataclass(frozen=True)
class Case:
    root: Path
    alias: Path
    keeper: Path
    request: Path
    proof: Path
    checkpoint: Path
    original: Path
    seal: Path


@pytest.fixture
def case(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Case:
    monkeypatch.setattr(aliases, "ROOT", tmp_path)
    monkeypatch.setattr(aliases, "WORK", tmp_path)
    keeper = tmp_path / "keeper.mov"
    keeper.write_bytes(b"synthetic retained packets")
    alias = tmp_path / "alias.mov"
    os.link(keeper, alias)
    info = alias.stat()
    row = {
        "path": str(alias),
        "kind": "file",
        "dev": info.st_dev,
        "inode": info.st_ino,
        "nlink": 2,
        "bytes": info.st_size,
        "mtimeNs": info.st_mtime_ns,
        "ctimeNs": info.st_ctime_ns,
        "linkTarget": None,
        "classification": "eligible",
    }
    inventory = write(tmp_path / "inventory.json", {"rows": [row]})
    policy = write(
        tmp_path / "policy.json",
        {
            "eligiblePaths": [str(alias)],
            "eligibleTrees": [],
            "keepPaths": [str(keeper)],
        },
    )
    original = write(
        tmp_path / "original.json",
        {
            "kind": "native-admission-registry",
            "complete": True,
            "outputs": [{"path": str(alias), "owner": "custodian", "state": "retained"}],
        },
    )
    proof_path = tmp_path / "retired.json"
    retirement_request = write(
        tmp_path / "retire-request.json",
        {
            "version": 1,
            "kind": "verified-hardlink-publication-retirement",
            "alias": str(alias),
            "keeper": str(keeper),
            "output": str(proof_path),
            "policy": pin(policy),
            "inventory": pin(inventory),
        },
    )
    disposition = write(
        tmp_path / "disposition.json",
        {
            "version": 1,
            "kind": "verified-hardlink-alias-disposition",
            "request": pin(retirement_request),
            "rows": [row],
        },
    )
    log = write(
        tmp_path / "retirement.jsonl",
        {
            "path": str(alias),
            "removed": True,
            "kind": "file",
            "bytes": info.st_size,
        },
    )
    alias.unlink()
    proof = write(
        proof_path,
        {
            "version": 1,
            "kind": "verified-hardlink-publication-retired",
            "request": pin(retirement_request),
            "alias": str(alias),
            "keeper": pin(keeper),
            "disposition": pin(disposition),
            "retirementLog": pin(log),
            "logicalAliasBytes": info.st_size,
            "physicalBytesReclaimed": 0,
            "originalOuterStatusRewritten": False,
            "qualityApproved": False,
        },
    )
    suffix = (str(retirement_request), "--sha256", file_digest(retirement_request))
    stage = Stage(
        name="retire-alias",
        command=(sys.executable, "-m", "replay.verified_hardlink_publication", "run", *suffix),
        verify_command=(
            sys.executable,
            "-m",
            "replay.verified_hardlink_publication",
            "verify",
            *suffix,
        ),
        inputs=(retirement_request, inventory, policy),
        outputs=(proof, disposition, log),
        timeout_seconds=60,
    )
    workflow = write(tmp_path / "workflow.json", {"version": 1})
    monkeypatch.setattr(aliases, "load_stages", lambda path: (stage,))
    checkpoint = write(
        tmp_path / "complete.json",
        {
            "status": "complete",
            "fingerprint": _fingerprint(stage),
            "outputs": {
                str(proof): file_digest(proof),
                str(disposition): file_digest(disposition),
                str(log): file_digest(log),
            },
        },
    )
    request = write(
        tmp_path / "seal-request.json",
        {
            "version": 1,
            "kind": "native-retired-hardlink-seal-request",
            "originalRegistry": pin(original),
            "workflow": pin(workflow),
            "stage": "retire-alias",
            "completedStage": pin(checkpoint),
            "retirementRequest": pin(retirement_request),
            "retirementProof": pin(proof),
            "alias": str(alias),
            "keeper": str(keeper),
        },
    )
    return Case(
        tmp_path, alias, keeper, request, proof, checkpoint, original, tmp_path / "seal.json"
    )


def test_completed_alias_seals_without_media_backend(case: Case) -> None:
    result = aliases.seal(case.request, case.seal)
    assert result.owner == "custodian"
    assert not case.alias.exists()
    assert case.keeper.stat().st_nlink == 1
    document = json.loads(case.seal.read_text())
    assert document["qualityApproved"] is False
    assert document["physicalBytesReclaimed"] == 0
    aliases.verify(case.seal, alias=case.alias, owner="custodian")


@pytest.mark.parametrize("status", ["running", "failed"])
def test_first_command_receipt_is_not_completed_verification(case: Case, status: str) -> None:
    checkpoint = json.loads(case.checkpoint.read_text())
    checkpoint["status"] = status
    write(case.checkpoint, checkpoint)
    request = json.loads(case.request.read_text())
    request["completedStage"] = pin(case.checkpoint)
    write(case.request, request)
    with pytest.raises(ReplayError, match="Completed post-command"):
        aliases.seal(case.request, case.seal)


@pytest.mark.parametrize("field,value", [("fingerprint", "0" * 64), ("outputs", {})])
def test_corrupt_completed_provenance(case: Case, field: str, value: object) -> None:
    checkpoint = json.loads(case.checkpoint.read_text())
    checkpoint[field] = value
    write(case.checkpoint, checkpoint)
    request = json.loads(case.request.read_text())
    request["completedStage"] = pin(case.checkpoint)
    write(case.request, request)
    with pytest.raises(ReplayError, match="Completed post-command"):
        aliases.seal(case.request, case.seal)


def test_sealed_identity_rejects_content_change_with_restored_mtime(case: Case) -> None:
    aliases.seal(case.request, case.seal)
    before = case.keeper.stat()
    case.keeper.write_bytes(b"changed retained packets!")
    os.utime(case.keeper, ns=(before.st_atime_ns, before.st_mtime_ns))
    with pytest.raises(ReplayError, match="keeper identity"):
        aliases.verify(case.seal, alias=case.alias, owner="custodian")


def test_alias_reappeared_as_dangling_symlink(case: Case) -> None:
    aliases.seal(case.request, case.seal)
    case.alias.symlink_to(case.root / "absent")
    with pytest.raises(ReplayError, match="reappeared"):
        aliases.verify(case.seal, alias=case.alias, owner="custodian")


def test_keeper_replacement_and_extra_link_rejected(case: Case) -> None:
    aliases.seal(case.request, case.seal)
    os.link(case.keeper, case.root / "new-link")
    with pytest.raises(ReplayError, match="single-link"):
        aliases.verify(case.seal, alias=case.alias, owner="custodian")


def test_wrong_custodian_and_changed_evidence_rejected(case: Case) -> None:
    aliases.seal(case.request, case.seal)
    with pytest.raises(ReplayError, match="alias/owner"):
        aliases.verify(case.seal, alias=case.alias, owner="different")
    case.proof.write_text("{}", encoding="utf-8")
    with pytest.raises(ReplayError):
        aliases.verify(case.seal, alias=case.alias, owner="custodian")


def test_occupied_seal_is_not_overwritten(case: Case) -> None:
    case.seal.write_bytes(b"original unique evidence")
    with pytest.raises(FileExistsError):
        aliases.seal(case.request, case.seal)
    assert case.seal.read_bytes() == b"original unique evidence"


def test_bounded_duplicate_json_and_scope_rejected(case: Case) -> None:
    case.request.write_text('{"version":1,"version":1}', encoding="utf-8")
    with pytest.raises(ReplayError, match="Duplicate"):
        aliases.seal(case.request, case.seal)
    with pytest.raises(ReplayError, match="canonical"):
        aliases.reference({"path": str(case.root / "../escaped.json"), "sha256": "0" * 64})


def test_cli_seal_then_verify(case: Case, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(sys, "argv", ["aliases", "seal", str(case.request), str(case.seal)])
    aliases.main()
    monkeypatch.setattr(sys, "argv", ["aliases", "verify", str(case.seal)])
    aliases.main()


@pytest.mark.parametrize(
    "field", ["device", "inode", "bytes", "modifiedAtNs", "changedAtNs", "nlink"]
)
def test_every_sealed_keeper_identity_field_is_bound(case: Case, field: str) -> None:
    aliases.seal(case.request, case.seal)
    raw = json.loads(case.seal.read_text())
    raw["keeperIdentity"][field] += 1
    write(case.seal, raw)
    with pytest.raises(ReplayError):
        aliases.verify(case.seal, alias=case.alias, owner="custodian")


@pytest.mark.parametrize(
    "field", ["keeper", "keeperSHA256", "alias", "originalRegistry", "request", "completedStage"]
)
def test_sealed_body_relationship_mutations_rejected(case: Case, field: str) -> None:
    aliases.seal(case.request, case.seal)
    raw = json.loads(case.seal.read_text())
    if field in ("keeper", "alias"):
        raw[field] = str(case.root / "other.mov")
    elif field == "keeperSHA256":
        raw[field] = "0" * 64
    elif field == "completedStage":
        raw[field]["command"] = ["not-the-completed-command"]
    else:
        raw[field]["sha256"] = "0" * 64
    write(case.seal, raw)
    with pytest.raises(ReplayError):
        aliases.verify(case.seal, alias=case.alias, owner="custodian")


def test_alias_reappearance_during_verification(
    case: Case, monkeypatch: pytest.MonkeyPatch
) -> None:
    aliases.seal(case.request, case.seal)
    original = aliases._identity

    def identity(path: Path) -> dict[str, int]:
        result = original(path)
        case.alias.write_bytes(b"unexpected new alias")
        return result

    monkeypatch.setattr(aliases, "_identity", identity)
    with pytest.raises(ReplayError, match="reappeared"):
        aliases.verify(case.seal, alias=case.alias, owner="custodian")


def test_seal_rejects_unverified_kind(case: Case) -> None:
    request = json.loads(case.request.read_text())
    request["kind"] = "command-only-retirement"
    write(case.request, request)
    with pytest.raises(ReplayError, match="Unsupported"):
        aliases.seal(case.request, case.seal)


def test_seal_rejects_wrong_stage_and_original_row(case: Case) -> None:
    request = json.loads(case.request.read_text())
    request["stage"] = "unverified-stage"
    write(case.request, request)
    with pytest.raises(ReplayError, match="one exact"):
        aliases.seal(case.request, case.seal)
    request["stage"] = "retire-alias"
    original = json.loads(case.original.read_text())
    original["outputs"][0]["state"] = "planned"
    write(case.original, original)
    request["originalRegistry"] = pin(case.original)
    write(case.request, request)
    with pytest.raises(ReplayError, match="Original retained"):
        aliases.seal(case.request, case.seal)


def test_reference_rejects_malformed_hash_and_text(case: Case) -> None:
    with pytest.raises(ReplayError, match="immutable"):
        aliases.reference({"path": str(case.original), "sha256": "invalid"})
    with pytest.raises(ReplayError, match="bounded"):
        aliases.reference({"path": str(case.original), "sha256": 1})


def test_cli_accepts_repository_relative_arguments(
    case: Case, monkeypatch: pytest.MonkeyPatch
) -> None:
    request = case.request.relative_to(Path.cwd())
    output = case.seal.relative_to(Path.cwd())
    monkeypatch.setattr(sys, "argv", ["aliases", "seal", str(request), str(output)])
    aliases.main()
    monkeypatch.setattr(sys, "argv", ["aliases", "verify", str(output)])
    aliases.main()
