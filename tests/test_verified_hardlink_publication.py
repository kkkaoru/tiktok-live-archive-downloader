"""Only a proven, specifically approved publication alias may be removed."""

import json
import os
from fractions import Fraction
from pathlib import Path

import pytest

from replay import leaf_retirement as leaves
from replay import measured_job_admission as admission
from replay import verified_hardlink_publication as publication
from replay.core import ReplayError
from replay.fcp_part_assembly import _reference
from replay.stages import file_digest


@pytest.fixture
def alias_request(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    monkeypatch.setattr(publication, "ROOT", tmp_path)
    work = tmp_path / "edits/work"
    work.mkdir(parents=True)
    keeper = work / "keeper.mov"
    keeper.write_bytes(b"verified synthetic full media")
    alias = work / "pending.mov"
    os.link(keeper, alias)
    info = alias.stat()
    inventory = work / "inventory.json"
    inventory.write_text(
        json.dumps(
            {
                "rows": [
                    {
                        "path": str(alias),
                        "kind": "file",
                        "classification": "hold",
                        "dev": info.st_dev,
                        "inode": info.st_ino,
                        "nlink": info.st_nlink,
                        "bytes": info.st_size,
                        "mtimeNs": info.st_mtime_ns,
                        "ctimeNs": info.st_ctime_ns,
                        "linkTarget": None,
                    }
                ]
            }
        )
    )
    approval = work / "approval.md"
    approval.write_text("Owner approves only the redundant pending alias.")
    status = work / "producer.status"
    status.write_text("1\n")
    policy = work / "policy.json"
    policy.write_text(
        json.dumps(
            {
                "eligiblePaths": [str(alias)],
                "eligibleTrees": [],
                "keepPaths": [str(keeper)],
                "approval": _reference(approval),
                "terminalProofs": [_reference(status)],
            }
        )
    )
    assembly_receipt = work / "assembly.json"
    assembly_receipt.write_text(json.dumps({"output": _reference(keeper)}))
    child = work / "child.json"
    child.write_text(
        json.dumps({"status": "complete", "outputs": {str(keeper): file_digest(keeper)}})
    )
    current = work / "current.json"
    current.write_text(json.dumps({"source": str(keeper)}))
    capture = work / "capture.json"
    capture.write_text("{}")
    parent = admission.Parent(123, "100:1", keeper, "a" * 64)
    completed = admission.Completion(
        capture,
        file_digest(capture),
        current,
        file_digest(current),
        child,
        status,
        "b" * 64,
        (str(keeper),),
        (123,),
    )
    monkeypatch.setattr(
        admission, "load_completion_capture", lambda path, digest: (parent, completed)
    )
    monkeypatch.setattr(admission, "probe", lambda command: "1 1\n")
    monkeypatch.setattr(leaves, "lsof_open_among", lambda paths: set())
    monkeypatch.setattr(
        publication, "clean_source", lambda config: (work / "original.mov", keeper, Fraction(2))
    )
    lease = work / ".workflow.lock"
    lease.write_bytes(b"")
    request = work / "request.json"
    request.write_text(
        json.dumps(
            {
                "version": 1,
                "kind": "verified-hardlink-publication-retirement",
                "alias": str(alias),
                "keeper": str(keeper),
                "policy": _reference(policy),
                "inventory": _reference(inventory),
                "assembly": {"assemblyReceipt": _reference(assembly_receipt)},
                "childReceipt": _reference(child),
                "captures": [_reference(capture)],
                "leases": [str(lease)],
                "currentReferences": [_reference(current)],
                "output": str(work / "retired.json"),
            }
        )
    )
    return request


def test_retire_and_verify_only_alias(alias_request: Path) -> None:
    digest = file_digest(alias_request)
    output = publication.execute(alias_request, sha256=digest)
    assert alias_request.with_name("keeper.mov").read_bytes() == b"verified synthetic full media"
    assert alias_request.with_name("keeper.mov").stat().st_nlink == 1
    assert not alias_request.with_name("pending.mov").exists()
    assert json.loads(output.read_text())["physicalBytesReclaimed"] == 0
    assert publication.execute(alias_request, sha256=digest, verify=True) == output
    with pytest.raises(ReplayError, match="already exists"):
        publication.execute(alias_request, sha256=digest)


@pytest.mark.parametrize(
    "field,value", [("version", 2), ("captures", []), ("leases", []), ("currentReferences", [])]
)
def test_invalid_request(alias_request: Path, field: str, value: object) -> None:
    data = json.loads(alias_request.read_text())
    data[field] = value
    alias_request.write_text(json.dumps(data))
    with pytest.raises(ReplayError):
        publication.execute(alias_request, sha256=file_digest(alias_request))
    assert alias_request.with_name("pending.mov").exists()


@pytest.mark.parametrize("result", [None, {"open"}])
def test_fd_guard(
    alias_request: Path, monkeypatch: pytest.MonkeyPatch, result: set[str] | None
) -> None:
    monkeypatch.setattr(leaves, "lsof_open_among", lambda paths: result)
    with pytest.raises(ReplayError, match="FD check"):
        publication.execute(alias_request, sha256=file_digest(alias_request))
    assert alias_request.with_name("pending.mov").exists()


def test_actor_group_present(alias_request: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(admission, "probe", lambda command: "456 123\n")
    with pytest.raises(ReplayError, match="not genuinely terminal"):
        publication.execute(alias_request, sha256=file_digest(alias_request))


def test_current_reference_protects_alias(alias_request: Path) -> None:
    current = alias_request.with_name("current.json")
    current.write_text(
        json.dumps({"nested": [{"media": str(alias_request.with_name("pending.mov"))}]})
    )
    data = json.loads(alias_request.read_text())
    data["currentReferences"] = [_reference(current)]
    alias_request.write_text(json.dumps(data))
    with pytest.raises(ReplayError, match="still references"):
        publication.execute(alias_request, sha256=file_digest(alias_request))


def test_keeper_changed(alias_request: Path) -> None:
    alias_request.with_name("keeper.mov").write_bytes(b"tampered")
    with pytest.raises(ReplayError, match="changed"):
        publication.execute(alias_request, sha256=file_digest(alias_request))


@pytest.mark.parametrize("name", ["retired.json", "keeper.mov"])
def test_retained_tamper(alias_request: Path, name: str) -> None:
    digest = file_digest(alias_request)
    publication.execute(alias_request, sha256=digest)
    alias_request.with_name(name).write_text("{}")
    with pytest.raises(ReplayError):
        publication.execute(alias_request, sha256=digest, verify=True)


def test_wrong_approval(alias_request: Path) -> None:
    policy = alias_request.with_name("policy.json")
    data = json.loads(policy.read_text())
    data["eligiblePaths"] = []
    policy.write_text(json.dumps(data))
    request = json.loads(alias_request.read_text())
    request["policy"] = _reference(policy)
    alias_request.write_text(json.dumps(request))
    with pytest.raises(ReplayError, match="Owner policy"):
        publication.execute(alias_request, sha256=file_digest(alias_request))


def test_scope_and_relative_path(alias_request: Path) -> None:
    with pytest.raises(ReplayError, match="checkout"):
        publication._local(str(alias_request.parents[2] / "escape"))
    with pytest.raises(ReplayError, match="approved alias"):
        publication._relative(None)
    assert publication._relative("edits/work/pending.mov").name == "pending.mov"


def test_cli(alias_request: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        "sys.argv",
        ["publication", "run", str(alias_request), "--sha256", file_digest(alias_request)],
    )
    publication.main()
