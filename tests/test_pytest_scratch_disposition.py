"""Synthetic terminal and external-evidence gates; no unlink/application calls."""

import fcntl
import hashlib
import os
from dataclasses import dataclass
from pathlib import Path
from unittest.mock import patch

import pytest

from replay import leaf_inventory
from replay import pytest_lifecycle as lifecycle
from replay import pytest_scratch_disposition as module
from replay.leaf_inventory import Limits, Plan
from replay.native_sequence import ContractError
from replay.sequence_journal import decode, encode


def ref(path: Path) -> dict[str, str]:
    return {"path": str(path), "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}


def write(path: Path, value: object) -> dict[str, str]:
    path.write_bytes(encode(value))
    return ref(path)


class Owner:
    def __init__(self, active: bool = False) -> None:
        self.active = active

    def inactive(self, pid: int, pgid: int) -> bool:
        return not self.active


@dataclass
class Run:
    checkout: Path
    work: Path
    request: Path
    output: Path
    base: Path
    python: Path
    fixture: Path


@pytest.fixture
def run_data(tmp_path: Path) -> Run:
    checkout = tmp_path / "checkout"
    work = checkout / "edits/work"
    work.mkdir(parents=True)
    base, python = work / "base", work / "python"
    base.mkdir()
    python.mkdir()
    fixture = base / "fixture"
    fixture.mkdir()
    (fixture / "synthetic.txt").write_text("test", encoding="utf-8")
    (python / "unknown.txt").write_text("keep", encoding="utf-8")
    evidence = work / "evidence"
    evidence.mkdir()
    output = work / "disposition"
    output.mkdir()
    owner = {"pid": 123456, "birth": "original", "pgid": 123456}
    source = work / "source.py"
    source.write_text("# synthetic source", encoding="utf-8")
    locks = work / "pytest-default/locks"
    locks.mkdir(parents=True)
    lock = locks / hashlib.sha256(os.fsencode(base)).hexdigest()
    lock.write_bytes(b"")
    info = lock.stat()
    start = write(
        evidence / "start.json",
        {
            "version": 1,
            "kind": "pytest-lifecycle-start",
            "owner": owner,
            "base": str(base),
            "pythonTmp": str(python),
            "baseLock": {"path": str(lock), "dev": info.st_dev, "inode": info.st_ino},
        },
    )
    finish = write(
        evidence / "finish.json",
        {
            "kind": "pytest-lifecycle-finish",
            "startSHA256": start["sha256"],
            "exitStatus": 0,
            "fixtures": {str(fixture): "test_a"},
            "outcomes": {"test_a": {"setup": "passed", "call": "passed", "teardown": "passed"}},
        },
    )
    terminal = write(
        evidence / "terminal.json",
        {
            "kind": "pytest-process-terminal",
            "owner": owner,
            "startSHA256": start["sha256"],
            "exitStatus": 0,
            "waitStatusObserved": True,
            "sourcePins": [ref(source)],
        },
    )
    coverage = write(
        evidence / "coverage.json", {"files": {"src/a.py": {}}, "totals": {"percent_covered": 100}}
    )
    approval = write(
        evidence / "approval.json",
        {
            "kind": "pytest-synthetic-retirement-approval",
            "startSHA256": start["sha256"],
            "disposablePaths": [],
        },
    )
    proof = write(evidence / "proof.json", {"verified": True})
    request = work / "request.json"
    write(
        request,
        {
            "version": 1,
            "kind": "pytest-scratch-finalization",
            "start": start,
            "finish": finish,
            "terminal": terminal,
            "coverage": coverage,
            "approval": approval,
            "proofs": [proof],
            "sourcePins": [ref(source)],
        },
    )
    return Run(checkout, work, request, output, base, python, fixture)


def change(data: Run, key: str, updates: dict[str, object]) -> None:
    raw = decode(data.request.read_bytes())
    reference = raw[key]
    assert isinstance(reference, dict)
    path = Path(str(reference["path"]))
    value = decode(path.read_bytes())
    value.update(updates)
    raw[key] = write(path, value)
    data.request.write_bytes(encode(raw))


def test_success_prepares_only_known_synthetic_leaves(run_data: Run) -> None:
    path = module.prepare(
        run_data.request, run_data.output, checkout=run_data.checkout, owner=Owner()
    )
    result = decode(path.read_bytes())
    rows = result["rows"]
    assert isinstance(rows, list)
    assert sorted(row["classification"] for row in rows) == ["eligible", "hold"]
    assert (run_data.fixture / "synthetic.txt").read_text(encoding="utf-8") == "test"
    assert decode((run_data.output / "prepared.json").read_bytes())["deletionPerformed"] is False
    with pytest.raises(ContractError):
        module.prepare(run_data.request, run_data.output, checkout=run_data.checkout, owner=Owner())


def test_failed_session_default_holds_everything(run_data: Run) -> None:
    change(run_data, "finish", {"exitStatus": 1})
    change(run_data, "terminal", {"exitStatus": 1})
    path = module.prepare(
        run_data.request, run_data.output, checkout=run_data.checkout, owner=Owner()
    )
    rows = decode(path.read_bytes())["rows"]
    assert isinstance(rows, list)
    assert [row["classification"] for row in rows] == ["keep", "hold"]


def test_explicit_passed_fixture_in_failed_run_only(run_data: Run) -> None:
    change(run_data, "finish", {"exitStatus": 1})
    change(run_data, "terminal", {"exitStatus": 1})
    change(run_data, "approval", {"disposablePaths": [str(run_data.fixture)]})
    path = module.prepare(
        run_data.request, run_data.output, checkout=run_data.checkout, owner=Owner()
    )
    rows = decode(path.read_bytes())["rows"]
    assert isinstance(rows, list)
    assert sorted(row["classification"] for row in rows) == ["eligible", "hold"]


@pytest.mark.parametrize(
    "key,updates",
    [
        ("terminal", {"waitStatusObserved": False}),
        ("terminal", {"exitStatus": 2}),
        ("coverage", {"files": {}}),
        ("approval", {"kind": "unknown"}),
        ("approval", {"disposablePaths": [1]}),
        ("start", {"kind": "unknown"}),
    ],
)
def test_incomplete_or_wrong_external_evidence_refused(
    run_data: Run, key: str, updates: dict[str, object]
) -> None:
    change(run_data, key, updates)
    with pytest.raises(ContractError):
        module.prepare(run_data.request, run_data.output, checkout=run_data.checkout, owner=Owner())
    assert (run_data.fixture / "synthetic.txt").is_file()
    assert not (run_data.output / "prepared.json").exists()


def test_live_owner_and_missing_original_lock_refused(run_data: Run) -> None:
    with pytest.raises(ContractError):
        module.prepare(
            run_data.request, run_data.output, checkout=run_data.checkout, owner=Owner(active=True)
        )
    lock = (
        run_data.work
        / "pytest-default/locks"
        / hashlib.sha256(os.fsencode(run_data.base)).hexdigest()
    )
    lock.unlink()
    with pytest.raises(FileNotFoundError):
        module.prepare(run_data.request, run_data.output, checkout=run_data.checkout, owner=Owner())


def test_proof_inside_pytest_is_not_preserved_evidence(run_data: Run) -> None:
    raw = decode(run_data.request.read_bytes())
    raw["proofs"] = [write(run_data.base / "proof.json", {"verified": True})]
    run_data.request.write_bytes(encode(raw))
    with pytest.raises(ContractError):
        module.prepare(run_data.request, run_data.output, checkout=run_data.checkout, owner=Owner())


def test_owner_observation_errors_never_mean_absent() -> None:
    with patch.object(os, "kill", side_effect=PermissionError), pytest.raises(PermissionError):
        module.SystemOwnerProbe().inactive(123, 123)
    with patch.object(os, "kill"):
        assert module.SystemOwnerProbe().inactive(123, 123) is False
    with patch.object(os, "kill", side_effect=ProcessLookupError), patch.object(os, "killpg"):
        assert module.SystemOwnerProbe().inactive(123, 123) is False
    with (
        patch.object(os, "kill", side_effect=ProcessLookupError),
        patch.object(os, "killpg", side_effect=ProcessLookupError),
    ):
        assert module.SystemOwnerProbe().inactive(123, 123) is True


def test_cli_delegates_only_prepare(run_data: Run) -> None:
    with patch.object(
        module, "prepare", return_value=run_data.output / "disposition.json"
    ) as prepared:
        assert module.main([str(run_data.request), str(run_data.output)]) == 0
    prepared.assert_called_once_with(run_data.request, run_data.output)


@pytest.mark.parametrize(
    "field,value",
    [
        ("proofs", []),
        ("sourcePins", [{"path": "/private/no-read.py", "sha256": "no"}]),
        ("kind", "unknown"),
        ("terminal", {"path": "invalid"}),
    ],
)
def test_invalid_parent_request_refused(run_data: Run, field: str, value: object) -> None:
    raw = decode(run_data.request.read_bytes())
    raw[field] = value
    run_data.request.write_bytes(encode(raw))
    with pytest.raises(ContractError):
        module.prepare(run_data.request, run_data.output, checkout=run_data.checkout, owner=Owner())
    assert not (run_data.output / "prepared.json").exists()


def test_replaced_original_lock_and_unknown_fixture_refused(run_data: Run) -> None:
    lock = (
        run_data.work
        / "pytest-default/locks"
        / hashlib.sha256(os.fsencode(run_data.base)).hexdigest()
    )
    replacement = lock.with_suffix(".replacement")
    replacement.write_bytes(b"replacement")
    replacement.replace(lock)
    with pytest.raises(ContractError, match="lock identity"):
        module.prepare(run_data.request, run_data.output, checkout=run_data.checkout, owner=Owner())


def test_fixture_outside_both_actual_roots_refused(run_data: Run) -> None:
    change(run_data, "finish", {"fixtures": {str(run_data.work / "not-scratch"): "test_a"}})
    with pytest.raises(ContractError, match="outside actual scratch"):
        module.prepare(run_data.request, run_data.output, checkout=run_data.checkout, owner=Owner())


def test_output_inside_scratch_and_invalid_owner_refused(run_data: Run) -> None:
    child = run_data.python / "output"
    child.mkdir()
    with pytest.raises(ContractError, match="overlaps"):
        module.prepare(run_data.request, child, checkout=run_data.checkout, owner=Owner())


def test_shared_hardlink_is_held_by_original_policy_builder(run_data: Run) -> None:
    os.link(run_data.fixture / "synthetic.txt", run_data.work / "external-link.txt")
    output = module.prepare(
        run_data.request, run_data.output, checkout=run_data.checkout, owner=Owner()
    )
    rows = decode(output.read_bytes())["rows"]
    assert isinstance(rows, list)
    assert [row["classification"] for row in rows] == ["hold", "hold"]
    assert (run_data.work / "external-link.txt").is_file()


def test_live_lock_prevents_inventory_and_preserves_all_inputs(run_data: Run) -> None:
    lock = (
        run_data.work
        / "pytest-default/locks"
        / hashlib.sha256(os.fsencode(run_data.base)).hexdigest()
    )
    with lock.open("rb") as stream:
        fcntl.flock(stream.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        with pytest.raises(BlockingIOError):
            module.prepare(
                run_data.request, run_data.output, checkout=run_data.checkout, owner=Owner()
            )
    assert list(run_data.output.iterdir()) == []


def test_new_activity_during_disposition_leaves_no_ready_receipt(run_data: Run) -> None:
    owner = Owner()
    with (
        patch.object(owner, "inactive", side_effect=[True, False]),
        pytest.raises(ContractError, match="authority changed"),
    ):
        module.prepare(run_data.request, run_data.output, checkout=run_data.checkout, owner=owner)
    assert (run_data.output / "disposition.json").is_file()
    assert not (run_data.output / "prepared.json").exists()
    assert (run_data.fixture / "synthetic.txt").is_file()


def test_large_inventory_uses_bounded_metadata_domain(run_data: Run) -> None:
    original = leaf_inventory.inventory

    def large_inventory(
        plan: Plan, output: Path, *, limits: Limits | None = None
    ) -> dict[str, int]:
        result = original(plan, output, limits=limits)
        with output.open("ab") as stream:
            stream.write(b" " * (18 * 1024 * 1024))
        return result

    with patch.object(leaf_inventory, "inventory", side_effect=large_inventory):
        path = module.prepare(
            run_data.request, run_data.output, checkout=run_data.checkout, owner=Owner()
        )
    assert path.name == "disposition.json"
    assert (run_data.output / "prepared.json").is_file()
    assert (run_data.output / "inventory.json").stat().st_size > 18 * 1024 * 1024


@pytest.mark.parametrize("tampered", [False, True])
def test_sharded_terminal_payload_is_required_by_finalizer(run_data: Run, tampered: bool) -> None:
    request = decode(run_data.request.read_bytes())
    finish = decode((run_data.work / "evidence/finish.json").read_bytes())
    path = run_data.work / "evidence/finish-sharded.json"
    lifecycle._write_finish(path, finish)
    request["finish"] = ref(path)
    run_data.request.write_bytes(encode(request))
    if tampered:
        path.with_name("finish-sharded-0000.fragment").write_bytes(b"changed")
        with pytest.raises(ContractError, match="fragment identity"):
            module.prepare(
                run_data.request, run_data.output, checkout=run_data.checkout, owner=Owner()
            )
        assert not (run_data.output / "prepared.json").exists()
    else:
        result = module.prepare(
            run_data.request, run_data.output, checkout=run_data.checkout, owner=Owner()
        )
        assert result.name == "disposition.json"
        assert (run_data.output / "prepared.json").is_file()
