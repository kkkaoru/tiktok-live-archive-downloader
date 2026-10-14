"""Prepare exact pytest leaf dispositions after terminal evidence; never delete."""

import argparse
import fcntl
import hashlib
import os
from pathlib import Path
from typing import Protocol

from replay import leaf_disposition, leaf_inventory
from replay.native_sequence import ContractError
from replay.pytest_lifecycle import canonical, load_finish
from replay.recordings import record
from replay.sequence_journal import Pin, bounded_read, decode, encode, publish

CHECKOUT = Path(__file__).resolve().parents[2]
EVIDENCE_SUFFIXES = frozenset(
    {".json", ".jsonl", ".log", ".status", ".sha256", ".txt", ".md", ".py", ".sh", ".toml", ".lock"}
)


class OwnerProbe(Protocol):
    def inactive(self, pid: int, pgid: int) -> bool: ...


class SystemOwnerProbe:
    def inactive(self, pid: int, pgid: int) -> bool:
        try:
            os.kill(pid, 0)
        except ProcessLookupError:
            pass
        else:
            return False
        try:
            os.killpg(pgid, 0)
        except ProcessLookupError:
            return True
        return False


def _pin(raw: object, roots: tuple[Path, ...], work: Path) -> Pin:
    value = record(raw)
    name, digest = value.get("path"), value.get("sha256")
    if set(value) != {"path", "sha256"} or not isinstance(name, str) or not isinstance(digest, str):
        raise ContractError("Evidence requires an exact metadata pin")
    path = canonical(Path(name), work)
    if path.suffix not in EVIDENCE_SUFFIXES:
        raise ContractError("Only declared code and metadata evidence may be read")
    if any(path.is_relative_to(root) for root in roots):
        raise ContractError("Required proof must remain outside pytest scratch")
    result = Pin(path, digest)
    result.verify()
    return result


def _reference(pin: Pin) -> leaf_disposition.Reference:
    return leaf_disposition.Reference(pin.path, pin.sha256)


def _int(raw: dict[str, object], key: str) -> int:
    value = raw.get(key)
    if type(value) is not int or not isinstance(value, int) or value <= 0:
        raise ContractError("Owner requires exact positive PID and process group")
    return value


def prepare(
    request: Path, output: Path, *, checkout: Path = CHECKOUT, owner: OwnerProbe | None = None
) -> Path:
    """Validate original terminal authority then delegate inventory and policy building."""
    work = checkout / "edits/work"
    canonical(request, work)
    canonical(output, work)
    if not output.is_dir() or any(output.iterdir()):
        raise ContractError("Disposition output directory must be fresh and exclusive")
    original = bounded_read(request)
    raw = decode(original)
    if (
        set(raw)
        != {
            "version",
            "kind",
            "start",
            "finish",
            "terminal",
            "coverage",
            "approval",
            "proofs",
            "sourcePins",
        }
        or raw.get("version") != 1
        or raw.get("kind") != "pytest-scratch-finalization"
    ):
        raise ContractError("Invalid parent finalization request")
    start_ref = _pin(raw["start"], (), work)
    start = decode(bounded_read(start_ref.path))
    if start.get("kind") != "pytest-lifecycle-start" or start.get("version") != 1:
        raise ContractError("Actual lifecycle start required")
    roots = (
        canonical(Path(str(start.get("base"))), work),
        canonical(Path(str(start.get("pythonTmp"))), work),
    )
    if roots[0] == roots[1] or any(
        output.is_relative_to(root) or root.is_relative_to(output) for root in roots
    ):
        raise ContractError("Evidence output overlaps actual scratch")
    refs = tuple(
        _pin(raw[key], roots, work)
        for key in ("start", "finish", "terminal", "coverage", "approval")
    )
    extra: list[Pin] = []
    for key in ("proofs", "sourcePins"):
        rows = raw[key]
        if not isinstance(rows, list) or not rows or len(rows) > 4096:
            raise ContractError("Required retained proof/source inventory is incomplete")
        for row in rows:
            source = record(row).get("path")
            if key == "sourcePins" and (
                not isinstance(source, str)
                or not any(
                    Path(source).is_relative_to(checkout / prefix)
                    for prefix in ("src", "tests", "scripts", "docs", "edits/work")
                )
            ):
                raise ContractError("Source pins must reference declared code or work evidence")
            extra.append(_pin(row, roots, checkout if key == "sourcePins" else work))
    finish = load_finish(refs[1].path)
    terminal, coverage, approval = (decode(bounded_read(ref.path)) for ref in refs[2:])
    owner_row = record(start.get("owner"))
    pid, pgid = _int(owner_row, "pid"), _int(owner_row, "pgid")
    if (
        not isinstance(owner_row.get("birth"), str)
        or not owner_row["birth"]
        or terminal.get("kind") != "pytest-process-terminal"
        or terminal.get("owner") != owner_row
        or terminal.get("waitStatusObserved") is not True
        or terminal.get("startSHA256") != start_ref.sha256
        or finish.get("startSHA256") != start_ref.sha256
        or finish.get("kind") != "pytest-lifecycle-finish"
        or type(finish.get("exitStatus")) is not int
        or terminal.get("exitStatus") != finish.get("exitStatus")
        or terminal.get("sourcePins") != raw["sourcePins"]
    ):
        raise ContractError("Actual parent exit and plugin outcome do not agree")
    if (
        not isinstance(coverage.get("files"), dict)
        or not coverage["files"]
        or not isinstance(coverage.get("totals"), dict)
    ):
        raise ContractError("Retained coverage JSON is incomplete")
    if (
        approval.get("kind") != "pytest-synthetic-retirement-approval"
        or approval.get("startSHA256") != start_ref.sha256
    ):
        raise ContractError("Explicit original run approval required")
    if not (SystemOwnerProbe() if owner is None else owner).inactive(pid, pgid):
        raise ContractError("Pytest owner or process group remains active")
    lock = work / "pytest-default/locks" / hashlib.sha256(os.fsencode(roots[0])).hexdigest()
    canonical(lock, work)
    descriptor = os.open(lock, os.O_RDONLY | os.O_NOFOLLOW)
    try:
        lock_info = os.fstat(descriptor)
        if start.get("baseLock") != {
            "path": str(lock),
            "dev": lock_info.st_dev,
            "inode": lock_info.st_ino,
        } or (lock.stat().st_dev, lock.stat().st_ino) != (lock_info.st_dev, lock_info.st_ino):
            raise ContractError("Original scratch lock identity changed")
        fcntl.flock(descriptor, fcntl.LOCK_EX | fcntl.LOCK_NB)
        fixtures, outcomes = record(finish.get("fixtures")), record(finish.get("outcomes"))
        eligible: list[Path] = []
        keep: list[Path] = []
        explicit = approval.get("disposablePaths", [])
        if not isinstance(explicit, list) or any(not isinstance(item, str) for item in explicit):
            raise ContractError("Approved disposable paths must be exact strings")
        for name, node in fixtures.items():
            path = canonical(Path(name), work)
            if not isinstance(node, str) or not any(
                path.is_relative_to(root) and path != root for root in roots
            ):
                raise ContractError("Fixture ownership is outside actual scratch")
            phases = record(outcomes.get(node))
            passed = phases == {"setup": "passed", "call": "passed", "teardown": "passed"}
            if passed and (finish["exitStatus"] == 0 or name in explicit):
                eligible.append(path)
            else:
                keep.append(path)
        plan_path = output / "inventory-plan.json"
        publish(
            plan_path,
            encode({"version": 1, "roots": [str(root.relative_to(checkout)) for root in roots]}),
        )
        inventory_path = output / "inventory.json"
        leaf_inventory.inventory(
            leaf_inventory.load_plan(plan_path, checkout=checkout), inventory_path
        )
        inventory_ref = leaf_disposition.metadata_reference(inventory_path, checkout=checkout)
        authority = output / "authority.json"
        publish(
            authority,
            encode(
                {
                    "version": 1,
                    "kind": "pytest-terminal-authority",
                    "requestSHA256": hashlib.sha256(original).hexdigest(),
                    "evidence": [
                        {"path": str(ref.path), "sha256": ref.sha256} for ref in (*refs, *extra)
                    ],
                    "ownerInactive": True,
                    "lockIdentityVerified": True,
                }
            ),
        )
        policy = leaf_disposition.Policy(
            owner=f"pytest:{pid}:{owner_row['birth']}",
            eligible_trees=tuple(eligible),
            keep_trees=tuple(keep),
            approval=_reference(refs[4]),
            terminal_proofs=(
                *tuple(_reference(ref) for ref in refs[:4]),
                leaf_disposition.Reference(
                    authority, hashlib.sha256(bounded_read(authority)).hexdigest()
                ),
            ),
        )
        policy_ref = leaf_disposition.write_policy(
            policy, output / "policy.json", checkout=checkout
        )
        disposition = output / "disposition.json"
        leaf_disposition.build_disposition(
            inventory_ref, policy_ref, disposition, checkout=checkout
        )
        disposition_ref = leaf_disposition.metadata_reference(disposition, checkout=checkout)
        for ref in (*refs, *extra):
            ref.verify()
        load_finish(refs[1].path)
        if (
            bounded_read(request) != original
            or not (SystemOwnerProbe() if owner is None else owner).inactive(pid, pgid)
            or (lock.stat().st_dev, lock.stat().st_ino) != (lock_info.st_dev, lock_info.st_ino)
        ):
            raise ContractError("Finalization authority changed")
        publish(
            output / "prepared.json",
            encode(
                {
                    "version": 1,
                    "kind": "pytest-scratch-disposition-prepared",
                    "requestSHA256": hashlib.sha256(original).hexdigest(),
                    "disposition": {
                        "path": str(disposition),
                        "sha256": disposition_ref.sha256,
                    },
                    "deletionPerformed": False,
                    "autoApplyVerified": False,
                }
            ),
        )
        return disposition
    finally:
        os.close(descriptor)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("request", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args(argv)
    print(prepare(args.request, args.output))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
