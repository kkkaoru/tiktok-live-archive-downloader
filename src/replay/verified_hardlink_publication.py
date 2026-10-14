"""Retire one explicitly approved publication alias while retaining verified bytes."""

import argparse
import re
from contextlib import ExitStack
from pathlib import Path

from replay import cleanup_lane_successor as locks
from replay import code_inventory_memo_run as memo
from replay import leaf_retirement as leaves
from replay import measured_job_admission as admission
from replay import retained_publication_support as retained_publication
from replay.artifact_json import _json, _path, _pin
from replay.core import ReplayError
from replay.fcp_part_assembly import _reference, _save
from replay.fcp_transformed_source import clean_source
from replay.recordings import record
from replay.stages import file_digest

ROOT = Path(__file__).resolve().parents[2]
FIELDS = frozenset(
    {
        "version",
        "kind",
        "alias",
        "keeper",
        "policy",
        "inventory",
        "assembly",
        "childReceipt",
        "captures",
        "leases",
        "currentReferences",
        "output",
    }
)


def _local(value: object) -> Path:
    path = _path(value)
    if not path.is_relative_to(ROOT / "edits/work"):
        raise ReplayError("Publication retirement must remain in checkout edits/work")
    return path


def _relative(value: object) -> Path:
    if not isinstance(value, str):
        raise ReplayError("Expected exact approved alias path")
    path = Path(value)
    return _local(str(path if path.is_absolute() else ROOT / path))


def _mentions(value: object, alias: str) -> bool:
    if isinstance(value, str):
        return alias in value
    if isinstance(value, dict):
        return any(_mentions(key, alias) or _mentions(item, alias) for key, item in value.items())
    if isinstance(value, list):
        return any(_mentions(item, alias) for item in value)
    return False


def _actors(references: object) -> list[dict[str, object]]:
    if not isinstance(references, list) or not 1 <= len(references) <= 8:
        raise ReplayError("Require explicit genuine terminal actor captures")
    result: list[dict[str, object]] = []
    for value in references:
        capture = _pin(value)
        parent, completion = admission.load_completion_capture(capture, file_digest(capture))
        before = admission.parse_groups(admission.probe(["/bin/ps", "-A", "-o", "pid=,pgid="]))
        status = completion.status.read_bytes()
        if re.fullmatch(rb"(?:0|[1-9][0-9]{0,2})\n?", status) is None or any(
            pid == parent.pid or group in completion.process_groups for pid, group in before
        ):
            raise ReplayError("Captured actor or owned group is not genuinely terminal and absent")
        result.append(
            {
                "capture": _reference(capture),
                "parentPID": parent.pid,
                "parentBirth": parent.birth,
                "groupsAbsent": list(completion.process_groups),
                "status": _reference(completion.status),
                "terminalExit": int(status),
            }
        )
    return result


def _load(request: Path, sha256: str) -> dict[str, object]:
    _local(str(request))
    _pin({"path": str(request), "sha256": sha256})
    data = _json(request)
    version = data.get("version")
    fields = FIELDS | {"publicationOperation"} if version == 2 else FIELDS
    if (
        set(data) != fields
        or version not in (1, 2)
        or data.get("kind") != "verified-hardlink-publication-retirement"
    ):
        raise ReplayError("Invalid publication retirement request")
    return data


def _checked(data: dict[str, object], *, retired: bool) -> tuple[Path, Path, dict[str, object]]:
    alias, keeper = _local(data.get("alias")), _local(data.get("keeper"))
    policy = _json(_pin(data.get("policy")))
    _pin(policy.get("approval"))
    approved = policy.get("eligiblePaths")
    kept = policy.get("keepPaths")
    if (
        approved != [str(alias)]
        or not isinstance(kept, list)
        or str(keeper) not in kept
        or policy.get("eligibleTrees") != []
    ):
        raise ReplayError("Owner policy does not approve exactly this alias and retained keeper")
    terminal = policy.get("terminalProofs")
    if not isinstance(terminal, list) or not terminal:
        raise ReplayError("Owner policy lacks pinned terminal evidence")
    tuple(_pin(value) for value in terminal)
    rows = _json(_pin(data.get("inventory"))).get("rows")
    if not isinstance(rows, list):
        raise ReplayError("Expected original immutable alias inventory")
    matches = [record(row) for row in rows if _relative(record(row).get("path")) == alias]
    if len(matches) != 1:
        raise ReplayError("Inventory does not identify exactly one approved alias")
    row = {**matches[0], "path": str(alias), "classification": "eligible"}
    metadata = leaves._row(row)
    info = keeper.lstat()
    if (
        (info.st_dev, info.st_ino, info.st_size, info.st_mtime_ns)
        != (metadata.dev, metadata.inode, metadata.size, metadata.mtime_ns)
        or metadata.nlink != 2
        or info.st_nlink != (1 if retired else 2)
    ):
        raise ReplayError("Retained publication inode, content identity or link count changed")
    if retired:
        if alias.exists() or alias.is_symlink():
            raise ReplayError("Retired publication alias reappeared")
    elif leaves.static_hold(metadata, (ROOT / "edits/work",)) is not None:
        raise ReplayError("Approved alias identity changed")
    assembly = record(data.get("assembly"))
    _, clean, _ = clean_source(assembly)
    child = _json(_pin(data.get("childReceipt")))
    receipt = _json(_pin(assembly.get("assemblyReceipt")))
    keeper_pin = _reference(keeper)
    completed_child = (
        child.get("status") == "complete"
        and record(child.get("outputs")).get(str(keeper)) == keeper_pin["sha256"]
    )
    if data.get("version") == 2:
        job = retained_publication.pending_job(
            _pin(data.get("publicationOperation")), _pin(data.get("childReceipt"))
        )
        completed_child = clean == job.root / "clean-source.mov"
    if clean != keeper or receipt.get("output") != keeper_pin or not completed_child:
        raise ReplayError("Genuine completed child does not prove the retained publication")
    refs = data.get("currentReferences")
    if not isinstance(refs, list) or not 1 <= len(refs) <= 64:
        raise ReplayError("Require current producer and consumer reference closure")
    representations = (str(alias), alias.relative_to(ROOT).as_posix(), alias.as_uri())
    if any(any(_mentions(_json(_pin(value)), name) for name in representations) for value in refs):
        raise ReplayError("Current metadata still references the publication alias")
    return alias, keeper, row


def execute(request: Path, *, sha256: str, verify: bool = False) -> Path:
    """Reconcile a publication boundary, never media failure or producer completion."""
    data = _load(request, sha256)
    output = _local(data.get("output"))
    paths = data.get("leases")
    if not isinstance(paths, list) or not 1 <= len(paths) <= 8:
        raise ReplayError("Require original producer and child writer leases")
    with ExitStack() as stack:
        for path in sorted({_local(value) for value in paths}):
            stack.enter_context(locks._unlocked(path))
        actors = _actors(data.get("captures"))
        if data.get("version") == 2:
            stack.enter_context(memo.capture_scope())
            stack.enter_context(
                retained_publication.pending_child_ready(
                    _pin(data.get("publicationOperation")), _pin(data.get("childReceipt"))
                )
            )
        if verify:
            alias, keeper, row = _checked(data, retired=True)
            proof = _json(output)
            disposition = _pin(proof.get("disposition"))
            log = _pin(proof.get("retirementLog"))
            if _json(disposition) != {
                "version": 1,
                "kind": "verified-hardlink-alias-disposition",
                "request": _reference(request),
                "rows": [row],
            } or _json(log) != {
                "path": str(alias),
                "removed": True,
                "kind": "file",
                "bytes": row["bytes"],
            }:
                raise ReplayError("Publication disposition or executor log changed")
            if (
                set(proof)
                != {
                    "version",
                    "kind",
                    "request",
                    "alias",
                    "keeper",
                    "actors",
                    "disposition",
                    "retirementLog",
                    "logicalAliasBytes",
                    "physicalBytesReclaimed",
                    "originalOuterStatusRewritten",
                    "qualityApproved",
                }
                or proof.get("version") != 1
                or proof.get("kind") != "verified-hardlink-publication-retired"
                or proof.get("logicalAliasBytes") != row["bytes"]
                or proof.get("originalOuterStatusRewritten") is not False
                or proof.get("qualityApproved") is not False
                or proof.get("request") != _reference(request)
                or proof.get("keeper") != _reference(keeper)
                or proof.get("actors") != actors
                or proof.get("alias") != str(alias)
                or proof.get("physicalBytesReclaimed") != 0
            ):
                raise ReplayError("Publication retirement proof changed")
            return output
        if output.exists():
            raise ReplayError("Publication retirement proof already exists")
        alias, keeper, row = _checked(data, retired=False)
        if leaves.lsof_open_among((str(alias), str(keeper))) != set():
            raise ReplayError("Alias or retained publication is open or FD check failed")
        disposition = output.with_suffix(".disposition.json")
        _save(
            disposition,
            {
                "version": 1,
                "kind": "verified-hardlink-alias-disposition",
                "request": _reference(request),
                "rows": [row],
            },
        )
        if _actors(data.get("captures")) != actors or file_digest(request) != sha256:
            raise ReplayError("Actor or approval evidence changed before retirement")
        result = leaves.retire(
            disposition,
            file_digest(disposition),
            (ROOT / "edits/work",),
            output.with_suffix(".retirement.jsonl"),
            apply=True,
            probe=leaves.lsof_open_among,
        )
        if result.retired != 1 or result.held:
            raise ReplayError("Verified publication alias was held; inspect retirement log")
        _checked(data, retired=True)
        _save(
            output,
            {
                "version": 1,
                "kind": "verified-hardlink-publication-retired",
                "request": _reference(request),
                "alias": str(alias),
                "keeper": _reference(keeper),
                "actors": actors,
                "disposition": _reference(disposition),
                "retirementLog": _reference(output.with_suffix(".retirement.jsonl")),
                "logicalAliasBytes": result.logical_bytes,
                "physicalBytesReclaimed": 0,
                "originalOuterStatusRewritten": False,
                "qualityApproved": False,
            },
        )
    return output


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("run", "verify"))
    parser.add_argument("request", type=Path)
    parser.add_argument("--sha256", required=True)
    args = parser.parse_args()
    print(execute(args.request, sha256=args.sha256, verify=args.mode == "verify"))


if __name__ == "__main__":
    main()
