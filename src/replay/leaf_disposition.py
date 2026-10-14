"""Classify pinned leaf metadata using an explicit, independently approved policy.

No leaf is opened, hashed, discovered or removed. Eligibility is an owner decision;
the retirement consumer must still check live identities, references and handles.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import stat
import sys
from collections import Counter
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import TypeGuard

CHECKOUT = Path(__file__).resolve().parents[2]
MAX_JSON_BYTES = 128 * 1024 * 1024
PROTECTED = frozenset({"private", "downloads", ".git", "models"})
METADATA_SUFFIXES = frozenset({".json", ".jsonl", ".log", ".status", ".sha256", ".txt", ".md"})
POLICY_KEYS = frozenset(
    {
        "version",
        "owner",
        "eligibleTrees",
        "eligiblePaths",
        "keepTrees",
        "keepPaths",
        "approval",
        "terminalProofs",
    }
)
ROW_KEYS = frozenset(
    {
        "path",
        "kind",
        "classification",
        "dev",
        "inode",
        "nlink",
        "bytes",
        "mtimeNs",
        "ctimeNs",
        "linkTarget",
        "linkBytes",
        "blocks512",
        "allocatedBytes",
        "mode",
        "statBytes",
        "specialType",
    }
)


@dataclass(frozen=True, slots=True)
class Reference:
    path: Path
    sha256: str


@dataclass(frozen=True, slots=True)
class Policy:
    owner: str
    eligible_trees: tuple[Path, ...] = ()
    eligible_paths: tuple[Path, ...] = ()
    keep_trees: tuple[Path, ...] = ()
    keep_paths: tuple[Path, ...] = ()
    approval: Reference | None = None
    terminal_proofs: tuple[Reference, ...] = ()


@dataclass(frozen=True, slots=True)
class Summary:
    rows: int
    eligible: int
    keep: int
    hold: int
    eligible_logical_bytes: int
    eligible_allocated_bytes: int

    def json(self) -> dict[str, int]:
        return {
            "rows": self.rows,
            "eligible": self.eligible,
            "keep": self.keep,
            "hold": self.hold,
            "eligibleLogicalBytes": self.eligible_logical_bytes,
            "eligibleAllocatedBytes": self.eligible_allocated_bytes,
        }


@dataclass(frozen=True, slots=True)
class _Document:
    reference: Reference
    identity: tuple[int, ...]
    data: bytes


@dataclass(frozen=True, slots=True)
class _DecisionIndex:
    eligible_trees: frozenset[Path]
    eligible_paths: frozenset[Path]
    keep_trees: frozenset[Path]
    keep_paths: frozenset[Path]


def _string_object(value: object) -> TypeGuard[dict[str, object]]:
    return isinstance(value, dict) and all(isinstance(key, str) for key in value)


def _object(value: object) -> dict[str, object]:
    if not _string_object(value):
        raise ValueError("Metadata value must be an object with string keys")
    return value


def _pairs(items: list[tuple[str, object]]) -> dict[str, object]:
    result: dict[str, object] = {}
    for key, value in items:
        if key in result:
            raise ValueError("Duplicate JSON object key")
        result[key] = value
    return result


def _json(data: bytes) -> dict[str, object]:
    value: object = json.loads(data, object_pairs_hook=_pairs)
    return _object(value)


def _integer(value: object, name: str, *, minimum: int | None = 0) -> int:
    if type(value) is not int or (minimum is not None and value < minimum):
        raise ValueError(f"{name} must be an exact integer in range")
    return value


def _digest(value: object) -> str:
    if not isinstance(value, str) or re.fullmatch("[0-9a-f]{64}", value) is None:
        raise ValueError("SHA-256 must be 64 lowercase hexadecimal characters")
    return value


def _relative_text(value: object) -> Path:
    if (
        not isinstance(value, str)
        or not value
        or any(character in value for character in "\x00\r\n")
    ):
        raise ValueError("Paths must be nonempty canonical checkout-relative strings")
    path = Path(value)
    if (
        path.is_absolute()
        or path.as_posix() != value
        or any(part in {"", ".", ".."} for part in value.split("/"))
    ):
        raise ValueError("Paths must be canonical checkout-relative strings")
    return path


def _relative(path: Path, checkout: Path) -> Path:
    if ".." in path.parts:
        raise ValueError("Parent traversal is forbidden")
    absolute = path if path.is_absolute() else checkout / path
    try:
        relative = absolute.relative_to(checkout)
    except ValueError as error:
        raise ValueError("Path escapes checkout") from error
    return _relative_text(relative.as_posix())


def _directory(path: Path, checkout: Path) -> tuple[int, int, int]:
    relative = path.relative_to(checkout)
    current = checkout
    for part in ("", *relative.parts):
        if part:
            current = current / part
        info = current.lstat()
        if not stat.S_ISDIR(info.st_mode):
            raise ValueError("Directory ancestor is missing, linked or not a directory")
    return info.st_dev, info.st_ino, info.st_mode


def _identity(info: os.stat_result) -> tuple[int, ...]:
    return (
        info.st_dev,
        info.st_ino,
        info.st_mode,
        info.st_nlink,
        info.st_size,
        info.st_mtime_ns,
        info.st_ctime_ns,
    )


def _checkout(path: Path | None) -> Path:
    root = CHECKOUT if path is None else path.absolute()
    if root.resolve(strict=True) != root:
        raise ValueError("Checkout must be an existing canonical directory without aliases")
    _directory(root, root)
    return root


def _read(reference: Reference, *, checkout: Path, limit: int) -> _Document:
    return _read_path(
        reference.path,
        checkout=checkout,
        limit=limit,
        expected_sha256=_digest(reference.sha256),
    )


def _read_path(
    input_path: Path, *, checkout: Path, limit: int, expected_sha256: str | None
) -> _Document:
    relative = _relative(input_path, checkout)
    if PROTECTED.intersection(relative.parts) or relative.suffix not in METADATA_SUFFIXES:
        raise ValueError("Only declared unprotected metadata references may be read")
    path = checkout / relative
    _directory(path.parent, checkout)
    initial = _identity(path.lstat())
    if not stat.S_ISREG(initial[2]) or initial[4] > limit:
        raise ValueError("Metadata input must be a bounded regular file")
    with os.fdopen(os.open(path, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK), "rb") as stream:
        before = _identity(os.fstat(stream.fileno()))
        if before != initial:
            raise ValueError("Metadata input identity changed before reading")
        data = stream.read(limit + 1)
        if len(data) > limit:
            raise ValueError("Metadata input exceeds its byte budget")
        if _identity(os.fstat(stream.fileno())) != before or _identity(path.lstat()) != before:
            raise ValueError("Metadata input changed during reading")
    digest = hashlib.sha256(data).hexdigest()
    if expected_sha256 is not None and digest != expected_sha256:
        raise ValueError("Metadata input differs from pinned SHA-256")
    return _Document(Reference(path, digest), before, data)


def _leaf_metadata(document: _Document) -> None:
    raw = _json(document.data)
    if (
        _integer(raw.get("version"), "version") != 1
        or raw.get("kind") not in ("metadata-only-leaf-inventory", "owner-policy-leaf-disposition")
        or raw.get("mediaContentsRead") is not False
        or not isinstance(raw.get("rows"), list)
    ):
        raise ValueError("Large metadata references are limited to leaf inventory/disposition")


def metadata_reference(path: Path, *, checkout: Path | None = None) -> Reference:
    """Bind newly generated leaf metadata, with a domain-specific 128MiB ceiling."""
    document = _read_path(
        path, checkout=_checkout(checkout), limit=MAX_JSON_BYTES, expected_sha256=None
    )
    _leaf_metadata(document)
    return document.reference


def verify_reference(reference: Reference, *, checkout: Path | None = None) -> None:
    """Verify pinned leaf metadata without widening other domains' read budgets."""
    document = _read(reference, checkout=_checkout(checkout), limit=MAX_JSON_BYTES)
    _leaf_metadata(document)


def _verify(documents: Sequence[_Document], *, checkout: Path, limit: int) -> None:
    for document in documents:
        current = _read(document.reference, checkout=checkout, limit=limit)
        if current.identity != document.identity:
            raise ValueError("Metadata input identity changed before publication")


def _reference_json(reference: Reference, checkout: Path) -> dict[str, str]:
    return {
        "path": _relative(reference.path, checkout).as_posix(),
        "sha256": _digest(reference.sha256),
    }


def _reference(value: object, checkout: Path) -> Reference:
    raw = _object(value)
    if set(raw) != {"path", "sha256"}:
        raise ValueError("Metadata reference must contain only path and sha256")
    return Reference(checkout / _relative_text(raw["path"]), _digest(raw["sha256"]))


def _paths(value: object) -> tuple[Path, ...]:
    if not isinstance(value, list):
        raise ValueError("Policy path collections must be arrays")
    paths = tuple(_relative_text(item) for item in value)
    if len(set(paths)) != len(paths):
        raise ValueError("Policy paths must be distinct")
    return paths


def policy_document(policy: Policy, *, checkout: Path | None = None) -> dict[str, object]:
    """Represent explicit owner intent; this function never infers eligibility."""
    root = _checkout(checkout)
    if not isinstance(policy.owner, str) or not policy.owner.strip():
        raise ValueError("Policy requires a nonempty owner")
    if (policy.eligible_trees or policy.eligible_paths) and (
        policy.approval is None or not policy.terminal_proofs
    ):
        raise ValueError("Eligibility requires approval and terminal proof references")
    fields = {
        "eligibleTrees": policy.eligible_trees,
        "eligiblePaths": policy.eligible_paths,
        "keepTrees": policy.keep_trees,
        "keepPaths": policy.keep_paths,
    }
    document: dict[str, object] = {"version": 1, "owner": policy.owner}
    for key, paths in fields.items():
        values = [_relative(path, root).as_posix() for path in paths]
        _paths(values)
        document[key] = values
    document["approval"] = (
        None if policy.approval is None else _reference_json(policy.approval, root)
    )
    document["terminalProofs"] = [
        _reference_json(reference, root) for reference in policy.terminal_proofs
    ]
    return document


def _policy(raw: dict[str, object], checkout: Path) -> Policy:
    if set(raw) != POLICY_KEYS or _integer(raw["version"], "version") != 1:
        raise ValueError("Unsupported or malformed policy schema")
    owner = raw["owner"]
    proofs = raw["terminalProofs"]
    if not isinstance(owner, str) or not isinstance(proofs, list):
        raise ValueError("Policy owner and terminal proofs are malformed")
    policy = Policy(
        owner=owner,
        eligible_trees=_paths(raw["eligibleTrees"]),
        eligible_paths=_paths(raw["eligiblePaths"]),
        keep_trees=_paths(raw["keepTrees"]),
        keep_paths=_paths(raw["keepPaths"]),
        approval=None if raw["approval"] is None else _reference(raw["approval"], checkout),
        terminal_proofs=tuple(_reference(item, checkout) for item in proofs),
    )
    policy_document(policy, checkout=checkout)
    return policy


def _proofs(policy: Policy, *, checkout: Path, limit: int) -> list[_Document]:
    references = policy.terminal_proofs
    if policy.approval is not None:
        references = (policy.approval, *references)
    return [_read(reference, checkout=checkout, limit=limit) for reference in references]


def _publish(
    payload: dict[str, object],
    output: Path,
    *,
    checkout: Path,
    documents: Sequence[_Document],
    limit: int,
) -> Reference:
    relative = _relative(output, checkout)
    if relative.parts[:2] != ("edits", "work") or PROTECTED.intersection(relative.parts):
        raise ValueError("Output must be under unprotected checkout edits/work")
    path = checkout / relative
    if any(path == document.reference.path for document in documents):
        raise ValueError("Output cannot replace an input")
    before = _directory(path.parent, checkout)
    data = (json.dumps(payload, ensure_ascii=True, indent=2) + "\n").encode()
    if len(data) > limit:
        raise ValueError("Disposition output exceeds its byte budget")
    _verify(documents, checkout=checkout, limit=limit)
    descriptor = os.open(path.parent, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW)
    try:
        info = os.fstat(descriptor)
        if (info.st_dev, info.st_ino, info.st_mode) != before or _directory(
            path.parent, checkout
        ) != before:
            raise ValueError("Output directory identity changed")
        with os.fdopen(
            os.open(
                path.name,
                os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW,
                0o600,
                dir_fd=descriptor,
            ),
            "wb",
        ) as stream:
            stream.write(data)
            stream.flush()
            os.fsync(stream.fileno())
        _verify(documents, checkout=checkout, limit=limit)
        if _directory(path.parent, checkout) != before:
            raise ValueError("Output directory identity changed during publication")
        os.fsync(descriptor)
    finally:
        os.close(descriptor)
    return Reference(path, hashlib.sha256(data).hexdigest())


def write_policy(policy: Policy, output: Path, *, checkout: Path | None = None) -> Reference:
    """Publish a new explicit policy, retaining bound approval and terminal proofs."""
    root = _checkout(checkout)
    document = policy_document(policy, checkout=root)
    return _publish(
        document,
        output,
        checkout=root,
        documents=_proofs(policy, checkout=root, limit=MAX_JSON_BYTES),
        limit=MAX_JSON_BYTES,
    )


def _inside(path: Path, trees: Sequence[Path]) -> bool:
    return any(path.is_relative_to(tree) for tree in trees)


def _indexed_inside(path: Path, trees: frozenset[Path]) -> bool:
    """Test the original tree union without scanning unrelated policy siblings."""
    return path in trees or any(parent in trees for parent in path.parents)


def _row(value: object, roots: Sequence[Path]) -> dict[str, object]:
    raw = _object(value)
    if set(raw) != ROW_KEYS:
        raise ValueError("Inventory row schema is unsupported")
    path = _relative_text(raw["path"])
    if not _inside(path, roots) or path in roots:
        raise ValueError("Inventory leaf is outside the exact declared roots")
    if raw["kind"] not in ("file", "symlink", "special") or raw["classification"] != "hold":
        raise ValueError("Inventory must contain unclassified known leaf kinds")
    for key in (
        "dev",
        "inode",
        "nlink",
        "bytes",
        "linkBytes",
        "blocks512",
        "allocatedBytes",
        "mode",
        "statBytes",
    ):
        _integer(raw[key], key)
    for key in ("mtimeNs", "ctimeNs"):
        _integer(raw[key], key, minimum=None)
    blocks = _integer(raw["blocks512"], "blocks512")
    if raw["allocatedBytes"] != blocks * 512 or raw["inode"] == 0 or raw["nlink"] == 0:
        raise ValueError("Inventory allocation or identity is inconsistent")
    target = raw["linkTarget"]
    mode = _integer(raw["mode"], "mode")
    special = raw["specialType"]
    if raw["kind"] == "symlink":
        if (
            not stat.S_ISLNK(mode)
            or not isinstance(target, str)
            or raw["bytes"] != 0
            or raw["linkBytes"] != raw["statBytes"]
            or special is not None
        ):
            raise ValueError("Symlink metadata is malformed")
    elif target is not None or raw["linkBytes"] != 0:
        raise ValueError("Regular file metadata is malformed")
    elif raw["kind"] == "file":
        if not stat.S_ISREG(mode) or raw["bytes"] != raw["statBytes"] or special is not None:
            raise ValueError("Regular file mode or size metadata is inconsistent")
    else:
        expected_type = "unknown"
        for predicate, name in (
            (stat.S_ISFIFO, "fifo"),
            (stat.S_ISSOCK, "socket"),
            (stat.S_ISCHR, "character"),
            (stat.S_ISBLK, "block"),
        ):
            if predicate(mode):
                expected_type = name
        if (
            stat.S_ISREG(mode)
            or stat.S_ISLNK(mode)
            or stat.S_ISDIR(mode)
            or raw["bytes"] != 0
            or special != expected_type
        ):
            raise ValueError("Special file metadata is malformed")
    return raw


def _classification(
    path: Path, row: dict[str, object], index: _DecisionIndex, duplicates: set[tuple[int, int]]
) -> tuple[str, str]:
    if row["kind"] == "special":
        return "hold", "special file is never eligible"
    if path in index.keep_paths or _indexed_inside(path, index.keep_trees):
        return "keep", "explicit owner preservation"
    if PROTECTED.intersection(path.parts):
        return "hold", "protected path component"
    if row["kind"] != "file" or row["nlink"] != 1:
        return "hold", "linked or uncertain leaf"
    identity = (_integer(row["dev"], "dev"), _integer(row["inode"], "inode"))
    if identity in duplicates:
        return "hold", "duplicate inode observations"
    if path in index.eligible_paths or _indexed_inside(path, index.eligible_trees):
        return "eligible", "explicit approved owner policy"
    return "hold", "no explicit eligibility decision"


def _inventory(document: _Document) -> tuple[tuple[Path, ...], list[dict[str, object]]]:
    raw = _json(document.data)
    if (
        _integer(raw.get("version"), "version") != 1
        or raw.get("kind") != "metadata-only-leaf-inventory"
        or raw.get("classification") != "hold"
        or raw.get("mediaContentsRead") is not False
    ):
        raise ValueError("Unsupported metadata-only inventory contract")
    roots = _paths(raw.get("roots"))
    if not roots or any(PROTECTED.intersection(path.parts) for path in roots):
        raise ValueError("Inventory roots are empty or protected")
    for index, root in enumerate(roots):
        if _inside(root, roots[:index]) or any(
            other.is_relative_to(root) for other in roots[:index]
        ):
            raise ValueError("Inventory roots overlap")
    values = raw.get("rows")
    if not isinstance(values, list):
        raise ValueError("Inventory needs a rows array")
    rows = [_row(value, roots) for value in values]
    if len({row["path"] for row in rows}) != len(rows):
        raise ValueError("Duplicate inventory leaf path")
    return roots, rows


def build_disposition(
    inventory: Reference,
    policy: Reference,
    output: Path,
    *,
    checkout: Path | None = None,
    max_json_bytes: int = MAX_JSON_BYTES,
) -> Summary:
    """Write a pinned, non-overwriting disposition; never inspect leaf contents."""
    limit = _integer(max_json_bytes, "max_json_bytes", minimum=1)
    if limit > MAX_JSON_BYTES:
        raise ValueError("JSON limit cannot exceed the fixed 128MiB ceiling")
    root = _checkout(checkout)
    source = _read(inventory, checkout=root, limit=limit)
    owner = _read(policy, checkout=root, limit=limit)
    roots, rows = _inventory(source)
    intent = _policy(_json(owner.data), root)
    relative_output = _relative(output, root)
    if _inside(relative_output, roots):
        raise ValueError("Disposition output must be outside inventoried roots")
    documents = [source, owner, *_proofs(intent, checkout=root, limit=limit)]
    index = _DecisionIndex(
        eligible_trees=frozenset(intent.eligible_trees),
        eligible_paths=frozenset(intent.eligible_paths),
        keep_trees=frozenset(intent.keep_trees),
        keep_paths=frozenset(intent.keep_paths),
    )
    counts = Counter((_integer(row["dev"], "dev"), _integer(row["inode"], "inode")) for row in rows)
    duplicates = {identity for identity, count in counts.items() if count > 1}
    output_rows: list[dict[str, object]] = []
    totals: Counter[str] = Counter()
    for row in rows:
        path = _relative_text(row["path"])
        classification, reason = _classification(path, row, index, duplicates)
        output_rows.append(
            dict(row, path=str(root / path), classification=classification, reason=reason)
        )
        totals[classification] += 1
        if classification == "eligible":
            totals["logical"] += _integer(row["bytes"], "bytes")
            totals["allocated"] += _integer(row["allocatedBytes"], "allocatedBytes")
    summary = Summary(
        len(rows),
        totals["eligible"],
        totals["keep"],
        totals["hold"],
        totals["logical"],
        totals["allocated"],
    )
    payload: dict[str, object] = {
        "version": 1,
        "kind": "owner-policy-leaf-disposition",
        "inventory": _reference_json(source.reference, root),
        "policy": _reference_json(owner.reference, root),
        "ownerPolicy": policy_document(intent, checkout=root),
        "roots": [str(root / path) for path in roots],
        "mediaContentsRead": False,
        "leavesRemoved": 0,
        "allocationNote": "Metadata allocation is not an observed free-space gain.",
        "totals": summary.json(),
        "rows": output_rows,
    }
    _publish(payload, output, checkout=root, documents=documents, limit=limit)
    return summary


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("inventory", type=Path)
    parser.add_argument("policy", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--inventory-sha256", required=True)
    parser.add_argument("--policy-sha256", required=True)
    args = parser.parse_args(argv)
    try:
        summary = build_disposition(
            Reference(args.inventory, args.inventory_sha256),
            Reference(args.policy, args.policy_sha256),
            args.output,
        )
    except (OSError, ValueError) as error:
        print(f"Leaf disposition failed: {error}", file=sys.stderr)
        return 1
    print(json.dumps(summary.json(), sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
