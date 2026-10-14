"""Bounded, metadata-only inventory of explicitly selected checkout directories."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import stat
import sys
from collections.abc import Iterator, Sequence
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path

from replay.disk_budget import BLOCK_BYTES

CHECKOUT = Path(__file__).resolve().parents[2]
PROTECTED = frozenset({"private", "downloads", ".git", "models"})
MAX_PLAN_BYTES = 1024 * 1024
MAX_ROOTS = 1024
DEFAULT_MAX_ENTRIES = 1_000_000
DEFAULT_MAX_METADATA_BYTES = 128 * 1024 * 1024
DEFAULT_MAX_DEPTH = 128


@dataclass(frozen=True)
class Limits:
    max_entries: int = DEFAULT_MAX_ENTRIES
    max_metadata_bytes: int = DEFAULT_MAX_METADATA_BYTES
    max_depth: int = DEFAULT_MAX_DEPTH

    def validate(self) -> None:
        values = (self.max_entries, self.max_metadata_bytes, self.max_depth)
        if any(type(value) is not int or value <= 0 for value in values):
            raise ValueError("Inventory limits must be positive integers")


@dataclass(frozen=True)
class Identity:
    dev: int
    inode: int
    mode: int
    nlink: int
    size: int
    mtime_ns: int
    ctime_ns: int
    blocks: int

    @classmethod
    def of(cls, info: os.stat_result) -> Identity:
        return cls(
            info.st_dev,
            info.st_ino,
            info.st_mode,
            info.st_nlink,
            info.st_size,
            info.st_mtime_ns,
            info.st_ctime_ns,
            info.st_blocks,
        )


@dataclass(frozen=True)
class Plan:
    path: Path
    sha256: str
    identity: Identity
    checkout: Path
    roots: tuple[str, ...]


@dataclass(frozen=True)
class Leaf:
    path: str
    identity: Identity
    link_target: str | None

    def json(self) -> dict[str, object]:
        info = self.identity
        regular = stat.S_ISREG(info.mode)
        special = not regular and self.link_target is None
        return {
            "path": self.path,
            "kind": "symlink" if self.link_target is not None else "file" if regular else "special",
            "classification": "hold",
            "mode": info.mode,
            "dev": info.dev,
            "inode": info.inode,
            "nlink": info.nlink,
            "bytes": info.size if regular else 0,
            "statBytes": info.size,
            "linkBytes": info.size if self.link_target is not None else 0,
            "mtimeNs": info.mtime_ns,
            "ctimeNs": info.ctime_ns,
            "blocks512": info.blocks,
            "allocatedBytes": info.blocks * BLOCK_BYTES,
            "linkTarget": self.link_target,
            "specialType": _special_type(info.mode) if special else None,
        }


def _special_type(mode: int) -> str:
    if stat.S_ISFIFO(mode):
        return "fifo"
    if stat.S_ISSOCK(mode):
        return "socket"
    if stat.S_ISCHR(mode):
        return "character"
    if stat.S_ISBLK(mode):
        return "block"
    return "unknown"


def _relative(value: object) -> str:
    if not isinstance(value, str) or not value or "\x00" in value:
        raise ValueError("Inventory roots must be nonempty relative paths")
    path = Path(value)
    if (
        path.is_absolute()
        or path.as_posix() != value
        or any(part in {".", "..", ""} for part in value.split("/"))
    ):
        raise ValueError("Inventory roots must be canonical checkout-relative paths")
    if PROTECTED.intersection(path.parts):
        raise ValueError("Inventory roots cannot include protected data or models")
    return value


def _directory(path: Path, checkout: Path) -> Identity:
    try:
        relative = path.relative_to(checkout)
    except ValueError as error:
        raise ValueError("Path escapes checkout") from error
    current = checkout
    for part in ("", *relative.parts):
        if part:
            current = current / part
        info = current.lstat()
        if not stat.S_ISDIR(info.st_mode):
            raise ValueError(f"Directory ancestor is missing, linked or not a directory: {current}")
    if path.resolve() != path:
        raise ValueError("Directory ancestors must be canonical and unlinked")
    return Identity.of(info)


@contextmanager
def _directory_fd(path: Path, checkout: Path) -> Iterator[int]:
    """Open each checkout-relative ancestor separately, without following links."""
    expected = _directory(path, checkout)
    descriptor = os.open(checkout, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW)
    try:
        for part in path.relative_to(checkout).parts:
            child = os.open(part, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW, dir_fd=descriptor)
            os.close(descriptor)
            descriptor = child
        if Identity.of(os.fstat(descriptor)) != expected:
            raise ValueError("Directory identity changed while opening ancestors")
        yield descriptor
    finally:
        os.close(descriptor)


def _read_plan(path: Path, checkout: Path) -> tuple[bytes, Identity]:
    with (
        _directory_fd(path.parent, checkout) as parent,
        os.fdopen(os.open(path.name, os.O_RDONLY | os.O_NOFOLLOW, dir_fd=parent), "rb") as stream,
    ):
        before = Identity.of(os.fstat(stream.fileno()))
        if not stat.S_ISREG(before.mode) or before.size > MAX_PLAN_BYTES:
            raise ValueError("Inventory plan must be a bounded regular file")
        data = stream.read(MAX_PLAN_BYTES + 1)
        if len(data) > MAX_PLAN_BYTES:
            raise ValueError("Inventory plan exceeds its byte budget")
        if Identity.of(os.fstat(stream.fileno())) != before or Identity.of(path.lstat()) != before:
            raise ValueError("Inventory plan changed during reading")
    return data, before


def load_plan(path: Path, *, checkout: Path = CHECKOUT) -> Plan:
    """Validate a small exact-root plan without reading any leaf contents."""
    checkout = Path(os.path.abspath(checkout))
    path = Path(os.path.abspath(path))
    data, identity = _read_plan(path, checkout)
    raw: object = json.loads(data)
    if not isinstance(raw, dict) or set(raw) != {"version", "roots"}:
        raise ValueError("Inventory plan must contain only version and roots")
    if type(raw["version"]) is not int or raw["version"] != 1:
        raise ValueError("Unsupported inventory plan version")
    values = raw["roots"]
    if not isinstance(values, list) or not 1 <= len(values) <= MAX_ROOTS:
        raise ValueError("Inventory plan requires a bounded nonempty root list")
    roots = tuple(_relative(value) for value in values)
    if len(set(roots)) != len(roots):
        raise ValueError("Inventory roots must be distinct")
    for index, root in enumerate(roots):
        if any(
            root.startswith(other + "/") or other.startswith(root + "/") for other in roots[:index]
        ):
            raise ValueError("Inventory roots must not overlap")
        _directory(checkout / root, checkout)
    return Plan(path, hashlib.sha256(data).hexdigest(), identity, checkout, roots)


class _Walk:
    def __init__(self, plan: Plan, limits: Limits) -> None:
        self.plan = plan
        self.limits = limits
        self.leaves: list[Leaf] = []
        self.directories: dict[Path, Identity] = {}
        self.by_directory: dict[Path, list[Leaf]] = {}
        self.excluded: list[str] = []
        self.entries = 0
        self.metadata_bytes = 4096 + sum(len(root.encode()) for root in plan.roots)

    def _budget(self, value: object) -> None:
        self.metadata_bytes += len(json.dumps(value, ensure_ascii=True).encode()) + 2
        if self.metadata_bytes > self.limits.max_metadata_bytes:
            raise ValueError("Inventory metadata exceeds its byte budget")

    def directory(self, path: Path, depth: int = 0) -> None:
        if depth > self.limits.max_depth:
            raise ValueError("Inventory exceeds its directory depth budget")
        before = _directory(path, self.plan.checkout)
        with _directory_fd(path, self.plan.checkout) as descriptor:
            if Identity.of(os.fstat(descriptor)) != before:
                raise ValueError("Inventory directory changed before scanning")
            self.directories[path] = before
            parent_leaves: list[Leaf] = []
            self.by_directory[path] = parent_leaves
            with os.scandir(descriptor) as listing:
                for entry in listing:
                    self.entries += 1
                    if self.entries > self.limits.max_entries:
                        raise ValueError("Inventory exceeds its entry budget")
                    child = path / entry.name
                    relative = child.relative_to(self.plan.checkout).as_posix()
                    info = Identity.of(entry.stat(follow_symlinks=False))
                    if stat.S_ISDIR(info.mode):
                        if entry.name in PROTECTED:
                            self._budget(relative)
                            self.excluded.append(relative)
                        else:
                            self.directory(child, depth + 1)
                    else:
                        target = (
                            os.readlink(entry.name, dir_fd=descriptor)
                            if stat.S_ISLNK(info.mode)
                            else None
                        )
                        if (
                            Identity.of(
                                os.stat(entry.name, dir_fd=descriptor, follow_symlinks=False)
                            )
                            != info
                        ):
                            raise ValueError("Inventory leaf changed during observation")
                        leaf = Leaf(relative, info, target)
                        self._budget(leaf.json())
                        self.leaves.append(leaf)
                        parent_leaves.append(leaf)
            if (
                Identity.of(os.fstat(descriptor)) != before
                or _directory(path, self.plan.checkout) != before
            ):
                raise ValueError("Inventory directory changed during scanning")

    def finish(self) -> None:
        self._directories_current()
        for parent, leaves in self.by_directory.items():
            if not leaves:
                continue
            with _directory_fd(parent, self.plan.checkout) as descriptor:
                for leaf in leaves:
                    name = Path(leaf.path).name
                    info = os.stat(name, dir_fd=descriptor, follow_symlinks=False)
                    if Identity.of(info) != leaf.identity:
                        raise ValueError("Inventory leaf changed before publication")
                    if (
                        leaf.link_target is not None
                        and os.readlink(name, dir_fd=descriptor) != leaf.link_target
                    ):
                        raise ValueError("Inventory link changed before publication")
                if (
                    Identity.of(os.fstat(descriptor)) != self.directories[parent]
                    or _directory(parent, self.plan.checkout) != self.directories[parent]
                ):
                    raise ValueError("Inventory parent changed during terminal leaf checks")
        self._directories_current()

    def _directories_current(self) -> None:
        for path, identity in self.directories.items():
            if _directory(path, self.plan.checkout) != identity:
                raise ValueError("Inventory directory membership changed before publication")


def _totals(leaves: Sequence[Leaf]) -> dict[str, int]:
    unique: dict[tuple[int, int], Identity] = {}
    logical = 0
    allocated = 0
    files = 0
    links = 0
    special = 0
    for leaf in leaves:
        info = leaf.identity
        allocated += info.blocks * BLOCK_BYTES
        if stat.S_ISREG(info.mode):
            files += 1
            logical += info.size
        elif leaf.link_target is not None:
            links += 1
        else:
            special += 1
        key = (info.dev, info.inode)
        if key in unique and unique[key] != info:
            raise ValueError("Conflicting observations for one inode")
        unique[key] = info
    unique_logical = sum(info.size for info in unique.values() if stat.S_ISREG(info.mode))
    return {
        "leaves": len(leaves),
        "files": files,
        "symlinks": links,
        "special": special,
        "uniqueInodes": len(unique),
        "logicalBytes": logical,
        "uniqueLogicalBytes": unique_logical,
        "duplicateLogicalBytes": logical - unique_logical,
        "allocatedBytes": allocated,
        "uniqueAllocatedBytes": sum(info.blocks * BLOCK_BYTES for info in unique.values()),
    }


def _output_parent(output: Path, plan: Plan) -> Identity:
    relative = output.relative_to(plan.checkout)
    if relative.parts[:2] != ("edits", "work") or PROTECTED.intersection(relative.parts):
        raise ValueError("Inventory output must be under unprotected edits/work")
    if any(
        relative.as_posix() == root or relative.as_posix().startswith(root + "/")
        for root in plan.roots
    ):
        raise ValueError("Inventory output must be outside inventoried roots")
    if output == plan.path:
        raise ValueError("Inventory output cannot replace its plan")
    if os.path.lexists(output):
        raise FileExistsError("Inventory output already exists")
    return _directory(output.parent, plan.checkout)


def _same_directory(left: Identity, right: Identity) -> bool:
    return (left.dev, left.inode, left.mode) == (right.dev, right.inode, right.mode)


def _chunks(payload: dict[str, object], leaves: Sequence[Leaf]) -> Iterator[bytes]:
    header = json.dumps(payload, ensure_ascii=True, indent=2)
    yield (header[:-2] + ',\n  "rows": [').encode()
    for index, leaf in enumerate(leaves):
        prefix = "\n" if index == 0 else ",\n"
        row = json.dumps(leaf.json(), ensure_ascii=True, indent=2)
        yield (prefix + "\n".join("    " + line for line in row.splitlines())).encode()
    yield b"\n  ]\n}\n"


def inventory(plan: Plan, output: Path, *, limits: Limits | None = None) -> dict[str, int]:
    """Publish an immutable private inventory; every observed leaf remains on hold."""
    limits = Limits() if limits is None else limits
    limits.validate()
    if load_plan(plan.path, checkout=plan.checkout) != plan:
        raise ValueError("Inventory plan no longer matches its validated request")
    output = Path(os.path.abspath(output))
    parent = _output_parent(output, plan)
    walk = _Walk(plan, limits)
    for root in plan.roots:
        walk.directory(plan.checkout / root)
    walk.finish()
    if load_plan(plan.path, checkout=plan.checkout) != plan:
        raise ValueError("Inventory plan changed before publication")
    totals = _totals(walk.leaves)
    payload = {
        "version": 1,
        "kind": "metadata-only-leaf-inventory",
        "plan": {"path": plan.path.relative_to(plan.checkout).as_posix(), "sha256": plan.sha256},
        "roots": list(plan.roots),
        "classification": "hold",
        "mediaContentsRead": False,
        "limits": limits.__dict__,
        "entriesObserved": walk.entries,
        "directoriesObserved": len(walk.directories),
        "excludedDirectories": sorted(walk.excluded),
        "totals": totals,
        "allocationNote": (
            "Inode deduplicated st_blocks; shared APFS extents are not measured free space."
        ),
    }
    ordered = sorted(walk.leaves, key=lambda leaf: leaf.path)
    if sum(len(chunk) for chunk in _chunks(payload, ordered)) > limits.max_metadata_bytes:
        raise ValueError("Serialized inventory exceeds its metadata byte budget")
    with _directory_fd(output.parent, plan.checkout) as descriptor:
        if not _same_directory(Identity.of(os.fstat(descriptor)), parent) or not _same_directory(
            _directory(output.parent, plan.checkout), parent
        ):
            raise ValueError("Inventory output ancestor changed")
        with os.fdopen(
            os.open(
                output.name,
                os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW,
                0o600,
                dir_fd=descriptor,
            ),
            "wb",
        ) as stream:
            for chunk in _chunks(payload, ordered):
                stream.write(chunk)
            stream.flush()
            os.fsync(stream.fileno())
        os.fsync(descriptor)
    return totals


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("plan", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--max-entries", type=int, default=DEFAULT_MAX_ENTRIES)
    parser.add_argument("--max-metadata-bytes", type=int, default=DEFAULT_MAX_METADATA_BYTES)
    parser.add_argument("--max-depth", type=int, default=DEFAULT_MAX_DEPTH)
    args = parser.parse_args(argv)
    try:
        totals = inventory(
            load_plan(args.plan),
            args.output,
            limits=Limits(args.max_entries, args.max_metadata_bytes, args.max_depth),
        )
    except (OSError, ValueError) as error:
        print(f"Leaf inventory failed: {error}", file=sys.stderr)
        return 1
    print(json.dumps(totals, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
