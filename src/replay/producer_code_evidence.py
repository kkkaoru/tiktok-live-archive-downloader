"""Read pinned historical code bytes without executing or substituting data.

Domain consumers validate the returned binding fields against their immutable
artifacts. This helper only permits explicitly allowlisted Python producer roles.
"""

import hashlib
import json
import os
import re
import stat
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

from replay.recordings import is_record

MAX_BYTES = 1_048_576
MAX_ROLES = 256
RESERVED_FIELDS = frozenset({"version", "protocol", "archives"})


@dataclass(frozen=True, slots=True)
class CodeArchive:
    path: Path
    sha256: str


@dataclass(frozen=True, slots=True)
class CodeEvidence:
    archives: dict[str, CodeArchive]
    document: dict[str, object]
    inputs: dict[str, str]


def _unique_object(pairs: list[tuple[str, object]]) -> dict[str, object]:
    if len({key for key, _ in pairs}) != len(pairs):
        raise ValueError("Producer evidence JSON keys must be unique")
    return dict(pairs)


def _record(value: object) -> dict[str, object]:
    if not is_record(value):
        raise ValueError("Producer evidence requires JSON objects")
    return value


def _digest(value: object) -> str:
    if not isinstance(value, str) or re.fullmatch(r"[a-f0-9]{64}", value) is None:
        raise ValueError("Producer code evidence requires explicit SHA256 pins")
    return value


def _path(value: object, *, root: Path) -> Path:
    if (
        not isinstance(value, str)
        or not value
        or len(value) > 4096
        or any(char in value for char in "\0\r\n")
    ):
        raise ValueError("Producer evidence requires canonical absolute paths")
    path = Path(value)
    if (
        not path.is_absolute()
        or str(path) != value
        or ".." in path.parts
        or path.resolve() != path
        or not path.is_relative_to(root)
    ):
        raise ValueError("Producer evidence requires canonical repository paths")
    return path


def _bound_bytes(path: Path, digest: str) -> bytes:
    expected = _digest(digest)
    descriptor = os.open(path, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK)
    try:
        before = os.fstat(descriptor)
        if not stat.S_ISREG(before.st_mode) or not 0 < before.st_size <= MAX_BYTES:
            raise ValueError("Producer evidence must be a bounded nonempty regular file")
        with os.fdopen(descriptor, "rb", closefd=False) as stream:
            data = stream.read(MAX_BYTES + 1)
        after = os.fstat(descriptor)
        named = path.lstat()
    finally:
        os.close(descriptor)
    identity = (
        before.st_dev,
        before.st_ino,
        before.st_size,
        before.st_mtime_ns,
        before.st_ctime_ns,
    )
    if (
        not data
        or len(data) > MAX_BYTES
        or any(
            identity != (info.st_dev, info.st_ino, info.st_size, info.st_mtime_ns, info.st_ctime_ns)
            for info in (after, named)
        )
    ):
        raise ValueError("Producer evidence changed during reading")
    if hashlib.sha256(data).hexdigest() != expected:
        raise ValueError("Pinned producer evidence bytes changed")
    return data


def _roles(source_names: Sequence[str], *, root: Path) -> frozenset[str]:
    if not 1 <= len(source_names) <= MAX_ROLES or len(set(source_names)) != len(source_names):
        raise ValueError("Producer code roles must be a bounded unique nonempty allowlist")
    if any(re.fullmatch(r"[a-z_][a-z0-9_]{0,127}\.py", name) is None for name in source_names):
        raise ValueError("Only Python producer code roles can be archived")
    return frozenset(
        str(_path(str(root / "src" / "replay" / name), root=root)) for name in source_names
    )


def read_archives(
    descriptor: Path,
    descriptor_sha256: str,
    *,
    protocol: str,
    source_names: Sequence[str],
    root: Path,
    binding_fields: frozenset[str],
) -> CodeEvidence:
    """Read exact manifest/archive pins and retain caller-validated artifact context.

    ``binding_fields`` is a fixed consumer contract, such as ``{"windows"}``.
    The consumer must validate those fields and require each historical producer
    input to match its archive SHA before accepting any media or measured result.
    Neither this helper nor historical code bytes grant delivery approval.
    """
    if not root.is_absolute() or root.resolve() != root:
        raise ValueError("Producer evidence root must be canonical and absolute")
    if re.fullmatch(r"[a-z0-9][a-z0-9_-]{0,63}", protocol) is None or (
        not binding_fields
        or binding_fields.intersection(RESERVED_FIELDS)
        or any(re.fullmatch(r"[a-zA-Z][a-zA-Z0-9]{0,63}", key) is None for key in binding_fields)
    ):
        raise ValueError("Producer evidence requires an explicit protocol and binding contract")
    roles = _roles(source_names, root=root)
    descriptor = _path(str(descriptor), root=root)
    work = root / "edits" / "work"
    if not descriptor.is_relative_to(work):
        raise ValueError("Producer evidence manifests must stay under repository work")
    raw = _record(
        json.loads(_bound_bytes(descriptor, descriptor_sha256), object_pairs_hook=_unique_object)
    )
    if (
        set(raw) != RESERVED_FIELDS | binding_fields
        or type(raw.get("version")) is not int
        or raw.get("version") != 1
        or raw.get("protocol") != protocol
    ):
        raise ValueError("Producer evidence manifest differs from the declared protocol")
    rows = raw.get("archives")
    if not isinstance(rows, list) or not 1 <= len(rows) <= len(roles):
        raise ValueError("Producer evidence requires bounded nonempty archives")
    archives: dict[str, CodeArchive] = {}
    inputs = {str(descriptor): descriptor_sha256}
    for value in rows:
        row = _record(value)
        source = row.get("sourcePath")
        if set(row) != {"sourcePath", "path", "sha256"} or not isinstance(source, str):
            raise ValueError("Producer archive fields differ from the code-only contract")
        if source not in roles or source in archives:
            raise ValueError("Only unique allowlisted producer code may be archived")
        path = _path(row.get("path"), root=root)
        if not path.is_relative_to(work) or str(path) in inputs:
            raise ValueError("Producer archives must be distinct repository work evidence")
        digest = _digest(row.get("sha256"))
        _bound_bytes(path, digest)
        archives[source] = CodeArchive(path, digest)
        inputs[str(path)] = digest
    # Catch changes after parsing and checking all referenced archives.
    for name, digest in inputs.items():
        _bound_bytes(Path(name), digest)
    return CodeEvidence(archives, raw, inputs)
