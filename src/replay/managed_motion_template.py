"""Bounded closed-library Motion body replacement; never edits a library database.

The caller must exclusively own app control. FD observations are point-in-time
checks, not an OS lock against an unrelated application reopening the library.
An interrupted apply is reconciled by verify, never by repeating the mutation.
"""

import argparse
import fcntl
import hashlib
import os
import stat
import subprocess
import time
from pathlib import Path
from typing import Protocol

from replay.fcp_native_queue import SystemHost as ProcessHost
from replay.native_sequence import ContractError
from replay.sequence_journal import bounded_read, decode, encode, identity, publish

CHECKOUT = Path(__file__).resolve().parents[2]
MAX_FILES = 4096
MAX_TOTAL = 64 * 1024 * 1024
MAX_AGE_NS = 30_000_000_000


class Host(Protocol):
    def now(self) -> int: ...

    def birth(self, pid: int) -> str: ...

    def closed(self, library: Path) -> bool: ...


class SystemHost:
    def now(self) -> int:
        return time.time_ns()

    def birth(self, pid: int) -> str:
        return ProcessHost().birth(pid)

    def closed(self, library: Path) -> bool:
        result = subprocess.run(
            ["/usr/sbin/lsof", "-nP", "+D", str(library), "-Fn"],
            capture_output=True,
            timeout=10,
            check=False,
        )
        return result.returncode == 1 and not result.stdout and not result.stderr


def _object(value: object) -> dict[str, object]:
    if not isinstance(value, dict) or any(not isinstance(key, str) for key in value):
        raise ContractError("Expected object")
    return {key: item for key, item in value.items() if isinstance(key, str)}


def _string(value: object) -> str:
    if not isinstance(value, str) or not value:
        raise ContractError("Expected nonempty string")
    return value


def _integer(value: object) -> int:
    if type(value) is not int:
        raise ContractError("Expected integer")
    return value


def _path(value: object) -> Path:
    path = Path(_string(value))
    if not path.is_absolute() or path != path.resolve() or not path.is_relative_to(CHECKOUT):
        raise ContractError("Expected canonical checkout path")
    return path


def _bytes(path: Path) -> bytes:
    before = path.lstat()
    if before.st_nlink != 1 or before.st_size == 0:
        raise ContractError("Expected nonempty single-link metadata")
    result = bounded_read(path)
    if identity(path.lstat()) != identity(before):
        raise ContractError("Metadata changed")
    return result


def reference(path: Path) -> dict[str, object]:
    _path(str(path))
    return {"path": str(path), "sha256": hashlib.sha256(_bytes(path)).hexdigest()}


def _ref(value: object) -> Path:
    item = _object(value)
    path = _path(item.get("path"))
    if reference(path) != item:
        raise ContractError("Reference changed")
    return path


def _read(path: Path) -> dict[str, object]:
    return decode(_bytes(path))


def _blob(path: Path, remaining: int) -> str:
    """Hash an opaque library blob with bounded memory and the remaining total budget."""
    _path(str(path))
    initial = path.lstat()
    detail = (
        f"library blob {path} (mode={stat.S_IFMT(initial.st_mode):o}, "
        f"bytes={initial.st_size}, remaining={remaining})"
    )
    if (
        not stat.S_ISREG(initial.st_mode)
        or initial.st_nlink != 1
        or not 0 <= initial.st_size <= remaining <= MAX_TOTAL
    ):
        raise ContractError(f"Invalid or over-budget {detail}")
    descriptor = os.open(path, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK)
    try:
        before = os.fstat(descriptor)
        if identity(before) != identity(initial) or before.st_nlink != 1:
            raise ContractError(f"Opening identity changed: {detail}")
        digest = hashlib.sha256()
        total = 0
        while True:
            chunk = os.read(descriptor, min(65536, remaining + 1 - total))
            if not chunk:
                break
            total += len(chunk)
            if total > remaining:
                raise ContractError(f"Growing blob exceeds aggregate budget: {detail}")
            digest.update(chunk)
        terminal = os.fstat(descriptor)
        current = path.lstat()
        if (
            total != initial.st_size
            or identity(terminal) != identity(initial)
            or identity(current) != identity(initial)
            or terminal.st_nlink != 1
            or current.st_nlink != 1
            or path != path.resolve()
        ):
            raise ContractError(f"Reading identity changed: {detail}")
        return digest.hexdigest()
    finally:
        os.close(descriptor)


def _snapshot(library: Path) -> dict[str, object]:
    """Hash only a bounded small managed library; never follow linked media."""
    root = library.lstat()
    result: dict[str, object] = {".": {"directory": [root.st_dev, root.st_ino]}}
    pending = [library]
    total = 0
    while pending:
        parent = pending.pop()
        before = parent.lstat()
        if not stat.S_ISDIR(before.st_mode) or parent != parent.resolve():
            raise ContractError("Library directory changed")
        for path in sorted(parent.iterdir()):
            if len(result) >= MAX_FILES:
                raise ContractError("Library inventory exceeds bound")
            info = path.lstat()
            key = str(path.relative_to(library))
            if stat.S_ISDIR(info.st_mode):
                result[key] = {"directory": [info.st_dev, info.st_ino]}
                pending.append(path)
            elif stat.S_ISLNK(info.st_mode):
                result[key] = {"link": os.readlink(path), "identity": list(identity(info))}
            elif stat.S_ISREG(info.st_mode):
                digest = _blob(path, MAX_TOTAL - total)
                total += info.st_size
                if identity(path.lstat()) != identity(info):
                    raise ContractError(
                        f"Library blob changed during inventory: {path}, bytes={info.st_size}"
                    )
                result[key] = {
                    "identity": list(identity(info)),
                    "ref": {"path": str(path), "sha256": digest},
                }
            else:
                raise ContractError("Unsupported library node")
        if identity(parent.lstat()) != identity(before):
            raise ContractError("Library changed during inventory")
    return result


def _load(config: Path) -> tuple[dict[str, object], Path, Path, Path, Path]:
    document = _read(config)
    if document.get("version") != 1 or document.get("kind") != "managed-motion-template-update":
        raise ContractError("Unsupported configuration")
    library = _path(document.get("library"))
    target = _path(document.get("target"))
    backup = _path(document.get("backup"))
    original = _ref(document.get("original"))
    replacement = _ref(document.get("replacement"))
    if (
        not library.is_dir()
        or library.suffix != ".fcpbundle"
        or target == library
        or not target.is_relative_to(library)
        or target.suffix != ".moti"
        or original.is_relative_to(library)
        or replacement.is_relative_to(library)
        or backup.is_relative_to(library)
        or backup in (original, replacement, config)
        or not backup.parent.is_dir()
        or _bytes(original) == _bytes(replacement)
    ):
        raise ContractError("Invalid managed template paths or identical bodies")
    control = _read(_ref(document.get("control")))
    actor = _object(document.get("actor"))
    if control != {
        "version": 1,
        "kind": "managed-template-exclusive-control",
        "library": str(library),
        "actor": actor,
        "soleOperator": True,
        "noQueuedLibraryOpen": True,
    }:
        raise ContractError("Exclusive app control is not bound")
    close_request = _read(_ref(document.get("closeRequest")))
    response = _read(_ref(document.get("closeResponse")))
    data = _object(response.get("data"))
    payload = _object(data.get("structuredContent"))
    if (
        close_request != {"bundleID": "com.apple.FinalCutApp", "library": library.stem}
        or response.get("ok") is not True
        or data.get("isError") is not False
        or payload.get("pid") != actor.get("pid")
        or _object(payload.get("finalCut")).get("libraryClosed") is not True
    ):
        raise ContractError("Original native close is not verified")
    return document, library, target, backup, replacement


def _boundary(document: dict[str, object], library: Path, host: Host) -> dict[str, object]:
    actor = _object(document.get("actor"))
    pid = _integer(actor.get("pid"))
    if pid <= 0 or host.birth(pid) != _string(actor.get("birth")) or not host.closed(library):
        raise ContractError("App identity changed or library handles are open/unknown")
    if host.birth(pid) != actor.get("birth"):
        raise ContractError("App identity changed during handle observation")
    return {"actor": actor, "library": str(library), "noOpenHandles": True}


def capture(config: Path, output: Path, *, host: Host | None = None) -> None:
    """Record fresh real closure and full bounded database/reference identities."""
    actual = host or SystemHost()
    config_ref = reference(config)
    document, library, target, backup, _ = _load(config)
    started = actual.now()
    before = _boundary(document, library, actual)
    if backup.exists() or backup.is_symlink():
        raise ContractError("Backup already occupied")
    if _bytes(target) != _bytes(_ref(document.get("original"))):
        raise ContractError("Managed body differs from original")
    snapshot = _snapshot(library)
    after = _boundary(document, library, actual)
    _ref(config_ref)
    if actual.now() - started > MAX_AGE_NS:
        raise ContractError("Capture expired")
    publish(
        _path(str(output)),
        encode(
            {
                "version": 1,
                "config": config_ref,
                "observedAtNs": started,
                "snapshot": snapshot,
                "before": before,
                "after": after,
                "code": reference(Path(__file__).resolve()),
            }
        ),
    )


def _unchanged(snapshot: dict[str, object], current: dict[str, object], key: str) -> None:
    if {k: v for k, v in snapshot.items() if k != key} != {
        k: v for k, v in current.items() if k != key
    }:
        raise ContractError("Library database/reference inventory changed")


def verify(config: Path, operation: Path, *, host: Host | None = None) -> dict[str, object]:
    """Read-only reconciliation; never performs a replacement or creates evidence."""
    actual = host or SystemHost()
    intent = _read(operation / "intent.json")
    if intent.get("config") != reference(config) or intent.get("code") != reference(
        Path(__file__).resolve()
    ):
        raise ContractError("Operation generation changed")
    observation = _read(_ref(intent.get("capture")))
    if observation.get("config") != intent.get("config"):
        raise ContractError("Operation capture configuration differs")
    document, library, target, backup, replacement = _load(config)
    before = _boundary(document, library, actual)
    if _bytes(backup) != _bytes(_ref(document.get("original"))) or _bytes(target) != _bytes(
        replacement
    ):
        raise ContractError("Backup or replacement body differs")
    _unchanged(
        _object(observation.get("snapshot")), _snapshot(library), str(target.relative_to(library))
    )
    after = _boundary(document, library, actual)
    return {
        "version": 1,
        "bodyUpdated": True,
        "databaseUnchanged": True,
        "qualityApproved": False,
        "target": reference(target),
        "backup": reference(backup),
        "intent": reference(operation / "intent.json"),
        "before": before,
        "after": after,
    }


def apply(config: Path, capture_path: Path, operation: Path, *, host: Host | None = None) -> None:
    """One immutable intent, exclusive old-body backup, and atomic exact-slot update."""
    actual = host or SystemHost()
    document, library, target, backup, replacement = _load(config)
    _path(str(operation))
    if operation.is_relative_to(library) or not operation.parent.is_dir():
        raise ContractError("Operation must be outside the library")
    observation = _read(capture_path)
    config_ref = reference(config)
    if observation.get("config") != config_ref or observation.get("code") != reference(
        Path(__file__).resolve()
    ):
        raise ContractError("Capture configuration mismatch")
    age = actual.now() - _integer(observation.get("observedAtNs"))
    if not 0 <= age <= MAX_AGE_NS:
        raise ContractError("Capture expired")
    # Shared library-specific advisory lock; independent app control is still required.
    lock = (
        CHECKOUT
        / "edits"
        / "work"
        / ("managed-template-" + hashlib.sha256(str(library).encode()).hexdigest() + ".lock")
    )
    lock_fd = os.open(lock, os.O_CREAT | os.O_RDWR | os.O_NOFOLLOW | os.O_NONBLOCK, 0o600)
    with os.fdopen(lock_fd, "a+b") as handle:
        lock_info = os.fstat(handle.fileno())
        if not stat.S_ISREG(lock_info.st_mode) or lock_info.st_nlink != 1:
            raise ContractError("Invalid library advisory lock")
        fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
        _boundary(document, library, actual)
        snapshot = _object(observation.get("snapshot"))
        if snapshot != _snapshot(library):
            raise ContractError("Captured library changed")
        old = _bytes(target)
        new = _bytes(replacement)
        if old != _bytes(_ref(document.get("original"))):
            raise ContractError("Old body changed")
        operation.mkdir(mode=0o700)
        publish(
            operation / "intent.json",
            encode(
                {
                    "config": config_ref,
                    "capture": reference(capture_path),
                    "code": reference(Path(__file__).resolve()),
                }
            ),
        )
        publish(backup, old)
        staged = operation / "replacement.moti"
        publish(staged, new)
        if staged.stat().st_dev != target.stat().st_dev:
            raise ContractError("Atomic replacement requires same filesystem")
        _load(config)
        _ref(config_ref)
        _boundary(document, library, actual)
        if snapshot != _snapshot(library) or _bytes(staged) != new:
            raise ContractError("Library or staged body changed before commit")
        if actual.now() - _integer(observation.get("observedAtNs")) > MAX_AGE_NS:
            raise ContractError("Capture expired before commit")
        os.replace(staged, target)
        descriptor = os.open(target.parent, os.O_RDONLY)
        try:
            os.fsync(descriptor)
        finally:
            os.close(descriptor)
        result = verify(config, operation, host=actual)
        publish(operation / "result.json", encode(result))


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("capture", "apply", "verify"))
    parser.add_argument("config", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--capture", type=Path)
    args = parser.parse_args(argv)
    if args.action == "capture":
        capture(args.config, args.output)
    elif args.action == "verify":
        print(encode(verify(args.config, args.output)).decode())
    elif args.capture is None:
        parser.error("apply requires --capture")
    else:
        apply(args.config, args.capture, args.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
