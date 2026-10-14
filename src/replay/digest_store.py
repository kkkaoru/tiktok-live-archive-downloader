"""Advisory private SHA-256 reuse for unchanged inputs in intermediate operations.

Entries bind canonical source paths and exact stat identities with a private
store-key HMAC. This detects entry corruption, not a compromised same-UID owner,
privileged metadata rollback, media validity or provenance. Fresh initial,
terminal and resume acceptance must hash bytes independently; `fresh=True` and
CLI `--fresh` always do so. No existing workflow enables this store implicitly.
"""

from __future__ import annotations

import hashlib
import hmac
import json
import os
import re
import secrets
import stat
import sys
import time
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path
from tempfile import NamedTemporaryFile

from replay.core import ReplayError
from replay.recordings import record
from replay.stages import execution_lock

CHUNK_BYTES = 8 * 1024 * 1024
MINIMUM_BYTES = 64 * 1024 * 1024
MAX_ENTRY_BYTES = 8192
KEY_BYTES = 32
LOCK_SECONDS = 60.0
LOCK_BACKOFF = 0.01
SHA256 = re.compile(r"[a-f0-9]{64}")
Identity = tuple[int, int, int, int, int]


def identity(path: Path) -> Identity:
    if path.absolute() != path.resolve():
        raise ReplayError("Digest source must be canonical and must not traverse links")
    status = path.lstat()
    if not stat.S_ISREG(status.st_mode):
        raise ReplayError("Digest source must be a regular file")
    return _identity(status)


def _identity(status: os.stat_result) -> Identity:
    return (status.st_dev, status.st_ino, status.st_size, status.st_mtime_ns, status.st_ctime_ns)


def hash_file(path: Path) -> str:
    """Hash actual bytes and reject replacement or modification during the read."""
    path = path.absolute()
    before = identity(path)
    hasher = hashlib.sha256()
    descriptor = os.open(path, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK)
    with os.fdopen(descriptor, "rb") as stream:
        if _identity(os.fstat(stream.fileno())) != before:
            raise ReplayError("Digest source changed before the byte read")
        remaining = before[2]
        while remaining:
            chunk = stream.read(min(CHUNK_BYTES, remaining))
            if not chunk:
                raise ReplayError("Digest source changed during the byte read")
            hasher.update(chunk)
            remaining -= len(chunk)
        if (
            stream.read(1)
            or _identity(os.fstat(stream.fileno())) != before
            or identity(path) != before
        ):
            raise ReplayError("Digest source changed during the byte read")
    return hasher.hexdigest()


def _entry(store: Path, path: Path) -> Path:
    return store / f"{hashlib.sha256(str(path).encode('utf-8')).hexdigest()}.json"


def _store(store: Path) -> Path:
    root = Path(__file__).resolve().parents[2]
    path = store.absolute()
    if (
        Path.cwd().resolve() != root
        or path != path.resolve()
        or not path.is_relative_to(root / "edits" / "work")
        or not path.parent.is_dir()
    ):
        raise ReplayError("Digest store must have a canonical repository work parent")
    path.mkdir(mode=0o700, exist_ok=True)
    status = path.lstat()
    if (
        not stat.S_ISDIR(status.st_mode)
        or status.st_uid != os.getuid()
        or stat.S_IMODE(status.st_mode) != 0o700
    ):
        raise ReplayError("Digest store must be an owned private directory")
    return path


def _private(path: Path, status: os.stat_result) -> Identity:
    if (
        path != path.resolve()
        or not stat.S_ISREG(status.st_mode)
        or status.st_uid != os.getuid()
        or stat.S_IMODE(status.st_mode) != 0o600
        or status.st_nlink != 1
    ):
        raise ReplayError("Digest metadata must be an owned private single-link regular file")
    return _identity(status)


def _read_private(path: Path, *, limit: int) -> bytes:
    before = _private(path, path.lstat())
    if before[2] > limit:
        raise ReplayError("Digest metadata exceeds its byte budget")
    descriptor = os.open(path, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK)
    with os.fdopen(descriptor, "rb") as stream:
        if _private(path, os.fstat(stream.fileno())) != before:
            raise ReplayError("Digest metadata changed before reading")
        data = stream.read(limit + 1)
        if (
            len(data) > limit
            or _identity(os.fstat(stream.fileno())) != before
            or _private(path, path.lstat()) != before
        ):
            raise ReplayError("Digest metadata changed during reading")
    return data


@contextmanager
def _lease(store: Path, name: str) -> Iterator[None]:
    deadline = time.monotonic() + LOCK_SECONDS
    while True:
        with execution_lock(store, name) as acquired:
            if acquired:
                yield
                return
        if time.monotonic() >= deadline:
            raise ReplayError("Digest cache lease exceeded its finite deadline")
        time.sleep(LOCK_BACKOFF)


def _key(store: Path) -> bytes:
    path = store / "store.key"
    with _lease(store, "store-key"):
        try:
            value = _read_private(path, limit=KEY_BYTES)
        except FileNotFoundError:
            value = secrets.token_bytes(KEY_BYTES)
            descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW, 0o600)
            with os.fdopen(descriptor, "wb") as stream:
                stream.write(value)
                stream.flush()
                os.fsync(stream.fileno())
        if len(value) != KEY_BYTES:
            raise ReplayError("Digest store key is incomplete; it was not replaced")
    return value


def _payload(path: Path, before: Identity, digest: str) -> dict[str, object]:
    return {"version": 1, "path": str(path), "identity": list(before), "sha256": digest}


def _signature(key: bytes, payload: dict[str, object]) -> str:
    data = json.dumps(payload, sort_keys=True, separators=(",", ":"), allow_nan=False).encode(
        "utf-8"
    )
    return hmac.new(key, data, hashlib.sha256).hexdigest()


def _load(entry: Path, *, key: bytes, path: Path, before: Identity) -> str | None:
    try:
        data = _read_private(entry, limit=MAX_ENTRY_BYTES)
    except FileNotFoundError:
        return None
    try:
        saved = record(json.loads(data.decode("utf-8")))
    except (ValueError, UnicodeError, ReplayError):
        return None
    digest, signature = saved.get("sha256"), saved.get("mac")
    if (
        set(saved) != {"version", "path", "identity", "sha256", "mac"}
        or not isinstance(digest, str)
        or SHA256.fullmatch(digest) is None
        or not isinstance(signature, str)
        or SHA256.fullmatch(signature) is None
    ):
        return None
    expected = _payload(path, before, digest)
    saved_identity = saved.get("identity")
    if (
        type(saved.get("version")) is not int
        or saved.get("version") != 1
        or saved.get("path") != str(path)
        or not isinstance(saved_identity, list)
        or any(type(value) is not int for value in saved_identity)
        or saved_identity != list(before)
        or not hmac.compare_digest(signature, _signature(key, expected))
    ):
        return None
    return digest


def _publish(entry: Path, *, key: bytes, path: Path, before: Identity, digest: str) -> None:
    payload = _payload(path, before, digest)
    data = json.dumps({**payload, "mac": _signature(key, payload)}, allow_nan=False).encode("utf-8")
    if len(data) > MAX_ENTRY_BYTES:
        raise ReplayError("Digest entry exceeds its byte budget")
    pending: Path | None = None
    try:
        with NamedTemporaryFile(
            dir=entry.parent, prefix="digest-", suffix=".pending", delete=False
        ) as stream:
            pending = Path(stream.name)
            os.fchmod(stream.fileno(), 0o600)
            stream.write(data)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(pending, entry)
    finally:
        if pending is not None:
            pending.unlink(missing_ok=True)


def cached_digest(store: Path, path: Path, *, fresh: bool = False) -> str:
    """Reuse authenticated stat-bound intermediate metadata, or hash bytes fresh.

    A cache hit is not fresh cryptographic acceptance. Callers retain independent
    initial/terminal/resume checks and domain validation. No cache error authorizes
    skipping those checks or changing an existing completion receipt.
    """
    path = path.absolute()
    before = identity(path)
    if fresh or before[2] < MINIMUM_BYTES:
        return hash_file(path)
    store = _store(store)
    key = _key(store)
    entry = _entry(store, path)
    with _lease(store, entry.stem):
        saved = _load(entry, key=key, path=path, before=before)
        if saved is not None:
            if identity(path) != before:
                raise ReplayError("Digest source changed during cache lookup")
            return saved
        digest = hash_file(path)
        if identity(path) != before:
            raise ReplayError("Digest source changed before cache publication")
        _publish(entry, key=key, path=path, before=before, digest=digest)
        if identity(path) != before:
            raise ReplayError("Digest source changed during cache publication")
        return digest


def main(argv: list[str] | None = None) -> int:
    arguments = sys.argv[1:] if argv is None else argv
    fresh = bool(arguments and arguments[0] == "--fresh")
    if fresh:
        arguments = arguments[1:]
    if len(arguments) < 2:
        sys.stderr.write("usage: python -m replay.digest_store [--fresh] STORE FILE...\n")
        return 2
    store = Path(arguments[0])
    for name in arguments[1:]:
        try:
            print(f"{cached_digest(store, Path(name), fresh=fresh)}  {name}")
        except (OSError, ReplayError, ValueError) as error:
            sys.stderr.write(f"{name}: {error}\n")
            return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
