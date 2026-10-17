"""Offload a delivery's render intermediates to R2 before they are retired, and restore them.

`replay.delivery_retention --apply` hands its eligible rows to `offload` before
`leaf_retirement` unlinks anything. Each recording keeps one manifest,
`<viewerRoot>/evidence/intermediate-offload.json`, listing every offloaded file by `root` and
`relative` path, never as one absolute media path, which `storage_disposition` would read as a
binding reference. Entries record SHA-256, size, mode, mtime, the R2 key and a state:

- `pending`: hashed and recorded, upload not yet confirmed.
- `uploaded`: the object exists with the same size and SHA-256 metadata.
- `retired`: the local file was unlinked after upload.
- `restored`: the file was downloaded back and its SHA-256 matched.

Objects live under `intermediates/<recording>/<sha256>` in a dedicated bucket whose lifecycle
rule deletes them 90 days after upload; `expiresAt` comes from the object's own upload time,
so a rerun that finds an existing object never extends it. Every step can be rerun: an
object that already matches is not uploaded again, and a restore never overwrites a file.
"""

from __future__ import annotations

import argparse
import fcntl
import json
import os
import sys
import tempfile
from collections.abc import Callable, Iterator, Mapping, Sequence
from contextlib import contextmanager
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime, timedelta
from enum import StrEnum
from pathlib import Path
from typing import Protocol

import httpx

from replay import disk_budget, leaf_retirement, r2_worker
from replay.core import ReplayError
from replay.r2_worker import RemoteObject
from replay.recordings import record

BUCKET = "replay-intermediates"
KEY_PREFIX = "intermediates"
EVIDENCE_DIRECTORY = "evidence"
MANIFEST_NAME = "intermediate-offload.json"
LOCK_NAME = "intermediate-offload.lock"
PARTIAL_SUFFIX = ".offload-partial"
VERSION = 1
LIFETIME = timedelta(days=90)
SHA256_METADATA = "sha256"

Clock = Callable[[], datetime]


class State(StrEnum):
    PENDING = "pending"
    UPLOADED = "uploaded"
    RETIRED = "retired"
    RESTORED = "restored"


class ObjectStore(Protocol):
    def head(self, key: str) -> RemoteObject | None: ...

    def put(self, key: str, path: Path, metadata: Mapping[str, str]) -> None: ...

    def get(self, key: str, destination: Path) -> int: ...


@dataclass(slots=True)
class Entry:
    root: str
    relative: str
    key: str
    sha256: str
    size: int
    mode: int
    mtime_ns: int
    state: State
    uploaded_at: str | None = None
    expires_at: str | None = None

    @property
    def path(self) -> Path:
        return Path(self.root) / self.relative


@dataclass(slots=True)
class Manifest:
    recording: str
    bucket: str
    entries: dict[tuple[str, str], Entry] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class OffloadSummary:
    uploaded: int
    reused: int
    skipped: dict[str, str]


ENTRY_FIELDS: Mapping[str, str] = {
    "root": "root",
    "relative": "relative",
    "key": "key",
    "sha256": "sha256",
    "size": "bytes",
    "mode": "mode",
    "mtime_ns": "mtimeNs",
    "state": "state",
    "uploaded_at": "uploadedAt",
    "expires_at": "expiresAt",
}
REUSABLE_HASH_STATES = frozenset({State.PENDING, State.UPLOADED})


def manifest_path(viewer_root: Path) -> Path:
    return viewer_root / EVIDENCE_DIRECTORY / MANIFEST_NAME


def object_key(recording: str, sha256: str) -> str:
    return f"{KEY_PREFIX}/{recording}/{sha256}"


def _text(data: Mapping[str, object], key: str) -> str:
    value = data.get(key)
    if not isinstance(value, str) or not value:
        raise ReplayError(f"Offload manifest entry field {key} must be text")
    return value


def _integer(data: Mapping[str, object], key: str) -> int:
    value = data.get(key)
    if type(value) is not int:
        raise ReplayError(f"Offload manifest entry field {key} must be an integer")
    return value


def _timestamp(data: Mapping[str, object], key: str) -> str | None:
    value = data.get(key)
    if value is not None and not isinstance(value, str):
        raise ReplayError(f"Offload manifest entry field {key} must be a timestamp")
    return value


def _entry(raw: object) -> Entry:
    data = record(raw)
    try:
        state = State(_text(data, "state"))
    except ValueError as error:
        raise ReplayError("Offload manifest entry has an unknown state") from error
    return Entry(
        root=_text(data, "root"),
        relative=_text(data, "relative"),
        key=_text(data, "key"),
        sha256=_text(data, "sha256"),
        size=_integer(data, "bytes"),
        mode=_integer(data, "mode"),
        mtime_ns=_integer(data, "mtimeNs"),
        state=state,
        uploaded_at=_timestamp(data, "uploadedAt"),
        expires_at=_timestamp(data, "expiresAt"),
    )


def load(path: Path, *, recording: str, bucket: str = BUCKET) -> Manifest:
    """The recording's manifest, or an empty one; a manifest of another recording is refused."""
    if not path.exists():
        return Manifest(recording, bucket)
    data = record(json.loads(path.read_text(encoding="utf-8")))
    if data.get("version") != VERSION or data.get("recording") != recording:
        raise ReplayError(f"Offload manifest belongs to another recording or version: {path}")
    stored_bucket, files = data.get("bucket"), data.get("files")
    if not isinstance(stored_bucket, str) or not isinstance(files, list):
        raise ReplayError("Offload manifest needs a bucket and a files array")
    entries = (_entry(raw) for raw in files)
    return Manifest(
        recording, stored_bucket, {(item.root, item.relative): item for item in entries}
    )


def _serialized(entry: Entry) -> dict[str, object]:
    values: dict[str, object] = {**asdict(entry), "state": str(entry.state)}
    return {key: values[name] for name, key in ENTRY_FIELDS.items()}


def _entry_order(entry: Entry) -> tuple[str, str]:
    return entry.root, entry.relative


def save(manifest: Manifest, path: Path) -> None:
    """Atomically replace the manifest: same-directory temporary file, fsync, rename."""
    files = [_serialized(entry) for entry in sorted(manifest.entries.values(), key=_entry_order)]
    body = {
        "version": VERSION,
        "recording": manifest.recording,
        "bucket": manifest.bucket,
        "files": files,
    }
    path.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        "w", encoding="utf-8", dir=path.parent, prefix=f".{path.name}.", delete=False
    ) as stream:
        json.dump(body, stream, indent=2, sort_keys=True)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    os.chmod(stream.name, 0o600)
    os.replace(stream.name, path)


@contextmanager
def locked(viewer_root: Path) -> Iterator[None]:
    """One offload or restore per recording at a time; the lock file is never unlinked."""
    lock = viewer_root / EVIDENCE_DIRECTORY / LOCK_NAME
    lock.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
    descriptor = os.open(lock, os.O_CREAT | os.O_RDWR | os.O_NOFOLLOW, 0o600)
    with os.fdopen(descriptor, "a+b") as stream:
        try:
            fcntl.flock(stream, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as error:
            raise ReplayError(
                "Another process is offloading or restoring this recording"
            ) from error
        try:
            yield
        finally:
            fcntl.flock(stream, fcntl.LOCK_UN)


def _split(path: Path, roots: Sequence[Path]) -> tuple[str, str]:
    for root in roots:
        if path.is_relative_to(root):
            return str(root), str(path.relative_to(root))
    raise ReplayError(f"Offload target is outside the allowed roots: {path.name}")


def _matches(remote: RemoteObject | None, entry: Entry) -> bool:
    return (
        remote is not None
        and remote.size == entry.size
        and remote.metadata.get(SHA256_METADATA) == entry.sha256
    )


def _hashed(row: leaf_retirement.Row, roots: Sequence[Path], manifest: Manifest) -> Entry | None:
    """The row's entry, hashing only when no matching hash is recorded; None if it changed."""
    root, relative = _split(Path(row.path), roots)
    known = manifest.entries.get((root, relative))
    if (
        known is not None
        and known.state in REUSABLE_HASH_STATES
        and (known.size, known.mtime_ns) == (row.size, row.mtime_ns)
    ):
        return known
    digest = leaf_retirement.sha256_of(Path(row.path))
    if leaf_retirement.static_hold(row, roots) is not None:
        return None
    mode = os.lstat(row.path).st_mode & 0o7777
    key = object_key(manifest.recording, digest)
    return Entry(root, relative, key, digest, row.size, mode, row.mtime_ns, State.PENDING)


def _confirm(entry: Entry, store: ObjectStore) -> bool:
    """Upload unless an identical object exists; True when an upload happened."""
    remote = store.head(entry.key)
    uploaded = not _matches(remote, entry)
    if uploaded:
        store.put(entry.key, entry.path, {SHA256_METADATA: entry.sha256})
        remote = store.head(entry.key)
    if remote is None or not _matches(remote, entry):
        raise ReplayError(f"R2 object does not match {entry.relative} after upload")
    entry.state = State.UPLOADED
    entry.uploaded_at = remote.uploaded.isoformat()
    entry.expires_at = (remote.uploaded + LIFETIME).isoformat()
    return uploaded


def offload(
    rows: Sequence[leaf_retirement.Row],
    *,
    recording: str,
    roots: Sequence[Path],
    viewer_root: Path,
    store: ObjectStore,
) -> OffloadSummary:
    """Record, upload and confirm every unchanged row; rows that changed are left to be held."""
    path = manifest_path(viewer_root)
    skipped: dict[str, str] = {}
    uploaded = reused = 0
    with locked(viewer_root):
        manifest = load(path, recording=recording)
        for row in rows:
            reason = leaf_retirement.static_hold(row, roots)
            entry = None if reason is not None else _hashed(row, roots, manifest)
            if entry is None:
                skipped[row.path] = reason or "identity changed while hashing"
                continue
            manifest.entries[(entry.root, entry.relative)] = entry
            save(manifest, path)
            if _confirm(entry, store):
                uploaded += 1
            else:
                reused += 1
            save(manifest, path)
    return OffloadSummary(uploaded, reused, skipped)


def mark_retired(viewer_root: Path, *, recording: str) -> int:
    """Mark uploaded entries whose local file is gone; returns how many changed."""
    path = manifest_path(viewer_root)
    with locked(viewer_root):
        manifest = load(path, recording=recording)
        gone = [
            entry
            for entry in manifest.entries.values()
            if entry.state is State.UPLOADED and not os.path.lexists(entry.path)
        ]
        for entry in gone:
            entry.state = State.RETIRED
        if gone:
            save(manifest, path)
    return len(gone)


def _restore_entry(entry: Entry, store: ObjectStore, now: datetime) -> None:
    destination = entry.path
    if os.path.lexists(destination):
        if not destination.is_file() or leaf_retirement.sha256_of(destination) != entry.sha256:
            raise ReplayError(f"A different file already exists at {entry.relative}")
        return
    if entry.expires_at is not None and now >= datetime.fromisoformat(entry.expires_at):
        raise ReplayError(f"{entry.relative} expired from R2 at {entry.expires_at}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    disk_budget.require_headroom([destination.parent], entry.size)
    partial = destination.with_name(destination.name + PARTIAL_SUFFIX)
    partial.unlink(missing_ok=True)
    try:
        store.get(entry.key, partial)
        if leaf_retirement.sha256_of(partial) != entry.sha256:
            raise ReplayError(f"Downloaded {entry.relative} does not match its SHA-256")
        os.chmod(partial, entry.mode)
        os.utime(partial, ns=(entry.mtime_ns, entry.mtime_ns))
        os.link(partial, destination)
    finally:
        partial.unlink(missing_ok=True)


def restore(
    viewer_root: Path,
    *,
    recording: str,
    store: ObjectStore,
    relative: str | None = None,
    clock: Clock = lambda: datetime.now(UTC),
) -> list[Path]:
    """Download retired entries back, verify SHA-256 and restore mode and mtime."""
    path = manifest_path(viewer_root)
    restored: list[Path] = []
    with locked(viewer_root):
        manifest = load(path, recording=recording)
        wanted = [
            entry
            for entry in manifest.entries.values()
            if entry.state in {State.RETIRED, State.UPLOADED}
            and (relative is None or entry.relative == relative)
        ]
        if relative is not None and not wanted:
            raise ReplayError(f"No offloaded entry matches {relative}")
        if relative is not None and len(wanted) > 1:
            raise ReplayError(f"{relative} exists under several roots; restore them all instead")
        for entry in wanted:
            _restore_entry(entry, store, clock())
            entry.state = State.RESTORED
            save(manifest, path)
            restored.append(entry.path)
    return restored


def status(viewer_root: Path, *, recording: str) -> dict[str, int]:
    manifest = load(manifest_path(viewer_root), recording=recording)
    counts = dict.fromkeys((str(state) for state in State), 0)
    for entry in manifest.entries.values():
        counts[str(entry.state)] += 1
    return counts


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="replay.intermediate_offload", description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    for name in ("restore", "status"):
        command = commands.add_parser(name)
        command.add_argument("viewer_root", type=Path, help="edits/<recording> of the delivery")
    commands.choices["restore"].add_argument("--relative", help="restore only this entry")
    args = parser.parse_args(argv)
    viewer_root = args.viewer_root.absolute()
    recording = viewer_root.name
    try:
        if args.command == "status":
            print(json.dumps(status(viewer_root, recording=recording), sort_keys=True))
            return 0
        bucket = load(manifest_path(viewer_root), recording=recording).bucket
        with r2_worker.session(bucket) as store:
            restored = restore(
                viewer_root, recording=recording, store=store, relative=args.relative
            )
    except (ReplayError, OSError, ValueError, httpx.HTTPError) as error:
        print(f"INTERMEDIATE_OFFLOAD_ERROR {error}", file=sys.stderr)
        return 1
    print(json.dumps({"restored": [str(path) for path in restored]}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
