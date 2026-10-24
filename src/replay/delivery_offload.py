"""R2-only files of idle recordings: the offload record, verified offloading and fetching back.

A recording's delivery (or original, or candidate) may live only in the web viewer's R2 bucket
(``media/<name>``, ``media-original/<name>``) once it is idle. ``edits/<recording>/offload.json``
records, per file, its role, the local path it had, the R2 key, its size and SHA-256, how and when
the R2 copy was verified, and the evidence. The SHA-256 is never assumed: the R2 object is read
back in full (``r2_worker.WorkerStore.digest``) and must equal the local file's before anything is
deleted. Paths that were hardlinks of the same file (a delivery and ``final/candidate.mp4``) are
recorded with it and restored with it.

Fetching back (``replay.r2_fetch``) checks the disk budget first, downloads into a hidden file in
the target folder, verifies the SHA-256 and links it into place without overwriting anything.

    python -m replay.delivery_offload RECORDING --role delivery --path MOVIE --key media/NAME \
        [--link final/candidate.mp4 ...] [--evidence pipeline-evidence.json] [--dry-run]
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from collections.abc import Callable, Sequence
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Final, Protocol

from replay import disk_budget, r2_worker
from replay.core import ReplayError, write_private_text
from replay.recordings import record

RECORD_NAME: Final[str] = "offload.json"
ROLES: Final[frozenset[str]] = frozenset({"delivery", "original", "candidate"})
BUCKET: Final[str] = "transcript-viewer-media"
METHOD: Final[str] = "r2-full-readback-sha256"
# The web-viewer sync streams each upload's SHA-256 and checks every part's MD5 against its ETag;
# with the object's size and multipart ETag still matching, no read-back is needed.
UPLOAD_METHOD: Final[str] = "upload-sha256-size-etag"
METHODS: Final[frozenset[str]] = frozenset({METHOD, UPLOAD_METHOD})
PARTIAL_SUFFIX: Final[str] = ".r2-partial"
BLOCK_BYTES: Final[int] = 1 << 20
REPOSITORY: Final[Path] = Path(__file__).resolve().parents[2]
SYNC_STATE: Final[Path] = REPOSITORY / "web-viewer" / "sync-state.json"

Clock = Callable[[], datetime]


@dataclass(frozen=True, slots=True)
class Uploaded:
    """What the web-viewer sync recorded when it sent a key (``sync-state.json`` ``uploads``)."""

    size: int
    sha256: str
    etag: str


class Store(Protocol):
    def head(self, key: str) -> r2_worker.RemoteObject | None: ...

    def digest(self, key: str) -> tuple[int, str]: ...

    def get(self, key: str, destination: Path) -> int: ...


@dataclass(slots=True)
class Entry:
    """One offloaded file."""

    role: str
    local_path: str
    key: str
    bytes: int
    sha256: str
    verified_at: str
    links: list[str] = field(default_factory=list)
    evidence: dict[str, object] = field(default_factory=dict)
    bucket: str = BUCKET
    method: str = METHOD
    mode: int = 0o600
    mtime_ns: int = 0
    fetched_at: list[str] = field(default_factory=list)

    def document(self) -> dict[str, object]:
        data = asdict(self)
        return {
            "role": data["role"], "localPath": data["local_path"], "bucket": data["bucket"],
            "key": data["key"], "bytes": data["bytes"], "sha256": data["sha256"],
            "verification": {"method": data["method"], "verifiedAt": data["verified_at"]},
            "links": data["links"], "evidence": data["evidence"], "mode": data["mode"],
            "mtimeNs": data["mtime_ns"], "fetchedAt": data["fetched_at"],
        }  # fmt: skip


def edit_root(recording: str, *, root: Path = REPOSITORY / "edits") -> Path:
    """``edits/replay-<id>`` for a TikTok replay ID, else ``edits/<name>`` (``youtube-…``)."""
    name = f"replay-{recording}" if recording.isdigit() else recording
    if "/" in name or name.startswith("."):
        raise ReplayError(f"Invalid recording name: {recording}")
    return root / name


def record_path(root: Path) -> Path:
    return root / RECORD_NAME


def _text(item: dict[str, object], key: str) -> str:
    value = item.get(key)
    if not isinstance(value, str) or not value:
        raise ReplayError(f"Offload record entry needs {key}")
    return value


def _integer(item: dict[str, object], key: str) -> int:
    value = item.get(key)
    if type(value) is not int or value < 0:
        raise ReplayError(f"Offload record entry needs a non-negative integer {key}")
    return value


def _entry(value: object) -> Entry:
    item = record(value)
    verification = record(item.get("verification"))
    role = _text(item, "role")
    sha = _text(item, "sha256")
    links, fetched, evidence = item.get("links"), item.get("fetchedAt"), item.get("evidence")
    if (
        role not in ROLES
        or len(sha) != 64
        or verification.get("method") not in METHODS
        or not isinstance(links, list)
        or not all(isinstance(link, str) for link in links)
        or not isinstance(fetched, list)
        or not all(isinstance(stamp, str) for stamp in fetched)
        or not isinstance(evidence, dict)
    ):
        raise ReplayError("Malformed offload record entry")
    return Entry(
        role=role, local_path=_text(item, "localPath"), key=_text(item, "key"),
        bytes=_integer(item, "bytes"), sha256=sha,
        verified_at=_text(verification, "verifiedAt"), links=[str(link) for link in links],
        method=_text(verification, "method"),
        evidence={str(key): value for key, value in evidence.items()},
        bucket=_text(item, "bucket"), mode=_integer(item, "mode"),
        mtime_ns=_integer(item, "mtimeNs"), fetched_at=[str(stamp) for stamp in fetched],
    )  # fmt: skip


def load(root: Path) -> list[Entry]:
    """The recording's offloaded files; none when it has no record."""
    path = record_path(root)
    if not path.exists():
        return []
    data = record(json.loads(path.read_text(encoding="utf-8")))
    entries = data.get("files")
    if data.get("version") != 1 or not isinstance(entries, list):
        raise ReplayError(f"Unsupported offload record: {path}")
    return [_entry(value) for value in entries]


def save(root: Path, entries: Sequence[Entry]) -> None:
    """Replace the record atomically (written beside, then renamed)."""
    document = {"version": 1, "files": [entry.document() for entry in entries]}
    write_private_text(record_path(root), json.dumps(document, ensure_ascii=False, indent=1) + "\n")


def offloaded(root: Path, path: Path, role: str | None = None) -> Entry | None:
    """The record entry naming a local path or one of its hardlinks, if it was offloaded."""
    wanted = path.resolve()  # the pipeline and the CLI may spell one path differently
    for entry in load(root):
        named = any(Path(name).resolve() == wanted for name in (entry.local_path, *entry.links))
        if named and (role is None or entry.role == role):
            return entry
    return None


def sha256_of(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(BLOCK_BYTES), b""):
            digest.update(block)
    return digest.hexdigest()


def _evidence(path: Path | None, sha256: str) -> dict[str, object]:
    """A pipeline evidence index naming the delivery: its movie SHA-256 must match too."""
    if path is None:
        return {}
    data = record(json.loads(path.read_text(encoding="utf-8")))
    movie = record(data.get("movie", {}))
    if movie.get("sha256") != sha256:
        raise ReplayError(f"{path} records another movie SHA-256 than the offloaded file")
    return {"path": str(path), "sha256": sha256_of(path), "movieSha256": sha256}


def uploaded(key: str, state: Path | None = None) -> Uploaded | None:
    """The sync's record of uploading ``key``, if the sync state has one."""
    state = SYNC_STATE if state is None else state
    if not state.is_file():
        return None
    item = record(record(json.loads(state.read_text(encoding="utf-8"))).get("uploads", {})).get(key)
    if item is None:
        return None
    data = record(item)
    size, sha, etag = data.get("size"), data.get("sha256"), data.get("etag")
    if type(size) is not int or not isinstance(sha, str) or not isinstance(etag, str):
        raise ReplayError(f"Malformed upload record of {key}")
    return Uploaded(size, sha, etag)


def _verified(store: Store, key: str, size: int, sha256: str, sent: Uploaded | None) -> str:
    """How the R2 copy was shown equal to the local file; raises when it differs."""
    remote = store.head(key)
    if remote is None or remote.size != size:
        raise ReplayError(f"R2 object {key} is missing or of another size")
    etag = None if remote.etag is None else remote.etag.strip('"')
    if sent is not None and etag is not None and sent == Uploaded(size, sha256, etag):
        return UPLOAD_METHOD
    if store.digest(key) != (size, sha256):  # uploaded before hashes were recorded
        raise ReplayError(f"R2 object {key} differs from the local file; nothing is deleted")
    return METHOD


def offload(
    root: Path,
    *,
    role: str,
    path: Path,
    key: str,
    store: Store,
    links: Sequence[Path] = (),
    evidence: Path | None = None,
    sent: Uploaded | None = None,
    dry_run: bool = False,
    clock: Clock = lambda: datetime.now(UTC),
) -> Entry:
    """Verify the R2 copy against the local file, record it, then delete the local file and its
    hardlinks (unless ``dry_run``). The copy is verified by the SHA-256 the sync recorded while
    uploading it (``sent``) when R2's size and ETag still match that upload, else by a full
    read-back."""
    if role not in ROLES:
        raise ReplayError(f"Unknown offload role: {role}")
    if not path.is_file() or path.is_symlink():
        raise ReplayError(f"Not a regular local file: {path}")
    if offloaded(root, path) is not None:
        raise ReplayError(f"{path} is already recorded as offloaded")
    for link in links:
        if not link.is_file() or not link.samefile(path):
            raise ReplayError(f"{link} is not a hardlink of {path}")
    stat = path.stat()
    if stat.st_nlink != 1 + len(links):
        raise ReplayError(f"{path} has hardlinks the offload does not name")
    local = sha256_of(path)
    method = _verified(store, key, stat.st_size, local, sent)
    if path.stat().st_mtime_ns != stat.st_mtime_ns:
        raise ReplayError(f"{path} changed while it was verified; nothing is deleted")
    entry = Entry(
        role=role, local_path=str(path), key=key, bytes=stat.st_size, sha256=local,
        verified_at=clock().isoformat(), links=[str(link) for link in links],
        evidence=_evidence(evidence, local), method=method, mode=stat.st_mode & 0o777,
        mtime_ns=stat.st_mtime_ns,
    )  # fmt: skip
    if dry_run:
        return entry
    save(root, [*load(root), entry])
    for name in (*links, path):
        name.unlink()
    return entry


def download(entry: Entry, store: Store, destination: Path) -> Path:
    """Download an offloaded file to a new ``destination``, verified against its record."""
    destination.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
    disk_budget.require_headroom([destination.parent], entry.bytes)
    partial = destination.with_name(f".{destination.name}{PARTIAL_SUFFIX}")
    partial.unlink(missing_ok=True)
    try:
        if store.get(entry.key, partial) != entry.bytes or sha256_of(partial) != entry.sha256:
            raise ReplayError(f"The R2 copy of {destination} does not match its record")
        os.chmod(partial, entry.mode)
        os.utime(partial, ns=(entry.mtime_ns, entry.mtime_ns))
        os.link(partial, destination)  # never replaces an existing file
    finally:
        partial.unlink(missing_ok=True)
    return destination


def _restore(entry: Entry, store: Store) -> Path:
    destination = Path(entry.local_path)
    if os.path.lexists(destination):
        if not destination.is_file() or sha256_of(destination) != entry.sha256:
            raise ReplayError(f"A different file already exists at {destination}")
    else:
        download(entry, store, destination)
    for name in map(Path, entry.links):
        if os.path.lexists(name):
            if not name.samefile(destination):
                raise ReplayError(f"{name} exists and is not the restored file")
            continue
        name.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
        os.link(destination, name)
    return destination


def fetch(
    root: Path,
    *,
    role: str,
    store: Store,
    path: Path | None = None,
    clock: Clock = lambda: datetime.now(UTC),
) -> list[Path]:
    """Bring a role's offloaded files back (or only ``path``), verified, with their links."""
    entries = load(root)
    wanted = [
        entry for entry in entries
        if entry.role == role and (path is None or entry.local_path == str(path))
    ]  # fmt: skip
    if not wanted:
        raise ReplayError(f"No offloaded {role} recorded for {root.name}")
    restored = []
    for entry in wanted:
        restored.append(_restore(entry, store))
        entry.fetched_at.append(clock().isoformat())
        save(root, entries)
    return restored


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("recording")
    parser.add_argument("--role", choices=sorted(ROLES), required=True)
    parser.add_argument("--path", type=Path, required=True)
    parser.add_argument("--key", required=True)
    parser.add_argument("--link", type=Path, action="append", default=[])
    parser.add_argument("--evidence", type=Path)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--edit-root", type=Path, help="the recording's edit root, if not edits/")
    args = parser.parse_args(argv)
    with r2_worker.session(BUCKET) as store:
        entry = offload(
            args.edit_root or edit_root(args.recording), role=args.role, path=args.path.absolute(),
            key=args.key, store=store, links=[link.absolute() for link in args.link],
            evidence=args.evidence, sent=uploaded(args.key), dry_run=args.dry_run,
        )  # fmt: skip
    print(json.dumps({"dryRun": args.dry_run, **entry.document()}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
