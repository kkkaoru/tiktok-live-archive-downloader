"""Immutable code snapshots: a running pipeline step and later edits under ``src/`` coexist.

When enabled (``edits/work/code-snapshots/settings.json`` ``{"enabled": true}``) the pipeline
runs each step from a read-only copy of ``src/replay`` at ``code-snapshots/<content sha>/replay``
(deduplicated by content): ``PYTHONPATH`` points at the snapshot, so the step, its producers and
their lazy imports keep the code they started with while ``src/`` is edited. Modules that pin
their own files (``Path(__file__)``) and the paging closure (``recording_caption_steps``) then
pin the snapshot's immutable files; the step receipt records the snapshot sha.

Steps whose producers pin runtime file paths across resumed runs (the Compressor batch's runtime
receipts, the separation and ASR plan code inputs, the cleanup chunk pins) are not snapshotted
until those producers learn to accept a snapshot path: resuming their state from a different
path would read as changed code.

A running step holds an flock-held ``in-use`` marker in its snapshot. ``prune`` deletes a
snapshot that no live marker, no receipt (``codeSnapshot``) and no newer-than-``keep_hours``
creation still needs; run it with ``--dry-run`` first (disk-maintenance rules).
"""

from __future__ import annotations

import argparse
import fcntl
import hashlib
import json
import os
import shutil
import stat
import time
from collections.abc import Generator, Sequence
from contextlib import contextmanager
from pathlib import Path

from replay.recordings import record

VARIABLE = "REPLAY_CODE_SNAPSHOT"
# Steps whose producers pin runtime code paths across resumed runs (see the module docstring).
UNSNAPSHOTTED = frozenset({"caption-render", "cleanup", "separation", "asr", "cleanup-bootstrap"})
SKIPPED = ("__pycache__",)
KEEP_HOURS = 24.0


def root() -> Path:
    return Path(__file__).resolve().parents[2] / "edits" / "work" / "code-snapshots"


def enabled(directory: Path | None = None) -> bool:
    path = (root() if directory is None else directory) / "settings.json"
    if not path.is_file():
        return False
    return record(json.loads(path.read_text(encoding="utf-8"))).get("enabled") is True


def _files(package: Path) -> list[Path]:
    return sorted(
        path
        for path in package.rglob("*")
        if path.is_file() and not any(part in SKIPPED for part in path.relative_to(package).parts)
    )


def content_sha(package: Path) -> str:
    """The sha256 of every file's relative path and bytes under ``package``."""
    digest = hashlib.sha256()
    for path in _files(package):
        digest.update(str(path.relative_to(package)).encode() + b"\0")
        digest.update(hashlib.sha256(path.read_bytes()).digest())
    return digest.hexdigest()


def materialise(package: Path, directory: Path | None = None) -> tuple[str, Path]:
    """The snapshot (sha, PYTHONPATH root) of ``package``, written read-only once."""
    place = root() if directory is None else directory
    sha = content_sha(package)
    target = place / sha
    if not (target / package.name).is_dir():
        staged = place / f".{sha}.{os.getpid()}.partial"
        shutil.rmtree(staged, ignore_errors=True)
        shutil.copytree(package, staged / package.name, ignore=shutil.ignore_patterns(*SKIPPED))
        if content_sha(staged / package.name) != sha:
            shutil.rmtree(staged, ignore_errors=True)
            raise ValueError("The code changed while its snapshot was taken; take it again")
        for path in (staged / package.name).rglob("*"):
            path.chmod(stat.S_IRUSR | stat.S_IRGRP | (stat.S_IXUSR if path.is_dir() else 0))
        (staged / package.name).chmod(stat.S_IRUSR | stat.S_IXUSR)
        try:
            staged.rename(target)
        except OSError:  # another process published the same content first
            shutil.rmtree(staged, ignore_errors=True)
    return sha, target


@contextmanager
def in_use(snapshot: Path) -> Generator[None]:
    """Mark the snapshot as used by this running step until the block ends."""
    marker = snapshot.parent / f"{snapshot.name}.in-use.{os.getpid()}.{time.time_ns()}"
    descriptor = os.open(marker, os.O_RDWR | os.O_CREAT, 0o600)
    fcntl.flock(descriptor, fcntl.LOCK_EX)
    try:
        yield
    finally:
        marker.unlink(missing_ok=True)
        os.close(descriptor)


def _live(marker: Path) -> bool:
    descriptor = os.open(marker, os.O_RDONLY)
    try:
        fcntl.flock(descriptor, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        return True
    finally:
        os.close(descriptor)
    marker.unlink(missing_ok=True)
    return False


def referenced(directory: Path, receipts: Path) -> set[str]:
    """Snapshot shas a running step (live marker) or a step receipt still names."""
    used = {
        marker.name.split(".in-use.")[0] for marker in directory.glob("*.in-use.*") if _live(marker)
    }
    for receipt in receipts.glob("*/steps/*.json"):
        try:
            sha = record(json.loads(receipt.read_text(encoding="utf-8"))).get("codeSnapshot")
        except (OSError, ValueError):
            continue
        if isinstance(sha, str):
            used.add(sha)
    return used


def prune(
    directory: Path, receipts: Path, *, keep_hours: float = KEEP_HOURS, dry_run: bool = True
) -> list[str]:
    """Snapshots nothing needs any more (deleted unless ``dry_run``)."""
    used = referenced(directory, receipts)
    cutoff = time.time() - keep_hours * 3600
    unused = [
        path
        for path in sorted(directory.iterdir())
        if path.is_dir() and not path.name.startswith(".") and path.name not in used
        and path.stat().st_mtime < cutoff
    ]  # fmt: skip
    if not dry_run:
        for path in unused:
            for item in path.rglob("*"):
                item.chmod(stat.S_IRWXU)
            path.chmod(stat.S_IRWXU)
            shutil.rmtree(path)
    return [path.name for path in unused]


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("prune",))
    parser.add_argument("--recordings", type=Path, default=Path("edits/work/recordings"))
    parser.add_argument("--keep-hours", type=float, default=KEEP_HOURS)
    parser.add_argument("--apply", action="store_true", help="delete (default: list only)")
    args = parser.parse_args(argv)
    found = prune(root(), args.recordings, keep_hours=args.keep_hours, dry_run=not args.apply)
    print(json.dumps({"unused": found, "deleted": args.apply}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
