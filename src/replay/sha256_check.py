"""Verify a `sha256sum`-style manifest with parallel hashlib reads.

The CLI requires a pinned manifest, canonical scopes and explicit measured
memory admission limits. Each line is `<64 hex digits>  <absolute path>`.
Fresh cryptographic reads preserve deterministic manifest order. The workflow
runner supplies the process deadline; admission limits are not an OS RSS quota.
"""

from __future__ import annotations

import argparse
import hashlib
import os
import re
import stat
import sys
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path

CHUNK_BYTES = 8 * 1024 * 1024
WORKERS = 4
MAX_MANIFEST_BYTES = 1024 * 1024
MAX_ENTRIES = 256
LINE = re.compile(r"^([0-9a-f]{64})  (.+)$")


@dataclass(frozen=True)
class DigestLimits:
    """Measured admission budget; this is not an operating-system RSS quota."""

    workers: int
    memory_budget_bytes: int
    baseline_rss_bytes: int
    worker_rss_bytes: int

    def __post_init__(self) -> None:
        values = (
            self.workers,
            self.memory_budget_bytes,
            self.baseline_rss_bytes,
            self.worker_rss_bytes,
        )
        if any(type(value) is not int or value <= 0 for value in values):
            raise ValueError("Digest limits must be positive integers")
        if self.workers > WORKERS:
            raise ValueError("At most four digest workers are supported")
        self.effective_workers(1)

    def effective_workers(self, entry_count: int) -> int:
        if type(entry_count) is not int or not 1 <= entry_count <= MAX_ENTRIES:
            raise ValueError("Invalid digest entry count")
        # stream.read can temporarily retain both the previous and next buffer.
        per_worker = max(self.worker_rss_bytes, 2 * CHUNK_BYTES)
        available = (self.memory_budget_bytes - self.baseline_rss_bytes) // per_worker
        if available < 1:
            raise ValueError("Digest memory budget cannot admit one worker")
        return min(self.workers, entry_count, available)


def _canonical(path: Path) -> None:
    if not path.is_absolute() or ".." in path.parts or path.resolve() != path:
        raise ValueError("Digest paths must be absolute and canonical")
    if any(part.is_symlink() for part in (path, *path.parents)):
        raise ValueError("Digest paths must not contain symlinks")


def _identity(path: Path) -> tuple[int, int, int, int, int]:
    _canonical(path)
    info = path.stat(follow_symlinks=False)
    if not stat.S_ISREG(info.st_mode):
        raise ValueError("Digest input must be a regular file")
    return _stat_identity(info)


def _stat_identity(info: os.stat_result) -> tuple[int, int, int, int, int]:
    return info.st_dev, info.st_ino, info.st_size, info.st_mtime_ns, info.st_ctime_ns


def digest(path: Path) -> str:
    before = _identity(path)
    hasher = hashlib.sha256()
    descriptor = os.open(path, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK)
    with os.fdopen(descriptor, "rb") as stream:
        if _stat_identity(os.fstat(stream.fileno())) != before:
            raise ValueError("Digest input changed before reading")
        while chunk := stream.read(CHUNK_BYTES):
            hasher.update(chunk)
        if _stat_identity(os.fstat(stream.fileno())) != before:
            raise ValueError("Digest input changed during reading")
    if _identity(path) != before:
        raise ValueError("Digest input changed after reading")
    return hasher.hexdigest()


def parse(manifest: str) -> list[tuple[str, Path]]:
    if not manifest or len(manifest.encode("utf-8")) > MAX_MANIFEST_BYTES:
        raise ValueError("Manifest must be nonempty and bounded")
    entries: list[tuple[str, Path]] = []
    seen: set[Path] = set()
    for line in manifest.splitlines():
        match = LINE.match(line)
        if match is None:
            raise ValueError(f"Malformed manifest line: {line[:80]!r}")
        path = Path(match.group(2))
        if str(path) != match.group(2):
            raise ValueError("Manifest path spelling must be canonical")
        if "\0" in str(path) or path in seen or len(entries) >= MAX_ENTRIES:
            raise ValueError("Duplicate, invalid or excessive manifest entries")
        _canonical(path)
        seen.add(path)
        entries.append((match.group(1), path))
    return entries


def mismatches(entries: list[tuple[str, Path]], *, limits: DigestLimits) -> list[Path]:
    snapshots: dict[Path, tuple[int, int, int, int, int]] = {}
    for _, path in entries:
        try:
            snapshots[path] = _identity(path)
        except (OSError, ValueError):
            continue

    def check(entry: tuple[str, Path]) -> Path | None:
        expected, path = entry
        try:
            return None if digest(path) == expected else path
        except (OSError, ValueError):
            return path

    with ThreadPoolExecutor(max_workers=limits.effective_workers(len(entries))) as pool:
        failed = {path for path in pool.map(check, entries) if path is not None}
    for _, path in entries:
        try:
            if snapshots.get(path) != _identity(path):
                failed.add(path)
        except (OSError, ValueError):
            failed.add(path)
    return [path for _, path in entries if path in failed]


def verify(
    manifest: Path, *, manifest_sha256: str, scopes: tuple[Path, ...], limits: DigestLimits
) -> list[Path]:
    """Verify a pinned bounded manifest and fresh file bytes without persistent memo."""
    if re.fullmatch(r"[0-9a-f]{64}", manifest_sha256) is None or not scopes:
        raise ValueError("Expected manifest SHA256 and explicit scopes")
    for scope in scopes:
        _canonical(scope)
        if not scope.is_dir():
            raise ValueError("Digest scope must be a directory")
    if not any(manifest.is_relative_to(scope) for scope in scopes):
        raise ValueError("Manifest is outside declared scopes")
    before = _identity(manifest)
    if before[2] > MAX_MANIFEST_BYTES:
        raise ValueError("Manifest exceeds byte limit")
    descriptor = os.open(manifest, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK)
    with os.fdopen(descriptor, "rb") as stream:
        if _stat_identity(os.fstat(stream.fileno())) != before:
            raise ValueError("Manifest changed before reading")
        raw = stream.read(MAX_MANIFEST_BYTES + 1)
    if len(raw) > MAX_MANIFEST_BYTES or hashlib.sha256(raw).hexdigest() != manifest_sha256:
        raise ValueError("Manifest hash or size differs")
    if _identity(manifest) != before:
        raise ValueError("Manifest changed while reading")
    entries = parse(raw.decode("utf-8"))
    if any(
        not any(path.is_relative_to(scope) for scope in scopes)
        for path in (manifest, *(path for _, path in entries))
    ):
        raise ValueError("Digest path is outside declared scopes")
    failed = mismatches(entries, limits=limits)
    if _identity(manifest) != before or digest(manifest) != manifest_sha256:
        raise ValueError("Manifest changed during verification")
    return failed


def main(argv: list[str]) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("manifest", type=Path)
    parser.add_argument("--manifest-sha256", required=True)
    parser.add_argument("--scope", action="append", type=Path, required=True)
    parser.add_argument("--workers", type=int, required=True)
    parser.add_argument("--memory-budget-bytes", type=int, required=True)
    parser.add_argument("--baseline-rss-bytes", type=int, required=True)
    parser.add_argument("--worker-rss-bytes", type=int, required=True)
    try:
        args = parser.parse_args(argv)
        limits = DigestLimits(
            args.workers, args.memory_budget_bytes, args.baseline_rss_bytes, args.worker_rss_bytes
        )
        failed = verify(
            args.manifest,
            manifest_sha256=args.manifest_sha256,
            scopes=tuple(args.scope),
            limits=limits,
        )
    except SystemExit as error:
        return int(error.code) if isinstance(error.code, int) else 2
    except (OSError, ValueError) as error:
        sys.stderr.write(f"{error}\n")
        return 2
    for path in failed:
        sys.stderr.write(f"{path}: FAILED\n")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
