"""Bounded, process-local digests for immutable inputs within one operation.

Only explicitly snapshotted inputs are cached. New outputs always use the
caller's ordinary hashing, and reuse starts with an empty context. Terminal
verification reads every input again, independent of the cached digests.
The largest fitting inputs receive cache slots; full inventory order is retained.
"""

import hashlib
import os
import stat
from collections.abc import Callable, Iterable, Iterator
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass, replace
from pathlib import Path

from replay.core import ReplayError

MAX_FILES = 1024
# A dependency closure may exceed cache capacity. Overflow is read freshly;
# its identity and start/terminal digest still remain mandatory.
MAX_INPUTS = 4096
MAX_BYTES = 256 * 1024**3
READ_BYTES = 1024 * 1024
Identity = tuple[int, int, int, int, int]


def _identity(value: os.stat_result) -> Identity:
    if not stat.S_ISREG(value.st_mode) or value.st_size <= 0:
        raise ReplayError("Cached inputs must be nonempty regular files")
    return value.st_dev, value.st_ino, value.st_size, value.st_mtime_ns, value.st_ctime_ns


def _canonical(path: Path) -> Path:
    lexical = path.absolute()
    if lexical != lexical.resolve(strict=True):
        raise ReplayError("Cached inputs must have matching lexical and resolved paths")
    return lexical


@dataclass(frozen=True)
class Entry:
    identity: Identity
    sha256: str | None = None


class InputHashCache:
    """Cache fixed inputs with identity checks; never persist or adopt cached state."""

    def __init__(
        self,
        paths: Iterable[Path],
        *,
        max_files: int = MAX_FILES,
        max_bytes: int = MAX_BYTES,
        allow_uncached_overflow: bool = False,
    ) -> None:
        if (
            type(max_files) is not int
            or not 1 <= max_files <= MAX_FILES
            or type(max_bytes) is not int
            or not 1 <= max_bytes <= MAX_BYTES
            or type(allow_uncached_overflow) is not bool
        ):
            raise ReplayError("Invalid bounded input hash cache budgets")
        self._entries: dict[Path, Entry] = {}
        self._uncached: set[Path] = set()
        for index, raw in enumerate(paths):
            if index >= (MAX_INPUTS if allow_uncached_overflow else MAX_FILES):
                raise ReplayError("Input hash cache has excessive input references")
            path = _canonical(raw)
            if path in self._entries:
                continue
            self._entries[path] = Entry(_identity(path.lstat()))
        if not self._entries:
            raise ReplayError("Input hash cache requires explicit inputs")
        total = cached_files = 0
        # Admission order must not change the manifest order of reads or proofs.
        priority = sorted(
            self._entries, key=lambda path: (-self._entries[path].identity[2], str(path))
        )
        for path in priority:
            size = self._entries[path].identity[2]
            if cached_files >= max_files or total + size > max_bytes:
                if not allow_uncached_overflow:
                    raise ReplayError("Input hash cache exceeds its file or byte budget")
                self._uncached.add(path)
            else:
                total += size
                cached_files += 1
        self.hash_reads = 0
        self.bytes_read = 0
        self.cache_hits = 0

    def _guard(self, path: Path, expected: Identity) -> None:
        if _canonical(path) != path or _identity(path.lstat()) != expected:
            raise ReplayError("Input changed within the hash cache operation")

    def guard(self) -> None:
        """Check every input between substeps, including unused cached inputs."""
        for path, entry in self._entries.items():
            self._guard(path, entry.identity)

    def _read(self, path: Path, entry: Entry, *, fresh: bool) -> str:
        self._guard(path, entry.identity)
        with os.fdopen(os.open(path, os.O_RDONLY | os.O_NOFOLLOW), "rb") as handle:
            if _identity(os.fstat(handle.fileno())) != entry.identity:
                raise ReplayError("Input changed before opening the cached descriptor")
            if not fresh and entry.sha256 is not None:
                digest = entry.sha256
                self.cache_hits += 1
            else:
                value = hashlib.sha256()
                while block := handle.read(READ_BYTES):
                    value.update(block)
                    self.bytes_read += len(block)
                digest = value.hexdigest()
                self.hash_reads += 1
            if _identity(os.fstat(handle.fileno())) != entry.identity:
                raise ReplayError("Input descriptor changed during cached hashing")
        self._guard(path, entry.identity)
        return digest

    def digest(self, path: Path, ordinary: Callable[[Path], str]) -> str:
        """Hash allowlisted input once; keep normal hashing for generated artifacts."""
        lexical = path.absolute()
        entry = self._entries.get(lexical)
        if entry is None:
            return ordinary(path)
        digest = self._read(lexical, entry, fresh=lexical in self._uncached)
        self._entries[lexical] = replace(entry, sha256=digest)
        return digest

    def prime(self) -> None:
        """Read all explicit inputs cryptographically before any production side effect."""
        for path, entry in tuple(self._entries.items()):
            self._entries[path] = replace(entry, sha256=self._read(path, entry, fresh=True))

    def verify(self) -> dict[str, str]:
        """Read every input again and compare real SHA256 bytes at the terminal boundary."""
        verified: dict[str, str] = {}
        for path, entry in self._entries.items():
            if entry.sha256 is None:
                raise ReplayError("Terminal hash verification requires primed inputs")
            if self._read(path, entry, fresh=True) != entry.sha256:
                raise ReplayError("Input bytes changed at terminal hash verification")
            verified[str(path)] = entry.sha256
        return verified


_active: ContextVar[InputHashCache | None] = ContextVar("replay_input_hash_cache", default=None)


@contextmanager
def input_hash_context(cache: InputHashCache) -> Iterator[InputHashCache]:
    """Limit reuse to one explicitly owned lexical scope in this process."""
    if _active.get() is not None:
        raise ReplayError("Nested input hash contexts are unsupported")
    token = _active.set(cache)
    try:
        yield cache
    finally:
        _active.reset(token)


def cached_digest(path: Path, ordinary: Callable[[Path], str]) -> str:
    """Preserve ordinary behavior unless a bounded operation explicitly opts in."""
    cache = _active.get()
    return ordinary(path) if cache is None else cache.digest(path, ordinary)
