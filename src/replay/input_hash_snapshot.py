"""Run-scoped full-content verification for explicitly pinned shared inputs.

Intermediate digest lookups are provisional: consumers must finish successfully
before publishing results. This is neither a persistent cache nor an atomic
filesystem snapshot. Endpoint hashes cannot prove that bytes were never changed
and completely restored between observations; consumers needing that guarantee
must read immutable inputs or coordinate writers separately.
"""

import re
import stat
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path

from replay.stages import StageError, file_digest

MAX_INPUTS = 4096


@dataclass(frozen=True, slots=True)
class _Identity:
    device: int
    inode: int
    size: int
    modified_ns: int
    changed_ns: int
    mode: int
    links: int


@dataclass(frozen=True, slots=True)
class _Input:
    digest: str
    identity: _Identity


class InputHashSnapshot:
    """Hash each distinct canonical path at capture and finish, without global state.

    Keep per-receipt/configuration checks outside this shared-input scope. A
    failed or finished scope cannot be reused. Distinct hard-link paths remain
    distinct inputs; this utility grants no directory or source authorization.
    """

    def __init__(self, inputs: dict[Path, _Input]) -> None:
        self._inputs = inputs.copy()
        self._closed = False

    @classmethod
    def capture(cls, bindings: Iterable[tuple[Path, str]]) -> "InputHashSnapshot":
        """Validate expected pins and perform one initial full read per path."""
        expected: dict[Path, str] = {}
        for path, digest in bindings:
            if re.fullmatch(r"[a-f0-9]{64}", digest) is None:
                raise StageError("Shared inputs require lowercase SHA256 pins")
            if path in expected and expected[path] != digest:
                raise StageError("Conflicting shared input pins")
            expected[path] = digest
            if len(expected) > MAX_INPUTS:
                raise StageError("Too many distinct shared inputs")
        if not expected:
            raise StageError("Expected at least one shared input")
        inputs: dict[Path, _Input] = {}
        for path, digest in expected.items():
            identity = _identity(path)
            _verify_content(path, _Input(digest, identity))
            inputs[path] = _Input(digest, identity)
        # Earlier inputs must still have their identity after later reads.
        for path, item in inputs.items():
            _require_identity(path, item.identity)
        return cls(inputs)

    def checked_digest(self, path: Path, expected: str) -> str:
        """Return a provisional captured hash after checking pin and file identity.

        This avoids repeated full reads within one scope, but is not independent
        content verification. Successful finish is required for acceptance.
        """
        self._require_open()
        item = self._inputs.get(path)
        if item is None or item.digest != expected:
            self._closed = True
            raise StageError("Unknown shared input or changed expected pin")
        try:
            _require_identity(path, item.identity)
        except (OSError, StageError):
            self._closed = True
            raise
        return item.digest

    def finish(self) -> dict[Path, str]:
        """Rehash all inputs once and close; return pins only after full validation."""
        self._require_open()
        self._closed = True
        for path, item in self._inputs.items():
            _verify_content(path, item)
        for path, item in self._inputs.items():
            _require_identity(path, item.identity)
        return {path: item.digest for path, item in self._inputs.items()}

    def _require_open(self) -> None:
        if self._closed:
            raise StageError("Shared input verification scope is closed")


def _identity(path: Path) -> _Identity:
    if not path.is_absolute() or path != path.resolve(strict=True):
        raise StageError("Shared input must be a canonical unlinked absolute path")
    value = path.lstat()
    if not stat.S_ISREG(value.st_mode) or value.st_size <= 0:
        raise StageError("Shared input must be a nonempty regular file")
    return _Identity(
        value.st_dev,
        value.st_ino,
        value.st_size,
        value.st_mtime_ns,
        value.st_ctime_ns,
        value.st_mode,
        value.st_nlink,
    )


def _require_identity(path: Path, expected: _Identity) -> None:
    if _identity(path) != expected:
        raise StageError("Shared input identity changed")


def _verify_content(path: Path, item: _Input) -> None:
    _require_identity(path, item.identity)
    actual = file_digest(path)
    _require_identity(path, item.identity)
    if actual != item.digest:
        raise StageError("Shared input SHA256 differs from its expected pin")
