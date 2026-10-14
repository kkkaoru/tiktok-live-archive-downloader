"""Bounded immutable checkpoint journal with advisory-exclusive CAS and pins."""

import fcntl
import hashlib
import json
import os
import secrets
import stat
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path

from replay.native_sequence import Checkpoint, Config, ContractError, Phase

MAX_BYTES = 8 * 1024 * 1024
PHASES: tuple[Phase, ...] = ("prepared", "started", "published", "retired", "released", "uncertain")
TRANSITIONS = {
    "prepared": {"started"},
    "started": {"published", "uncertain"},
    "published": {"retired"},
    "retired": {"released"},
    "released": set(),
    "uncertain": set(),
}


def identity(info: os.stat_result) -> tuple[int, int, int, int, int]:
    return info.st_dev, info.st_ino, info.st_size, info.st_mtime_ns, info.st_ctime_ns


def bounded_read(path: Path) -> bytes:
    """Read one regular metadata FD within budget; detect grow/replace/symlink races."""
    if not path.is_absolute() or path != path.resolve():
        raise ContractError("Expected canonical metadata path")
    descriptor = os.open(path, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK)
    try:
        before = os.fstat(descriptor)
        if not stat.S_ISREG(before.st_mode) or before.st_size > MAX_BYTES:
            raise ContractError("Expected bounded regular metadata")
        if identity(path.lstat()) != identity(before):
            raise ContractError("Metadata changed before read")
        chunks: list[bytes] = []
        total = 0
        while True:
            chunk = os.read(descriptor, min(65536, MAX_BYTES + 1 - total))
            if not chunk:
                break
            chunks.append(chunk)
            total += len(chunk)
            if total > MAX_BYTES:
                raise ContractError("Metadata grew beyond byte budget")
        if (
            total != before.st_size
            or identity(os.fstat(descriptor)) != identity(before)
            or identity(path.lstat()) != identity(before)
            or path != path.resolve()
        ):
            raise ContractError("Metadata changed during read")
        return b"".join(chunks)
    finally:
        os.close(descriptor)


def unique_pairs(pairs: list[tuple[str, object]]) -> dict[str, object]:
    result: dict[str, object] = {}
    for key, value in pairs:
        if key in result:
            raise ContractError("Duplicate JSON key")
        result[key] = value
    return result


def invalid_constant(value: str) -> object:
    raise ContractError("Nonfinite JSON value")


def decode(data: bytes | str) -> dict[str, object]:
    if len(data) > MAX_BYTES:
        raise ContractError("JSON exceeds byte budget")
    return _record(
        json.loads(data, object_pairs_hook=unique_pairs, parse_constant=invalid_constant)
    )


@dataclass(frozen=True)
class Pin:
    path: Path
    sha256: str

    def verify(self) -> None:
        if hashlib.sha256(bounded_read(self.path)).hexdigest() != self.sha256:
            raise ContractError("Pinned metadata/code changed")


def encode(value: object) -> bytes:
    data = (
        json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False) + "\n"
    ).encode()
    if len(data) > MAX_BYTES:
        raise ContractError("Journal item exceeds byte budget")
    return data


def publish(path: Path, data: bytes) -> None:
    """Atomic no-replace publication; all temporary bytes live beside target."""
    if (
        len(data) > MAX_BYTES
        or path != path.resolve()
        or path.is_symlink()
        or not path.parent.is_dir()
    ):
        raise ContractError("Noncanonical journal destination")
    directory = os.open(path.parent, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW)
    parent = os.fstat(directory)
    name = ".journal-" + secrets.token_hex(16)
    created = False
    try:
        descriptor = os.open(
            name, os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW, 0o600, dir_fd=directory
        )
        created = True
        with os.fdopen(descriptor, "wb") as stream:
            stream.write(data)
            stream.flush()
            os.fsync(stream.fileno())
        if path.parent != path.parent.resolve() or (
            path.parent.stat().st_dev,
            path.parent.stat().st_ino,
        ) != (parent.st_dev, parent.st_ino):
            raise ContractError("Publication parent changed")
        os.link(name, path.name, src_dir_fd=directory, dst_dir_fd=directory, follow_symlinks=False)
        os.fsync(directory)
        if bounded_read(path) != data:
            raise ContractError("Published bytes changed")
    finally:
        if created:
            os.unlink(name, dir_fd=directory)
        os.close(directory)


def _record(value: object) -> dict[str, object]:
    if not isinstance(value, dict) or any(not isinstance(key, str) for key in value):
        raise ContractError("Expected JSON object")
    return {str(key): item for key, item in value.items()}


def _checkpoint(raw: object) -> Checkpoint:
    row = _record(raw)
    config = _record(row.get("config"))
    tail = config.get("tail")
    identity, digest, proof = config.get("identity"), config.get("digest"), row.get("proof")
    if (
        set(row) != {"config", "phase", "proof"}
        or set(config) != {"identity", "tail", "digest"}
        or not isinstance(tail, list)
        or not tail
        or any(type(item) is not int or item <= 0 for item in tail)
        or not isinstance(identity, str)
        or not isinstance(digest, str)
        or not isinstance(proof, str)
    ):
        raise ContractError("Invalid checkpoint shape")
    selected: list[int] = []
    for item in tail:
        if not isinstance(item, int):
            raise ContractError("Invalid tail")
        selected.append(item)
    for phase in PHASES:
        if row.get("phase") == phase:
            return Checkpoint(Config(identity, tuple(selected), digest), phase, proof)
    raise ContractError("Invalid phase")


def _value(checkpoint: Checkpoint) -> dict[str, object]:
    return {
        "config": {
            "identity": checkpoint.config.identity,
            "tail": list(checkpoint.config.tail),
            "digest": checkpoint.config.digest,
        },
        "phase": checkpoint.phase,
        "proof": checkpoint.proof,
    }


class DurableJournal:
    def __init__(self, root: Path, pins: tuple[Pin, ...]) -> None:
        if not root.is_absolute() or root != root.resolve() or not root.is_dir() or not pins:
            raise ContractError("Existing canonical journal root and code/input pins required")
        if len(pins) > 1024 or len({pin.path for pin in pins}) != len(pins):
            raise ContractError("Invalid pin closure")
        own = Path(__file__).resolve()
        required = {own, own.with_name("native_sequence.py"), own.with_name("sequence_adapters.py")}
        if not required <= {pin.path for pin in pins}:
            raise ContractError("Controller, journal and adapters must be pinned")
        self.root, self.pins = root, pins
        self.pin_bytes = encode([{"path": str(pin.path), "sha256": pin.sha256} for pin in pins])
        with self._locked():
            path = root / "generation.json"
            if not path.exists():
                publish(path, self.pin_bytes)
            if bounded_read(path) != self.pin_bytes:
                raise ContractError("Journal generation pins changed")

    @contextmanager
    def _locked(self) -> Iterator[None]:
        descriptor = os.open(self.root / ".lock", os.O_RDWR | os.O_CREAT | os.O_NOFOLLOW, 0o600)
        try:
            fcntl.flock(descriptor, fcntl.LOCK_EX | fcntl.LOCK_NB)
            lock_identity = identity(os.fstat(descriptor))
            if (
                identity((self.root / ".lock").lstat()) != lock_identity
                or self.root != self.root.resolve()
            ):
                raise ContractError("Journal lock identity changed")
            for pin in self.pins:
                pin.verify()
            generation = self.root / "generation.json"
            if generation.exists() and (
                generation.is_symlink() or bounded_read(generation) != self.pin_bytes
            ):
                raise ContractError("Journal generation changed")
            yield
            if (
                identity((self.root / ".lock").lstat()) != lock_identity
                or self.root != self.root.resolve()
            ):
                raise ContractError("Journal lock replaced during transaction")
            for pin in self.pins:
                pin.verify()
        finally:
            os.close(descriptor)

    def _history(self, part: int) -> tuple[Checkpoint | None, str, int]:
        if type(part) is not int or part <= 0:
            raise ContractError("Positive part required")
        paths = sorted(self.root.glob(f"part-{part}-*.json"))
        if len(paths) > 6:
            raise ContractError("Too many transitions")
        previous, saved = "", None
        for index, path in enumerate(paths):
            if path.name != f"part-{part}-{index:02d}.json" or path.is_symlink():
                raise ContractError("Checkpoint gap or symlink")
            data = bounded_read(path)
            raw = decode(data)
            if (
                raw.get("previous") != previous
                or raw.get("generation") != hashlib.sha256(self.pin_bytes).hexdigest()
            ):
                raise ContractError("Checkpoint chain changed")
            current = _checkpoint(raw.get("checkpoint"))
            self._transition(part, saved, current)
            saved, previous = current, hashlib.sha256(data).hexdigest()
        return saved, previous, len(paths)

    @staticmethod
    def _transition(part: int, old: Checkpoint | None, new: Checkpoint) -> None:
        if new.config.tail[0] != part:
            raise ContractError("Checkpoint part differs from queue tail")
        if old is None:
            if new.phase != "prepared":
                raise ContractError("Initial checkpoint must be prepared")
        elif old.config != new.config or new.phase not in TRANSITIONS[old.phase]:
            raise ContractError("Invalid immutable checkpoint transition")

    def get(self, part: int) -> Checkpoint | None:
        with self._locked():
            return self._history(part)[0]

    def put(self, part: int, old: Checkpoint | None, new: Checkpoint) -> None:
        with self._locked():
            actual, previous, ordinal = self._history(part)
            if actual != old:
                raise ContractError("Exclusive journal CAS conflict")
            self._transition(part, old, new)
            publish(
                self.root / f"part-{part}-{ordinal:02d}.json",
                encode(
                    {
                        "generation": hashlib.sha256(self.pin_bytes).hexdigest(),
                        "previous": previous,
                        "checkpoint": _value(new),
                    }
                ),
            )
