"""Revalidate a complete retirement batch without creating or repairing artifacts.

An existing shared lock protects the completed metadata. Every domain is checked
again within one fresh input digest scope; checkpoints never replace verification.
"""

import argparse
import fcntl
import json
import os
import stat
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import BinaryIO

from replay import part_master_publication as publication
from replay import part_retirement_verification_batch as batch
from replay.core import ReplayError
from replay.input_hash_cache import InputHashCache, input_hash_context
from replay.recordings import record
from replay.stages import file_digest

Identity = tuple[int, int, int, int, int, int, int]
COUNTERS = frozenset({"hashReads", "bytesRead", "cacheHits"})
METRIC_SCOPE = "InputHashCache payload reads; bounded metadata parsing excluded"


@dataclass(frozen=True)
class Checkpoint:
    part: int
    artifact: batch.Reference


@dataclass(frozen=True)
class Request:
    artifact: batch.Reference
    manifest: batch.Reference
    operation: batch.Reference
    receipt: batch.Reference
    checkpoints: tuple[Checkpoint, ...]

    @property
    def output(self) -> Path:
        return self.receipt.path.parent

    @property
    def metadata(self) -> tuple[batch.Reference, ...]:
        return (
            self.artifact,
            self.manifest,
            self.operation,
            self.receipt,
            *(checkpoint.artifact for checkpoint in self.checkpoints),
        )


@dataclass(frozen=True)
class Verification:
    phase: batch.Phase
    parts: tuple[int, ...]
    receipt: batch.Reference
    inputs: dict[str, str]
    hash_reads: int
    bytes_read: int
    cache_hits: int

    def json(self) -> dict[str, object]:
        return {
            "version": 1,
            "kind": "readonly-retirement-verification",
            "phase": self.phase,
            "parts": list(self.parts),
            "receipt": self.receipt.json(),
            "inputs": self.inputs,
            "retirementDomainVerified": True,
            "initialAndTerminalBytesVerified": True,
            "sourceAbsenceObserved": self.phase == "after-delete",
            "deletionPerformed": False,
            "filesWritten": False,
            "releaseDecision": None,
            "qualityApproved": False,
            "deliveryApproved": False,
            "hashReads": self.hash_reads,
            "bytesRead": self.bytes_read,
            "cacheHits": self.cache_hits,
            "physicalIOBytes": None,
            "metricScope": METRIC_SCOPE,
        }


def _work_ref(value: object) -> batch.Reference:
    ref = batch._ref(value)
    batch._path(str(ref.path), work=True)
    return ref


def _checkpoint(value: object) -> Checkpoint:
    raw = record(value)
    part = raw.get("part")
    if (
        set(raw) != {"part", "artifact"}
        or type(part) is not int
        or not 1 <= part <= batch.MAX_PARTS
    ):
        raise ReplayError("Readonly checkpoint requires a bounded part and exact artifact")
    return Checkpoint(part, _work_ref(raw["artifact"]))


def load(path: Path) -> Request:
    """Read an exact request binding every artifact of an already complete batch."""
    artifact = batch._current(batch._path(str(path), work=True))
    raw = batch._read(artifact)
    rows = raw.get("checkpoints")
    if (
        set(raw) != {"version", "kind", "manifest", "operation", "receipt", "checkpoints"}
        or type(raw.get("version")) is not int
        or raw.get("version") != 1
        or raw.get("kind") != "readonly-retirement-verification"
        or not isinstance(rows, list)
        or not 1 <= len(rows) <= batch.MAX_PARTS
    ):
        raise ReplayError("Invalid readonly retirement request")
    request = Request(
        artifact,
        _work_ref(raw["manifest"]),
        _work_ref(raw["operation"]),
        _work_ref(raw["receipt"]),
        tuple(_checkpoint(row) for row in rows),
    )
    if (
        request.operation.path != request.output / "operation.json"
        or request.receipt.path != request.output / "receipt.json"
        or any(
            row.artifact.path != request.output / f"part-{row.part:04d}.json"
            for row in request.checkpoints
        )
        or len({row.part for row in request.checkpoints}) != len(request.checkpoints)
        or len({ref.path for ref in request.metadata}) != len(request.metadata)
    ):
        raise ReplayError("Readonly request must bind distinct exact batch artifacts")
    return request


def _identity(value: os.stat_result) -> Identity:
    return (
        value.st_dev,
        value.st_ino,
        value.st_mode,
        value.st_nlink,
        value.st_size,
        value.st_mtime_ns,
        value.st_ctime_ns,
    )


def _directory(request: Request) -> Identity:
    root = batch._path(str(request.output), work=True)
    value = root.lstat()
    if not stat.S_ISDIR(value.st_mode):
        raise ReplayError("Readonly retirement requires an existing regular directory")
    expected = {"operation.json", "operation.lock", "receipt.json"} | {
        f"part-{row.part:04d}.json" for row in request.checkpoints
    }
    children = tuple(root.iterdir())
    if {child.name for child in children} != expected or any(
        batch._path(str(child), work=True) != child for child in children
    ):
        raise ReplayError("Readonly retirement requires complete exact output membership")
    return _identity(value)


def _lock_identity(path: Path) -> Identity:
    batch._path(str(path), work=True)
    value = path.lstat()
    if (
        not stat.S_ISREG(value.st_mode)
        or value.st_nlink != 1
        or value.st_mode & 0o077
        or value.st_size != 0
    ):
        raise ReplayError("Readonly retirement lock must be existing private empty evidence")
    return _identity(value)


def _guard(request: Request, handle: BinaryIO, *, directory: Identity, lock: Identity) -> None:
    if (
        _directory(request) != directory
        or _lock_identity(request.output / "operation.lock") != lock
        or _identity(os.fstat(handle.fileno())) != lock
    ):
        raise ReplayError("Readonly retirement ownership or output identity changed")


@contextmanager
def _locked(request: Request) -> Iterator[tuple[BinaryIO, Identity, Identity]]:
    directory = _directory(request)
    path = request.output / "operation.lock"
    lock = _lock_identity(path)
    with os.fdopen(os.open(path, os.O_RDONLY | os.O_NOFOLLOW), "rb") as handle:
        if _identity(os.fstat(handle.fileno())) != lock:
            raise ReplayError("Readonly retirement lock changed before opening")
        try:
            fcntl.flock(handle, fcntl.LOCK_SH | fcntl.LOCK_NB)
        except BlockingIOError as error:
            raise ReplayError("Readonly retirement batch is already owned") from error
        _guard(request, handle, directory=directory, lock=lock)
        yield handle, directory, lock
        _guard(request, handle, directory=directory, lock=lock)


def _inputs(request: Request, refs: tuple[batch.Reference, ...]) -> tuple[batch.Reference, ...]:
    inventory = batch.Inventory()
    for ref in (*refs, *request.metadata):
        inventory.add(ref.json())
    return inventory.ordered()


def _matches(actual: dict[str, object], expected: dict[str, object]) -> bool:
    # Python equality considers True equal to 1; serialized JSON preserves types.
    return json.dumps(actual, sort_keys=True) == json.dumps(expected, sort_keys=True)


def _receipt(
    request: Request,
    *,
    phase: batch.Phase,
    parts: tuple[batch.Part, ...],
    refs: tuple[batch.Reference, ...],
) -> None:
    raw = batch._read(request.receipt)
    stable: dict[str, object] = {
        "version": 1,
        "manifest": request.manifest.json(),
        "phase": phase,
        "kind": "verified-retirement-batch",
        "parts": [part.number for part in parts],
        "inputs": {str(ref.path): ref.sha256 for ref in (*refs, request.manifest)},
        "retirementDomainVerified": True,
        "initialAndTerminalBytesVerified": True,
        "sourceAbsenceObserved": phase == "after-delete",
        "deletionPerformed": False,
        "releaseDecision": None,
        "qualityApproved": False,
        "deliveryApproved": False,
        "physicalIOBytes": None,
        "metricScope": METRIC_SCOPE,
    }
    if not _matches({key: value for key, value in raw.items() if key not in COUNTERS}, stable):
        raise ReplayError("Readonly retirement receipt differs from the completed batch")
    for key in COUNTERS:
        value = raw.get(key)
        if type(value) is not int or value < 0:
            raise ReplayError("Completed retirement counters must remain nonnegative integers")


def _checkpoint_check(
    request: Request, checkpoint: Checkpoint, part: batch.Part, *, phase: batch.Phase
) -> None:
    expected: dict[str, object] = {
        "version": 1,
        "manifest": request.manifest.json(),
        "phase": phase,
        "part": part.number,
        "verification": part.config.json(),
        "retirementDomainVerified": True,
    }
    if not _matches(batch._read(checkpoint.artifact), expected):
        raise ReplayError("Readonly retirement checkpoint differs from the completed part")


def verify(request: Request) -> Verification:
    """Freshly check every original domain and pinned artifact without file writes."""
    with _locked(request) as (handle, directory, lock):
        phase, parts, refs = batch._manifest(request.manifest.path, request.manifest.sha256)
        if tuple(row.part for row in request.checkpoints) != tuple(part.number for part in parts):
            raise ReplayError("Readonly checkpoints must preserve complete manifest order")
        inputs = _inputs(request, refs)
        cache = InputHashCache((ref.path for ref in inputs), allow_uncached_overflow=True)
        cache.prime()
        with input_hash_context(cache):
            if any(file_digest(ref.path) != ref.sha256 for ref in inputs):
                raise ReplayError("Readonly retirement input digest changed")
            if load(request.artifact.path) != request:
                raise ReplayError("Readonly retirement request model differs from its pinned bytes")
            if batch._manifest(request.manifest.path, request.manifest.sha256) != (
                phase,
                parts,
                refs,
            ):
                raise ReplayError("Readonly retirement discovery changed after initial hashing")
            operation = {"version": 1, "manifest": request.manifest.json(), "phase": phase}
            if not _matches(batch._read(request.operation), operation):
                raise ReplayError("Readonly retirement operation differs from the manifest")
            _receipt(request, phase=phase, parts=parts, refs=refs)
            for checkpoint, part in zip(request.checkpoints, parts, strict=True):
                _checkpoint_check(request, checkpoint, part, phase=phase)
                cache.guard()
                _guard(request, handle, directory=directory, lock=lock)
                batch._source(part.source, phase)
                publication.verify_retirement(part.config.path)
                batch._source(part.source, phase)
                cache.guard()
                _guard(request, handle, directory=directory, lock=lock)
            verified = cache.verify()
            if verified != {str(ref.path): ref.sha256 for ref in inputs}:
                raise ReplayError("Readonly retirement terminal input inventory changed")
            for part in parts:
                batch._source(part.source, phase)
            cache.guard()
            _guard(request, handle, directory=directory, lock=lock)
            return Verification(
                phase,
                tuple(part.number for part in parts),
                request.receipt,
                verified,
                cache.hash_reads,
                cache.bytes_read,
                cache.cache_hits,
            )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("request", type=Path)
    args = parser.parse_args()
    print(json.dumps(verify(load(args.request)).json(), sort_keys=True))


if __name__ == "__main__":
    main()
