"""Pin a readonly request from an existing complete retirement batch.

Only bounded metadata is read. This bridge never runs a retirement domain,
creates batch checkpoints, hashes media, authorizes deletion or repairs outputs.
"""

import argparse
import hashlib
import json
import os
import stat
from contextlib import ExitStack
from pathlib import Path

from replay import part_retirement_verification_batch as batch
from replay import part_retirement_verification_readonly as readonly
from replay.core import ReplayError
from replay.input_hash_cache import InputHashCache, input_hash_context
from replay.stages import file_digest


def _pin(path: Path) -> batch.Reference:
    path = batch._path(str(path), work=True)
    value = path.lstat()
    identity = readonly._identity(value)
    if (
        not stat.S_ISREG(value.st_mode)
        or value.st_nlink != 1
        or not 0 < value.st_size <= batch.MAX_JSON_BYTES
    ):
        raise ReplayError("Request inputs must be bounded single-link metadata files")
    with os.fdopen(os.open(path, os.O_RDONLY | os.O_NOFOLLOW), "rb") as handle:
        if readonly._identity(os.fstat(handle.fileno())) != identity:
            raise ReplayError("Request metadata changed before opening")
        data = handle.read(batch.MAX_JSON_BYTES + 1)
        if (
            len(data) > batch.MAX_JSON_BYTES
            or readonly._identity(os.fstat(handle.fileno())) != identity
        ):
            raise ReplayError("Request metadata changed during its bounded read")
    if batch._path(str(path), work=True) != path or readonly._identity(path.lstat()) != identity:
        raise ReplayError("Request metadata path changed after reading")
    return batch.Reference(path, hashlib.sha256(data).hexdigest())


def _parent(output: Path, descriptor: int) -> None:
    path = batch._path(str(output.parent), work=True)
    value = path.lstat()
    opened = os.fstat(descriptor)
    if not stat.S_ISDIR(value.st_mode) or (value.st_dev, value.st_ino, value.st_mode) != (
        opened.st_dev,
        opened.st_ino,
        opened.st_mode,
    ):
        raise ReplayError("Request output parent identity changed")


def _publish(output: Path, data: bytes) -> None:
    # A directory descriptor prevents a replaced parent alias from redirecting Save.
    with ExitStack() as stack:
        parent = os.open(output.parent, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW)
        stack.callback(os.close, parent)
        _parent(output, parent)
        descriptor = os.open(
            output.name,
            os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW,
            0o600,
            dir_fd=parent,
        )
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(data)
            handle.flush()
            os.fsync(handle.fileno())
            value = os.fstat(handle.fileno())
            _parent(output, parent)
            if (
                batch._path(str(output), work=True) != output
                or value.st_nlink != 1
                or value.st_mode & 0o077
                or readonly._identity(output.lstat()) != readonly._identity(value)
            ):
                raise ReplayError("New request output identity changed")
        os.fsync(parent)
        _parent(output, parent)


def _value(
    manifest: batch.Reference,
    *,
    operation: batch.Reference,
    receipt: batch.Reference,
    checkpoints: tuple[readonly.Checkpoint, ...],
) -> dict[str, object]:
    return {
        "version": 1,
        "kind": "readonly-retirement-verification",
        "manifest": manifest.json(),
        "operation": operation.json(),
        "receipt": receipt.json(),
        "checkpoints": [{"part": row.part, "artifact": row.artifact.json()} for row in checkpoints],
    }


def prepare(
    manifest: Path,
    expected_sha256: str,
    *,
    completed: Path,
    output: Path,
) -> Path:
    """Exclusively create, or preserve identical, a metadata-only verification request."""
    manifest = batch._path(str(manifest), work=True)
    completed = batch._path(str(completed), work=True)
    output = batch._path(str(output), work=True)
    expected = batch._ref({"path": str(manifest), "sha256": expected_sha256})
    if _pin(manifest) != expected:
        raise ReplayError("Request manifest differs from its expected SHA256")
    if output.is_relative_to(completed) or not output.parent.is_dir():
        raise ReplayError("Request output must have an existing parent outside the batch directory")
    phase, parts, refs = batch._manifest(manifest, expected_sha256)
    operation = _pin(completed / "operation.json")
    receipt = _pin(completed / "receipt.json")
    checkpoints = tuple(
        readonly.Checkpoint(part.number, _pin(completed / f"part-{part.number:04d}.json"))
        for part in parts
    )
    value = _value(expected, operation=operation, receipt=receipt, checkpoints=checkpoints)
    data = (json.dumps(value, ensure_ascii=False, indent=2) + "\n").encode("utf-8")
    if len(data) > batch.MAX_JSON_BYTES:
        raise ReplayError("Generated request exceeds the bounded metadata budget")
    # This planned output reference is checked against actual bytes after Save.
    request = readonly.Request(
        batch.Reference(output, hashlib.sha256(data).hexdigest()),
        expected,
        operation,
        receipt,
        checkpoints,
    )
    metadata = (expected, operation, receipt, *(row.artifact for row in checkpoints))
    with readonly._locked(request) as (handle, directory, lock):
        cache = InputHashCache((ref.path for ref in metadata), allow_uncached_overflow=True)
        cache.prime()
        with input_hash_context(cache):
            if any(file_digest(ref.path) != ref.sha256 for ref in metadata):
                raise ReplayError("Request batch metadata changed after pinning")
            if batch._manifest(manifest, expected_sha256) != (phase, parts, refs):
                raise ReplayError("Request batch discovery changed")
            expected_operation = {"version": 1, "manifest": expected.json(), "phase": phase}
            if not readonly._matches(batch._read(operation), expected_operation):
                raise ReplayError("Request operation differs from the complete manifest")
            readonly._receipt(request, phase=phase, parts=parts, refs=refs)
            for checkpoint, part in zip(checkpoints, parts, strict=True):
                readonly._checkpoint_check(request, checkpoint, part, phase=phase)
            cache.verify()
            if batch._manifest(manifest, expected_sha256) != (phase, parts, refs):
                raise ReplayError("Request batch discovery changed at the terminal boundary")
            cache.guard()
            readonly._guard(request, handle, directory=directory, lock=lock)
            if output.exists():
                if _pin(output) != request.artifact:
                    raise ReplayError("Occupied request output must preserve exact identical bytes")
                if output.stat().st_mode & 0o077:
                    raise ReplayError("Occupied request output must retain private permissions")
            else:
                _publish(output, data)
            if readonly.load(output) != request:
                raise ReplayError("Request output readback differs from its planned pins")
            cache.guard()
    return output


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("manifest", type=Path)
    parser.add_argument("expected_sha256")
    parser.add_argument("completed", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    print(
        prepare(args.manifest, args.expected_sha256, completed=args.completed, output=args.output)
    )


if __name__ == "__main__":
    main()
