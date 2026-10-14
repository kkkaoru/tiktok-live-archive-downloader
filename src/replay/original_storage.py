"""Organize completed originals by explicit same-volume publication, preserving old names.

Publication and completion-record replacement are separate operations. An
immutable intent receipt permits reconciliation after either interruption;
neither original aliases nor historical reference records are deleted here.
"""

import argparse
import ctypes
import fcntl
import json
import os
import re
import stat
from pathlib import Path

from replay.core import ReplayError
from replay.jobs import already_complete, file_sha256, write_record
from replay.recordings import record

CLONE_NOFOLLOW_ANY = 0x0008


def _canonical(path: Path, *, exists: bool) -> Path:
    absolute = path.absolute()
    if absolute.resolve(strict=exists) != absolute:
        raise ReplayError("Storage paths must be canonical and must not traverse symlinks")
    return absolute


def _read(path: Path) -> dict[str, object]:
    if path.is_symlink() or not path.is_file() or path.stat().st_size > 65536:
        raise ReplayError("Expected bounded regular storage record")
    return record(json.loads(path.read_text(encoding="utf-8")))


def _identity(path: Path, *, method: str = "hardlink") -> dict[str, int]:
    value = path.lstat()
    if not stat.S_ISREG(value.st_mode) or value.st_size <= 0:
        raise ReplayError("Original must be a nonempty regular file")
    identity = {
        "device": value.st_dev,
        "inode": value.st_ino,
        "size": value.st_size,
        "mtimeNs": value.st_mtime_ns,
    }
    if method == "clone":
        if value.st_nlink != 1:
            raise ReplayError("Clone source must retain a single link")
        identity.update(ctimeNs=value.st_ctime_ns, links=value.st_nlink)
    return identity


def _clone_file(source: Path, destination: Path) -> None:
    """Darwin clonefile only: atomic exclusive publication, never a copying fallback."""
    library = ctypes.CDLL("/usr/lib/libSystem.B.dylib", use_errno=True)
    clone = library.clonefile
    clone.argtypes = (ctypes.c_char_p, ctypes.c_char_p, ctypes.c_uint32)
    clone.restype = ctypes.c_int
    # SDK sys/clonefile.h: reject symlinks in every pathname component.
    if clone(os.fsencode(source), os.fsencode(destination), CLONE_NOFOLLOW_ANY) != 0:
        error = ctypes.get_errno()
        raise OSError(error, os.strerror(error), str(destination))


def inspect(
    *,
    root: Path,
    replay_id: str,
    source: Path,
    destination: Path,
    expected_sha256: str,
    method: str = "hardlink",
) -> dict[str, object]:
    """Read-only preflight: no directories, links, locks or receipts are created."""
    if method not in ("hardlink", "clone"):
        raise ReplayError("Unsupported original publication method")
    if (
        re.fullmatch(r"[0-9]{1,32}", replay_id) is None
        or re.fullmatch(r"[0-9a-f]{64}", expected_sha256) is None
    ):
        raise ReplayError("Expected stable replay ID and original SHA256")
    root = _canonical(root, exists=True)
    source = _canonical(source, exists=True)
    destination = _canonical(destination, exists=False)
    parent = _canonical(destination.parent, exists=True)
    if not root.is_dir() or not parent.is_dir() or source == destination:
        raise ReplayError("Expected distinct original names and existing directories")
    job = root / f"{replay_id}.json"
    raw = _read(job)
    identity = _identity(source, method=method)
    if (
        raw.get("status") != "complete"
        or raw.get("output") not in (str(source), str(destination))
        or raw.get("size") != identity["size"]
        or not already_complete(job)
    ):
        raise ReplayError("Completed recording does not match the proposed storage mapping")
    if parent.stat().st_dev != identity["device"]:
        raise ReplayError("Same-volume organization cannot copy across devices")
    if destination.exists() or destination.is_symlink():
        destination_identity = _identity(destination, method=method)
        if method == "hardlink" and not os.path.samefile(source, destination):
            raise ReplayError("Destination exists and is not the same original inode")
        if method == "clone" and (
            os.path.samefile(source, destination)
            or destination_identity["device"] != identity["device"]
            or file_sha256(destination) != expected_sha256
            or _identity(destination, method=method) != destination_identity
        ):
            raise ReplayError("Clone destination is not an independent matching original")
    if file_sha256(source) != expected_sha256 or _identity(source, method=method) != identity:
        raise ReplayError("Original content or identity changed during preflight")
    result: dict[str, object] = {
        "version": 1,
        "replayId": replay_id,
        "job": str(job),
        "source": str(source),
        "destination": str(destination),
        "sha256": expected_sha256,
        "identity": identity,
    }
    if method == "clone":
        result["method"] = method
    return result


def organize(
    *,
    root: Path,
    replay_id: str,
    source: Path,
    destination: Path,
    expected_sha256: str,
    receipt: Path,
    method: str = "hardlink",
) -> dict[str, object]:
    """Explicit apply; preserve old name and reconcile only this exact saved intent."""
    planned = inspect(
        root=root,
        replay_id=replay_id,
        source=source,
        destination=destination,
        expected_sha256=expected_sha256,
        method=method,
    )
    job = root.absolute() / f"{replay_id}.json"
    receipt = _canonical(receipt, exists=False)
    _canonical(receipt.parent, exists=True)
    lock_path = job.with_suffix(".lock")
    if receipt in (job, lock_path, source.absolute(), destination.absolute()):
        raise ReplayError("Storage receipt must be separate from originals and job records")
    descriptor = os.open(lock_path, os.O_RDWR | os.O_CREAT | os.O_NOFOLLOW, 0o600)
    with os.fdopen(descriptor, "a+b") as lock:
        if (
            not stat.S_ISREG(os.fstat(lock.fileno()).st_mode)
            or os.fstat(lock.fileno()).st_nlink != 1
        ):
            raise ReplayError("Storage lock must be a single-link regular file")
        os.fchmod(lock.fileno(), 0o600)
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        current = inspect(
            root=root,
            replay_id=replay_id,
            source=source,
            destination=destination,
            expected_sha256=expected_sha256,
            method=method,
        )
        if current != planned:
            raise ReplayError("Original changed before storage lock acquisition")
        if receipt.exists():
            intent = _read(receipt)
            if intent.get("mapping") != planned:
                raise ReplayError("Existing storage intent belongs to a different mapping")
            recorded = _read(job)
            published = {
                "status": "complete",
                "output": str(destination.absolute()),
                "size": source.stat().st_size,
            }
            if file_sha256(job) != intent.get("originalJobSha256") and recorded != published:
                raise ReplayError("Completion record changed outside this storage intent")
        else:
            if destination.exists():
                raise ReplayError("Existing alias without this intent requires independent review")
            with receipt.open("x", encoding="utf-8") as stream:
                os.fchmod(stream.fileno(), 0o600)
                json.dump(
                    {
                        "mapping": planned,
                        "originalJobSha256": file_sha256(job),
                        "oldAliasMustRemain": True,
                    },
                    stream,
                    sort_keys=True,
                )
                stream.flush()
                os.fsync(stream.fileno())
        if not destination.exists():
            if method == "clone":
                _clone_file(source, destination)
            else:
                os.link(source, destination, follow_symlinks=False)
        verified = inspect(
            root=root,
            replay_id=replay_id,
            source=source,
            destination=destination,
            expected_sha256=expected_sha256,
            method=method,
        )
        if verified != planned:
            raise ReplayError("Original changed during storage publication")
        if _read(job).get("output") != str(destination.absolute()):
            write_record(job, destination, "complete")
        final = inspect(
            root=root,
            replay_id=replay_id,
            source=source,
            destination=destination,
            expected_sha256=expected_sha256,
            method=method,
        )
        if (
            final != planned
            or _read(job).get("output") != str(destination.absolute())
            or (method == "hardlink" and not os.path.samefile(source, destination))
        ):
            raise ReplayError("Published original or completion record failed verification")
        result: dict[str, object] = {
            "version": 1,
            "status": "organized-with-old-alias-retained",
            "mapping": planned,
            "receipt": str(receipt),
            "additionalMediaBytes": 0,
            "oldAliasDeleted": False,
        }
        if method == "clone":
            result.update(
                method=method,
                storageSemantics="clonefile copy-on-write; metadata allocation not measured",
                sourceIdentityUnchanged=True,
                destinationSingleLink=True,
            )
        return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("check", "apply"))
    parser.add_argument("root", type=Path)
    parser.add_argument("replay_id")
    parser.add_argument("source", type=Path)
    parser.add_argument("destination", type=Path)
    parser.add_argument("sha256")
    parser.add_argument("--receipt", type=Path)
    parser.add_argument("--method", choices=("hardlink", "clone"), default="hardlink")
    args = parser.parse_args()
    if args.mode == "apply":
        if args.receipt is None:
            parser.error("apply requires --receipt for interruption reconciliation")
        result = organize(
            root=args.root,
            replay_id=args.replay_id,
            source=args.source,
            destination=args.destination,
            expected_sha256=args.sha256,
            receipt=args.receipt,
            method=args.method,
        )
    else:
        result = inspect(
            root=args.root,
            replay_id=args.replay_id,
            source=args.source,
            destination=args.destination,
            expected_sha256=args.sha256,
            method=args.method,
        )
    print(json.dumps(result, sort_keys=True))


if __name__ == "__main__":
    main()
