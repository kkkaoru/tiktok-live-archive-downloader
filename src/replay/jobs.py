"""Replay-ID deduplication with cross-process locks and durable job records."""

import fcntl
import json
import os
import re
import tempfile
from collections.abc import Callable
from pathlib import Path

from replay.core import ReplayError, private_directory


def record_path(root: Path, replay_id: str) -> Path:
    if re.fullmatch(r"[0-9]{1,32}", replay_id) is None:
        raise ReplayError("Replay ID must contain 1-32 digits.")
    private_directory(root)
    return root / f"{replay_id}.json"


def write_record(path: Path, output: Path, status: str) -> None:
    size = output.stat().st_size if status == "complete" else 0
    with tempfile.NamedTemporaryFile(
        mode="w", encoding="utf-8", dir=path.parent, delete=False
    ) as stream:
        temporary = Path(stream.name)
        try:
            json.dump({"status": status, "output": str(output.resolve()), "size": size}, stream)
            stream.flush()
            os.fsync(stream.fileno())
            os.replace(temporary, path)
        finally:
            temporary.unlink(missing_ok=True)


def already_complete(path: Path) -> bool:
    if not path.exists():
        return False
    data: object = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ReplayError("Invalid replay job record.")
    status: object = data.get("status")
    output: object = data.get("output")
    size: object = data.get("size")
    if status not in ("running", "failed", "complete") or not isinstance(output, str):
        raise ReplayError("Invalid replay job record.")
    target = Path(output)
    if status == "complete":
        if (
            type(size) is not int
            or size <= 0
            or not target.is_file()
            or target.stat().st_size != size
        ):
            raise ReplayError(
                "Completed replay file is missing or changed; verify it before retrying."
            )
        return True
    if target.exists():
        raise ReplayError(
            "Interrupted replay has an output file; "
            "verify and register it instead of downloading again."
        )
    return False


def download_once(
    *, root: Path, replay_id: str, output: Path, download: Callable[[], None]
) -> bool:
    """Skip completed/busy IDs; download must publish its output atomically."""
    return prepared_download_once(
        root=root, replay_id=replay_id, prepare=lambda: (output, download)
    )


def prepared_download_once(
    *, root: Path, replay_id: str, prepare: Callable[[], tuple[Path, Callable[[], None]]]
) -> bool:
    """Resolve a metadata-derived output only after locking and checking completion."""
    path = record_path(root, replay_id)
    with path.with_suffix(".lock").open("a+b") as lock:
        os.fchmod(lock.fileno(), 0o600)
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            return False
        if already_complete(path):
            return False
        output, download = prepare()
        if output.exists():
            raise ReplayError("Output already exists and is not registered for this replay.")
        write_record(path, output, "running")
        try:
            download()
            if not output.is_file() or output.stat().st_size == 0:
                raise ReplayError("Download did not publish a nonempty video.")
            write_record(path, output, "complete")
        except BaseException:
            write_record(path, output, "failed")
            raise
        return True


def rename_completed(*, root: Path, replay_id: str, destination: Path) -> bool:
    """Rename a verified completed output without overwriting or invalidating its record.

    Link first, publish the new job path, then remove the old name. On interruption,
    at least one valid name remains; any leftover alias requires explicit review.
    """
    path = record_path(root, replay_id)
    with path.with_suffix(".lock").open("a+b") as lock:
        os.fchmod(lock.fileno(), 0o600)
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        if not already_complete(path):
            raise ReplayError("Only a verified completed recording can be renamed.")
        data: object = json.loads(path.read_text(encoding="utf-8"))
        output: object = data.get("output") if isinstance(data, dict) else None
        if not isinstance(output, str):
            raise ReplayError("Invalid replay job record.")
        source = Path(output)
        destination = destination.parent.resolve() / destination.name
        if source.is_symlink() or destination.parent != source.parent:
            raise ReplayError("Rename requires a regular file in the same directory.")
        if source == destination:
            return False
        # link() atomically refuses existing paths, including dangling symlinks.
        os.link(source, destination)
        write_record(path, destination, "complete")
        source.unlink()
        return True


def register_completed(*, root: Path, replay_id: str, output: Path) -> None:
    """Adopt an existing video only after the caller independently verifies it."""
    path = record_path(root, replay_id)
    with path.with_suffix(".lock").open("a+b") as lock:
        os.fchmod(lock.fileno(), 0o600)
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        if not output.is_file() or output.stat().st_size == 0:
            raise ReplayError("Cannot register a missing or empty video.")
        write_record(path, output, "complete")
