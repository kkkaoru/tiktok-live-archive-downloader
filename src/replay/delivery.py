"""Stable processed-delivery names from source-bound broadcast evidence."""

import fcntl
import json
import os
import tempfile
from collections.abc import Iterator
from contextlib import contextmanager
from datetime import datetime
from pathlib import Path

from replay.core import ReplayError
from replay.edit_completion import is_superseded
from replay.recordings import record
from replay.stages import file_digest


def delivery_stem(replay_root: Path) -> str:
    """Reject missing dates before creating any processing directory or media."""
    probe = replay_root / "evidence" / "source-probe.json"
    if probe.is_symlink() or probe.resolve().parent != replay_root.resolve() / "evidence":
        raise ReplayError("Broadcast evidence must remain inside its source directory")
    try:
        if probe.stat().st_size > 1_048_576:
            raise ReplayError("Broadcast evidence exceeds the size limit")
        data = record(json.loads(probe.read_text(encoding="utf-8")))
        filename = record(data.get("format")).get("filename")
        if not isinstance(filename, str):
            raise ReplayError("Missing source filename in broadcast evidence")
        if not Path(filename).is_absolute() and filename != f"downloads/{Path(filename).name}":
            raise ReplayError("Unscoped source filename in broadcast evidence")
        suffix = f"_JST_{replay_root.name}.mp4"
        name = Path(filename).name
        if not replay_root.name.startswith("replay-") or not name.endswith(suffix):
            raise ReplayError("Broadcast evidence belongs to another source")
        stamp = name.removesuffix(suffix)
        date = datetime.strptime(stamp, "%Y-%m-%d_%H-%M-%S")
        if date.strftime("%Y-%m-%d_%H-%M-%S") != stamp:
            raise ReplayError("Broadcast date must use its canonical format")
    except (OSError, UnicodeError, ValueError) as error:
        raise ReplayError(
            "Missing or invalid broadcast date; refusing an undated delivery"
        ) from error
    return f"{name.removesuffix('.mp4')}-review"


def require_new_delivery(replay_root: Path) -> str:
    """Return the stable name only when this source has no dated final delivery."""
    stem = delivery_stem(replay_root)
    suffix = f"_JST_{replay_root.name}-review"
    deliveries = list((replay_root / "revisions").glob(f"**/delivery/*{suffix}.mp4"))
    if any(not is_superseded(replay_root, path) for path in deliveries) or (
        not deliveries and any((replay_root / "evidence").glob(f"*{suffix}-selection.json"))
    ):
        raise ReplayError("This source already has a dated delivery; refusing another revision")
    return stem


def retire_processed_source(source: Path, output: Path, before: os.stat_result) -> None:
    """Retire only the unchanged legacy delivery after recording its verified successor."""
    root = output.parents[3]
    if (
        source.name != "full-review.mp4"
        or source.parent.name != "delivery"
        or not source.is_relative_to(root / "revisions")
        or source.resolve() != source
        or output.resolve() != output
        or output.name != f"{delivery_stem(root)}.mp4"
        or not output.is_file()
        or output.stat().st_size == 0
    ):
        raise ReplayError("Only a scoped processed delivery may be retired")
    current = source.stat()
    if (current.st_dev, current.st_ino, current.st_size, current.st_mtime_ns) != (
        before.st_dev,
        before.st_ino,
        before.st_size,
        before.st_mtime_ns,
    ):
        raise ReplayError("Source delivery changed during processing; keep both for inspection")
    digest = file_digest(output)
    marker = root / "evidence" / f"{output.stem}-selection.json"
    if marker.exists() or marker.is_symlink():
        # A superseded incomplete delivery kept the plain marker; never replace it.
        marker = marker.with_name(f"{output.stem}-selection-{digest[:16]}.json")
    receipt = {
        "status": "selected-verified-successor",
        "retired_source": str(source),
        "delivery": str(output),
        "sha256": digest,
        "size": output.stat().st_size,
    }
    with tempfile.NamedTemporaryFile(mode="w", encoding="utf-8", dir=marker.parent) as stream:
        json.dump(receipt, stream, sort_keys=True)
        stream.flush()
        os.fsync(stream.fileno())
        os.link(stream.name, marker)
    if source.stat() != current:
        raise ReplayError("Source delivery changed before retirement; keep both for inspection")
    source.unlink()


@contextmanager
def delivery_lock(replay_root: Path) -> Iterator[None]:
    """Serialize all revision families for a single source; never unlink the lock."""
    delivery_stem(replay_root)
    path = replay_root / "evidence" / f"{replay_root.name}.delivery.lock"
    descriptor = os.open(path, os.O_CREAT | os.O_RDWR | os.O_NOFOLLOW, 0o600)
    with os.fdopen(descriptor, "a+b") as stream:
        try:
            fcntl.flock(stream, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as error:
            raise ReplayError("Another process owns this source delivery") from error
        try:
            yield
        finally:
            fcntl.flock(stream, fcntl.LOCK_UN)
