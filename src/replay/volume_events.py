"""Recovery of a step that failed because a volume detached under it.

The external SSD disconnected four times (10-22 04:07, 10-24 08:01, 18:24:57 and 18:54:16, all
logged by diskarbitrationd). Writes then failed with FileNotFoundError, EIO, ENXIO or ENODEV, and
Compressor, holding a handle to the old mount, failed every later job ("結果フォルダ…にアクセスで
きません"). Before such a step runs again (``step_retry`` allows one retry):

- ``await_return``: every volume mounted at the attempt's start is mounted again, with the same
  volume UUID (a different disk at the same path is refused);
- a detach (the mount's device changed or was missing) while Compressor runs restarts it
  (``compressor_guard.restart``);
- ``recheck``: receipts written shortly before the failure are re-hashed against the files they
  pin; a receipt whose file changed or vanished is moved aside, so its work is done again.
"""

from __future__ import annotations

import hashlib
import json
import os
import plistlib
import subprocess
import time
from collections.abc import Callable, Iterator
from dataclasses import dataclass
from pathlib import Path

from replay import compressor_batch, compressor_guard

VOLUMES = Path("/Volumes")
WAIT_SECONDS = 600.0
POLL_SECONDS = 5.0
RECENT_SECONDS = 300.0  # receipts written this long before a failure are checked again
DISKUTIL = "/usr/sbin/diskutil"


@dataclass(frozen=True, slots=True)
class Volume:
    uuid: str
    device: int


Snapshot = dict[str, Volume]


def volume_uuid(mount: Path) -> str | None:
    """The mounted volume's UUID (``diskutil info``), or None when it cannot be read."""
    done = subprocess.run(
        [DISKUTIL, "info", "-plist", str(mount)], capture_output=True, check=False, timeout=30
    )
    try:
        value = plistlib.loads(done.stdout).get("VolumeUUID")
    except plistlib.InvalidFileException:
        return None
    return value if isinstance(value, str) else None


def snapshot(
    root: Path = VOLUMES,
    *,
    uuid: Callable[[Path], str | None] = volume_uuid,
    mounted: Callable[[Path], bool] = os.path.ismount,
) -> Snapshot:
    """The mounted volumes under ``root`` (the boot volume's symlink excluded)."""
    found: Snapshot = {}
    for path in sorted(root.iterdir()) if root.is_dir() else ():
        if path.is_symlink() or not mounted(path):
            continue
        identity = uuid(path)
        if identity is not None:
            found[str(path)] = Volume(identity, path.stat().st_dev)
    return found


def detached(before: Snapshot, after: Snapshot) -> bool:
    """True when a volume of ``before`` is missing or was mounted again (a new device)."""
    return any(
        after.get(mount) is None or after[mount].device != volume.device
        for mount, volume in before.items()
    )


def await_return(
    before: Snapshot,
    *,
    current: Callable[[], Snapshot] = snapshot,
    sleep: Callable[[float], None] = time.sleep,
    clock: Callable[[], float] = time.monotonic,
    wait: float = WAIT_SECONDS,
) -> str | None:
    """None once every volume of ``before`` is mounted with its UUID again; else why not."""
    deadline = clock() + wait
    while True:
        now = current()
        for mount, volume in before.items():
            if mount in now and now[mount].uuid != volume.uuid:
                return f"{mount} is a different volume now"
        if all(mount in now for mount in before):
            return None
        if clock() >= deadline:
            missing = sorted(mount for mount in before if mount not in now)
            return f"{', '.join(missing)} did not return within {wait:.0f} s"
        sleep(POLL_SECONDS)


def file_sha256(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def _pins(value: object) -> Iterator[tuple[str, str]]:
    if isinstance(value, dict):
        path, digest = value.get("path"), value.get("sha256")
        if isinstance(path, str) and isinstance(digest, str):
            yield path, digest
        for item in value.values():
            yield from _pins(item)
    elif isinstance(value, list):
        for item in value:
            yield from _pins(item)


def _intact(receipt: Path, hasher: Callable[[Path], str]) -> bool:
    try:
        data = json.loads(receipt.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return False  # a receipt cut off by the detach
    return all(
        Path(path).is_file() and hasher(Path(path)) == digest for path, digest in _pins(data)
    )


def recheck(
    roots: tuple[Path, ...], *, since: float, hasher: Callable[[Path], str] = file_sha256
) -> list[Path]:
    """Move aside each receipt under ``roots`` written at or after ``since`` (epoch seconds)
    whose pinned files changed or vanished; returns where they went."""
    retired = []
    for root in roots:
        for receipt in sorted(root.rglob("*.json")) if root.is_dir() else ():
            if not receipt.is_file() or receipt.stat().st_mtime < since:
                continue
            if not _intact(receipt, hasher):
                aside = receipt.with_name(f"{receipt.stem}.unverified-{time.time_ns()}.json")
                receipt.rename(aside)
                retired.append(aside)
    return retired


def compressor_running() -> bool:
    found = subprocess.run(["/usr/bin/pgrep", "-x", "Compressor"], capture_output=True, check=False)
    return found.returncode == 0


def restart_compressor() -> None:
    compressor_guard.restart(compressor_batch.COMPRESSOR.parents[2])


def recover(
    before: Snapshot,
    roots: tuple[Path, ...],
    *,
    failed_at: float,
    current: Callable[[], Snapshot] = snapshot,
    running: Callable[[], bool] = compressor_running,
    restart: Callable[[], None] = restart_compressor,
    sleep: Callable[[float], None] = time.sleep,
    clock: Callable[[], float] = time.monotonic,
    hasher: Callable[[Path], str] = file_sha256,
) -> str | None:
    """Make the failed step safe to run again; None when it is, else why not.

    ``failed_at`` is the failure's wall-clock time (epoch seconds).
    """
    gone = detached(before, current())  # at the failure: missing, or already mounted again
    reason = await_return(before, current=current, sleep=sleep, clock=clock)
    if reason is not None:
        return reason
    if (gone or detached(before, current())) and running():
        restart()  # it still holds a handle to the old mount
    for aside in recheck(roots, since=failed_at - RECENT_SECONDS, hasher=hasher):
        print(f"receipt moved aside after a detach: {aside}", flush=True)
    return None
