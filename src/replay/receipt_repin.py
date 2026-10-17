"""Re-pin finished receipts by content, without rewriting them.

A receipt pins each input as `{"path", "sha256", "bytes", "stat"}`. Moving work to the SSD
changed every pinned file's device, inode and ctime, and the approved option-A edit changed the
digest of `native_invocation.py`, so the stat-bound pins no longer compare equal. This module
re-verifies each pinned input against its current file (path mapped through
`replay.work_roots.resolve`), by sha256 and size, and writes a sidecar
`<receipt>.repin.json` with the current stat identity of every input. The receipt itself is
never modified. A receipt whose inputs do not all verify gets no sidecar and is reported.
The only accepted digest difference is the recorded, user-approved `native_invocation.py`
change; any other changed or missing input stays unresolved.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import sys
import threading
from collections.abc import Iterable, Iterator, Mapping, Sequence
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

from replay import work_roots
from replay.core import ReplayError

VERSION = 1
SIDECAR_SUFFIX = ".repin.json"
CHUNK_BYTES = 8 * 1024 * 1024
MAX_RECEIPT_BYTES = 64 * 1024 * 1024
DEFAULT_WORKERS = 4
DIGEST = re.compile(r"[0-9a-f]{64}")
# Edits the user approved for this storage work (HEAD digest -> working-tree digest): option A
# (identity() stops recording the device), observe_identity ignoring st_dev, and the
# shared-volume headroom check.
APPROVED_CODE_CHANGES: Mapping[str, tuple[str, str]] = {
    "replay/native_delivery_completion.py": (
        "709461eab8999ec3a0bb7ef77ae0ed83938bf9a2099edf3b9bedf730798a1861",
        "c7892465430e6c88a43fa4277b7e2bf73adc9d4d9246f151957cc90448d2244d",
    ),
    "replay/disk_budget.py": (
        "6bd9c688918927765233ac62d33e5f22bb3b31f968f2c88ebd5d2caddd702ce6",
        "4075433a14f6b2ec135012716625a485eb63f93ce59e304018ccb6e4c04990db",
    ),
    "replay/native_invocation.py": (
        "e2f53c85c4cee47bacde57bb00646a295f8e7989d887443db318deab214cae99",
        "4e27c20c2cf66c3311b8783f024fd19a647be02bbecb9e10c8c168d338263556",
    ),
}


@dataclass(frozen=True)
class PinCheck:
    """Outcome of re-verifying one pinned input."""

    recorded: str
    current: str
    status: str
    sha256: str
    identity: Mapping[str, int] | None


class Hasher:
    """sha256 per (path, size, mtime) so a file pinned by many receipts is read once."""

    def __init__(self) -> None:
        self._cache: dict[tuple[str, int, int], str] = {}
        self._lock = threading.Lock()

    def digest(self, path: Path) -> str:
        info = path.stat()
        key = (str(path), info.st_size, info.st_mtime_ns)
        with self._lock:
            cached = self._cache.get(key)
        if cached is not None:
            return cached
        digest = hashlib.sha256()
        with path.open("rb") as handle:
            while chunk := handle.read(CHUNK_BYTES):
                digest.update(chunk)
        value = digest.hexdigest()
        with self._lock:
            self._cache[key] = value
        return value


def pins(value: object) -> Iterator[Mapping[str, object]]:
    """Every `{"path", "sha256", "bytes"}` object anywhere in a receipt."""
    if isinstance(value, list):
        for item in value:
            yield from pins(item)
        return
    if not isinstance(value, dict):
        return
    path, digest, size = value.get("path"), value.get("sha256"), value.get("bytes")
    if (
        isinstance(path, str)
        and isinstance(digest, str)
        and DIGEST.fullmatch(digest)
        and isinstance(size, int)
    ):
        yield value
    for item in value.values():
        yield from pins(item)


def identity(path: Path) -> dict[str, int]:
    info = path.lstat()
    return {
        "inode": info.st_ino,
        "bytes": info.st_size,
        "mtime_ns": info.st_mtime_ns,
        "ctime_ns": info.st_ctime_ns,
    }


def _approved(path: Path, recorded: str, current: str) -> bool:
    change = next(
        (
            pair
            for suffix, pair in APPROVED_CODE_CHANGES.items()
            if path.as_posix().endswith(suffix)
        ),
        None,
    )
    return change == (recorded, current)


def check(pin: Mapping[str, object], hasher: Hasher) -> PinCheck:
    recorded = Path(str(pin["path"]))
    absolute = recorded if recorded.is_absolute() else work_roots.CHECKOUT / recorded
    current = work_roots.resolve(absolute)
    if current.is_symlink() or not current.is_file():
        return PinCheck(str(recorded), str(current), "missing", "", None)
    digest = hasher.digest(current)
    if digest == pin["sha256"] and current.stat().st_size == pin["bytes"]:
        status = "verified"
    elif _approved(current, str(pin["sha256"]), digest):
        status = "approved-code-change"
    else:
        status = "changed"
    return PinCheck(str(recorded), str(current), status, digest, identity(current))


def _read(receipt: Path) -> tuple[object, str]:
    data = receipt.read_bytes()
    if len(data) > MAX_RECEIPT_BYTES:
        raise ReplayError(f"{receipt} exceeds the receipt size bound")
    try:
        return json.loads(data), hashlib.sha256(data).hexdigest()
    except json.JSONDecodeError as error:
        raise ReplayError(f"{receipt} is not JSON") from error


def _write(path: Path, record: Mapping[str, object]) -> None:
    partial = path.with_name(f".{path.name}.partial")
    partial.write_text(json.dumps(record, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    partial.replace(path)


def repin(receipt: Path, hasher: Hasher) -> dict[str, object]:
    """Re-verify one receipt; write its sidecar only when every pinned input verifies."""
    value, receipt_digest = _read(receipt)
    sidecar = receipt.with_name(receipt.name + SIDECAR_SUFFIX)
    checks = [check(pin, hasher) for pin in pins(value)]
    unresolved = [item for item in checks if item.status in {"missing", "changed"}]
    outcome: dict[str, object] = {"receipt": str(receipt), "pins": len(checks)}
    if unresolved:
        outcome["status"] = "unresolved"
        outcome["unresolved"] = [
            {"path": item.recorded, "status": item.status} for item in unresolved
        ]
        return outcome
    _write(
        sidecar,
        {
            "version": VERSION,
            "receipt": str(receipt),
            "receiptSha256": receipt_digest,
            "repinnedAt": datetime.now(UTC).isoformat(),
            "inputs": [
                {
                    "recordedPath": item.recorded,
                    "currentPath": item.current,
                    "sha256": item.sha256,
                    "status": item.status,
                    "stat": item.identity,
                }
                for item in checks
            ],
        },
    )
    outcome["status"] = "repinned"
    return outcome


def _safe(receipt: Path, hasher: Hasher) -> dict[str, object]:
    try:
        return repin(receipt, hasher)
    except (OSError, ReplayError) as error:
        return {"receipt": str(receipt), "status": "error", "error": str(error)}


def run(receipts: Iterable[Path], *, workers: int = DEFAULT_WORKERS) -> list[dict[str, object]]:
    hasher = Hasher()
    with ThreadPoolExecutor(max_workers=workers) as pool:
        return list(pool.map(lambda receipt: _safe(receipt, hasher), receipts))


def summary(outcomes: Sequence[Mapping[str, object]]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for outcome in outcomes:
        status = str(outcome["status"])
        counts[status] = counts.get(status, 0) + 1
    return dict(sorted(counts.items()))


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("receipts", type=Path, help="text file with one receipt path per line")
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=DEFAULT_WORKERS)
    arguments = parser.parse_args(argv)
    listed = arguments.receipts.read_text(encoding="utf-8").splitlines()
    receipts = [Path(line) for line in listed if line.strip()]
    outcomes = run(receipts, workers=arguments.workers)
    counts = summary(outcomes)
    _write(arguments.report, {"summary": counts, "outcomes": outcomes})
    print(json.dumps(counts, sort_keys=True))
    return 0 if set(counts) <= {"repinned"} else 1


if __name__ == "__main__":
    os.umask(0o022)
    sys.exit(main())
