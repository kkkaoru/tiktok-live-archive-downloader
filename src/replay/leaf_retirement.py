"""Retire exactly the leaves an owner approved, with per-row checks and a log.

The input is an owner-signed disposition file whose SHA-256 the caller pins. Only rows
classed `eligible` are considered; nothing is discovered by pattern, no directory is
removed and no symlink target is followed. Immediately before each unlink the row is
checked against the live file: allowed root, canonical parent, device, inode, link
count, size, mtime, ctime, symlink target and open handles. Any mismatch holds only
that row. Open handles are probed with one lsof call per chunk of exact paths and fail
closed. The default is a dry run.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import stat
import subprocess
from collections import Counter
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import TextIO

from replay.core import ReplayError

ELIGIBLE = "eligible"
LSOF_TIMEOUT_SECONDS = 900
CHUNK_SIZE = 500


@dataclass(frozen=True, slots=True)
class Row:
    path: str
    kind: str
    dev: int
    inode: int
    nlink: int
    size: int
    mtime_ns: int
    ctime_ns: int
    link_target: str | None


@dataclass(frozen=True, slots=True)
class Summary:
    eligible: int
    retired: int
    logical_bytes: int
    held: dict[str, int]


OpenProbe = Callable[[str], bool | None]
BatchProbe = Callable[[Sequence[str]], set[str] | None]


def sha256_of(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _integer(raw: dict[str, object], name: str) -> int:
    value = raw.get(name)
    if type(value) is not int:
        raise ReplayError(f"Disposition field {name} must be an integer")
    return value


def _row(raw: object) -> Row:
    if not isinstance(raw, dict):
        raise ReplayError("Disposition rows must be objects")
    path, kind = raw.get("path"), raw.get("kind")
    if not isinstance(path, str) or kind not in {"file", "symlink"}:
        raise ReplayError("Disposition row needs a path and a file or symlink kind")
    target = raw.get("linkTarget")
    if kind == "symlink" and not isinstance(target, str):
        raise ReplayError("Symlink rows need their exact target")
    size = _integer(raw, "bytes") if kind == "file" else 0
    return Row(
        path,
        str(kind),
        _integer(raw, "dev"),
        _integer(raw, "inode"),
        _integer(raw, "nlink"),
        size,
        _integer(raw, "mtimeNs"),
        _integer(raw, "ctimeNs"),
        target if isinstance(target, str) else None,
    )


def load_eligible(dispositions: Path, expected_sha256: str) -> tuple[Row, ...]:
    """Verify the pinned SHA-256, then return only the eligible rows."""
    if sha256_of(dispositions) != expected_sha256:
        raise ReplayError("Disposition file SHA-256 differs from the pinned value")
    document = json.loads(dispositions.read_text(encoding="utf-8"))
    rows = document.get("rows") if isinstance(document, dict) else None
    if not isinstance(rows, list):
        raise ReplayError("Disposition file needs a rows array")
    return tuple(
        _row(raw) for raw in rows if isinstance(raw, dict) and raw.get("classification") == ELIGIBLE
    )


def lsof_is_open(path: str) -> bool | None:
    """True when a process holds `path`, False when none does, None when lsof failed.

    lsof exits 1 with no output when nothing matches; any other outcome (a different exit
    status, stderr text or a timeout) is a failed probe and must never read as "closed".
    """
    try:
        done = subprocess.run(
            ["lsof", "-F", "n", "--", path],
            capture_output=True,
            text=True,
            timeout=LSOF_TIMEOUT_SECONDS,
            check=False,
        )
    except subprocess.TimeoutExpired:
        return None
    if done.returncode == 0 and done.stdout.strip():
        return True
    if done.returncode == 1 and not done.stdout.strip() and not done.stderr.strip():
        return False
    return None


def lsof_open_among(paths: Sequence[str]) -> set[str] | None:
    """The exact paths among `paths` that a process holds open; None when the probe failed.

    One lsof call covers a whole chunk. With several paths lsof exits 1 even when some are
    open, so the verdict comes from its output: names listed are open, an empty output with
    exit 1 and no stderr means none is open. Output naming anything outside the chunk, any
    stderr text, another exit status or a timeout make the probe unusable, and the caller
    must hold the entire chunk.
    """
    try:
        done = subprocess.run(
            ["lsof", "-F", "n", "--", *paths],
            capture_output=True,
            text=True,
            timeout=LSOF_TIMEOUT_SECONDS,
            check=False,
        )
    except subprocess.TimeoutExpired:
        return None
    if done.stderr.strip() or done.returncode not in {0, 1}:
        return None
    names = {line[1:] for line in done.stdout.splitlines() if line.startswith("n")}
    if not names:
        return set() if done.returncode == 1 and not done.stdout.strip() else None
    return names if names <= set(paths) else None


def _matches(row: Row, info: os.stat_result) -> bool:
    identity = (info.st_dev, info.st_ino, info.st_nlink, info.st_mtime_ns, info.st_ctime_ns)
    wanted = (row.dev, row.inode, row.nlink, row.mtime_ns, row.ctime_ns)
    return identity == wanted


def _inside(path: Path, roots: Sequence[Path]) -> bool:
    return ".." not in path.parts and any(path.is_relative_to(root) for root in roots)


def static_hold(row: Row, roots: Sequence[Path]) -> str | None:
    """Why this row must be held before any open-handle probe, or None."""
    path = Path(row.path)
    if not _inside(path, roots):
        return "outside the allowed roots"
    if os.path.realpath(path.parent) != str(path.parent):
        return "ancestor is not canonical"
    try:
        info = os.lstat(path)
    except FileNotFoundError:
        return "already absent"
    if not _matches(row, info):
        return "identity changed"
    if row.kind == "file" and (not stat.S_ISREG(info.st_mode) or info.st_size != row.size):
        return "not the recorded regular file"
    if row.kind == "symlink" and (
        not stat.S_ISLNK(info.st_mode) or os.readlink(path) != row.link_target
    ):
        return "symlink changed"
    return None


def check(row: Row, roots: Sequence[Path], is_open: OpenProbe) -> str | None:
    """Why this row must be held, or None when it may be unlinked."""
    reason = static_hold(row, roots)
    if reason is not None:
        return reason
    state = is_open(row.path)
    if state is None:
        return "open-handle check failed"
    return "open handle" if state else None


def _hold(row: Row, reason: str, held: Counter[str], stream: TextIO) -> None:
    held[reason] += 1
    stream.write(json.dumps({"path": row.path, "held": reason}) + "\n")


def _remove(row: Row, stream: TextIO) -> None:
    os.unlink(row.path)
    record = {"path": row.path, "removed": True, "kind": row.kind, "bytes": row.size}
    stream.write(json.dumps(record) + "\n")


def _retire_chunk(
    chunk: Sequence[Row],
    roots: Sequence[Path],
    probe: BatchProbe,
    apply: bool,
    held: Counter[str],
    stream: TextIO,
) -> tuple[int, int]:
    """Check, probe once for the whole chunk, recheck each row, then unlink."""
    static = {row.path: static_hold(row, roots) for row in chunk}
    candidates = [row.path for row in chunk if static[row.path] is None]
    open_set = probe(candidates) if candidates else set()
    retired = logical = 0
    for row in chunk:
        reason = static[row.path]
        if reason is None and open_set is None:
            reason = "open-handle check failed"
        if reason is None and row.path in (open_set or set()):
            reason = "open handle"
        if reason is None:
            reason = static_hold(row, roots)
        if reason is not None:
            _hold(row, reason, held, stream)
            continue
        if apply:
            _remove(row, stream)
        retired += 1
        logical += row.size
    return retired, logical


def retire(
    dispositions: Path,
    expected_sha256: str,
    roots: Sequence[Path],
    log: Path,
    *,
    apply: bool,
    probe: BatchProbe = lsof_open_among,
    chunk_size: int = CHUNK_SIZE,
) -> Summary:
    rows = load_eligible(dispositions, expected_sha256)
    held: Counter[str] = Counter()
    retired = logical = 0
    ordered = sorted(rows, key=lambda row: (row.kind == "symlink", row.path))
    with log.open("a", encoding="utf-8") as stream:
        for start in range(0, len(ordered), chunk_size):
            done, size = _retire_chunk(
                ordered[start : start + chunk_size], roots, probe, apply, held, stream
            )
            retired += done
            logical += size
    return Summary(len(rows), retired, logical, dict(held))


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="replay.leaf_retirement")
    parser.add_argument("dispositions", type=Path)
    parser.add_argument("--sha256", required=True)
    parser.add_argument("--root", action="append", type=Path, required=True)
    parser.add_argument("--log", type=Path, required=True)
    parser.add_argument("--apply", action="store_true")
    args = parser.parse_args(argv)
    summary = retire(args.dispositions, args.sha256, args.root, args.log, apply=args.apply)
    print(
        json.dumps(
            {
                "mode": "apply" if args.apply else "dry-run",
                "eligible": summary.eligible,
                "retired": summary.retired,
                "logicalBytes": summary.logical_bytes,
                "held": summary.held,
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
