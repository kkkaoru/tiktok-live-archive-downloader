"""Offload quiet scratch files under `edits/work` to R2, then retire them to free disk space.

Each top-level `edits/work/<name>` is one unit. Its manifest is the same idempotency file
deliveries use, `<name>/evidence/intermediate-offload.json`, under recording id `<name>`, so
`replay.intermediate_offload restore edits/work/<name>` and `status` work unchanged and the
objects share the `intermediates/` lifecycle rule (deleted 90 days after upload).

A file is a candidate only when it is a single-link regular file with at least `MIN_BYTES`
allocated, mostly allocated rather than sparse (a sparse file would upload its full length
to free almost nothing), and whose mtime and ctime are both older than the quiet period
(seven days by default). Git
metadata, the offload manifests and locks, partial downloads and this tool's own directories
are never candidates. Only rows whose upload was confirmed reach `leaf_retirement`, which
rechecks identity and open handles immediately before each unlink. A failure leaves that
directory as it was and the sweep continues with the next one. The default is a dry run.
"""

from __future__ import annotations

import argparse
import json
import os
import stat
import sys
from collections.abc import Sequence
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from pathlib import Path
from zoneinfo import ZoneInfo

import httpx

from replay import intermediate_offload, leaf_retirement, r2_worker
from replay.core import ReplayError

WORK_ROOT = r2_worker.REPOSITORY / "edits" / "work"
LOG_DIRECTORY = "work-offload"
QUIET_PERIOD = timedelta(days=7)
MIN_BYTES = 1 << 20
BLOCK_BYTES = 512
JST = ZoneInfo("Asia/Tokyo")
EXCLUDED_UNITS = frozenset({r2_worker.PROJECT_ROOT.name, LOG_DIRECTORY})
EXCLUDED_DIRECTORIES = frozenset({".git"})
PACK_INDEX_NAME = "small-files-offload.json"
PACK_LOCK_NAME = "small-files-offload.lock"
EXCLUDED_NAMES = frozenset(
    {
        intermediate_offload.MANIFEST_NAME,
        intermediate_offload.LOCK_NAME,
        PACK_INDEX_NAME,
        PACK_LOCK_NAME,
    }
)


@dataclass(slots=True)
class UnitResult:
    name: str
    candidates: int
    candidate_bytes: int
    uploaded: int = 0
    reused: int = 0
    retired: int = 0
    retired_bytes: int = 0
    held: dict[str, int] = field(default_factory=dict)
    error: str | None = None


def _excluded(name: str) -> bool:
    return (
        name in EXCLUDED_NAMES
        or name.startswith((f".{intermediate_offload.MANIFEST_NAME}.", f".{PACK_INDEX_NAME}."))
        or name.endswith(intermediate_offload.PARTIAL_SUFFIX)
    )


def row_of(path: str, info: os.stat_result) -> leaf_retirement.Row:
    return leaf_retirement.Row(
        path,
        "file",
        info.st_dev,
        info.st_ino,
        info.st_nlink,
        info.st_size,
        info.st_mtime_ns,
        info.st_ctime_ns,
        None,
    )


def _sized(info: os.stat_result, *, small: bool) -> bool:
    """Large files go up one by one; small non-empty ones are packed by `replay.work_pack`."""
    if small:
        return 0 < info.st_size < MIN_BYTES
    allocated = info.st_blocks * BLOCK_BYTES
    return allocated >= MIN_BYTES and allocated * 2 >= info.st_size


def _quiet(info: os.stat_result, cutoff_ns: int, *, small: bool) -> bool:
    return (
        stat.S_ISREG(info.st_mode)
        and info.st_nlink == 1
        and _sized(info, small=small)
        and info.st_mtime_ns < cutoff_ns
        and info.st_ctime_ns < cutoff_ns
    )


def scan(unit: Path, cutoff_ns: int, *, small: bool = False) -> list[leaf_retirement.Row]:
    """Candidate rows of one unit, read with lstat; symlinks are never followed."""
    rows: list[leaf_retirement.Row] = []
    for directory, subdirectories, names in os.walk(unit):
        subdirectories[:] = sorted(
            name for name in subdirectories if name not in EXCLUDED_DIRECTORIES
        )
        for name in sorted(names):
            if _excluded(name):
                continue
            path = os.path.join(directory, name)
            try:
                info = os.lstat(path)
            except FileNotFoundError:
                continue
            if _quiet(info, cutoff_ns, small=small):
                rows.append(row_of(path, info))
    return rows


def units(work_root: Path, skip: frozenset[str] = frozenset()) -> list[Path]:
    excluded = EXCLUDED_UNITS | skip
    return sorted(
        entry
        for entry in work_root.iterdir()
        if entry.name not in excluded and entry.is_dir() and not entry.is_symlink()
    )


def _offload_unit(
    result: UnitResult,
    unit: Path,
    rows: Sequence[leaf_retirement.Row],
    store: intermediate_offload.ObjectStore,
    log: Path,
    probe: leaf_retirement.BatchProbe,
) -> None:
    summary = intermediate_offload.offload(
        rows, recording=unit.name, roots=(unit,), viewer_root=unit, store=store
    )
    result.uploaded, result.reused = summary.uploaded, summary.reused
    confirmed = [row for row in rows if row.path not in summary.skipped]
    retired = leaf_retirement.retire_rows(confirmed, (unit,), log, apply=True, probe=probe)
    result.retired, result.retired_bytes = retired.retired, retired.logical_bytes
    result.held = dict(retired.held)
    for reason in summary.skipped.values():
        result.held[reason] = result.held.get(reason, 0) + 1
    intermediate_offload.mark_retired(unit, recording=unit.name)


def sweep(
    work_root: Path,
    *,
    cutoff: datetime,
    store: intermediate_offload.ObjectStore | None,
    log: Path,
    probe: leaf_retirement.BatchProbe = leaf_retirement.lsof_open_among,
    skip: frozenset[str] = frozenset(),
) -> list[UnitResult]:
    """Plan every unit except `skip`; with a store, offload and retire each one independently."""
    cutoff_ns = int(cutoff.timestamp() * 1_000_000_000)
    results: list[UnitResult] = []
    for unit in units(work_root, skip):
        rows = scan(unit, cutoff_ns)
        if not rows:
            continue
        result = UnitResult(unit.name, len(rows), sum(row.size for row in rows))
        results.append(result)
        if store is None:
            continue
        try:
            _offload_unit(result, unit, rows, store, log, probe)
        except (ReplayError, OSError, ValueError, httpx.HTTPError) as error:
            result.error = str(error)
    return results


def report(results: Sequence[UnitResult], *, cutoff: datetime, applied: bool) -> dict[str, object]:
    return {
        "mode": "apply" if applied else "dry-run",
        "cutoff": cutoff.astimezone(JST).isoformat(),
        "units": len(results),
        "candidates": sum(result.candidates for result in results),
        "candidateBytes": sum(result.candidate_bytes for result in results),
        "uploaded": sum(result.uploaded for result in results),
        "reused": sum(result.reused for result in results),
        "retired": sum(result.retired for result in results),
        "retiredBytes": sum(result.retired_bytes for result in results),
        "failed": {result.name: result.error for result in results if result.error is not None},
    }


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="replay.work_offload", description=__doc__)
    parser.add_argument("--work-root", type=Path, default=WORK_ROOT)
    parser.add_argument("--quiet-days", type=int, default=QUIET_PERIOD.days)
    parser.add_argument("--apply", action="store_true")
    parser.add_argument("--skip", action="append", default=[], help="unit name to leave alone")
    args = parser.parse_args(argv)
    work_root = Path(os.path.realpath(args.work_root))
    skip = frozenset(args.skip)
    now = datetime.now(JST)
    cutoff = now - timedelta(days=args.quiet_days)
    log_directory = work_root / LOG_DIRECTORY
    log_directory.mkdir(mode=0o700, exist_ok=True)
    log = log_directory / f"{now.strftime('%Y%m%dT%H%M%S')}.jsonl"
    try:
        if args.apply:
            with r2_worker.session(intermediate_offload.BUCKET) as store:
                results = sweep(work_root, cutoff=cutoff, store=store, log=log, skip=skip)
        else:
            results = sweep(work_root, cutoff=cutoff, store=None, log=log, skip=skip)
    except (ReplayError, OSError, httpx.HTTPError) as error:
        print(f"WORK_OFFLOAD_ERROR {error}", file=sys.stderr)
        return 1
    summary = report(results, cutoff=cutoff, applied=args.apply)
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 1 if summary["failed"] else 0


if __name__ == "__main__":
    raise SystemExit(main())
