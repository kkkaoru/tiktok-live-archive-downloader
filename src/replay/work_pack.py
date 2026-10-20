"""Pack quiet small scratch files under `edits/work` into zstd tars, offload them, retire.

`replay.work_offload` uploads files of at least 1 MiB one by one. Hundreds of thousands of
smaller files would each cost several Worker round trips, so this module packs each unit's
quiet small files (same age and link rules, `0 < size < 1 MiB`) into `.tar.zst` packs under
`edits/work/work-offload/packs/<unit>/`. zstd suits this text-heavy scratch (JSON, CSV,
Markdown): a high ratio at a speed that keeps packing and restore well below upload time.

Packs follow the unit's layout so later changes stay local: one group per first-level
subdirectory (files directly in the unit form the `.` group), split into packs of at most
`PACK_BYTES` uncompressed, in path order. Restoring or reworking one area touches only that
group's packs, and `restore --relative PATH` downloads only the pack holding PATH. Each pack
is offloaded through `intermediate_offload`, so it is recorded in the unit's
`evidence/intermediate-offload.json` and expires 90 days after upload.

The unit's pack index, `evidence/small-files-offload.json`, lists every pack: its path under
the pack root, group, compression, SHA-256, state (`pending`, `uploaded`, `retired`,
`restored`) and members (`relative`, SHA-256, size, mode, mtime). A member is listed only
when its file was unchanged after it was read, and only listed members of an uploaded pack
reach `leaf_retirement`, which rechecks identity and open handles before each unlink; the
local pack is then retired the same way. An interrupted run is finished on the next one: a
`pending` pack is discarded locally and repacked, an `uploaded` pack retires its members
that still match. Restore never overwrites a file.
"""

from __future__ import annotations

import argparse
import fcntl
import hashlib
import io
import json
import os
import re
import sys
import tarfile
import tempfile
from collections.abc import Callable, Iterator, Mapping, Sequence
from contextlib import contextmanager
from dataclasses import dataclass, field
from datetime import UTC, datetime, timedelta
from enum import StrEnum
from pathlib import Path
from typing import BinaryIO

import httpx
import zstandard

from replay import disk_budget, intermediate_offload, leaf_retirement, r2_worker, work_offload
from replay.core import ReplayError
from replay.recordings import record

PACK_DIRECTORY = "packs"
INDEX_NAME = work_offload.PACK_INDEX_NAME
LOCK_NAME = work_offload.PACK_LOCK_NAME
VERSION = 1
PACK_BYTES = 256 * 1024 * 1024
ZSTD = "zstd"
UNCOMPRESSED = "none"
ZSTD_LEVEL = 10
ROOT_GROUP = "."
SLUG_LIMIT = 80
UNSAFE_SLUG = re.compile(r"[^A-Za-z0-9._-]")
ERRORS = (
    ReplayError,
    OSError,
    ValueError,
    tarfile.TarError,
    zstandard.ZstdError,
    httpx.HTTPError,
)

Clock = Callable[[], datetime]


class State(StrEnum):
    PENDING = "pending"
    UPLOADED = "uploaded"
    RETIRED = "retired"
    RESTORED = "restored"


@dataclass(frozen=True, slots=True)
class Member:
    relative: str
    sha256: str
    size: int
    mode: int
    mtime_ns: int


@dataclass(slots=True)
class Pack:
    relative: str
    sha256: str
    size: int
    state: State
    members: list[Member]
    group: str = ROOT_GROUP
    compression: str = UNCOMPRESSED


@dataclass(slots=True)
class UnitResult:
    name: str
    candidates: int
    candidate_bytes: int
    packs: int = 0
    packed: int = 0
    packed_bytes: int = 0
    retired: int = 0
    retired_bytes: int = 0
    held: dict[str, int] = field(default_factory=dict)
    error: str | None = None


def pack_root(work_root: Path) -> Path:
    return work_root / work_offload.LOG_DIRECTORY / PACK_DIRECTORY


def index_path(unit: Path) -> Path:
    return unit / intermediate_offload.EVIDENCE_DIRECTORY / INDEX_NAME


def group_of(relative: str) -> str:
    parts = Path(relative).parts
    return parts[0] if len(parts) > 1 else ROOT_GROUP


def slug(group: str) -> str:
    return "_root" if group == ROOT_GROUP else UNSAFE_SLUG.sub("_", group)[:SLUG_LIMIT]


def _member(raw: object) -> Member:
    data = record(raw)
    relative, digest = data.get("relative"), data.get("sha256")
    size, mode, mtime = data.get("bytes"), data.get("mode"), data.get("mtimeNs")
    if not isinstance(relative, str) or not isinstance(digest, str):
        raise ReplayError("Pack member needs a relative path and SHA-256")
    if type(size) is not int or type(mode) is not int or type(mtime) is not int:
        raise ReplayError("Pack member needs integer bytes, mode and mtimeNs")
    return Member(relative, digest, size, mode, mtime)


def _pack(raw: object) -> Pack:
    data = record(raw)
    relative, digest, size = data.get("relative"), data.get("sha256"), data.get("bytes")
    members = data.get("members")
    group, compression = data.get("group", ROOT_GROUP), data.get("compression", UNCOMPRESSED)
    if not isinstance(relative, str) or not isinstance(digest, str) or type(size) is not int:
        raise ReplayError("Pack needs a relative path, SHA-256 and bytes")
    if not isinstance(members, list):
        raise ReplayError("Pack needs a members array")
    if not isinstance(group, str) or compression not in {ZSTD, UNCOMPRESSED}:
        raise ReplayError("Pack has an unknown group or compression")
    try:
        state = State(str(data.get("state")))
    except ValueError as error:
        raise ReplayError("Pack has an unknown state") from error
    return Pack(
        relative, digest, size, state, [_member(item) for item in members], group, compression
    )


def load(unit: Path) -> list[Pack]:
    path = index_path(unit)
    if not path.exists():
        return []
    data = record(json.loads(path.read_text(encoding="utf-8")))
    packs = data.get("packs")
    if data.get("version") != VERSION or data.get("unit") != unit.name:
        raise ReplayError(f"Pack index belongs to another unit or version: {path}")
    if not isinstance(packs, list):
        raise ReplayError("Pack index needs a packs array")
    return [_pack(item) for item in packs]


def _serialized(pack: Pack) -> dict[str, object]:
    return {
        "relative": pack.relative,
        "group": pack.group,
        "compression": pack.compression,
        "sha256": pack.sha256,
        "bytes": pack.size,
        "state": str(pack.state),
        "members": [
            {
                "relative": member.relative,
                "sha256": member.sha256,
                "bytes": member.size,
                "mode": member.mode,
                "mtimeNs": member.mtime_ns,
            }
            for member in pack.members
        ],
    }


def save(unit: Path, packs: Sequence[Pack]) -> None:
    """Atomically replace the index: same-directory temporary file, fsync, rename."""
    path = index_path(unit)
    body = {"version": VERSION, "unit": unit.name, "packs": [_serialized(p) for p in packs]}
    path.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        "w", encoding="utf-8", dir=path.parent, prefix=f".{path.name}.", delete=False
    ) as stream:
        json.dump(body, stream, sort_keys=True)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    os.chmod(stream.name, 0o600)
    os.replace(stream.name, path)


@contextmanager
def locked(unit: Path) -> Iterator[None]:
    """One pack or restore per unit at a time; the lock file is never unlinked."""
    lock = unit / intermediate_offload.EVIDENCE_DIRECTORY / LOCK_NAME
    lock.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
    descriptor = os.open(lock, os.O_CREAT | os.O_RDWR | os.O_NOFOLLOW, 0o600)
    with os.fdopen(descriptor, "a+b") as stream:
        try:
            fcntl.flock(stream, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as error:
            raise ReplayError("Another process is packing or restoring this unit") from error
        try:
            yield
        finally:
            fcntl.flock(stream, fcntl.LOCK_UN)


def chunks(
    unit: Path, rows: Sequence[leaf_retirement.Row]
) -> list[tuple[str, list[leaf_retirement.Row]]]:
    """Rows grouped by first-level subdirectory, each group split at `PACK_BYTES`."""
    grouped: dict[str, list[leaf_retirement.Row]] = {}
    for row in sorted(rows, key=lambda item: item.path):
        grouped.setdefault(group_of(str(Path(row.path).relative_to(unit))), []).append(row)
    result: list[tuple[str, list[leaf_retirement.Row]]] = []
    for group in sorted(grouped):
        current: list[leaf_retirement.Row] = []
        total = 0
        for row in grouped[group]:
            if current and total + row.size > PACK_BYTES:
                result.append((group, current))
                current, total = [], 0
            current.append(row)
            total += row.size
        result.append((group, current))
    return result


def build(
    unit: Path, rows: Sequence[leaf_retirement.Row], archive: Path
) -> tuple[list[Member], list[leaf_retirement.Row]]:
    """Write a `.tar.zst` and return the members whose files did not change while read."""
    members: list[Member] = []
    kept: list[leaf_retirement.Row] = []
    archive.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
    compressor = zstandard.ZstdCompressor(level=ZSTD_LEVEL, threads=-1)
    with archive.open("xb") as raw:
        with (
            compressor.stream_writer(raw, closefd=False) as compressed,
            tarfile.open(fileobj=compressed, mode="w|", format=tarfile.PAX_FORMAT) as tar,
        ):
            for row in rows:
                member = _add(tar, unit, row)
                if member is not None:
                    members.append(member)
                    kept.append(row)
        raw.flush()
        os.fsync(raw.fileno())
    return members, kept


def _add(tar: tarfile.TarFile, unit: Path, row: leaf_retirement.Row) -> Member | None:
    try:
        data = Path(row.path).read_bytes()
    except FileNotFoundError:
        return None
    if len(data) != row.size or leaf_retirement.static_hold(row, (unit,)) is not None:
        return None
    relative = str(Path(row.path).relative_to(unit))
    mode = os.lstat(row.path).st_mode & 0o7777
    info = tarfile.TarInfo(relative)
    info.size, info.mode, info.mtime = row.size, mode, row.mtime_ns // 1_000_000_000
    tar.addfile(info, io.BytesIO(data))
    return Member(relative, hashlib.sha256(data).hexdigest(), row.size, mode, row.mtime_ns)


def _archive_row(path: Path) -> leaf_retirement.Row:
    return work_offload.row_of(str(path), os.lstat(path))


def _still_matching(unit: Path, member: Member) -> leaf_retirement.Row | None:
    """A row for a member left behind by an interrupted run, if it is still the packed file."""
    path = unit / member.relative
    try:
        info = os.lstat(path)
    except FileNotFoundError:
        return None
    row = work_offload.row_of(str(path), info)
    if (row.size, row.mtime_ns, row.nlink) != (member.size, member.mtime_ns, 1):
        return None
    if leaf_retirement.static_hold(row, (unit,)) is not None:
        return None
    return row if leaf_retirement.sha256_of(path) == member.sha256 else None


@dataclass(frozen=True, slots=True)
class Context:
    unit: Path
    packs_root: Path
    store: intermediate_offload.ObjectStore
    log: Path
    probe: leaf_retirement.BatchProbe


def _finish(
    context: Context, pack: Pack, rows: Sequence[leaf_retirement.Row], result: UnitResult
) -> None:
    """Retire an uploaded pack's members, then its local file."""
    retired = leaf_retirement.retire_rows(
        rows, (context.unit,), context.log, apply=True, probe=context.probe
    )
    result.retired += retired.retired
    result.retired_bytes += retired.logical_bytes
    for reason, count in retired.held.items():
        result.held[reason] = result.held.get(reason, 0) + count
    archive = context.packs_root / pack.relative
    if os.path.lexists(archive):
        leaf_retirement.retire_rows(
            [_archive_row(archive)],
            (context.packs_root,),
            context.log,
            apply=True,
            probe=context.probe,
        )
    intermediate_offload.mark_retired(context.unit, recording=context.unit.name)
    pack.state = State.RETIRED


def _resume(context: Context, packs: list[Pack], result: UnitResult) -> None:
    """Finish what an interrupted run left: discard pending packs, retire uploaded ones."""
    for pack in [item for item in packs if item.state is State.PENDING]:
        (context.packs_root / pack.relative).unlink(missing_ok=True)
        packs.remove(pack)
        save(context.unit, packs)
    for pack in packs:
        if pack.state is State.UPLOADED:
            rows = [_still_matching(context.unit, member) for member in pack.members]
            _finish(context, pack, [row for row in rows if row is not None], result)
            save(context.unit, packs)


def _pack_chunk(
    context: Context,
    packs: list[Pack],
    group: str,
    rows: Sequence[leaf_retirement.Row],
    name: str,
    result: UnitResult,
) -> None:
    archive = context.packs_root / context.unit.name / name
    members, kept = build(context.unit, rows, archive)
    pack = Pack(
        str(archive.relative_to(context.packs_root)),
        leaf_retirement.sha256_of(archive),
        archive.stat().st_size,
        State.PENDING,
        members,
        group,
        ZSTD,
    )
    packs.append(pack)
    save(context.unit, packs)
    offloaded = intermediate_offload.offload(
        [_archive_row(archive)],
        recording=context.unit.name,
        roots=(context.packs_root,),
        viewer_root=context.unit,
        store=context.store,
    )
    if offloaded.skipped:
        raise ReplayError(f"Pack {pack.relative} changed before it was confirmed in R2")
    pack.state = State.UPLOADED
    save(context.unit, packs)
    result.packs += 1
    result.packed += len(members)
    result.packed_bytes += pack.size
    _finish(context, pack, kept, result)
    save(context.unit, packs)


def _pack_unit(context: Context, cutoff_ns: int, stamp: str, result: UnitResult) -> None:
    with locked(context.unit):
        packs = load(context.unit)
        _resume(context, packs, result)
        rows = work_offload.scan(context.unit, cutoff_ns, small=True)
        for number, (group, chunk) in enumerate(chunks(context.unit, rows)):
            name = f"{stamp}-{number:04d}-{slug(group)}.tar.zst"
            _pack_chunk(context, packs, group, chunk, name, result)


def sweep(
    work_root: Path,
    *,
    cutoff: datetime,
    store: intermediate_offload.ObjectStore | None,
    log: Path,
    stamp: str,
    probe: leaf_retirement.BatchProbe = leaf_retirement.lsof_open_among,
    skip: frozenset[str] = frozenset(),
) -> list[UnitResult]:
    """Plan every unit except `skip`; with a store, pack, offload and retire each one."""
    cutoff_ns = int(cutoff.timestamp() * 1_000_000_000)
    results: list[UnitResult] = []
    for unit in work_offload.units(work_root, skip):
        rows = work_offload.scan(unit, cutoff_ns, small=True)
        unfinished = any(pack.state is State.UPLOADED for pack in load(unit))
        if not rows and not unfinished:
            continue
        result = UnitResult(unit.name, len(rows), sum(row.size for row in rows))
        results.append(result)
        if store is None:
            continue
        context = Context(unit, pack_root(work_root), store, log, probe)
        try:
            _pack_unit(context, cutoff_ns, stamp, result)
        except ERRORS as error:
            result.error = str(error)
    return results


def _restore_member(member: Member, data: bytes, unit: Path) -> bool:
    """Link one member back; False when a file already occupies its path."""
    destination = unit / member.relative
    if os.path.lexists(destination):
        return False
    if len(data) != member.size or hashlib.sha256(data).hexdigest() != member.sha256:
        raise ReplayError(f"Pack member {member.relative} does not match its SHA-256")
    destination.parent.mkdir(parents=True, exist_ok=True)
    partial = destination.with_name(destination.name + intermediate_offload.PARTIAL_SUFFIX)
    partial.unlink(missing_ok=True)
    try:
        with partial.open("xb") as stream:
            stream.write(data)
            stream.flush()
            os.fsync(stream.fileno())
        os.chmod(partial, member.mode)
        os.utime(partial, ns=(member.mtime_ns, member.mtime_ns))
        os.link(partial, destination)
    finally:
        partial.unlink(missing_ok=True)
    return True


@contextmanager
def _opened(pack: Pack, archive: Path) -> Iterator[tarfile.TarFile]:
    with archive.open("rb") as raw:
        stream: BinaryIO = raw
        if pack.compression == ZSTD:
            stream = zstandard.ZstdDecompressor().stream_reader(raw)
        with tarfile.open(fileobj=stream, mode="r|") as tar:
            yield tar


def _extract(pack: Pack, archive: Path, unit: Path, wanted: set[str]) -> tuple[int, int]:
    """Stream the pack once and restore the members named in `wanted`."""
    members = {member.relative: member for member in pack.members if member.relative in wanted}
    restored = occupied = 0
    with _opened(pack, archive) as tar:
        for info in tar:
            member = members.pop(info.name, None)
            if member is None:
                continue
            stream = tar.extractfile(info)
            if stream is None:
                raise ReplayError(f"Pack member {member.relative} is not a regular file")
            if _restore_member(member, stream.read(), unit):
                restored += 1
            else:
                occupied += 1
    if members:
        raise ReplayError(f"Pack {pack.relative} lacks {len(members)} listed members")
    return restored, occupied


def _download(
    pack: Pack, unit: Path, packs_root: Path, store: intermediate_offload.ObjectStore, now: datetime
) -> Path:
    """The pack's verified bytes beside its pack path; the caller unlinks the result."""
    manifest = intermediate_offload.load(
        intermediate_offload.manifest_path(unit), recording=unit.name
    )
    entry = manifest.entries.get((str(packs_root), pack.relative))
    if entry is None:
        raise ReplayError(f"Pack {pack.relative} has no offload entry")
    if entry.expires_at is not None and now >= datetime.fromisoformat(entry.expires_at):
        raise ReplayError(f"Pack {pack.relative} expired from R2 at {entry.expires_at}")
    target = packs_root / (pack.relative + intermediate_offload.PARTIAL_SUFFIX)
    target.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
    disk_budget.require_headroom([target.parent], pack.size)
    target.unlink(missing_ok=True)
    store.get(entry.key, target)
    if leaf_retirement.sha256_of(target) != pack.sha256:
        target.unlink()
        raise ReplayError(f"Restored pack {pack.relative} does not match its SHA-256")
    return target


def restore(
    unit: Path,
    *,
    work_root: Path,
    store: intermediate_offload.ObjectStore,
    relative: str | None = None,
    clock: Clock = lambda: datetime.now(UTC),
) -> dict[str, int]:
    """Link back offloaded members: all of them, or only `relative` from its one pack."""
    packs_root = pack_root(work_root)
    restored = occupied = 0
    with locked(unit):
        packs = load(unit)
        offloaded = [pack for pack in packs if pack.state in {State.UPLOADED, State.RETIRED}]
        if relative is not None:
            offloaded = [
                pack
                for pack in offloaded
                if any(member.relative == relative for member in pack.members)
            ][-1:]
            if not offloaded:
                raise ReplayError(f"No offloaded pack holds {relative}")
        for pack in offloaded:
            wanted = {member.relative for member in pack.members}
            archive = _download(pack, unit, packs_root, store, clock())
            try:
                done, kept = _extract(
                    pack, archive, unit, wanted if relative is None else {relative}
                )
            finally:
                archive.unlink(missing_ok=True)
            restored, occupied = restored + done, occupied + kept
            if relative is None:
                pack.state = State.RESTORED
                save(unit, packs)
    return {"restored": restored, "alreadyPresent": occupied}


def report(results: Sequence[UnitResult], *, cutoff: datetime, applied: bool) -> dict[str, object]:
    return {
        "mode": "apply" if applied else "dry-run",
        "cutoff": cutoff.astimezone(work_offload.JST).isoformat(),
        "units": len(results),
        "candidates": sum(result.candidates for result in results),
        "candidateBytes": sum(result.candidate_bytes for result in results),
        "packs": sum(result.packs for result in results),
        "packed": sum(result.packed for result in results),
        "packedBytes": sum(result.packed_bytes for result in results),
        "retired": sum(result.retired for result in results),
        "retiredBytes": sum(result.retired_bytes for result in results),
        "held": _held(results),
        "failed": {result.name: result.error for result in results if result.error is not None},
    }


def _held(results: Sequence[UnitResult]) -> Mapping[str, int]:
    total: dict[str, int] = {}
    for result in results:
        for reason, count in result.held.items():
            total[reason] = total.get(reason, 0) + count
    return total


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="replay.work_pack", description=__doc__)
    parser.add_argument("--work-root", type=Path, default=work_offload.WORK_ROOT)
    commands = parser.add_subparsers(dest="command")
    sweep_command = commands.add_parser("sweep")
    sweep_command.add_argument("--quiet-days", type=int, default=work_offload.QUIET_PERIOD.days)
    sweep_command.add_argument("--apply", action="store_true")
    sweep_command.add_argument(
        "--skip", action="append", default=[], help="unit name to leave alone"
    )
    restore_command = commands.add_parser("restore")
    restore_command.add_argument("unit", type=Path, help="edits/work/<unit>")
    restore_command.add_argument("--relative", help="restore only this member")
    args = parser.parse_args(argv)
    work_root = Path(os.path.realpath(args.work_root))
    now = datetime.now(work_offload.JST)
    try:
        if args.command == "restore":
            unit = Path(os.path.realpath(args.unit))
            with r2_worker.session(intermediate_offload.BUCKET) as store:
                counts = restore(unit, work_root=work_root, store=store, relative=args.relative)
            print(json.dumps(counts, sort_keys=True))
            return 0
        quiet_days = getattr(args, "quiet_days", work_offload.QUIET_PERIOD.days)
        applied = bool(getattr(args, "apply", False))
        skip = frozenset(getattr(args, "skip", []))
        cutoff = now - timedelta(days=quiet_days)
        stamp = now.strftime("%Y%m%dT%H%M%S")
        log_directory = work_root / work_offload.LOG_DIRECTORY
        log_directory.mkdir(mode=0o700, exist_ok=True)
        log = log_directory / f"pack-{stamp}.jsonl"
        if applied:
            with r2_worker.session(intermediate_offload.BUCKET) as store:
                results = sweep(
                    work_root, cutoff=cutoff, store=store, log=log, stamp=stamp, skip=skip
                )
        else:
            results = sweep(work_root, cutoff=cutoff, store=None, log=log, stamp=stamp, skip=skip)
    except ERRORS as error:
        print(f"WORK_PACK_ERROR {error}", file=sys.stderr)
        return 1
    summary = report(results, cutoff=cutoff, applied=applied)
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 1 if summary["failed"] else 0


if __name__ == "__main__":
    raise SystemExit(main())
