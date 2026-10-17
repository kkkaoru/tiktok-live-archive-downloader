"""Measure real disk use and refuse new large work below the free-space floor.

Sizes are physical: allocated blocks, each inode counted once, symlinks never followed,
so sparse test files and hardlinks do not inflate a report. Nothing here deletes a
file; retirement stays an owner decision (see `replay.leaf_retirement`).

The floor follows the repository policy: the larger of 30 GiB and 5% of the volume.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import stat
from collections.abc import Iterable, Iterator, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path

from replay.core import ReplayError

GIB = 1024**3
FLOOR_MIN_BYTES = 30 * GIB
FLOOR_FRACTION = 0.05
BLOCK_BYTES = 512
DEFAULT_MIN_BYTES = 30 * 1024 * 1024
DEFAULT_TOP = 15
PROTECTED_NAMES = frozenset({"private", "downloads", ".git"})


@dataclass(frozen=True, slots=True)
class Headroom:
    path: Path
    total: int
    free: int
    floor: int
    need: int

    @property
    def spare(self) -> int:
        return self.free - self.need - self.floor


@dataclass(frozen=True, slots=True)
class Usage:
    files: int
    allocated: int


@dataclass(frozen=True, slots=True)
class Report:
    allocated: int
    by_directory: dict[str, Usage]
    by_extension: dict[str, Usage]


def floor_bytes(total: int) -> int:
    """The larger of 30 GiB and 5% of the volume."""
    return max(FLOOR_MIN_BYTES, int(total * FLOOR_FRACTION))


def measure(path: Path, need: int = 0) -> Headroom:
    usage = shutil.disk_usage(path)
    return Headroom(path, usage.total, usage.free, floor_bytes(usage.total), need)


def require_headroom(paths: Iterable[Path], need: int = 0) -> list[Headroom]:
    """Return the measurements, or raise when any volume cannot keep its floor."""
    volumes = [measure(path, need) for path in paths]
    short = [volume for volume in volumes if volume.spare < 0]
    if short:
        worst = min(short, key=lambda volume: volume.spare)
        raise ReplayError(_shortage(worst))
    return volumes


def require_shared_headroom(needs: Mapping[Path, int]) -> list[Headroom]:
    """Like `require_headroom`, but paths on one volume share a single free-space pool.

    Each path states its own peak need; needs of paths whose volume is the same device
    are summed and checked once against that volume's floor, so scratch and output on one
    disk cannot both claim the same free bytes.
    """
    grouped: dict[int, list[Path]] = {}
    for path in needs:
        grouped.setdefault(path.stat().st_dev, []).append(path)
    volumes = [measure(paths[0], sum(needs[path] for path in paths)) for paths in grouped.values()]
    short = [volume for volume in volumes if volume.spare < 0]
    if short:
        worst = min(short, key=lambda volume: volume.spare)
        raise ReplayError(_shortage(worst))
    return volumes


def _shortage(worst: Headroom) -> str:
    return (
        f"Free space on {worst.path} is {worst.free / GIB:.1f} GiB; "
        f"{(worst.need + worst.floor) / GIB:.1f} GiB is required "
        f"(need {worst.need / GIB:.1f} + floor {worst.floor / GIB:.1f}). "
        "Retire approved intermediates before starting new large work."
    )


def _extension(name: str) -> str:
    return Path(name).suffix.lower() or "(none)"


def _add(table: dict[str, Usage], key: str, size: int) -> None:
    current = table.get(key, Usage(0, 0))
    table[key] = Usage(current.files + 1, current.allocated + size)


def _entries(
    directory: Path, parts: tuple[str, ...]
) -> Iterator[tuple[tuple[str, ...], str, os.stat_result]]:
    with os.scandir(directory) as listing:
        entries = list(listing)
    for entry in entries:
        info = entry.stat(follow_symlinks=False)
        if not stat.S_ISDIR(info.st_mode):
            yield parts, entry.name, info
        elif entry.name not in PROTECTED_NAMES:
            yield from _entries(Path(entry.path), (*parts, entry.name))


def physical_usage(root: Path, *, min_bytes: int = DEFAULT_MIN_BYTES, depth: int = 3) -> Report:
    """Walk `root` without following symlinks or entering protected user-data folders."""
    seen: set[tuple[int, int]] = set()
    by_directory: dict[str, Usage] = {}
    by_extension: dict[str, Usage] = {}
    total = 0
    for parts, name, info in _entries(root, ()):
        size = info.st_blocks * BLOCK_BYTES
        identity = (info.st_dev, info.st_ino)
        if not stat.S_ISREG(info.st_mode) or size < min_bytes or identity in seen:
            continue
        seen.add(identity)
        total += size
        _add(by_directory, "/".join(parts[:depth]) or ".", size)
        _add(by_extension, _extension(name), size)
    return Report(total, by_directory, by_extension)


def _top(table: dict[str, Usage], count: int) -> list[dict[str, object]]:
    ranked = sorted(table.items(), key=lambda item: item[1].allocated, reverse=True)
    return [
        {"name": name, "files": usage.files, "allocatedBytes": usage.allocated}
        for name, usage in ranked[:count]
    ]


def render(report: Report, count: int) -> str:
    return json.dumps(
        {
            "allocatedBytes": report.allocated,
            "byDirectory": _top(report.by_directory, count),
            "byExtension": _top(report.by_extension, count),
        },
        indent=2,
    )


def volume_table(volumes: Sequence[Headroom]) -> dict[str, dict[str, int]]:
    return {
        str(volume.path): {
            "totalBytes": volume.total,
            "freeBytes": volume.free,
            "floorBytes": volume.floor,
            "spareBytes": volume.spare,
        }
        for volume in volumes
    }


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="replay.disk_budget")
    commands = parser.add_subparsers(dest="command", required=True)
    report = commands.add_parser("report")
    report.add_argument("root", type=Path)
    report.add_argument("--min-mib", type=int, default=DEFAULT_MIN_BYTES // (1024 * 1024))
    report.add_argument("--depth", type=int, default=3)
    report.add_argument("--top", type=int, default=DEFAULT_TOP)
    preflight = commands.add_parser("preflight")
    preflight.add_argument("--path", type=Path, action="append", required=True)
    preflight.add_argument("--need-gib", type=float, default=0.0)
    args = parser.parse_args(argv)
    if args.command == "report":
        usage = physical_usage(args.root, min_bytes=args.min_mib * 1024 * 1024, depth=args.depth)
        print(render(usage, args.top))
        return 0
    try:
        volumes = require_headroom(args.path, int(args.need_gib * GIB))
    except ReplayError as error:
        print(f"DISK_BUDGET_ERROR {error}")
        return 1
    print(json.dumps(volume_table(volumes), indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
