"""The two places large work may live, and where relocated work folders went.

Work lives under the checkout's `edits/work/` or, for large render/offload work, under the
approved external root on the SSD (`REPLAY_EXTERNAL_WORK_ROOT`, the same root
`scripts/tmux-job.sh` accepts). Completed folders moved from `edits/work/<NAME>` to the
external root are listed in `edits/work/RELOCATED-TO-SSD.tsv`; `resolve` maps a recorded old
path to its current location so receipts that still name the old path can be read. No
symlink or extra mount is involved, and a relocated row must land inside the external root.
"""

from __future__ import annotations

import argparse
import csv
import os
import sys
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path

from replay.core import ReplayError

CHECKOUT = Path(__file__).resolve().parents[2]
CHECKOUT_WORK = CHECKOUT / "edits" / "work"
DEFAULT_EXTERNAL_WORK = Path("/Volumes/m.2-ssd/movie/replay-work")
EXTERNAL_ENVIRONMENT = "REPLAY_EXTERNAL_WORK_ROOT"
INDEX_NAME = "RELOCATED-TO-SSD.tsv"
INDEX_COLUMNS = ("name", "old_path", "new_path", "files", "bytes", "moved_at")


@dataclass(frozen=True)
class Relocation:
    """One work folder moved from the checkout to the external root."""

    name: str
    old_path: Path
    new_path: Path
    files: int
    bytes: int
    moved_at: str


def external_work(environment: Mapping[str, str] | None = None) -> Path:
    variables = os.environ if environment is None else environment
    return Path(variables.get(EXTERNAL_ENVIRONMENT, str(DEFAULT_EXTERNAL_WORK)))


def roots(*, work: Path = CHECKOUT_WORK, external: Path | None = None) -> tuple[Path, Path]:
    return (work, external_work() if external is None else external)


def within_work(path: Path, *, work: Path = CHECKOUT_WORK, external: Path | None = None) -> bool:
    """True when the path is under either work root (no `..` escape, no linked leaf)."""
    if path.is_symlink():
        return False
    candidate = path.resolve()
    return any(
        candidate.is_relative_to(root.resolve()) for root in roots(work=work, external=external)
    )


def _relocation(row: Mapping[str, str], external: Path) -> Relocation:
    try:
        relocation = Relocation(
            name=row["name"],
            old_path=Path(row["old_path"]),
            new_path=Path(row["new_path"]),
            files=int(row["files"]),
            bytes=int(row["bytes"]),
            moved_at=row["moved_at"],
        )
    except (KeyError, ValueError) as error:
        raise ReplayError(f"Malformed relocation row: {row}") from error
    if (
        relocation.new_path != external / relocation.name
        or relocation.old_path.name != relocation.name
    ):
        raise ReplayError(
            f"Relocation {relocation.name} must map a work folder into the external root"
        )
    return relocation


def relocations(
    *, work: Path = CHECKOUT_WORK, external: Path | None = None
) -> dict[Path, Relocation]:
    """The relocation index keyed by old folder path; empty when nothing was relocated."""
    index = work / INDEX_NAME
    if not index.is_file():
        return {}
    target = external_work() if external is None else external
    with index.open(encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle, delimiter="\t")
        if tuple(reader.fieldnames or ()) != INDEX_COLUMNS:
            raise ReplayError(f"{index} does not have the columns {', '.join(INDEX_COLUMNS)}")
        rows = [_relocation(row, target) for row in reader]
    return {row.old_path: row for row in rows}


def resolve(path: Path, *, work: Path = CHECKOUT_WORK, external: Path | None = None) -> Path:
    """Current location of a recorded work path: unchanged unless its folder was relocated."""
    index = relocations(work=work, external=external)
    folder = next((old for old in index if path == old or path.is_relative_to(old)), None)
    if folder is None:
        return path
    return index[folder].new_path / path.relative_to(folder)


def missing(*, work: Path = CHECKOUT_WORK, external: Path | None = None) -> list[str]:
    """Names of relocated folders whose new directory is absent, or whose old path reappeared."""
    index = relocations(work=work, external=external).values()
    return sorted(row.name for row in index if not row.new_path.is_dir() or row.old_path.exists())


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    resolver = commands.add_parser("resolve", help="print the current location of recorded paths")
    resolver.add_argument("paths", nargs="+", type=Path)
    commands.add_parser("check", help="verify every relocated folder exists only at its new path")
    arguments = parser.parse_args(argv)
    if arguments.command == "resolve":
        print("\n".join(str(resolve(path)) for path in arguments.paths))
        return 0
    problems = missing()
    print("\n".join(problems) if problems else "ok")
    return 1 if problems else 0


if __name__ == "__main__":
    sys.exit(main())
