"""Decide from evidence which heavy media files may be retired, on every volume.

The planner walks the given roots without following symlinks and classifies each large
media file by a fixed rule order. A file is `eligible` only when its owning recording passes
`edit_completion.completion_report` and no JSON receipt, workflow or configuration under the
scanned roots names the file. A file that something names is `hold` (bound by a reference), and
a file of an unfinished, unverifiable or unowned recording is `keep`. Dated deliveries, Viewer
inputs, models, tools, protected data and small evidence are always `keep`.

The output is a `leaf_retirement` disposition file plus a per-reason summary; nothing is
deleted here. Retirement stays with `replay.leaf_retirement`, which rechecks identity and open
handles per row.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import stat
import sys
from collections import Counter
from collections.abc import Callable, Iterator, Sequence
from dataclasses import dataclass, field
from pathlib import Path

from replay import disk_budget, edit_completion, leaf_retirement
from replay.core import ReplayError

HEAVY_SUFFIXES = frozenset(
    {".mov", ".mp4", ".m4v", ".mkv", ".wav", ".m4a", ".aac", ".flac", ".pt", ".ckpt", ".npy"}
)
KEEP_NAMES = frozenset(
    {
        "private",
        "downloads",
        ".git",
        "models",
        ".tools",
        ".venv",
        "node_modules",
        "sounds",
        "bgm-library",
    }
)
PROXY_NAME = "source-proxy-native-clock.mp4"
DELIVERY_DIRECTORY = "delivery"
MIN_BYTES = 64 * 1024 * 1024
MAX_JSON_BYTES = 64 * 1024 * 1024
OWNER = re.compile(r"(?:replay-\d{15,}|youtube-[A-Za-z0-9_-]{11}|(?<![\d-])\d{16,}(?!\d))")
MEDIA_REFERENCE = re.compile(r'"(/[^"\\]+?\.(?:mov|mp4|m4v|mkv|wav|m4a|aac|flac|pt|ckpt|npy))"')
SKIPPED_JSON = re.compile(r"^(dispositions|storage-plan)([-.].*)?\.json$")

Completion = Callable[[str], bool]


@dataclass(frozen=True, slots=True)
class Entry:
    path: Path
    size: int
    length: int
    dev: int
    inode: int
    nlink: int
    mtime_ns: int
    ctime_ns: int
    mode: int


@dataclass(slots=True)
class Plan:
    rows: list[dict[str, object]] = field(default_factory=list)
    bytes_by_reason: Counter[str] = field(default_factory=Counter)
    files_by_reason: Counter[str] = field(default_factory=Counter)


def owner_of(path: Path) -> str | None:
    """The recording a path belongs to, normalised to `replay-<id>` or `youtube-<id>`."""
    for part in path.parts:
        match = OWNER.fullmatch(part)
        if match is not None:
            value = match.group(0)
            return value if not value.isdigit() else f"replay-{value}"
    for part in path.parts:
        match = OWNER.search(part)
        if match is not None and not match.group(0).isdigit():
            return match.group(0)
    return None


def walk(root: Path) -> Iterator[Entry]:
    """Regular files under `root`; symlinks are neither followed nor reported."""
    pending = [root]
    while pending:
        directory = pending.pop()
        try:
            children = list(os.scandir(directory))
        except OSError:
            continue
        for child in children:
            if child.name in disk_budget.PROTECTED_NAMES:
                continue
            info = child.stat(follow_symlinks=False)
            if stat.S_ISDIR(info.st_mode):
                pending.append(Path(child.path))
            elif stat.S_ISREG(info.st_mode):
                yield Entry(
                    Path(child.path),
                    info.st_blocks * disk_budget.BLOCK_BYTES,
                    info.st_size,
                    info.st_dev,
                    info.st_ino,
                    info.st_nlink,
                    info.st_mtime_ns,
                    info.st_ctime_ns,
                    info.st_mode,
                )


def references(roots: Sequence[Path]) -> set[str]:
    """Absolute media paths named by any bounded JSON file under the roots."""
    named: set[str] = set()
    for root in roots:
        for entry in walk(root):
            path = entry.path
            if path.suffix != ".json" or SKIPPED_JSON.match(path.name):
                continue
            if entry.size > MAX_JSON_BYTES:
                continue
            try:
                text = path.read_text(encoding="utf-8", errors="replace")
            except OSError:
                continue
            named.update(MEDIA_REFERENCE.findall(text))
    return named


def delivery_targets(roots: Sequence[Path]) -> frozenset[str]:
    """Real paths of media that delivery-directory symlinks publish to the Viewer."""
    targets: set[str] = set()
    for root in roots:
        for directory, _subdirectories, names in os.walk(root):
            if Path(directory).name != DELIVERY_DIRECTORY:
                continue
            for name in names:
                link = Path(directory, name)
                if link.is_symlink():
                    targets.add(os.path.realpath(link))
    return frozenset(targets)


def completes(edits: Path, assumed: frozenset[str] = frozenset()) -> Completion:
    """Memoised completion gate of one recording under `edits`.

    `assumed` lists recordings the owner explicitly treats as complete; it is a what-if
    input for a dry classification, never a statement that the gate passed.
    """
    known: dict[str, bool] = dict.fromkeys(assumed, True)

    def gate(owner: str) -> bool:
        if owner not in known:
            try:
                known[owner] = bool(edit_completion.completion_report(edits / owner)["complete"])
            except (ReplayError, OSError, ValueError, KeyError):
                known[owner] = False
        return known[owner]

    return gate


def decide(
    entry: Entry, named: set[str], complete: Completion, delivered: frozenset[str] = frozenset()
) -> tuple[str, str]:
    """Classification and reason, from a fixed rule order."""
    path = entry.path
    if path.suffix.lower() not in HEAVY_SUFFIXES or entry.size < MIN_BYTES:
        return "keep", "small or non-media evidence"
    if any(part in KEEP_NAMES for part in path.parts):
        return "keep", "protected, model or tool data"
    if path.parent.name == DELIVERY_DIRECTORY:
        return "keep", "dated delivery"
    if str(path) in delivered:
        return "keep", "linked dated delivery"
    if path.name == PROXY_NAME:
        return "keep", "reused caption proxy"
    owner = owner_of(path)
    if owner is None:
        return "hold", "no owning recording"
    if not complete(owner):
        return "keep", "recording is unfinished or unverifiable"
    if str(path) in named:
        return "hold", "named by a JSON receipt or workflow"
    return "eligible", "unreferenced media of a completed recording"


def row(entry: Entry, classification: str, reason: str) -> dict[str, object]:
    return {
        "path": str(entry.path),
        "kind": "file",
        "dev": entry.dev,
        "inode": entry.inode,
        "nlink": entry.nlink,
        "bytes": entry.length,
        "mtimeNs": entry.mtime_ns,
        "ctimeNs": entry.ctime_ns,
        "classification": classification,
        "reason": reason,
    }


def plan(
    roots: Sequence[Path], scan: Sequence[Path], edits: Path, assumed: frozenset[str] = frozenset()
) -> Plan:
    """Classify every large file under `roots`; `scan` bounds the reference index."""
    named = references(scan)
    delivered = delivery_targets(scan)
    complete = completes(edits, assumed)
    result = Plan()
    seen: set[tuple[int, int]] = set()
    for root in roots:
        for entry in walk(root):
            identity = (entry.dev, entry.inode)
            if identity in seen:
                continue
            seen.add(identity)
            classification, reason = decide(entry, named, complete, delivered)
            if classification == "keep" and reason == "small or non-media evidence":
                continue
            key = f"{classification}: {reason}"
            result.bytes_by_reason[key] += entry.size
            result.files_by_reason[key] += 1
            if classification != "keep":
                result.rows.append(row(entry, classification, reason))
    return result


def write(result: Plan, directory: Path) -> tuple[Path, str]:
    directory.mkdir(mode=0o700, parents=True, exist_ok=True)
    rows = sorted(result.rows, key=lambda item: str(item["path"]))
    body = json.dumps({"rows": rows}, indent=1, sort_keys=True)
    path = directory / "dispositions.json"
    path.write_text(body, encoding="utf-8")
    return path, leaf_retirement.sha256_of(path)


def summary(result: Plan) -> dict[str, dict[str, float | int]]:
    return {
        key: {"files": result.files_by_reason[key], "gib": round(size / 2**30, 2)}
        for key, size in sorted(result.bytes_by_reason.items())
    }


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("roots", nargs="+", type=Path)
    parser.add_argument("--edits", type=Path, required=True)
    parser.add_argument("--scan", action="append", type=Path, default=[])
    parser.add_argument("--assume-complete", action="append", default=[])
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args(argv)
    # The completion gate compares absolute candidate paths with `root.parent`.
    roots = [path.resolve() for path in args.roots]
    scan = [path.resolve() for path in args.scan]
    result = plan(roots, [*roots, *scan], args.edits.resolve(), frozenset(args.assume_complete))
    path, digest = write(result, args.out)
    print(json.dumps({"dispositions": str(path), "sha256": digest, "summary": summary(result)}))
    return 0


if __name__ == "__main__":
    sys.exit(main())
