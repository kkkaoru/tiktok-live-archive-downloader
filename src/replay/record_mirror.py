"""Keep a second copy of the small, irreplaceable per-recording records on another volume.

Large intermediates are rebuilt from the restore recipe (`replay.delivery_restore`); what
cannot be rebuilt is the edit configuration, completion records and transcripts. Only small
text files are mirrored, copies are verified by size, and nothing is ever deleted.
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from collections.abc import Iterator, Sequence
from pathlib import Path

from replay.core import ReplayError

SUFFIXES = frozenset({".json", ".srt", ".vtt", ".txt"})
SKIPPED_DIRECTORIES = frozenset({"evidence", "proxy"})
MAX_FILE_BYTES = 8 * 1024 * 1024
PATTERNS = ("replay-*", "youtube-*")


def _files(directory: Path) -> Iterator[Path]:
    for entry in sorted(directory.iterdir()):
        if entry.is_symlink():
            continue
        if entry.is_dir():
            if entry.name not in SKIPPED_DIRECTORIES:
                yield from _files(entry)
        elif entry.suffix.lower() in SUFFIXES and entry.stat().st_size <= MAX_FILE_BYTES:
            yield entry


def recordings(edits: Path) -> list[Path]:
    return sorted(path for pattern in PATTERNS for path in edits.glob(pattern) if path.is_dir())


def volume(path: Path) -> int:
    """Device of the nearest existing ancestor, so a not-yet-created mirror is judged too."""
    existing = next(candidate for candidate in (path, *path.parents) if candidate.exists())
    return existing.stat().st_dev


def _copy_directory_modes(source: Path, target: Path, edits: Path) -> None:
    """Keep private directories private; TranscriptViewer rejects a group-readable summary store."""
    while source != edits:
        shutil.copymode(source, target)
        source, target = source.parent, target.parent


def mirror(edits: Path, destination: Path) -> dict[str, int]:
    """Copy new or changed records; refuse a destination on the same volume as the source."""
    if volume(destination) == volume(edits):
        raise ReplayError("The mirror must live on a different volume than the records")
    copied = unchanged = 0
    for recording in recordings(edits):
        for source in _files(recording):
            target = destination / source.relative_to(edits)
            target.parent.mkdir(parents=True, exist_ok=True)
            _copy_directory_modes(source.parent, target.parent, edits)
            if target.is_file() and target.stat().st_size == source.stat().st_size:
                unchanged += 1
                continue
            partial = target.with_name(f"{target.name}.partial")
            shutil.copy2(source, partial)
            partial.replace(target)
            copied += 1
    return {"copied": copied, "unchanged": unchanged}


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--edits", type=Path, default=Path("edits"))
    parser.add_argument("destination", type=Path)
    args = parser.parse_args(argv)
    try:
        print(json.dumps(mirror(args.edits, args.destination)))
    except ReplayError as error:
        print(error, file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
