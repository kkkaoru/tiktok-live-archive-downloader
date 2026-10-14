"""Run one replay module's CLI with per-process reuse of large-file digests.

Bounded jobs verify the same multi-gigabyte source once per foreground window,
so most of their time is spent re-hashing identical bytes. This wrapper runs
`python -m replay.digest_memo MODULE ARGS...`: inside that single process, a
large file's SHA-256 is computed once and reused only while its device, inode,
size, modification and change times are unchanged; any change rehashes. The
wrapped modules and their recorded code bindings stay byte-identical, and every
other check they perform is unchanged.
"""

from __future__ import annotations

import importlib
import os
import sys
from collections.abc import Callable
from pathlib import Path
from types import ModuleType

from replay import stages

MEMO_MINIMUM_BYTES = 64 * 1024 * 1024
ALLOWED_MODULES = frozenset({"replay.caption_measured_job", "replay.foreground_collection"})

Digest = Callable[[Path], str]


def memoized(digest: Digest) -> Digest:
    cache: dict[tuple[int, int, int, int, int], str] = {}

    def identity(path: Path) -> tuple[int, int, int, int, int]:
        status = os.lstat(path)
        return (
            status.st_dev,
            status.st_ino,
            status.st_size,
            status.st_mtime_ns,
            status.st_ctime_ns,
        )

    def wrapped(path: Path) -> str:
        before = identity(path)
        if before[2] < MEMO_MINIMUM_BYTES:
            return digest(path)
        cached = cache.get(before)
        if cached is not None:
            return cached
        value = digest(path)
        if identity(path) == before:
            cache[before] = value
        return value

    return wrapped


def install(original: Digest) -> list[str]:
    """Replace every loaded replay module's reference to `original`; return names."""
    replacement = memoized(original)
    patched: list[str] = []
    for name, module in list(sys.modules.items()):
        if not name.startswith("replay.") or not isinstance(module, ModuleType):
            continue
        if getattr(module, "file_digest", None) is original:
            module.__dict__["file_digest"] = replacement
            patched.append(name)
    return sorted(patched)


def main(argv: list[str] | None = None) -> int:
    arguments = sys.argv[1:] if argv is None else argv
    if not arguments or arguments[0] not in ALLOWED_MODULES:
        print(
            f"usage: digest_memo {{{','.join(sorted(ALLOWED_MODULES))}}} ARGS...", file=sys.stderr
        )
        return 2
    module = importlib.import_module(arguments[0])
    install(stages.file_digest)
    entry = module.main
    sys.argv = [arguments[0], *arguments[1:]]
    result = entry()
    return result if isinstance(result, int) else 0


if __name__ == "__main__":
    raise SystemExit(main())
