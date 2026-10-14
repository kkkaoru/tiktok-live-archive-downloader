"""Run one replay module's CLI with cross-process reuse of large-file digests.

`python -m replay.digest_store_run STORE MODULE ARGS...` is `replay.digest_memo`
with the memo kept on disk: a large file's SHA-256 is computed once per STORE and
reused by every later process (for example one process per foreground window or per
workflow stage) while its device, inode, size, modification and change times are
unchanged. Wrapped modules and their recorded code bindings stay byte-identical; any
change to the file identity rehashes. Same module allow-list as `digest_memo`.
"""

from __future__ import annotations

import importlib
import sys
from collections.abc import Callable
from pathlib import Path
from types import ModuleType

from replay import digest_memo, digest_store, stages

Digest = Callable[[Path], str]


def default_store() -> Path:
    """The checkout-local digest store shared by every measurement and workflow process."""
    return Path(__file__).resolve().parents[2] / "edits" / "work" / "digest-store"


def persistent(original: Digest, store: Path) -> Digest:
    def wrapped(path: Path) -> str:
        if path.stat().st_size < digest_store.MINIMUM_BYTES:
            return original(path)
        return digest_store.cached_digest(store, path)

    return wrapped


def install(original: Digest, store: Path) -> list[str]:
    """Replace every loaded replay module's reference to `original`; return names."""
    replacement = persistent(original, store)
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
    if len(arguments) < 2 or arguments[1] not in digest_memo.ALLOWED_MODULES:
        allowed = ",".join(sorted(digest_memo.ALLOWED_MODULES))
        print(f"usage: digest_store_run STORE {{{allowed}}} ARGS...", file=sys.stderr)
        return 2
    module = importlib.import_module(arguments[1])
    install(stages.file_digest, Path(arguments[0]))
    entry = module.main
    sys.argv = [arguments[1], *arguments[2:]]
    result = entry()
    return result if isinstance(result, int) else 0


if __name__ == "__main__":
    raise SystemExit(main())
