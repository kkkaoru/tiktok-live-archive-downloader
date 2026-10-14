"""Capture bounded local Python import dependencies without importing producers.

The inventory covers static replay imports, including lazy and relative imports.
It does not discover runtime-selected modules, native binaries or third-party
dependencies; callers must declare those separately and coordinate source writers.
"""

import argparse
import ast
import hashlib
import json
import os
import re
import stat
from dataclasses import dataclass
from pathlib import Path

PACKAGE = Path(__file__).resolve().parent
MAX_MODULES = 512
MAX_SOURCE_BYTES = 2 * 1024 * 1024
MAX_TOTAL_BYTES = 16 * 1024 * 1024
MODULE_PATTERN = re.compile(r"(?:[a-z][a-z0-9_]*|__init__)")


@dataclass(frozen=True, slots=True)
class Source:
    path: Path
    sha256: str
    identity: tuple[int, int, int, int, int]


def _identity(info: os.stat_result) -> tuple[int, int, int, int, int]:
    return info.st_dev, info.st_ino, info.st_size, info.st_mtime_ns, info.st_ctime_ns


def _read(path: Path) -> tuple[bytes, Source]:
    before = path.lstat()
    if (
        path.resolve(strict=True) != path
        or not stat.S_ISREG(before.st_mode)
        or before.st_size > MAX_SOURCE_BYTES
    ):
        raise ValueError("Code dependency must be bounded canonical regular source")
    descriptor = os.open(path, os.O_RDONLY | os.O_NOFOLLOW)
    with os.fdopen(descriptor, "rb") as stream:
        if _identity(os.fstat(stream.fileno())) != _identity(before):
            raise ValueError("Code dependency changed before reading")
        data = stream.read(MAX_SOURCE_BYTES + 1)
        if len(data) > MAX_SOURCE_BYTES or _identity(path.lstat()) != _identity(before):
            raise ValueError("Code dependency changed during reading")
    return data, Source(path, hashlib.sha256(data).hexdigest(), _identity(before))


def _local_names(node: ast.AST) -> tuple[str, ...]:
    if isinstance(node, ast.Import):
        return tuple(alias.name for alias in node.names)
    if not isinstance(node, ast.ImportFrom):
        return ()
    if node.level:
        if node.level != 1:
            raise ValueError("Relative code import escapes the flat replay package")
        return (
            (f"replay.{node.module}",)
            if node.module is not None
            else tuple(f"replay.{alias.name}" for alias in node.names)
        )
    if node.module == "replay":
        return tuple(f"replay.{alias.name}" for alias in node.names)
    return (node.module,) if node.module is not None else ()


def _dependency(name: str) -> str | None:
    if name == "replay":
        return "__init__"
    if not name.startswith("replay."):
        return None
    module = name.removeprefix("replay.")
    if MODULE_PATTERN.fullmatch(module) is None:
        raise ValueError("Code import is outside the flat replay module scope")
    return module


def capture(modules: tuple[str, ...], *, package: Path = PACKAGE) -> tuple[Source, ...]:
    """Read each static local dependency once, then recheck all source identities.

    No Python code is executed. Endpoint identity checks are not an atomic
    snapshot; a caller's coordinated before/after content checks remain required.
    """
    if not modules or package != package.resolve(strict=True) or not package.is_dir():
        raise ValueError("Expected modules and a canonical package directory")
    if any(MODULE_PATTERN.fullmatch(name) is None for name in modules):
        raise ValueError("Expected flat replay module names")
    pending = set(modules) | {"__init__"}
    found: dict[str, Source] = {}
    total = 0
    while pending:
        name = min(pending)
        pending.remove(name)
        if name in found:
            continue
        if len(found) >= MAX_MODULES:
            raise ValueError("Code dependency module budget exceeded")
        data, source = _read(package / f"{name}.py")
        total += len(data)
        if total > MAX_TOTAL_BYTES:
            raise ValueError("Code dependency aggregate byte budget exceeded")
        found[name] = source
        tree = ast.parse(data, filename=str(source.path))
        for node in ast.walk(tree):
            for imported in _local_names(node):
                dependency = _dependency(imported)
                if dependency is not None and dependency not in found:
                    pending.add(dependency)
    for source in found.values():
        if (
            source.path.resolve(strict=True) != source.path
            or _identity(source.path.lstat()) != source.identity
        ):
            raise ValueError("Code dependency changed during closure capture")
    return tuple(found[name] for name in sorted(found))


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("modules", nargs="+", help="Flat replay module names")
    parser.add_argument("--json", action="store_true", help="Emit an inputs JSON inventory")
    args = parser.parse_args(argv)
    sources = capture(tuple(args.modules) + ("code_dependency_inventory",))
    if args.json:
        print(json.dumps({"inputs": {str(item.path): item.sha256 for item in sources}}))
    else:
        print("\n".join(f"{item.sha256}  {item.path}" for item in sources))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
