"""Opt-in immutable scopes for repeated static source-closure inventories.

Consumers explicitly call the memo's capture method. Existing capture aliases
and live producers are never intercepted. A scope fails closed on package or
source identity changes; the next scope performs a fresh ordinary capture.
These endpoint checks are not an atomic snapshot or a substitute for source
writer coordination and recorded generation/content verification.
"""

import stat
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path
from typing import Protocol

from replay import code_dependency_inventory as inventory

MAX_CLOSURES = 64
Identity = tuple[int, int, int, int, int]


class Capture(Protocol):
    """The ordinary capture signature, including its explicit package bound."""

    def __call__(
        self, modules: tuple[str, ...], *, package: Path = inventory.PACKAGE
    ) -> tuple[inventory.Source, ...]: ...


def _identity(path: Path, *, directory: bool = False) -> Identity:
    info = path.lstat()
    if path.resolve(strict=True) != path or not (
        stat.S_ISDIR(info.st_mode) if directory else stat.S_ISREG(info.st_mode)
    ):
        raise ValueError("Inventory memo requires canonical ordinary paths")
    return inventory._identity(info)


class InventoryMemo:
    """Cache exact Source tuples while every observed identity remains immutable."""

    def __init__(
        self, package: Path = inventory.PACKAGE, *, capture_function: Capture | None = None
    ) -> None:
        self.package = package
        self._package_identity = _identity(package, directory=True)
        self._implementation = Path(inventory.__file__).resolve()
        self._implementation_identity = _identity(self._implementation)
        self._closures: dict[tuple[str, ...], tuple[inventory.Source, ...]] = {}
        self._sources: dict[Path, inventory.Source] = {}
        self._closed = False
        self._capture = capture_function

    def check(self) -> None:
        """Recheck directory additions/removals and all five source identity fields."""
        if self._closed:
            raise ValueError("Inventory memo scope is closed")
        try:
            if (
                _identity(self.package, directory=True) != self._package_identity
                or _identity(self._implementation) != self._implementation_identity
                or any(_identity(path) != source.identity for path, source in self._sources.items())
            ):
                raise ValueError("Code changed within inventory memo scope")
        except (OSError, ValueError):
            self.close()
            raise

    def capture(self, modules: tuple[str, ...]) -> tuple[inventory.Source, ...]:
        """Use ordinary bounded capture on the first call; never execute source code."""
        self.check()
        key = tuple(sorted(set(modules)))
        if key not in self._closures:
            if len(self._closures) >= MAX_CLOSURES:
                raise ValueError("Inventory memo closure budget exceeded")
            capture = inventory.capture if self._capture is None else self._capture
            sources = capture(modules, package=self.package)
            for source in sources:
                previous = self._sources.get(source.path)
                if previous is not None and previous != source:
                    self.close()
                    raise ValueError("Code changed during inventory memo capture")
                self._sources[source.path] = source
            self.check()
            self._closures[key] = sources
        result = self._closures[key]
        self.check()
        return result

    def close(self) -> None:
        """Discard cached inventories; no persistent cache or filesystem writes."""
        self._closed = True
        self._closures.clear()
        self._sources.clear()


@contextmanager
def inventory_memo(package: Path = inventory.PACKAGE) -> Iterator[InventoryMemo]:
    """Before/after checked ownership scope for future explicit callers."""
    memo = InventoryMemo(package)
    try:
        yield memo
        memo.check()
    finally:
        memo.close()
