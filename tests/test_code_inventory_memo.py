"""Exact inventory preservation and fail-closed immutable memo scopes."""

import os
from pathlib import Path

import pytest

from replay import code_dependency_inventory as inventory
from replay import code_inventory_memo as memo


@pytest.fixture
def package(tmp_path: Path) -> Path:
    root = tmp_path / "replay"
    root.mkdir()
    (root / "__init__.py").write_bytes(b"")
    (root / "entry.py").write_text("from .helper import value\n", encoding="utf-8")
    (root / "helper.py").write_text("value = 1\n", encoding="utf-8")
    return root


def test_outputs_match_and_only_one_ast_capture(
    package: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    expected = inventory.capture(("entry",), package=package)
    original = inventory.capture
    calls: list[tuple[str, ...]] = []

    def capture(
        modules: tuple[str, ...], *, package: Path = inventory.PACKAGE
    ) -> tuple[inventory.Source, ...]:
        calls.append(modules)
        return original(modules, package=package)

    monkeypatch.setattr(inventory, "capture", capture)
    with memo.inventory_memo(package) as cache:
        for _ in range(20):
            assert cache.capture(("entry",)) == expected
        assert cache.capture(("entry", "helper")) == expected
        assert cache.capture(("helper", "entry", "entry")) == expected
    assert calls == [("entry",), ("entry", "helper")]
    with pytest.raises(ValueError, match="closed"):
        cache.capture(("entry",))


@pytest.mark.parametrize("field", range(5))
def test_every_identity_field_invalidates(
    package: Path, monkeypatch: pytest.MonkeyPatch, field: int
) -> None:
    cache = memo.InventoryMemo(package)
    cache.capture(("entry",))
    original = memo._identity

    def changed(path: Path, *, directory: bool = False) -> memo.Identity:
        identity = original(path, directory=directory)
        if path.name == "helper.py":
            values = [value + (1 if index == field else 0) for index, value in enumerate(identity)]
            return values[0], values[1], values[2], values[3], values[4]
        return identity

    monkeypatch.setattr(memo, "_identity", changed)
    with pytest.raises(ValueError, match="Code changed"):
        cache.capture(("entry",))
    with pytest.raises(ValueError, match="closed"):
        cache.check()


@pytest.mark.parametrize("mutation", ["add", "remove", "dependency", "touch", "symlink"])
def test_source_and_dependency_changes_never_reuse(package: Path, mutation: str) -> None:
    cache = memo.InventoryMemo(package)
    before = cache.capture(("entry",))
    helper = package / "helper.py"
    if mutation == "add":
        (package / "new.py").write_bytes(b"value = 2\n")
    elif mutation == "remove":
        helper.unlink()
    elif mutation == "dependency":
        (package / "entry.py").write_text("from . import new\n", encoding="utf-8")
    elif mutation == "touch":
        os.utime(helper, ns=(helper.stat().st_atime_ns, helper.stat().st_mtime_ns + 1))
    else:
        helper.unlink()
        helper.symlink_to("entry.py")
    with pytest.raises((OSError, ValueError)):
        cache.capture(("entry",))
    assert before
    if mutation == "touch":
        with memo.inventory_memo(package) as fresh:
            assert fresh.capture(("entry",)) == inventory.capture(("entry",), package=package)


def test_exit_rechecks_source_and_failed_capture_is_not_cached(package: Path) -> None:
    with pytest.raises(ValueError, match="Code changed"), memo.inventory_memo(package) as cache:
        cache.capture(("entry",))
        (package / "helper.py").write_bytes(b"value = 2\n")
    with memo.inventory_memo(package) as fresh:
        with pytest.raises(ValueError):
            fresh.capture(())
        assert fresh.capture(("entry",)) == inventory.capture(("entry",), package=package)


def test_budget_and_canonical_package(package: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    alias = package.parent / "alias"
    alias.symlink_to(package, target_is_directory=True)
    with pytest.raises(ValueError, match="canonical"):
        memo.InventoryMemo(alias)
    monkeypatch.setattr(memo, "MAX_CLOSURES", 1)
    with memo.inventory_memo(package) as cache:
        cache.capture(("entry",))
        with pytest.raises(ValueError, match="budget"):
            cache.capture(("helper",))


@pytest.mark.parametrize("before_read", [False, True])
def test_changed_during_capture_is_rejected(
    package: Path, monkeypatch: pytest.MonkeyPatch, before_read: bool
) -> None:
    with pytest.raises(ValueError, match="Code changed"), memo.inventory_memo(package) as cache:
        cache.capture(("entry",))
        original = inventory.capture

        def changed(
            modules: tuple[str, ...], *, package: Path = inventory.PACKAGE
        ) -> tuple[inventory.Source, ...]:
            if before_read:
                (package / "helper.py").write_bytes(b"value = 3\n")
            result = original(modules, package=package)
            if not before_read:
                (package / "helper.py").write_bytes(b"value = 3\n")
            return result

        monkeypatch.setattr(inventory, "capture", changed)
        cache.capture(("helper",))
