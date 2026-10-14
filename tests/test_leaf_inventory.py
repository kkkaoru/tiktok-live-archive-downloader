"""Metadata-only inventory regressions using synthetic checkout trees."""

from __future__ import annotations

import hashlib
import json
import os
import socket
import stat
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass, replace
from pathlib import Path

import pytest

from replay import leaf_inventory as module
from replay.leaf_retirement import load_eligible


@dataclass(frozen=True)
class Tree:
    checkout: Path
    root: Path
    plan: Path
    output: Path

    def load(self) -> module.Plan:
        return module.load_plan(self.plan, checkout=self.checkout)

    def write_plan(self, value: object) -> None:
        self.plan.write_text(json.dumps(value), encoding="utf-8")


@pytest.fixture
def tree(tmp_path: Path) -> Tree:
    work = tmp_path / "edits" / "work"
    work.mkdir(parents=True)
    root = work / "selected"
    root.mkdir()
    plan = work / "plan.json"
    plan.write_text(json.dumps({"version": 1, "roots": ["edits/work/selected"]}), encoding="utf-8")
    return Tree(tmp_path, root, plan, work / "inventory.json")


def _json(path: Path) -> dict[str, object]:
    value: object = json.loads(path.read_bytes())
    assert isinstance(value, dict)
    assert all(isinstance(key, str) for key in value)
    return value


def _rows(path: Path) -> list[dict[str, object]]:
    value = _json(path)["rows"]
    assert isinstance(value, list)
    result: list[dict[str, object]] = []
    for item in value:
        assert isinstance(item, dict)
        assert all(isinstance(key, str) for key in item)
        result.append(item)
    return result


def test_regular_inventory_preserves_exact_nanoseconds_and_private_mode(tree: Tree) -> None:
    leaf = tree.root / "record.bin"
    leaf.write_bytes(b"data")
    timestamp = 1_791_587_898_870_780_088
    os.utime(leaf, ns=(timestamp, timestamp))
    info = leaf.lstat()
    totals = module.inventory(tree.load(), tree.output)
    row = _rows(tree.output)[0]
    assert row["mtimeNs"] == timestamp
    assert row["ctimeNs"] == info.st_ctime_ns
    assert type(row["mtimeNs"]) is int
    assert row["dev"] == info.st_dev
    assert row["inode"] == info.st_ino
    assert row["bytes"] == 4
    assert row["classification"] == "hold"
    assert totals["logicalBytes"] == 4
    assert totals["uniqueAllocatedBytes"] == info.st_blocks * 512
    assert stat.S_IMODE(tree.output.stat().st_mode) == 0o600
    assert _json(tree.output)["mediaContentsRead"] is False
    assert load_eligible(tree.output, hashlib.sha256(tree.output.read_bytes()).hexdigest()) == ()


def test_symlink_targets_are_recorded_but_never_followed(tree: Tree, tmp_path: Path) -> None:
    target = tmp_path / "outside"
    target.mkdir()
    (target / "secret.bin").write_bytes(b"not inventoried")
    link = tree.root / "alias"
    link.symlink_to(target, target_is_directory=True)
    (tree.root / "missing").symlink_to("absent.bin")
    totals = module.inventory(tree.load(), tree.output)
    rows = _rows(tree.output)
    assert totals["leaves"] == 2
    assert totals["logicalBytes"] == 0
    assert rows[0]["kind"] == "symlink"
    assert rows[0]["linkTarget"] == str(target)
    assert rows[1]["linkTarget"] == "absent.bin"
    assert _json(tree.output)["directoriesObserved"] == 1


def test_hardlinks_are_deduplicated_by_device_and_inode(tree: Tree) -> None:
    source = tree.root / "a.bin"
    source.write_bytes(b"abcdef")
    os.link(source, tree.root / "b.bin")
    totals = module.inventory(tree.load(), tree.output)
    assert totals["files"] == 2
    assert totals["uniqueInodes"] == 1
    assert totals["logicalBytes"] == 12
    assert totals["uniqueLogicalBytes"] == 6
    assert totals["duplicateLogicalBytes"] == 6
    assert totals["allocatedBytes"] == totals["uniqueAllocatedBytes"] * 2
    assert _rows(tree.output)[0]["nlink"] == 2


@pytest.mark.parametrize("name", ["models", "private", "downloads", ".git"])
def test_protected_nested_directories_are_excluded(tree: Tree, name: str) -> None:
    protected = tree.root / name
    protected.mkdir()
    (protected / "must-not-list.bin").write_bytes(b"private")
    (tree.root / "ordinary.bin").write_bytes(b"public")
    module.inventory(tree.load(), tree.output)
    assert _json(tree.output)["excludedDirectories"] == [f"edits/work/selected/{name}"]
    assert len(_rows(tree.output)) == 1


def test_multiple_roots_have_stable_sorted_rows(tree: Tree) -> None:
    other = tree.root.parent / "other"
    other.mkdir()
    (other / "z.bin").write_bytes(b"z")
    (tree.root / "b.bin").write_bytes(b"b")
    tree.write_plan({"version": 1, "roots": ["edits/work/selected", "edits/work/other"]})
    module.inventory(tree.load(), tree.output)
    assert [row["path"] for row in _rows(tree.output)] == [
        "edits/work/other/z.bin",
        "edits/work/selected/b.bin",
    ]


@pytest.mark.parametrize(
    "value",
    [
        None,
        [],
        {"version": 1, "roots": [], "extra": True},
        {"version": True, "roots": ["edits/work/selected"]},
        {"version": 2, "roots": ["edits/work/selected"]},
        {"version": 1, "roots": "edits/work/selected"},
        {"version": 1, "roots": []},
        {"version": 1, "roots": [None]},
        {"version": 1, "roots": [""]},
        {"version": 1, "roots": ["\x00"]},
        {"version": 1, "roots": ["/absolute"]},
        {"version": 1, "roots": ["../escape"]},
        {"version": 1, "roots": ["edits//work/selected"]},
        {"version": 1, "roots": ["edits/work/./selected"]},
        {"version": 1, "roots": ["edits/work/selected/"]},
        {"version": 1, "roots": ["private/selected"]},
        {"version": 1, "roots": ["downloads/selected"]},
        {"version": 1, "roots": ["edits/models/selected"]},
        {"version": 1, "roots": ["edits/work/selected", "edits/work/selected"]},
        {"version": 1, "roots": ["edits/work/selected", "edits/work/selected/nested"]},
        {"version": 1, "roots": ["edits/work/selected/nested", "edits/work/selected"]},
    ],
)
def test_invalid_plans_fail_before_leaf_observation(tree: Tree, value: object) -> None:
    (tree.root / "nested").mkdir()
    tree.write_plan(value)
    with pytest.raises(ValueError):
        tree.load()
    assert not tree.output.exists()


def test_root_count_is_bounded(tree: Tree, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(module, "MAX_ROOTS", 0)
    with pytest.raises(ValueError, match="bounded nonempty"):
        tree.load()


@pytest.mark.parametrize("payload", [b"not json", b"\xff"])
def test_invalid_plan_encoding(tree: Tree, payload: bytes) -> None:
    tree.plan.write_bytes(payload)
    with pytest.raises(ValueError):
        tree.load()


def test_linked_root_rejected(tree: Tree) -> None:
    alias = tree.root.parent / "linked"
    alias.symlink_to(tree.root, target_is_directory=True)
    tree.write_plan({"version": 1, "roots": ["edits/work/linked"]})
    with pytest.raises(ValueError, match="linked"):
        tree.load()


def test_linked_ancestor_rejected(tree: Tree) -> None:
    alias = tree.checkout / "alias"
    alias.symlink_to(tree.root.parent, target_is_directory=True)
    tree.write_plan({"version": 1, "roots": ["alias/selected"]})
    with pytest.raises(ValueError, match="linked"):
        tree.load()


def test_plan_outside_checkout_rejected(tree: Tree) -> None:
    with pytest.raises(ValueError, match="escapes"):
        module.load_plan(tree.plan, checkout=tree.root)


def test_linked_plan_rejected(tree: Tree) -> None:
    alias = tree.plan.with_name("linked-plan.json")
    alias.symlink_to(tree.plan)
    with pytest.raises(OSError):
        module.load_plan(alias, checkout=tree.checkout)


def test_plan_file_limit(tree: Tree, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(module, "MAX_PLAN_BYTES", 1)
    with pytest.raises(ValueError, match="bounded regular"):
        tree.load()


def test_plan_is_not_a_regular_file(tree: Tree) -> None:
    tree.plan.unlink()
    tree.plan.mkdir()
    with pytest.raises((ValueError, IsADirectoryError)):
        tree.load()


def test_loaded_plan_cannot_be_modified_in_memory(tree: Tree) -> None:
    plan = replace(tree.load(), sha256="0" * 64)
    with pytest.raises(ValueError, match="validated request"):
        module.inventory(plan, tree.output)


def test_output_collision_is_nonoverwriting(tree: Tree) -> None:
    tree.output.write_bytes(b"retained")
    with pytest.raises(FileExistsError):
        module.inventory(tree.load(), tree.output)
    assert tree.output.read_bytes() == b"retained"


def test_dangling_output_collision(tree: Tree) -> None:
    tree.output.symlink_to("absent-output.json")
    with pytest.raises(FileExistsError):
        module.inventory(tree.load(), tree.output)
    assert tree.output.is_symlink()


@pytest.mark.parametrize("location", ["outside.json", "private/output.json"])
def test_output_scope(tree: Tree, location: str) -> None:
    with pytest.raises(ValueError, match="under unprotected"):
        module.inventory(tree.load(), tree.checkout / location)


def test_output_inside_scanned_root_rejected(tree: Tree) -> None:
    with pytest.raises(ValueError, match="outside inventoried"):
        module.inventory(tree.load(), tree.root / "output.json")


def test_linked_output_parent_rejected(tree: Tree) -> None:
    alias = tree.root.parent / "output-alias"
    alias.symlink_to(tree.root.parent, target_is_directory=True)
    with pytest.raises(ValueError, match="linked"):
        module.inventory(tree.load(), alias / "output.json")


@pytest.mark.parametrize("limits", [module.Limits(0), module.Limits(-1), module.Limits(True)])
def test_invalid_limits(tree: Tree, limits: module.Limits) -> None:
    with pytest.raises(ValueError, match="positive integers"):
        module.inventory(tree.load(), tree.output, limits=limits)


def test_entry_limit_stops_without_output(tree: Tree) -> None:
    (tree.root / "a").write_bytes(b"a")
    (tree.root / "b").write_bytes(b"b")
    with pytest.raises(ValueError, match="entry budget"):
        module.inventory(tree.load(), tree.output, limits=module.Limits(max_entries=1))
    assert not tree.output.exists()


def test_depth_limit_stops_without_output(tree: Tree) -> None:
    (tree.root / "one" / "two").mkdir(parents=True)
    with pytest.raises(ValueError, match="depth budget"):
        module.inventory(tree.load(), tree.output, limits=module.Limits(max_depth=1))
    assert not tree.output.exists()


def test_metadata_limit_stops_without_output(tree: Tree) -> None:
    (tree.root / "a").write_bytes(b"a")
    with pytest.raises(ValueError, match="metadata exceeds"):
        module.inventory(tree.load(), tree.output, limits=module.Limits(max_metadata_bytes=1))
    assert not tree.output.exists()


def test_serialized_metadata_limit(tree: Tree, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(module._Walk, "_budget", lambda self, value: None)
    with pytest.raises(ValueError, match="Serialized inventory"):
        module.inventory(tree.load(), tree.output, limits=module.Limits(max_metadata_bytes=1))


def test_fifo_is_metadata_only_and_always_hold(tree: Tree, monkeypatch: pytest.MonkeyPatch) -> None:
    os.mkfifo(tree.root / "fifo")
    original = os.open

    def observed(
        path: str | bytes | os.PathLike[str] | os.PathLike[bytes],
        flags: int,
        mode: int = 0o777,
        *,
        dir_fd: int | None = None,
    ) -> int:
        if os.fspath(path) == "fifo":
            raise AssertionError("A metadata inventory must never open a FIFO")
        return original(path, flags, mode, dir_fd=dir_fd)

    monkeypatch.setattr(os, "open", observed)
    totals = module.inventory(tree.load(), tree.output)
    row = _rows(tree.output)[0]
    assert row["kind"] == "special"
    assert row["specialType"] == "fifo"
    assert row["classification"] == "hold"
    assert row["bytes"] == row["linkBytes"] == 0
    assert row["linkTarget"] is None
    assert stat.S_ISFIFO(tree.root.joinpath("fifo").lstat().st_mode)
    assert totals["special"] == 1
    assert totals["files"] == 0


def test_socket_is_metadata_only_and_always_hold(
    tree: Tree, monkeypatch: pytest.MonkeyPatch
) -> None:
    # Relative binding avoids Darwin's short AF_UNIX address limit without external scratch.
    monkeypatch.chdir(tree.root)
    with socket.socket(socket.AF_UNIX, socket.SOCK_STREAM) as endpoint:
        endpoint.bind("control.socket")
        module.inventory(tree.load(), tree.output)
    row = _rows(tree.output)[0]
    assert row["kind"] == "special"
    assert row["specialType"] == "socket"
    assert row["mode"] == tree.root.joinpath("control.socket").lstat().st_mode
    assert row["classification"] == "hold"


@pytest.mark.parametrize(
    ("mode", "expected"),
    [(stat.S_IFCHR, "character"), (stat.S_IFBLK, "block"), (0, "unknown")],
)
def test_other_special_types_remain_hold(mode: int, expected: str) -> None:
    identity = module.Identity(1, 2, mode, 1, 3, 4, 5, 0)
    row = module.Leaf("synthetic", identity, None).json()
    assert row["specialType"] == expected
    assert row["statBytes"] == 3
    assert row["bytes"] == 0
    assert row["classification"] == "hold"


def test_leaf_mutation_before_publication_is_rejected(
    tree: Tree, monkeypatch: pytest.MonkeyPatch
) -> None:
    path = tree.root / "a"
    path.write_bytes(b"original")
    original = module._Walk.finish

    def changed(walk: module._Walk) -> None:
        path.write_bytes(b"changed")
        original(walk)

    monkeypatch.setattr(module._Walk, "finish", changed)
    with pytest.raises(ValueError, match="leaf changed"):
        module.inventory(tree.load(), tree.output)
    assert not tree.output.exists()


def test_membership_change_before_publication_is_rejected(
    tree: Tree, monkeypatch: pytest.MonkeyPatch
) -> None:
    original = module._Walk.finish

    def changed(walk: module._Walk) -> None:
        (tree.root / "new").write_bytes(b"new")
        original(walk)

    monkeypatch.setattr(module._Walk, "finish", changed)
    with pytest.raises(ValueError, match="membership changed"):
        module.inventory(tree.load(), tree.output)


def test_plan_change_before_publication_is_rejected(
    tree: Tree, monkeypatch: pytest.MonkeyPatch
) -> None:
    original = module._Walk.finish

    def changed(walk: module._Walk) -> None:
        original(walk)
        tree.plan.write_text('{"version": 1, "roots": ["edits/work/selected"]}\n')

    monkeypatch.setattr(module._Walk, "finish", changed)
    with pytest.raises(ValueError, match="plan changed"):
        module.inventory(tree.load(), tree.output)


def test_output_ancestor_replacement_is_rejected(
    tree: Tree, monkeypatch: pytest.MonkeyPatch
) -> None:
    parent = tree.root.parent / "publication"
    parent.mkdir()
    output = parent / "result.json"
    original = module._Walk.finish

    def changed(walk: module._Walk) -> None:
        original(walk)
        parent.rename(parent.with_name("old-publication"))
        parent.mkdir()

    monkeypatch.setattr(module._Walk, "finish", changed)
    with pytest.raises(ValueError, match="output ancestor changed"):
        module.inventory(tree.load(), output)
    assert not output.exists()


def test_conflicting_hardlink_snapshot_is_rejected(tree: Tree) -> None:
    path = tree.root / "a"
    path.write_bytes(b"a")
    identity = module.Identity.of(path.lstat())
    with pytest.raises(ValueError, match="Conflicting"):
        module._totals(
            [module.Leaf("a", identity, None), module.Leaf("b", replace(identity, size=2), None)]
        )


def test_main_success(
    tree: Tree, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    original = module.load_plan

    def public(path: Path, *, checkout: Path = tree.checkout) -> module.Plan:
        return original(path, checkout=checkout)

    monkeypatch.setattr(module, "load_plan", public)
    assert module.main([str(tree.plan), str(tree.output)]) == 0
    assert '"leaves": 0' in capsys.readouterr().out


def test_main_failure(tree: Tree, capsys: pytest.CaptureFixture[str]) -> None:
    assert module.main([str(tree.plan), str(tree.output)]) == 1
    assert "Leaf inventory failed:" in capsys.readouterr().err


def test_regular_leaf_contents_are_never_opened(
    tree: Tree, monkeypatch: pytest.MonkeyPatch
) -> None:
    leaf = tree.root / "movie.bin"
    leaf.write_bytes(b"unread content")
    original = os.open

    def observed(
        path: str | bytes | os.PathLike[str] | os.PathLike[bytes],
        flags: int,
        mode: int = 0o777,
        *,
        dir_fd: int | None = None,
    ) -> int:
        if os.fspath(path) in {str(leaf), "movie.bin"}:
            raise AssertionError("Inventory tried to open regular leaf content")
        return original(path, flags, mode, dir_fd=dir_fd)

    monkeypatch.setattr(os, "open", observed)
    assert module.inventory(tree.load(), tree.output)["files"] == 1


def test_ancestor_replaced_by_symlink_before_open_is_not_followed(
    tree: Tree, monkeypatch: pytest.MonkeyPatch
) -> None:
    outside = tree.checkout / "outside"
    outside.mkdir()
    (outside / "must-not-observe.bin").write_bytes(b"private")
    plan = tree.load()
    original = os.open

    def replaced(
        path: str | bytes | os.PathLike[str] | os.PathLike[bytes],
        flags: int,
        mode: int = 0o777,
        *,
        dir_fd: int | None = None,
    ) -> int:
        if os.fspath(path) == "selected":
            tree.root.rename(tree.root.with_name("original-selected"))
            tree.root.symlink_to(outside, target_is_directory=True)
        return original(path, flags, mode, dir_fd=dir_fd)

    monkeypatch.setattr(os, "open", replaced)
    with pytest.raises(OSError):
        module.inventory(plan, tree.output)
    assert not tree.output.exists()
    assert (outside / "must-not-observe.bin").read_bytes() == b"private"


def test_symlink_changed_during_observation_is_rejected(
    tree: Tree, monkeypatch: pytest.MonkeyPatch
) -> None:
    link = tree.root / "alias"
    link.symlink_to("old")
    original = os.readlink

    def changed(path: str | os.PathLike[str], *, dir_fd: int | None = None) -> str:
        result = original(path, dir_fd=dir_fd)
        if os.fspath(path) == "alias":
            link.unlink()
            link.symlink_to("new")
        return result

    monkeypatch.setattr(os, "readlink", changed)
    with pytest.raises(ValueError, match="leaf changed during observation"):
        module.inventory(tree.load(), tree.output)
    assert not tree.output.exists()


def test_terminal_checks_reuse_one_descriptor_per_parent(
    tree: Tree, monkeypatch: pytest.MonkeyPatch
) -> None:
    (tree.root / "a").write_bytes(b"a")
    (tree.root / "b").write_bytes(b"b")
    (tree.root / "c").write_bytes(b"c")
    (tree.root / "empty").mkdir()
    walk = module._Walk(tree.load(), module.Limits())
    walk.directory(tree.root)
    opened: list[Path] = []
    original = module._directory_fd

    @contextmanager
    def observed(path: Path, checkout: Path) -> Iterator[int]:
        opened.append(path)
        with original(path, checkout) as descriptor:
            yield descriptor

    monkeypatch.setattr(module, "_directory_fd", observed)
    walk.finish()
    assert opened == [tree.root]


@pytest.mark.parametrize("change", ["add", "delete", "ctime", "replace", "link"])
def test_terminal_group_mutations_fail_closed(
    tree: Tree, monkeypatch: pytest.MonkeyPatch, change: str
) -> None:
    (tree.root / "a").write_bytes(b"a")
    (tree.root / "b").write_bytes(b"b")
    walk = module._Walk(tree.load(), module.Limits())
    walk.directory(tree.root)
    original = os.stat
    changed = False

    def observed(
        path: str | bytes | os.PathLike[str] | os.PathLike[bytes] | int,
        *,
        dir_fd: int | None = None,
        follow_symlinks: bool = True,
    ) -> os.stat_result:
        nonlocal changed
        if not changed and path in {"a", "b"}:
            changed = True
            remaining = "b" if path == "a" else "a"
            if change == "add":
                (tree.root / "new").write_bytes(b"new")
            elif change == "delete":
                (tree.root / remaining).unlink()
            elif change == "ctime":
                (tree.root / remaining).chmod(0o400)
            else:
                old = tree.root.with_name("old-selected")
                tree.root.rename(old)
                if change == "replace":
                    tree.root.mkdir()
                else:
                    tree.root.symlink_to(old, target_is_directory=True)
        return original(path, dir_fd=dir_fd, follow_symlinks=follow_symlinks)

    monkeypatch.setattr(os, "stat", observed)
    with pytest.raises((ValueError, OSError)):
        walk.finish()
    assert not tree.output.exists()
