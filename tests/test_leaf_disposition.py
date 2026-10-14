"""Owner decisions classify metadata; leaf contents never enter this operation."""

from __future__ import annotations

import hashlib
import json
import os
import stat
from dataclasses import dataclass, replace
from pathlib import Path
from typing import TypeGuard

import pytest

from replay import leaf_disposition as module
from replay.leaf_retirement import load_eligible


def _object(value: object) -> TypeGuard[dict[str, object]]:
    return isinstance(value, dict) and all(isinstance(key, str) for key in value)


def _document(path: Path) -> dict[str, object]:
    value: object = json.loads(path.read_bytes())
    assert _object(value)
    return value


def _first_row(path: Path) -> dict[str, object]:
    values = _document(path)["rows"]
    assert isinstance(values, list)
    value: object = values[0]
    assert _object(value)
    return value


def _row(**changes: object) -> dict[str, object]:
    return (
        dict(
            path="edits/work/selected/generated.bin",
            kind="file",
            classification="hold",
            dev=7,
            inode=9,
            nlink=1,
            bytes=19,
            statBytes=19,
            mode=stat.S_IFREG | 0o600,
            specialType=None,
            mtimeNs=1_791_587_898_870_780_088,
            ctimeNs=1_791_587_898_870_780_099,
            linkTarget=None,
            linkBytes=0,
            blocks512=8,
            allocatedBytes=4096,
        )
        | changes
    )


def _inventory(*rows: dict[str, object], **changes: object) -> dict[str, object]:
    return (
        dict(
            version=1,
            kind="metadata-only-leaf-inventory",
            roots=["edits/work/selected"],
            classification="hold",
            mediaContentsRead=False,
            rows=list(rows),
        )
        | changes
    )


@dataclass(frozen=True)
class Tree:
    checkout: Path
    work: Path
    inventory_path: Path
    policy_path: Path
    output: Path
    approval: module.Reference
    terminal: module.Reference

    def policy(self) -> module.Policy:
        return module.Policy(
            owner="synthetic gate owner",
            eligible_trees=(Path("edits/work/selected"),),
            approval=self.approval,
            terminal_proofs=(self.terminal,),
        )

    def reference(self, path: Path, data: bytes) -> module.Reference:
        path.write_bytes(data)
        return module.Reference(path, hashlib.sha256(data).hexdigest())

    def inventory(self, document: object) -> module.Reference:
        return self.reference(self.inventory_path, json.dumps(document).encode())

    def bind_policy(self, policy: module.Policy) -> module.Reference:
        return module.write_policy(policy, self.policy_path, checkout=self.checkout)

    def run(self, document: object, policy: module.Policy | None = None) -> module.Summary:
        source = self.inventory(document)
        intent = self.bind_policy(self.policy() if policy is None else policy)
        return module.build_disposition(source, intent, self.output, checkout=self.checkout)


@pytest.fixture
def tree(tmp_path: Path) -> Tree:
    work = tmp_path / "edits" / "work"
    (work / "selected").mkdir(parents=True)
    approval = work / "approval.json"
    terminal = work / "terminal.status"
    approval.write_bytes(b'{"approved":true}')
    terminal.write_bytes(b"0\n")
    return Tree(
        tmp_path,
        work,
        work / "inventory.json",
        work / "policy.json",
        work / "disposition.json",
        module.Reference(approval, hashlib.sha256(approval.read_bytes()).hexdigest()),
        module.Reference(terminal, hashlib.sha256(terminal.read_bytes()).hexdigest()),
    )


def test_eligible_metadata_is_pinned_compatible_and_keeps_exact_integer_identity(
    tree: Tree,
) -> None:
    result = tree.run(_inventory(_row()))
    row = _first_row(tree.output)
    assert result == module.Summary(1, 1, 0, 0, 19, 4096)
    assert row["path"] == str(tree.work / "selected" / "generated.bin")
    assert row["mtimeNs"] == 1_791_587_898_870_780_088
    assert type(row["mtimeNs"]) is int
    assert _document(tree.output)["mediaContentsRead"] is False
    assert _document(tree.output)["leavesRemoved"] == 0
    assert _document(tree.output)["ownerPolicy"] == module.policy_document(
        tree.policy(), checkout=tree.checkout
    )
    assert stat.S_IMODE(tree.output.stat().st_mode) == 0o600
    eligible = load_eligible(tree.output, hashlib.sha256(tree.output.read_bytes()).hexdigest())
    assert len(eligible) == 1
    assert eligible[0].mtime_ns == row["mtimeNs"]
    assert not (tree.work / "selected" / "generated.bin").exists()


def test_default_policy_holds_without_claiming_terminal_success(tree: Tree) -> None:
    result = tree.run(_inventory(_row()), module.Policy(owner="owner"))
    assert result.hold == 1
    assert result.eligible == 0
    assert _first_row(tree.output)["reason"] == "no explicit eligibility decision"


@pytest.mark.parametrize("preservation", ["tree", "path"])
def test_explicit_keep_has_priority_over_eligibility(tree: Tree, preservation: str) -> None:
    policy = tree.policy()
    if preservation == "tree":
        policy = replace(policy, keep_trees=(Path("edits/work/selected"),))
    else:
        policy = replace(policy, keep_paths=(Path("edits/work/selected/generated.bin"),))
    assert tree.run(_inventory(_row()), policy).keep == 1
    assert _first_row(tree.output)["classification"] == "keep"


def test_exact_path_decision_does_not_promote_sibling(tree: Tree) -> None:
    policy = replace(
        tree.policy(),
        eligible_trees=(),
        eligible_paths=(Path("edits/work/selected/generated.bin"),),
    )
    result = tree.run(
        _inventory(_row(), _row(path="edits/work/selected/other.bin", inode=10)), policy
    )
    assert result.eligible == 1
    assert result.hold == 1


@pytest.mark.parametrize("component", ["models", "private", "downloads", ".git"])
def test_protected_descendants_remain_held(tree: Tree, component: str) -> None:
    assert tree.run(_inventory(_row(path=f"edits/work/selected/{component}/leaf"))).hold == 1
    assert _first_row(tree.output)["reason"] == "protected path component"


@pytest.mark.parametrize(
    "row",
    [
        _row(nlink=2),
        _row(
            kind="symlink",
            mode=stat.S_IFLNK | 0o777,
            bytes=0,
            statBytes=5,
            linkBytes=5,
            linkTarget="alias",
        ),
        _row(kind="special", mode=stat.S_IFIFO | 0o600, bytes=0, statBytes=0, specialType="fifo"),
        _row(kind="special", mode=stat.S_IFSOCK | 0o600, bytes=0, specialType="socket"),
        _row(kind="special", mode=stat.S_IFCHR | 0o600, bytes=0, specialType="character"),
        _row(kind="special", mode=stat.S_IFBLK | 0o600, bytes=0, specialType="block"),
        _row(kind="special", mode=0, bytes=0, specialType="unknown"),
    ],
)
def test_uncertain_and_special_leaves_are_never_promoted(
    tree: Tree, row: dict[str, object]
) -> None:
    assert tree.run(_inventory(row)).hold == 1
    assert load_eligible(tree.output, hashlib.sha256(tree.output.read_bytes()).hexdigest()) == ()


def test_duplicate_inode_observations_hold_even_with_claimed_single_links(tree: Tree) -> None:
    result = tree.run(_inventory(_row(), _row(path="edits/work/selected/alias.bin")))
    assert result.hold == 2
    assert result.eligible == 0


def test_special_file_remains_hold_even_when_explicitly_preserved(tree: Tree) -> None:
    special = _row(kind="special", mode=stat.S_IFIFO, bytes=0, specialType="fifo")
    policy = replace(tree.policy(), keep_trees=(Path("edits/work/selected"),))
    result = tree.run(_inventory(special), policy)
    assert result.hold == 1
    assert result.keep == 0
    assert _first_row(tree.output)["reason"] == "special file is never eligible"


@pytest.mark.parametrize(
    "row",
    [
        _row(extra=True),
        _row(path="edits/work/selected"),
        _row(path="edits/work/selected-other/leaf"),
        _row(path="../escape"),
        _row(kind="directory"),
        _row(classification="eligible"),
        _row(inode=0),
        _row(nlink=0),
        _row(dev=True),
        _row(bytes=1.0),
        _row(ctimeNs="1"),
        _row(blocks512=-1),
        _row(allocatedBytes=1),
        _row(linkTarget="bad"),
        _row(linkBytes=1),
        _row(mode=stat.S_IFDIR),
        _row(statBytes=18),
        _row(specialType="fifo"),
        _row(kind="symlink", mode=stat.S_IFLNK, bytes=0),
        _row(kind="symlink", mode=stat.S_IFREG, bytes=0, statBytes=1, linkBytes=1, linkTarget="x"),
        _row(kind="special", mode=stat.S_IFDIR, bytes=0, specialType="unknown"),
        _row(kind="special", mode=stat.S_IFIFO, bytes=0, specialType="socket"),
        _row(kind="special", mode=stat.S_IFIFO, bytes=1, specialType="fifo"),
    ],
)
def test_malformed_or_preclassified_rows_refuse_before_publication(
    tree: Tree, row: dict[str, object]
) -> None:
    with pytest.raises(ValueError):
        tree.run(_inventory(row))
    assert not tree.output.exists()


@pytest.mark.parametrize(
    "document",
    [
        [],
        _inventory(version=True),
        _inventory(version=2),
        _inventory(kind="other"),
        _inventory(classification="eligible"),
        _inventory(mediaContentsRead=True),
        _inventory(roots=[]),
        _inventory(roots="edits/work/selected"),
        _inventory(roots=["private/data"]),
        _inventory(roots=["edits/work/selected", "edits/work/selected"]),
        _inventory(roots=["edits/work/selected", "edits/work/selected/nested"]),
        _inventory(roots=["edits/work/selected/nested", "edits/work/selected"]),
        _inventory(rows=None),
        _inventory(_row(), _row()),
        _inventory(rows=[1]),
    ],
)
def test_invalid_inventory_contract_is_not_laundered_by_owner_approval(
    tree: Tree, document: object
) -> None:
    with pytest.raises(ValueError):
        tree.run(document)
    assert not tree.output.exists()


@pytest.mark.parametrize(
    "path", ["/outside", "../escape", "edits//work/x", "edits/work/./x", "edits/work/x/", "", "a\n"]
)
def test_policy_paths_require_canonical_component_containment(tree: Tree, path: str) -> None:
    policy = module.policy_document(tree.policy(), checkout=tree.checkout)
    policy["eligibleTrees"] = [path]
    source = tree.inventory(_inventory(_row()))
    reference = tree.reference(tree.policy_path, json.dumps(policy).encode())
    with pytest.raises(ValueError):
        module.build_disposition(source, reference, tree.output, checkout=tree.checkout)


@pytest.mark.parametrize(
    "change", ["owner", "approval", "proofs", "duplicate", "unknown", "version"]
)
def test_policy_rejects_missing_authority_or_unknown_contract(tree: Tree, change: str) -> None:
    policy = module.policy_document(tree.policy(), checkout=tree.checkout)
    if change == "owner":
        policy["owner"] = " "
    elif change == "approval":
        policy["approval"] = None
    elif change == "proofs":
        policy["terminalProofs"] = []
    elif change == "duplicate":
        policy["eligiblePaths"] = ["edits/work/selected/a", "edits/work/selected/a"]
    elif change == "unknown":
        policy["unreviewed"] = True
    else:
        policy["version"] = True
    source = tree.inventory(_inventory(_row()))
    reference = tree.reference(tree.policy_path, json.dumps(policy).encode())
    with pytest.raises(ValueError):
        module.build_disposition(source, reference, tree.output, checkout=tree.checkout)


@pytest.mark.parametrize(
    "value",
    [
        {"path": "edits/work/approval.json", "sha256": "bad"},
        {"path": "edits/work/approval.json", "sha256": "a" * 64, "extra": True},
        None,
    ],
)
def test_malformed_proof_reference_is_refused(tree: Tree, value: object) -> None:
    policy = module.policy_document(tree.policy(), checkout=tree.checkout)
    policy["terminalProofs"] = [value]
    source = tree.inventory(_inventory(_row()))
    reference = tree.reference(tree.policy_path, json.dumps(policy).encode())
    with pytest.raises(ValueError):
        module.build_disposition(source, reference, tree.output, checkout=tree.checkout)


def test_shared_policy_can_cover_other_batches_without_promoting_unregistered_roots(
    tree: Tree,
) -> None:
    policy = replace(tree.policy(), eligible_trees=(Path("edits/work/selected-other"),))
    result = tree.run(_inventory(_row()), policy)
    assert result.hold == 1
    assert result.eligible == 0
    assert _first_row(tree.output)["reason"] == "no explicit eligibility decision"


def test_policy_union_remains_bound_to_each_inventory_batch(tree: Tree) -> None:
    policy = replace(
        tree.policy(),
        eligible_trees=(Path("edits/work/selected"), Path("edits/work/selected-other")),
    )
    result = tree.run(_inventory(_row()), policy)
    assert result.eligible == 1
    assert _document(tree.output)["roots"] == [str(tree.work / "selected")]


@pytest.mark.parametrize(
    ("path", "trees", "expected"),
    [
        ("edits/work/batch/part/leaf", (), False),
        ("edits/work/batch/part", ("edits/work/batch/part",), True),
        ("edits/work/batch/part/nested/leaf", ("edits/work/batch/part",), True),
        ("edits/work/batch/part-other/leaf", ("edits/work/batch/part",), False),
        ("edits/work/batch/part", ("edits/work/batch/part/nested",), False),
        ("edits/work/batch-other/part/leaf", ("edits/work/batch",), False),
        ("edits/work/batch/part/leaf", ("edits/work/other", "edits/work/batch"), True),
        (
            "edits/work/batch/part/leaf",
            ("edits/work/batch/part", "edits/work/batch"),
            True,
        ),
    ],
)
def test_indexed_ancestor_union_matches_component_containment(
    path: str, trees: tuple[str, ...], expected: bool
) -> None:
    original_trees = tuple(Path(value) for value in trees)
    assert module._inside(Path(path), original_trees) is expected
    assert module._indexed_inside(Path(path), frozenset(original_trees)) is expected


def test_many_sibling_decisions_preserve_unregistered_paths_and_keep_precedence(
    tree: Tree,
) -> None:
    policy = replace(
        tree.policy(),
        eligible_trees=tuple(
            Path(f"edits/work/selected/part-{index:04d}") for index in range(4096)
        ),
        keep_trees=(Path("edits/work/selected/part-4095/keep"),),
        keep_paths=(Path("edits/work/selected/part-4095/exact.bin"),),
    )
    result = tree.run(
        _inventory(
            _row(path="edits/work/selected/part-4095/nested/leaf", inode=11),
            _row(path="edits/work/selected/part-4095-other/leaf", inode=12),
            _row(path="edits/work/selected/unregistered.bin", inode=13),
            _row(path="edits/work/selected/part-4095/keep/leaf", inode=14),
            _row(path="edits/work/selected/part-4095/exact.bin", inode=15),
        ),
        policy,
    )
    assert result == module.Summary(5, 1, 2, 2, 19, 4096)
    rows = _document(tree.output)["rows"]
    assert isinstance(rows, list)
    assert [row["classification"] for row in rows if _object(row)] == [
        "eligible",
        "hold",
        "hold",
        "keep",
        "keep",
    ]


def test_membership_work_is_bounded_by_path_depth_not_sibling_count(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    trees = frozenset(Path(f"edits/work/batch/part-{index:04d}") for index in range(4096))
    original_hash = Path.__hash__
    hash_calls = 0

    def counted_hash(path: Path) -> int:
        nonlocal hash_calls
        hash_calls += 1
        return original_hash(path)

    monkeypatch.setattr(Path, "__hash__", counted_hash)
    assert module._indexed_inside(Path("edits/work/batch/part-4095/nested/leaf"), trees)
    assert hash_calls == 3
    assert not module._indexed_inside(Path("edits/work/batch/part-4095-other/nested/leaf"), trees)
    assert hash_calls == 10


def test_proof_bytes_must_match_the_approved_reference(tree: Tree) -> None:
    tree.terminal.path.write_bytes(b"1\n")
    with pytest.raises(ValueError, match="SHA-256"):
        tree.bind_policy(tree.policy())
    assert not tree.policy_path.exists()


@pytest.mark.parametrize("payload", [b"{}", b'{"rows":[],"rows":[]}', b"not json", b"\xff"])
def test_malformed_raw_inventory_metadata(tree: Tree, payload: bytes) -> None:
    source = tree.reference(tree.inventory_path, payload)
    intent = tree.bind_policy(tree.policy())
    with pytest.raises(ValueError):
        module.build_disposition(source, intent, tree.output, checkout=tree.checkout)


@pytest.mark.parametrize("limit", [0, True, module.MAX_JSON_BYTES + 1, 10])
def test_metadata_budgets_are_hard_refusals(tree: Tree, limit: int) -> None:
    source = tree.inventory(_inventory(_row()))
    intent = tree.bind_policy(tree.policy())
    with pytest.raises(ValueError):
        module.build_disposition(
            source, intent, tree.output, checkout=tree.checkout, max_json_bytes=limit
        )
    assert not tree.output.exists()


def test_output_budget_accounts_for_the_result_expansion(tree: Tree) -> None:
    source = tree.inventory(_inventory(_row()))
    intent = tree.bind_policy(tree.policy())
    limit = max(source.path.stat().st_size, intent.path.stat().st_size)
    with pytest.raises(ValueError, match="output exceeds"):
        module.build_disposition(
            source, intent, tree.output, checkout=tree.checkout, max_json_bytes=limit
        )


@pytest.mark.parametrize("destination", ["selected", "source", "outside", "protected"])
def test_output_cannot_escape_or_enter_the_selected_leaf_scope(
    tree: Tree, destination: str
) -> None:
    source = tree.inventory(_inventory(_row()))
    intent = tree.bind_policy(tree.policy())
    output = tree.output
    if destination == "selected":
        output = tree.work / "selected" / "output.json"
    elif destination == "source":
        output = source.path
    elif destination == "outside":
        output = tree.checkout.parent / "output.json"
    else:
        output = tree.work / "models" / "output.json"
    with pytest.raises(ValueError):
        module.build_disposition(source, intent, output, checkout=tree.checkout)


def test_exclusive_output_does_not_replace_existing_evidence(tree: Tree) -> None:
    tree.output.write_bytes(b"retained")
    with pytest.raises(FileExistsError):
        tree.run(_inventory(_row()))
    assert tree.output.read_bytes() == b"retained"


def test_linked_metadata_and_output_ancestors_are_rejected(tree: Tree) -> None:
    alias = tree.work / "alias"
    alias.symlink_to(tree.work / "selected", target_is_directory=True)
    with pytest.raises(ValueError, match="ancestor"):
        module.write_policy(
            module.Policy(owner="owner"), alias / "output.json", checkout=tree.checkout
        )


@pytest.mark.parametrize("kind", ["fifo", "symlink", "directory"])
def test_metadata_input_nonregular_types_are_refused_without_opening(tree: Tree, kind: str) -> None:
    if kind == "fifo":
        os.mkfifo(tree.inventory_path)
    elif kind == "symlink":
        tree.inventory_path.symlink_to(tree.approval.path)
    else:
        tree.inventory_path.mkdir()
    intent = tree.bind_policy(tree.policy())
    with pytest.raises(ValueError, match="regular"):
        module.build_disposition(
            module.Reference(tree.inventory_path, "a" * 64),
            intent,
            tree.output,
            checkout=tree.checkout,
        )


@pytest.mark.parametrize("path", ["private/input.json", "edits/work/input.bin"])
def test_only_unprotected_declared_metadata_is_read(tree: Tree, path: str) -> None:
    intent = tree.bind_policy(tree.policy())
    with pytest.raises(ValueError, match="metadata"):
        module.build_disposition(
            module.Reference(Path(path), "a" * 64), intent, tree.output, checkout=tree.checkout
        )


def test_same_byte_replacement_of_inventory_is_detected_before_publication(
    tree: Tree, monkeypatch: pytest.MonkeyPatch
) -> None:
    source = tree.inventory(_inventory(_row()))
    intent = tree.bind_policy(tree.policy())
    original = module._proofs

    def replace_inventory(
        policy: module.Policy, *, checkout: Path, limit: int
    ) -> list[module._Document]:
        data = source.path.read_bytes()
        replacement = source.path.with_name("replacement.json")
        replacement.write_bytes(data)
        replacement.replace(source.path)
        return original(policy, checkout=checkout, limit=limit)

    monkeypatch.setattr(module, "_proofs", replace_inventory)
    with pytest.raises(ValueError, match="identity changed"):
        module.build_disposition(source, intent, tree.output, checkout=tree.checkout)
    assert not tree.output.exists()


def test_ctime_change_with_same_bytes_inode_size_and_mtime_is_refused(
    tree: Tree, monkeypatch: pytest.MonkeyPatch
) -> None:
    source = tree.inventory(_inventory(_row()))
    intent = tree.bind_policy(tree.policy())
    before = source.path.stat()
    original = module._proofs

    def change_ctime(
        policy: module.Policy, *, checkout: Path, limit: int
    ) -> list[module._Document]:
        source.path.chmod(0o400)
        source.path.chmod(stat.S_IMODE(before.st_mode))
        return original(policy, checkout=checkout, limit=limit)

    monkeypatch.setattr(module, "_proofs", change_ctime)
    with pytest.raises(ValueError, match="identity changed"):
        module.build_disposition(source, intent, tree.output, checkout=tree.checkout)
    after = source.path.stat()
    assert (after.st_ino, after.st_size, after.st_mtime_ns) == (
        before.st_ino,
        before.st_size,
        before.st_mtime_ns,
    )
    assert after.st_ctime_ns != before.st_ctime_ns
    assert not tree.output.exists()


def test_input_replaced_between_lstat_and_open_is_refused(
    tree: Tree, monkeypatch: pytest.MonkeyPatch
) -> None:
    source = tree.inventory(_inventory(_row()))
    intent = tree.bind_policy(tree.policy())
    replacement = tree.work / "replacement.json"
    replacement.write_bytes(source.path.read_bytes())
    original = os.open

    def replace_on_open(
        path: str | bytes | os.PathLike[str] | os.PathLike[bytes],
        flags: int,
        mode: int = 0o777,
        *,
        dir_fd: int | None = None,
    ) -> int:
        if Path(os.fsdecode(path)) == source.path:
            replacement.replace(source.path)
        return original(path, flags, mode, dir_fd=dir_fd)

    monkeypatch.setattr(os, "open", replace_on_open)
    with pytest.raises(ValueError, match="identity changed before reading"):
        module.build_disposition(source, intent, tree.output, checkout=tree.checkout)
    assert not tree.output.exists()


def test_postwrite_input_mutation_is_refused_and_partial_output_preserved(
    tree: Tree, monkeypatch: pytest.MonkeyPatch
) -> None:
    source = tree.inventory(_inventory(_row()))
    intent = tree.bind_policy(tree.policy())
    original = os.fsync
    changed = False

    def mutate_on_fsync(descriptor: int) -> None:
        nonlocal changed
        original(descriptor)
        if not changed:
            changed = True
            source.path.write_bytes(b"changed metadata")

    monkeypatch.setattr(os, "fsync", mutate_on_fsync)
    with pytest.raises(ValueError, match="SHA-256"):
        module.build_disposition(source, intent, tree.output, checkout=tree.checkout)
    assert tree.output.is_file()
    assert _document(tree.output)["leavesRemoved"] == 0


def test_changed_output_parent_refuses_before_creating_any_file(
    tree: Tree, monkeypatch: pytest.MonkeyPatch
) -> None:
    source = tree.inventory(_inventory(_row()))
    intent = tree.bind_policy(tree.policy())
    original = os.open

    def replace_parent(
        path: str | bytes | os.PathLike[str] | os.PathLike[bytes],
        flags: int,
        mode: int = 0o777,
        *,
        dir_fd: int | None = None,
    ) -> int:
        if Path(os.fsdecode(path)) == tree.work and flags & os.O_DIRECTORY:
            tree.work.rename(tree.checkout / "old-work")
            tree.work.mkdir()
        return original(path, flags, mode, dir_fd=dir_fd)

    monkeypatch.setattr(os, "open", replace_parent)
    with pytest.raises(ValueError, match="directory identity"):
        module.build_disposition(source, intent, tree.output, checkout=tree.checkout)
    assert not tree.output.exists()


def test_noncanonical_checkout_alias_is_rejected(tree: Tree) -> None:
    alias = tree.work / "checkout-alias"
    alias.symlink_to(tree.checkout, target_is_directory=True)
    with pytest.raises(ValueError, match="canonical"):
        module.policy_document(module.Policy(owner="owner"), checkout=alias)


def test_cli_reports_totals_and_error_without_promoting_unapproved_leaves(
    tree: Tree, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    source = tree.inventory(_inventory(_row()))
    intent = tree.bind_policy(module.Policy(owner="owner"))
    monkeypatch.setattr(module, "CHECKOUT", tree.checkout)
    args = [
        str(source.path),
        str(intent.path),
        str(tree.output),
        "--inventory-sha256",
        source.sha256,
        "--policy-sha256",
        intent.sha256,
    ]
    assert module.main(args) == 0
    assert '"hold": 1' in capsys.readouterr().out
    assert module.main(args) == 1
    assert "Leaf disposition failed" in capsys.readouterr().err


@pytest.mark.parametrize("kind", ["metadata-only-leaf-inventory", "owner-policy-leaf-disposition"])
def test_domain_reference_accepts_metadata_above_journal_budget_without_widening_it(
    tree: Tree, kind: str
) -> None:
    payload = _inventory(kind=kind, padding="x" * (9 * 1024 * 1024))
    original = tree.inventory(payload)
    bound = module.metadata_reference(original.path, checkout=tree.checkout)
    assert bound == original
    assert bound.path.stat().st_size > 8 * 1024 * 1024
    module.verify_reference(bound, checkout=tree.checkout)


@pytest.mark.parametrize(
    "document",
    [
        _inventory(kind="arbitrary-large-metadata"),
        _inventory(kind={"unknown": True}),
        _inventory(kind=[]),
        _inventory(version=True),
        _inventory(mediaContentsRead=True),
        _inventory(rows=None),
    ],
)
def test_domain_reference_refuses_unknown_metadata_contract(tree: Tree, document: object) -> None:
    reference = tree.inventory(document)
    with pytest.raises(ValueError):
        module.metadata_reference(reference.path, checkout=tree.checkout)
    with pytest.raises(ValueError):
        module.verify_reference(reference, checkout=tree.checkout)


def test_domain_reference_revalidates_bytes_and_refuses_links(tree: Tree) -> None:
    original = tree.inventory(_inventory())
    bound = module.metadata_reference(original.path, checkout=tree.checkout)
    original.path.write_bytes(b"changed")
    with pytest.raises(ValueError, match="SHA-256"):
        module.verify_reference(bound, checkout=tree.checkout)
    alias = tree.work / "alias.json"
    alias.symlink_to(original.path)
    with pytest.raises(ValueError, match="regular"):
        module.metadata_reference(alias, checkout=tree.checkout)


def test_domain_reference_retains_the_fixed_metadata_ceiling(
    tree: Tree, monkeypatch: pytest.MonkeyPatch
) -> None:
    original = tree.inventory(_inventory(padding="x" * 4096))
    monkeypatch.setattr(module, "MAX_JSON_BYTES", 1024)
    with pytest.raises(ValueError, match="bounded regular"):
        module.metadata_reference(original.path, checkout=tree.checkout)


def test_domain_reference_refuses_growth_between_identity_check_and_read(
    tree: Tree, monkeypatch: pytest.MonkeyPatch
) -> None:
    original = tree.inventory(_inventory())
    fstat = os.fstat
    grew = False
    monkeypatch.setattr(module, "MAX_JSON_BYTES", 1024)

    def grow_on_first_fstat(descriptor: int) -> os.stat_result:
        nonlocal grew
        info = fstat(descriptor)
        if not grew:
            grew = True
            with original.path.open("ab") as stream:
                stream.write(b" " * 2048)
        return info

    monkeypatch.setattr(os, "fstat", grow_on_first_fstat)
    with pytest.raises(ValueError, match="exceeds its byte budget"):
        module.metadata_reference(original.path, checkout=tree.checkout)


def test_domain_reference_refuses_same_bytes_replacement_during_read(
    tree: Tree, monkeypatch: pytest.MonkeyPatch
) -> None:
    original = tree.inventory(_inventory())
    replacement = tree.work / "replacement.json"
    replacement.write_bytes(original.path.read_bytes())
    fstat = os.fstat
    calls = 0

    def replace_on_terminal_fstat(descriptor: int) -> os.stat_result:
        nonlocal calls
        calls += 1
        if calls == 2:
            replacement.replace(original.path)
        return fstat(descriptor)

    monkeypatch.setattr(os, "fstat", replace_on_terminal_fstat)
    with pytest.raises(ValueError, match="changed during reading"):
        module.metadata_reference(original.path, checkout=tree.checkout)


def test_domain_reference_rejects_duplicate_json_fields(tree: Tree) -> None:
    original = tree.reference(
        tree.inventory_path,
        b'{"version":1,"kind":"metadata-only-leaf-inventory","mediaContentsRead":false,"rows":[],"rows":[]}',
    )
    with pytest.raises(ValueError, match="Duplicate JSON"):
        module.metadata_reference(original.path, checkout=tree.checkout)
    with pytest.raises(ValueError, match="Duplicate JSON"):
        module.verify_reference(original, checkout=tree.checkout)
