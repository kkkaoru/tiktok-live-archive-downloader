import fcntl
import json
import os
import sys
from pathlib import Path

import pytest

from replay import original_storage as storage
from replay.core import ReplayError
from replay.jobs import already_complete, download_once, file_sha256, write_record


@pytest.fixture
def paths(tmp_path: Path) -> tuple[Path, Path, Path, Path]:
    root = tmp_path / "jobs"
    root.mkdir()
    source = tmp_path / "original.mp4"
    source.write_bytes(b"synthetic original")
    directory = tmp_path / "per-video"
    directory.mkdir()
    write_record(root / "123.json", source, "complete")
    return root, source, directory / "original.mp4", tmp_path / "intent.json"


def test_same_inode_and_completion_keep_old_reference(paths: tuple[Path, Path, Path, Path]) -> None:
    root, source, destination, receipt = paths
    digest = file_sha256(source)
    plan = storage.inspect(
        root=root, replay_id="123", source=source, destination=destination, expected_sha256=digest
    )
    assert not destination.exists()
    assert not receipt.exists()
    result = storage.organize(
        root=root,
        replay_id="123",
        source=source,
        destination=destination,
        expected_sha256=digest,
        receipt=receipt,
    )
    assert result["additionalMediaBytes"] == 0
    assert result["oldAliasDeleted"] is False
    assert result["mapping"] == plan
    assert os.path.samefile(source, destination)
    assert source.read_bytes() == b"synthetic original"
    assert already_complete(root / "123.json")
    assert (
        download_once(
            root=root,
            replay_id="123",
            output=source,
            download=lambda: pytest.fail("must not redownload an organized original"),
        )
        is False
    )
    assert json.loads((root / "123.json").read_text())["output"] == str(destination)
    assert (
        storage.organize(
            root=root,
            replay_id="123",
            source=source,
            destination=destination,
            expected_sha256=digest,
            receipt=receipt,
        )
        == result
    )


@pytest.mark.parametrize("phase", ["link", "record", "after-record"])
def test_interruption_reconciles_exact_intent(
    paths: tuple[Path, Path, Path, Path], monkeypatch: pytest.MonkeyPatch, phase: str
) -> None:
    root, source, destination, receipt = paths
    digest = file_sha256(source)
    original_link, original_write = os.link, write_record

    def interrupted_link(src: Path, dst: Path, *, follow_symlinks: bool) -> None:
        raise OSError("synthetic link interruption")

    def interrupted_record(path: Path, output: Path, status: str) -> None:
        if phase == "after-record":
            original_write(path, output, status)
        raise OSError("synthetic record interruption")

    if phase == "link":
        monkeypatch.setattr(os, "link", interrupted_link)
    else:
        monkeypatch.setattr(storage, "write_record", interrupted_record)
    with pytest.raises(OSError):
        storage.organize(
            root=root,
            replay_id="123",
            source=source,
            destination=destination,
            expected_sha256=digest,
            receipt=receipt,
        )
    assert source.exists()
    assert already_complete(root / "123.json")
    assert receipt.exists()
    monkeypatch.setattr(os, "link", original_link)
    monkeypatch.setattr(storage, "write_record", original_write)
    storage.organize(
        root=root,
        replay_id="123",
        source=source,
        destination=destination,
        expected_sha256=digest,
        receipt=receipt,
    )
    assert os.path.samefile(source, destination)


@pytest.mark.parametrize(
    "change",
    [
        "id",
        "digest",
        "content",
        "target",
        "symlink",
        "parent-symlink",
        "empty",
        "job",
        "same",
        "receipt",
    ],
)
def test_reject_invalid_paths_and_identity(
    paths: tuple[Path, Path, Path, Path], change: str
) -> None:
    root, source, destination, receipt = paths
    digest = file_sha256(source)
    replay_id = "123"
    if change == "id":
        replay_id = "../123"
    elif change == "digest":
        digest = "invalid"
    elif change == "content":
        digest = "0" * 64
    elif change == "target":
        destination.write_bytes(b"different")
    elif change == "symlink":
        destination.symlink_to(source)
    elif change == "parent-symlink":
        alias = root.parent / "alias"
        alias.symlink_to(destination.parent, target_is_directory=True)
        destination = alias / destination.name
    elif change == "empty":
        source.write_bytes(b"")
    elif change == "job":
        (root / "123.json").write_text('{"status":"failed"}')
    elif change == "same":
        destination = source
    elif change == "receipt":
        receipt = destination
    with pytest.raises(ReplayError):
        storage.organize(
            root=root,
            replay_id=replay_id,
            source=source,
            destination=destination,
            expected_sha256=digest,
            receipt=receipt,
        )


def test_no_unregistered_alias_adoption(paths: tuple[Path, Path, Path, Path]) -> None:
    root, source, destination, receipt = paths
    os.link(source, destination)
    with pytest.raises(ReplayError, match="without this intent"):
        storage.organize(
            root=root,
            replay_id="123",
            source=source,
            destination=destination,
            expected_sha256=file_sha256(source),
            receipt=receipt,
        )


def test_cli_preflight(
    paths: tuple[Path, Path, Path, Path], monkeypatch: pytest.MonkeyPatch
) -> None:
    root, source, destination, receipt = paths
    monkeypatch.setattr(
        sys,
        "argv",
        ["storage", "check", str(root), "123", str(source), str(destination), file_sha256(source)],
    )
    storage.main()
    assert not receipt.exists()
    assert not destination.exists()


def test_cross_device_refused(
    paths: tuple[Path, Path, Path, Path], monkeypatch: pytest.MonkeyPatch
) -> None:
    root, source, destination, receipt = paths
    identity = storage._identity

    def other_device(path: Path, *, method: str = "hardlink") -> dict[str, int]:
        result = identity(path, method=method)
        result["device"] = -1
        return result

    monkeypatch.setattr(storage, "_identity", other_device)
    with pytest.raises(ReplayError, match="across devices"):
        storage.organize(
            root=root,
            replay_id="123",
            source=source,
            destination=destination,
            expected_sha256=file_sha256(source),
            receipt=receipt,
        )


def test_busy_id_lock(paths: tuple[Path, Path, Path, Path]) -> None:
    root, source, destination, receipt = paths
    with (root / "123.lock").open("a+b") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        with pytest.raises(BlockingIOError):
            storage.organize(
                root=root,
                replay_id="123",
                source=source,
                destination=destination,
                expected_sha256=file_sha256(source),
                receipt=receipt,
            )
    assert not destination.exists()


def test_link_publication_detects_source_mutation(
    paths: tuple[Path, Path, Path, Path], monkeypatch: pytest.MonkeyPatch
) -> None:
    root, source, destination, receipt = paths
    original_link = os.link

    def changing_link(src: Path, dst: Path, *, follow_symlinks: bool) -> None:
        original_link(src, dst, follow_symlinks=follow_symlinks)
        source.write_bytes(b"different original")

    monkeypatch.setattr(os, "link", changing_link)
    with pytest.raises(ReplayError, match="changed"):
        storage.organize(
            root=root,
            replay_id="123",
            source=source,
            destination=destination,
            expected_sha256=file_sha256(source),
            receipt=receipt,
        )
    assert json.loads((root / "123.json").read_text())["output"] == str(source)


@pytest.mark.parametrize("change", ["intent", "job", "lock"])
def test_reconciliation_rejects_conflicting_records(
    paths: tuple[Path, Path, Path, Path], change: str
) -> None:
    root, source, destination, receipt = paths
    digest = file_sha256(source)
    storage.organize(
        root=root,
        replay_id="123",
        source=source,
        destination=destination,
        expected_sha256=digest,
        receipt=receipt,
    )
    if change == "intent":
        receipt.write_text('{"mapping":{}}')
    elif change == "job":
        write_record(root / "123.json", source, "complete")
        raw = json.loads((root / "123.json").read_text())
        raw["unexpected"] = "outside transaction"
        (root / "123.json").write_text(json.dumps(raw))
    else:
        os.link(root / "123.lock", root / "lock-alias")
    with pytest.raises(ReplayError):
        storage.organize(
            root=root,
            replay_id="123",
            source=source,
            destination=destination,
            expected_sha256=digest,
            receipt=receipt,
        )


@pytest.mark.parametrize("receipt_option", [False, True])
def test_cli_apply(
    paths: tuple[Path, Path, Path, Path], monkeypatch: pytest.MonkeyPatch, receipt_option: bool
) -> None:
    root, source, destination, receipt = paths
    argv = [
        "storage",
        "apply",
        str(root),
        "123",
        str(source),
        str(destination),
        file_sha256(source),
    ]
    if receipt_option:
        argv.extend(["--receipt", str(receipt)])
    monkeypatch.setattr(sys, "argv", argv)
    if receipt_option:
        storage.main()
        assert os.path.samefile(source, destination)
    else:
        with pytest.raises(SystemExit):
            storage.main()


def test_bounded_record(paths: tuple[Path, Path, Path, Path]) -> None:
    root, source, destination, receipt = paths
    (root / "123.json").write_bytes(b" " * 65537)
    with pytest.raises(ReplayError, match="bounded"):
        storage.inspect(
            root=root,
            replay_id="123",
            source=source,
            destination=destination,
            expected_sha256=file_sha256(source),
        )
