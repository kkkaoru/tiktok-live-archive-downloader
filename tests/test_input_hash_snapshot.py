import hashlib
import os
from pathlib import Path
from unittest.mock import Mock

import pytest

from replay import input_hash_snapshot
from replay.input_hash_snapshot import InputHashSnapshot
from replay.stages import StageError, file_digest


def test_duplicate_inputs_hash_only_once_at_each_endpoint(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    path = tmp_path / "input"
    path.write_bytes(b"abc")
    digest = hashlib.sha256(b"abc").hexdigest()
    read = Mock(wraps=file_digest)
    monkeypatch.setattr(input_hash_snapshot, "file_digest", read)
    snapshot = InputHashSnapshot.capture([(path, digest), (path, digest)])
    assert read.call_count == 1
    assert snapshot.checked_digest(path, digest) == (
        "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad"
    )
    snapshot.checked_digest(path, digest)
    assert read.call_count == 1
    assert list(snapshot.finish().values()) == [
        "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad"
    ]
    assert read.call_count == 2
    with pytest.raises(StageError, match="closed"):
        snapshot.finish()
    with pytest.raises(StageError, match="closed"):
        snapshot.checked_digest(path, digest)


@pytest.mark.parametrize("fault", ["empty", "pin", "conflict", "limit"])
def test_invalid_bindings_fail_before_reading(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, fault: str
) -> None:
    path = tmp_path / "missing"
    digest = "a" * 64
    bindings = [(path, digest)]
    if fault == "empty":
        bindings = []
    elif fault == "pin":
        bindings = [(path, "invalid")]
    elif fault == "conflict":
        bindings.append((path, "b" * 64))
    else:
        monkeypatch.setattr(input_hash_snapshot, "MAX_INPUTS", 1)
        bindings.append((tmp_path / "other", digest))
    with pytest.raises(StageError):
        InputHashSnapshot.capture(bindings)


@pytest.mark.parametrize("fault", ["empty", "directory", "link", "parent_link", "relative"])
def test_only_canonical_nonempty_files_are_accepted(tmp_path: Path, fault: str) -> None:
    source = tmp_path / "source"
    source.write_bytes(b"abc")
    path = tmp_path / "invalid"
    if fault == "empty":
        path.touch()
    elif fault == "directory":
        path.mkdir()
    elif fault == "link":
        path.symlink_to(source)
    elif fault == "parent_link":
        path.symlink_to(tmp_path, target_is_directory=True)
        path /= "source"
    else:
        path = Path("relative")
    with pytest.raises(StageError):
        InputHashSnapshot.capture([(path, hashlib.sha256(b"abc").hexdigest())])


def test_expected_content_is_checked_at_capture(tmp_path: Path) -> None:
    path = tmp_path / "input"
    path.write_bytes(b"abc")
    with pytest.raises(StageError, match="SHA256"):
        InputHashSnapshot.capture([(path, "a" * 64)])


@pytest.mark.parametrize("fault", ["replace", "size", "mtime", "mode", "delete"])
def test_identity_changes_poison_the_scope(tmp_path: Path, fault: str) -> None:
    path = tmp_path / "input"
    path.write_bytes(b"abc")
    digest = hashlib.sha256(b"abc").hexdigest()
    snapshot = InputHashSnapshot.capture([(path, digest)])
    if fault == "replace":
        replacement = tmp_path / "replacement"
        replacement.write_bytes(b"abc")
        replacement.replace(path)
    elif fault == "size":
        path.write_bytes(b"abcd")
    elif fault == "mtime":
        before = path.stat()
        os.utime(path, ns=(before.st_atime_ns, before.st_mtime_ns + 1_000_000))
    elif fault == "mode":
        path.chmod(0o400)
    else:
        path.unlink()
    with pytest.raises((StageError, FileNotFoundError)):
        snapshot.checked_digest(path, digest)
    with pytest.raises(StageError, match="closed"):
        snapshot.finish()


@pytest.mark.parametrize("unknown", [False, True])
def test_new_path_or_conflicting_lookup_pin_poison_scope(tmp_path: Path, unknown: bool) -> None:
    path = tmp_path / "input"
    path.write_bytes(b"abc")
    snapshot = InputHashSnapshot.capture([(path, hashlib.sha256(b"abc").hexdigest())])
    with pytest.raises(StageError, match="Unknown"):
        snapshot.checked_digest(tmp_path / "other" if unknown else path, "0" * 64)
    with pytest.raises(StageError, match="closed"):
        snapshot.finish()


def test_finish_rehashes_even_if_metadata_appears_unchanged(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    path = tmp_path / "input"
    path.write_bytes(b"abc")
    snapshot = InputHashSnapshot.capture([(path, hashlib.sha256(b"abc").hexdigest())])
    identity = input_hash_snapshot._identity(path)
    path.write_bytes(b"xyz")
    # Simulate a metadata observation that misses the change; final bytes must
    # still be checked, rather than treating unchanged stat as content proof.
    monkeypatch.setattr(input_hash_snapshot, "_identity", Mock(return_value=identity))
    with pytest.raises(StageError, match="SHA256"):
        snapshot.finish()
    with pytest.raises(StageError, match="closed"):
        snapshot.finish()


def test_change_during_hash_fails(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    path = tmp_path / "input"
    path.write_bytes(b"abc")

    def changed_read(value: Path) -> str:
        digest = file_digest(value)
        value.write_bytes(b"replacement")
        return digest

    monkeypatch.setattr(input_hash_snapshot, "file_digest", changed_read)
    with pytest.raises(StageError, match="identity"):
        InputHashSnapshot.capture([(path, hashlib.sha256(b"abc").hexdigest())])


@pytest.mark.parametrize("during_finish", [False, True])
def test_earlier_file_change_while_hashing_later_file_is_detected(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, during_finish: bool
) -> None:
    first, second = tmp_path / "first", tmp_path / "second"
    first.write_bytes(b"abc")
    second.write_bytes(b"abc")
    bindings = [
        (first, hashlib.sha256(b"abc").hexdigest()),
        (second, hashlib.sha256(b"abc").hexdigest()),
    ]
    snapshot = InputHashSnapshot.capture(bindings) if during_finish else None

    def changed_read(value: Path) -> str:
        digest = file_digest(value)
        if value == second:
            first.write_bytes(b"changed")
        return digest

    monkeypatch.setattr(input_hash_snapshot, "file_digest", changed_read)
    with pytest.raises(StageError, match="identity"):
        if snapshot is None:
            InputHashSnapshot.capture(bindings)
        else:
            snapshot.finish()
