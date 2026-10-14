from __future__ import annotations

import hashlib
import json
import os
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import IO

import pytest

from replay import digest_store
from replay.core import ReplayError
from replay.stages import execution_lock


@pytest.fixture(autouse=True)
def small_threshold(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(digest_store, "MINIMUM_BYTES", 4)


def test_reuses_digest_until_identity_changes(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    media = tmp_path / "media.bin"
    media.write_bytes(b"first-bytes")
    store = tmp_path / "store"
    calls: list[Path] = []
    original = digest_store.hash_file

    def measured(path: Path) -> str:
        calls.append(path)
        return original(path)

    monkeypatch.setattr(digest_store, "hash_file", measured)
    first = digest_store.cached_digest(store, media)
    assert first == hashlib.sha256(b"first-bytes").hexdigest()
    assert digest_store.cached_digest(store, media) == first
    assert len(calls) == 1
    media.write_bytes(b"other-bytes")
    assert digest_store.cached_digest(store, media) == hashlib.sha256(b"other-bytes").hexdigest()
    assert len(calls) == 2
    os.utime(media, ns=(1, 1))
    digest_store.cached_digest(store, media)
    assert len(calls) == 3
    assert oct(next(store.glob("*.json")).stat().st_mode & 0o777) == "0o600"


def test_small_files_and_corrupt_entries_are_rehashed(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    media = tmp_path / "media.bin"
    media.write_bytes(b"123456789")
    store = tmp_path / "store"
    value = digest_store.cached_digest(store, media)
    next(store.glob("*.json")).write_text("not json", encoding="utf-8")
    assert digest_store.cached_digest(store, media) == value
    monkeypatch.setattr(digest_store, "MINIMUM_BYTES", 1 << 30)
    assert digest_store.cached_digest(tmp_path / "unused", media) == value
    assert not (tmp_path / "unused").exists()


def test_identity_change_during_hashing_is_not_cached(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    media = tmp_path / "media.bin"
    media.write_bytes(b"abcdefgh")
    original = digest_store.hash_file

    def racing(path: Path) -> str:
        value = original(path)
        path.write_bytes(b"changed-while-hashing")
        return value

    monkeypatch.setattr(digest_store, "hash_file", racing)
    with pytest.raises(ReplayError, match="changed before cache publication"):
        digest_store.cached_digest(tmp_path / "store", media)
    assert not list((tmp_path / "store").glob("*.json"))


def test_cli(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    media = tmp_path / "media.bin"
    media.write_bytes(b"cli-bytes")
    assert digest_store.main([str(tmp_path / "s"), str(media)]) == 0
    assert capsys.readouterr().out.startswith(hashlib.sha256(b"cli-bytes").hexdigest())
    assert digest_store.main([str(tmp_path / "s"), str(tmp_path / "missing")]) == 1
    assert digest_store.main(["only"]) == 2


def test_valid_hex_digest_tampering_is_rehashed_not_trusted(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    source = tmp_path / "source"
    source.write_bytes(b"authenticated source")
    store = tmp_path / "store"
    expected = digest_store.cached_digest(store, source)
    entry = next(store.glob("*.json"))
    saved = json.loads(entry.read_text(encoding="utf-8"))
    saved["sha256"] = "0" * 64
    entry.write_text(json.dumps(saved), encoding="utf-8")
    calls: list[Path] = []
    original = digest_store.hash_file

    def measured(path: Path) -> str:
        calls.append(path)
        return original(path)

    monkeypatch.setattr(digest_store, "hash_file", measured)
    assert digest_store.cached_digest(store, source) == expected
    assert len(calls) == 1


def test_fresh_always_reads_bytes_even_when_cache_matches(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    source = tmp_path / "source"
    source.write_bytes(b"fresh bytes")
    store = tmp_path / "store"
    digest_store.cached_digest(store, source)
    calls: list[Path] = []
    original = digest_store.hash_file

    def measured(path: Path) -> str:
        calls.append(path)
        return original(path)

    monkeypatch.setattr(digest_store, "hash_file", measured)
    assert (
        digest_store.cached_digest(store, source, fresh=True)
        == hashlib.sha256(b"fresh bytes").hexdigest()
    )
    assert len(calls) == 1


def test_fresh_boundary_does_not_read_or_repair_a_broken_cache(tmp_path: Path) -> None:
    source = tmp_path / "source"
    source.write_bytes(b"fresh source")
    store = tmp_path / "store"
    digest_store.cached_digest(store, source)
    key = store / "store.key"
    key.write_bytes(b"retained incomplete key")
    expected = hashlib.sha256(b"fresh source").hexdigest()
    assert digest_store.cached_digest(store, source, fresh=True) == expected
    assert key.read_bytes() == b"retained incomplete key"


def test_cli_fresh_has_no_cache_side_effects(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    source = tmp_path / "source"
    source.write_bytes(b"fresh cli source")
    store = tmp_path / "unused-store"
    assert digest_store.main(["--fresh", str(store), str(source)]) == 0
    assert capsys.readouterr().out.startswith(hashlib.sha256(b"fresh cli source").hexdigest())
    assert not store.exists()


@pytest.mark.parametrize("source_kind", ["link", "parent-link", "directory"])
def test_source_links_and_nonregular_files_are_rejected(tmp_path: Path, source_kind: str) -> None:
    source = tmp_path / "source"
    if source_kind == "link":
        original = tmp_path / "original"
        original.write_bytes(b"original")
        source.symlink_to(original)
    elif source_kind == "parent-link":
        original_dir = tmp_path / "original-dir"
        original_dir.mkdir()
        (original_dir / "source").write_bytes(b"original")
        source.symlink_to(original_dir, target_is_directory=True)
        source = source / "source"
    else:
        source.mkdir()
    with pytest.raises(ReplayError, match="canonical|regular"):
        digest_store.hash_file(source)


def test_source_replacement_before_open_is_rejected(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    source = tmp_path / "source"
    source.write_bytes(b"before")
    original = os.open

    def replaced(path: Path, flags: int, *args: object, **kwargs: object) -> int:
        source.rename(tmp_path / "retained")
        source.write_bytes(b"replacement")
        return original(path, flags)

    monkeypatch.setattr(os, "open", replaced)
    with pytest.raises(ReplayError, match="changed before the byte read"):
        digest_store.hash_file(source)
    assert (tmp_path / "retained").read_bytes() == b"before"


def test_mutation_during_actual_byte_read_is_rejected(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    source = tmp_path / "source"
    source.write_bytes(b"before")
    original = os.fdopen

    def fdopen(descriptor: int, _mode: str) -> IO[bytes]:
        stream = original(descriptor, "rb")
        read = stream.read

        def mutate(size: int) -> bytes:
            data = read(size)
            if data:
                source.write_bytes(b"replacement")
            return data

        monkeypatch.setattr(stream, "read", mutate)
        return stream

    monkeypatch.setattr(os, "fdopen", fdopen)
    with pytest.raises(ReplayError, match="changed during the byte read"):
        digest_store.hash_file(source)


def test_source_changes_during_cache_hit_are_rejected(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    source = tmp_path / "source"
    source.write_bytes(b"before")
    store = tmp_path / "store"
    digest_store.cached_digest(store, source)
    original = digest_store._load

    def changed(
        entry: Path, *, key: bytes, path: Path, before: digest_store.Identity
    ) -> str | None:
        result = original(entry, key=key, path=path, before=before)
        source.write_bytes(b"after")
        return result

    monkeypatch.setattr(digest_store, "_load", changed)
    with pytest.raises(ReplayError, match="changed during cache lookup"):
        digest_store.cached_digest(store, source)


@pytest.mark.parametrize("kind", ["link", "hardlink", "mode", "oversize"])
def test_unsafe_cache_entry_is_not_read(tmp_path: Path, kind: str) -> None:
    source = tmp_path / "source"
    source.write_bytes(b"cache bytes")
    store = tmp_path / "store"
    digest_store.cached_digest(store, source)
    entry = next(store.glob("*.json"))
    retained = store / "retained"
    if kind == "link":
        entry.rename(retained)
        entry.symlink_to(retained)
    elif kind == "hardlink":
        os.link(entry, retained)
    elif kind == "mode":
        entry.chmod(0o644)
    else:
        entry.write_bytes(b"x" * 8193)
    with pytest.raises(ReplayError, match="private|budget"):
        digest_store.cached_digest(store, source)


@pytest.mark.parametrize("unsafe", ["outside", "link", "mode"])
def test_store_scope_and_permissions_are_enforced(tmp_path: Path, unsafe: str) -> None:
    source = tmp_path / "source"
    source.write_bytes(b"cache bytes")
    store = tmp_path / "store"
    if unsafe == "outside":
        store = Path("/tmp/forbidden-digest-store")
    elif unsafe == "link":
        target = tmp_path / "target"
        target.mkdir(mode=0o700)
        store.symlink_to(target, target_is_directory=True)
    else:
        store.mkdir(mode=0o755)
        store.chmod(0o755)
    with pytest.raises(ReplayError, match="canonical|private"):
        digest_store.cached_digest(store, source)


def test_concurrent_same_input_has_one_verified_hash(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    source = tmp_path / "source"
    source.write_bytes(b"concurrent cache bytes")
    store = tmp_path / "store"
    calls: list[Path] = []
    original = digest_store.hash_file

    def slow(path: Path) -> str:
        calls.append(path)
        time.sleep(0.01)
        return original(path)

    def digest(_worker: int) -> str:
        return digest_store.cached_digest(store, source)

    monkeypatch.setattr(digest_store, "hash_file", slow)
    with ThreadPoolExecutor(max_workers=4) as pool:
        values = list(pool.map(digest, [1, 2, 3, 4]))
    assert len(set(values)) == 1
    assert len(calls) == 1
    assert len(list(store.glob("*.json"))) == 1
    assert list(store.glob("*.pending")) == []


def test_old_fixed_pending_link_is_never_used(tmp_path: Path) -> None:
    source = tmp_path / "source"
    source.write_bytes(b"cache bytes")
    store = tmp_path / "store"
    store.mkdir(mode=0o700)
    original = tmp_path / "original"
    original.write_bytes(b"preserved")
    pending = digest_store._entry(store, source).with_suffix(".pending")
    pending.symlink_to(original)
    digest_store.cached_digest(store, source)
    assert original.read_bytes() == b"preserved"
    assert pending.is_symlink()


@pytest.mark.parametrize("key_kind", ["short", "mode", "link"])
def test_invalid_store_key_is_preserved_and_not_silently_regenerated(
    tmp_path: Path, key_kind: str
) -> None:
    source = tmp_path / "source"
    source.write_bytes(b"cache bytes")
    store = tmp_path / "store"
    digest_store.cached_digest(store, source)
    key = store / "store.key"
    if key_kind == "short":
        key.write_bytes(b"short")
    elif key_kind == "mode":
        key.chmod(0o644)
    else:
        retained = store / "retained-key"
        key.rename(retained)
        key.symlink_to(retained)
    with pytest.raises(ReplayError, match="incomplete|private"):
        digest_store.cached_digest(store, source)


def test_failed_publication_removes_only_its_own_temporary(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    source = tmp_path / "source"
    source.write_bytes(b"cache bytes")
    store = tmp_path / "store"

    def failed(_source: Path, _target: Path) -> None:
        raise OSError("synthetic publication failure")

    monkeypatch.setattr(os, "replace", failed)
    with pytest.raises(OSError, match="publication failure"):
        digest_store.cached_digest(store, source)
    assert list(store.glob("*.pending")) == []
    assert list(store.glob("*.json")) == []
    assert (store / "store.key").stat().st_mode & 0o777 == 0o600


def test_cache_metadata_replacement_before_read_is_rejected(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    metadata = tmp_path / "metadata"
    metadata.write_bytes(b"old metadata")
    metadata.chmod(0o600)
    original = os.open

    def replaced(path: Path, flags: int) -> int:
        metadata.rename(tmp_path / "retained")
        metadata.write_bytes(b"new metadata")
        metadata.chmod(0o600)
        return original(path, flags)

    monkeypatch.setattr(os, "open", replaced)
    with pytest.raises(ReplayError, match="changed before reading"):
        digest_store._read_private(metadata, limit=100)


def test_cache_metadata_modification_during_read_is_rejected(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    metadata = tmp_path / "metadata"
    metadata.write_bytes(b"old metadata")
    metadata.chmod(0o600)
    original = os.fdopen

    def fdopen(descriptor: int, _mode: str) -> IO[bytes]:
        stream = original(descriptor, "rb")
        read = stream.read

        def changed(size: int) -> bytes:
            data = read(size)
            metadata.write_bytes(b"new metadata")
            return data

        monkeypatch.setattr(stream, "read", changed)
        return stream

    monkeypatch.setattr(os, "fdopen", fdopen)
    with pytest.raises(ReplayError, match="changed during reading"):
        digest_store._read_private(metadata, limit=100)


def test_cache_lease_timeout_is_finite(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    clock = [0.0]
    monkeypatch.setattr(time, "monotonic", lambda: clock[0])
    monkeypatch.setattr(time, "sleep", lambda _seconds: clock.__setitem__(0, clock[0] + 1))
    with (
        execution_lock(tmp_path, "busy"),
        pytest.raises(ReplayError, match="finite deadline"),
        digest_store._lease(tmp_path, "busy"),
    ):
        raise RuntimeError("must not acquire")
