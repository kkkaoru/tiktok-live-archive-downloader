from __future__ import annotations

import hashlib
import sys
from collections.abc import Callable
from pathlib import Path
from types import ModuleType

import pytest

from replay import digest_store, digest_store_run, stages
from replay.core import ReplayError


class DigestHolder(ModuleType):
    def __init__(self, name: str, digest: Callable[[Path], str]) -> None:
        super().__init__(name)
        self.file_digest = digest


class CLIRecorder(ModuleType):
    def __init__(self, name: str) -> None:
        super().__init__(name)
        self.calls: list[list[str]] = []

    def main(self) -> int:
        self.calls.append(sys.argv[:])
        return 0


@pytest.fixture(autouse=True)
def small_threshold(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(digest_store, "MINIMUM_BYTES", 4)


def test_persistent_digest_survives_a_second_install(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    media = tmp_path / "media.bin"
    media.write_bytes(b"large-enough")
    store = tmp_path / "store"
    expected = hashlib.sha256(b"large-enough").hexdigest()
    first = digest_store_run.persistent(stages.file_digest, store)
    assert first(media) == expected
    calls: list[Path] = []

    def never(path: Path) -> str:  # a fresh process must not hash again
        calls.append(path)
        return "wrong"

    original = digest_store.hash_file

    def measured(path: Path) -> str:
        calls.append(path)
        return original(path)

    monkeypatch.setattr(digest_store, "hash_file", measured)
    second = digest_store_run.persistent(never, store)
    assert second(media) == expected
    assert calls == []


def test_small_files_use_the_original_function(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(digest_store, "MINIMUM_BYTES", 1 << 30)
    media = tmp_path / "small.bin"
    media.write_bytes(b"x")
    wrapped = digest_store_run.persistent(lambda p: "original", tmp_path / "unused")
    assert wrapped(media) == "original"
    assert not (tmp_path / "unused").exists()


def test_install_patches_references_and_main_runs_the_module(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    def sentinel(_path: Path) -> str:
        return "original"

    # The unique sentinel prevents install from changing unrelated module references.
    monkeypatch.setattr(stages, "file_digest", sentinel)
    holder = DigestHolder("replay.fake_holder", sentinel)
    monkeypatch.setitem(sys.modules, "replay.fake_holder", holder)
    assert "replay.fake_holder" in digest_store_run.install(sentinel, tmp_path / "s")
    assert holder.file_digest is not sentinel

    fake = CLIRecorder("replay.caption_measured_job")
    monkeypatch.setitem(sys.modules, "replay.caption_measured_job", fake)
    monkeypatch.setattr(sys, "argv", sys.argv[:])
    assert (
        digest_store_run.main([str(tmp_path / "s"), "replay.caption_measured_job", "a", "b"]) == 0
    )
    assert fake.calls == [["replay.caption_measured_job", "a", "b"]]


def test_persistent_rehashes_immediate_same_size_mutation(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    source = tmp_path / "source"
    source.write_bytes(b"old-bytes")
    calls: list[Path] = []
    original = digest_store.hash_file

    def measured(path: Path) -> str:
        calls.append(path)
        return original(path)

    monkeypatch.setattr(digest_store, "hash_file", measured)
    wrapped = digest_store_run.persistent(stages.file_digest, tmp_path / "store")
    assert wrapped(source) == hashlib.sha256(b"old-bytes").hexdigest()
    source.write_bytes(b"new-bytes")
    assert wrapped(source) == hashlib.sha256(b"new-bytes").hexdigest()
    assert calls == [source, source]


def test_mutation_during_persistent_hashing_is_rejected(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    source = tmp_path / "source"
    source.write_bytes(b"original")
    original = digest_store.hash_file

    def racing(path: Path) -> str:
        value = original(path)
        path.write_bytes(b"mutated-after-read")
        return value

    monkeypatch.setattr(digest_store, "hash_file", racing)
    store = tmp_path / "store"
    wrapped = digest_store_run.persistent(stages.file_digest, store)
    with pytest.raises(ReplayError, match="changed before cache publication"):
        wrapped(source)
    assert not list(store.glob("*.json"))


def test_independent_fresh_boundary_still_reads_actual_bytes_after_install(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    source = tmp_path / "source"
    source.write_bytes(b"fresh-boundary")
    store = tmp_path / "store"

    def sentinel(path: Path) -> str:
        return digest_store.hash_file(path)

    monkeypatch.setattr(stages, "file_digest", sentinel)
    assert digest_store_run.install(sentinel, store) == ["replay.stages"]
    assert stages.file_digest(source) == hashlib.sha256(b"fresh-boundary").hexdigest()
    calls: list[Path] = []
    original = digest_store.hash_file

    def measured(path: Path) -> str:
        calls.append(path)
        return original(path)

    monkeypatch.setattr(digest_store, "hash_file", measured)
    assert stages.file_digest(source) == hashlib.sha256(b"fresh-boundary").hexdigest()
    assert calls == []
    assert (
        digest_store.cached_digest(store, source, fresh=True)
        == hashlib.sha256(b"fresh-boundary").hexdigest()
    )
    assert calls == [source]


def test_usage_error(capsys: pytest.CaptureFixture[str]) -> None:
    assert digest_store_run.main(["only"]) == 2
    assert digest_store_run.main(["store", "os"]) == 2
    assert "usage" in capsys.readouterr().err


def test_default_store_is_inside_the_checkout_work_area() -> None:
    store = digest_store_run.default_store()
    assert store.parts[-3:] == ("edits", "work", "digest-store")
    assert store.is_absolute()
