import fcntl
import hashlib
import json
import os
import stat
import sys
from collections.abc import Iterable, Iterator
from contextlib import ExitStack, contextmanager
from pathlib import Path
from typing import BinaryIO, NoReturn

import pytest
import test_part_retirement_verification_readonly as fixtures

from replay import part_master_publication as publication
from replay import part_retirement_request as request
from replay import part_retirement_verification_batch as batch
from replay import part_retirement_verification_readonly as readonly
from replay.core import ReplayError
from replay.input_hash_cache import InputHashCache
from replay.recordings import record


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


@pytest.fixture
def completed(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> fixtures.Setup:
    return fixtures._complete(tmp_path, monkeypatch)


def _prepare(setup: fixtures.Setup, *, output: Path | None = None) -> Path:
    return request.prepare(
        setup.fixture.work / "manifest.json",
        _sha(setup.fixture.work / "manifest.json"),
        completed=setup.fixture.work / "result",
        output=setup.fixture.work / "generated-request.json" if output is None else output,
    )


def _forbidden(*args: object, **kwargs: object) -> NoReturn:
    raise AssertionError("Request bridge attempted a producer or media-domain check")


@pytest.mark.parametrize("count", [1, 2, 3])
@pytest.mark.parametrize("phase", ["before-delete", "after-delete"])
def test_actual_metadata_generates_request_accepted_by_unchanged_consumer(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, count: int, phase: batch.Phase
) -> None:
    setup = fixtures._complete(tmp_path, monkeypatch, count=count, phase=phase)
    root = setup.fixture.work / "result"
    before = {path: path.read_bytes() for path in root.iterdir()}
    with monkeypatch.context() as guard:
        guard.setattr(batch, "run", _forbidden)
        guard.setattr(publication, "verify_retirement", _forbidden)
        generated = _prepare(setup)
    assert generated.stat().st_mode & 0o777 == 0o600
    assert {path: path.read_bytes() for path in root.iterdir()} == before
    result = readonly.verify(readonly.load(generated))
    assert result.phase == phase
    assert result.parts == tuple(range(1, count + 1))
    assert result.json()["deletionPerformed"] is False
    assert result.json()["qualityApproved"] is False
    assert result.json()["deliveryApproved"] is False


def test_preparation_never_opens_or_hashes_media(
    completed: fixtures.Setup, monkeypatch: pytest.MonkeyPatch
) -> None:
    forbidden = {*completed.fixture.movies, *completed.fixture.sources, completed.fixture.voice}
    original_open = os.open

    def open_metadata(
        path: str | bytes | os.PathLike[str] | os.PathLike[bytes],
        flags: int,
        mode: int = 0o777,
        *,
        dir_fd: int | None = None,
    ) -> int:
        assert path not in forbidden
        return original_open(path, flags, mode, dir_fd=dir_fd)

    def metadata_cache(paths: Iterable[Path], *, allow_uncached_overflow: bool) -> InputHashCache:
        bound = tuple(paths)
        assert not forbidden.intersection(bound)
        return InputHashCache(bound, allow_uncached_overflow=allow_uncached_overflow)

    monkeypatch.setattr(os, "open", open_metadata)
    monkeypatch.setattr(request, "InputHashCache", metadata_cache)
    monkeypatch.setattr(batch, "run", _forbidden)
    monkeypatch.setattr(publication, "verify_retirement", _forbidden)
    assert _prepare(completed).is_file()


def test_completed_identical_request_is_preserved_without_writing(
    completed: fixtures.Setup, monkeypatch: pytest.MonkeyPatch
) -> None:
    output = _prepare(completed)
    identity = output.stat()
    original = output.read_bytes()
    monkeypatch.setattr(request, "_publish", _forbidden)
    assert _prepare(completed) == output
    assert output.read_bytes() == original
    assert output.stat() == identity


@pytest.mark.parametrize("change", ["bytes", "public", "hardlink", "symlink"])
def test_occupied_request_requires_exact_bytes_mode_and_link_identity(
    completed: fixtures.Setup, change: str
) -> None:
    output = _prepare(completed)
    if change == "bytes":
        output.write_bytes(b"different result")
    elif change == "public":
        output.chmod(0o644)
    elif change == "hardlink":
        os.link(output, completed.fixture.work / "request-link.json")
    else:
        output.unlink()
        output.symlink_to(completed.path)
    original = output.read_bytes()
    with pytest.raises(ReplayError):
        _prepare(completed)
    assert output.read_bytes() == original


@pytest.mark.parametrize("sha256", ["0" * 64, "invalid", "A" * 64])
def test_expected_manifest_hash_is_required(completed: fixtures.Setup, sha256: str) -> None:
    with pytest.raises(ReplayError):
        request.prepare(
            completed.fixture.work / "manifest.json",
            sha256,
            completed=completed.fixture.work / "result",
            output=completed.fixture.work / "generated-request.json",
        )
    assert not (completed.fixture.work / "generated-request.json").exists()


@pytest.mark.parametrize(
    "name", ["operation.json", "receipt.json", "part-0001.json", "operation.lock"]
)
def test_missing_complete_metadata_is_not_repaired(completed: fixtures.Setup, name: str) -> None:
    missing = completed.fixture.work / "result" / name
    missing.unlink()
    with pytest.raises((ReplayError, FileNotFoundError)):
        _prepare(completed)
    assert not missing.exists()
    assert not (completed.fixture.work / "generated-request.json").exists()


@pytest.mark.parametrize("target", ["manifest", "batch", "parent", "inside"])
def test_missing_or_unrelated_directories_are_rejected(
    completed: fixtures.Setup, target: str
) -> None:
    output = completed.fixture.work / "generated-request.json"
    if target == "manifest":
        (completed.fixture.work / "manifest.json").unlink()
    elif target == "batch":
        (completed.fixture.work / "result").rename(completed.fixture.work / "retained-result")
    elif target == "parent":
        output = completed.fixture.work / "missing" / "request.json"
    else:
        output = completed.fixture.work / "result" / "request.json"
    with pytest.raises((ReplayError, FileNotFoundError)):
        _prepare(completed, output=output)
    assert not output.exists()


@pytest.mark.parametrize("name", ["operation.json", "receipt.json", "part-0001.json"])
@pytest.mark.parametrize("change", ["wrong-version", "extra-key"])
def test_changed_completed_semantics_are_rejected(
    completed: fixtures.Setup, name: str, change: str
) -> None:
    path = completed.fixture.work / "result" / name
    raw = record(json.loads(path.read_text()))
    if change == "wrong-version":
        raw["version"] = True
    else:
        raw["unknown"] = "unrelated state"
    path.write_text(json.dumps(raw), encoding="utf-8")
    with pytest.raises(ReplayError, match="differs"):
        _prepare(completed)
    assert not (completed.fixture.work / "generated-request.json").exists()


def test_manifest_order_cannot_be_changed_independently_of_completion(
    completed: fixtures.Setup,
) -> None:
    path = completed.fixture.work / "manifest.json"
    raw = record(json.loads(path.read_text()))
    parts = raw["parts"]
    assert isinstance(parts, list)
    raw["parts"] = list(reversed(parts))
    path.write_text(json.dumps(raw), encoding="utf-8")
    with pytest.raises(ReplayError, match="differs"):
        _prepare(completed)
    assert not (completed.fixture.work / "generated-request.json").exists()


def test_unrelated_batch_member_is_rejected(completed: fixtures.Setup) -> None:
    (completed.fixture.work / "result" / "unexpected").write_bytes(b"unrelated")
    with pytest.raises(ReplayError, match="membership"):
        _prepare(completed)


def test_busy_batch_owner_is_not_interrupted(completed: fixtures.Setup) -> None:
    with (completed.fixture.work / "result" / "operation.lock").open("rb") as owner:
        fcntl.flock(owner, fcntl.LOCK_EX | fcntl.LOCK_NB)
        with pytest.raises(ReplayError, match="already owned"):
            _prepare(completed)
    assert not (completed.fixture.work / "generated-request.json").exists()


@pytest.mark.parametrize("change", ["empty", "oversize", "directory", "hardlink", "symlink"])
def test_pinned_metadata_is_bounded_regular_and_single_link(
    completed: fixtures.Setup, change: str
) -> None:
    path = completed.fixture.work / "result" / "operation.json"
    if change == "empty":
        path.write_bytes(b"")
    elif change == "oversize":
        with path.open("wb") as handle:
            handle.truncate(batch.MAX_JSON_BYTES + 1)
    elif change == "directory":
        path.unlink()
        path.mkdir()
    elif change == "hardlink":
        os.link(path, completed.fixture.work / "linked-operation.json")
    else:
        path.unlink()
        path.symlink_to(completed.path)
    with pytest.raises(ReplayError):
        _prepare(completed)


def test_output_ancestor_alias_is_rejected(completed: fixtures.Setup) -> None:
    alias = completed.fixture.work / "alias"
    alias.symlink_to(completed.fixture.work, target_is_directory=True)
    with pytest.raises(ReplayError, match="symlinks"):
        _prepare(completed, output=alias / "generated-request.json")
    assert not (completed.fixture.work / "generated-request.json").exists()


def test_metadata_swap_between_lstat_and_open_is_rejected(
    completed: fixtures.Setup, monkeypatch: pytest.MonkeyPatch
) -> None:
    selected = completed.fixture.work / "manifest.json"
    original = os.open

    def changed(
        path: str | bytes | os.PathLike[str] | os.PathLike[bytes],
        flags: int,
        mode: int = 0o777,
        *,
        dir_fd: int | None = None,
    ) -> int:
        if path == selected:
            selected.write_bytes(b"changed before open")
        return original(path, flags, mode, dir_fd=dir_fd)

    monkeypatch.setattr(os, "open", changed)
    with pytest.raises(ReplayError, match="before opening"):
        _prepare(completed)


def test_metadata_changes_during_bounded_read_are_rejected(
    completed: fixtures.Setup, monkeypatch: pytest.MonkeyPatch
) -> None:
    original = os.fstat
    calls = 0

    def changed(descriptor: int) -> os.stat_result:
        nonlocal calls
        calls += 1
        if calls == 2:
            (completed.fixture.work / "manifest.json").write_bytes(b"during bounded read")
        return original(descriptor)

    monkeypatch.setattr(os, "fstat", changed)
    with pytest.raises(ReplayError, match="during"):
        _prepare(completed)


def test_metadata_changes_after_pin_are_rejected_before_publication(
    completed: fixtures.Setup, monkeypatch: pytest.MonkeyPatch
) -> None:
    original = request._pin
    selected = completed.fixture.work / "result" / "operation.json"

    def changed(path: Path) -> batch.Reference:
        ref = original(path)
        if path == selected:
            selected.write_bytes(b"changed after pin")
        return ref

    monkeypatch.setattr(request, "_pin", changed)
    with pytest.raises(ReplayError, match="changed after pinning"):
        _prepare(completed)
    assert not (completed.fixture.work / "generated-request.json").exists()


def test_metadata_replaced_after_read_is_rejected(
    completed: fixtures.Setup, monkeypatch: pytest.MonkeyPatch
) -> None:
    original = os.fdopen

    @contextmanager
    def changed(descriptor: int, mode: str) -> Iterator[BinaryIO]:
        assert mode == "rb"
        with original(descriptor, "rb") as handle:
            yield handle
        (completed.fixture.work / "manifest.json").write_bytes(b"changed after read")

    monkeypatch.setattr(os, "fdopen", changed)
    with pytest.raises(ReplayError, match="after reading"):
        _prepare(completed)


def test_terminal_metadata_mutation_is_rejected_before_publication(
    completed: fixtures.Setup, monkeypatch: pytest.MonkeyPatch
) -> None:
    original = InputHashCache.verify

    def changed(cache: InputHashCache) -> dict[str, str]:
        (completed.fixture.work / "result" / "part-0001.json").write_bytes(b"changed terminal")
        return original(cache)

    monkeypatch.setattr(InputHashCache, "verify", changed)
    with pytest.raises(ReplayError, match="changed"):
        _prepare(completed)
    assert not (completed.fixture.work / "generated-request.json").exists()


def test_output_parent_descriptor_does_not_accept_other_directory(
    completed: fixtures.Setup,
) -> None:
    other = completed.fixture.work / "other"
    other.mkdir()
    with ExitStack() as stack:
        descriptor = os.open(other, os.O_RDONLY | os.O_DIRECTORY)
        stack.callback(os.close, descriptor)
        with pytest.raises(ReplayError, match="parent identity"):
            request._parent(completed.fixture.work / "request.json", descriptor)


def test_output_ancestor_replaced_after_descriptor_open_never_receives_request(
    completed: fixtures.Setup, monkeypatch: pytest.MonkeyPatch
) -> None:
    parent = completed.fixture.work / "destination"
    parent.mkdir()
    other = completed.fixture.work / "other"
    other.mkdir()
    original = os.open

    def changed(
        path: str | bytes | os.PathLike[str] | os.PathLike[bytes],
        flags: int,
        mode: int = 0o777,
        *,
        dir_fd: int | None = None,
    ) -> int:
        descriptor = original(path, flags, mode, dir_fd=dir_fd)
        if path == parent:
            parent.rename(completed.fixture.work / "retained-destination")
            parent.symlink_to(other, target_is_directory=True)
        return descriptor

    monkeypatch.setattr(os, "open", changed)
    with pytest.raises(ReplayError, match="symlinks"):
        _prepare(completed, output=parent / "request.json")
    assert not (other / "request.json").exists()
    assert not (completed.fixture.work / "retained-destination" / "request.json").exists()


def test_exclusive_publication_never_overwrites_racing_writer(
    completed: fixtures.Setup, monkeypatch: pytest.MonkeyPatch
) -> None:
    original = request._publish

    def raced(output: Path, data: bytes) -> None:
        output.write_bytes(b"another writer")
        original(output, data)

    monkeypatch.setattr(request, "_publish", raced)
    with pytest.raises(FileExistsError):
        _prepare(completed)
    assert (completed.fixture.work / "generated-request.json").read_bytes() == b"another writer"


def test_new_request_file_and_parent_are_fsynced(
    completed: fixtures.Setup, monkeypatch: pytest.MonkeyPatch
) -> None:
    original = os.fsync
    kinds: list[str] = []

    def observed(descriptor: int) -> None:
        value = os.fstat(descriptor)
        kinds.append("directory" if stat.S_ISDIR(value.st_mode) else "file")
        original(descriptor)

    monkeypatch.setattr(os, "fsync", observed)
    _prepare(completed)
    assert kinds == ["file", "directory"]


def test_generated_request_budget_is_enforced_before_output(
    completed: fixtures.Setup, monkeypatch: pytest.MonkeyPatch
) -> None:
    def oversize(
        manifest: batch.Reference,
        *,
        operation: batch.Reference,
        receipt: batch.Reference,
        checkpoints: tuple[readonly.Checkpoint, ...],
    ) -> dict[str, object]:
        return {"oversize": "x" * (batch.MAX_JSON_BYTES + 1)}

    monkeypatch.setattr(request, "_value", oversize)
    with pytest.raises(ReplayError, match="Generated request exceeds"):
        _prepare(completed)
    assert not (completed.fixture.work / "generated-request.json").exists()


def test_actual_request_readback_must_match_planned_pins(
    completed: fixtures.Setup, monkeypatch: pytest.MonkeyPatch
) -> None:
    actual = readonly.load(completed.path)

    def retained(path: Path) -> readonly.Request:
        return actual

    monkeypatch.setattr(readonly, "load", retained)
    with pytest.raises(ReplayError, match="readback differs"):
        _prepare(completed)


def test_output_changed_after_file_fsync_is_rejected(
    completed: fixtures.Setup, monkeypatch: pytest.MonkeyPatch
) -> None:
    original = request._parent
    calls = 0

    def changed(output: Path, descriptor: int) -> None:
        nonlocal calls
        calls += 1
        if calls == 2:
            output.write_bytes(b"changed after file fsync")
        original(output, descriptor)

    monkeypatch.setattr(request, "_parent", changed)
    with pytest.raises(ReplayError, match="output identity changed"):
        _prepare(completed)


def test_cli_uses_generic_four_positional_arguments(
    completed: fixtures.Setup, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    output = completed.fixture.work / "generated-request.json"
    manifest = completed.fixture.work / "manifest.json"
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "part-retirement-request",
            str(manifest),
            _sha(manifest),
            str(completed.fixture.work / "result"),
            str(output),
        ],
    )
    request.main()
    assert capsys.readouterr().out == f"{output}\n"
    assert readonly.verify(readonly.load(output)).parts == (1, 2)
