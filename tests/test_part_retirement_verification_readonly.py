import builtins
import fcntl
import hashlib
import json
import os
import sys
from collections.abc import Iterable
from dataclasses import dataclass, replace
from pathlib import Path
from typing import BinaryIO, NoReturn, TextIO

import pytest
import test_part_retirement_verification_batch as fixtures

from replay import part_master_publication as publication
from replay import part_retirement_verification_batch as batch
from replay import part_retirement_verification_readonly as readonly
from replay.core import ReplayError
from replay.input_hash_cache import InputHashCache
from replay.recordings import record


@dataclass(frozen=True)
class Setup:
    fixture: fixtures.Fixture
    path: Path


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _save(path: Path, value: dict[str, object]) -> Path:
    path.write_text(json.dumps(value), encoding="utf-8")
    return path


def _reference(path: Path) -> batch.Reference:
    return batch.Reference(path, _sha(path))


def _rebind(setup: Setup, target: str, path: Path) -> readonly.Request:
    raw = record(json.loads(setup.path.read_text()))
    if target == "checkpoint":
        rows = raw["checkpoints"]
        assert isinstance(rows, list)
        rows[0] = {**record(rows[0]), "artifact": _reference(path).json()}
    else:
        raw[target] = _reference(path).json()
    _save(setup.path, raw)
    return readonly.load(setup.path)


def _complete(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    *,
    phase: batch.Phase = "before-delete",
    count: int = 2,
) -> Setup:
    fixture = fixtures._fixture(tmp_path, monkeypatch, count)
    if phase == "after-delete":
        tuple(map(Path.unlink, fixture.sources))
    manifest = fixture.manifest(phase)
    receipt = batch.run(manifest, _sha(manifest), fixture.work / "result")
    path = _save(
        fixture.work / "readonly-request.json",
        {
            "version": 1,
            "kind": "readonly-retirement-verification",
            "manifest": _reference(manifest).json(),
            "operation": _reference(receipt.parent / "operation.json").json(),
            "receipt": _reference(receipt).json(),
            "checkpoints": [
                {
                    "part": number,
                    "artifact": _reference(receipt.parent / f"part-{number:04d}.json").json(),
                }
                for number in range(1, count + 1)
            ],
        },
    )
    return Setup(fixture, path)


@pytest.fixture
def completed(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Setup:
    return _complete(tmp_path, monkeypatch)


def _forbidden(*args: object, **kwargs: object) -> NoReturn:
    raise AssertionError("Readonly consumer attempted a producer or filesystem write")


def _no_writes(monkeypatch: pytest.MonkeyPatch) -> list[int]:
    original_open = os.open
    original_builtin_open = builtins.open
    original_path_open = Path.open
    flags_seen: list[int] = []

    def read_open(
        path: str | bytes | os.PathLike[str] | os.PathLike[bytes],
        flags: int,
        mode: int = 0o777,
        *,
        dir_fd: int | None = None,
    ) -> int:
        assert not flags & (os.O_CREAT | os.O_WRONLY | os.O_RDWR | os.O_TRUNC | os.O_APPEND)
        flags_seen.append(flags)
        return original_open(path, flags, mode, dir_fd=dir_fd)

    def read_builtin_open(file: str | bytes | int | Path, mode: str = "r") -> BinaryIO | TextIO:
        if mode == "rb":
            return original_builtin_open(file, "rb")
        if mode == "r":
            return original_builtin_open(file, "r", encoding="utf-8")
        return _forbidden(file, mode)

    def read_path_open(
        path: Path,
        mode: str = "r",
        buffering: int = -1,
        encoding: str | None = None,
        errors: str | None = None,
        newline: str | None = None,
    ) -> BinaryIO | TextIO:
        if mode == "rb":
            return original_path_open(path, "rb", buffering=buffering)
        if mode == "r":
            return original_path_open(
                path, "r", buffering=buffering, encoding=encoding, errors=errors, newline=newline
            )
        return _forbidden(path, mode)

    monkeypatch.setattr(os, "open", read_open)
    monkeypatch.setattr(builtins, "open", read_builtin_open)
    monkeypatch.setattr(Path, "open", read_path_open)
    monkeypatch.setattr(Path, "mkdir", _forbidden)
    monkeypatch.setattr(Path, "touch", _forbidden)
    monkeypatch.setattr(Path, "unlink", _forbidden)
    monkeypatch.setattr(Path, "rename", _forbidden)
    monkeypatch.setattr(Path, "replace", _forbidden)
    monkeypatch.setattr(Path, "write_bytes", _forbidden)
    monkeypatch.setattr(Path, "write_text", _forbidden)
    monkeypatch.setattr(batch, "run", _forbidden)
    monkeypatch.setattr(batch, "save_new", _forbidden)
    monkeypatch.setattr(publication, "publish", _forbidden)
    monkeypatch.setattr(publication, "prepare_retirement", _forbidden)
    return flags_seen


@pytest.mark.parametrize("phase", ["before-delete", "after-delete"])
@pytest.mark.parametrize("count", [1, 2, 3])
def test_complete_domains_are_readonly_and_preserve_history(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, phase: batch.Phase, count: int
) -> None:
    setup = _complete(tmp_path, monkeypatch, phase=phase, count=count)
    request = readonly.load(setup.path)
    before = {ref.path: ref.path.read_bytes() for ref in request.metadata}
    names = tuple(sorted(child.name for child in request.output.iterdir()))
    with monkeypatch.context() as guard:
        flags = _no_writes(guard)
        result = readonly.verify(request)
    assert flags
    assert result.phase == phase
    assert result.parts == tuple(range(1, count + 1))
    assert result.hash_reads == 2 * len(result.inputs)
    assert result.bytes_read == 2 * sum(Path(path).stat().st_size for path in result.inputs)
    assert result.cache_hits > count
    assert {ref.path: ref.path.read_bytes() for ref in request.metadata} == before
    assert tuple(sorted(child.name for child in request.output.iterdir())) == names
    assert result.json()["filesWritten"] is False
    assert result.json()["deletionPerformed"] is False
    assert result.json()["releaseDecision"] is None
    assert result.json()["qualityApproved"] is False
    assert result.json()["deliveryApproved"] is False
    assert result.json()["physicalIOBytes"] is None
    assert result.json()["sourceAbsenceObserved"] == (phase == "after-delete")


@pytest.mark.parametrize(
    "value",
    [
        {},
        {"version": True},
        {"version": 1, "kind": "other"},
        {"version": 1, "kind": "readonly-retirement-verification", "checkpoints": []},
    ],
)
def test_invalid_request_is_rejected(completed: Setup, value: dict[str, object]) -> None:
    _save(completed.path, value)
    with pytest.raises(ReplayError, match="request"):
        readonly.load(completed.path)


@pytest.mark.parametrize(
    "value",
    [{}, {"part": True, "artifact": {}}, {"part": 0, "artifact": {}}, {"part": 1025}],
)
def test_invalid_checkpoint_is_rejected(completed: Setup, value: dict[str, object]) -> None:
    raw = record(json.loads(completed.path.read_text()))
    raw["checkpoints"] = [value]
    _save(completed.path, raw)
    with pytest.raises(ReplayError, match="checkpoint"):
        readonly.load(completed.path)


@pytest.mark.parametrize("target", ["operation", "receipt", "checkpoint", "duplicate", "alias"])
def test_request_artifacts_are_exact_distinct_and_canonical(completed: Setup, target: str) -> None:
    raw = record(json.loads(completed.path.read_text()))
    rows = raw["checkpoints"]
    assert isinstance(rows, list)
    if target in {"operation", "receipt"}:
        raw[target] = raw["manifest"]
    elif target == "checkpoint":
        rows[0] = {"part": 3, "artifact": record(rows[0])["artifact"]}
    elif target == "duplicate":
        rows.append(rows[0])
    else:
        alias = completed.fixture.work / "alias"
        alias.symlink_to(completed.fixture.work / "manifest.json")
        raw["manifest"] = {"path": str(alias), "sha256": "0" * 64}
    _save(completed.path, raw)
    with pytest.raises(ReplayError):
        readonly.load(completed.path)


@pytest.mark.parametrize(
    "name", ["operation.json", "operation.lock", "receipt.json", "part-0001.json"]
)
def test_missing_completed_artifact_never_recreates_or_writes(
    completed: Setup, monkeypatch: pytest.MonkeyPatch, name: str
) -> None:
    request = readonly.load(completed.path)
    missing = request.output / name
    missing.unlink()
    with monkeypatch.context() as guard:
        _no_writes(guard)
        with pytest.raises(ReplayError, match="membership"):
            readonly.verify(request)
    assert not missing.exists()


def test_missing_output_root_never_creates_it(
    completed: Setup, monkeypatch: pytest.MonkeyPatch
) -> None:
    request = readonly.load(completed.path)
    request.output.rename(request.output.with_name("retained-original"))
    with monkeypatch.context() as guard:
        _no_writes(guard)
        with pytest.raises(FileNotFoundError):
            readonly.verify(request)
    assert not request.output.exists()


@pytest.mark.parametrize("change", ["extra", "linked", "root-file"])
def test_output_membership_and_kind_are_strict(completed: Setup, change: str) -> None:
    request = readonly.load(completed.path)
    if change == "extra":
        (request.output / "unexpected").write_bytes(b"unknown")
    elif change == "linked":
        path = request.output / "receipt.json"
        path.unlink()
        path.symlink_to(completed.path)
    else:
        request.output.rename(request.output.with_name("retained-original"))
        request.output.write_bytes(b"not a directory")
    with pytest.raises(ReplayError):
        readonly.verify(request)


@pytest.mark.parametrize("change", ["public", "nonempty", "hardlinked"])
def test_lock_requires_original_private_regular_file(completed: Setup, change: str) -> None:
    request = readonly.load(completed.path)
    lock = request.output / "operation.lock"
    if change == "public":
        lock.chmod(0o644)
    elif change == "nonempty":
        lock.write_bytes(b"not empty")
    else:
        os.link(lock, completed.fixture.work / "linked-lock")
    with pytest.raises(ReplayError, match="private empty"):
        readonly.verify(request)


def test_exclusive_owner_is_not_interrupted_or_written(
    completed: Setup, monkeypatch: pytest.MonkeyPatch
) -> None:
    request = readonly.load(completed.path)
    with (request.output / "operation.lock").open("rb") as owner:
        fcntl.flock(owner, fcntl.LOCK_EX | fcntl.LOCK_NB)
        with monkeypatch.context() as guard:
            _no_writes(guard)
            with pytest.raises(ReplayError, match="already owned"):
                readonly.verify(request)


def test_lock_replacement_between_check_and_open_is_rejected_readonly(
    completed: Setup, monkeypatch: pytest.MonkeyPatch
) -> None:
    request = readonly.load(completed.path)
    lock = request.output / "operation.lock"
    original_unlink = Path.unlink
    original_open = os.open
    with monkeypatch.context() as guard:
        _no_writes(guard)
        guarded_open = os.open

        def raced_open(
            path: str | bytes | os.PathLike[str] | os.PathLike[bytes],
            flags: int,
            mode: int = 0o777,
            *,
            dir_fd: int | None = None,
        ) -> int:
            if path == lock:
                original_unlink(lock)
                os.close(original_open(lock, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600))
            return guarded_open(path, flags, mode, dir_fd=dir_fd)

        guard.setattr(os, "open", raced_open)
        with pytest.raises(ReplayError, match="before opening"):
            readonly.verify(request)


@pytest.mark.parametrize(
    "target", ["operation", "receipt", "checkpoint", "source", "movie", "voice", "code"]
)
def test_changed_bytes_are_detected(completed: Setup, target: str) -> None:
    request = readonly.load(completed.path)
    selected = {
        "operation": request.operation.path,
        "receipt": request.receipt.path,
        "checkpoint": request.checkpoints[0].artifact.path,
        "source": completed.fixture.sources[0],
        "movie": completed.fixture.movies[0],
        "voice": completed.fixture.voice,
        "code": batch.ROOT / "src/replay/core.py",
    }[target]
    selected.write_bytes(b"changed")
    with pytest.raises((ReplayError, json.JSONDecodeError)):
        readonly.verify(request)


@pytest.mark.parametrize("target", ["operation", "receipt", "checkpoint"])
def test_rebound_metadata_must_preserve_original_semantics(completed: Setup, target: str) -> None:
    request = readonly.load(completed.path)
    selected = {
        "operation": request.operation.path,
        "receipt": request.receipt.path,
        "checkpoint": request.checkpoints[0].artifact.path,
    }[target]
    raw = record(json.loads(selected.read_text()))
    raw["phase"] = "after-delete"
    _save(selected, raw)
    request = _rebind(completed, target, selected)
    with pytest.raises(ReplayError, match="differs"):
        readonly.verify(request)


@pytest.mark.parametrize("counter", [None, -1, True, "1"])
def test_original_receipt_counter_types_are_validated(completed: Setup, counter: object) -> None:
    request = readonly.load(completed.path)
    raw = record(json.loads(request.receipt.path.read_text()))
    raw["hashReads"] = counter
    _save(request.receipt.path, raw)
    request = _rebind(completed, "receipt", request.receipt.path)
    with pytest.raises(ReplayError, match="counters"):
        readonly.verify(request)


@pytest.mark.parametrize("target", ["operation", "receipt", "checkpoint"])
def test_json_boolean_cannot_substitute_integer_in_completed_metadata(
    completed: Setup, target: str
) -> None:
    request = readonly.load(completed.path)
    selected = {
        "operation": request.operation.path,
        "receipt": request.receipt.path,
        "checkpoint": request.checkpoints[0].artifact.path,
    }[target]
    raw = record(json.loads(selected.read_text()))
    raw["version"] = True
    _save(selected, raw)
    request = _rebind(completed, target, selected)
    with pytest.raises(ReplayError, match="differs"):
        readonly.verify(request)


def test_owned_model_must_match_actual_pinned_request_bytes(completed: Setup) -> None:
    request = readonly.load(completed.path)
    changed = replace(request, operation=request.receipt)
    with pytest.raises(ReplayError, match="request model"):
        readonly.verify(changed)


def test_discovery_change_after_initial_hashing_is_rejected(
    completed: Setup, monkeypatch: pytest.MonkeyPatch
) -> None:
    request = readonly.load(completed.path)
    original = batch._manifest
    calls = 0

    def changed(
        path: Path, digest: str
    ) -> tuple[batch.Phase, tuple[batch.Part, ...], tuple[batch.Reference, ...]]:
        nonlocal calls
        calls += 1
        phase, parts, refs = original(path, digest)
        return ("after-delete", parts, refs) if calls == 2 else (phase, parts, refs)

    monkeypatch.setattr(batch, "_manifest", changed)
    with monkeypatch.context() as guard:
        _no_writes(guard)
        with pytest.raises(ReplayError, match="discovery changed"):
            readonly.verify(request)


def test_incomplete_or_reordered_checkpoint_membership_is_not_adopted(completed: Setup) -> None:
    raw = record(json.loads(completed.path.read_text()))
    rows = raw["checkpoints"]
    assert isinstance(rows, list)
    raw["checkpoints"] = list(reversed(rows))
    _save(completed.path, raw)
    with pytest.raises(ReplayError, match="manifest order"):
        readonly.verify(readonly.load(completed.path))


@pytest.mark.parametrize("target", ["movie", "voice", "metadata", "lock", "missing-checkpoint"])
def test_changes_during_domains_are_rejected_without_repair(
    completed: Setup, monkeypatch: pytest.MonkeyPatch, target: str
) -> None:
    request = readonly.load(completed.path)
    original = publication.verify_retirement
    original_unlink = Path.unlink
    lock = request.output / "operation.lock"

    def mutate(config: Path) -> None:
        original(config)
        if target == "missing-checkpoint":
            original_unlink(request.checkpoints[0].artifact.path)
        elif target == "lock":
            original_unlink(lock)
            with os.fdopen(
                original_os_open(lock, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600), "wb"
            ):
                pass
        else:
            selected = {
                "movie": completed.fixture.movies[0],
                "voice": completed.fixture.voice,
                "metadata": request.operation.path,
            }[target]
            with os.fdopen(original_os_open(selected, os.O_WRONLY | os.O_TRUNC), "wb") as actor:
                actor.write(b"changed")

    original_os_open = os.open
    monkeypatch.setattr(publication, "verify_retirement", mutate)
    with monkeypatch.context() as guard:
        _no_writes(guard)
        with pytest.raises((ReplayError, FileNotFoundError)):
            readonly.verify(request)
    assert not (request.output / "new-receipt.json").exists()


def test_after_delete_reappearance_is_not_accepted(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    setup = _complete(tmp_path, monkeypatch, phase="after-delete")
    request = readonly.load(setup.path)
    original = publication.verify_retirement

    def reappear(config: Path) -> None:
        original(config)
        setup.fixture.sources[0].write_bytes(b"returned")

    monkeypatch.setattr(publication, "verify_retirement", reappear)
    with pytest.raises(ReplayError, match="absence"):
        readonly.verify(request)


def test_terminal_bytes_are_fresh_even_when_domain_hashes_are_cached(
    completed: Setup, monkeypatch: pytest.MonkeyPatch
) -> None:
    request = readonly.load(completed.path)
    original = InputHashCache.verify

    def changed(cache: InputHashCache) -> dict[str, str]:
        completed.fixture.movies[0].write_bytes(b"terminal mutation")
        return original(cache)

    monkeypatch.setattr(InputHashCache, "verify", changed)
    with pytest.raises(ReplayError, match="changed"):
        readonly.verify(request)


def test_source_phase_is_checked_again_after_terminal_hashing(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    setup = _complete(tmp_path, monkeypatch, phase="after-delete")
    request = readonly.load(setup.path)
    original = InputHashCache.verify

    def returned(cache: InputHashCache) -> dict[str, str]:
        verified = original(cache)
        setup.fixture.sources[0].write_bytes(b"returned after the terminal boundary")
        return verified

    monkeypatch.setattr(InputHashCache, "verify", returned)
    with pytest.raises(ReplayError, match="absence"):
        readonly.verify(request)


def test_terminal_inventory_must_be_exact(
    completed: Setup, monkeypatch: pytest.MonkeyPatch
) -> None:
    original = InputHashCache.verify

    def changed(cache: InputHashCache) -> dict[str, str]:
        verified = original(cache)
        verified.pop(str(completed.path))
        return verified

    monkeypatch.setattr(InputHashCache, "verify", changed)
    with pytest.raises(ReplayError, match="terminal input inventory"):
        readonly.verify(readonly.load(completed.path))


def test_uncached_overflow_preserves_fresh_verification(
    completed: Setup, monkeypatch: pytest.MonkeyPatch
) -> None:
    def limited(paths: Iterable[Path], *, allow_uncached_overflow: bool) -> InputHashCache:
        return InputHashCache(paths, max_files=1, allow_uncached_overflow=allow_uncached_overflow)

    monkeypatch.setattr(readonly, "InputHashCache", limited)
    result = readonly.verify(readonly.load(completed.path))
    assert result.phase == "before-delete"
    assert result.parts == (1, 2)
    assert result.hash_reads > 2 * len(result.inputs)


def test_cli_has_no_output_or_resume_option(
    completed: Setup, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.setattr(sys, "argv", ["readonly", str(completed.path)])
    readonly.main()
    raw = record(json.loads(capsys.readouterr().out))
    assert raw["kind"] == "readonly-retirement-verification"
    assert raw["parts"] == [1, 2]
    assert raw["filesWritten"] is False
    assert raw["retirementDomainVerified"] is True
    assert raw["qualityApproved"] is False
    assert raw["deliveryApproved"] is False
    monkeypatch.setattr(sys, "argv", ["readonly", str(completed.path), "--resume"])
    with pytest.raises(SystemExit):
        readonly.main()
