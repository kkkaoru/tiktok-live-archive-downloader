"""Bounded Darwin clonefile compatibility proof using synthetic media only."""

import ctypes
import errno
import json
import os
import struct
from pathlib import Path

import pytest

from replay import original_storage as storage
from replay.core import ReplayError
from replay.jobs import file_sha256, write_record


def test_native_clone_preserves_active_source_identity(tmp_path: Path) -> None:
    source = tmp_path / "synthetic-source.bin"
    destination = tmp_path / "synthetic-clone.bin"
    source.write_bytes(b"synthetic storage clone\n" * 65536)
    before = source.stat()
    library = ctypes.CDLL("/usr/lib/libSystem.B.dylib", use_errno=True)
    clone = library.clonefile
    clone.argtypes = (ctypes.c_char_p, ctypes.c_char_p, ctypes.c_uint32)
    clone.restype = ctypes.c_int
    storage._clone_file(source, destination)
    after = source.stat()
    diagnostic: dict[str, object] = {
        "result": 0,
        "sourceCtimeUnchanged": before.st_ctime_ns == after.st_ctime_ns,
        "sourceNlinkUnchanged": before.st_nlink == after.st_nlink,
        "sourceInodeUnchanged": before.st_ino == after.st_ino,
        "sourceMtimeUnchanged": before.st_mtime_ns == after.st_mtime_ns,
        "sourceBytes": before.st_size,
    }
    (tmp_path / "clone-observation.json").write_text(json.dumps(diagnostic), encoding="utf-8")
    assert destination.stat().st_nlink == 1
    assert not os.path.samefile(source, destination)
    assert destination.read_bytes() == b"synthetic storage clone\n" * 65536
    # ATTR_CMNEXT_CLONEID in forkattr, FSOPT_ATTR_CMN_EXTENDED: SDK sys/attr.h.
    attrs = ctypes.create_string_buffer(struct.pack("=HHIIIII", 5, 0, 0, 0, 0, 0, 0x100))
    source_attributes = ctypes.create_string_buffer(12)
    clone_attributes = ctypes.create_string_buffer(12)
    getattrlist = library.getattrlist
    getattrlist.argtypes = (
        ctypes.c_char_p,
        ctypes.c_void_p,
        ctypes.c_void_p,
        ctypes.c_size_t,
        ctypes.c_ulong,
    )
    getattrlist.restype = ctypes.c_int
    assert getattrlist(os.fsencode(source), attrs, source_attributes, 12, 0x20) == 0
    assert getattrlist(os.fsencode(destination), attrs, clone_attributes, 12, 0x20) == 0
    diagnostic["sourceCloneId"] = struct.unpack("=IQ", source_attributes.raw)[1]
    diagnostic["destinationCloneId"] = struct.unpack("=IQ", clone_attributes.raw)[1]
    diagnostic["cloneIdsEqual"] = source_attributes.raw == clone_attributes.raw
    diagnostic["destinationLinks"] = destination.stat().st_nlink
    diagnostic["exclusiveSecondResult"] = clone(
        os.fsencode(source), os.fsencode(destination), 0x0008
    )
    diagnostic["exclusiveSecondErrno"] = ctypes.get_errno()
    (tmp_path / "clone-observation.json").write_text(json.dumps(diagnostic), encoding="utf-8")
    assert diagnostic["cloneIdsEqual"] is True
    assert diagnostic["sourceCloneId"] != 0
    assert diagnostic["exclusiveSecondResult"] == -1
    assert diagnostic["exclusiveSecondErrno"] == 17
    assert diagnostic["sourceCtimeUnchanged"] is True
    assert diagnostic["sourceNlinkUnchanged"] is True
    assert diagnostic["sourceInodeUnchanged"] is True
    assert diagnostic["sourceMtimeUnchanged"] is True
    destination.write_bytes(b"independent copy-on-write change")
    assert source.read_bytes() == b"synthetic storage clone\n" * 65536
    assert destination.read_bytes() == b"independent copy-on-write change"


@pytest.fixture
def clone_paths(tmp_path: Path) -> tuple[Path, Path, Path, Path]:
    jobs = tmp_path / "jobs"
    jobs.mkdir()
    source = tmp_path / "source.bin"
    source.write_bytes(b"original clone fixture")
    write_record(jobs / "456.json", source, "complete")
    return jobs, source, tmp_path / "destination.bin", tmp_path / "intent.json"


def test_clone_organization_and_resume(clone_paths: tuple[Path, Path, Path, Path]) -> None:
    root, source, destination, receipt = clone_paths
    before = source.stat()
    result = storage.organize(
        root=root,
        replay_id="456",
        source=source,
        destination=destination,
        expected_sha256=file_sha256(source),
        receipt=receipt,
        method="clone",
    )
    assert result["sourceIdentityUnchanged"] is True
    assert result["destinationSingleLink"] is True
    assert result["method"] == "clone"
    assert source.stat().st_ctime_ns - before.st_ctime_ns == 0
    assert source.stat().st_nlink == 1
    assert destination.stat().st_nlink == 1
    assert not source.samefile(destination)
    assert json.loads((root / "456.json").read_text())["output"] == str(destination)
    resumed = storage.organize(
        root=root,
        replay_id="456",
        source=source,
        destination=destination,
        expected_sha256=file_sha256(source),
        receipt=receipt,
        method="clone",
    )
    assert resumed["method"] == "clone"
    assert destination.read_bytes() == b"original clone fixture"


def test_clone_collision_is_not_overwritten(clone_paths: tuple[Path, Path, Path, Path]) -> None:
    root, source, destination, receipt = clone_paths
    destination.write_bytes(b"unrelated destination")
    with pytest.raises(ReplayError, match="independent matching"):
        storage.organize(
            root=root,
            replay_id="456",
            source=source,
            destination=destination,
            expected_sha256=file_sha256(source),
            receipt=receipt,
            method="clone",
        )
    assert destination.read_bytes() == b"unrelated destination"
    assert not receipt.exists()
    with pytest.raises(FileExistsError):
        storage._clone_file(source, destination)
    assert destination.read_bytes() == b"unrelated destination"


def test_unsupported_clone_has_no_copy_or_link_fallback(
    clone_paths: tuple[Path, Path, Path, Path], monkeypatch: pytest.MonkeyPatch
) -> None:
    root, source, destination, receipt = clone_paths

    def unsupported(_source: Path, _destination: Path) -> None:
        raise OSError(errno.ENOTSUP, "synthetic unsupported volume")

    monkeypatch.setattr(storage, "_clone_file", unsupported)
    with pytest.raises(OSError, match="unsupported volume"):
        storage.organize(
            root=root,
            replay_id="456",
            source=source,
            destination=destination,
            expected_sha256=file_sha256(source),
            receipt=receipt,
            method="clone",
        )
    assert not destination.exists()
    assert source.stat().st_nlink == 1
    assert source.read_bytes() == b"original clone fixture"
    assert receipt.exists()
    assert json.loads((root / "456.json").read_text())["output"] == str(source)


@pytest.mark.parametrize("after_record", [False, True])
def test_clone_interrupted_record_reconciles(
    clone_paths: tuple[Path, Path, Path, Path],
    monkeypatch: pytest.MonkeyPatch,
    after_record: bool,
) -> None:
    root, source, destination, receipt = clone_paths

    def interrupted(path: Path, output: Path, status: str) -> None:
        if after_record:
            write_record(path, output, status)
        raise OSError("synthetic record interruption")

    monkeypatch.setattr(storage, "write_record", interrupted)
    with pytest.raises(OSError, match="record interruption"):
        storage.organize(
            root=root,
            replay_id="456",
            source=source,
            destination=destination,
            expected_sha256=file_sha256(source),
            receipt=receipt,
            method="clone",
        )
    monkeypatch.setattr(storage, "write_record", write_record)
    result = storage.organize(
        root=root,
        replay_id="456",
        source=source,
        destination=destination,
        expected_sha256=file_sha256(source),
        receipt=receipt,
        method="clone",
    )
    assert result["method"] == "clone"
    assert source.stat().st_nlink == 1
    assert destination.stat().st_nlink == 1


def test_clone_source_identity_change_is_rejected(
    clone_paths: tuple[Path, Path, Path, Path],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    root, source, destination, receipt = clone_paths
    original = storage._clone_file

    def changed(src: Path, dst: Path) -> None:
        original(src, dst)
        os.link(src, src.with_suffix(".unexpected-link"))

    monkeypatch.setattr(storage, "_clone_file", changed)
    with pytest.raises(ReplayError, match="single link"):
        storage.organize(
            root=root,
            replay_id="456",
            source=source,
            destination=destination,
            expected_sha256=file_sha256(source),
            receipt=receipt,
            method="clone",
        )
    assert json.loads((root / "456.json").read_text())["output"] == str(source)


def test_invalid_method_is_rejected(clone_paths: tuple[Path, Path, Path, Path]) -> None:
    root, source, destination, receipt = clone_paths
    with pytest.raises(ReplayError, match="Unsupported"):
        storage.organize(
            root=root,
            replay_id="456",
            source=source,
            destination=destination,
            expected_sha256=file_sha256(source),
            receipt=receipt,
            method="copy",
        )
    assert not destination.exists()


def test_identical_existing_clone_without_intent_is_not_adopted(
    clone_paths: tuple[Path, Path, Path, Path],
) -> None:
    root, source, destination, receipt = clone_paths
    storage._clone_file(source, destination)
    with pytest.raises(ReplayError, match="without this intent"):
        storage.organize(
            root=root,
            replay_id="456",
            source=source,
            destination=destination,
            expected_sha256=file_sha256(source),
            receipt=receipt,
            method="clone",
        )
    assert not receipt.exists()
    assert destination.read_bytes() == b"original clone fixture"


def test_changed_clone_cannot_resume(clone_paths: tuple[Path, Path, Path, Path]) -> None:
    root, source, destination, receipt = clone_paths
    storage.organize(
        root=root,
        replay_id="456",
        source=source,
        destination=destination,
        expected_sha256=file_sha256(source),
        receipt=receipt,
        method="clone",
    )
    destination.write_bytes(b"different clone bytes!")
    with pytest.raises(ReplayError, match="independent matching"):
        storage.organize(
            root=root,
            replay_id="456",
            source=source,
            destination=destination,
            expected_sha256=file_sha256(source),
            receipt=receipt,
            method="clone",
        )
    assert source.read_bytes() == b"original clone fixture"
