import json
import os
from collections.abc import Mapping
from datetime import UTC, datetime
from pathlib import Path

import pytest

from replay import disk_budget, intermediate_offload, leaf_retirement, r2_worker
from replay.core import ReplayError
from replay.r2_worker import RemoteObject
from replay.storage_disposition import MEDIA_REFERENCE

UPLOADED = datetime(2026, 10, 17, 3, 0, tzinfo=UTC)
SYNTHETIC_SHA = "690758e29609deeb256114e268bd1ede8c4e77eb2406044e4943433a52a8598d"
OTHER_SHA = "a3ead5eedad5df82318c51685dbc1c147a36d1ff8584fc82de6b08d0bf63a795"


class FakeStore:
    """In-memory R2 double; `corrupt` stores different bytes than were sent."""

    def __init__(self, *, corrupt: bool = False) -> None:
        self.objects: dict[str, tuple[bytes, dict[str, str]]] = {}
        self.puts: list[str] = []
        self.gets: list[str] = []
        self.corrupt = corrupt

    def head(self, key: str) -> RemoteObject | None:
        stored = self.objects.get(key)
        if stored is None:
            return None
        return RemoteObject(len(stored[0]), UPLOADED, dict(stored[1]))

    def put(self, key: str, path: Path, metadata: Mapping[str, str]) -> None:
        self.puts.append(key)
        data = path.read_bytes()
        self.objects[key] = (data + b"!" if self.corrupt else data, dict(metadata))

    def get(self, key: str, destination: Path) -> int:
        self.gets.append(key)
        data = self.objects[key][0]
        with destination.open("xb") as stream:
            stream.write(data)
        return len(data)


def media(path: Path, data: bytes = b"synthetic media") -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(data)
    return path


def row_of(path: Path) -> leaf_retirement.Row:
    info = os.lstat(path)
    return leaf_retirement.Row(
        str(path),
        "file",
        info.st_dev,
        info.st_ino,
        info.st_nlink,
        info.st_size,
        info.st_mtime_ns,
        info.st_ctime_ns,
        None,
    )


def offloaded(tmp_path: Path, store: FakeStore) -> Path:
    """One rendered intermediate under `work`, offloaded for recording replay-1."""
    source = media(tmp_path / "work" / "render" / "av-master.mov")
    os.chmod(source, 0o640)
    os.utime(source, ns=(1_700_000_000_000_000_000, 1_700_000_000_000_000_000))
    intermediate_offload.offload(
        [row_of(source)],
        recording="replay-1",
        roots=(tmp_path / "work",),
        viewer_root=tmp_path / "edits" / "replay-1",
        store=store,
    )
    return source


def manifest(tmp_path: Path) -> dict[str, object]:
    path = tmp_path / "edits" / "replay-1" / "evidence" / "intermediate-offload.json"
    document: dict[str, object] = json.loads(path.read_text(encoding="utf-8"))
    return document


def test_offload_uploads_and_records_one_manifest_entry(tmp_path: Path) -> None:
    store = FakeStore()
    offloaded(tmp_path, store)
    assert store.puts == [f"intermediates/replay-1/{SYNTHETIC_SHA}"]
    assert manifest(tmp_path) == {
        "bucket": "replay-intermediates",
        "files": [
            {
                "bytes": 15,
                "expiresAt": "2027-01-15T03:00:00+00:00",
                "key": f"intermediates/replay-1/{SYNTHETIC_SHA}",
                "mode": 0o640,
                "mtimeNs": 1_700_000_000_000_000_000,
                "relative": "render/av-master.mov",
                "root": str(tmp_path / "work"),
                "sha256": SYNTHETIC_SHA,
                "state": "uploaded",
                "uploadedAt": "2026-10-17T03:00:00+00:00",
            }
        ],
        "recording": "replay-1",
        "version": 1,
    }


def test_manifest_names_no_absolute_media_path(tmp_path: Path) -> None:
    offloaded(tmp_path, FakeStore())
    path = tmp_path / "edits" / "replay-1" / "evidence" / "intermediate-offload.json"
    assert MEDIA_REFERENCE.findall(path.read_text(encoding="utf-8")) == []


def test_offload_rerun_reuses_the_matching_object(tmp_path: Path) -> None:
    store = FakeStore()
    source = offloaded(tmp_path, store)
    summary = intermediate_offload.offload(
        [row_of(source)],
        recording="replay-1",
        roots=(tmp_path / "work",),
        viewer_root=tmp_path / "edits" / "replay-1",
        store=store,
    )
    assert summary == intermediate_offload.OffloadSummary(uploaded=0, reused=1, skipped={})
    assert store.puts == [f"intermediates/replay-1/{SYNTHETIC_SHA}"]


def test_offload_skips_a_row_that_changed_after_planning(tmp_path: Path) -> None:
    source = media(tmp_path / "work" / "joined-video.mp4")
    planned = row_of(source)
    source.write_bytes(b"other bytes")
    store = FakeStore()
    summary = intermediate_offload.offload(
        [planned],
        recording="replay-1",
        roots=(tmp_path / "work",),
        viewer_root=tmp_path / "edits" / "replay-1",
        store=store,
    )
    assert summary.skipped == {str(source): "identity changed"}
    assert store.puts == []


def test_offload_skips_a_row_that_changes_while_hashing(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    source = media(tmp_path / "work" / "selected.wav")
    holds = iter([None, "identity changed"])
    monkeypatch.setattr(leaf_retirement, "static_hold", lambda _row, _roots: next(holds))
    summary = intermediate_offload.offload(
        [row_of(source)],
        recording="replay-1",
        roots=(tmp_path / "work",),
        viewer_root=tmp_path / "edits" / "replay-1",
        store=FakeStore(),
    )
    assert summary.skipped == {str(source): "identity changed while hashing"}


def test_offload_fails_closed_when_the_object_does_not_match(tmp_path: Path) -> None:
    with pytest.raises(ReplayError, match="does not match render/av-master.mov after upload"):
        offloaded(tmp_path, FakeStore(corrupt=True))
    assert manifest(tmp_path)["files"] == [
        {
            "bytes": 15,
            "expiresAt": None,
            "key": f"intermediates/replay-1/{SYNTHETIC_SHA}",
            "mode": 0o640,
            "mtimeNs": 1_700_000_000_000_000_000,
            "relative": "render/av-master.mov",
            "root": str(tmp_path / "work"),
            "sha256": SYNTHETIC_SHA,
            "state": "pending",
            "uploadedAt": None,
        }
    ]


def test_offload_rejects_a_row_outside_the_roots(tmp_path: Path) -> None:
    source = media(tmp_path / "elsewhere" / "part.mov")
    with pytest.raises(ReplayError, match="outside the allowed roots"):
        intermediate_offload._split(source, (tmp_path / "work",))


def test_mark_retired_marks_only_unlinked_uploads(tmp_path: Path) -> None:
    source = offloaded(tmp_path, FakeStore())
    assert (
        intermediate_offload.mark_retired(tmp_path / "edits" / "replay-1", recording="replay-1")
        == 0
    )
    source.unlink()
    assert (
        intermediate_offload.mark_retired(tmp_path / "edits" / "replay-1", recording="replay-1")
        == 1
    )
    assert intermediate_offload.status(tmp_path / "edits" / "replay-1", recording="replay-1") == {
        "pending": 0,
        "restored": 0,
        "retired": 1,
        "uploaded": 0,
    }


def test_restore_downloads_and_restores_mode_and_mtime(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(disk_budget, "require_headroom", lambda _paths, _need: [])
    store = FakeStore()
    source = offloaded(tmp_path, store)
    source.unlink()
    intermediate_offload.mark_retired(tmp_path / "edits" / "replay-1", recording="replay-1")
    restored = intermediate_offload.restore(
        tmp_path / "edits" / "replay-1",
        recording="replay-1",
        store=store,
        clock=lambda: UPLOADED,
    )
    assert restored == [source]
    assert source.read_bytes() == b"synthetic media"
    assert oct(source.stat().st_mode & 0o7777) == "0o640"
    assert source.stat().st_mtime_ns == 1_700_000_000_000_000_000
    assert list(source.parent.iterdir()) == [source]
    assert intermediate_offload.status(tmp_path / "edits" / "replay-1", recording="replay-1") == {
        "pending": 0,
        "restored": 1,
        "retired": 0,
        "uploaded": 0,
    }


def test_restore_accepts_an_identical_existing_file_without_downloading(tmp_path: Path) -> None:
    store = FakeStore()
    source = offloaded(tmp_path, store)
    restored = intermediate_offload.restore(
        tmp_path / "edits" / "replay-1", recording="replay-1", store=store
    )
    assert restored == [source]
    assert store.gets == []


def test_restore_never_overwrites_a_different_file(tmp_path: Path) -> None:
    store = FakeStore()
    source = offloaded(tmp_path, store)
    source.write_bytes(b"other bytes")
    with pytest.raises(ReplayError, match="different file already exists at render/av-master.mov"):
        intermediate_offload.restore(
            tmp_path / "edits" / "replay-1", recording="replay-1", store=store
        )
    assert source.read_bytes() == b"other bytes"


def test_restore_reports_an_expired_object(tmp_path: Path) -> None:
    store = FakeStore()
    offloaded(tmp_path, store).unlink()
    with pytest.raises(ReplayError, match="expired from R2 at 2027-01-15T03:00:00"):
        intermediate_offload.restore(
            tmp_path / "edits" / "replay-1",
            recording="replay-1",
            store=store,
            clock=lambda: datetime(2027, 1, 16, tzinfo=UTC),
        )


def test_restore_rejects_a_download_with_another_hash(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(disk_budget, "require_headroom", lambda _paths, _need: [])
    store = FakeStore()
    source = offloaded(tmp_path, store)
    source.unlink()
    store.objects[f"intermediates/replay-1/{SYNTHETIC_SHA}"] = (b"other bytes", {})
    with pytest.raises(ReplayError, match="does not match its SHA-256"):
        intermediate_offload.restore(
            tmp_path / "edits" / "replay-1",
            recording="replay-1",
            store=store,
            clock=lambda: UPLOADED,
        )
    assert list(source.parent.iterdir()) == []


def test_restore_of_an_unknown_entry_fails(tmp_path: Path) -> None:
    offloaded(tmp_path, FakeStore())
    with pytest.raises(ReplayError, match="No offloaded entry matches render/missing.mov"):
        intermediate_offload.restore(
            tmp_path / "edits" / "replay-1",
            recording="replay-1",
            store=FakeStore(),
            relative="render/missing.mov",
        )


def test_load_returns_an_empty_manifest_when_none_exists(tmp_path: Path) -> None:
    loaded = intermediate_offload.load(tmp_path / "missing.json", recording="replay-1")
    assert loaded == intermediate_offload.Manifest("replay-1", "replay-intermediates", {})


def test_load_refuses_another_recording(tmp_path: Path) -> None:
    offloaded(tmp_path, FakeStore())
    path = tmp_path / "edits" / "replay-1" / "evidence" / "intermediate-offload.json"
    with pytest.raises(ReplayError, match="another recording or version"):
        intermediate_offload.load(path, recording="replay-2")


def test_load_needs_a_bucket_and_files(tmp_path: Path) -> None:
    path = tmp_path / "manifest.json"
    path.write_text('{"version": 1, "recording": "replay-1", "files": {}}', encoding="utf-8")
    with pytest.raises(ReplayError, match="needs a bucket and a files array"):
        intermediate_offload.load(path, recording="replay-1")


def test_load_rejects_malformed_entries(tmp_path: Path) -> None:
    path = tmp_path / "manifest.json"
    entry = {
        "root": "/r",
        "relative": "a.mov",
        "key": "k",
        "sha256": "s",
        "bytes": 1,
        "mode": 420,
        "mtimeNs": 1,
        "state": "uploaded",
    }
    document = {"version": 1, "recording": "replay-1", "bucket": "b"}
    path.write_text(json.dumps({**document, "files": [{**entry, "bytes": "1"}]}), encoding="utf-8")
    with pytest.raises(ReplayError, match="field bytes must be an integer"):
        intermediate_offload.load(path, recording="replay-1")
    path.write_text(json.dumps({**document, "files": [{**entry, "key": ""}]}), encoding="utf-8")
    with pytest.raises(ReplayError, match="field key must be text"):
        intermediate_offload.load(path, recording="replay-1")
    path.write_text(
        json.dumps({**document, "files": [{**entry, "expiresAt": 5}]}), encoding="utf-8"
    )
    with pytest.raises(ReplayError, match="field expiresAt must be a timestamp"):
        intermediate_offload.load(path, recording="replay-1")
    path.write_text(
        json.dumps({**document, "files": [{**entry, "state": "gone"}]}), encoding="utf-8"
    )
    with pytest.raises(ReplayError, match="unknown state"):
        intermediate_offload.load(path, recording="replay-1")


def test_a_second_process_cannot_take_the_recording_lock(tmp_path: Path) -> None:
    with intermediate_offload.locked(tmp_path), pytest.raises(ReplayError, match="Another process"):
        intermediate_offload.offload(
            [], recording="replay-1", roots=(), viewer_root=tmp_path, store=FakeStore()
        )


def test_main_prints_status(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    offloaded(tmp_path, FakeStore())
    assert intermediate_offload.main(["status", str(tmp_path / "edits" / "replay-1")]) == 0
    assert json.loads(capsys.readouterr().out) == {
        "pending": 0,
        "restored": 0,
        "retired": 0,
        "uploaded": 1,
    }


def test_main_restores_through_a_worker_session(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    store = FakeStore()
    source = offloaded(tmp_path, store)
    buckets: list[str] = []

    class Session:
        def __init__(self, bucket: str) -> None:
            buckets.append(bucket)

        def __enter__(self) -> FakeStore:
            return store

        def __exit__(self, *_exc: object) -> None:
            return None

    monkeypatch.setattr(r2_worker, "session", Session)
    assert intermediate_offload.main(["restore", str(tmp_path / "edits" / "replay-1")]) == 0
    assert buckets == ["replay-intermediates"]
    assert json.loads(capsys.readouterr().out) == {"restored": [str(source)]}


def test_main_reports_errors(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    path = tmp_path / "edits" / "replay-1" / "evidence" / "intermediate-offload.json"
    path.parent.mkdir(parents=True)
    path.write_text('{"version": 2}', encoding="utf-8")
    assert intermediate_offload.main(["restore", str(tmp_path / "edits" / "replay-1")]) == 1
    assert "INTERMEDIATE_OFFLOAD_ERROR" in capsys.readouterr().err


def test_restore_refuses_a_relative_path_under_several_roots(tmp_path: Path) -> None:
    store = FakeStore()
    work = media(tmp_path / "work" / "av-master.mov")
    storage = media(tmp_path / "storage" / "av-master.mov", b"other bytes")
    intermediate_offload.offload(
        [row_of(work), row_of(storage)],
        recording="replay-1",
        roots=(tmp_path / "work", tmp_path / "storage"),
        viewer_root=tmp_path / "edits" / "replay-1",
        store=store,
    )
    with pytest.raises(ReplayError, match="av-master.mov exists under several roots"):
        intermediate_offload.restore(
            tmp_path / "edits" / "replay-1",
            recording="replay-1",
            store=store,
            relative="av-master.mov",
        )
