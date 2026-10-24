import hashlib
import json
import os
from datetime import UTC, datetime
from pathlib import Path

import pytest

from replay import delivery_offload, disk_budget, r2_worker
from replay.core import ReplayError
from replay.delivery_offload import Entry

MOVIE = b"delivered movie bytes"
MOVIE_SHA = "6c3f4b4bdc6a0e3a2d74f2d0a8d0d1ab7a1d10d2c35d9f5e4a6b1d9c5a0e0b11"


def clock() -> datetime:
    return datetime(2026, 10, 24, 9, 0, tzinfo=UTC)


class Store:
    def __init__(
        self, objects: dict[str, bytes], *, readback: bytes | None = None, etag: str | None = None
    ) -> None:
        self.objects = objects
        self.readback = readback
        self.etag = etag
        self.calls: list[str] = []

    def head(self, key: str) -> r2_worker.RemoteObject | None:
        self.calls.append(f"head {key}")
        if key not in self.objects:
            return None
        return r2_worker.RemoteObject(len(self.objects[key]), clock(), {}, self.etag)

    def digest(self, key: str) -> tuple[int, str]:
        self.calls.append(f"digest {key}")
        data = self.objects[key] if self.readback is None else self.readback
        return len(data), hashlib.sha256(data).hexdigest()

    def get(self, key: str, destination: Path) -> int:
        self.calls.append(f"get {key}")
        data = self.objects[key] if self.readback is None else self.readback
        with destination.open("xb") as stream:
            stream.write(data)
        return len(data)


@pytest.fixture(autouse=True)
def roomy(monkeypatch: pytest.MonkeyPatch) -> list[tuple[list[Path], int]]:
    seen: list[tuple[list[Path], int]] = []

    def require(paths: list[Path], need: int) -> list[disk_budget.Headroom]:
        seen.append((list(paths), need))
        return []

    monkeypatch.setattr(disk_budget, "require_headroom", require)
    return seen


def delivery(tmp_path: Path) -> tuple[Path, Path, Path]:
    root = tmp_path / "edits" / "replay-1"
    movie = tmp_path / "store" / "delivery" / "2026-10-01_JST_replay-1-review.mp4"
    candidate = root / "media" / "final" / "candidate.mp4"
    movie.parent.mkdir(parents=True)
    candidate.parent.mkdir(parents=True)
    movie.write_bytes(MOVIE)
    os.chmod(movie, 0o640)
    os.utime(movie, ns=(1_700_000_000_000_000_000, 1_700_000_000_000_000_000))
    os.link(movie, candidate)
    return root, movie, candidate


def test_edit_root_names_replays_and_other_recordings(tmp_path: Path) -> None:
    assert delivery_offload.edit_root("123", root=tmp_path) == tmp_path / "replay-123"
    assert delivery_offload.edit_root("youtube-abc", root=tmp_path) == tmp_path / "youtube-abc"


def test_edit_root_refuses_paths(tmp_path: Path) -> None:
    with pytest.raises(ReplayError, match="Invalid recording name"):
        delivery_offload.edit_root("../x", root=tmp_path)


def test_load_without_a_record_is_empty(tmp_path: Path) -> None:
    assert delivery_offload.load(tmp_path) == []


def test_offload_verifies_the_readback_records_and_deletes(tmp_path: Path) -> None:
    root, movie, candidate = delivery(tmp_path)
    store = Store({"media/m.mp4": MOVIE})
    entry = delivery_offload.offload(
        root, role="delivery", path=movie, key="media/m.mp4", store=store,
        links=[candidate], clock=clock,
    )  # fmt: skip
    assert store.calls == ["head media/m.mp4", "digest media/m.mp4"]
    assert not movie.exists()
    assert not candidate.exists()
    assert entry.sha256 == hashlib.sha256(MOVIE).hexdigest()
    document = json.loads((root / "offload.json").read_text(encoding="utf-8"))
    assert document == {
        "version": 1,
        "files": [
            {
                "role": "delivery", "localPath": str(movie),
                "bucket": "transcript-viewer-media", "key": "media/m.mp4", "bytes": 21,
                "sha256": entry.sha256,
                "verification": {
                    "method": "r2-full-readback-sha256",
                    "verifiedAt": "2026-10-24T09:00:00+00:00",
                },
                "links": [str(candidate)], "evidence": {}, "mode": 0o640,
                "mtimeNs": 1_700_000_000_000_000_000, "fetchedAt": [],
            }
        ],
    }  # fmt: skip
    assert delivery_offload.load(root) == [entry]
    assert delivery_offload.offloaded(root, movie, "delivery") == entry
    assert delivery_offload.offloaded(root, movie, "original") is None


def test_offload_dry_run_keeps_everything(tmp_path: Path) -> None:
    root, movie, candidate = delivery(tmp_path)
    delivery_offload.offload(
        root, role="delivery", path=movie, key="media/m.mp4", store=Store({"media/m.mp4": MOVIE}),
        links=[candidate], dry_run=True, clock=clock,
    )  # fmt: skip
    assert movie.read_bytes() == MOVIE
    assert candidate.exists()
    assert not (root / "offload.json").exists()


def test_offload_keeps_the_file_when_the_readback_differs(tmp_path: Path) -> None:
    root, movie, candidate = delivery(tmp_path)
    store = Store({"media/m.mp4": MOVIE}, readback=b"delivered movie bytez")
    with pytest.raises(ReplayError, match="differs"):
        delivery_offload.offload(
            root, role="delivery", path=movie, key="media/m.mp4", store=store, links=[candidate]
        )
    assert movie.read_bytes() == MOVIE
    assert not (root / "offload.json").exists()


def test_offload_refuses_a_missing_or_resized_object(tmp_path: Path) -> None:
    root, movie, candidate = delivery(tmp_path)
    with pytest.raises(ReplayError, match="missing or of another size"):
        delivery_offload.offload(
            root, role="delivery", path=movie, key="media/m.mp4", store=Store({}),
            links=[candidate],
        )  # fmt: skip
    with pytest.raises(ReplayError, match="missing or of another size"):
        delivery_offload.offload(
            root, role="delivery", path=movie, key="media/m.mp4",
            store=Store({"media/m.mp4": b"short"}), links=[candidate],
        )  # fmt: skip


def test_offload_refuses_unnamed_hardlinks(tmp_path: Path) -> None:
    root, movie, _candidate = delivery(tmp_path)
    with pytest.raises(ReplayError, match="hardlinks the offload does not name"):
        delivery_offload.offload(
            root, role="delivery", path=movie, key="media/m.mp4",
            store=Store({"media/m.mp4": MOVIE}),
        )  # fmt: skip


def test_offload_refuses_a_link_of_another_file(tmp_path: Path) -> None:
    root, movie, candidate = delivery(tmp_path)
    other = tmp_path / "other.mp4"
    other.write_bytes(MOVIE)
    with pytest.raises(ReplayError, match="is not a hardlink"):
        delivery_offload.offload(
            root, role="delivery", path=movie, key="media/m.mp4",
            store=Store({"media/m.mp4": MOVIE}), links=[candidate, other],
        )  # fmt: skip


def test_offload_refuses_bad_roles_missing_files_and_repeats(tmp_path: Path) -> None:
    root, movie, candidate = delivery(tmp_path)
    store = Store({"media/m.mp4": MOVIE})
    with pytest.raises(ReplayError, match="Unknown offload role"):
        delivery_offload.offload(root, role="final", path=movie, key="media/m.mp4", store=store)
    with pytest.raises(ReplayError, match="Not a regular local file"):
        delivery_offload.offload(
            root, role="delivery", path=tmp_path / "none.mp4", key="media/m.mp4", store=store
        )
    delivery_offload.save(root, [Entry("delivery", str(movie), "media/m.mp4", 21, "a" * 64, "t")])
    with pytest.raises(ReplayError, match="already recorded"):
        delivery_offload.offload(
            root, role="delivery", path=movie, key="media/m.mp4", store=store, links=[candidate]
        )


def test_offload_pins_matching_pipeline_evidence(tmp_path: Path) -> None:
    root, movie, candidate = delivery(tmp_path)
    sha = hashlib.sha256(MOVIE).hexdigest()
    evidence = tmp_path / "pipeline-evidence.json"
    evidence.write_text(json.dumps({"movie": {"sha256": sha}}), encoding="utf-8")
    entry = delivery_offload.offload(
        root, role="delivery", path=movie, key="media/m.mp4", store=Store({"media/m.mp4": MOVIE}),
        links=[candidate], evidence=evidence, dry_run=True,
    )  # fmt: skip
    assert entry.evidence == {
        "path": str(evidence),
        "sha256": delivery_offload.sha256_of(evidence),
        "movieSha256": sha,
    }


def test_offload_refuses_evidence_of_another_movie(tmp_path: Path) -> None:
    root, movie, candidate = delivery(tmp_path)
    evidence = tmp_path / "pipeline-evidence.json"
    evidence.write_text(json.dumps({"movie": {"sha256": MOVIE_SHA}}), encoding="utf-8")
    with pytest.raises(ReplayError, match="another movie SHA-256"):
        delivery_offload.offload(
            root, role="delivery", path=movie, key="media/m.mp4",
            store=Store({"media/m.mp4": MOVIE}), links=[candidate], evidence=evidence,
        )  # fmt: skip


def offloaded(tmp_path: Path) -> tuple[Path, Path, Path]:
    root, movie, candidate = delivery(tmp_path)
    delivery_offload.offload(
        root, role="delivery", path=movie, key="media/m.mp4", store=Store({"media/m.mp4": MOVIE}),
        links=[candidate], clock=clock,
    )  # fmt: skip
    return root, movie, candidate


def test_fetch_restores_the_file_its_links_and_metadata(
    tmp_path: Path, roomy: list[tuple[list[Path], int]]
) -> None:
    root, movie, candidate = offloaded(tmp_path)
    restored = delivery_offload.fetch(
        root, role="delivery", store=Store({"media/m.mp4": MOVIE}), clock=clock
    )
    assert restored == [movie]
    assert movie.read_bytes() == MOVIE
    assert candidate.samefile(movie)
    assert movie.stat().st_mode & 0o777 == 0o640
    assert movie.stat().st_mtime_ns == 1_700_000_000_000_000_000
    assert roomy == [([movie.parent], 21)]
    assert sorted(path.name for path in movie.parent.iterdir()) == [movie.name]
    assert delivery_offload.load(root)[0].fetched_at == ["2026-10-24T09:00:00+00:00"]


def test_fetch_keeps_a_matching_file_and_only_restores_links(tmp_path: Path) -> None:
    root, movie, candidate = offloaded(tmp_path)
    movie.write_bytes(MOVIE)
    store = Store({"media/m.mp4": MOVIE})
    assert delivery_offload.fetch(root, role="delivery", store=store, path=movie) == [movie]
    assert store.calls == []
    assert candidate.samefile(movie)


def test_fetch_never_overwrites_a_different_file(tmp_path: Path) -> None:
    root, movie, _candidate = offloaded(tmp_path)
    movie.write_bytes(b"something else")
    with pytest.raises(ReplayError, match="A different file already exists"):
        delivery_offload.fetch(root, role="delivery", store=Store({"media/m.mp4": MOVIE}))
    assert movie.read_bytes() == b"something else"


def test_fetch_refuses_a_link_that_is_another_file(tmp_path: Path) -> None:
    root, movie, candidate = offloaded(tmp_path)
    candidate.write_bytes(b"other")
    with pytest.raises(ReplayError, match="is not the restored file"):
        delivery_offload.fetch(root, role="delivery", store=Store({"media/m.mp4": MOVIE}))
    assert movie.read_bytes() == MOVIE


def test_fetch_discards_a_mismatched_download(tmp_path: Path) -> None:
    root, movie, candidate = offloaded(tmp_path)
    store = Store({"media/m.mp4": MOVIE}, readback=b"delivered movie bytez")
    with pytest.raises(ReplayError, match="does not match its record"):
        delivery_offload.fetch(root, role="delivery", store=store)
    assert not movie.exists()
    assert not candidate.exists()
    assert list(movie.parent.iterdir()) == []


def test_fetch_needs_a_recorded_role(tmp_path: Path) -> None:
    root, _movie, _candidate = offloaded(tmp_path)
    with pytest.raises(ReplayError, match="No offloaded original recorded for replay-1"):
        delivery_offload.fetch(root, role="original", store=Store({}))


def test_load_refuses_unsupported_and_malformed_records(tmp_path: Path) -> None:
    (tmp_path / "offload.json").write_text('{"version": 2, "files": []}', encoding="utf-8")
    with pytest.raises(ReplayError, match="Unsupported offload record"):
        delivery_offload.load(tmp_path)
    (tmp_path / "offload.json").write_text(
        '{"version": 1, "files": [{"role": "delivery", "localPath": "/x", "key": "k",'
        ' "bytes": 1, "sha256": "short", "verification": {"method": "r2-full-readback-sha256",'
        ' "verifiedAt": "t"}, "links": [], "evidence": {}, "bucket": "b", "mode": 384,'
        ' "mtimeNs": 0, "fetchedAt": []}]}',
        encoding="utf-8",
    )
    with pytest.raises(ReplayError, match="Malformed offload record entry"):
        delivery_offload.load(tmp_path)


def test_load_refuses_missing_text_and_negative_numbers(tmp_path: Path) -> None:
    (tmp_path / "offload.json").write_text(
        '{"version": 1, "files": [{"role": "", "verification": {}}]}', encoding="utf-8"
    )
    with pytest.raises(ReplayError, match="needs role"):
        delivery_offload.load(tmp_path)
    (tmp_path / "offload.json").write_text(
        '{"version": 1, "files": [{"role": "delivery", "localPath": "/x", "key": "k",'
        ' "bytes": -1, "sha256": "' + "a" * 64 + '", "verification": {"method":'
        ' "r2-full-readback-sha256", "verifiedAt": "t"}, "links": [], "evidence": {},'
        ' "bucket": "b", "mode": 384, "mtimeNs": 0, "fetchedAt": []}]}',
        encoding="utf-8",
    )
    with pytest.raises(ReplayError, match="non-negative integer bytes"):
        delivery_offload.load(tmp_path)


def test_main_offloads_through_a_worker_session(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    root, movie, candidate = delivery(tmp_path)
    store = Store({"media/m.mp4": MOVIE})

    class Session:
        def __init__(self, bucket: str) -> None:
            assert bucket == "transcript-viewer-media"

        def __enter__(self) -> Store:
            return store

        def __exit__(self, *_exc: object) -> None:
            return None

    monkeypatch.setattr(r2_worker, "session", Session)
    monkeypatch.setattr(delivery_offload, "SYNC_STATE", tmp_path / "sync-state.json")
    monkeypatch.setattr(
        delivery_offload, "edit_root", lambda name: tmp_path / "edits" / f"replay-{name}"
    )
    code = delivery_offload.main(
        ["1", "--role", "delivery", "--path", str(movie), "--key", "media/m.mp4",
         "--link", str(candidate), "--dry-run"]
    )  # fmt: skip
    output = json.loads(capsys.readouterr().out)
    assert code == 0
    assert output["dryRun"] is True
    assert output["localPath"] == str(movie)
    assert movie.exists()
    assert not (root / "offload.json").exists()


def test_offloaded_matches_another_spelling_of_the_path(tmp_path: Path) -> None:
    root, movie, candidate = offloaded(tmp_path)
    (tmp_path / "alias").symlink_to(tmp_path / "store")
    spelled = tmp_path / "alias" / "delivery" / movie.name
    entry = delivery_offload.offloaded(root, spelled)
    assert entry is not None
    assert entry.local_path == str(movie)
    assert delivery_offload.offloaded(root, candidate) == entry


def test_offload_trusts_the_upload_hash_while_size_and_etag_match(tmp_path: Path) -> None:
    root, movie, candidate = delivery(tmp_path)
    store = Store({"media/m.mp4": MOVIE}, etag='"abc-1"')
    sent = delivery_offload.Uploaded(21, hashlib.sha256(MOVIE).hexdigest(), "abc-1")
    entry = delivery_offload.offload(
        root, role="delivery", path=movie, key="media/m.mp4", store=store,
        links=[candidate], sent=sent, dry_run=True,
    )  # fmt: skip
    assert store.calls == ["head media/m.mp4"]
    assert entry.method == "upload-sha256-size-etag"
    assert entry.document()["verification"] == {
        "method": "upload-sha256-size-etag",
        "verifiedAt": entry.verified_at,
    }


def test_offload_reads_back_when_the_etag_or_hash_differs(tmp_path: Path) -> None:
    root, movie, candidate = delivery(tmp_path)
    store = Store({"media/m.mp4": MOVIE}, etag="other-1")
    sent = delivery_offload.Uploaded(21, hashlib.sha256(MOVIE).hexdigest(), "abc-1")
    entry = delivery_offload.offload(
        root, role="delivery", path=movie, key="media/m.mp4", store=store,
        links=[candidate], sent=sent, clock=clock,
    )  # fmt: skip
    assert store.calls == ["head media/m.mp4", "digest media/m.mp4"]
    assert entry.method == "r2-full-readback-sha256"
    assert delivery_offload.load(root) == [entry]


def test_offload_refuses_a_file_changed_during_verification(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root, movie, candidate = delivery(tmp_path)

    class Touching(Store):
        def digest(self, key: str) -> tuple[int, str]:
            os.utime(movie, ns=(5, 5))
            return super().digest(key)

    with pytest.raises(ReplayError, match="changed while it was verified"):
        delivery_offload.offload(
            root, role="delivery", path=movie, key="media/m.mp4",
            store=Touching({"media/m.mp4": MOVIE}), links=[candidate],
        )  # fmt: skip
    assert movie.exists()


def test_uploaded_reads_the_sync_record(tmp_path: Path) -> None:
    state = tmp_path / "sync-state.json"
    assert delivery_offload.uploaded("media/m.mp4", state) is None
    state.write_text(
        '{"uploads": {"media/m.mp4": {"size": 21, "sha256": "s", "etag": "e-1"}}}',
        encoding="utf-8",
    )
    assert delivery_offload.uploaded("media/m.mp4", state) == delivery_offload.Uploaded(
        21, "s", "e-1"
    )
    assert delivery_offload.uploaded("media/other.mp4", state) is None
    state.write_text('{"uploads": {"media/m.mp4": {"size": "21"}}}', encoding="utf-8")
    with pytest.raises(ReplayError, match="Malformed upload record of media/m.mp4"):
        delivery_offload.uploaded("media/m.mp4", state)
