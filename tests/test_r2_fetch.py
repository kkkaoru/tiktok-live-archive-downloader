import hashlib
import json
from contextlib import AbstractContextManager
from datetime import UTC, datetime
from pathlib import Path

import pytest

from replay import delivery_offload, disk_budget, r2_fetch, r2_worker
from replay.core import ReplayError
from replay.delivery_offload import Entry

MOVIE = b"delivered movie bytes"


class Store:
    def head(self, key: str) -> r2_worker.RemoteObject | None:
        return r2_worker.RemoteObject(len(MOVIE), datetime(2026, 10, 24, tzinfo=UTC), {})

    def digest(self, key: str) -> tuple[int, str]:
        return len(MOVIE), hashlib.sha256(MOVIE).hexdigest()

    def get(self, key: str, destination: Path) -> int:
        destination.write_bytes(MOVIE)
        return len(MOVIE)


class Session(AbstractContextManager[delivery_offload.Store]):
    buckets: list[str] = []

    def __init__(self, bucket: str) -> None:
        Session.buckets.append(bucket)

    def __enter__(self) -> delivery_offload.Store:
        return Store()

    def __exit__(self, *_exc: object) -> None:
        return None


@pytest.fixture(autouse=True)
def roomy(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(disk_budget, "require_headroom", lambda _paths, _need: [])
    Session.buckets = []


def recorded(tmp_path: Path) -> tuple[Path, Path]:
    root = tmp_path / "replay-1"
    movie = tmp_path / "delivery" / "m.mp4"
    entry = Entry(
        "delivery", str(movie), "media/m.mp4", len(MOVIE), hashlib.sha256(MOVIE).hexdigest(), "t"
    )
    delivery_offload.save(root, [entry])
    return root, movie


def test_restore_if_offloaded_fetches_a_missing_recorded_file(tmp_path: Path) -> None:
    root, movie = recorded(tmp_path)
    assert r2_fetch.restore_if_offloaded(movie, root=root, session=Session) is True
    assert movie.read_bytes() == MOVIE
    assert Session.buckets == ["transcript-viewer-media"]


def test_restore_if_offloaded_leaves_present_and_unrecorded_files(tmp_path: Path) -> None:
    root, movie = recorded(tmp_path)
    assert (
        r2_fetch.restore_if_offloaded(tmp_path / "other.mp4", root=root, session=Session) is False
    )
    movie.parent.mkdir()
    movie.write_bytes(b"present")
    assert r2_fetch.restore_if_offloaded(movie, root=root, session=Session) is False
    assert Session.buckets == []


def test_main_fetches_a_role(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    root, movie = recorded(tmp_path)
    monkeypatch.setattr(delivery_offload, "edit_root", lambda _name: root)
    assert r2_fetch.main(["1", "--role", "delivery"], session=Session) == 0
    assert json.loads(capsys.readouterr().out) == {"restored": [str(movie)]}
    assert movie.read_bytes() == MOVIE


def test_main_downloads_verified_copies_into_a_folder_without_restoring(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    root, movie = recorded(tmp_path)
    into = tmp_path / "scratch"
    code = r2_fetch.main(
        ["1", "--role", "delivery", "--edit-root", str(root), "--into", str(into)], session=Session
    )
    assert code == 0
    assert json.loads(capsys.readouterr().out) == {"restored": [str(into / "m.mp4")]}
    assert (into / "m.mp4").read_bytes() == MOVIE
    assert not movie.exists()


def test_fetch_copy_needs_a_recorded_role(tmp_path: Path) -> None:
    root, _movie = recorded(tmp_path)
    with pytest.raises(ReplayError, match="No offloaded original recorded for replay-1"):
        r2_fetch.fetch_copy(root, role="original", into=tmp_path / "s", session=Session)
