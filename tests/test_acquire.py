import fcntl
from collections.abc import Iterator
from pathlib import Path
from unittest.mock import Mock

import pytest

from replay.acquire import (
    DownloadOptions,
    DownloadSummary,
    acquire_all,
    acquire_one,
    recording_rows,
)
from replay.core import ReplayError
from replay.jobs import record_path, register_completed
from replay.recordings import Notice, Recording


def test_rows_do_not_expose_media_urls(tmp_path: Path) -> None:
    api = Mock()
    api.notices.return_value = [Notice(replay_id="1", anchor_id="2", user_type="1")]
    api.resolve.return_value = Recording(
        replay_id="1",
        title="Example",
        available=True,
        duration=10,
        media_url="https://v16m.tiktokcdn.com/a.m3u8?secret=1",
    )
    assert recording_rows(api, anchor_id="2", jobs=tmp_path) == [
        {
            "replay_id": "1",
            "anchor_id": "2",
            "title": "Example",
            "available": True,
            "duration": 10,
            "completed": False,
        }
    ]


def test_saved_but_unavailable_row(tmp_path: Path) -> None:
    output = tmp_path / "old.mp4"
    output.write_bytes(b"complete")
    register_completed(root=tmp_path / "jobs", replay_id="1", output=output)
    api = Mock()
    api.notices.return_value = [Notice(replay_id="1", anchor_id="2", user_type="1")]
    api.resolve.return_value = Recording(replay_id="1", title="Unavailable", available=False)
    assert recording_rows(api, anchor_id="2", jobs=tmp_path / "jobs") == [
        {
            "replay_id": "1",
            "anchor_id": "2",
            "title": "Unavailable",
            "available": False,
            "duration": None,
            "completed": True,
        }
    ]


def test_complete_skips_even_url_resolution(tmp_path: Path) -> None:
    output = tmp_path / "old.mp4"
    output.write_bytes(b"complete")
    register_completed(root=tmp_path / "jobs", replay_id="1", output=output)
    api = Mock()
    assert (
        acquire_one(
            api,
            notice=Notice(replay_id="1", anchor_id="2", user_type="1"),
            options=DownloadOptions(store=tmp_path / "candidates", output=tmp_path),
        )
        is False
    )
    api.resolve.assert_not_called()
    assert output.read_bytes() == b"complete"


def test_busy_skips_resolution(tmp_path: Path) -> None:
    lock = record_path(tmp_path / "jobs", "1").with_suffix(".lock")
    api = Mock()
    with lock.open("a+b") as stream:
        fcntl.flock(stream.fileno(), fcntl.LOCK_EX)
        assert (
            acquire_one(
                api,
                notice=Notice(replay_id="1", anchor_id="2", user_type="1"),
                options=DownloadOptions(store=tmp_path / "candidates", output=tmp_path),
            )
            is False
        )
    api.resolve.assert_not_called()


def test_batch_and_credential_isolation(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    output = tmp_path / "old.mp4"
    output.write_bytes(b"complete")
    register_completed(root=tmp_path / "jobs", replay_id="3", output=output)
    api = Mock()
    api.notices.return_value = [
        Notice(replay_id="1", anchor_id="2", user_type="1"),
        Notice(replay_id="2", anchor_id="2", user_type="1"),
        Notice(replay_id="3", anchor_id="2", user_type="1"),
    ]
    api.resolve.side_effect = [
        Recording(
            replay_id="1",
            title="Example",
            available=True,
            duration=10,
            media_url="https://v16m.tiktokcdn.com/a.m3u8",
            start_time=1704067200,
        ),
        Recording(replay_id="2", title="Expired", available=False),
    ]
    downloader = Mock()
    downloader.return_value.download.side_effect = lambda path: path.write_bytes(b"video")
    monkeypatch.setattr("replay.acquire.Downloader", downloader)
    summary = acquire_all(
        api,
        anchor_id="2",
        options=DownloadOptions(store=tmp_path / "candidates", output=tmp_path / "output"),
    )
    assert summary == DownloadSummary(saved=1, skipped=1, unavailable=1)
    assert (tmp_path / "output" / "2024-01-01_09-00-00_JST_replay-1.mp4").read_bytes() == b"video"
    assert api.resolve.call_count == 2
    assert downloader.call_args.kwargs["candidate"].headers == {}
    assert "cookie" not in downloader.call_args.kwargs["client"].headers
    assert list(downloader.call_args.kwargs["client"].cookies) == []


def test_partial_catalog_does_not_start_downloads(tmp_path: Path) -> None:
    def partial() -> Iterator[Notice]:
        yield Notice(replay_id="1", anchor_id="2", user_type="1")
        raise ReplayError("incomplete")

    api = Mock()
    api.notices.return_value = partial()
    with pytest.raises(ReplayError, match="incomplete"):
        acquire_all(
            api,
            anchor_id="2",
            options=DownloadOptions(store=tmp_path / "candidates", output=tmp_path / "output"),
        )
    api.resolve.assert_not_called()
    assert not (tmp_path / "output").exists()


@pytest.mark.parametrize("workers,max_bytes", [(0, 1), (17, 1), (1, 0)])
def test_invalid_options(tmp_path: Path, workers: int, max_bytes: int) -> None:
    with pytest.raises(ReplayError, match="Invalid worker"):
        acquire_all(
            Mock(),
            anchor_id="2",
            options=DownloadOptions(
                store=tmp_path / "candidates", output=tmp_path, workers=workers, max_bytes=max_bytes
            ),
        )
