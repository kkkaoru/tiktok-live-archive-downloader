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


def test_rows_do_not_expose_creator_id_or_media_urls(tmp_path: Path) -> None:
    api = Mock()
    api.notices.return_value = [Notice(replay_id="1", anchor_id="2", user_type="1")]
    api.resolve.return_value = Recording(
        replay_id="1",
        title="Example",
        available=True,
        duration=10,
        media_url="https://v16m.tiktokcdn.com/a.m3u8?secret=1",
        start_time=1704067200,
    )
    assert recording_rows(api, anchor_id="2", jobs=tmp_path) == [
        {
            "replay_id": "1",
            "title": "Example",
            "available": True,
            "duration": 10,
            "start_time": 1704067200,
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
            "title": "Unavailable",
            "available": False,
            "duration": None,
            "start_time": None,
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


@pytest.mark.parametrize("replay_id", [None, "1"])
def test_partial_catalog_does_not_start_downloads(tmp_path: Path, replay_id: str | None) -> None:
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
            replay_id=replay_id,
        )
    api.resolve.assert_not_called()
    assert not (tmp_path / "output").exists()


def test_selected_replay_downloads_only_observed_match(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    api = Mock()
    api.notices.return_value = [
        Notice(replay_id="1", anchor_id="2", user_type="1"),
        Notice(replay_id="3", anchor_id="2", user_type="1"),
    ]
    operation = Mock(return_value=True)
    monkeypatch.setattr("replay.acquire.acquire_one", operation)
    options = DownloadOptions(store=tmp_path / "candidates", output=tmp_path / "output")
    assert acquire_all(api, anchor_id="2", options=options, replay_id="3") == DownloadSummary(
        saved=1, skipped=0, unavailable=0
    )
    operation.assert_called_once_with(
        api, notice=Notice(replay_id="3", anchor_id="2", user_type="1"), options=options
    )
    api.notices.assert_called_once_with(anchor_id="2")


def test_unobserved_replay_stops_before_resolution_or_output(tmp_path: Path) -> None:
    api = Mock()
    api.notices.return_value = [Notice(replay_id="1", anchor_id="2", user_type="1")]
    with pytest.raises(ReplayError, match="not in the authorized creator's notices"):
        acquire_all(
            api,
            anchor_id="2",
            options=DownloadOptions(store=tmp_path / "candidates", output=tmp_path / "output"),
            replay_id="3",
        )
    api.resolve.assert_not_called()
    assert not (tmp_path / "output").exists()


@pytest.mark.parametrize("replay_id", ["", "../1", "abc", "1" * 33])
def test_invalid_selected_id_stops_before_api(tmp_path: Path, replay_id: str) -> None:
    api = Mock()
    with pytest.raises(ReplayError, match="Invalid numeric"):
        acquire_all(
            api,
            anchor_id="2",
            options=DownloadOptions(store=tmp_path / "candidates", output=tmp_path / "output"),
            replay_id=replay_id,
        )
    api.notices.assert_not_called()
    assert not (tmp_path / "output").exists()


def test_selected_completed_replay_is_not_redownloaded(tmp_path: Path) -> None:
    output = tmp_path / "old.mp4"
    output.write_bytes(b"complete")
    register_completed(root=tmp_path / "jobs", replay_id="1", output=output)
    api = Mock()
    api.notices.return_value = [Notice(replay_id="1", anchor_id="2", user_type="1")]
    assert acquire_all(
        api,
        anchor_id="2",
        options=DownloadOptions(store=tmp_path / "candidates", output=tmp_path / "output"),
        replay_id="1",
    ) == DownloadSummary(saved=0, skipped=1, unavailable=0)
    api.resolve.assert_not_called()
    assert output.read_bytes() == b"complete"


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
