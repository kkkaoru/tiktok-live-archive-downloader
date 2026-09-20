import fcntl
import json
import os
from pathlib import Path
from unittest.mock import Mock

import pytest

from replay.acquire import DownloadOptions, acquire_one
from replay.core import ReplayError
from replay.jobs import already_complete, register_completed, rename_completed
from replay.recordings import (
    Notice,
    Recording,
    broadcast_datetime,
    recording_filename,
    recording_from_json,
)


def test_filename_uses_jst_not_machine_timezone(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("TZ", "Pacific/Honolulu")
    replay = Recording(
        replay_id="123", title="Ignored / title", available=True, start_time=1704067200
    )
    assert recording_filename(replay) == "2024-01-01_09-00-00_JST_replay-123.mp4"
    assert broadcast_datetime(1704067200).isoformat() == "2024-01-01T09:00:00+09:00"


def test_filenames_sort_across_jst_new_year_and_disambiguate_ids() -> None:
    newer = Recording(replay_id="123", title="", available=True, start_time=1704034800)
    older = Recording(replay_id="123", title="", available=True, start_time=1704034799)
    same_time = Recording(replay_id="456", title="", available=True, start_time=1704034800)
    assert sorted(
        [recording_filename(newer), recording_filename(older), recording_filename(same_time)]
    ) == [
        "2023-12-31_23-59-59_JST_replay-123.mp4",
        "2024-01-01_00-00-00_JST_replay-123.mp4",
        "2024-01-01_00-00-00_JST_replay-456.mp4",
    ]


@pytest.mark.parametrize(
    "start",
    [
        None,
        0,
        -1,
        True,
        "1704067200",
        1704067200.0,
        float("nan"),
        float("inf"),
        1704067200000,
        253402300799,
        10**400,
    ],
)
def test_bad_or_missing_start_never_produces_a_guessed_filename(start: object) -> None:
    with pytest.raises(ReplayError, match="start time"):
        replay = recording_from_json(
            {
                "replays": [
                    {
                        "id": "123",
                        "title": "Example",
                        "available": True,
                        "m3u8_url": "https://v16m.tiktokcdn.com/a.m3u8",
                        "hls_video_meta_info": {"duration": 10},
                        "start_time": start,
                    }
                ]
            },
            replay_id="123",
        )
        recording_filename(replay)


def test_filename_rejects_invalid_id() -> None:
    with pytest.raises(ReplayError, match="numeric"):
        recording_filename(
            Recording(replay_id="../bad", title="", available=True, start_time=1704067200)
        )


def test_no_start_time_stops_before_media_or_job_publication(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    api = Mock()
    api.resolve.return_value = Recording(
        replay_id="1", title="", available=True, media_url="https://v16m.tiktokcdn.com/a.m3u8"
    )
    downloader = Mock()
    monkeypatch.setattr("replay.acquire.Downloader", downloader)
    with pytest.raises(ReplayError, match="start time"):
        acquire_one(
            api,
            notice=Notice(replay_id="1", anchor_id="2", user_type="1"),
            options=DownloadOptions(store=tmp_path / "candidates", output=tmp_path),
        )
    downloader.assert_not_called()
    assert not (tmp_path / "jobs/1.json").exists()
    assert list(tmp_path.glob("*.mp4")) == []


def test_timestamped_output_collision_is_not_overwritten(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    output = tmp_path / "2024-01-01_09-00-00_JST_replay-1.mp4"
    output.write_bytes(b"existing")
    api = Mock()
    api.resolve.return_value = Recording(
        replay_id="1",
        title="",
        available=True,
        media_url="https://v16m.tiktokcdn.com/a.m3u8",
        start_time=1704067200,
    )
    downloader = Mock()
    monkeypatch.setattr("replay.acquire.Downloader", downloader)
    with pytest.raises(ReplayError, match="not registered"):
        acquire_one(
            api,
            notice=Notice(replay_id="1", anchor_id="2", user_type="1"),
            options=DownloadOptions(store=tmp_path / "candidates", output=tmp_path),
        )
    downloader.assert_not_called()
    assert output.read_bytes() == b"existing"


def test_rename_preserves_file_identity_and_deduplication(tmp_path: Path) -> None:
    source = tmp_path / "old.mp4"
    destination = tmp_path / "new.mp4"
    source.write_bytes(b"video")
    os.link(source, tmp_path / "identity.mp4")
    register_completed(root=tmp_path / "jobs", replay_id="1", output=source)
    assert rename_completed(root=tmp_path / "jobs", replay_id="1", destination=destination) is True
    assert not source.exists()
    assert destination.read_bytes() == b"video"
    assert destination.samefile(tmp_path / "identity.mp4")
    assert already_complete(tmp_path / "jobs/1.json") is True
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
    assert rename_completed(root=tmp_path / "jobs", replay_id="1", destination=destination) is False


def test_rename_refuses_existing_destination(tmp_path: Path) -> None:
    source = tmp_path / "old.mp4"
    destination = tmp_path / "new.mp4"
    source.write_bytes(b"original")
    destination.write_bytes(b"unrelated")
    register_completed(root=tmp_path / "jobs", replay_id="1", output=source)
    with pytest.raises(FileExistsError):
        rename_completed(root=tmp_path / "jobs", replay_id="1", destination=destination)
    assert source.read_bytes() == b"original"
    assert destination.read_bytes() == b"unrelated"
    assert already_complete(tmp_path / "jobs/1.json") is True


def test_rename_refuses_dangling_destination_symlink(tmp_path: Path) -> None:
    source = tmp_path / "old.mp4"
    destination = tmp_path / "new.mp4"
    source.write_bytes(b"original")
    destination.symlink_to("absent")
    register_completed(root=tmp_path / "jobs", replay_id="1", output=source)
    with pytest.raises(FileExistsError):
        rename_completed(root=tmp_path / "jobs", replay_id="1", destination=destination)
    assert source.read_bytes() == b"original"
    assert destination.is_symlink()


def test_rename_refuses_another_directory(tmp_path: Path) -> None:
    source = tmp_path / "old.mp4"
    source.write_bytes(b"original")
    register_completed(root=tmp_path / "jobs", replay_id="1", output=source)
    with pytest.raises(ReplayError, match="same directory"):
        rename_completed(
            root=tmp_path / "jobs", replay_id="1", destination=tmp_path / "jobs/new.mp4"
        )
    assert source.read_bytes() == b"original"


def test_rename_refuses_source_symlink(tmp_path: Path) -> None:
    source = tmp_path / "old.mp4"
    source.write_bytes(b"original")
    link = tmp_path / "link.mp4"
    link.symlink_to(source)
    (tmp_path / "1.json").write_text(
        json.dumps({"status": "complete", "output": str(link), "size": 8}), encoding="utf-8"
    )
    with pytest.raises(ReplayError, match="regular file"):
        rename_completed(root=tmp_path, replay_id="1", destination=tmp_path / "new.mp4")
    assert source.read_bytes() == b"original"


def test_rename_without_completion_is_rejected(tmp_path: Path) -> None:
    with pytest.raises(ReplayError, match="completed"):
        rename_completed(root=tmp_path, replay_id="1", destination=tmp_path / "new.mp4")


def test_rename_respects_lock(tmp_path: Path) -> None:
    with (tmp_path / "1.lock").open("a+b") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        with pytest.raises(BlockingIOError):
            rename_completed(root=tmp_path, replay_id="1", destination=tmp_path / "new.mp4")


def test_record_write_failure_keeps_original_data_and_record(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    source = tmp_path / "old.mp4"
    destination = tmp_path / "new.mp4"
    source.write_bytes(b"original")
    register_completed(root=tmp_path / "jobs", replay_id="1", output=source)
    monkeypatch.setattr("replay.jobs.write_record", Mock(side_effect=OSError("disk")))
    with pytest.raises(OSError, match="disk"):
        rename_completed(root=tmp_path / "jobs", replay_id="1", destination=destination)
    assert source.read_bytes() == b"original"
    assert destination.samefile(source)
    assert already_complete(tmp_path / "jobs/1.json") is True


@pytest.mark.parametrize("data", [[], {"output": 1}])
def test_corrupted_record_after_validation_is_rejected(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, data: object
) -> None:
    (tmp_path / "1.json").write_text(json.dumps(data), encoding="utf-8")
    monkeypatch.setattr("replay.jobs.already_complete", lambda path: True)
    with pytest.raises(ReplayError, match="Invalid replay job"):
        rename_completed(root=tmp_path, replay_id="1", destination=tmp_path / "new.mp4")
