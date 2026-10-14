import json
import subprocess
import sys
from pathlib import Path

import pytest

from replay.core import ReplayError
from replay.youtube_delivery import download_original, main, new_edit_directory, source_stem


@pytest.mark.parametrize(
    "video_id,data",
    [
        ("../invalid", {}),
        ("abcdefghijk", {"id": "other"}),
        ("abcdefghijk", {"id": "abcdefghijk", "is_live": True}),
        ("abcdefghijk", {"id": "abcdefghijk"}),
        ("abcdefghijk", {"id": "abcdefghijk", "timestamp": True}),
        ("abcdefghijk", {"id": "abcdefghijk", "timestamp": -1}),
        ("abcdefghijk", {"id": "abcdefghijk", "timestamp": float("nan")}),
        ("abcdefghijk", {"id": "abcdefghijk", "timestamp": 1e100}),
    ],
)
def test_invalid_publication(video_id: str, data: dict[str, object]) -> None:
    with pytest.raises(ReplayError):
        source_stem(video_id, data)


def test_release_time_wins_over_upload_and_edits_are_unique(tmp_path: Path) -> None:
    data = {"id": "abcdefghijk", "release_timestamp": 1704067200, "timestamp": 1}
    assert source_stem("abcdefghijk", data) == "2024-01-01_09-00-00_JST_youtube-abcdefghijk"
    first = new_edit_directory(tmp_path, video_id="abcdefghijk", metadata=data)
    second = new_edit_directory(tmp_path, video_id="abcdefghijk", metadata=data)
    assert first != second
    assert first.name.startswith("2024-01-01_09-00-00_JST_youtube-abcdefghijk-edit-")
    assert first.is_dir() and second.is_dir()
    assert first.stat().st_mode & 0o777 == 0o700


def test_invalid_edit_has_no_side_effects(tmp_path: Path) -> None:
    with pytest.raises(ReplayError):
        new_edit_directory(tmp_path / "missing", video_id="abcdefghijk", metadata={})
    assert not (tmp_path / "missing").exists()


def test_download_contract(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    calls: list[list[str]] = []

    def run(args: list[str], **kwargs: object) -> subprocess.CompletedProcess[str]:
        calls.append(args)
        return subprocess.CompletedProcess(args, 0, '{"id":"abcdefghijk","timestamp":1704067200}')

    monkeypatch.setattr(subprocess, "run", run)
    directory = download_original(
        url="https://www.youtube.com/watch?v=abcdefghijk",
        root=tmp_path,
        ffmpeg_directory=Path("/ffmpeg"),
    )
    assert directory.name == "2024-01-01_09-00-00_JST_youtube-abcdefghijk"
    assert len(calls) == 2
    assert calls[0] == [
        "yt-dlp",
        "--ignore-config",
        "--js-runtimes",
        "node",
        "--no-playlist",
        "--skip-download",
        "--dump-single-json",
        "https://www.youtube.com/watch?v=abcdefghijk",
    ]
    assert calls[1][-2] == str(directory / "2024-01-01_09-00-00_JST_youtube-abcdefghijk.%(ext)s")
    assert "--no-overwrites" in calls[1]
    assert calls[1][calls[1].index("--merge-output-format") + 1] == "mp4"
    assert calls[1][calls[1].index("--remux-video") + 1] == "mp4"
    assert calls[1][calls[1].index("-f") + 1] == ("bv*+ba/b")
    assert "mkv" not in calls[1]
    assert json.loads((directory / "publication.json").read_text()) == {
        "id": "abcdefghijk",
        "release_timestamp": None,
        "timestamp": 1704067200,
    }
    with pytest.raises(ReplayError, match="already exists"):
        download_original(
            url="https://www.youtube.com/watch?v=abcdefghijk",
            root=tmp_path,
            ffmpeg_directory=Path("/ffmpeg"),
        )
    assert len(calls) == 2


@pytest.mark.parametrize("extension", ["mkv", "mp4"])
def test_legacy_original_stops_before_network(tmp_path: Path, extension: str) -> None:
    (tmp_path / f"abcdefghijk.{extension}").write_bytes(b"original")
    with pytest.raises(ReplayError, match="already exists"):
        download_original(
            url="https://youtube.com/watch?v=abcdefghijk",
            root=tmp_path,
            ffmpeg_directory=Path("/ffmpeg"),
        )


def test_bad_url_stops_before_lock(tmp_path: Path) -> None:
    with pytest.raises(ReplayError, match="canonical"):
        download_original(
            url="https://example.com/watch?v=abcdefghijk",
            root=tmp_path,
            ffmpeg_directory=Path("/ffmpeg"),
        )
    assert list(tmp_path.iterdir()) == []


def test_bad_json_retains_only_lock(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    def run(args: list[str], **kwargs: object) -> subprocess.CompletedProcess[str]:
        return subprocess.CompletedProcess(args, 0, "invalid-json")

    monkeypatch.setattr(subprocess, "run", run)
    with pytest.raises(ReplayError, match="JSON"):
        download_original(
            url="https://youtube.com/watch?v=abcdefghijk",
            root=tmp_path,
            ffmpeg_directory=Path("/ffmpeg"),
        )
    assert list(tmp_path.glob("*_JST*")) == []


def test_main_usage(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(sys, "argv", ["youtube_delivery"])
    with pytest.raises(SystemExit, match="Usage"):
        main()


def test_main_failure(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.setattr(sys, "argv", ["youtube_delivery", "bad-url", str(tmp_path), "/ffmpeg"])
    with pytest.raises(SystemExit, match="canonical"):
        main()


def test_main_success(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    def download(*, url: str, root: Path, ffmpeg_directory: Path) -> Path:
        return root

    monkeypatch.setattr("replay.youtube_delivery.download_original", download)
    monkeypatch.setattr(sys, "argv", ["youtube_delivery", "url", str(tmp_path), "/ffmpeg"])
    main()
