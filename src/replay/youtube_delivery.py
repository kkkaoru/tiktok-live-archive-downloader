"""Source-dated, non-overwriting YouTube originals and unique edit directories."""

import fcntl
import json
import math
import subprocess
import sys
import tempfile
from datetime import datetime, timedelta, timezone
from pathlib import Path

from replay.core import ReplayError
from replay.recordings import record

JST = timezone(timedelta(hours=9))
# Browsers whose signed-in session yt-dlp may read (members-only videos); never a cookie file.
BROWSERS = frozenset({"brave", "chrome", "chromium", "edge", "firefox", "safari"})


def source_stem(video_id: str, metadata: object) -> str:
    """Use publication time, never IDs, file times or a date-only approximation."""
    if len(video_id) != 11 or any(
        c not in "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_-"
        for c in video_id
    ):
        raise ReplayError("Invalid YouTube video ID")
    data = record(metadata)
    if data.get("id") != video_id or data.get("is_live") is True:
        raise ReplayError("YouTube metadata does not describe the finished requested video")
    stamp = data.get("release_timestamp")
    if stamp is None:
        stamp = data.get("timestamp")
    if (
        isinstance(stamp, bool)
        or not isinstance(stamp, (int, float))
        or not math.isfinite(stamp)
        or stamp <= 0
    ):
        raise ReplayError("Missing verified YouTube publication timestamp")
    try:
        date = datetime.fromtimestamp(stamp, JST)
    except (ValueError, OverflowError, OSError) as error:
        raise ReplayError("Invalid YouTube publication timestamp") from error
    return f"{date:%Y-%m-%d_%H-%M-%S}_JST_youtube-{video_id}"


def new_edit_directory(root: Path, *, video_id: str, metadata: object) -> Path:
    """Atomically reserve a unique dated revision; use its name as the file stem."""
    stem = source_stem(video_id, metadata)
    root.mkdir(parents=True, exist_ok=True, mode=0o700)
    return Path(tempfile.mkdtemp(prefix=f"{stem}-edit-", dir=root))


def _yt_dlp(cookies_from_browser: str | None) -> list[str]:
    if cookies_from_browser is not None and cookies_from_browser not in BROWSERS:
        raise ReplayError("Unsupported browser for signed-in YouTube access")
    common = ["yt-dlp", "--ignore-config", "--js-runtimes", "node", "--no-playlist"]
    if cookies_from_browser is not None:
        common += ["--cookies-from-browser", cookies_from_browser]
    return common


def _metadata(common: list[str], url: str) -> object:
    result = subprocess.run(
        [*common, "--skip-download", "--dump-single-json", url],
        capture_output=True,
        text=True,
        check=True,
        timeout=120,
    )
    try:
        metadata: object = json.loads(result.stdout)
    except json.JSONDecodeError as error:
        raise ReplayError("Invalid YouTube metadata JSON") from error
    return metadata


def channel_record(metadata: object) -> dict[str, object]:
    data = record(metadata)
    channel = {key: data.get(key) for key in ("channel_id", "channel", "uploader_id")}
    if not isinstance(channel["channel_id"], str) or not channel["channel_id"]:
        raise ReplayError("YouTube metadata has no channel ID")
    return channel


def write_channel(directory: Path, channel: dict[str, object]) -> None:
    """Record the uploading channel beside the original; never replace a different record."""
    path = directory / "channel.json"
    if path.exists():
        if json.loads(path.read_text(encoding="utf-8")) != channel:
            raise ReplayError("channel.json already records a different channel")
        return
    with path.open("x", encoding="utf-8") as evidence:
        json.dump(channel, evidence, ensure_ascii=False)


def backfill_channel(directory: Path, *, cookies_from_browser: str | None = None) -> None:
    """Fetch metadata only for an existing original directory and add its channel.json."""
    video_id = directory.name.rsplit("_JST_youtube-", 1)[-1]
    source_stem(video_id, {"id": video_id, "timestamp": 1})
    if not (directory / "publication.json").is_file():
        raise ReplayError("Not a YouTube original directory")
    common = _yt_dlp(cookies_from_browser)
    with (directory.parent / f".youtube-{video_id}.lock").open("a", encoding="utf-8") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        metadata = _metadata(common, f"https://www.youtube.com/watch?v={video_id}")
        if source_stem(video_id, metadata) != directory.name:
            raise ReplayError("YouTube metadata does not match the original directory")
        write_channel(directory, channel_record(metadata))


def download_original(
    *, url: str, root: Path, ffmpeg_directory: Path, cookies_from_browser: str | None = None
) -> Path:
    """Lock by stable ID before metadata lookup; never reacquire a legacy original.

    ``cookies_from_browser`` (an authorized, signed-in browser) lets yt-dlp read that session for
    a members-only video; the cookies never pass through this process or its arguments.
    """
    common = _yt_dlp(cookies_from_browser)
    video_id = url.rsplit("=", 1)[-1]
    if url not in (
        f"https://www.youtube.com/watch?v={video_id}",
        f"https://youtube.com/watch?v={video_id}",
    ):
        raise ReplayError("Expected a canonical YouTube watch URL")
    # Validate the ID before using it in any filesystem path.
    source_stem(video_id, {"id": video_id, "timestamp": 1})
    with (root / f".youtube-{video_id}.lock").open("a", encoding="utf-8") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        if any(
            (root / f"{video_id}.{extension}").exists() for extension in ("mp4", "mkv")
        ) or tuple(root.glob(f"*_JST_youtube-{video_id}")):
            raise ReplayError(
                "Original already exists; verify or migrate it, do not download again"
            )
        metadata = _metadata(common, url)
        stem = source_stem(video_id, metadata)
        channel = channel_record(metadata)
        directory = root / stem
        directory.mkdir(mode=0o700)
        with (directory / "publication.json").open("x", encoding="utf-8") as evidence:
            data = record(metadata)
            json.dump(
                {key: data.get(key) for key in ("id", "release_timestamp", "timestamp")}, evidence
            )
        write_channel(directory, channel)
        subprocess.run(
            [
                *common,
                "--ffmpeg-location",
                str(ffmpeg_directory),
                "--no-overwrites",
                "--match-filter",
                "!is_live",
                "-f",
                "bv*+ba/b",
                "--merge-output-format",
                "mp4",
                "--remux-video",
                "mp4",
                "-o",
                str(directory / f"{stem}.%(ext)s"),
                url,
            ],
            check=True,
        )
        return directory


def main() -> None:
    if sys.argv[1:2] == ["channel"] and len(sys.argv) in (3, 4):
        try:
            backfill_channel(
                Path(sys.argv[2]), cookies_from_browser=sys.argv[3] if len(sys.argv) == 4 else None
            )
        except (ReplayError, OSError, ValueError, subprocess.SubprocessError) as error:
            raise SystemExit(str(error)) from error
        return
    if len(sys.argv) not in (4, 5):
        raise SystemExit(
            "Usage: python -m replay.youtube_delivery URL ROOT FFMPEG_DIRECTORY [BROWSER]"
            " | channel ORIGINAL_DIRECTORY [BROWSER]"
        )
    try:
        download_original(
            url=sys.argv[1],
            root=Path(sys.argv[2]),
            ffmpeg_directory=Path(sys.argv[3]),
            cookies_from_browser=sys.argv[4] if len(sys.argv) == 5 else None,
        )
    except (ReplayError, OSError, subprocess.SubprocessError) as error:
        raise SystemExit(str(error)) from error


if __name__ == "__main__":
    main()
