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


def download_original(*, url: str, root: Path, ffmpeg_directory: Path) -> Path:
    """Lock by stable ID before metadata lookup; never reacquire a legacy original."""
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
        common = ["yt-dlp", "--ignore-config", "--js-runtimes", "node", "--no-playlist"]
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
        stem = source_stem(video_id, metadata)
        directory = root / stem
        directory.mkdir(mode=0o700)
        with (directory / "publication.json").open("x", encoding="utf-8") as evidence:
            data = record(metadata)
            json.dump(
                {key: data.get(key) for key in ("id", "release_timestamp", "timestamp")}, evidence
            )
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
    if len(sys.argv) != 4:
        raise SystemExit("Usage: python -m replay.youtube_delivery URL ROOT FFMPEG_DIRECTORY")
    try:
        download_original(
            url=sys.argv[1], root=Path(sys.argv[2]), ffmpeg_directory=Path(sys.argv[3])
        )
    except (ReplayError, OSError, subprocess.SubprocessError) as error:
        raise SystemExit(str(error)) from error


if __name__ == "__main__":
    main()
