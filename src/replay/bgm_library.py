"""A BGM library kept apart from TikTok/YouTube replays: audio, thumbnail and searchable facts.

Each track lives in `ROOT/<video id>/` as `<title> [<id>].m4a` (title, channel and artwork
embedded), `<title> [<id>].jpg` and `track.json`. The audio is 24-bit Apple Lossless: YouTube's
best audio stream (usually Opus) decoded once, so no second lossy generation is added and Final
Cut Pro, Logic Pro and AVFoundation all read it. `loudness` is the audio's mean level, measured
like a replay's per-second levels (`replay.excitement`), so TranscriptViewer sets a BGM under a
video from stored numbers instead of decoding audio. A track is either the video's own audio or,
for a talk stream, a looping BGM rebuilt locally (`replay.bgm_loop`) and marked `loop`. A track
is staged in a hidden directory and renamed into place only after its audio decodes, so a listed
track is always complete. Only an allowlist of metadata is stored; signed media URLs never are.
No cookies or login are used.
"""

from __future__ import annotations

import argparse
import fcntl
import json
import os
import shutil
import subprocess
import sys
from collections.abc import Callable, Sequence
from pathlib import Path

from replay.bgm_loop import opened
from replay.core import ReplayError
from replay.excitement import DEFAULT_CACHE, cached_levels, mean_level
from replay.recordings import record
from replay.youtube_delivery import source_stem

DEFAULT_ROOT = Path("edits/bgm-library")
# `{"version": 1, "root": "/absolute/library"}`: the tracks live there (an external SSD).
ROOT_CONFIG = Path("edits/bgm-library.json")
# The locked environment's yt-dlp, so the Viewer can run this module without a shell PATH.
YT_DLP_EXECUTABLE = Path(sys.executable).with_name("yt-dlp")
YT_DLP_FLAGS = ("--ignore-config", "--js-runtimes", "node", "--no-playlist")
METADATA_TIMEOUT_SECONDS = 120
DOWNLOAD_TIMEOUT_SECONDS = 1800
DECODE_TIMEOUT_SECONDS = 600
ENCODE_TIMEOUT_SECONDS = 1800
# 24 bits in 32-bit planar samples: decoded Opus or a float loop keeps its resolution.
LOSSLESS_CODEC = ("-c:a", "alac", "-sample_fmt", "s32p", "-bits_per_raw_sample", "24")
SOURCE_STEM = ".source"
SECONDS_PER_MINUTE = 60
LEVEL_DIGITS = 2

# Fills the staging directory; returns the metadata, audio, thumbnail and record overrides.
Stage = Callable[[Path], tuple[object, Path, Path, dict[str, object]]]


def video_id(url: str) -> str:
    """The ID of a canonical YouTube watch URL; anything else is refused."""
    identifier = url.rsplit("=", 1)[-1]
    if url not in (
        f"https://www.youtube.com/watch?v={identifier}",
        f"https://youtube.com/watch?v={identifier}",
    ):
        raise ReplayError(f"Expected a canonical YouTube watch URL: {url}")
    source_stem(identifier, {"id": identifier, "timestamp": 1})  # validates the ID characters
    return identifier


def _text(data: dict[str, object], key: str) -> str | None:
    value = data.get(key)
    return value if isinstance(value, str) and value else None


def _number(data: dict[str, object], key: str) -> float | None:
    value = data.get(key)
    return float(value) if isinstance(value, (int, float)) and not isinstance(value, bool) else None


def track_record(
    metadata: object, *, identifier: str, audio: Path, thumbnail: Path, loop: bool = False
) -> dict[str, object]:
    """The stored, allowlisted facts of one downloaded track."""
    data = record(metadata)
    title = _text(data, "title")
    if data.get("id") != identifier or title is None:
        raise ReplayError(f"{identifier}: metadata does not describe the requested video")
    return {
        "version": 1,
        "id": identifier,
        "title": title,
        "channel": _channel(data),
        "channelURL": _text(data, "channel_url") or _text(data, "uploader_url"),
        "url": f"https://www.youtube.com/watch?v={identifier}",
        "duration": _number(data, "duration"),
        "timestamp": _number(data, "timestamp"),
        "audio": audio.name,
        "thumbnail": thumbnail.name,
        "loop": loop,
    }


def _only(directory: Path, suffix: str) -> Path:
    found = sorted(path for path in directory.iterdir() if path.suffix.lower() == suffix)
    if len(found) != 1:
        raise ReplayError(f"Expected one {suffix} file in {directory.name}, found {len(found)}")
    return found[0]


def loudness(audio: Path, *, cache: Path = DEFAULT_CACHE) -> float | None:
    """Mean level (dBFS) of the track's audible seconds; None for a silent track."""
    level = mean_level(cached_levels(audio, cache=cache))
    return None if level is None else round(level, LEVEL_DIGITS)


def measure_library(root: Path, *, force: bool = False) -> list[str]:
    """Add `loudness` to every track record lacking it (or all with `force`)."""
    lines: list[str] = []
    for path in sorted(root.glob("*/track.json")):
        value = record(json.loads(path.read_text(encoding="utf-8")))
        audio = value.get("audio")
        if "loudness" in value and not force:
            lines.append(f"present {path.parent.name}")
            continue
        if not isinstance(audio, str) or Path(audio).name != audio:
            lines.append(f"failed {path.parent.name} (invalid audio name)")
            continue
        value["loudness"] = loudness(path.parent / audio)
        temporary = path.with_name(".track.json.partial")
        temporary.write_text(
            json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
        )
        os.replace(temporary, path)
        lines.append(f"measured {path.parent.name} {value['loudness']}")
    return lines


def decodes(audio: Path, ffmpeg: str) -> bool:
    """The whole audio stream decodes without a reported error."""
    result = subprocess.run(
        [ffmpeg, "-v", "error", "-nostdin", "-i", str(audio), "-map", "0:a:0", "-f", "null", "-"],
        capture_output=True, text=True, timeout=DECODE_TIMEOUT_SECONDS, check=False,
    )  # fmt: skip
    return result.returncode == 0 and not result.stderr.strip()


def _yt_dlp(url: str, identifier: str, arguments: Sequence[str], timeout: int) -> object:
    done = subprocess.run(
        [str(YT_DLP_EXECUTABLE), *YT_DLP_FLAGS, *arguments, url],
        capture_output=True, text=True, timeout=timeout, check=False,
    )  # fmt: skip
    if done.returncode != 0:
        raise ReplayError(f"{identifier}: yt-dlp failed: {done.stderr.strip()[-400:]}")
    try:
        metadata: object = json.loads(done.stdout)
    except json.JSONDecodeError as error:
        raise ReplayError(f"{identifier}: invalid yt-dlp metadata") from error
    return metadata


def _encode(
    audio: Path, thumbnail: Path, *, identifier: str, ffmpeg: str, tags: dict[str, str]
) -> Path:
    """Apple Lossless `<thumbnail stem>.m4a` of `audio`, with the artwork and `tags` embedded."""
    encoded = thumbnail.with_suffix(".m4a")
    metadata = [part for key, value in tags.items() for part in ("-metadata", f"{key}={value}")]
    done = subprocess.run(
        [ffmpeg, "-v", "error", "-nostdin", "-i", str(audio), "-i", str(thumbnail),
         "-map", "0:a:0", "-map", "1:v:0", *LOSSLESS_CODEC,
         "-c:v", "copy", "-disposition:v", "attached_pic", *metadata,
         "-movflags", "+faststart", str(encoded)],
        capture_output=True, text=True, timeout=ENCODE_TIMEOUT_SECONDS, check=False,
    )  # fmt: skip
    if done.returncode != 0:
        raise ReplayError(f"{identifier}: encoding failed: {done.stderr.strip()[-400:]}")
    return encoded


def _channel(data: dict[str, object]) -> str:
    return _text(data, "channel") or _text(data, "uploader") or ""


def library_root(config: Path = ROOT_CONFIG) -> Path:
    """The configured library root, else the checkout's `edits/bgm-library`."""
    if not config.exists():
        return DEFAULT_ROOT
    if config.is_symlink() or not config.is_file():
        raise ReplayError(f"{config} must be a plain file")
    data = record(json.loads(config.read_text(encoding="utf-8")))
    root = data.get("root")
    if data.get("version") != 1 or not isinstance(root, str) or not Path(root).is_absolute():
        raise ReplayError(f"{config} must give version 1 and an absolute root")
    return Path(root)


def _publish(
    identifier: str, *, root: Path, ffmpeg: str, stage: Stage, replace: bool = False
) -> tuple[str, str]:
    """Stage one track unless present, check its audio decodes, then rename it into place.

    With `replace`, a complete existing track is swapped out only once the new one is staged.
    """
    # Never parents: an unmounted volume must fail, not grow a stand-in on the boot disk.
    root.mkdir(mode=0o700, exist_ok=True)
    final = root / identifier
    with (root / f".bgm-{identifier}.lock").open("a", encoding="utf-8") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        present = (final / "track.json").is_file()
        if present and not replace:
            return identifier, "present"
        if final.exists() and not present:
            raise ReplayError(f"{final} exists without track.json; inspect it before retrying")
        staging = root / f".partial-{identifier}"
        shutil.rmtree(staging, ignore_errors=True)  # only ever our own interrupted attempt
        staging.mkdir(mode=0o700)
        metadata, audio, thumbnail, overrides = stage(staging)
        if not decodes(audio, ffmpeg):
            raise ReplayError(f"{identifier}: track audio does not decode")
        value = track_record(
            metadata, identifier=identifier, audio=audio, thumbnail=thumbnail,
            loop=overrides.pop("loop", False) is True,
        )  # fmt: skip
        value.update(overrides)
        value["loudness"] = loudness(audio)
        with (staging / "track.json").open("x", encoding="utf-8") as stream:
            json.dump(value, stream, ensure_ascii=False, indent=2)
            stream.write("\n")
        if not present:
            staging.rename(final)
            return identifier, "added"
        retired = root / f".replaced-{identifier}"
        shutil.rmtree(retired, ignore_errors=True)  # only ever our own interrupted swap
        final.rename(retired)
        staging.rename(final)
        shutil.rmtree(retired)
        return identifier, "replaced"


def add(url: str, *, root: Path, ffmpeg: str, replace: bool = False) -> tuple[str, str]:
    """Download one track unless present (or `replace`); return its ID and status."""
    identifier = video_id(url)

    def stage(staging: Path) -> tuple[object, Path, Path, dict[str, object]]:
        metadata = _yt_dlp(
            url, identifier,
            ["--no-simulate", "--dump-single-json", "--no-overwrites",
             "--ffmpeg-location", str(Path(ffmpeg).parent),
             "-f", "ba", "--write-thumbnail", "--convert-thumbnails", "jpg",
             "-o", str(staging / f"{SOURCE_STEM}.%(ext)s"),
             "-o", f"thumbnail:{staging / '%(title)s [%(id)s].%(ext)s'}"],
            DOWNLOAD_TIMEOUT_SECONDS,
        )  # fmt: skip
        sources = sorted(staging.glob(f"{SOURCE_STEM}.*"))
        if len(sources) != 1:
            raise ReplayError(f"Expected one downloaded audio stream, found {len(sources)}")
        thumbnail = _only(staging, ".jpg")
        data = record(metadata)
        encoded = _encode(
            sources[0], thumbnail, identifier=identifier, ffmpeg=ffmpeg,
            tags={"title": _text(data, "title") or identifier, "artist": _channel(data),
                  "comment": f"Audio of https://www.youtube.com/watch?v={identifier}"},
        )  # fmt: skip
        sources[0].unlink()
        return metadata, encoded, thumbnail, {}

    return _publish(identifier, root=root, ffmpeg=ffmpeg, stage=stage, replace=replace)


def add_loop(
    url: str, audio: Path, *, root: Path, ffmpeg: str, replace: bool = False
) -> tuple[str, str]:
    """Publish a BGM loop rebuilt from the video; only metadata and thumbnail are fetched."""
    identifier = video_id(url)
    with opened(audio) as source:
        seconds = source.frames / source.rate

    def stage(staging: Path) -> tuple[object, Path, Path, dict[str, object]]:
        metadata = _yt_dlp(
            url, identifier,
            ["--no-simulate", "--skip-download", "--dump-single-json", "--no-overwrites",
             "--ffmpeg-location", str(Path(ffmpeg).parent),
             "--write-thumbnail", "--convert-thumbnails", "jpg",
             "-o", str(staging / "%(title)s [%(id)s].%(ext)s")],
            METADATA_TIMEOUT_SECONDS,
        )  # fmt: skip
        thumbnail = _only(staging, ".jpg")
        data = record(metadata)
        title = f"{_text(data, 'title') or identifier}（BGM・"
        title += f"{round(seconds / SECONDS_PER_MINUTE)}分ループ）"
        encoded = _encode(
            audio, thumbnail, identifier=identifier, ffmpeg=ffmpeg,
            tags={"title": title, "artist": _channel(data),
                  "comment": f"BGM rebuilt from https://www.youtube.com/watch?v={identifier}"},
        )  # fmt: skip
        return metadata, encoded, thumbnail, {"title": title, "duration": seconds, "loop": True}

    return _publish(identifier, root=root, ffmpeg=ffmpeg, stage=stage, replace=replace)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    adding = commands.add_parser("add", help="download tracks from canonical YouTube URLs")
    adding.add_argument("urls", nargs="+")
    adding.add_argument("--root", type=Path, help="default: edits/bgm-library.json or edits")
    adding.add_argument("--ffmpeg", default=shutil.which("ffmpeg") or "ffmpeg")
    adding.add_argument("--replace", action="store_true", help="re-download existing tracks")
    looping = commands.add_parser("add-loop", help="publish a rebuilt BGM loop of a video")
    looping.add_argument("url")
    looping.add_argument("audio", type=Path, help="PCM WAV from replay.bgm_loop")
    looping.add_argument("--root", type=Path, help="default: edits/bgm-library.json or edits")
    looping.add_argument("--ffmpeg", default=shutil.which("ffmpeg") or "ffmpeg")
    looping.add_argument("--replace", action="store_true", help="swap out an existing track")
    measuring = commands.add_parser("loudness", help="store each track's mean level")
    measuring.add_argument("--root", type=Path, help="default: edits/bgm-library.json or edits")
    measuring.add_argument("--force", action="store_true", help="measure tracks that have one")
    args = parser.parse_args(argv)
    args.root = args.root or library_root()
    if args.command == "loudness":
        lines = measure_library(args.root, force=args.force)
        print("\n".join(lines), flush=True)
        return int(any(line.startswith("failed") for line in lines))
    if args.command == "add-loop":
        identifier, state = add_loop(
            args.url, args.audio, root=args.root, ffmpeg=args.ffmpeg, replace=args.replace
        )
        print(json.dumps({"url": args.url, "id": identifier, "status": state}), flush=True)
        return 0
    status = 0
    for url in args.urls:
        try:
            identifier, state = add(url, root=args.root, ffmpeg=args.ffmpeg, replace=args.replace)
            print(json.dumps({"url": url, "id": identifier, "status": state}), flush=True)
        except (ReplayError, OSError, subprocess.SubprocessError) as error:
            print(json.dumps({"url": url, "status": "failed", "error": str(error)}), flush=True)
            status = 1
    return status


if __name__ == "__main__":
    sys.exit(main())
