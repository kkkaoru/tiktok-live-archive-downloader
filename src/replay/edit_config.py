"""Shared recording configuration validation, independent of delivery assessment."""

from __future__ import annotations

import hashlib
import json
import re
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

from replay.core import ReplayError
from replay.recordings import record
from replay.youtube_delivery import source_stem

CONFIG_NAME = "edit-config.json"
REQUIRED_STAGES = (
    "bgm_removal",
    "silence_cut",
    "source_caption_mask",
    "transcription",
    "dynamic_captions",
)
CONFIG_KEYS = frozenset(
    {"version", "replay_id", "source", "source_size", "engine", "executor_scope", "stages"}
)
# Optional in every config version: an explicit external directory that holds this
# recording's large deliveries and native evidence instead of the checkout.
STORAGE_KEY = "storage_root"
YOUTUBE_CONFIG_KEYS = frozenset(
    {
        "version",
        "video_id",
        "source",
        "source_size",
        "source_sha256",
        "publication",
        "publication_sha256",
        "engine",
        "executor_scope",
        "stages",
    }
)
MEDIA_SUFFIXES = frozenset({".mp4", ".mov", ".m4v"})
MAX_JSON_BYTES = 64 * 1024 * 1024


@dataclass(frozen=True)
class EditConfig:
    replay_id: str
    source: Path
    source_size: int
    engine: Path
    executor_scope: Path
    platform: Literal["tiktok", "youtube"] = "tiktok"
    storage_root: Path | None = None


def _read_json(path: Path, *, maximum_bytes: int = MAX_JSON_BYTES) -> object:
    if path.is_symlink() or not path.is_file():
        raise ReplayError(f"Missing evidence: {path.name}")
    if path.stat().st_size > maximum_bytes:
        raise ReplayError(f"Evidence exceeds the size limit: {path.name}")
    return json.loads(path.read_text(encoding="utf-8"))


def _replay_id(root: Path) -> str:
    match = re.fullmatch(r"replay-([0-9]{1,32})", root.name)
    if match is None:
        raise ReplayError("Expected an edits/replay-<id> directory")
    return match[1]


def native_media_path(path: Path) -> Path:
    """Resolve a media input and require a suffix that native AVFoundation tools open."""
    resolved = path.resolve(strict=True)
    if not resolved.is_file() or resolved.suffix.lower() not in MEDIA_SUFFIXES:
        raise ReplayError(f"Media input must resolve to a .mp4/.mov/.m4v file: {path.name}")
    return resolved


def storage_root(value: object, *, root: Path, source: Path) -> Path:
    """Validate an explicit external storage directory for one recording.

    It must be an existing real directory named after the recording, outside the
    account store and not containing or inside the original download directory.
    """
    if not isinstance(value, str) or not value or not Path(value).is_absolute():
        raise ReplayError("Storage root must be an absolute path")
    path = Path(value)
    if path.is_symlink() or not path.is_dir() or path.resolve() != path:
        raise ReplayError("Storage root must be an existing, unlinked directory")
    if path.name != root.name:
        raise ReplayError("Storage root must be named after its recording directory")
    private = root.parents[1] / "private"
    originals = source.resolve().parent
    if (
        path.is_relative_to(private.resolve())
        or path.is_relative_to(originals)
        or originals.is_relative_to(path)
    ):
        raise ReplayError("Storage root must stay outside account and original media storage")
    return path


def _storage(data: dict[str, object], *, root: Path, source: Path) -> Path | None:
    if STORAGE_KEY not in data:
        return None
    return storage_root(data[STORAGE_KEY], root=root, source=source)


def _digest(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def _youtube_inputs(
    root: Path, source: Path, publication: Path, *, reader: Callable[[Path], object] = _read_json
) -> tuple[str, Path]:
    match = re.fullmatch(r"youtube-([A-Za-z0-9_-]{11})", root.name)
    if match is None:
        raise ReplayError("Expected an edits/youtube-<id> directory")
    video_id = match[1]
    stem = source_stem(video_id, reader(publication))
    media = source.resolve(strict=True)
    if (
        not media.is_file()
        or media.suffix.lower() not in MEDIA_SUFFIXES | {".mkv", ".webm"}
        or media.stem != stem
        or publication.resolve().parent != media.parent
        or media.stat().st_size <= 0
    ):
        raise ReplayError("Expected a dated original beside its publication evidence")
    return video_id, media


def _load_youtube_config(
    root: Path,
    data: dict[str, object],
    *,
    reader: Callable[[Path], object] = _read_json,
    digest: Callable[[Path], str] = _digest,
) -> EditConfig:
    if set(data) - {STORAGE_KEY} != YOUTUBE_CONFIG_KEYS or type(data.get("version")) is not int:
        raise ReplayError("YouTube edit config has unknown, missing or unsupported fields")
    stages = record(data["stages"])
    if set(stages) != set(REQUIRED_STAGES) or any(value is not True for value in stages.values()):
        raise ReplayError("Edit config must enable every required stage")
    source, publication, size = data["source"], data["publication"], data["source_size"]
    engine, scope = data["engine"], data["executor_scope"]
    if (
        not isinstance(source, str)
        or not isinstance(publication, str)
        or type(size) is not int
        or not isinstance(engine, str)
        or not isinstance(scope, str)
    ):
        raise ReplayError("Invalid YouTube edit config input fields")
    video_id, media = _youtube_inputs(root, Path(source), Path(publication), reader=reader)
    if data["video_id"] != video_id:
        raise ReplayError("Edit config does not belong to this recording")
    if media.stat().st_size != size or digest(media) != data["source_sha256"]:
        raise ReplayError("YouTube edit config source changed")
    if digest(Path(publication)) != data["publication_sha256"]:
        raise ReplayError("YouTube publication evidence changed")
    if not Path(engine).is_dir() or not Path(scope).is_dir():
        raise ReplayError("Edit engine and Executor scope directories must exist")
    return EditConfig(
        video_id,
        media,
        size,
        Path(engine),
        Path(scope),
        platform="youtube",
        storage_root=_storage(data, root=root, source=media),
    )


def load_config(
    root: Path,
    *,
    reader: Callable[[Path], object] = _read_json,
    digest: Callable[[Path], str] = _digest,
) -> EditConfig:
    """Validate a per-recording config; every required stage stays enabled."""
    data = record(reader(root / CONFIG_NAME))
    if data.get("version") == 2:
        return _load_youtube_config(root, data, reader=reader, digest=digest)
    if set(data) - {STORAGE_KEY} != CONFIG_KEYS or data.get("version") != 1:
        raise ReplayError("Edit config has unknown, missing or unsupported fields")
    stages = data.get("stages")
    if stages != dict.fromkeys(REQUIRED_STAGES, True):
        raise ReplayError("Edit config must enable every required stage")
    replay_id, source, size = data["replay_id"], data["source"], data["source_size"]
    engine, scope = data["engine"], data["executor_scope"]
    if (
        not isinstance(replay_id, str)
        or replay_id != _replay_id(root)
        or not isinstance(source, str)
        or type(size) is not int
        or not isinstance(engine, str)
        or not isinstance(scope, str)
    ):
        raise ReplayError("Edit config does not belong to this recording")
    media = native_media_path(Path(source))
    if media.stat().st_size != size:
        raise ReplayError("Edit config source size changed")
    return EditConfig(
        replay_id,
        media,
        size,
        Path(engine),
        Path(scope),
        storage_root=_storage(data, root=root, source=media),
    )
