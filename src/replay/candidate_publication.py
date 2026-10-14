"""Publish an explicitly unverified candidate without copying or encoding media.

The selected revision is a Viewer choice only. Required-stage completion records
are neither created nor weakened. Existing media and revisions are never replaced.
"""

from __future__ import annotations

import argparse
import fcntl
import json
import os
import re
import tempfile
from dataclasses import dataclass
from pathlib import Path

from replay.core import ReplayError
from replay.delivery_output_names import delivery_stem
from replay.edit_completion import EditConfig, load_config
from replay.fcp_caption_srt import srt_text
from replay.recordings import record
from replay.stages import file_digest

SELECTION_NAME = "candidate-selection.json"
MAX_SELECTION_BYTES = 64 * 1024


@dataclass(frozen=True)
class Publication:
    edit_root: Path
    library: Path
    movie: Path
    fcpxml: Path
    revision: str
    previous_selection_sha256: str | None = None


def _plain(path: Path) -> Path:
    absolute = path.absolute()
    if absolute != absolute.resolve():
        raise ReplayError("Candidate paths must not traverse symbolic links or parent segments")
    if any(part in {"private", "downloads"} for part in absolute.parts):
        raise ReplayError("Candidate paths must stay outside private and original download storage")
    return absolute


def _stem(config: EditConfig, root: Path) -> str:
    return delivery_stem(config.source, root.name)


def _write_new(path: Path, body: str) -> None:
    with path.open("x", encoding="utf-8") as stream:
        os.fchmod(stream.fileno(), 0o600)
        stream.write(body)
        stream.flush()
        os.fsync(stream.fileno())


def _json(value: object) -> str:
    return json.dumps(value, ensure_ascii=False, allow_nan=False, indent=2) + "\n"


def _previous(path: Path, expected: str | None) -> str | None:
    if not path.exists():
        if expected is not None:
            raise ReplayError("Expected previous candidate selection is missing")
        return None
    if path.is_symlink() or not path.is_file() or path.stat().st_size > MAX_SELECTION_BYTES:
        raise ReplayError("Existing candidate selection is not a bounded regular file")
    if expected is None or file_digest(path) != expected:
        raise ReplayError("Existing selection requires its exact previous SHA-256")
    return path.read_text(encoding="utf-8")


def _probe(path: Path, source: Path) -> None:
    if path.exists():
        if not path.is_file() or path.stat().st_size > 1_048_576:
            raise ReplayError("Source probe must be a bounded regular file")
        data = record(json.loads(path.read_text(encoding="utf-8")))
        if record(data.get("format")).get("filename") != str(source):
            raise ReplayError("Existing source probe belongs to a different original")


def publish(options: Publication) -> Path:
    """Create a new immutable candidate revision, then atomically select it for Viewer."""
    root, library, movie, fcpxml = map(
        _plain, (options.edit_root, options.library, options.movie, options.fcpxml)
    )
    config_sha = file_digest(root / "edit-config.json")
    config = load_config(root)
    folder = _plain(library / root.name)
    if not library.is_dir() or config.storage_root != folder:
        raise ReplayError("Explicit library must match the edit config storage root")
    if (
        not movie.is_file()
        or not movie.is_relative_to(library)
        or movie == config.source
        or movie.suffix.lower() not in {".mp4", ".m4v", ".mov"}
        or movie.stat().st_size <= 0
    ):
        raise ReplayError("Expected nonempty candidate media inside the explicit library")
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_-]{0,95}", options.revision):
        raise ReplayError("Invalid candidate revision identifier")
    stem = _stem(config, root)
    # Validate and bind all subtitle inputs before reserving any output directory.
    before = file_digest(fcpxml)
    subtitles = srt_text(fcpxml)
    if file_digest(fcpxml) != before:
        raise ReplayError("FCPXML changed during caption export")
    media_stat = movie.stat()
    revision = _plain(folder / "revisions" / options.revision)
    delivery = _plain(revision / "delivery")
    evidence = _plain(folder / "evidence")
    _probe(_plain(evidence / "source-probe.json"), config.source)
    evidence.mkdir(mode=0o700, parents=True, exist_ok=True)
    lock = _plain(evidence / ".candidate-publication.lock")
    with lock.open("a", encoding="utf-8") as stream:
        fcntl.flock(stream.fileno(), fcntl.LOCK_EX)
        return _publish_locked(
            options,
            folder=folder,
            delivery=delivery,
            movie=movie,
            stem=stem,
            subtitles=subtitles,
            fcpxml_sha256=before,
            config_sha256=config_sha,
            source=config.source,
            media_stat=media_stat,
        )


def _publish_locked(
    options: Publication,
    *,
    folder: Path,
    delivery: Path,
    movie: Path,
    stem: str,
    subtitles: str,
    fcpxml_sha256: str,
    config_sha256: str,
    source: Path,
    media_stat: os.stat_result,
) -> Path:
    selection = folder / "evidence" / SELECTION_NAME
    previous = _previous(selection, options.previous_selection_sha256)
    if file_digest(options.edit_root / "edit-config.json") != config_sha256:
        raise ReplayError("Edit config changed during publication")
    # The revision may already contain Compressor's master. Only its new delivery
    # directory is reserved; existing final products cannot be overwritten.
    delivery.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
    delivery.mkdir(mode=0o700)
    published_movie = delivery / f"{stem}{movie.suffix.lower()}"
    transcript = delivery / f"{stem}.srt"
    if movie.stat() != media_stat:
        raise ReplayError("Candidate media changed before publication")
    _write_new(transcript, subtitles)
    published_movie.symlink_to(os.path.relpath(movie, delivery))
    manifest: dict[str, object] = {
        "version": 1,
        "status": "unverified-candidate",
        "movie": published_movie.relative_to(folder).as_posix(),
        "movieSize": media_stat.st_size,
        "movieTarget": os.path.relpath(movie, folder),
        "transcript": transcript.relative_to(folder).as_posix(),
        "transcriptSha256": file_digest(transcript),
        "fcpxml": str(options.fcpxml.absolute()),
        "fcpxmlSha256": fcpxml_sha256,
        "editConfigSha256": config_sha256,
        "movieMtimeNanoseconds": media_stat.st_mtime_ns,
        "renderConsumptionVerified": False,
        "requiredStagesVerified": False,
    }
    content = _json(manifest)
    _write_new(delivery / "candidate.json", content)
    if previous is not None:
        _write_new(delivery / "previous-selection.json", previous)
    probe = _plain(folder / "evidence" / "source-probe.json")
    if not probe.exists():
        _write_new(probe, _json({"format": {"filename": str(source)}}))
    if (
        movie.stat() != media_stat
        or file_digest(options.fcpxml) != fcpxml_sha256
        or file_digest(options.edit_root / "edit-config.json") != config_sha256
    ):
        raise ReplayError("Candidate inputs changed before selection; partial revision retained")
    with tempfile.NamedTemporaryFile(
        mode="w",
        encoding="utf-8",
        dir=selection.parent,
        prefix=".candidate-selection-",
        delete=False,
    ) as stream:
        stream.write(content)
        stream.flush()
        os.fsync(stream.fileno())
        pending = Path(stream.name)
    os.replace(pending, selection)
    return selection


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fcpxml", required=True, type=Path)
    parser.add_argument("--srt-output", type=Path)
    parser.add_argument("--edit-root", type=Path)
    parser.add_argument("--library", type=Path)
    parser.add_argument("--movie", type=Path)
    parser.add_argument("--revision")
    parser.add_argument("--previous-selection-sha256")
    args = parser.parse_args(argv)
    try:
        if args.srt_output is not None:
            if any(
                value is not None
                for value in (
                    args.edit_root,
                    args.library,
                    args.movie,
                    args.revision,
                    args.previous_selection_sha256,
                )
            ):
                raise ReplayError("SRT-only export cannot change candidate selection")
            output = _plain(args.srt_output)
            _write_new(output, srt_text(_plain(args.fcpxml)))
        else:
            if any(
                value is None
                for value in (
                    args.edit_root,
                    args.library,
                    args.movie,
                    args.revision,
                )
            ):
                raise ReplayError("Publication requires edit root, library, movie and revision")
            output = publish(
                Publication(
                    args.edit_root,
                    args.library,
                    args.movie,
                    args.fcpxml,
                    args.revision,
                    args.previous_selection_sha256,
                )
            )
        print(output)
        return 0
    except (ReplayError, OSError, ValueError) as error:
        print(f"Candidate publication failed: {error}")
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
