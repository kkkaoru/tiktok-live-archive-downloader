"""Retire a completed recording's heavy intermediates, so finished recordings stop growing disk use.

``recording_pipeline.run`` calls ``retire`` once every step of a run has finished, the summary
step's completion gate passed and the web viewer holds the delivery. It removes an explicit
manifest built from the step receipts, never a glob over a work tree:

- ``cleaned-source``: the cleanup's media directory (assembled clean source and chunk media);
- ``comment-clean``: a YouTube recording's comment-cleaned source movie;
- ``band-tiles``: the cleanup chunks' audit movies (their receipts and configs stay);
- ``vocals``: the separated voice ``vocals.wav``;
- ``tempo``: the paging's tempo audio;
- ``moti``: the caption parts' Motion projects (``parts.json`` and ``jobs.json`` stay).

Receipts, configurations, hashes, the delivery and its SRT, summaries, clips and every evidence
file stay. An entry is refused, never removed, when the evidence a completion gate reads names it
or a file inside it, when it holds the delivery's inode, when it is a symbolic link, or when it
lies outside the recording's own work and media roots. Every removal and refusal is appended to
``steps/logs/retirement.jsonl``; ``steps/retired-intermediates.json`` keeps every retired entry
with the step that produced it and the steps that read it, so a later run that needs one forces
its producer again (``forced_producers``) instead of trusting a receipt whose output is gone.

``retainIntermediates`` keeps everything; ``retirementHold`` names recordings kept for now.
"""

import json
import os
import time
from collections.abc import Iterable, Iterator, Sequence
from dataclasses import dataclass
from pathlib import Path

from replay.core import ReplayError
from replay.recording_steps import Pipeline, read_receipt, remove
from replay.recording_viewer_steps import delivery, require_uploaded
from replay.recordings import record

RECORD = "retired-intermediates.json"
LOG = "retirement.jsonl"
CLEANUP_MEDIA = "source-caption-cleanup"
COMMENT_CLEAN = "source-comment-clean"


@dataclass(frozen=True, slots=True)
class Entry:
    """One retirable intermediate: what it is, which step wrote it and which steps read it."""

    category: str
    path: Path
    producer: str
    consumers: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class Settings:
    """The driver's retirement options (DEFAULTS ``retainIntermediates``/``retirementHold``)."""

    retain: bool = False
    hold: tuple[str, ...] = ()


def outputs(pipeline: Pipeline, step: str) -> dict[str, object]:
    path = pipeline.steps / f"{step}.json"
    return record(read_receipt(pipeline, step).get("outputs")) if path.is_file() else {}


def pinned(value: object) -> Path | None:
    path = record(value).get("path") if isinstance(value, dict) else None
    return Path(path) if isinstance(path, str) else None


def cleanup_work(pipeline: Pipeline) -> Path | None:
    """The cleanup's work directory, named by the description its receipt pins."""
    description = pinned(outputs(pipeline, "cleanup").get("description"))
    if description is None or not description.is_file():
        return None
    work = record(json.loads(description.read_text(encoding="utf-8"))).get("work")
    return Path(work) if isinstance(work, str) else None


def manifest(pipeline: Pipeline) -> list[Entry]:
    """The recording's retirable intermediates, resolved from its receipts."""
    entries: list[Entry] = []
    cleaned = outputs(pipeline, "cleanup")
    if isinstance(cleaned.get("chunkMedia"), str):
        media = Path(str(cleaned["chunkMedia"])).parent
        entries.append(Entry("cleaned-source", media, "cleanup", ("cleanup", "compose")))
    elif "commentClean" in cleaned and isinstance(cleaned.get("cleanSource"), str):
        movie = Path(str(cleaned["cleanSource"]))
        entries.append(Entry("comment-clean", movie, "cleanup", ("compose",)))
    work = cleanup_work(pipeline)
    if work is not None:
        audits = sorted((work / "state").glob("chunk-*/attempt-*/audit/band-tiles.mov"))
        entries += [Entry("band-tiles", path, "cleanup", ("cleanup",)) for path in audits]
    vocals = pinned(outputs(pipeline, "separation").get("vocals"))
    if vocals is not None:
        entries.append(Entry("vocals", vocals, "separation", ("asr", "cut-plan", "paging")))
    tempo = pipeline.work / "paging-v1" / "prepared" / "tempo" / "candidate.wav"
    entries.append(Entry("tempo", tempo, "paging", ("onsets", "compose", "finish")))
    moti = pipeline.work / "caption-parts-v1" / "moti"
    entries.append(Entry("moti", moti, "caption-parts", ("caption-render",)))
    return entries


def paths_in(value: object) -> Iterator[Path]:
    """Every absolute path string inside a JSON value."""
    if isinstance(value, str) and value.startswith("/"):
        yield Path(value)
    elif isinstance(value, dict):
        for item in value.values():
            yield from paths_in(item)
    elif isinstance(value, list):
        for item in value:
            yield from paths_in(item)


def evidence_files(pipeline: Pipeline, storage: Path) -> list[Path]:
    """The files a completion gate of this recording reads its bindings from."""
    files = [pipeline.path_of("editRoot") / "edit-config.json"]
    files += sorted((storage / "revisions").glob("*/metadata/*.json"))
    description = pinned(outputs(pipeline, "cleanup").get("description"))
    files += [description] if description is not None else []
    files += sorted((pipeline.work / "caption-absence-v1").glob("*.json"))
    return [path for path in files if path.is_file()]


def named(files: Iterable[Path]) -> set[Path]:
    found: set[Path] = set()
    for path in files:
        try:
            found.update(paths_in(json.loads(path.read_text(encoding="utf-8"))))
        except ValueError as error:
            raise ReplayError(f"Unreadable evidence {path}; nothing is retired") from error
    return found


def files_of(path: Path) -> Iterator[Path]:
    if path.is_file():
        yield path
        return
    for base, _directories, names in os.walk(path):
        yield from (Path(base, name) for name in names)


def freed_bytes(path: Path) -> int:
    """Bytes removing ``path`` frees: inodes whose every link lies inside it."""
    seen: dict[tuple[int, int], tuple[int, int, int]] = {}
    for item in files_of(path):
        if item.is_symlink():
            continue
        status = item.stat()
        key = (status.st_dev, status.st_ino)
        count, size, links = seen.get(key, (0, status.st_size, status.st_nlink))
        seen[key] = (count + 1, size, links)
    return sum(size for count, size, links in seen.values() if count == links)


def refusal(
    entry: Entry, *, evidence: set[Path], movie: tuple[int, int], roots: Sequence[Path]
) -> str | None:
    """Why ``entry`` must stay, or None when it may be removed."""
    path = entry.path
    if path.is_symlink():
        return "a symbolic link"
    if not any(path.is_relative_to(root) and path != root for root in roots):
        return "outside the recording's work and media roots"
    cited = sorted(str(item) for item in evidence if item == path or item.is_relative_to(path))
    if cited:
        return f"named by completion evidence: {cited[0]}"
    for item in files_of(path):
        status = item.lstat()
        if (status.st_dev, status.st_ino) == movie:
            return "holds the delivery"
    return None


def append_log(pipeline: Pipeline, row: dict[str, object]) -> None:
    pipeline.logs.mkdir(mode=0o700, parents=True, exist_ok=True)
    with (pipeline.logs / LOG).open("a", encoding="utf-8") as stream:
        stream.write(json.dumps({"time": time.time(), **row}, ensure_ascii=False) + "\n")


def recorded(pipeline: Pipeline) -> list[Entry]:
    """Every intermediate retired from this recording so far."""
    path = pipeline.steps / RECORD
    if not path.is_file():
        return []
    rows = record(json.loads(path.read_text(encoding="utf-8"))).get("retired")
    entries: list[Entry] = []
    for row in map(record, rows if isinstance(rows, list) else []):
        consumers = row.get("consumers")
        names = tuple(str(name) for name in consumers) if isinstance(consumers, list) else ()
        entries.append(
            Entry(str(row["category"]), Path(str(row["path"])), str(row["producer"]), names)
        )
    return entries


def save_record(pipeline: Pipeline, entries: Sequence[Entry], settings: Settings) -> None:
    rows = [
        {"category": e.category, "path": str(e.path), "producer": e.producer,
         "consumers": list(e.consumers)}
        for e in entries
    ]  # fmt: skip
    value = {
        "version": 1,
        "retired": rows,
        "settings": {"retainIntermediates": settings.retain, "retirementHold": list(settings.hold)},
    }
    target = pipeline.steps / RECORD
    staged = target.with_name(f".{target.name}.partial")
    staged.write_text(json.dumps(value, indent=1) + "\n", encoding="utf-8")
    os.replace(staged, target)


def ready(pipeline: Pipeline) -> str | None:
    """Why retirement must wait, or None once the gate passed and the viewer holds the movie."""
    gate = outputs(pipeline, "summary").get("gate")
    if not isinstance(gate, dict) or gate.get("complete") is not True:
        return "the summary step's completion gate has not passed"
    if not (pipeline.steps / "upload.json").is_file():
        return "the delivery has not been uploaded"
    movie = delivery(pipeline)
    try:
        require_uploaded(movie.stem, movie)
    except (ReplayError, OSError, ValueError) as error:
        return f"the web viewer does not hold the delivery ({error})"
    return None


def retire(pipeline: Pipeline, settings: Settings) -> dict[str, object]:
    """Remove the recording's retirable intermediates; return what was removed or kept."""
    name = pipeline.path_of("editRoot").name
    if settings.retain or name in settings.hold:
        reason = "retainIntermediates" if settings.retain else "retirementHold"
        append_log(pipeline, {"retained": reason})
        return {"retained": reason}
    waiting = ready(pipeline)
    if waiting is not None:
        append_log(pipeline, {"waiting": waiting})
        return {"waiting": waiting}
    movie = delivery(pipeline)
    storage = movie.parents[3]
    status = movie.stat()
    evidence = named(evidence_files(pipeline, storage))
    work = cleanup_work(pipeline)
    roots = [
        pipeline.work, *([work] if work is not None else []),
        storage.parent / CLEANUP_MEDIA, storage.parent / COMMENT_CLEAN,
    ]  # fmt: skip
    retired = recorded(pipeline)
    removed: list[dict[str, object]] = []
    for entry in manifest(pipeline):
        if not entry.path.exists() and not entry.path.is_symlink():
            continue
        why = refusal(entry, evidence=evidence, movie=(status.st_dev, status.st_ino), roots=roots)
        if why is not None:
            append_log(pipeline, {"kept": str(entry.path), "category": entry.category, "why": why})
            continue
        freed = freed_bytes(entry.path)
        remove(entry.path)
        row = {"path": str(entry.path), "bytes": freed, "reason": entry.category}
        append_log(pipeline, {"removed": row})
        removed.append(row)
        retired = [*(item for item in retired if item.path != entry.path), entry]
        save_record(pipeline, retired, settings)
    return {"removed": removed, "bytes": sum(int(str(row["bytes"])) for row in removed)}


def forced_producers(pipeline: Pipeline, pending: set[str]) -> list[tuple[str, str, Path]]:
    """(producer, consumer, path) for retired outputs a step about to run would read.

    Only producers whose receipt still stands are returned: forcing one removes it, so a driver
    that forces these and asks again stops once every needed intermediate will be produced anew.
    """
    found: list[tuple[str, str, Path]] = []
    for entry in recorded(pipeline):
        if entry.path.exists() or not (pipeline.steps / f"{entry.producer}.json").is_file():
            continue
        found += [(entry.producer, name, entry.path) for name in entry.consumers if name in pending]
    return found
