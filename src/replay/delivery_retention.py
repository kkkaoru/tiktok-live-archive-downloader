"""Retire a delivery's regenerable render intermediates once its delivery is accepted.

After `replay.delivery_render.publish` registers a dated delivery, the alpha movies, native
parts, joined video, A/V master, selected voice track and Compressor output stay on the
startup disk (`work`) and the external SSD (`storage`) although nothing needs them. Targets
come only from producer declarations (Compressor job `output`, workflow stage `outputs` and
native `render.response.json`); nothing is found by file-name pattern. They are written to a
SHA-pinned disposition file and unlinked by `replay.leaf_retirement`, which rechecks
identity, canonical parents and open handles per row. The default is a dry run.

Files an FCPXML timeline references are kept, as are JSON, receipts, models, the dated
delivery, its transcript, verification report and Viewer links. A receipt is written after
a fully applied retirement; `is_retired` lets unpinned launchers skip the render pipeline,
which would otherwise regenerate the removed media.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import stat
import sys
from collections.abc import Iterable, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from urllib.parse import unquote, urlparse
from xml.etree import ElementTree as ET

from replay import delivery_render, disk_budget, edit_completion, leaf_retirement
from replay.core import ReplayError
from replay.recordings import record

RECEIPT_NAME = "retention-receipt.json"
RETENTION_DIRECTORY = "retention"
MEDIA_SUFFIXES = frozenset({".mov", ".mp4", ".m4v", ".mkv", ".wav", ".m4a", ".aac"})
WORKFLOW_NAMES = ("join-workflow.json", "final-mux-workflow.json", "audio-workflow.json")
JOB_FILES = ("alpha-jobs.json", "final-job.json")
MAX_JSON_BYTES = 256 * 1024 * 1024


@dataclass(frozen=True, slots=True)
class Candidate:
    path: Path
    reason: str


@dataclass(frozen=True, slots=True)
class Outcome:
    mode: str
    eligible: int
    retired: int
    logical_bytes: int
    held: dict[str, int]
    skipped: dict[str, str]
    free_before: dict[str, int]
    free_after: dict[str, int]


def _read(path: Path) -> object:
    if path.stat().st_size > MAX_JSON_BYTES:
        raise ReplayError(f"JSON exceeds its size limit: {path.name}")
    return json.loads(path.read_text(encoding="utf-8"))


def _strings(rows: object, key: str) -> list[str]:
    if not isinstance(rows, list):
        return []
    values = (record(row).get(key) for row in rows)
    return [value for value in values if isinstance(value, str)]


def _variant(requested: Path) -> Path:
    """The Compressor output actually written: the request, or its sole same-stem sibling."""
    if requested.is_file():
        return requested
    siblings = [
        path
        for path in requested.parent.glob(f"{requested.stem}.*")
        if path.is_file() and not path.name.startswith(".")
    ]
    return siblings[0] if len(siblings) == 1 else requested


def _job_outputs(path: Path) -> list[Path]:
    jobs = record(_read(path)).get("jobs")
    return [_variant(Path(value)) for value in _strings(jobs, "output")]


def _workflow_outputs(path: Path) -> list[Path]:
    stages = record(_read(path)).get("stages")
    outputs = (record(stage).get("outputs") for stage in stages) if isinstance(stages, list) else ()
    return [Path(value) for listed in outputs if isinstance(listed, list) for value in listed]


def _native_parts(config: delivery_render.Config) -> list[Path]:
    paths: list[Path] = []
    for request, _frames in delivery_render.rendered_parts(config):
        response = record(_read(request.with_name("render.response.json")))
        paths.append(delivery_render._path(response.get("outputPath")))
    return paths


def declared(config: delivery_render.Config) -> list[Candidate]:
    """Every media output a producer declared, with the file that declared it."""
    found: list[Candidate] = []
    for name in JOB_FILES:
        if (config.work / name).is_file():
            found += [Candidate(p, name) for p in _job_outputs(config.work / name)]
    for name in WORKFLOW_NAMES:
        if (config.work / name).is_file():
            found += [Candidate(p, name) for p in _workflow_outputs(config.work / name)]
    found += [Candidate(p, "render.response.json") for p in _native_parts(config)]
    return found


def delivery_movie(config: delivery_render.Config) -> Path:
    directory = config.delivery_root / "revisions" / "native-v1" / "delivery"
    return directory / f"{delivery_render.delivery_stem(config)}.mp4"


def require_accepted(config: delivery_render.Config) -> Path:
    """The dated delivery, only when it was verified, transcribed and registered."""
    movie = delivery_movie(config)
    if not movie.is_file() or not movie.with_suffix(".srt").is_file():
        raise ReplayError("Retention requires the dated delivery and its transcript")
    report = record(_read(movie.with_name(f"{movie.stem}.verification.json")))
    audio = report.get("audioPackets")
    if (
        report.get("framesMatchCutPlan") is not True
        or report.get("fullDecodePassed") is not True
        or type(audio) is not int
        or audio <= 0
    ):
        raise ReplayError("Retention requires a passed delivery media verification")
    failures = edit_completion.assess_delivery(movie, root=config.viewer_root)
    if failures:
        raise ReplayError("Retention requires every edit stage: " + ", ".join(sorted(failures)))
    linked = config.viewer_root / "revisions" / "native-v1" / "delivery" / movie.name
    if not linked.is_symlink() or linked.resolve() != movie.resolve():
        raise ReplayError("Retention requires the registered Viewer link")
    return movie


def fcpxml_references(timelines: Iterable[Path]) -> set[Path]:
    """Local media paths named by `src` attributes of the given FCPXML files."""
    referenced: set[Path] = set()
    for timeline in timelines:
        for element in ET.parse(timeline).getroot().iter():
            parsed = urlparse(element.get("src", ""))
            if parsed.scheme == "file" and parsed.path:
                referenced.add(Path(unquote(parsed.path)))
    return referenced


def _protected(config: delivery_render.Config) -> tuple[Path, ...]:
    return (config.delivery_root, config.viewer_root, config.source)


def _row(path: Path) -> dict[str, object]:
    info = os.lstat(path)
    return {
        "path": str(path),
        "kind": "file",
        "dev": info.st_dev,
        "inode": info.st_ino,
        "nlink": info.st_nlink,
        "bytes": info.st_size,
        "mtimeNs": info.st_mtime_ns,
        "ctimeNs": info.st_ctime_ns,
        "classification": leaf_retirement.ELIGIBLE,
    }


def _skip_reason(path: Path, kept: set[Path], protected: Sequence[Path]) -> str | None:
    if path.suffix.lower() not in MEDIA_SUFFIXES:
        return "not a media output"
    if path in kept or path.resolve() in kept:
        return "referenced by an FCPXML timeline"
    if any(path.is_relative_to(root) for root in protected):
        return "inside a delivery or Viewer root"
    try:
        info = os.lstat(path)
    except FileNotFoundError:
        return "already absent"
    return None if stat.S_ISREG(info.st_mode) else "not a regular file"


def plan(
    config: delivery_render.Config, timelines: Sequence[Path] = ()
) -> tuple[list[dict[str, object]], dict[str, str]]:
    """Disposition rows for every retirable file and the reason each other one is kept."""
    kept = fcpxml_references(timelines)
    protected = _protected(config)
    rows: dict[str, dict[str, object]] = {}
    skipped: dict[str, str] = {}
    for candidate in declared(config):
        reason = _skip_reason(candidate.path, kept, protected)
        if reason is not None:
            skipped[str(candidate.path)] = reason
        else:
            rows[str(candidate.path)] = _row(candidate.path)
    return list(rows.values()), skipped


def _write_dispositions(directory: Path, rows: list[dict[str, object]]) -> tuple[Path, str]:
    body = json.dumps({"rows": rows}, indent=2, sort_keys=True)
    digest = hashlib.sha256(body.encode("utf-8")).hexdigest()
    directory.mkdir(mode=0o700, parents=True, exist_ok=True)
    path = directory / f"dispositions-{digest[:16]}.json"
    if not path.exists():
        with path.open("x", encoding="utf-8") as stream:
            stream.write(body)
    return path, leaf_retirement.sha256_of(path)


def _free(config: delivery_render.Config) -> dict[str, int]:
    return {
        "work": disk_budget.measure(config.work).free,
        "storage": disk_budget.measure(config.storage).free,
    }


def is_retired(config_path: Path) -> bool:
    """True once a retention receipt exists, so launchers must not rerun rendering."""
    if not config_path.is_file():
        return False
    try:
        raw = record(_read(config_path))
    except (ReplayError, ValueError):
        return False  # A malformed configuration carries no receipt; its owner reports it.
    work = raw.get("work")
    return isinstance(work, str) and (Path(work) / RETENTION_DIRECTORY / RECEIPT_NAME).is_file()


def retire(config_path: Path, timelines: Sequence[Path] = (), *, apply: bool = False) -> Outcome:
    config = delivery_render.load_config(config_path)
    require_accepted(config)
    rows, skipped = plan(config, timelines)
    directory = config.work / RETENTION_DIRECTORY
    dispositions, digest = _write_dispositions(directory, rows)
    before = _free(config)
    summary = leaf_retirement.retire(
        dispositions,
        digest,
        (config.work, config.storage),
        directory / "retirement.log",
        apply=apply,
    )
    outcome = Outcome(
        "apply" if apply else "dry-run",
        summary.eligible,
        summary.retired,
        summary.logical_bytes,
        summary.held,
        skipped,
        before,
        _free(config),
    )
    if apply and summary.retired == summary.eligible:
        receipt = directory / RECEIPT_NAME
        if not receipt.exists():
            with receipt.open("x", encoding="utf-8") as stream:
                json.dump({**asdict(outcome), "dispositions": str(dispositions)}, stream, indent=2)
    return outcome


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("config", type=Path)
    parser.add_argument("--fcpxml", action="append", type=Path, default=[])
    parser.add_argument("--apply", action="store_true")
    args = parser.parse_args(argv)
    try:
        outcome = retire(args.config.absolute(), args.fcpxml, apply=args.apply)
    except (ReplayError, OSError, ValueError, ET.ParseError) as error:
        print(f"DELIVERY_RETENTION_ERROR {error}", file=sys.stderr)
        return 1
    print(json.dumps(asdict(outcome), indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
