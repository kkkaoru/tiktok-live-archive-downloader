"""Resume bounded current-pixel chunks under one run-local source fingerprint."""

import argparse
import json
import os
import stat
import subprocess
from dataclasses import dataclass
from fractions import Fraction
from itertools import pairwise
from pathlib import Path

from replay import head_image_collection as collection
from replay import tracking_observations as storage
from replay.artifact_json import _pin
from replay.core import ReplayError
from replay.fcp_visible_arc import VisibleArc
from replay.head_image_plan import MAX_CELLS, _fraction
from replay.recordings import record
from replay.stages import _uncached_file_digest as file_digest
from replay.stages import execution_lock

MAX_CHUNKS = 2048
MAX_TOTAL_CELLS = MAX_CHUNKS * MAX_CELLS
MAX_INPUTS = 65_536
MANIFEST_BYTES = 16 * 1024 * 1024
SUMMARY_BYTES = 64 * 1024 * 1024
MAX_RUNS = 4096
BOUNDARIES = frozenset({"first", "budget", "source-cut", "source-gap", "zoom"})
WORK_ROOT = Path(__file__).resolve().parents[2] / "edits" / "work"
CODE_FILES = (
    "head_image_chunks.py",
    "artifact_json.py",
    "foreground_evidence.py",
    "input_hash_cache.py",
    "vertical_captions.py",
)


@dataclass(frozen=True, slots=True)
class FileStamp:
    device: int
    inode: int
    size: int
    modified_ns: int
    changed_ns: int

    def encoded(self) -> dict[str, int]:
        return {
            "device": self.device,
            "inode": self.inode,
            "size": self.size,
            "modifiedNS": self.modified_ns,
            "changedNS": self.changed_ns,
        }


@dataclass(frozen=True, slots=True)
class Chunk:
    job: collection.ImageJob
    boundary: str
    reuse: Path | None


@dataclass(frozen=True, slots=True)
class Manifest:
    config: Path
    chunks: tuple[Chunk, ...]
    inputs: dict[str, str]
    ranges: tuple[tuple[Fraction, Fraction], ...]

    @property
    def source(self) -> Path:
        return self.chunks[0].job.source

    @property
    def frames(self) -> int:
        return sum(len(chunk.job.cells) for chunk in self.chunks)


def _stamp(path: Path) -> FileStamp:
    info = path.lstat()
    if not stat.S_ISREG(info.st_mode) or path.resolve() != path:
        raise ReplayError("Chunk inputs must remain canonical regular files")
    return FileStamp(info.st_dev, info.st_ino, info.st_size, info.st_mtime_ns, info.st_ctime_ns)


def _ranges(value: object) -> tuple[tuple[Fraction, Fraction], ...]:
    if not isinstance(value, list) or not 1 <= len(value) <= MAX_CHUNKS:
        raise ReplayError("Expected bounded explicit source coverage ranges")
    if any(set(record(row)) != {"start", "end"} for row in value):
        raise ReplayError("Source coverage ranges require exact start and end clocks")
    result = tuple(
        (_fraction(record(row).get("start")), _fraction(record(row).get("end"))) for row in value
    )
    if any(not 0 <= left < right <= 86400 for left, right in result) or any(
        end >= left for (_start, end), (left, _end) in pairwise(result)
    ):
        raise ReplayError("Source coverage ranges must be ordered, separate and positive")
    return result


def _validate_coverage(
    chunks: tuple[Chunk, ...], ranges: tuple[tuple[Fraction, Fraction], ...]
) -> None:
    actual: list[tuple[Fraction, Fraction]] = []
    previous: Fraction | None = None
    total = 0
    for index, chunk in enumerate(chunks):
        cells = chunk.job.cells
        if any(left.end != right.start for left, right in pairwise(cells)):
            raise ReplayError("A chunk must contain complete consecutive actual source cells")
        start, end = cells[0].start, cells[-1].end
        if index == 0:
            if chunk.boundary != "first":
                raise ReplayError("First chunk requires a first boundary")
        elif previous is None or start < previous or chunk.boundary == "first":
            raise ReplayError("Chunk clocks overlap or repeat an initial boundary")
        elif (start > previous) != (chunk.boundary == "source-gap"):
            raise ReplayError("Source gaps must be explicit; other boundaries cannot hide holes")
        if actual and actual[-1][1] == start:
            actual[-1] = (actual[-1][0], end)
        else:
            actual.append((start, end))
        total += len(cells)
        previous = end
    if tuple(actual) != ranges or total > MAX_TOTAL_CELLS:
        raise ReplayError("Actual chunks do not exactly cover the declared source ranges")


def _merge_inputs(inputs: dict[str, str], additions: dict[str, str]) -> None:
    if any(name in inputs and inputs[name] != digest for name, digest in additions.items()):
        raise ReplayError("Conflicting chunk dependency fingerprints")
    inputs.update(additions)
    if len(inputs) > MAX_INPUTS:
        raise ReplayError("Chunk dependency count exceeds its budget")


def load_manifest(config: Path) -> Manifest:
    """Keep the existing 128-cell and 16-batch limits in each immutable chunk."""
    config = storage._path(str(config.absolute()))
    data = record(json.loads(storage._read(config, MANIFEST_BYTES)))
    rows = data.get("chunks")
    if (
        set(data) != {"version", "ranges", "chunks"}
        or type(data["version"]) is not int
        or data["version"] != 1
        or not isinstance(rows, list)
        or not 1 <= len(rows) <= MAX_CHUNKS
    ):
        raise ReplayError("Invalid bounded current-image chunk manifest")
    chunks: list[Chunk] = []
    inputs = {str(config): file_digest(config)}
    inputs.update(
        {
            str(path): file_digest(path)
            for path in (Path(__file__).with_name(name) for name in CODE_FILES)
        }
    )
    identity: tuple[Path, str, Path, str, Path, str, int, int] | None = None
    for row in rows:
        item = record(row)
        boundary = item.get("boundary")
        if (
            set(item) != {"config", "boundary", "reuse"}
            or not isinstance(boundary, str)
            or boundary not in BOUNDARIES
        ):
            raise ReplayError("Expected explicit pinned chunk and boundary")
        path = _pin(item["config"])
        job = collection.load_job(path)
        current = (
            job.source,
            job.inputs[str(job.source)],
            job.native,
            job.inputs[str(job.native)],
            job.sandbox,
            job.inputs[str(job.sandbox)],
            job.width,
            job.height,
        )
        if identity is not None and current != identity:
            raise ReplayError("Chunks must share exact source, native, sandbox and canvas pins")
        identity = current
        reuse = None if item["reuse"] is None else _pin(item["reuse"])
        if reuse is not None and reuse.name != "provenance.json":
            raise ReplayError("Reusable chunk must pin its immutable provenance")
        _merge_inputs(inputs, job.inputs)
        if reuse is not None:
            inputs[str(reuse)] = file_digest(reuse)
        if len(inputs) > MAX_INPUTS:
            raise ReplayError("Chunk dependency count exceeds its budget")
        chunks.append(Chunk(job, boundary, reuse))
    result = Manifest(config, tuple(chunks), inputs, _ranges(data["ranges"]))
    _validate_coverage(result.chunks, result.ranges)
    return result


def _fingerprint(manifest: Manifest) -> dict[Path, FileStamp]:
    """Rehash all declared inputs, including source, once at each run boundary."""
    stamps = {Path(name): _stamp(Path(name)) for name in manifest.inputs}
    if not os.access(manifest.chunks[0].job.native, os.X_OK):
        raise ReplayError("Current-image native executable is not executable")
    if any(file_digest(Path(name)) != digest for name, digest in manifest.inputs.items()):
        raise ReplayError("Chunk source or declared dependency content changed")
    _stable(stamps, tuple(stamps))
    return stamps


def _stable(stamps: dict[Path, FileStamp], paths: tuple[Path, ...]) -> None:
    if any(_stamp(path) != stamps[path] for path in paths):
        raise ReplayError("Run-local file identity changed; no chunk can adopt cached hashes")


def _output(path: Path) -> Path:
    output = storage._checked_output(path)
    if not output.is_relative_to(WORK_ROOT) or output == WORK_ROOT:
        raise ReplayError("Current-image chunk output must be a child of repository edits/work")
    return output


def _run_directory(output: Path) -> Path:
    runs = output / "runs"
    runs.mkdir(mode=0o700, exist_ok=True)
    if runs.resolve() != runs:
        raise ReplayError("Linked run ledger is not accepted")
    for index in range(MAX_RUNS):
        directory = runs / f"run-{index:04d}"
        try:
            directory.mkdir(mode=0o700)
        except FileExistsError as error:
            if not directory.is_dir() or directory.is_symlink():
                raise ReplayError("Run ledger contains linked or non-directory storage") from error
        else:
            return directory
    raise ReplayError("Run ledger exceeds its immutable attempt budget")


def _verified_chunk(job: collection.ImageJob, evidence: Path) -> VisibleArc:
    storage._checked_output(evidence)
    expected, arc = collection._receipt(job, evidence)
    if (
        json.loads(storage._read(evidence / "inputs.json", storage.CONFIG_BYTES)) != job.inputs
        or json.loads(storage._read(evidence / "provenance.json", storage.CONFIG_BYTES)) != expected
    ):
        raise ReplayError("Completed chunk raw evidence or provenance changed")
    arc.inputs.update(
        {str(evidence / "provenance.json"): file_digest(evidence / "provenance.json")}
    )
    return arc


def _collect(job: collection.ImageJob, evidence: Path) -> None:
    """Retain incomplete attempts; never overwrite or silently retry a native batch."""
    storage._checked_output(evidence)
    if evidence.exists():
        if (evidence / "provenance.json").is_file():
            _verified_chunk(job, evidence)
            return
        raise ReplayError("Incomplete chunk retained; inspect it before a new declared attempt")
    evidence.mkdir(mode=0o700)
    storage._write(evidence / "inputs.json", job.inputs)
    for index, cells in enumerate(collection._batches(job)):
        directory = evidence / f"batch-{index:04d}"
        directory.mkdir(mode=0o700)
        request = directory / "request.json"
        storage._write(request, job.request(cells))
        with (
            (directory / "response.json").open("xb") as stdout,
            (directory / "stderr.log").open("xb") as stderr,
        ):
            os.fchmod(stdout.fileno(), 0o600)
            os.fchmod(stderr.fileno(), 0o600)
            subprocess.run(
                (
                    "/usr/bin/sandbox-exec",
                    "-f",
                    str(job.sandbox),
                    str(job.native),
                    "measure-media",
                    "video_foreground_measure",
                    str(request),
                ),
                stdin=subprocess.DEVNULL,
                stdout=stdout,
                stderr=stderr,
                timeout=collection.BATCH_TIMEOUT_SECONDS,
                check=True,
            )
        collection._batch_cells(job, directory, cells)


def _binding(chunk: Chunk, evidence: Path, arc: VisibleArc) -> dict[str, object]:
    return {
        "config": str(chunk.job.config),
        "configSHA256": chunk.job.inputs[str(chunk.job.config)],
        "boundary": chunk.boundary,
        "evidence": str(evidence),
        "provenanceSHA256": file_digest(evidence / "provenance.json"),
        "frames": len(arc.cells),
        "supportedFrames": sum(cell.visible_top is not None for cell in arc.cells),
        "missingFaceFrames": sum(cell.face is None for cell in chunk.job.cells),
    }


def collect(config: Path, output: Path, *, limit: int | None = None) -> dict[str, object]:
    """Resume complete chunks; stat caching never crosses a new invocation."""
    manifest = load_manifest(config)
    if limit is not None and (type(limit) is not int or not 1 <= limit <= MAX_CHUNKS):
        raise ReplayError("Expected a bounded positive new-chunk limit")
    output = _output(output)
    output.mkdir(mode=0o700, exist_ok=True)
    with execution_lock(output, "current-image-chunks") as acquired:
        if not acquired:
            raise ReplayError("Current-image chunk namespace is already owned")
        return _collect_locked(manifest, output, limit=limit)


def _collect_locked(manifest: Manifest, output: Path, *, limit: int | None) -> dict[str, object]:
    stamps = _fingerprint(manifest)
    pin = output / "manifest-inputs.json"
    if pin.exists():
        if json.loads(storage._read(pin, SUMMARY_BYTES)) != manifest.inputs:
            raise ReplayError("Resume namespace belongs to different input fingerprints")
    else:
        storage._write(pin, manifest.inputs)
    run = _run_directory(output)
    storage._write(
        run / "start.json",
        {"inputs": manifest.inputs, "sourceStat": stamps[manifest.source].encoded()},
    )
    bindings: list[dict[str, object]] = []
    created = 0
    for index, chunk in enumerate(manifest.chunks):
        evidence = chunk.reuse.parent if chunk.reuse is not None else output / f"chunk-{index:04d}"
        if not evidence.exists() and limit is not None and created >= limit:
            break
        paths = (
            tuple(Path(name) for name in chunk.job.inputs)
            + tuple(Path(__file__).with_name(name) for name in CODE_FILES)
            + (manifest.config,)
        )
        _stable(stamps, paths)
        existed = evidence.exists()
        if chunk.reuse is None:
            _collect(chunk.job, evidence)
        _stable(stamps, paths)
        if not (evidence / "provenance.json").exists():
            receipt, _arc = collection._receipt(chunk.job, evidence)
            _stable(stamps, paths)
            storage._write(evidence / "provenance.json", receipt)
        arc = _verified_chunk(chunk.job, evidence)
        _stable(stamps, paths)
        bindings.append(_binding(chunk, evidence, arc))
        created += not existed
    ending = _fingerprint(manifest)
    _stable(stamps, tuple(ending))
    result: dict[str, object] = {
        "version": 1,
        "inputs": manifest.inputs,
        "sourceSHA256": manifest.inputs[str(manifest.source)],
        "sourceStat": ending[manifest.source].encoded(),
        "chunks": bindings,
        "complete": len(bindings) == len(manifest.chunks),
        "frames": sum(len(chunk.job.cells) for chunk in manifest.chunks[: len(bindings)]),
        "candidateOnly": True,
        "semanticHeadVerified": False,
        "captionPlacementApproved": False,
    }
    storage._write(run / "result.json", result)
    if result["complete"] is True:
        summary = output / "summary.json"
        if summary.exists():
            prior = record(json.loads(storage._read(summary, SUMMARY_BYTES)))
            if _content_summary(prior) != _content_summary(result):
                raise ReplayError("Completed summary changed; no replacement is allowed")
        else:
            storage._write(summary, result)
    return result


def _content_summary(value: dict[str, object]) -> dict[str, object]:
    """A new run rehashes source bytes; historical stats are evidence only."""
    return {key: item for key, item in value.items() if key != "sourceStat"}


def _summary(value: object) -> dict[str, object]:
    data = record(value)
    if (
        set(data)
        != {
            "version",
            "inputs",
            "sourceSHA256",
            "sourceStat",
            "chunks",
            "complete",
            "frames",
            "candidateOnly",
            "semanticHeadVerified",
            "captionPlacementApproved",
        }
        or type(data["version"]) is not int
        or data["version"] != 1
    ):
        raise ReplayError("Invalid current-image chunk summary schema")
    stamp = record(data["sourceStat"])
    if set(stamp) != {"device", "inode", "size", "modifiedNS", "changedNS"} or any(
        type(number) is not int or number < 0 for number in stamp.values()
    ):
        raise ReplayError("Invalid historical source identity evidence")
    return data


def verify(config: Path, output: Path) -> VisibleArc:
    """Domain-check every raw chunk and fresh full hashes before joining its cells."""
    manifest = load_manifest(config)
    stamps = _fingerprint(manifest)
    output = _output(output)
    summary = _summary(json.loads(storage._read(output / "summary.json", SUMMARY_BYTES)))
    arcs: list[VisibleArc] = []
    bindings: list[dict[str, object]] = []
    for index, chunk in enumerate(manifest.chunks):
        evidence = chunk.reuse.parent if chunk.reuse is not None else output / f"chunk-{index:04d}"
        arc = _verified_chunk(chunk.job, evidence)
        arcs.append(arc)
        bindings.append(_binding(chunk, evidence, arc))
    if (
        summary.get("inputs") != manifest.inputs
        or summary.get("sourceSHA256") != manifest.inputs[str(manifest.source)]
        or summary.get("chunks") != bindings
        or summary.get("frames") != manifest.frames
        or summary.get("complete") is not True
        or summary.get("candidateOnly") is not True
        or summary.get("semanticHeadVerified") is not False
        or summary.get("captionPlacementApproved") is not False
    ):
        raise ReplayError("Chunk summary differs from current complete raw evidence")
    ending = _fingerprint(manifest)
    _stable(stamps, tuple(ending))
    inputs = dict(manifest.inputs)
    inputs.update({path: digest for arc in arcs for path, digest in arc.inputs.items()})
    inputs[str(output / "summary.json")] = file_digest(output / "summary.json")
    return VisibleArc(tuple(cell for arc in arcs for cell in arc.cells), inputs)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("check", "collect", "verify"))
    parser.add_argument("config", type=Path)
    parser.add_argument("output", type=Path, nargs="?")
    parser.add_argument("--limit", type=int)
    args = parser.parse_args()
    if args.action == "check":
        manifest = load_manifest(args.config)
        print(
            json.dumps(
                {
                    "chunks": len(manifest.chunks),
                    "frames": manifest.frames,
                    "inputs": list(manifest.inputs),
                }
            )
        )
        return
    if args.output is None:
        parser.error("collection and verification require a repository-local output directory")
    if args.action == "collect":
        result = collect(args.config, args.output, limit=args.limit)
        print(json.dumps({key: result[key] for key in ("complete", "frames", "candidateOnly")}))
    else:
        arc = verify(args.config, args.output)
        print(f"Verified {len(arc.cells)} current cells; semantic head remains unapproved")


if __name__ == "__main__":
    main()
