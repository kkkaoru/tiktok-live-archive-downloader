"""Build immutable recognition-token summary inputs on the edited output clock.

The display SRT is checked against the selected FCPXML. It is not substituted
for recognition text, and media identity does not certify a rendered delivery.
Batch proofs keep workflow input arrays bounded while retaining every token pin.
"""

import argparse
import math
import sys
from dataclasses import dataclass
from itertools import batched
from pathlib import Path

from replay.caption_measured_job import _json as strict_json
from replay.caption_native_paging import read_tokens
from replay.caption_token_sources import TokenSource, load_sources
from replay.core import ReplayError
from replay.delivery_captions import _write_new
from replay.fcp_caption_srt import srt_text
from replay.foreground_evidence import _bound_bytes
from replay.recordings import record
from replay.stages import file_digest

ROOT = Path(__file__).resolve().parents[2]
MAX_MANIFEST_BYTES = 4 << 20
MAX_REFLOW_BYTES = 64 << 20
MAX_SRT_BYTES = 16 << 20
MAX_BATCH_PINS = 240
MAX_JOBS = 10000
MAX_SEGMENTS = 512
MAX_FRAMES = 10000000
MAX_TOKENS = 1000000
TOKEN_CLOCK_TOLERANCE = 0.000001
CONFIG_FIELDS = frozenset(
    {"version", "captions", "segmentIndex", "reflow", "displayXML", "displaySRT", "media"}
)
CODE_FILES = (
    "summary_caption_inputs.py",
    "caption_native_paging.py",
    "caption_token_sources.py",
    "caption_measured_job.py",
    "fcp_caption_srt.py",
    "delivery_render.py",
    "delivery_output_names.py",
)


@dataclass(frozen=True)
class SummaryInputs:
    config: Path
    manifest: dict[str, object]
    direct: dict[str, str]
    tokens: dict[str, str]
    token_count: int
    media_identity: dict[str, object]


@dataclass(frozen=True)
class Preparation:
    captions: Path
    reflow: Path
    xml: Path
    media: Path
    srt: Path
    token_sources: Path | None = None


def _path(value: object, *, media: bool = False) -> Path:
    if not isinstance(value, str) or not value or "\0" in value:
        raise ReplayError("Expected an explicit canonical summary input path")
    path = Path(value)
    if (
        not path.is_absolute()
        or path != path.resolve()
        or not path.is_file()
        or any(part in {"private", "downloads"} for part in path.parts)
        or (not media and not path.is_relative_to(ROOT))
    ):
        raise ReplayError("Summary input paths must be regular canonical files in scope")
    return path


def _read(path: Path, digest: str, limit: int = MAX_MANIFEST_BYTES) -> dict[str, object]:
    return record(strict_json(_bound_bytes(path, digest, limit)))


def _pin(value: object, inputs: dict[str, str]) -> Path:
    row = record(value)
    path = _path(row.get("path"))
    digest = row.get("sha256")
    if set(row) != {"path", "sha256"} or not isinstance(digest, str):
        raise ReplayError("Expected a path and SHA256 summary input pin")
    if file_digest(path) != digest:
        raise ReplayError("Summary input hash changed")
    inputs[str(path)] = digest
    return path


def _reference(path: Path) -> dict[str, str]:
    return {"path": str(path), "sha256": file_digest(path)}


def _new_work_file(path: Path) -> Path:
    if (
        path != path.resolve()
        or not path.is_relative_to(ROOT / "edits" / "work")
        or not path.parent.is_dir()
        or path.exists()
        or path.is_symlink()
    ):
        raise ReplayError("Summary preparation requires a new canonical repository work file")
    return path


def configure(output: Path, preparation: Preparation) -> Path:
    """Export exact display captions and pin already-projected recognition inputs."""
    _new_work_file(output)
    _new_work_file(preparation.srt)
    index = _path(str(preparation.captions / "segments.json"))
    reflow = _path(str(preparation.reflow))
    xml = _path(str(preparation.xml))
    media = _path(str(preparation.media), media=True)
    text = srt_text(xml)
    with preparation.srt.open("x", encoding="utf-8") as stream:
        stream.write(text)
    preparation.srt.chmod(0o600)
    config: dict[str, object] = {
        "version": 1,
        "captions": str(index.parent),
        "segmentIndex": _reference(index),
        "reflow": _reference(reflow),
        "displayXML": _reference(xml),
        "displaySRT": _reference(preparation.srt),
        "media": str(media),
    }
    if preparation.token_sources is not None:
        config["tokenSources"] = _reference(_path(str(preparation.token_sources)))
    _write_new(output, config)
    load(output)
    return output


def _integer(value: object, *, minimum: int = 0, maximum: int = MAX_FRAMES) -> int:
    if type(value) is not int or not minimum <= value <= maximum:
        raise ReplayError("Summary index requires bounded integer clocks and identifiers")
    return value


def _entries(index: dict[str, object]) -> tuple[list[object], int]:
    entries = index.get("segments")
    rate = _integer(index.get("rate"), minimum=1, maximum=240)
    if (
        type(index.get("version")) is not int
        or index.get("version") != 1
        or not isinstance(entries, list)
        or not 1 <= len(entries) <= MAX_SEGMENTS
    ):
        raise ReplayError("Expected a bounded version1 caption segment index")
    return entries, rate


def _job_source(
    directory: Path, row: dict[str, object], job: int, sources: tuple[TokenSource, ...]
) -> tuple[Path, TokenSource | None]:
    replacement = next(
        (source for source in sources if source.segment == row["segment"] and source.job == job),
        None,
    )
    path = directory / f"tokens-{job:04d}.json" if replacement is None else replacement.tokens
    return _path(str(path)), replacement


def _selected_job(
    directory: Path,
    row: dict[str, object],
    *,
    job: int,
    bindings: dict[str, object],
    sources: tuple[TokenSource, ...],
    duration: float,
) -> tuple[dict[str, object], dict[str, str], int]:
    # Empty jobs retain their exact source identity and membership. The shared
    # reader validates the full segment before this per-job manifest is assembled.
    path, replacement = _job_source(directory, row, job, sources)
    expected = bindings.get(str(path))
    if not isinstance(expected, str):
        raise ReplayError("Existing candidate does not bind its recognized tokens")
    pins = {} if replacement is None else replacement.verify(bindings)
    pins[str(path)] = expected
    document = _read(path, expected)
    if type(document.get("version")) is not int or document.get("version") != 1:
        raise ReplayError("Summary recognition files require version1")
    values = document.get("tokens")
    if document.get("clock") != "output" or not isinstance(values, list):
        raise ReplayError("Summary recognition files require output-clock token arrays")
    if not values:
        return {"job": job, "tokens": {"path": str(path), "sha256": expected}}, pins, 0
    selected = {**row, "jobs": [job]}
    tokens, selected_pins = read_tokens(directory, selected, bindings, sources)
    pins.update(selected_pins)
    previous = 0.0
    for token in tokens:
        if (
            not math.isfinite(token.start)
            or not math.isfinite(token.end)
            or not previous <= token.start <= token.end <= duration + TOKEN_CLOCK_TOLERANCE
            or not token.text.strip()
            or len(token.text.encode("utf-8")) > 16384
            or not token.utterance
            or len(token.utterance.encode("utf-8")) > 512
        ):
            raise ReplayError("Summary tokens must retain valid segment-local recognized clocks")
        previous = token.start
    return {"job": job, "tokens": {"path": str(path), "sha256": pins[str(path)]}}, pins, len(tokens)


def _jobs(
    directory: Path,
    entries: list[object],
    *,
    bindings: dict[str, object],
    sources: tuple[TokenSource, ...],
    rate: int,
) -> tuple[list[dict[str, object]], dict[str, str], int]:
    jobs: list[dict[str, object]] = []
    pins: dict[str, str] = {}
    segments: set[int] = set()
    seen: set[int] = set()
    count = 0
    for value in entries:
        row = record(value)
        segment = _integer(row.get("segment"))
        if segment in segments:
            raise ReplayError("Summary segment identifiers must be unique")
        segments.add(segment)
        _integer(row.get("offsetFrame"))
        duration = _integer(row.get("frames"), minimum=1) / rate
        members = row.get("jobs")
        if not isinstance(members, list) or not 1 <= len(members) <= 128:
            raise ReplayError("Expected bounded caption job membership")
        selected, additions, recognized = _segment_jobs(
            directory, row, members=members, bindings=bindings, sources=sources, duration=duration
        )
        identifiers = {_integer(item["job"]) for item in selected}
        if seen.intersection(identifiers) or len(identifiers) != len(selected):
            raise ReplayError("Summary jobs must belong to exactly one segment")
        seen.update(identifiers)
        jobs.extend(selected)
        pins.update(additions)
        count += recognized
        if len(jobs) > MAX_JOBS or count > MAX_TOKENS:
            raise ReplayError("Summary recognition input budget exceeded")
    return jobs, pins, count


def _segment_jobs(
    directory: Path,
    row: dict[str, object],
    *,
    members: list[object],
    bindings: dict[str, object],
    sources: tuple[TokenSource, ...],
    duration: float,
) -> tuple[list[dict[str, object]], dict[str, str], int]:
    identifiers = [_integer(member, maximum=MAX_JOBS - 1) for member in members]
    for member in identifiers:
        _job_source(directory, row, member, sources)
    # Keep the native segment's nonempty/4096-token budget. An empty individual
    # recognition job is valid only as part of a nonempty recognized segment.
    read_tokens(directory, {**row, "jobs": identifiers}, bindings, sources)
    result: list[dict[str, object]] = []
    pins: dict[str, str] = {}
    count = 0
    for member in identifiers:
        item, additions, recognized = _selected_job(
            directory,
            row,
            job=member,
            bindings=bindings,
            sources=sources,
            duration=duration,
        )
        result.append(item)
        pins.update(additions)
        count += recognized
    return result, pins, count


def load(config: Path) -> SummaryInputs:
    """Validate the consumer contract without changing token text or any clock."""
    config = _path(str(config))
    direct = {
        str(Path(__file__).with_name(name).resolve()): file_digest(Path(__file__).with_name(name))
        for name in CODE_FILES
    }
    direct[str(config)] = file_digest(config)
    raw = _read(config, direct[str(config)])
    if (
        set(raw) not in (CONFIG_FIELDS, CONFIG_FIELDS | {"tokenSources"})
        or type(raw.get("version")) is not int
        or raw.get("version") != 1
    ):
        raise ReplayError("Expected the version1 edited summary configuration")
    paths = {
        name: _pin(raw[name], direct)
        for name in ("segmentIndex", "reflow", "displayXML", "displaySRT")
    }
    index = _read(paths["segmentIndex"], direct[str(paths["segmentIndex"])])
    reflow = _read(paths["reflow"], direct[str(paths["reflow"])], MAX_REFLOW_BYTES)
    bindings = record(reflow.get("inputs"))
    if bindings.get(str(paths["segmentIndex"])) != direct[str(paths["segmentIndex"])]:
        raise ReplayError("Reflow does not bind the selected summary segment index")
    entries, rate = _entries(index)
    directory = raw.get("captions")
    if not isinstance(directory, str) or Path(directory) != paths["segmentIndex"].parent:
        raise ReplayError("Caption directory must be the selected segment index parent")
    sources: tuple[TokenSource, ...] = ()
    source_pins: dict[str, str] = {}
    if "tokenSources" in raw:
        source_path = _pin(raw["tokenSources"], direct)
        sources, source_pins = load_sources(source_path, entries)
    jobs, token_pins, count = _jobs(
        Path(directory), entries, bindings=bindings, sources=sources, rate=rate
    )
    token_pins.update(source_pins)
    display = _bound_bytes(paths["displaySRT"], direct[str(paths["displaySRT"])], MAX_SRT_BYTES)
    if display.decode("utf-8") != srt_text(paths["displayXML"]):
        raise ReplayError("Display SRT differs from the selected FCPXML caption clock or text")
    media = _path(raw.get("media"), media=True)
    metadata = media.stat()
    manifest: dict[str, object] = {
        "version": 1,
        "clockDomain": "edited-output",
        "media": str(media),
        "displaySRT": raw["displaySRT"],
        "segmentIndex": raw["segmentIndex"],
        "reflow": raw["reflow"],
        "jobs": jobs,
        "candidateOnly": True,
        "recognitionAccuracyVerified": False,
        "renderedDeliveryVerified": False,
    }
    return SummaryInputs(
        config,
        manifest,
        direct,
        token_pins,
        count,
        {"path": str(media), "size": metadata.st_size, "mtimeNs": metadata.st_mtime_ns},
    )


def _output(directory: Path, *, new: bool) -> Path:
    if (
        directory != directory.resolve()
        or not directory.is_relative_to(ROOT / "edits" / "work")
        or (new and directory.exists())
        or (not new and not directory.is_dir())
    ):
        raise ReplayError("Summary work requires a canonical repository work directory")
    return directory


def _batch_proof(path: Path) -> dict[str, object]:
    path = _path(str(path))
    descriptor = _read(path, file_digest(path))
    values = record(descriptor.get("inputs"))
    if (
        set(descriptor) != {"version", "kind", "config", "inputs"}
        or descriptor.get("version") != 1
        or descriptor.get("kind") != "summary-token-input-batch-v1"
        or not 1 <= len(values) <= MAX_BATCH_PINS
    ):
        raise ReplayError("Expected a bounded summary token input batch")
    pins: dict[str, str] = {}
    _pin(descriptor.get("config"), pins)
    for name, digest in values.items():
        member = _pin({"path": name, "sha256": digest}, pins)
        _bound_bytes(member, pins[name], MAX_REFLOW_BYTES)
    return {
        "version": 1,
        "kind": "verified-summary-token-batch-v1",
        "batch": _reference(path),
        "inputs": pins,
        "producer": _reference(Path(__file__).resolve()),
        "candidateOnly": True,
    }


def verify_batch(path: Path, proof: Path, *, existing: bool = False) -> None:
    """Check every transitive token file, then publish or compare an immutable proof."""
    expected = _batch_proof(path)
    if existing:
        if _read(_path(str(proof)), file_digest(proof)) != expected:
            raise ReplayError("Summary batch proof differs from current pinned inputs")
        return
    _write_new(_new_work_file(proof), expected)


def _batch_paths(inputs: SummaryInputs, directory: Path) -> tuple[tuple[Path, Path], ...]:
    return tuple(
        (
            directory / "batches" / f"batch-{number:04d}.json",
            directory / "batches" / f"batch-{number:04d}.proof.json",
        )
        for number, _ in enumerate(batched(inputs.tokens.items(), MAX_BATCH_PINS))
    )


def plan(config: Path, directory: Path) -> Path:
    """Declare token batches and a consumer stage without weakening workflow limits."""
    directory = _output(directory, new=True)
    inputs = load(config)
    directory.mkdir(mode=0o700)
    (directory / "batches").mkdir(mode=0o700)
    stages: list[dict[str, object]] = []
    commands = [sys.executable, "-m", "replay.summary_caption_inputs"]
    batches = _batch_paths(inputs, directory)
    for (descriptor, proof), members in zip(
        batches, batched(inputs.tokens.items(), MAX_BATCH_PINS), strict=True
    ):
        values = dict(members)
        _write_new(
            descriptor,
            {
                "version": 1,
                "kind": "summary-token-input-batch-v1",
                "config": _reference(inputs.config),
                "inputs": values,
            },
        )
        stages.append(
            {
                "name": descriptor.stem,
                "command": [*commands, "batch", str(descriptor), str(proof)],
                "inputs": [
                    str(descriptor),
                    str(inputs.config),
                    str(Path(__file__).resolve()),
                    *values,
                ],
                "outputs": [str(proof)],
                "timeout_seconds": 300,
                "verify_command": [*commands, "batch", str(descriptor), str(proof), "--verify"],
            }
        )
    stages.append(
        {
            "name": "summary-input-manifest",
            "command": [*commands, "produce", str(inputs.config), str(directory)],
            "inputs": [*inputs.direct, *(str(path) for pair in batches for path in pair)],
            "outputs": [str(directory / "manifest.json"), str(directory / "receipt.json")],
            "timeout_seconds": 600,
            "verify_command": [*commands, "verify", str(inputs.config), str(directory)],
        }
    )
    graph = directory / "workflow.json"
    _write_new(graph, {"version": 1, "stages": stages})
    return graph


def _documents(config: Path, directory: Path) -> tuple[dict[str, object], dict[str, object]]:
    inputs = load(config)
    pins = inputs.direct | inputs.tokens
    expected_members = tuple(batched(inputs.tokens.items(), MAX_BATCH_PINS))
    for (descriptor, proof), members in zip(
        _batch_paths(inputs, directory), expected_members, strict=True
    ):
        raw = _read(_path(str(descriptor)), file_digest(descriptor))
        if raw.get("inputs") != dict(members) or raw.get("config") != _reference(inputs.config):
            raise ReplayError("Summary batch does not cover the selected recognition inputs")
        verify_batch(descriptor, proof, existing=True)
        pins.update({str(descriptor): file_digest(descriptor), str(proof): file_digest(proof)})
    receipt: dict[str, object] = {
        "version": 1,
        "kind": "canonical-caption-summary-input-v1",
        "clockDomain": "edited-output",
        "config": _reference(inputs.config),
        "inputs": pins,
        "recognizedTokens": inputs.token_count,
        "mediaIdentity": inputs.media_identity,
        "mediaBinding": "identity-only-not-render-verification",
        "segmentOffsetApplications": 1,
        "recognitionClocksChanged": False,
        "displaySRTMatchesXML": True,
        "candidateOnly": True,
        "recognitionAccuracyVerified": False,
        "summaryQualityVerified": False,
        "renderedDeliveryVerified": False,
    }
    return inputs.manifest, receipt


def produce(config: Path, directory: Path, *, existing: bool = False) -> Path:
    """Recheck all batches at the consumer boundary; no unseen files are adopted."""
    directory = _output(directory, new=False)
    manifest, receipt = _documents(config, directory)
    output = directory / "manifest.json"
    saved = directory / "receipt.json"
    if existing:
        if (
            _read(_path(str(output)), file_digest(output)) != manifest
            or _read(_path(str(saved)), file_digest(saved)) != receipt
        ):
            raise ReplayError("Summary manifest or receipt differs from its current input contract")
        return output
    if output.exists() or saved.exists():
        raise ReplayError("Summary input outputs must be new immutable files")
    _write_new(output, manifest)
    _write_new(saved, receipt)
    return output


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="operation", required=True)
    configure_command = commands.add_parser("configure")
    configure_command.add_argument("config", type=Path)
    configure_command.add_argument("--captions", type=Path, required=True)
    configure_command.add_argument("--reflow", type=Path, required=True)
    configure_command.add_argument("--xml", type=Path, required=True)
    configure_command.add_argument("--media", type=Path, required=True)
    configure_command.add_argument("--srt", type=Path, required=True)
    configure_command.add_argument("--token-sources", type=Path)
    for operation in ("plan", "produce", "verify"):
        child = commands.add_parser(operation)
        child.add_argument("config", type=Path)
        child.add_argument("directory", type=Path)
    batch = commands.add_parser("batch")
    batch.add_argument("descriptor", type=Path)
    batch.add_argument("proof", type=Path)
    batch.add_argument("--verify", action="store_true")
    args = parser.parse_args()
    if args.operation == "configure":
        print(
            configure(
                args.config,
                Preparation(
                    args.captions, args.reflow, args.xml, args.media, args.srt, args.token_sources
                ),
            )
        )
    elif args.operation == "batch":
        verify_batch(args.descriptor, args.proof, existing=args.verify)
    elif args.operation == "plan":
        print(plan(args.config, args.directory))
    else:
        print(produce(args.config, args.directory, existing=args.operation == "verify"))


if __name__ == "__main__":
    main()
