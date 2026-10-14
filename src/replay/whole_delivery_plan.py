"""Prepare complete whole-delivery graphs from pinned, ordered part evidence.

Only metadata, source code and configured executables are hashed here. Actual
media bytes, all domain verifiers and checkpoint fingerprints remain mandatory
at execution. Preparation reuses only byte-identical immutable metadata.
"""

import argparse
import hashlib
import json
import os
import stat
from collections.abc import Iterator
from dataclasses import dataclass
from fractions import Fraction
from pathlib import Path

from replay import fcp_cleanup_adoption as cleanup
from replay import fcp_part_assembly as assembly
from replay import part_master_publication as publication_domain
from replay import part_retirement_verification_batch as retirement_batch
from replay import part_retirement_verification_readonly as retirement_readonly
from replay import sha256_check
from replay import whole_technical_batch as technical
from replay.core import ReplayError
from replay.fcp_part_assembly import Timeline
from replay.fcp_part_followup import MAX_PARTS
from replay.input_hash_cache import MAX_INPUTS
from replay.recordings import record
from replay.stages import _read_receipt, file_digest
from replay.workflow import MAX_STAGES, load_stages, parse_stages, stage_graph

ROOT = Path(__file__).resolve().parents[2]
# Each member may require its own two-argument --scope plus fixed digest flags.
# Keep both declared paths and the complete command below workflow's 256 limit.
GROUP_INPUTS = 96
MAX_REFERENCE_DEPTH = 64
# Array values include frame clocks, not just input-file references. A JSON
# element and separator require at least two bytes within the existing byte cap.
MAX_METADATA_ITEMS = technical.MAX_METADATA_BYTES // 2
TOOL_NAMES = frozenset({"python", "env", "ffmpeg", "ffprobe", "jq", "cmp", "shasum"})
TARGETS = ("whole-frame-clock", "whole-metadata", "joined-pcm-hash")


@dataclass(frozen=True)
class Tool:
    alias: Path
    resolved: technical.Reference


@dataclass(frozen=True)
class Part:
    movie: technical.Reference
    proof: technical.Reference
    publication: technical.Reference
    retirement: technical.Reference
    verifier: technical.Reference


@dataclass(frozen=True)
class Facts:
    config: technical.Reference
    assembly: technical.Reference
    timeline: Timeline
    voice: technical.Reference
    cleanup_configs: tuple[Path, ...]
    parts: tuple[Part, ...]
    tools: dict[str, Tool]
    inputs: dict[str, str]
    master: Path
    threads: int
    digest_limits: sha256_check.DigestLimits
    retirement_batch: retirement_readonly.Request | None
    existing_assembly: tuple[technical.Reference, technical.Reference] | None


class Arguments(argparse.Namespace):
    command: str
    config: Path
    output: Path
    plan: Path
    source_state: Path
    state: Path


def _path(value: object, *, work: bool = False) -> Path:
    if not isinstance(value, str):
        raise ReplayError("Whole plan requires a canonical path")
    return technical._path(value, work=work)


def _reference(value: object) -> technical.Reference:
    return technical._reference(value)


def _refs(value: object) -> tuple[technical.Reference, ...]:
    if not isinstance(value, list) or not 1 <= len(value) <= MAX_INPUTS:
        raise ReplayError("Whole plan requires bounded nonempty references")
    refs = tuple(_reference(row) for row in value)
    if len({ref.path for ref in refs}) != len(refs):
        raise ReplayError("Whole plan references must be unique")
    return refs


def _bind(inputs: dict[str, str], ref: technical.Reference) -> None:
    if inputs.get(str(ref.path), ref.sha256) != ref.sha256:
        raise ReplayError("Whole plan has conflicting input identities")
    inputs[str(ref.path)] = ref.sha256
    if len(inputs) >= MAX_INPUTS:
        raise ReplayError("Whole plan transitive input budget exceeded")


def _metadata(ref: technical.Reference) -> dict[str, object]:
    if not ref.path.is_relative_to(ROOT / "edits"):
        raise ReplayError("Whole plan metadata must be inside checkout edits")
    return technical._metadata(ref.path, ref.sha256)


def _regular(path: Path) -> None:
    info = path.lstat()
    if not stat.S_ISREG(info.st_mode) or info.st_size <= 0:
        raise ReplayError("Whole plan input must be a nonempty regular file")


def _nested_refs(value: object, *, depth: int = 0) -> Iterator[technical.Reference]:
    if depth > MAX_REFERENCE_DEPTH:
        raise ReplayError("Whole plan metadata reference nesting is excessive")
    if isinstance(value, dict):
        row = record(value)
        if {"path", "sha256"}.issubset(row):
            yield _reference({"path": row["path"], "sha256": row["sha256"]})
        else:
            for child in row.values():
                yield from _nested_refs(child, depth=depth + 1)
    elif isinstance(value, list):
        if len(value) > MAX_METADATA_ITEMS:
            raise ReplayError("Whole plan metadata array exceeds its item budget")
        for child in value:
            yield from _nested_refs(child, depth=depth + 1)


def _collect(
    ref: technical.Reference,
    inputs: dict[str, str],
    visited: set[Path],
    replacements: dict[Path, technical.Reference],
) -> dict[str, object]:
    replacement = replacements.get(ref.path)
    if replacement is not None:
        if replacement.sha256 != ref.sha256:
            raise ReplayError("Historical source and published replacement hashes differ")
        ref = replacement
    _bind(inputs, ref)
    if ref.path in visited:
        return {}
    visited.add(ref.path)
    if ref.path.suffix != ".json":
        return {}
    _regular(ref.path)
    if not ref.path.is_relative_to(ROOT / "edits"):
        raise ReplayError("Whole plan metadata must be inside checkout edits")
    data = technical._bounded_bytes(ref.path)
    if hashlib.sha256(data).hexdigest() != ref.sha256:
        raise ReplayError("Whole plan metadata fingerprint changed")
    document: object = json.loads(
        data, object_pairs_hook=technical._unique, parse_constant=technical._nonfinite
    )
    if not isinstance(document, (dict, list)):
        raise ReplayError("Whole closure JSON must be an object or bounded array")
    for child in _nested_refs(document):
        _collect(child, inputs, visited, replacements)
    return record(document) if isinstance(document, dict) else {}


def source_inventory() -> tuple[technical.Reference, ...]:
    """Return current code facts, including lazy readers and cleanup domains."""
    paths = {
        *technical.reader_code_paths(),
        *(ROOT / "src/replay" / name for name in cleanup.CODE_NAMES),
        *_digest_code_paths(),
        Path(__file__).resolve(),
        Path(retirement_batch.__file__).resolve(),
        Path(retirement_readonly.__file__).resolve(),
    }
    return tuple(technical.Reference(path, file_digest(path)) for path in sorted(paths))


def _digest_code_paths() -> tuple[Path, ...]:
    """The bounded SHA reader has only standard-library dependencies."""
    helper = Path(sha256_check.__file__).resolve()
    return helper, helper.with_name("__init__.py")


def _digest_limits(value: object) -> sha256_check.DigestLimits:
    fields = ("workers", "memoryBudgetBytes", "baselineRSSBytes", "workerRSSBytes")
    row = record(value)
    if set(row) != set(fields):
        raise ReplayError("Whole closure digest limit fields differ")
    workers, memory, baseline, worker = (row[field] for field in fields)
    if (
        type(workers) is not int
        or type(memory) is not int
        or type(baseline) is not int
        or type(worker) is not int
    ):
        raise ReplayError("Whole closure digest limits must be explicit integers")
    try:
        return sha256_check.DigestLimits(
            workers=workers,
            memory_budget_bytes=memory,
            baseline_rss_bytes=baseline,
            worker_rss_bytes=worker,
        )
    except ValueError as error:
        raise ReplayError("Whole closure digest memory admission failed") from error


def _tools(value: object) -> dict[str, Tool]:
    rows = record(value)
    if set(rows) != TOOL_NAMES:
        raise ReplayError("Whole plan requires every configured tool")
    result: dict[str, Tool] = {}
    for name, raw in rows.items():
        row = record(raw)
        if set(row) != {"path", "sha256"} or not isinstance(row["path"], str):
            raise ReplayError("Whole plan tool fields differ")
        alias, resolved = technical._tool((row["path"],))
        digest = technical._sha(row["sha256"])
        if not os.access(resolved, os.X_OK) or file_digest(resolved) != digest:
            raise ReplayError("Whole plan executable fingerprint changed")
        result[name] = Tool(alias, technical.Reference(resolved, digest))
    return result


def _part(
    value: object,
    *,
    number: int,
    row: dict[str, object],
    facts: assembly.Timeline,
    assembly_ref: technical.Reference,
    inputs: dict[str, str],
    visited: set[Path],
    replacements: dict[Path, technical.Reference],
) -> Part:
    entry = record(value)
    if set(entry) != {"part", "publication", "retirement", "retirementVerifyConfig"} or (
        type(entry["part"]) is not int or entry["part"] != number
    ):
        raise ReplayError("Whole plan must include every ordered part identity")
    publication, retirement, verifier = (
        _reference(entry[name]) for name in ("publication", "retirement", "retirementVerifyConfig")
    )
    published = _collect(publication, inputs, visited, replacements)
    retired = _collect(retirement, inputs, visited, replacements)
    verification = _collect(verifier, inputs, visited, replacements)
    movie = _reference(published.get("movie"))
    proof = _reference(published.get("frameProof"))
    frame = _metadata(proof)
    completed = _metadata(_reference(published.get("completion")))
    if (
        type(published.get("version")) is not int
        or published.get("version") != 1
        or published.get("kind") != "published-native-part"
        or published.get("sourcePreserved") is not True
        or str(movie.path) != row.get("movie")
        or str(proof.path) != row.get("proof")
        or type(frame.get("version")) is not int
        or frame.get("version") != 1
        or type(frame.get("part")) is not int
        or frame.get("part") != number
        or frame.get("movie") != movie.json()
        or frame.get("partition")
        != {"path": str(facts.manifest), "sha256": inputs[str(facts.manifest)]}
        or type(completed.get("version")) is not int
        or completed.get("version") != 1
        or completed.get("kind") != "verified-native-part"
        or type(completed.get("part")) is not int
        or completed.get("part") != number
        or completed.get("completionVerified") is not True
        or completed.get("shareObservedIdle") is not True
        or type(completed.get("fullAVDecodeExitCode")) is not int
        or completed.get("fullAVDecodeExitCode") != 0
        or completed.get("movie") != published.get("source")
        or _reference(completed.get("movie")).sha256 != movie.sha256
        or type(retired.get("version")) is not int
        or retired.get("version") != 1
        or retired.get("kind") != "verified-staging-retirement"
        or type(retired.get("part")) is not int
        or retired.get("part") != number
        or retired.get("assembly") != assembly_ref.json()
        or retired.get("stagingRetirementAuthorized") is not True
        or retired.get("deletionPerformed") is not False
        or retired.get("publication") != publication.json()
        or verification != {"version": 1, "retirement": retirement.json()}
        or type(verification.get("version")) is not int
    ):
        raise ReplayError("Whole plan part completion/publication/retirement binding differs")
    if any(
        retired.get(key) != published.get(key)
        for key in ("source", "movie", "completion", "frameProof", "intent")
    ):
        raise ReplayError("Whole plan retirement lineage differs from publication")
    original_proof = _reference(completed.get("frameProof"))
    source_frame = _metadata(original_proof)
    intent = _reference(published.get("intent"))
    intended = _metadata(intent)
    retirement_config = _metadata(_reference(retired.get("config")))
    if frame != {
        **source_frame,
        "movie": movie.json(),
        "sourceFrameProof": original_proof.json(),
        "publicationIntent": intent.json(),
    }:
        raise ReplayError("Published frame proof differs from original native evidence")
    if (
        source_frame.get("movie") != completed.get("movie")
        or source_frame.get("part") != number
        or intended.get("source") != completed.get("movie")
        or intended.get("destination") != str(movie.path)
        or intended.get("completion") != published.get("completion")
        or type(retirement_config.get("version")) is not int
        or retirement_config.get("version") != 1
        or retirement_config.get("retireStaging") is not True
        or retirement_config.get("publication") != retired.get("publication")
        or retirement_config.get("assembly") != retired.get("assembly")
        or retirement_config.get("receiptOutput") != str(retirement.path)
    ):
        raise ReplayError("Whole part publication or retirement intent differs")
    source = _reference(published["source"]).path
    source_identity = retired.get("sourceIdentity")
    destination_identity = list(publication_domain._identity(movie.path))
    if (
        source == movie.path
        or not isinstance(source_identity, list)
        or len(source_identity) != 4
        or any(type(value) is not int or value < 0 for value in source_identity)
        or source_identity[:2] == destination_identity[:2]
        or retired.get("destinationIdentity") != destination_identity
        or (source.exists() and list(publication_domain._identity(source)) != source_identity)
    ):
        raise ReplayError("Whole part retirement physical identities differ")
    assembly._check_video(record(frame.get("observed")), facts, number - 1)
    completion_inputs = record(completed.get("inputs"))
    if not completion_inputs:
        raise ReplayError("Whole part completion input closure is empty")
    for name, digest in completion_inputs.items():
        selected = movie.path if name == str(_reference(published["source"]).path) else _path(name)
        _collect(
            technical.Reference(selected, technical._sha(digest)), inputs, visited, replacements
        )
    return Part(movie, proof, publication, retirement, verifier)


def _retirement_adoption(
    value: object,
    parts: tuple[Part, ...],
    inputs: dict[str, str],
    visited: set[Path],
    replacements: dict[Path, technical.Reference],
) -> retirement_readonly.Request:
    ref = _reference(value)
    _collect(ref, inputs, visited, replacements)
    request = retirement_readonly.load(ref.path)
    phase, selected, refs = retirement_batch._manifest(
        request.manifest.path, request.manifest.sha256
    )
    expected = tuple(range(1, len(parts) + 1))
    if (
        phase != "after-delete"
        or tuple(row.number for row in selected) != expected
        or tuple(row.part for row in request.checkpoints) != expected
        or tuple(row.config.json() for row in selected)
        != tuple(part.verifier.json() for part in parts)
    ):
        raise ReplayError("Whole readonly retirement batch must cover every ordered current part")
    operation = {"version": 1, "manifest": request.manifest.json(), "phase": phase}
    if not retirement_readonly._matches(retirement_batch._read(request.operation), operation):
        raise ReplayError("Whole readonly retirement operation differs")
    retirement_readonly._receipt(request, phase=phase, parts=selected, refs=refs)
    for checkpoint, part in zip(request.checkpoints, selected, strict=True):
        retirement_readonly._checkpoint_check(request, checkpoint, part, phase=phase)
    for item in (*request.metadata, *refs):
        _bind(inputs, technical.Reference(item.path, item.sha256))
    return request


def _assembly_adoption(
    value: object,
    *,
    assembly_ref: technical.Reference,
    parts: tuple[Part, ...],
    raw: dict[str, object],
    inputs: dict[str, str],
) -> tuple[technical.Reference, technical.Reference]:
    row = record(value)
    if set(row) != {"assembly", "xml", "lineage"} or row["assembly"] != assembly_ref.json():
        raise ReplayError("Existing assembly must bind the current assembly configuration")
    xml, lineage = _reference(row["xml"]), _reference(row["lineage"])
    if lineage.path != xml.path.with_suffix(".lineage.json"):
        raise ReplayError("Existing assembly lineage path differs")
    _regular(xml.path)
    if file_digest(xml.path) != xml.sha256:
        raise ReplayError("Existing assembly XML bytes changed")
    document = _metadata(lineage)
    if (
        document.get("assemblyXML") != xml.json()
        or document.get("partition") != raw.get("partition")
        or document.get("voice") != raw.get("voice")
        or document.get("voiceReceipt") != raw.get("voiceReceipt")
        or document.get("masterProofs") != [part.proof.json() for part in parts]
        or document.get("partAudioDisabled") is not True
        or document.get("voiceOutputStart") != "0s"
    ):
        raise ReplayError("Existing assembly lineage differs from ordered current inputs")
    _bind(inputs, xml)
    _bind(inputs, lineage)
    return xml, lineage


def _facts(config: Path) -> Facts:
    config = _path(str(config), work=True)
    document = technical._metadata(config)
    required = {
        "version",
        "assembly",
        "cleanupConfigs",
        "parts",
        "tools",
        "sourceInventory",
        "wholeMovie",
        "decoderThreads",
        "closureDigest",
    }
    optional = {"verifiedReadonlyRetirementBatch", "verifiedExistingAssembly"}
    if not required <= set(document) <= required | optional or (
        type(document["version"]) is not int or document["version"] != 1
    ):
        raise ReplayError("Whole plan configuration fields differ")
    digest_limits = _digest_limits(document["closureDigest"])
    assembly_ref = _reference(document["assembly"])
    configs = _refs(document["cleanupConfigs"])
    declared_code = _refs(document["sourceInventory"])
    current_code = source_inventory()
    if {ref.path: ref.sha256 for ref in declared_code} != {
        ref.path: ref.sha256 for ref in current_code
    }:
        raise ReplayError("Whole plan current source inventory differs")
    tools = _tools(document["tools"])
    threads = document["decoderThreads"]
    assembly.FrameProbeTool.from_reference(
        tools["ffprobe"].resolved.json(), decoder_threads=threads
    )
    if type(threads) is not int:
        raise ReplayError("Whole plan decoder threads must be an integer")
    inputs = {str(config): file_digest(config)}
    visited: set[Path] = set()
    raw = _collect(assembly_ref, inputs, visited, {})
    partition = _reference(raw.get("partition"))
    partition_metadata = _metadata(partition)
    if any(
        type(document.get("version")) is not int or document.get("version") != 1
        for document in (raw, partition_metadata)
    ):
        raise ReplayError("Whole assembly/partition version differs")
    plan = assembly.timeline(partition.path)
    values, rows = document["parts"], raw.get("parts")
    if (
        not isinstance(values, list)
        or not isinstance(rows, list)
        or not (1 <= len(values) == len(rows) == len(configs) == len(plan.parts) <= MAX_PARTS)
    ):
        raise ReplayError("Whole plan must cover the complete nonempty partition")
    expected = [
        *configs,
        *(
            _reference(record(item).get(key))
            for item in values
            for key in ("publication", "retirement", "retirementVerifyConfig")
        ),
    ]
    missing = [str(ref.path) for ref in expected if not ref.path.is_file() or ref.path.is_symlink()]
    missing.extend(
        str(_path(record(row).get("proof")))
        for row in rows
        if not _path(record(row).get("proof")).is_file()
    )
    if missing:
        raise ReplayError("Missing required whole metadata: " + ", ".join(missing))
    publications = [_metadata(_reference(record(value).get("publication"))) for value in values]
    replacements = {
        _reference(row.get("source")).path: _reference(row.get("movie")) for row in publications
    }
    cleanup_facts = cleanup._discover(
        {
            "version": 1,
            "semantic": {"path": str(plan.semantic), "sha256": file_digest(plan.semantic)},
            "configs": [ref.json() for ref in configs],
        }
    )
    for name, digest in cleanup_facts.inputs.items():
        _bind(inputs, technical.Reference(_path(name), digest))
    # The original cleanup consumer owns this exact domain boundary. Historical
    # producer children (for example OCR acquisition batches) are not new whole
    # consumer dependencies. Every declared current node remains pinned, and the
    # unchanged cleanup verifier still checks the full domain at execution.
    visited.update(Path(name) for name in cleanup_facts.inputs)
    parts = tuple(
        _part(
            value,
            number=number,
            row=record(row),
            facts=plan,
            assembly_ref=assembly_ref,
            inputs=inputs,
            visited=visited,
            replacements=replacements,
        )
        for number, (value, row) in enumerate(zip(values, rows, strict=True), start=1)
    )
    batch = (
        _retirement_adoption(
            document["verifiedReadonlyRetirementBatch"], parts, inputs, visited, replacements
        )
        if "verifiedReadonlyRetirementBatch" in document
        else None
    )
    existing_assembly = (
        _assembly_adoption(
            document["verifiedExistingAssembly"],
            assembly_ref=assembly_ref,
            parts=parts,
            raw=raw,
            inputs=inputs,
        )
        if "verifiedExistingAssembly" in document
        else None
    )
    voice = _reference(raw.get("voice"))
    receipt = _metadata(_reference(raw.get("voiceReceipt")))
    rate, samples, channels = (
        receipt.get("sampleRate"),
        receipt.get("outputSamples"),
        receipt.get("channels"),
    )
    if (
        type(rate) is not int
        or rate <= 0
        or type(samples) is not int
        or Fraction(samples, rate) != plan.duration
        or type(channels) is not int
        or not 1 <= channels <= 64
        or receipt.get("outputSHA256") != voice.sha256
        or (1 / plan.frame).denominator != 1
    ):
        raise ReplayError("Whole plan processed voice or integer frame clock differs")
    for ref in (*current_code, *(tool.resolved for tool in tools.values())):
        _bind(inputs, ref)
    for name in inputs:
        _regular(_path(name))
    return Facts(
        technical.Reference(config, inputs[str(config)]),
        assembly_ref,
        plan,
        voice,
        tuple(ref.path for ref in configs),
        parts,
        tools,
        inputs,
        _path(document["wholeMovie"]),
        threads,
        digest_limits,
        batch,
        existing_assembly,
    )


def _write(path: Path, value: object) -> None:
    _path(str(path), work=True)
    data = (json.dumps(value, sort_keys=True, ensure_ascii=False, indent=2) + "\n").encode("utf-8")
    if path.exists():
        if technical._bounded_bytes(path) != data:
            raise ReplayError("Existing preparation metadata differs; preserve the partial output")
        return
    with path.open("xb") as stream:
        os.fchmod(stream.fileno(), 0o600)
        stream.write(data)
        stream.flush()
        os.fsync(stream.fileno())


def _stage(
    name: str,
    command: list[str],
    inputs: list[Path],
    outputs: list[Path],
    verifier: list[str],
    *,
    adopt: bool = False,
) -> dict[str, object]:
    paths = list(dict.fromkeys(inputs))
    if len(paths) > 256:
        raise ReplayError("Whole stage direct input budget exceeded")
    return {
        "name": name,
        "command": command,
        "inputs": list(map(str, paths)),
        "outputs": list(map(str, outputs)),
        "timeout_seconds": 7200,
        "verify_command": verifier,
        "adopt_existing": adopt,
    }


def _cleanup_manifest(facts: Facts, output: Path) -> dict[str, object]:
    if not output.exists():
        cleanup.prepare(facts.cleanup_configs, facts.timeline.semantic, output)
    document = technical._metadata(output)
    expected = {
        "version": 1,
        "semantic": {
            "path": str(facts.timeline.semantic),
            "sha256": file_digest(facts.timeline.semantic),
        },
        "configs": [
            {"path": str(path), "sha256": file_digest(path)} for path in facts.cleanup_configs
        ],
    }
    if {key: document.get(key) for key in expected} != expected or set(document) != {
        *expected,
        "inputs",
    }:
        raise ReplayError("Existing cleanup preparation belongs to different inputs")
    batch = cleanup._discover(document)
    if cleanup._inventory(document["inputs"]) != batch.inputs:
        raise ReplayError("Prepared cleanup closure differs from current metadata")
    return document


def _graph(
    facts: Facts, output: Path, cleanup_path: Path
) -> tuple[dict[str, object], dict[str, list[str]], dict[str, str]]:
    inputs = dict(facts.inputs)
    for ref in _refs(technical._metadata(cleanup_path)["inputs"]):
        _bind(inputs, ref)
    _bind(inputs, technical.Reference(cleanup_path, file_digest(cleanup_path)))
    retired = {str(part.retirement.path) for part in facts.parts}
    if facts.existing_assembly is not None:
        retired.update(str(ref.path) for ref in facts.existing_assembly)
    grouped = [(name, digest) for name, digest in sorted(inputs.items()) if name not in retired]
    groups: list[Path] = []
    stages: list[dict[str, object]] = []
    tool = {name: str(value.alias) for name, value in facts.tools.items()}
    binaries = [value.resolved.path for value in facts.tools.values()]
    python = [tool["python"], "-B", "-m"]
    digest_inputs = [
        facts.tools["env"].resolved.path,
        facts.tools["python"].resolved.path,
        *_digest_code_paths(),
    ]
    for offset in range(0, len(grouped), GROUP_INPUTS):
        group = output / f"closure-{len(groups) + 1:04d}.sha256"
        content = "".join(
            f"{digest}  {name}\n" for name, digest in grouped[offset : offset + GROUP_INPUTS]
        )
        if group.exists():
            if technical._bounded_bytes(group) != content.encode("utf-8"):
                raise ReplayError("Existing closure metadata differs")
        else:
            with group.open("x", encoding="utf-8") as stream:
                stream.write(content)
        members = [Path(name) for name, digest in grouped[offset : offset + GROUP_INPUTS]]
        scopes = sorted({output, *(path.parent for path in members)})
        limits = facts.digest_limits
        command = [
            str(facts.tools["env"].resolved.path),
            f"PYTHONPATH={ROOT / 'src'}",
            str(facts.tools["python"].resolved.path),
            "-B",
            "-m",
            "replay.sha256_check",
            str(group),
            "--manifest-sha256",
            file_digest(group),
            "--workers",
            str(limits.workers),
            "--memory-budget-bytes",
            str(limits.memory_budget_bytes),
            "--baseline-rss-bytes",
            str(limits.baseline_rss_bytes),
            "--worker-rss-bytes",
            str(limits.worker_rss_bytes),
            *(argument for scope in scopes for argument in ("--scope", str(scope))),
        ]
        stages.append(
            _stage(
                f"closure-{len(groups) + 1:04d}",
                command,
                [*members, *digest_inputs],
                [group],
                command,
                adopt=True,
            )
        )
        groups.append(group)
    for group in groups:
        _bind(inputs, technical.Reference(group, file_digest(group)))
    common = [*groups, *binaries]
    clean_receipt = output / "cleanup-receipt.json"
    cleanup_args = [str(cleanup_path), str(clean_receipt), "--sha256", file_digest(cleanup_path)]
    stages.append(
        _stage(
            "all-part-local-repair",
            [*python, "replay.fcp_cleanup_adoption", "adopt", *cleanup_args],
            [*common, cleanup_path],
            [clean_receipt],
            [*python, "replay.fcp_cleanup_adoption", "verify", *cleanup_args],
        )
    )
    retire_names: list[str] = []
    if facts.retirement_batch is not None:
        request = facts.retirement_batch
        name = "all-part-readonly-retirement"
        command = [
            *python,
            "replay.part_retirement_verification_readonly",
            str(request.artifact.path),
        ]
        stages.append(
            _stage(
                name,
                command,
                [*common, clean_receipt, *(ref.path for ref in request.metadata)],
                [part.retirement.path for part in facts.parts],
                command,
                adopt=True,
            )
        )
        retire_names.append(name)
    for number, part in enumerate(facts.parts if facts.retirement_batch is None else (), start=1):
        name = f"part-{number:04d}-retirement"
        command = [
            *python,
            "replay.part_master_publication",
            str(part.verifier.path),
            "--verify-retirement",
        ]
        stages.append(
            _stage(
                name,
                command,
                [
                    *common,
                    clean_receipt,
                    part.verifier.path,
                    part.publication.path,
                    part.proof.path,
                    part.movie.path,
                ],
                [part.retirement.path],
                command,
                adopt=True,
            )
        )
        retire_names.append(name)
    xml = output / "whole-assembly.fcpxml"
    lineage = xml.with_suffix(".lineage.json")
    if facts.existing_assembly is not None:
        xml, lineage = (ref.path for ref in facts.existing_assembly)
    masters = [part.movie.path for part in facts.parts]
    proofs = [part.proof.path for part in facts.parts]
    retirements = [part.retirement.path for part in facts.parts]
    assembly_command = [
        *python,
        "replay.fcp_part_assembly",
        "assemble",
        str(facts.assembly.path),
        str(xml),
        "--name",
        "Whole delivery",
    ]
    assembly_verify = [
        *python,
        "replay.fcp_part_assembly",
        "verify-assembly",
        str(facts.assembly.path),
        str(xml),
    ]
    if facts.existing_assembly is not None:
        assembly_verify = [
            *python,
            "replay.whole_delivery_plan",
            "verify-existing-assembly",
            str(facts.config.path),
        ]
        assembly_command = assembly_verify
    stages.append(
        _stage(
            "whole-assembly",
            assembly_command,
            [*common, facts.assembly.path, clean_receipt, *masters, *proofs, *retirements],
            [xml, lineage],
            assembly_verify,
            adopt=facts.existing_assembly is not None,
        )
    )
    rate = (1 / facts.timeline.frame).numerator
    join_plan = output / "join-plan.json"
    _write(
        join_plan,
        [
            {
                "sourcePath": str(part.movie.path),
                "durationFrames": int(
                    Fraction(str(row["duration"]).removesuffix("s")) / facts.timeline.frame
                ),
            }
            for part, row in zip(facts.parts, facts.timeline.parts, strict=True)
        ],
    )
    _bind(inputs, technical.Reference(join_plan, file_digest(join_plan)))
    expected_metadata, expected_pcm = (
        output / "voice-metadata.json",
        output / "expected-pcm.streamhash",
    )
    stages.append(
        _stage(
            "expected-voice-metadata",
            [
                tool["ffprobe"],
                "-v",
                "error",
                "-show_streams",
                "-of",
                "json",
                "-o",
                str(expected_metadata),
                str(facts.voice.path),
            ],
            [*common, facts.voice.path],
            [expected_metadata],
            [],
        )
    )
    pcm_args = [tool["ffmpeg"], "-nostdin", "-v", "error", "-n", "-i"]
    pcm_suffix = ["-map", "0:a:0", "-c:a", "pcm_s32le", "-f", "streamhash", "-hash", "sha256"]
    stages.append(
        _stage(
            "expected-voice-pcm",
            [*pcm_args, str(facts.voice.path), *pcm_suffix, str(expected_pcm)],
            [*common, facts.voice.path],
            [expected_pcm],
            [],
        )
    )
    command = [
        tool["env"],
        f"PATH={facts.tools['ffmpeg'].alias.parent}:/usr/bin:/bin",
        *python,
        "replay.delivery_render",
        "join-files",
        str(join_plan),
        "--output",
        str(facts.master),
        "--rate",
        str(rate),
        "--audio",
        str(facts.voice.path),
    ]
    full_av = [
        tool["ffmpeg"],
        "-nostdin",
        "-v",
        "error",
        "-xerror",
        "-protocol_whitelist",
        "file",
        "-f",
        "mov",
        "-enable_drefs",
        "0",
        "-use_absolute_path",
        "0",
        "-i",
        str(facts.master),
        "-map",
        "0:v:0",
        "-map",
        "0:a:0",
        "-f",
        "null",
        "-",
    ]
    stages.append(
        _stage(
            "whole-lossless-join",
            command,
            [
                *common,
                xml,
                lineage,
                clean_receipt,
                join_plan,
                *masters,
                *proofs,
                *retirements,
                facts.voice.path,
                expected_metadata,
                expected_pcm,
                *technical.reader_code_paths(),
            ],
            [facts.master],
            full_av,
        )
    )
    frames, metadata, pcm = (
        output / "whole-frames.json",
        output / "whole-metadata.json",
        output / "joined-pcm.streamhash",
    )
    width, height = facts.timeline.format.get("width"), facts.timeline.format.get("height")
    frame_check = (
        "($plan[0]|map(.durationFrames)|add) as $n | .streams[0] as $s | "
        '($s.time_base|split("/")|map(tonumber)) as $tb | '
        f"($s.width=={width} and $s.height=={height} and $s.start_pts==0 "
        f'and $s.avg_frame_rate=="{rate}/1" and ($s.nb_frames|tonumber)==$n '
        f"and ($s.duration_ts*$tb[0]*{rate})==($n*$tb[1]) and (.frames|length)==$n "
        f"and (.frames|to_entries|all(.value.best_effort_timestamp*$tb[0]*{rate}==.key*$tb[1])))"
    )
    stages.append(
        _stage(
            TARGETS[0],
            [
                tool["ffprobe"],
                "-v",
                "error",
                "-threads",
                str(facts.threads),
                "-select_streams",
                "v:0",
                "-show_frames",
                "-show_streams",
                "-show_entries",
                "frame=best_effort_timestamp,duration:stream=codec_name,width,height,time_base,start_pts,duration_ts,nb_frames,avg_frame_rate",
                "-of",
                "json",
                "-o",
                str(frames),
                str(facts.master),
            ],
            [*common, facts.master, join_plan],
            [frames],
            [tool["jq"], "-e", "--slurpfile", "plan", str(join_plan), frame_check, str(frames)],
        )
    )
    metadata_check = (
        '$expected[0].streams[0] as $e | [.streams[]|select(.codec_type=="audio")] as $a | '
        "($a|length)==1 and ($a[0]|{codec_name,channels,sample_rate,time_base,duration_ts})"
        "==($e|{codec_name,channels,sample_rate,time_base,duration_ts}) and $a[0].start_pts==0"
    )
    stages.append(
        _stage(
            TARGETS[1],
            [
                tool["ffprobe"],
                "-v",
                "error",
                "-show_streams",
                "-of",
                "json",
                "-o",
                str(metadata),
                str(facts.master),
            ],
            [*common, facts.master, expected_metadata],
            [metadata],
            [
                tool["jq"],
                "-e",
                "--slurpfile",
                "expected",
                str(expected_metadata),
                metadata_check,
                str(metadata),
            ],
        )
    )
    stages.append(
        _stage(
            TARGETS[2],
            [*pcm_args, str(facts.master), *pcm_suffix, str(pcm)],
            [*common, facts.master, expected_pcm],
            [pcm],
            [tool["cmp"], str(expected_pcm), str(pcm)],
        )
    )
    if len(stages) > MAX_STAGES:
        raise ReplayError("Whole plan exceeds the existing workflow stage budget")
    document = {"version": 1, "stages": stages}
    stage_graph(parse_stages(document))
    roles = {
        "wholeJoin": ["whole-lossless-join"],
        "fullAV": ["whole-lossless-join"],
        "clock": ["whole-assembly"],
        "cleanup": ["all-part-local-repair"],
        "retirement": retire_names,
    }
    return document, roles, inputs


def prepare(config: Path, output: Path) -> Path:
    """Prepare or resume identical metadata; never run a probe or domain command."""
    facts = _facts(config)
    output = _path(str(output), work=True)
    if (
        facts.master.exists()
        or facts.master.is_relative_to(output)
        or not facts.master.parent.is_dir()
    ):
        raise ReplayError("Whole media output must be absent, separate and have an existing parent")
    if output.exists() and not output.is_dir():
        raise ReplayError("Whole preparation output must be a directory")
    generated = (
        "cleanup-receipt.json",
        "whole-assembly.fcpxml",
        "whole-assembly.lineage.json",
        "voice-metadata.json",
        "expected-pcm.streamhash",
        "whole-frames.json",
        "whole-metadata.json",
        "joined-pcm.streamhash",
    )
    if any((output / name).exists() or (output / name).is_symlink() for name in generated):
        raise ReplayError("Existing production outputs require original workflow recovery")
    output.mkdir(mode=0o700, exist_ok=True)
    _write(output / "preparation.json", {"version": 1, "config": facts.config.json()})
    clean = output / "cleanup-manifest.json"
    _cleanup_manifest(facts, clean)
    graph, roles, inputs = _graph(facts, output, clean)
    workflow = output / "workflow.json"
    _write(workflow, graph)
    metadata = {
        "version": 1,
        "config": facts.config.json(),
        "workflow": {"path": str(workflow), "sha256": file_digest(workflow)},
        "roles": roles,
        "inputs": [{"path": name, "sha256": digest} for name, digest in sorted(inputs.items())],
        "candidateOnly": True,
        "mediaBytesVerified": False,
        "qualityApproved": False,
        "syncApproved": False,
        "performanceTargetAchieved": False,
    }
    result = output / "plan.json"
    _write(result, metadata)
    _facts(config)
    load_stages(workflow)
    return result


def prepare_technical(plan: Path, *, source_state: Path, state: Path, output: Path) -> Path:
    """Bind actual completed ancestor receipts, then delegate technical preparation."""
    saved = technical._metadata(_path(str(plan), work=True))
    if (
        set(saved)
        != {
            "version",
            "config",
            "workflow",
            "roles",
            "inputs",
            "candidateOnly",
            "mediaBytesVerified",
            "qualityApproved",
            "syncApproved",
            "performanceTargetAchieved",
        }
        or type(saved["version"]) is not int
        or saved["version"] != 1
    ):
        raise ReplayError("Saved whole plan schema differs")
    config_ref = _reference(saved["config"])
    facts = _facts(config_ref.path)
    if facts.config != config_ref:
        raise ReplayError("Saved whole configuration changed")
    original = _reference(saved.get("workflow"))
    technical._metadata(original.path, original.sha256)
    stages = load_stages(original.path)
    joins = tuple(stage for stage in stages if stage.name == "whole-lossless-join")
    if (
        len(joins) != 1
        or joins[0].outputs != (facts.master,)
        or not set(technical.reader_code_paths()).issubset(joins[0].inputs)
        or joins[0].verify_command
        != (
            str(facts.tools["ffmpeg"].alias),
            "-nostdin",
            "-v",
            "error",
            "-xerror",
            "-protocol_whitelist",
            "file",
            "-f",
            "mov",
            "-enable_drefs",
            "0",
            "-use_absolute_path",
            "0",
            "-i",
            str(facts.master),
            "-map",
            "0:v:0",
            "-map",
            "0:a:0",
            "-f",
            "null",
            "-",
        )
    ):
        raise ReplayError("Saved whole graph lacks canonical original full-AV evidence binding")
    graph = stage_graph(stages)
    ancestors = technical._ancestors(graph, TARGETS)
    refs = {str(ref.path): ref.sha256 for ref in _refs(saved.get("inputs"))}
    if any(refs.get(name) != digest for name, digest in facts.inputs.items()):
        raise ReplayError("Saved whole inventory omits current required evidence")
    source_state = _path(str(source_state), work=True)
    for stage in stages:
        if stage.name not in ancestors:
            continue
        path = source_state / f"{stage.name}.json"
        receipt = record(_read_receipt(path))
        values = record(receipt.get("outputs"))
        if receipt.get("status") != "complete" or set(values) != {
            str(path) for path in stage.outputs
        }:
            raise ReplayError("Original whole ancestors are not complete")
        for name, digest in values.items():
            ref = technical.Reference(_path(name), technical._sha(digest))
            _regular(ref.path)
            _bind(refs, ref)
        _bind(refs, technical.Reference(path, file_digest(path)))
    for stage in stages:
        checkpoint = source_state / f"{stage.name}.json"
        if stage.name in TARGETS and checkpoint.exists():
            _bind(refs, technical.Reference(checkpoint, file_digest(checkpoint)))
    for stage in stages:
        for path in stage.inputs:
            if str(path) not in refs and path not in {
                out for item in stages if item.name in TARGETS for out in item.outputs
            }:
                raise ReplayError(
                    "Original workflow input is missing from the exact whole inventory"
                )
    _bind(refs, original)
    for ref in source_inventory():
        _bind(refs, ref)
    output = _path(str(output), work=True)
    output.mkdir(mode=0o700)
    config = output / "config.json"
    _write(
        config,
        {
            "version": 1,
            "workflow": original.json(),
            "sourceState": str(source_state),
            "targets": list(TARGETS),
            "state": str(_path(str(state), work=True)),
            "cwd": str(ROOT),
            "admission": {
                "roles": saved["roles"],
                "fullAVReuse": {
                    "version": 1,
                    "kind": "completed-canonical-ffmpeg-full-av",
                    "stages": ["whole-lossless-join"],
                },
                "inputs": [
                    {"path": name, "sha256": digest} for name, digest in sorted(refs.items())
                ],
            },
        },
    )
    manifest = output / "manifest.json"
    technical.prepare(config, manifest=manifest, workflow=output / "technical-workflow.json")
    return manifest


def verify_existing_assembly(config: Path) -> None:
    """Bind immutable adoption bytes, then run the original assembly domain once."""
    facts = _facts(config)
    if facts.existing_assembly is None:
        raise ReplayError("Existing assembly verification requires an explicit adoption")
    assembly.verify_assembly(facts.assembly.path, facts.existing_assembly[0].path)


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    prepare_parser = sub.add_parser("prepare")
    prepare_parser.add_argument("config", type=Path)
    prepare_parser.add_argument("output", type=Path)
    inventory_parser = sub.add_parser("source-inventory")
    inventory_parser.add_argument("output", type=Path)
    adoption_parser = sub.add_parser("verify-existing-assembly")
    adoption_parser.add_argument("config", type=Path)
    technical_parser = sub.add_parser("prepare-technical")
    technical_parser.add_argument("plan", type=Path)
    technical_parser.add_argument("--source-state", required=True, type=Path)
    technical_parser.add_argument("--state", required=True, type=Path)
    technical_parser.add_argument("--output", required=True, type=Path)
    args = Arguments()
    parser.parse_args(argv, namespace=args)
    if args.command == "source-inventory":
        _write(args.output, [ref.json() for ref in source_inventory()])
        print(args.output)
    elif args.command == "verify-existing-assembly":
        verify_existing_assembly(args.config)
        print(args.config)
    elif args.command == "prepare-technical":
        print(
            prepare_technical(
                args.plan, source_state=args.source_state, state=args.state, output=args.output
            )
        )
    else:
        print(prepare(args.config, args.output))


if __name__ == "__main__":
    main()
