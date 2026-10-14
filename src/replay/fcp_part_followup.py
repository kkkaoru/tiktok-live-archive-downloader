"""Prepare one observed native part's verification and exclusive publication graph.

This factory never operates apps, submits rendering, deletes media or infers a
successful dispatch. Explicit recovery remains the existing chronology verifier.
"""

import argparse
import sys
import time
from dataclasses import dataclass
from pathlib import Path

from replay.artifact_json import _json, _path, _pin
from replay.core import ReplayError
from replay.fcp_part_assembly import FrameProbeTool, _reference, _save, probe, verify
from replay.input_hash_cache import MAX_INPUTS, InputHashCache, input_hash_context
from replay.native_export_receipt import complete
from replay.part_master_publication import _completed, publish, verify_published
from replay.recordings import record
from replay.source_caption_binding import _consumer, clean_source

MAX_PARTS = 1024
FRAME_TIMEOUT_SECONDS = 1800
COMPLETION_TIMEOUT_SECONDS = 7400
PUBLICATION_TIMEOUT_SECONDS = 1800
CODE_NAMES = (
    "fcp_part_followup.py",
    "input_hash_cache.py",
    "fcp_part_assembly.py",
    "fcp_partition.py",
    "fcp_caption_srt.py",
    "native_export_receipt.py",
    "fcp_native_parts.py",
    "part_master_publication.py",
    "artifact_json.py",
    "source_caption_binding.py",
    "fcp_transformed_source.py",
    "delivery_source_assembly.py",
    "subtitle_clock.py",
    "fcp_project.py",
    "fcp_proxy_binding.py",
    "frame_cell_extent.py",
    "recordings.py",
    "stages.py",
    "workflow.py",
)
PIN_NAMES = (
    "assembly",
    "publishedAssembly",
    "exportRequest",
    "exportResponse",
    "background",
    "ffmpeg",
    "ffprobe",
    "sourceCleanup",
)


@dataclass(frozen=True)
class Followup:
    part: int
    paths: dict[str, Path]
    movie: Path
    proof: Path
    destination: Path
    published_proof: Path
    publication: Path
    inputs: tuple[Path, ...]
    recover: bool
    audio_rate: int
    audio_channels: int
    semantic: Path


def _member(assembly: dict[str, object], part: int) -> dict[str, object]:
    rows = assembly.get("parts")
    if not isinstance(rows, list) or not 1 <= part <= len(rows) <= MAX_PARTS:
        raise ReplayError("Followup requires an existing bounded assembly part")
    return record(rows[part - 1])


def frame_probe_tool(config_path: Path) -> FrameProbeTool:
    """Consume the existing pinned executable; zero selects automatic decoding."""
    config = _json(config_path)
    return FrameProbeTool.from_reference(
        config.get("ffprobe"), decoder_threads=config.get("probeDecoderThreads", 0)
    )


def _approved_original(
    config: dict[str, object], cleanup: dict[str, object], original: Path
) -> Path:
    """Authorize only an explicitly pinned alias of the validated historical source.

    Historical producer paths remain immutable provenance, not general filesystem
    authority. Canonical callers retain their existing contract. Alias callers
    must name the exact canonical destination independently; equal bytes at a
    different destination do not authorize retargeting the historical alias.
    """
    if "approvedOriginal" not in config:
        return _path(str(original))
    approval = record(config["approvedOriginal"])
    canonical = _path(approval.get("path"))
    manifest = _json(_pin(cleanup.get("assemblyManifest")))
    chunks = manifest.get("chunks")
    if not isinstance(chunks, list) or not 1 <= len(chunks) <= 512:
        raise ReplayError("Approved original requires bounded pinned producer evidence")
    producer = _json(_pin(record(chunks[0]).get("producer")))
    if producer.get("source") != str(original) or producer.get("sourceSHA256") != approval.get(
        "sha256"
    ):
        raise ReplayError("Approved original differs from the validated historical producer")
    if not original.is_absolute() or original.resolve() != canonical or not canonical.is_file():
        raise ReplayError("Approved original alias target changed or is not a regular file")
    _pin(approval)
    # Recheck resolution after reading bytes, including same-byte target swaps.
    if original.resolve() != canonical:
        raise ReplayError("Approved original alias target changed during validation")
    return canonical


def _load(config_path: Path, *, allow_existing: bool = False) -> Followup:
    config = _json(config_path)
    part, recover = config.get("part"), config.get("recover", False)
    if config.get("version") != 1 or type(part) is not int or part < 1 or type(recover) is not bool:
        raise ReplayError("Expected version-one part configuration and explicit recovery mode")
    paths = {name: _pin(config.get(name)) for name in PIN_NAMES}
    frame_probe_tool(config_path).verify()
    audio_rate, audio_channels = config.get("audioSampleRate"), config.get("audioChannels")
    if (
        type(audio_rate) is not int
        or audio_rate <= 0
        or type(audio_channels) is not int
        or not (1 <= audio_channels <= 64)
    ):
        raise ReplayError("Followup requires explicit valid audio properties")
    if recover:
        paths["projectObservation"] = _pin(config.get("projectObservation"))
    elif "projectObservation" in config:
        raise ReplayError("Project chronology evidence requires explicit recovery mode")
    staged, published = _json(paths["assembly"]), _json(paths["publishedAssembly"])
    if (
        staged.get("version") != 1
        or published.get("version") != 1
        or any(
            staged.get(name) != published.get(name)
            for name in ("partition", "voice", "voiceReceipt")
        )
    ):
        raise ReplayError("Staging and published assembly clocks or voice evidence differ")
    metadata = tuple(_pin(staged.get(name)) for name in ("partition", "voice", "voiceReceipt"))
    partition = _json(metadata[0])
    rows = partition.get("parts")
    if not isinstance(rows, list) or not 1 <= part <= len(rows) <= MAX_PARTS:
        raise ReplayError("Followup requires the corresponding pinned partition")
    xml, semantic = _pin(rows[part - 1]), _pin(partition.get("source"))
    cleanup = _json(paths["sourceCleanup"])
    historical_original, clean, _ = clean_source(cleanup)
    original = _approved_original(config, cleanup, historical_original)
    if _pin(cleanup.get("consumer")) != semantic:
        raise ReplayError("Followup cleanup binding differs from its semantic XML")
    _consumer(xml, "fcpxml", cleanup, clean)
    _approved_original(config, cleanup, historical_original)
    cleanup_pins = tuple(
        _pin(cleanup.get(name))
        for name in ("assemblyManifest", "assemblyReceipt", "cleanIndex", "voiceReceipt")
    )
    source, target = _member(staged, part), _member(published, part)
    movie, proof = _path(source.get("movie")), _path(source.get("proof"))
    destination, published_proof = _path(target.get("movie")), _path(target.get("proof"))
    publication = _path(config.get("receiptOutput"))
    if not movie.is_file() or movie.is_symlink() or movie.stat().st_size == 0:
        raise ReplayError("Followup needs the observed nonempty staging movie")
    if len({movie, proof, destination, published_proof, publication}) != 5 or any(
        (path.exists() and not allow_existing) or path.is_symlink() or not path.parent.is_dir()
        for path in (proof, destination, published_proof, publication)
    ):
        raise ReplayError("Followup outputs must be distinct, absent and have existing parents")
    code = tuple(Path(__file__).with_name(name) for name in CODE_NAMES)
    inputs = tuple(
        dict.fromkeys(
            (
                config_path,
                *paths.values(),
                *metadata,
                xml,
                semantic,
                movie,
                *cleanup_pins,
                original,
                clean,
                *code,
            )
        )
    )
    return Followup(
        part,
        paths,
        movie,
        proof,
        destination,
        published_proof,
        publication,
        inputs,
        recover,
        audio_rate,
        audio_channels,
        semantic,
    )


def validate_cleanup(config_path: Path, semantic: Path) -> tuple[Path, int, bool]:
    """Revalidate transitive repair evidence before work, adoption or verification.

    This intentionally permits existing outputs: completed media can be checked
    without recreating it, and workflow reuse still validates current provenance.
    """
    config_path, semantic = _path(str(config_path)), _path(str(semantic))
    config = _json(config_path)
    part, recover = config.get("part"), config.get("recover", False)
    if config.get("version") != 1 or type(part) is not int or part < 1 or type(recover) is not bool:
        raise ReplayError("Expected version-one part configuration and explicit recovery mode")
    assembly = _pin(config.get("assembly"))
    metadata = _json(assembly)
    _member(metadata, part)
    partition = _json(_pin(metadata.get("partition")))
    rows = partition.get("parts")
    if not isinstance(rows, list) or not 1 <= part <= len(rows) <= MAX_PARTS:
        raise ReplayError("Followup requires the corresponding pinned partition")
    if _pin(partition.get("source")) != semantic:
        raise ReplayError("Runtime semantic XML differs from the pinned partition")
    xml = _pin(rows[part - 1])
    cleanup = _json(_pin(config.get("sourceCleanup")))
    original, clean, _ = clean_source(cleanup)
    _approved_original(config, cleanup, original)
    if _pin(cleanup.get("consumer")) != semantic:
        raise ReplayError("Runtime cleanup binding differs from its semantic XML")
    _consumer(semantic, "fcpxml", cleanup, clean)
    _consumer(xml, "fcpxml", cleanup, clean)
    _approved_original(config, cleanup, original)
    return assembly, part, recover


def _command(module: str, *arguments: str) -> list[str]:
    return [sys.executable, "-B", "-m", f"replay.{module}", *arguments]


def prepare(config_path: Path, output: Path, *, integrated: bool = False) -> Path:
    """Generate a fresh graph only after actual response, idle and media exist."""
    config_path, output = _path(str(config_path)), _path(str(output))
    plan = _load(config_path)
    if output.exists() or output.is_symlink() or not output.parent.is_dir():
        raise ReplayError("Followup graph requires a fresh canonical output directory")
    output.mkdir(mode=0o700)
    completion_config, completion = output / "completion-config.json", output / "completion.json"
    publication_config = output / "publication-config.json"
    _save(
        completion_config,
        {
            "version": 1,
            "part": plan.part,
            "audioSampleRate": plan.audio_rate,
            "audioChannels": plan.audio_channels,
            **{
                name: _reference(path)
                for name, path in plan.paths.items()
                if name not in ("publishedAssembly", "sourceCleanup")
            },
        },
    )
    common = [str(path) for path in plan.inputs]
    published = str(plan.paths["publishedAssembly"])
    number = str(plan.part)
    runtime = (str(config_path), str(plan.semantic))
    stages: list[dict[str, object]] = [
        {
            "name": "frames",
            "command": _command("fcp_part_followup", "frames", *runtime),
            "inputs": common,
            "outputs": [str(plan.proof)],
            "timeout_seconds": FRAME_TIMEOUT_SECONDS,
            "verify_command": _command("fcp_part_followup", "verify-frames", *runtime),
        },
        {
            "name": "complete",
            "command": _command(
                "fcp_part_followup", "complete", *runtime, str(completion_config), str(completion)
            ),
            "inputs": [*common, str(completion_config), str(plan.proof)],
            "outputs": [str(completion)],
            "timeout_seconds": COMPLETION_TIMEOUT_SECONDS,
            "verify_command": [
                "/usr/bin/jq",
                "-e",
                ".completionVerified == true and .fullAVDecodeExitCode == 0 "
                "and .shareObservedIdle == true",
                str(completion),
            ],
        },
        {
            "name": "publication-config",
            "command": _command(
                "fcp_part_followup",
                "publication-config",
                *runtime,
                str(completion),
                str(plan.destination),
                str(plan.publication),
                str(plan.published_proof),
                str(publication_config),
            ),
            "inputs": [*common, str(completion), str(plan.proof)],
            "outputs": [str(publication_config)],
            "timeout_seconds": PUBLICATION_TIMEOUT_SECONDS,
        },
        {
            "name": "publish",
            "command": _command("fcp_part_followup", "publish", *runtime, str(publication_config)),
            "inputs": [*common, str(completion), str(plan.proof), str(publication_config)],
            "outputs": [str(plan.destination), str(plan.publication), str(plan.published_proof)],
            "timeout_seconds": PUBLICATION_TIMEOUT_SECONDS,
            "verify_command": _command(
                "fcp_part_followup",
                "verify-publication",
                *runtime,
                published,
                number,
                str(plan.publication),
            ),
        },
    ]
    if integrated:
        operation = output / "operation.json"
        terminal = output / "verified-followup.json"
        inventory = tuple(
            dict.fromkeys(
                (*plan.inputs, completion_config, *_cleanup_inputs(plan.paths["sourceCleanup"]))
            )
        )
        _save(
            operation,
            {
                "version": 1,
                "kind": "bounded-native-part-followup",
                "config": str(config_path),
                "semantic": str(plan.semantic),
                "completionConfig": str(completion_config),
                "inputs": [_reference(path) for path in inventory],
            },
        )
        stages = [
            {
                "name": "followup",
                "command": _command(
                    "fcp_part_followup", "run-integrated", str(operation), str(plan.semantic)
                ),
                # The operation pins the complete closure, whose runtime and
                # reuse verifiers read every referenced byte. Keep workflow's
                # existing 256 direct-path guard rather than relaxing it.
                "inputs": [*common, str(completion_config), str(operation)],
                "outputs": list(
                    map(
                        str,
                        (
                            plan.proof,
                            completion,
                            publication_config,
                            plan.destination,
                            plan.published_proof,
                            plan.publication,
                            terminal,
                        ),
                    )
                ),
                "timeout_seconds": FRAME_TIMEOUT_SECONDS
                + COMPLETION_TIMEOUT_SECONDS
                + 2 * PUBLICATION_TIMEOUT_SECONDS,
                "verify_command": _command(
                    "fcp_part_followup", "verify-integrated", str(operation), str(plan.semantic)
                ),
            }
        ]
    graph = output / "workflow.json"
    _save(
        graph,
        {
            "version": 1,
            "stages": stages,
            "sourceCaptionProduction": {
                "bindings": {stage["name"]: str(plan.paths["sourceCleanup"]) for stage in stages}
            },
        },
    )
    return graph


def _cleanup_inputs(binding: Path) -> tuple[Path, ...]:
    """Inventory only current cleanup references, never historical source strings.

    The caller also validates clean_source; these paths do not replace that
    domain verifier. Explicit references are bounded and pinned at preparation.
    """
    config = _json(binding)
    manifest = _json(_pin(config.get("assemblyManifest")))
    index = _json(_pin(manifest.get("index")))
    rows, chunks = index.get("ranges"), manifest.get("chunks")
    if (
        not isinstance(rows, list)
        or not isinstance(chunks, list)
        or not 1 <= len(rows) == len(chunks) <= 512
    ):
        raise ReplayError("Followup cleanup inventory requires bounded matching chunks")
    paths = [_pin(manifest.get("index"))]
    for row, chunk in zip(rows, chunks, strict=True):
        data = record(chunk)
        paths.extend((_pin(data.get("candidate")), _pin(data.get("producer"))))
        paths.append(_path(record(row).get("media")))
    receipt = _json(_pin(config.get("assemblyReceipt")))
    evidence = record(receipt.get("probeEvidence"))
    if not 1 <= len(evidence) <= 1024:
        raise ReplayError("Followup cleanup probe inventory is missing or excessive")
    paths.extend(_path(path) for path in evidence)
    return tuple(dict.fromkeys(paths))


def _operation(path: Path, semantic: Path) -> tuple[dict[str, object], tuple[Path, ...]]:
    path, semantic = _path(str(path)), _path(str(semantic))
    data = _json(path)
    refs = data.get("inputs")
    if (
        data.get("version") != 1
        or data.get("kind") != "bounded-native-part-followup"
        or data.get("semantic") != str(semantic)
        or not isinstance(refs, list)
        or not 1 <= len(refs) < MAX_INPUTS
    ):
        raise ReplayError("Invalid integrated followup input inventory or semantic binding")
    inputs = tuple(_path(record(ref).get("path")) for ref in refs)
    if len(set(inputs)) != len(inputs) or path in inputs:
        raise ReplayError("Integrated followup input inventory must be unique")
    return data, inputs


def _integrated_plan(
    operation: Path, semantic: Path, data: dict[str, object], inputs: tuple[Path, ...]
) -> Followup:
    refs = data["inputs"]
    if not isinstance(refs, list):
        raise ReplayError("Integrated followup references are absent")
    for ref in refs:
        _pin(ref)
    config = _path(data.get("config"))
    plan = _load(config, allow_existing=True)
    expected = {
        *plan.inputs,
        operation.parent / "completion-config.json",
        *_cleanup_inputs(plan.paths["sourceCleanup"]),
    }
    if set(inputs) != expected or data.get("completionConfig") != str(
        operation.parent / "completion-config.json"
    ):
        raise ReplayError(
            "Integrated input inventory differs from the current complete dependency closure"
        )
    validate_cleanup(config, semantic)
    return plan


def _integrated_outputs(operation: Path, plan: Followup) -> tuple[Path, ...]:
    return (
        plan.proof,
        operation.parent / "completion.json",
        operation.parent / "publication-config.json",
        plan.destination,
        plan.published_proof,
        plan.publication,
        operation.parent / "verified-followup.json",
    )


def run_integrated(operation: Path, semantic: Path) -> None:
    """Execute existing verifiers once in one process; never dispatch or retry apps.

    Partial effects are retained and require explicit reconciliation. No failed
    operation is resumed by guessing which immutable outputs are safe to reuse.
    """
    operation = _path(str(operation))
    data, inputs = _operation(operation, semantic)
    cache = InputHashCache((*inputs, operation), allow_uncached_overflow=True)
    started = time.monotonic()
    with input_hash_context(cache):
        cache.prime()
        plan = _integrated_plan(operation, semantic, data, inputs)
        outputs = _integrated_outputs(operation, plan)
        prior_effects = (
            *outputs,
            plan.publication.with_suffix(".intent.json"),
            plan.destination.with_name(f".{plan.destination.name}.publication-partial"),
        )
        if any(path.exists() or path.is_symlink() for path in prior_effects):
            raise ReplayError("Integrated followup has prior effects; reconcile before a new run")
        config = _path(data.get("config"))
        assembly = plan.paths["assembly"]
        completion, publication = outputs[1:3]
        for name in ("frames", "complete", "publication-config", "publish"):
            cache.guard()
            validate_cleanup(config, semantic)
            print(f"FOLLOWUP_SUBSTEP {name} start {time.monotonic() - started:.3f}", flush=True)
            if name == "frames":
                probe(assembly, part=plan.part, tool=frame_probe_tool(config))
                verify(assembly, part=plan.part)
            elif name == "complete":
                complete(_path(data.get("completionConfig")), completion, recover=plan.recover)
            elif name == "publication-config":
                publication_config(
                    completion,
                    destination=plan.destination,
                    receipt=plan.publication,
                    proof=plan.published_proof,
                    output=publication,
                )
            else:
                publish(publication)
                verify_publication(
                    plan.paths["publishedAssembly"], part=plan.part, publication=plan.publication
                )
            cache.guard()
            print(f"FOLLOWUP_SUBSTEP {name} complete {time.monotonic() - started:.3f}", flush=True)
        validate_cleanup(config, semantic)
        verified = cache.verify()
        _save(
            outputs[-1],
            {
                "version": 1,
                "kind": "verified-integrated-native-followup",
                "operation": _reference(operation),
                "part": plan.part,
                "inputs": verified,
                "outputs": [_reference(path) for path in outputs[:-1]],
                "hashReads": cache.hash_reads,
                "bytesRead": cache.bytes_read,
                "cacheHits": cache.cache_hits,
                "qualityApproved": False,
            },
        )


def verify_integrated(operation: Path, semantic: Path) -> None:
    """Revalidate published artifacts and all input bytes with a fresh cache."""
    operation = _path(str(operation))
    data, inputs = _operation(operation, semantic)
    cache = InputHashCache((*inputs, operation), allow_uncached_overflow=True)
    with input_hash_context(cache):
        cache.prime()
        plan = _integrated_plan(operation, semantic, data, inputs)
        outputs = _integrated_outputs(operation, plan)
        terminal = _json(outputs[-1])
        if (
            terminal.get("version") != 1
            or terminal.get("kind") != "verified-integrated-native-followup"
            or terminal.get("part") != plan.part
            or terminal.get("operation") != _reference(operation)
            or terminal.get("outputs") != [_reference(path) for path in outputs[:-1]]
            or terminal.get("qualityApproved") is not False
        ):
            raise ReplayError("Integrated followup terminal evidence differs from its operation")
        _completed(outputs[1])
        verify_publication(
            plan.paths["publishedAssembly"], part=plan.part, publication=plan.publication
        )
        validate_cleanup(_path(data.get("config")), semantic)
        if terminal.get("inputs") != cache.verify():
            raise ReplayError("Integrated followup terminal input hashes changed")


def publication_config(
    completion: Path, *, destination: Path, receipt: Path, proof: Path, output: Path
) -> None:
    """Pin only independently completed bytes; publication revalidates all pins."""
    completion, destination = _path(str(completion)), _path(str(destination))
    receipt, proof, output = _path(str(receipt)), _path(str(proof)), _path(str(output))
    _completed(completion)
    _save(
        output,
        {
            "version": 1,
            "completion": _reference(completion),
            "destination": str(destination),
            "receiptOutput": str(receipt),
            "frameProofOutput": str(proof),
        },
    )


def verify_publication(assembly: Path, *, part: int, publication: Path) -> None:
    """Keep assembly clock and original completion/intent validation mandatory."""
    proof = verify(assembly, part=part)
    verify_published(publication, proof, part=part)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    subcommands = parser.add_subparsers(dest="operation", required=True)
    create = subcommands.add_parser("prepare")
    create.add_argument("config", type=Path)
    create.add_argument("output", type=Path)
    create.add_argument("--integrated", action="store_true")
    for name in ("run-integrated", "verify-integrated"):
        integrated = subcommands.add_parser(name)
        integrated.add_argument("operation_file", type=Path)
        integrated.add_argument("semantic", type=Path)
    for name in ("frames", "verify-frames", "validate-cleanup"):
        check = subcommands.add_parser(name)
        check.add_argument("config", type=Path)
        check.add_argument("semantic", type=Path)
    finish = subcommands.add_parser("complete")
    finish.add_argument("config", type=Path)
    finish.add_argument("semantic", type=Path)
    finish.add_argument("completion_config", type=Path)
    finish.add_argument("output", type=Path)
    publish_command = subcommands.add_parser("publish")
    publish_command.add_argument("config", type=Path)
    publish_command.add_argument("semantic", type=Path)
    publish_command.add_argument("publication_config", type=Path)
    bind = subcommands.add_parser("publication-config")
    bind.add_argument("config", type=Path)
    bind.add_argument("semantic", type=Path)
    for name in ("completion", "destination", "receipt", "proof", "output"):
        bind.add_argument(name, type=Path)
    check = subcommands.add_parser("verify-publication")
    check.add_argument("config", type=Path)
    check.add_argument("semantic", type=Path)
    check.add_argument("assembly", type=Path)
    check.add_argument("part", type=int)
    check.add_argument("publication", type=Path)
    args = parser.parse_args()
    if args.operation == "prepare":
        print(prepare(args.config, args.output, integrated=args.integrated))
        return
    if args.operation in ("run-integrated", "verify-integrated"):
        function = run_integrated if args.operation == "run-integrated" else verify_integrated
        function(args.operation_file, args.semantic)
        return
    assembly, part, recover = validate_cleanup(args.config, args.semantic)
    if args.operation == "frames":
        probe(assembly, part=part, tool=frame_probe_tool(args.config))
    elif args.operation == "verify-frames":
        verify(assembly, part=part)
    elif args.operation == "complete":
        complete(args.completion_config, args.output, recover=recover)
    elif args.operation == "publish":
        publish(args.publication_config)
    elif args.operation == "publication-config":
        publication_config(
            args.completion,
            destination=args.destination,
            receipt=args.receipt,
            proof=args.proof,
            output=args.output,
        )
    elif args.operation == "verify-publication":
        if args.part != part or _pin(_json(args.config).get("publishedAssembly")) != args.assembly:
            raise ReplayError("Publication verification differs from the selected part")
        verify_publication(args.assembly, part=args.part, publication=args.publication)


if __name__ == "__main__":
    main()
