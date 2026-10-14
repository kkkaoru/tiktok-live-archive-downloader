"""Synthetic metadata preparation never reads media or operates Apple apps."""

import json
import sys
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Literal

import pytest

from replay import fcp_cleanup_adoption as cleanup
from replay import fcp_part_assembly as assembly_domain
from replay import part_retirement_verification_batch as retirement_batch
from replay import part_retirement_verification_readonly as retirement_readonly
from replay import stage_recovery, stages, workflow
from replay import whole_delivery_plan as factory
from replay import whole_technical_batch as technical
from replay.core import ReplayError
from replay.recordings import record
from replay.stages import Stage, StageError, file_digest


def completed_readonly_batch(example: "Example", monkeypatch: pytest.MonkeyPatch) -> Path:
    """Create synthetic complete metadata using the original domain discovery."""
    monkeypatch.setattr(retirement_batch, "ROOT", example.root)
    monkeypatch.setattr(retirement_batch, "CODE_NAMES", ())
    monkeypatch.setattr(retirement_batch, "reader_code_paths", lambda: (example.code,))
    manifest = example.work / "retirement-manifest.json"
    configs = tuple(
        example.work / f"verify-retirement-{number}.json" for number in range(1, example.count + 1)
    )
    retirement_batch.prepare(configs, manifest, phase="after-delete")
    phase, parts, refs = retirement_batch._manifest(manifest, file_digest(manifest))
    output = example.work / "completed-batch"
    output.mkdir()
    operation = write(
        output / "operation.json", {"version": 1, "manifest": pin(manifest), "phase": phase}
    )
    receipt = write(
        output / "receipt.json",
        {
            "version": 1,
            "manifest": pin(manifest),
            "phase": phase,
            "kind": "verified-retirement-batch",
            "parts": list(range(1, example.count + 1)),
            "inputs": {
                str(ref.path): ref.sha256
                for ref in (*refs, retirement_batch.Reference(manifest, file_digest(manifest)))
            },
            "retirementDomainVerified": True,
            "initialAndTerminalBytesVerified": True,
            "sourceAbsenceObserved": True,
            "deletionPerformed": False,
            "releaseDecision": None,
            "qualityApproved": False,
            "deliveryApproved": False,
            "physicalIOBytes": None,
            "metricScope": retirement_readonly.METRIC_SCOPE,
            "hashReads": 1,
            "bytesRead": 1,
            "cacheHits": 1,
        },
    )
    checkpoints = tuple(
        write(
            output / f"part-{part.number:04d}.json",
            {
                "version": 1,
                "manifest": pin(manifest),
                "phase": phase,
                "part": part.number,
                "verification": part.config.json(),
                "retirementDomainVerified": True,
            },
        )
        for part in parts
    )
    request = write(
        example.work / "readonly-request.json",
        {
            "version": 1,
            "kind": "readonly-retirement-verification",
            "manifest": pin(manifest),
            "operation": pin(operation),
            "receipt": pin(receipt),
            "checkpoints": [
                {"part": number, "artifact": pin(path)}
                for number, path in enumerate(checkpoints, start=1)
            ],
        },
    )
    example.change("verifiedReadonlyRetirementBatch", pin(request))
    return request


def existing_assembly(example: "Example") -> tuple[Path, Path]:
    output = example.work / "existing-whole.fcpxml"
    output.write_bytes(b"<fcpxml/>")
    raw = read(example.assembly)
    lineage = write(
        output.with_suffix(".lineage.json"),
        {
            "version": 1,
            "assemblyXML": pin(output),
            "partition": raw["partition"],
            "voice": raw["voice"],
            "voiceReceipt": raw["voiceReceipt"],
            "masterProofs": [pin(path) for path in example.proofs],
            "partAudioDisabled": True,
            "voiceOutputStart": "0s",
        },
    )
    example.change(
        "verifiedExistingAssembly",
        {"assembly": pin(example.assembly), "xml": pin(output), "lineage": pin(lineage)},
    )
    return output, lineage


def write(path: Path, value: object) -> Path:
    path.write_text(json.dumps(value, sort_keys=True), encoding="utf-8")
    return path


def read(path: Path) -> dict[str, object]:
    return record(json.loads(path.read_text(encoding="utf-8")))


def pin(path: Path) -> dict[str, str]:
    return {"path": str(path), "sha256": file_digest(path)}


def xml(path: Path, *, rate: int, width: int, texts: tuple[str, ...]) -> Path:
    clips = "".join(
        f'<asset-clip ref="a" start="0s" offset="{index}/{rate}s" duration="1/{rate}s">'
        f'<title offset="0s" duration="1/{rate}s"><text>{text}</text></title></asset-clip>'
        for index, text in enumerate(texts)
    )
    path.write_text(
        '<fcpxml><resources><format id="f" '
        f'width="{width}" height="1920" frameDuration="1/{rate}s"/></resources>'
        '<library><event><project><sequence format="f" '
        f'duration="{len(texts)}/{rate}s"><spine>{clips}</spine>'
        "</sequence></project></event></library></fcpxml>",
        encoding="utf-8",
    )
    return path


@dataclass(frozen=True)
class Example:
    root: Path
    work: Path
    config: Path
    output: Path
    assembly: Path
    partition: Path
    voice_receipt: Path
    publications: tuple[Path, ...]
    retirements: tuple[Path, ...]
    proofs: tuple[Path, ...]
    media: tuple[Path, ...]
    code: Path
    count: int
    rate: int
    width: int

    def prepare(self) -> Path:
        return factory.prepare(self.config, self.output)

    def change(self, key: str, value: object) -> None:
        document = read(self.config)
        document[key] = value
        write(self.config, document)


@pytest.fixture(params=[(2, 30, 1080), (3, 25, 640)])
def example(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, request: pytest.FixtureRequest
) -> Example:
    topology: object = request.param
    assert isinstance(topology, tuple) and len(topology) == 3
    count, rate, width = topology
    assert isinstance(count, int) and isinstance(rate, int) and isinstance(width, int)
    work = tmp_path / "edits/work"
    work.mkdir(parents=True)
    code_dir = tmp_path / "src/replay"
    code_dir.mkdir(parents=True)
    _ = [(code_dir / name).write_bytes(name.encode()) for name in cleanup.CODE_NAMES]
    code = code_dir / "synthetic_reader.py"
    code.write_bytes(b"synthetic code input\n")
    helper = code_dir / "sha256_check.py"
    initializer = code_dir / "__init__.py"
    helper.write_bytes(Path(factory.__file__).with_name("sha256_check.py").read_bytes())
    initializer.write_bytes(Path(factory.__file__).with_name("__init__.py").read_bytes())
    monkeypatch.setattr(factory, "ROOT", tmp_path)
    monkeypatch.setattr(technical, "ROOT", tmp_path)
    monkeypatch.setattr(cleanup, "ROOT", tmp_path)
    monkeypatch.setattr(technical, "reader_code_paths", lambda: (code,))
    monkeypatch.setattr(factory, "_digest_code_paths", lambda: (helper, initializer))
    monkeypatch.setattr(
        factory,
        "source_inventory",
        lambda: tuple(
            technical.Reference(path, file_digest(path)) for path in (code, helper, initializer)
        ),
    )
    voice = work / "voice.wav"
    voice.write_bytes(b"synthetic selected voice, not a real recording")
    voice_receipt = write(
        work / "voice-receipt.json",
        {
            "outputSHA256": file_digest(voice),
            "sampleRate": rate,
            "outputSamples": count,
            "channels": 2,
        },
    )
    semantic = xml(
        work / "semantic.fcpxml",
        rate=rate,
        width=width,
        texts=tuple(f"Caption {number}" for number in range(1, count + 1)),
    )
    part_xmls = tuple(
        xml(work / f"part-{number}.fcpxml", rate=rate, width=width, texts=(f"Caption {number}",))
        for number in range(1, count + 1)
    )
    partition = write(
        work / "partition.json",
        {
            "version": 1,
            "source": pin(semantic),
            "duration": f"{count}/{rate}s",
            "parts": [
                {
                    **pin(path),
                    "globalStart": f"{number}/{rate}s",
                    "duration": f"1/{rate}s",
                    "titles": 1,
                }
                for number, path in enumerate(part_xmls)
            ],
        },
    )
    media = tuple(work / f"published-{number}.mov" for number in range(1, count + 1))
    _ = [
        path.write_bytes(f"synthetic native part {number}".encode())
        for number, path in enumerate(media, start=1)
    ]
    proofs = tuple(work / f"published-proof-{number}.json" for number in range(1, count + 1))
    assembly = write(
        work / "assembly.json",
        {
            "version": 1,
            "partition": pin(partition),
            "voice": pin(voice),
            "voiceReceipt": pin(voice_receipt),
            "parts": [
                {"movie": str(movie), "proof": str(proof)}
                for movie, proof in zip(media, proofs, strict=True)
            ],
        },
    )
    chunk = work / "chunk.mov"
    chunk.write_bytes(b"synthetic locally repaired chunk")
    producer = write(work / "chunk-producer.json", {"version": 1})
    candidate = write(work / "chunk-candidate.json", {"videoSHA256": file_digest(chunk)})
    index = write(work / "chunk-index.json", {"ranges": [{"media": str(chunk)}]})
    source_manifest = write(
        work / "source-manifest.json",
        {"index": pin(index), "chunks": [{"candidate": pin(candidate), "producer": pin(producer)}]},
    )
    probe = write(work / "source-probe.json", {"version": 1})
    clean_index = write(work / "clean-index.json", {"version": 1})
    clean_receipt = write(
        work / "clean-receipt.json",
        {"output": pin(chunk), "probeEvidence": {str(probe): file_digest(probe)}},
    )
    binding = write(
        work / "cleanup-binding.json",
        {
            "consumer": {**pin(semantic), "kind": "fcpxml"},
            "assemblyManifest": pin(source_manifest),
            "assemblyReceipt": pin(clean_receipt),
            "cleanIndex": pin(clean_index),
            "voiceReceipt": pin(voice_receipt),
        },
    )
    configs = tuple(
        write(
            work / f"cleanup-part-{number}.json",
            {
                "version": 1,
                "part": number,
                "assembly": pin(assembly),
                "sourceCleanup": pin(binding),
            },
        )
        for number in range(1, count + 1)
    )

    def publication(number: int) -> tuple[Path, Path, Path]:
        movie, proof = media[number - 1], proofs[number - 1]
        historical = work / f"absent-staging-{number}.mov"
        source = {"path": str(historical), "sha256": file_digest(movie)}
        observed = {
            "streams": [
                {
                    "codec_type": "video",
                    "width": width,
                    "height": 1920,
                    "start_pts": 0,
                    "avg_frame_rate": f"{rate}/1",
                    "duration_ts": 1,
                    "time_base": f"1/{rate}",
                    "nb_read_frames": "1",
                }
            ],
            "frames": [{"best_effort_timestamp": 0}],
        }
        original = write(
            work / f"original-proof-{number}.json",
            {
                "version": 1,
                "part": number,
                "movie": source,
                "partition": pin(partition),
                "observed": observed,
            },
        )
        completed = write(
            work / f"completion-{number}.json",
            {
                "version": 1,
                "kind": "verified-native-part",
                "part": number,
                "completionVerified": True,
                "shareObservedIdle": True,
                "fullAVDecodeExitCode": 0,
                "movie": source,
                "frameProof": pin(original),
                "inputs": {str(historical): source["sha256"], str(original): file_digest(original)},
            },
        )
        intent = write(
            work / f"intent-{number}.json",
            {"source": source, "destination": str(movie), "completion": pin(completed)},
        )
        write(
            proof,
            {
                **read(original),
                "movie": pin(movie),
                "sourceFrameProof": pin(original),
                "publicationIntent": pin(intent),
            },
        )
        published = write(
            work / f"publication-{number}.json",
            {
                "version": 1,
                "kind": "published-native-part",
                "sourcePreserved": True,
                "source": source,
                "movie": pin(movie),
                "completion": pin(completed),
                "frameProof": pin(proof),
                "intent": pin(intent),
            },
        )
        retirement_config = write(
            work / f"retirement-config-{number}.json",
            {
                "version": 1,
                "retireStaging": True,
                "assembly": pin(assembly),
                "publication": pin(published),
                "receiptOutput": str(work / f"retirement-{number}.json"),
            },
        )
        retired = write(
            work / f"retirement-{number}.json",
            {
                "version": 1,
                "kind": "verified-staging-retirement",
                "part": number,
                "config": pin(retirement_config),
                "assembly": pin(assembly),
                "publication": pin(published),
                "stagingRetirementAuthorized": True,
                "deletionPerformed": False,
                "sourceIdentity": [
                    movie.stat().st_dev,
                    movie.stat().st_ino + 1,
                    movie.stat().st_size,
                    movie.stat().st_mtime_ns,
                ],
                "destinationIdentity": [
                    movie.stat().st_dev,
                    movie.stat().st_ino,
                    movie.stat().st_size,
                    movie.stat().st_mtime_ns,
                ],
                **{
                    key: read(published)[key]
                    for key in ("source", "movie", "completion", "frameProof", "intent")
                },
            },
        )
        verification = write(
            work / f"verify-retirement-{number}.json", {"version": 1, "retirement": pin(retired)}
        )
        return published, retired, verification

    publications = tuple(publication(number) for number in range(1, count + 1))
    tool = Path(sys.executable).resolve()
    config = write(
        work / "plan-config.json",
        {
            "version": 1,
            "assembly": pin(assembly),
            "cleanupConfigs": [pin(path) for path in configs],
            "parts": [
                {
                    "part": number,
                    "publication": pin(pub),
                    "retirement": pin(ret),
                    "retirementVerifyConfig": pin(verify),
                }
                for number, (pub, ret, verify) in enumerate(publications, start=1)
            ],
            "tools": {
                name: pin(Path("/usr/bin/env").resolve() if name == "env" else tool)
                for name in factory.TOOL_NAMES
            },
            "sourceInventory": [pin(code), pin(helper), pin(initializer)],
            "wholeMovie": str(work / "whole-master.mov"),
            "decoderThreads": 0,
            "closureDigest": {
                "workers": 4,
                "memoryBudgetBytes": 268435456,
                "baselineRSSBytes": 67108864,
                "workerRSSBytes": 16777216,
            },
        },
    )
    return Example(
        tmp_path,
        work,
        config,
        work / "prepared",
        assembly,
        partition,
        voice_receipt,
        tuple(row[0] for row in publications),
        tuple(row[1] for row in publications),
        proofs,
        (*media, chunk, voice),
        code,
        count,
        rate,
        width,
    )


def graph(example: Example) -> tuple[Path, tuple[Stage, ...]]:
    plan = example.prepare()
    path = Path(str(record(read(plan)["workflow"])["path"]))
    return plan, workflow.load_stages(path)


def test_adopt_complete_readonly_batch_and_existing_assembly_once(
    example: Example, monkeypatch: pytest.MonkeyPatch
) -> None:
    request = completed_readonly_batch(example, monkeypatch)
    xml_path, lineage = existing_assembly(example)
    original_request = request.read_bytes()
    plan, stages_ = graph(example)
    retirement = next(stage for stage in stages_ if stage.name == "all-part-readonly-retirement")
    assembled = next(stage for stage in stages_ if stage.name == "whole-assembly")
    assert retirement.adopt_existing is True
    assert retirement.command == retirement.verify_command
    assert retirement.command[-2:] == ("replay.part_retirement_verification_readonly", str(request))
    assert retirement.outputs == example.retirements
    assert not any(stage.name.startswith("part-") for stage in stages_)
    assert assembled.adopt_existing is True
    assert assembled.command == assembled.verify_command
    assert "verify-existing-assembly" in assembled.command
    assert "assemble" not in assembled.command
    assert assembled.outputs == (xml_path, lineage)
    assert record(read(plan)["roles"])["retirement"] == ["all-part-readonly-retirement"]
    assert request.read_bytes() == original_request
    assert example.prepare() == plan


def test_existing_assembly_wrapper_checks_pins_then_original_domain_once(
    example: Example, monkeypatch: pytest.MonkeyPatch
) -> None:
    output, lineage = existing_assembly(example)
    calls: list[tuple[Path, Path]] = []
    monkeypatch.setattr(
        assembly_domain, "verify_assembly", lambda config, xml: calls.append((config, xml))
    )
    factory.main(["verify-existing-assembly", str(example.config)])
    assert calls == [(example.assembly, output)]
    # Even coordinated replacement of XML and its original lineage cannot change
    # the pinned adoption after plan creation.
    output.write_bytes(b"<changed/>")
    write(lineage, {**read(lineage), "assemblyXML": pin(output)})
    with pytest.raises(ReplayError):
        factory.verify_existing_assembly(example.config)
    assert len(calls) == 1


def test_assembly_wrapper_requires_explicit_hook(example: Example) -> None:
    with pytest.raises(ReplayError, match="explicit adoption"):
        factory.verify_existing_assembly(example.config)


def test_pinned_observation_array_collects_nested_references(example: Example) -> None:
    leaf = write(example.work / "observation-child.json", {"width": 640})
    array = write(
        example.work / "observations.json",
        [{"actualTimeSeconds": 1, "lines": [], "nested": [pin(leaf)]}],
    )
    inputs: dict[str, str] = {}
    result = factory._collect(technical.Reference(array, file_digest(array)), inputs, set(), {})
    assert result == {}
    assert set(inputs) == {str(array), str(leaf)}
    # Domain object readers remain strict even though closure traversal supports arrays.
    with pytest.raises(ReplayError):
        factory._metadata(technical.Reference(array, file_digest(array)))


@pytest.mark.parametrize("value", [[{"path": "relative.json", "sha256": "0" * 64}], 17, None])
def test_collection_rejects_invalid_array_reference_and_scalar_roots(
    example: Example, value: object
) -> None:
    path = write(example.work / "invalid-closure.json", value)
    with pytest.raises(ReplayError):
        factory._collect(technical.Reference(path, file_digest(path)), {}, set(), {})


def test_scalar_clock_array_is_not_a_file_reference_inventory(example: Example) -> None:
    path = write(example.work / "clock-array.json", {"timestamps": list(range(5000))})
    inputs: dict[str, str] = {}
    factory._collect(technical.Reference(path, file_digest(path)), inputs, set(), {})
    assert len(inputs) == 1


def test_original_cleanup_boundary_preserves_producer_without_reopening_history(
    example: Example,
) -> None:
    producer = example.work / "chunk-producer.json"
    write(
        producer,
        {
            "version": 1,
            "historicalOCR": {
                "path": str(example.work / "absent-old-ocr.json"),
                "sha256": "0" * 64,
            },
        },
    )
    manifest = example.work / "source-manifest.json"
    document = read(manifest)
    chunks = document["chunks"]
    assert isinstance(chunks, list)
    chunks[0] = {**record(chunks[0]), "producer": pin(producer)}
    write(manifest, document)
    binding = example.work / "cleanup-binding.json"
    write(binding, {**read(binding), "assemblyManifest": pin(manifest)})
    config = read(example.config)
    references = config["cleanupConfigs"]
    assert isinstance(references, list)
    paths = tuple(Path(str(record(ref)["path"])) for ref in references)
    _ = [write(path, {**read(path), "sourceCleanup": pin(binding)}) for path in paths]
    example.change("cleanupConfigs", [pin(path) for path in paths])
    saved, stages_ = graph(example)
    inventory = read(saved)["inputs"]
    assert isinstance(inventory, list)
    names = {str(record(ref)["path"]) for ref in inventory}
    assert str(producer) in names
    assert str(example.work / "absent-old-ocr.json") not in names
    stage = next(item for item in stages_ if item.name == "all-part-local-repair")
    assert "replay.fcp_cleanup_adoption" in stage.verify_command
    assert "verify" in stage.verify_command


@pytest.mark.parametrize("missing", [False, True])
def test_original_cleanup_domain_still_rejects_changed_required_input(
    example: Example, missing: bool
) -> None:
    candidate = example.work / "chunk-candidate.json"
    if missing:
        candidate.unlink()
    else:
        write(candidate, {"videoSHA256": "0" * 64})
    with pytest.raises((ReplayError, FileNotFoundError)):
        example.prepare()


def test_distinct_parent_scopes_are_segmented_without_dropping_pins(example: Example) -> None:
    facts = factory._facts(example.config)
    parents = tuple(example.work / f"parent-{number:04d}" for number in range(220))
    tuple(map(Path.mkdir, parents))
    leaves = tuple(write(path / "evidence.json", {"version": 1}) for path in parents)
    expanded = replace(
        facts, inputs={**facts.inputs, **{str(path): file_digest(path) for path in leaves}}
    )
    example.output.mkdir()
    clean = example.output / "cleanup-manifest.json"
    factory._cleanup_manifest(expanded, clean)
    document, _, inputs = factory._graph(expanded, example.output, clean)
    parsed = workflow.parse_stages(document)
    workflow.stage_graph(parsed)
    closures = tuple(stage for stage in parsed if stage.name.startswith("closure-"))
    assert max(len(stage.command) for stage in closures) <= 256
    assert max(len(stage.inputs) for stage in closures) <= 256
    assert len(parsed) <= 512
    assert {str(path) for path in leaves}.issubset(inputs)
    assert set(leaves).issubset({path for stage in closures for path in stage.inputs})
    assert len([path for stage in closures for path in stage.inputs if path in leaves]) == 220


def test_collection_array_item_limit(example: Example, monkeypatch: pytest.MonkeyPatch) -> None:
    path = write(example.work / "too-many-observations.json", [{}, {}, {}])
    monkeypatch.setattr(factory, "MAX_METADATA_ITEMS", 2)
    with pytest.raises(ReplayError, match="array exceeds"):
        factory._collect(technical.Reference(path, file_digest(path)), {}, set(), {})


def test_collection_preserves_byte_limit(example: Example, monkeypatch: pytest.MonkeyPatch) -> None:
    path = write(example.work / "too-large-observations.json", [{"text": "abcdef"}])
    monkeypatch.setattr(technical, "MAX_METADATA_BYTES", 4)
    with pytest.raises(ReplayError):
        factory._collect(technical.Reference(path, file_digest(path)), {}, set(), {})


@pytest.mark.parametrize("payload", ["[", '[{"x":1,"x":2}]', '[{"x":NaN}]'])
def test_collection_strict_json_parsing(example: Example, payload: str) -> None:
    path = example.work / "malformed-array.json"
    path.write_text(payload, encoding="utf-8")
    with pytest.raises((ReplayError, json.JSONDecodeError)):
        factory._collect(technical.Reference(path, file_digest(path)), {}, set(), {})


@pytest.mark.parametrize(
    "change", ["order", "missing", "operation", "receipt", "checkpoint", "stale"]
)
def test_readonly_batch_rejects_incomplete_changed_or_reordered_metadata(
    example: Example, monkeypatch: pytest.MonkeyPatch, change: str
) -> None:
    request = completed_readonly_batch(example, monkeypatch)
    document = read(request)
    if change in {"order", "missing"}:
        checkpoints = document["checkpoints"]
        assert isinstance(checkpoints, list)
        document["checkpoints"] = (
            list(reversed(checkpoints)) if change == "order" else checkpoints[:-1]
        )
    elif change == "stale":
        write(
            example.work / "verify-retirement-1.json",
            {"version": 1, "retirement": pin(example.retirements[-1])},
        )
    elif change == "checkpoint":
        checkpoints = document["checkpoints"]
        assert isinstance(checkpoints, list)
        first = record(checkpoints[0])
        path = Path(str(record(first["artifact"])["path"]))
        write(path, {**read(path), "retirementDomainVerified": False})
        checkpoints[0] = {**first, "artifact": pin(path)}
    else:
        path = Path(str(record(document[change])["path"]))
        write(path, {**read(path), "phase": "before-delete"})
        document[change] = pin(path)
    write(request, document)
    example.change("verifiedReadonlyRetirementBatch", pin(request))
    with pytest.raises(ReplayError):
        example.prepare()


@pytest.mark.parametrize(
    "change", ["assembly", "lineage-path", "xml-bytes", "proof-order", "voice"]
)
def test_existing_assembly_requires_exact_current_provenance(example: Example, change: str) -> None:
    xml_path, lineage = existing_assembly(example)
    config = read(example.config)
    hook = record(config["verifiedExistingAssembly"])
    if change == "assembly":
        hook["assembly"] = pin(example.partition)
    elif change == "lineage-path":
        other = write(example.work / "other-lineage.json", read(lineage))
        hook["lineage"] = pin(other)
    elif change == "xml-bytes":
        xml_path.write_bytes(b"<changed/>")
    else:
        document = read(lineage)
        document["masterProofs" if change == "proof-order" else "voice"] = [
            pin(path) for path in reversed(example.proofs)
        ]
        write(lineage, document)
        hook["lineage"] = pin(lineage)
    example.change("verifiedExistingAssembly", hook)
    with pytest.raises(ReplayError):
        example.prepare()


def test_prepare_covers_every_part_with_current_domains_without_media_hashes(
    example: Example, monkeypatch: pytest.MonkeyPatch
) -> None:
    original = file_digest
    seen: list[Path] = []

    def digest(path: Path) -> str:
        if path in example.media:
            pytest.fail("Metadata preparation read media bytes")
        seen.append(path)
        return original(path)

    monkeypatch.setattr(factory, "file_digest", digest)
    plan, stages_ = graph(example)
    document = read(plan)
    roles = record(document["roles"])
    assert roles["retirement"] == [
        f"part-{number:04d}-retirement" for number in range(1, example.count + 1)
    ]
    assert document["candidateOnly"] is True
    assert document["mediaBytesVerified"] is False
    assert document["performanceTargetAchieved"] is False
    assert len(stages_) >= example.count + 8
    declared = document["inputs"]
    assert isinstance(declared, list)
    assert {str(path) for path in example.media}.issubset(
        {str(record(row)["path"]) for row in declared}
    )
    assert not any("absent-staging" in str(record(row)["path"]) for row in declared)
    target = next(stage for stage in stages_ if stage.name == "whole-frame-clock")
    join = next(stage for stage in stages_ if stage.name == "whole-lossless-join")
    assert example.code in join.inputs
    assert "-protocol_whitelist" in join.verify_command
    assert join.verify_command[join.verify_command.index("-enable_drefs") + 1] == "0"
    assert target.command[target.command.index("-threads") + 1] == "0"
    assert f"$s.width=={example.width}" in target.verify_command[-2]
    assert f'$s.avg_frame_rate=="{example.rate}/1"' in target.verify_command[-2]
    assert (
        workflow.main(
            [
                str(example.output / "workflow.json"),
                "--state",
                str(example.work / "state"),
                "--check",
            ]
        )
        == 0
    )
    assert seen


def test_byte_identical_partial_preparation_is_reused(
    example: Example, monkeypatch: pytest.MonkeyPatch
) -> None:
    ordinary = cleanup.prepare

    def interrupted(configs: tuple[Path, ...], semantic: Path, output: Path) -> None:
        ordinary(configs, semantic, output)
        raise RuntimeError("synthetic stop after metadata publication")

    monkeypatch.setattr(cleanup, "prepare", interrupted)
    with pytest.raises(RuntimeError, match="synthetic stop"):
        example.prepare()
    before = (example.output / "cleanup-manifest.json").read_bytes()
    monkeypatch.setattr(
        cleanup, "prepare", lambda *args: pytest.fail("Completed metadata repeated")
    )
    plan = example.prepare()
    assert (example.output / "cleanup-manifest.json").read_bytes() == before
    plan_before = plan.read_bytes()
    assert example.prepare() == plan
    assert plan.read_bytes() == plan_before


@pytest.mark.parametrize(
    "artifact",
    [
        "preparation.json",
        "closure-0001.sha256",
        "cleanup-manifest.json",
        "join-plan.json",
        "workflow.json",
        "plan.json",
    ],
)
def test_changed_partial_metadata_is_never_overwritten(example: Example, artifact: str) -> None:
    example.prepare()
    target = example.output / artifact
    original = target.read_bytes()
    if artifact == "cleanup-manifest.json":
        altered = read(target)
        altered["version"] = 2
        write(target, altered)
    else:
        target.write_bytes(original + b" ")
    before = target.read_bytes()
    with pytest.raises(ReplayError):
        example.prepare()
    assert target.read_bytes() == before


@pytest.mark.parametrize(
    "key,value",
    [
        ("version", True),
        ("version", 2),
        ("parts", []),
        ("cleanupConfigs", []),
        ("decoderThreads", True),
        ("decoderThreads", 65),
        ("sourceInventory", []),
        ("unexpected", True),
    ],
)
def test_illegal_configuration_is_rejected_before_writes(
    example: Example, key: str, value: object
) -> None:
    example.change(key, value)
    with pytest.raises(ReplayError):
        example.prepare()
    assert not example.output.exists()


def test_missing_retirements_reports_all_paths_without_fabrication(example: Example) -> None:
    list(map(Path.unlink, example.retirements))
    with pytest.raises(ReplayError, match="Missing required") as error:
        example.prepare()
    assert all(str(path) in str(error.value) for path in example.retirements)
    assert not example.output.exists()


def test_part_identity_mismatch_is_rejected(example: Example) -> None:
    doc = read(example.config)
    rows = doc["parts"]
    assert isinstance(rows, list)
    rows[0] = {**record(rows[0]), "part": 2}
    write(example.config, doc)
    with pytest.raises(ReplayError, match="ordered part"):
        example.prepare()


def test_declared_cleanup_pin_cannot_be_silently_refreshed(example: Example) -> None:
    document = read(example.config)
    refs = document["cleanupConfigs"]
    assert isinstance(refs, list)
    refs[0] = {**record(refs[0]), "sha256": "a" * 64}
    write(example.config, document)
    with pytest.raises(ReplayError, match="fingerprint changed"):
        example.prepare()
    assert not example.output.exists()


def test_missing_cleanup_producer_refuses_before_preparation_side_effects(example: Example) -> None:
    (example.work / "chunk-producer.json").unlink()
    with pytest.raises(FileNotFoundError):
        example.prepare()
    assert not example.output.exists()


@pytest.mark.parametrize(
    "field,value",
    [("width", 900), ("start_pts", 1), ("nb_read_frames", "2"), ("avg_frame_rate", "60/1")],
)
def test_changed_native_geometry_or_clock_is_rejected(
    example: Example, field: str, value: object
) -> None:
    document = read(example.proofs[0])
    observed = record(document["observed"])
    streams_ = observed["streams"]
    assert isinstance(streams_, list)
    record(streams_[0])[field] = value
    write(example.proofs[0], document)
    with pytest.raises(ReplayError, match="fingerprint"):
        example.prepare()


def test_source_inventory_change_is_rejected(example: Example) -> None:
    example.code.write_bytes(b"changed current reader")
    with pytest.raises(ReplayError, match="current source inventory"):
        example.prepare()


@pytest.mark.parametrize("kind", ["symlink", "private", "downloads", "parent", "relative"])
def test_metadata_scope_rejects_before_read(example: Example, kind: str) -> None:
    target = {
        "symlink": example.work / "linked.json",
        "private": example.root / "private/x.json",
        "downloads": example.root / "downloads/x.json",
        "parent": example.work / "../x.json",
        "relative": Path("relative.json"),
    }[kind]
    if kind == "symlink":
        target.symlink_to(example.assembly)
    example.change("assembly", {"path": str(target), "sha256": "a" * 64})
    with pytest.raises(ReplayError):
        example.prepare()
    assert not example.output.exists()


@pytest.mark.parametrize("tool_problem", ["missing", "extra", "changed", "relative"])
def test_tool_contract_is_pinned_and_complete(example: Example, tool_problem: str) -> None:
    tools = record(read(example.config)["tools"])
    if tool_problem == "missing":
        tools.pop("ffprobe")
    elif tool_problem == "extra":
        tools["unused"] = tools["ffprobe"]
    elif tool_problem == "changed":
        tools["ffprobe"] = {**record(tools["ffprobe"]), "sha256": "a" * 64}
    else:
        tools["ffprobe"] = {"path": "ffprobe", "sha256": "a" * 64}
    example.change("tools", tools)
    with pytest.raises(ReplayError):
        example.prepare()


def test_exclusive_master_and_output_directory_are_required(example: Example) -> None:
    master = Path(str(read(example.config)["wholeMovie"]))
    master.write_bytes(b"unowned output")
    with pytest.raises(ReplayError, match="absent"):
        example.prepare()
    assert master.read_bytes() == b"unowned output"


def test_current_inventory_has_every_transitive_reader(monkeypatch: pytest.MonkeyPatch) -> None:
    small = Path(factory.__file__).resolve()
    monkeypatch.setattr(technical, "reader_code_paths", lambda: (small,))
    refs = factory.source_inventory()
    paths = {ref.path for ref in refs}
    assert small in paths
    assert factory.ROOT / "src/replay/sha256_check.py" in paths
    assert factory.ROOT / "src/replay/__init__.py" in paths
    assert all(factory.ROOT / "src/replay" / name in paths for name in cleanup.CODE_NAMES)


def test_closure_uses_bounded_sha_reader_with_explicit_code_tool_and_manifest_pins(
    example: Example,
) -> None:
    _, stages_ = graph(example)
    closure = next(stage for stage in stages_ if stage.name == "closure-0001")
    assert closure.command[:6] == (
        str(Path("/usr/bin/env").resolve()),
        f"PYTHONPATH={example.root / 'src'}",
        str(Path(sys.executable).resolve()),
        "-B",
        "-m",
        "replay.sha256_check",
    )
    assert closure.command[closure.command.index("--manifest-sha256") + 1] == file_digest(
        closure.outputs[0]
    )
    assert closure.command[closure.command.index("--workers") + 1] == "4"
    assert closure.command[closure.command.index("--memory-budget-bytes") + 1] == "268435456"
    assert closure.command[closure.command.index("--baseline-rss-bytes") + 1] == "67108864"
    assert closure.command[closure.command.index("--worker-rss-bytes") + 1] == "16777216"
    assert closure.verify_command == closure.command
    assert closure.adopt_existing is True
    assert closure.timeout_seconds == 7200
    assert Path(sys.executable).resolve() in closure.inputs
    assert Path("/usr/bin/env").resolve() in closure.inputs
    assert example.root / "src/replay/sha256_check.py" in closure.inputs
    assert example.root / "src/replay/__init__.py" in closure.inputs
    arguments = iter(closure.command[closure.command.index("--scope") :])
    scopes = tuple(Path(path) for flag, path in zip(arguments, arguments, strict=True))
    assert example.output in scopes
    assert all(path.parent in scopes for path in closure.inputs)


@pytest.mark.parametrize(
    "limits",
    [
        None,
        {},
        {"workers": 1},
        {
            "workers": True,
            "memoryBudgetBytes": 268435456,
            "baselineRSSBytes": 67108864,
            "workerRSSBytes": 16777216,
        },
        {
            "workers": 5,
            "memoryBudgetBytes": 268435456,
            "baselineRSSBytes": 67108864,
            "workerRSSBytes": 16777216,
        },
        {
            "workers": 1,
            "memoryBudgetBytes": 67108864,
            "baselineRSSBytes": 67108864,
            "workerRSSBytes": 16777216,
        },
    ],
)
def test_explicit_closure_memory_limits_refuse_before_metadata_writes(
    example: Example, limits: object
) -> None:
    example.change("closureDigest", limits)
    with pytest.raises((ReplayError, ValueError)):
        example.prepare()
    assert not example.output.exists()


def test_generated_closure_command_verifies_real_synthetic_bytes_and_reuses_checkpoint(
    example: Example,
) -> None:
    _, stages_ = graph(example)
    closure = next(stage for stage in stages_ if stage.name == "closure-0001")
    state = example.work / "closure-state"
    source = example.output / "workflow.json"
    workflow.run_plan(source, state_directory=state, targets=(closure.name,))
    checkpoint = (state / "closure-0001.json").read_bytes()
    group = closure.outputs[0].read_bytes()
    assert read(state / "closure-0001.json")["status"] == "complete"
    workflow.run_plan(source, state_directory=state, targets=(closure.name,))
    assert (state / "closure-0001.json").read_bytes() == checkpoint
    assert closure.outputs[0].read_bytes() == group


def test_changed_closure_manifest_cannot_be_adopted_under_original_pin(example: Example) -> None:
    _, stages_ = graph(example)
    closure = next(stage for stage in stages_ if stage.name == "closure-0001")
    closure.outputs[0].write_bytes(closure.outputs[0].read_bytes() + b"\n")
    before = closure.outputs[0].read_bytes()
    with pytest.raises(StageError):
        workflow.run_plan(
            example.output / "workflow.json",
            state_directory=example.work / "bad-closure-state",
            targets=(closure.name,),
        )
    assert closure.outputs[0].read_bytes() == before


def test_completed_closure_refuses_helper_code_mutation(example: Example) -> None:
    _, stages_ = graph(example)
    closure = next(stage for stage in stages_ if stage.name == "closure-0001")
    state = example.work / "helper-state"
    workflow.run_plan(
        example.output / "workflow.json", state_directory=state, targets=(closure.name,)
    )
    checkpoint = (state / "closure-0001.json").read_bytes()
    (example.root / "src/replay/sha256_check.py").write_bytes(b"changed synthetic SHA helper")
    with pytest.raises(StageError, match="Completed stage"):
        workflow.run_plan(
            example.output / "workflow.json", state_directory=state, targets=(closure.name,)
        )
    assert (state / "closure-0001.json").read_bytes() == checkpoint


def test_group_recovery_preserves_prefix_and_reuses_identical_prepared_metadata(
    example: Example, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(factory, "GROUP_INPUTS", 8)
    plan, stages_ = graph(example)
    first = next(stage for stage in stages_ if stage.name == "closure-0001")
    second = next(stage for stage in stages_ if stage.name == "closure-0002")
    state = example.work / "group-state"
    source = example.output / "workflow.json"
    first_bytes, second_bytes = first.outputs[0].read_bytes(), second.outputs[0].read_bytes()
    interrupted = True

    def execute(command: tuple[str, ...], log: Path, timeout: int) -> None:
        if interrupted and command == second.verify_command:
            raise RuntimeError("synthetic interrupted group verifier")
        stages.execute_command(command, log, timeout)

    def run_stage(
        stage: Stage, *, state_directory: Path
    ) -> Literal["completed", "reused", "adopted", "busy"]:
        return stages.run_stage(stage, state_directory=state_directory, execute=execute)

    monkeypatch.setattr(workflow, "run_stage", run_stage)
    with pytest.raises(RuntimeError, match="interrupted group"):
        workflow.run_plan(source, state_directory=state, targets=(first.name, second.name))
    checkpoint = (state / "closure-0001.json").read_bytes()
    with stages.execution_lock(state, ".workflow") as workflow_free:
        assert workflow_free
        with stages.execution_lock(state, second.name) as stage_free:
            assert stage_free
            preserved = stage_recovery.recover(source, state)
    assert len(preserved) == 2
    assert not second.outputs[0].exists()
    assert first.outputs[0].read_bytes() == first_bytes
    assert (state / "closure-0001.json").read_bytes() == checkpoint
    assert example.prepare() == plan
    assert second.outputs[0].read_bytes() == second_bytes
    interrupted = False
    workflow.run_plan(source, state_directory=state, targets=(first.name, second.name))
    assert (state / "closure-0001.json").read_bytes() == checkpoint
    assert first.outputs[0].read_bytes() == first_bytes
    assert second.outputs[0].read_bytes() == second_bytes
    assert read(state / "closure-0002.json")["status"] == "complete"


def test_reference_conflict_and_bounds_are_preserved(example: Example) -> None:
    refs = {str(example.code): "a" * 64}
    with pytest.raises(ReplayError, match="conflicting"):
        factory._bind(refs, technical.Reference(example.code, "b" * 64))
    with pytest.raises(ReplayError, match="unique"):
        factory._refs([pin(example.code), pin(example.code)])
    with pytest.raises(ReplayError, match="nesting"):
        list(factory._nested_refs({}, depth=65))
    with pytest.raises(ReplayError, match="canonical"):
        factory._path(42)


def test_stage_budget_keeps_existing_workflow_limit(example: Example) -> None:
    with pytest.raises(ReplayError, match="input budget"):
        factory._stage(
            "oversized",
            [str(example.code)],
            [example.work / str(number) for number in range(257)],
            [],
            [],
        )


def test_cli_uses_shared_prepare(example: Example, capsys: pytest.CaptureFixture[str]) -> None:
    factory.main(["prepare", str(example.config), str(example.output)])
    assert capsys.readouterr().out.strip() == str(example.output / "plan.json")


class SyntheticExecutor:
    """Model native effects while using the real workflow/checkpoint/recovery core."""

    def __init__(self, stages_: tuple[Stage, ...]) -> None:
        self.stages = stages_
        self.calls: list[str] = []
        self.interrupt = False

    def execute(self, command: tuple[str, ...], log: Path, timeout: int) -> None:
        verified = next((stage for stage in self.stages if command == stage.verify_command), None)
        if verified is not None:
            if any(
                not path.is_file() or path.read_bytes() == b"partial" for path in verified.outputs
            ):
                raise StageError("Synthetic domain refused uncertain output")
            return
        stage = next(stage for stage in self.stages if command == stage.command)
        self.calls.append(stage.name)
        _ = [
            path.write_bytes(
                b"partial"
                if self.interrupt and stage.name == "whole-metadata"
                else f"synthetic {stage.name}".encode()
            )
            for path in stage.outputs
        ]
        if self.interrupt and stage.name == "whole-metadata":
            raise RuntimeError("synthetic interrupted whole metadata")

    def run_stage(
        self, stage: Stage, *, state_directory: Path
    ) -> Literal["completed", "reused", "adopted", "busy"]:
        return stages.run_stage(stage, state_directory=state_directory, execute=self.execute)


def complete_ancestors(example: Example, monkeypatch: pytest.MonkeyPatch) -> tuple[Path, Path]:
    plan, stages_ = graph(example)
    executor = SyntheticExecutor(stages_)
    monkeypatch.setattr(workflow, "run_stage", executor.run_stage)
    state = example.work / "original-state"
    workflow.run_plan(
        example.output / "workflow.json", state_directory=state, targets=("whole-lossless-join",)
    )
    return plan, state


def test_technical_preparation_delegates_exact_original_stages_without_media_hash(
    example: Example, monkeypatch: pytest.MonkeyPatch
) -> None:
    plan, original_state = complete_ancestors(example, monkeypatch)
    original = file_digest

    def digest(path: Path) -> str:
        if path.suffix in {".mov", ".wav"}:
            pytest.fail("Technical metadata preparation hashed media")
        return original(path)

    monkeypatch.setattr(factory, "file_digest", digest)
    manifest = factory.prepare_technical(
        plan,
        source_state=original_state,
        state=example.work / "technical-state",
        output=example.work / "technical",
    )
    prepared = technical.load(manifest, file_digest(manifest))
    original_stages = workflow.load_stages(example.output / "workflow.json")
    assert prepared.plan.stages == tuple(
        stage for stage in original_stages if stage.name in factory.TARGETS
    )
    assert len(prepared.plan.stages) == 3
    config_document = read(example.work / "technical/config.json")
    assert record(config_document["admission"])["fullAVReuse"] == {
        "version": 1,
        "kind": "completed-canonical-ffmpeg-full-av",
        "stages": ["whole-lossless-join"],
    }
    assert all(stage.verify_command for stage in prepared.plan.stages)
    assert not (example.work / "technical-state").exists()
    assert not (example.output / "whole-frames.json").exists()


def test_actual_workflow_interruption_recovery_keeps_complete_prefix_and_bytes(
    example: Example, monkeypatch: pytest.MonkeyPatch
) -> None:
    _, stages_ = graph(example)
    executor = SyntheticExecutor(stages_)
    executor.interrupt = True
    monkeypatch.setattr(workflow, "run_stage", executor.run_stage)
    source = example.output / "workflow.json"
    state = example.work / "original-state"
    with pytest.raises(RuntimeError, match="interrupted"):
        workflow.run_plan(source, state_directory=state, targets=factory.TARGETS)
    prefix = (state / "whole-frame-clock.json").read_bytes()
    frames = (example.output / "whole-frames.json").read_bytes()
    publication = example.publications[0].read_bytes()
    earlier_calls = executor.calls.copy()
    with pytest.raises(StageError, match="uncertain"):
        workflow.run_plan(source, state_directory=state, targets=factory.TARGETS)
    assert executor.calls == earlier_calls
    with stages.execution_lock(state, ".workflow") as free:
        assert free
        with stages.execution_lock(state, "whole-metadata") as stage_free:
            assert stage_free
            preserved = stage_recovery.recover(source, state)
    assert len(preserved) == 2
    assert (state / "whole-frame-clock.json").read_bytes() == prefix
    assert (example.output / "whole-frames.json").read_bytes() == frames
    assert example.publications[0].read_bytes() == publication
    executor.calls.clear()
    executor.interrupt = False
    workflow.run_plan(source, state_directory=state, targets=factory.TARGETS)
    assert executor.calls == ["whole-metadata", "joined-pcm-hash"]
    assert (state / "whole-frame-clock.json").read_bytes() == prefix
    assert (example.output / "whole-frames.json").read_bytes() == frames
    executor.calls.clear()
    workflow.run_plan(source, state_directory=state, targets=factory.TARGETS)
    assert executor.calls == []


@pytest.mark.parametrize("tamper", ["schema", "inventory", "config", "workflow", "ancestor"])
def test_technical_admission_refuses_missing_or_changed_closure_before_writes(
    example: Example, monkeypatch: pytest.MonkeyPatch, tamper: str
) -> None:
    plan, state = complete_ancestors(example, monkeypatch)
    saved = read(plan)
    if tamper == "schema":
        saved["version"] = True
        write(plan, saved)
    elif tamper == "inventory":
        saved["inputs"] = [pin(example.code)]
        write(plan, saved)
    elif tamper == "config":
        example.change("decoderThreads", 1)
    elif tamper == "workflow":
        (example.output / "workflow.json").write_bytes(b"{}")
    else:
        write(state / "whole-lossless-join.json", {"status": "running", "fingerprint": "a" * 64})
    output = example.work / "technical"
    with pytest.raises(ReplayError):
        factory.prepare_technical(
            plan, source_state=state, state=example.work / "next-state", output=output
        )
    assert not output.exists()


def test_completed_target_checkpoints_are_pinned_for_technical_reuse(
    example: Example, monkeypatch: pytest.MonkeyPatch
) -> None:
    plan, stages_ = graph(example)
    executor = SyntheticExecutor(stages_)
    monkeypatch.setattr(workflow, "run_stage", executor.run_stage)
    state = example.work / "original-state"
    workflow.run_plan(
        example.output / "workflow.json", state_directory=state, targets=factory.TARGETS
    )
    manifest = factory.prepare_technical(
        plan,
        source_state=state,
        state=example.work / "next-state",
        output=example.work / "technical",
    )
    selected = technical.load(manifest, file_digest(manifest)).plan
    assert {state / f"{name}.json" for name in factory.TARGETS}.issubset(
        {ref.path for ref in selected.inputs}
    )


def test_preparation_does_not_adopt_unknown_production_outputs(example: Example) -> None:
    example.prepare()
    output = example.output / "whole-metadata.json"
    output.write_bytes(b"unknown prior effect")
    with pytest.raises(ReplayError, match="workflow recovery"):
        example.prepare()
    assert output.read_bytes() == b"unknown prior effect"


def test_original_stages_keep_partial_no_verifier_refusal(example: Example) -> None:
    _, stages_ = graph(example)
    expected = next(stage for stage in stages_ if stage.name == "expected-voice-pcm")
    expected.outputs[0].write_bytes(b"unknown partial voice analysis")
    with pytest.raises(StageError, match="read-only provenance"):
        stages.run_stage(expected, state_directory=example.work / "state")


def test_non_regular_and_outside_metadata_are_rejected(example: Example) -> None:
    with pytest.raises(ReplayError, match="regular"):
        factory._regular(example.work)
    with pytest.raises(ReplayError, match="checkout edits"):
        factory._metadata(technical.Reference(example.code, file_digest(example.code)))


def test_historical_replacement_hash_mismatch_is_rejected(example: Example) -> None:
    ref = technical.Reference(example.media[0], "a" * 64)
    with pytest.raises(ReplayError, match="replacement hashes"):
        factory._collect(ref, {}, set(), {ref.path: technical.Reference(ref.path, "b" * 64)})


def refresh_references(value: object) -> object:
    if isinstance(value, list):
        return [refresh_references(child) for child in value]
    if not isinstance(value, dict):
        return value
    document = record(value)
    if {"path", "sha256"}.issubset(document):
        path = Path(str(document["path"]))
        if path.is_file():
            return {**document, "sha256": file_digest(path)}
    return {
        key: file_digest(Path(key))
        if key.startswith("/") and Path(key).is_file()
        else refresh_references(child)
        for key, child in document.items()
    }


def refresh_metadata(example: Example) -> None:
    paths = sorted(example.work.glob("*.json"))

    def refresh_path(path: Path) -> bool:
        original = read(path)
        refreshed = refresh_references(original)
        if refreshed == original:
            return False
        write(path, refreshed)
        return True

    def settle(remaining: int) -> None:
        changed = any([refresh_path(path) for path in paths])
        if not changed:
            return
        if remaining <= 0:
            pytest.fail("Synthetic metadata contains a reference cycle")
        settle(remaining - 1)

    settle(len(paths))


@pytest.mark.parametrize(
    "field,value",
    [("width", 900), ("start_pts", 1), ("nb_read_frames", "2"), ("avg_frame_rate", "60/1")],
)
def test_rehashed_evidence_still_rejects_wrong_native_clock_and_geometry(
    example: Example, field: str, value: object
) -> None:
    original_path = example.work / "original-proof-1.json"
    original, published = read(original_path), read(example.proofs[0])
    observed = record(original["observed"])
    streams_ = observed["streams"]
    assert isinstance(streams_, list)
    record(streams_[0])[field] = value
    published["observed"] = observed
    write(original_path, original)
    write(example.proofs[0], published)
    refresh_metadata(example)
    with pytest.raises(ReplayError, match="clock or geometry"):
        example.prepare()


@pytest.mark.parametrize(
    "artifact,field,value",
    [
        ("intent-1.json", "destination", "wrong movie"),
        ("retirement-config-1.json", "retireStaging", False),
        ("retirement-1.json", "sourceIdentity", [0, 0]),
        ("retirement-1.json", "destinationIdentity", [0, 0, 0, 0]),
        ("retirement-1.json", "deletionPerformed", True),
        ("completion-1.json", "inputs", {}),
        ("completion-1.json", "completionVerified", False),
    ],
)
def test_rehashed_unsupported_lineage_is_rejected_without_approval(
    example: Example, artifact: str, field: str, value: object
) -> None:
    path = example.work / artifact
    document = read(path)
    document[field] = value
    write(path, document)
    refresh_metadata(example)
    with pytest.raises(ReplayError):
        example.prepare()
    assert not example.output.exists()


def test_many_declared_groups_preserve_full_transitive_closure(
    example: Example, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(factory, "GROUP_INPUTS", 8)
    plan, stages_ = graph(example)
    assert len([stage for stage in stages_ if stage.name.startswith("closure-")]) > 1
    declarations = read(plan)["inputs"]
    assert isinstance(declarations, list)
    refs = {str(record(row)["path"]) for row in declarations}
    assert {str(path) for path in example.retirements}.issubset(refs)
    assert all(len(stage.inputs) <= 256 for stage in stages_)


def test_input_and_stage_closure_bounds_are_not_relaxed(
    example: Example, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(factory, "MAX_INPUTS", 1)
    with pytest.raises(ReplayError, match="budget"):
        factory._bind({}, technical.Reference(example.code, "a" * 64))
    monkeypatch.setattr(factory, "MAX_INPUTS", 4096)
    monkeypatch.setattr(factory, "MAX_STAGES", 1)
    with pytest.raises(ReplayError, match="stage budget"):
        example.prepare()


def test_change_during_preparation_is_rejected_with_partial_metadata_preserved(
    example: Example, monkeypatch: pytest.MonkeyPatch
) -> None:
    ordinary = cleanup.prepare

    def changed(configs: tuple[Path, ...], semantic: Path, output: Path) -> None:
        ordinary(configs, semantic, output)
        example.code.write_bytes(b"concurrent code mutation")

    monkeypatch.setattr(cleanup, "prepare", changed)
    with pytest.raises(ReplayError, match="current source inventory"):
        example.prepare()
    assert (example.output / "cleanup-manifest.json").is_file()
    assert not Path(str(read(example.config)["wholeMovie"])).exists()


def test_cli_exports_exact_current_inventory(
    example: Example, capsys: pytest.CaptureFixture[str]
) -> None:
    output = example.work / "source-inventory.json"
    factory.main(["source-inventory", str(output)])
    assert capsys.readouterr().out.strip() == str(output)
    assert json.loads(output.read_text(encoding="utf-8")) == [
        pin(example.code),
        pin(example.root / "src/replay/sha256_check.py"),
        pin(example.root / "src/replay/__init__.py"),
    ]


def test_cli_technical_prepare_uses_existing_metadata_api(
    example: Example, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    plan, state = complete_ancestors(example, monkeypatch)
    capsys.readouterr()
    output = example.work / "technical"
    factory.main(
        [
            "prepare-technical",
            str(plan),
            "--source-state",
            str(state),
            "--state",
            str(example.work / "new-state"),
            "--output",
            str(output),
        ]
    )
    assert capsys.readouterr().out.strip() == str(output / "manifest.json")
    assert (output / "manifest.json").is_file()


@pytest.mark.parametrize("change", ["reader", "protocol", "drefs", "master"])
def test_noncanonical_original_graph_cannot_request_fullav_reuse(
    example: Example, monkeypatch: pytest.MonkeyPatch, change: str
) -> None:
    plan, state = complete_ancestors(example, monkeypatch)
    path = example.output / "workflow.json"
    document = read(path)
    rows = document["stages"]
    assert isinstance(rows, list)
    join = next(record(row) for row in rows if record(row)["name"] == "whole-lossless-join")
    if change == "reader":
        values = join["inputs"]
        assert isinstance(values, list)
        join["inputs"] = [value for value in values if value != str(example.code)]
    elif change == "master":
        join["outputs"] = [str(example.work / "other.mov")]
    else:
        values = join["verify_command"]
        assert isinstance(values, list)
        position = values.index("-protocol_whitelist" if change == "protocol" else "-enable_drefs")
        values[position + 1] = "file,http" if change == "protocol" else "1"
    write(path, document)
    saved = read(plan)
    saved["workflow"] = pin(path)
    write(plan, saved)
    output = example.work / "technical"
    with pytest.raises(ReplayError, match="canonical original full-AV"):
        factory.prepare_technical(
            plan, source_state=state, state=example.work / "next-state", output=output
        )
    assert not output.exists()
