"""Synthetic full-part cleanup batches preserve every existing domain guard."""

import copy
import hashlib
import json
import sys
from collections.abc import Iterable
from dataclasses import dataclass, replace
from pathlib import Path
from xml.etree import ElementTree as ET

import pytest

from replay import delivery_source_assembly as source
from replay import fcp_cleanup_adoption as batch
from replay import fcp_part_followup as followup
from replay.core import ReplayError
from replay.fcp_project import GAUSSIAN_UID
from replay.input_hash_cache import InputHashCache, input_hash_context
from replay.recordings import record
from replay.stages import file_digest


def write(path: Path, value: object) -> Path:
    path.write_text(json.dumps(value), encoding="utf-8")
    return path


def pin(path: Path) -> dict[str, str]:
    return {"path": str(path), "sha256": file_digest(path)}


def number(value: object) -> int:
    assert isinstance(value, int)
    return value


@dataclass(frozen=True)
class Plan:
    manifest: Path
    configs: tuple[Path, ...]
    semantic: Path
    clean: Path
    probe: Path
    chunk_media: tuple[Path, ...]
    chunk_producers: tuple[Path, ...]
    partition: Path
    assembly: Path
    xmls: tuple[Path, ...]

    def adopt(self, name: str = "receipt.json") -> Path:
        output = self.manifest.parent / name
        batch.adopt(self.manifest, file_digest(self.manifest), output)
        return output


@pytest.fixture(params=[(2, 3, "work"), (4, 1, "work"), (2, 3, "replay"), (4, 1, "replay")])
def plan(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, request: pytest.FixtureRequest) -> Plan:
    topology: object = request.param
    assert isinstance(topology, tuple) and len(topology) == 3
    part_count, chunk_count, layout = topology
    assert isinstance(part_count, int) and isinstance(chunk_count, int) and isinstance(layout, str)
    monkeypatch.setattr(batch, "ROOT", tmp_path)
    code = tmp_path / "src/replay"
    code.mkdir(parents=True)
    _ = [(code / name).write_bytes(name.encode()) for name in batch.CODE_NAMES]
    work = tmp_path / "edits/work"
    work.mkdir(parents=True)
    evidence = work if layout == "work" else tmp_path / "edits/replay-synthetic"
    evidence.mkdir(parents=True, exist_ok=True)
    clean = work / "clean.mov"
    clean.write_bytes(b"assembled current repaired bytes")
    historical_source = work / "never-open-historical-original.mp4"

    def make_chunk(index: int) -> tuple[dict[str, object], dict[str, object], Path, Path]:
        directory = evidence / "proxy" / f"chunk-{index:03d}"
        directory.mkdir(parents=True)
        producer_directory = evidence / "evidence"
        producer_directory.mkdir(exist_ok=True)
        media = directory / "candidate.mp4"
        media.write_bytes(f"verified current chunk {index}".encode())
        producer = write(
            producer_directory / f"producer-{index:03d}.json",
            {
                "source": str(historical_source),
                "sourceSHA256": "a" * 64,
                "outputDirectory": str(directory),
                "clock": {
                    "pts": list(range(index * 4, index * 4 + 4)),
                    "timeBase": "1/30",
                    "lastDuration": 1,
                },
            },
        )
        candidate = write(
            directory / "candidate.json",
            {
                "inputs": {str(producer): file_digest(producer), str(historical_source): "a" * 64},
                "videoSHA256": file_digest(media),
            },
        )
        return (
            {
                "media": str(media),
                "sourceFirstPTS": index * 4,
                "sourceEndPTS": index * 4 + 4,
                "frames": 4,
                "timeBase": "1/30",
            },
            {"candidate": pin(candidate), "producer": pin(producer)},
            media,
            producer,
        )

    chunks = tuple(make_chunk(index) for index in range(chunk_count))
    index = write(evidence / "index.json", {"ranges": [row for row, _, _, _ in chunks]})
    source_manifest = write(
        evidence / "source-manifest.json",
        {"version": 1, "index": pin(index), "chunks": [ref for _, ref, _, _ in chunks]},
    )
    bindings: dict[str, str] = {}
    source.load_chunks(source_manifest, sha256=file_digest(source_manifest), bindings=bindings)
    clean_index = write(
        evidence / "clean-index.json",
        {
            "ranges": [
                {
                    "media": str(clean),
                    "sha256": file_digest(clean),
                    "sourceFirstPTS": 0,
                    "sourceEndPTS": chunk_count * 4,
                    "frames": chunk_count * 4,
                    "timeBase": "1/30",
                }
            ]
        },
    )
    probe = write(evidence / "probe.json", {"packetsAndPTSVerified": True})
    receipt = write(
        evidence / "source-receipt.json",
        {
            "exactPresentationClockVerified": True,
            "compressedPacketsIdentical": True,
            "indexSHA256": file_digest(clean_index),
            "inputs": bindings,
            "probeEvidence": {str(probe): file_digest(probe)},
            "output": pin(clean),
            "frames": chunk_count * 4,
        },
    )
    voice_receipt = write(evidence / "voice-receipt.json", {"candidateOnly": True})
    semantic = work / "semantic.fcpxml"
    semantic.write_text(
        '<fcpxml><resources><asset id="r1" hasVideo="1">'
        f'<media-rep src="{clean.as_uri()}"/></asset></resources></fcpxml>',
        encoding="utf-8",
    )

    def make_xml(number: int) -> Path:
        path = work / f"part-{number:03d}.fcpxml"
        path.write_bytes(semantic.read_bytes())
        return path

    xmls = tuple(make_xml(number) for number in range(1, part_count + 1))
    partition = write(
        work / "partition.json",
        {"version": 1, "source": pin(semantic), "parts": [pin(path) for path in xmls]},
    )
    assembly = write(
        work / "assembly.json",
        {
            "version": 1,
            "partition": pin(partition),
            "parts": [{"movie": "not-yet-rendered"} for _ in xmls],
        },
    )
    cleanup = write(
        work / "cleanup.json",
        {
            "version": 1,
            "consumer": {**pin(semantic), "kind": "fcpxml"},
            "assemblyManifest": pin(source_manifest),
            "assemblyReceipt": pin(receipt),
            "cleanIndex": pin(clean_index),
            "voiceReceipt": pin(voice_receipt),
        },
    )

    def make_config(number: int) -> Path:
        return write(
            work / f"config-{number:03d}.json",
            {
                "version": 1,
                "part": number,
                "assembly": pin(assembly),
                "sourceCleanup": pin(cleanup),
                "recover": number == part_count,
            },
        )

    configs = tuple(make_config(number) for number in range(1, part_count + 1))
    manifest = work / "batch.json"
    batch.prepare(configs, semantic, manifest)
    return Plan(
        manifest,
        configs,
        semantic,
        clean,
        probe,
        tuple(row[2] for row in chunks),
        tuple(row[3] for row in chunks),
        partition,
        assembly,
        xmls,
    )


def test_every_part_domain_runs_once_and_current_media_have_two_full_reads(
    plan: Plan, monkeypatch: pytest.MonkeyPatch
) -> None:
    original = followup.validate_cleanup
    calls: list[Path] = []

    def validate(config: Path, semantic: Path) -> tuple[Path, int, bool]:
        calls.append(config)
        return original(config, semantic)

    monkeypatch.setattr(followup, "validate_cleanup", validate)
    output = plan.adopt()
    receipt = record(json.loads(output.read_text()))
    inputs = record(receipt["inputs"])
    assert calls == list(plan.configs)
    assert receipt["hashReads"] == 2 * len(inputs)
    assert number(receipt["cacheHits"]) > 0
    assert receipt["bytesRead"] == 2 * sum(Path(name).stat().st_size for name in inputs)
    assert set(map(str, (*plan.chunk_media, plan.clean))).issubset(inputs)
    assert not any("never-open-historical" in name for name in inputs)
    assert receipt["historicalSourceOpened"] is False
    assert receipt["deliveryComplete"] is False
    assert receipt["qualityApproved"] is False
    assert receipt["nativeRenderVerified"] is False
    assert receipt["initialAndTerminalBytesVerified"] is True
    before = output.read_bytes()
    batch.verify(plan.manifest, file_digest(plan.manifest), output)
    assert calls == [*plan.configs, *plan.configs]
    assert output.read_bytes() == before
    with pytest.raises(ReplayError, match="must be absent"):
        plan.adopt()


@pytest.mark.parametrize("artifact", ["clean", "probe", "chunk", "producer", "semantic", "code"])
def test_change_between_parts_stops_the_batch_before_next_part(
    plan: Plan, monkeypatch: pytest.MonkeyPatch, artifact: str
) -> None:
    target = {
        "clean": plan.clean,
        "probe": plan.probe,
        "chunk": plan.chunk_media[0],
        "producer": plan.chunk_producers[0],
        "semantic": plan.semantic,
        "code": batch.ROOT / "src/replay/stages.py",
    }[artifact]
    original = followup.validate_cleanup
    calls: list[Path] = []

    def validate(config: Path, semantic: Path) -> tuple[Path, int, bool]:
        calls.append(config)
        result = original(config, semantic)
        target.write_bytes(target.read_bytes() + b"changed")
        return result

    monkeypatch.setattr(followup, "validate_cleanup", validate)
    with pytest.raises(ReplayError, match="changed"):
        plan.adopt()
    assert calls == [plan.configs[0]]
    assert not (plan.manifest.parent / "receipt.json").exists()


def test_terminal_full_byte_boundary_and_fresh_resume_reject_changed_media(
    plan: Plan, monkeypatch: pytest.MonkeyPatch
) -> None:
    original = InputHashCache.verify

    def mutate(cache: InputHashCache) -> dict[str, str]:
        plan.clean.write_bytes(b"changed at the terminal boundary")
        return original(cache)

    monkeypatch.setattr(InputHashCache, "verify", mutate)
    with pytest.raises(ReplayError, match="changed"):
        plan.adopt()
    assert not (plan.manifest.parent / "receipt.json").exists()


def test_readonly_reuse_starts_with_empty_cache_and_rejects_transitive_change(plan: Plan) -> None:
    output = plan.adopt()
    old = output.read_bytes()
    plan.probe.write_bytes(plan.probe.read_bytes() + b" ")
    with pytest.raises(ReplayError, match="fingerprint changed"):
        batch.verify(plan.manifest, file_digest(plan.manifest), output)
    assert output.read_bytes() == old


@pytest.mark.parametrize("mode", ["missing", "extra", "duplicate", "wrongsha", "self", "empty"])
def test_explicit_inventory_cannot_omit_add_or_alias_dependencies(plan: Plan, mode: str) -> None:
    document = record(json.loads(plan.manifest.read_text()))
    rows = document["inputs"]
    assert isinstance(rows, list)
    if mode == "missing":
        document["inputs"] = rows[1:]
    elif mode == "extra":
        other = write(plan.manifest.parent / "unrelated.json", {"not": "an input"})
        document["inputs"] = [*rows, pin(other)]
    elif mode == "duplicate":
        document["inputs"] = [*rows, rows[0]]
    elif mode == "wrongsha":
        changed = copy.deepcopy(rows)
        record(changed[0])["sha256"] = "0" * 64
        document["inputs"] = changed
    elif mode == "self":
        document["inputs"] = [*rows, pin(plan.manifest)]
    else:
        document["inputs"] = []
    write(plan.manifest, document)
    with pytest.raises(ReplayError, match="inventory|closure"):
        plan.adopt()


@pytest.mark.parametrize(
    "mode", ["repeated", "reordered", "missing", "empty", "wrongpart", "extra", "bool", "recover"]
)
def test_full_ordered_partition_and_minimal_configs_are_required(plan: Plan, mode: str) -> None:
    document = record(json.loads(plan.manifest.read_text()))
    refs = document["configs"]
    assert isinstance(refs, list)
    if mode == "repeated":
        document["configs"] = [refs[0] for _ in refs]
    elif mode == "reordered":
        document["configs"] = list(reversed(refs))
    elif mode == "missing":
        document["configs"] = refs[:-1]
    elif mode == "empty":
        document["configs"] = []
    else:
        config = record(json.loads(plan.configs[0].read_text()))
        key, value = {
            "wrongpart": ("part", 0),
            "extra": ("unsafeFallback", True),
            "bool": ("part", True),
            "recover": ("recover", "true"),
        }[mode]
        config[key] = value
        write(plan.configs[0], config)
        document["configs"] = [pin(plan.configs[0]), *refs[1:]]
    write(plan.manifest, document)
    with pytest.raises(ReplayError, match="part|configuration|partition"):
        plan.adopt()


@pytest.mark.parametrize("defect", ["raw-source", "gaussian"])
def test_raw_source_or_gaussian_part_is_still_rejected_by_existing_domain(
    plan: Plan, defect: str
) -> None:
    root = ET.fromstring(plan.xmls[0].read_bytes())
    resources = root.find("resources")
    assert resources is not None
    if defect == "raw-source":
        media = resources.find("asset/media-rep")
        assert media is not None
        media.set("src", (plan.manifest.parent / "raw-unrepaired.mp4").as_uri())
    else:
        ET.SubElement(resources, "effect", {"id": "blur", "uid": GAUSSIAN_UID})
        ET.SubElement(root, "filter-video", {"ref": "blur"})
    plan.xmls[0].write_bytes(ET.tostring(root))
    partition = record(json.loads(plan.partition.read_text()))
    partition["parts"] = [pin(path) for path in plan.xmls]
    write(plan.partition, partition)
    assembly = record(json.loads(plan.assembly.read_text()))
    assembly["partition"] = pin(plan.partition)
    write(plan.assembly, assembly)

    def rebind(path: Path) -> Path:
        document = record(json.loads(path.read_text()))
        document["assembly"] = pin(plan.assembly)
        return write(path, document)

    configs = tuple(rebind(path) for path in plan.configs)
    prepared = plan.manifest.parent / "defective-batch.json"
    batch.prepare(configs, plan.semantic, prepared)
    with pytest.raises(ReplayError, match="unverified source|Gaussian"):
        batch.adopt(prepared, file_digest(prepared), plan.manifest.parent / "receipt.json")
    assert not (plan.manifest.parent / "receipt.json").exists()


def test_domain_identity_mismatch_is_not_hidden(
    plan: Plan, monkeypatch: pytest.MonkeyPatch
) -> None:
    original = followup.validate_cleanup

    def wrong(config: Path, semantic: Path) -> tuple[Path, int, bool]:
        assembly, part, recover = original(config, semantic)
        return assembly, part + 1, recover

    monkeypatch.setattr(followup, "validate_cleanup", wrong)
    with pytest.raises(ReplayError, match="different part identity"):
        plan.adopt()


def test_duplicate_part_xml_cannot_masquerade_as_full_partition(plan: Plan) -> None:
    partition = record(json.loads(plan.partition.read_text()))
    partition["parts"] = [pin(plan.xmls[0]) for _ in plan.xmls]
    write(plan.partition, partition)
    assembly = record(json.loads(plan.assembly.read_text()))
    assembly["partition"] = pin(plan.partition)
    write(plan.assembly, assembly)

    def rebind(path: Path) -> Path:
        config = record(json.loads(path.read_text()))
        config["assembly"] = pin(plan.assembly)
        return write(path, config)

    with pytest.raises(ReplayError, match="XML paths must be unique"):
        batch.prepare(
            tuple(rebind(path) for path in plan.configs),
            plan.semantic,
            plan.manifest.parent / "never.json",
        )


def test_uncached_overflow_is_read_fresh_without_dropping_any_inputs(
    plan: Plan, monkeypatch: pytest.MonkeyPatch
) -> None:
    normal = record(json.loads(plan.adopt("normal.json").read_text()))

    def limited(paths: Iterable[Path], *, allow_uncached_overflow: bool) -> InputHashCache:
        assert allow_uncached_overflow is True
        return InputHashCache(paths, max_files=1, max_bytes=1, allow_uncached_overflow=True)

    monkeypatch.setattr(batch, "InputHashCache", limited)
    output = plan.adopt("overflow.json")
    overflow = record(json.loads(output.read_text()))
    assert overflow["inputs"] == normal["inputs"]
    assert number(overflow["hashReads"]) > number(normal["hashReads"])
    assert number(overflow["bytesRead"]) > number(normal["bytesRead"])
    batch.verify(plan.manifest, file_digest(plan.manifest), output)


def test_nested_cache_scope_is_refused_without_receipt(plan: Plan) -> None:
    cache = InputHashCache([plan.clean])
    cache.prime()
    with input_hash_context(cache), pytest.raises(ReplayError, match="Nested"):
        plan.adopt()
    assert not (plan.manifest.parent / "receipt.json").exists()


@pytest.mark.parametrize("target", ["manifest", "code", "media"])
def test_symlinked_inputs_cannot_enter_inventory_or_cache(plan: Plan, target: str) -> None:
    path = {
        "manifest": plan.manifest,
        "code": batch.ROOT / "src/replay/stages.py",
        "media": plan.clean,
    }[target]
    destination = path.with_suffix(path.suffix + ".real")
    expected = file_digest(plan.manifest)
    path.rename(destination)
    path.symlink_to(destination)
    with pytest.raises(ReplayError, match="canonical|resolved"):
        batch.adopt(plan.manifest, expected, plan.manifest.parent / "receipt.json")


@pytest.mark.parametrize("field", ["version", "extra", "inputs"])
def test_manifest_schema_rejects_invalid_or_unbounded_fields(plan: Plan, field: str) -> None:
    document = record(json.loads(plan.manifest.read_text()))
    document[field] = {"version": True, "extra": True, "inputs": [pin(plan.clean)] * 4096}[field]
    write(plan.manifest, document)
    with pytest.raises(ReplayError, match="manifest fields|inventory"):
        plan.adopt()


def test_duplicate_json_keys_and_pinned_mutation_are_refused_before_domains(plan: Plan) -> None:
    expected = file_digest(plan.manifest)
    plan.manifest.write_bytes(b'{"version":1,"version":1}')
    with pytest.raises(ReplayError, match="fingerprint changed"):
        batch.adopt(plan.manifest, expected, plan.manifest.parent / "never.json")
    with pytest.raises(ReplayError, match="unique"):
        batch.adopt(plan.manifest, file_digest(plan.manifest), plan.manifest.parent / "never.json")


@pytest.mark.parametrize(
    "approval", ["qualityApproved", "deliveryComplete", "nativeRenderVerified"]
)
def test_fresh_verification_refuses_tampered_approval(plan: Plan, approval: str) -> None:
    output = plan.adopt()
    document = record(json.loads(output.read_text()))
    document[approval] = True
    write(output, document)
    with pytest.raises(ReplayError, match="receipt differs"):
        batch.verify(plan.manifest, file_digest(plan.manifest), output)


def test_cli_prepare_adopt_and_readonly_verify(plan: Plan, monkeypatch: pytest.MonkeyPatch) -> None:
    prepared = plan.manifest.parent / "cli-batch.json"
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "batch",
            "prepare",
            str(prepared),
            "--semantic",
            str(plan.semantic),
            *[value for path in plan.configs for value in ("--part-config", str(path))],
        ],
    )
    batch.main()
    output = plan.manifest.parent / "cli-receipt.json"
    monkeypatch.setattr(
        sys,
        "argv",
        ["batch", "adopt", str(prepared), str(output), "--sha256", file_digest(prepared)],
    )
    batch.main()
    before = output.read_bytes()
    monkeypatch.setattr(
        sys,
        "argv",
        ["batch", "verify", str(prepared), str(output), "--sha256", file_digest(prepared)],
    )
    batch.main()
    assert output.read_bytes() == before
    assert hashlib.sha256(prepared.read_bytes()).hexdigest() == file_digest(prepared)


@pytest.mark.parametrize("directory", ["private", "downloads", "outside", "root-file"])
def test_unapproved_metadata_scope_is_rejected_before_opening(
    plan: Plan, monkeypatch: pytest.MonkeyPatch, directory: str
) -> None:
    target = (
        batch.ROOT.parent / "outside.json"
        if directory == "outside"
        else batch.ROOT / directory / "hidden.json"
    )

    def unexpected(_path: Path) -> bytes:
        pytest.fail("Unapproved metadata must never be opened")

    monkeypatch.setattr(batch, "_bytes", unexpected)
    with pytest.raises(ReplayError, match="canonical"):
        batch.Inventory().metadata({"path": str(target), "sha256": "a" * 64})


def test_escaping_evidence_symlink_is_rejected_before_read(
    plan: Plan, monkeypatch: pytest.MonkeyPatch
) -> None:
    evidence = batch.ROOT / "edits/replay-link"
    evidence.symlink_to(batch.ROOT.parent)

    def unexpected(_path: Path) -> bytes:
        pytest.fail("Escaping metadata links must never be opened")

    monkeypatch.setattr(batch, "_bytes", unexpected)
    with pytest.raises(ReplayError, match="canonical"):
        batch.Inventory().metadata({"path": str(evidence / "outside.json"), "sha256": "a" * 64})


def test_new_configs_and_receipts_stay_work_only_even_when_evidence_is_replay(plan: Plan) -> None:
    directory = batch.ROOT / "edits/replay-controls"
    directory.mkdir()
    control = directory / "new-config.json"
    control.write_bytes(plan.configs[0].read_bytes())
    with pytest.raises(ReplayError, match="canonical"):
        batch.prepare(
            (control, *plan.configs[1:]), plan.semantic, plan.manifest.parent / "never.json"
        )
    with pytest.raises(ReplayError, match="canonical"):
        batch.prepare(plan.configs, plan.semantic, directory / "new-operation.json")
    with pytest.raises(ReplayError, match="canonical"):
        batch.adopt(plan.manifest, file_digest(plan.manifest), directory / "new-receipt.json")
    assert not (directory / "new-operation.json").exists()
    assert not (directory / "new-receipt.json").exists()


def approved_plan(plan: Plan) -> tuple[Plan, Path, Path]:
    """Bind a synthetic canonical original to historical alias producer evidence."""
    directory = plan.manifest.parent
    original = directory / "approved-original.mp4"
    original.write_bytes(b"explicitly approved original bytes")
    digest = file_digest(original)
    historical = directory / "never-open-historical-original.mp4"
    historical.symlink_to(original)
    config = record(json.loads(plan.configs[0].read_text()))
    binding_path = reference_path(config["sourceCleanup"])
    binding = record(json.loads(binding_path.read_text()))
    manifest_path = reference_path(binding["assemblyManifest"])
    manifest = record(json.loads(manifest_path.read_text()))
    rows = manifest["chunks"]
    assert isinstance(rows, list)

    def rebind_chunk(value: object) -> dict[str, object]:
        chunk = record(value)
        producer = reference_path(chunk["producer"])
        candidate = reference_path(chunk["candidate"])
        data = record(json.loads(producer.read_text()))
        data["sourceSHA256"] = digest
        write(producer, data)
        candidate_data = record(json.loads(candidate.read_text()))
        record(candidate_data["inputs"])[str(producer)] = file_digest(producer)
        record(candidate_data["inputs"])[str(historical)] = digest
        write(candidate, candidate_data)
        return {"candidate": pin(candidate), "producer": pin(producer)}

    manifest["chunks"] = [rebind_chunk(value) for value in rows]
    write(manifest_path, manifest)
    bindings: dict[str, str] = {}
    source.load_chunks(manifest_path, sha256=file_digest(manifest_path), bindings=bindings)
    receipt_path = reference_path(binding["assemblyReceipt"])
    receipt = record(json.loads(receipt_path.read_text()))
    receipt["inputs"] = bindings
    write(receipt_path, receipt)
    binding["assemblyManifest"] = pin(manifest_path)
    binding["assemblyReceipt"] = pin(receipt_path)
    write(binding_path, binding)

    def approve(path: Path) -> Path:
        data = record(json.loads(path.read_text()))
        data["sourceCleanup"] = pin(binding_path)
        data["approvedOriginal"] = pin(original)
        return write(path, data)

    _ = [approve(path) for path in plan.configs]
    prepared = directory / "approved-batch.json"
    batch.prepare(plan.configs, plan.semantic, prepared)
    return replace(plan, manifest=prepared), original, historical


def reference_path(value: object) -> Path:
    path = record(value).get("path")
    assert isinstance(path, str)
    return Path(path)


def test_explicit_approved_original_joins_full_byte_and_domain_verification(plan: Plan) -> None:
    approved, original, historical = approved_plan(plan)
    document = record(json.loads(approved.manifest.read_text()))
    rows = document["inputs"]
    assert isinstance(rows, list)
    assert pin(original) in rows
    assert str(historical) not in [record(row)["path"] for row in rows]
    receipt = approved.adopt("approved-receipt.json")
    verified = record(json.loads(receipt.read_text()))
    assert record(verified["inputs"])[str(original)] == file_digest(original)
    assert verified["hashReads"] == 2 * len(record(verified["inputs"]))
    assert verified["cleanupDomainVerified"] is True
    assert verified["qualityApproved"] is False
    old = receipt.read_bytes()
    batch.verify(approved.manifest, file_digest(approved.manifest), receipt)
    assert receipt.read_bytes() == old


@pytest.mark.parametrize("defect", ["malformed", "unknown", "hash", "escape", "private"])
def test_invalid_approved_reference_is_rejected_before_preparation_output(
    plan: Plan, defect: str
) -> None:
    config = record(json.loads(plan.configs[0].read_text()))
    ref: object = {
        "malformed": None,
        "unknown": {**pin(plan.clean), "unsafe": True},
        "hash": {"path": str(plan.clean), "sha256": "invalid"},
        "escape": {"path": str(plan.manifest.parent / "../escape.mov"), "sha256": "a" * 64},
        "private": {"path": str(batch.ROOT / "private/unopened.mov"), "sha256": "a" * 64},
    }[defect]
    config["approvedOriginal"] = ref
    write(plan.configs[0], config)
    target = plan.manifest.parent / "invalid-approved-batch.json"
    with pytest.raises(ReplayError):
        batch.prepare(plan.configs, plan.semantic, target)
    assert not target.exists()


@pytest.mark.parametrize("defect", ["wrong-hash", "wrong-target", "target-swap"])
def test_approval_does_not_bypass_original_hash_or_same_byte_alias_target(
    plan: Plan, defect: str
) -> None:
    approved, original, historical = approved_plan(plan)
    other = plan.manifest.parent / "different-original.mp4"
    other.write_bytes(original.read_bytes())
    if defect == "target-swap":
        historical.unlink()
        historical.symlink_to(other)
    else:

        def change(path: Path) -> Path:
            data = record(json.loads(path.read_text()))
            data["approvedOriginal"] = (
                {"path": str(original), "sha256": "a" * 64}
                if defect == "wrong-hash"
                else pin(other)
            )
            return write(path, data)

        _ = [change(path) for path in plan.configs]
        target = plan.manifest.parent / "changed-approved-batch.json"
        batch.prepare(plan.configs, plan.semantic, target)
        approved = replace(approved, manifest=target)
    with pytest.raises(ReplayError, match="fingerprint|alias target"):
        approved.adopt("invalid-approved-receipt.json")
    assert not (plan.manifest.parent / "invalid-approved-receipt.json").exists()
