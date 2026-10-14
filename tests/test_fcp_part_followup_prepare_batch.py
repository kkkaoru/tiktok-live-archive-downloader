import fcntl
import json
import os
from collections.abc import Iterable
from dataclasses import dataclass
from fractions import Fraction
from pathlib import Path

import pytest

from replay import fcp_part_followup as canonical
from replay import fcp_part_followup_prepare_batch as batch
from replay import stages
from replay.core import ReplayError
from replay.input_hash_cache import MAX_FILES, Entry, InputHashCache, input_hash_context
from replay.recordings import record


@dataclass(frozen=True)
class Case:
    root: Path
    request: Path
    configs: tuple[Path, ...]
    shared: tuple[Path, ...]
    count: int


def write(path: Path, raw: dict[str, object]) -> Path:
    path.write_text(json.dumps(raw), encoding="utf-8")
    return path


def read(path: Path) -> dict[str, object]:
    return record(json.loads(path.read_text(encoding="utf-8")))


def pin(path: Path) -> dict[str, object]:
    return {"path": str(path), "sha256": stages.file_digest(path)}


def members(raw: object) -> list[object]:
    assert isinstance(raw, list)
    return raw


def media(path: Path, size: int) -> Path:
    path.write_bytes(b"x" * size)
    return path


def part_config(root: Path, number: int, refs: dict[str, object]) -> Path:
    native = media(root / f"native-{number}.mov", 100 + number)
    return write(
        root / f"config-{number}.json",
        {
            "version": 1,
            "part": number,
            **refs,
            "exportRequest": pin(write(root / f"export-{number}.json", {"movie": str(native)})),
            "exportResponse": pin(write(root / f"response-{number}.json", {"native": number})),
            "background": pin(write(root / f"idle-{number}.json", {"idle": True})),
            "audioSampleRate": 48000 if number == 1 else 44100,
            "audioChannels": 2,
            "probeDecoderThreads": 0,
            "receiptOutput": str(root / f"publication-{number}.json"),
        },
    )


@pytest.fixture(params=[(2, 2), (3, 1)])
def case(request: pytest.FixtureRequest, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Case:
    assert isinstance(request.param, tuple)
    count, chunks_count = request.param
    assert isinstance(count, int) and isinstance(chunks_count, int)
    original = media(tmp_path / "original.mov", 16384)
    clean = media(tmp_path / "clean.mov", 32768)
    voice = media(tmp_path / "voice.wav", 24576)
    chunks = tuple(media(tmp_path / f"chunk-{n}.mov", 4096) for n in range(chunks_count))
    evidence = write(tmp_path / "evidence.json", {"version": 1})
    semantic = tmp_path / "semantic.fcpxml"
    semantic.write_text(
        f'<fcpxml><resources><asset hasVideo="1"><media-rep src="{clean.as_uri()}"/>'
        "</asset></resources><library><event><project name='Full'><sequence/>"
        "</project></event></library></fcpxml>",
        encoding="utf-8",
    )
    xmls = tuple(tmp_path / f"part-{n}.fcpxml" for n in range(1, count + 1))
    tuple(path.write_bytes(semantic.read_bytes()) for path in xmls)
    partition = write(
        tmp_path / "partition.json",
        {"version": 1, "source": pin(semantic), "parts": [pin(path) for path in xmls]},
    )
    common = {
        "version": 1,
        "partition": pin(partition),
        "voice": pin(voice),
        "voiceReceipt": pin(evidence),
    }
    assembly = write(
        tmp_path / "staging.json",
        {
            **common,
            "parts": [
                {
                    "movie": str(tmp_path / f"native-{n}.mov"),
                    "proof": str(tmp_path / f"proof-{n}.json"),
                }
                for n in range(1, count + 1)
            ],
        },
    )
    published = write(
        tmp_path / "published.json",
        {
            **common,
            "parts": [
                {
                    "movie": str(tmp_path / f"ssd-{n}.mov"),
                    "proof": str(tmp_path / f"ssd-proof-{n}.json"),
                }
                for n in range(1, count + 1)
            ],
        },
    )
    cleanup = write(
        tmp_path / "cleanup.json",
        {
            "version": 1,
            "consumer": {**pin(semantic), "kind": "fcpxml"},
            "assemblyManifest": pin(evidence),
            "assemblyReceipt": pin(evidence),
            "cleanIndex": pin(evidence),
            "voiceReceipt": pin(evidence),
        },
    )
    executable = media(tmp_path / "ffprobe", 24)
    executable.chmod(0o700)

    def clean_source(raw: dict[str, object]) -> tuple[Path, Path, Fraction]:
        assert raw["consumer"] == {**pin(semantic), "kind": "fcpxml"}
        stages.file_digest(original)
        stages.file_digest(clean)
        stages.file_digest(clean)
        return original, clean, Fraction(count)

    def cleanup_inputs(path: Path) -> tuple[Path, ...]:
        assert path == cleanup
        tuple(map(stages.file_digest, chunks))
        return (*chunks, evidence)

    monkeypatch.setattr(canonical, "clean_source", clean_source)
    monkeypatch.setattr(canonical, "_cleanup_inputs", cleanup_inputs)
    refs: dict[str, object] = {
        "assembly": pin(assembly),
        "publishedAssembly": pin(published),
        "sourceCleanup": pin(cleanup),
        "ffmpeg": pin(executable),
        "ffprobe": pin(executable),
    }
    configs = tuple(part_config(tmp_path, n, refs) for n in range(1, count + 1))
    inputs = tuple(
        dict.fromkeys(path for config in configs for path in canonical._load(config).inputs)
    )
    path = write(
        tmp_path / "request.json",
        {
            "version": 1,
            "kind": "native-followup-preparation-request",
            "maxReadyParts": count,
            "maxNativeBytes": 10000,
            "parts": [
                {"part": n, "config": pin(config), "output": str(tmp_path / f"prepared-{n}")}
                for n, config in enumerate(configs, 1)
            ],
            "inputs": [pin(path) for path in (*inputs, *chunks)],
        },
    )
    return Case(tmp_path, path, configs, (original, clean, voice, *chunks), count)


def test_delegates_real_factory_with_original_clocks_and_complete_proofs(case: Case) -> None:
    result = batch.prepare(case.request, case.root / "prepare")
    assert result.prepared == case.count
    assert result.adopted == 0
    assert result.cache_hits > 0
    graph = read(case.root / "prepared-1/workflow.json")
    stage = record(members(graph["stages"])[0])
    assert stage["name"] == "followup"
    assert len(members(stage["outputs"])) == 7
    operation = read(case.root / "prepared-1/operation.json")
    assert operation["semantic"] == str(case.root / "semantic.fcpxml")
    assert read(case.root / "prepared-1/completion-config.json")["audioSampleRate"] == 48000
    assert read(case.root / "prepared-2/completion-config.json")["audioSampleRate"] == 44100
    acceptance = record(read(result.receipt)["acceptance"])
    assert acceptance["qualityApproved"] is False
    assert acceptance["deliveryApproved"] is False
    assert acceptance["workflowsExecuted"] is False
    assert len(record(acceptance["generatedMetadata"])) == case.count * 4
    metrics = record(read(result.receipt)["metrics"])
    assert metrics["childHashBytes"] is None
    assert metrics["physicalIOBytes"] is None


def test_common_read_reduction_is_counted_against_same_ordinary_factory(
    case: Case, monkeypatch: pytest.MonkeyPatch
) -> None:
    ordinary = stages._uncached_file_digest
    ordinary_bytes: list[int] = []

    def counted(path: Path) -> str:
        if path in case.shared:
            ordinary_bytes.append(path.stat().st_size)
        return ordinary(path)

    monkeypatch.setattr(stages, "_uncached_file_digest", counted)
    tuple(
        canonical.prepare(config, case.root / f"ordinary-{n}", integrated=True)
        for n, config in enumerate(case.configs, 1)
    )
    baseline = sum(ordinary_bytes)
    ordinary_bytes.clear()
    reads: list[int] = []
    reader = InputHashCache._read

    def read_cache(self: InputHashCache, path: Path, entry: Entry, *, fresh: bool) -> str:
        if path in case.shared and (fresh or entry.sha256 is None):
            reads.append(path.stat().st_size)
        return reader(self, path, entry, fresh=fresh)

    monkeypatch.setattr(InputHashCache, "_read", read_cache)
    result = batch.prepare(case.request, case.root / "prepare")
    assert ordinary_bytes == []
    assert sum(reads) == 2 * sum(path.stat().st_size for path in case.shared)
    assert sum(reads) < baseline
    assert result.bytes_read > sum(reads)
    write(
        case.root / "synthetic-read-comparison.json",
        {
            "ordinaryCommonBytes": baseline,
            "ownedCacheCommonBytes": sum(reads),
            "physicalIOBytes": None,
        },
    )


def test_fresh_resume_adopts_complete_preparation_without_reexecuting_factory(
    case: Case, monkeypatch: pytest.MonkeyPatch
) -> None:
    first = batch.prepare(case.request, case.root / "prepare")
    old = first.receipt.read_bytes()

    def forbidden(config: Path, output: Path, *, integrated: bool = False) -> Path:
        pytest.fail("Verified complete preparation must not rerun its factory")

    monkeypatch.setattr(canonical, "prepare", forbidden)
    second = batch.prepare(case.request, case.root / "prepare", resume=True)
    assert second.prepared == 0
    assert second.adopted == case.count
    assert second.hash_reads > 0
    assert second.receipt.read_bytes() == old


def test_partial_resume_only_prepares_missing_unoccupied_part(
    case: Case, monkeypatch: pytest.MonkeyPatch
) -> None:
    prepare = canonical.prepare
    stopped: list[bool] = []

    def interrupt(config: Path, output: Path, *, integrated: bool = False) -> Path:
        if config == case.configs[1] and not stopped:
            stopped.append(True)
            raise ReplayError("Synthetic interruption before second factory")
        return prepare(config, output, integrated=integrated)

    monkeypatch.setattr(canonical, "prepare", interrupt)
    with pytest.raises(ReplayError, match="interruption"):
        batch.prepare(case.request, case.root / "prepare")
    checkpoint = case.root / "prepare/checkpoints/part-0001.json"
    old = checkpoint.read_bytes()
    result = batch.prepare(case.request, case.root / "prepare", resume=True)
    assert result.adopted == 1
    assert result.prepared == case.count - 1
    assert checkpoint.read_bytes() == old


def test_uncheckpointed_occupied_partial_is_preserved_and_never_retried(
    case: Case, monkeypatch: pytest.MonkeyPatch
) -> None:
    save = batch._save_atomic

    def interrupt(path: Path, raw: dict[str, object]) -> None:
        if path.name == "part-0002.json":
            raise ReplayError("Synthetic failure before checkpoint")
        save(path, raw)

    monkeypatch.setattr(batch, "_save_atomic", interrupt)
    with pytest.raises(ReplayError, match="before checkpoint"):
        batch.prepare(case.request, case.root / "prepare")
    old = (case.root / "prepared-2/workflow.json").read_bytes()
    with pytest.raises(ReplayError, match="uncertain"):
        batch.prepare(case.request, case.root / "prepare", resume=True)
    assert (case.root / "prepared-2/workflow.json").read_bytes() == old
    assert not (case.root / "prepare/prepared.json").exists()


@pytest.mark.parametrize(
    ("key", "value"),
    [
        ("version", True),
        ("kind", "other"),
        ("maxReadyParts", 0),
        ("maxNativeBytes", -1),
        ("parts", []),
        ("inputs", []),
    ],
)
def test_invalid_request_is_rejected_before_operation_side_effects(
    case: Case, key: str, value: object
) -> None:
    raw = read(case.request)
    raw[key] = value
    write(case.request, raw)
    with pytest.raises(ReplayError, match="bounded"):
        batch.prepare(case.request, case.root / "prepare")
    assert not (case.root / "prepare").exists()


@pytest.mark.parametrize("kind", ["missing", "extra", "changed", "duplicate"])
def test_exact_explicit_closure_is_mandatory(case: Case, kind: str) -> None:
    raw = read(case.request)
    refs = members(raw["inputs"])
    if kind == "missing":
        refs.pop()
    elif kind == "extra":
        refs.append(pin(media(case.root / "extra.bin", 10)))
    elif kind == "changed":
        record(refs[-1])["sha256"] = "a" * 64
    else:
        refs.append(refs[-1])
    write(case.request, raw)
    with pytest.raises(ReplayError, match="closure|fingerprints|unique"):
        batch.prepare(case.request, case.root / "prepare")
    assert not (case.root / "prepare").exists()


def test_config_identity_and_native_byte_budget_are_enforced(case: Case) -> None:
    raw = read(case.request)
    record(members(raw["parts"])[0])["part"] = 99
    write(case.request, raw)
    with pytest.raises(ReplayError, match="assembly part"):
        batch.prepare(case.request, case.root / "prepare")
    record(members(raw["parts"])[0])["part"] = 1
    raw["maxNativeBytes"] = 1
    write(case.request, raw)
    with pytest.raises(ReplayError, match="byte budget"):
        batch.prepare(case.request, case.root / "prepare")


def test_namespaces_cannot_collide_with_inputs_or_each_other(case: Case) -> None:
    raw = read(case.request)
    record(members(raw["parts"])[1])["output"] = str(case.root / "prepared-1/child")
    write(case.request, raw)
    with pytest.raises(ReplayError, match="namespaces"):
        batch.prepare(case.request, case.root / "prepare")
    record(members(raw["parts"])[1])["output"] = str(case.root / "prepared-2")
    write(case.request, raw)
    with pytest.raises(ReplayError, match="namespaces"):
        batch.prepare(case.request, case.root)


def test_symlink_private_paths_and_duplicate_json_are_rejected(case: Case) -> None:
    linked = case.root / "linked.json"
    linked.symlink_to(case.request)
    with pytest.raises(ReplayError, match="canonical"):
        batch.prepare(linked, case.root / "prepare")
    with pytest.raises(ReplayError, match="private"):
        batch._ref({"path": str(batch.ROOT / "private/unopened.bin"), "sha256": "a" * 64})
    case.request.write_text('{"version":1,"version":1}', encoding="utf-8")
    with pytest.raises(ReplayError, match="unique"):
        batch.prepare(case.request, case.root / "prepare")


def test_nested_cache_is_rejected_before_preparation_effects(case: Case) -> None:
    cache = InputHashCache((case.request,))
    with input_hash_context(cache), pytest.raises(ReplayError, match="Nested"):
        batch.prepare(case.request, case.root / "prepare")
    assert not (case.root / "prepare").exists()


def test_midpart_mutation_preserves_pending_evidence_without_acceptance(
    case: Case, monkeypatch: pytest.MonkeyPatch
) -> None:
    prepare = canonical.prepare

    def mutate(config: Path, output: Path, *, integrated: bool = False) -> Path:
        graph = prepare(config, output, integrated=integrated)
        case.shared[1].write_bytes(b"changed cleanup")
        return graph

    monkeypatch.setattr(canonical, "prepare", mutate)
    with pytest.raises(ReplayError, match="changed"):
        batch.prepare(case.request, case.root / "prepare")
    assert (case.root / "prepared-1/workflow.json").exists()
    assert not (case.root / "prepare/prepared.json").exists()


def test_terminal_mutation_with_preserved_size_and_mtime_stops(
    case: Case, monkeypatch: pytest.MonkeyPatch
) -> None:
    verify = InputHashCache.verify

    def mutate(self: InputHashCache) -> dict[str, str]:
        source = case.shared[0]
        old = source.stat()
        source.write_bytes(b"z" * old.st_size)
        os.utime(source, ns=(old.st_atime_ns, old.st_mtime_ns))
        return verify(self)

    monkeypatch.setattr(InputHashCache, "verify", mutate)
    with pytest.raises(ReplayError, match="changed"):
        batch.prepare(case.request, case.root / "prepare")
    assert not (case.root / "prepare/prepared.json").exists()


def test_occupied_root_requires_resume_and_changed_metadata_cannot_be_adopted(case: Case) -> None:
    first = batch.prepare(case.request, case.root / "prepare")
    with pytest.raises(ReplayError, match="resume"):
        batch.prepare(case.request, case.root / "prepare")
    graph = case.root / "prepared-1/workflow.json"
    raw = read(graph)
    raw["diagnostic"] = "changed"
    write(graph, raw)
    with pytest.raises(ReplayError, match="checkpoint"):
        batch.prepare(case.request, case.root / "prepare", resume=True)
    assert first.receipt.exists()


def test_current_producer_change_cannot_reuse_old_preparation(case: Case) -> None:
    batch.prepare(case.request, case.root / "prepare")
    raw = read(case.request)
    raw["maxNativeBytes"] = 9999
    write(case.request, raw)
    with pytest.raises(ReplayError, match="operation differs"):
        batch.prepare(case.request, case.root / "prepare", resume=True)


def test_operation_lease_rejects_other_owner_and_linked_lock(case: Case) -> None:
    output = case.root / "prepare"
    batch.prepare(case.request, output)
    with (output / "prepare.lock").open("rb") as stream:
        fcntl.flock(stream, fcntl.LOCK_EX | fcntl.LOCK_NB)
        with pytest.raises(ReplayError, match="owned"):
            batch.prepare(case.request, output, resume=True)
    (output / "prepare.lock").unlink()
    (output / "prepare.lock").hardlink_to(case.root / "original.mov")
    with pytest.raises(ReplayError, match="single-link"):
        batch.prepare(case.request, output, resume=True)


def test_missing_original_lease_cannot_be_recreated_by_resume(case: Case) -> None:
    output = case.root / "prepare"
    first = batch.prepare(case.request, output)
    original = first.receipt.read_bytes()
    lease = output / "prepare.lock"
    lease.unlink()
    with pytest.raises(ReplayError, match="original preparation lease"):
        batch.prepare(case.request, output, resume=True)
    assert not lease.exists()
    assert first.receipt.read_bytes() == original


def test_cli_preserves_readonly_preparation_boundary(
    case: Case, capsys: pytest.CaptureFixture[str]
) -> None:
    batch.main([str(case.request), str(case.root / "prepare")])
    assert record(json.loads(capsys.readouterr().out))["adopted"] == 0
    batch.main([str(case.request), str(case.root / "prepare"), "--resume"])
    assert record(json.loads(capsys.readouterr().out))["prepared"] == 0


def test_one_ready_part_keeps_factory_identity_without_a_throughput_promise(case: Case) -> None:
    raw = read(case.request)
    raw["parts"] = [members(raw["parts"])[0]]
    raw["maxReadyParts"] = 1
    plan = canonical._load(case.configs[0])
    paths = tuple(
        dict.fromkeys((*plan.inputs, *canonical._cleanup_inputs(plan.paths["sourceCleanup"])))
    )
    raw["inputs"] = [pin(path) for path in paths]
    write(case.request, raw)
    result = batch.prepare(case.request, case.root / "single")
    assert result.prepared == 1
    assert result.adopted == 0
    operation = read(case.root / "prepared-1/operation.json")
    assert operation["config"] == str(case.root / "config-1.json")
    assert record(read(result.receipt)["acceptance"])["qualityApproved"] is False


def test_overflow_paths_remain_fresh_instead_of_exceeding_cache_budgets(
    case: Case, monkeypatch: pytest.MonkeyPatch
) -> None:
    class TinyCache(InputHashCache):
        def __init__(self, paths: Iterable[Path], *, allow_uncached_overflow: bool = False) -> None:
            super().__init__(
                paths,
                max_files=1 if allow_uncached_overflow else MAX_FILES,
                allow_uncached_overflow=allow_uncached_overflow,
            )

    monkeypatch.setattr(batch, "InputHashCache", TinyCache)
    result = batch.prepare(case.request, case.root / "prepare")
    acceptance = record(read(result.receipt)["acceptance"])
    assert result.hash_reads > 2 * len(record(acceptance["inputs"]))
    assert acceptance["workflowsExecuted"] is False


def test_prior_metadata_changed_by_later_factory_cannot_receive_acceptance(
    case: Case, monkeypatch: pytest.MonkeyPatch
) -> None:
    prepare = canonical.prepare

    def mutate(config: Path, output: Path, *, integrated: bool = False) -> Path:
        graph = prepare(config, output, integrated=integrated)
        if config == case.configs[1]:
            old_graph = case.root / "prepared-1/workflow.json"
            raw = read(old_graph)
            raw["diagnostic"] = "changed after checkpoint"
            write(old_graph, raw)
        return graph

    monkeypatch.setattr(canonical, "prepare", mutate)
    with pytest.raises(ReplayError, match="immutable checkpoint"):
        batch.prepare(case.request, case.root / "prepare")
    assert not (case.root / "prepare/prepared.json").exists()


def test_generated_metadata_is_guarded_during_fresh_input_terminal_hashing(
    case: Case, monkeypatch: pytest.MonkeyPatch
) -> None:
    verify = InputHashCache.verify
    mutated: list[bool] = []

    def mutate(self: InputHashCache) -> dict[str, str]:
        if not mutated:
            mutated.append(True)
            graph = case.root / "prepared-1/workflow.json"
            raw = read(graph)
            raw["late"] = "changed during input hash"
            write(graph, raw)
        return verify(self)

    monkeypatch.setattr(InputHashCache, "verify", mutate)
    with pytest.raises(ReplayError, match="changed"):
        batch.prepare(case.request, case.root / "prepare")
    assert not (case.root / "prepare/prepared.json").exists()


def test_completed_receipt_cannot_be_overwritten_or_promoted(case: Case) -> None:
    first = batch.prepare(case.request, case.root / "prepare")
    raw = read(first.receipt)
    record(raw["acceptance"])["qualityApproved"] = True
    write(first.receipt, raw)
    changed = first.receipt.read_bytes()
    with pytest.raises(ReplayError, match="receipt changed"):
        batch.prepare(case.request, case.root / "prepare", resume=True)
    assert first.receipt.read_bytes() == changed


def test_downstream_output_collisions_are_rejected_before_metadata_creation(case: Case) -> None:
    raw = read(case.root / "staging.json")
    record(members(raw["parts"])[1])["proof"] = str(case.root / "proof-1.json")
    write(case.root / "staging.json", raw)
    first = read(case.configs[0])
    second = read(case.configs[1])
    first["assembly"] = pin(case.root / "staging.json")
    second["assembly"] = pin(case.root / "staging.json")
    write(case.configs[0], first)
    write(case.configs[1], second)
    raw_request = read(case.request)
    raw_request["parts"] = members(raw_request["parts"])[:2]
    record(members(raw_request["parts"])[0])["config"] = pin(case.configs[0])
    record(members(raw_request["parts"])[1])["config"] = pin(case.configs[1])
    plans = (canonical._load(case.configs[0]), canonical._load(case.configs[1]))
    paths = tuple(
        dict.fromkeys(
            (
                *plans[0].inputs,
                *plans[1].inputs,
                *canonical._cleanup_inputs(plans[0].paths["sourceCleanup"]),
            )
        )
    )
    raw_request["inputs"] = [pin(path) for path in paths]
    write(case.request, raw_request)
    with pytest.raises(ReplayError, match="namespaces"):
        batch.prepare(case.request, case.root / "prepare")
    assert not (case.root / "prepare").exists()
