import json
import sys
from fractions import Fraction
from pathlib import Path

import pytest

from replay import fcp_part_followup as followup
from replay import source_caption_binding, workflow
from replay.artifact_json import _pin
from replay.core import ReplayError
from replay.fcp_part_assembly import FrameProbeTool, _reference, _save
from replay.stages import Stage, StageError
from replay.workflow import main as workflow_main


@pytest.fixture
def alias_plan(plan: Path, monkeypatch: pytest.MonkeyPatch) -> tuple[Path, Path]:
    original = plan.parent / "original.mov"
    alias = plan.parent / "historical-original.mov"
    alias.symlink_to(original)
    producer = plan.parent / "alias-producer.json"
    _save(producer, {"source": str(alias), "sourceSHA256": _reference(original)["sha256"]})
    manifest = plan.parent / "alias-manifest.json"
    _save(manifest, {"chunks": [{"producer": _reference(producer)}]})
    cleanup = plan.parent / "cleanup.json"
    cleanup_data = json.loads(cleanup.read_text())
    cleanup_data["assemblyManifest"] = _reference(manifest)
    cleanup.write_text(json.dumps(cleanup_data))
    data = json.loads(plan.read_text())
    data["sourceCleanup"] = _reference(cleanup)
    data["approvedOriginal"] = _reference(original)
    plan.write_text(json.dumps(data))
    checked = source_caption_binding.clean_source

    def aliased(config: dict[str, object]) -> tuple[Path, Path, Fraction]:
        _, clean, duration = checked(config)
        return alias, clean, duration

    monkeypatch.setattr(followup, "clean_source", aliased)
    monkeypatch.setattr(followup, "_cleanup_inputs", lambda _: (manifest, producer))
    return plan, alias


def test_approved_alias_inventory_is_canonical_and_historical_evidence_unchanged(
    alias_plan: tuple[Path, Path],
) -> None:
    config, alias = alias_plan
    originals = tuple(
        config.parent / name for name in ("alias-producer.json", "staging.json", "published.json")
    )
    before = [path.read_bytes() for path in originals]
    graph = followup.prepare(config, config.parent / "alias-integrated", integrated=True)
    operation = graph.parent / "operation.json"
    semantic = config.parent / "semantic.fcpxml"
    data, inputs = followup._operation(operation, semantic)
    loaded = followup._integrated_plan(operation, semantic, data, inputs)
    assert alias not in loaded.inputs and alias.resolve() in loaded.inputs
    assert len(inputs) == len(set(inputs))
    assert [path.read_bytes() for path in originals] == before


@pytest.mark.parametrize(
    "mode", ["unapproved", "other-target", "changed-bytes", "missing", "linked-approval"]
)
def test_approved_alias_rejects_unknown_changed_or_escaping_source(
    alias_plan: tuple[Path, Path], mode: str
) -> None:
    config, alias = alias_plan
    data = json.loads(config.read_text())
    if mode == "unapproved":
        del data["approvedOriginal"]
    elif mode == "linked-approval":
        data["approvedOriginal"]["path"] = str(alias)
    elif mode == "changed-bytes":
        data["approvedOriginal"]["sha256"] = "0" * 64
    else:
        alias.unlink()
        if mode == "other-target":
            outside = config.parent / "unapproved-scope"
            outside.mkdir()
            other = outside / "same-bytes.mov"
            other.write_bytes((config.parent / "original.mov").read_bytes())
            alias.symlink_to(other)
    config.write_text(json.dumps(data))
    with pytest.raises(ReplayError):
        followup.prepare(config, config.parent / "refused", integrated=True)
    assert not (config.parent / "refused").exists()


@pytest.mark.parametrize("field", ["source", "sourceSHA256", "chunks"])
def test_approved_alias_requires_exact_pinned_producer(
    alias_plan: tuple[Path, Path], field: str
) -> None:
    config, _ = alias_plan
    producer, manifest, cleanup = (
        config.parent / name
        for name in ("alias-producer.json", "alias-manifest.json", "cleanup.json")
    )
    raw = json.loads(producer.read_text())
    raw[field] = "unrelated"  # source fields are evidence, never authority.
    producer.write_text(json.dumps(raw))
    manifest.write_text(
        json.dumps({"chunks": [] if field == "chunks" else [{"producer": _reference(producer)}]})
    )
    clean = json.loads(cleanup.read_text())
    clean["assemblyManifest"] = _reference(manifest)
    cleanup.write_text(json.dumps(clean))
    data = json.loads(config.read_text())
    data["sourceCleanup"] = _reference(cleanup)
    config.write_text(json.dumps(data))
    with pytest.raises(ReplayError, match="producer"):
        followup.validate_cleanup(config, config.parent / "semantic.fcpxml")


def test_approved_alias_rechecks_target_after_consumer(
    alias_plan: tuple[Path, Path], monkeypatch: pytest.MonkeyPatch
) -> None:
    config, alias = alias_plan
    other = config.parent / "same-bytes.mov"
    other.write_bytes(alias.read_bytes())

    def swap(*args: object) -> None:
        alias.unlink()
        alias.symlink_to(other)

    monkeypatch.setattr(followup, "_consumer", swap)
    with pytest.raises(ReplayError, match="target changed"):
        followup.validate_cleanup(config, config.parent / "semantic.fcpxml")


def test_approved_alias_rechecks_same_byte_retarget_after_hash(
    alias_plan: tuple[Path, Path], monkeypatch: pytest.MonkeyPatch
) -> None:
    config, alias = alias_plan
    other = config.parent / "identical.mov"
    other.write_bytes(alias.read_bytes())
    target = alias.resolve()

    def swapped_pin(value: object) -> Path:
        result = _pin(value)
        if result == target:
            alias.unlink()
            alias.symlink_to(other)
        return result

    monkeypatch.setattr(followup, "_pin", swapped_pin)
    with pytest.raises(ReplayError, match="during validation"):
        followup.validate_cleanup(config, config.parent / "semantic.fcpxml")


def test_approved_alias_completed_verification_is_read_only_and_rejects_retarget(
    alias_plan: tuple[Path, Path], integrated_core: list[str]
) -> None:
    config, alias = alias_plan
    graph = followup.prepare(config, config.parent / "alias-completed", integrated=True)
    operation, semantic = graph.parent / "operation.json", config.parent / "semantic.fcpxml"
    followup.run_integrated(operation, semantic)
    outputs = [Path(path) for path in json.loads(graph.read_text())["stages"][0]["outputs"]]
    before = [path.read_bytes() for path in outputs]
    followup.verify_integrated(operation, semantic)
    assert integrated_core == ["frames", "complete", "publish"]
    assert [path.read_bytes() for path in outputs] == before
    other = config.parent / "other-original.mov"
    other.write_bytes(alias.read_bytes())
    alias.unlink()
    alias.symlink_to(other)
    with pytest.raises(ReplayError, match="target changed"):
        followup.verify_integrated(operation, semantic)
    assert integrated_core == ["frames", "complete", "publish"]
    assert [path.read_bytes() for path in outputs] == before


@pytest.mark.parametrize("threads", [0, 1, 64])
def test_standalone_frames_uses_the_pinned_tool_and_explicit_policy(
    plan: Path, monkeypatch: pytest.MonkeyPatch, threads: int
) -> None:
    data = json.loads(plan.read_text())
    data["probeDecoderThreads"] = threads
    plan.write_text(json.dumps(data))
    observed: list[FrameProbeTool] = []

    def probe(config: Path, *, part: int, tool: FrameProbeTool) -> Path:
        assert config == plan.parent / "staging.json" and part == 1
        tool.verify()
        observed.append(tool)
        return plan.parent / "proof.json"

    monkeypatch.setattr(followup, "probe", probe)
    monkeypatch.setattr(
        sys, "argv", ["followup", "frames", str(plan), str(plan.parent / "semantic.fcpxml")]
    )
    followup.main()
    assert len(observed) == 1
    assert observed[0].path == plan.parent / "ffprobe"
    assert observed[0].decoder_threads == threads
    assert observed[0].sha256 == data["ffprobe"]["sha256"]


@pytest.mark.parametrize("threads", [-1, 65, True, "0", None])
def test_invalid_probe_policy_fails_before_factory_effects(plan: Path, threads: object) -> None:
    data = json.loads(plan.read_text())
    data["probeDecoderThreads"] = threads
    plan.write_text(json.dumps(data))
    output = plan.parent / "invalid-policy"
    with pytest.raises(ReplayError, match="threads"):
        followup.prepare(plan, output, integrated=True)
    assert not output.exists()


def test_changed_probe_policy_invalidates_pinned_operation_without_dispatch(
    integrated_plan: tuple[Path, Path], integrated_core: list[str]
) -> None:
    config, graph = integrated_plan
    data = json.loads(config.read_text())
    data["probeDecoderThreads"] = 1
    config.write_text(json.dumps(data))
    with pytest.raises(ReplayError, match="changed"):
        followup.run_integrated(graph.parent / "operation.json", config.parent / "semantic.fcpxml")
    assert integrated_core == []


@pytest.fixture
def plan(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    movie = tmp_path / "observed.mov"
    movie.write_bytes(b"actual observed native bytes")
    original, clean, voice = (
        tmp_path / "original.mov",
        tmp_path / "clean.mov",
        tmp_path / "voice.wav",
    )
    original.write_bytes(b"original")
    clean.write_bytes(b"locally repaired")
    voice.write_bytes(b"processed voice")
    semantic = tmp_path / "semantic.fcpxml"
    semantic.write_text(
        f'<fcpxml><resources><asset hasVideo="1"><media-rep src="{clean.as_uri()}"/>'
        "</asset></resources></fcpxml>",
        encoding="utf-8",
    )
    xml = tmp_path / "part.fcpxml"
    xml.write_bytes(semantic.read_bytes())
    evidence = tmp_path / "evidence.json"
    _save(evidence, {"version": 1})
    executable = tmp_path / "ffprobe"
    executable.write_bytes(b"synthetic pinned ffprobe")
    executable.chmod(0o700)
    partition = tmp_path / "partition.json"
    _save(partition, {"version": 1, "source": _reference(semantic), "parts": [_reference(xml)]})
    staging, published = tmp_path / "staging.json", tmp_path / "published.json"
    common = {
        "version": 1,
        "partition": _reference(partition),
        "voice": _reference(voice),
        "voiceReceipt": _reference(evidence),
    }
    _save(
        staging, {**common, "parts": [{"movie": str(movie), "proof": str(tmp_path / "proof.json")}]}
    )
    _save(
        published,
        {
            **common,
            "parts": [
                {"movie": str(tmp_path / "ssd.mov"), "proof": str(tmp_path / "ssd-proof.json")}
            ],
        },
    )
    cleanup = tmp_path / "cleanup.json"
    _save(
        cleanup,
        {
            "version": 1,
            "consumer": {**_reference(semantic), "kind": "fcpxml"},
            "assemblyManifest": _reference(evidence),
            "assemblyReceipt": _reference(evidence),
            "cleanIndex": _reference(evidence),
            "voiceReceipt": _reference(evidence),
        },
    )
    transitive = [original, clean]
    for name in ("chunk", "producer", "probe"):
        item = tmp_path / f"{name}.json"
        item.write_text(name, encoding="utf-8")
        transitive.append(item)
    cleanup_data = json.loads(cleanup.read_text(encoding="utf-8"))
    cleanup_data["syntheticTransitive"] = [_reference(path) for path in transitive]
    cleanup.write_text(json.dumps(cleanup_data), encoding="utf-8")

    def checked_cleanup(config: dict[str, object]) -> tuple[Path, Path, Fraction]:
        pins = config["syntheticTransitive"]
        assert isinstance(pins, list)
        for pin in pins:
            _pin(pin)
        return original, clean, Fraction(3)

    monkeypatch.setattr(followup, "clean_source", checked_cleanup)
    monkeypatch.setattr(source_caption_binding, "clean_source", checked_cleanup)
    config = tmp_path / "config.json"
    _save(
        config,
        {
            "version": 1,
            "part": 1,
            "audioSampleRate": 48000,
            "audioChannels": 2,
            "assembly": _reference(staging),
            "publishedAssembly": _reference(published),
            "sourceCleanup": _reference(cleanup),
            "exportRequest": _reference(evidence),
            "exportResponse": _reference(evidence),
            "background": _reference(evidence),
            "ffmpeg": _reference(evidence),
            "ffprobe": _reference(executable),
            "receiptOutput": str(tmp_path / "publication.json"),
        },
    )
    return config


@pytest.fixture
def integrated_plan(plan: Path, monkeypatch: pytest.MonkeyPatch) -> tuple[Path, Path]:
    extras = tuple(plan.parent / name for name in ("chunk.json", "producer.json", "probe.json"))
    monkeypatch.setattr(followup, "_cleanup_inputs", lambda _: extras)
    graph = followup.prepare(plan, plan.parent / "integrated", integrated=True)
    return plan, graph


@pytest.fixture
def integrated_core(plan: Path, monkeypatch: pytest.MonkeyPatch) -> list[str]:
    calls: list[str] = []
    root = plan.parent

    def probe(_: Path, *, part: int, tool: FrameProbeTool) -> Path:
        assert part == 1
        assert tool.path == root / "ffprobe" and tool.decoder_threads == 0
        tool.verify()
        calls.append("frames")
        path = root / "proof.json"
        _save(path, {"movie": _reference(root / "observed.mov"), "part": part})
        return path

    def verify(_: Path, *, part: int) -> Path:
        assert part == 1
        return root / "proof.json"

    def complete(config: Path, output: Path, *, recover: bool) -> None:
        assert json.loads(config.read_text())["part"] == 1
        assert recover is False
        calls.append("complete")
        _save(
            output,
            {"completionVerified": True, "fullAVDecodeExitCode": 0, "qualityApproved": False},
        )

    def completed(path: Path) -> None:
        assert json.loads(path.read_text())["completionVerified"] is True

    def publish(config: Path) -> Path:
        calls.append("publish")
        raw = json.loads(config.read_text())
        destination = Path(raw["destination"])
        destination.write_bytes((root / "observed.mov").read_bytes())
        _save(Path(raw["frameProofOutput"]), {"movie": _reference(destination), "part": 1})
        _save(
            Path(raw["receiptOutput"]), {"movie": _reference(destination), "qualityApproved": False}
        )
        return Path(raw["receiptOutput"])

    def verify_publication(_: Path, *, part: int, publication: Path) -> None:
        assert part == 1
        _pin(json.loads(publication.read_text())["movie"])

    monkeypatch.setattr(followup, "probe", probe)
    monkeypatch.setattr(followup, "verify", verify)
    monkeypatch.setattr(followup, "complete", complete)
    monkeypatch.setattr(followup, "_completed", completed)
    monkeypatch.setattr(followup, "publish", publish)
    monkeypatch.setattr(followup, "verify_publication", verify_publication)
    return calls


def test_integrated_graph_and_fresh_reuse_preserve_all_substeps(
    integrated_plan: tuple[Path, Path], integrated_core: list[str]
) -> None:
    plan, graph = integrated_plan
    raw = json.loads(graph.read_text())
    assert len(raw["stages"]) == 1
    stage = raw["stages"][0]
    assert stage["name"] == "followup"
    assert len(stage["outputs"]) == 7
    assert raw["sourceCaptionProduction"]["bindings"] == {
        "followup": str(plan.parent / "cleanup.json")
    }
    for name in ("chunk.json", "producer.json", "probe.json"):
        refs = json.loads((graph.parent / "operation.json").read_text())["inputs"]
        assert str(plan.parent / name) in {ref["path"] for ref in refs}
    assert workflow_main([str(graph), "--state", str(plan.parent / "preflight"), "--check"]) == 0
    operation, semantic = graph.parent / "operation.json", plan.parent / "semantic.fcpxml"
    followup.run_integrated(operation, semantic)
    assert integrated_core == ["frames", "complete", "publish"]
    receipt = json.loads((graph.parent / "verified-followup.json").read_text())
    assert receipt["qualityApproved"] is False
    assert receipt["hashReads"] == 2 * len(receipt["inputs"])
    assert receipt["cacheHits"] > 0
    followup.verify_integrated(operation, semantic)
    assert integrated_core == ["frames", "complete", "publish"]
    with pytest.raises(ReplayError, match="prior effects"):
        followup.run_integrated(operation, semantic)


@pytest.mark.parametrize("boundary", ["probe", "complete", "publication_config", "publish"])
@pytest.mark.parametrize("name", ["clean.mov", "chunk.json", "producer.json", "probe.json"])
def test_integrated_mutation_at_every_substep_is_rejected(
    integrated_plan: tuple[Path, Path],
    integrated_core: list[str],
    monkeypatch: pytest.MonkeyPatch,
    boundary: str,
    name: str,
) -> None:
    plan, graph = integrated_plan
    original = getattr(followup, boundary)

    def mutated(*args: object, **kwargs: object) -> object:
        result = original(*args, **kwargs)
        target = plan.parent / name
        target.write_bytes(target.read_bytes() + b" changed")
        return result

    monkeypatch.setattr(followup, boundary, mutated)
    with pytest.raises(ReplayError, match="changed"):
        followup.run_integrated(graph.parent / "operation.json", plan.parent / "semantic.fcpxml")
    assert not (graph.parent / "verified-followup.json").exists()
    if boundary != "publish":
        assert not (plan.parent / "publication.json").exists()


@pytest.mark.parametrize("artifact", ["intent", "partial"])
def test_integrated_uncertain_publication_effects_never_retried(
    integrated_plan: tuple[Path, Path], integrated_core: list[str], artifact: str
) -> None:
    plan, graph = integrated_plan
    path = plan.parent / (
        "publication.intent.json" if artifact == "intent" else ".ssd.mov.publication-partial"
    )
    path.write_bytes(b"uncertain side effect")
    with pytest.raises(ReplayError, match="prior effects"):
        followup.run_integrated(graph.parent / "operation.json", plan.parent / "semantic.fcpxml")
    assert not integrated_core
    assert path.read_bytes() == b"uncertain side effect"


def test_integrated_partial_failure_preserves_evidence_and_refuses_completion(
    integrated_plan: tuple[Path, Path], integrated_core: list[str], monkeypatch: pytest.MonkeyPatch
) -> None:
    plan, graph = integrated_plan
    intent = plan.parent / "publication.intent.json"

    def failed(_: Path) -> Path:
        intent.write_bytes(b"uncertain publication")
        raise ReplayError("publication interrupted")

    monkeypatch.setattr(followup, "publish", failed)
    operation, semantic = graph.parent / "operation.json", plan.parent / "semantic.fcpxml"
    with pytest.raises(ReplayError, match="interrupted"):
        followup.run_integrated(operation, semantic)
    assert intent.exists() and (plan.parent / "proof.json").exists()
    assert (graph.parent / "completion.json").exists()
    with pytest.raises(FileNotFoundError):
        followup.verify_integrated(operation, semantic)
    with pytest.raises(ReplayError, match="prior effects"):
        followup.run_integrated(operation, semantic)


@pytest.mark.parametrize(
    "name", ["clean.mov", "chunk.json", "producer.json", "probe.json", "ssd.mov"]
)
def test_integrated_reuse_hashes_current_bytes_again(
    integrated_plan: tuple[Path, Path], integrated_core: list[str], name: str
) -> None:
    plan, graph = integrated_plan
    operation, semantic = graph.parent / "operation.json", plan.parent / "semantic.fcpxml"
    followup.run_integrated(operation, semantic)
    target = plan.parent / name
    target.write_bytes(target.read_bytes() + b" changed after completion")
    with pytest.raises(ReplayError):
        followup.verify_integrated(operation, semantic)


@pytest.mark.parametrize(
    "change", ["version", "kind", "semantic", "empty", "duplicate", "self", "closure", "completion"]
)
def test_integrated_inventory_cannot_omit_or_substitute_dependencies(
    integrated_plan: tuple[Path, Path], integrated_core: list[str], change: str
) -> None:
    plan, graph = integrated_plan
    operation = graph.parent / "operation.json"
    raw = json.loads(operation.read_text())
    if change in {"version", "kind", "semantic"}:
        raw[change] = "wrong"
    elif change == "empty":
        raw["inputs"] = []
    elif change == "duplicate":
        raw["inputs"].append(raw["inputs"][0])
    elif change == "self":
        raw["inputs"].append({"path": str(operation), "sha256": "a" * 64})
    elif change == "closure":
        raw["inputs"] = [
            ref for ref in raw["inputs"] if ref["path"] != str(plan.parent / "chunk.json")
        ]
    else:
        raw["completionConfig"] = str(plan.parent / "evidence.json")
    operation.write_text(json.dumps(raw))
    with pytest.raises(ReplayError):
        followup.run_integrated(operation, plan.parent / "semantic.fcpxml")
    assert not integrated_core


@pytest.mark.parametrize("change", ["kind", "outputs", "inputs", "quality"])
def test_integrated_terminal_receipt_cannot_claim_success_without_exact_proof(
    integrated_plan: tuple[Path, Path], integrated_core: list[str], change: str
) -> None:
    plan, graph = integrated_plan
    operation, semantic = graph.parent / "operation.json", plan.parent / "semantic.fcpxml"
    followup.run_integrated(operation, semantic)
    path = graph.parent / "verified-followup.json"
    raw = json.loads(path.read_text())
    raw[{"quality": "qualityApproved"}.get(change, change)] = True if change == "quality" else []
    path.write_text(json.dumps(raw))
    with pytest.raises(ReplayError):
        followup.verify_integrated(operation, semantic)


@pytest.mark.parametrize("failure", [None, "rows", "chunks", "count", "probes"])
def test_cleanup_inventory_uses_only_current_bounded_reference_paths(
    tmp_path: Path, failure: str | None
) -> None:
    media = tmp_path / "chunk.mp4"
    media.write_bytes(b"repaired source chunk")
    candidate, producer, probe = (
        tmp_path / f"{name}.json" for name in ("candidate", "producer", "probe")
    )
    for path in (candidate, producer, probe):
        _save(path, {"historicalSource": "/historical/unavailable-original.mp4"})
    index = tmp_path / "index.json"
    _save(index, {"ranges": None if failure == "rows" else [{"media": str(media)}]})
    manifest = tmp_path / "manifest.json"
    refs = {"candidate": _reference(candidate), "producer": _reference(producer)}
    _save(
        manifest,
        {
            "index": _reference(index),
            "chunks": None if failure == "chunks" else [refs] * (2 if failure == "count" else 1),
        },
    )
    receipt, binding = tmp_path / "receipt.json", tmp_path / "binding.json"
    _save(
        receipt,
        {"probeEvidence": {} if failure == "probes" else {str(probe): _reference(probe)["sha256"]}},
    )
    _save(
        binding, {"assemblyManifest": _reference(manifest), "assemblyReceipt": _reference(receipt)}
    )
    if failure:
        with pytest.raises(ReplayError):
            followup._cleanup_inputs(binding)
    else:
        assert followup._cleanup_inputs(binding) == (index, candidate, producer, media, probe)


@pytest.mark.parametrize("operation", ["run-integrated", "verify-integrated"])
def test_integrated_cli_uses_existing_immutable_operation(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, operation: str
) -> None:
    calls: list[tuple[Path, Path]] = []
    function = "run_integrated" if operation == "run-integrated" else "verify_integrated"
    monkeypatch.setattr(followup, function, lambda left, right: calls.append((left, right)))
    plan, semantic = tmp_path / "operation.json", tmp_path / "semantic.fcpxml"
    monkeypatch.setattr(sys, "argv", ["followup", operation, str(plan), str(semantic)])
    followup.main()
    assert calls == [(plan, semantic)]


def test_large_cleanup_inventory_preserves_bounded_workflow_and_uncached_bytes(
    plan: Path, integrated_core: list[str], monkeypatch: pytest.MonkeyPatch
) -> None:
    root = plan.parent
    chunks: list[dict[str, object]] = []
    ranges: list[dict[str, object]] = []
    probes: dict[str, object] = {}
    for number in range(220):
        media, producer, candidate = (
            root / f"{name}-{number}" for name in ("media", "producer", "candidate")
        )
        for path in (media, producer, candidate):
            path.write_bytes(b"bounded synthetic cleanup dependency")
        chunks.append({"producer": _reference(producer), "candidate": _reference(candidate)})
        ranges.append({"media": str(media)})
        for kind in range(3):
            probe = root / f"probe-{number}-{kind}"
            probe.write_bytes(b"synthetic frame or packet evidence")
            probes[str(probe)] = _reference(probe)["sha256"]
    index, manifest, receipt = (
        root / f"large-{name}.json" for name in ("index", "manifest", "receipt")
    )
    _save(index, {"ranges": ranges})
    _save(manifest, {"index": _reference(index), "chunks": chunks})
    _save(receipt, {"probeEvidence": probes})
    cleanup = root / "cleanup.json"
    data = json.loads(cleanup.read_text())
    data["assemblyManifest"], data["assemblyReceipt"] = _reference(manifest), _reference(receipt)
    cleanup.write_text(json.dumps(data))
    config = json.loads(plan.read_text())
    config["sourceCleanup"] = _reference(cleanup)
    plan.write_text(json.dumps(config))
    graph = followup.prepare(plan, root / "large-operation", integrated=True)
    stage = json.loads(graph.read_text())["stages"][0]
    assert len(stage["inputs"]) < 256
    assert workflow_main([str(graph), "--state", str(root / "large-check"), "--check"]) == 0
    operation, semantic = graph.parent / "operation.json", root / "semantic.fcpxml"
    followup.run_integrated(operation, semantic)
    followup.verify_integrated(operation, semantic)
    # A path beyond the 1024 cached entries is still pinned, guarded and hashed.
    (root / "probe-219-2").write_bytes(b"changed after publication")
    with pytest.raises(ReplayError, match="changed"):
        followup.verify_integrated(operation, semantic)


def test_graph_preserves_all_verifiers_and_automatic_pin_transition(plan: Path) -> None:
    graph = followup.prepare(plan, plan.parent / "followup")
    raw = json.loads(graph.read_text(encoding="utf-8"))
    assert [stage["name"] for stage in raw["stages"]] == [
        "frames",
        "complete",
        "publication-config",
        "publish",
    ]
    assert raw["stages"][0]["command"][-2:] == [str(plan), str(plan.parent / "semantic.fcpxml")]
    assert raw["stages"][1]["command"][3] == "replay.fcp_part_followup"
    assert "--recover" not in raw["stages"][1]["command"]
    assert raw["stages"][1]["verify_command"][2] == (
        ".completionVerified == true and .fullAVDecodeExitCode == 0 and .shareObservedIdle == true"
    )
    assert raw["stages"][3]["verify_command"][4] == "verify-publication"
    assert str(plan.parent / "cleanup.json") in raw["stages"][3]["inputs"]
    assert str(plan.parent / "clean.mov") in raw["stages"][3]["inputs"]
    assert raw["sourceCaptionProduction"]["bindings"] == {
        stage["name"]: str(plan.parent / "cleanup.json") for stage in raw["stages"]
    }
    for stage in raw["stages"]:
        assert str(plan.parent / "semantic.fcpxml") in stage["command"]
    completion = json.loads((graph.parent / "completion-config.json").read_text(encoding="utf-8"))
    assert completion["audioSampleRate"] == 48000
    assert completion["audioChannels"] == 2
    assert workflow_main([str(graph), "--state", str(plan.parent / "state"), "--check"]) == 0


def test_explicit_recovery_only_delegates_existing_chronology(plan: Path) -> None:
    raw = json.loads(plan.read_text(encoding="utf-8"))
    raw["recover"] = True
    raw["projectObservation"] = raw["background"]
    plan.write_text(json.dumps(raw), encoding="utf-8")
    graph = followup.prepare(plan, plan.parent / "recovery")
    stages = json.loads(graph.read_text(encoding="utf-8"))["stages"]
    assert stages[1]["command"][4] == "complete"
    assert stages[1]["command"][3] == "replay.fcp_part_followup"
    assert "projectObservation" in json.loads(
        (graph.parent / "completion-config.json").read_text(encoding="utf-8")
    )


@pytest.mark.parametrize(
    "change",
    [
        "version",
        "part",
        "recover",
        "audio-rate",
        "channels",
        "missing-cleanup",
        "non-recovery-project",
    ],
)
def test_invalid_or_unpinned_inputs_rejected(plan: Path, change: str) -> None:
    raw = json.loads(plan.read_text(encoding="utf-8"))
    if change in ("version", "part", "recover"):
        raw[change] = "1"
    elif change == "audio-rate":
        raw["audioSampleRate"] = True
    elif change == "channels":
        raw["audioChannels"] = 0
    elif change == "missing-cleanup":
        del raw["sourceCleanup"]
    else:
        raw["projectObservation"] = raw["background"]
    plan.write_text(json.dumps(raw), encoding="utf-8")
    with pytest.raises(ReplayError):
        followup.prepare(plan, plan.parent / "refused")
    assert not (plan.parent / "refused").exists()


@pytest.mark.parametrize(
    "change",
    [
        "assembly-version",
        "clock",
        "voice",
        "parts",
        "partition",
        "cleanup-consumer",
        "raw-source",
        "missing-movie",
        "empty-movie",
        "same-output",
        "occupied",
        "missing-parent",
        "graph-exists",
    ],
)
def test_binding_output_and_clock_guard_failures(plan: Path, change: str) -> None:
    raw = json.loads(plan.read_text(encoding="utf-8"))
    assembly = Path(raw["publishedAssembly"]["path"])
    data = json.loads(assembly.read_text(encoding="utf-8"))
    if change == "assembly-version":
        data["version"] = 2
    elif change == "clock":
        data["partition"] = raw["background"]
    elif change == "voice":
        data["voiceReceipt"] = None
    elif change == "parts":
        data["parts"] = []
    elif change == "same-output":
        data["parts"][0]["movie"] = str(plan.parent / "observed.mov")
    elif change == "missing-parent":
        data["parts"][0]["movie"] = str(plan.parent / "absent/ssd.mov")
    elif change == "partition":
        path = Path(data["partition"]["path"])
        path.write_text('{"parts":[]}', encoding="utf-8")
        data["partition"] = _reference(path)
        staged_path = Path(raw["assembly"]["path"])
        staged = json.loads(staged_path.read_text(encoding="utf-8"))
        staged["partition"] = _reference(path)
        staged_path.write_text(json.dumps(staged), encoding="utf-8")
        raw["assembly"] = _reference(staged_path)
    elif change == "cleanup-consumer":
        path = Path(raw["sourceCleanup"]["path"])
        cleanup = json.loads(path.read_text(encoding="utf-8"))
        cleanup["consumer"] = raw["background"]
        path.write_text(json.dumps(cleanup), encoding="utf-8")
        raw["sourceCleanup"] = _reference(path)
    elif change == "raw-source":
        path = plan.parent / "part.fcpxml"
        path.write_text(
            '<fcpxml><resources><asset hasVideo="1"/></resources></fcpxml>', encoding="utf-8"
        )
    elif change == "missing-movie":
        (plan.parent / "observed.mov").unlink()
    elif change == "empty-movie":
        (plan.parent / "observed.mov").write_bytes(b"")
    elif change == "occupied":
        (plan.parent / "ssd.mov").write_bytes(b"existing")
    else:
        (plan.parent / "output").mkdir()
    assembly.write_text(json.dumps(data), encoding="utf-8")
    raw["publishedAssembly"] = _reference(assembly)
    plan.write_text(json.dumps(raw), encoding="utf-8")
    with pytest.raises(ReplayError):
        followup.prepare(plan, plan.parent / "output")


def test_publication_config_pins_completed_receipt_without_publication(plan: Path) -> None:
    movie = plan.parent / "observed.mov"
    proof = plan.parent / "completed-proof.json"
    _save(proof, {"part": 1, "movie": _reference(movie)})
    completion = plan.parent / "completion.json"
    _save(
        completion,
        {
            "version": 1,
            "kind": "verified-native-part",
            "part": 1,
            "completionVerified": True,
            "shareObservedIdle": True,
            "fullAVDecodeExitCode": 0,
            "movie": _reference(movie),
            "frameProof": _reference(proof),
            "inputs": {str(movie): _reference(movie)["sha256"]},
        },
    )
    output = plan.parent / "publication-config.json"
    followup.publication_config(
        completion,
        destination=plan.parent / "ssd.mov",
        receipt=plan.parent / "receipt.json",
        proof=plan.parent / "ssd-proof.json",
        output=output,
    )
    data = json.loads(output.read_text(encoding="utf-8"))
    assert data["version"] == 1
    assert data["completion"] == _reference(completion)
    assert not (plan.parent / "ssd.mov").exists()
    with pytest.raises(FileExistsError):
        followup.publication_config(
            completion,
            destination=plan.parent / "ssd.mov",
            receipt=plan.parent / "receipt.json",
            proof=plan.parent / "ssd-proof.json",
            output=output,
        )
    movie.write_bytes(b"changed")
    with pytest.raises(ReplayError):
        followup.publication_config(
            completion,
            destination=plan.parent / "ssd.mov",
            receipt=plan.parent / "receipt.json",
            proof=plan.parent / "ssd-proof.json",
            output=plan.parent / "changed-config.json",
        )


@pytest.mark.parametrize(
    "operation",
    [
        "prepare",
        "publication-config",
        "verify-publication",
        "frames",
        "verify-frames",
        "complete",
        "publish",
        "validate-cleanup",
    ],
)
def test_cli_routes_without_gui_or_submit(
    plan: Path, monkeypatch: pytest.MonkeyPatch, operation: str
) -> None:
    calls: list[str] = []
    if operation == "prepare":
        arguments = [str(plan), str(plan.parent / "out")]
    elif operation == "publication-config":
        arguments = [str(plan), str(plan.parent / "semantic.fcpxml"), *([str(plan)] * 5)]
        monkeypatch.setattr(followup, "_completed", lambda path: ({}, path))
        monkeypatch.setattr(
            followup, "_save", lambda path, document: calls.append("publication-config")
        )
    elif operation == "verify-publication":
        arguments = [
            str(plan),
            str(plan.parent / "semantic.fcpxml"),
            str(plan.parent / "published.json"),
            "1",
            str(plan),
        ]
        monkeypatch.setattr(followup, "verify", lambda assembly, *, part: plan)
        monkeypatch.setattr(
            followup,
            "verify_published",
            lambda publication, proof, *, part: calls.append("verify-publication"),
        )
    else:
        arguments = [str(plan), str(plan.parent / "semantic.fcpxml")]
        if operation == "complete":
            arguments.extend([str(plan), str(plan)])
            monkeypatch.setattr(
                followup, "complete", lambda config, output, *, recover: calls.append("complete")
            )
        elif operation == "publish":
            arguments.append(str(plan))
            monkeypatch.setattr(followup, "publish", lambda config: calls.append("publish"))
        elif operation == "frames":
            monkeypatch.setattr(
                followup, "probe", lambda config, *, part, tool: calls.append("frames")
            )
        elif operation == "verify-frames":
            monkeypatch.setattr(
                followup, "verify", lambda config, *, part: calls.append("verify-frames")
            )
    monkeypatch.setattr(sys, "argv", ["followup", operation, *arguments])
    followup.main()
    if operation == "prepare":
        assert (plan.parent / "out/workflow.json").is_file()
    elif operation == "publication-config":
        assert calls == ["publication-config"]
    elif operation == "validate-cleanup":
        assert calls == []
    else:
        assert calls == [operation]


@pytest.mark.parametrize(
    "changed", ["cleanup.json", "clean.mov", "chunk.json", "producer.json", "probe.json"]
)
@pytest.mark.parametrize("result", ["completed", "reused", "adopted"])
def test_workflow_revalidates_cleanup_before_execution_reuse_and_adoption(
    plan: Path, monkeypatch: pytest.MonkeyPatch, changed: str, result: str
) -> None:
    graph = followup.prepare(plan, plan.parent / "followup")
    calls: list[str] = []

    def observed_stage(stage: Stage, *, state_directory: Path) -> str:
        calls.append(stage.name)
        return result

    monkeypatch.setattr(workflow, "run_stage", observed_stage)
    workflow.run_plan(graph, state_directory=plan.parent / "valid-state", targets=("frames",))
    assert calls == ["frames"]
    calls.clear()
    if changed == "cleanup.json":
        cleanup = json.loads((plan.parent / changed).read_text(encoding="utf-8"))
        cleanup["version"] = 2
        (plan.parent / changed).write_text(json.dumps(cleanup), encoding="utf-8")
    else:
        (plan.parent / changed).write_bytes(b"changed after preparation")
    with pytest.raises(StageError, match="Source-caption repair binding failed"):
        workflow.run_plan(graph, state_directory=plan.parent / "changed-state", targets=("frames",))
    assert calls == []


@pytest.mark.parametrize(
    "operation",
    ["frames", "verify-frames", "complete", "publish", "publication-config", "verify-publication"],
)
def test_direct_wrapper_rejects_changed_transitive_cleanup_before_effect(
    plan: Path, monkeypatch: pytest.MonkeyPatch, operation: str
) -> None:
    arguments = [str(plan), str(plan.parent / "semantic.fcpxml")]
    if operation == "complete":
        arguments.extend([str(plan), str(plan)])
    elif operation == "publish":
        arguments.append(str(plan))
    elif operation == "publication-config":
        arguments.extend([str(plan)] * 5)
    elif operation == "verify-publication":
        arguments.extend([str(plan.parent / "published.json"), "1", str(plan)])
    (plan.parent / "producer.json").write_bytes(b"changed after preparation")
    monkeypatch.setattr(sys, "argv", ["followup", operation, *arguments])
    with pytest.raises(ReplayError, match="hash"):
        followup.main()


def test_existing_outputs_do_not_prevent_readonly_cleanup_revalidation(plan: Path) -> None:
    (plan.parent / "ssd.mov").write_bytes(b"existing published output")
    assert followup.validate_cleanup(plan, plan.parent / "semantic.fcpxml") == (
        plan.parent / "staging.json",
        1,
        False,
    )
    with pytest.raises(ReplayError, match="Runtime semantic XML differs"):
        followup.validate_cleanup(plan, plan.parent / "part.fcpxml")
    raw = json.loads(plan.read_text(encoding="utf-8"))
    raw["part"] = False
    plan.write_text(json.dumps(raw), encoding="utf-8")
    with pytest.raises(ReplayError, match="version-one"):
        followup.validate_cleanup(plan, plan.parent / "semantic.fcpxml")


def test_runtime_cleanup_requires_selected_part_and_bound_semantic(plan: Path) -> None:
    raw = json.loads(plan.read_text(encoding="utf-8"))
    cleanup = Path(raw["sourceCleanup"]["path"])
    data = json.loads(cleanup.read_text(encoding="utf-8"))
    data["consumer"] = _reference(plan.parent / "part.fcpxml")
    cleanup.write_text(json.dumps(data), encoding="utf-8")
    raw["sourceCleanup"] = _reference(cleanup)
    plan.write_text(json.dumps(raw), encoding="utf-8")
    with pytest.raises(ReplayError, match="Runtime cleanup binding differs"):
        followup.validate_cleanup(plan, plan.parent / "semantic.fcpxml")


def test_publication_verifier_cannot_choose_another_part(
    plan: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "followup",
            "verify-publication",
            str(plan),
            str(plan.parent / "semantic.fcpxml"),
            str(plan.parent / "published.json"),
            "2",
            str(plan),
        ],
    )
    with pytest.raises(ReplayError, match="selected part"):
        followup.main()
