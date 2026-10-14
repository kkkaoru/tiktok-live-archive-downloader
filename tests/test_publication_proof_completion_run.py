"""Current boundary integration retains real assembly, material and clock checks."""

import fcntl
import json
import os
from dataclasses import dataclass
from pathlib import Path

import pytest
from test_recovery_source_finalization import Case
from test_recovery_source_finalization import case as case
from test_retained_cleanup_finalization import ready_case as ready_case
from test_retained_publication_support import published_case as published_case

from replay import cleanup_finalization_workflow as common
from replay import cleanup_lane_successor as successor
from replay import leaf_retirement as leaves
from replay import native_retired_hardlink_verification as custody
from replay import publication_proof_completion_seal as seal
from replay import retained_cleanup_finalization as retained
from replay import verified_hardlink_publication as retirement
from replay import workflow
from replay.core import ReplayError
from replay.input_hash_cache import InputHashCache, input_hash_context
from replay.recordings import record
from replay.stages import Stage, file_digest, run_stage


@dataclass(frozen=True)
class BoundaryCase:
    job: common.Job
    history: seal.PublicationHistory
    request: Path
    output: Path
    source: Path
    clock: Path
    proof: Path
    candidate: Path


@dataclass(frozen=True)
class SourceHistoryCase:
    graph: Path
    state: Path
    source: Path
    historical_sha256: str
    output: Path


@pytest.fixture
def boundary_case(published_case: Case, monkeypatch: pytest.MonkeyPatch) -> BoundaryCase:
    c = published_case
    job = retained._load(c.operation)
    work = c.plan.parent
    monkeypatch.setattr(seal, "ROOT", Path.cwd())
    settings = common._read(c.settings)
    profile = work / "profile"
    profile.mkdir()
    profile_input = profile / "inputs.json"
    profile_input.write_bytes(b"{}")
    settings.update(
        {
            "classificationEvidence": common._pin(c.model),
            "profileInputs": common._pin(profile_input),
            "profileComplete": common._pin(profile_input),
        }
    )
    c.settings.write_text(json.dumps(settings), encoding="utf-8")
    data = common._read(c.plan)
    original = Path(str(data["originalPlan"]))
    prior = common._read(original)
    prior["settings"] = common._pin(c.settings)
    original.write_text(json.dumps(prior), encoding="utf-8")
    manifest = common._read(c.output / "finalized/manifest.json")
    refs = successor._list(manifest["chunks"])
    rows = [record(value) for value in successor._list(data["chunks"])]
    legacy_rows: list[dict[str, object]] = []
    for index, row in enumerate(rows):
        chunk = Path(str(row["chunk"]))
        selected = common._read(chunk)
        selected["settings"] = common._pin(c.settings)
        chunk.write_text(json.dumps(selected), encoding="utf-8")
        (chunk.parent / "ocr-batches.json").write_bytes(b"[]")
        (chunk.parent / "admission.json").write_bytes(b"{}")
        (chunk.parent / "eligibility.json").write_bytes(b"{}")
        candidate_ref = record(record(refs[index])["candidate"])
        receipt = common._read(Path(str(candidate_ref["path"])))
        media = chunk.parent / "repair/candidate.mp4"
        repair = {
            "receipt": candidate_ref,
            "media": {"path": str(media), "sha256": receipt["videoSHA256"]},
        }
        row.update(
            {
                "originalChunk": str(chunk),
                "adoptedProof": None,
                "repairEvidence": repair if row["mode"] == "retained-completed" else None,
                "proof": str(work / f"verified-{index:05d}.json"),
                "oldState": str(c.states[index % len(c.states)]),
                "oldStageNames": [f"segment-{index}"],
            }
        )
        legacy_rows.append(
            {
                "original": str(chunk),
                "chunk": str(chunk),
                "lane": row["lane"],
                "oldStages": [{"state": str(work / "original-state"), "name": "old-repair"}],
            }
        )
    legacy_request = work / "legacy-request.json"
    legacy_request.write_text('{"oldWorkers":[]}', encoding="utf-8")
    handoff = work / "handoff.json"
    handoff.write_text('{"evidence":[]}', encoding="utf-8")
    legacy = work / "successor.json"
    legacy.write_text(
        json.dumps(
            {
                "request": str(legacy_request),
                "originalPlan": str(original),
                "handoff": str(handoff),
                "settings": str(c.settings),
                "chunks": legacy_rows,
                "workflows": [str(path) for path in c.graphs],
                "states": [str(path) for path in c.states],
                "oldStates": [str(work / "original-state")],
                "inputs": {str(legacy_request): file_digest(legacy_request)},
            }
        ),
        encoding="utf-8",
    )
    data.update(
        {
            "chunks": rows,
            "successor": str(legacy),
            "handoff": str(handoff),
            "oldStates": [str(work / "original-state")],
            "inputs": {
                str(path): file_digest(path)
                for path in (c.source, c.model, c.settings, original, legacy, handoff, *c.segments)
            },
        }
    )
    c.plan.write_text(json.dumps(data), encoding="utf-8")
    request_data = common._read(c.request)
    request_data["generation"] = common._pin(c.plan)
    c.request.write_text(json.dumps(request_data), encoding="utf-8")
    operation = common._read(c.operation)
    operation["request"] = common._pin(c.request)
    c.operation.write_text(json.dumps(operation), encoding="utf-8")
    for row in rows:
        chunk = Path(str(row["chunk"]))
        candidate = chunk.parent / "repair/candidate.json"
        candidate_data = common._read(candidate)
        proof = {
            "version": 1,
            "generation": common._pin(c.plan),
            "index": row["index"],
            "mode": row["mode"],
            "originalChunk": row["originalChunk"],
            "adoptedProof": None,
            "verification": {
                "version": 1,
                "kind": "readonly-original-cleanup-admission",
                "chunk": common._pin(chunk),
                "outputs": {
                    str(chunk.parent / name): file_digest(chunk.parent / name)
                    for name in (
                        "ocr-batches.json",
                        "admission.json",
                        "eligibility.json",
                        "candidate-config.json",
                    )
                },
                "repairVerified": True,
                "originalStageCompleted": False,
                "qualityApproved": False,
            },
            "repairEvidence": {
                "receipt": common._pin(candidate),
                "media": {
                    "path": str(chunk.parent / "repair/candidate.mp4"),
                    "sha256": candidate_data["videoSHA256"],
                },
            },
            "qualityApproved": False,
        }
        Path(str(row["proof"])).write_text(json.dumps(proof), encoding="utf-8")
    lineage_path = c.output / "finalized/lineage.json"
    lineage = common._read(lineage_path)
    lineage.update({"generation": common._pin(c.plan), "selectedResults": rows})
    lineage_path.write_text(json.dumps(lineage), encoding="utf-8")
    finalized_path = c.output / "finalized.json"
    finalized = common._read(finalized_path)
    finalized.update(
        {
            "operation": common._pin(c.operation),
            "outputs": {
                str(c.output / "finalized" / name): file_digest(c.output / "finalized" / name)
                for name in ("index.json", "manifest.json", "lineage.json")
            },
        }
    )
    finalized_path.write_text(json.dumps(finalized), encoding="utf-8")
    common._assembled(job)
    keeper = c.output / "clean-source.mov"
    alias = c.output / "retired-staging.mov"
    historical = {
        str(path): file_digest(path)
        for path in (
            c.operation,
            c.plan,
            c.request,
            finalized_path,
            lineage_path,
            c.output / "finalized/manifest.json",
            c.output / "finalized/index.json",
        )
    }
    child = Stage(
        name="observed-child",
        command=("synthetic-readonly",),
        inputs=tuple(Path(name) for name in historical),
        outputs=(
            keeper,
            c.output / "evidence/receipt.json",
            c.output / "evidence/single-source-index.json",
        ),
        timeout_seconds=60,
        verify_command=("synthetic-readonly",),
        adopt_existing=True,
    )
    child_graph = work / "observed-child.json"
    child_graph.write_text(
        json.dumps(
            {
                "version": 1,
                "stages": [
                    {
                        "name": child.name,
                        "command": list(child.command),
                        "inputs": [str(p) for p in child.inputs],
                        "outputs": [str(p) for p in child.outputs],
                        "timeout_seconds": child.timeout_seconds,
                        "verify_command": list(child.verify_command),
                        "adopt_existing": True,
                    }
                ],
            }
        ),
        encoding="utf-8",
    )
    child_state = work / "observed-child-state"

    def readonly(command: tuple[str, ...], log: Path, timeout: int) -> None:
        common._assembled(job)
        log.write_bytes(b"real synthetic readonly assembly domain")

    run_stage(child, state_directory=child_state, execute=readonly)
    checkpoint = child_state / "observed-child.json"
    custody_path = work / "custody.json"
    custody_path.write_bytes(b"{}")
    capture = work / "capture.json"
    capture.write_bytes(b"{}")
    history = seal.PublicationHistory(
        job,
        {"leases": [str(work / "custody.lock")]},
        historical,
        {},
        child,
        child,
        {str(p): file_digest(p) for p in child.outputs},
        (custody.Reference(checkpoint, file_digest(checkpoint)),),
        alias,
        keeper,
    )

    def synthetic_history(
        *,
        operation: Path,
        custody_seal: Path,
        publication_workflow: Path,
        adoption_checkpoint: Path,
        child_workflow: Path,
        child_checkpoint: Path,
    ) -> seal.PublicationHistory:
        custody._absent(alias)
        custody._identity(keeper)
        return history

    monkeypatch.setattr(seal, "load_publication_history", synthetic_history)
    monkeypatch.setattr(retirement, "_actors", lambda references: [{"terminalExit": 0}])
    monkeypatch.setattr(leaves, "lsof_open_among", lambda paths: set())
    request = work / "current-request.json"
    request.write_text(
        json.dumps(
            {
                "version": 1,
                "kind": seal.BOUNDARY_KIND,
                "operation": common._pin(c.operation),
                "custodySeal": common._pin(custody_path),
                "publicationWorkflow": common._pin(child_graph),
                "adoptionCheckpoint": common._pin(checkpoint),
                "childWorkflow": common._pin(child_graph),
                "childCheckpoint": common._pin(checkpoint),
                "captures": [common._pin(capture)],
            }
        ),
        encoding="utf-8",
    )
    return BoundaryCase(
        job,
        history,
        request,
        work / "current-boundary",
        c.source,
        Path(str(record(settings["sourceClock"])["path"])),
        Path(str(rows[0]["proof"])),
        Path(str(rows[0]["chunk"])).parent / "repair/candidate.mp4",
    )


def test_real_current_boundary_and_verify(boundary_case: BoundaryCase) -> None:
    c = boundary_case
    before = c.job.operation.read_bytes()
    graph = seal.prepare_current_boundary(c.request, c.output)
    assert len(workflow.load_stages(graph)) == 1
    result = seal.execute_current_boundary(c.output / "operation.json")
    completion = common._read(result)
    assert completion["historicalReaderEqualsCurrent"] is False
    assert completion["editCompletion"] is False
    assert c.job.operation.read_bytes() == before
    assert not (c.job.root / "completion.json").exists()
    assert seal.execute_current_boundary(c.output / "operation.json", verify=True) == result


@pytest.mark.parametrize("material", ["source", "candidate", "clock"])
def test_current_changed_material_rejected(boundary_case: BoundaryCase, material: str) -> None:
    c = boundary_case
    seal.prepare_current_boundary(c.request, c.output)
    paths = {"source": c.source, "candidate": c.candidate, "clock": c.clock}
    paths[material].write_bytes(b"changed actual material")
    with pytest.raises((ReplayError, ValueError)):
        seal.execute_current_boundary(c.output / "operation.json")
    assert not (c.output / "completion.json").exists()


def test_real_assembly_probe_change_rejected(boundary_case: BoundaryCase) -> None:
    c = boundary_case
    seal.prepare_current_boundary(c.request, c.output)
    (c.job.root / "evidence/probe.json").write_bytes(b"changed probe")
    with pytest.raises(ReplayError, match="hash changed"):
        seal.execute_current_boundary(c.output / "operation.json")


def test_keeper_hardlink_alias_rejected(boundary_case: BoundaryCase) -> None:
    c = boundary_case
    seal.prepare_current_boundary(c.request, c.output)
    os.link(c.history.keeper, c.output / "unexpected-alias.mov")
    with pytest.raises(ReplayError, match="single-link"):
        seal.execute_current_boundary(c.output / "operation.json")


def test_changed_current_reader_rejected(boundary_case: BoundaryCase) -> None:
    c = boundary_case
    seal.prepare_current_boundary(c.request, c.output)
    operation = c.output / "operation.json"
    data = common._read(operation)
    data["currentReader"] = {}
    operation.write_text(json.dumps(data), encoding="utf-8")
    with pytest.raises(ReplayError, match="reader or operation"):
        seal.execute_current_boundary(operation)


def test_current_namespace_does_not_overwrite(boundary_case: BoundaryCase) -> None:
    c = boundary_case
    seal.prepare_current_boundary(c.request, c.output)
    occupied = c.output / "completion.json"
    occupied.write_bytes(b"preserve occupied result")
    with pytest.raises(ReplayError, match="outputs already exist"):
        seal.execute_current_boundary(c.output / "operation.json")
    assert occupied.read_bytes() == b"preserve occupied result"
    with pytest.raises(ReplayError, match="namespace already exists"):
        seal.prepare_current_boundary(c.request, c.output)


def test_current_writer_lease_rejected(boundary_case: BoundaryCase) -> None:
    c = boundary_case
    seal.prepare_current_boundary(c.request, c.output)
    lease = c.job.lanes[0][1] / ".workflow.lock"
    with lease.open("w", encoding="utf-8") as stream:
        fcntl.flock(stream, fcntl.LOCK_EX | fcntl.LOCK_NB)
        with pytest.raises(ValueError, match="active"):
            seal.execute_current_boundary(c.output / "operation.json")


def test_current_lineage_recovery_proof_change_rejected(boundary_case: BoundaryCase) -> None:
    c = boundary_case
    proof = common._read(c.proof)
    proof["qualityApproved"] = True
    c.proof.write_text(json.dumps(proof), encoding="utf-8")
    with pytest.raises(ReplayError, match="recovery proof differs"):
        seal.current_lineage_validation(c.job, c.history.historical_inputs)


def test_lineage_requires_authenticated_metadata(boundary_case: BoundaryCase) -> None:
    with pytest.raises(ReplayError, match="authenticated historical pin"):
        seal.current_lineage_validation(boundary_case.job, {})


def test_bound_finalized_metadata_must_match_output_pins(boundary_case: BoundaryCase) -> None:
    c = boundary_case
    path = c.job.root / "finalized.json"
    document = common._read(path)
    document["physicalSourceFrames"] = 1
    path.write_text(json.dumps(document), encoding="utf-8")
    historical = dict(c.history.historical_inputs)
    historical[str(path)] = file_digest(path)
    with pytest.raises(ReplayError, match="Finalized publication differs"):
        seal.current_lineage_validation(c.job, historical)


def test_bound_lineage_approval_is_not_delivery_acceptance(boundary_case: BoundaryCase) -> None:
    c = boundary_case
    path = c.job.root / "finalized/lineage.json"
    document = common._read(path)
    document["qualityApproved"] = True
    path.write_text(json.dumps(document), encoding="utf-8")
    historical = dict(c.history.historical_inputs)
    historical[str(path)] = file_digest(path)
    finalized = c.job.root / "finalized.json"
    checkpoint = common._read(finalized)
    record(checkpoint["outputs"])[str(path)] = file_digest(path)
    finalized.write_text(json.dumps(checkpoint), encoding="utf-8")
    historical[str(finalized)] = file_digest(finalized)
    with pytest.raises(ReplayError, match="lineage selection differs"):
        seal.current_lineage_validation(c.job, historical)


def test_retained_repair_cannot_differ_from_bound_lineage(boundary_case: BoundaryCase) -> None:
    c = boundary_case
    metadata = common._read(c.job.plan)
    row = dict(record(successor._list(metadata["chunks"])[0]))
    row["repairEvidence"] = {}
    manifest = common._read(c.job.root / "finalized/manifest.json")
    indexed = common._read(c.job.root / "finalized/index.json")
    with pytest.raises(ReplayError, match="repair evidence differs"):
        seal._lineage_current_proof(
            job=c.job,
            row=row,
            reference=record(successor._list(manifest["chunks"])[0]),
            indexed=record(successor._list(indexed["ranges"])[0]),
            pins={str(c.job.plan): file_digest(c.job.plan)},
        )


def test_descriptor_guard_rejects_open_endpoint(
    boundary_case: BoundaryCase, monkeypatch: pytest.MonkeyPatch
) -> None:
    c = boundary_case
    seal.prepare_current_boundary(c.request, c.output)
    monkeypatch.setattr(leaves, "lsof_open_among", lambda paths: {str(c.history.keeper)})
    with pytest.raises(ReplayError, match="descriptor"):
        seal.execute_current_boundary(c.output / "operation.json")


def test_boundary_verify_rejects_changed_completion(boundary_case: BoundaryCase) -> None:
    c = boundary_case
    seal.prepare_current_boundary(c.request, c.output)
    output = seal.execute_current_boundary(c.output / "operation.json")
    output.write_bytes(b"{}")
    with pytest.raises(ReplayError, match="completion proof changed"):
        seal.execute_current_boundary(c.output / "operation.json", verify=True)


def test_selected_union_matches_original_contract(boundary_case: BoundaryCase) -> None:
    c = boundary_case
    metadata = common._read(c.job.plan)
    legacy = common._read(Path(str(metadata["successor"])))
    legacy_request = Path(str(legacy["request"]))
    legacy_request.write_text(
        json.dumps({"oldWorkers": [{"workflow": common._pin(c.job.lanes[0][0])}]}),
        encoding="utf-8",
    )
    evidence = c.job.plan.parent / "handoff-evidence.json"
    evidence.write_bytes(b"{}")
    Path(str(legacy["handoff"])).write_text(
        json.dumps({"evidence": [common._pin(evidence)]}), encoding="utf-8"
    )
    legacy["inputs"] = {
        str(evidence): file_digest(evidence),
        str(c.proof): file_digest(c.proof),
        str(c.candidate): file_digest(c.candidate),
        str(c.job.plan.parent / "unconsumed/absent.json"): "a" * 64,
    }
    expected = set().union(
        *(
            successor._selected_inputs(legacy, index)
            for index in range(len(successor._list(legacy["chunks"])))
        )
    )
    assert seal._scope_selected_input_union(legacy) == expected


def test_producer_historical_fingerprint_must_match(boundary_case: BoundaryCase) -> None:
    c = boundary_case
    graph, state, _ = c.job.lanes[0]
    checkpoint = state / "segment-0.json"
    document = common._read(checkpoint)
    document["fingerprint"] = "a" * 64
    checkpoint.write_text(json.dumps(document), encoding="utf-8")
    with pytest.raises(ReplayError, match="historical fingerprint changed"):
        seal.current_dependency_pins(c.job)


def test_current_pin_snapshot_identity_guard(boundary_case: BoundaryCase) -> None:
    c = boundary_case
    pins = {str(c.source): file_digest(c.source)}
    snapshots = seal._boundary_identities(pins)
    cache = InputHashCache((c.job.plan,))
    cache.prime()
    c.source.write_bytes(b"changed source identity")
    with pytest.raises(ReplayError, match="identity changed"):
        seal._boundary_terminal(pins, cache, snapshots)


@pytest.fixture
def many_current_pins(boundary_case: BoundaryCase) -> dict[str, str]:
    directory = boundary_case.job.plan.parent / "terminal-inputs"
    directory.mkdir()
    pins: dict[str, str] = {}
    for index in range(1025):
        path = directory / f"input-{index:04d}.json"
        path.write_bytes(b"x")
        pins[str(path)] = file_digest(path)
    return pins


def test_every_uncached_terminal_dependency_is_hashed(many_current_pins: dict[str, str]) -> None:
    pins = many_current_pins
    cache = InputHashCache(tuple(Path(name) for name in list(pins)[:1024]))
    identities = seal._boundary_identities(pins)
    with input_hash_context(cache):
        cache.prime()
        terminal = seal._boundary_terminal(pins, cache, identities)
    assert len(terminal) == 1025
    assert list(terminal.values())[-1] == (
        "2d711642b726b04401627ca9fbac32f5c8530fb1903cc4db02258717921a4881"
    )


def test_uncached_mutation_is_not_hidden_by_hash_subset(many_current_pins: dict[str, str]) -> None:
    pins = many_current_pins
    cache = InputHashCache(tuple(Path(name) for name in list(pins)[:1024]))
    identities = seal._boundary_identities(pins)
    with input_hash_context(cache):
        cache.prime()
        Path(list(pins)[-1]).write_bytes(b"changed uncached input")
        with pytest.raises(ReplayError, match="identity changed"):
            seal._boundary_terminal(pins, cache, identities)


@pytest.fixture
def source_history_case(boundary_case: BoundaryCase) -> SourceHistoryCase:
    directory = Path.cwd() / "src/replay"
    directory.mkdir(parents=True)
    source = directory / "historical_reader.py"
    source.write_bytes(b"def observed():\n    return 1\n")
    historical_sha = file_digest(source)
    output = boundary_case.job.plan.parent / "source-history-output.json"
    output.write_bytes(b"{}")
    graph = boundary_case.job.plan.parent / "source-history-graph.json"
    graph.write_text(
        json.dumps(
            {
                "version": 1,
                "stages": [
                    {
                        "name": "source-history",
                        "command": ["synthetic-readonly"],
                        "inputs": [str(source)],
                        "outputs": [str(output)],
                        "verify_command": ["synthetic-readonly"],
                        "timeout_seconds": 60,
                        "adopt_existing": True,
                    }
                ],
            }
        ),
        encoding="utf-8",
    )
    state = boundary_case.job.plan.parent / "source-history-state"
    stage = workflow.load_stages(graph)[0]

    def readonly(command: tuple[str, ...], log: Path, timeout: int) -> None:
        log.write_bytes(b"synthetic metadata verifier; source never executed")

    run_stage(stage, state_directory=state, execute=readonly)
    source.write_bytes(b"def observed():\n    return 2\n")
    return SourceHistoryCase(graph, state, source, historical_sha, output)


def test_source_history_is_not_replaced_with_current_bytes(
    source_history_case: SourceHistoryCase,
) -> None:
    c = source_history_case
    pins = {str(c.output): file_digest(c.output)}
    seal._scope_stage_pins(
        pins, graph=c.graph, state=c.state, recorded_sources={str(c.source): c.historical_sha256}
    )
    assert str(c.source) not in pins
    assert file_digest(c.source) != c.historical_sha256


def test_missing_historical_source_pin_cannot_use_current_bytes(
    source_history_case: SourceHistoryCase,
) -> None:
    c = source_history_case
    with pytest.raises(ReplayError, match="historical source binding"):
        seal._scope_stage_pins({str(c.output): file_digest(c.output)}, graph=c.graph, state=c.state)


def test_conflicting_generation_source_history_rejected(
    source_history_case: SourceHistoryCase,
) -> None:
    c = source_history_case
    with pytest.raises(ReplayError, match="Conflicting historical generation"):
        seal._scope_source_table(
            {
                "inputs": {str(c.source): c.historical_sha256},
                "readerCode": {str(c.source): file_digest(c.source)},
            }
        )


def test_material_reference_cannot_masquerade_as_source_history(
    source_history_case: SourceHistoryCase,
) -> None:
    with pytest.raises(ReplayError, match="material reference"):
        seal._scope_reference({}, common._pin(source_history_case.source))


def test_authenticated_empty_retained_log_is_checked_at_terminal(
    boundary_case: BoundaryCase,
) -> None:
    c = boundary_case
    log = c.job.plan.parent / "retained-stderr.log"
    log.write_bytes(b"")
    pins = {str(log): "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855"}
    identities = seal._boundary_identities(pins)
    cache = InputHashCache((c.source,))
    with input_hash_context(cache):
        cache.prime()
        terminal = seal._boundary_terminal(pins, cache, identities)
    assert list(terminal.values()) == [
        "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855"
    ]


def test_empty_retained_log_requires_exact_empty_hash(boundary_case: BoundaryCase) -> None:
    log = boundary_case.job.plan.parent / "retained-stderr.log"
    log.write_bytes(b"")
    with pytest.raises(ReplayError, match="bound regular"):
        seal._boundary_identities({str(log): "a" * 64})


def test_empty_retained_log_mutation_is_rejected(boundary_case: BoundaryCase) -> None:
    c = boundary_case
    log = c.job.plan.parent / "retained-stderr.log"
    log.write_bytes(b"")
    pins = {str(log): "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855"}
    identities = seal._boundary_identities(pins)
    cache = InputHashCache((c.source,))
    cache.prime()
    log.write_bytes(b"changed original retained log")
    with pytest.raises(ReplayError, match="identity changed"):
        seal._boundary_terminal(pins, cache, identities)


def test_empty_material_reference_remains_rejected(boundary_case: BoundaryCase) -> None:
    empty = boundary_case.job.plan.parent / "empty-model.bin"
    empty.write_bytes(b"")
    with pytest.raises(ReplayError, match="material references must be nonempty"):
        seal._scope_reference(
            {},
            {
                "path": str(empty),
                "sha256": "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855",
            },
        )


@pytest.mark.parametrize("change", ["missing", "changed"])
def test_declared_child_graph_must_match_completed_parent(
    boundary_case: BoundaryCase, change: str
) -> None:
    c = boundary_case
    metadata = common._read(c.job.plan)
    child = Path(successor._strings(metadata["workflows"])[0])
    record(metadata["inputs"]).pop(str(child))
    c.job.plan.write_text(json.dumps(metadata), encoding="utf-8")
    request = common._ref(common._read(c.job.operation)["request"])
    request_document = common._read(request)
    request_document["generation"] = common._pin(c.job.plan)
    request.write_text(json.dumps(request_document), encoding="utf-8")
    operation = common._read(c.job.operation)
    operation["request"] = common._pin(request)
    c.job.operation.write_text(json.dumps(operation), encoding="utf-8")
    if change == "missing":
        child.unlink()
        with pytest.raises(FileNotFoundError):
            seal.current_dependency_pins(c.job)
    else:
        child.write_bytes(child.read_bytes() + b"\n")
        with pytest.raises(ReplayError, match="historical fingerprint changed"):
            seal.current_dependency_pins(c.job)


def test_descriptor_probe_uses_keeper_between_typed_custody_checks(
    boundary_case: BoundaryCase, monkeypatch: pytest.MonkeyPatch
) -> None:
    c = boundary_case
    calls: list[str] = []
    probed: list[str] = []
    original = seal.load_publication_history

    def typed_history(
        *,
        operation: Path,
        custody_seal: Path,
        publication_workflow: Path,
        adoption_checkpoint: Path,
        child_workflow: Path,
        child_checkpoint: Path,
    ) -> seal.PublicationHistory:
        calls.append("custody")
        return original(
            operation=operation,
            custody_seal=custody_seal,
            publication_workflow=publication_workflow,
            adoption_checkpoint=adoption_checkpoint,
            child_workflow=child_workflow,
            child_checkpoint=child_checkpoint,
        )

    def actors(references: object) -> list[dict[str, object]]:
        calls.append("actors")
        return [{"terminalExit": 0}]

    def probe(paths: list[str]) -> set[str]:
        calls.append("fd")
        probed.extend(paths)
        return set()

    monkeypatch.setattr(seal, "load_publication_history", typed_history)
    monkeypatch.setattr(retirement, "_actors", actors)
    monkeypatch.setattr(leaves, "lsof_open_among", probe)
    assert seal._boundary_guard(c.history, common._read(c.request)) == [{"terminalExit": 0}]
    assert calls == ["custody", "actors", "fd", "custody"]
    assert probed == [str(c.history.keeper)]


def test_typed_alias_absence_is_required_before_descriptor_probe(
    boundary_case: BoundaryCase, monkeypatch: pytest.MonkeyPatch
) -> None:
    c = boundary_case
    calls: list[str] = []

    def probe(paths: list[str]) -> set[str]:
        calls.extend(paths)
        return set()

    monkeypatch.setattr(leaves, "lsof_open_among", probe)
    os.link(c.history.keeper, c.history.alias)
    with pytest.raises(ReplayError, match="Retired alias reappeared"):
        seal._boundary_guard(c.history, common._read(c.request))
    assert calls == []


def test_typed_alias_absence_is_required_after_descriptor_probe(
    boundary_case: BoundaryCase, monkeypatch: pytest.MonkeyPatch
) -> None:
    c = boundary_case

    def probe(paths: list[str]) -> set[str]:
        os.link(c.history.keeper, c.history.alias)
        return set()

    monkeypatch.setattr(leaves, "lsof_open_among", probe)
    with pytest.raises(ReplayError, match="Retired alias reappeared"):
        seal._boundary_guard(c.history, common._read(c.request))


@pytest.mark.parametrize("descriptor", ["open", "unknown", "deleted-name"])
def test_keeper_descriptor_open_or_unobservable_remains_rejected(
    boundary_case: BoundaryCase, monkeypatch: pytest.MonkeyPatch, descriptor: str
) -> None:
    c = boundary_case
    observations: dict[str, set[str] | None] = {
        "open": {str(c.history.keeper)},
        "unknown": None,
        "deleted-name": {str(c.history.alias) + " (deleted)"},
    }
    monkeypatch.setattr(leaves, "lsof_open_among", lambda paths: observations[descriptor])
    with pytest.raises(ReplayError, match="open or unobservable descriptor"):
        seal._boundary_guard(c.history, common._read(c.request))
