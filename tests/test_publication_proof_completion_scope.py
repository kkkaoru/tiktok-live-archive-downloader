"""Current selected material and writer leases remain bound across source history."""

import json
from pathlib import Path

import pytest
from test_recovery_source_finalization import Case
from test_recovery_source_finalization import case as case

from replay import cleanup_finalization_workflow as common
from replay import cleanup_lane_successor as successor
from replay import publication_proof_completion_seal as seal
from replay.core import ReplayError
from replay.recordings import record
from replay.stages import file_digest


@pytest.fixture
def scoped_job(case: Case, monkeypatch: pytest.MonkeyPatch) -> common.Job:
    c = case
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
    rows = [record(value) for value in successor._list(data["chunks"])]
    request = work / "legacy-request.json"
    request.write_text('{"oldWorkers":[]}', encoding="utf-8")
    handoff = work / "handoff.json"
    handoff.write_text('{"evidence":[]}', encoding="utf-8")
    legacy_rows: list[dict[str, object]] = []
    manifest_rows: list[dict[str, object]] = []
    for index, row in enumerate(rows):
        chunk = Path(str(row["chunk"]))
        chunk_data = common._read(chunk)
        chunk_data["settings"] = common._pin(c.settings)
        chunk.write_text(json.dumps(chunk_data), encoding="utf-8")
        media = chunk.parent / "candidate.mp4"
        receipt = chunk.parent / "candidate.json"
        proof = work / f"verified-{index}.json"
        media.write_bytes(b"synthetic candidate")
        receipt.write_bytes(b"synthetic repair receipt")
        proof.write_text(
            json.dumps(
                {
                    "repairEvidence": {
                        "receipt": common._pin(receipt),
                        "media": common._pin(media),
                    }
                }
            ),
            encoding="utf-8",
        )
        row.update(
            {
                "proof": str(proof),
                "adoptedProof": None,
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
        manifest_rows.append({"producer": common._pin(proof), "candidate": common._pin(media)})
    old_source = Path.cwd() / "src/replay/historical_reader.py"
    legacy = work / "successor.json"
    legacy.write_text(
        json.dumps(
            {
                "request": str(request),
                "originalPlan": str(original),
                "handoff": str(handoff),
                "settings": str(c.settings),
                "chunks": legacy_rows,
                "workflows": [str(path) for path in c.graphs],
                "states": [str(path) for path in c.states],
                "oldStates": [str(work / "original-state")],
                "inputs": {
                    str(request): file_digest(request),
                    str(work / "unused/never-produced.json"): "b" * 64,
                },
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
    record(data["inputs"])[str(old_source)] = "a" * 64
    c.plan.write_text(json.dumps(data), encoding="utf-8")
    c.request.write_text(json.dumps({"generation": common._pin(c.plan)}), encoding="utf-8")
    c.output.mkdir()
    c.operation.write_text(json.dumps({"request": common._pin(c.request)}), encoding="utf-8")
    final = c.output / "finalized"
    final.mkdir()
    (final / "manifest.json").write_text(json.dumps({"chunks": manifest_rows}), encoding="utf-8")
    return common.Job(
        c.operation,
        c.output,
        c.plan,
        tuple(zip(c.graphs, c.states, c.statuses, strict=True)),
        {},
        (record(settings["native"]), record(settings["decoder"])),
        60,
    )


def test_selected_scope_excludes_unconsumed_missing_history(scoped_job: common.Job) -> None:
    pins = seal.current_dependency_pins(scoped_job)
    seal.verify_current_dependency_pins(pins)
    assert not any("never-produced" in name for name in pins)
    assert not any("historical_reader" in name for name in pins)
    assert str(scoped_job.root / "finalized/manifest.json") in pins


@pytest.mark.parametrize("field", ["source", "model", "sourceClock"])
def test_bound_material_changes_rejected(scoped_job: common.Job, field: str) -> None:
    pins = seal.current_dependency_pins(scoped_job)
    settings = common._read(Path(str(common._read(scoped_job.plan)["settings"])))
    Path(str(record(settings[field])["path"])).write_bytes(b"changed")
    with pytest.raises(ReplayError, match="dependency bytes changed"):
        seal.verify_current_dependency_pins(pins)


def test_candidate_cannot_repin_current_checkpoint(scoped_job: common.Job) -> None:
    manifest = common._read(scoped_job.root / "finalized/manifest.json")
    first = record(successor._list(manifest["chunks"])[0])
    candidate = Path(str(record(first["candidate"])["path"]))
    candidate.write_bytes(b"changed candidate")
    pins = seal.current_dependency_pins(scoped_job)
    with pytest.raises(ReplayError, match="dependency bytes changed"):
        seal.verify_current_dependency_pins(pins)


def test_parent_nonzero_rejected(scoped_job: common.Job) -> None:
    scoped_job.lanes[0][2].write_bytes(b"1\n")
    with pytest.raises(ReplayError, match="terminal zero"):
        seal.current_dependency_pins(scoped_job)


def test_derived_leases_include_producers_and_originals(scoped_job: common.Job) -> None:
    explicit = scoped_job.root / "publication.lock"
    leases = seal.publication_leases(scoped_job, {"leases": [str(explicit)]})
    assert explicit in leases
    assert scoped_job.plan.parent / "recovery-lane-0.lock" in leases
    assert scoped_job.plan.parent / "original-state/old-repair.lock" in leases
    assert scoped_job.lanes[0][1] / "segment-0.lock" in leases
    assert scoped_job.lanes[0][1] / ".workflow.lock" in leases


def test_historical_source_cannot_be_a_current_material_pin(scoped_job: common.Job) -> None:
    name = str(Path.cwd() / "src/replay/historical_reader.py")
    with pytest.raises(ReplayError, match="Historical source"):
        seal.verify_current_dependency_pins({name: "a" * 64})


def test_conflicting_consumed_binding_rejected(scoped_job: common.Job) -> None:
    pins = {str(scoped_job.plan): "a" * 64}
    with pytest.raises(ReplayError, match="Conflicting"):
        seal._scope_pin(pins, scoped_job.plan, "b" * 64)


def test_malformed_material_reference_rejected(scoped_job: common.Job) -> None:
    with pytest.raises(ReplayError, match="exact current dependency"):
        seal._scope_reference({}, {"path": str(scoped_job.plan)})


def test_selected_producer_must_exist(scoped_job: common.Job) -> None:
    graph, state, _ = scoped_job.lanes[0]
    with pytest.raises(ReplayError, match="Stage is missing"):
        seal._scope_stage_pins({}, graph=graph, state=state, names={"never-produced"})


def test_consumed_producer_inputs_need_expected_binding(scoped_job: common.Job) -> None:
    graph, state, _ = scoped_job.lanes[0]
    with pytest.raises(ReplayError, match="authenticated binding"):
        seal._scope_stage_pins({}, graph=graph, state=state)


def test_running_producer_cannot_supply_outputs(scoped_job: common.Job) -> None:
    graph, state, _ = scoped_job.lanes[0]
    (state / "segment-0.json").write_text('{"status":"running","outputs":{}}', encoding="utf-8")
    with pytest.raises(ReplayError, match="genuine complete"):
        seal._scope_stage_pins({}, graph=graph, state=state)


def test_adopted_proof_remains_a_consumed_binding(scoped_job: common.Job) -> None:
    data = common._read(scoped_job.plan)
    rows = successor._list(data["chunks"])
    first = record(rows[0])
    first["mode"] = "adopted-existing"
    adopted = scoped_job.plan.parent / "adopted-proof.json"
    adopted.write_bytes(b"real synthetic adopted proof")
    first["adoptedProof"] = common._pin(adopted)
    scoped_job.plan.write_text(json.dumps(data), encoding="utf-8")
    request = common._ref(common._read(scoped_job.operation)["request"])
    request.write_text(json.dumps({"generation": common._pin(scoped_job.plan)}), encoding="utf-8")
    scoped_job.operation.write_text(json.dumps({"request": common._pin(request)}), encoding="utf-8")
    pins = seal.current_dependency_pins(scoped_job)
    adopted.write_bytes(b"changed adopted proof")
    with pytest.raises(ReplayError, match="dependency bytes changed"):
        seal.verify_current_dependency_pins(pins)
