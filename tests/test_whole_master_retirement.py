"""Synthetic retirement lifecycle; no recordings, apps or real-media cleanup."""

import hashlib
import json
import os
import subprocess
import time
from dataclasses import dataclass
from pathlib import Path

import pytest

from replay import native_join
from replay import whole_master_retirement as retirement
from replay.code_dependency_inventory import Source, capture
from replay.core import ReplayError
from replay.input_hash_cache import InputHashCache
from replay.recordings import record
from replay.stages import execution_lock


@dataclass
class Case:
    root: Path
    request: Path
    config: dict[str, object]
    master: Path
    part: Path
    voice: Path
    plan: Path
    historical: Path
    disposition: Path
    state: Path
    owner_state: Path
    output: Path


@pytest.fixture
def case(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Case:
    root = tmp_path / "checkout"
    work = root / "edits/work/retirement"
    work.mkdir(parents=True)
    code = root / "src/replay"
    code.mkdir(parents=True)
    tools = work / "tools"
    tools.mkdir()
    monkeypatch.setattr(retirement, "ROOT", root)
    monkeypatch.setattr(retirement, "TOOL_ROOT", tools)
    names = (
        "whole_master_retirement.py",
        "native_export_receipt.py",
        "input_hash_cache.py",
        "core.py",
        "recordings.py",
        "stages.py",
        "delivery_render.py",
        "native_join.py",
        "media.py",
        "subtitle_clock.py",
    )
    created = [(code / name).write_bytes(b"synthetic code\n") for name in names]
    assert len(created) == 10
    (tools / "ffmpeg").write_bytes(b"synthetic encoder")
    (tools / "ffprobe").write_bytes(b"synthetic decoder")
    master, part, second, voice = (
        work / name for name in ("whole.mov", "part-a.mov", "part-b.mov", "voice.wav")
    )
    master.write_bytes(b"synthetic original whole")
    part.write_bytes(b"synthetic native part A")
    second.write_bytes(b"synthetic native part B")
    voice.write_bytes(b"synthetic edited voice")
    ref_master = {"path": str(master), "sha256": hashlib.sha256(master.read_bytes()).hexdigest()}
    domain: dict[str, object] = {
        "frames": 5,
        "rate": 30,
        "width": 64,
        "height": 64,
        "sampleRate": 44100,
        "channels": 2,
        "sampleFrames": 7350,
        "pcm32SHA256": "f" * 64,
    }
    historical = work / "historical.json"
    facts = {
        **domain,
        "masterPath": str(master),
        "masterSHA256": ref_master["sha256"],
        "masterBytes": 24,
        "fullAVVerified": True,
        "allPTSVerified": True,
        "wholeVoicePCMEqual": True,
    }
    # Keep the expected master size explicit and verify the actual fixture value.
    assert master.stat().st_size == 24
    historical.write_text(json.dumps(facts), encoding="utf-8")
    historical_ref = {
        "path": str(historical),
        "sha256": hashlib.sha256(historical.read_bytes()).hexdigest(),
    }
    retained = {
        name: work / f"{name}.json"
        for name in (
            "voiceProof",
            "assembly",
            "assemblyLineage",
            "editableXML",
            "wholeWorkflow",
            "sourceBefore",
            "sourceAfter",
        )
    }
    created_metadata = [
        path.write_text('{"retained":true}', encoding="utf-8") for path in retained.values()
    ]
    assert len(created_metadata) == 7
    reference = work / "reference-inspection.json"
    reference.write_text('{"completed":true,"materializedMasterRequired":false}', encoding="utf-8")
    reference_ref = {
        "path": str(reference),
        "sha256": hashlib.sha256(reference.read_bytes()).hexdigest(),
    }
    disposition = work / "owner.json"
    disposition.write_text(
        json.dumps(
            {
                "version": 1,
                "master": ref_master,
                "owner": "synthetic completed owner",
                "ownerReleased": True,
                "observedAtNs": time.time_ns(),
                "references": {
                    name: {
                        "scope": "explicit synthetic namespace",
                        "resolved": True,
                        "requiresMaterializedMaster": False,
                        "evidence": [reference_ref],
                    }
                    for name in ("current", "queued", "resume", "fcp", "viewer", "diagnostic")
                },
            }
        ),
        encoding="utf-8",
    )
    plan = work / "join.json"
    plan.write_text(
        json.dumps(
            [
                {"sourcePath": str(part), "durationFrames": 2},
                {"sourcePath": str(second), "durationFrames": 3},
            ]
        ),
        encoding="utf-8",
    )
    state, owner_state = work / "state", work / "owner-state"
    state.mkdir()
    owner_state.mkdir()
    config: dict[str, object] = {
        "version": 1,
        "master": ref_master,
        "masterBytes": 24,
        "joinPlan": {"path": str(plan), "sha256": hashlib.sha256(plan.read_bytes()).hexdigest()},
        "parts": [
            {
                "media": {
                    "path": str(path),
                    "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                },
                "frames": frames,
            }
            for path, frames in ((part, 2), (second, 3))
        ],
        "voice": {"path": str(voice), "sha256": hashlib.sha256(voice.read_bytes()).hexdigest()},
        "retained": {
            name: {"path": str(path), "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}
            for name, path in retained.items()
        },
        "code": [
            {
                "path": str(code / name),
                "sha256": hashlib.sha256((code / name).read_bytes()).hexdigest(),
            }
            for name in names
        ],
        "tools": [
            {"path": str(path), "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}
            for path in (tools / "ffmpeg", tools / "ffprobe")
        ],
        "domain": domain,
        "historical": {role: {"artifact": historical_ref, "pointer": f"/{role}"} for role in facts},
        "disposition": {
            "path": str(disposition),
            "sha256": hashlib.sha256(disposition.read_bytes()).hexdigest(),
        },
        "state": str(state),
        "leases": [str(owner_state)],
    }
    request = work / "request.json"
    request.write_text(json.dumps(config), encoding="utf-8")
    return Case(
        root,
        request,
        config,
        master,
        part,
        voice,
        plan,
        historical,
        disposition,
        state,
        owner_state,
        work / "intent.json",
    )


def test_prepare_verifies_fresh_inputs_without_deleting_and_readonly_reentry(case: Case) -> None:
    assert retirement.run(case.request, case.output) == case.output
    result = record(json.loads(case.output.read_text(encoding="utf-8")))
    assert result["kind"] == "whole-master-retirement-intent"
    assert result["masterMaterialized"] is True
    assert result["deletionAuthorized"] is False
    assert result["freshWholeVerified"] is False
    assert result["rebuildPolicy"] == "new-nonoverwriting-revision"
    assert result["qualityApproved"] is False
    assert case.master.read_bytes() == b"synthetic original whole"
    assert case.output.stat().st_mode & 0o777 == 0o600
    before = case.output.read_bytes()
    assert retirement.run(case.request, case.output, verify=True) == case.output
    assert case.output.read_bytes() == before
    with pytest.raises(ReplayError, match="occupied"):
        retirement.run(case.request, case.output)


def test_after_delete_preserves_history_and_reports_no_materialized_whole(
    case: Case, monkeypatch: pytest.MonkeyPatch
) -> None:
    retirement.run(case.request, case.output)
    intent = record(json.loads(case.output.read_text(encoding="utf-8")))
    historical_before, intent_before = case.historical.read_bytes(), case.output.read_bytes()
    deletion = case.output.parent / "deletion.json"
    deletion.write_text(
        json.dumps(
            {
                "version": 1,
                "intent": {
                    "path": str(case.output),
                    "sha256": hashlib.sha256(intent_before).hexdigest(),
                },
                "master": case.config["master"],
                "masterIdentity": intent["masterIdentity"],
                "removed": True,
                "operation": "exact-path-unlink",
                "observedAtNs": time.time_ns(),
                "disposition": case.config["disposition"],
            }
        ),
        encoding="utf-8",
    )
    case.master.unlink()  # Only this synthetic fixture; production module has no unlink.
    outcome = case.output.parent / "outcome.json"
    assert retirement.run(case.request, outcome, intent=case.output, deletion=deletion) == outcome
    result = record(json.loads(outcome.read_text(encoding="utf-8")))
    assert result["kind"] == "retired-reconstructable-whole-master"
    assert result["masterMaterialized"] is False
    assert result["historicalWholeVerified"] is True
    assert result["freshWholeVerified"] is False
    assert result["byteIdenticalRebuildVerified"] is False
    assert (
        retirement.run(case.request, outcome, intent=case.output, deletion=deletion, verify=True)
        == outcome
    )
    assert case.historical.read_bytes() == historical_before
    assert case.output.read_bytes() == intent_before
    assert case.part.is_file()
    assert case.voice.is_file()
    later = time.time_ns() + 2 * 3600 * 1_000_000_000
    with monkeypatch.context() as clock:
        clock.setattr(time, "time_ns", lambda: later)
        # A future consumer validates immutable retirement history and fresh inputs,
        # not a new deletion admission using stale observations.
        assert (
            retirement.run(
                case.request, outcome, intent=case.output, deletion=deletion, verify=True
            )
            == outcome
        )
        with pytest.raises(ReplayError, match="stale"):
            retirement.run(
                case.request,
                case.output.parent / "repeat-outcome.json",
                intent=case.output,
                deletion=deletion,
            )
    with pytest.raises(FileNotFoundError):
        retirement.run(case.request, case.output.parent / "new-intent.json")


@pytest.mark.parametrize("role", ["current", "queued", "resume", "fcp", "viewer", "diagnostic"])
def test_each_materialized_reference_is_required_to_be_released(case: Case, role: str) -> None:
    owner = record(json.loads(case.disposition.read_text(encoding="utf-8")))
    references = record(owner["references"])
    record(references[role])["requiresMaterializedMaster"] = True
    # record returns the same checked dict; write the changed owned fixture.
    case.disposition.write_text(json.dumps(owner), encoding="utf-8")
    case.config["disposition"] = {
        "path": str(case.disposition),
        "sha256": hashlib.sha256(case.disposition.read_bytes()).hexdigest(),
    }
    case.request.write_text(json.dumps(case.config), encoding="utf-8")
    with pytest.raises(ReplayError, match="active, unresolved or required"):
        retirement.run(case.request, case.output)
    assert case.master.exists()
    assert not case.output.exists()


def test_active_declared_owner_and_duplicate_retirement_owner_are_not_interrupted(
    case: Case,
) -> None:
    with execution_lock(case.owner_state, ".workflow") as owner:
        assert owner is True
        with pytest.raises(ReplayError, match="workflow owner is still active"):
            retirement.run(case.request, case.output)
    name = hashlib.sha256(str(case.master).encode()).hexdigest()
    with execution_lock(case.state, name) as owned:
        assert owned is True
        with pytest.raises(ReplayError, match="namespace is already owned"):
            retirement.run(case.request, case.output)
    assert case.master.exists()
    assert not case.output.exists()


@pytest.mark.parametrize("name", ["part-a.mov", "voice.wav", "historical.json", "tools/ffmpeg"])
def test_changed_or_missing_retained_bytes_refuse_admission(case: Case, name: str) -> None:
    (case.output.parent / name).write_bytes(b"changed")
    with pytest.raises(ReplayError, match="changed"):
        retirement.run(case.request, case.output)
    assert case.master.exists()
    assert not case.output.exists()


def test_missing_part_is_not_implicitly_rebuilt(case: Case) -> None:
    case.part.unlink()
    with pytest.raises(FileNotFoundError):
        retirement.run(case.request, case.output)
    assert not case.output.exists()
    assert case.master.exists()


def test_ordered_plan_cannot_be_substituted(case: Case) -> None:
    rows: object = json.loads(case.plan.read_text(encoding="utf-8"))
    assert isinstance(rows, list)
    case.plan.write_text(json.dumps(list(reversed(rows))), encoding="utf-8")
    case.config["joinPlan"] = {
        "path": str(case.plan),
        "sha256": hashlib.sha256(case.plan.read_bytes()).hexdigest(),
    }
    case.request.write_text(json.dumps(case.config), encoding="utf-8")
    with pytest.raises(ReplayError, match="Ordered join plan differs"):
        retirement.run(case.request, case.output)


def test_historical_domain_cannot_be_promoted_by_a_new_pin(case: Case) -> None:
    historical = record(json.loads(case.historical.read_text(encoding="utf-8")))
    historical["fullAVVerified"] = False
    case.historical.write_text(json.dumps(historical), encoding="utf-8")
    new_ref = {
        "path": str(case.historical),
        "sha256": hashlib.sha256(case.historical.read_bytes()).hexdigest(),
    }
    claims = record(case.config["historical"])
    case.config["historical"] = {
        name: {"artifact": new_ref, "pointer": record(item)["pointer"]}
        for name, item in claims.items()
    }
    case.request.write_text(json.dumps(case.config), encoding="utf-8")
    with pytest.raises(ReplayError, match="Historical whole domain differs: fullAVVerified"):
        retirement.run(case.request, case.output)


@pytest.mark.parametrize(
    "field,value",
    [
        ("version", True),
        ("masterBytes", 0),
        ("parts", []),
        ("leases", []),
        ("domain", {}),
        ("retained", {}),
        ("historical", {}),
    ],
)
def test_invalid_request_is_rejected_before_effect(case: Case, field: str, value: object) -> None:
    case.config[field] = value
    case.request.write_text(json.dumps(case.config), encoding="utf-8")
    with pytest.raises(ReplayError):
        retirement.load_request(case.request)
    assert not case.output.exists()


def test_recipe_and_clock_shape_are_required(case: Case) -> None:
    record(case.config["domain"])["sampleFrames"] = 7351
    case.request.write_text(json.dumps(case.config), encoding="utf-8")
    with pytest.raises(ReplayError, match="clocks differ"):
        retirement.load_request(case.request)
    record(case.config["domain"])["sampleFrames"] = 7350
    case.config["leases"] = [str(case.owner_state), str(case.owner_state)]
    case.request.write_text(json.dumps(case.config), encoding="utf-8")
    with pytest.raises(ReplayError, match="ownership namespace differs"):
        retirement.load_request(case.request)


def test_retained_dependency_on_the_target_master_is_not_reconstructable(case: Case) -> None:
    record(case.config["retained"])["editableXML"] = case.config["master"]
    case.request.write_text(json.dumps(case.config), encoding="utf-8")
    with pytest.raises(ReplayError, match="still requires the master"):
        retirement.run(case.request, case.output)


def test_scope_symlink_hardlink_and_output_parent_guards(case: Case) -> None:
    with pytest.raises(ReplayError, match="scope"):
        retirement.run(case.request, case.root / "outside.json")
    linked = case.output.parent / "linked.json"
    linked.symlink_to(case.request)
    with pytest.raises(ReplayError, match="link"):
        retirement.load_request(linked)
    alias = case.output.parent / "master-alias.mov"
    os.link(case.master, alias)
    with pytest.raises(ReplayError, match="size differs"):
        retirement.run(case.request, case.output)
    with pytest.raises(ReplayError, match="incomplete"):
        retirement.run(case.request, case.output.parent / "missing/result.json")
    assert case.master.exists()


@pytest.mark.parametrize("value", ["not-a-digest", "0" * 63, "F" * 64])
def test_sha_pins_must_be_lowercase_full_sha256(case: Case, value: str) -> None:
    record(case.config["voice"])["sha256"] = value
    case.request.write_text(json.dumps(case.config), encoding="utf-8")
    with pytest.raises(ReplayError, match="SHA256"):
        retirement.load_request(case.request)


def test_duplicate_json_and_nonfinite_values_are_not_accepted(case: Case) -> None:
    case.request.write_text('{"version":1,"version":1}', encoding="utf-8")
    with pytest.raises(ReplayError, match="duplicate"):
        retirement.load_request(case.request)
    case.request.write_text('{"version":NaN}', encoding="utf-8")
    with pytest.raises(ReplayError, match="Nonfinite"):
        retirement.load_request(case.request)


def test_metadata_budget_nonregular_and_stale_owner_guards(
    case: Case, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(retirement, "MAX_METADATA_BYTES", 1)
    with pytest.raises(ReplayError, match="byte budget"):
        retirement.load_request(case.request)
    monkeypatch.setattr(retirement, "MAX_METADATA_BYTES", 8 * 1024**2)
    owner = record(json.loads(case.disposition.read_text(encoding="utf-8")))
    owner["observedAtNs"] = 1
    case.disposition.write_text(json.dumps(owner), encoding="utf-8")
    case.config["disposition"] = {
        "path": str(case.disposition),
        "sha256": hashlib.sha256(case.disposition.read_bytes()).hexdigest(),
    }
    case.request.write_text(json.dumps(case.config), encoding="utf-8")
    with pytest.raises(ReplayError, match="stale"):
        retirement.run(case.request, case.output)


def test_missing_reference_scope_or_owner_release_refuses_admission(case: Case) -> None:
    owner = record(json.loads(case.disposition.read_text(encoding="utf-8")))
    record(owner["references"]).pop("diagnostic")
    case.disposition.write_text(json.dumps(owner), encoding="utf-8")
    case.config["disposition"] = {
        "path": str(case.disposition),
        "sha256": hashlib.sha256(case.disposition.read_bytes()).hexdigest(),
    }
    case.request.write_text(json.dumps(case.config), encoding="utf-8")
    with pytest.raises(ReplayError, match="diagnostic scopes are required"):
        retirement.run(case.request, case.output)
    owner["ownerReleased"] = False
    case.disposition.write_text(json.dumps(owner), encoding="utf-8")
    case.config["disposition"] = {
        "path": str(case.disposition),
        "sha256": hashlib.sha256(case.disposition.read_bytes()).hexdigest(),
    }
    case.request.write_text(json.dumps(case.config), encoding="utf-8")
    with pytest.raises(ReplayError, match="Completed owner disposition"):
        retirement.run(case.request, case.output)


def test_master_identity_race_is_rejected_before_publication(
    case: Case, monkeypatch: pytest.MonkeyPatch
) -> None:
    original = InputHashCache.verify

    def changed(cache: InputHashCache) -> dict[str, str]:
        verified = original(cache)
        case.master.write_bytes(b"same logical media, changed container")
        return verified

    monkeypatch.setattr(InputHashCache, "verify", changed)
    with pytest.raises(ReplayError, match="identity changed"):
        retirement.run(case.request, case.output)
    assert not case.output.exists()


def test_complete_requires_actual_absence_and_bound_external_result(case: Case) -> None:
    retirement.run(case.request, case.output)
    prepared = record(json.loads(case.output.read_text(encoding="utf-8")))
    deletion = case.output.parent / "deletion.json"
    removal: dict[str, object] = {
        "version": 1,
        "intent": {
            "path": str(case.output),
            "sha256": hashlib.sha256(case.output.read_bytes()).hexdigest(),
        },
        "master": case.config["master"],
        "masterIdentity": prepared["masterIdentity"],
        "removed": True,
        "operation": "exact-path-unlink",
        "observedAtNs": time.time_ns(),
        "disposition": case.config["disposition"],
    }
    deletion.write_text(json.dumps(removal), encoding="utf-8")
    with pytest.raises(ReplayError, match="actually be absent"):
        retirement.run(
            case.request, case.output.parent / "outcome.json", intent=case.output, deletion=deletion
        )
    case.master.unlink()
    removal["removed"] = False
    deletion.write_text(json.dumps(removal), encoding="utf-8")
    with pytest.raises(ReplayError, match="External deletion result"):
        retirement.run(
            case.request, case.output.parent / "outcome.json", intent=case.output, deletion=deletion
        )
    removal["removed"] = True
    removal["observedAtNs"] = 1
    deletion.write_text(json.dumps(removal), encoding="utf-8")
    with pytest.raises(ReplayError, match="chronology"):
        retirement.run(
            case.request, case.output.parent / "outcome.json", intent=case.output, deletion=deletion
        )


def test_unknown_historical_pointer_and_conflicting_inventory_are_rejected(case: Case) -> None:
    record(record(case.config["historical"])["frames"])["pointer"] = "/missing"
    case.request.write_text(json.dumps(case.config), encoding="utf-8")
    with pytest.raises(ReplayError, match="recorded value"):
        retirement.run(case.request, case.output)
    record(record(case.config["historical"])["frames"])["pointer"] = "/frames"
    record(case.config["retained"])["editableXML"] = {"path": str(case.voice), "sha256": "0" * 64}
    case.request.write_text(json.dumps(case.config), encoding="utf-8")
    with pytest.raises(ReplayError, match="Conflicting"):
        retirement.run(case.request, case.output)


def test_cli_and_malformed_intent_are_explicit_failures(
    case: Case, capsys: pytest.CaptureFixture[str]
) -> None:
    assert retirement.main(["prepare", str(case.request), str(case.output)]) == 0
    assert '"mediaDeleted": false' in capsys.readouterr().out
    with pytest.raises(SystemExit) as missing:
        retirement.main(["complete", str(case.request), str(case.output.parent / "result.json")])
    assert missing.value.code == 2
    with pytest.raises(SystemExit) as wrong_phase:
        retirement.main(
            ["prepare", str(case.request), str(case.output), "--intent", str(case.output)]
        )
    assert wrong_phase.value.code == 2
    result = record(json.loads(case.output.read_text(encoding="utf-8")))
    result["preparedAtNs"] = "invented"
    case.output.write_text(json.dumps(result), encoding="utf-8")
    with pytest.raises(SystemExit) as invalid:
        retirement.main(["verify", str(case.request), str(case.output)])
    assert invalid.value.code == 1


def test_readonly_evidence_rejects_changed_approval(case: Case) -> None:
    retirement.run(case.request, case.output)
    result = record(json.loads(case.output.read_text(encoding="utf-8")))
    result["qualityApproved"] = True
    case.output.write_text(json.dumps(result), encoding="utf-8")
    with pytest.raises(ReplayError, match="Immutable retirement evidence differs"):
        retirement.run(case.request, case.output, verify=True)


@pytest.fixture
def supplemental_case(case: Case, monkeypatch: pytest.MonkeyPatch) -> tuple[Path, Path, Path]:
    retirement.run(case.request, case.output)
    prepared = record(json.loads(case.output.read_text(encoding="utf-8")))
    deletion = case.output.parent / "external-deletion.json"
    deletion.write_text(
        json.dumps(
            {
                "version": 1,
                "intent": {
                    "path": str(case.output),
                    "sha256": hashlib.sha256(case.output.read_bytes()).hexdigest(),
                },
                "master": case.config["master"],
                "masterIdentity": prepared["masterIdentity"],
                "removed": True,
                "operation": "exact-path-unlink",
                "observedAtNs": time.time_ns(),
                "disposition": case.config["disposition"],
            }
        ),
        encoding="utf-8",
    )
    case.master.unlink()
    now = time.time_ns() + 2 * retirement.MAX_DISPOSITION_AGE_NS
    monkeypatch.setattr(time, "time_ns", lambda: now)
    owner = record(json.loads(case.disposition.read_text(encoding="utf-8")))
    owner["observedAtNs"] = now
    current = case.output.parent / "current-owner.json"
    current.write_text(json.dumps(owner), encoding="utf-8")
    return deletion, current, case.output.parent / "supplemental-outcome.json"


def test_supplemental_completion_preserves_originals_and_readonly_contract(
    case: Case, supplemental_case: tuple[Path, Path, Path]
) -> None:
    deletion, current, outcome = supplemental_case
    before = (case.request.read_bytes(), case.output.read_bytes(), deletion.read_bytes())
    with pytest.raises(ReplayError, match="stale"):
        retirement.run(case.request, outcome, intent=case.output, deletion=deletion)
    assert (
        retirement.main(
            [
                "complete",
                str(case.request),
                str(outcome),
                "--intent",
                str(case.output),
                "--deletion",
                str(deletion),
                "--current-owner",
                str(current),
            ]
        )
        == 0
    )
    result = record(json.loads(outcome.read_text(encoding="utf-8")))
    assert result["version"] == 2
    assert result["qualityApproved"] is False
    assert result["freshWholeVerified"] is False
    assert result["masterMaterialized"] is False
    assert result["currentOwner"] == {
        "path": str(current),
        "sha256": hashlib.sha256(current.read_bytes()).hexdigest(),
    }
    assert (case.request.read_bytes(), case.output.read_bytes(), deletion.read_bytes()) == before
    saved = outcome.read_bytes()
    retirement.run(
        case.request,
        outcome,
        intent=case.output,
        deletion=deletion,
        current_owner=current,
        verify=True,
    )
    assert outcome.read_bytes() == saved
    with pytest.raises(ReplayError, match="Immutable retirement evidence differs"):
        retirement.run(case.request, outcome, intent=case.output, deletion=deletion, verify=True)


@pytest.mark.parametrize(
    "fault", ["stale", "future", "integer", "master", "active", "scope", "unknown"]
)
def test_supplemental_owner_rejects_invalid_admission(
    case: Case, supplemental_case: tuple[Path, Path, Path], fault: str
) -> None:
    deletion, current, outcome = supplemental_case
    owner = record(json.loads(current.read_text(encoding="utf-8")))
    if fault == "stale":
        owner["observedAtNs"] = 1
    elif fault == "future":
        owner["observedAtNs"] = time.time_ns() + 1
    elif fault == "integer":
        owner["observedAtNs"] = True
    elif fault == "master":
        record(owner["master"])["sha256"] = "0" * 64
    elif fault == "active":
        record(record(owner["references"])["current"])["requiresMaterializedMaster"] = True
    elif fault == "scope":
        del record(owner["references"])["resume"]
    else:
        owner["unexpected"] = True
    current.write_text(json.dumps(owner), encoding="utf-8")
    with pytest.raises(ReplayError):
        retirement.run(
            case.request, outcome, intent=case.output, deletion=deletion, current_owner=current
        )
    assert not outcome.exists()


@pytest.mark.parametrize("fault", ["owner", "evidence", "resurrection", "expiry"])
def test_supplemental_terminal_mutation_refuses_publication(
    case: Case,
    supplemental_case: tuple[Path, Path, Path],
    monkeypatch: pytest.MonkeyPatch,
    fault: str,
) -> None:
    deletion, current, outcome = supplemental_case
    original = retirement._domains
    instant = time.time_ns()

    def changed(request: retirement.Request) -> None:
        original(request)
        if fault == "owner":
            current.write_text("{}", encoding="utf-8")
        elif fault == "evidence":
            (current.parent / "reference-inspection.json").write_text("{}", encoding="utf-8")
        elif fault == "resurrection":
            case.master.write_bytes(b"resurrected")
        else:
            monkeypatch.setattr(
                time, "time_ns", lambda: instant + retirement.MAX_DISPOSITION_AGE_NS + 1
            )

    monkeypatch.setattr(retirement, "_domains", changed)
    with pytest.raises(ReplayError):
        retirement.run(
            case.request, outcome, intent=case.output, deletion=deletion, current_owner=current
        )
    assert not outcome.exists()


def test_supplemental_owner_cannot_change_recipe_or_prepare(case: Case) -> None:
    with pytest.raises(ReplayError, match="requires original intent"):
        retirement.run(case.request, case.output, current_owner=case.disposition)


@pytest.mark.parametrize("fault", ["request", "deletion", "frames"])
def test_supplemental_owner_does_not_override_original_domain(
    case: Case, supplemental_case: tuple[Path, Path, Path], fault: str
) -> None:
    deletion, current, outcome = supplemental_case
    if fault == "request":
        case.request.write_text(json.dumps(case.config, indent=2), encoding="utf-8")
    elif fault == "deletion":
        row = record(json.loads(deletion.read_text(encoding="utf-8")))
        row["masterIdentity"] = [1, 2, 24, 4, 5, 1]
        deletion.write_text(json.dumps(row), encoding="utf-8")
    else:
        case.historical.write_text('{"frames":999}', encoding="utf-8")
    with pytest.raises(ReplayError):
        retirement.run(
            case.request, outcome, intent=case.output, deletion=deletion, current_owner=current
        )
    assert not outcome.exists()


@pytest.fixture
def migration_case(case: Case, supplemental_case: tuple[Path, Path, Path]) -> Path:
    deletion, current, outcome = supplemental_case
    source = case.root / "src/replay/whole_master_retirement.py"
    archive = current.parent / "original-source.txt"
    archive.write_bytes(source.read_bytes())
    source.write_text("# new supplemental owner runtime\n", encoding="utf-8")
    (source.parent / "__init__.py").write_text("# synthetic package\n", encoding="utf-8")
    transition = outcome.parent / "runtime-transition.json"
    transition.write_text(
        json.dumps(
            {
                "version": 1,
                "scope": "supplemental-current-owner-v1",
                "request": {
                    "path": str(case.request),
                    "sha256": hashlib.sha256(case.request.read_bytes()).hexdigest(),
                },
                "intent": {
                    "path": str(case.output),
                    "sha256": hashlib.sha256(case.output.read_bytes()).hexdigest(),
                },
                "deletion": {
                    "path": str(deletion),
                    "sha256": hashlib.sha256(deletion.read_bytes()).hexdigest(),
                },
                "changes": [
                    {
                        "original": {
                            "path": str(source),
                            "sha256": hashlib.sha256(archive.read_bytes()).hexdigest(),
                        },
                        "archive": {
                            "path": str(archive),
                            "sha256": hashlib.sha256(archive.read_bytes()).hexdigest(),
                        },
                        "current": {
                            "path": str(source),
                            "sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
                        },
                    }
                ],
                "runtime": [
                    {"path": str(item.path), "sha256": item.sha256}
                    for item in capture(("whole_master_retirement",), package=source.parent)
                ],
            }
        ),
        encoding="utf-8",
    )
    return transition


def test_explicit_runtime_migration_reads_archive_but_never_executes_it(
    case: Case, supplemental_case: tuple[Path, Path, Path], migration_case: Path
) -> None:
    deletion, current, outcome = supplemental_case
    with pytest.raises(ReplayError, match="input bytes changed"):
        retirement.run(
            case.request, outcome, intent=case.output, deletion=deletion, current_owner=current
        )
    retirement.run(
        case.request,
        outcome,
        intent=case.output,
        deletion=deletion,
        current_owner=current,
        runtime_transition=migration_case,
    )
    result = record(json.loads(outcome.read_text(encoding="utf-8")))
    assert result["version"] == 2
    assert result["runtimeTransition"] == {
        "path": str(migration_case),
        "sha256": hashlib.sha256(migration_case.read_bytes()).hexdigest(),
    }
    retirement.run(
        case.request,
        outcome,
        intent=case.output,
        deletion=deletion,
        current_owner=current,
        runtime_transition=migration_case,
        verify=True,
    )
    assert (current.parent / "original-source.txt").read_bytes() == b"synthetic code\n"


@pytest.mark.parametrize(
    "fault",
    [
        "request",
        "scope",
        "missing",
        "fields",
        "archive-sha",
        "archive-bytes",
        "current",
        "unlisted",
        "runtime-extra",
        "runtime-duplicate",
        "wrong-module",
    ],
)
def test_runtime_migration_rejects_inexact_or_unlisted_changes(
    case: Case, supplemental_case: tuple[Path, Path, Path], migration_case: Path, fault: str
) -> None:
    deletion, current, outcome = supplemental_case
    raw = record(json.loads(migration_case.read_text(encoding="utf-8")))
    changes = raw["changes"]
    runtime = raw["runtime"]
    assert isinstance(changes, list) and isinstance(runtime, list)
    change = record(changes[0])
    if fault == "request":
        record(raw["request"])["sha256"] = "0" * 64
    elif fault == "scope":
        raw["scope"] = "allow-all-code-changes"
    elif fault == "missing":
        raw["changes"] = []
    elif fault == "fields":
        change["unexpected"] = True
    elif fault == "archive-sha":
        record(change["archive"])["sha256"] = "0" * 64
    elif fault == "archive-bytes":
        (current.parent / "original-source.txt").write_bytes(b"changed")
    elif fault == "current":
        (case.root / "src/replay/whole_master_retirement.py").write_text(
            "# changed\n", encoding="utf-8"
        )
    elif fault == "unlisted":
        (case.root / "src/replay/core.py").write_bytes(b"changed")
    elif fault == "runtime-extra":
        runtime.append(record(change["archive"]))
    elif fault == "runtime-duplicate":
        runtime.append(runtime[0])
    else:
        record(change["original"])["path"] = str(case.root / "src/replay/core.py")
    migration_case.write_text(json.dumps(raw), encoding="utf-8")
    with pytest.raises(ReplayError):
        retirement.run(
            case.request,
            outcome,
            intent=case.output,
            deletion=deletion,
            current_owner=current,
            runtime_transition=migration_case,
        )
    assert not outcome.exists()


def test_runtime_migration_requires_current_owner(
    case: Case, supplemental_case: tuple[Path, Path, Path], migration_case: Path
) -> None:
    deletion, current, outcome = supplemental_case
    assert current.exists()
    with pytest.raises(ReplayError, match="requires explicit current owner"):
        retirement.run(
            case.request,
            outcome,
            intent=case.output,
            deletion=deletion,
            runtime_transition=migration_case,
        )


def test_runtime_migration_rechecks_dependency_membership_at_terminal(
    case: Case,
    supplemental_case: tuple[Path, Path, Path],
    migration_case: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    deletion, current, outcome = supplemental_case
    original = capture
    calls = 0

    def changed(modules: tuple[str, ...], *, package: Path) -> tuple[Source, ...]:
        nonlocal calls
        calls += 1
        sources = original(modules, package=package)
        return sources if calls == 1 else sources[1:]

    monkeypatch.setattr(retirement, "capture", changed)
    with pytest.raises(ReplayError, match="runtime closure differs"):
        retirement.run(
            case.request,
            outcome,
            intent=case.output,
            deletion=deletion,
            current_owner=current,
            runtime_transition=migration_case,
        )
    assert not outcome.exists()


@pytest.mark.parametrize("target", ["archive", "current"])
def test_runtime_migration_mutation_after_initial_read_fails(
    case: Case,
    supplemental_case: tuple[Path, Path, Path],
    migration_case: Path,
    monkeypatch: pytest.MonkeyPatch,
    target: str,
) -> None:
    deletion, current, outcome = supplemental_case
    original = retirement._domains

    def mutate(request: retirement.Request) -> None:
        original(request)
        changed = (
            current.parent / "original-source.txt"
            if target == "archive"
            else case.root / "src/replay/whole_master_retirement.py"
        )
        changed.write_bytes(b"# changed after initial admission\n")

    monkeypatch.setattr(retirement, "_domains", mutate)
    with pytest.raises(ReplayError):
        retirement.run(
            case.request,
            outcome,
            intent=case.output,
            deletion=deletion,
            current_owner=current,
            runtime_transition=migration_case,
        )
    assert not outcome.exists()


def test_supplemental_readonly_verify_requires_fresh_matching_owner(
    case: Case, supplemental_case: tuple[Path, Path, Path], monkeypatch: pytest.MonkeyPatch
) -> None:
    deletion, current, outcome = supplemental_case
    retirement.run(
        case.request, outcome, intent=case.output, deletion=deletion, current_owner=current
    )
    saved = outcome.read_bytes()
    later = time.time_ns() + retirement.MAX_DISPOSITION_AGE_NS + 1
    monkeypatch.setattr(time, "time_ns", lambda: later)
    with pytest.raises(ReplayError, match="stale"):
        retirement.run(
            case.request,
            outcome,
            intent=case.output,
            deletion=deletion,
            current_owner=current,
            verify=True,
        )
    assert outcome.read_bytes() == saved


def test_tiny_repeated_rebuild_separates_container_bytes_packet_pts_and_pcm(tmp_path: Path) -> None:
    first, second, voice = tmp_path / "one.mov", tmp_path / "two.mov", tmp_path / "voice.wav"
    subprocess.run(
        [
            "ffmpeg",
            "-nostdin",
            "-v",
            "error",
            "-f",
            "lavfi",
            "-i",
            "color=blue:s=64x64:r=30",
            "-frames:v",
            "2",
            "-c:v",
            "prores_ks",
            "-profile:v",
            "1",
            "-video_track_timescale",
            "600",
            str(first),
        ],
        check=True,
        capture_output=True,
        timeout=20,
    )
    subprocess.run(
        [
            "ffmpeg",
            "-nostdin",
            "-v",
            "error",
            "-f",
            "lavfi",
            "-i",
            "color=red:s=64x64:r=30",
            "-frames:v",
            "3",
            "-c:v",
            "prores_ks",
            "-profile:v",
            "1",
            "-video_track_timescale",
            "600",
            str(second),
        ],
        check=True,
        capture_output=True,
        timeout=20,
    )
    subprocess.run(
        [
            "ffmpeg",
            "-nostdin",
            "-v",
            "error",
            "-f",
            "lavfi",
            "-i",
            "sine=frequency=440:sample_rate=44100",
            "-af",
            "atrim=end_sample=7350",
            "-ac",
            "2",
            "-c:a",
            "pcm_s32le",
            str(voice),
        ],
        check=True,
        capture_output=True,
        timeout=20,
    )
    parts = (native_join.PartFile(first, 2), native_join.PartFile(second, 3))
    left, right = tmp_path / "revision-a.mov", tmp_path / "revision-b.mov"
    assert native_join.join(parts, rate=30, output=left, audio=voice) == 5
    assert native_join.join(parts, rate=30, output=right, audio=voice) == 5
    left_packets = subprocess.run(
        [
            "ffprobe",
            "-v",
            "error",
            "-select_streams",
            "v:0",
            "-show_packets",
            "-show_entries",
            "packet=pts,dts,duration,data_hash",
            "-show_data_hash",
            "sha256",
            "-of",
            "json",
            str(left),
        ],
        check=True,
        capture_output=True,
        timeout=20,
    ).stdout
    right_packets = subprocess.run(
        [
            "ffprobe",
            "-v",
            "error",
            "-select_streams",
            "v:0",
            "-show_packets",
            "-show_entries",
            "packet=pts,dts,duration,data_hash",
            "-show_data_hash",
            "sha256",
            "-of",
            "json",
            str(right),
        ],
        check=True,
        capture_output=True,
        timeout=20,
    ).stdout
    assert left_packets == right_packets
    packet_data = record(json.loads(left_packets))
    packets = packet_data["packets"]
    assert isinstance(packets, list)
    assert [record(item)["pts"] for item in packets] == [0, 20, 40, 60, 80]
    left_pcm = subprocess.run(
        [
            "ffmpeg",
            "-nostdin",
            "-v",
            "error",
            "-xerror",
            "-i",
            str(left),
            "-map",
            "0:a:0",
            "-c:a",
            "pcm_s32le",
            "-f",
            "s32le",
            "-",
        ],
        check=True,
        capture_output=True,
        timeout=20,
    ).stdout
    right_pcm = subprocess.run(
        [
            "ffmpeg",
            "-nostdin",
            "-v",
            "error",
            "-xerror",
            "-i",
            str(right),
            "-map",
            "0:a:0",
            "-c:a",
            "pcm_s32le",
            "-f",
            "s32le",
            "-",
        ],
        check=True,
        capture_output=True,
        timeout=20,
    ).stdout
    voice_pcm = subprocess.run(
        [
            "ffmpeg",
            "-nostdin",
            "-v",
            "error",
            "-i",
            str(voice),
            "-c:a",
            "pcm_s32le",
            "-f",
            "s32le",
            "-",
        ],
        check=True,
        capture_output=True,
        timeout=20,
    ).stdout
    assert left_pcm == right_pcm == voice_pcm
    assert len(left_pcm) == 58800
    assert (
        hashlib.sha256(left.read_bytes()).hexdigest()
        == hashlib.sha256(right.read_bytes()).hexdigest()
    )
    full_av = subprocess.run(
        [
            "ffmpeg",
            "-nostdin",
            "-v",
            "error",
            "-xerror",
            "-i",
            str(left),
            "-map",
            "0:v:0",
            "-map",
            "0:a:0",
            "-f",
            "null",
            "-",
        ],
        check=True,
        capture_output=True,
        timeout=20,
    )
    assert full_av.returncode == 0
    with pytest.raises(ReplayError, match="Refusing to overwrite"):
        native_join.join(parts, rate=30, output=left, audio=voice)
