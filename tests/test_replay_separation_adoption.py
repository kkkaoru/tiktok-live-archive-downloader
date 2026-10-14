"""Synthetic retained outcomes prove adoption without replaying ML inference."""

from __future__ import annotations

import hashlib
import json
import os
import wave
from dataclasses import dataclass
from pathlib import Path

import pytest

from replay import replay_clock_adoption as retained
from replay import replay_separation_adoption as adoption
from replay.bgm_separation import MODEL_NAME, MODEL_REVISION
from replay.core import ReplayError
from replay.delivery_audio import verify_material
from replay.recordings import record
from replay.stages import StageError, run_stage
from replay.workflow import load_stages
from replay.workflow import main as workflow_main


def _json(path: Path, value: object) -> Path:
    path.write_text(json.dumps(value), encoding="utf-8")
    return path


def _ref(path: Path) -> dict[str, str]:
    return {"path": str(path), "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}


def _pcm(path: Path, *, count: int = 882, rate: int = 44100, channels: int = 2) -> None:
    with wave.open(str(path), "wb") as stream:
        stream.setnchannels(channels)
        stream.setsampwidth(2)
        stream.setframerate(rate)
        stream.writeframes(b"\x01\0" * channels * count)


@dataclass(frozen=True)
class Fixture:
    root: Path
    config: Path
    work: Path
    source: Path
    code: Path

    def data(self) -> dict[str, object]:
        return record(json.loads(self.config.read_bytes()))

    def save(self, data: object) -> None:
        _json(self.config, data)

    def role(self, name: str) -> Path:
        return Path(str(record(record(self.data()["roles"])[name])["path"]))

    def prepare(self) -> retained.Prepared:
        return adoption.prepare(self.config, self.work)

    def seal(self) -> tuple[Path, Path]:
        self.prepare()
        plan, receipt = self.work / "plan.json", self.work / "adoption-evidence.json"
        adoption.seal(plan, receipt)
        return plan, receipt


@pytest.fixture
def fixture(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Fixture:
    root = tmp_path / "checkout"
    work = root / "edits" / "work"
    models = root / "edits" / "models"
    work.mkdir(parents=True)
    models.mkdir()
    monkeypatch.setattr(adoption, "ROOT", root)
    monkeypatch.setattr(retained, "ROOT", root)
    code = root / "reader.py"
    code.write_bytes(b"synthetic-reader-only\n")
    monkeypatch.setattr(
        retained, "_code", lambda modules: (retained.Pin(code, _ref(code)["sha256"]),)
    )
    source = work / "original.mp4"
    source.write_bytes(b"synthetic-original-no-decoder\n")
    model_input = work / "prepared.wav"
    _pcm(model_input)
    separation = work / "separation"
    separation.mkdir()
    vocals = separation / "vocals.wav"
    _pcm(vocals)
    result = _json(
        separation / "result.json",
        {
            "source": str(model_input),
            "gainDB": 0,
            "preparedInput": True,
            "device": "cpu",
            "modelInput": str(model_input),
            "vocals": str(vocals),
            "separation": "synthetic-no-inference",
            "timestampCorrection": False,
            "loudnessNormalization": False,
            "deliveryComplete": False,
        },
    )
    audio = _json(
        work / "audio-clock.json",
        {
            "streams": [{"sample_rate": "44100", "time_base": "1/90000"}],
            "frames": [{"pts": 900, "nb_samples": 441}, {"pts": 3600, "nb_samples": 441}],
        },
    )
    clock_workflow = _json(
        work / "clock-workflow.json",
        {
            "version": 1,
            "stages": [
                {
                    "name": "adopt-clock",
                    "command": ["synthetic-clock-check"],
                    "verify_command": ["synthetic-clock-check"],
                    "inputs": [str(source), str(code)],
                    "outputs": [str(audio)],
                    "timeout_seconds": 30,
                    "adopt_existing": True,
                }
            ],
        },
    )
    stage = load_stages(clock_workflow)[0]
    state = work / "clock-state"

    def execute_clock(command: tuple[str, ...], log: Path, timeout: int) -> None:
        log.write_bytes(b"")

    assert run_stage(stage, state_directory=state, execute=execute_clock) == "adopted"
    clock_receipt = state / "adopt-clock.json"
    weights, config = models / "weights-blob", models / "config-blob"
    weights.write_bytes(b"synthetic-model-only")
    config.write_bytes(b"synthetic-config-only")
    monkeypatch.setattr(adoption, "MODEL_SHA256", _ref(weights)["sha256"])
    monkeypatch.setattr(adoption, "MODEL_CONFIG_SHA256", _ref(config)["sha256"])
    provider = root / "provider.METADATA"
    provider.write_bytes(b"Name: synthetic-provider\nVersion: 1\n")
    snapshots: dict[str, object] = {}
    maps = {
        "code": {str(code): _ref(code)["sha256"]},
        "model": {str(weights): _ref(weights)["sha256"], str(config): _ref(config)["sha256"]},
        "provider": {str(provider): _ref(provider)["sha256"]},
    }
    for role, pins in maps.items():
        for suffix in ("Before", "Terminal"):
            path = _json(work / f"{role}-{suffix}.json", {"inputs": pins})
            snapshots[f"{role}{suffix}"] = {"evidence": _ref(path), "format": "input-inventory"}
    status, log, phases = work / "direct.status", work / "direct.log", work / "PHASES.jsonl"
    status.write_bytes(b"0\n")
    log.write_bytes(b"original progress and /usr/bin/time diagnostics\n")
    # Epoch elapsed and monotonic elapsed intentionally differ.
    phases.write_text(
        json.dumps(
            {"phase": "original-separation", "start": 100, "end": 101.1, "seconds": 1, "exit": 0}
        )
        + "\n",
        encoding="utf-8",
    )
    prepared_config = _json(
        work / "adoption-config.json",
        {
            "version": 1,
            "protocol": adoption.PROTOCOL,
            "roles": {
                "source": _ref(source),
                "modelInput": _ref(model_input),
                "vocals": _ref(vocals),
                "result": _ref(result),
                "audioClock": _ref(audio),
            },
            "tools": {"python": _ref(code), "env": _ref(code)},
            "model": {
                "name": MODEL_NAME,
                "revision": MODEL_REVISION,
                "weights": _ref(weights),
                "config": _ref(config),
            },
            "providerInputs": [_ref(provider)],
            "boundaries": snapshots,
            "terminal": {
                "status": _ref(status),
                "log": _ref(log),
                "phases": _ref(phases),
                "phase": "original-separation",
            },
            "clockStage": {
                "workflow": _ref(clock_workflow),
                "receipt": _ref(clock_receipt),
                "stageName": "adopt-clock",
                "inputs": [_ref(source), _ref(code)],
            },
            "timeoutSeconds": 120,
        },
    )
    code.chmod(0o700)
    return Fixture(root, prepared_config, work / "reconciled", source, code)


def test_ordinary_graphs_adopt_real_stage_without_reissuing_generation(
    fixture: Fixture, capsys: pytest.CaptureFixture[str]
) -> None:
    originals = {
        path: path.read_bytes()
        for path in (
            fixture.source,
            fixture.role("modelInput"),
            fixture.role("vocals"),
            fixture.role("result"),
            fixture.config.parent / "PHASES.jsonl",
        )
    }
    prepared = fixture.prepare()
    for graph in (prepared.evidence_workflow, prepared.adoption_workflow):
        assert workflow_main([str(graph), "--state", str(fixture.work / "checked"), "--check"]) == 0
        capsys.readouterr()
    evidence, adopted = (
        load_stages(prepared.evidence_workflow)[0],
        load_stages(prepared.adoption_workflow)[0],
    )
    calls: list[str] = []

    def execute(command: tuple[str, ...], log: Path, timeout: int) -> None:
        action, plan, receipt = command[-3:]
        log.write_bytes(b"")
        calls.append(action)
        if action == "seal":
            adoption.seal(Path(plan), Path(receipt))
        else:
            adoption.verify(Path(plan), Path(receipt))

    assert (
        run_stage(evidence, state_directory=fixture.work / "evidence-state", execute=execute)
        == "completed"
    )
    assert (
        run_stage(adopted, state_directory=fixture.work / "adoption-state", execute=execute)
        == "adopted"
    )
    assert (
        run_stage(adopted, state_directory=fixture.work / "adoption-state", execute=execute)
        == "reused"
    )
    assert calls == ["seal", "verify", "verify", "verify"]
    assert fixture.source in adopted.inputs
    assert fixture.role("modelInput") in adopted.inputs
    assert set(adopted.outputs) == {fixture.role("vocals"), fixture.role("result")}
    assert {path: path.read_bytes() for path in originals} == originals
    assert not (fixture.role("result").parent / "producer.json").exists()
    proof = record(json.loads((fixture.work / "adoption-evidence.json").read_bytes()))
    assert proof["currentInputBytesVerified"] is True
    assert proof["currentPCMClockVerified"] is True
    assert proof["historicalGenerationBindingFullyVerified"] is False
    assert proof["originalGenerationReceiptReissued"] is False
    assert proof["speechOrMusicQualityApproved"] is False
    assert record(record(proof["domains"])["nativeAudioClock"])["firstPTS"] == 900
    # This volume is case-insensitive: lowercase phases can alias original PHASES.
    # The new timing destination must instead have its own distinct work parent.
    assert str(fixture.work / "phases.jsonl") in evidence.command
    assert fixture.work != (fixture.config.parent / "PHASES.jsonl").parent


def test_plan_never_hashes_or_decodes_media(
    fixture: Fixture, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(
        adoption, "digest", lambda path: pytest.fail("prepare performed a full hash")
    )
    monkeypatch.setattr(
        adoption, "verify_material", lambda source, directory: pytest.fail("prepare decoded PCM")
    )
    fixture.prepare()


def test_descriptive_before_records_are_retained_without_fake_snapshot(fixture: Fixture) -> None:
    data = fixture.data()
    metadata = _json(
        fixture.config.parent / "launch-metadata.json",
        {"historicalEnvironmentRecorded": False, "pinnedModel": MODEL_NAME},
    )
    boundaries = record(data["boundaries"])
    for name in boundaries:
        boundaries[name] = {"evidence": _ref(metadata), "format": "retained-metadata"}
    fixture.save(data)
    _, receipt = fixture.seal()
    proof = record(json.loads(receipt.read_bytes()))
    records = record(record(proof["domains"])["retainedBoundaryEvidence"])
    assert all(record(value)["inputInventoryVerified"] is False for value in records.values())
    assert proof["historicalInitialCodeSnapshotVerified"] is False


def test_immutable_copy_provenance_can_be_bound_without_repinning_original_phase(
    fixture: Fixture,
) -> None:
    data = fixture.data()
    original = fixture.config.parent / "PHASES.jsonl"
    original_bytes = original.read_bytes()
    snapshot = fixture.config.parent / "immutable-original-phases.jsonl"
    snapshot.write_bytes(original_bytes)
    provenance = _json(
        fixture.config.parent / "phase-copy.json",
        {
            "originalPath": str(original),
            "originalSHA256": _ref(original)["sha256"],
            "copy": _ref(snapshot),
            "exactBytesCopied": True,
        },
    )
    record(data["terminal"])["phases"] = _ref(snapshot)
    data["supportingEvidence"] = [_ref(provenance)]
    fixture.save(data)
    plan, receipt = fixture.seal()
    original.write_bytes(original_bytes + b'{"phase":"later-process"}\n')
    adoption.verify(plan, receipt)
    proof = record(json.loads(receipt.read_bytes()))
    assert str(original) not in record(proof["inputs"])
    assert str(provenance) in record(proof["inputs"])
    provenance.write_bytes(b"{}")
    with pytest.raises(ReplayError, match="bytes changed"):
        adoption.verify(plan, receipt)


def test_text_boundary_inventory_is_supported(fixture: Fixture) -> None:
    data = fixture.data()
    boundaries = record(data["boundaries"])
    row = record(boundaries["codeBefore"])
    path = Path(str(record(row["evidence"])["path"]))
    path.write_text(f"{_ref(fixture.code)['sha256']}  {fixture.code}\n", encoding="utf-8")
    row["evidence"] = _ref(path)
    fixture.save(data)
    fixture.seal()


def test_original_source_decode_route_remains_an_honest_current_candidate(fixture: Fixture) -> None:
    old_input = fixture.role("modelInput")
    result = fixture.role("result")
    model_input = result.parent / "model-input.wav"
    old_input.rename(model_input)
    raw = record(json.loads(result.read_bytes()))
    raw.update(
        {
            "source": str(fixture.source),
            "preparedInput": False,
            "modelInput": str(model_input),
            "gainDB": -0.1,
        }
    )
    _json(result, raw)
    data = fixture.data()
    record(data["roles"])["modelInput"] = _ref(model_input)
    record(data["roles"])["result"] = _ref(result)
    fixture.save(data)
    _, receipt = fixture.seal()
    proof = record(json.loads(receipt.read_bytes()))
    assert record(proof["domains"])["preparedInput"] is False
    assert proof["historicalSourceToPCMContentVerified"] is False


def _model_alias_inventory(fixture: Fixture) -> Path:
    data = fixture.data()
    model = record(data["model"])
    weights = Path(str(record(model["weights"])["path"]))
    config = Path(str(record(model["config"])["path"]))
    aliases = weights.parent / "snapshots"
    aliases.mkdir()
    weight_alias, config_alias = aliases / "model.safetensors", aliases / "config.json"
    weight_alias.symlink_to(weights)
    config_alias.symlink_to(config)
    inventory = fixture.config.parent / "initial-source-model.sha256"
    inventory.write_text(
        f"{_ref(fixture.source)['sha256']}  {fixture.source}\n"
        f"{_ref(weights)['sha256']}  {weight_alias}\n"
        f"{_ref(config)['sha256']}  {config_alias}\n",
        encoding="utf-8",
    )
    boundaries = record(data["boundaries"])
    for name in ("modelBefore", "modelTerminal"):
        boundaries[name] = {"evidence": _ref(inventory), "format": "input-inventory"}
    fixture.save(data)
    return weight_alias


def test_original_hf_aliases_and_combined_source_model_inventory_are_preserved(
    fixture: Fixture,
) -> None:
    alias = _model_alias_inventory(fixture)
    original = (fixture.config.parent / "initial-source-model.sha256").read_bytes()
    _, receipt = fixture.seal()
    proof = record(json.loads(receipt.read_bytes()))
    boundary = record(record(record(proof["domains"])["retainedBoundaryEvidence"])["modelBefore"])
    assert record(boundary["resolvedReferences"])[str(alias)] == str(alias.resolve())
    assert (fixture.config.parent / "initial-source-model.sha256").read_bytes() == original


def test_model_alias_changes_during_readonly_verification_are_rejected(
    fixture: Fixture, monkeypatch: pytest.MonkeyPatch
) -> None:
    alias = _model_alias_inventory(fixture)
    real_verify = verify_material
    other = alias.parent / "other-weights"
    other.write_bytes(alias.read_bytes())

    def verify_and_retarget(source: Path, directory: Path) -> None:
        real_verify(source, directory)
        alias.unlink()
        alias.symlink_to(other)

    monkeypatch.setattr(adoption, "verify_material", verify_and_retarget)
    with pytest.raises(ReplayError, match="inventory alias changed"):
        fixture.seal()


def test_truncated_pcm_is_rejected_even_if_header_and_declared_hash_are_rebound(
    fixture: Fixture,
) -> None:
    vocals = fixture.role("vocals")
    vocals.write_bytes(vocals.read_bytes()[:-4])
    data = fixture.data()
    record(data["roles"])["vocals"] = _ref(vocals)
    fixture.save(data)
    with pytest.raises(ReplayError, match="Truncated"):
        fixture.seal()


@pytest.mark.parametrize("role", ["source", "modelInput", "vocals", "result", "audioClock"])
def test_changed_role_bytes_refuse_adoption(fixture: Fixture, role: str) -> None:
    fixture.prepare()
    fixture.role(role).write_bytes(b"changed original bytes")
    with pytest.raises((ReplayError, StageError), match="changed|complete"):
        adoption.seal(fixture.work / "plan.json", fixture.work / "adoption-evidence.json")


@pytest.mark.parametrize(
    "field,value",
    [
        ("device", "mps"),
        ("preparedInput", "true"),
        ("timestampCorrection", True),
        ("loudnessNormalization", True),
        ("deliveryComplete", True),
        ("gainDB", 1),
        ("gainDB", float("nan")),
        ("modelInput", "/not/the/input.wav"),
        ("vocals", "/not/the/vocals.wav"),
        ("source", "/wrong/source.wav"),
    ],
)
def test_policy_or_binding_mutation_is_rejected(
    fixture: Fixture, field: str, value: object
) -> None:
    path = fixture.role("result")
    raw = record(json.loads(path.read_bytes()))
    raw[field] = value
    _json(path, raw)
    data = fixture.data()
    record(data["roles"])["result"] = _ref(path)
    fixture.save(data)
    with pytest.raises(ReplayError, match="policy|binding"):
        fixture.seal()


@pytest.mark.parametrize("count,rate,channels", [(881, 44100, 2), (882, 48000, 2), (882, 44100, 1)])
def test_vocals_sample_clock_and_format_must_match(
    fixture: Fixture, count: int, rate: int, channels: int
) -> None:
    path = fixture.role("vocals")
    _pcm(path, count=count, rate=rate, channels=channels)
    data = fixture.data()
    record(data["roles"])["vocals"] = _ref(path)
    fixture.save(data)
    with pytest.raises(ReplayError, match="sample clock|PCM format"):
        fixture.seal()


def test_original_clock_duration_is_not_a_global_shift(fixture: Fixture) -> None:
    path = fixture.role("audioClock")
    raw = record(json.loads(path.read_bytes()))
    raw["frames"] = [{"pts": 900, "nb_samples": 48000}]
    _json(path, raw)
    data = fixture.data()
    record(data["roles"])["audioClock"] = _ref(path)
    stage = record(data["clockStage"])
    receipt = Path(str(record(stage["receipt"])["path"]))
    old = record(json.loads(receipt.read_bytes()))
    old["outputs"] = {str(path): _ref(path)["sha256"]}
    _json(receipt, old)
    stage["receipt"] = _ref(receipt)
    fixture.save(data)
    with pytest.raises(ReplayError, match="native decoded audio duration"):
        fixture.seal()


@pytest.mark.parametrize("kind", ["code", "model", "provider"])
def test_changed_boundary_inventory_is_not_opaque_fallback(fixture: Fixture, kind: str) -> None:
    data = fixture.data()
    boundary = record(record(data["boundaries"])[f"{kind}Terminal"])
    path = Path(str(record(boundary["evidence"])["path"]))
    _json(path, {"inputs": {str(fixture.code): "0" * 64}})
    boundary["evidence"] = _ref(path)
    fixture.save(data)
    with pytest.raises(ReplayError, match="boundaries changed"):
        fixture.seal()


@pytest.mark.parametrize(
    "fault",
    [
        "nonzero-status",
        "failed-phase",
        "missing-phase",
        "duplicate-phase",
        "partial-phase",
        "nan",
        "negative",
        "extra-field",
    ],
)
def test_terminal_evidence_is_finite_complete_and_unambiguous(fixture: Fixture, fault: str) -> None:
    data = fixture.data()
    terminal = record(data["terminal"])
    status = Path(str(record(terminal["status"])["path"]))
    phases = Path(str(record(terminal["phases"])["path"]))
    row = record(json.loads(phases.read_bytes()))
    if fault == "nonzero-status":
        status.write_bytes(b"1\n")
    elif fault == "failed-phase":
        row["exit"] = 1
    elif fault == "missing-phase":
        row["phase"] = "other"
    elif fault == "nan":
        row["seconds"] = float("nan")
    elif fault == "negative":
        row["seconds"] = -1
    elif fault == "extra-field":
        row["newField"] = True
    content = json.dumps(row) + "\n"
    if fault == "duplicate-phase":
        content *= 2
    if fault == "partial-phase":
        content = content.rstrip("\n")
    phases.write_text(content, encoding="utf-8")
    terminal["status"], terminal["phases"] = _ref(status), _ref(phases)
    fixture.save(data)
    with pytest.raises(ReplayError, match="phase|successful"):
        fixture.seal()


def test_failed_direct_process_with_partial_outputs_does_not_launch_inference(
    fixture: Fixture, monkeypatch: pytest.MonkeyPatch
) -> None:
    data = fixture.data()
    terminal = record(data["terminal"])
    status = Path(str(record(terminal["status"])["path"]))
    status.write_bytes(b"1\n")
    terminal["status"] = _ref(status)
    fixture.save(data)
    called: list[object] = []
    monkeypatch.setattr(adoption, "verify_material", lambda *args: called.append(args))
    with pytest.raises(ReplayError, match="successful"):
        fixture.seal()
    assert not (fixture.work / "adoption-evidence.json").exists()
    assert not called


def test_evidence_receipt_is_immutable_and_exclusive(fixture: Fixture) -> None:
    plan, receipt = fixture.seal()
    before = receipt.read_bytes()
    adoption.verify(plan, receipt)
    assert receipt.read_bytes() == before
    with pytest.raises(ReplayError, match="new evidence"):
        adoption.seal(plan, receipt)
    raw = record(json.loads(before))
    raw["historicalGenerationBindingFullyVerified"] = True
    _json(receipt, raw)
    with pytest.raises(ReplayError, match="evidence changed"):
        adoption.verify(plan, receipt)


def test_changed_terminal_bytes_or_source_identity_fail_readonly_verification(
    fixture: Fixture,
) -> None:
    plan, receipt = fixture.seal()
    fixture.source.touch()
    # A content-identical identity change before a new operation is allowed only
    # after rehash; changed content remains rejected without rewriting evidence.
    adoption.verify(plan, receipt)
    fixture.source.write_bytes(b"new-source")
    with pytest.raises(ReplayError, match="bytes changed"):
        adoption.verify(plan, receipt)


@pytest.mark.parametrize(
    "fault",
    [
        "wrong-model",
        "wrong-weights",
        "missing-role",
        "overlap",
        "bad-boundary",
        "no-provider",
        "bad-phase",
        "bad-timeout",
        "extra-field",
        "clock-running",
        "clock-missing-source",
        "clock-wrong-output",
        "clock-no-verifier",
        "duplicate-inputs",
    ],
)
def test_invalid_configuration_or_predecessor_refuses_preparation(
    fixture: Fixture, fault: str
) -> None:
    data = fixture.data()
    if fault == "wrong-model":
        record(data["model"])["name"] = "other-model"
    elif fault == "wrong-weights":
        record(record(data["model"])["weights"])["sha256"] = "0" * 64
    elif fault == "missing-role":
        del record(data["roles"])["result"]
    elif fault == "overlap":
        record(data["roles"])["vocals"] = record(data["roles"])["source"]
    elif fault == "bad-boundary":
        record(record(data["boundaries"])["modelBefore"])["format"] = "trust-me"
    elif fault == "no-provider":
        data["providerInputs"] = []
    elif fault == "bad-phase":
        record(data["terminal"])["phase"] = ""
    elif fault == "bad-timeout":
        data["timeoutSeconds"] = True
    elif fault == "extra-field":
        data["skipMusicRemoval"] = True
    elif fault == "clock-running":
        row = record(data["clockStage"])
        receipt = Path(str(record(row["receipt"])["path"]))
        _json(receipt, {"status": "running"})
        row["receipt"] = _ref(receipt)
    elif fault == "duplicate-inputs":
        row = record(data["clockStage"])
        row["inputs"] = [_ref(fixture.source), _ref(fixture.source)]
    else:
        row = record(data["clockStage"])
        path = Path(str(record(row["workflow"])["path"]))
        raw = record(json.loads(path.read_bytes()))
        stages = raw["stages"]
        assert isinstance(stages, list)
        stage = record(stages[0])
        if fault == "clock-missing-source":
            stage["inputs"] = [str(fixture.code)]
            row["inputs"] = [_ref(fixture.code)]
        elif fault == "clock-wrong-output":
            stage["outputs"] = [str(fixture.config.parent / "different-clock.json")]
        else:
            stage["verify_command"] = []
            stage["adopt_existing"] = False
        _json(path, raw)
        row["workflow"] = _ref(path)
    fixture.save(data)
    with pytest.raises((ReplayError, StageError)):
        fixture.prepare()


def test_changed_plan_input_membership_and_config_pin_are_rejected(fixture: Fixture) -> None:
    fixture.prepare()
    plan = fixture.work / "plan.json"
    raw = record(json.loads(plan.read_bytes()))
    raw["inputs"] = [_ref(fixture.source)]
    _json(plan, raw)
    with pytest.raises(ReplayError, match="membership"):
        adoption.seal(plan, fixture.work / "adoption-evidence.json")
    fixture.config.write_bytes(b"{}")
    with pytest.raises(ReplayError, match="Configuration pin"):
        adoption.seal(plan, fixture.work / "adoption-evidence.json")


def test_occupied_and_escaping_paths_are_never_overwritten(fixture: Fixture) -> None:
    fixture.prepare()
    with pytest.raises(ReplayError, match="new reconciliation"):
        fixture.prepare()
    plan = fixture.work / "plan.json"
    with pytest.raises(ReplayError, match="plan work root"):
        adoption.seal(plan, fixture.config.parent / "different.json")
    with pytest.raises(ReplayError, match="escapes plan"):
        adoption.verify(plan, fixture.config.parent / "different.json")
    with pytest.raises(ReplayError, match="escapes repository"):
        adoption.prepare(fixture.config, fixture.root / "not-work")


def test_json_duplicate_keys_and_linked_inputs_are_rejected(fixture: Fixture) -> None:
    fixture.config.write_bytes(b'{"version":1,"version":1}')
    with pytest.raises(ReplayError, match="Duplicate"):
        fixture.prepare()
    fixture.config.unlink()
    os.link(fixture.source, fixture.config)
    with pytest.raises(ReplayError, match="single-link"):
        fixture.prepare()


def test_cli_prepare_seal_verify_and_error(
    fixture: Fixture, capsys: pytest.CaptureFixture[str]
) -> None:
    assert adoption.main(["prepare", str(fixture.config), str(fixture.work)]) == 0
    assert "sequence.json" in capsys.readouterr().out
    plan, receipt = fixture.work / "plan.json", fixture.work / "adoption-evidence.json"
    assert adoption.main(["seal", str(plan), str(receipt)]) == 0
    assert adoption.main(["verify", str(plan), str(receipt)]) == 0
    assert adoption.main(["seal", str(plan), str(receipt)]) == 1
    assert "SEPARATION_ADOPTION_ERROR" in capsys.readouterr().err
