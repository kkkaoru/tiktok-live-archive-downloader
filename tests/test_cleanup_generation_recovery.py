"""Synthetic generations retain old results and run only missing production."""

import json
import shutil
import subprocess
import time
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path

import pytest
from pytest import MonkeyPatch
from test_subtitle_cleanup_workflow import Usage, document
from test_subtitle_cleanup_workflow import chunk as chunk
from test_subtitle_cleanup_workflow import settings as settings

from replay import cleanup_generation_recovery as recovery
from replay import cleanup_lane_successor as old
from replay import subtitle_admission_verification as admission
from replay import subtitle_candidate, subtitle_observations
from replay import subtitle_cleanup_assembly as assembly
from replay import subtitle_cleanup_workflow as cleanup
from replay.local_asr import fingerprint
from replay.sequence_journal import bounded_read, encode
from replay.workflow import parse_stages

PREPARATION_PATHS = recovery._preparation_paths
PROCESS_ABSENCE = recovery._process_absence


@pytest.fixture
def recovery_request(chunk: Path, monkeypatch: MonkeyPatch) -> Path:
    monkeypatch.setattr(old, "_path", lambda value, **kwargs: Path(str(value)))
    monkeypatch.setattr(recovery, "_code", lambda: {})
    monkeypatch.setattr(old, "_quiescent", lambda *args, **kwargs: None)
    legacy: dict[str, object] = {
        "chunks": [{"index": 0, "chunk": str(chunk), "lane": 0, "successorStages": ["old"]}],
        "states": [str(chunk.parent / "old-state")],
        "workflows": [str(chunk.parent / "old-workflow.json")],
        "settings": str(Path.cwd() / "settings.json"),
        "originalPlan": str(chunk),
    }
    monkeypatch.setattr(old, "_load", lambda *args: legacy)
    monkeypatch.setattr(recovery, "_prior_complete", lambda *args, **kwargs: None)
    monkeypatch.setattr(
        recovery, "_preparation_paths", lambda path: (path, Path.cwd() / "source.mp4")
    )

    @contextmanager
    def owned(plan: Path, index: int) -> Iterator[tuple[dict[str, object], dict[str, object]]]:
        yield legacy, {"chunk": str(chunk)}

    monkeypatch.setattr(old, "_owned", owned)
    owner = document(chunk.parent / "owner.json", {"synthetic": True})
    legacy_ref = document(chunk.parent / "legacy.json", legacy)
    path = chunk.parent / "recovery_request.json"
    document(
        path, {"version": 1, "successor": legacy_ref, "handoff": owner, "workers": 1, "adopted": []}
    )
    return path


def render_fixture(config: Path) -> Path:
    raw = json.loads(config.read_text())
    directory = Path(raw["outputDirectory"])
    directory.mkdir()
    movie = directory / "candidate.mp4"
    movie.write_bytes(b"synthetic repaired media")
    source, native = Path(raw["source"]), Path(raw["nativeExecutable"])
    document(
        directory / "candidate.json",
        {
            "inputs": {
                str(config): fingerprint(config),
                str(source): fingerprint(source),
                str(native): fingerprint(native),
            },
            "render": {"frames": len(raw["clock"]["pts"])},
            "sourceStart": "0",
            "videoSHA256": fingerprint(movie),
        },
    )
    return directory / "candidate.json"


def test_pending_generation_actual_pure_domains(
    recovery_request: Path, monkeypatch: MonkeyPatch
) -> None:
    monkeypatch.setattr(subtitle_candidate, "render_candidate", render_fixture)
    plan = recovery.prepare(recovery_request, recovery_request.parent / "new")
    raw = json.loads(plan.read_text())
    assert raw["workers"] == 1
    assert raw["chunks"][0]["mode"] == "pending"
    assert (
        json.loads((plan.parent / "chunk-00000/chunk.json").read_text())["maximumActiveChunks"] == 1
    )
    recovery.phase(plan, 0)
    recovery.phase(plan, 0, verify=True)
    result = json.loads((plan.parent / "verified-00000.json").read_text())
    assert result["verification"]["repairVerified"] is True
    assert result["qualityApproved"] is False
    assert not (recovery_request.parent / "repair/candidate.mp4").exists()
    with pytest.raises(FileExistsError):
        recovery.phase(plan, 0)


def test_retained_and_adopted_proofs(
    recovery_request: Path, chunk: Path, monkeypatch: MonkeyPatch
) -> None:
    cleanup.build_candidate(chunk)
    render_fixture(chunk.parent / "candidate-config.json")
    monkeypatch.setattr(recovery, "_prior_complete", lambda *args: None)
    plan = recovery.prepare(recovery_request, recovery_request.parent / "retained")
    recovery.phase(plan, 0)
    assert json.loads(plan.read_text())["chunks"][0]["mode"] == "retained-completed"
    proof = document(
        chunk.parent / "proof.json", admission.verify_admission(chunk, verify_repair=True)
    )
    raw = json.loads(recovery_request.read_text())
    raw["adopted"] = [{"index": 0, "proof": proof}]
    recovery_request.write_text(json.dumps(raw), encoding="utf-8")
    adopted = recovery.prepare(recovery_request, recovery_request.parent / "adopted")
    recovery.phase(adopted, 0)
    recovery.phase(adopted, 0, verify=True)
    assert json.loads(adopted.read_text())["chunks"][0]["mode"] == "adopted-existing"


@pytest.mark.parametrize("mutation", ["version", "extra", "workers", "duplicate", "index"])
def test_bad_request_refused(recovery_request: Path, mutation: str) -> None:
    raw = json.loads(recovery_request.read_text())
    if mutation == "version":
        raw["version"] = 2
    elif mutation == "extra":
        raw["unknown"] = True
    elif mutation == "workers":
        raw["workers"] = 0
    elif mutation == "duplicate":
        raw["adopted"] = [{"index": 0, "proof": {}}, {"index": 0, "proof": {}}]
    else:
        raw["adopted"] = [{"index": 1, "proof": {}}]
    recovery_request.write_text(json.dumps(raw), encoding="utf-8")
    with pytest.raises(ValueError):
        recovery.prepare(recovery_request, recovery_request.parent / "invalid")


def test_partial_old_repair_held(recovery_request: Path, chunk: Path) -> None:
    (chunk.parent / "repair").mkdir()
    (chunk.parent / "repair/candidate.mp4").write_bytes(b"partial")
    with pytest.raises(ValueError, match="Partial previous"):
        recovery.prepare(recovery_request, recovery_request.parent / "partial")


def test_adopted_missing_or_changed_refused(recovery_request: Path, chunk: Path) -> None:
    raw = json.loads(recovery_request.read_text())
    raw["adopted"] = [{"index": 0, "proof": document(chunk.parent / "proof.json", {})}]
    recovery_request.write_text(json.dumps(raw), encoding="utf-8")
    with pytest.raises(ValueError, match="existing repair"):
        recovery.prepare(recovery_request, recovery_request.parent / "missing")
    cleanup.build_candidate(chunk)
    render_fixture(chunk.parent / "candidate-config.json")
    with pytest.raises(ValueError, match="proof differs"):
        recovery.prepare(recovery_request, recovery_request.parent / "changed")


def test_plan_code_input_and_proof_mutations(
    recovery_request: Path, monkeypatch: MonkeyPatch
) -> None:
    monkeypatch.setattr(subtitle_candidate, "render_candidate", render_fixture)
    plan = recovery.prepare(recovery_request, recovery_request.parent / "new")
    recovery.phase(plan, 0)
    proof = plan.parent / "verified-00000.json"
    proof.write_text("{}", encoding="utf-8")
    with pytest.raises(ValueError, match="result differs"):
        recovery.phase(plan, 0, verify=True)
    with pytest.raises(ValueError, match="Unknown recovery"):
        recovery.phase(plan, 1, verify=True)
    original = plan.read_bytes()
    raw = json.loads(original)
    raw["kind"] = "other"
    plan.write_text(json.dumps(raw), encoding="utf-8")
    with pytest.raises(ValueError, match="Unsupported"):
        recovery._load(plan)
    plan.write_bytes(original)
    monkeypatch.setattr(recovery, "_code", lambda: {"changed": "0" * 64})
    with pytest.raises(ValueError, match="code changed"):
        recovery._load(plan)
    monkeypatch.setattr(recovery, "_code", lambda: {})
    recovery_request.write_text("{}", encoding="utf-8")
    with pytest.raises(ValueError, match="input changed"):
        recovery._load(plan)


def test_owner_leases_are_required(recovery_request: Path, monkeypatch: MonkeyPatch) -> None:
    plan = recovery.prepare(recovery_request, recovery_request.parent / "new")

    @contextmanager
    def locked(path: Path, name: str) -> Iterator[bool]:
        yield False

    monkeypatch.setattr(recovery, "execution_lock", locked)
    with pytest.raises(ValueError, match="lane already active"):
        recovery.phase(plan, 0)

    @contextmanager
    def original_locked(path: Path, name: str) -> Iterator[bool]:
        yield name.startswith("recovery-lane")

    monkeypatch.setattr(recovery, "execution_lock", original_locked)
    with pytest.raises(ValueError, match="Previous generation chunk"):
        recovery.phase(plan, 0)


def test_ocr_capacity_and_source_guards(recovery_request: Path, monkeypatch: MonkeyPatch) -> None:
    plan = recovery.prepare(recovery_request, recovery_request.parent / "new")
    chunk = plan.parent / "chunk-00000/chunk.json"
    (chunk.parent / "ocr/complete.json").unlink()
    monkeypatch.setattr(shutil, "disk_usage", lambda path: Usage(100, 100, 0))
    with pytest.raises(ValueError, match="concurrency capacity"):
        recovery._pending(chunk)
    monkeypatch.setattr(shutil, "disk_usage", lambda path: Usage(100, 0, 10**12))
    calls: list[str] = []
    monkeypatch.setattr(
        subtitle_observations, "collect_observations", lambda *args, **kwargs: calls.append("ocr")
    )
    monkeypatch.setattr(cleanup, "build_candidate", lambda path: None)
    monkeypatch.setattr(subtitle_candidate, "render_candidate", lambda path: path)
    recovery._pending(chunk)
    assert calls == ["ocr"]
    (Path.cwd() / "source.mp4").write_bytes(b"different")
    with pytest.raises(ValueError, match="Source changed"):
        recovery._pending(chunk)


def test_cli_required_args(recovery_request: Path, monkeypatch: MonkeyPatch) -> None:
    assert recovery.main(["prepare", str(recovery_request)]) == 1
    assert recovery.main(["run", str(recovery_request)]) == 1
    output = recovery_request.parent / "cli"
    assert recovery.main(["prepare", str(recovery_request), "--output", str(output)]) == 0
    monkeypatch.setattr(subtitle_candidate, "render_candidate", render_fixture)
    assert recovery.main(["run", str(output / "generation.json"), "--index", "0"]) == 0
    assert recovery.main(["verify", str(output / "generation.json"), "--index", "0"]) == 0


def test_original_checkpoint_binding(recovery_request: Path, monkeypatch: MonkeyPatch) -> None:
    legacy = old._read(recovery_request.parent / "legacy.json", {})
    row = old._object(old._list(legacy["chunks"])[0])
    # Restore the real checkpoint verifier, independent of the fixture's fake owner.
    monkeypatch.undo()
    monkeypatch.setattr(recovery, "load_stages", lambda path: ())
    with pytest.raises(ValueError, match="Missing previous"):
        recovery._prior_complete(legacy, row)
    row["lane"] = -1
    with pytest.raises(ValueError, match="Invalid previous lane"):
        recovery._prior_complete(legacy, row)


def test_new_assembly_uses_selected_results(
    recovery_request: Path, monkeypatch: MonkeyPatch
) -> None:
    monkeypatch.setattr(subtitle_candidate, "render_candidate", render_fixture)
    plan = recovery.prepare(recovery_request, recovery_request.parent / "new")
    recovery.phase(plan, 0)
    monkeypatch.setattr(old, "_complete", lambda *args, **kwargs: None)
    monkeypatch.setattr(assembly, "_bytes", lambda path, **kwargs: bounded_read(path))
    # Use the real canonical assembly row verifier on the new chunk, with a
    # synthetic whole clock equal to this single bounded chunk.
    selected = plan.parent / "chunk-00000/chunk.json"
    data, settings, clock = cleanup._chunk(selected)
    monkeypatch.setattr(recovery, "clock_chunks", lambda *args, **kwargs: (clock,))
    manifest = recovery.assemble(plan, plan.parent / "assembly")
    value = json.loads(manifest.read_text())
    assert len(value["chunks"]) == 1
    assert json.loads((manifest.parent / "lineage.json").read_text())["deliveryComplete"] is False
    assert data["maximumActiveChunks"] == 1
    assert settings["version"] == 1
    monkeypatch.setattr(recovery, "clock_chunks", lambda *args, **kwargs: ())
    with pytest.raises(ValueError, match="whole original clock"):
        recovery.assemble(plan, plan.parent / "bad-assembly")


@pytest.mark.parametrize(
    "field,value", [("lane", 8), ("mode", "fake"), ("index", 2), ("workers", 0)]
)
def test_generation_selection_mutations(recovery_request: Path, field: str, value: object) -> None:
    plan = recovery.prepare(recovery_request, recovery_request.parent / "new")
    raw = json.loads(plan.read_text())
    if field == "workers":
        raw[field] = value
    else:
        raw["chunks"][0][field] = value
    plan.write_text(json.dumps(raw), encoding="utf-8")
    with pytest.raises(ValueError, match="Recovery"):
        recovery._load(plan)


@pytest.mark.parametrize("field", ["clock", "maximumActiveChunks", "ocrDirectory"])
def test_changed_pending_config_even_with_new_pin(recovery_request: Path, field: str) -> None:
    plan = recovery.prepare(recovery_request, recovery_request.parent / "new")
    raw = json.loads(plan.read_text())
    chunk = Path(raw["chunks"][0]["chunk"])
    selected = json.loads(chunk.read_text())
    selected[field] = 7
    chunk.write_text(json.dumps(selected), encoding="utf-8")
    raw["inputs"][str(chunk)] = fingerprint(chunk)
    plan.write_text(json.dumps(raw), encoding="utf-8")
    with pytest.raises(ValueError, match="Recovery"):
        recovery._load(plan)


def test_retained_media_mutation(recovery_request: Path, chunk: Path) -> None:
    cleanup.build_candidate(chunk)
    render_fixture(chunk.parent / "candidate-config.json")
    plan = recovery.prepare(recovery_request, recovery_request.parent / "retained")
    (chunk.parent / "repair/candidate.mp4").write_bytes(b"changed")
    with pytest.raises(ValueError, match="repair bytes changed"):
        recovery.phase(plan, 0)


def test_completed_ocr_requires_genuine_checkpoint(
    recovery_request: Path, monkeypatch: MonkeyPatch
) -> None:
    def incomplete(*args: object, **kwargs: object) -> None:
        raise ValueError("Incomplete OCR Stage")

    monkeypatch.setattr(recovery, "_prior_complete", incomplete)
    with pytest.raises(ValueError, match="Incomplete OCR Stage"):
        recovery.prepare(recovery_request, recovery_request.parent / "new")


def test_prior_ocr_and_complete_stage_selection(
    recovery_request: Path, monkeypatch: MonkeyPatch
) -> None:
    legacy = old._read(recovery_request.parent / "legacy.json", {})
    row = old._object(old._list(legacy["chunks"])[0])
    monkeypatch.undo()
    with pytest.raises(ValueError, match="completed OCR Stage"):
        recovery._prior_complete(legacy, row, ocr_only=True)
    row["successorStages"] = ["old-ocr"]
    stages = parse_stages(
        {
            "version": 1,
            "stages": [
                {
                    "name": "old-ocr",
                    "timeout_seconds": 10,
                    "command": ["true"],
                    "inputs": [str(recovery_request)],
                    "outputs": [str(recovery_request.parent / "synthetic-output.json")],
                    "verify_command": ["true"],
                }
            ],
        }
    )
    monkeypatch.setattr(recovery, "load_stages", lambda path: stages)
    checked: list[str] = []
    monkeypatch.setattr(old, "_complete", lambda stage, state: checked.append(stage.name))
    recovery._prior_complete(legacy, row, ocr_only=True)
    assert checked == ["old-ocr"]


def test_preparation_cache_scope_is_declared_and_existing(
    recovery_request: Path, chunk: Path, monkeypatch: MonkeyPatch
) -> None:
    legacy_path = recovery_request.parent / "legacy.json"
    legacy = json.loads(legacy_path.read_text())
    legacy["chunks"][0]["successorStages"] = ["old-ocr"]
    reference = document(legacy_path, legacy)
    raw = json.loads(recovery_request.read_text())
    raw["successor"] = reference
    recovery_request.write_text(json.dumps(raw), encoding="utf-8")
    stages = parse_stages(
        {
            "version": 1,
            "stages": [
                {
                    "name": "old-ocr",
                    "command": ["true"],
                    "verify_command": ["true"],
                    "timeout_seconds": 10,
                    "inputs": [str(chunk)],
                    "outputs": [str(chunk.parent / "ocr/complete.json")],
                }
            ],
        }
    )
    monkeypatch.setattr(recovery, "load_stages", lambda path: stages)
    paths = PREPARATION_PATHS(recovery_request)
    assert chunk in paths
    assert chunk.parent / "ocr/complete.json" in paths
    cleanup.build_candidate(chunk)
    render_fixture(chunk.parent / "candidate-config.json")
    assert chunk in PREPARATION_PATHS(recovery_request)
    (chunk.parent / "repair/candidate.mp4").unlink()
    (chunk.parent / "ocr/complete.json").unlink()
    assert chunk not in PREPARATION_PATHS(recovery_request)


@pytest.fixture
def terminal_request(recovery_request: Path, monkeypatch: MonkeyPatch) -> Path:
    legacy_path = recovery_request.parent / "legacy.json"
    legacy = json.loads(legacy_path.read_text())
    workflow = Path(legacy["workflows"][0])
    document(workflow, {"version": 1, "stages": []})
    Path(legacy["states"][0]).mkdir()
    status, log = (
        recovery_request.parent / "terminal.status",
        recovery_request.parent / "terminal.log",
    )
    status.write_text("1\n", encoding="utf-8")
    log.write_text("Original failed job; preserved.\n", encoding="utf-8")
    request_path = recovery_request.parent / "terminal-request.json"
    document(
        request_path,
        {
            "version": 1,
            "kind": "terminal-generation-observation-request",
            "successor": old._pin(legacy_path),
            "maximumAgeSeconds": 3600,
            "oldWorkersDisabled": True,
            "terminals": [
                {
                    "workflow": old._pin(workflow),
                    "state": legacy["states"][0],
                    "status": old._pin(status),
                    "log": old._pin(log),
                }
            ],
        },
    )
    monkeypatch.setattr(recovery, "_process_absence", lambda paths: None)
    return request_path


def test_terminal_observation_no_fabricated_birth(terminal_request: Path) -> None:
    output = terminal_request.parent / "owner-observed.json"
    assert recovery.main(["observe-owner", str(terminal_request), "--output", str(output)]) == 0
    proof = json.loads(output.read_text())
    assert proof["historicalActorBirthRecorded"] is False
    assert proof["matchingProcesses"] == []
    assert "actors" not in proof
    states = tuple(Path(name) for name in proof["states"])
    recovery._owner_guard(proof, states, fresh=True)
    recovery._owner_guard(proof, states, fresh=False)
    assert (terminal_request.parent / "terminal.status").read_text() == "1\n"
    with pytest.raises(FileExistsError):
        recovery.observe_owner(terminal_request, output)


@pytest.mark.parametrize("mutation", ["version", "disabled", "age", "coverage", "state", "status"])
def test_terminal_owner_bad_contract(terminal_request: Path, mutation: str) -> None:
    raw = json.loads(terminal_request.read_text())
    if mutation == "version":
        raw["version"] = 3
    elif mutation == "disabled":
        raw["oldWorkersDisabled"] = False
    elif mutation == "age":
        raw["maximumAgeSeconds"] = 0
    elif mutation == "coverage":
        raw["terminals"] = []
    elif mutation == "state":
        raw["terminals"][0]["state"] = str(terminal_request.parent)
    else:
        path = terminal_request.parent / "terminal.status"
        path.write_text("256\n", encoding="utf-8")
        raw["terminals"][0]["status"] = old._pin(path)
    terminal_request.write_text(json.dumps(raw), encoding="utf-8")
    with pytest.raises(ValueError):
        recovery._terminal_observation(terminal_request)


def test_terminal_owner_expiry_mutation_and_active_guard(
    terminal_request: Path, monkeypatch: MonkeyPatch
) -> None:
    output = terminal_request.parent / "owner-fresh.json"
    recovery.observe_owner(terminal_request, output)
    proof = json.loads(output.read_text())
    states = tuple(Path(name) for name in proof["states"])
    proof["observedAtNs"] = 0
    with pytest.raises(ValueError, match="expired"):
        recovery._owner_guard(proof, states, fresh=True)
    proof["observedAtNs"] = time.time_ns()
    proof["maximumAgeSeconds"] = "bad"
    with pytest.raises(ValueError, match="owner clock"):
        recovery._owner_guard(proof, states, fresh=True)
    proof = json.loads(output.read_text())
    proof["historicalActorBirthRecorded"] = True
    with pytest.raises(ValueError, match="evidence changed"):
        recovery._owner_guard(proof, states, fresh=False)
    (terminal_request.parent / "terminal.log").write_text("changed", encoding="utf-8")
    with pytest.raises(ValueError, match="Terminal reference changed"):
        recovery._terminal_observation(terminal_request)


@pytest.mark.parametrize(
    "code,stdout,stderr", [(0, b"123\n", b""), (2, b"", b"error"), (1, b"", b"error")]
)
def test_scoped_process_negative_must_be_certain(
    monkeypatch: MonkeyPatch, tmp_path: Path, code: int, stdout: bytes, stderr: bytes
) -> None:
    monkeypatch.setattr(
        subprocess,
        "run",
        lambda *args, **kwargs: subprocess.CompletedProcess([], code, stdout, stderr),
    )
    with pytest.raises(ValueError, match="absence"):
        PROCESS_ABSENCE((tmp_path,))


def test_scoped_process_absence_and_empty_scope(monkeypatch: MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.setattr(
        subprocess, "run", lambda *args, **kwargs: subprocess.CompletedProcess([], 1, b"", b"")
    )
    PROCESS_ABSENCE((tmp_path,))
    with pytest.raises(ValueError, match="bounded process"):
        PROCESS_ABSENCE(())


def test_graph_segments_preserve_exact_order_and_bound(monkeypatch: MonkeyPatch) -> None:
    items: list[dict[str, object]] = [
        {"name": str(index), "payload": "x" * 40} for index in range(5)
    ]
    monkeypatch.setattr(recovery, "MAX_PLAN_BYTES", 180)
    segments = recovery._graph_segments(items)
    assert len(segments) > 1
    assert [item for segment in segments for item in segment] == items
    assert all(len(encode({"version": 1, "stages": segment})) <= 180 for segment in segments)
    assert recovery._graph_segments([]) == []
    with pytest.raises(ValueError, match="Single recovery stage"):
        recovery._graph_segments([{"payload": "x" * 180}])


def test_parent_segment_verifier_preserves_child_domain_checks(
    recovery_request: Path, monkeypatch: MonkeyPatch
) -> None:
    plan = recovery.prepare(recovery_request, recovery_request.parent / "segments")
    checks: list[int] = []
    monkeypatch.setattr(old, "_complete", lambda *args, **kwargs: None)
    monkeypatch.setattr(recovery, "phase", lambda plan, index, **kwargs: checks.append(index))
    recovery.verify_segment(plan, 0)
    assert checks == [0]
    assert recovery.main(["verify-segment", str(plan), "--index", "0"]) == 0
    with pytest.raises(ValueError, match="Unknown recovery segment"):
        recovery.verify_segment(plan, 1)
    parent = json.loads((plan.parent / "lane-00.json").read_text())
    assert parent["stages"][0]["command"][3] == "replay.workflow"
    assert parent["stages"][0]["verify_command"][4] == "verify-segment"
    raw = json.loads(plan.read_text())
    child = Path(raw["workflows"][0])
    body = json.loads(child.read_text())
    body["stages"][0]["name"] = "invented"
    child.write_text(json.dumps(body))
    with pytest.raises(ValueError, match="Invalid recovery segment stage"):
        recovery.verify_segment(plan, 0)


def test_reviewed_retained_self_pin_is_readonly(tmp_path: Path) -> None:
    code = tmp_path / "cleanup_generation_recovery.py"
    code.write_bytes(b"current reader")
    current = {str(code): fingerprint(code)}
    recorded = {str(code): recovery.RETAINED_RECOVERY_SHA}
    recovery._retained_code(recorded, current)
    with pytest.raises(ValueError, match="Unreviewed"):
        recovery._retained_code({str(code): "0" * 64}, current)
    code.write_bytes(b"changed during reading")
    with pytest.raises(ValueError, match="Recorded code changed"):
        recovery._retained_code(recorded, current)
    assert not recovery._retained_digest(str(code), recovery.RETAINED_RECOVERY_SHA, "0" * 64)
    actual = str(Path(recovery.__file__).resolve())
    assert recovery._retained_digest(actual, recovery.RETAINED_RECOVERY_SHA, "0" * 64)


def test_retained_receipts_keep_data_and_output_pins(tmp_path: Path) -> None:
    from replay.stages import Stage

    source, output = tmp_path / "model", tmp_path / "result"
    source.write_bytes(b"model bytes")
    output.write_bytes(b"completed bytes")
    code = Path(recovery.__file__).resolve()
    stage = Stage(
        name="retained",
        command=("never-executed",),
        inputs=(source, code),
        outputs=(output,),
        timeout_seconds=1,
        verify_command=("domain-verifier",),
    )
    pins = {str(source): fingerprint(source), str(code): recovery.RETAINED_RECOVERY_SHA}
    receipt = {
        "status": "complete",
        "fingerprint": old._historical_fingerprint(stage, pins),
        "outputs": {str(output): fingerprint(output)},
    }
    document(tmp_path / "retained.json", receipt)
    recovery.retained_complete(stage, tmp_path, {str(code): recovery.RETAINED_RECOVERY_SHA})
    source.write_bytes(b"changed model")
    with pytest.raises(ValueError, match="complete Stage"):
        recovery.retained_complete(stage, tmp_path, {str(code): recovery.RETAINED_RECOVERY_SHA})
    with pytest.raises(ValueError, match="Unreviewed retained Stage input"):
        recovery.retained_complete(stage, tmp_path, {str(source): pins[str(source)]})


def test_retained_load_never_authorizes_execution(
    recovery_request: Path, monkeypatch: MonkeyPatch
) -> None:
    plan = recovery.prepare(recovery_request, recovery_request.parent / "readonly")
    raw = json.loads(plan.read_text())
    raw["readerCode"] = {"unknown": "0" * 64}
    plan.write_text(json.dumps(raw))
    with pytest.raises(ValueError, match="Unreviewed"):
        recovery._load(plan, retained=True)
    with pytest.raises(ValueError, match="code changed"):
        recovery.phase(plan, 0)
    with pytest.raises(ValueError, match="code changed"):
        recovery.assemble(plan, plan.parent / "must-not-exist")
    assert not (plan.parent / "must-not-exist").exists()


def test_retained_successor_reader_and_leases(tmp_path: Path, monkeypatch: MonkeyPatch) -> None:
    chunk = tmp_path / "chunk.json"
    chunk.write_bytes(b"chunk")
    (tmp_path / "times.json").write_bytes(b"times")
    settings = tmp_path / "settings.json"
    settings.write_bytes(b"{}")
    data: dict[str, object] = {
        "version": 1,
        "readerCode": {},
        "inputs": {str(chunk): fingerprint(chunk)},
        "oldStates": [],
        "handoff": str(settings),
        "settings": str(settings),
        "chunks": [
            {
                "chunk": str(chunk),
                "original": str(chunk),
                "prefix": [],
                "mode": "complete-original",
                "oldStages": [],
            }
        ],
    }
    plan = tmp_path / "successor.json"
    document(plan, data)
    monkeypatch.setattr(old, "_code", lambda: {})
    monkeypatch.setattr(old, "_path", lambda value, **kwargs: Path(str(value)))
    monkeypatch.setattr(old, "_selected_inputs", lambda *args: {str(chunk)})
    monkeypatch.setattr(old, "_quiescent", lambda *args, **kwargs: None)
    monkeypatch.setattr(old, "_old_stages", lambda *args: ())
    monkeypatch.setattr(
        cleanup,
        "_settings",
        lambda path: {
            field: {"path": str(settings)}
            for field in (
                "source",
                "native",
                "decoder",
                "model",
                "sourceClock",
                "classificationEvidence",
                "profileInputs",
                "profileComplete",
            )
        },
    )
    with recovery._retained_successor_owned(plan, 0):
        pass
    from replay.core import ReplayError

    with (
        pytest.raises(ReplayError, match="Input changed"),
        recovery._retained_successor_owned(plan, 0),
    ):
        chunk.write_bytes(b"tampered")
    raw = json.loads(plan.read_text())
    raw["version"] = 2
    plan.write_text(json.dumps(raw))
    with pytest.raises(ValueError, match="Unsupported retained successor"):
        recovery._retained_successor(plan, 0)


def test_fresh_owner_observation_is_explicit_and_bound(
    terminal_request: Path, monkeypatch: MonkeyPatch
) -> None:
    monkeypatch.setattr(recovery, "_code", lambda: {})
    observation = recovery._terminal_observation(terminal_request)
    observation["observedAtNs"] = time.time_ns()
    replacement = terminal_request.parent / "replacement-owner.json"
    document(replacement, observation)
    saved = json.loads(json.dumps(observation))
    for values in saved["identities"].values():
        values[0] += 1
    states = tuple(Path(name) for name in saved["states"])
    with pytest.raises(ValueError, match="owner evidence changed"):
        recovery._owner_guard(saved, states, fresh=False, retained=True)
    recovery._owner_guard(saved, states, fresh=False, retained=True, owner_observation=replacement)
    invalid = dict(observation)
    invalid["maximumAgeSeconds"] = 1
    document(replacement, invalid)
    with pytest.raises(ValueError, match="owner evidence changed"):
        recovery._owner_guard(
            saved, states, fresh=False, retained=True, owner_observation=replacement
        )
    with pytest.raises(ValueError, match="verification only"):
        recovery.phase(terminal_request, 0, owner_observation=replacement)
