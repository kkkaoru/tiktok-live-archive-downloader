import hashlib
import json
import os
from dataclasses import dataclass
from pathlib import Path

import pytest

from replay import replay_clock_adoption as adoption
from replay.core import ReplayError
from replay.recordings import record
from replay.stages import run_stage
from replay.workflow import load_stages, stage_graph
from replay.workflow import main as workflow_main


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _json(path: Path, value: object) -> None:
    path.write_text(json.dumps(value), encoding="utf-8")


def _ref(path: Path) -> dict[str, str]:
    return {"path": str(path), "sha256": _sha(path)}


def _tool(path: Path) -> dict[str, str]:
    path.write_bytes(b"synthetic executable")
    path.chmod(0o700)
    return _ref(path)


@dataclass(frozen=True)
class Fixture:
    root: Path
    work: Path
    config: Path
    source: Path
    code: Path

    def data(self) -> dict[str, object]:
        return record(json.loads(self.config.read_bytes()))

    def prepare(self) -> adoption.Prepared:
        return adoption.prepare(self.config, self.work)

    def revise(self, role: str, value: object) -> None:
        data = self.data()
        path = Path(str(record(data[role])["path"]))
        _json(path, value)
        data[role] = _ref(path)
        _json(self.config, data)


@pytest.fixture
def fixture(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Fixture:
    root = tmp_path / "checkout"
    evidence = root / "edits" / "work" / "original"
    evidence.mkdir(parents=True)
    code = root / "src" / "replay" / "synthetic.py"
    code.parent.mkdir(parents=True)
    code.write_bytes(b"# current and after-dispatch synthetic producer\n")
    monkeypatch.setattr(adoption, "ROOT", root)
    monkeypatch.setattr(adoption, "_code", lambda modules: (adoption.Pin(code, _sha(code)),))
    source = evidence / "source.mp4"
    source.write_bytes(b"synthetic source, never decoded")
    tool_root = root / "tools"
    tool_root.mkdir()
    tools = {
        "python": _tool(tool_root / "python"),
        "ffprobe": _tool(tool_root / "ffprobe"),
        "env": _tool(tool_root / "env"),
    }
    video = evidence / "source-video-clock.json"
    audio = evidence / "source-audio-clock.json"
    _json(
        video,
        {
            "streams": [{"time_base": "1/10"}],
            "frames": [
                {"pts": 0, "duration": 3},
                {"pts": 3, "duration": 6},
                {"pts": 9, "duration": 2},
            ],
        },
    )
    _json(
        audio,
        {
            "streams": [{"time_base": "1/48000", "sample_rate": "48000"}],
            "frames": [
                {"pts": 17, "nb_samples": 1024},
                {"pts": 1025, "nb_samples": 1024},
                {"pts": 2049, "nb_samples": 512},
            ],
        },
    )
    video_status, audio_status = evidence / "video.status", evidence / "audio.status"
    video_status.write_bytes(b"0\n")
    audio_status.write_bytes(b"0\n")
    video_log, audio_log = evidence / "video.log", evidence / "audio.log"
    video_log.write_bytes(b"")
    audio_log.write_bytes(b"")
    phases = evidence / "phases.jsonl"
    phases.write_text(
        '{"phase":"decoded-video","start":100,"end":105,"seconds":5,"exit":0}\n'
        '{"phase":"decoded-audio","start":100.1,"end":100.6,"seconds":0.5,"exit":0}\n',
        encoding="utf-8",
    )
    dispatch = evidence / "dispatch.json"
    _json(
        dispatch,
        {
            "source": str(source),
            "producer": "replay.delivery_media",
            "initialTransitiveCodeSnapshotCaptured": False,
            "sourceDigestCapturedAtDispatch": False,
            "jobs": {"video": "video-direct", "audio": "audio-direct"},
        },
    )
    outcome = evidence / "outcome.json"
    _json(
        outcome,
        {
            "video": {
                "job": "video-direct",
                "exit": 0,
                "seconds": 5,
                "sha256": _sha(video),
                "frameCount": 3,
                "timeBase": "1/10",
                "firstPTS": 0,
                "lastPTS": 9,
                "lastDuration": 2,
            },
            "audio": {
                "job": "audio-direct",
                "exit": 0,
                "seconds": 0.5,
                "sha256": _sha(audio),
                "frameCount": 3,
                "sampleRate": 48000,
                "timeBase": "1/48000",
                "firstPTS": 17,
                "lastPTS": 2049,
                "decodedSamples": 2560,
            },
            "initialCodeSnapshotBeforeDispatch": False,
            "domainChecksPerformedByOriginalProducer": True,
            "afterDispatchToTerminalSourceClosureEqual": True,
            "sourceSHA256": _sha(source),
            "clockOffsetsModified": False,
            "synchronizationApproved": False,
            "deliveryComplete": False,
        },
    )
    after, terminal = evidence / "after.json", evidence / "terminal.json"
    _json(after, {"inputs": {str(code): _sha(code)}})
    _json(terminal, {"inputs": {str(code): _sha(code)}})
    info = source.stat()
    identity = {
        "dev": info.st_dev,
        "inode": info.st_ino,
        "bytes": info.st_size,
        "mtimeNs": info.st_mtime_ns,
        "ctimeNs": info.st_ctime_ns,
        "nlink": info.st_nlink,
    }
    identity_evidence = evidence / "source-identity.json"
    _json(identity_evidence, {"source": {"path": str(source), **identity}})
    digest_evidence = evidence / "source-digest.log"
    digest_evidence.write_text(f"{_sha(source)}  {source}\n", encoding="utf-8")
    config = evidence / "adoption.json"
    _json(
        config,
        {
            "version": 1,
            "protocol": "retained-clock-adoption-v1",
            "source": {**_ref(source), "identity": identity},
            "tools": tools,
            "clocks": [
                {
                    "kind": "video",
                    "output": _ref(video),
                    "status": _ref(video_status),
                    "log": _ref(video_log),
                    "job": "video-direct",
                    "phase": "decoded-video",
                },
                {
                    "kind": "audio",
                    "output": _ref(audio),
                    "status": _ref(audio_status),
                    "log": _ref(audio_log),
                    "job": "audio-direct",
                    "phase": "decoded-audio",
                },
            ],
            "phases": _ref(phases),
            "dispatch": _ref(dispatch),
            "outcome": _ref(outcome),
            "codeAfterDispatch": _ref(after),
            "codeTerminal": _ref(terminal),
            "sourceIdentityEvidence": _ref(identity_evidence),
            "sourceDigestEvidence": _ref(digest_evidence),
            "timeoutSeconds": 120,
        },
    )
    return Fixture(root, root / "edits" / "work" / "reconciled", config, source, code)


def _seal(fixture: Fixture) -> tuple[Path, Path]:
    fixture.prepare()
    plan, receipt = fixture.work / "plan.json", fixture.work / "adoption-evidence.json"
    adoption.seal(plan, receipt)
    return plan, receipt


def test_two_graphs_create_real_complete_state_without_decoding(fixture: Fixture) -> None:
    prepared = fixture.prepare()
    evidence = load_stages(prepared.evidence_workflow)[0]
    adopt = load_stages(prepared.adoption_workflow)[0]
    calls: list[str] = []

    def execute(command: tuple[str, ...], log: Path, timeout: int) -> None:
        action, plan, receipt = command[-3:]
        calls.append(action)
        log.write_bytes(b"")
        if action == "seal":
            adoption.seal(Path(plan), Path(receipt))
        else:
            adoption.verify(Path(plan), Path(receipt))

    assert stage_graph((evidence,)) == {"seal-retained-clocks": ()}
    assert stage_graph((adopt,)) == {"adopt-retained-clocks": ()}
    assert (
        run_stage(evidence, state_directory=fixture.work / "evidence-state", execute=execute)
        == "completed"
    )
    assert (
        run_stage(adopt, state_directory=fixture.work / "adoption-state", execute=execute)
        == "adopted"
    )
    assert (
        run_stage(adopt, state_directory=fixture.work / "adoption-state", execute=execute)
        == "reused"
    )
    assert calls == ["seal", "verify", "verify", "verify"]
    state = record(
        json.loads((fixture.work / "adoption-state" / "adopt-retained-clocks.json").read_bytes())
    )
    assert state["status"] == "complete"
    assert len(record(state["outputs"])) == 2
    assert fixture.source in adopt.inputs
    proof = record(json.loads((fixture.work / "adoption-evidence.json").read_bytes()))
    assert proof["currentInputBytesVerified"] is True
    assert proof["historicalGenerationBindingFullyVerified"] is False
    assert proof["historicalSourceToClockBindingVerified"] is False
    assert proof["fullSourceClockCoverageIndependentlyVerified"] is False
    assert proof["historicalInitialCodeSnapshotVerified"] is False
    assert proof["historicalSourceDigestAtDispatchVerified"] is False
    assert proof["historicalOriginalArgvRecorded"] is False
    assert proof["historicalToolPinsAtDispatchVerified"] is False
    assert proof["originalGenerationReceiptReissued"] is False
    assert proof["deliveryComplete"] is False
    assert not (fixture.work / "video-clock.producer.json").exists()


def test_generated_graphs_pass_standard_readonly_preflight(
    fixture: Fixture, capsys: pytest.CaptureFixture[str]
) -> None:
    prepared = fixture.prepare()
    cases: tuple[tuple[Path, dict[str, list[str]]], ...] = (
        (prepared.evidence_workflow, {"seal-retained-clocks": []}),
        (prepared.adoption_workflow, {"adopt-retained-clocks": []}),
    )
    for graph, expected in cases:
        assert workflow_main([str(graph), "--state", str(fixture.work / "checked"), "--check"]) == 0
        assert json.loads(capsys.readouterr().out) == expected
    assert not (fixture.work / "adoption-evidence.json").exists()


def test_contrasting_nonzero_video_origin_and_audio_support_are_preserved(fixture: Fixture) -> None:
    data = fixture.data()
    clocks = data["clocks"]
    assert isinstance(clocks, list)
    video_row, audio_row = record(clocks[0]), record(clocks[1])
    video = Path(str(record(video_row["output"])["path"]))
    audio = Path(str(record(audio_row["output"])["path"]))
    _json(
        video,
        {
            "streams": [{"time_base": "1/90000"}],
            "frames": [
                {"pts": 7, "duration": 4},
                {"pts": 11, "duration": 3},
                {"pts": 14, "duration": 6},
                {"pts": 20, "duration": 2},
            ],
        },
    )
    _json(
        audio,
        {
            "streams": [{"time_base": "1/48000", "sample_rate": "48000"}],
            "frames": [
                {"pts": 25440, "nb_samples": 2048},
                {"pts": 27488, "nb_samples": 2048},
            ],
        },
    )
    video_row["output"], audio_row["output"] = _ref(video), _ref(audio)
    data["clocks"] = [video_row, audio_row]
    outcome_path = Path(str(record(data["outcome"])["path"]))
    outcome = record(json.loads(outcome_path.read_bytes()))
    outcome["video"] = {
        "job": "video-direct",
        "exit": 0,
        "seconds": 5,
        "sha256": _sha(video),
        "frameCount": 4,
        "timeBase": "1/90000",
        "firstPTS": 7,
        "lastPTS": 20,
        "lastDuration": 2,
    }
    outcome["audio"] = {
        "job": "audio-direct",
        "exit": 0,
        "seconds": 0.5,
        "sha256": _sha(audio),
        "frameCount": 2,
        "sampleRate": 48000,
        "timeBase": "1/48000",
        "firstPTS": 25440,
        "lastPTS": 27488,
        "decodedSamples": 4096,
    }
    _json(outcome_path, outcome)
    data["outcome"] = _ref(outcome_path)
    _json(fixture.config, data)
    plan, receipt = _seal(fixture)
    original = receipt.read_bytes()
    adoption.verify(plan, receipt)
    assert receipt.read_bytes() == original
    proof = record(json.loads(original))
    video_summary = record(record(record(proof["clocks"])["video"])["summary"])
    audio_summary = record(record(record(proof["clocks"])["audio"])["summary"])
    assert video_summary == {
        "frameCount": 4,
        "timeBase": "1/90000",
        "firstPTS": 7,
        "lastPTS": 20,
        "lastDuration": 2,
    }
    assert audio_summary == {
        "frameCount": 2,
        "sampleRate": 48000,
        "timeBase": "1/48000",
        "firstPTS": 25440,
        "lastPTS": 27488,
        "decodedSamples": 4096,
    }


@pytest.mark.parametrize(
    "field,value",
    [
        ("version", True),
        ("version", 2),
        ("protocol", "generation-v1"),
        ("unknown", 1),
        ("timeoutSeconds", 0),
        ("timeoutSeconds", True),
        ("timeoutSeconds", 86401),
        ("clocks", []),
        ("clocks", {}),
        ("tools", {}),
        ("source", {}),
    ],
)
def test_invalid_configuration_refuses_before_output(
    fixture: Fixture, field: str, value: object
) -> None:
    data = fixture.data()
    data[field] = value
    _json(fixture.config, data)
    with pytest.raises((ReplayError, KeyError)):
        fixture.prepare()
    assert not fixture.work.exists()


@pytest.mark.parametrize("role", ["source", "tools", "clocks", "phases"])
def test_malformed_pin_fields_refuse(fixture: Fixture, role: str) -> None:
    data = fixture.data()
    if role == "source":
        source = record(data[role])
        source["sha256"] = "bad"
        data[role] = source
    elif role == "tools":
        tools = record(data[role])
        tools["ffprobe"] = {"path": "relative", "sha256": "0" * 64}
        data[role] = tools
    elif role == "clocks":
        clocks = data[role]
        assert isinstance(clocks, list)
        row = record(clocks[0])
        row["unknown"] = True
        data[role] = [row, clocks[1]]
    else:
        data[role] = {"path": str(fixture.source), "sha256": True}
    _json(fixture.config, data)
    with pytest.raises(ReplayError):
        fixture.prepare()


@pytest.mark.parametrize(
    "mutation",
    [
        "duplicate",
        "failed",
        "elapsed",
        "partial",
        "nan",
        "extra",
        "reversed",
        "negative",
        "overrun",
    ],
)
def test_false_or_incomplete_phase_evidence_refuses(fixture: Fixture, mutation: str) -> None:
    data = fixture.data()
    phases = Path(str(record(data["phases"])["path"]))
    if mutation == "duplicate":
        phases.write_bytes(phases.read_bytes() + phases.read_bytes().splitlines()[0] + b"\n")
    elif mutation == "partial":
        phases.write_bytes(phases.read_bytes().rstrip(b"\n"))
    else:
        row = record(json.loads(phases.read_bytes().splitlines()[0]))
        if mutation == "failed":
            row["exit"] = 1
        elif mutation == "elapsed":
            row["seconds"] = 4
        elif mutation == "nan":
            row["seconds"] = float("nan")
        elif mutation == "reversed":
            row["end"] = 99
        elif mutation == "negative":
            row["seconds"] = -1
        elif mutation == "overrun":
            row["seconds"] = 86401
        else:
            row["invented"] = True
        phases.write_text(
            json.dumps(row) + "\n" + phases.read_text(encoding="utf-8").splitlines()[1] + "\n",
            encoding="utf-8",
        )
    data["phases"] = _ref(phases)
    _json(fixture.config, data)
    fixture.prepare()
    with pytest.raises(ReplayError):
        adoption.seal(fixture.work / "plan.json", fixture.work / "adoption-evidence.json")
    assert not (fixture.work / "adoption-evidence.json").exists()


@pytest.mark.parametrize("wall_end", [105.005, 104.995, 108])
def test_diagnostic_epochs_do_not_replace_monotonic_elapsed(
    fixture: Fixture, wall_end: float
) -> None:
    data = fixture.data()
    phases = Path(str(record(data["phases"])["path"]))
    rows = [record(json.loads(line)) for line in phases.read_text(encoding="utf-8").splitlines()]
    rows[0]["end"] = wall_end
    phases.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")
    data["phases"] = _ref(phases)
    _json(fixture.config, data)
    fixture.prepare()
    receipt = fixture.work / "adoption-evidence.json"
    adoption.seal(fixture.work / "plan.json", receipt)
    adoption.verify(fixture.work / "plan.json", receipt)
    adopted = record(json.loads(receipt.read_bytes()))
    video = record(record(adopted["clocks"])["video"])
    assert record(video["phase"]) == rows[0]
    assert record(video["phase"])["seconds"] == 5
    assert adopted["historicalGenerationBindingFullyVerified"] is False
    assert adopted["clockOffsetsModified"] is False


def test_false_elapsed_is_not_validated_by_matching_diagnostic_epoch(fixture: Fixture) -> None:
    data = fixture.data()
    phases = Path(str(record(data["phases"])["path"]))
    rows = [record(json.loads(line)) for line in phases.read_text(encoding="utf-8").splitlines()]
    rows[0].update(end=104, seconds=4)
    phases.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")
    data["phases"] = _ref(phases)
    _json(fixture.config, data)
    fixture.prepare()
    with pytest.raises(ReplayError, match="Original clock outcome differs"):
        adoption.seal(fixture.work / "plan.json", fixture.work / "adoption-evidence.json")


@pytest.mark.parametrize(
    "kind,field,value",
    [
        ("video", "pts", 0),
        ("video", "duration", -1),
        ("video", "duration", True),
        ("audio", "pts", 17),
        ("audio", "nb_samples", 0),
        ("audio", "nb_samples", True),
    ],
)
def test_invalid_retained_clock_is_rejected_by_real_domain(
    fixture: Fixture, kind: str, field: str, value: object
) -> None:
    data = fixture.data()
    clocks = data["clocks"]
    assert isinstance(clocks, list)
    index = 0 if kind == "video" else 1
    row = record(clocks[index])
    output = Path(str(record(row["output"])["path"]))
    clock = record(json.loads(output.read_bytes()))
    frames = clock["frames"]
    assert isinstance(frames, list)
    frame = record(frames[1])
    frame[field] = value
    frames[1] = frame
    _json(output, clock)
    row["output"] = _ref(output)
    clocks[index] = row
    _json(fixture.config, data)
    fixture.prepare()
    with pytest.raises((ReplayError, ValueError)):
        adoption.seal(fixture.work / "plan.json", fixture.work / "adoption-evidence.json")


@pytest.mark.parametrize(
    "role,field,value",
    [
        ("dispatch", "initialTransitiveCodeSnapshotCaptured", True),
        ("dispatch", "sourceDigestCapturedAtDispatch", True),
        ("dispatch", "producer", "unknown"),
        ("dispatch", "source", "other"),
        ("outcome", "initialCodeSnapshotBeforeDispatch", True),
        ("outcome", "clockOffsetsModified", True),
        ("outcome", "deliveryComplete", True),
        ("outcome", "sourceSHA256", "0" * 64),
        ("outcome", "audio", {}),
    ],
)
def test_historical_limitations_and_original_roles_cannot_be_promoted(
    fixture: Fixture, role: str, field: str, value: object
) -> None:
    path = Path(str(record(fixture.data()[role])["path"]))
    evidence = record(json.loads(path.read_bytes()))
    evidence[field] = value
    fixture.revise(role, evidence)
    fixture.prepare()
    with pytest.raises(ReplayError):
        adoption.seal(fixture.work / "plan.json", fixture.work / "adoption-evidence.json")


@pytest.mark.parametrize("fault", ["source", "code", "output", "status", "log", "tool", "missing"])
def test_changed_bytes_and_missing_outputs_never_regenerate(fixture: Fixture, fault: str) -> None:
    plan, receipt = _seal(fixture)
    data = fixture.data()
    clocks = data["clocks"]
    assert isinstance(clocks, list)
    video = record(clocks[0])
    if fault == "source":
        fixture.source.write_bytes(b"changed source")
    elif fault == "code":
        fixture.code.write_bytes(b"# reader changed")
    elif fault == "tool":
        tool = record(record(data["tools"])["ffprobe"])
        Path(str(tool["path"])).write_bytes(b"changed executable")
    else:
        field = "output" if fault == "missing" else fault
        path = Path(str(record(video[field])["path"]))
        if fault == "missing":
            path.unlink()
        else:
            path.write_bytes(b"changed artifact")
    previous = receipt.read_bytes()
    with pytest.raises((ReplayError, OSError)):
        adoption.verify(plan, receipt)
    assert receipt.read_bytes() == previous


def test_mutation_during_domain_validation_fails_terminal_bytes(
    fixture: Fixture, monkeypatch: pytest.MonkeyPatch
) -> None:
    fixture.prepare()
    original = adoption._domains

    def mutate(job: adoption.Job) -> tuple[dict[str, object], dict[str, str]]:
        result = original(job)
        fixture.source.write_bytes(b"mutated during validation")
        return result

    monkeypatch.setattr(adoption, "_domains", mutate)
    with pytest.raises(ReplayError, match="bytes changed"):
        adoption.seal(fixture.work / "plan.json", fixture.work / "adoption-evidence.json")
    assert not (fixture.work / "adoption-evidence.json").exists()


def test_plan_mutation_after_parse_is_not_sealed(
    fixture: Fixture, monkeypatch: pytest.MonkeyPatch
) -> None:
    fixture.prepare()
    original = adoption._load

    def mutate(plan: Path) -> tuple[adoption.Job, tuple[adoption.Pin, ...]]:
        result = original(plan)
        plan.write_bytes(plan.read_bytes() + b" ")
        return result

    monkeypatch.setattr(adoption, "_load", mutate)
    with pytest.raises(ReplayError, match="bytes changed"):
        adoption.seal(fixture.work / "plan.json", fixture.work / "adoption-evidence.json")


def test_same_bytes_replaced_source_fails_original_identity(fixture: Fixture) -> None:
    fixture.prepare()
    replacement = fixture.source.with_name("replacement.mp4")
    replacement.write_bytes(fixture.source.read_bytes())
    replacement.replace(fixture.source)
    with pytest.raises(ReplayError, match="source identity changed"):
        adoption.seal(fixture.work / "plan.json", fixture.work / "adoption-evidence.json")


@pytest.mark.parametrize("fault", ["symlink", "hardlink", "outside", "directory", "oversized"])
def test_unsafe_metadata_rejected_before_publication(fixture: Fixture, fault: str) -> None:
    if fault == "outside":
        with pytest.raises(ReplayError, match="escapes"):
            adoption.prepare(fixture.config, fixture.root / "outside")
        return
    if fault == "directory":
        fixture.config.unlink()
        fixture.config.mkdir()
    elif fault == "oversized":
        fixture.config.write_bytes(b" " * (adoption.MAX_METADATA_BYTES + 1))
    elif fault == "hardlink":
        os.link(fixture.config, fixture.config.with_name("alias.json"))
    else:
        target = fixture.config.with_name("target.json")
        fixture.config.rename(target)
        fixture.config.symlink_to(target)
    with pytest.raises(ReplayError):
        fixture.prepare()
    assert not fixture.work.exists()


def test_duplicate_keys_and_occupied_outputs_are_not_overwritten(fixture: Fixture) -> None:
    fixture.prepare()
    with pytest.raises(ReplayError, match="new reconciliation"):
        fixture.prepare()
    plan, receipt = fixture.work / "plan.json", fixture.work / "adoption-evidence.json"
    adoption.seal(plan, receipt)
    before = receipt.read_bytes()
    with pytest.raises(ReplayError, match="absent evidence"):
        adoption.seal(plan, receipt)
    assert receipt.read_bytes() == before
    receipt.write_bytes(b'{"version":1,"version":1}')
    with pytest.raises(ReplayError, match="Duplicate"):
        adoption.verify(plan, receipt)


@pytest.mark.parametrize("fault", ["extra", "missing", "duplicate", "hash", "protocol"])
def test_plan_membership_is_exact(fixture: Fixture, fault: str) -> None:
    fixture.prepare()
    plan = fixture.work / "plan.json"
    raw = record(json.loads(plan.read_bytes()))
    inputs = raw["inputs"]
    assert isinstance(inputs, list)
    if fault == "extra":
        other = fixture.config.with_name("unrelated.json")
        other.write_bytes(b"{}")
        inputs.append(_ref(other))
    elif fault == "missing":
        inputs.pop()
    elif fault == "duplicate":
        inputs.append(inputs[0])
    elif fault == "hash":
        raw["config"] = {"path": str(fixture.config), "sha256": "0" * 64}
    else:
        raw["protocol"] = "original-generation"
    _json(plan, raw)
    with pytest.raises(ReplayError):
        adoption.seal(plan, fixture.work / "adoption-evidence.json")


def test_changed_producer_closure_refuses(fixture: Fixture) -> None:
    fixture.revise("codeTerminal", {"inputs": {str(fixture.code): "0" * 64}})
    fixture.prepare()
    with pytest.raises(ReplayError, match="producer code changed"):
        adoption.seal(fixture.work / "plan.json", fixture.work / "adoption-evidence.json")


@pytest.mark.parametrize(
    "role,content", [("status", b"1\n"), ("log", b"ffprobe reported an error\n")]
)
def test_repinned_failed_process_record_still_fails_domain(
    fixture: Fixture, role: str, content: bytes
) -> None:
    data = fixture.data()
    clocks = data["clocks"]
    assert isinstance(clocks, list)
    row = record(clocks[0])
    path = Path(str(record(row[role])["path"]))
    path.write_bytes(content)
    row[role] = _ref(path)
    clocks[0] = row
    _json(fixture.config, data)
    fixture.prepare()
    with pytest.raises(ReplayError, match="Original"):
        adoption.seal(fixture.work / "plan.json", fixture.work / "adoption-evidence.json")


def test_repinned_source_digest_evidence_cannot_name_another_source(fixture: Fixture) -> None:
    data = fixture.data()
    path = Path(str(record(data["sourceDigestEvidence"])["path"]))
    path.write_text("0" * 64 + "  /different/source.mp4\n", encoding="utf-8")
    data["sourceDigestEvidence"] = _ref(path)
    _json(fixture.config, data)
    fixture.prepare()
    with pytest.raises(ReplayError, match="source digest evidence differs"):
        adoption.seal(fixture.work / "plan.json", fixture.work / "adoption-evidence.json")


def test_repinned_original_clock_summary_cannot_hide_different_pts(fixture: Fixture) -> None:
    path = Path(str(record(fixture.data()["outcome"])["path"]))
    raw = record(json.loads(path.read_bytes()))
    audio = record(raw["audio"])
    audio["firstPTS"] = 0
    raw["audio"] = audio
    fixture.revise("outcome", raw)
    fixture.prepare()
    with pytest.raises(ReplayError, match="clock outcome differs"):
        adoption.seal(fixture.work / "plan.json", fixture.work / "adoption-evidence.json")


def test_metadata_preparation_never_hashes_media(
    fixture: Fixture, monkeypatch: pytest.MonkeyPatch
) -> None:
    def refuse_digest(path: Path) -> str:
        raise RuntimeError("prepare must not hash any supplied media or tool input")

    monkeypatch.setattr(adoption, "digest", refuse_digest)
    fixture.prepare()
    assert (fixture.work / "evidence-workflow.json").exists()
    assert (fixture.work / "adoption-workflow.json").exists()
    assert not (fixture.work / "adoption-evidence.json").exists()


def test_current_tool_alias_target_change_is_not_authorized_by_equal_bytes(
    fixture: Fixture,
) -> None:
    data = fixture.data()
    tools = record(data["tools"])
    python = record(tools["python"])
    old = Path(str(python["path"]))
    invocation = old.with_name("python-alias")
    invocation.symlink_to(old)
    python["path"] = str(invocation)
    tools["python"] = python
    _json(fixture.config, data)
    fixture.prepare()
    other = old.with_name("equal-python")
    other.write_bytes(old.read_bytes())
    other.chmod(0o700)
    invocation.unlink()
    invocation.symlink_to(other)
    with pytest.raises(ReplayError, match="membership"):
        adoption.seal(fixture.work / "plan.json", fixture.work / "adoption-evidence.json")


def test_receipt_changed_during_readonly_validation_is_not_overwritten(
    fixture: Fixture, monkeypatch: pytest.MonkeyPatch
) -> None:
    plan, receipt = _seal(fixture)
    original = adoption._verified

    def mutate(path: Path) -> dict[str, object]:
        result = original(path)
        receipt.write_bytes(receipt.read_bytes() + b" ")
        return result

    monkeypatch.setattr(adoption, "_verified", mutate)
    with pytest.raises(ReplayError, match="receipt changed during"):
        adoption.verify(plan, receipt)
    assert receipt.read_bytes().endswith(b" ")


def test_receipt_tamper_fails_readonly_reentry(fixture: Fixture) -> None:
    plan, receipt = _seal(fixture)
    raw = record(json.loads(receipt.read_bytes()))
    raw["historicalGenerationBindingFullyVerified"] = True
    _json(receipt, raw)
    before = receipt.read_bytes()
    with pytest.raises(ReplayError, match="evidence changed"):
        adoption.verify(plan, receipt)
    assert receipt.read_bytes() == before


def test_public_cli_prepare_seal_verify_and_error(
    fixture: Fixture, capsys: pytest.CaptureFixture[str]
) -> None:
    assert adoption.main(["prepare", str(fixture.config), str(fixture.work)]) == 0
    plan, receipt = fixture.work / "plan.json", fixture.work / "adoption-evidence.json"
    assert adoption.main(["seal", str(plan), str(receipt)]) == 0
    assert adoption.main(["verify", str(plan), str(receipt)]) == 0
    assert adoption.main(["seal", str(plan), str(receipt)]) == 1
    assert "CLOCK_ADOPTION_ERROR" in capsys.readouterr().err


def test_static_reader_closure_contains_own_and_domain_modules() -> None:
    paths = {pin.path.name for pin in adoption._code(("replay_clock_adoption",))}
    assert {
        "replay_clock_adoption.py",
        "delivery_media.py",
        "audio_clock.py",
        "sha256_check.py",
    } <= paths
    assert "workflow.py" not in paths
    assert "fcp_head_refresh.py" not in paths
