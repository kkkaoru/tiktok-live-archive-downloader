import hashlib
import json
import subprocess
import wave
from dataclasses import dataclass
from fractions import Fraction
from pathlib import Path

import pytest

from replay import asr_assembly, isolated_cpu_asr
from replay import replay_preparation_plan as initial
from replay import replay_transcription_plan as factory
from replay.core import ReplayError
from replay.recordings import record
from replay.stages import Stage, StageError, _fingerprint, execution_lock, run_stage
from replay.workflow import parse_stages, stage_graph


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _json(path: Path, data: object) -> None:
    path.write_text(json.dumps(data), encoding="utf-8")


def _data(path: Path) -> dict[str, object]:
    return record(json.loads(path.read_bytes()))


def _pin(path: Path) -> dict[str, str]:
    return {"path": str(path), "sha256": _sha(path)}


def _wave(path: Path, channels: int = 2, rate: int = 44100, samples: int = 4410) -> None:
    with wave.open(str(path), "wb") as stream:
        stream.setparams((channels, 2, rate, 0, "NONE", "not compressed"))
        stream.writeframes(b"\0" * samples * channels * 2)


def _stage_complete(stage: Stage, state: Path) -> None:
    state.mkdir(exist_ok=True)
    _json(
        state / f"{stage.name}.json",
        {
            "status": "complete",
            "fingerprint": _fingerprint(stage),
            "outputs": {str(path): _sha(path) for path in stage.outputs},
        },
    )


@dataclass(frozen=True)
class Fixture:
    root: Path
    config: Path
    work: Path
    prepared: initial.Preparation
    model: Path
    code: Path

    def plan(self) -> Path:
        factory.prepare(self.config, self.work)
        return self.work / "transcription.json"


@pytest.fixture
def fixture(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Fixture:
    root = tmp_path / "checkout"
    work = root / "edits" / "work"
    work.mkdir(parents=True)
    (root / ".python-version").write_text("3.12.13\n", encoding="utf-8")
    (root / "pyproject.toml").write_bytes(b"[project]\n")
    (root / "uv.lock").write_bytes(b"version=1\n")
    source = root / "source.mp4"
    source.write_bytes(b"synthetic original only")
    code = root / "src/replay/batch_asr.py"
    code.parent.mkdir(parents=True)
    code.write_text("# synthetic batch\n", encoding="utf-8")
    code.with_name("asr_runtime.py").write_text("# synthetic runtime\n", encoding="utf-8")
    code.with_name("isolated_cpu_asr.py").write_text("# synthetic isolated\n", encoding="utf-8")
    code.with_name("asr_assembly.py").write_bytes(Path(asr_assembly.__file__).read_bytes())
    model = root / "edits/models/large-v3"
    model.mkdir(parents=True)
    (model / "config.json").write_bytes(b'{"synthetic":true}')
    (model / "weights.npz").write_bytes(b"synthetic nonTurbo weights")
    prepared_root = work / "preparation"
    prepared_root.mkdir()
    (prepared_root / "state").mkdir()
    _json(prepared_root / "preparation.json", {"synthetic": True})
    _json(
        prepared_root / "video-clock.json",
        {
            "streams": [{"time_base": "1/90000"}],
            "frames": [{"pts": 90000, "duration": 18000}],
        },
    )
    _json(
        prepared_root / "audio-clock.json",
        {
            "streams": [{"sample_rate": "44100", "time_base": "1/44100"}],
            "frames": [{"pts": 44100, "nb_samples": 2205}, {"pts": 49100, "nb_samples": 2205}],
        },
    )
    _json(prepared_root / "video-clock.producer.json", {"synthetic": "video"})
    _json(prepared_root / "audio-clock.producer.json", {"synthetic": "audio"})
    separation = prepared_root / "separation"
    separation.mkdir()
    _wave(separation / "vocals.wav")
    _wave(separation / "model-input.wav")
    _json(separation / "result.json", {"synthetic": "separated"})
    _json(separation / "producer.json", {"synthetic": "completed"})
    _json(separation / "before-model-import-descriptors.json", {"externalWritableRegular": []})
    _json(separation / "after-model-import-descriptors.json", {"externalWritableRegular": []})
    _json(separation / "after-separation-descriptors.json", {"externalWritableRegular": []})
    executable = root / "python"
    executable.write_bytes(b"synthetic executable")
    executable.chmod(0o700)
    tool = initial.Tool(executable, initial.Pin(executable, _sha(executable)))
    settings = initial.Settings(
        {"python": tool, "ffmpeg": tool, "env": tool}, model, "cpu", 120, 9000
    )
    prepared = initial.Preparation(
        root / "edits/replay-example",
        work / "initial-tools.json",
        prepared_root,
        settings,
        source,
        (initial.Pin(source, _sha(source)), initial.Pin(code, _sha(code))),
    )
    stages = [
        {
            "name": "source-video-clock",
            "command": ["synthetic", "video"],
            "inputs": [str(source)],
            "outputs": [
                str(prepared_root / "video-clock.json"),
                str(prepared_root / "video-clock.producer.json"),
            ],
            "timeout_seconds": 120,
        },
        {
            "name": "source-audio-clock",
            "command": ["synthetic", "audio"],
            "inputs": [str(source)],
            "outputs": [
                str(prepared_root / "audio-clock.json"),
                str(prepared_root / "audio-clock.producer.json"),
            ],
            "timeout_seconds": 120,
        },
        {
            "name": "source-voice-separation",
            "command": ["synthetic", "separation"],
            "inputs": [str(source)],
            "outputs": [
                str(separation / "vocals.wav"),
                str(separation / "model-input.wav"),
                str(separation / "result.json"),
            ],
            "timeout_seconds": 9000,
        },
    ]
    workflow: dict[str, object] = {"version": 1, "stages": stages}
    _json(prepared_root / "workflow.json", workflow)
    parsed = parse_stages(workflow)
    _stage_complete(parsed[0], prepared_root / "state")
    _stage_complete(parsed[1], prepared_root / "state")
    _stage_complete(parsed[2], prepared_root / "state")
    config = work / "transcription-settings.json"
    _json(
        config,
        {
            "version": 1,
            "preparation": _pin(prepared_root / "preparation.json"),
            "predecessorState": str(prepared_root / "state"),
            "model": str(model),
            "fps": 30,
            "timeouts": {
                "mono": 120,
                "recognize": 3600,
                "assemble": 120,
                "child": 300,
                "quiet": 300,
            },
        },
    )
    monkeypatch.setattr(factory, "ROOT", root)
    monkeypatch.setattr(initial, "ROOT", root)
    monkeypatch.setattr(factory, "DEFAULT_WEIGHTS_SHA256", _sha(model / "weights.npz"))
    monkeypatch.setattr(factory, "DEFAULT_CONFIG_SHA256", _sha(model / "config.json"))
    monkeypatch.setattr(factory, "_code_paths", lambda: (code,))
    monkeypatch.setattr(initial, "_load", lambda path: prepared)
    monkeypatch.setattr(initial, "_verify_producer", lambda plan, kind: None)
    monkeypatch.setattr(initial, "_domain", lambda plan, kind: None)
    return Fixture(root, config, work / "transcription", prepared, model, code)


def _mono(fixture: Fixture, path: Path) -> None:
    plan = factory._load(path)
    _wave(plan.mono_root / "mono.wav", 1, 16000, 1600)
    _json(factory._proof(plan, "mono"), factory._producer_data(plan, "mono"))
    stage = parse_stages(_data(plan.work / "workflow.json"))[0]
    _stage_complete(stage, plan.work / "state")


def _raw() -> dict[str, object]:
    return {
        "language": "ja",
        "text": "声",
        "segments": [
            {
                "start": 0.01,
                "end": 0.08,
                "text": "声",
                "words": [{"word": "声", "start": 0.01, "end": 0.08, "probability": 1}],
            }
        ],
    }


def _collection(plan: factory.Plan, *, complete: bool = True) -> None:
    plan.collection.mkdir(exist_ok=True)
    _json(plan.collection / "inputs.json", factory._definition(plan))
    _json(plan.collection / "000000000000.raw.json", _raw())
    digest = _sha(plan.collection / "000000000000.raw.json")
    _json(
        plan.collection / "000000000000.clock.json",
        {
            "startSample": 0,
            "endSample": 1600,
            "sampleRate": 16000,
            "humanReviewed": False,
            "emptyRecognitionIsNotSilence": True,
            "rawSHA256": digest,
        },
    )
    if complete:
        _json(
            plan.collection / "complete.json",
            {"rawSHA256": [digest], "windows": 1, "deliveryComplete": False},
        )


def _policy(plan: factory.Plan) -> None:
    policy = plan.work / "asr-policy"
    policy.mkdir()
    _json(
        policy / "started.json",
        {
            "version": 1,
            "producerSHA256": _sha(fixture_path(plan, "isolated_cpu_asr.py")),
            "processPolicy": "fresh-cpu-child-per-missing-window",
            "cachePolicy": "unchanged-provider-default",
            "inputs": factory._definition(plan)["inputs"],
            "manifestSHA256": _sha(plan.collection / "inputs.json"),
            "limitNewWindows": None,
            "childTimeoutSeconds": 300,
        },
    )
    _json(
        policy / "result.json",
        {
            "version": 1,
            "recognitionComplete": True,
            "deliveryComplete": False,
            "verifiedWindows": 1,
            "totalWindows": 1,
            "producerSHA256": _sha(fixture_path(plan, "isolated_cpu_asr.py")),
        },
    )


def fixture_path(plan: factory.Plan, name: str) -> Path:
    return plan.prepared.source.parent / "src/replay" / name


def _recognition(fixture: Fixture, path: Path) -> None:
    _mono(fixture, path)
    plan = factory._load(path)
    _collection(plan)
    _policy(plan)
    _json(factory._proof(plan, "recognize"), factory._producer_data(plan, "recognize"))


def _assembly(fixture: Fixture, path: Path) -> None:
    _recognition(fixture, path)
    factory.produce(path, "assemble")
    plan = factory._load(path)
    _stage_complete(parse_stages(_data(plan.work / "workflow.json"))[2], plan.work / "state")


def _explicit_config(fixture: Fixture) -> None:
    before = fixture.prepared.work
    _json(
        before / "separation/result.json",
        {
            "source": str(fixture.prepared.source),
            "modelInput": str(before / "separation/model-input.wav"),
            "vocals": str(before / "separation/vocals.wav"),
            "preparedInput": False,
            "timestampCorrection": False,
            "loudnessNormalization": False,
            "deliveryComplete": False,
        },
    )
    workflow = _data(before / "workflow.json")
    stages = workflow["stages"]
    assert isinstance(stages, list)
    record(stages[0])["verify_command"] = ["synthetic", "verify-video"]
    record(stages[0])["outputs"] = [str(before / "video-clock.json")]
    record(stages[1])["verify_command"] = ["synthetic", "verify-audio"]
    record(stages[1])["outputs"] = [str(before / "audio-clock.json")]
    record(stages[2])["verify_command"] = ["synthetic", "verify-separation"]
    _json(before / "workflow.json", workflow)
    parsed = parse_stages(workflow)
    _stage_complete(parsed[0], before / "state")
    _stage_complete(parsed[1], before / "state")
    _stage_complete(parsed[2], before / "state")
    python = fixture.root / ".venv/bin/python"
    python.parent.mkdir(parents=True)
    python.symlink_to(fixture.root / "python")
    python.parent.parent.joinpath("pyvenv.cfg").write_text(
        "version_info = 3.12.13\n", encoding="utf-8"
    )
    config = _data(fixture.config)
    config.pop("preparation")
    config.pop("predecessorState")
    config["tools"] = {
        "python": {"path": str(python), "sha256": _sha(fixture.root / "python")},
        "env": _pin(fixture.root / "python"),
        "ffmpeg": _pin(fixture.root / "python"),
    }
    workflow_pin = _pin(before / "workflow.json")
    config["predecessor"] = {
        "roles": {
            "source": _pin(fixture.prepared.source),
            "videoClock": _pin(before / "video-clock.json"),
            "audioClock": _pin(before / "audio-clock.json"),
            "modelInput": _pin(before / "separation/model-input.wav"),
            "vocals": _pin(before / "separation/vocals.wav"),
            "separationResult": _pin(before / "separation/result.json"),
        },
        "stages": {
            "audio": {
                "workflow": workflow_pin,
                "receipt": _pin(before / "state/source-audio-clock.json"),
                "stageName": "source-audio-clock",
            },
            "video": {
                "workflow": workflow_pin,
                "receipt": _pin(before / "state/source-video-clock.json"),
                "stageName": "source-video-clock",
            },
            "separation": {
                "workflow": workflow_pin,
                "receipt": _pin(before / "state/source-voice-separation.json"),
                "stageName": "source-voice-separation",
            },
        },
    }
    _json(fixture.config, config)


def test_explicit_real_stage_adoption_needs_no_initial_factory_record(fixture: Fixture) -> None:
    _explicit_config(fixture)
    (fixture.prepared.work / "preparation.json").unlink()
    (fixture.prepared.work / "audio-clock.producer.json").unlink()
    (fixture.prepared.work / "video-clock.producer.json").unlink()
    path = fixture.plan()
    plan = factory._load(path)
    assert plan.settings.preparation is None
    assert plan.prepared.source == fixture.prepared.source
    assert plan.prepared.audio_receipt == fixture.prepared.work / "state/source-audio-clock.json"


def test_two_clocks_can_share_genuine_adoption_stage_and_history_limits(fixture: Fixture) -> None:
    _explicit_config(fixture)
    before = fixture.prepared.work
    workflow = _data(before / "workflow.json")
    stages = workflow["stages"]
    assert isinstance(stages, list)
    clock = record(stages[0])
    clock["name"] = "adopt-retained-clocks"
    clock["outputs"] = [str(before / "video-clock.json"), str(before / "audio-clock.json")]
    proof = before / "adoption-evidence.json"
    _json(
        proof,
        {
            "protocol": "retained-clock-adoption-v1",
            "historicalGenerationBindingFullyVerified": False,
        },
    )
    clock["inputs"] = [str(fixture.prepared.source), str(proof)]
    workflow["stages"] = [clock, stages[2]]
    _json(before / "workflow.json", workflow)
    parsed = parse_stages(workflow)
    adoption_state = before / "adoption-state"
    adoption_state.mkdir()
    _stage_complete(parsed[0], adoption_state)
    _stage_complete(parsed[1], before / "state")
    config = _data(fixture.config)
    roles = record(record(config["predecessor"])["stages"])
    original = {
        "workflow": _pin(before / "workflow.json"),
        "receipt": _pin(adoption_state / "adopt-retained-clocks.json"),
        "stageName": "adopt-retained-clocks",
    }
    roles["audio"], roles["video"] = original, original
    record(roles["separation"])["workflow"] = _pin(before / "workflow.json")
    _json(fixture.config, config)
    expected = proof.read_bytes()
    plan = factory._load(fixture.plan())
    assert plan.prepared.audio_receipt == adoption_state / "adopt-retained-clocks.json"
    assert initial.Pin(proof, _sha(proof)) in plan.pins
    assert proof.read_bytes() == expected


@pytest.mark.parametrize("field", ["source", "audioClock", "modelInput", "vocals"])
def test_explicit_adoption_rejects_changed_role_pin(fixture: Fixture, field: str) -> None:
    _explicit_config(fixture)
    data = _data(fixture.config)
    role = record(record(record(data["predecessor"])["roles"])[field])
    role["sha256"] = "0" * 64
    _json(fixture.config, data)
    with pytest.raises((ReplayError, StageError)):
        fixture.plan()


def test_explicit_domain_verifier_or_source_binding_missing_is_rejected(fixture: Fixture) -> None:
    _explicit_config(fixture)
    before = fixture.prepared.work
    workflow = _data(before / "workflow.json")
    stages = workflow["stages"]
    assert isinstance(stages, list)
    record(stages[0])["verify_command"] = []
    _json(before / "workflow.json", workflow)
    _stage_complete(parse_stages(workflow)[0], before / "state")
    data = _data(fixture.config)
    explicit_stages = record(record(data["predecessor"])["stages"])
    record(explicit_stages["video"])["workflow"] = _pin(before / "workflow.json")
    record(explicit_stages["video"])["receipt"] = _pin(before / "state/source-video-clock.json")
    _json(fixture.config, data)
    with pytest.raises(ReplayError):
        fixture.plan()


def test_prepare_real_dependencies_candidate_flags_and_no_inference(fixture: Fixture) -> None:
    path = fixture.plan()
    plan = factory._load(path)
    graph = stage_graph(parse_stages(_data(fixture.work / "workflow.json")))
    assert graph == {
        "vocal-mono-pcm": (),
        "vocal-asr": ("vocal-mono-pcm",),
        "vocal-asr-assembly": ("vocal-mono-pcm", "vocal-asr"),
    }
    assert _data(path)["deliveryComplete"] is False
    assert plan.settings.model == fixture.model
    assert not (fixture.work / "asr").exists()
    assert "NUMBA_CACHE_DIR=" + str(fixture.work / "numba") in factory._environment(plan)
    with pytest.raises(ReplayError, match="absent"):
        factory.prepare(fixture.config, fixture.work)


@pytest.mark.parametrize(
    "role", ["source-audio-clock", "source-video-clock", "source-voice-separation"]
)
def test_direct_cli_or_running_predecessor_is_not_adopted(fixture: Fixture, role: str) -> None:
    (fixture.prepared.work / "state" / f"{role}.json").unlink()
    with pytest.raises(ReplayError, match="genuine workflow"):
        fixture.plan()
    assert not fixture.work.exists()


@pytest.mark.parametrize(
    "field,value",
    [
        ("version", True),
        ("fps", True),
        ("fps", 121),
        ("runtimeInputs", {}),
        ("runtimeInputs", [None]),
    ],
)
def test_settings_invalid_fields(fixture: Fixture, field: str, value: object) -> None:
    data = _data(fixture.config)
    data[field] = value
    _json(fixture.config, data)
    with pytest.raises(ReplayError):
        fixture.plan()


@pytest.mark.parametrize("field,value", [("child", 3601), ("recognize", 0), ("mono", True)])
def test_deadline_contract(fixture: Fixture, field: str, value: object) -> None:
    data = _data(fixture.config)
    record(data["timeouts"])[field] = value
    _json(fixture.config, data)
    with pytest.raises(ReplayError):
        fixture.plan()


def test_changed_model_and_original_input_rejected(fixture: Fixture) -> None:
    (fixture.model / "weights.npz").write_bytes(b"substitution")
    with pytest.raises(StageError, match="SHA256"):
        fixture.plan()


def test_multiple_weights_no_silent_model_choice(fixture: Fixture) -> None:
    (fixture.model / "weights.safetensors").write_bytes(b"other model")
    with pytest.raises(ReplayError, match="Exactly one"):
        fixture.plan()


@pytest.mark.parametrize("target", ["workflow.json", "transcription.json"])
def test_pinned_generated_plan_tampering(fixture: Fixture, target: str) -> None:
    path = fixture.plan()
    _json(fixture.work / target, {})
    with pytest.raises(ReplayError):
        factory._load(path)


def test_real_stereo_to_mono_command_and_exclusive_source_bound_proof(
    fixture: Fixture, monkeypatch: pytest.MonkeyPatch
) -> None:
    path = fixture.plan()
    commands: list[tuple[str, ...]] = []

    def run(command: tuple[str, ...], *, stdin: int, check: bool, timeout: int) -> None:
        commands.append(command)
        assert stdin == subprocess.DEVNULL and check and timeout == 120
        _wave(Path(command[-1]), 1, 16000, 1600)

    monkeypatch.setattr(subprocess, "run", run)
    factory.produce(path, "mono")
    factory.verify(path, "mono")
    assert commands == [
        factory.mono_command(
            fixture.prepared.work / "separation/vocals.wav",
            fixture.work / "mono.wav",
            fixture.prepared.settings.tools["ffmpeg"].path,
        )
    ]
    assert not any("async" in part or "atrim" in part or "apad" in part for part in commands[0])
    assert _data(fixture.work / "mono.producer.json")["outputs"] == {
        str(fixture.work / "mono.wav"): _sha(fixture.work / "mono.wav")
    }
    assert _data(fixture.work / "mono.producer.json")["sampleCounts"] == {
        "stereoInputSamples": 4410,
        "stereoInputRate": 44100,
        "monoOutputSamples": 1600,
        "monoOutputRate": 16000,
        "durationPreserving": True,
    }
    with pytest.raises(ReplayError, match="Occupied"):
        factory.produce(path, "mono")


@pytest.mark.parametrize("samples", [1598, 1602])
def test_time_stretch_or_shortened_pcm_rejected(fixture: Fixture, samples: int) -> None:
    path = fixture.plan()
    plan = factory._load(path)
    _wave(fixture.work / "mono.wav", 1, 16000, samples)
    with pytest.raises(ReplayError, match="duration"):
        factory._mono_domain(plan)


@pytest.mark.parametrize("channels,rate", [(2, 16000), (1, 48000)])
def test_pcm_format_rejected(fixture: Fixture, channels: int, rate: int) -> None:
    fixture.plan()
    path = fixture.work / "bad.wav"
    _wave(path, channels, rate, 10)
    with pytest.raises(ReplayError, match="format"):
        factory._pcm(path, 1, 16000)


def test_truncated_pcm_rejected(fixture: Fixture) -> None:
    fixture.plan()
    path = fixture.work / "bad.wav"
    _wave(path, 1, 16000, 10)
    path.write_bytes(path.read_bytes()[:-2])
    with pytest.raises(ReplayError, match="Truncated"):
        factory._pcm(path, 1, 16000)


def test_full_asr_binds_native_runtime_and_independent_policy(
    fixture: Fixture, monkeypatch: pytest.MonkeyPatch
) -> None:
    path = fixture.plan()
    _mono(fixture, path)

    def run(
        source: Path,
        output: Path,
        policy: Path,
        *,
        timeout: int,
        model: Path,
        weights_sha256: str,
        config_sha256: str,
    ) -> Path:
        assert source == fixture.work / "mono.wav" and output == fixture.work / "asr"
        assert policy == fixture.work / "asr-policy" and timeout == 300
        assert model == fixture.model and weights_sha256 == _sha(model / "weights.npz")
        assert config_sha256 == _sha(model / "config.json")
        plan = factory._load(path)
        _collection(plan)
        _policy(plan)
        return policy / "result.json"

    monkeypatch.setattr(isolated_cpu_asr, "run", run)
    factory.produce(path, "recognize")
    factory.verify(path, "recognize")
    assert len(record(_data(fixture.work / "recognize.producer.json")["collectionEvidence"])) == 4
    factory.produce(path, "assemble")
    factory.verify(path, "assemble")
    labels = _data(fixture.work / "labels.json")
    assert labels["clock"] == "input-pcm-not-projected"
    assert labels["labels"] == [
        {"text": "声", "start": 0.01, "end": 0.08, "provenance": "000000000000:0"}
    ]


@pytest.mark.parametrize(
    "filename", ["complete.json", "000000000000.clock.json", "000000000000.raw.json", "inputs.json"]
)
def test_changed_asr_evidence_not_finished(fixture: Fixture, filename: str) -> None:
    path = fixture.plan()
    _recognition(fixture, path)
    _json(fixture.work / "asr" / filename, {})
    with pytest.raises((ReplayError, StageError)):
        factory.verify(path, "recognize")


def test_resume_preserves_raw_and_prior_running_state_with_new_policy(fixture: Fixture) -> None:
    path = fixture.plan()
    _mono(fixture, path)
    old = factory._load(path)
    _collection(old, complete=False)
    asr_stage = parse_stages(_data(fixture.work / "workflow.json"))[1]
    state = fixture.work / "state/vocal-asr.json"
    _json(state, {"status": "running", "fingerprint": _fingerprint(asr_stage)})
    original = state.read_bytes()
    raw = (old.collection / "000000000000.raw.json").read_bytes()
    resumed_work = fixture.work.parent / "resumed"
    result = factory.prepare_resume(path, resumed_work)
    resumed = factory._load(resumed_work / "transcription.json")
    assert resumed.collection == old.collection and resumed.mono_root == fixture.work
    assert (
        state.read_bytes() == original
        and (old.collection / "000000000000.raw.json").read_bytes() == raw
    )
    assert len(parse_stages(_data(result))) == 3
    stages = parse_stages(_data(result))
    assert stages[0].adopt_existing is True
    assert old.collection / "inputs.json" not in stages[1].outputs
    assert old.collection / "complete.json" in stages[1].outputs
    factory.verify(resumed_work / "transcription.json", "mono")


def test_resume_real_completed_windows_are_reused_and_final_receipt_is_new(
    fixture: Fixture, monkeypatch: pytest.MonkeyPatch
) -> None:
    path = fixture.plan()
    _mono(fixture, path)
    old = factory._load(path)
    _collection(old, complete=False)
    stage = parse_stages(_data(fixture.work / "workflow.json"))[1]
    _json(
        fixture.work / "state/vocal-asr.json",
        {"status": "running", "fingerprint": _fingerprint(stage)},
    )
    raw_path = old.collection / "000000000000.raw.json"
    old_bytes = raw_path.read_bytes()
    resumed_work = fixture.work.parent / "resumed"
    factory.prepare_resume(path, resumed_work)
    resumed_path = resumed_work / "transcription.json"

    def run(
        source: Path,
        output: Path,
        policy: Path,
        *,
        timeout: int,
        model: Path,
        weights_sha256: str,
        config_sha256: str,
    ) -> Path:
        assert output == old.collection and source == old.mono_root / "mono.wav"
        assert raw_path.read_bytes() == old_bytes
        _json(
            output / "complete.json",
            {"rawSHA256": [_sha(raw_path)], "windows": 1, "deliveryComplete": False},
        )
        _policy(factory._load(resumed_path))
        return policy / "result.json"

    monkeypatch.setattr(isolated_cpu_asr, "run", run)
    factory.produce(resumed_path, "recognize")
    factory.verify(resumed_path, "recognize")
    factory.produce(resumed_path, "assemble")
    assert raw_path.read_bytes() == old_bytes
    assert _data(resumed_work / "recognize.producer.json")["candidateOnly"] is True
    assert not (old.work / "recognize.producer.json").exists()


@pytest.mark.parametrize(
    "start,rate,base,gap", [(0, 44100, "1/44100", 2205), (90000, 48000, "1/90000", 5000)]
)
def test_different_native_origins_and_timebases_remain_real(
    fixture: Fixture, start: int, rate: int, base: str, gap: int
) -> None:
    audio = fixture.prepared.work / "audio-clock.json"
    _json(
        audio,
        {
            "streams": [{"sample_rate": str(rate), "time_base": base}],
            "frames": [
                {"pts": start, "nb_samples": rate // 20},
                {"pts": start + gap, "nb_samples": rate // 20},
            ],
        },
    )
    workflow = fixture.prepared.work / "workflow.json"
    _stage_complete(parse_stages(_data(workflow))[1], fixture.prepared.work / "state")
    path = fixture.plan()
    _assembly(fixture, path)
    factory.prepare_successor(path)
    factory.produce(path, "lineage")
    factory.produce(path, "projection")
    labels = _data(fixture.work / "source-labels.json")["labels"]
    assert isinstance(labels, list)
    expected = 0.01 if start == 0 else 1.01
    assert record(labels[0])["start"] == expected
    assert record(labels[0])["end"] == pytest.approx((start + gap) * float(Fraction(base)) + 0.03)


def test_active_workflow_resume_refused(fixture: Fixture) -> None:
    path = fixture.plan()
    with execution_lock(fixture.work / "state", ".workflow") as acquired:
        assert acquired
        with pytest.raises(ReplayError, match="still owns"):
            factory.prepare_resume(path, fixture.work.parent / "resumed")


def test_successor_pins_after_completion_projects_nonzero_gap_once(
    fixture: Fixture, monkeypatch: pytest.MonkeyPatch
) -> None:
    path = fixture.plan()
    _assembly(fixture, path)
    workflow = factory.prepare_successor(path)
    graph = stage_graph(parse_stages(_data(workflow)))
    assert graph == {
        "source-lineage": (),
        "source-projection": ("source-lineage",),
        "source-quiet": ("source-lineage", "source-projection"),
    }
    factory.produce(path, "lineage")
    factory.verify(path, "lineage")
    factory.produce(path, "projection")
    factory.verify(path, "projection")
    labels = _data(fixture.work / "source-labels.json")["labels"]
    assert isinstance(labels, list)
    projected = record(labels[0])
    assert projected["start"] == 1.01
    assert projected["end"] == pytest.approx(49100 / 44100 + 0.03)
    assert projected["text"] == "声"

    def peaks(
        command: list[str], *, stdin: int, stdout: object, stderr: object, check: bool, timeout: int
    ) -> None:
        assert command[0] == str(fixture.prepared.settings.tools["ffmpeg"].path)
        assert hasattr(stdout, "write")
        stdout.write(
            b"frame:0 pts:0 pts_time:0\nlavfi.astats.Overall.Peak_level=0\n"
            b"frame:1 pts:882 pts_time:0.02\nlavfi.astats.Overall.Peak_level=0\n"
            b"frame:2 pts:1764 pts_time:0.04\nlavfi.astats.Overall.Peak_level=0\n"
            b"frame:3 pts:2646 pts_time:0.06\nlavfi.astats.Overall.Peak_level=0\n"
            b"frame:4 pts:3528 pts_time:0.08\nlavfi.astats.Overall.Peak_level=0\n"
        )

    monkeypatch.setattr(subprocess, "run", peaks)
    factory.produce(path, "quiet")
    factory.verify(path, "quiet")
    assert _data(fixture.work / "quiet/cuts.json")["nonSpeechRemovalVerified"] is False
    with pytest.raises(ReplayError, match="occupied"):
        factory.prepare_successor(path)


@pytest.mark.parametrize(
    "action", ["prepare", "produce", "verify", "prepare-successor", "prepare-resume"]
)
def test_cli_boundaries(fixture: Fixture, action: str) -> None:
    if action == "prepare":
        assert factory.main([action, str(fixture.config), str(fixture.work)]) == 0
    else:
        path = fixture.plan()
        arguments = [action, str(path)]
        if action in ("produce", "verify"):
            arguments.append("assemble")
        if action == "prepare-resume":
            arguments.append(str(fixture.work.parent / "resumed"))
        assert factory.main(arguments) == 1


def test_complete_stage_reuse_uses_domain_verifier(fixture: Fixture) -> None:
    path = fixture.plan()
    _mono(fixture, path)
    stage = parse_stages(_data(fixture.work / "workflow.json"))[0]
    commands: list[tuple[str, ...]] = []

    def execute(command: tuple[str, ...], log: Path, timeout: int) -> None:
        commands.append(command)
        factory.verify(path, "mono")

    assert run_stage(stage, state_directory=fixture.work / "state", execute=execute) == "reused"
    assert commands == [stage.verify_command]
    (fixture.work / "mono.wav").write_bytes(b"changed")
    with pytest.raises(StageError, match="changed"):
        run_stage(stage, state_directory=fixture.work / "state", execute=execute)
