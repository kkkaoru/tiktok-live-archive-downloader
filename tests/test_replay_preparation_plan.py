import hashlib
import json
import os
import subprocess
import sys
import wave
from dataclasses import dataclass
from pathlib import Path

import pytest

from replay import replay_preparation_plan as planner
from replay.bgm_separation import MODEL_NAME, MODEL_REVISION
from replay.core import ReplayError
from replay.edit_completion import REQUIRED_STAGES
from replay.input_hash_snapshot import InputHashSnapshot
from replay.recordings import record
from replay.stages import StageError, run_stage
from replay.workflow import parse_stages, stage_graph


def test_generated_graph_passes_original_external_workflow_check(fixture: "Fixture") -> None:
    graph = fixture.plan()
    result = subprocess.run(
        [
            sys.executable,
            "-B",
            "-m",
            "replay.workflow",
            str(graph),
            "--state",
            str(fixture.work / "state"),
            "--check",
        ],
        capture_output=True,
        text=True,
        check=False,
        timeout=20,
    )
    assert result.returncode == 0, result.stderr
    assert "source-video-clock" in result.stdout
    assert "source-audio-clock" in result.stdout
    assert "source-voice-separation" in result.stdout
    assert not (fixture.work / "state").exists()


def test_external_workflow_check_rejects_state_overlapping_outputs(fixture: "Fixture") -> None:
    graph = fixture.plan()
    result = subprocess.run(
        [
            sys.executable,
            "-B",
            "-m",
            "replay.workflow",
            str(graph),
            "--state",
            str(fixture.work),
            "--check",
        ],
        capture_output=True,
        text=True,
        check=False,
        timeout=20,
    )
    assert result.returncode != 0
    assert result.stderr == "Workflow stopped (StageError); inspect its private state.\n"
    assert not (fixture.work / "state").exists()


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _json(path: Path, data: object) -> None:
    path.write_text(json.dumps(data), encoding="utf-8")


def _tool(path: Path) -> dict[str, str]:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(b"synthetic executable " + path.name.encode())
    path.chmod(0o700)
    return {"path": str(path), "sha256": _sha(path)}


@dataclass(frozen=True)
class Fixture:
    root: Path
    edit: Path
    settings: Path
    work: Path
    source: Path
    code: Path
    weights: Path
    config: Path

    def settings_data(self) -> dict[str, object]:
        return record(json.loads(self.settings.read_bytes()))

    def plan(self) -> Path:
        return planner.prepare(self.edit, self.settings, self.work)


@pytest.fixture
def fixture(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Fixture:
    root = tmp_path / "checkout"
    work = root / "edits" / "work"
    work.mkdir(parents=True)
    root.joinpath(".python-version").write_text("3.12.13\n", encoding="utf-8")
    root.joinpath("pyproject.toml").write_text("[project]\n", encoding="utf-8")
    root.joinpath("uv.lock").write_text("version = 1\n", encoding="utf-8")
    code = root / "src" / "replay" / "synthetic.py"
    code.parent.mkdir(parents=True)
    code.write_text("# synthetic source input\n", encoding="utf-8")
    python_target = root / "python-target"
    python_pin = _tool(python_target)
    python = root / ".venv" / "bin" / "python"
    python.parent.mkdir(parents=True)
    python.symlink_to(python_target)
    python_pin["path"] = str(python)
    python.parent.parent.joinpath("pyvenv.cfg").write_text(
        "version_info = 3.12.13\n", encoding="utf-8"
    )
    tools = {
        "python": python_pin,
        "ffmpeg": _tool(root / "tools" / "ffmpeg"),
        "ffprobe": _tool(root / "tools" / "ffprobe"),
        "env": _tool(root / "tools" / "env"),
    }
    home = root / "edits" / "models" / "local-hf"
    snapshot = (
        home / "hub" / ("models--" + MODEL_NAME.replace("/", "--")) / "snapshots" / MODEL_REVISION
    )
    snapshot.mkdir(parents=True)
    weights, config = snapshot / "model.safetensors", snapshot / "config.json"
    weights.write_bytes(b"synthetic weights")
    config.write_bytes(b'{"synthetic":true}')
    monkeypatch.setattr(planner, "ROOT", root)
    monkeypatch.setattr(planner, "MODEL_SHA256", _sha(weights))
    monkeypatch.setattr(planner, "MODEL_CONFIG_SHA256", _sha(config))
    monkeypatch.setattr(planner, "_code_paths", lambda: (code,))
    settings = work / "tools.json"
    _json(
        settings,
        {
            "version": 1,
            "tools": tools,
            "hfHome": str(home),
            "device": "cpu",
            "clockTimeoutSeconds": 120,
            "separationTimeoutSeconds": 7200,
        },
    )
    source = root / "source.mp4"
    source.write_bytes(b"synthetic recording only")
    edit = root / "edits" / "replay-123"
    edit.mkdir()
    _json(
        edit / "edit-config.json",
        {
            "version": 1,
            "replay_id": "123",
            "source": str(source),
            "source_size": source.stat().st_size,
            "engine": str(root),
            "executor_scope": str(root),
            "stages": dict.fromkeys(REQUIRED_STAGES, True),
        },
    )
    return Fixture(root, edit, settings, work / "prepared", source, code, weights, config)


def _video(start: int = 0, base: str = "1/90000") -> dict[str, object]:
    return {
        "streams": [{"time_base": base}],
        "frames": [{"pts": start, "duration": 3000}, {"pts": start + 3030, "duration": 2970}],
    }


def _audio(start: int = 0, base: str = "1/48000") -> dict[str, object]:
    return {
        "streams": [{"time_base": base, "sample_rate": "48000"}],
        "frames": [{"pts": start, "nb_samples": 32}, {"pts": start + 40, "nb_samples": 32}],
    }


def _clocks(fixture: Fixture) -> None:
    _json(fixture.work / "video-clock.json", _video())
    _json(fixture.work / "audio-clock.json", _audio())
    _seal(fixture, "video")
    _seal(fixture, "audio")


def _seal(fixture: Fixture, kind: planner.Kind) -> None:
    preparation = planner._load(fixture.work / "preparation.json")
    _json(planner._producer_path(preparation, kind), planner._producer_data(preparation, kind))


def _pcm(path: Path, samples: int) -> None:
    with wave.open(str(path), "wb") as stream:
        stream.setnchannels(2)
        stream.setsampwidth(2)
        stream.setframerate(44100)
        stream.writeframes(b"\0" * samples * 4)


def _separation(fixture: Fixture, samples: int = 59, *, seal: bool = True) -> None:
    folder = fixture.work / "separation"
    folder.mkdir()
    _pcm(folder / "model-input.wav", samples)
    _pcm(folder / "vocals.wav", samples)
    _json(
        folder / "result.json",
        {
            "source": str(fixture.source),
            "device": "cpu",
            "preparedInput": False,
            "modelInput": str(folder / "model-input.wav"),
            "vocals": str(folder / "vocals.wav"),
            "timestampCorrection": False,
            "loudnessNormalization": False,
            "deliveryComplete": False,
        },
    )
    _json(folder / "before-model-import-descriptors.json", {"externalWritableRegular": []})
    _json(folder / "after-model-import-descriptors.json", {"externalWritableRegular": []})
    _json(folder / "after-separation-descriptors.json", {"externalWritableRegular": []})
    if seal:
        _seal(fixture, "separation")


def test_prepare_is_plan_only_with_exact_dependencies_and_private_outputs(fixture: Fixture) -> None:
    path = fixture.plan()
    stages = parse_stages(json.loads(path.read_bytes()))
    assert stage_graph(stages) == {
        "source-video-clock": (),
        "source-audio-clock": (),
        "source-voice-separation": ("source-video-clock", "source-audio-clock"),
    }
    assert len(stages) == 3
    assert stages[2].outputs == tuple(
        fixture.work / "separation" / name
        for name in (
            "model-input.wav",
            "vocals.wav",
            "result.json",
            "before-model-import-descriptors.json",
            "after-model-import-descriptors.json",
            "after-separation-descriptors.json",
            "producer.json",
        )
    )
    assert fixture.source in stages[0].inputs
    assert fixture.code in stages[2].inputs
    assert fixture.weights in stages[2].inputs
    assert fixture.config in stages[2].inputs
    assert fixture.root / ".venv" / "pyvenv.cfg" in stages[2].inputs
    assert "replay.phase_timer" in stages[0].command
    assert "produce" in stages[2].command
    assert "HF_HUB_OFFLINE=1" in stages[2].command
    assert str(fixture.root / ".venv" / "bin" / "python") in stages[2].command
    assert not fixture.work.joinpath("video-clock.json").exists()
    assert not fixture.work.joinpath("separation").exists()
    assert path.stat().st_mode & 0o777 == 0o600
    assert fixture.work.stat().st_mode & 0o777 == 0o700
    assert stages[0].adopt_existing is False
    assert (
        record(json.loads(fixture.work.joinpath("preparation.json").read_bytes()))[
            "deliveryComplete"
        ]
        is False
    )


def test_separate_python_preserves_venv_alias_and_declares_both_configs(fixture: Fixture) -> None:
    target = fixture.root / "separation-python-target"
    pin = _tool(target)
    invocation = fixture.root / "edits" / ".venv-provider" / "bin" / "python"
    invocation.parent.mkdir(parents=True)
    invocation.symlink_to(target)
    config = invocation.parent.parent / "pyvenv.cfg"
    config.write_text("version = 3.12.13\n", encoding="utf-8")
    pin["path"] = str(invocation)
    data = fixture.settings_data()
    record(data["tools"])["separationPython"] = pin
    _json(fixture.settings, data)
    stages = parse_stages(json.loads(fixture.plan().read_bytes()))
    assert str(invocation) in stages[2].command
    assert config in stages[2].inputs
    assert target in stages[2].inputs


def test_hf_snapshot_blob_aliases_are_resolved_and_bound(fixture: Fixture) -> None:
    blob = fixture.weights.parents[2] / "blobs"
    blob.mkdir()
    weight_target, config_target = blob / "weights", blob / "configuration"
    fixture.weights.rename(weight_target)
    fixture.config.rename(config_target)
    fixture.weights.symlink_to(weight_target)
    fixture.config.symlink_to(config_target)
    stages = parse_stages(json.loads(fixture.plan().read_bytes()))
    assert weight_target in stages[2].inputs
    assert config_target in stages[2].inputs
    assert fixture.weights not in stages[2].inputs


@pytest.mark.parametrize("device", ["cpu", "mps"])
def test_explicit_device_is_preserved(fixture: Fixture, device: str) -> None:
    data = fixture.settings_data()
    data["device"] = device
    _json(fixture.settings, data)
    stages = parse_stages(json.loads(fixture.plan().read_bytes()))
    assert planner.read_settings(fixture.settings).device == device
    assert len(stages[2].outputs) == (7 if device == "cpu" else 4)


@pytest.mark.parametrize(
    "kind,start,base",
    [
        ("video", 0, "1/90000"),
        ("video", 180000, "1/90000"),
        ("video", 100, "1/1000"),
        ("audio", 0, "1/48000"),
        ("audio", 480000, "1/48000"),
        ("audio", 700, "1/1000"),
    ],
)
def test_clock_check_preserves_real_starts_timebases_and_gaps(
    fixture: Fixture, kind: planner.Kind, start: int, base: str
) -> None:
    fixture.plan()
    data = _video(start, base) if kind == "video" else _audio(start, base)
    path = fixture.work / f"{kind}-clock.json"
    _json(path, data)
    _seal(fixture, kind)
    original = path.read_bytes()
    planner.verify_media_clocks(fixture.work / "preparation.json", kind)
    assert path.read_bytes() == original


def test_separation_domain_is_read_only_and_accepts_exact_resampling_rounding(
    fixture: Fixture,
) -> None:
    fixture.plan()
    _clocks(fixture)
    _separation(fixture)
    before = fixture.work.joinpath("separation", "result.json").read_bytes()
    planner.verify_media_clocks(fixture.work / "preparation.json", "separation")
    assert fixture.work.joinpath("separation", "result.json").read_bytes() == before


def test_separation_must_match_source_decoded_sample_count(fixture: Fixture) -> None:
    fixture.plan()
    _clocks(fixture)
    _separation(fixture, samples=65)
    with pytest.raises(ReplayError, match="decoded source sample clock"):
        planner.verify_media_clocks(fixture.work / "preparation.json", "separation")


@pytest.mark.parametrize(
    "field,value", [("device", "mps"), ("preparedInput", True), ("loudnessNormalization", True)]
)
def test_separation_policy_is_bound(fixture: Fixture, field: str, value: object) -> None:
    fixture.plan()
    _clocks(fixture)
    _separation(fixture)
    path = fixture.work / "separation" / "result.json"
    data = record(json.loads(path.read_bytes()))
    data[field] = value
    _json(path, data)
    _seal(fixture, "separation")
    with pytest.raises(ReplayError, match="policy changed"):
        planner.verify_media_clocks(fixture.work / "preparation.json", "separation")


def test_same_size_changed_source_is_not_reused(fixture: Fixture) -> None:
    fixture.plan()
    _clocks(fixture)
    fixture.source.write_bytes(b"x" * fixture.source.stat().st_size)
    with pytest.raises(ReplayError, match="input hashes"):
        planner.verify_media_clocks(fixture.work / "preparation.json", "video")


def test_changed_code_is_not_reused(fixture: Fixture) -> None:
    fixture.plan()
    _clocks(fixture)
    fixture.code.write_text("# changed producer\n", encoding="utf-8")
    with pytest.raises(ReplayError, match="input hashes"):
        planner.verify_media_clocks(fixture.work / "preparation.json", "audio")


def test_existing_partial_root_is_preserved_and_refused(fixture: Fixture) -> None:
    fixture.work.mkdir()
    partial = fixture.work / "video-clock.json"
    partial.write_bytes(b"partial")
    with pytest.raises(ReplayError, match="occupied/partial"):
        fixture.plan()
    assert partial.read_bytes() == b"partial"


def test_existing_plan_is_not_regenerated(fixture: Fixture) -> None:
    workflow = fixture.plan()
    previous = workflow.read_bytes()
    with pytest.raises(ReplayError, match="occupied/partial"):
        fixture.plan()
    assert workflow.read_bytes() == previous


@pytest.mark.parametrize(
    "field,value",
    [
        ("version", True),
        ("device", "auto"),
        ("clockTimeoutSeconds", 0),
        ("clockTimeoutSeconds", True),
        ("separationTimeoutSeconds", 86401),
        ("unexpected", "value"),
    ],
)
def test_settings_reject_invalid_policies(fixture: Fixture, field: str, value: object) -> None:
    data = fixture.settings_data()
    data[field] = value
    _json(fixture.settings, data)
    with pytest.raises(ReplayError):
        fixture.plan()


@pytest.mark.parametrize("role", ["python", "ffmpeg", "ffprobe", "env"])
def test_changed_tool_bytes_reject_plan_without_outputs(fixture: Fixture, role: str) -> None:
    tool = record(record(fixture.settings_data()["tools"])[role])
    Path(str(tool["path"])).write_bytes(b"changed executable")
    with pytest.raises(StageError, match="expected pin"):
        fixture.plan()
    assert not fixture.work.exists()


def test_mismatching_model_pin_does_not_create_outputs(fixture: Fixture) -> None:
    fixture.weights.write_bytes(b"different model")
    with pytest.raises(StageError, match="expected pin"):
        fixture.plan()
    assert not fixture.work.exists()


def test_wrong_python_patch_is_refused(fixture: Fixture) -> None:
    fixture.root.joinpath(".venv", "pyvenv.cfg").write_text(
        "version_info = 3.13.0\n", encoding="utf-8"
    )
    with pytest.raises(ReplayError, match="patch version"):
        fixture.plan()


def test_unknown_tool_is_refused(fixture: Fixture) -> None:
    data = fixture.settings_data()
    record(data["tools"])["other"] = {"path": "x", "sha256": "a" * 64}
    _json(fixture.settings, data)
    with pytest.raises(ReplayError, match="unknown preparation tool"):
        fixture.plan()


def test_model_alias_cannot_escape_cache(fixture: Fixture) -> None:
    outside = fixture.root / "outside.safetensors"
    fixture.weights.rename(outside)
    fixture.weights.symlink_to(outside)
    with pytest.raises(ReplayError, match="scope"):
        fixture.plan()


def test_work_symlink_escape_is_refused(fixture: Fixture) -> None:
    fixture.work.symlink_to(fixture.root)
    with pytest.raises(ReplayError, match="canonical unlinked"):
        fixture.plan()


def test_work_must_stay_in_repository(fixture: Fixture) -> None:
    with pytest.raises(ReplayError, match="scope"):
        planner.prepare(fixture.edit, fixture.settings, fixture.root / "outside-work")


def test_duplicate_json_keys_are_not_silently_accepted(fixture: Fixture) -> None:
    fixture.settings.write_text('{"version":1,"version":1}', encoding="utf-8")
    with pytest.raises(ReplayError, match="Duplicate JSON key"):
        fixture.plan()


@pytest.mark.parametrize(
    "value",
    [
        {"streams": [], "frames": []},
        {"streams": [{"time_base": "1/90000"}], "frames": [{"pts": 0, "duration": 0}]},
        {"streams": [{"time_base": "2/1"}], "frames": [{"pts": 0, "duration": 1}]},
        {
            "streams": [{"time_base": "1/90000"}],
            "frames": [{"pts": 0, "duration": -1}, {"pts": 1, "duration": 1}],
        },
        {
            "streams": [{"time_base": "1/90000"}],
            "frames": [{"pts": 0, "duration": 1}, {"pts": 0, "duration": 1}],
        },
    ],
)
def test_invalid_video_clock_is_refused(fixture: Fixture, value: object) -> None:
    fixture.plan()
    _json(fixture.work / "video-clock.json", value)
    _seal(fixture, "video")
    with pytest.raises(ReplayError):
        planner.verify_media_clocks(fixture.work / "preparation.json", "video")


def test_partial_clock_is_refused_without_repair_or_overwrite(fixture: Fixture) -> None:
    fixture.plan()
    path = fixture.work / "audio-clock.json"
    path.write_bytes(b'{"streams":[')
    _seal(fixture, "audio")
    with pytest.raises(ValueError):
        planner.verify_media_clocks(fixture.work / "preparation.json", "audio")
    assert path.read_bytes() == b'{"streams":['


def test_workflow_tampering_is_refused(fixture: Fixture) -> None:
    fixture.plan()
    _json(fixture.work / "workflow.json", {"version": 1, "stages": []})
    with pytest.raises(ReplayError, match="Workflow no longer"):
        planner.verify_media_clocks(fixture.work / "preparation.json", "video")


@pytest.mark.parametrize(
    "field,value",
    [("deliveryComplete", True), ("candidateOnly", False), ("inputs", []), ("version", True)],
)
def test_plan_schema_does_not_promote_or_drop_pins(
    fixture: Fixture, field: str, value: object
) -> None:
    fixture.plan()
    path = fixture.work / "preparation.json"
    data = record(json.loads(path.read_bytes()))
    data[field] = value
    _json(path, data)
    with pytest.raises(ReplayError):
        planner.verify_media_clocks(path, "video")


def test_cli_prepare_and_readonly_verify(
    fixture: Fixture, capsys: pytest.CaptureFixture[str]
) -> None:
    assert (
        planner.main(["prepare", str(fixture.edit), str(fixture.settings), str(fixture.work)]) == 0
    )
    _clocks(fixture)
    assert (
        planner.main(["verify-media-clocks", str(fixture.work / "preparation.json"), "audio"]) == 0
    )
    assert "delivery remains unapproved" in capsys.readouterr().out


def test_cli_error_retains_failure(fixture: Fixture, capsys: pytest.CaptureFixture[str]) -> None:
    fixture.work.mkdir()
    assert (
        planner.main(["prepare", str(fixture.edit), str(fixture.settings), str(fixture.work)]) == 1
    )
    assert "REPLAY_PREPARATION_ERROR" in capsys.readouterr().err


def test_transitive_source_inventory_includes_lazy_imports_and_cycles(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    original = planner._code_paths
    directory = tmp_path / "src" / "replay"
    directory.mkdir(parents=True)
    directory.joinpath("__init__.py").write_text("# package\n", encoding="utf-8")
    directory.joinpath("replay_preparation_plan.py").write_text(
        "from replay import core\nimport replay.core\nfrom pathlib import Path\n", encoding="utf-8"
    )
    directory.joinpath("delivery_media.py").write_text(
        "from replay.core import ReplayError\n", encoding="utf-8"
    )
    directory.joinpath("delivery_audio.py").write_text(
        "def lazy():\n    from replay.core import ReplayError\n", encoding="utf-8"
    )
    directory.joinpath("phase_timer.py").write_text("# timer\n", encoding="utf-8")
    directory.joinpath("workflow.py").write_text("# workflow\n", encoding="utf-8")
    directory.joinpath("core.py").write_text(
        "from replay import replay_preparation_plan\n", encoding="utf-8"
    )
    monkeypatch.setattr(planner, "ROOT", tmp_path)
    assert original() == (
        directory / "__init__.py",
        directory / "core.py",
        directory / "delivery_audio.py",
        directory / "delivery_media.py",
        directory / "phase_timer.py",
        directory / "replay_preparation_plan.py",
    )


def test_metadata_read_rejects_hardlinks(fixture: Fixture) -> None:
    os.link(fixture.settings, fixture.settings.with_name("second-link.json"))
    with pytest.raises(ReplayError, match="single-link metadata"):
        fixture.plan()


def test_explicit_runtime_files_are_pinned(fixture: Fixture) -> None:
    runtime = fixture.root / "runtime-metadata"
    runtime.write_bytes(b"synthetic provider identity")
    data = fixture.settings_data()
    data["runtimeInputs"] = [{"path": str(runtime), "sha256": _sha(runtime)}]
    _json(fixture.settings, data)
    stages = parse_stages(json.loads(fixture.plan().read_bytes()))
    assert runtime in stages[2].inputs
    _clocks(fixture)
    runtime.write_bytes(b"changed provider identity")
    with pytest.raises(StageError, match="expected pin"):
        planner.verify_media_clocks(fixture.work / "preparation.json", "audio")


@pytest.mark.parametrize(
    "value",
    [
        None,
        "invalid",
        [{"path": "/outside", "sha256": "a" * 64}],
        [{"path": "relative", "sha256": "a" * 64}],
    ],
)
def test_runtime_pins_require_explicit_bounded_local_inputs(
    fixture: Fixture, value: object
) -> None:
    data = fixture.settings_data()
    data["runtimeInputs"] = value
    _json(fixture.settings, data)
    with pytest.raises(ReplayError):
        fixture.plan()


@pytest.mark.parametrize("kind", ["video", "audio"])
def test_successful_shared_clock_call_seals_producer_evidence(
    fixture: Fixture, kind: planner.Kind, monkeypatch: pytest.MonkeyPatch
) -> None:
    fixture.plan()
    calls: list[list[str]] = []

    def successful(arguments: list[str]) -> int:
        calls.append(arguments)
        _json(Path(arguments[2]), _video() if kind == "video" else _audio())
        return 0

    monkeypatch.setattr(planner, "media_main", successful)
    assert planner.main(["produce", str(fixture.work / "preparation.json"), kind]) == 0
    planner.verify_media_clocks(fixture.work / "preparation.json", kind)
    assert calls == [
        [
            "clock" if kind == "video" else "audio-clock",
            str(fixture.source),
            str(fixture.work / f"{kind}-clock.json"),
            "--ffprobe",
            str(fixture.root / "tools" / "ffprobe"),
        ]
    ]
    receipt = record(json.loads(fixture.work.joinpath(f"{kind}-clock.producer.json").read_bytes()))
    assert receipt["sharedProducerCompleted"] is True
    assert receipt["deliveryComplete"] is False


def test_successful_shared_separation_seals_evidence(
    fixture: Fixture, monkeypatch: pytest.MonkeyPatch
) -> None:
    fixture.plan()
    _clocks(fixture)
    calls: list[tuple[Path, Path, str, str]] = []

    def successful(source: Path, directory: Path, *, ffmpeg: str, device: str) -> Path:
        calls.append((source, directory, ffmpeg, device))
        _separation(fixture, seal=False)
        return directory / "result.json"

    monkeypatch.setattr(planner, "separate", successful)
    planner.produce(fixture.work / "preparation.json", "separation")
    planner.verify_media_clocks(fixture.work / "preparation.json", "separation")
    assert calls == [
        (fixture.source, fixture.work / "separation", str(fixture.root / "tools" / "ffmpeg"), "cpu")
    ]


def test_valid_json_without_successful_producer_is_not_adopted(fixture: Fixture) -> None:
    fixture.plan()
    _json(fixture.work / "video-clock.json", _video())
    with pytest.raises(FileNotFoundError):
        planner.verify_media_clocks(fixture.work / "preparation.json", "video")


def test_failed_clock_call_preserves_partial_output_without_success_evidence(
    fixture: Fixture, monkeypatch: pytest.MonkeyPatch
) -> None:
    fixture.plan()

    def failed(arguments: list[str]) -> int:
        _json(Path(arguments[2]), _video())
        return 1

    monkeypatch.setattr(planner, "media_main", failed)
    with pytest.raises(ReplayError, match="producer failed"):
        planner.produce(fixture.work / "preparation.json", "video")
    assert fixture.work.joinpath("video-clock.json").is_file()
    assert not fixture.work.joinpath("video-clock.producer.json").exists()
    with pytest.raises(ReplayError, match="occupied"):
        planner.produce(fixture.work / "preparation.json", "video")


def test_altered_clock_does_not_match_its_producer_evidence(fixture: Fixture) -> None:
    fixture.plan()
    _clocks(fixture)
    _json(fixture.work / "video-clock.json", _video(180000))
    with pytest.raises(ReplayError, match="producer evidence"):
        planner.verify_media_clocks(fixture.work / "preparation.json", "video")


def test_generated_stage_reuses_only_verified_immutable_record(fixture: Fixture) -> None:
    stage = parse_stages(json.loads(fixture.plan().read_bytes()))[0]
    calls: list[str] = []

    def execute(command: tuple[str, ...], _log: Path, _timeout: int) -> None:
        if "produce" in command:
            calls.append("producer")
            _json(fixture.work / "video-clock.json", _video())
            _seal(fixture, "video")
        else:
            calls.append("verifier")
            planner.verify_media_clocks(fixture.work / "preparation.json", "video")

    state = fixture.work / "state"
    assert run_stage(stage, state_directory=state, execute=execute) == "completed"
    assert run_stage(stage, state_directory=state, execute=execute) == "reused"
    assert calls == ["producer", "verifier", "verifier"]
    fixture.source.write_bytes(b"x" * fixture.source.stat().st_size)
    with pytest.raises(StageError, match="inputs changed"):
        run_stage(stage, state_directory=state, execute=execute)
    assert calls == ["producer", "verifier", "verifier"]


def test_interrupted_valid_clock_cannot_be_recovered_without_sealed_evidence(
    fixture: Fixture,
) -> None:
    stage = parse_stages(json.loads(fixture.plan().read_bytes()))[0]

    def interrupted(_command: tuple[str, ...], _log: Path, _timeout: int) -> None:
        _json(fixture.work / "video-clock.json", _video())
        raise StageError("Interrupted synthetic producer")

    state = fixture.work / "state"
    with pytest.raises(StageError, match="Interrupted"):
        run_stage(stage, state_directory=state, execute=interrupted)

    def verify(_command: tuple[str, ...], _log: Path, _timeout: int) -> None:
        planner.verify_media_clocks(fixture.work / "preparation.json", "video")

    with pytest.raises(FileNotFoundError):
        run_stage(stage, state_directory=state, execute=verify)
    assert (
        record(json.loads(state.joinpath("source-video-clock.json").read_bytes()))["status"]
        == "running"
    )


def test_alias_changes_at_terminal_preflight_prevent_plan_publication(
    fixture: Fixture, monkeypatch: pytest.MonkeyPatch
) -> None:
    replacement = fixture.root / "same-python-target"
    replacement.write_bytes(fixture.root.joinpath("python-target").read_bytes())
    replacement.chmod(0o700)
    original_finish = InputHashSnapshot.finish

    def changed_alias(snapshot: InputHashSnapshot) -> dict[Path, str]:
        result = original_finish(snapshot)
        invocation = fixture.root / ".venv" / "bin" / "python"
        invocation.unlink()
        invocation.symlink_to(replacement)
        return result

    monkeypatch.setattr(InputHashSnapshot, "finish", changed_alias)
    with pytest.raises(ReplayError, match="alias or runtime settings changed"):
        fixture.plan()
    assert not fixture.work.exists()


def test_source_clock_receipt_remains_required_for_separation_verification(
    fixture: Fixture,
) -> None:
    fixture.plan()
    _clocks(fixture)
    _separation(fixture)
    fixture.work.joinpath("audio-clock.producer.json").unlink()
    with pytest.raises(FileNotFoundError):
        planner.verify_media_clocks(fixture.work / "preparation.json", "separation")


def test_input_inventory_budget_is_enforced(
    fixture: Fixture, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(planner, "MAX_INPUTS", 4)
    with pytest.raises(ReplayError, match="input closure exceeds"):
        fixture.plan()
    assert not fixture.work.exists()


def test_configured_python_version_must_be_an_exact_patch(fixture: Fixture) -> None:
    fixture.root.joinpath(".python-version").write_text("3.12\n", encoding="utf-8")
    with pytest.raises(ReplayError, match="configured Python patch version"):
        fixture.plan()
