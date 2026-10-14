"""Retained measurements authenticate old Stages and protect new recognition."""

import fcntl
import json
import sys
import wave
from dataclasses import asdict, dataclass
from math import ceil
from pathlib import Path

import pytest

from replay import asr_clock_projection as projection
from replay import caption_continuation_verify as verifier
from replay import code_dependency_inventory as inventory
from replay import continuous_quiet_plan as quiet
from replay import retained_quiet_peaks as retained
from replay.audio_clock import AudioFrameClock
from replay.core import ReplayError
from replay.delivery_media import read_clock
from replay.quiet_cuts import quiet_cut_frames
from replay.recordings import record
from replay.stages import Stage, StageError, _fingerprint, file_digest, run_stage
from replay.waveform_quiet import measured_quiet_runs


@dataclass(frozen=True)
class Case:
    request: Path
    config: Path
    current: Path
    cuts: Path
    peaks: Path
    output: Path
    workflow: Path
    checkpoint: Path
    closure: Path
    stage: Stage
    model: Path
    native: Path
    settings: Path


def write(path: Path, data: object) -> dict[str, str]:
    path.write_text(json.dumps(data), encoding="utf-8")
    return {"path": str(path), "sha256": file_digest(path)}


def reference(path: Path) -> dict[str, str]:
    return {"path": str(path), "sha256": file_digest(path)}


@pytest.fixture(params=[16000, 44100])
def case(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, request: pytest.FixtureRequest) -> Case:
    monkeypatch.setattr(retained, "ROOT", tmp_path)
    parameter = request.param
    if isinstance(parameter, tuple) and len(parameter) == 2:
        rate, large_clock = parameter
    else:
        rate, large_clock = parameter, False
    if type(rate) is not int:
        raise ValueError("Expected synthetic measurement rate")
    work = tmp_path / "edits/work"
    work.mkdir(parents=True)
    mono, stereo = work / "mono.wav", work / "stereo.wav"
    for path, channels in ((mono, 1), (stereo, 2)):
        with wave.open(str(path), "wb") as stream:
            stream.setparams((channels, 2, rate, 0, "NONE", "not compressed"))
            stream.writeframes(b"\0" * (rate * 2 * channels * 2))
    audio, video = work / "audio.json", work / "video.json"
    write(
        audio,
        {
            "streams": [{"sample_rate": str(rate), "time_base": f"1/{rate}"}],
            "frames": [{"pts": 0, "nb_samples": rate}, {"pts": rate, "nb_samples": rate}],
            "format": {
                "tags": {
                    "syntheticProbeNote": "x" * (8 * 1024**2 + 1024) if large_clock else "synthetic"
                }
            },
        },
    )
    write(video, {"streams": [{"time_base": "1/30"}], "frames": [{"pts": 0, "duration": 60}]})
    projection, labels, lineage = (
        work / "projection.json",
        work / "labels.json",
        work / "lineage.json",
    )
    write(projection, {"artifacts": {"pcm": reference(mono), "audioClock": reference(audio)}})
    write(
        labels,
        {
            "clock": "source-presentation",
            "configSHA256": file_digest(projection),
            "parentSHA256": {str(mono): file_digest(mono)},
            "labels": [{"text": "声", "start": 0.9, "end": 1.1, "provenance": "native:0"}],
        },
    )
    write(
        lineage,
        {
            "clock": "continuous-decoded-pcm",
            "stereoPCM": reference(stereo),
            "monoPCM": reference(mono),
            "audioClock": reference(audio),
            "transforms": ["concatenate-decoded-frames", "duration-preserving-resample"],
        },
    )
    config = work / "quiet-config.json"
    data: dict[str, object] = {
        "version": 1,
        "fps": 30,
        "pcmClock": "continuous-decoded-pcm",
        "artifacts": {
            "labels": reference(labels),
            "stereoPCM": reference(stereo),
            "audioClock": reference(audio),
            "videoClock": reference(video),
            "projectionConfig": reference(projection),
            "lineageEvidence": reference(lineage),
        },
    }
    write(config, data)
    output_root = work / "quiet"
    output_root.mkdir()
    cuts, peaks = output_root / "cuts.json", output_root / "peaks.txt"
    current, current_labels = work / "current-config.json", work / "current-labels.json"
    updated = quiet._read(labels)
    updated["labels"] = [{"text": "声", "start": 0.7, "end": 1.2, "provenance": "native:1"}]
    write(current_labels, updated)
    current_data = dict(data)
    current_data["artifacts"] = {**record(data["artifacts"]), "labels": reference(current_labels)}
    write(current, current_data)
    sources = {str(row.path): row.sha256 for row in inventory.capture(retained.PURE_MODULES)}
    closure = work / "old-code.json"
    write(closure, {"inputs": sources})
    native, model, settings = work / "native", work / "model.bin", work / "settings.json"
    native.write_bytes(b"synthetic measured binary")
    model.write_bytes(b"synthetic pinned model")
    write(settings, {"policy": "synthetic unchanged settings"})
    artifacts = record(data["artifacts"])
    stage = Stage(
        name="measure-source-quiet",
        command=(
            sys.executable,
            "-B",
            "-m",
            "replay.continuous_quiet_plan",
            str(config),
            str(output_root),
            "--ffmpeg",
            str(native),
        ),
        inputs=(
            config,
            closure,
            model,
            native,
            settings,
            mono,
            *(Path(str(record(value)["path"])) for value in artifacts.values()),
            *(Path(name) for name in sources),
        ),
        outputs=(cuts, peaks),
        timeout_seconds=60,
        verify_command=(
            sys.executable,
            "-B",
            "-m",
            "replay.caption_continuation_verify",
            "quiet",
            str(config),
            str(cuts),
        ),
    )
    # sourceClock artifacts and mono are distinct, but the role list includes no mono.
    graph = work / "workflow.json"
    write(
        graph,
        {
            "version": 1,
            "stages": [
                {
                    "name": stage.name,
                    "command": list(stage.command),
                    "inputs": [str(path) for path in stage.inputs],
                    "outputs": [str(path) for path in stage.outputs],
                    "timeout_seconds": stage.timeout_seconds,
                    "verify_command": list(stage.verify_command),
                }
            ],
        },
    )
    state = work / "state"

    def execute(command: tuple[str, ...], log: Path, timeout: int) -> None:
        if command == stage.command:
            window = rate // 50
            peaks.write_text(
                "".join(
                    f"frame:{index} pts:{index * window} pts_time:{index * 0.02}\n"
                    "lavfi.astats.Overall.Peak_level=-inf\n"
                    for index in range(100)
                ),
                encoding="utf-8",
            )
            clock = AudioFrameClock.from_probe(quiet._read(audio))
            with peaks.open(encoding="utf-8") as lines:
                spans = measured_quiet_runs(
                    lines, sample_count=rate * 2, sample_rate=rate, window_samples=window
                )
            mapped = quiet.source_quiet(spans, clock, measurement_rate=rate)
            video_clock = read_clock(video)
            frames = ceil(
                (video_clock.pts[-1] + video_clock.last_duration) * video_clock.time_base * 30
            )
            selected = quiet_cut_frames(
                mapped, quiet._labels(quiet._read(labels)), source_frames=frames, fps=30
            )
            write(
                cuts,
                {
                    "version": 1,
                    "clock": "source-presentation",
                    "fps": 30,
                    "sourceFrames": frames,
                    "cuts": [asdict(cut) for cut in selected],
                    "sourceQuiet": [asdict(span) for span in mapped],
                    "parentSHA256": {
                        str(record(value)["path"]): record(value)["sha256"]
                        for value in artifacts.values()
                    },
                    "configSHA256": file_digest(config),
                    "peakSHA256": file_digest(peaks),
                    "plannerSHA256": file_digest(Path(quiet.__file__)),
                    "command": [str(native), "synthetic-measurement"],
                    "lineageIsCallerDeclared": True,
                    "nonSpeechRemovalVerified": False,
                    "humanReviewed": False,
                    "deliveryComplete": False,
                },
            )
        else:
            verifier.verify("quiet", config, cuts)
        log.write_bytes(
            b"genuine normal synthetic measurement and post-command domain check; no FFmpeg"
        )

    run_stage(stage, state_directory=state, execute=execute)
    (state / ".workflow.lock").write_bytes(b"")
    checkpoint = state / (stage.name + ".json")
    request_path = work / "request.json"
    target = output_root / "retained-cuts.json"
    write(
        request_path,
        {
            "version": 1,
            "kind": retained.KIND,
            "baseline": {
                "config": reference(config),
                "cuts": reference(cuts),
                "peaks": reference(peaks),
                "workflow": reference(graph),
                "checkpoint": reference(checkpoint),
                "readerClosure": reference(closure),
            },
            "currentConfig": reference(current),
            "output": str(target),
        },
    )
    return Case(
        request_path,
        config,
        current,
        cuts,
        peaks,
        target,
        graph,
        checkpoint,
        closure,
        stage,
        model,
        native,
        settings,
    )


def test_normal_retained_recompute_verify_and_nonoverwrite(case: Case) -> None:
    before = case.cuts.read_bytes()
    result = retained.run(case.request, sha256=file_digest(case.request))
    actual = quiet._read(result)
    assert actual["cuts"] == [
        {"source_start_frame": 18, "source_end_frame": 39, "output_start_frame": 0}
    ]
    assert record(actual["retainedDerivation"])["waveformMeasurementRepeated"] is False
    assert record(actual["retainedDerivation"])["historicalReaderEqualsCurrent"] is False
    assert case.cuts.read_bytes() == before
    assert retained.run(case.request, verify=True) == result
    with pytest.raises(FileExistsError):
        retained.run(case.request)


def test_historical_fingerprint_matches_canonical_stage(case: Case) -> None:
    inputs = {str(path): file_digest(path) for path in case.stage.inputs}
    assert retained._historical_fingerprint(case.stage, inputs) == _fingerprint(case.stage)


@pytest.mark.parametrize(
    "artifact", ["peaks", "config", "cuts", "checkpoint", "closure", "model", "native", "settings"]
)
def test_original_dependency_tampering_rejected(case: Case, artifact: str) -> None:
    paths = {
        "peaks": case.peaks,
        "config": case.config,
        "cuts": case.cuts,
        "checkpoint": case.checkpoint,
        "closure": case.closure,
        "model": case.model,
        "native": case.native,
        "settings": case.settings,
    }
    paths[artifact].write_bytes(b"changed original artifact")
    with pytest.raises((ValueError, ReplayError, StageError)):
        retained.run(case.request)
    assert not case.output.exists()


def test_request_pin_is_exact(case: Case) -> None:
    with pytest.raises(ReplayError, match="request pin changed"):
        retained.run(case.request, sha256="a" * 64)


@pytest.mark.parametrize(
    "field,value", [("fps", 60), ("version", True), ("pcmClock", "source-presentation")]
)
def test_current_policy_cannot_change(case: Case, field: str, value: object) -> None:
    data = quiet._read(case.current)
    data[field] = value
    write(case.current, data)
    request = quiet._read(case.request)
    request["currentConfig"] = reference(case.current)
    write(case.request, request)
    with pytest.raises(ReplayError):
        retained.run(case.request)


def test_old_stage_fingerprint_cannot_be_invented(case: Case) -> None:
    checkpoint = quiet._read(case.checkpoint)
    checkpoint["fingerprint"] = "a" * 64
    write(case.checkpoint, checkpoint)
    request = quiet._read(case.request)
    record(request["baseline"])["checkpoint"] = reference(case.checkpoint)
    write(case.request, request)
    with pytest.raises(ReplayError, match="genuine post-command"):
        retained.run(case.request)


def test_original_writer_lease_is_preserved(case: Case) -> None:
    lock = case.checkpoint.parent / ".workflow.lock"
    with lock.open("rb") as stream:
        fcntl.flock(stream, fcntl.LOCK_EX | fcntl.LOCK_NB)
        with pytest.raises(ValueError, match="active"):
            retained.run(case.request)


def test_current_derived_output_tamper_is_rejected(case: Case) -> None:
    retained.run(case.request)
    case.output.write_bytes(b"{}")
    with pytest.raises(ReplayError, match="derivation changed"):
        retained.run(case.request, verify=True)


def test_output_must_remain_adjacent_to_retained_peaks(case: Case) -> None:
    request = quiet._read(case.request)
    request["output"] = str(case.request.parent / "elsewhere.json")
    write(case.request, request)
    with pytest.raises(ReplayError, match="adjacent"):
        retained.run(case.request)


def test_duplicate_and_nonfinite_request_rejected(case: Case) -> None:
    case.request.write_bytes(b'{"version":1,"version":1,"value":NaN}')
    with pytest.raises((ValueError, ReplayError)):
        retained.run(case.request)


def test_normal_stage_definition_cannot_change(case: Case) -> None:
    graph = quiet._read(case.workflow)
    rows = graph["stages"]
    if not isinstance(rows, list):
        raise ValueError("Missing synthetic stages")
    record(rows[0])["command"] = ["invented-producer"]
    write(case.workflow, graph)
    request = quiet._read(case.request)
    record(request["baseline"])["workflow"] = reference(case.workflow)
    write(case.request, request)
    with pytest.raises(ReplayError, match="command or verifier"):
        retained.run(case.request)


def test_historical_algorithm_pin_cannot_be_replaced(case: Case) -> None:
    closure = quiet._read(case.closure)
    record(closure["inputs"])[str(Path(quiet.__file__))] = "a" * 64
    write(case.closure, closure)
    request = quiet._read(case.request)
    record(request["baseline"])["readerClosure"] = reference(case.closure)
    write(case.request, request)
    with pytest.raises(ReplayError, match="algorithms changed"):
        retained.run(case.request)


def test_non_source_cannot_enter_historical_reader_table(case: Case) -> None:
    write(case.closure, {"inputs": {str(case.model): file_digest(case.model)}})
    request = quiet._read(case.request)
    record(request["baseline"])["readerClosure"] = reference(case.closure)
    write(case.request, request)
    with pytest.raises(ReplayError, match="non-source pin"):
        retained.run(case.request)


def test_current_pcm_lineage_must_stay_same_clock(case: Case) -> None:
    config = quiet._read(case.current)
    lineage_path = case.current.parent / "changed-lineage.json"
    old_lineage = Path(str(record(record(config["artifacts"])["lineageEvidence"])["path"]))
    lineage = quiet._read(old_lineage)
    lineage["transforms"] = ["invented-stretch"]
    record(config["artifacts"])["lineageEvidence"] = write(lineage_path, lineage)
    write(case.current, config)
    request = quiet._read(case.request)
    request["currentConfig"] = reference(case.current)
    write(case.request, request)
    with pytest.raises(ReplayError, match="same decoded PCM lineage"):
        retained.run(case.request)


@pytest.mark.parametrize("version", [True, 2])
def test_request_version_requires_exact_integer(case: Case, version: object) -> None:
    request = quiet._read(case.request)
    request["version"] = version
    write(case.request, request)
    with pytest.raises(ReplayError, match="Invalid retained quiet request"):
        retained.run(case.request)


def test_cli_run_and_verify_use_exact_request_pin(
    case: Case, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.setattr(
        sys, "argv", ["retained", "run", str(case.request), "--sha256", file_digest(case.request)]
    )
    retained.main()
    assert str(case.output) in capsys.readouterr().out
    monkeypatch.setattr(
        sys,
        "argv",
        ["retained", "verify", str(case.request), "--sha256", file_digest(case.request)],
    )
    retained.main()
    assert str(case.output) in capsys.readouterr().out


def test_cli_invalid_pin_is_a_failure(case: Case, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(sys, "argv", ["retained", "run", str(case.request), "--sha256", "invalid"])
    with pytest.raises(SystemExit) as result:
        retained.main()
    assert result.value.code == 1
    assert not case.output.exists()


def test_path_and_reference_validation(case: Case) -> None:
    with pytest.raises(ReplayError, match="canonical absolute"):
        retained._path("relative")
    with pytest.raises(ReplayError, match="explicit canonical"):
        retained._path(1)
    with pytest.raises(ReplayError, match="checkout-local"):
        retained._path(str(case.request.parent.parent / "elsewhere.json"))
    with pytest.raises(ReplayError, match="exact retained artifact"):
        retained._reference({"path": str(case.model)}, {})
    with pytest.raises(ReplayError, match="Conflicting"):
        retained._pin({case.model: "a" * 64}, case.model, "b" * 64)


def test_missing_old_source_hash_cannot_use_current_bytes(case: Case) -> None:
    closure = quiet._read(case.closure)
    record(closure["inputs"]).pop(str(retained.PACKAGE / "stages.py"))
    write(case.closure, closure)
    request = quiet._read(case.request)
    record(request["baseline"])["readerClosure"] = reference(case.closure)
    write(case.request, request)
    with pytest.raises(ReplayError, match="historical source binding"):
        retained.run(case.request)


def test_new_projection_cannot_substitute_recognition_voice(case: Case) -> None:
    data = quiet._read(case.current)
    artifacts = record(data["artifacts"])
    old_projection = Path(str(record(artifacts["projectionConfig"])["path"]))
    projection = quiet._read(old_projection)
    original_mono = Path(str(record(record(projection["artifacts"])["pcm"])["path"]))
    other_mono = case.current.parent / "other-mono.wav"
    other_mono.write_bytes(original_mono.read_bytes())
    record(projection["artifacts"])["pcm"] = reference(other_mono)
    artifacts["projectionConfig"] = write(case.current.parent / "new-projection.json", projection)
    labels_path = Path(str(record(artifacts["labels"])["path"]))
    labels = quiet._read(labels_path)
    labels["configSHA256"] = record(artifacts["projectionConfig"])["sha256"]
    labels["parentSHA256"] = {str(other_mono): file_digest(other_mono)}
    artifacts["labels"] = write(labels_path, labels)
    old_lineage = Path(str(record(artifacts["lineageEvidence"])["path"]))
    lineage = quiet._read(old_lineage)
    lineage["monoPCM"] = reference(other_mono)
    artifacts["lineageEvidence"] = write(case.current.parent / "new-lineage.json", lineage)
    write(case.current, data)
    request = quiet._read(case.request)
    request["currentConfig"] = reference(case.current)
    write(case.request, request)
    with pytest.raises(ReplayError, match="mono PCM identity changed"):
        retained.run(case.request)


@pytest.mark.parametrize("case", [(16000, True), (44100, True)], indirect=True)
def test_audio_clock_role_keeps_existing_128mib_budget(case: Case) -> None:
    config = quiet._read(case.current)
    clock = Path(str(record(record(config["artifacts"])["audioClock"])["path"]))
    assert clock.stat().st_size > 8 * 1024**2
    with pytest.raises(ValueError, match="bounded regular metadata"):
        retained._read(clock)
    output = retained.run(case.request, sha256=file_digest(case.request))
    verifier.verify("quiet", case.current, output)
    assert quiet._read(output)["sourceFrames"] == 60


@pytest.mark.parametrize("case", [(16000, True)], indirect=True)
def test_large_audio_clock_tamper_remains_rejected(case: Case) -> None:
    config = quiet._read(case.current)
    clock = Path(str(record(record(config["artifacts"])["audioClock"])["path"]))
    with clock.open("ab") as stream:
        stream.write(b" ")
    with pytest.raises(ReplayError, match="input bytes changed"):
        retained.run(case.request)
    assert not case.output.exists()


def test_audio_clock_reader_oversize_and_bad_domain(
    case: Case, monkeypatch: pytest.MonkeyPatch
) -> None:
    job = retained._load(case.request, None)
    data, paths = retained._config(job.current, job.pins)
    # Lower only the existing role reader's limit for a bounded oversize fixture.
    monkeypatch.setattr(projection, "MAX_JSON_BYTES", 1)
    with pytest.raises(ReplayError, match="byte budget"):
        retained._derive(job, data, paths, {})
    monkeypatch.setattr(projection, "MAX_JSON_BYTES", 128 * 1024**2)
    write(
        paths["audioClock"],
        {
            "streams": [{"sample_rate": "16000", "time_base": "1/16000"}],
            "frames": [{"pts": 0, "nb_samples": 0}],
        },
    )
    with pytest.raises(ValueError, match="native decoded-audio clock"):
        retained._derive(job, data, paths, {})
