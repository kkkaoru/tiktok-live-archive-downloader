import json
import wave
from pathlib import Path

import pytest

from replay.clock_contract import _artifact_binding, _rational, _wave, main, verify
from replay.core import ReplayError
from replay.stages import StageError, file_digest


def write(path: Path, value: object) -> Path:
    path.write_text(json.dumps(value), encoding="utf-8")
    return path


def bind(path: Path) -> dict[str, str]:
    return {"path": str(path), "sha256": file_digest(path)}


def wav(path: Path, *, samples: int = 800, width: int = 2) -> Path:
    with wave.open(str(path), "wb") as output:
        output.setparams((2, width, 8000, 0, "NONE", "not compressed"))
        output.writeframes(b"\0" * samples * 2 * width)
    return path


def repin(config: Path, role: str, value: object) -> None:
    document = json.loads(config.read_text(encoding="utf-8"))
    path = Path(document["artifacts"][role]["path"])
    write(path, value)
    document["artifacts"][role] = bind(path)
    write(config, document)


@pytest.fixture
def tempo(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    monkeypatch.chdir(tmp_path)
    (tmp_path / "edits/work").mkdir(parents=True)
    clock = write(
        tmp_path / "clock.json",
        {
            "streams": [{"sample_rate": "8000", "time_base": "1/8000"}],
            "frames": [{"pts": 0, "nb_samples": 800}],
        },
    )
    source = wav(tmp_path / "source.wav")
    output = wav(tmp_path / "output.wav")
    cuts = write(
        tmp_path / "cuts.json",
        {
            "fps": 30,
            "clock": "source-presentation",
            "cuts": [{"source_start_frame": 0, "source_end_frame": 3, "output_start_frame": 0}],
            "parentSHA256": {str(clock): file_digest(clock), str(source): file_digest(source)},
        },
    )
    producer = write(
        tmp_path / "producer.json",
        {
            "coordinateSpace": "native-presentation-seconds",
            "source": bind(source),
            "clock": bind(clock),
            "intervals": [
                {"start": {"value": 0, "timescale": 30}, "end": {"value": 3, "timescale": 30}}
            ],
        },
    )
    receipt = write(
        tmp_path / "producer-receipt.json",
        {
            "kind": "context-atempo-exact-cut-slot-candidate",
            "inputs": {
                str(producer): file_digest(producer),
                str(source): file_digest(source),
                str(clock): file_digest(clock),
                str(cuts): file_digest(cuts),
            },
            "output": str(output),
            "outputSHA256": file_digest(output),
            "sampleRate": 8000,
            "channels": 2,
            "outputStartSample": 0,
            "outputEndSample": 800,
            "outputSamples": 800,
            "maximumRateChangeAllowed": 0.5,
            "cuts": [
                {
                    "source_start_sample": 0,
                    "source_end_sample": 800,
                    "output_start_sample": 0,
                    "output_end_sample": 800,
                }
            ],
        },
    )
    return write(
        tmp_path / "config.json",
        {
            "version": 1,
            "operation": "tempo-cut-audio",
            "previousApplications": 0,
            "inputClock": "continuous-decoded-pcm",
            "outputClock": "edited-cut-pcm",
            "artifacts": {
                "producerConfig": bind(producer),
                "producerReceipt": bind(receipt),
                "inputAudio": bind(source),
                "outputAudio": bind(output),
                "audioClock": bind(clock),
                "cuts": bind(cuts),
            },
        },
    )


@pytest.fixture
def recognition(tempo: Path) -> Path:
    root = tempo.parent
    original = write(
        root / "input-labels.json",
        {
            "version": 1,
            "clock": "input-pcm-not-projected",
            "labels": [{"text": "声", "start": 0.01, "end": 0.05, "provenance": "raw:0"}],
        },
    )
    clock = root / "clock.json"
    config = write(
        root / "projection.json",
        {
            "inputClock": "continuous-decoded-pcm",
            "transforms": ["concatenate-decoded-frames", "duration-preserving-resample"],
            "artifacts": {"labels": bind(original), "audioClock": bind(clock)},
        },
    )
    projected = write(
        root / "source-labels.json",
        {
            "clock": "source-presentation",
            "configSHA256": file_digest(config),
            "parentSHA256": {str(original): file_digest(original), str(clock): file_digest(clock)},
            "labels": [{"text": "声", "start": 0.01, "end": 0.05, "provenance": "raw:0"}],
        },
    )
    return write(
        root / "recognition-contract.json",
        {
            "version": 1,
            "operation": "recognition-source",
            "previousApplications": 0,
            "inputClock": "continuous-decoded-pcm",
            "outputClock": "source-presentation",
            "artifacts": {
                "producerConfig": bind(config),
                "inputLabels": bind(original),
                "outputLabels": bind(projected),
                "audioClock": bind(clock),
            },
        },
    )


def test_tempo_receipt_and_fresh_reuse_check(
    tempo: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    output = tempo.parent / "edits/work/contract.json"
    assert main([str(tempo), "--output", str(output)]) == 0
    assert main([str(tempo), "--check", str(output)]) == 0
    result = json.loads(output.read_text(encoding="utf-8"))
    assert result["inputClock"] == "continuous-decoded-pcm"
    assert result["outputClock"] == "edited-cut-pcm"
    assert result["outputSamples"] == 800
    assert result["outputFrames"] == 3
    assert result["applicationCount"] == 1
    assert result["cutApplications"] == 1
    assert result["AVSyncVerified"] is False
    assert main([str(tempo), "--inspect"]) == 0
    inspection = json.loads(capsys.readouterr().out)
    assert inspection["inspectionAnchors"] == [
        {
            "cutIndex": 0,
            "sourceStart": "0",
            "sourceEnd": "1/10",
            "inputStartSample": 0,
            "inputEndSample": 800,
            "outputStartSample": 0,
            "outputEndSample": 800,
        }
    ]
    result["outputSamples"] = 799
    write(output, result)
    with pytest.raises(ReplayError, match="cannot be reused"):
        main([str(tempo), "--check", str(output)])


def test_recognition_recomputes_exact_native_clock(recognition: Path) -> None:
    result = verify(recognition)
    assert result["labelCount"] == 1
    assert result["sourceAudioFirstTime"] == "0"
    assert result["sourceAudioLastTime"] == "1/10"
    assert result["cutApplications"] == 0
    assert result["outputClock"] == "source-presentation"


@pytest.mark.parametrize(
    "field,value",
    [
        ("previousApplications", 1),
        ("previousApplications", True),
        ("operation", "shift"),
        ("version", 2),
        ("inputClock", "source-presentation"),
        ("outputClock", "continuous-decoded-pcm"),
    ],
)
def test_rejects_double_application_and_unknown_contract(
    tempo: Path, field: str, value: object
) -> None:
    config = json.loads(tempo.read_text(encoding="utf-8"))
    config[field] = value
    write(tempo, config)
    with pytest.raises(ReplayError, match="unapplied"):
        verify(tempo)


def test_rejects_changed_audio_pin(tempo: Path) -> None:
    wav(tempo.parent / "source.wav", samples=799)
    with pytest.raises(StageError):
        verify(tempo)


def test_conflicting_role_pins_are_rejected(tempo: Path) -> None:
    config = json.loads(tempo.read_text(encoding="utf-8"))
    config["artifacts"]["outputAudio"] = {
        "path": str(tempo.parent / "source.wav"),
        "sha256": "0" * 64,
    }
    write(tempo, config)
    with pytest.raises(StageError, match="Conflicting"):
        verify(tempo)


def test_rejects_other_cut_revision(tempo: Path) -> None:
    data = json.loads((tempo.parent / "cuts.json").read_text(encoding="utf-8"))
    data["cuts"][0]["source_end_frame"] = 2
    repin(tempo, "cuts", data)
    with pytest.raises(ReplayError, match="cut revision") as failure:
        verify(tempo)
    assert "continuous-decoded-pcm" in str(failure.value)
    assert "edited-cut-pcm" in str(failure.value)
    assert str(tempo.parent / "source.wav") in str(failure.value)
    assert str(tempo.parent / "output.wav") in str(failure.value)


def test_rejects_cut_offset_and_other_audio_lineage(tempo: Path) -> None:
    data = json.loads((tempo.parent / "cuts.json").read_text(encoding="utf-8"))
    data["cuts"][0]["output_start_frame"] = 1
    repin(tempo, "cuts", data)
    with pytest.raises(ReplayError, match="contiguous"):
        verify(tempo)
    data["cuts"][0]["output_start_frame"] = 0
    data["parentSHA256"] = {}
    repin(tempo, "cuts", data)
    with pytest.raises(ReplayError, match="separated audio"):
        verify(tempo)


@pytest.mark.parametrize(
    "field,value,match",
    [
        ("coordinateSpace", "edited-cut-pcm", "native cuts"),
        ("intervals", None, "selection"),
    ],
)
def test_invalid_tempo_config(tempo: Path, field: str, value: object, match: str) -> None:
    data = json.loads((tempo.parent / "producer.json").read_text(encoding="utf-8"))
    data[field] = value
    repin(tempo, "producerConfig", data)
    with pytest.raises(ReplayError, match=match):
        verify(tempo)


@pytest.mark.parametrize(
    "field,value,match",
    [
        ("inputs", {}, "exact input"),
        ("maximumRateChangeAllowed", True, "rate-change"),
        ("cuts", [], "every cut"),
        ("cuts", [{"source_start_sample": 1}], "placements"),
        ("outputSamples", 799, "full clock"),
        ("outputStartSample", 1, "full clock"),
        ("outputSHA256", "0" * 64, "full clock"),
    ],
)
def test_invalid_tempo_receipt(tempo: Path, field: str, value: object, match: str) -> None:
    data = json.loads((tempo.parent / "producer-receipt.json").read_text(encoding="utf-8"))
    data[field] = value
    repin(tempo, "producerReceipt", data)
    with pytest.raises(ReplayError, match=match):
        verify(tempo)


def test_wrong_output_sample_count_even_when_hash_rebound(tempo: Path) -> None:
    path = wav(tempo.parent / "output.wav", samples=799)
    config = json.loads(tempo.read_text(encoding="utf-8"))
    config["artifacts"]["outputAudio"] = bind(path)
    write(tempo, config)
    with pytest.raises(ReplayError, match="full clock"):
        verify(tempo)


def test_recognition_rejects_preprojected_input_and_changed_timestamps(recognition: Path) -> None:
    output = recognition.parent / "source-labels.json"
    data = json.loads(output.read_text(encoding="utf-8"))
    data["labels"][0]["start"] = 0.02
    repin(recognition, "outputLabels", data)
    with pytest.raises(ReplayError, match="exact single"):
        verify(recognition)
    config = json.loads((recognition.parent / "projection.json").read_text(encoding="utf-8"))
    config["inputClock"] = "source-presentation"
    repin(recognition, "producerConfig", config)
    with pytest.raises(ReplayError, match="unprojected"):
        verify(recognition)


def test_schema_double_projection_is_rejected(recognition: Path) -> None:
    source = recognition.parent / "input-labels.json"
    original = json.loads(source.read_text(encoding="utf-8"))
    original["clock"] = "source-presentation"
    repin(recognition, "inputLabels", original)
    producer = json.loads((recognition.parent / "projection.json").read_text(encoding="utf-8"))
    producer["artifacts"]["labels"] = bind(source)
    repin(recognition, "producerConfig", producer)
    with pytest.raises(ReplayError, match="unprojected assembled"):
        verify(recognition)


def test_nonzero_source_origin_is_piecewise_verified(recognition: Path) -> None:
    clock = recognition.parent / "clock.json"
    data = json.loads(clock.read_text(encoding="utf-8"))
    data["frames"][0]["pts"] = 2000
    repin(recognition, "audioClock", data)
    producer = json.loads((recognition.parent / "projection.json").read_text(encoding="utf-8"))
    producer["artifacts"]["audioClock"] = bind(clock)
    repin(recognition, "producerConfig", producer)
    output = json.loads((recognition.parent / "source-labels.json").read_text(encoding="utf-8"))
    output["parentSHA256"][str(clock)] = file_digest(clock)
    output["configSHA256"] = file_digest(recognition.parent / "projection.json")
    output["labels"][0]["start"] = 0.26
    output["labels"][0]["end"] = 0.3
    repin(recognition, "outputLabels", output)
    result = verify(recognition)
    assert result["sourceAudioFirstTime"] == "1/4"
    assert result["sourceAudioLastTime"] == "7/20"
    assert result["inputOrigin"] == "0"
    assert result["outputOrigin"] == "0"


def test_different_output_sample_rate_is_not_reused(tempo: Path) -> None:
    path = tempo.parent / "output.wav"
    with wave.open(str(path), "wb") as stream:
        stream.setparams((2, 2, 16000, 0, "NONE", "not compressed"))
        stream.writeframes(b"\0" * 3200)
    config = json.loads(tempo.read_text(encoding="utf-8"))
    config["artifacts"]["outputAudio"] = bind(path)
    write(tempo, config)
    receipt = json.loads((tempo.parent / "producer-receipt.json").read_text(encoding="utf-8"))
    receipt["outputSHA256"] = file_digest(path)
    repin(tempo, "producerReceipt", receipt)
    with pytest.raises(ReplayError, match="full clock"):
        verify(tempo)


def test_swapped_raw_and_projected_artifacts_are_not_reused(recognition: Path) -> None:
    config = json.loads(recognition.read_text(encoding="utf-8"))
    config["artifacts"]["inputLabels"], config["artifacts"]["outputLabels"] = (
        config["artifacts"]["outputLabels"],
        config["artifacts"]["inputLabels"],
    )
    write(recognition, config)
    with pytest.raises(ReplayError, match="unprojected"):
        verify(recognition)


def test_rejects_external_paths_and_bad_artifact_schema(tempo: Path) -> None:
    link = tempo.parent / "linked.wav"
    link.symlink_to(tempo.parent / "source.wav")
    with pytest.raises(ReplayError, match="canonical"):
        _artifact_binding({"path": str(link), "sha256": "0" * 64})
    with pytest.raises(ReplayError, match="explicit"):
        _artifact_binding({"path": 3, "sha256": "bad"})
    with pytest.raises(ReplayError, match="rational"):
        _rational({"value": 1, "timescale": False})
    with pytest.raises(ReplayError, match="PCM16"):
        _wave(wav(tempo.parent / "byte.wav", width=1))


def test_explicit_readonly_media_root(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    checkout = tmp_path / "checkout"
    checkout.mkdir()
    media = tmp_path / "media"
    media.mkdir()
    audio = wav(media / "input.wav")
    monkeypatch.chdir(checkout)
    with pytest.raises(ReplayError, match="canonical"):
        _artifact_binding(bind(audio))
    assert _artifact_binding(bind(audio), media_root=media) == (audio, file_digest(audio))
    with pytest.raises(ValueError, match="explicit source root"):
        _artifact_binding(bind(audio), media_root=checkout)
    link = media / "link.wav"
    link.symlink_to(audio)
    with pytest.raises((ValueError, ReplayError)):
        _artifact_binding({"path": str(link), "sha256": file_digest(audio)}, media_root=media)


def test_media_root_must_be_canonical(tempo: Path) -> None:
    with pytest.raises(ReplayError, match="Media root"):
        verify(tempo, media_root=Path("relative"))
    result = verify(tempo, media_root=tempo.parent)
    assert result["authorizedReadOnlyMediaRoot"] == str(tempo.parent)


@pytest.fixture
def legacy_tempo(tempo: Path) -> Path:
    cuts = tempo.parent / "cuts.json"
    data = json.loads(cuts.read_text())
    data.pop("fps")
    data.pop("clock")
    data.pop("parentSHA256")
    data.update(grid="historical descriptive grid", audioSHA256="a" * 64, inputs={})
    repin(tempo, "cuts", data)
    producer = tempo.parent / "producer.json"
    data = json.loads(producer.read_text())
    data["evidence"] = [bind(cuts)]
    repin(tempo, "producerConfig", data)
    receipt = tempo.parent / "producer-receipt.json"
    data = json.loads(receipt.read_text())
    data["inputs"][str(producer)] = file_digest(producer)
    data["inputs"][str(cuts)] = file_digest(cuts)
    repin(tempo, "producerReceipt", data)
    data = json.loads(tempo.read_text())
    data["legacyCutBinding"] = {
        "schema": "protected-frame-plan-completed-selection",
        "frameRate": 30,
    }
    write(tempo, data)
    return tempo


def test_completed_selection_binds_legacy_cut_schema(legacy_tempo: Path) -> None:
    result = verify(legacy_tempo)
    assert result["cutClockBinding"] == "completed-selection-and-receipt"
    assert result["historicalCutAudioSHA256"] == "a" * 64
    assert result["outputSamples"] == 800
    assert result["AVSyncVerified"] is False


def test_legacy_grid_cannot_be_guessed(legacy_tempo: Path) -> None:
    data = json.loads(legacy_tempo.read_text())
    data["legacyCutBinding"]["frameRate"] = 25
    write(legacy_tempo, data)
    with pytest.raises(ReplayError, match="timescales"):
        verify(legacy_tempo)


def test_legacy_cut_requires_producer_evidence(legacy_tempo: Path) -> None:
    producer = legacy_tempo.parent / "producer.json"
    data = json.loads(producer.read_text())
    data["evidence"] = []
    repin(legacy_tempo, "producerConfig", data)
    with pytest.raises(ReplayError, match="completed-selection binding"):
        verify(legacy_tempo)
