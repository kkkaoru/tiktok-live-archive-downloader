import json
import wave
from fractions import Fraction
from pathlib import Path

import pytest

from replay.asr_clock_projection import main, project_labels, run
from replay.audio_clock import AudioFrameClock
from replay.caption_repair import ClockLabel
from replay.core import ReplayError
from replay.stages import file_digest


def write(path: Path, value: object) -> Path:
    path.write_text(json.dumps(value), encoding="utf-8")
    return path


def bind(path: Path) -> dict[str, str]:
    return {"path": str(path), "sha256": file_digest(path)}


@pytest.fixture
def config(tmp_path: Path) -> Path:
    pcm = tmp_path / "pcm.wav"
    with wave.open(str(pcm), "wb") as stream:
        stream.setparams((1, 2, 16000, 0, "NONE", "not compressed"))
        stream.writeframes(b"\0" * 640)
    clock = write(
        tmp_path / "clock.json",
        {
            "streams": [{"sample_rate": "16000", "time_base": "1/16000"}],
            "frames": [{"pts": 11808, "nb_samples": 160}, {"pts": 12000, "nb_samples": 160}],
        },
    )
    manifest = write(
        tmp_path / "inputs.json",
        {
            "clock": "input-pcm-not-projected",
            "inputs": {str(pcm): file_digest(pcm)},
            "samples": 320,
            "sampleRate": 16000,
        },
    )
    labels = write(
        tmp_path / "labels.json",
        {
            "version": 1,
            "clock": "input-pcm-not-projected",
            "parentSHA256": {str(manifest): file_digest(manifest)},
            "labels": [{"text": "声", "start": 0.005, "end": 0.015, "provenance": "window:0"}],
        },
    )
    clock_receipt = write(
        tmp_path / "clock-receipt.json",
        {
            "status": "complete",
            "outputs": {str(clock): file_digest(clock)},
        },
    )
    pcm_receipt = write(
        tmp_path / "pcm-receipt.json",
        {
            "status": "complete",
            "outputs": {str(pcm): file_digest(pcm)},
        },
    )
    lineage = write(tmp_path / "lineage.json", {"test": "synthetic duration-preserving input"})
    return write(
        tmp_path / "config.json",
        {
            "version": 1,
            "inputClock": "continuous-decoded-pcm",
            "transforms": ["concatenate-decoded-frames", "duration-preserving-resample"],
            "artifacts": {
                "labels": bind(labels),
                "pcm": bind(pcm),
                "audioClock": bind(clock),
                "asrInputs": bind(manifest),
                "clockReceipt": bind(clock_receipt),
                "pcmReceipt": bind(pcm_receipt),
                "lineageEvidence": bind(lineage),
            },
        },
    )


def test_cli_projects_gap_with_nonzero_origin_and_preserves_raw(config: Path) -> None:
    original = (config.parent / "labels.json").read_bytes()
    output = config.parent / "projected.json"
    assert main([str(config), str(output)]) == 0
    result = json.loads(output.read_text())
    assert result["labels"] == [
        {"text": "声", "start": 0.743, "end": 0.755, "provenance": "window:0"}
    ]
    assert result["clock"] == "source-presentation"
    assert result["deliveryComplete"] is False
    assert (config.parent / "labels.json").read_bytes() == original


def test_point_and_boundary_sides_are_distinct() -> None:
    clock = AudioFrameClock((7380, 8000), (100, 100), 10000, Fraction(1, 10000))
    labels = (ClockLabel("終", 0, 0.01, "a"), ClockLabel("点", 0.01, 0.01, "b"))
    result = project_labels(labels, clock)
    assert result[0].end == 0.748
    assert result[1].start == result[1].end == 0.8


def test_overlap_rejects_inverted_interval() -> None:
    clock = AudioFrameClock((0, 1), (100, 100), 10000, Fraction(1, 10000))
    with pytest.raises(ValueError, match="overlap"):
        project_labels((ClockLabel("声", 0.009, 0.011, "a"),), clock)


def test_onset_reversal_is_not_sorted_or_repaired() -> None:
    clock = AudioFrameClock((0,), (1000,), 10000, Fraction(1, 10000))
    with pytest.raises(ReplayError, match="onsets"):
        project_labels((ClockLabel("後", 0.05, 0.06, "b"), ClockLabel("前", 0, 0.01, "a")), clock)


@pytest.mark.parametrize(
    "field,value",
    [
        ("version", 2),
        ("inputClock", "source-presentation"),
        ("transforms", ["pad"]),
        ("artifacts", {}),
    ],
)
def test_contract_rejected(config: Path, field: str, value: object) -> None:
    data = json.loads(config.read_text())
    data[field] = value
    write(config, data)
    with pytest.raises(ReplayError):
        run(config, config.parent / "out.json")


@pytest.mark.parametrize(
    "name,field,value",
    [
        ("labels", "clock", "source-presentation"),
        ("labels", "labels", {}),
        ("labels", "labels", [{"text": "声", "start": True, "end": 0.01, "provenance": "a"}]),
        ("labels", "labels", [{"text": 1}]),
        ("labels", "parentSHA256", {}),
        ("asrInputs", "inputs", {}),
        ("asrInputs", "clock", "source-presentation"),
        ("asrInputs", "samples", 321),
        ("clockReceipt", "status", "running"),
        ("pcmReceipt", "outputs", {}),
    ],
)
def test_bad_bound_evidence(config: Path, name: str, field: str, value: object) -> None:
    data = json.loads(config.read_text())
    path = Path(data["artifacts"][name]["path"])
    document = json.loads(path.read_text())
    document[field] = value
    write(path, document)
    data["artifacts"][name] = bind(path)
    if name == "asrInputs":
        labels = config.parent / "labels.json"
        assembled = json.loads(labels.read_text())
        assembled["parentSHA256"] = {str(path): file_digest(path)}
        write(labels, assembled)
        data["artifacts"]["labels"] = bind(labels)
    write(config, data)
    with pytest.raises(ReplayError):
        run(config, config.parent / "out.json")


def test_existing_output_is_preserved(config: Path) -> None:
    output = write(config.parent / "out.json", {"keep": True})
    with pytest.raises(ReplayError, match="new output"):
        run(config, output)
    assert json.loads(output.read_text()) == {"keep": True}


def test_malformed_binding(config: Path) -> None:
    data = json.loads(config.read_text())
    data["artifacts"]["pcm"] = {"path": 5}
    write(config, data)
    with pytest.raises(ReplayError, match="artifact"):
        run(config, config.parent / "out.json")
