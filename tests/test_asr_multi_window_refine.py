"""Independent CPU observations retain native clocks and exact zero-gap evidence."""

import json
import wave
from pathlib import Path

import pytest

from replay import asr_multi_window_refine as multi
from replay import asr_window_refine as single
from replay.asr_window import extract
from replay.stages import StageError, file_digest


def write(path: Path, value: object) -> dict[str, str]:
    path.write_text(json.dumps(value), encoding="utf-8")
    return {"path": str(path), "sha256": file_digest(path)}


@pytest.fixture
def config(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    source = tmp_path / "source.wav"
    with wave.open(str(source), "wb") as audio:
        audio.setparams((1, 2, 16000, 0, "NONE", "not compressed"))
        audio.writeframes(b"\0\0" * 64000)
    model = tmp_path / "model"
    model.mkdir()
    (model / "weights.npz").write_bytes(b"synthetic")
    (model / "config.json").write_text("{}")
    monkeypatch.setattr(single, "DEFAULT_MODEL", model)
    monkeypatch.setattr(single, "DEFAULT_WEIGHTS_SHA256", file_digest(model / "weights.npz"))
    monkeypatch.setattr(single, "DEFAULT_CONFIG_SHA256", file_digest(model / "config.json"))
    model_pins = {str(path): file_digest(path) for path in model.iterdir()}
    original_inputs = write(
        tmp_path / "original-inputs.json",
        {"inputs": {str(source): file_digest(source), **model_pins}, "samples": 64000},
    )
    original = write(
        tmp_path / "original-labels.json",
        {
            "version": 1,
            "clock": "input-pcm-not-projected",
            "labels": [
                {"text": "旧", "start": 1, "end": 2.4, "provenance": "old:0"},
                {"text": "次", "start": 3, "end": 3.2, "provenance": "old:1"},
            ],
            "parentSHA256": {original_inputs["path"]: original_inputs["sha256"]},
        },
    )
    windows: list[object] = []
    for index, start in enumerate((4000, 32000)):
        directory = tmp_path / f"window-{index}"
        clip = extract(source, directory, start_sample=start, sample_count=8000)
        onset = 0 if index == 0 else 0.1
        raw = write(
            directory / "raw.json",
            {
                "text": "声",
                "language": "ja",
                "segments": [
                    {
                        "text": "声",
                        "start": onset,
                        "end": 0.3,
                        "words": [{"word": "声", "start": onset, "end": 0.3, "probability": 0.9}],
                    }
                ],
            },
        )
        clock = write(
            directory / "clock.json",
            {
                "startSample": 0,
                "endSample": 8000,
                "sampleRate": 16000,
                "humanReviewed": False,
                "emptyRecognitionIsNotSilence": True,
                "rawSHA256": raw["sha256"],
            },
        )
        complete = write(
            directory / "complete.json",
            {"rawSHA256": [raw["sha256"]], "windows": 1, "deliveryComplete": False},
        )
        inputs = write(
            directory / "inputs.json",
            {
                "version": 1,
                "samples": 8000,
                "sampleRate": 16000,
                "windowSamples": 960000,
                "strideSamples": 480000,
                "clock": "input-pcm-not-projected",
                "device": "cpu",
                "dtypeContract": "provider-default-fp16-unchanged",
                "inputs": {str(clip): file_digest(clip), **model_pins},
            },
        )
        windows.append(
            {
                "raw": raw,
                "clock": clock,
                "complete": complete,
                "inputs": inputs,
                "audio": {"path": str(clip), "sha256": file_digest(clip)},
                "window": {
                    "path": str(directory / "window.json"),
                    "sha256": file_digest(directory / "window.json"),
                },
            }
        )
    prior = write(tmp_path / "prior-observation.json", {"retained": True})
    path = tmp_path / "config.json"
    write(
        path,
        {
            "version": 1,
            "artifacts": {"labels": original, "originalInputs": original_inputs},
            "windows": windows,
            "replacement": {
                "firstLabel": 0,
                "stopLabel": 1,
                "startSample": 4000,
                "endSample": 40000,
            },
            "priorObservations": [prior],
        },
    )
    return path


def test_candidate_exact_native_clocks_uncertainty_and_zero_gap(
    config: Path, tmp_path: Path
) -> None:
    original_path = tmp_path / "original-labels.json"
    before = original_path.read_bytes()
    output = tmp_path / "candidate.json"
    assert multi.run(config, output) == 3
    assert multi.verify(config, output) == 3
    document = json.loads(output.read_text())
    assert document["labels"] == [
        {"text": "声", "start": 0.25, "end": 0.55, "provenance": "000000004000:0"},
        {"text": "声", "start": 2.1, "end": 2.3, "provenance": "000000032000:0"},
        {"text": "次", "start": 3, "end": 3.2, "provenance": "old:1"},
    ]
    assert original_path.read_bytes() == before
    assert document["candidateOnly"] is True
    assert document["speechAccuracyVerified"] is False
    refinement = document["refinement"]
    assert refinement["unrecognizedGaps"] == [
        {
            "startSample": 12000,
            "endSample": 32000,
            "samples": 20000,
            "exactDigitalZero": True,
            "ASRObservedThisGap": False,
        }
    ]
    assert refinement["uncertainOnsets"][0]["start"] == 0.25
    assert refinement["nativeClocksAdjusted"] is False
    assert multi.main([str(config), str(output), "--verify"]) == 0
    with pytest.raises(ValueError, match="fresh"):
        multi.run(config, output)
    output.write_text("{}")
    with pytest.raises(ValueError, match="differs"):
        multi.verify(config, output)


@pytest.mark.parametrize(
    "mutation",
    ["schema", "base", "parent", "windows", "prior", "roles", "coordinates", "envelope", "source"],
)
def test_corrupt_contract_rejected(config: Path, tmp_path: Path, mutation: str) -> None:
    document = json.loads(config.read_text())
    if mutation == "schema":
        document["version"] = 2
    elif mutation == "base":
        document["artifacts"].pop("labels")
    elif mutation == "parent":
        original = tmp_path / "original-labels.json"
        data = json.loads(original.read_text())
        data["parentSHA256"] = {}
        document["artifacts"]["labels"] = write(original, data)
    elif mutation == "windows":
        document["windows"] = []
    elif mutation == "prior":
        document["priorObservations"] = None
    elif mutation == "roles":
        document["windows"][0].pop("raw")
    elif mutation == "coordinates":
        document["replacement"]["startSample"] = True
    elif mutation == "envelope":
        document["replacement"]["endSample"] -= 1
    else:
        (tmp_path / "window-0/window.wav").write_bytes(b"corrupt")
    write(config, document)
    with pytest.raises((ValueError, StageError)):
        multi.run(config, tmp_path / "bad.json")


def test_exact_gap_rejects_nonzero_samples(config: Path, tmp_path: Path) -> None:
    source = tmp_path / "source.wav"
    data = bytearray(source.read_bytes())
    data[44 + 16000 * 2] = 1
    source.write_bytes(data)
    # Source/clip bindings remain truthful after the synthetic gap modification.
    document = json.loads(config.read_text())
    original_inputs = tmp_path / "original-inputs.json"
    inputs = json.loads(original_inputs.read_text())
    inputs["inputs"][str(source)] = file_digest(source)
    original_ref = write(original_inputs, inputs)
    document["artifacts"]["originalInputs"] = original_ref
    original = tmp_path / "original-labels.json"
    labels = json.loads(original.read_text())
    labels["parentSHA256"][original_ref["path"]] = original_ref["sha256"]
    document["artifacts"]["labels"] = write(original, labels)
    for index in range(2):
        window_path = tmp_path / f"window-{index}/window.json"
        window = json.loads(window_path.read_text())
        window["sourceSHA256"] = file_digest(source)
        document["windows"][index]["window"] = write(window_path, window)
    write(config, document)
    with pytest.raises(ValueError, match="digital zero"):
        multi.run(config, tmp_path / "bad.json")
