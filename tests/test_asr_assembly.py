import json
import runpy
import sys
import wave
from pathlib import Path

import numpy as np
import pytest
from pytest import MonkeyPatch

from replay import asr_assembly, batch_asr
from replay.audio_windows import AudioWindow
from replay.caption_repair import ClockLabel
from replay.core import ReplayError
from replay.local_asr import fingerprint


@pytest.fixture
def collection(tmp_path: Path, monkeypatch: MonkeyPatch) -> Path:
    source = tmp_path / "source.wav"
    with wave.open(str(source), "wb") as audio:
        audio.setparams((1, 2, 16000, 0, "NONE", "not compressed"))
        audio.writeframes(b"\0\0" * (61 * 16000))
    model = tmp_path / "model"
    model.mkdir()
    (model / "weights.npz").write_bytes(b"synthetic")
    (model / "config.json").write_text("{}", encoding="utf-8")

    def predict(samples: object, model: Path) -> object:
        assert isinstance(samples, np.ndarray)
        start = 1 if len(samples) == 960000 else 15
        return {
            "language": "ja",
            "text": "声点",
            "segments": [
                {
                    "start": start,
                    "end": start + 3,
                    "text": "声点",
                    "words": [
                        {"word": "声", "start": start, "end": start + 1, "probability": 0.9},
                        {"word": "点", "start": start + 3, "end": start + 3, "probability": 0.8},
                    ],
                }
            ],
        }

    monkeypatch.setattr(batch_asr, "_predict", predict)
    output = tmp_path / "raw"
    batch_asr.transcribe_windows(
        source,
        output,
        model=model,
        config_sha256=fingerprint(model / "config.json"),
        weights_sha256=fingerprint(model / "weights.npz"),
    )
    return output


def test_complete_collection_preserves_points_and_provenance(
    collection: Path, tmp_path: Path
) -> None:
    output = tmp_path / "assembled.json"
    assert asr_assembly.assemble(collection, output) == 4
    saved = json.loads(output.read_text(encoding="utf-8"))
    assert saved["labels"] == [
        {"text": "声", "start": 1.0, "end": 2.0, "provenance": "000000000000:0"},
        {"text": "点", "start": 4.0, "end": 4.0, "provenance": "000000000000:1"},
        {"text": "声", "start": 45.0, "end": 46.0, "provenance": "000000480000:0"},
        {"text": "点", "start": 48.0, "end": 48.0, "provenance": "000000480000:1"},
    ]
    assert saved["unresolvedPoints"] == 2
    assert saved["emptyWindows"] == 0
    assert saved["deliveryComplete"] is False
    assert saved["clock"] == "input-pcm-not-projected"
    assert output.stat().st_mode & 0o777 == 0o600
    with pytest.raises(ReplayError, match="new output"):
        asr_assembly.assemble(collection, output)


@pytest.mark.parametrize(
    "field,value", [("samples", True), ("samples", 0), ("sampleRate", 1), ("clock", "projected")]
)
def test_invalid_manifest(collection: Path, tmp_path: Path, field: str, value: object) -> None:
    path = collection / "inputs.json"
    data = json.loads(path.read_text(encoding="utf-8"))
    data[field] = value
    path.write_text(json.dumps(data), encoding="utf-8")
    with pytest.raises(ReplayError, match="definition"):
        asr_assembly.assemble(collection, tmp_path / "out.json")


@pytest.mark.parametrize(
    "filename", ["complete.json", "000000000000.clock.json", "000000000000.raw.json"]
)
def test_changed_or_missing_evidence(collection: Path, tmp_path: Path, filename: str) -> None:
    (collection / filename).write_text("{}", encoding="utf-8")
    with pytest.raises(ReplayError):
        asr_assembly.assemble(collection, tmp_path / "out.json")
    assert not (tmp_path / "out.json").exists()


@pytest.mark.parametrize("bindings", [{}, {"relative": "bad"}])
def test_invalid_input_bindings(collection: Path, tmp_path: Path, bindings: dict[str, str]) -> None:
    path = collection / "inputs.json"
    data = json.loads(path.read_text(encoding="utf-8"))
    data["inputs"] = bindings
    path.write_text(json.dumps(data), encoding="utf-8")
    with pytest.raises(ReplayError):
        asr_assembly.assemble(collection, tmp_path / "out.json")


def test_linked_evidence(collection: Path, tmp_path: Path) -> None:
    path = collection / "complete.json"
    saved = tmp_path / "complete.json"
    path.rename(saved)
    path.symlink_to(saved)
    with pytest.raises(ReplayError, match="regular"):
        asr_assembly.assemble(collection, tmp_path / "out.json")


def test_empty_and_half_open_ownership() -> None:
    assert (
        asr_assembly._labels(
            {"language": "ja", "text": "", "segments": []},
            window=AudioWindow(0, 960000),
            owner_start=0,
            owner_end=45,
        )
        == []
    )
    raw: dict[str, object] = {
        "language": "ja",
        "text": "声",
        "segments": [
            {
                "start": 44,
                "end": 46,
                "text": "声",
                "words": [{"word": "声", "start": 44, "end": 46, "probability": 1}],
            }
        ],
    }
    assert (
        asr_assembly._labels(raw, window=AudioWindow(0, 960000), owner_start=0, owner_end=45) == []
    )


def test_changed_during_read(collection: Path, tmp_path: Path, monkeypatch: MonkeyPatch) -> None:
    original = asr_assembly._labels

    def change(
        raw: dict[str, object], *, window: AudioWindow, owner_start: float, owner_end: float
    ) -> list[ClockLabel]:
        (collection / "inputs.json").write_text("{}", encoding="utf-8")
        return original(raw, window=window, owner_start=owner_start, owner_end=owner_end)

    monkeypatch.setattr(asr_assembly, "_labels", change)
    with pytest.raises(ReplayError, match="during assembly"):
        asr_assembly.assemble(collection, tmp_path / "out.json")


def test_cli(collection: Path, tmp_path: Path) -> None:
    assert asr_assembly.main([str(collection), str(tmp_path / "cli.json")]) == 0


def test_module_help(monkeypatch: MonkeyPatch) -> None:
    monkeypatch.setattr(sys, "argv", ["asr_assembly", "--help"])
    monkeypatch.delitem(sys.modules, "replay.asr_assembly")
    with pytest.raises(SystemExit) as error:
        runpy.run_module("replay.asr_assembly", run_name="__main__")
    assert error.value.code == 0
