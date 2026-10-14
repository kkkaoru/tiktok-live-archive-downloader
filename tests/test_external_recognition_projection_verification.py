"""Exact direct-producer projection with unchanged originals and no inference."""

import json
from pathlib import Path
from unittest.mock import MagicMock

import pytest
from test_external_recognition_projection import _write

from replay import external_recognition_projection as producer
from replay import external_recognition_projection_verification as verifier
from replay.core import ReplayError
from replay.native_invocation import pin
from replay.native_sequence import ContractError


@pytest.fixture
def setup(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> tuple[Path, Path]:
    monkeypatch.chdir(tmp_path)
    folder = tmp_path / "edits/work/direct-projection"
    folder.mkdir(parents=True)
    paths = _write(folder)
    config = Path(paths["config"])
    output = folder / "source-labels.json"
    assert producer.run(config, output) == 2
    return config, output


def test_exact_projection_and_scratch_removal(setup: tuple[Path, Path]) -> None:
    config, output = setup
    original = pin(output)
    assert verifier.verify(config, output) == 2
    assert pin(output) == original
    assert list((config.parent.parent / "asr-assembly-verification").iterdir()) == []


@pytest.mark.parametrize(
    "field,value",
    [
        ("labels", []),
        ("clock", "output"),
        ("humanReviewed", True),
        ("deliveryComplete", True),
        ("unresolvedPoints", 1),
        ("measuredDuration", {}),
    ],
)
def test_output_corruption_rejected(setup: tuple[Path, Path], field: str, value: object) -> None:
    config, output = setup
    document = json.loads(output.read_text())
    document[field] = value
    output.write_text(json.dumps(document), encoding="utf-8")
    original = pin(output)
    with pytest.raises(ReplayError, match="exact native-clock"):
        verifier.verify(config, output)
    assert pin(output) == original
    assert list((config.parent.parent / "asr-assembly-verification").iterdir()) == []


@pytest.mark.parametrize("name", ["labels.json", "mono.wav", "audio-clock.json", "inputs.json"])
def test_changed_input_hash_rejected(setup: tuple[Path, Path], name: str) -> None:
    config, output = setup
    path = config.parent / name
    path.write_bytes(path.read_bytes() + b" ")
    with pytest.raises(ReplayError, match="digest"):
        verifier.verify(config, output)


def test_missing_role_file_and_link_rejected(setup: tuple[Path, Path]) -> None:
    config, output = setup
    document = json.loads(config.read_text())
    del document["artifacts"]["pcm"]
    config.write_text(json.dumps(document), encoding="utf-8")
    with pytest.raises(ReplayError, match="complete direct"):
        verifier.verify(config, output)
    output.unlink()
    with pytest.raises(ReplayError):
        verifier.verify(config, output)


def test_no_fabricated_clocks_and_preserved_overlapping_source_anchors(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.chdir(tmp_path)
    folder = tmp_path / "edits/work/direct-projection"
    folder.mkdir(parents=True)
    paths = _write(folder, frames=2, per=8000)
    clock = folder / "audio-clock.json"
    clock.write_text(
        json.dumps(
            {
                "streams": [{"sample_rate": "16000", "time_base": "1/16000"}],
                "frames": [{"pts": 8480, "nb_samples": 8000}, {"pts": 16380, "nb_samples": 8000}],
            }
        ),
        encoding="utf-8",
    )
    labels = folder / "labels.json"
    document = json.loads(labels.read_text())
    document["labels"].append({"text": "点", "start": 0.5, "end": 0.5, "provenance": "w:2"})
    labels.write_text(json.dumps(document), encoding="utf-8")
    config = Path(paths["config"])
    data = json.loads(config.read_text())
    data["artifacts"]["audioClock"]["sha256"] = pin(clock)["sha256"]
    data["artifacts"]["labels"]["sha256"] = pin(labels)["sha256"]
    config.write_text(json.dumps(data), encoding="utf-8")
    output = folder / "source-labels.json"
    assert producer.run(config, output) == 3
    assert verifier.verify(config, output) == 3
    assert json.loads(output.read_text())["labels"] == [
        {"text": "あ", "start": 0.53, "end": 1.03, "provenance": "w:0"},
        {"text": "い", "start": 1.02375, "end": 1.52375, "provenance": "w:1"},
        {"text": "点", "start": 1.02375, "end": 1.02375, "provenance": "w:2"},
    ]


def test_input_mutation_during_verification_detected(
    setup: tuple[Path, Path], monkeypatch: pytest.MonkeyPatch
) -> None:
    config, output = setup
    run = producer.run

    def changing(config_path: Path, target: Path) -> int:
        count = run(config_path, target)
        output.chmod(0o640)
        return count

    monkeypatch.setattr(producer, "run", changing)
    with pytest.raises(ReplayError, match="changed during"):
        verifier.verify(config, output)


def test_nonfinite_output_rejected(setup: tuple[Path, Path]) -> None:
    config, output = setup
    output.write_text('{"labels":NaN}', encoding="utf-8")
    with pytest.raises(ContractError, match="Nonfinite"):
        verifier.verify(config, output)


def test_cli(monkeypatch: pytest.MonkeyPatch) -> None:
    verify = MagicMock(return_value=2)
    monkeypatch.setattr(verifier, "verify", verify)
    assert verifier.main(["config", "output"]) == 0
    verify.assert_called_once_with(Path("config"), Path("output"))
