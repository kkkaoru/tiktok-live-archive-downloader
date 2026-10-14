"""Exact native-clock verification with synthetic overlapping ASR groups."""

import json
from pathlib import Path
from unittest.mock import MagicMock

import pytest

from replay import asr_assembly
from replay import asr_assembly_verification as verifier
from replay.core import ReplayError
from replay.local_asr import fingerprint
from replay.native_invocation import durable, pin
from replay.native_sequence import ContractError
from replay.recordings import record


def publish_raw(source: Path, *, start: int, end: int, speech_start: int) -> str:
    raw = source / f"{start:012d}.raw.json"
    durable(
        raw,
        {
            "language": "ja",
            "text": "声点",
            "segments": [
                {
                    "start": speech_start,
                    "end": speech_start + 3,
                    "text": "声点",
                    "words": [
                        {
                            "word": "声",
                            "start": speech_start,
                            "end": speech_start + 1,
                            "probability": 0.9,
                        },
                        {
                            "word": "点",
                            "start": speech_start + 3,
                            "end": speech_start + 3,
                            "probability": 0.8,
                        },
                    ],
                },
            ],
        },
    )
    digest = fingerprint(raw)
    durable(
        source / f"{start:012d}.clock.json",
        {
            "startSample": start,
            "endSample": end,
            "sampleRate": 16000,
            "humanReviewed": False,
            "emptyRecognitionIsNotSilence": True,
            "rawSHA256": digest,
        },
    )
    return digest


@pytest.fixture
def setup(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> tuple[Path, Path]:
    monkeypatch.chdir(tmp_path)
    source = tmp_path / "edits/work/collection"
    source.mkdir(parents=True)
    input_path = source.parent / "synthetic-input.bin"
    input_path.write_bytes(b"synthetic bound input")
    durable(
        source / "inputs.json",
        {
            "version": 1,
            "samples": 976000,
            "sampleRate": 16000,
            "windowSamples": 960000,
            "strideSamples": 480000,
            "clock": "input-pcm-not-projected",
            "inputs": {str(input_path): fingerprint(input_path)},
        },
    )
    first = publish_raw(source, start=0, end=960000, speech_start=1)
    second = publish_raw(source, start=480000, end=976000, speech_start=15)
    durable(
        source / "complete.json",
        {"rawSHA256": [first, second], "windows": 2, "deliveryComplete": False},
    )
    labels = source.parent / "labels.json"
    assert asr_assembly.assemble(source, labels) == 4
    return source, labels


def replace(path: Path, value: dict[str, object]) -> None:
    path.write_text(json.dumps(value), encoding="utf-8")


def test_exact_conservation_and_originals_untouched(setup: tuple[Path, Path]) -> None:
    source, labels = setup
    original_labels = pin(labels)
    original_raw = pin(source / "000000480000.raw.json")
    assert verifier.verify(source, labels) == 4
    assert pin(labels) == original_labels
    assert pin(source / "000000480000.raw.json") == original_raw
    assert list((source.parent / "asr-assembly-verification").iterdir()) == []
    document = asr_assembly._read(labels)
    assert document["labels"] == [
        {"text": "声", "start": 1.0, "end": 2.0, "provenance": "000000000000:0"},
        {"text": "点", "start": 4.0, "end": 4.0, "provenance": "000000000000:1"},
        {"text": "声", "start": 45.0, "end": 46.0, "provenance": "000000480000:0"},
        {"text": "点", "start": 48.0, "end": 48.0, "provenance": "000000480000:1"},
    ]
    assert document["unresolvedPoints"] == 2
    assert document["humanReviewed"] is False
    assert document["speechAccuracyVerified"] is False
    assert document["deliveryComplete"] is False


@pytest.mark.parametrize(
    "field,value",
    [
        ("labels", []),
        ("clock", "projected"),
        ("humanReviewed", True),
        ("speechAccuracyVerified", True),
        ("deliveryComplete", True),
        ("unresolvedPoints", 0),
        ("version", True),
        ("assemblySHA256", "a" * 64),
    ],
)
def test_changed_label_document_rejected_and_scratch_removed(
    setup: tuple[Path, Path], field: str, value: object
) -> None:
    source, labels = setup
    document = asr_assembly._read(labels)
    document[field] = value
    replace(labels, document)
    original = pin(labels)
    with pytest.raises(ReplayError, match="exact native-clock"):
        verifier.verify(source, labels)
    assert pin(labels) == original
    assert list((source.parent / "asr-assembly-verification").iterdir()) == []


def test_shifted_word_clock_rejected(setup: tuple[Path, Path]) -> None:
    source, labels = setup
    document = asr_assembly._read(labels)
    words = document["labels"]
    assert isinstance(words, list)
    first = record(words[0])
    first["start"] = 1.1
    words[0] = first
    replace(labels, document)
    with pytest.raises(ReplayError, match="exact native-clock"):
        verifier.verify(source, labels)


@pytest.mark.parametrize(
    "name", ["complete.json", "000000000000.raw.json", "000000480000.clock.json"]
)
def test_corrupt_source_rejected(setup: tuple[Path, Path], name: str) -> None:
    source, labels = setup
    replace(source / name, {"corrupt": True})
    with pytest.raises(ReplayError):
        verifier.verify(source, labels)
    assert list((source.parent / "asr-assembly-verification").iterdir()) == []


@pytest.mark.parametrize(
    "name", ["complete.json", "000000000000.raw.json", "000000480000.clock.json"]
)
def test_missing_source_window_rejected(setup: tuple[Path, Path], name: str) -> None:
    source, labels = setup
    (source / name).unlink()
    with pytest.raises(FileNotFoundError):
        verifier.verify(source, labels)
    assert not (source.parent / "asr-assembly-verification").exists()


def test_bound_input_change_rejected(setup: tuple[Path, Path]) -> None:
    source, labels = setup
    (source.parent / "synthetic-input.bin").write_bytes(b"changed")
    with pytest.raises(ReplayError, match="fingerprint"):
        verifier.verify(source, labels)
    assert list((source.parent / "asr-assembly-verification").iterdir()) == []


def test_formatting_is_part_of_exact_assembler_bytes(setup: tuple[Path, Path]) -> None:
    source, labels = setup
    labels.write_bytes(labels.read_bytes() + b"\n")
    with pytest.raises(ReplayError, match="exact native-clock"):
        verifier.verify(source, labels)


def test_duplicate_nonfinite_and_oversized_labels_rejected(setup: tuple[Path, Path]) -> None:
    source, labels = setup
    labels.write_text('{"version":1,"version":1}', encoding="utf-8")
    with pytest.raises(ContractError, match="Duplicate"):
        verifier.verify(source, labels)
    labels.write_text('{"value":NaN}', encoding="utf-8")
    with pytest.raises(ContractError, match="Nonfinite"):
        verifier.verify(source, labels)
    labels.write_bytes(b" " * (8 * 1024**2 + 1))
    with pytest.raises(ContractError, match="bounded"):
        verifier.verify(source, labels)


def test_linked_input_and_scratch_rejected(setup: tuple[Path, Path]) -> None:
    source, labels = setup
    linked = source.parent / "linked-labels.json"
    linked.symlink_to(labels)
    with pytest.raises(ReplayError, match="linked"):
        verifier.verify(source, linked)
    destination = source.parent / "scratch-target"
    destination.mkdir()
    (source.parent / "asr-assembly-verification").symlink_to(destination)
    with pytest.raises(ReplayError, match="linked"):
        verifier.verify(source, labels)
    assert list(destination.iterdir()) == []


def test_original_mutation_during_reassembly_detected(
    setup: tuple[Path, Path], monkeypatch: pytest.MonkeyPatch
) -> None:
    source, labels = setup
    assemble = asr_assembly.assemble

    def changing(source_collection: Path, output: Path) -> int:
        count = assemble(source_collection, output)
        labels.chmod(0o640)
        return count

    monkeypatch.setattr(asr_assembly, "assemble", changing)
    with pytest.raises(ReplayError, match="changed during"):
        verifier.verify(source, labels)
    assert list((source.parent / "asr-assembly-verification").iterdir()) == []


def test_cli(monkeypatch: pytest.MonkeyPatch) -> None:
    verify = MagicMock(return_value=4)
    monkeypatch.setattr(verifier, "verify", verify)
    assert verifier.main(["source", "labels.json"]) == 0
    verify.assert_called_once_with(Path("source"), Path("labels.json"))
