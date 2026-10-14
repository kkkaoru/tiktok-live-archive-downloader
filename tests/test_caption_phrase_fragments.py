"""Dependent display grouping retains every original recognition anchor."""

import hashlib
import json
from pathlib import Path
from unittest.mock import MagicMock

import pytest

from replay import caption_phrase_fragments as phrase
from replay.core import ReplayError
from replay.recordings import record


def rows(texts: list[str]) -> list[object]:
    return [
        {
            "text": text,
            "start": index / 10,
            "end": index / 10 + 0.01,
            "utterance": f"000000000000:{index}:cut-0",
        }
        for index, text in enumerate(texts)
    ]


@pytest.mark.parametrize(
    "texts,expected",
    [
        (["食べ", "る", "人"], ["食べる", "人"]),
        (["走り", "まし", "た"], ["走りました"]),
        (["猫", "の", "人"], ["猫の", "人"]),
        (["食べ", "られる", "じゃ", "ん", "食べ", "る"], ["食べられるじゃん", "食べる"]),
        (["人", "さん"], ["人さん"]),
        (["猫", "。", "の"], ["猫", "。", "の"]),
    ],
)
def test_pos_grouping(texts: list[str], expected: list[str]) -> None:
    original = rows(texts)
    snapshot = json.dumps(original)
    grouped, decisions = phrase.group_fragments(original)
    assert [row["text"] for row in map(record, grouped)] == expected
    assert json.dumps(original) == snapshot
    assert "".join(expected) == "".join(texts)
    for decision in decisions:
        originals = decision["originalRows"]
        assert isinstance(originals, list)
        assert record(decision["displayRow"])["start"] == record(originals[0])["start"]
        assert decision["firstActualAnchor"] == record(originals[0])["start"]


@pytest.mark.parametrize("boundary", ["gap", "cut", "context", "unknown"])
def test_hard_boundaries(boundary: str) -> None:
    values = rows(["食べ", "る"])
    second = record(values[1])
    if boundary == "gap":
        second.update(start=1, end=2)
    else:
        second["utterance"] = {
            "cut": "000000000000:1:cut-1",
            "context": "speaker:000000000000:1:cut-0",
            "unknown": "other",
        }[boundary]
    assert phrase.group_fragments(values)[0] == values


def test_empty_coarse_and_budgets(monkeypatch: pytest.MonkeyPatch) -> None:
    assert phrase.group_fragments([]) == ([], [])
    values = rows(["食べる人"])
    assert phrase.group_fragments(values)[0] == values
    with pytest.raises(ReplayError, match="budget"):
        phrase.group_fragments(values, maximum_gap=float("nan"))
    backwards = rows(["猫", "人"])
    record(backwards[1])["start"] = -1
    with pytest.raises((ReplayError, ValueError)):
        phrase.group_fragments(backwards)
    tokenizer = MagicMock()
    tokenizer.tokenize.return_value = []
    monkeypatch.setattr(phrase, "_tokenizer", lambda: tokenizer)
    with pytest.raises(ReplayError, match="Morphology"):
        phrase.group_fragments(values)


def ref(path: Path) -> dict[str, str]:
    return {"path": str(path), "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}


def write(path: Path, document: object) -> dict[str, str]:
    path.write_text(json.dumps(document), encoding="utf-8")
    return ref(path)


@pytest.fixture
def directory(tmp_path: Path) -> Path:
    values = rows(["食べ", "る", "人"])
    labels = write(tmp_path / "labels.json", {"clock": "source-presentation", "labels": values})
    cuts = write(tmp_path / "cuts.json", {"version": 1})
    projected = write(
        tmp_path / "projected.json",
        {
            "version": 1,
            "clock": "output",
            "cuts": cuts,
            "labels": labels,
            "boundaryConflicts": [],
            "tokens": values,
        },
    )
    config = write(tmp_path / "config.json", {"projectedTokens": projected, "cuts": cuts})
    tokens = write(
        tmp_path / "tokens-0000.json", {"version": 1, "clock": "output", "tokens": values}
    )
    inputs = {projected["path"]: projected["sha256"], labels["path"]: labels["sha256"]}
    binding = write(
        tmp_path / "recognition-0000.json",
        {
            "version": 1,
            "kind": "recognition-inputs",
            "segment": 0,
            "jobs": [0],
            "measurementsPerformed": False,
            "inputs": {**inputs, tokens["path"]: tokens["sha256"]},
        },
    )
    index = write(
        tmp_path / "segments.json",
        {
            "version": 1,
            "rate": 30,
            "segments": [
                {"segment": 0, "offsetFrame": 0, "jobs": [0], "recognitionInput": binding}
            ],
        },
    )
    write(tmp_path / "receipt.json", {"segments": index, "config": config, "inputs": inputs})
    return tmp_path


def test_prepare_verify_provenance(directory: Path) -> None:
    before = {path: path.read_bytes() for path in directory.glob("*.json")}
    output = directory / "grouped"
    mapping = phrase.prepare(directory, output, (0,))
    assert mapping.name == "token-sources.json"
    phrase.verify(directory, output, (0,))
    assert all(path.read_bytes() == content for path, content in before.items())
    adopted = json.loads((output / "adopted-0000-0000.json").read_text())
    assert adopted["originalRows"] == rows(["食べ", "る", "人"])
    assert adopted["originalAnchorsUnchanged"] is True
    assert adopted["speechAccuracyVerified"] is False
    assert phrase.main([str(directory), str(output), "--segment", "0", "--verify"]) == 0
    assert phrase.main([str(directory), str(output), "--segment", "0"]) == 1
    (output / "tokens-0000-0000.json").write_text("{}")
    with pytest.raises((ReplayError, ValueError)):
        phrase.verify(directory, output, (0,))


@pytest.mark.parametrize(
    "mutation",
    ["rate", "duplicate", "absent", "coordinates", "job", "anchor", "contract", "selected"],
)
def test_source_corruption_rejected(directory: Path, mutation: str) -> None:
    index_path = directory / "segments.json"
    index = json.loads(index_path.read_text())
    selected: tuple[int, ...] = (0,)
    if mutation == "rate":
        index["rate"] = 60
    elif mutation == "duplicate":
        index["segments"] *= 2
    elif mutation == "absent":
        selected = (5,)
    elif mutation == "selected":
        selected = (0, 0)
    elif mutation == "coordinates":
        index["segments"][0]["offsetFrame"] = -1
    elif mutation == "job":
        index["segments"][0]["jobs"] = ["0"]
    elif mutation in {"anchor", "contract"}:
        binding_path = directory / "recognition-0000.json"
        binding = json.loads(binding_path.read_text())
        if mutation == "anchor":
            token_path = directory / "tokens-0000.json"
            tokens = json.loads(token_path.read_text())
            tokens["tokens"][0]["text"] = "別"
            token_ref = write(token_path, tokens)
            binding["inputs"][token_ref["path"]] = token_ref["sha256"]
        else:
            binding["inputs"][str(directory / "labels.json")] = "0" * 64
        index["segments"][0]["recognitionInput"] = write(binding_path, binding)
    index_ref = write(index_path, index)
    receipt_path = directory / "receipt.json"
    receipt = json.loads(receipt_path.read_text())
    receipt["segments"] = index_ref
    write(receipt_path, receipt)
    with pytest.raises((ReplayError, ValueError)):
        phrase.prepare(directory, directory / "bad", selected)


def test_selection_audit_and_no_output(directory: Path) -> None:
    files, entries = phrase._artifacts(directory, directory / "audit", ())
    assert entries
    audit = json.loads(files[directory / "audit" / "audit.json"])
    assert audit["segments"][0]["groups"] == 1
    assert json.loads(files[directory / "audit" / "token-sources.json"])["replacements"] == []
    with pytest.raises(ReplayError):
        phrase.prepare(directory, directory / "empty", ())
    with pytest.raises(ReplayError, match="work storage"):
        phrase._artifacts(directory, Path.cwd() / "outside", (0,))


def test_long_auxiliary_chain_retains_all_fine_anchors() -> None:
    values = rows(["カシスミルク", "で", "ござい", "ます", "みたい", "な"])
    for index, value in enumerate(values):
        record(value).update(start=index * 0.6, end=(index + 1) * 0.6)
    grouped, decisions = phrase.group_fragments(values)
    assert grouped == values
    assert decisions == []


def test_only_subframe_dependent_tail_changes() -> None:
    values = rows(["食べ", "られる", "じゃ", "ん", "食べ", "る"])
    spans = [(0, 0.3), (0.3, 0.5), (0.5, 1), (1, 1.017), (1.017, 1.2), (1.2, 1.5)]
    for value, (start, end) in zip(values, spans, strict=True):
        record(value).update(start=start, end=end)
    grouped, decisions = phrase.group_fragments(values)
    assert [record(value)["text"] for value in grouped] == [
        "食べ",
        "られる",
        "じゃん",
        "食べ",
        "る",
    ]
    assert grouped[:2] == values[:2]
    assert grouped[-2:] == values[-2:]
    assert decisions[0]["originalRows"] == values[2:4]
    assert record(grouped[2])["start"] == 0.5


def test_no_coalescence_over_existing_focused_budget() -> None:
    values = rows(["猫", "の"])
    record(values[0]).update(start=0, end=3)
    record(values[1]).update(start=3, end=3.01)
    assert phrase.group_fragments(values)[0] == values


def test_preserves_native_fine_fragment_scaling_contract() -> None:
    from collections.abc import Iterable
    from fractions import Fraction

    from replay.caption_measured_job import _token
    from replay.caption_native_paging import paged_cues
    from replay.caption_timing import ScaledCaptionCue

    class Measurements:
        width = 100.0
        expected_point_size: float | None = 20.0

        def prepare_lines(self, texts: Iterable[str]) -> None:
            tuple(texts)

        def measure(self, text: str) -> tuple[float, float]:
            return max(map(len, text.split("\n"))) * 20.0, len(text.split("\n")) * 20.0

    values = rows(list("こぉぉぉぉぉぉぉぉぉぉ"))
    for index, value in enumerate(values):
        record(value).update(start=index * 0.4, end=(index + 1) * 0.4)
    grouped, decisions = phrase.group_fragments(values)
    assert grouped == values
    assert decisions == []
    cues, _, _ = paged_cues(
        tuple(_token(value) for value in grouped),
        layout=Measurements(),
        rate=30,
        duration=Fraction(5),
    )
    assert len(cues) == 1
    assert isinstance(cues[0], ScaledCaptionCue)
    assert cues[0].scale.fragments == 11
    assert cues[0].scale.maximum_fragment_seconds == pytest.approx(0.4)
