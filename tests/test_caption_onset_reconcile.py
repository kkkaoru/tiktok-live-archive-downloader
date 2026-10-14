"""Duplicate onset selection preserves original clocks and overlay evidence."""

import hashlib
import json
from pathlib import Path

import pytest

from replay import caption_onset_reconcile as reconcile
from replay.core import ReplayError


def ref(path: Path) -> dict[str, str]:
    return {"path": str(path), "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}


def write(path: Path, value: object) -> dict[str, str]:
    path.write_text(json.dumps(value), encoding="utf-8")
    return ref(path)


@pytest.fixture
def directory(tmp_path: Path) -> Path:
    first = {"text": "a", "start": 1.0, "end": 1.2, "utterance": "000000000000:1:cut-0"}
    second = {"text": "a", "start": 1.0, "end": 1.3, "utterance": "000000480000:2:cut-0"}
    labels = write(
        tmp_path / "labels.json",
        {
            "clock": "source-presentation",
            "labels": [
                {"provenance": "000000000000:1", "text": "a", "start": 4.0, "end": 4.2},
                {"provenance": "000000480000:2", "text": "a", "start": 4.0, "end": 4.3},
            ],
        },
    )
    cuts = write(tmp_path / "cuts.json", {"version": 1})
    projected = write(
        tmp_path / "projected.json",
        {
            "version": 1,
            "clock": "output",
            "cuts": cuts,
            "labels": labels,
            "boundaryConflicts": [],
            "tokens": [first, second],
        },
    )
    config = write(
        tmp_path / "config.json", {"version": 1, "projectedTokens": projected, "cuts": cuts}
    )
    tokens = write(
        tmp_path / "tokens-0000.json", {"version": 1, "clock": "output", "tokens": [first, second]}
    )
    binding = write(
        tmp_path / "recognition-0000.json",
        {
            "version": 1,
            "kind": "recognition-inputs",
            "segment": 0,
            "jobs": [0],
            "inputs": {
                tokens["path"]: tokens["sha256"],
                projected["path"]: projected["sha256"],
                labels["path"]: labels["sha256"],
            },
            "measurementsPerformed": False,
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
    write(
        tmp_path / "receipt.json",
        {
            "segments": index,
            "config": config,
            "inputs": {projected["path"]: projected["sha256"], labels["path"]: labels["sha256"]},
        },
    )
    return tmp_path


def test_prepare_verify_intact_longer_and_overlay(directory: Path) -> None:
    original = (directory / "tokens-0000.json").read_bytes()
    output = directory / "reconciled"
    mapping = reconcile.prepare(directory, output)
    reconcile.verify(directory, output)
    assert json.loads((output / "tokens-0000-0000.json").read_text())["tokens"] == [
        {"text": "a", "start": 1.0, "end": 1.3, "utterance": "000000480000:2:cut-0"}
    ]
    assert (directory / "tokens-0000.json").read_bytes() == original
    manifest = json.loads((output / "adopted-0000-0000.json").read_text())
    assert manifest["decisions"][0]["chosenIndex"] == 1
    assert manifest["timestampsChanged"] is False
    assert len(json.loads(mapping.read_text())["replacements"]) == 1
    assert reconcile.main([str(directory), str(output), "--verify"]) == 0
    assert reconcile.main([str(directory), str(output)]) == 1
    (output / "combined-0000.json").write_text("{}", encoding="utf-8")
    with pytest.raises((ReplayError, ValueError)):
        reconcile.verify(directory, output)


def test_pure_idempotent_tie_and_ordinary() -> None:
    values: list[object] = [
        {"text": "x", "start": 1, "end": 2, "utterance": "000000000000:0:cut-3"},
        {"text": "x", "start": 1, "end": 2, "utterance": "000000480000:0:cut-3"},
        {"text": "y", "start": 3, "end": 4, "utterance": "normal"},
    ]
    selected, decisions = reconcile.reconcile_tokens(
        values, known_owners=frozenset({"000000000000:0", "000000480000:0"})
    )
    assert decisions[0]["chosenIndex"] == 0
    assert selected == [
        {"text": "x", "start": 1, "end": 2, "utterance": "000000000000:0:cut-3"},
        {"text": "y", "start": 3, "end": 4, "utterance": "normal"},
    ]
    assert reconcile.reconcile_tokens(selected, known_owners=frozenset())[1] == []


@pytest.mark.parametrize(
    "mutation", ["text", "unknown", "samewindow", "cut", "context", "absent", "zero", "backwards"]
)
def test_ambiguous_duplicate_rejected(mutation: str) -> None:
    first: dict[str, object] = {
        "text": "x",
        "start": 1,
        "end": 2,
        "utterance": "000000000000:0:cut-3",
    }
    second: dict[str, object] = {
        "text": "x",
        "start": 1,
        "end": 3,
        "utterance": "000000480000:0:cut-3",
    }
    owners = frozenset({"000000000000:0", "000000480000:0"})
    if mutation == "text":
        second["text"] = "different"
    elif mutation == "unknown":
        second["utterance"] = "unknown:cut-3"
    elif mutation == "samewindow":
        second["utterance"] = "000000000000:1:cut-3"
    elif mutation == "cut":
        second["utterance"] = "000000480000:0:cut-4"
    elif mutation == "context":
        second["utterance"] = "speaker:000000480000:0:cut-3"
    elif mutation == "absent":
        owners = frozenset()
    elif mutation == "zero":
        second["end"] = 1
    else:
        second["start"] = 0.5
    with pytest.raises((ReplayError, ValueError)):
        reconcile.reconcile_tokens([first, second], known_owners=owners)


def test_raw_window_and_unknown_cut() -> None:
    values: list[object] = [
        {"text": "x", "start": 1, "end": 2, "utterance": "raw/1.raw.json:cut-0"},
        {"text": "x", "start": 1, "end": 3, "utterance": "raw/2.raw.json:cut-0"},
    ]
    assert (
        reconcile.reconcile_tokens(
            values, known_owners=frozenset({"raw/1.raw.json", "raw/2.raw.json"})
        )[1][0]["chosenIndex"]
        == 1
    )
    with pytest.raises(ReplayError, match="owner or cut"):
        reconcile._window("unknown")
    with pytest.raises(ReplayError, match="count"):
        reconcile.reconcile_tokens([], known_owners=frozenset())


def test_bound_input_mutation_and_limits(directory: Path, tmp_path: Path) -> None:
    reader = reconcile.Reader()
    reader.read(directory / "segments.json")
    (directory / "segments.json").write_text("{}", encoding="utf-8")
    with pytest.raises(ReplayError, match="inputs changed"):
        reader.verify()
    with pytest.raises(ReplayError, match="binding changed"):
        reader.read(directory / "segments.json", "0" * 64)
    reader.total = reconcile.MAX_TOTAL_BYTES
    with pytest.raises(ReplayError, match="budget"):
        reader.read(directory / "segments.json")
    with pytest.raises(ReplayError, match="reference"):
        reconcile.Reader().ref({"path": str(directory)})
    with pytest.raises(ReplayError, match="checkout-local"):
        reconcile._path(Path("relative"))
    with pytest.raises(ReplayError, match="rows"):
        reconcile._values({}, "tokens")


def test_cli_producer(directory: Path) -> None:
    assert reconcile.main([str(directory), str(directory / "cli")]) == 0


def test_empty_recognition_job_needs_no_reconciliation(tmp_path: Path) -> None:
    single = {"text": "a", "start": 1.0, "end": 1.2, "utterance": "000000000000:1:cut-0"}
    labels = write(
        tmp_path / "labels.json",
        {
            "clock": "source-presentation",
            "labels": [{"provenance": "000000000000:1", "text": "a", "start": 4.0, "end": 4.2}],
        },
    )
    cuts = write(tmp_path / "cuts.json", {"version": 1})
    projected = write(
        tmp_path / "projected.json",
        {
            "version": 1,
            "clock": "output",
            "cuts": cuts,
            "labels": labels,
            "boundaryConflicts": [],
            "tokens": [single],
        },
    )
    config = write(
        tmp_path / "config.json", {"version": 1, "projectedTokens": projected, "cuts": cuts}
    )
    empty = write(tmp_path / "tokens-0000.json", {"version": 1, "clock": "output", "tokens": []})
    ordinary = write(
        tmp_path / "tokens-0001.json", {"version": 1, "clock": "output", "tokens": [single]}
    )
    binding = write(
        tmp_path / "recognition-0000.json",
        {
            "version": 1,
            "kind": "recognition-inputs",
            "segment": 0,
            "jobs": [0, 1],
            "inputs": {
                empty["path"]: empty["sha256"],
                ordinary["path"]: ordinary["sha256"],
                projected["path"]: projected["sha256"],
                labels["path"]: labels["sha256"],
            },
            "measurementsPerformed": False,
        },
    )
    index = write(
        tmp_path / "segments.json",
        {
            "version": 1,
            "rate": 30,
            "segments": [
                {"segment": 0, "offsetFrame": 0, "jobs": [0, 1], "recognitionInput": binding}
            ],
        },
    )
    write(
        tmp_path / "receipt.json",
        {
            "segments": index,
            "config": config,
            "inputs": {projected["path"]: projected["sha256"], labels["path"]: labels["sha256"]},
        },
    )
    output = tmp_path / "reconciled"
    mapping = reconcile.prepare(tmp_path, output)
    reconcile.verify(tmp_path, output)
    assert json.loads(mapping.read_text())["replacements"] == []
    assert json.loads((output / "reconciliation.json").read_text())["decisions"] == []


def test_source_onset_before_projection_must_match() -> None:
    decisions: list[dict[str, object]] = [
        {
            "candidates": [
                {"text": "a", "start": 1, "end": 2, "utterance": "000000000000:1:cut-0"},
                {"text": "a", "start": 1, "end": 3, "utterance": "000000480000:2:cut-0"},
            ]
        }
    ]
    labels: list[object] = [
        {"provenance": "000000000000:1", "text": "a", "start": 4, "end": 5},
        {"provenance": "000000480000:2", "text": "a", "start": 4.5, "end": 6},
    ]
    with pytest.raises(ReplayError, match="before cut projection"):
        reconcile._source_decisions(decisions, labels)
    with pytest.raises(ReplayError, match="Ambiguous original"):
        reconcile._source_decisions(decisions, [])


@pytest.mark.parametrize("digest", [None, "", "0" * 63, "A" * 64])
def test_reference_requires_explicit_digest(directory: Path, digest: object) -> None:
    with pytest.raises(ReplayError, match="reference"):
        reconcile.Reader().ref({"path": str(directory / "segments.json"), "sha256": digest})
