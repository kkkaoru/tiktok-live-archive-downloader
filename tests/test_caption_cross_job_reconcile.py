"""Cross-job overlap reconciliation retains actual clocks and original ownership."""

import json
from pathlib import Path

import pytest

from replay import caption_cross_job_reconcile as cross
from replay.caption_onset_reconcile import Reader, _source_decisions, _values, reconcile_tokens
from replay.core import ReplayError
from replay.recordings import record
from replay.stages import file_digest


def write(path: Path, value: object) -> dict[str, str]:
    path.write_text(json.dumps(value), encoding="utf-8")
    return {"path": str(path), "sha256": file_digest(path)}


@pytest.fixture
def fixture(tmp_path: Path) -> tuple[Path, Path, Path, Path]:
    collection = tmp_path / "collection"
    collection.mkdir()
    parents: dict[str, str] = {}
    rawhashes: list[str] = []
    labels: list[object] = []
    tokens: list[object] = []
    for index, start in enumerate((0, 480000)):
        local_start = 44.88 if index == 0 else 14.88
        local_end = 45.10 if index == 0 else 15.12
        raw = write(
            collection / f"{start:012d}.raw.json",
            {
                "language": "ja",
                "text": "数",
                "segments": [
                    {
                        "start": local_start,
                        "end": local_end,
                        "text": "数",
                        "words": [
                            {
                                "word": "数",
                                "start": local_start,
                                "end": local_end,
                                "probability": 0.99 if index == 0 else 0.97,
                            }
                        ],
                    }
                ],
            },
        )
        rawhashes.append(raw["sha256"])
        clock = write(
            collection / f"{start:012d}.clock.json",
            {
                "startSample": start,
                "endSample": start + 960000,
                "sampleRate": 16000,
                "humanReviewed": False,
                "emptyRecognitionIsNotSilence": True,
                "rawSHA256": raw["sha256"],
            },
        )
        parents.update({raw["path"]: raw["sha256"], clock["path"]: clock["sha256"]})
        label = {
            "text": "数",
            "start": local_start + start / 16000,
            "end": local_end + start / 16000,
            "provenance": f"{start:012d}:0",
        }
        labels.append(label)
        tokens.append(
            {key: value for key, value in label.items() if key != "provenance"}
            | {"utterance": f"{start:012d}:0:cut-0"}
        )
    manifest = write(
        collection / "inputs.json",
        {
            "version": 1,
            "sampleRate": 16000,
            "samples": 1440000,
            "windowSamples": 960000,
            "strideSamples": 480000,
            "clock": "input-pcm-not-projected",
        },
    )
    complete = write(
        collection / "complete.json",
        {"rawSHA256": rawhashes, "windows": 2, "deliveryComplete": False},
    )
    parents.update({manifest["path"]: manifest["sha256"], complete["path"]: complete["sha256"]})
    pcm_labels = tmp_path / "pcm-labels.json"
    pcm_ref = write(
        pcm_labels,
        {
            "version": 1,
            "clock": "input-pcm-not-projected",
            "labels": labels,
            "parentSHA256": parents,
        },
    )
    source = write(
        tmp_path / "source-labels.json",
        {
            "clock": "source-presentation",
            "labels": labels,
            "parentSHA256": {pcm_ref["path"]: pcm_ref["sha256"]},
        },
    )
    projected = write(tmp_path / "projected.json", {"clock": "output", "labels": source})
    config = write(tmp_path / "config.json", {"projectedTokens": projected})
    directory = tmp_path / "inputs"
    directory.mkdir()
    token_refs = [
        write(
            directory / f"tokens-{index:04d}.json",
            {"version": 1, "clock": "output", "tokens": [token]},
        )
        for index, token in enumerate(tokens)
    ]
    adopted = write(tmp_path / "prior-adopted.json", {"version": 1, "tokens": token_refs[0]})
    bindings = {ref["path"]: ref["sha256"] for ref in [*token_refs, adopted]}
    recognition = write(
        directory / "recognition-0000.json",
        {
            "version": 1,
            "kind": "recognition-inputs",
            "segment": 0,
            "jobs": [0, 1],
            "inputs": bindings,
            "measurementsPerformed": False,
        },
    )
    index_ref = write(
        directory / "segments.json",
        {
            "rate": 30,
            "segments": [
                {
                    "segment": 0,
                    "offsetFrame": 0,
                    "frames": 2700,
                    "jobs": [0, 1],
                    "recognitionInput": recognition,
                }
            ],
        },
    )
    write(directory / "receipt.json", {"segments": index_ref, "config": config})
    mapping = tmp_path / "mapping.json"
    write(
        mapping,
        {
            "version": 1,
            "replacements": [
                {"segment": 0, "job": 0, "tokens": token_refs[0], "adoptedManifest": adopted}
            ],
        },
    )
    return directory, mapping, collection, pcm_labels


def decision_for(fixture: tuple[Path, Path, Path, Path]) -> dict[str, object]:
    directory, _, _, labels = fixture
    values = [
        json.loads((directory / f"tokens-{index:04d}.json").read_text())["tokens"][0]
        for index in range(2)
    ]
    _, decisions = reconcile_tokens(
        values, known_owners=frozenset({"000000000000:0", "000000480000:0"})
    )
    _source_decisions(decisions, json.loads(labels.read_text())["labels"])
    return decisions[0]


def test_prepare_verify_full_cross_job_and_empty_original_job(
    fixture: tuple[Path, Path, Path, Path], tmp_path: Path
) -> None:
    directory, mapping, collection, labels = fixture
    originals = {path: path.read_bytes() for path in directory.glob("*.json")}
    output = tmp_path / "reconciled"
    cross.prepare(directory, mapping, collection, labels, output)
    cross.verify(directory, mapping, collection, labels, output)
    assert json.loads((output / "tokens-0000-0000.json").read_text())["tokens"] == []
    assert all(path.read_bytes() == content for path, content in originals.items())
    audit = json.loads((output / "audit.json").read_text())
    decision = audit["segments"][0]["decisions"][0]
    assert decision["selectedCanonicalOwner"] == "000000480000:0:cut-0"
    assert decision["ownershipProofs"][0]["ownershipEnd"] == 45
    assert decision["ownershipProofs"][0]["originalMidpoint"] < 45
    assert decision["ownershipProofs"][1]["originalMidpoint"] >= 45
    assert decision["extentOrOwnerSpliced"] is False
    assert decision["semanticConfidenceRankingUsed"] is False
    args = [str(directory), str(mapping), str(collection), str(labels), str(output)]
    assert cross.main([*args, "--verify"]) == 0
    assert cross.main(args) == 1
    (output / "audit.json").write_text("{}")
    with pytest.raises((ReplayError, ValueError)):
        cross.verify(directory, mapping, collection, labels, output)


@pytest.mark.parametrize(
    "mutation", ["parent", "coverage", "raw", "clock", "absentgroup", "label", "owner", "selection"]
)
def test_original_proof_corruption_rejected(
    fixture: tuple[Path, Path, Path, Path], mutation: str
) -> None:
    _, _, collection, labels = fixture
    decision = decision_for(fixture)
    document = json.loads(labels.read_text())
    if mutation == "parent":
        document["parentSHA256"] = {}
        write(labels, document)
    elif mutation == "coverage":
        complete = json.loads((collection / "complete.json").read_text())
        complete["windows"] = 0
        ref = write(collection / "complete.json", complete)
        document["parentSHA256"][ref["path"]] = ref["sha256"]
        write(labels, document)
    elif mutation == "raw":
        (collection / "000000000000.raw.json").write_text("{}")
    elif mutation == "clock":
        ref = write(collection / "000000000000.clock.json", {"broken": True})
        document["parentSHA256"][ref["path"]] = ref["sha256"]
        write(labels, document)
    elif mutation == "absentgroup":
        record(_values(decision, "sourceCandidates")[0])["provenance"] = "000000000000:9"
    elif mutation == "label":
        record(_values(decision, "sourceCandidates")[0])["end"] = 46
    elif mutation == "owner":
        record(_values(decision, "sourceCandidates")[0])["provenance"] = "unknown"
    else:
        decision["chosenIndex"] = 9
    with pytest.raises((ReplayError, ValueError)):
        cross.ownership_proof(decision, reader=Reader(), collection=collection, labels=labels)


@pytest.mark.parametrize("mutation", ["text", "cut", "window", "context"])
def test_ambiguous_equal_onsets_preserved_as_failure(
    fixture: tuple[Path, Path, Path, Path], mutation: str
) -> None:
    directory, _, _, _ = fixture
    values = [
        json.loads((directory / f"tokens-{index:04d}.json").read_text())["tokens"][0]
        for index in range(2)
    ]
    if mutation == "text":
        values[1]["text"] = "別"
    else:
        values[1]["utterance"] = {
            "cut": "000000480000:0:cut-1",
            "window": "000000000000:1:cut-0",
            "context": "speaker:000000480000:0:cut-0",
        }[mutation]
    with pytest.raises(ReplayError):
        reconcile_tokens(values, known_owners=frozenset({"000000000000:0", "000000480000:0"}))
    values[1]["start"] += 0.01
    values[1]["end"] += 0.01
    assert reconcile_tokens(values, known_owners=frozenset())[0] == values


def test_output_and_projected_binding_guard(
    fixture: tuple[Path, Path, Path, Path], tmp_path: Path
) -> None:
    directory, mapping, collection, labels = fixture
    with pytest.raises(ReplayError, match="work storage"):
        cross._artifacts(directory, mapping, collection, labels, Path.cwd() / "outside")
    source = tmp_path / "source-labels.json"
    doc = json.loads(source.read_text())
    doc["parentSHA256"] = {}
    source_ref = write(source, doc)
    projected = tmp_path / "projected.json"
    projected_ref = write(projected, {"clock": "output", "labels": source_ref})
    config_ref = write(tmp_path / "config.json", {"projectedTokens": projected_ref})
    receipt = json.loads((directory / "receipt.json").read_text())
    receipt["config"] = config_ref
    write(directory / "receipt.json", receipt)
    with pytest.raises(ReplayError, match="original PCM"):
        cross.prepare(directory, mapping, collection, labels, tmp_path / "bad")
