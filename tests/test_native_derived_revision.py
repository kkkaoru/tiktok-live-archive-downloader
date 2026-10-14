"""Synthetic immutable revision binding regressions."""

import json
from pathlib import Path

import pytest

from replay import native_audio_revision_evidence as audio
from replay import native_delivery_evidence as evidence
from replay import native_derived_revision as derived
from replay.stages import file_digest


def ref(name: str) -> dict[str, str]:
    return {"path": "/fixture/" + name, "sha256": "a" * 64}


@pytest.fixture
def case(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> dict[str, object]:
    proof: dict[str, object] = {
        "originalDescriptor": ref("descriptor"),
        "originalMovie": ref("original"),
        "candidate": ref("candidate"),
        "numericalAudioMeasurementVerified": True,
        "qualityApproved": False,
    }
    target = tmp_path / "proof.json"
    target.write_text(json.dumps(proof), encoding="utf-8")
    monkeypatch.setattr(audio, "assess", lambda request, reader: proof)
    return {
        "version": 1,
        "kind": "payload-identical-video-audio-revision",
        "request": ref("request"),
        "proof": {"path": str(target), "sha256": file_digest(target)},
        "codePins": {
            str(Path(derived.__file__).resolve()): file_digest(Path(derived.__file__)),
            str(Path(audio.__file__).resolve()): file_digest(Path(audio.__file__)),
        },
    }


def test_binding_preserves_false_approval(case: dict[str, object]) -> None:
    result = derived.validate(
        case, descriptor=ref("descriptor"), movie=ref("candidate"), reader=evidence.MetadataReader()
    )
    assert result["qualityApproved"] is False


@pytest.mark.parametrize(
    ("key", "value"),
    [("version", True), ("kind", "unknown"), ("extra", 0), ("codePins", {})],
)
def test_schema_and_code_reject(case: dict[str, object], key: str, value: object) -> None:
    case[key] = value
    with pytest.raises(ValueError):
        derived.validate(
            case,
            descriptor=ref("descriptor"),
            movie=ref("candidate"),
            reader=evidence.MetadataReader(),
        )


@pytest.mark.parametrize(("descriptor", "movie"), [("other", "candidate"), ("descriptor", "other")])
def test_cross_candidate_reject(case: dict[str, object], descriptor: str, movie: str) -> None:
    with pytest.raises(ValueError):
        derived.validate(
            case, descriptor=ref(descriptor), movie=ref(movie), reader=evidence.MetadataReader()
        )


@pytest.mark.parametrize("mutation", ["different", "same_movie", "unverified"])
def test_recomputed_proof_guards(
    case: dict[str, object], mutation: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    proof_ref = evidence.reference(case["proof"])
    target = Path(proof_ref["path"])
    proof = evidence.MetadataReader().read(proof_ref)
    if mutation == "same_movie":
        proof["originalMovie"] = ref("candidate")
    elif mutation == "unverified":
        proof["numericalAudioMeasurementVerified"] = False
    else:
        proof["qualityApproved"] = True
    if mutation != "different":
        target.write_text(json.dumps(proof), encoding="utf-8")
        case["proof"] = {"path": str(target), "sha256": file_digest(target)}
    monkeypatch.setattr(audio, "assess", lambda request, reader: proof)
    with pytest.raises(ValueError):
        derived.validate(
            case,
            descriptor=ref("descriptor"),
            movie=ref("candidate"),
            reader=evidence.MetadataReader(),
        )
