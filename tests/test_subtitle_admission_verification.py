"""Contrast original producer capacity with strict read-only artifact admission."""

import json
import shutil
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path

import pytest
from pytest import MonkeyPatch
from test_subtitle_cleanup_workflow import Usage
from test_subtitle_cleanup_workflow import chunk as chunk
from test_subtitle_cleanup_workflow import settings as settings

from replay import cleanup_lane_successor as successor
from replay import subtitle_admission_verification as reader
from replay import subtitle_cleanup_workflow as cleanup


def test_low_space_verifies_existing_but_builder_refuses(
    chunk: Path, monkeypatch: MonkeyPatch
) -> None:
    cleanup.build_candidate(chunk)
    before = (chunk.parent / "candidate-config.json").read_bytes()
    monkeypatch.setattr(shutil, "disk_usage", lambda path: Usage(100, 100, 0))
    result = reader.verify_admission(chunk)
    assert result["originalStageCompleted"] is False
    assert result["qualityApproved"] is False
    assert result["repairVerified"] is False
    assert (chunk.parent / "candidate-config.json").read_bytes() == before
    with pytest.raises(ValueError, match="Insufficient free space"):
        cleanup.build_candidate(chunk)


@pytest.mark.parametrize("name", reader.ARTIFACTS)
def test_partial_artifacts_are_not_created(chunk: Path, name: str) -> None:
    cleanup.build_candidate(chunk)
    path = chunk.parent / name
    path.unlink()
    with pytest.raises(ValueError, match="Invalid bounded"):
        reader.verify_admission(chunk)
    assert not path.exists()


@pytest.mark.parametrize("name", reader.ARTIFACTS)
def test_changed_artifact_refused(chunk: Path, name: str) -> None:
    cleanup.build_candidate(chunk)
    (chunk.parent / name).write_text("{}", encoding="utf-8")
    with pytest.raises(ValueError, match="artifact differs"):
        reader.verify_admission(chunk)


def test_source_change_refused(chunk: Path) -> None:
    cleanup.build_candidate(chunk)
    (Path.cwd() / "source.mp4").write_bytes(b"changed")
    with pytest.raises(ValueError, match="source changed before"):
        reader.verify_admission(chunk)


def test_model_change_refused(chunk: Path) -> None:
    cleanup.build_candidate(chunk)
    (Path.cwd() / "model").write_bytes(b"changed")
    with pytest.raises(ValueError):
        reader.verify_admission(chunk)


def test_invalid_worker_budget_refused(chunk: Path) -> None:
    cleanup.build_candidate(chunk)
    data = json.loads(chunk.read_text())
    data["maximumActiveChunks"] = 0
    chunk.write_text(json.dumps(data), encoding="utf-8")
    with pytest.raises(ValueError, match="active chunk"):
        reader.verify_admission(chunk)


def test_repair_requires_real_receipt(chunk: Path) -> None:
    cleanup.build_candidate(chunk)
    with pytest.raises(ValueError, match="Invalid bounded"):
        reader.verify_admission(chunk, verify_repair=True)


def test_endpoint_mutation_refused(chunk: Path, monkeypatch: MonkeyPatch) -> None:
    cleanup.build_candidate(chunk)
    original = reader._verify_artifacts

    def changed(path: Path) -> dict[str, str]:
        result = original(path)
        path.write_text(path.read_text() + " ", encoding="utf-8")
        return result

    monkeypatch.setattr(reader, "_verify_artifacts", changed)
    with pytest.raises(ValueError, match="changed during"):
        reader.verify_admission(chunk)


def test_source_endpoint_mutation_refused(chunk: Path, monkeypatch: MonkeyPatch) -> None:
    cleanup.build_candidate(chunk)
    original = reader._verify_artifacts

    def changed(path: Path) -> dict[str, str]:
        result = original(path)
        (Path.cwd() / "source.mp4").write_bytes(b"changed")
        return result

    monkeypatch.setattr(reader, "_verify_artifacts", changed)
    with pytest.raises(ValueError, match="source changed after"):
        reader.verify_admission(chunk)


def test_cli_and_owner_binding(
    chunk: Path, monkeypatch: MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    cleanup.build_candidate(chunk)

    @contextmanager
    def owned(plan: Path, index: int) -> Iterator[tuple[dict[str, object], dict[str, object]]]:
        if index != 1:
            raise ValueError("Owner active")
        yield {}, {"chunk": str(chunk)}

    monkeypatch.setattr(successor, "_owned", owned)
    monkeypatch.setattr(successor, "_path", lambda value: Path(str(value)))
    assert reader.verify_successor(chunk, 1)["originalStageCompleted"] is False
    assert reader.main([str(chunk), "--successor", str(chunk), "--index", "1"]) == 0
    assert json.loads(capsys.readouterr().out)["kind"] == "readonly-original-cleanup-admission"
    assert reader.main([str(chunk)]) == 0
    assert reader.main([str(chunk), "--index", "1"]) == 1
    assert reader.main([str(chunk), "--successor", str(chunk), "--index", "2"]) == 1
    assert (
        reader.main([str(chunk.parent / "different"), "--successor", str(chunk), "--index", "1"])
        == 1
    )


def test_symlink_artifact_refused(chunk: Path) -> None:
    cleanup.build_candidate(chunk)
    artifact = chunk.parent / "admission.json"
    moved = chunk.parent / "moved.json"
    artifact.rename(moved)
    artifact.symlink_to(moved)
    with pytest.raises(ValueError, match="canonical"):
        reader.verify_admission(chunk)


def test_exclusive_proof_and_consumer(chunk: Path) -> None:
    cleanup.build_candidate(chunk)
    proof = chunk.parent / "new-proof.json"
    assert reader.main([str(chunk), "--proof", str(proof)]) == 0
    assert reader.main([str(chunk), "--proof", str(proof)]) == 1
    assert reader.main([str(chunk), "--proof", str(proof), "--verify-proof"]) == 0
    proof.write_text("{}", encoding="utf-8")
    assert reader.main([str(chunk), "--proof", str(proof), "--verify-proof"]) == 1
    assert reader.main([str(chunk), "--verify-proof"]) == 1
