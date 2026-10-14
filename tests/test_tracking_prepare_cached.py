"""Synthetic preparation only; no native execution."""

import json
from pathlib import Path

import pytest

from replay import tracking_batch
from replay import tracking_prepare_cached as cached
from replay.core import ReplayError
from replay.recordings import record
from replay.stages import file_digest

pytest_plugins = ("test_tracking_batch",)


def test_actual_prepare_reuses_and_seals(manifest: Path, tmp_path: Path) -> None:
    output = tmp_path / "cached"
    cached.main([str(manifest), str(output), "--unseeded"])
    proof = record(json.loads((output / "cache-provenance.json").read_text()))
    inputs = record(proof["inputs"])
    assert proof["hashReads"] == 2 * len(inputs)
    assert isinstance(proof["cacheHits"], int) and proof["cacheHits"] > 2
    assert proof["nativeExecuted"] is False
    assert proof["terminalFreshHashesVerified"] is True
    assert record(proof["outputs"])[str(output / "workflow.json")] == file_digest(
        output / "workflow.json"
    )
    with pytest.raises(FileExistsError):
        cached.prepare(manifest, output)


@pytest.mark.parametrize("field", ["source", "nativeExecutable", "clock", "seeds", "eof"])
def test_invalid_path(manifest: Path, tmp_path: Path, field: str) -> None:
    raw = record(json.loads(manifest.read_text()))
    raw[field] = "relative" if field in {"source", "nativeExecutable"} else {"path": 3}
    manifest.write_text(json.dumps(raw))
    with pytest.raises(ValueError, match="Explicit"):
        cached.prepare(manifest, tmp_path / "out")


def test_identity_mutation_and_no_receipt(
    manifest: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    original = tracking_batch.prepare

    def mutate(path: Path, output: Path, *, unseeded: bool = False) -> Path:
        result = original(path, output, unseeded=unseeded)
        (tmp_path / "source.bin").write_bytes(b"changed")
        return result

    monkeypatch.setattr(tracking_batch, "prepare", mutate)
    output = tmp_path / "out"
    with pytest.raises(ReplayError, match="changed"):
        cached.prepare(manifest, output)
    assert not (output / "cache-provenance.json").exists()
    assert (output / "binding.json").exists()


def test_symlink_rejected(manifest: Path, tmp_path: Path) -> None:
    raw = record(json.loads(manifest.read_text()))
    alias = tmp_path / "alias"
    alias.symlink_to(tmp_path / "source.bin")
    raw["source"] = str(alias)
    manifest.write_text(json.dumps(raw))
    with pytest.raises(ReplayError, match="resolved"):
        cached.prepare(manifest, tmp_path / "out")
