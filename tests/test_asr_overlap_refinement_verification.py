"""Exact overlap verification uses the real producer on synthetic metadata."""

from pathlib import Path

import pytest
from test_asr_label_refine import config as config

from replay import asr_label_refine
from replay import asr_overlap_refinement_verification as verifier
from replay.core import ReplayError
from replay.stages import StageError


def test_exact_producer_preserves_originals(config: Path, tmp_path: Path) -> None:
    labels = tmp_path / "candidate.json"
    asr_label_refine.run(config, labels)
    originals = {path: path.read_bytes() for path in tmp_path.glob("*.json")}
    assert verifier.verify(config, labels) == 4
    assert all(path.read_bytes() == content for path, content in originals.items())
    assert verifier.main([str(config), str(labels)]) == 0


@pytest.mark.parametrize("corruption", [b"{}", b"null", b"invalid"])
def test_corrupt_candidate_rejected(config: Path, tmp_path: Path, corruption: bytes) -> None:
    labels = tmp_path / "candidate.json"
    labels.write_bytes(corruption)
    with pytest.raises((ReplayError, ValueError)):
        verifier.verify(config, labels)


def test_changed_original_window_rejected(config: Path, tmp_path: Path) -> None:
    labels = tmp_path / "candidate.json"
    asr_label_refine.run(config, labels)
    (tmp_path / "raw.json").write_bytes(b"{}")
    with pytest.raises((ReplayError, ValueError, StageError)):
        verifier.verify(config, labels)


def test_escaping_work_and_missing_candidate(config: Path, tmp_path: Path) -> None:
    with pytest.raises((OSError, ReplayError)):
        verifier.verify(config, tmp_path / "missing.json")
    with pytest.raises(ReplayError):
        verifier.verify(Path.cwd() / "README.md", config)
