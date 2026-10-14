import json
from pathlib import Path

import pytest

from replay import artifact_json
from replay.core import ReplayError
from replay.stages import StageError, file_digest


def test_reflow_accepts_large_lineage_without_raising_default_budget(tmp_path: Path) -> None:
    path = tmp_path / "reflow.json"
    path.write_text(
        json.dumps({"version": 1, "lineage": "x" * (9 * 1024 * 1024)}), encoding="utf-8"
    )
    result = artifact_json.caption_reflow_json(path)
    assert result["version"] == 1
    lineage = result["lineage"]
    assert isinstance(lineage, str)
    assert len(lineage) == 9437184
    with pytest.raises(ValueError, match="byte budget"):
        artifact_json._json(path)
    assert artifact_json.MAX_JSON_BYTES == 8388608


def test_reflow_budget_remains_bounded(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    path = tmp_path / "reflow.json"
    path.write_text('{"version":1}', encoding="utf-8")
    monkeypatch.setattr(artifact_json, "MAX_REFLOW_JSON_BYTES", 12)
    with pytest.raises(ValueError, match="byte budget"):
        artifact_json.caption_reflow_json(path)


def test_reflow_rejects_changed_digest(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    path = tmp_path / "reflow.json"
    path.write_text('{"version":1}', encoding="utf-8")
    monkeypatch.setattr(artifact_json, "file_digest", lambda path: "0" * 64)
    with pytest.raises(ValueError, match="changed"):
        artifact_json.caption_reflow_json(path)


def test_reflow_requires_object(tmp_path: Path) -> None:
    path = tmp_path / "reflow.json"
    path.write_text("[]", encoding="utf-8")
    with pytest.raises(ReplayError):
        artifact_json.caption_reflow_json(path)


def test_reflow_rejects_symlink(tmp_path: Path) -> None:
    path = tmp_path / "reflow.json"
    path.write_text('{"version":1}', encoding="utf-8")
    link = tmp_path / "linked.json"
    link.symlink_to(path)
    with pytest.raises(StageError):
        artifact_json.caption_reflow_json(link)


@pytest.mark.parametrize("value", [None, "", "bad\x00path", "relative.json"])
def test_canonical_artifact_path_required(value: object) -> None:
    with pytest.raises(ReplayError, match="canonical"):
        artifact_json._path(value)


def test_pinned_small_artifact_and_wrong_hash(tmp_path: Path) -> None:
    path = tmp_path / "metadata.json"
    path.write_text('{"version":1}', encoding="utf-8")
    assert artifact_json._pin({"path": str(path), "sha256": file_digest(path)}) == path
    assert artifact_json._json(path) == {"version": 1}
    with pytest.raises(ReplayError, match="hash changed"):
        artifact_json._pin({"path": str(path), "sha256": "0" * 64})
