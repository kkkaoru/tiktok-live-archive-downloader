"""The leaf configuration contract must not depend on caption/render producers."""

import json
from pathlib import Path

import pytest

from replay import edit_completion, edit_config, replay_preparation_plan
from replay.code_dependency_inventory import capture
from replay.core import ReplayError


def test_shared_type_and_leaf_dependency() -> None:
    assert edit_completion.EditConfig is edit_config.EditConfig
    modules = {item.path.stem for item in capture(("edit_config",))}
    assert "edit_config" in modules
    assert not modules.intersection(
        {"edit_completion", "head_observations", "head_geometry", "foreground_collection"}
    )


def test_actual_preparation_producer_closure_excludes_delivery_and_runner() -> None:
    modules = {path.stem for path in replay_preparation_plan._code_paths()}
    assert {"edit_config", "delivery_audio", "delivery_media", "phase_timer"} <= modules
    assert not modules.intersection(
        {"workflow", "edit_completion", "head_observations", "foreground_collection"}
    )


@pytest.fixture
def recording(tmp_path: Path) -> Path:
    root = tmp_path / "edits" / "replay-27"
    root.mkdir(parents=True)
    source = tmp_path / "source.mov"
    source.write_bytes(b"original")
    (root / "edit-config.json").write_text(
        json.dumps(
            {
                "version": 1,
                "replay_id": "27",
                "source": str(source),
                "source_size": 8,
                "engine": str(tmp_path),
                "executor_scope": str(tmp_path),
                "stages": dict.fromkeys(edit_config.REQUIRED_STAGES, True),
            }
        ),
        encoding="utf-8",
    )
    return root


def test_direct_and_compatibility_loader_same_value(recording: Path) -> None:
    assert edit_config.load_config(recording) == edit_completion.load_config(recording)
    assert edit_config.load_config(recording).replay_id == "27"


@pytest.mark.parametrize("stage", edit_config.REQUIRED_STAGES)
def test_every_stage_remains_mandatory(recording: Path, stage: str) -> None:
    path = recording / "edit-config.json"
    data = json.loads(path.read_text(encoding="utf-8"))
    data["stages"][stage] = False
    path.write_text(json.dumps(data), encoding="utf-8")
    with pytest.raises(ReplayError, match="every required stage"):
        edit_config.load_config(recording)
    with pytest.raises(ReplayError, match="every required stage"):
        edit_completion.load_config(recording)


def test_compatibility_reader_limit_remains_patchable(
    recording: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(edit_completion, "MAX_JSON_BYTES", 1)
    with pytest.raises(ReplayError, match="size limit"):
        edit_completion.load_config(recording)
    assert edit_config.load_config(recording).source_size == 8


def test_youtube_direct_loader_and_digest_hook(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root = tmp_path / "edits" / "youtube-abcdefghijk"
    source = tmp_path / "2024-01-01_09-00-00_JST_youtube-abcdefghijk.mkv"
    source.write_bytes(b"original")
    publication = tmp_path / "publication.json"
    publication.write_text('{"id":"abcdefghijk","release_timestamp":1704067200}', encoding="utf-8")
    edit_completion.create_youtube_config(
        root, source=source, publication=publication, engine=tmp_path, executor_scope=tmp_path
    )
    assert edit_config.load_config(root) == edit_completion.load_config(root)
    assert edit_config.load_config(root).platform == "youtube"
    monkeypatch.setattr(edit_completion, "_digest", lambda path: "changed")
    with pytest.raises(ReplayError, match="source changed"):
        edit_completion.load_config(root)
    assert edit_config.load_config(root).source_size == 8
