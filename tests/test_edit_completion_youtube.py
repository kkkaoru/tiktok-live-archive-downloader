"""Synthetic configuration/discovery tests, not download or native-render approval."""

import json
from pathlib import Path

import pytest

from replay.core import ReplayError
from replay.edit_completion import (
    completion_report,
    create_youtube_config,
    dated_deliveries,
    load_config,
    main,
)


@pytest.fixture
def original(tmp_path: Path) -> tuple[Path, Path, Path]:
    directory = tmp_path / "original"
    directory.mkdir()
    source = directory / "2024-01-01_09-00-00_JST_youtube-abcdefghijk.mkv"
    source.write_bytes(b"synthetic original")
    publication = directory / "publication.json"
    publication.write_text('{"id":"abcdefghijk","release_timestamp":1704067200}')
    return tmp_path / "youtube-abcdefghijk", source, publication


def test_youtube_config_binds_original_without_approving_empty_library(
    original: tuple[Path, Path, Path],
) -> None:
    root, source, publication = original
    path = create_youtube_config(
        root, source=source, publication=publication, engine=root.parent, executor_scope=root.parent
    )
    config = load_config(root)
    assert config.platform == "youtube"
    assert config.replay_id == "abcdefghijk"
    assert config.source.suffix == ".mkv"
    assert config.source_size == 18
    assert path.stat().st_mode & 0o777 == 0o600
    assert completion_report(root) == {
        "replay": "youtube-abcdefghijk",
        "complete": False,
        "state": "incomplete",
        "deliveries": [],
    }
    assert main(["check", str(root)]) == 2
    with pytest.raises(FileExistsError):
        create_youtube_config(
            root,
            source=source,
            publication=publication,
            engine=root.parent,
            executor_scope=root.parent,
        )


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("version", 2.0, "unsupported"),
        ("video_id", "abcdefghijx", "does not belong"),
        ("source", 1, "input fields"),
        ("source_size", True, "input fields"),
        ("source_size", 1, "source changed"),
        ("source_sha256", "0" * 64, "source changed"),
        ("publication_sha256", "0" * 64, "publication evidence changed"),
        ("stages", {}, "every required stage"),
        ("extra", True, "unsupported"),
    ],
)
def test_youtube_config_rejects_invalid_fields(
    original: tuple[Path, Path, Path],
    field: str,
    value: object,
    message: str,
) -> None:
    root, source, publication = original
    path = create_youtube_config(
        root, source=source, publication=publication, engine=root.parent, executor_scope=root.parent
    )
    data = json.loads(path.read_text())
    data[field] = value
    path.write_text(json.dumps(data))
    with pytest.raises(ReplayError, match=message):
        load_config(root)


@pytest.mark.parametrize("value", [False, 1])
def test_youtube_config_requires_literal_true_stages(
    original: tuple[Path, Path, Path],
    value: object,
) -> None:
    root, source, publication = original
    path = create_youtube_config(
        root, source=source, publication=publication, engine=root.parent, executor_scope=root.parent
    )
    data = json.loads(path.read_text())
    data["stages"]["source_caption_mask"] = value
    path.write_text(json.dumps(data))
    with pytest.raises(ReplayError, match="every required stage"):
        load_config(root)


def test_youtube_config_rejects_same_size_source_change(
    original: tuple[Path, Path, Path],
) -> None:
    root, source, publication = original
    create_youtube_config(
        root, source=source, publication=publication, engine=root.parent, executor_scope=root.parent
    )
    source.write_bytes(b"changed!! original")
    with pytest.raises(ReplayError, match="source changed"):
        load_config(root)


def test_youtube_config_rejects_changed_publication_bytes(
    original: tuple[Path, Path, Path],
) -> None:
    root, source, publication = original
    create_youtube_config(
        root, source=source, publication=publication, engine=root.parent, executor_scope=root.parent
    )
    publication.write_text('{"id":"abcdefghijk","release_timestamp":1704067200,"changed":true}')
    with pytest.raises(ReplayError, match="publication evidence changed"):
        load_config(root)


def test_youtube_config_rejects_unsupported_recording_root(
    original: tuple[Path, Path, Path],
) -> None:
    root, source, publication = original
    with pytest.raises(ReplayError, match="edits/youtube"):
        create_youtube_config(
            root.parent / "replay-7",
            source=source,
            publication=publication,
            engine=root.parent,
            executor_scope=root.parent,
        )


def test_youtube_config_requires_dated_original_and_local_publication(
    original: tuple[Path, Path, Path],
) -> None:
    root, source, publication = original
    moved = source.with_name("undated.mkv")
    source.rename(moved)
    with pytest.raises(ReplayError, match="dated original"):
        create_youtube_config(
            root,
            source=moved,
            publication=publication,
            engine=root.parent,
            executor_scope=root.parent,
        )


def test_youtube_config_requires_engine_directory(
    original: tuple[Path, Path, Path],
) -> None:
    root, source, publication = original
    with pytest.raises(ReplayError, match="directories must exist"):
        create_youtube_config(
            root,
            source=source,
            publication=publication,
            engine=root / "missing",
            executor_scope=root.parent,
        )


def test_youtube_config_rejects_removed_scope(original: tuple[Path, Path, Path]) -> None:
    root, source, publication = original
    scope = root.parent / "scope"
    scope.mkdir()
    create_youtube_config(
        root, source=source, publication=publication, engine=root.parent, executor_scope=scope
    )
    scope.rmdir()
    with pytest.raises(ReplayError, match="directories must exist"):
        load_config(root)


def test_youtube_cli_initialization_keeps_native_delivery_incomplete(
    original: tuple[Path, Path, Path],
) -> None:
    root, source, publication = original
    assert (
        main(
            [
                "init-youtube",
                str(root),
                "--source",
                str(source),
                "--publication",
                str(publication),
                "--engine",
                str(root.parent),
                "--executor-scope",
                str(root.parent),
            ]
        )
        == 0
    )
    directory = root / "revisions/native/delivery"
    directory.mkdir(parents=True)
    movie = directory / "2024-01-01_09-00-00_JST_youtube-abcdefghijk-review.mov"
    movie.write_bytes(b"not a verified native render")
    assert len(dated_deliveries(root)) == 1
    assert completion_report(root)["complete"] is False
    assert main(["check", str(root)]) == 2


@pytest.mark.parametrize(
    "metadata",
    [
        {"id": "abcdefghijx", "release_timestamp": 1704067200},
        {"id": "abcdefghijk", "release_timestamp": 1704067200, "is_live": True},
        {"id": "abcdefghijk", "upload_date": "20240101"},
    ],
)
def test_youtube_config_rejects_unverified_publication(
    original: tuple[Path, Path, Path],
    metadata: dict[str, object],
) -> None:
    root, source, publication = original
    publication.write_text(json.dumps(metadata))
    with pytest.raises(ReplayError):
        create_youtube_config(
            root,
            source=source,
            publication=publication,
            engine=root.parent,
            executor_scope=root.parent,
        )
    assert not (root / "edit-config.json").exists()
