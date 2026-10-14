import json
from dataclasses import replace
from pathlib import Path

import pytest

from replay import candidate_publication as candidate
from replay.core import ReplayError
from replay.edit_completion import load_config
from replay.fcp_caption_srt import srt_text
from replay.stages import file_digest


@pytest.fixture
def options(tmp_path: Path) -> candidate.Publication:
    root = tmp_path / "edits" / "replay-123"
    root.mkdir(parents=True)
    originals = tmp_path / "originals"
    originals.mkdir()
    source = originals / "2026-01-02_03-04-05_JST_replay-123.mp4"
    source.write_bytes(b"original")
    library = tmp_path / "ssd-library"
    folder = library / "replay-123"
    folder.mkdir(parents=True)
    movie = folder / "master.m4v"
    movie.write_bytes(b"candidate")
    (root / "edit-config.json").write_text(
        json.dumps(
            {
                "version": 1,
                "replay_id": "123",
                "source": str(source),
                "source_size": 8,
                "engine": str(tmp_path),
                "executor_scope": str(tmp_path),
                "storage_root": str(folder),
                "stages": dict.fromkeys(
                    (
                        "bgm_removal",
                        "silence_cut",
                        "source_caption_mask",
                        "transcription",
                        "dynamic_captions",
                    ),
                    True,
                ),
            }
        ),
        encoding="utf-8",
    )
    xml = tmp_path / "final.fcpxml"
    xml.write_text(
        '<fcpxml><library><event><project><sequence duration="2s"><spine>'
        '<asset-clip offset="0s" start="20s" duration="2s">'
        '<title offset="21s" duration="1s"><text>完成候補</text></title>'
        "</asset-clip></spine></sequence></project></event></library></fcpxml>",
        encoding="utf-8",
    )
    return candidate.Publication(root, library, movie, xml, "first")


def test_publication_preserves_original_and_selects_relative_candidate(
    options: candidate.Publication,
) -> None:
    result = candidate.publish(options)
    manifest = json.loads(result.read_text(encoding="utf-8"))
    assert manifest["status"] == "unverified-candidate"
    assert manifest["movie"] == (
        "revisions/first/delivery/2026-01-02_03-04-05_JST_replay-123-review.m4v"
    )
    assert manifest["movieTarget"] == "master.m4v"
    assert manifest["movieSize"] == 9
    assert manifest["requiredStagesVerified"] is False
    folder = options.library / "replay-123"
    linked = folder / manifest["movie"]
    assert linked.is_symlink()
    assert linked.readlink().as_posix() == "../../../master.m4v"
    assert linked.read_bytes() == b"candidate"
    assert (folder / manifest["transcript"]).read_text(encoding="utf-8") == (
        "1\n00:00:01,000 --> 00:00:02,000\n完成候補\n"
    )
    assert options.movie.read_bytes() == b"candidate"
    assert not (options.edit_root / "evidence").exists()


def test_two_revisions_require_explicit_selection_and_preserve_previous(
    options: candidate.Publication,
) -> None:
    selected = candidate.publish(options)
    old = selected.read_bytes()
    old_sha = file_digest(selected)
    second = replace(options, revision="second")
    with pytest.raises(ReplayError, match="exact previous"):
        candidate.publish(second)
    assert selected.read_bytes() == old
    candidate.publish(replace(second, previous_selection_sha256=old_sha))
    assert json.loads(selected.read_text(encoding="utf-8"))["movie"] == (
        "revisions/second/delivery/2026-01-02_03-04-05_JST_replay-123-review.m4v"
    )
    assert (
        options.library / "replay-123/revisions/second/delivery/previous-selection.json"
    ).read_bytes() == old
    assert (
        options.library / "replay-123/revisions/first/delivery/candidate.json"
    ).read_bytes() == old


def test_republication_cannot_overwrite_unique_artifacts(options: candidate.Publication) -> None:
    selected = candidate.publish(options)
    with pytest.raises(FileExistsError):
        candidate.publish(replace(options, previous_selection_sha256=file_digest(selected)))


@pytest.mark.parametrize("revision", ["..", "../escape", "a/b", "", "x" * 97])
def test_invalid_revision(options: candidate.Publication, revision: str) -> None:
    with pytest.raises(ReplayError, match="revision"):
        candidate.publish(replace(options, revision=revision))


def test_missing_previous_selection(options: candidate.Publication) -> None:
    with pytest.raises(ReplayError, match="previous candidate selection is missing"):
        candidate.publish(replace(options, previous_selection_sha256="a" * 64))


def test_symlink_input_refused(options: candidate.Publication) -> None:
    link = options.movie.with_name("linked.mp4")
    link.symlink_to(options.movie)
    with pytest.raises(ReplayError, match="symbolic"):
        candidate.publish(replace(options, movie=link))


def test_outside_library_refused(options: candidate.Publication) -> None:
    movie = options.edit_root / "outside.mp4"
    movie.write_bytes(b"outside")
    with pytest.raises(ReplayError, match="inside the explicit library"):
        candidate.publish(replace(options, movie=movie))


def test_wrong_library_refused(options: candidate.Publication) -> None:
    with pytest.raises(ReplayError, match="match the edit config"):
        candidate.publish(replace(options, library=options.library.parent))


def test_private_path_refused(tmp_path: Path) -> None:
    with pytest.raises(ReplayError, match="private"):
        candidate._plain(tmp_path / "private" / "file")


@pytest.mark.parametrize(
    "name",
    [
        "undated.mp4",
        "2026-99-01_00-00-00_JST_replay-123.mp4",
        "2026-1-02_03-04-05_JST_replay-123.mp4",
    ],
)
def test_invalid_original_date(options: candidate.Publication, name: str) -> None:
    config = load_config(options.edit_root)
    with pytest.raises(ReplayError, match="source date|source date and identifier"):
        candidate._stem(replace(config, source=Path(name)), options.edit_root)


def test_changed_xml_is_not_published(
    options: candidate.Publication,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    original = srt_text

    def change(path: Path) -> str:
        text = original(path)
        path.write_text(path.read_text(encoding="utf-8") + " ", encoding="utf-8")
        return text

    monkeypatch.setattr(candidate, "srt_text", change)
    with pytest.raises(ReplayError, match="changed during"):
        candidate.publish(options)


def test_changed_media_is_not_selected(
    options: candidate.Publication,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    original = candidate._previous

    def change(path: Path, expected: str | None) -> str | None:
        result = original(path, expected)
        options.movie.write_bytes(b"changed")
        return result

    monkeypatch.setattr(candidate, "_previous", change)
    with pytest.raises(ReplayError, match="changed before"):
        candidate.publish(options)


def test_invalid_existing_selection(options: candidate.Publication) -> None:
    evidence = options.library / "replay-123/evidence"
    evidence.mkdir()
    (evidence / "candidate-selection.json").mkdir()
    with pytest.raises(ReplayError, match="bounded regular"):
        candidate.publish(options)


def test_wrong_source_probe_cannot_replace_current_selection(
    options: candidate.Publication,
) -> None:
    evidence = options.library / "replay-123/evidence"
    evidence.mkdir()
    (evidence / "source-probe.json").write_text(
        '{"format":{"filename":"/different.mp4"}}', encoding="utf-8"
    )
    with pytest.raises(ReplayError, match="different original"):
        candidate.publish(options)
    assert not (evidence / "candidate-selection.json").exists()


def test_invalid_probe_directory(options: candidate.Publication) -> None:
    evidence = options.library / "replay-123/evidence"
    (evidence / "source-probe.json").mkdir(parents=True)
    with pytest.raises(ReplayError, match="bounded regular"):
        candidate.publish(options)


def test_output_directory_escape_refused(options: candidate.Publication) -> None:
    outside = options.edit_root / "outside"
    outside.mkdir()
    (options.library / "replay-123/revisions").symlink_to(outside)
    with pytest.raises(ReplayError, match="symbolic"):
        candidate.publish(options)
    assert list(outside.iterdir()) == []


def test_changed_config_refused(
    options: candidate.Publication, monkeypatch: pytest.MonkeyPatch
) -> None:
    original = candidate._previous

    def change(path: Path, expected: str | None) -> str | None:
        value = original(path, expected)
        config = options.edit_root / "edit-config.json"
        config.write_text(config.read_text(encoding="utf-8") + " ", encoding="utf-8")
        return value

    monkeypatch.setattr(candidate, "_previous", change)
    with pytest.raises(ReplayError, match="config changed"):
        candidate.publish(options)


def test_media_change_during_writes_keeps_previous_selection(
    options: candidate.Publication, monkeypatch: pytest.MonkeyPatch
) -> None:
    original = candidate._write_new

    def change(path: Path, body: str) -> None:
        original(path, body)
        if path.name == "candidate.json":
            options.movie.write_bytes(b"still encoding")

    monkeypatch.setattr(candidate, "_write_new", change)
    with pytest.raises(ReplayError, match="partial revision retained"):
        candidate.publish(options)
    assert not (options.library / "replay-123/evidence/candidate-selection.json").exists()


def test_cli_export_and_no_overwrite(options: candidate.Publication) -> None:
    srt = options.fcpxml.with_suffix(".srt")
    assert candidate.main(["--fcpxml", str(options.fcpxml), "--srt-output", str(srt)]) == 0
    assert candidate.main(["--fcpxml", str(options.fcpxml), "--srt-output", str(srt)]) == 1
    assert srt.read_text(encoding="utf-8") == ("1\n00:00:01,000 --> 00:00:02,000\n完成候補\n")


def test_cli_publication(options: candidate.Publication) -> None:
    assert (
        candidate.main(
            [
                "--fcpxml",
                str(options.fcpxml),
                "--edit-root",
                str(options.edit_root),
                "--library",
                str(options.library),
                "--movie",
                str(options.movie),
                "--revision",
                "first",
            ]
        )
        == 0
    )


def test_cli_incomplete(options: candidate.Publication) -> None:
    assert candidate.main(["--fcpxml", str(options.fcpxml)]) == 1
    assert (
        candidate.main(
            ["--fcpxml", str(options.fcpxml), "--srt-output", "file.srt", "--revision", "bad"]
        )
        == 1
    )
