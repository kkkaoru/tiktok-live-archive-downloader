import json
from collections.abc import Sequence
from pathlib import Path

import pytest

from replay.core import ReplayError
from replay.viewer_summary_publish import (
    load_saved_summary,
    main,
    merge_summaries,
    upload_objects,
)

SAVED = {
    "version": 1,
    "overview": "全体の要約です。",
    "sections": [
        {"startSeconds": 0, "endSeconds": 600, "text": "最初の区間です。"},
        {"startSeconds": 600, "endSeconds": 900.5, "text": "次の区間です。"},
    ],
}


def write_saved(transcript: Path, saved: object) -> None:
    folder = transcript.with_name(transcript.name + ".summaries")
    folder.mkdir()
    (folder / "current.json").write_text(json.dumps(saved), encoding="utf-8")


def make_viewer(tmp_path: Path, *, summary: object = None) -> tuple[Path, Path]:
    web = tmp_path / "web"
    (web / "data").mkdir(parents=True)
    library = tmp_path / "library"
    library.mkdir()
    movie = library / "a-review.mp4"
    movie.write_bytes(b"")
    transcript = library / "a-review.srt"
    transcript.write_text("1\n", encoding="utf-8")
    catalog = {
        "version": 1,
        "videos": [
            {
                "id": "a-review",
                "localPath": str(movie),
                "transcriptPath": str(transcript),
                "hasSummary": False,
            }
        ],
    }
    (web / "data" / "catalog.json").write_text(json.dumps(catalog), encoding="utf-8")
    data = {"id": "a-review", "cues": [], "summary": summary}
    (web / "data" / "a-review.json").write_text(json.dumps(data), encoding="utf-8")
    return web, transcript


def test_load_saved_summary_converts_to_viewer_shape(tmp_path: Path) -> None:
    transcript = tmp_path / "t.srt"
    write_saved(transcript, SAVED)
    assert load_saved_summary(transcript) == {
        "overview": "全体の要約です。",
        "sections": [
            {"start": 0.0, "end": 600.0, "text": "最初の区間です。"},
            {"start": 600.0, "end": 900.5, "text": "次の区間です。"},
        ],
    }


def test_load_saved_summary_returns_none_without_saved_file(tmp_path: Path) -> None:
    assert load_saved_summary(tmp_path / "t.srt") is None


def test_load_saved_summary_rejects_wrong_version(tmp_path: Path) -> None:
    transcript = tmp_path / "t.srt"
    write_saved(transcript, {**SAVED, "version": 2})
    with pytest.raises(ReplayError, match="unsupported version"):
        load_saved_summary(transcript)


def test_load_saved_summary_rejects_missing_sections(tmp_path: Path) -> None:
    transcript = tmp_path / "t.srt"
    write_saved(transcript, {**SAVED, "sections": []})
    with pytest.raises(ReplayError, match="no sections"):
        load_saved_summary(transcript)


def test_load_saved_summary_rejects_reversed_section(tmp_path: Path) -> None:
    transcript = tmp_path / "t.srt"
    write_saved(
        transcript, {**SAVED, "sections": [{"startSeconds": 9, "endSeconds": 1, "text": "x"}]}
    )
    with pytest.raises(ReplayError, match="reversed section"):
        load_saved_summary(transcript)


def test_load_saved_summary_rejects_boolean_time(tmp_path: Path) -> None:
    transcript = tmp_path / "t.srt"
    write_saved(
        transcript, {**SAVED, "sections": [{"startSeconds": True, "endSeconds": 2, "text": "x"}]}
    )
    with pytest.raises(ReplayError, match="invalid section time"):
        load_saved_summary(transcript)


def test_load_saved_summary_rejects_blank_text(tmp_path: Path) -> None:
    transcript = tmp_path / "t.srt"
    write_saved(transcript, {**SAVED, "overview": "  "})
    with pytest.raises(ReplayError, match="empty text"):
        load_saved_summary(transcript)


def test_merge_summaries_updates_data_and_catalog(tmp_path: Path) -> None:
    web, transcript = make_viewer(tmp_path)
    write_saved(transcript, SAVED)
    assert merge_summaries(web) == ["a-review"]
    data = json.loads((web / "data" / "a-review.json").read_text(encoding="utf-8"))
    catalog = json.loads((web / "data" / "catalog.json").read_text(encoding="utf-8"))
    assert data["summary"]["overview"] == "全体の要約です。"
    assert catalog["videos"][0]["hasSummary"] is True


def test_merge_summaries_is_idempotent(tmp_path: Path) -> None:
    web, transcript = make_viewer(tmp_path)
    write_saved(transcript, SAVED)
    merge_summaries(web)
    assert merge_summaries(web) == []


def test_merge_summaries_replaces_an_older_summary(tmp_path: Path) -> None:
    older = {"overview": "古い要約", "sections": [{"start": 0, "end": 1, "text": "古い"}]}
    web, transcript = make_viewer(tmp_path, summary=older)
    write_saved(transcript, SAVED)
    assert merge_summaries(web) == ["a-review"]


def test_merge_summaries_skips_video_without_saved_summary(tmp_path: Path) -> None:
    web, _ = make_viewer(tmp_path)
    assert merge_summaries(web) == []
    catalog = json.loads((web / "data" / "catalog.json").read_text(encoding="utf-8"))
    assert catalog["videos"][0]["hasSummary"] is False


def test_merge_summaries_skips_entry_without_transcript_path(tmp_path: Path) -> None:
    web, _ = make_viewer(tmp_path)
    catalog = {"version": 1, "videos": [{"id": "a-review", "transcriptPath": None}]}
    (web / "data" / "catalog.json").write_text(json.dumps(catalog), encoding="utf-8")
    assert merge_summaries(web) == []


def test_merge_summaries_rejects_unsafe_video_id(tmp_path: Path) -> None:
    web, transcript = make_viewer(tmp_path)
    write_saved(transcript, SAVED)
    catalog = {
        "version": 1,
        "videos": [{"id": "../x", "transcriptPath": str(transcript)}],
    }
    (web / "data" / "catalog.json").write_text(json.dumps(catalog), encoding="utf-8")
    with pytest.raises(ReplayError, match="invalid video id"):
        merge_summaries(web)


def test_merge_summaries_rejects_catalog_without_videos(tmp_path: Path) -> None:
    web, _ = make_viewer(tmp_path)
    (web / "data" / "catalog.json").write_text(json.dumps({"version": 1}), encoding="utf-8")
    with pytest.raises(ReplayError, match="no video list"):
        merge_summaries(web)


def test_upload_objects_puts_data_then_catalog(tmp_path: Path) -> None:
    calls: list[Sequence[str]] = []

    def runner(argv: Sequence[str]) -> int:
        calls.append(argv)
        return 0

    upload_objects(tmp_path, ["a-review"], bucket="bucket", runner=runner)
    assert [call[4] for call in calls] == ["data/a-review.json", "catalog.json"]
    assert calls[0][:4] == ["cf", "r2", "objects", "put"]
    assert calls[0][5:7] == ["--bucket-name", "bucket"]


def test_upload_objects_stops_at_first_failure(tmp_path: Path) -> None:
    calls: list[Sequence[str]] = []

    def runner(argv: Sequence[str]) -> int:
        calls.append(argv)
        return 1

    with pytest.raises(ReplayError, match="Upload failed for data/a-review.json"):
        upload_objects(tmp_path, ["a-review"], bucket="bucket", runner=runner)
    assert len(calls) == 1


def test_main_dry_run_merges_without_uploading(tmp_path: Path) -> None:
    web, transcript = make_viewer(tmp_path)
    write_saved(transcript, SAVED)
    calls: list[Sequence[str]] = []

    def runner(argv: Sequence[str]) -> int:
        calls.append(argv)
        return 0

    before = {path.name: path.read_bytes() for path in (web / "data").iterdir()}
    assert main(["--web-dir", str(web), "--bucket", "b", "--dry-run"], runner) == 0
    assert calls == []
    # A dry run changes nothing, so the real run still uploads the same video.
    assert {path.name: path.read_bytes() for path in (web / "data").iterdir()} == before
    assert main(["--web-dir", str(web), "--bucket", "b"], runner) == 0
    assert len(calls) == 2


def test_main_uploads_changed_objects(tmp_path: Path) -> None:
    web, transcript = make_viewer(tmp_path)
    write_saved(transcript, SAVED)
    calls: list[Sequence[str]] = []

    def runner(argv: Sequence[str]) -> int:
        calls.append(argv)
        return 0

    assert main(["--web-dir", str(web), "--bucket", "b"], runner) == 0
    assert len(calls) == 2


def test_main_uploads_nothing_when_nothing_changed(tmp_path: Path) -> None:
    web, _ = make_viewer(tmp_path)
    calls: list[Sequence[str]] = []

    def runner(argv: Sequence[str]) -> int:
        calls.append(argv)
        return 0

    assert main(["--web-dir", str(web), "--bucket", "b"], runner) == 0
    assert calls == []


def test_merge_summaries_can_include_unchanged_videos(tmp_path: Path) -> None:
    web, transcript = make_viewer(tmp_path)
    write_saved(transcript, SAVED)
    merge_summaries(web)
    assert merge_summaries(web, include_unchanged=True) == ["a-review"]


def test_main_all_uploads_unchanged_videos(tmp_path: Path) -> None:
    web, transcript = make_viewer(tmp_path)
    write_saved(transcript, SAVED)
    merge_summaries(web)
    calls: list[Sequence[str]] = []

    def runner(argv: Sequence[str]) -> int:
        calls.append(argv)
        return 0

    assert main(["--web-dir", str(web), "--bucket", "b", "--all"], runner) == 0
    assert len(calls) == 2
