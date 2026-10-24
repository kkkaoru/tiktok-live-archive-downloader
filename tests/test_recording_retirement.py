"""Retirement of a completed recording's intermediates: explicit manifest, guards and re-runs."""

import json
import os
from pathlib import Path

import pytest
from test_recording_viewer_steps import published

from replay import recording_retirement as retirement
from replay import recording_steps as steps
from replay import recording_viewer_steps as viewer
from replay.core import ReplayError


def write(path: Path, text: str = "x") -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")
    return path


def completed(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> steps.Pipeline:
    """A published, gated and uploaded recording with every retirable intermediate."""
    pipeline = published(tmp_path)
    movie = viewer.delivery(pipeline)
    media = tmp_path / "deliveries/source-caption-cleanup/replay-123-p1"
    write(media / "assembly/source.mov", "clean")
    (media / "assembly/source.mov.assembly").mkdir()
    os.link(media / "assembly/source.mov", media / "assembly/source.mov.assembly/source.mov")
    write(media / "sampled/chunk-00001/part.mp4", "chunk")
    clean_work = tmp_path / "clean-replay-123-p1"
    write(clean_work / "state/chunk-00001/chunk-receipt.json", '{"accepted": true}')
    write(clean_work / "state/chunk-00001/attempt-sampled/audit/band-tiles.mov", "tiles")
    description = write(
        tmp_path / "clean.description.json",
        json.dumps({"work": str(clean_work), "mediaRoot": str(media.parent)}),
    )
    write(
        tmp_path / "steps/cleanup.json",
        json.dumps({"outputs": {
            "description": {"path": str(description), "sha256": "d"},
            "cleanSource": str(media / "assembly/source.mov"),
            "chunkMedia": str(media / "sampled"),
        }}),
    )  # fmt: skip
    vocals = pipeline.work / "separation-v1/pipeline/vocals.wav"
    write(
        tmp_path / "steps/separation.json",
        json.dumps({"outputs": {"vocals": {"path": str(vocals), "sha256": "v"}}}),
    )
    write(pipeline.work / "paging-v1/prepared/tempo/candidate.wav", "tempo")
    write(pipeline.work / "caption-parts-v1/moti/w000-part-0000.moti", "moti")
    write(pipeline.work / "caption-parts-v1/parts.json", "[]")
    write(tmp_path / "edit/edit-config.json", json.dumps({"source": str(tmp_path / "source.mp4")}))
    write(tmp_path / "steps/summary.json", json.dumps({"outputs": {"gate": {"complete": True}}}))
    write(tmp_path / "steps/upload.json", "{}")
    for name in ("paging", "caption-parts"):
        write(tmp_path / f"steps/{name}.json", "{}")
    found = movie.stat()
    state = {"sources": {movie.stem: {"movie": {
        "size": found.st_size, "mtimeNs": str(found.st_mtime_ns)}}}}  # fmt: skip
    monkeypatch.setattr(viewer, "WEB_VIEWER", tmp_path / "web")
    write(tmp_path / "web/sync-state.json", json.dumps(state))
    return pipeline


def logged(pipeline: steps.Pipeline) -> list[dict[str, object]]:
    lines = (pipeline.logs / retirement.LOG).read_text(encoding="utf-8").splitlines()
    return [{k: v for k, v in json.loads(line).items() if k != "time"} for line in lines]


def test_a_completed_recording_retires_its_manifest_and_keeps_receipts_and_evidence(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    pipeline = completed(tmp_path, monkeypatch)
    result = retirement.retire(pipeline, retirement.Settings())
    media = tmp_path / "deliveries/source-caption-cleanup/replay-123-p1"
    tiles = tmp_path / "clean-replay-123-p1/state/chunk-00001/attempt-sampled/audit/band-tiles.mov"
    work = pipeline.work
    assert result == {
        "removed": [
            {"path": str(media), "bytes": 10, "reason": "cleaned-source"},
            {"path": str(tiles), "bytes": 5, "reason": "band-tiles"},
            {"path": str(work / "separation-v1/pipeline/vocals.wav"), "bytes": 1,
             "reason": "vocals"},
            {"path": str(work / "paging-v1/prepared/tempo/candidate.wav"), "bytes": 5,
             "reason": "tempo"},
            {"path": str(work / "caption-parts-v1/moti"), "bytes": 4, "reason": "moti"},
        ],
        "bytes": 25,
    }  # fmt: skip
    assert [row["removed"] for row in logged(pipeline)] == result["removed"]
    assert (tmp_path / "clean-replay-123-p1/state/chunk-00001/chunk-receipt.json").is_file()
    assert (work / "caption-parts-v1/parts.json").is_file()
    assert viewer.delivery(pipeline).is_file()
    assert (tmp_path / "steps/cleanup.json").is_file()
    saved = json.loads((tmp_path / "steps" / retirement.RECORD).read_text(encoding="utf-8"))
    assert [row["category"] for row in saved["retired"]] == [
        "cleaned-source", "band-tiles", "vocals", "tempo", "moti",
    ]  # fmt: skip
    assert saved["settings"] == {"retainIntermediates": False, "retirementHold": []}
    # A second retirement finds nothing left and keeps the record.
    assert retirement.retire(pipeline, retirement.Settings()) == {"removed": [], "bytes": 0}
    assert len(retirement.recorded(pipeline)) == 5


def test_evidence_the_delivery_inode_links_and_foreign_paths_are_kept(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    pipeline = completed(tmp_path, monkeypatch)
    media = tmp_path / "deliveries/source-caption-cleanup/replay-123-p1"
    # The gate's evidence names a file inside the cleaned-source directory.
    write(
        tmp_path / "deliveries/replay-123/revisions/old/metadata/native-evidence.json",
        json.dumps({"inputs": [{"path": str(media / "assembly/source.mov")}]}),
    )
    os.link(viewer.delivery(pipeline), pipeline.work / "caption-parts-v1/moti/copy.mp4")
    tempo = pipeline.work / "paging-v1/prepared/tempo/candidate.wav"
    tempo.unlink()
    tempo.symlink_to(tmp_path / "source.mp4")
    outside = write(tmp_path / "elsewhere/vocals.wav")
    write(
        tmp_path / "steps/separation.json",
        json.dumps({"outputs": {"vocals": {"path": str(outside), "sha256": "v"}}}),
    )
    removed = retirement.retire(pipeline, retirement.Settings())["removed"]
    assert isinstance(removed, list)
    assert [row["reason"] for row in removed] == ["band-tiles"]
    assert [(row["category"], row["why"]) for row in logged(pipeline) if "kept" in row] == [
        ("cleaned-source", f"named by completion evidence: {media / 'assembly/source.mov'}"),
        ("vocals", "outside the recording's work and media roots"),
        ("tempo", "a symbolic link"),
        ("moti", "holds the delivery"),
    ]
    assert media.is_dir() and outside.is_file() and tempo.is_symlink()


def test_retain_hold_and_an_unfinished_recording_remove_nothing(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    pipeline = completed(tmp_path, monkeypatch)
    vocals = pipeline.work / "separation-v1/pipeline/vocals.wav"
    assert retirement.retire(pipeline, retirement.Settings(retain=True)) == {
        "retained": "retainIntermediates"
    }
    assert retirement.retire(pipeline, retirement.Settings(hold=("edit",))) == {
        "retained": "retirementHold"
    }
    write(tmp_path / "web/sync-state.json", json.dumps({"sources": {}}))
    waiting = retirement.retire(pipeline, retirement.Settings())
    assert str(waiting["waiting"]).startswith("the web viewer does not hold the delivery")
    (tmp_path / "steps/upload.json").unlink()
    assert retirement.retire(pipeline, retirement.Settings()) == {
        "waiting": "the delivery has not been uploaded"
    }
    write(tmp_path / "steps/summary.json", json.dumps({"outputs": {"gate": {"complete": False}}}))
    assert retirement.retire(pipeline, retirement.Settings()) == {
        "waiting": "the summary step's completion gate has not passed"
    }
    assert vocals.is_file()
    assert [next(iter(row)) for row in logged(pipeline)] == [
        "retained", "retained", "waiting", "waiting", "waiting",
    ]  # fmt: skip


def test_a_later_run_forces_the_producers_of_retired_inputs_its_steps_need(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    pipeline = completed(tmp_path, monkeypatch)
    retirement.retire(pipeline, retirement.Settings())
    media = tmp_path / "deliveries/source-caption-cleanup/replay-123-p1"
    tempo = pipeline.work / "paging-v1/prepared/tempo/candidate.wav"
    assert retirement.forced_producers(pipeline, {"compose"}) == [
        ("cleanup", "compose", media),
        ("paging", "compose", tempo),
    ]
    assert retirement.forced_producers(pipeline, {"summary", "clips"}) == []
    assert retirement.forced_producers(pipeline, set()) == []
    # Once forced, a producer's receipt is gone; once re-made, its output exists again.
    (tmp_path / "steps/cleanup.json").unlink()
    write(tempo)
    assert retirement.forced_producers(pipeline, {"compose", "caption-render"}) == [
        ("caption-parts", "caption-render", pipeline.work / "caption-parts-v1/moti"),
    ]


def test_a_comment_cleaned_source_is_retired_for_a_youtube_recording(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    pipeline = completed(tmp_path, monkeypatch)
    movie = write(tmp_path / "deliveries/source-comment-clean/replay-123/clean.mp4", "comment")
    write(
        tmp_path / "steps/cleanup.json",
        json.dumps({"outputs": {
            "captionFree": {"path": "/s", "sha256": "s"},
            "commentClean": {"path": str(movie.with_name("comment-clean.json")), "sha256": "c"},
            "cleanSource": str(movie),
        }}),
    )  # fmt: skip
    assert retirement.manifest(pipeline)[0] == retirement.Entry(
        "comment-clean", movie, "cleanup", ("compose",)
    )
    removed = retirement.retire(pipeline, retirement.Settings())["removed"]
    assert isinstance(removed, list)
    assert removed[0] == {"path": str(movie), "bytes": 7, "reason": "comment-clean"}
    assert not movie.exists()


def test_unreadable_evidence_stops_retirement(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    pipeline = completed(tmp_path, monkeypatch)
    write(tmp_path / "edit/edit-config.json", "{not json")
    with pytest.raises(ReplayError, match="Unreadable evidence"):
        retirement.retire(pipeline, retirement.Settings())
    assert (pipeline.work / "separation-v1/pipeline/vocals.wav").is_file()


def test_a_recording_without_cleanup_or_separation_receipts_lists_only_its_work_files(
    tmp_path: Path,
) -> None:
    pipeline = published(tmp_path)
    (tmp_path / "steps/cleanup.json").unlink()
    assert [entry.category for entry in retirement.manifest(pipeline)] == ["tempo", "moti"]
    assert retirement.cleanup_work(pipeline) is None
    assert retirement.recorded(pipeline) == []
