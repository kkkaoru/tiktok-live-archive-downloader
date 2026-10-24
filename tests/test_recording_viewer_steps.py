"""Viewer extras of a published recording: summary, excitement and clip points."""

import json
import subprocess
import time
import urllib.error
import urllib.request
from collections.abc import Sequence
from pathlib import Path

import pytest
from test_recording_caption_steps import configure

from replay import delivery_offload, r2_fetch
from replay import recording_steps as steps
from replay import recording_viewer_steps as viewer
from replay.core import ReplayError


def published(tmp_path: Path) -> steps.Pipeline:
    """A pipeline whose movie was published with its transcript."""
    pipeline = steps.load(configure(tmp_path))
    folder = tmp_path / "deliveries/replay-123/revisions/pipeline-abc/delivery"
    folder.mkdir(parents=True)
    (folder / "2026-10-21_18-30-00_JST_replay-123-review.mp4").write_bytes(b"movie")
    (folder / "2026-10-21_18-30-00_JST_replay-123-review.srt").write_text("1\n", encoding="utf-8")
    movie = folder / "2026-10-21_18-30-00_JST_replay-123-review.mp4"
    (tmp_path / "steps/publish.json").write_text(
        json.dumps({"outputs": {"delivery": str(movie)}}), encoding="utf-8"
    )
    return pipeline


def web(tmp_path: Path, domain: str = "viewer.example.com,second.example.com") -> Path:
    root = tmp_path / "web"
    root.mkdir()
    (root / "secrets.json").write_text('{"CLIPS_API_TOKEN": "t0ken"}', encoding="utf-8")
    (root / "deploy.env").write_text(f"OTHER=1\nVIEWER_DOMAIN={domain}\n", encoding="utf-8")
    return root


def test_summary_generates_beside_the_transcript_then_syncs_data(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    pipeline = published(tmp_path)
    monkeypatch.setattr(viewer, "WEB_VIEWER", web(tmp_path))
    monkeypatch.setattr("shutil.which", lambda _name: "/bin/bun")
    monkeypatch.setattr(viewer, "gate", lambda _pipeline: {"complete": True})
    folder = tmp_path / "deliveries/replay-123/revisions/pipeline-abc/delivery"
    saved = folder / "2026-10-21_18-30-00_JST_replay-123-review.srt.summaries/current.json"
    commands: list[list[str]] = []

    def run(command: Sequence[str], _log: Path) -> None:
        commands.append(list(command))
        if "generate" in commands[-1]:
            saved.parent.mkdir()
            saved.write_text("{}", encoding="utf-8")

    outputs = viewer.summary(pipeline, run)
    assert commands == [
        [
            str(viewer.SUMMARY_CLI), "generate", "--library", str(tmp_path / "deliveries"),
            "--transcript", str(folder / "2026-10-21_18-30-00_JST_replay-123-review.srt"),
        ],
        ["/bin/bun", str(tmp_path / "web/src/bin/sync.ts"), "--data-only"],
    ]  # fmt: skip
    assert outputs == {
        "gate": {"complete": True},
        "summary": {
            "path": str(saved),
            "sha256": "44136fa355b3678a1146ad16f7e8649e94fb4fc21fe77e8310c060f61caaff8a",
        },
    }


def saving_run(folder: Path, commands: list[list[str]]) -> steps.Runner:
    saved = folder / "2026-10-21_18-30-00_JST_replay-123-review.srt.summaries/current.json"

    def run(command: Sequence[str], _log: Path) -> None:
        commands.append(list(command))
        if "generate" in commands[-1]:
            saved.parent.mkdir(exist_ok=True)
            saved.write_text("{}", encoding="utf-8")

    return run


def gated(tmp_path: Path) -> dict[str, object]:
    """A passed gate whose saved report is on disk, as ``gate`` returns it."""
    report = tmp_path / "steps/logs/gate-1.json"
    report.parent.mkdir(parents=True, exist_ok=True)
    report.write_text('{"complete": true}\n', encoding="utf-8")
    (tmp_path / "edit").mkdir(exist_ok=True)
    (tmp_path / "edit/edit-config.json").write_text("{}", encoding="utf-8")
    return {"complete": True, "report": steps.pin(report), "superseded": []}


def test_summarize_supersedes_then_gates_beside_the_generation_and_keeps_the_gate(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    pipeline = published(tmp_path)
    passed = gated(tmp_path)
    order: list[str] = []
    monkeypatch.setattr(viewer, "release_older", lambda _root, _published: order.append("release"))

    def gate(_pipeline: steps.Pipeline) -> dict[str, object]:
        order.append("gate")
        return passed

    monkeypatch.setattr(viewer, "gate", gate)
    folder = tmp_path / "deliveries/replay-123/revisions/pipeline-abc/delivery"
    commands: list[list[str]] = []
    outputs = viewer.summarize(pipeline, saving_run(folder, commands))
    assert [command[1] for command in commands] == ["generate"]
    assert order[0] == "release"
    assert sorted(order) == ["gate", "release"]
    assert outputs["gate"] == passed
    assert viewer.kept_gate(pipeline) == passed


def test_a_failed_generation_keeps_the_passed_gate_for_the_retry(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    pipeline = published(tmp_path)
    passed = gated(tmp_path)
    monkeypatch.setattr(viewer, "release_older", lambda _root, _published: None)
    monkeypatch.setattr(viewer, "gate", lambda _pipeline: passed)

    def failing(_command: Sequence[str], _log: Path) -> None:
        raise ReplayError("Producer failed (1)")

    with pytest.raises(ReplayError, match="Producer failed"):
        viewer.summarize(pipeline, failing)
    monkeypatch.setattr(viewer, "gate", lambda _pipeline: pytest.fail("gated again"))
    monkeypatch.setattr(
        viewer, "release_older", lambda _root, _published: pytest.fail("released again")
    )
    folder = tmp_path / "deliveries/replay-123/revisions/pipeline-abc/delivery"
    commands: list[list[str]] = []
    assert viewer.summarize(pipeline, saving_run(folder, commands))["gate"] == passed
    assert len(commands) == 1


def test_a_failed_gate_is_not_kept(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    pipeline = published(tmp_path)
    gated(tmp_path)
    monkeypatch.setattr(viewer, "release_older", lambda _root, _published: None)

    def failing(_pipeline: steps.Pipeline) -> dict[str, object]:
        raise ReplayError("The completion gate did not pass")

    monkeypatch.setattr(viewer, "gate", failing)
    folder = tmp_path / "deliveries/replay-123/revisions/pipeline-abc/delivery"
    with pytest.raises(ReplayError, match="did not pass"):
        viewer.summarize(pipeline, saving_run(folder, []))
    assert not (tmp_path / "steps/summarize-gate.json").exists()


def test_a_kept_gate_binds_the_publication_the_configuration_and_its_report(
    tmp_path: Path,
) -> None:
    pipeline = published(tmp_path)
    passed = gated(tmp_path)
    assert viewer.kept_gate(pipeline) is None
    kept = tmp_path / "steps/summarize-gate.json"
    binding = viewer.gate_binding(pipeline)
    kept.write_text(json.dumps({"binding": binding, "gate": passed}), encoding="utf-8")
    assert viewer.kept_gate(pipeline) == passed
    (tmp_path / "edit/edit-config.json").write_text('{"changed": 1}', encoding="utf-8")
    assert viewer.kept_gate(pipeline) is None
    kept.write_text(
        json.dumps({"binding": viewer.gate_binding(pipeline), "gate": passed}), encoding="utf-8"
    )
    (tmp_path / "steps/logs/gate-1.json").write_text("{}", encoding="utf-8")
    assert viewer.kept_gate(pipeline) is None
    (tmp_path / "steps/publish.json").write_text("{}", encoding="utf-8")
    assert viewer.kept_gate(pipeline) is None
    failed = {**passed, "complete": False}
    kept.write_text(
        json.dumps({"binding": viewer.gate_binding(pipeline), "gate": failed}), encoding="utf-8"
    )
    assert viewer.kept_gate(pipeline) is None


def test_releasing_older_revisions_leaves_the_publication_to_the_gate(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    pipeline = published(tmp_path)
    calls: list[tuple[Path, object]] = []

    def supersede(root: Path, assessed: object) -> list[Path]:
        calls.append((root, assessed))
        return []

    monkeypatch.setattr(viewer, "supersede_incomplete", supersede)
    monkeypatch.setattr(viewer, "delivery_label", lambda _root, path: f"storage:{path.name}")
    viewer.release_older(tmp_path / "edit", viewer.delivery(pipeline))
    assert calls == [
        (tmp_path / "edit", {"storage:2026-10-21_18-30-00_JST_replay-123-review.mp4": {}})
    ]


def test_summarize_after_an_older_summary_only_repeats_its_outputs(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    pipeline = published(tmp_path)
    monkeypatch.setattr(viewer, "gate", lambda _pipeline: pytest.fail("gated again"))
    older = {"gate": {"complete": True}, "summary": {"path": "s", "sha256": "0"}}
    (tmp_path / "steps/summary.json").write_text(json.dumps({"outputs": older}), encoding="utf-8")
    assert viewer.summarize(pipeline, lambda _command, _log: pytest.fail("ran")) == older


def summarized(tmp_path: Path, *, publish_digest: str, complete: bool | None) -> None:
    gated = {} if complete is None else {"gate": {"complete": complete, "report": "r"}}
    receipt = {"inputs": {str(tmp_path / "steps/publish.json"): publish_digest}, "outputs": gated}
    (tmp_path / "steps/summarize.json").write_text(json.dumps(receipt), encoding="utf-8")


def test_summary_reuses_the_gate_of_the_current_publication(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    pipeline = published(tmp_path)
    monkeypatch.setattr(viewer, "WEB_VIEWER", web(tmp_path))
    monkeypatch.setattr("shutil.which", lambda _name: "/bin/bun")
    monkeypatch.setattr(viewer, "gate", lambda _pipeline: pytest.fail("gated again"))
    summarized(
        tmp_path, publish_digest=steps.sha256(tmp_path / "steps/publish.json"), complete=True
    )
    folder = tmp_path / "deliveries/replay-123/revisions/pipeline-abc/delivery"
    commands: list[list[str]] = []
    outputs = viewer.summary(pipeline, saving_run(folder, commands))
    assert outputs["gate"] == {"complete": True, "report": "r"}
    assert commands[-1] == ["/bin/bun", str(tmp_path / "web/src/bin/sync.ts"), "--data-only"]


@pytest.mark.parametrize(("digest", "complete"), [("0" * 64, True), (None, False), (None, None)])
def test_summary_gates_itself_unless_summarize_passed_this_publication(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, digest: str | None, complete: bool | None
) -> None:
    pipeline = published(tmp_path)
    current = steps.sha256(tmp_path / "steps/publish.json")
    summarized(tmp_path, publish_digest=digest or current, complete=complete)
    monkeypatch.setattr(viewer, "gate", lambda _pipeline: {"complete": True, "fresh": True})
    assert viewer.summarized_gate(pipeline) is None
    monkeypatch.setattr(viewer, "WEB_VIEWER", web(tmp_path))
    monkeypatch.setattr("shutil.which", lambda _name: "/bin/bun")
    folder = tmp_path / "deliveries/replay-123/revisions/pipeline-abc/delivery"
    outputs = viewer.summary(pipeline, saving_run(folder, []))
    assert outputs["gate"] == {"complete": True, "fresh": True}


def test_upload_original_dry_runs_then_sends_only_the_originals(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    pipeline = published(tmp_path)
    monkeypatch.setattr(viewer, "WEB_VIEWER", web(tmp_path))
    monkeypatch.setattr("shutil.which", lambda _name: "/bin/bun")
    commands: list[list[str]] = []
    outputs = viewer.upload_original(pipeline, lambda command, _log: commands.append(list(command)))
    sync = ["/bin/bun", str(tmp_path / "web/src/bin/sync.ts"), "--originals-only"]
    assert commands == [[*sync, "--dry-run"], sync]
    assert outputs["originals"] == "sent"


def test_upload_original_after_the_delivery_upload_sends_nothing(tmp_path: Path) -> None:
    pipeline = published(tmp_path)
    (tmp_path / "steps/upload.json").write_text("{}", encoding="utf-8")
    outputs = viewer.upload_original(pipeline, lambda _command, _log: pytest.fail("synced"))
    assert outputs == {"originals": "sent with the delivery"}


def test_a_delivery_without_its_transcript_is_refused(tmp_path: Path) -> None:
    pipeline = published(tmp_path)
    folder = tmp_path / "deliveries/replay-123/revisions/pipeline-abc/delivery"
    (folder / "2026-10-21_18-30-00_JST_replay-123-review.srt").unlink()
    with pytest.raises(ReplayError, match="no transcript"):
        viewer.excitement(pipeline, lambda _command, _log: None)


def test_excitement_builds_from_the_delivered_transcript_and_movie(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    pipeline = published(tmp_path)
    folder = tmp_path / "deliveries/replay-123/revisions/pipeline-abc/delivery"
    sidecar = folder / "2026-10-21_18-30-00_JST_replay-123-review.srt.excitement.json"
    monkeypatch.setattr(
        subprocess,
        "run",
        lambda *args, **kwargs: subprocess.CompletedProcess(args, 0, "3806.866000\n", ""),
    )
    commands: list[list[str]] = []

    def run(command: Sequence[str], _log: Path) -> None:
        commands.append(list(command))
        sidecar.write_text("{}", encoding="utf-8")

    outputs = viewer.excitement(pipeline, run)
    assert commands == [
        [
            "/py", "-B", "-m", "replay.excitement", "build",
            "--transcript", str(folder / "2026-10-21_18-30-00_JST_replay-123-review.srt"),
            "--duration", "3806.866",
            "--movie", str(folder / "2026-10-21_18-30-00_JST_replay-123-review.mp4"),
        ]
    ]  # fmt: skip
    assert outputs == {
        "excitement": {
            "path": str(sidecar),
            "sha256": "44136fa355b3678a1146ad16f7e8649e94fb4fc21fe77e8310c060f61caaff8a",
        }
    }


def test_sync_without_bun_is_refused(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.setattr("shutil.which", lambda _name: None)
    with pytest.raises(ReplayError, match="bun is required"):
        viewer.sync(lambda _command, _log: None, tmp_path / "log")


def clip_data(root: Path) -> Path:
    data = root / "data" / "2026-10-21_18-30-00_JST_replay-123-review.json"
    data.parent.mkdir()
    data.write_text('{"id": "v"}', encoding="utf-8")
    return root


def test_clips_are_computed_on_the_mac_then_read_ready_without_a_reload(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    pipeline = published(tmp_path)
    root = clip_data(web(tmp_path))
    monkeypatch.setattr(viewer, "WEB_VIEWER", root)
    monkeypatch.setattr("shutil.which", lambda _name: "/bin/bun")
    commands: list[tuple[list[str], str]] = []
    seen: list[tuple[str, str, str | None]] = []

    def run(command: Sequence[str], log: Path) -> None:
        commands.append((list(command), log.name))

    def fetch(request: urllib.request.Request) -> dict[str, object]:
        seen.append((request.get_method(), request.full_url, request.get_header("Authorization")))
        return {"status": "ready", "record": {"clips": [{}, {}, {}]}}

    outputs = viewer.clips(pipeline, run, fetch=fetch)
    copied = pipeline.work / "clips-data.json"
    assert outputs == {"video": "2026-10-21_18-30-00_JST_replay-123-review", "clips": 3}
    assert commands == [
        (["/bin/bun", str(root / "src/bin/clips-mac.ts"), str(copied)], "clips.log")
    ]
    assert copied.read_text(encoding="utf-8") == '{"id": "v"}'
    assert seen == [
        (
            "GET",
            "https://viewer.example.com/api/clips/2026-10-21_18-30-00_JST_replay-123-review.json",
            "Bearer t0ken",
        )
    ]


def test_clips_not_ready_after_the_mac_run_fail(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    pipeline = published(tmp_path)
    monkeypatch.setattr(viewer, "WEB_VIEWER", clip_data(web(tmp_path)))
    monkeypatch.setattr("shutil.which", lambda _name: "/bin/bun")
    with pytest.raises(ReplayError, match="not ready after the Mac stored them: mismatch"):
        viewer.clips(pipeline, lambda _c, _l: None, fetch=lambda _request: {"status": "mismatch"})


def test_clips_without_bun_are_refused(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    pipeline = published(tmp_path)
    monkeypatch.setattr(viewer, "WEB_VIEWER", clip_data(web(tmp_path)))
    monkeypatch.setattr("shutil.which", lambda _name: None)
    with pytest.raises(ReplayError, match="bun is required to compute"):
        viewer.clips(pipeline, lambda _c, _l: None, fetch=lambda _request: {})


def test_viewer_settings_must_name_a_domain_and_a_token(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(viewer, "WEB_VIEWER", web(tmp_path, domain=""))
    with pytest.raises(ReplayError, match="CLIPS_API_TOKEN or VIEWER_DOMAIN"):
        viewer._viewer()


def test_a_ready_record_without_a_clip_list_counts_none(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    pipeline = published(tmp_path)
    monkeypatch.setattr(viewer, "WEB_VIEWER", clip_data(web(tmp_path)))
    monkeypatch.setattr("shutil.which", lambda _name: "/bin/bun")
    outputs = viewer.clips(
        pipeline, lambda _c, _l: None, fetch=lambda _request: {"status": "ready", "record": {}}
    )
    assert outputs == {"video": "2026-10-21_18-30-00_JST_replay-123-review", "clips": 0}


def test_the_gate_passes_then_supersedes_older_revisions(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    pipeline = published(tmp_path)
    monkeypatch.setattr(time, "time_ns", lambda: 7)
    rows = [
        {"delivery": "storage:new.mp4", "missingStages": {}},
        {"delivery": "storage:old.mp4", "missingStages": {"silence_cut": "missing"}},
    ]
    monkeypatch.setattr(
        viewer, "completion_report", lambda _root: {"complete": True, "deliveries": rows}
    )
    old = tmp_path / "edit/revisions/old/delivery/old.mp4"
    known: list[object] = []

    def supersede(_root: Path, assessed: object) -> list[Path]:
        known.append(assessed)
        return [old]

    monkeypatch.setattr(viewer, "supersede_incomplete", supersede)
    monkeypatch.setattr(viewer, "delivery_label", lambda _root, path: path.name)
    assert viewer.gate(pipeline) == {
        "complete": True,
        "report": {
            "path": str(tmp_path / "steps/logs/gate-7.json"),
            "sha256": "2181380284cd57ef614e829114606c911df4d49c18a09af2c56617df75b88a81",
        },
        "superseded": ["old.mp4"],
    }
    assert known == [{"storage:new.mp4": {}, "storage:old.mp4": {"silence_cut": "missing"}}]


def test_the_gate_fetches_an_offloaded_delivery_back_before_decoding_it(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    pipeline = published(tmp_path)
    order: list[str] = []

    def restore(path: Path, *, root: Path) -> bool:
        order.append(f"fetch {path.name} {root.name}")
        return True

    def report(_root: Path) -> dict[str, object]:
        order.append("gate")
        return {"complete": False}

    monkeypatch.setattr(r2_fetch, "restore_if_offloaded", restore)
    monkeypatch.setattr(viewer, "completion_report", report)
    with pytest.raises(ReplayError, match="did not pass"):
        viewer.gate(pipeline)
    assert order == ["fetch 2026-10-21_18-30-00_JST_replay-123-review.mp4 edit", "gate"]


def test_a_failing_gate_supersedes_nothing(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    pipeline = published(tmp_path)
    monkeypatch.setattr(viewer, "completion_report", lambda _root: {"complete": False})
    superseded: list[Path] = []

    def supersede(root: Path, _assessed: object) -> list[Path]:
        superseded.append(root)
        return []

    monkeypatch.setattr(viewer, "supersede_incomplete", supersede)
    with pytest.raises(ReplayError, match="completion gate did not pass"):
        viewer.gate(pipeline)
    assert superseded == []


def test_an_upload_that_left_its_movie_unsettled_fails(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(viewer, "WEB_VIEWER", tmp_path)
    movie = tmp_path / "m.mp4"
    movie.write_bytes(b"movie")
    found = movie.stat()
    state = tmp_path / "sync-state.json"
    state.write_text(
        json.dumps({"sources": {"m": {"movie": {"size": 5, "mtimeNs": str(found.st_mtime_ns)}}}}),
        encoding="utf-8",
    )
    viewer.require_uploaded("m", movie)
    state.write_text(
        json.dumps({"sources": {"m": {"movie": {"size": 4, "mtimeNs": str(found.st_mtime_ns)}}}}),
        encoding="utf-8",
    )
    with pytest.raises(ReplayError, match="did not upload m"):
        viewer.require_uploaded("m", movie)
    state.write_text(json.dumps({"sources": {}}), encoding="utf-8")
    with pytest.raises(ReplayError, match="did not upload m"):
        viewer.require_uploaded("m", movie)


def test_an_offloaded_movie_is_settled_by_its_record(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(viewer, "WEB_VIEWER", tmp_path)
    movie = tmp_path / "m.mp4"
    entry = delivery_offload.Entry(
        "delivery", str(movie), "media/m.mp4", 5, "a" * 64, "t", mtime_ns=1_700_000_000
    )
    delivery_offload.save(tmp_path / "replay-1", [entry])
    (tmp_path / "sync-state.json").write_text(
        json.dumps({"sources": {"m": {"movie": {"size": 5, "mtimeNs": "1700000000"}}}}),
        encoding="utf-8",
    )
    viewer.require_uploaded("m", movie, root=tmp_path / "replay-1")
    with pytest.raises(FileNotFoundError):
        viewer.require_uploaded("m", movie, root=tmp_path / "replay-2")
