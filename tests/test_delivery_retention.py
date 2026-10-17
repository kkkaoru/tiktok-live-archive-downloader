import json
from collections.abc import Mapping
from datetime import UTC, datetime
from pathlib import Path

import pytest

from replay import delivery_render, delivery_retention, disk_budget, edit_completion, r2_worker
from replay.core import ReplayError
from replay.r2_worker import RemoteObject

STEM = "2024-01-01_09-00-00_JST_replay-1-review"


class FakeStore:
    """In-memory R2 double for the offload that precedes every applied retirement."""

    def __init__(self) -> None:
        self.objects: dict[str, tuple[int, dict[str, str]]] = {}

    def head(self, key: str) -> RemoteObject | None:
        stored = self.objects.get(key)
        if stored is None:
            return None
        return RemoteObject(stored[0], datetime(2026, 10, 17, tzinfo=UTC), dict(stored[1]))

    def put(self, key: str, path: Path, metadata: Mapping[str, str]) -> None:
        self.objects[key] = (path.stat().st_size, dict(metadata))

    def get(self, key: str, destination: Path) -> int:
        raise AssertionError(f"retention never downloads {key} to {destination}")


def write(path: Path, value: object) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value), encoding="utf-8")
    return path


def media(path: Path, data: bytes = b"synthetic media") -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(data)
    return path


def configured(tmp_path: Path) -> Path:
    (tmp_path / "final.setting").write_text(
        '<setting><encoder name="H264iPod"><file-extension>m4v</file-extension>'
        "</encoder><filter-set/></setting>",
        encoding="utf-8",
    )
    return write(
        tmp_path / "render-config.json",
        {
            "version": 1,
            "recording": "replay-1",
            "captions": str(tmp_path / "captions"),
            "work": str(tmp_path / "work"),
            "storage": str(tmp_path / "storage"),
            "rate": 30,
            "width": 1080,
            "height": 1920,
            "cuts": str(tmp_path / "cuts.json"),
            "sourceChunks": str(tmp_path / "chunks.json"),
            "motion": {
                "template": str(tmp_path / "template.moti"),
                "expectedText": "Title",
                "finish": [],
            },
            "alphaPreset": str(tmp_path / "alpha.setting"),
            "encodePreset": str(tmp_path / "hevc.setting"),
            "native": str(tmp_path / "native"),
            "encoding": {"codec": "hevc"},
            "audio": {"source": str(tmp_path / "voice.wav"), "clock": str(tmp_path / "c.json")},
            "source": str(tmp_path / "2024-01-01_09-00-00_JST_replay-1.mp4"),
            "deliveryRoot": str(tmp_path / "deliveries" / "replay-1"),
            "finalPreset": str(tmp_path / "final.setting"),
            "calibration": str(tmp_path / "calibration.json"),
            "viewerRoot": str(tmp_path / "edits" / "replay-1"),
        },
    )


def produced(tmp_path: Path) -> delivery_render.Config:
    """Every producer declaration of a finished delivery with real synthetic media."""
    config = delivery_render.load_config(configured(tmp_path))
    work, storage = config.work, config.storage
    alpha = media(storage / "alpha" / "segment-0000-part-0000.mov")
    write(work / "alpha-jobs.json", {"jobs": [{"name": "alpha", "output": str(alpha)}]})
    part = media(work / "compose" / "parts-0000" / "part-0000.mov")
    request = write(work / "compose" / "parts-0000" / "part-0000.json", {})
    write(
        work / "compose" / "parts-0000" / "manifest.json",
        {"parts": [{"path": str(request), "frames": 60}]},
    )
    write(request.with_name("render.response.json"), {"outputPath": str(part)})
    joined = media(storage / "joined-video.mp4")
    write(work / "join-workflow.json", {"stages": [{"outputs": [str(joined)]}]})
    master = media(storage / "av-master.mov")
    write(work / "final-mux-workflow.json", {"stages": [{"outputs": [str(master)]}]})
    voice = media(work / "audio-selected" / "selected.wav")
    write(work / "audio-workflow.json", {"stages": [{"outputs": [str(voice)]}]})
    final = media(storage / "final" / f"{STEM}.m4v")
    write(work / "final-job.json", {"jobs": [{"name": "final", "output": str(final)}]})
    return config


def accepted(config: delivery_render.Config, monkeypatch: pytest.MonkeyPatch) -> Path:
    """A registered delivery whose verification and edit stages passed."""
    movie = media(delivery_retention.delivery_movie(config))
    movie.with_suffix(".srt").write_text("1\n0 --> 1\nhello\n", encoding="utf-8")
    write(
        movie.with_name(f"{movie.stem}.verification.json"),
        {"framesMatchCutPlan": True, "fullDecodePassed": True, "audioPackets": 5},
    )
    link = config.viewer_root / "revisions" / "native-v1" / "delivery" / movie.name
    link.parent.mkdir(parents=True)
    link.symlink_to(movie)
    monkeypatch.setattr(edit_completion, "assess_delivery", lambda _movie, *, root: {})
    return movie


def test_plan_lists_every_producer_declared_media_output(tmp_path: Path) -> None:
    config = produced(tmp_path)
    rows, skipped = delivery_retention.plan(config)
    assert sorted(Path(str(row["path"])).name for row in rows) == [
        "2024-01-01_09-00-00_JST_replay-1-review.m4v",
        "av-master.mov",
        "joined-video.mp4",
        "part-0000.mov",
        "segment-0000-part-0000.mov",
        "selected.wav",
    ]
    assert skipped == {}


def test_plan_keeps_fcpxml_referenced_media(tmp_path: Path) -> None:
    config = produced(tmp_path)
    timeline = tmp_path / "timeline.fcpxml"
    timeline.write_text(
        '<fcpxml><resources><asset id="r1"><media-rep kind="original-media" '
        f'src="file://{config.storage}/joined-video.mp4"/></asset></resources></fcpxml>',
        encoding="utf-8",
    )
    rows, skipped = delivery_retention.plan(config, [timeline])
    assert "joined-video.mp4" not in [Path(str(row["path"])).name for row in rows]
    assert skipped == {str(config.storage / "joined-video.mp4"): "referenced by an FCPXML timeline"}


def test_plan_skips_absent_inside_root_and_non_media_outputs(tmp_path: Path) -> None:
    config = produced(tmp_path)
    (config.storage / "av-master.mov").unlink()
    inside = media(config.delivery_root / "kept.mov")
    write(config.work / "audio-workflow.json", {"stages": [{"outputs": [str(inside)]}]})
    write(config.work / "join-workflow.json", {"stages": [{"outputs": ["/x/receipt.json"]}]})
    rows, skipped = delivery_retention.plan(config)
    assert sorted(Path(str(row["path"])).name for row in rows) == [
        "2024-01-01_09-00-00_JST_replay-1-review.m4v",
        "part-0000.mov",
        "segment-0000-part-0000.mov",
    ]
    assert skipped == {
        str(config.storage / "av-master.mov"): "already absent",
        str(inside): "inside a delivery or Viewer root",
        "/x/receipt.json": "not a media output",
    }


def test_plan_skips_a_symlinked_output(tmp_path: Path) -> None:
    config = produced(tmp_path)
    target = config.storage / "joined-video.mp4"
    other = media(tmp_path / "other.mp4")
    target.unlink()
    target.symlink_to(other)
    _rows, skipped = delivery_retention.plan(config)
    assert skipped == {str(target): "not a regular file"}


def test_require_accepted_returns_the_registered_delivery(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config = produced(tmp_path)
    movie = accepted(config, monkeypatch)
    assert delivery_retention.require_accepted(config) == movie


def test_require_accepted_needs_the_delivery_and_transcript(tmp_path: Path) -> None:
    config = produced(tmp_path)
    with pytest.raises(ReplayError, match="dated delivery and its transcript"):
        delivery_retention.require_accepted(config)


@pytest.mark.parametrize(
    "report",
    [
        {"framesMatchCutPlan": False, "fullDecodePassed": True, "audioPackets": 5},
        {"framesMatchCutPlan": True, "fullDecodePassed": False, "audioPackets": 5},
        {"framesMatchCutPlan": True, "fullDecodePassed": True, "audioPackets": 0},
        {"framesMatchCutPlan": True, "fullDecodePassed": True},
    ],
)
def test_require_accepted_rejects_a_failed_verification(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, report: dict[str, object]
) -> None:
    config = produced(tmp_path)
    movie = accepted(config, monkeypatch)
    write(movie.with_name(f"{movie.stem}.verification.json"), report)
    with pytest.raises(ReplayError, match="passed delivery media verification"):
        delivery_retention.require_accepted(config)


def test_require_accepted_rejects_unverified_edit_stages(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config = produced(tmp_path)
    accepted(config, monkeypatch)
    monkeypatch.setattr(
        edit_completion, "assess_delivery", lambda _movie, *, root: {"transcription": "missing"}
    )
    with pytest.raises(ReplayError, match="every edit stage: transcription"):
        delivery_retention.require_accepted(config)


def test_require_accepted_needs_the_viewer_link(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config = produced(tmp_path)
    movie = accepted(config, monkeypatch)
    (config.viewer_root / "revisions" / "native-v1" / "delivery" / movie.name).unlink()
    with pytest.raises(ReplayError, match="registered Viewer link"):
        delivery_retention.require_accepted(config)


def test_dry_run_changes_nothing_and_writes_no_receipt(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config = produced(tmp_path)
    accepted(config, monkeypatch)
    outcome = delivery_retention.retire(config.path)
    assert outcome.mode == "dry-run"
    assert outcome.eligible == 6
    assert (config.storage / "joined-video.mp4").is_file()
    assert (config.work / "audio-selected" / "selected.wav").is_file()
    assert delivery_retention.is_retired(config.path) is False


def test_apply_removes_intermediates_and_records_the_receipt(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config = produced(tmp_path)
    movie = accepted(config, monkeypatch)
    store = FakeStore()
    outcome = delivery_retention.retire(config.path, apply=True, store=store)
    assert (outcome.mode, outcome.eligible, outcome.retired, outcome.offloaded) == (
        "apply",
        6,
        6,
        6,
    )
    assert len(store.objects) == 1
    assert (
        json.loads(
            (config.viewer_root / "evidence" / "intermediate-offload.json").read_text(
                encoding="utf-8"
            )
        )["files"][0]["state"]
        == "retired"
    )
    assert outcome.logical_bytes == 6 * len(b"synthetic media")
    assert outcome.held == {}
    assert not (config.storage / "joined-video.mp4").exists()
    assert not (config.work / "audio-selected" / "selected.wav").exists()
    assert movie.is_file()
    assert (config.work / "alpha-jobs.json").is_file()
    assert delivery_retention.is_retired(config.path) is True


def test_apply_is_idempotent_after_retirement(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config = produced(tmp_path)
    accepted(config, monkeypatch)
    delivery_retention.retire(config.path, apply=True, store=FakeStore())
    again = delivery_retention.retire(config.path, apply=True, store=FakeStore())
    assert (again.eligible, again.retired) == (0, 0)


def test_apply_holds_a_row_changed_after_planning(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config = produced(tmp_path)
    accepted(config, monkeypatch)
    original = delivery_retention.plan

    def planned(
        configuration: delivery_render.Config, timelines: object = ()
    ) -> tuple[list[dict[str, object]], dict[str, str]]:
        result = original(configuration)
        (config.storage / "joined-video.mp4").write_bytes(b"changed after the plan was made")
        return result

    monkeypatch.setattr(delivery_retention, "plan", planned)
    outcome = delivery_retention.retire(config.path, apply=True, store=FakeStore())
    assert (outcome.offloaded, outcome.retired) == (5, 5)
    assert outcome.held == {"identity changed": 1}
    assert (config.storage / "joined-video.mp4").is_file()
    assert delivery_retention.is_retired(config.path) is False


def test_is_retired_is_false_without_a_config_or_work_directory(tmp_path: Path) -> None:
    assert delivery_retention.is_retired(tmp_path / "missing.json") is False
    assert delivery_retention.is_retired(write(tmp_path / "other.json", {"version": 1})) is False
    malformed = tmp_path / "malformed.json"
    malformed.write_text("changed", encoding="utf-8")
    assert delivery_retention.is_retired(malformed) is False
    assert delivery_retention.is_retired(write(tmp_path / "list.json", [])) is False


def test_main_prints_the_outcome(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    config = produced(tmp_path)
    accepted(config, monkeypatch)
    assert delivery_retention.main([str(config.path)]) == 0
    assert json.loads(capsys.readouterr().out)["eligible"] == 6


def test_main_reports_an_unaccepted_delivery(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    config = produced(tmp_path)
    assert delivery_retention.main([str(config.path), "--apply"]) == 1
    assert "DELIVERY_RETENTION_ERROR" in capsys.readouterr().err


def test_restore_rearms_a_retired_delivery(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    config = produced(tmp_path)
    accepted(config, monkeypatch)
    media(config.source)
    delivery_retention.retire(config.path, apply=True, store=FakeStore())
    assert delivery_retention.is_retired(config.path) is True
    monkeypatch.setattr(disk_budget, "require_headroom", lambda _paths: [])
    capsys.readouterr()
    assert delivery_retention.main([str(config.path), "--restore"]) == 0
    assert "reset" in capsys.readouterr().out
    assert delivery_retention.is_retired(config.path) is False


def test_apply_without_an_r2_store_unlinks_nothing(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config = produced(tmp_path)
    accepted(config, monkeypatch)
    with pytest.raises(ReplayError, match="needs an R2 store"):
        delivery_retention.retire(config.path, apply=True)
    assert (config.storage / "joined-video.mp4").is_file()
    assert delivery_retention.is_retired(config.path) is False


def test_main_applies_through_a_worker_session(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    config = produced(tmp_path)
    accepted(config, monkeypatch)
    buckets: list[str] = []

    class Session:
        def __init__(self, bucket: str) -> None:
            buckets.append(bucket)

        def __enter__(self) -> FakeStore:
            return FakeStore()

        def __exit__(self, *_exc: object) -> None:
            return None

    monkeypatch.setattr(r2_worker, "session", Session)
    assert delivery_retention.main([str(config.path), "--apply"]) == 0
    assert buckets == ["replay-intermediates"]
    assert json.loads(capsys.readouterr().out)["offloaded"] == 6
    assert delivery_retention.is_retired(config.path) is True
