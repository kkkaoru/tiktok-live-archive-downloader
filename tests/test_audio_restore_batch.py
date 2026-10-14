"""Synthetic tests for safe full-delivery batch selection."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from replay import audio_restore_batch as batch
from replay.core import ReplayError


def _source(repository: Path, replay: str = "replay-one", revision: str = "base") -> Path:
    source = repository / "edits" / replay / "revisions" / revision / "delivery" / "full-review.mp4"
    source.parent.mkdir(parents=True)
    source.touch()
    evidence = repository / "edits" / replay / "evidence"
    evidence.mkdir(exist_ok=True)
    (evidence / "source-probe.json").write_text(
        json.dumps({"format": {"filename": f"/synthetic/2024-01-01_09-00-00_JST_{replay}.mp4"}}),
        encoding="utf-8",
    )
    source.with_suffix(".srt").write_text("1\n00:00:01,000 --> 00:00:02,000\nvoice\n")
    return source


def _target(source: Path, repository: Path) -> Path:
    identifier = hashlib.sha256(
        str(source.relative_to(repository / "edits")).encode("utf-8")
    ).hexdigest()[:12]
    replay = source.relative_to(repository / "edits").parts[0]
    return source.parents[2] / (
        f"2024-01-01_09-00-00_JST_{replay}-review-{batch.DEFAULT_REVISION_NAME}-{identifier}"
    )


def test_discovery_is_sorted_and_excludes_its_own_revisions(tmp_path: Path) -> None:
    later = _source(tmp_path, "replay-z")
    earlier = _source(tmp_path, "replay-a")
    own_output = (
        tmp_path
        / "edits"
        / "replay-a"
        / "revisions"
        / f"{batch.DEFAULT_REVISION_NAME}-old"
        / "delivery"
        / "full-review.mp4"
    )
    own_output.parent.mkdir(parents=True)
    own_output.touch()
    own_revision = own_output.parents[1]
    metadata = own_revision / "metadata"
    metadata.mkdir()
    (metadata / "verification.json").write_text(
        json.dumps(
            {
                "status": "verified",
                "source": str(earlier.relative_to(tmp_path)),
                "delivery": str(own_output.relative_to(tmp_path)),
            }
        )
    )

    targets = batch.discover_current_deliveries(tmp_path)

    assert targets == ((earlier, _target(earlier, tmp_path)), (later, _target(later, tmp_path)))


def test_discovery_rejects_ambiguous_sources_missing_srt_and_bad_revision(
    tmp_path: Path,
) -> None:
    _source(tmp_path, revision="first")
    _source(tmp_path, revision="second")
    with pytest.raises(ReplayError, match="Multiple deliveries for one source"):
        batch.discover_current_deliveries(tmp_path)
    with pytest.raises(ValueError, match="one path component"):
        batch.discover_current_deliveries(tmp_path, "../outside")

    other_root = tmp_path / "other"
    missing = _source(other_root)
    missing.with_suffix(".srt").unlink()
    with pytest.raises(ReplayError, match="transcript is missing"):
        batch.discover_current_deliveries(other_root)


def test_verified_and_no_music_revisions_are_skipped(tmp_path: Path) -> None:
    source = _source(tmp_path)
    revision = _target(source, tmp_path)
    delivery = revision / "delivery" / source.name
    delivery.parent.mkdir(parents=True)
    delivery.touch()
    metadata = revision / "metadata"
    metadata.mkdir()
    report = {
        "status": "verified",
        "source": str(source.relative_to(tmp_path)),
        "delivery": str(delivery.relative_to(tmp_path)),
    }
    (metadata / "verification.json").write_text(json.dumps(report))
    assert batch.discover_current_deliveries(tmp_path) == ()

    delivery.unlink()
    (metadata / "verification.json").write_text(
        json.dumps({"status": "no-instrumental-content", "source": report["source"]})
    )
    assert batch.discover_current_deliveries(tmp_path) == ()
    delivery.touch()
    with pytest.raises(ReplayError, match="existing incomplete"):
        batch.discover_current_deliveries(tmp_path)


def test_dated_completed_output_is_idempotent(tmp_path: Path) -> None:
    source = _source(tmp_path)
    revision = _target(source, tmp_path)
    delivery = revision / "delivery" / "2024-01-01_09-00-00_JST_replay-one-review.mp4"
    delivery.parent.mkdir(parents=True)
    delivery.touch()
    metadata = revision / "metadata"
    metadata.mkdir()
    (metadata / "verification.json").write_text(
        json.dumps(
            {
                "status": "verified",
                "source": str(source.relative_to(tmp_path)),
                "delivery": str(delivery.relative_to(tmp_path)),
            }
        ),
        encoding="utf-8",
    )
    assert batch.discover_current_deliveries(tmp_path) == ()
    assert batch.discover_current_deliveries(tmp_path) == ()


def test_incomplete_existing_revision_is_never_reused(tmp_path: Path) -> None:
    source = _source(tmp_path)
    revision = _target(source, tmp_path)
    revision.mkdir()
    with pytest.raises(ReplayError, match="existing incomplete"):
        batch.discover_current_deliveries(tmp_path)


def test_existing_revision_symlink_cannot_escape_edits(tmp_path: Path) -> None:
    source = _source(tmp_path)
    revision = _target(source, tmp_path)
    revision.symlink_to(tmp_path / "outside")
    with pytest.raises(ReplayError, match="existing incomplete"):
        batch.discover_current_deliveries(tmp_path)


def test_verified_delivery_symlink_cannot_escape_edits(tmp_path: Path) -> None:
    source = _source(tmp_path)
    revision = _target(source, tmp_path)
    delivery = revision / "delivery" / source.name
    delivery.parent.mkdir(parents=True)
    outside = tmp_path / "outside.mp4"
    outside.touch()
    delivery.symlink_to(outside)
    metadata = revision / "metadata"
    metadata.mkdir()
    (metadata / "verification.json").write_text(
        json.dumps(
            {
                "status": "verified",
                "source": str(source.relative_to(tmp_path)),
                "delivery": str(delivery.relative_to(tmp_path)),
            }
        )
    )
    with pytest.raises(ReplayError, match="existing incomplete"):
        batch.discover_current_deliveries(tmp_path)


def test_invalid_existing_reports_are_not_treated_as_complete(tmp_path: Path) -> None:
    source = _source(tmp_path)
    revision = _target(source, tmp_path)
    metadata = revision / "metadata"
    metadata.mkdir(parents=True)
    report = metadata / "verification.json"
    for content in ("[]", "{", json.dumps({"status": "verified", "source": "wrong"})):
        report.write_text(content)
        with pytest.raises(ReplayError, match="existing incomplete"):
            batch.discover_current_deliveries(tmp_path)


def test_listening_approved_revisions_are_skipped_once_completed(tmp_path: Path) -> None:
    source = _source(tmp_path)
    revision = batch.discover_current_deliveries(tmp_path, batch.LISTENING_REVISION_NAME)[0][1]
    delivery = revision / "delivery" / source.name
    delivery.parent.mkdir(parents=True)
    delivery.touch()
    metadata = revision / "metadata"
    metadata.mkdir()
    (metadata / "verification.json").write_text(
        json.dumps(
            {
                "status": "listening-approved",
                "source": str(source.relative_to(tmp_path)),
                "delivery": str(delivery.relative_to(tmp_path)),
                "model": "mlx-community/mel-roformer-mlx@639de4848eb79a9e29cc8cba3c01170ddf165eef",
                "model_sha256": "86eee1472d825bf978bc6b6bb104bb71cb7950c4fea9dd192a8c55ed2db28b27",
                "full_audio_video_decode": "passed",
                "processing_minutes_per_audio_hour": 20.0,
            }
        ),
        encoding="utf-8",
    )
    assert batch.discover_current_deliveries(tmp_path, batch.LISTENING_REVISION_NAME) == ()
    data = json.loads((metadata / "verification.json").read_text(encoding="utf-8"))
    data["processing_minutes_per_audio_hour"] = 30.0
    (metadata / "verification.json").write_text(json.dumps(data), encoding="utf-8")
    with pytest.raises(ReplayError, match="existing incomplete"):
        batch.discover_current_deliveries(tmp_path, batch.LISTENING_REVISION_NAME)


def test_revision_root_rejects_paths_outside_revisions(tmp_path: Path) -> None:
    with pytest.raises(ReplayError, match="outside a revisions tree"):
        batch._revision_root(tmp_path / "full-review.mp4")


def test_batch_runs_sequentially_and_stops_after_first_failure(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    first = _source(tmp_path, "replay-a")
    second = _source(tmp_path, "replay-b")
    targets = ((first, tmp_path / "revision-a"), (second, tmp_path / "revision-b"))
    monkeypatch.setattr(batch, "discover_current_deliveries", lambda *args: targets)
    calls: list[Path] = []

    def restore(source: Path, revision: Path, device: str) -> dict[str, object]:
        calls.append(source)
        if source == second:
            raise ReplayError("quality gate failed")
        return {"status": "verified"}

    monkeypatch.setattr(batch, "restore_delivery", restore)
    with pytest.raises(ReplayError, match="quality gate failed"):
        batch.restore_current_deliveries(tmp_path, device="cpu")
    assert calls == [first, second]


def test_batch_returns_each_sequential_result(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source = _source(tmp_path)
    target = tmp_path / "new-revision"
    monkeypatch.setattr(batch, "discover_current_deliveries", lambda *args: ((source, target),))
    monkeypatch.setattr(
        batch,
        "restore_delivery",
        lambda actual_source, actual_target, device: {
            "source": str(actual_source),
            "revision": str(actual_target),
            "status": "verified",
        },
    )
    result = batch.restore_current_deliveries(tmp_path, device="cpu")
    assert result[0]["status"] == "verified"


def test_batch_uses_manual_processor_for_each_source(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    first = _source(tmp_path, "replay-a")
    second = _source(tmp_path, "replay-b")
    calls: list[Path] = []

    def approved(source: Path, revision: Path, device: str) -> dict[str, object]:
        calls.append(source)
        assert not revision.exists()
        return {"status": "listening-approved"}

    monkeypatch.setattr(batch, "apply_listening_approved", approved)
    results = batch.restore_current_deliveries(
        tmp_path, batch.LISTENING_REVISION_NAME, "mps", listening_approved=True
    )
    assert results == ({"status": "listening-approved"}, {"status": "listening-approved"})
    assert calls == [first, second]


def test_cli_reports_success_and_failure(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    monkeypatch.setattr(batch, "restore_current_deliveries", lambda *args, **kwargs: ())
    assert batch.main([]) == 0
    assert "[]" in capsys.readouterr().out

    def fail(*args: object, **kwargs: object) -> tuple[dict[str, object], ...]:
        raise ReplayError("blocked")

    monkeypatch.setattr(batch, "restore_current_deliveries", fail)
    assert batch.main([]) == 2
    assert "blocked" in capsys.readouterr().err
