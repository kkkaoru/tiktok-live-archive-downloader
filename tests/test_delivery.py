"""Synthetic source-bound naming checks; never infer dates from IDs or mtimes."""

import hashlib
import json
import subprocess
from pathlib import Path

import pytest

from replay.core import ReplayError
from replay.delivery import (
    delivery_lock,
    delivery_stem,
    require_new_delivery,
    retire_processed_source,
)


def test_stable_delivery_name(tmp_path: Path) -> None:
    root = tmp_path / "replay-123"
    evidence = root / "evidence"
    evidence.mkdir(parents=True)
    (evidence / "source-probe.json").write_text(
        json.dumps({"format": {"filename": "/original/2024-01-01_09-00-00_JST_replay-123.mp4"}}),
        encoding="utf-8",
    )
    assert delivery_stem(root) == "2024-01-01_09-00-00_JST_replay-123-review"
    assert delivery_stem(root) == "2024-01-01_09-00-00_JST_replay-123-review"


@pytest.mark.parametrize(
    "content",
    [
        "{}",
        "{",
        ' {"format":{"filename":null}}',
        '{"format":{"filename":"relative.mp4"}}',
        '{"format":{"filename":"/original/2024-01-01_09-00-00_JST_replay-456.mp4"}}',
        '{"format":{"filename":"/original/2024-02-30_09-00-00_JST_replay-123.mp4"}}',
        '{"format":{"filename":"/original/2024-1-1_09-00-00_JST_replay-123.mp4"}}',
    ],
)
def test_invalid_date_never_creates_output(tmp_path: Path, content: str) -> None:
    root = tmp_path / "replay-123"
    evidence = root / "evidence"
    evidence.mkdir(parents=True)
    (evidence / "source-probe.json").write_text(content, encoding="utf-8")
    with pytest.raises(ReplayError):
        delivery_stem(root)
    assert not (root / "delivery").exists()


def test_exact_checkout_relative_source_is_supported(tmp_path: Path) -> None:
    root = tmp_path / "replay-123"
    evidence = root / "evidence"
    evidence.mkdir(parents=True)
    (evidence / "source-probe.json").write_text(
        '{"format":{"filename":"downloads/2024-01-01_09-00-00_JST_replay-123.mp4"}}',
        encoding="utf-8",
    )
    assert delivery_stem(root) == "2024-01-01_09-00-00_JST_replay-123-review"


def test_source_lock_rejects_concurrent_writer_and_releases_after_failure(tmp_path: Path) -> None:
    root = tmp_path / "replay-123"
    evidence = root / "evidence"
    evidence.mkdir(parents=True)
    (evidence / "source-probe.json").write_text(
        '{"format":{"filename":"/original/2024-01-01_09-00-00_JST_replay-123.mp4"}}',
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="processor failed"), delivery_lock(root):
        with pytest.raises(ReplayError, match="Another process"), delivery_lock(root):
            pytest.fail("Concurrent writer entered")
        raise ValueError("processor failed")
    with delivery_lock(root):
        assert (evidence / "replay-123.delivery.lock").is_file()


def test_changed_date_or_missing_selected_video_never_authorizes_regeneration(
    tmp_path: Path,
) -> None:
    root = tmp_path / "replay-123"
    evidence = root / "evidence"
    evidence.mkdir(parents=True)
    (evidence / "source-probe.json").write_text(
        '{"format":{"filename":"/original/2024-01-02_09-00-00_JST_replay-123.mp4"}}',
        encoding="utf-8",
    )
    # Even an unreadable selection is a stop condition, not permission to regenerate.
    (evidence / "2024-01-01_09-00-00_JST_replay-123-review-selection.json").write_text(
        "{}", encoding="utf-8"
    )
    with pytest.raises(ReplayError, match="already has a dated delivery"):
        require_new_delivery(root)
    assert not (root / "revisions").exists()


def test_legacy_revision_writer_cannot_recreate_undated_duplicates(tmp_path: Path) -> None:
    script = Path(__file__).resolve().parents[1] / "scripts" / "build-expander-revision.sh"
    output = tmp_path / "undated-revision"
    result = subprocess.run(
        ["bash", str(script), str(tmp_path / "source.mp4"), str(output)],
        capture_output=True,
        text=True,
        check=False,
        timeout=10,
    )
    assert result.returncode == 2
    assert "publication is retired" in result.stderr
    assert not output.exists()


@pytest.mark.parametrize("changed", [False, True])
def test_retirement_protects_originals_and_changed_sources(tmp_path: Path, changed: bool) -> None:
    root = tmp_path / "replay-123"
    evidence = root / "evidence"
    evidence.mkdir(parents=True)
    (evidence / "source-probe.json").write_text(
        '{"format":{"filename":"/original/2024-01-01_09-00-00_JST_replay-123.mp4"}}',
        encoding="utf-8",
    )
    output = (
        root / "revisions" / "new" / "delivery" / "2024-01-01_09-00-00_JST_replay-123-review.mp4"
    )
    output.parent.mkdir(parents=True)
    output.write_bytes(b"verified output")
    source = (
        root / "revisions" / "old" / "delivery" / "full-review.mp4"
        if changed
        else tmp_path / "original.mp4"
    )
    source.parent.mkdir(parents=True, exist_ok=True)
    source.write_bytes(b"original input")
    before = source.stat()
    if changed:
        source.write_bytes(b"concurrently changed input")
    with pytest.raises(ReplayError):
        retire_processed_source(source, output, before)
    assert source.is_file()
    assert output.read_bytes() == b"verified output"
    assert not (evidence / "2024-01-01_09-00-00_JST_replay-123-review-selection.json").exists()


def test_missing_evidence(tmp_path: Path) -> None:
    with pytest.raises(ReplayError, match="broadcast date"):
        delivery_stem(tmp_path / "replay-123")


def test_symlink_evidence_is_rejected(tmp_path: Path) -> None:
    root = tmp_path / "replay-123"
    evidence = root / "evidence"
    evidence.mkdir(parents=True)
    outside = tmp_path / "probe.json"
    outside.write_text("{}", encoding="utf-8")
    (evidence / "source-probe.json").symlink_to(outside)
    with pytest.raises(ReplayError, match="inside"):
        delivery_stem(root)


def test_oversized_evidence_is_rejected(tmp_path: Path) -> None:
    root = tmp_path / "replay-123"
    evidence = root / "evidence"
    evidence.mkdir(parents=True)
    (evidence / "source-probe.json").write_bytes(b" " * 1_048_577)
    with pytest.raises(ReplayError, match="size limit"):
        delivery_stem(root)


def test_retirement_keeps_a_superseded_selection_marker(tmp_path: Path) -> None:
    root = tmp_path / "replay-123"
    evidence = root / "evidence"
    evidence.mkdir(parents=True)
    (evidence / "source-probe.json").write_text(
        '{"format":{"filename":"/original/2024-01-01_09-00-00_JST_replay-123.mp4"}}',
        encoding="utf-8",
    )
    previous = evidence / "2024-01-01_09-00-00_JST_replay-123-review-selection.json"
    previous.write_text('{"status": "selected-existing-delivery"}', encoding="utf-8")
    output = (
        root / "revisions" / "new" / "delivery" / "2024-01-01_09-00-00_JST_replay-123-review.mp4"
    )
    output.parent.mkdir(parents=True)
    output.write_bytes(b"verified output")
    source = root / "revisions" / "render" / "delivery" / "full-review.mp4"
    source.parent.mkdir(parents=True)
    source.write_bytes(b"captioned render")
    retire_processed_source(source, output, source.stat())
    assert not source.exists()
    assert previous.read_text(encoding="utf-8") == '{"status": "selected-existing-delivery"}'
    marker = evidence / (
        "2024-01-01_09-00-00_JST_replay-123-review-selection-"
        f"{hashlib.sha256(b'verified output').hexdigest()[:16]}.json"
    )
    assert json.loads(marker.read_text(encoding="utf-8"))["status"] == "selected-verified-successor"
