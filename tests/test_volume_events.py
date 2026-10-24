"""Recovery after a volume detached under a step: same volume back, Compressor, receipts."""

import hashlib
import json
import os
import plistlib
import subprocess
from pathlib import Path

import pytest

from replay import compressor_batch, compressor_guard
from replay import volume_events as events

SSD = events.Volume("UUID-A", 5)


def test_a_volume_snapshot_lists_mounted_volumes_with_their_uuid(tmp_path: Path) -> None:
    (tmp_path / "ssd").mkdir()
    (tmp_path / "plain").mkdir()
    (tmp_path / "Macintosh HD").symlink_to("/")
    found = events.snapshot(
        tmp_path,
        uuid=lambda path: "UUID-A" if path.name == "ssd" else None,
        mounted=lambda path: path.name in ("ssd", "Macintosh HD"),
    )
    assert found == {
        str(tmp_path / "ssd"): events.Volume("UUID-A", (tmp_path / "ssd").stat().st_dev)
    }
    assert events.snapshot(tmp_path / "absent") == {}


def test_the_volume_uuid_comes_from_diskutil(monkeypatch: pytest.MonkeyPatch) -> None:
    answers = [plistlib.dumps({"VolumeUUID": "UUID-A"}), b"not a plist", plistlib.dumps({})]

    def run(command: list[str], **_kw: object) -> subprocess.CompletedProcess[bytes]:
        return subprocess.CompletedProcess(command, 0, answers.pop(0), b"")

    monkeypatch.setattr(subprocess, "run", run)
    assert events.volume_uuid(Path("/Volumes/ssd")) == "UUID-A"
    assert events.volume_uuid(Path("/Volumes/ssd")) is None
    assert events.volume_uuid(Path("/Volumes/ssd")) is None


def test_a_missing_or_remounted_volume_counts_as_detached() -> None:
    assert events.detached({"/Volumes/ssd": SSD}, {"/Volumes/ssd": SSD}) is False
    assert events.detached({"/Volumes/ssd": SSD}, {}) is True
    assert events.detached({"/Volumes/ssd": SSD}, {"/Volumes/ssd": events.Volume("UUID-A", 6)})


def test_the_step_waits_until_the_same_volume_is_back() -> None:
    readings = iter([{}, {}, {"/Volumes/ssd": events.Volume("UUID-A", 6)}])
    waits: list[float] = []
    reason = events.await_return(
        {"/Volumes/ssd": SSD}, current=lambda: next(readings), sleep=waits.append,
        clock=lambda: 0.0,
    )  # fmt: skip
    assert reason is None
    assert waits == [5.0, 5.0]


def test_a_different_volume_at_the_mount_or_none_in_time_is_refused() -> None:
    other = {"/Volumes/ssd": events.Volume("UUID-B", 6)}
    assert events.await_return({"/Volumes/ssd": SSD}, current=lambda: other) == (
        "/Volumes/ssd is a different volume now"
    )
    times = iter([0.0, 700.0])
    assert (
        events.await_return(
            {"/Volumes/ssd": SSD}, current=dict, sleep=lambda _s: None, clock=lambda: next(times)
        )
        == "/Volumes/ssd did not return within 600 s"
    )


def receipt(path: Path, pinned: Path) -> Path:
    digest = hashlib.sha256(pinned.read_bytes()).hexdigest()
    path.write_text(json.dumps({"outputs": [{"path": str(pinned), "sha256": digest}]}), "utf-8")
    return path


def test_recent_receipts_whose_files_changed_are_moved_aside(tmp_path: Path) -> None:
    good, bad, old = (tmp_path / name for name in ("good.mov", "bad.mov", "old.mov"))
    for movie in (good, bad, old):
        movie.write_bytes(b"frames")
    receipt(tmp_path / "good.json", good)
    receipt(tmp_path / "bad.json", bad)
    receipt(tmp_path / "old.json", old)
    (tmp_path / "cut.json").write_text('{"outputs": [', "utf-8")  # cut off by the detach
    bad.write_bytes(b"fram")  # the detach truncated it after its receipt was written
    old.write_bytes(b"x")
    stamp = (tmp_path / "good.json").stat().st_mtime
    os.utime(tmp_path / "old.json", (stamp - 1000, stamp - 1000))
    retired = events.recheck((tmp_path, tmp_path / "absent"), since=stamp - 10)
    assert sorted(path.name.split(".")[0] for path in retired) == ["bad", "cut"]
    assert (tmp_path / "good.json").is_file() and (tmp_path / "old.json").is_file()


def test_recovery_restarts_compressor_after_a_detach_and_rechecks(tmp_path: Path) -> None:
    restarted: list[str] = []
    readings = iter([{}, {"/Volumes/ssd": events.Volume("UUID-A", 6)}] * 2)
    reason = events.recover(
        {"/Volumes/ssd": SSD}, (tmp_path,), failed_at=0.0, current=lambda: next(readings),
        running=lambda: True, restart=lambda: restarted.append("compressor"),
        sleep=lambda _s: None, clock=lambda: 0.0,
    )  # fmt: skip
    assert reason is None
    assert restarted == ["compressor"]


def test_recovery_without_a_detach_leaves_compressor_alone(tmp_path: Path) -> None:
    restarted: list[str] = []
    reason = events.recover(
        {"/Volumes/ssd": SSD}, (tmp_path,), failed_at=0.0, current=lambda: {"/Volumes/ssd": SSD},
        running=lambda: True, restart=lambda: restarted.append("compressor"),
    )  # fmt: skip
    assert reason is None
    assert restarted == []


def test_recovery_stops_when_the_volume_does_not_return(tmp_path: Path) -> None:
    times = iter([0.0, 700.0])
    reason = events.recover(
        {"/Volumes/ssd": SSD}, (tmp_path,), failed_at=0.0, current=dict,
        running=lambda: False, restart=lambda: None, sleep=lambda _s: None,
        clock=lambda: next(times),
    )  # fmt: skip
    assert reason == "/Volumes/ssd did not return within 600 s"


def test_compressor_is_found_by_its_process_name(monkeypatch: pytest.MonkeyPatch) -> None:
    codes = [0, 1]
    monkeypatch.setattr(
        subprocess, "run", lambda command, **_kw: subprocess.CompletedProcess(command, codes.pop(0))
    )
    assert events.compressor_running() is True
    assert events.compressor_running() is False


def test_the_compressor_restart_uses_the_guard(monkeypatch: pytest.MonkeyPatch) -> None:
    apps: list[Path] = []
    monkeypatch.setattr(compressor_guard, "restart", apps.append)
    events.restart_compressor()
    assert apps == [compressor_batch.COMPRESSOR.parents[2]]
