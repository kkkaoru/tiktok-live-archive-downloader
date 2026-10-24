"""Lease priority: least remaining GPU work first, from receipts and cleanup chunk receipts."""

import json
from pathlib import Path

import pytest

from replay import lease_priority


def receipt(recording: Path, step: str, wall: float) -> None:
    (recording / "steps").mkdir(parents=True, exist_ok=True)
    (recording / "steps" / f"{step}.json").write_text(json.dumps({"wallSeconds": wall}), "utf-8")


def chunks(work: Path, name: str, total: int, finished: int) -> None:
    clean = work / f"clean-{name}-p1"
    (clean / "configs").mkdir(parents=True)
    for index in range(total):
        (clean / "configs" / f"chunk-{index:05d}.json").write_text("{}", "utf-8")
    for index in range(finished):
        state = clean / "state" / f"chunk-{index:05d}"
        state.mkdir(parents=True)
        (state / "chunk-receipt.json").write_text("{}", "utf-8")


@pytest.fixture
def layout(tmp_path: Path) -> tuple[Path, Path]:
    recordings, work = tmp_path / "recordings", tmp_path
    done = recordings / "replay-1"
    for step in ("separation", "asr", "yolo-track", "onsets", "cleanup", "compose"):
        receipt(done, step, 100.0)
    receipt(done, "summary", 1.0)
    late = recordings / "replay-2"  # everything but compose: 100 s left
    for step in ("separation", "asr", "yolo-track", "onsets", "cleanup"):
        receipt(late, step, 300.0)
    early = recordings / "replay-3"  # half its cleanup (mean 200 s) and compose: 200 s
    for step in ("separation", "asr", "yolo-track", "onsets"):
        receipt(early, step, 300.0)
    chunks(work, "replay-3", total=4, finished=2)
    (recordings / "youtube-Ab-c" / "steps").mkdir(parents=True)  # nothing done yet
    (recordings / "unstarted").mkdir()
    return recordings, work


def test_recordings_are_ordered_by_remaining_gpu_work(layout: tuple[Path, Path]) -> None:
    recordings, work = layout
    assert lease_priority.ordered(recordings, work) == [
        ("2", 100.0),
        ("3", 200.0),
        ("youtube-Ab-c", pytest.approx(1233.333, abs=0.001)),
    ]


def test_apply_keeps_the_capacity_and_writes_the_order(tmp_path: Path) -> None:
    (tmp_path / "settings.json").write_text('{"capacity": 3, "priority": ["9"]}', "utf-8")
    assert lease_priority.apply(tmp_path, ["2", "3"]) == ["2", "3"]
    assert json.loads((tmp_path / "settings.json").read_text("utf-8")) == {
        "capacity": 3,
        "priority": ["2", "3"],
    }


def test_apply_stops_before_the_settings_size_limit(tmp_path: Path) -> None:
    kept = lease_priority.apply(tmp_path, [f"{index:019d}" for index in range(400)])
    assert len(kept) == 176
    assert len((tmp_path / "settings.json").read_bytes()) <= 4096


def test_main_prints_and_applies(
    layout: tuple[Path, Path], tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    recordings, work = layout
    leases = tmp_path / "leases"
    leases.mkdir()
    arguments = ["--recordings", str(recordings), "--work", str(work), "--leases", str(leases)]
    assert lease_priority.main(arguments) == 0
    assert not (leases / "settings.json").exists()
    assert lease_priority.main([*arguments, "--apply"]) == 0
    assert capsys.readouterr().out.splitlines()[-1] == "priority: 3 recordings"
    assert json.loads((leases / "settings.json").read_text("utf-8")) == {
        "capacity": 4,
        "priority": ["2", "3", "youtube-Ab-c"],
    }


def test_refresh_rewrites_only_a_changed_order(layout: tuple[Path, Path], tmp_path: Path) -> None:
    recordings, work = layout
    leases = tmp_path / "leases"
    leases.mkdir()
    (leases / "settings.json").write_text('{"capacity": 2, "priority": ["2", "3", "youtube-Ab-c"]}')
    before = (leases / "settings.json").stat().st_mtime_ns
    assert lease_priority.refresh(leases, recordings, work) == ["2", "3", "youtube-Ab-c"]
    assert (leases / "settings.json").stat().st_mtime_ns == before
    (leases / "settings.json").write_text('{"capacity": 2, "priority": ["3"]}', "utf-8")
    assert lease_priority.refresh(leases, recordings, work) == ["2", "3", "youtube-Ab-c"]
    assert json.loads((leases / "settings.json").read_text("utf-8"))["capacity"] == 2


def test_a_manual_priority_is_never_overwritten(layout: tuple[Path, Path], tmp_path: Path) -> None:
    recordings, work = layout
    (tmp_path / "settings.json").write_text('{"priority": ["9"], "manual": true}', "utf-8")
    assert lease_priority.refresh(tmp_path, recordings, work) is None
    assert json.loads((tmp_path / "settings.json").read_text("utf-8")) == {
        "priority": ["9"],
        "manual": True,
    }
