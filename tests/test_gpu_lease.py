"""Machine-wide GPU leases: flock slots served to one recording at a time, live settings."""

import json
import os
import subprocess
import sys
import threading
from pathlib import Path

import pytest

from replay import gpu_lease

HOLDER = """
import sys, time
from pathlib import Path
from replay import gpu_lease
waiter = gpu_lease.Waiter(Path(sys.argv[1]), sys.argv[2], start=int(sys.argv[3]))
descriptor = waiter.try_acquire()
print("held" if descriptor is not None else "waiting", flush=True)
time.sleep(60)
"""


def holder(directory: Path, replay_id: str, start: int) -> subprocess.Popen[str]:
    process = subprocess.Popen(
        [sys.executable, "-c", HOLDER, str(directory), replay_id, str(start)],
        stdout=subprocess.PIPE,
        text=True,
    )
    assert process.stdout is not None
    process.stdout.readline()
    return process


def test_missing_settings_mean_the_defaults(tmp_path: Path) -> None:
    assert gpu_lease.settings(tmp_path) == (4, ())


def test_settings_are_read_from_the_lease_directory(tmp_path: Path) -> None:
    (tmp_path / "settings.json").write_text('{"capacity": 2, "priority": ["9", "8"]}', "utf-8")
    assert gpu_lease.settings(tmp_path) == (2, ("9", "8"))


def test_invalid_settings_are_rejected(tmp_path: Path) -> None:
    (tmp_path / "settings.json").write_text('{"capacity": 0}', "utf-8")
    with pytest.raises(ValueError, match="capacity"):
        gpu_lease.settings(tmp_path)


def test_unknown_settings_keys_are_rejected(tmp_path: Path) -> None:
    (tmp_path / "settings.json").write_text('{"slots": 2}', "utf-8")
    with pytest.raises(ValueError, match="only capacity, priority and manual"):
        gpu_lease.settings(tmp_path)


def test_priority_must_list_text(tmp_path: Path) -> None:
    (tmp_path / "settings.json").write_text('{"priority": [9]}', "utf-8")
    with pytest.raises(ValueError, match="list replay IDs"):
        gpu_lease.settings(tmp_path)


def test_oversized_settings_are_rejected(tmp_path: Path) -> None:
    (tmp_path / "settings.json").write_text(json.dumps({"priority": ["1" * 5000]}), "utf-8")
    with pytest.raises(ValueError, match="too large"):
        gpu_lease.settings(tmp_path)


def test_a_replay_id_with_a_dot_is_rejected(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="without dots"):
        gpu_lease.Waiter(tmp_path, "a.b")


def test_the_start_time_comes_from_the_pipeline(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv(gpu_lease.STARTED_VARIABLE, "42")
    assert gpu_lease.started() == 42


def test_a_job_outside_a_pipeline_ranks_by_process_start(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv(gpu_lease.STARTED_VARIABLE, raising=False)
    assert gpu_lease.started() == gpu_lease.PROCESS_STARTED


def test_leases_stop_at_the_capacity_and_free_on_release(tmp_path: Path) -> None:
    (tmp_path / "settings.json").write_text('{"capacity": 2}', "utf-8")
    first = gpu_lease.Waiter(tmp_path, "1", start=1).try_acquire()
    second = gpu_lease.Waiter(tmp_path, "1", start=1).try_acquire()
    blocked = gpu_lease.Waiter(tmp_path, "1", start=1)
    assert first is not None
    assert second is not None
    assert blocked.try_acquire() is None
    gpu_lease.release(first)
    third = blocked.try_acquire()
    assert third is not None
    assert list((tmp_path / "waiting").iterdir()) == []
    gpu_lease.release(second)
    gpu_lease.release(third)


def test_threads_of_one_process_exclude_each_other(tmp_path: Path) -> None:
    (tmp_path / "settings.json").write_text('{"capacity": 1}', "utf-8")
    held = gpu_lease.Waiter(tmp_path, "1", start=1).try_acquire()
    seen: list[int | None] = []
    waiter = gpu_lease.Waiter(tmp_path, "1", start=1)
    thread = threading.Thread(target=lambda: seen.append(waiter.try_acquire()))
    thread.start()
    thread.join()
    waiter.close()
    assert held is not None
    assert seen == [None]
    gpu_lease.release(held)


def test_a_later_recording_yields_while_an_earlier_one_waits(tmp_path: Path) -> None:
    (tmp_path / "settings.json").write_text('{"capacity": 2}', "utf-8")
    busy = gpu_lease.Waiter(tmp_path, "1", start=1).try_acquire()
    assert busy is not None
    early = holder(tmp_path, "1", 1)  # holds the second lease
    waiting = holder(tmp_path, "1", 1)  # the earlier recording now waits
    later = gpu_lease.Waiter(tmp_path, "2", start=2)
    gpu_lease.release(busy)
    assert later.try_acquire() is None
    waiting.kill()
    waiting.wait()
    lease = later.try_acquire()
    assert lease is not None
    early.kill()
    early.wait()
    gpu_lease.release(lease)


def test_a_later_recording_uses_leases_the_earlier_one_leaves_idle(tmp_path: Path) -> None:
    (tmp_path / "settings.json").write_text('{"capacity": 2}', "utf-8")
    early = holder(tmp_path, "1", 1)
    lease = gpu_lease.Waiter(tmp_path, "2", start=2).try_acquire()
    assert lease is not None
    early.kill()
    early.wait()
    gpu_lease.release(lease)


def test_priority_ranks_ahead_of_the_start_time(tmp_path: Path) -> None:
    (tmp_path / "settings.json").write_text('{"capacity": 1, "priority": ["9"]}', "utf-8")
    busy = gpu_lease.Waiter(tmp_path, "5", start=5).try_acquire()
    assert busy is not None
    urgent = holder(tmp_path, "9", 9)
    older = gpu_lease.Waiter(tmp_path, "1", start=1)
    gpu_lease.release(busy)
    assert older.try_acquire() is None
    older.close()
    urgent.kill()
    urgent.wait()


def test_a_killed_holder_frees_its_lease(tmp_path: Path) -> None:
    (tmp_path / "settings.json").write_text('{"capacity": 1}', "utf-8")
    process = holder(tmp_path, "1", 1)
    waiter = gpu_lease.Waiter(tmp_path, "1", start=1)
    assert waiter.try_acquire() is None
    process.kill()
    process.wait()
    after = waiter.try_acquire()
    assert after is not None
    gpu_lease.release(after)


def test_the_lease_block_waits_for_a_free_lease(tmp_path: Path) -> None:
    (tmp_path / "settings.json").write_text('{"capacity": 1}', "utf-8")
    held = gpu_lease.Waiter(tmp_path, "1", start=1).try_acquire()
    assert held is not None
    timer = threading.Timer(0.05, gpu_lease.release, (held,))
    timer.start()
    with gpu_lease.lease(tmp_path, "1", poll_seconds=0.01):
        assert gpu_lease.Waiter(tmp_path, "1", start=1).try_acquire() is None
    timer.join()


def test_the_root_is_the_checkouts_whatever_the_working_directory(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    checkout = Path(gpu_lease.__file__).resolve().parents[2]
    monkeypatch.chdir(tmp_path)
    assert gpu_lease.root() == checkout / "edits/work/gpu-leases"


def test_newest_first_mirrors_broadcast_times() -> None:
    assert gpu_lease.newest_first(1792575000000000000) == 1207425000000000000
    with pytest.raises(ValueError, match="before 2065"):
        gpu_lease.newest_first(0)


def test_a_later_recording_takes_a_lease_beyond_the_earlier_ones_waiters(tmp_path: Path) -> None:
    (tmp_path / "settings.json").write_text('{"capacity": 3}', "utf-8")
    busy = gpu_lease.Waiter(tmp_path, "1", start=1).try_acquire()
    assert busy is not None
    early = holder(tmp_path, "1", 1)  # holds the second lease
    full = holder(tmp_path, "1", 1)  # holds the third lease
    waiting = holder(tmp_path, "1", 1)  # the earlier recording now waits
    gpu_lease.release(busy)
    owed = gpu_lease.Waiter(tmp_path, "2", start=2)
    assert owed.try_acquire() is None  # the one free lease is the earlier waiter's
    owed.close()
    full.kill()
    full.wait()
    later = gpu_lease.Waiter(tmp_path, "2", start=2).try_acquire()
    assert later is not None  # two free, one waiter ahead: one is left idle otherwise
    waiting.kill()
    waiting.wait()
    early.kill()
    early.wait()
    gpu_lease.release(later)


def test_holders_log_every_acquire_and_release_with_the_step(tmp_path: Path) -> None:
    with gpu_lease.lease(tmp_path, "7", step="compose"):
        pass
    rows = [json.loads(line) for line in (tmp_path / "holders.jsonl").read_text().splitlines()]
    assert [(row["event"], row["recording"], row["step"]) for row in rows] == [
        ("acquire", "7", "compose"),
        ("release", "7", "compose"),
    ]
    assert rows[0]["waitedSeconds"] >= 0
    assert rows[1]["heldSeconds"] >= 0
    assert rows[0]["pid"] == rows[1]["pid"]


def test_releasing_an_unlogged_descriptor_writes_no_holder_line(tmp_path: Path) -> None:
    descriptor = os.open(tmp_path / "other.lock", os.O_RDWR | os.O_CREAT, 0o600)
    gpu_lease.release(descriptor)
    assert not (tmp_path / "holders.jsonl").exists()


def test_an_unwritable_holder_log_neither_blocks_nor_leaks_the_lease(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    (tmp_path / "settings.json").write_text('{"capacity": 1}', "utf-8")
    (tmp_path / "holders.jsonl").mkdir()
    with gpu_lease.lease(tmp_path, "7", step="compose"):
        pass
    again = gpu_lease.Waiter(tmp_path, "7", start=1).try_acquire()
    assert again is not None
    gpu_lease.release(again)
    assert capsys.readouterr().err.count("GPU lease holder log not written") == 4


def test_a_manual_flag_is_accepted_and_must_be_a_boolean(tmp_path: Path) -> None:
    (tmp_path / "settings.json").write_text('{"priority": ["9"], "manual": true}', "utf-8")
    assert gpu_lease.settings(tmp_path) == (4, ("9",))
    (tmp_path / "settings.json").write_text('{"manual": "yes"}', "utf-8")
    with pytest.raises(ValueError, match="manual must be true or false"):
        gpu_lease.settings(tmp_path)
