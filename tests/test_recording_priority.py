"""Recording-first priority: ranking, manual order, urgent recordings and lease propagation."""

import json
import os
from pathlib import Path

import pytest

from replay import gpu_lease
from replay import gpu_orchestra as orchestra
from replay import recording_priority as priority

BROADCASTS = {"1": 1000, "2": 2000, "3": 3000, "4": 4000}


def test_auto_ranks_newest_broadcast_first_with_urgent_on_top_and_paused_last() -> None:
    assert priority.ranked(priority.Order(), BROADCASTS) == ["4", "3", "2", "1"]
    order = priority.Order(urgent=("1", "9"), paused=("3",))
    assert priority.ranked(order, BROADCASTS) == ["1", "4", "2", "3"]


def test_a_pinned_order_keeps_its_order_and_places_new_recordings_by_date() -> None:
    order = priority.Order("pinned", ("1", "3", "gone"))
    assert priority.ranked(order, BROADCASTS) == ["1", "3", "4", "2"]


def test_moving_clamps_the_position() -> None:
    assert priority.moved(["a", "b", "c"], "c", to=1) == ["c", "a", "b"]
    assert priority.moved(["a", "b", "c"], "a", to=9) == ["b", "c", "a"]
    assert priority.moved(["a", "b", "c"], "b", to=0) == ["b", "a", "c"]


def test_the_order_round_trips_and_bad_files_are_refused(tmp_path: Path) -> None:
    queue = tmp_path / "queue.json"
    assert priority.load(queue) == priority.Order()
    order = priority.Order("pinned", ("2",), ("3",), ("1",))
    priority.save(queue, order)
    assert priority.load(queue) == order
    assert priority.order_path(queue).name == "queue.order.json"
    priority.order_path(queue).write_text('{"version": 1, "mode": "random"}', "utf-8")
    with pytest.raises(ValueError, match="Unsupported queue order"):
        priority.load(queue)
    priority.order_path(queue).write_text('{"version": 1, "mode": "auto", "urgent": [1]}', "utf-8")
    with pytest.raises(ValueError, match="urgent must list"):
        priority.load(queue)


def test_enqueue_adds_a_recording_once(tmp_path: Path) -> None:
    queue = tmp_path / "queue.json"
    queue.write_text('{"version": 1, "slots": 1, "recordings": ["1"]}', "utf-8")
    priority.enqueue(queue, "2")
    priority.enqueue(queue, "2")
    assert json.loads(queue.read_text("utf-8"))["recordings"] == ["1", "2"]
    queue.write_text('{"version": 1, "slots": 1}', "utf-8")
    with pytest.raises(ValueError, match="lists no recordings"):
        priority.enqueue(queue, "2")


def test_publish_writes_every_lease_priority_and_the_urgent_list(tmp_path: Path) -> None:
    for name, text in (
        ("gpu-leases", '{"capacity": 4}'),
        ("compose-leases", '{"capacity": 1}'),
        ("compressor-leases", '{"capacity": 1, "priority": ["5"], "manual": true}'),
    ):
        (tmp_path / name).mkdir()
        (tmp_path / name / "settings.json").write_text(text, "utf-8")
    order = priority.Order(urgent=("2", "8"), paused=("1",))
    priority.publish(order, ["2", "4", "1"], tmp_path)
    assert json.loads((tmp_path / "gpu-leases/settings.json").read_text("utf-8")) == {
        "capacity": 4,
        "priority": ["2", "4"],
    }
    assert json.loads((tmp_path / "compose-leases/settings.json").read_text("utf-8")) == {
        "capacity": 1,
        "priority": ["2", "4"],
    }
    assert json.loads((tmp_path / "compressor-leases/settings.json").read_text("utf-8")) == {
        "capacity": 1,
        "priority": ["5"],
        "manual": True,
    }
    assert json.loads((tmp_path / "lease-urgent.json").read_text("utf-8")) == {"urgent": ["2"]}


def test_holders_report_the_jobs_holding_now_and_their_typical_time_left(tmp_path: Path) -> None:
    rows = [
        {"t": 100.0, "event": "acquire", "recording": "1", "step": "compose", "pid": 1},
        {"t": 400.0, "event": "release", "recording": "1", "step": "compose", "pid": 1,
         "heldSeconds": 300.0},
        {"t": 900.0, "event": "acquire", "recording": "2", "step": "compose", "pid": os.getpid()},
        {"t": 950.0, "event": "acquire", "recording": "3", "step": "yolo", "pid": os.getpid()},
        {"t": 960.0, "event": "acquire", "recording": "4", "step": "asr", "pid": 2**22 + 5},
    ]  # fmt: skip
    (tmp_path / "holders.jsonl").write_text("\n".join(map(json.dumps, rows)) + "\n")
    assert priority.holders(tmp_path, now=1000.0) == [
        {"recording": "2", "step": "compose", "heldSeconds": 100, "expectedRemainingSeconds": 200},
        {"recording": "3", "step": "yolo", "heldSeconds": 50, "expectedRemainingSeconds": None},
    ]
    assert priority.holders(tmp_path / "none") == []


def test_finished_urgent_recordings_leave_the_list() -> None:
    order = priority.Order(urgent=("1", "2"))
    assert priority.without(order, {"1"}).urgent == ("2",)


def lease_dirs(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> tuple[Path, Path]:
    work = tmp_path / "work"
    gpu, compose = work / "gpu-leases", work / "compose-leases"
    for directory, capacity in ((gpu, 2), (compose, 1)):
        directory.mkdir(parents=True)
        (directory / "settings.json").write_text(json.dumps({"capacity": capacity}), "utf-8")
    monkeypatch.setattr(gpu_lease, "root", lambda: gpu)
    monkeypatch.setattr(orchestra, "root", lambda: work / "gpu-orchestra")
    (work / "gpu-orchestra").mkdir()
    (work / "gpu-orchestra" / "settings.json").write_text('{"enabled": true}', "utf-8")
    return gpu, compose


def test_an_urgent_recording_gets_the_next_freed_gpu_units_compose_and_chunk_slots(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    gpu, compose = lease_dirs(tmp_path, monkeypatch)
    order = priority.Order()
    priority.publish(order, ["2", "1"], gpu.parent)
    held_chunk = gpu_lease.Waiter(gpu, "1", start=1, job="cleanup-chunk").try_acquire()
    held_onsets = gpu_lease.Waiter(gpu, "2", start=2, job="cleanup-chunk").try_acquire()
    held_compose = gpu_lease.Waiter(compose, "2", start=2).try_acquire()
    assert None not in (held_chunk, held_onsets, held_compose)
    # Recording 7 is made urgent while 1 and 2 hold every lease; all three keep waiting.
    order = priority.Order(urgent=("7",))
    priority.publish(order, ["7", "2", "1"], gpu.parent)
    others_chunk = gpu_lease.Waiter(gpu, "2", start=2, job="cleanup-chunk")
    others_compose = gpu_lease.Waiter(compose, "1", start=1)
    urgent_chunk = gpu_lease.Waiter(gpu, "7", start=7, job="cleanup-chunk")
    urgent_compose = gpu_lease.Waiter(compose, "7", start=7)
    assert urgent_chunk.try_acquire() is None
    assert urgent_compose.try_acquire() is None
    assert held_chunk is not None and held_compose is not None
    gpu_lease.release(held_chunk)
    gpu_lease.release(held_compose)
    assert others_chunk.try_acquire() is None  # the urgent recording waits: no grant to others
    assert others_compose.try_acquire() is None
    chunk = urgent_chunk.try_acquire()
    composing = urgent_compose.try_acquire()
    assert chunk is not None and composing is not None
    assert gpu_lease.urgent_waiting(gpu.parent, "2") is False  # nothing urgent waits any more
    others_chunk.close()
    others_compose.close()
    assert held_onsets is not None
    gpu_lease.release(chunk)
    gpu_lease.release(composing)
    gpu_lease.release(held_onsets)


def test_urgency_does_not_block_the_urgent_recording_itself(tmp_path: Path) -> None:
    (tmp_path / "gpu-leases" / "waiting").mkdir(parents=True)
    (tmp_path / "lease-urgent.json").write_text('{"urgent": ["7"]}', "utf-8")
    waiter = gpu_lease.Waiter(tmp_path / "gpu-leases", "7", start=7)
    assert waiter.try_acquire() is not None
    assert gpu_lease.urgent_waiting(tmp_path, "7") is False
    (tmp_path / "lease-urgent.json").write_text('{"urgent": "7"}', "utf-8")
    assert gpu_lease.urgent_waiting(tmp_path, "1") is False


def test_waiting_on_lists_the_lease_directories_a_recording_waits_in(tmp_path: Path) -> None:
    (tmp_path / "compose-leases").mkdir()
    (tmp_path / "compose-leases" / "settings.json").write_text('{"capacity": 1}', "utf-8")
    busy = gpu_lease.Waiter(tmp_path / "compose-leases", "1", start=1).try_acquire()
    waiter = gpu_lease.Waiter(tmp_path / "compose-leases", "7", start=7)
    assert waiter.try_acquire() is None
    assert priority.waiting_on(tmp_path, "7") == ["compose-leases"]
    assert priority.waiting_on(tmp_path, "1") == []
    waiter.close()
    assert busy is not None
    gpu_lease.release(busy)


def test_a_drain_records_its_reason_and_refuses_unknown_ones(tmp_path: Path) -> None:
    assert priority.drained(tmp_path, "1") is None
    priority.drain(tmp_path, "1", "abandon")
    assert priority.drained(tmp_path, "1") == "abandon"
    priority.drain_path(tmp_path, "1").write_text('{"reason": "other"}', "utf-8")
    assert priority.drained(tmp_path, "1") == "drain"
    with pytest.raises(ValueError, match="drain reason"):
        priority.drain(tmp_path, "1", "pause")
    priority.undrain(tmp_path, "1")
    assert priority.drained(tmp_path, "1") is None
