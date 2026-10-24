"""GPU orchestration: weighted units, class priority, work conservation, aging and legacy slots."""

import json
import subprocess
import sys
from pathlib import Path

import pytest

from replay import gpu_lease
from replay import gpu_orchestra as orchestra

SECOND = 10**9


@pytest.fixture
def leases(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """An enabled orchestra and a 4-slot lease directory, both private to the test."""
    monkeypatch.setattr(orchestra, "root", lambda: tmp_path / "orchestra")
    (tmp_path / "orchestra").mkdir()
    (tmp_path / "orchestra" / "settings.json").write_text('{"enabled": true}', "utf-8")
    directory = tmp_path / "leases"
    directory.mkdir()
    (directory / "settings.json").write_text('{"capacity": 4}', "utf-8")
    return directory


def waiting(
    job: str, since: int, weight: float, *, rank: int = 0, name: str = ""
) -> orchestra.Waiting:
    return orchestra.Waiting(job, "1", rank, 0, since, weight, Path(name or f"{job}-{since}"))


CURRENT = orchestra.Settings(True, 1.0, dict(orchestra.DEFAULT_WEIGHTS), 0.5, 600.0, 1800.0)


def test_defaults_are_disabled_and_settings_are_validated(tmp_path: Path) -> None:
    assert orchestra.settings(tmp_path) == orchestra.Settings(
        False, 1.0, dict(orchestra.DEFAULT_WEIGHTS), 0.5, 600.0, 1800.0
    )
    (tmp_path / "settings.json").write_text(
        '{"enabled": true, "overcommit": 0.25, "weights": {"onsets": 0.1}}', "utf-8"
    )
    current = orchestra.settings(tmp_path)
    assert (current.enabled, current.budget, current.weights["onsets"]) == (True, 1.25, 0.1)
    assert orchestra.enabled(tmp_path) is True


@pytest.mark.parametrize(
    ("text", "message"),
    [
        ('{"slots": 1}', "take agingSeconds"),
        ('{"enabled": 1}', "true or false"),
        ('{"weights": {"other": 1}}', "only known jobs"),
        ('{"budget": 0}', "budget must be"),
        ('{"overcommit": 2}', "overcommit must be"),
        (json.dumps({"weights": {"asr": 0.5}, "x": "1" * 5000}), "too large"),
    ],
)
def test_invalid_settings_are_refused(tmp_path: Path, text: str, message: str) -> None:
    (tmp_path / "settings.json").write_text(text, "utf-8")
    with pytest.raises(ValueError, match=message):
        orchestra.settings(tmp_path)


def test_cleanup_chunks_go_first_and_a_lighter_job_backfills_units_left_over() -> None:
    chunk = waiting("cleanup-chunk", 2 * SECOND, 0.5)
    yolo = waiting("yolo-track", 1 * SECOND, 0.5)
    onsets = waiting("onsets", 3 * SECOND, 0.25)
    queue = [yolo, chunk, onsets]
    now = 10 * SECOND
    # 0.25 running: the chunk takes 0.5, onsets the 0.25 left; yolo-track does not fit.
    assert orchestra.admitted(chunk, running=0.25, queue=queue, current=CURRENT, now_ns=now)
    assert orchestra.admitted(onsets, running=0.25, queue=queue, current=CURRENT, now_ns=now)
    assert not orchestra.admitted(yolo, running=0.25, queue=queue, current=CURRENT, now_ns=now)
    # 0.6 running: the chunk cannot fit, so the onsets job uses the units (work-conserving).
    assert not orchestra.admitted(chunk, running=0.6, queue=queue, current=CURRENT, now_ns=now)
    assert orchestra.admitted(onsets, running=0.6, queue=queue, current=CURRENT, now_ns=now)


def test_a_starving_waiter_holds_back_everything_behind_it() -> None:
    chunk = waiting("cleanup-chunk", 0, 0.5)
    onsets = waiting("onsets", 1, 0.25)
    late = 1801 * SECOND
    assert not orchestra.admitted(onsets, running=0.6, queue=[chunk, onsets], current=CURRENT,
                                  now_ns=late)  # fmt: skip


def test_a_job_heavier_than_the_budget_runs_alone_on_an_idle_gpu() -> None:
    huge = waiting("bench", 0, 1.5)
    assert orchestra.admitted(huge, running=0.0, queue=[huge], current=CURRENT, now_ns=SECOND)
    assert not orchestra.admitted(huge, running=0.1, queue=[huge], current=CURRENT, now_ns=SECOND)


def test_aging_raises_a_waiters_class() -> None:
    bench = waiting("bench", 0, 0.5)
    assert bench.key(SECOND, 600.0)[2] == 3
    assert bench.key(1201 * SECOND, 600.0)[2] == 1
    assert bench.key(10_000 * SECOND, 600.0)[2] == 0
    assert orchestra.admitted(waiting("x", 0, 0.5, name="none"), running=0.0, queue=[],
                              current=CURRENT, now_ns=SECOND) is False  # fmt: skip


def test_orchestrated_leases_hold_units_and_a_slot(leases: Path, tmp_path: Path) -> None:
    first = gpu_lease.Waiter(leases, "1", start=1, step="cleanup chunk-0", job="cleanup-chunk")
    second = gpu_lease.Waiter(leases, "1", start=1, step="cleanup chunk-1", job="cleanup-chunk")
    third = gpu_lease.Waiter(leases, "1", start=1, step="onsets", job="onsets")
    one, two = first.try_acquire(), second.try_acquire()
    assert one is not None and two is not None
    assert third.try_acquire() is None  # 1.0 of 1.0 units held, slots still free
    status = orchestra.status(tmp_path / "orchestra", leases, 4)
    assert status["unitsUsed"] == 1.0
    queued = status["waiting"]
    assert isinstance(queued, list)
    assert [row["units"] for row in queued] == [0.25]
    gpu_lease.release(one)
    three = third.try_acquire()
    assert three is not None
    rows = [json.loads(line) for line in (leases / "holders.jsonl").read_text().splitlines()]
    assert [row.get("units") for row in rows if row["event"] == "acquire"] == [0.5, 0.5, 0.25]
    gpu_lease.release(two)
    gpu_lease.release(three)
    assert orchestra.held(tmp_path / "orchestra") == []


def test_a_slot_held_by_earlier_code_counts_as_legacy_units(leases: Path) -> None:
    first = gpu_lease.Waiter(leases, "9", start=9).try_acquire()
    second = gpu_lease.Waiter(leases, "9", start=9).try_acquire()
    assert first is not None and second is not None  # 2 x 0.5 legacy units
    chunk = gpu_lease.Waiter(leases, "1", start=1, job="cleanup-chunk")
    assert chunk.try_acquire() is None
    gpu_lease.release(first)
    lease = chunk.try_acquire()
    assert lease is not None
    gpu_lease.release(lease)
    gpu_lease.release(second)


def test_units_without_a_free_slot_are_not_taken(leases: Path) -> None:
    (leases / "settings.json").write_text('{"capacity": 1}', "utf-8")
    (leases.parent / "orchestra" / "settings.json").write_text(
        '{"enabled": true, "legacyWeight": 0.01}', "utf-8"
    )
    busy = gpu_lease.Waiter(leases, "9", start=9).try_acquire()
    assert busy is not None
    waiter = gpu_lease.Waiter(leases, "1", start=1, job="onsets")
    assert waiter.try_acquire() is None
    assert orchestra.held(leases.parent / "orchestra") == []
    waiter.close()
    gpu_lease.release(busy)


def test_unknown_jobs_are_refused(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="Unknown orchestrated job"):
        orchestra.Ticket(tmp_path, "clocks", "1", rank=0, start=0)


def test_a_disabled_orchestra_leaves_gpu_lease_as_before(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(orchestra, "root", lambda: tmp_path / "none")
    descriptor = gpu_lease.Waiter(tmp_path, "1", start=1, job="cleanup-chunk").try_acquire()
    assert descriptor is not None
    rows = [json.loads(line) for line in (tmp_path / "holders.jsonl").read_text().splitlines()]
    assert rows[0]["units"] is None
    gpu_lease.release(descriptor)


def test_the_cli_reports_status_and_runs_a_benchmark_at_the_lowest_class(
    leases: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.setattr(gpu_lease, "root", lambda: leases)
    assert orchestra.main(["status"]) == 0
    assert json.loads(capsys.readouterr().out)["enabled"] is True
    code = orchestra.main(["run", "--units", "0.75", "--name", "probe", "--", sys.executable,
                           "-c", "raise SystemExit(3)"])  # fmt: skip
    assert code == 3
    rows = [json.loads(line) for line in (leases / "holders.jsonl").read_text().splitlines()]
    assert [(row["event"], row["step"], row.get("units")) for row in rows] == [
        ("acquire", "probe", 0.75),
        ("release", "probe", None),
    ]
    with pytest.raises(SystemExit):
        orchestra.main(["run"])


def test_the_cli_is_a_module() -> None:
    result = subprocess.run(
        [sys.executable, "-m", "replay.gpu_orchestra", "--help"], capture_output=True, text=True
    )
    assert result.returncode == 0


def test_the_higher_ranked_recording_goes_first_whatever_its_step() -> None:
    first = waiting("asr", 5 * SECOND, 0.5, rank=0, name="first")
    second = waiting("cleanup-chunk", 1 * SECOND, 0.5, rank=1, name="second")
    queue = [second, first]
    assert orchestra.admitted(first, running=0.5, queue=queue, current=CURRENT, now_ns=9 * SECOND)
    assert not orchestra.admitted(second, running=0.5, queue=queue, current=CURRENT,
                                  now_ns=9 * SECOND)  # fmt: skip


def test_a_lower_recording_borrows_only_short_jobs_while_a_higher_one_is_active() -> None:
    chunk = waiting("cleanup-chunk", 1 * SECOND, 0.25, rank=1, name="chunk")
    separation = waiting("separation", 1 * SECOND, 0.25, rank=1, name="separation")
    queue = [chunk, separation]
    busy = [(0, 0)]  # the top recording holds units but waits for nothing
    assert orchestra.admitted(chunk, running=0.5, queue=queue, current=CURRENT,
                              now_ns=9 * SECOND, busy=busy)  # fmt: skip
    assert not orchestra.admitted(separation, running=0.5, queue=queue, current=CURRENT,
                                  now_ns=9 * SECOND, busy=busy)  # fmt: skip
    assert orchestra.admitted(separation, running=0.5, queue=queue, current=CURRENT,
                              now_ns=9 * SECOND)  # the higher recording is idle  # fmt: skip


def test_short_jobs_are_configurable(tmp_path: Path) -> None:
    (tmp_path / "settings.json").write_text('{"shortJobs": ["bench"]}', "utf-8")
    assert orchestra.settings(tmp_path).short == frozenset({"bench"})
    (tmp_path / "settings.json").write_text('{"shortJobs": ["other"]}', "utf-8")
    with pytest.raises(ValueError, match="shortJobs"):
        orchestra.settings(tmp_path)
