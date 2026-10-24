"""Independent ready stages run together within slots and measured headroom; failures stop."""

import json
import os
import signal
import subprocess
import sys
import threading
from pathlib import Path

import pytest

from replay import memory_admission, source_caption_binding
from replay import workflow_parallel as parallel
from replay.core import ReplayError
from replay.stages import Stage, StageError

IDLE = parallel.Load(cpus=10, load=1.0, memory_free_percent=60, gpu_percent=10)
BUSY = parallel.Load(cpus=10, load=9.0, memory_free_percent=60, gpu_percent=10)
GPU_BUSY = parallel.Load(cpus=10, load=1.0, memory_free_percent=60, gpu_percent=90)


def stage(name: str) -> Stage:
    return Stage(
        name=name,
        command=("true",),
        inputs=(Path(f"/stage/{name}.in"),),
        outputs=(Path(f"/stage/{name}.out"),),
        timeout_seconds=10,
    )


A, B, C = stage("a"), stage("b"), stage("c")


class Peak:
    """A fake stage runner that records how many stages ran at once."""

    def __init__(self) -> None:
        self.active = 0
        self.peak = 0
        self.lock = threading.Lock()

    def __call__(self, item: Stage, state: Path) -> parallel.Result:
        with self.lock:
            self.active += 1
            self.peak = max(self.peak, self.active)
        threading.Event().wait(0.1)
        with self.lock:
            self.active -= 1
        return "completed"


def test_independent_stages_overlap(tmp_path: Path) -> None:
    barrier = threading.Barrier(2, timeout=10)

    def run(item: Stage, state: Path) -> parallel.Result:
        if item.name in {"a", "b"}:
            barrier.wait()  # passes only while a and b run at the same time
        return "completed"

    results = parallel.run_selected(
        (A, B, C), graph={"a": (), "b": (), "c": ("a", "b")}, state=tmp_path, bindings={},
        max_parallel=2, resources=parallel.StageSlots(), measure=lambda: IDLE, run=run,
    )  # fmt: skip
    assert results == {"a": "completed", "b": "completed", "c": "completed"}


def test_a_dependent_stage_waits_for_its_producer(tmp_path: Path) -> None:
    order: list[str] = []

    def run(item: Stage, state: Path) -> parallel.Result:
        order.append(item.name)
        return "reused"

    parallel.run_selected(
        (A, B), graph={"a": (), "b": ("a",)}, state=tmp_path, bindings={},
        max_parallel=4, resources=parallel.StageSlots(), measure=lambda: IDLE, run=run,
    )  # fmt: skip
    assert order == ["a", "b"]


def test_a_busy_machine_still_runs_the_always_stages(tmp_path: Path) -> None:
    peak = Peak()
    parallel.run_selected(
        (A, B, C), graph={"a": (), "b": (), "c": ()}, state=tmp_path, bindings={},
        max_parallel=4, resources=parallel.StageSlots(), measure=lambda: BUSY, run=peak,
        always=2, poll_seconds=0.01,
    )  # fmt: skip
    assert peak.peak == 2


def test_an_idle_machine_runs_beyond_the_always_stages(tmp_path: Path) -> None:
    peak = Peak()
    parallel.run_selected(
        (A, B, C), graph={"a": (), "b": (), "c": ()}, state=tmp_path, bindings={},
        max_parallel=4, resources=parallel.StageSlots(), measure=lambda: IDLE, run=peak,
        always=1, poll_seconds=0.01,
    )  # fmt: skip
    assert peak.peak == 3


def test_a_cpu_slot_caps_its_stages(tmp_path: Path) -> None:
    peak = Peak()
    slots = parallel.StageSlots({"io": 1}, {"a": "io", "b": "io"})
    parallel.run_selected(
        (A, B), graph={"a": (), "b": ()}, state=tmp_path, bindings={}, max_parallel=4,
        resources=slots, measure=lambda: IDLE, run=peak, poll_seconds=0.01,
    )  # fmt: skip
    assert peak.peak == 1


def test_a_second_gpu_stage_joins_an_idle_gpu_after_the_settle_time(tmp_path: Path) -> None:
    peak = Peak()
    slots = parallel.StageSlots({"gpu": 2}, {"a": "gpu", "b": "gpu"})
    parallel.run_selected(
        (A, B), graph={"a": (), "b": ()}, state=tmp_path, bindings={}, max_parallel=4,
        resources=slots, measure=lambda: IDLE, run=peak, poll_seconds=0.01, settle_seconds=0.0,
    )  # fmt: skip
    assert peak.peak == 2


def test_a_second_gpu_stage_waits_while_the_gpu_is_busy(tmp_path: Path) -> None:
    peak = Peak()
    slots = parallel.StageSlots({"gpu": 2}, {"a": "gpu", "b": "gpu"})
    parallel.run_selected(
        (A, B), graph={"a": (), "b": ()}, state=tmp_path, bindings={}, max_parallel=4,
        resources=slots, measure=lambda: GPU_BUSY, run=peak, poll_seconds=0.01,
        settle_seconds=0.0,
    )  # fmt: skip
    assert peak.peak == 1


def test_a_second_gpu_stage_waits_for_the_first_to_settle(tmp_path: Path) -> None:
    peak = Peak()
    slots = parallel.StageSlots({"gpu": 2}, {"a": "gpu", "b": "gpu"})
    parallel.run_selected(
        (A, B), graph={"a": (), "b": ()}, state=tmp_path, bindings={}, max_parallel=4,
        resources=slots, measure=lambda: IDLE, run=peak, poll_seconds=0.01,
        clock=lambda: 0.0, settle_seconds=60.0,
    )  # fmt: skip
    assert peak.peak == 1


def test_headroom_needs_low_load_and_free_memory() -> None:
    assert parallel.headroom(IDLE) is True
    assert parallel.headroom(BUSY) is False
    assert parallel.headroom(parallel.Load(10, 1.0, 5, 10)) is False


def test_paging_in_faster_than_the_limit_leaves_no_headroom() -> None:
    assert parallel.headroom(parallel.Load(10, 1.0, 80, 10, 1.9 * 1024**3)) is True
    assert parallel.headroom(parallel.Load(10, 1.0, 80, 10, 8 * 1024**3)) is False


def test_read_load_parses_memory_and_gpu(monkeypatch: pytest.MonkeyPatch) -> None:
    outputs = {
        parallel.MEMORY_LEVEL: "63\n",
        parallel.GPU_STATISTICS: '"Renderer Utilization %"=20,"Device Utilization %"=47',
    }

    def fake(command: tuple[str, ...], **_k: object) -> subprocess.CompletedProcess[str]:
        return subprocess.CompletedProcess(command, 0, outputs[command], "")

    monkeypatch.setattr(subprocess, "run", fake)
    monkeypatch.setattr(os, "cpu_count", lambda: 15)
    monkeypatch.setattr(os, "getloadavg", lambda: (3.5, 0.0, 0.0))
    monkeypatch.setattr(memory_admission, "swap_in_rate", lambda: 1024.0)
    assert parallel.read_load() == parallel.Load(15, 3.5, 63, 47, 1024.0)


def test_unreadable_measurements_count_as_no_headroom(monkeypatch: pytest.MonkeyPatch) -> None:
    def fake(command: tuple[str, ...], **_k: object) -> subprocess.CompletedProcess[str]:
        return subprocess.CompletedProcess(command, 1, "", "")

    monkeypatch.setattr(subprocess, "run", fake)
    monkeypatch.setattr(os, "cpu_count", lambda: None)
    monkeypatch.setattr(os, "getloadavg", lambda: (0.5, 0.0, 0.0))
    load = parallel.read_load()  # an unreadable vm_stat pages in without bound
    assert load == parallel.Load(1, 0.5, 0, 100, float("inf"))
    assert parallel.headroom(load) is False


def test_the_first_failure_stops_admission(tmp_path: Path) -> None:
    started: list[str] = []

    def run(item: Stage, state: Path) -> parallel.Result:
        started.append(item.name)
        if item.name == "a":
            raise StageError("boom")
        return "completed"

    with pytest.raises(StageError, match="a: StageError: boom"):
        parallel.run_selected(
            (A, B), graph={"a": (), "b": ("a",)}, state=tmp_path, bindings={},
            max_parallel=4, resources=parallel.StageSlots(), measure=lambda: IDLE, run=run,
        )  # fmt: skip
    assert started == ["a"]


def test_a_busy_stage_fails_without_a_duplicate(tmp_path: Path) -> None:
    with pytest.raises(StageError, match="another process owns this stage"):
        parallel.run_selected(
            (A,), graph={"a": ()}, state=tmp_path, bindings={}, max_parallel=1,
            resources=parallel.StageSlots(), measure=lambda: IDLE, run=lambda *_a: "busy",
            always=1,
        )  # fmt: skip


def test_a_stage_running_at_a_failure_finishes_and_its_failure_is_reported(
    tmp_path: Path,
) -> None:
    def run(item: Stage, state: Path) -> parallel.Result:
        if item.name == "a":
            raise StageError("first")
        threading.Event().wait(0.2)
        raise StageError("second")

    with pytest.raises(StageError, match="a: StageError: first; b: StageError: second"):
        parallel.run_selected(
            (A, B), graph={"a": (), "b": ()}, state=tmp_path, bindings={}, max_parallel=2,
            resources=parallel.StageSlots(), measure=lambda: IDLE, run=run, poll_seconds=0.01,
        )  # fmt: skip


def test_a_stage_running_at_a_failure_that_succeeds_is_kept(tmp_path: Path) -> None:
    def run(item: Stage, state: Path) -> parallel.Result:
        if item.name == "a":
            raise StageError("first")
        threading.Event().wait(0.1)
        return "completed"

    with pytest.raises(StageError, match="^Workflow stages failed: a: StageError: first$"):
        parallel.run_selected(
            (A, B), graph={"a": (), "b": ()}, state=tmp_path, bindings={}, max_parallel=2,
            resources=parallel.StageSlots(), measure=lambda: IDLE, run=run, poll_seconds=0.01,
        )  # fmt: skip


def test_invalid_parallel_counts_are_refused(tmp_path: Path) -> None:
    with pytest.raises(StageError, match="parallel stage count"):
        parallel.run_selected(
            (A,), graph={"a": ()}, state=tmp_path, bindings={}, max_parallel=1,
            resources=parallel.StageSlots(), run=lambda *_a: "completed", always=2,
        )  # fmt: skip


def test_a_failed_source_caption_binding_stops_before_the_stage(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    def refuse(*_a: object, **_k: object) -> None:
        raise ReplayError("stale binding")

    monkeypatch.setattr(source_caption_binding, "validate_binding", refuse)
    with pytest.raises(StageError, match="binding failed"):
        parallel.run_selected(
            (A,), graph={"a": ()}, state=tmp_path, bindings={"a": Path("/stage/a.in")},
            max_parallel=1, resources=parallel.StageSlots(), measure=lambda: IDLE,
            run=lambda *_a: "completed", always=1,
        )  # fmt: skip


def test_a_valid_source_caption_binding_lets_the_stage_run(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(source_caption_binding, "validate_binding", lambda *_a, **_k: None)
    assert parallel.run_selected(
        (A,), graph={"a": ()}, state=tmp_path, bindings={"a": Path("/stage/a.in")},
        max_parallel=1, resources=parallel.StageSlots(), measure=lambda: IDLE,
        run=lambda *_a: "completed", always=1,
    ) == {"a": "completed"}  # fmt: skip


def test_resources_file_is_validated(tmp_path: Path) -> None:
    path = tmp_path / "resources.json"
    path.write_text(json.dumps({"slots": {"gpu": 2}, "stages": {"a": "gpu"}}), encoding="utf-8")
    assert parallel.load_resources(path, ["a"]) == parallel.StageSlots({"gpu": 2}, {"a": "gpu"})
    assert parallel.load_resources(None, ["a"]) == parallel.StageSlots()


@pytest.mark.parametrize(
    "value",
    [
        [],
        {"slots": {}},
        {"slots": [], "stages": {}},
        {"slots": {"gpu": 0}, "stages": {}},
        {"slots": {"gpu": True}, "stages": {}},
        {"slots": {"gpu": 1}, "stages": {"x": "gpu"}},
        {"slots": {"gpu": 1}, "stages": {"a": "cpu"}},
    ],
)
def test_invalid_resource_files_are_refused(tmp_path: Path, value: object) -> None:
    path = tmp_path / "resources.json"
    path.write_text(json.dumps(value), encoding="utf-8")
    with pytest.raises(StageError):
        parallel.load_resources(path, ["a"])


def test_an_oversized_resource_file_is_refused(tmp_path: Path) -> None:
    path = tmp_path / "resources.json"
    path.write_text(" " * 70_000, encoding="utf-8")
    with pytest.raises(StageError, match="too large"):
        parallel.load_resources(path, ["a"])


def test_children_run_a_command_and_report_its_failure(tmp_path: Path) -> None:
    children = parallel.Children()
    children.execute((sys.executable, "-c", "print('ok')"), tmp_path / "ok.log", 30)
    assert (tmp_path / "ok.log").read_text(encoding="utf-8") == "ok\n"
    with pytest.raises(StageError, match="Stage command failed"):
        children.execute((sys.executable, "-c", "raise SystemExit(3)"), tmp_path / "x.log", 30)


def test_children_stop_all_kills_running_groups(tmp_path: Path) -> None:
    children = parallel.Children()
    failure: list[str] = []

    def target() -> None:
        try:
            children.execute(
                (sys.executable, "-c", "import time; time.sleep(60)"), tmp_path / "s.log", 120
            )
        except StageError as error:
            failure.append(str(error))

    thread = threading.Thread(target=target)
    thread.start()
    threading.Event().wait(0.5)
    children.stop_all()
    thread.join(timeout=10)
    assert failure == [f"Stage command failed; inspect {tmp_path / 's.log'}"]


def test_a_timed_out_child_is_killed(tmp_path: Path) -> None:
    children = parallel.Children()
    with pytest.raises(subprocess.TimeoutExpired):
        children.execute(
            (sys.executable, "-c", "import time; time.sleep(60)"), tmp_path / "t.log", 1
        )


def test_stop_all_ignores_an_already_finished_group(monkeypatch: pytest.MonkeyPatch) -> None:
    def gone(_group: int, _signal: int) -> None:
        raise ProcessLookupError

    monkeypatch.setattr(os, "killpg", gone)
    children = parallel.Children()
    children._groups.add(999_999)
    children.stop_all()
    assert children._groups == {999_999}


def write_plan(tmp_path: Path) -> Path:
    work = tmp_path / "work"
    work.mkdir()
    source = tmp_path / "source.txt"
    source.write_text("s", encoding="utf-8")
    script = "import pathlib,sys; pathlib.Path(sys.argv[1]).write_text('x', encoding='utf-8')"
    plan = {
        "version": 1,
        "stages": [
            {"name": "left", "command": [sys.executable, "-c", script, str(work / "l.txt")],
             "inputs": [str(source)], "outputs": [str(work / "l.txt")], "timeout_seconds": 30},
            {"name": "right", "command": [sys.executable, "-c", script, str(work / "r.txt")],
             "inputs": [str(source)], "outputs": [str(work / "r.txt")], "timeout_seconds": 30},
            {"name": "join", "command": [sys.executable, "-c", script, str(work / "j.txt")],
             "inputs": [str(work / "l.txt"), str(work / "r.txt")],
             "outputs": [str(work / "j.txt")], "timeout_seconds": 30},
        ],
    }  # fmt: skip
    path = tmp_path / "plan.json"
    path.write_text(json.dumps(plan), encoding="utf-8")
    return path


def test_the_cli_runs_a_real_plan_and_reuses_it(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    plan = write_plan(tmp_path)
    state = tmp_path / "state"
    assert parallel.main([str(plan), "--state", str(state), "--max-parallel", "2"]) == 0
    assert (tmp_path / "work/j.txt").read_text(encoding="utf-8") == "x"
    assert parallel.run_plan(plan, state_directory=state) == {
        "left": "reused",
        "right": "reused",
        "join": "reused",
    }
    assert "workflow wall seconds" in capsys.readouterr().out


def test_the_cli_reports_a_failure(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    plan = write_plan(tmp_path)
    resources = tmp_path / "resources.json"
    resources.write_text("{}", encoding="utf-8")
    state = tmp_path / "state"
    assert parallel.main([str(plan), "--state", str(state), "--resources", str(resources)]) == 1
    assert "Workflow stopped (StageError)" in capsys.readouterr().err


def test_sigterm_stops_the_children_and_fails_the_run(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    handlers: list[object] = []

    def run_plan(*_a: object, **_k: object) -> dict[str, parallel.Result]:
        handler = signal.getsignal(signal.SIGTERM)
        handlers.append(handler)
        if callable(handler):
            handler(signal.SIGTERM, None)
        return {}

    monkeypatch.setattr(parallel, "run_plan", run_plan)
    assert parallel.main([str(tmp_path / "plan.json"), "--state", str(tmp_path / "s")]) == 1
    assert "Workflow cancelled" in capsys.readouterr().err
    assert signal.getsignal(signal.SIGTERM) is not handlers[0]


def test_a_group_ignoring_sigterm_is_killed_after_the_grace(tmp_path: Path) -> None:
    children = parallel.Children()
    failure: list[str] = []
    stubborn = "import signal,time; signal.signal(signal.SIGTERM, signal.SIG_IGN); time.sleep(60)"

    def target() -> None:
        try:
            children.execute((sys.executable, "-c", stubborn), tmp_path / "k.log", 120)
        except StageError as error:
            failure.append(str(error))

    thread = threading.Thread(target=target)
    thread.start()
    threading.Event().wait(0.8)
    children.stop_all(grace_seconds=0.5)
    thread.join(timeout=10)
    assert failure == [f"Stage command failed; inspect {tmp_path / 'k.log'}"]
