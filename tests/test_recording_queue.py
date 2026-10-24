"""The queue runner keeps a few pipelines busy, newest broadcast first, and survives restarts."""

import json
import os
import subprocess
from datetime import datetime
from pathlib import Path

import pytest

from replay import gpu_lease, memory_admission
from replay import recording_pipeline as pipelines
from replay import recording_priority as priority
from replay import recording_queue as queue
from replay.core import ReplayError
from replay.stages import execution_lock

BROADCASTS = {"3": 3_000, "2": 2_000, "1": 1_000}
WINDOW_ALIVE = queue.window_alive  # the autouse fixture replaces it


@pytest.fixture(autouse=True)
def recordings(monkeypatch: pytest.MonkeyPatch) -> None:
    """Recordings 1-3 are bound with known broadcast times; any other is still downloading."""

    def bind(replay_id: str, _python: str, checkout: Path) -> Path:
        if replay_id not in BROADCASTS:
            raise subprocess.CalledProcessError(1, ["init"])
        return checkout / "edits" / f"replay-{replay_id}"

    monkeypatch.setattr(pipelines, "bind", bind)
    monkeypatch.setattr(pipelines, "broadcast", lambda root: BROADCASTS[root.name[7:]])
    monkeypatch.setattr(queue, "window_alive", lambda _directory, _name: False)
    monkeypatch.setattr(queue, "swap_thrashing", lambda: False)  # no real paging measurement


def checkout(tmp_path: Path) -> Path:
    code = tmp_path / "src/replay/module.py"
    code.parent.mkdir(parents=True)
    code.write_text("", encoding="utf-8")
    os.utime(code, (1_000, 1_000))
    return tmp_path


def defaults(tmp_path: Path) -> dict[str, object]:
    return {"root": str(tmp_path / "edits/work/recordings"), "python": "/py"}


def queue_file(
    tmp_path: Path, *recordings: str, slots: int | str = 2, paused: bool = False
) -> Path:
    path = tmp_path / "queue.json"
    data = {"version": 1, "slots": slots, "paused": paused, "recordings": list(recordings)}
    path.write_text(json.dumps(data), encoding="utf-8")
    return path


def recording(tmp_path: Path, replay_id: str) -> Path:
    directory = tmp_path / f"edits/work/recordings/replay-{replay_id}"
    (directory / "steps").mkdir(parents=True)
    return directory


def launched(tmp_path: Path, replay_id: str, *codes: str) -> None:
    """Dispatched launches with these exit codes (an empty code: no status yet)."""
    directory = tmp_path / f"edits/work/recordings/replay-{replay_id}"
    for number, code in enumerate(codes, start=1):
        (directory / f"pipeline-v{number}.launch").mkdir()
        if code:
            status = directory / f"pipeline-v{number}.status"
            status.write_text(f"{code}\n", encoding="utf-8")
            os.utime(status, (2_000, 2_000))


def finished(tmp_path: Path, replay_id: str) -> None:
    steps = tmp_path / f"edits/work/recordings/replay-{replay_id}/steps"
    for name in queue.STEP_NAMES:
        (steps / f"{name}.json").write_text("{}", encoding="utf-8")


def recorder(started: list[str]) -> queue.Launch:
    def launch(replay_id: str) -> str:
        started.append(replay_id)
        return f"pipeline-v{len(started)}"

    return launch


def test_ready_recordings_launch_newest_first_into_free_slots(tmp_path: Path) -> None:
    root = checkout(tmp_path)
    started: list[str] = []
    rows = queue.tick(
        queue_file(tmp_path, "1", "3", "2"),
        defaults(tmp_path),
        checkout=root,
        launch=recorder(started),
    )
    assert started == ["3", "2"]
    assert rows == [
        {"recording": "3", "state": "launched", "broadcast": 3000, "launch": "pipeline-v1"},
        {"recording": "2", "state": "launched", "broadcast": 2000, "launch": "pipeline-v2"},
        {"recording": "1", "state": "ready", "broadcast": 1000},
    ]


def test_while_paging_thrashes_only_one_pipeline_runs(tmp_path: Path) -> None:
    root = checkout(tmp_path)
    started: list[str] = []
    rows = queue.tick(
        queue_file(tmp_path, "1", "3", "2"), defaults(tmp_path), checkout=root,
        launch=recorder(started), thrashing=lambda: True,
    )  # fmt: skip
    assert started == ["3"]  # two slots free, but only one fills
    assert rows[1:] == [
        {"recording": "2", "state": "ready", "broadcast": 2000},
        {"recording": "1", "state": "ready", "broadcast": 1000},
    ]
    with execution_lock(recording(tmp_path, "3"), "pipeline"):  # its run is going
        rows = queue.tick(
            queue_file(tmp_path, "3", "2"),
            defaults(tmp_path),
            checkout=root,
            launch=recorder(started),
            thrashing=lambda: True,
        )
    assert [row["state"] for row in rows] == ["running", "ready"]
    assert started == ["3"]


def test_the_paging_rate_decides_thrashing(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.undo()
    monkeypatch.setattr(memory_admission, "swap_in_rate", lambda: 8.0 * 1024**3)
    assert queue.swap_thrashing() is True
    monkeypatch.setattr(memory_admission, "swap_in_rate", lambda: 0.5 * 1024**3)
    assert queue.swap_thrashing() is False


def test_volume_retries_are_counted_per_day(tmp_path: Path) -> None:
    logs = tmp_path / "steps/logs"
    logs.mkdir(parents=True)
    rows = {
        "compose-1792854000000000000.failure.json": {"class": "volume", "retried": True},
        "compose-1792854100000000000.failure.json": {"class": "volume", "retried": False},
        "paging-1792854200000000000.failure.json": {"class": "transient", "retried": True},
        "compose-1792700000000000000.failure.json": {"class": "volume", "retried": True},
    }
    for name, row in rows.items():
        (logs / name).write_text(json.dumps(row), encoding="utf-8")
    (logs / "compose-x.failure.json").write_text("{}", encoding="utf-8")
    (logs / "compose-1792854300000000000.failure.json").write_text("{", encoding="utf-8")
    day = datetime.fromtimestamp(1792854000).strftime("%Y-%m-%d")
    assert queue.volume_retries([pipelines.Layout(tmp_path)], day) == 1


def test_layouts_leave_out_an_invalid_recording(tmp_path: Path) -> None:
    rows: list[dict[str, object]] = [{"recording": "3"}, {"recording": "../x"}]
    found = queue.layouts(rows, defaults(tmp_path), checkout(tmp_path))
    assert found == [pipelines.Layout(tmp_path / "edits/work/recordings/replay-3")]


def test_a_running_pipeline_takes_a_slot(tmp_path: Path) -> None:
    root = checkout(tmp_path)
    directory = recording(tmp_path, "3")
    started: list[str] = []
    with execution_lock(directory, "pipeline"):
        rows = queue.tick(
            queue_file(tmp_path, "1", "2", "3"),
            defaults(tmp_path),
            checkout=root,
            launch=recorder(started),
        )
    assert started == ["2"]
    assert [row["state"] for row in rows] == ["running", "launched", "ready"]


def test_a_dispatched_run_counts_as_running_while_its_window_lives(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root = checkout(tmp_path)
    directory = recording(tmp_path, "3")
    launched(tmp_path, "3", "")
    monkeypatch.setattr(
        queue, "window_alive", lambda path, name: (path, name) == (directory, "pipeline-v1")
    )
    rows = queue.tick(
        queue_file(tmp_path, "3", slots=1), defaults(tmp_path), checkout=root, launch=str
    )
    assert rows == [{"recording": "3", "state": "running", "broadcast": 3000}]


def test_a_launch_cut_off_by_a_reboot_is_launched_again(tmp_path: Path) -> None:
    root = checkout(tmp_path)
    recording(tmp_path, "3")
    launched(tmp_path, "3", "")
    rows = queue.tick(
        queue_file(tmp_path, "3"),
        defaults(tmp_path),
        checkout=root,
        launch=lambda _r: "pipeline-v2",
    )
    assert rows == [
        {"recording": "3", "state": "launched", "broadcast": 3000, "launch": "pipeline-v2"}
    ]


def test_done_unbound_and_held_recordings_are_not_launched(tmp_path: Path) -> None:
    root = checkout(tmp_path)
    recording(tmp_path, "3")
    finished(tmp_path, "3")
    recording(tmp_path, "2")
    launched(tmp_path, "2", "1", "0", "1")
    started: list[str] = []
    rows = queue.tick(
        queue_file(tmp_path, "9", "2", "3", "1"),
        defaults(tmp_path),
        checkout=root,
        launch=recorder(started),
    )
    assert started == ["1"]
    assert rows == [
        {"recording": "3", "state": "done", "broadcast": 3000},
        {"recording": "2", "state": "held", "broadcast": 2000},
        {"recording": "1", "state": "launched", "broadcast": 1000, "launch": "pipeline-v1"},
        {"recording": "9", "state": "unbound", "broadcast": 0},
    ]


def test_a_code_change_after_the_failures_retries_a_held_recording(tmp_path: Path) -> None:
    root = checkout(tmp_path)
    recording(tmp_path, "2")
    launched(tmp_path, "2", "1", "1")
    os.utime(tmp_path / "src/replay/module.py", (3_000, 3_000))
    rows = queue.tick(
        queue_file(tmp_path, "2"), defaults(tmp_path), checkout=root, launch=lambda _r: "v3"
    )
    assert rows == [{"recording": "2", "state": "launched", "broadcast": 2000, "launch": "v3"}]


def test_unhold_resolves_the_earlier_failures_of_a_held_recording(tmp_path: Path) -> None:
    root = checkout(tmp_path)
    directory = recording(tmp_path, "2")
    launched(tmp_path, "2", "1", "1")
    held = queue.tick(
        queue_file(tmp_path, "2"), defaults(tmp_path), checkout=root, launch=lambda _r: "v3"
    )
    assert held == [{"recording": "2", "state": "held", "broadcast": 2000}]
    queue.unhold(pipelines.Layout(directory))
    rows = queue.tick(
        queue_file(tmp_path, "2"), defaults(tmp_path), checkout=root, launch=lambda _r: "v3"
    )
    assert rows == [{"recording": "2", "state": "launched", "broadcast": 2000, "launch": "v3"}]
    assert (directory / "pipeline-v1.status").read_text(encoding="utf-8") == "1\n"


def test_the_unhold_command_marks_the_recording(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.chdir(checkout(tmp_path))
    directory = recording(tmp_path, "2")
    monkeypatch.setattr(pipelines, "load_defaults", lambda _path: defaults(tmp_path))
    assert queue.main(["unhold", "2"]) == 0
    assert capsys.readouterr().out == "unheld 2\n"
    assert queue.resolved(pipelines.Layout(directory)) > 0


def test_a_paused_queue_launches_nothing(tmp_path: Path) -> None:
    root = checkout(tmp_path)
    rows = queue.tick(
        queue_file(tmp_path, "3", paused=True), defaults(tmp_path), checkout=root, launch=str
    )
    assert rows == [{"recording": "3", "state": "ready", "broadcast": 3000}]


def test_the_disk_floor_stops_launching_for_this_tick(tmp_path: Path) -> None:
    root = checkout(tmp_path)

    def refuse(_replay_id: str) -> str:
        raise subprocess.CalledProcessError(queue.DISK_FLOOR_EXIT, ["tmux-job.sh"])

    rows = queue.tick(
        queue_file(tmp_path, "3", "2"), defaults(tmp_path), checkout=root, launch=refuse
    )
    assert rows == [
        {"recording": "3", "state": "launch-refused", "broadcast": 3000},
        {"recording": "2", "state": "ready", "broadcast": 2000},
    ]


def test_another_refused_launch_moves_on_to_the_next_recording(tmp_path: Path) -> None:
    root = checkout(tmp_path)

    def launch(replay_id: str) -> str:
        if replay_id == "3":
            raise subprocess.CalledProcessError(3, ["tmux-job.sh"])
        return "pipeline-v1"

    rows = queue.tick(
        queue_file(tmp_path, "3", "2"), defaults(tmp_path), checkout=root, launch=launch
    )
    assert rows == [
        {"recording": "3", "state": "launch-refused", "broadcast": 3000},
        {"recording": "2", "state": "launched", "broadcast": 2000, "launch": "pipeline-v1"},
    ]


@pytest.mark.parametrize(
    ("data", "message"),
    [
        ({"version": 2, "slots": 1, "recordings": []}, "Unsupported"),
        ({"version": 1, "slots": 1, "recordings": [], "extra": 1}, "Unsupported"),
        ({"version": 1, "slots": 4, "recordings": []}, "slots must be"),
        ({"version": 1, "slots": 1, "paused": "yes", "recordings": []}, "paused must be"),
        ({"version": 1, "slots": 1, "recordings": {}}, "paused must be"),
        ({"version": 1, "slots": 1, "recordings": ["../x"]}, "TikTok replay ID"),
    ],
)
def test_an_invalid_queue_is_refused(tmp_path: Path, data: object, message: str) -> None:
    path = tmp_path / "queue.json"
    path.write_text(json.dumps(data), encoding="utf-8")
    with pytest.raises(ReplayError, match=message):
        queue.load_queue(path)


def bound(tmp_path: Path, replay_id: str) -> Path:
    """An edit root whose edit config names an existing source."""
    root = tmp_path / "edits" / f"replay-{replay_id}"
    root.mkdir(parents=True)
    (tmp_path / "source.mp4").write_bytes(b"video")
    source = json.dumps({"source": str(tmp_path / "source.mp4")})
    (root / "edit-config.json").write_text(source, encoding="utf-8")
    return root


def test_a_validated_broadcast_is_reused_until_its_inputs_change(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root = bound(tmp_path, "3")
    validated: list[str] = []

    def broadcast(edit_root: Path) -> int:
        validated.append(edit_root.name)
        return 3000

    monkeypatch.setattr(pipelines, "broadcast", broadcast)
    assert queue.cached_broadcast(root, tmp_path) == 3000
    assert queue.cached_broadcast(root, tmp_path) == 3000
    assert validated == ["replay-3"]
    (tmp_path / "source.mp4").write_bytes(b"other video")
    assert queue.cached_broadcast(root, tmp_path) == 3000
    assert validated == ["replay-3", "replay-3"]


def test_an_unreadable_broadcast_cache_is_validated_again(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root = bound(tmp_path, "3")
    cache = tmp_path / queue.BROADCASTS
    cache.parent.mkdir(parents=True)
    cache.write_text("[", encoding="utf-8")
    monkeypatch.setattr(pipelines, "broadcast", lambda _root: 3000)
    assert queue.cached_broadcast(root, tmp_path) == 3000
    assert json.loads(cache.read_text(encoding="utf-8"))["replay-3"]["ns"] == 3000


def test_add_appends_a_recording_once(tmp_path: Path) -> None:
    path = queue_file(tmp_path, "1", slots=3)
    assert queue.add(path, "2") is True
    assert queue.add(path, "2") is False
    assert json.loads(path.read_text(encoding="utf-8")) == {
        "version": 1, "slots": 3, "paused": False, "recordings": ["1", "2"]
    }  # fmt: skip


def test_window_alive_lists_the_recordings_tmux_windows(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    calls: list[tuple[list[str], Path]] = []

    def run(
        command: list[str], *, cwd: Path, **_options: object
    ) -> subprocess.CompletedProcess[str]:
        calls.append((command, cwd))
        return subprocess.CompletedProcess(command, 0, stdout="pipeline-v1\npipeline-v2\n")

    monkeypatch.setattr(subprocess, "run", run)
    assert WINDOW_ALIVE(tmp_path, "pipeline-v2") is True
    assert WINDOW_ALIVE(tmp_path, "pipeline-v3") is False
    assert calls[0] == (
        ["tmux", "-f", "/dev/null", "-S", "tmux.sock", "list-windows", "-a", "-F",
         "#{window_name}"],
        tmp_path,
    )  # fmt: skip


def test_the_launcher_dispatches_into_the_recordings_directory(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    calls: list[tuple[str, str, Path]] = []

    def dispatch(layout: pipelines.Layout, replay_id: str, path: Path, **options: object) -> str:
        calls.append((replay_id, str(options["python"]), layout.directory))
        return "pipeline-v1"

    monkeypatch.setattr(pipelines, "dispatch", dispatch)
    launch = queue.launcher(tmp_path / "defaults.json", defaults(tmp_path), tmp_path)
    assert launch("3") == "pipeline-v1"
    assert calls == [("3", "/py", tmp_path / "edits/work/recordings/replay-3")]
    assert (tmp_path / "edits/work/recordings/replay-3").is_dir()


def test_the_cli_runs_one_tick_and_adds(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    root = checkout(tmp_path)
    monkeypatch.chdir(root)
    monkeypatch.setattr(pipelines, "load_defaults", lambda _path: defaults(tmp_path))
    monkeypatch.setattr(queue, "launcher", lambda *_arguments: lambda _replay_id: "pipeline-v1")
    path = queue_file(tmp_path, slots=1)
    assert queue.main(["add", str(path), "3"]) == 0
    assert queue.main(["run", "--queue", str(path), "--defaults", "d.json", "--once"]) == 0
    printed = capsys.readouterr().out.splitlines()
    assert printed[0] == "added"
    assert json.loads(printed[1])["recordings"] == [
        {"recording": "3", "state": "launched", "broadcast": 3000, "launch": "pipeline-v1"}
    ]
    assert json.loads(printed[1])["volumeRetriesToday"] == 0


def test_the_cli_survives_a_broken_tick_and_polls_on(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    root = checkout(tmp_path)
    monkeypatch.chdir(root)
    monkeypatch.setattr(pipelines, "load_defaults", lambda _path: defaults(tmp_path))
    path = tmp_path / "queue.json"
    path.write_text("{", encoding="utf-8")  # caught mid-edit
    slept: list[float] = []

    def sleep(seconds: float) -> None:
        slept.append(seconds)
        if len(slept) == 1:
            queue_file(tmp_path, slots=1, paused=True)
            return
        raise KeyboardInterrupt

    with pytest.raises(KeyboardInterrupt):
        queue.main(
            ["run", "--queue", str(path), "--defaults", "d.json", "--poll", "5"], sleep=sleep
        )
    captured = capsys.readouterr()
    assert slept == [5.0, 5.0]
    assert captured.err.startswith("Queue tick failed (JSONDecodeError): ")
    assert json.loads(captured.out)["recordings"] == []


def test_the_cli_reports_a_failed_single_tick(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.chdir(checkout(tmp_path))
    monkeypatch.setattr(pipelines, "load_defaults", lambda _path: defaults(tmp_path))
    path = tmp_path / "queue.json"
    path.write_text("{}", encoding="utf-8")
    assert queue.main(["run", "--queue", str(path), "--defaults", "d.json", "--once"]) == 1
    assert (
        capsys.readouterr().err == "Queue tick failed (ReplayError): Unsupported recording queue\n"
    )


def test_unreadable_defaults_stop_the_runner(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    code = queue.main(["run", "--queue", "q.json", "--defaults", str(tmp_path / "none.json")])
    assert code == 1
    assert capsys.readouterr().err.startswith("Queue stopped (FileNotFoundError): ")


def test_one_unreadable_recording_does_not_stall_the_others(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root = checkout(tmp_path)
    original = pipelines.layout_of

    def layout_of(replay_id: str, settings: dict[str, object], where: Path) -> pipelines.Layout:
        if replay_id == "3":
            raise ReplayError("The pipeline root must live under the checkout's edits/work")
        return original(replay_id, settings, where)

    monkeypatch.setattr(pipelines, "layout_of", layout_of)
    rows = queue.tick(
        queue_file(tmp_path, "3", "2"), defaults(tmp_path), checkout=root, launch=lambda _r: "v1"
    )
    assert rows == [
        {"recording": "2", "state": "launched", "broadcast": 2000, "launch": "v1"},
        {
            "recording": "3",
            "state": "error",
            "broadcast": 0,
            "error": "ReplayError: The pipeline root must live under the checkout's edits/work",
        },
    ]


def test_the_cli_publishes_the_recording_rank_each_tick_when_asked(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root = checkout(tmp_path)
    monkeypatch.chdir(root)
    monkeypatch.setattr(pipelines, "load_defaults", lambda _path: defaults(tmp_path))
    monkeypatch.setattr(queue, "launcher", lambda *_arguments: lambda _replay_id: "pipeline-v1")
    monkeypatch.setattr(gpu_lease, "root", lambda: tmp_path / "work" / "gpu-leases")
    leases = tmp_path / "work" / "gpu-leases"
    leases.mkdir(parents=True)
    (leases / "settings.json").write_text('{"capacity": 3, "priority": ["9"]}', "utf-8")
    path = queue_file(tmp_path, "2", "3", slots=1, paused=True)
    arguments = ["run", "--queue", str(path), "--defaults", "d.json", "--once"]
    assert queue.main(arguments) == 0
    assert json.loads((leases / "settings.json").read_text("utf-8"))["priority"] == ["9"]
    assert queue.main([*arguments, "--order-leases"]) == 0
    assert json.loads((leases / "settings.json").read_text("utf-8")) == {
        "capacity": 3,
        "priority": ["3", "2"],
    }
    assert json.loads((tmp_path / "work" / "lease-urgent.json").read_text("utf-8")) == {
        "urgent": []
    }


def test_a_failed_lease_order_is_reported_and_the_queue_goes_on(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    def broken(*_arguments: object) -> None:
        raise ValueError("GPU lease settings are too large")

    monkeypatch.setattr(priority, "publish", broken)
    path = queue_file(tmp_path, "3")
    queue.order_leases(path, [{"recording": "3", "state": "ready", "broadcast": 3000}])
    assert capsys.readouterr().err == (
        "Lease order not refreshed (ValueError): GPU lease settings are too large\n"
    )


def test_auto_slots_admit_one_ready_recording_per_tick_while_memory_admits(tmp_path: Path) -> None:
    root = checkout(tmp_path)
    directory = recording(tmp_path, "3")
    started: list[str] = []
    with execution_lock(directory, "pipeline"):
        rows = queue.tick(
            queue_file(tmp_path, "1", "2", "3", slots="auto"),
            defaults(tmp_path),
            checkout=root,
            launch=recorder(started),
            admits=lambda: True,
        )
    assert started == ["2"]
    assert [row["state"] for row in rows] == ["running", "launched", "ready"]
    held = queue.tick(
        queue_file(tmp_path, "1", slots="auto"),
        defaults(tmp_path),
        checkout=root,
        launch=recorder(started),
        admits=lambda: False,
    )
    assert started == ["2"]
    assert [row["state"] for row in held] == ["ready"]


def test_memory_admits_needs_normal_pressure_free_memory_and_spare_reservations(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(memory_admission, "root", lambda: tmp_path)
    monkeypatch.setattr(memory_admission, "physical", lambda: 48 * 2**30)
    monkeypatch.setattr(memory_admission, "pressure", lambda: (95, 1))
    assert queue.memory_admits() is True
    monkeypatch.setattr(memory_admission, "pressure", lambda: (95, 2))
    assert queue.memory_admits() is False
    monkeypatch.setattr(memory_admission, "pressure", lambda: (10, 1))
    assert queue.memory_admits() is False
    monkeypatch.setattr(memory_admission, "pressure", lambda: (95, 1))
    (tmp_path / "settings.json").write_text('{"budgetGiB": 3}', "utf-8")
    assert queue.memory_admits() is False


def priority_command(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, *argv: str
) -> list[dict[str, object]]:
    monkeypatch.chdir(tmp_path)  # the test made the checkout
    monkeypatch.setattr(pipelines, "load_defaults", lambda _path: defaults(tmp_path))
    monkeypatch.setattr(gpu_lease, "root", lambda: tmp_path / "work" / "gpu-leases")
    path = tmp_path / "queue.json"
    if not path.exists():
        queue_file(tmp_path, "1", "2", "3")
    capture = []
    monkeypatch.setattr("builtins.print", lambda text, **_k: capture.append(text))
    assert queue.main([*argv, "--queue", str(path), "--defaults", "d.json"]) == 0
    rows = json.loads(capture[-1])
    assert isinstance(rows, list)
    return rows


def ranks(rows: list[dict[str, object]]) -> list[object]:
    return [row["recording"] for row in rows]


def test_the_priority_commands_reorder_pause_and_make_urgent(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    checkout(tmp_path)
    leases = tmp_path / "work" / "gpu-leases"
    leases.mkdir(parents=True)
    (leases / "settings.json").write_text('{"capacity": 2}', "utf-8")
    listed = priority_command(tmp_path, monkeypatch, "list")
    assert ranks(listed) == ["3", "2", "1"]
    assert listed[0]["broadcastJST"] == "1970-01-01T09:00:00.000003+09:00"
    assert ranks(priority_command(tmp_path, monkeypatch, "move", "1", "--top")) == ["1", "3", "2"]
    assert ranks(priority_command(tmp_path, monkeypatch, "move", "3", "--after", "2")) == [
        "1", "2", "3",
    ]  # fmt: skip
    assert ranks(priority_command(tmp_path, monkeypatch, "move", "3", "--before", "1")) == [
        "3", "1", "2",
    ]  # fmt: skip
    assert ranks(priority_command(tmp_path, monkeypatch, "move", "2", "--to", "2")) == [
        "3", "2", "1",
    ]  # fmt: skip
    paused = priority_command(tmp_path, monkeypatch, "pause", "3")
    assert ranks(paused) == ["2", "1", "3"]
    assert paused[2]["paused"] is True
    assert ranks(priority_command(tmp_path, monkeypatch, "resume", "3")) == ["3", "2", "1"]
    urgent = priority_command(tmp_path, monkeypatch, "urgent", "1")
    assert ranks(urgent) == ["1", "3", "2"]
    assert (urgent[0]["urgent"], urgent[0]["blockedBy"], urgent[0]["waitingFor"]) == (True, [], [])
    assert json.loads((leases / "settings.json").read_text("utf-8"))["priority"] == ["1", "3", "2"]
    assert ranks(priority_command(tmp_path, monkeypatch, "auto-order")) == ["1", "3", "2"]
    assert ranks(priority_command(tmp_path, monkeypatch, "pin-order")) == ["1", "3", "2"]


def test_moving_needs_a_place_and_queued_recordings(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.chdir(checkout(tmp_path))
    monkeypatch.setattr(pipelines, "load_defaults", lambda _path: defaults(tmp_path))
    monkeypatch.setattr(gpu_lease, "root", lambda: tmp_path / "work" / "gpu-leases")
    path = queue_file(tmp_path, "1", "2")
    common = ["--queue", str(path), "--defaults", "d.json"]
    assert queue.main(["move", "1", *common]) == 1
    assert queue.main(["move", "9", "--top", *common]) == 1
    assert queue.main(["move", "1", "--before", "8", *common]) == 1
    assert capsys.readouterr().err.splitlines() == [
        "Queue stopped (ReplayError): move needs --to, --top, --before or --after",
        "Queue stopped (ReplayError): 9 is not queued (or is urgent)",
        "Queue stopped (ReplayError): 8 is not queued",
    ]


def test_an_urgent_recording_is_enqueued_and_drops_out_once_done(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    checkout(tmp_path)
    queue_file(tmp_path, "1", "2")
    listed = priority_command(tmp_path, monkeypatch, "urgent", "3")
    assert ranks(listed) == ["3", "2", "1"]
    recording(tmp_path, "3")
    finished(tmp_path, "3")
    listed = priority_command(tmp_path, monkeypatch, "list")
    assert [(row["recording"], row["urgent"], row["state"]) for row in listed] == [
        ("3", False, "done"), ("2", False, "ready"), ("1", False, "ready"),
    ]  # fmt: skip


def test_the_current_step_is_the_newest_started_step_without_a_receipt(tmp_path: Path) -> None:
    logs = tmp_path / "steps" / "logs"
    logs.mkdir(parents=True)
    (logs / "clocks-1.run.log").write_text("", "utf-8")
    (logs / "cleanup-2.run.log").write_text("", "utf-8")
    os.utime(logs / "clocks-1.run.log", (1, 1))
    (tmp_path / "steps" / "cleanup.json").write_text("{}", "utf-8")
    assert queue.current_step(tmp_path / "steps") == "clocks"
    (tmp_path / "steps" / "clocks.json").write_text("{}", "utf-8")
    assert queue.current_step(tmp_path / "steps") is None


def test_drain_and_undrain_mark_a_recording_and_the_tick_skips_it(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    root = checkout(tmp_path)
    monkeypatch.setattr(gpu_lease, "root", lambda: tmp_path / "work" / "gpu-leases")
    assert queue.main(["drain", "3", "--reason", "relaunch"]) == 0
    assert priority.drained(tmp_path / "work", "3") == "relaunch"
    started: list[str] = []
    rows = queue.tick(queue_file(tmp_path, "3"), defaults(tmp_path), checkout=root,
                      launch=recorder(started))  # fmt: skip
    assert (started, rows[0]["state"]) == ([], "drained")
    assert queue.main(["undrain", "3"]) == 0
    assert priority.drained(tmp_path / "work", "3") is None
    assert capsys.readouterr().out.splitlines() == ["drained 3", "undrained 3"]


def test_drained_runs_do_not_count_as_failures(tmp_path: Path) -> None:
    root = checkout(tmp_path)
    recording(tmp_path, "3")
    launched(tmp_path, "3", "75", "75", "75")
    started: list[str] = []
    rows = queue.tick(queue_file(tmp_path, "3"), defaults(tmp_path), checkout=root,
                      launch=recorder(started))  # fmt: skip
    assert (started, rows[0]["state"]) == (["3"], "launched")
