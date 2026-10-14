from __future__ import annotations

import fcntl
import json
import os
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from threading import Barrier

import pytest

from replay import phase_timer
from replay.core import ReplayError


def test_records_phase_and_mirrors_exit(tmp_path: Path) -> None:
    log = tmp_path / "phases.jsonl"
    code = phase_timer.main(
        [str(log), "probe", "--", sys.executable, "-c", "import sys; sys.exit(3)"]
    )
    assert code == 3
    first = json.loads(log.read_text(encoding="utf-8").splitlines()[0])
    assert first["phase"] == "probe" and first["exit"] == 3
    assert first["end"] >= first["start"] and first["seconds"] >= 0


def test_appends_and_handles_missing_command(tmp_path: Path) -> None:
    log = tmp_path / "phases.jsonl"
    assert phase_timer.main([str(log), "a", "--", sys.executable, "-c", "pass"]) == 0
    assert phase_timer.main([str(log), "b", "--", "/no/such/command"]) == 127
    rows = [json.loads(line) for line in log.read_text(encoding="utf-8").splitlines()]
    assert [row["phase"] for row in rows] == ["a", "b"]
    assert oct(log.stat().st_mode & 0o777) == "0o600"


def test_usage_error() -> None:
    assert phase_timer.main(["only", "two"]) == 2


@pytest.mark.parametrize("phase", ["", "x" * 257, "bad\nphase", "bad\0phase"])
def test_invalid_phase_never_launches_child(tmp_path: Path, phase: str) -> None:
    with pytest.raises(ReplayError, match="bounded phase"):
        phase_timer.run_timed(tmp_path / "log", phase, ["missing"])
    assert not (tmp_path / "log").exists()


@pytest.mark.parametrize("command", [[], [""], ["bad\0command"]])
def test_invalid_command_never_opens_log(tmp_path: Path, command: list[str]) -> None:
    with pytest.raises(ReplayError, match="valid command"):
        phase_timer.run_timed(tmp_path / "log", "valid", command)
    assert not (tmp_path / "log").exists()


def test_log_scope_rejection_precedes_child_launch(monkeypatch: pytest.MonkeyPatch) -> None:
    calls: list[list[str]] = []
    monkeypatch.setattr(subprocess, "run", lambda command, **_kwargs: calls.append(command))
    assert phase_timer.main(["/tmp/forbidden.jsonl", "test", "--", "command"]) == 1
    assert calls == []


@pytest.mark.parametrize("artifact", ["link", "hardlink", "mode", "directory", "fifo"])
def test_unsafe_log_never_launches_child(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, artifact: str
) -> None:
    log = tmp_path / "log"
    original = tmp_path / "original"
    original.write_bytes(b"preserved")
    original.chmod(0o600)
    if artifact == "link":
        log.symlink_to(original)
    elif artifact == "hardlink":
        os.link(original, log)
    elif artifact == "mode":
        log.write_bytes(b"preserved")
        log.chmod(0o644)
    elif artifact == "directory":
        log.mkdir()
    else:
        os.mkfifo(log)
    calls: list[list[str]] = []
    monkeypatch.setattr(subprocess, "run", lambda command, **_kwargs: calls.append(command))
    assert phase_timer.main([str(log), "test", "--", "command"]) == 1
    assert calls == []
    assert original.read_bytes() == b"preserved"


def test_duration_is_monotonic_when_wall_clock_moves_backwards(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    clock = [100.0]
    epochs = iter([200.0, 150.0])
    monkeypatch.setattr(time, "time", lambda: next(epochs))
    monkeypatch.setattr(time, "monotonic", lambda: clock[0])

    def run(command: list[str], **_kwargs: object) -> subprocess.CompletedProcess[str]:
        clock[0] = 102.0
        return subprocess.CompletedProcess(command, 7)

    monkeypatch.setattr(subprocess, "run", run)
    log = tmp_path / "log"
    assert phase_timer.run_timed(log, "clock", ["command"]) == 7
    assert json.loads(log.read_text(encoding="utf-8")) == {
        "phase": "clock",
        "start": 200.0,
        "end": 150.0,
        "seconds": 2.0,
        "exit": 7,
    }


def test_short_writes_finish_one_complete_record(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    original = os.write
    monkeypatch.setattr(os, "write", lambda fd, data: original(fd, data[:3]))
    monkeypatch.setattr(
        subprocess, "run", lambda command, **_kwargs: subprocess.CompletedProcess(command, 0)
    )
    log = tmp_path / "log"
    assert phase_timer.run_timed(log, "partial", ["command"]) == 0
    assert json.loads(log.read_text(encoding="utf-8"))["phase"] == "partial"
    assert len(log.read_text(encoding="utf-8").splitlines()) == 1


def test_failed_append_reports_child_exit_without_relaunch(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    calls: list[list[str]] = []

    def run(command: list[str], **_kwargs: object) -> subprocess.CompletedProcess[str]:
        calls.append(command)
        return subprocess.CompletedProcess(command, 9)

    monkeypatch.setattr(subprocess, "run", run)
    monkeypatch.setattr(os, "write", lambda _fd, _data: 0)
    assert phase_timer.main([str(tmp_path / "log"), "failure", "--", "command"]) == 1
    assert calls == [["command"]]
    assert "after child exit 9" in capsys.readouterr().err


def test_partial_tail_is_preserved_and_rejects_next_child(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    log = tmp_path / "log"
    log.write_bytes(b'{"phase":"unfinished"')
    log.chmod(0o600)
    calls: list[list[str]] = []
    monkeypatch.setattr(subprocess, "run", lambda command, **_kwargs: calls.append(command))
    assert phase_timer.main([str(log), "next", "--", "command"]) == 1
    assert calls == []
    assert log.read_bytes() == b'{"phase":"unfinished"'


@pytest.mark.parametrize("_attempt", range(32))
def test_concurrent_appends_are_complete_rows(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, _attempt: int
) -> None:
    monkeypatch.setattr(
        subprocess, "run", lambda command, **_kwargs: subprocess.CompletedProcess(command, 0)
    )
    log = tmp_path / "log"
    ready = Barrier(4)

    def measured(phase: str) -> int:
        ready.wait(timeout=5)
        return phase_timer.run_timed(log, phase, ["command"])

    with ThreadPoolExecutor(max_workers=4) as pool:
        assert list(pool.map(measured, ["one", "two", "three", "four"])) == [0, 0, 0, 0]
    rows = [json.loads(line) for line in log.read_text(encoding="utf-8").splitlines()]
    assert sorted(row["phase"] for row in rows) == ["four", "one", "three", "two"]


def test_append_lock_timeout_is_finite(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    clock = [0.0]
    monkeypatch.setattr(time, "monotonic", lambda: clock[0])
    monkeypatch.setattr(time, "sleep", lambda seconds: clock.__setitem__(0, clock[0] + seconds))

    def busy(_descriptor: int, _operation: int) -> None:
        raise BlockingIOError("busy")

    monkeypatch.setattr(fcntl, "flock", busy)
    with (
        (tmp_path / "log").open("wb") as stream,
        pytest.raises(ReplayError, match="finite deadline"),
        phase_timer._append_lease(stream.fileno()),
    ):
        raise RuntimeError("must not acquire")


def test_signal_exit_uses_shell_status(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        subprocess, "run", lambda command, **_kwargs: subprocess.CompletedProcess(command, -9)
    )
    assert phase_timer.run_timed(tmp_path / "log", "signal", ["command"]) == 137


def test_first_open_enoent_rechecks_parent_and_runs_child_once(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    original_open = os.open
    opens: list[int | None] = []
    calls: list[list[str]] = []

    def open_log(
        path: str | Path, flags: int, mode: int = 0o777, *, dir_fd: int | None = None
    ) -> int:
        if path == "log":
            opens.append(dir_fd)
            if len(opens) == 1:
                raise FileNotFoundError(2, "Observed first-creation failure", path)
        return original_open(path, flags, mode, dir_fd=dir_fd)

    def run(command: list[str], **_kwargs: object) -> subprocess.CompletedProcess[str]:
        calls.append(command)
        return subprocess.CompletedProcess(command, 0)

    monkeypatch.setattr(os, "open", open_log)
    monkeypatch.setattr(subprocess, "run", run)
    log = tmp_path / "log"
    assert phase_timer.run_timed(log, "first-create", ["command"]) == 0
    assert len(opens) == 2 and opens[0] == opens[1]
    assert calls == [["command"]]
    assert json.loads(log.read_text(encoding="utf-8"))["phase"] == "first-create"
    assert "PHASE_TIMER_OPEN_RETRY errno=2" in capsys.readouterr().err


def test_persistent_open_enoent_is_finite_and_never_starts_child(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    original_open = os.open
    opens: list[int | None] = []
    calls: list[list[str]] = []

    def open_log(
        path: str | Path, flags: int, mode: int = 0o777, *, dir_fd: int | None = None
    ) -> int:
        if path == "log":
            opens.append(dir_fd)
            raise FileNotFoundError(2, "Persistent failure", path)
        return original_open(path, flags, mode, dir_fd=dir_fd)

    monkeypatch.setattr(os, "open", open_log)
    monkeypatch.setattr(subprocess, "run", lambda command, **_kwargs: calls.append(command))
    assert phase_timer.main([str(tmp_path / "log"), "failure", "--", "command"]) == 1
    assert len(opens) == 2 and opens[0] == opens[1]
    assert calls == []
    assert not (tmp_path / "log").exists()


def test_replaced_parent_rejects_open_retry_before_child(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    directory = tmp_path / "parent"
    directory.mkdir()
    original_open = os.open
    opens: list[int | None] = []
    calls: list[list[str]] = []

    def open_log(
        path: str | Path, flags: int, mode: int = 0o777, *, dir_fd: int | None = None
    ) -> int:
        if path == "log":
            opens.append(dir_fd)
            directory.rename(tmp_path / "retained-parent")
            directory.mkdir()
            raise FileNotFoundError(2, "Parent replaced", path)
        return original_open(path, flags, mode, dir_fd=dir_fd)

    monkeypatch.setattr(os, "open", open_log)
    monkeypatch.setattr(subprocess, "run", lambda command, **_kwargs: calls.append(command))
    with pytest.raises(ReplayError, match="parent changed"):
        phase_timer.run_timed(directory / "log", "failure", ["command"])
    assert len(opens) == 1
    assert calls == []
    assert not (directory / "log").exists()
    assert not (tmp_path / "retained-parent" / "log").exists()


def test_parent_replaced_after_open_rejects_child(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    directory = tmp_path / "parent"
    directory.mkdir()
    original_open = os.open
    calls: list[list[str]] = []

    def open_log(
        path: str | Path, flags: int, mode: int = 0o777, *, dir_fd: int | None = None
    ) -> int:
        descriptor = original_open(path, flags, mode, dir_fd=dir_fd)
        if path == "log":
            directory.rename(tmp_path / "retained-parent")
            directory.mkdir()
        return descriptor

    monkeypatch.setattr(os, "open", open_log)
    monkeypatch.setattr(subprocess, "run", lambda command, **_kwargs: calls.append(command))
    with pytest.raises(ReplayError, match="parent changed"):
        phase_timer.run_timed(directory / "log", "failure", ["command"])
    assert calls == []
    assert (tmp_path / "retained-parent" / "log").read_bytes() == b""
    assert not (directory / "log").exists()


def test_existing_log_enoent_is_not_retried_or_recreated(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    log = tmp_path / "log"
    log.write_bytes(b'{"phase":"preserved"}\n')
    log.chmod(0o600)
    original_open = os.open
    flags_seen: list[int] = []
    calls: list[list[str]] = []

    def open_log(
        path: str | Path, flags: int, mode: int = 0o777, *, dir_fd: int | None = None
    ) -> int:
        if path == "log":
            flags_seen.append(flags)
            raise FileNotFoundError(2, "Existing log missing during open", path)
        return original_open(path, flags, mode, dir_fd=dir_fd)

    monkeypatch.setattr(os, "open", open_log)
    monkeypatch.setattr(subprocess, "run", lambda command, **_kwargs: calls.append(command))
    assert phase_timer.main([str(log), "failure", "--", "command"]) == 1
    assert len(flags_seen) == 1
    assert not flags_seen[0] & os.O_CREAT
    assert calls == []
    assert log.read_bytes() == b'{"phase":"preserved"}\n'


def test_existing_log_removed_during_open_is_not_recreated(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    log = tmp_path / "log"
    log.write_bytes(b'{"phase":"preserved"}\n')
    log.chmod(0o600)
    original_open = os.open
    calls: list[list[str]] = []

    def open_log(
        path: str | Path, flags: int, mode: int = 0o777, *, dir_fd: int | None = None
    ) -> int:
        if path == "log":
            log.unlink()
        return original_open(path, flags, mode, dir_fd=dir_fd)

    monkeypatch.setattr(os, "open", open_log)
    monkeypatch.setattr(subprocess, "run", lambda command, **_kwargs: calls.append(command))
    assert phase_timer.main([str(log), "failure", "--", "command"]) == 1
    assert calls == []
    assert not log.exists()


def test_existing_log_replaced_before_open_rejects_child(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    log = tmp_path / "log"
    log.write_bytes(b'{"phase":"preserved"}\n')
    log.chmod(0o600)
    original_open = os.open
    calls: list[list[str]] = []

    def open_log(
        path: str | Path, flags: int, mode: int = 0o777, *, dir_fd: int | None = None
    ) -> int:
        if path == "log":
            log.rename(tmp_path / "retained-log")
            log.write_bytes(b'{"phase":"replacement"}\n')
            log.chmod(0o600)
        return original_open(path, flags, mode, dir_fd=dir_fd)

    monkeypatch.setattr(os, "open", open_log)
    monkeypatch.setattr(subprocess, "run", lambda command, **_kwargs: calls.append(command))
    with pytest.raises(ReplayError, match="existing log changed"):
        phase_timer.run_timed(log, "failure", ["command"])
    assert calls == []
    assert (tmp_path / "retained-log").read_bytes() == b'{"phase":"preserved"}\n'
    assert log.read_bytes() == b'{"phase":"replacement"}\n'
