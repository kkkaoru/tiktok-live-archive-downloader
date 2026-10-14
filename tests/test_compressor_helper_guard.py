import fcntl
import json
import os
import signal
import subprocess
import time
from dataclasses import asdict, replace
from pathlib import Path
from unittest.mock import Mock

import pytest

from replay import compressor_helper_guard as guard
from replay.core import ReplayError

HELPER = (
    "/Selected Compressor.app/Contents/Helpers/"
    "RegisterProExtension.app/Contents/MacOS/RegisterProExtension"
)


@pytest.fixture(autouse=True)
def no_live_processes(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(subprocess, "run", Mock(side_effect=AssertionError("Live ps forbidden")))
    monkeypatch.setattr(os, "kill", Mock(side_effect=AssertionError("Live signals forbidden")))


@pytest.fixture
def old() -> guard.Process:
    return guard.Process(42, 1, os.getuid(), 300, "Thu Oct 08 23:00:00 2026", HELPER)


@pytest.mark.parametrize(
    "text,expected", [("00:01", 1), ("05:00", 300), ("01:02:03", 3723), ("2-03:04:05", 183845)]
)
def test_elapsed(text: str, expected: int) -> None:
    assert guard.elapsed_seconds(text) == expected


@pytest.mark.parametrize(
    "text", ["x", "60:00", "00:60", "1-24:00:00", "-1:00", "1-00:01", "24:00:00"]
)
def test_bad_elapsed(text: str) -> None:
    with pytest.raises(guard.ElapsedParseError):
        guard.elapsed_seconds(text)


def test_parse_metadata() -> None:
    rows = guard.parse_processes(
        "\n 42 1 501 2-03:04:05 Thu Oct 8 23:00:00 2026 /App With Spaces/helper\n"
    )
    assert rows == [
        guard.Process(42, 1, 501, 183845, "Thu Oct 8 23:00:00 2026", "/App With Spaces/helper")
    ]


@pytest.mark.parametrize(
    "text",
    [
        "bad",
        "x 1 501 05:00 Thu Oct 8 23:00:00 2026 /x",
        "0 1 501 05:00 Thu Oct 8 23:00:00 2026 /x",
        "1 1 501 05:00 Bad Date 8 23:00:00 2026 /x",
        "1 1 501 05:00 Thu Oct 8 23:00:00 2026 /x\n1 1 501 05:00 Thu Oct 8 23:00:00 2026 /x",
    ],
)
def test_bad_metadata(text: str) -> None:
    with pytest.raises((ReplayError, ValueError)):
        guard.parse_processes(text)


def test_signed_system_uid_is_parsed_but_never_eligible() -> None:
    rows = guard.parse_processes("646 1 -2 20:16 Fri Oct 9 14:37:05 2026 /usr/libexec/dhcp6d\n")
    assert rows[0].uid == -2
    assert not guard.eligible(rows[0], HELPER)
    assert not guard.eligible(replace(rows[0], executable=HELPER), HELPER)


def test_low_count_appends_private_evidence(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, old: guard.Process
) -> None:
    monkeypatch.setattr(guard, "processes", Mock(return_value=[old] * 299))
    with guard.private_file(tmp_path / "events") as stream:
        guard.reclaim(tmp_path, HELPER, stream)
        guard.reclaim(tmp_path, HELPER, stream)
    records = (tmp_path / "events").read_text().splitlines()
    assert len(records) == 2
    assert json.loads(records[0])["count"] == 299
    assert (tmp_path / "events").stat().st_mode & 0o777 == 0o600


def test_old_only_cleanup_and_exit_verification(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, old: guard.Process
) -> None:
    young = replace(old, pid=43, age=299)
    query = Mock(side_effect=[[old] + [young] * 299, [old], [old, young], [young]])
    kill = Mock()
    sleep = Mock()
    monkeypatch.setattr(guard, "processes", query)
    monkeypatch.setattr(os, "kill", kill)
    monkeypatch.setattr(time, "sleep", sleep)
    with guard.private_file(tmp_path / "events") as stream:
        guard.reclaim(tmp_path, HELPER, stream)
    kill.assert_called_once_with(42, signal.SIGTERM)
    sleep.assert_called_once_with(1)
    assert '"event": "recovered"' in (tmp_path / "events").read_text()


@pytest.mark.parametrize(
    "change",
    [
        {"executable": HELPER + "-other"},
        {"executable": "/Other.app/Contents/MacOS/JobController"},
        {"executable": "/Other.app/Contents/MacOS/Transcoder"},
        {"executable": "/Other.app/Contents/MacOS/Renderer"},
        {"executable": "/Selected Compressor.app/Contents/MacOS/Compressor"},
        {"uid": -1},
        {"ppid": 2},
    ],
)
def test_scope_excludes_other_processes(old: guard.Process, change: dict[str, object]) -> None:
    values = {**asdict(old), **change}
    # Construct explicitly to keep the fixture boundary typed.
    process = guard.parse_processes(
        f"42 {values['ppid']} {max(0, int(str(values['uid'])))} "
        f"05:00 Thu Oct 8 23:00:00 2026 {values['executable']}"
    )[0]
    if "uid" in change:
        process = replace(old, uid=os.getuid() + 1)
    assert not guard.eligible(process, HELPER)


@pytest.mark.parametrize("change", ["start", "uid", "ppid", "path", "age", "gone"])
def test_recheck_blocks_changed_identity(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, old: guard.Process, change: str
) -> None:
    changed = {
        "start": replace(old, start="Fri Oct 09 00:00:00 2026"),
        "uid": replace(old, uid=os.getuid() + 1),
        "ppid": replace(old, ppid=2),
        "path": replace(old, executable=HELPER + "x"),
        "age": replace(old, age=1),
        "gone": old,
    }[change]
    young = replace(old, pid=43, age=1)
    monkeypatch.setattr(
        guard,
        "processes",
        Mock(side_effect=[[old] + [young] * 299, [] if change == "gone" else [changed], []]),
    )
    with guard.private_file(tmp_path / "events") as stream:
        guard.reclaim(tmp_path, HELPER, stream)
    assert '"event": "disappeared_or_changed"' in (tmp_path / "events").read_text()


@pytest.mark.parametrize("disappeared", [True, False])
def test_signal_failure(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, old: guard.Process, disappeared: bool
) -> None:
    young = replace(old, pid=43, age=1)
    monkeypatch.setattr(guard, "processes", Mock(side_effect=[[old] + [young] * 299, [old], []]))
    monkeypatch.setattr(
        os,
        "kill",
        Mock(side_effect=ProcessLookupError() if disappeared else PermissionError()),
    )
    with guard.private_file(tmp_path / "events") as stream:
        if disappeared:
            guard.reclaim(tmp_path, HELPER, stream)
        else:
            with pytest.raises(PermissionError):
                guard.reclaim(tmp_path, HELPER, stream)
    assert ("disappeared_before_signal" if disappeared else "signal_failed") in (
        tmp_path / "events"
    ).read_text()


def test_count_stays_high_bounded_wait(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, old: guard.Process
) -> None:
    monkeypatch.setattr(guard, "processes", Mock(return_value=[replace(old, age=1)] * 300))
    monkeypatch.setattr(time, "monotonic", Mock(side_effect=[0, 1, 29]))
    sleep = Mock()
    monkeypatch.setattr(time, "sleep", sleep)
    with (
        guard.private_file(tmp_path / "events") as stream,
        pytest.raises(ReplayError, match="could not"),
    ):
        guard.reclaim(tmp_path, HELPER, stream)
    sleep.assert_called_once_with(1)


@pytest.mark.parametrize(
    "code,stdout,stderr,pid,ok",
    [
        (0, "", "", 42, True),
        (0, "", "", None, False),
        (1, "", "", 42, True),
        (1, "", "", None, False),
        (0, "", "warning", 42, False),
        (0, "43 1 501 05:00 Thu Oct 8 23:00:00 2026 /x", "", 42, False),
    ],
)
def test_ps_contract(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    code: int,
    stdout: str,
    stderr: str,
    pid: int | None,
    ok: bool,
) -> None:
    runner = Mock(return_value=subprocess.CompletedProcess([], code, stdout, stderr))
    monkeypatch.setattr(subprocess, "run", runner)
    monkeypatch.setattr(time, "monotonic", lambda: 0)
    if ok:
        assert guard.processes(tmp_path, deadline=3, pid=pid) == []
    else:
        with pytest.raises(ReplayError):
            guard.processes(tmp_path, deadline=3, pid=pid)
    assert runner.call_args.kwargs["timeout"] == 2
    assert "args=" not in runner.call_args.args[0]
    assert runner.call_args.kwargs["env"]["TMPDIR"] == str(tmp_path)


def test_ps_deadline(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(time, "monotonic", lambda: 2)
    with pytest.raises(ReplayError, match="deadline"):
        guard.processes(tmp_path, deadline=1)


@pytest.mark.parametrize("pid", [None, 42])
def test_elapsed_reobservation_recovers(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    pid: int | None,
) -> None:
    runner = Mock(
        side_effect=[
            subprocess.CompletedProcess(
                [], 0, "42 1 501 00:60 Thu Oct 8 23:00:00 2026 /private-command", ""
            ),
            subprocess.CompletedProcess([], 0, "42 1 501 05:00 Thu Oct 8 23:00:00 2026 /x", ""),
        ]
    )
    sleep = Mock()
    monkeypatch.setattr(subprocess, "run", runner)
    monkeypatch.setattr(time, "sleep", sleep)
    monkeypatch.setattr(time, "monotonic", Mock(side_effect=[0, 0.05, 0.15]))
    assert guard.processes(tmp_path, deadline=1, pid=pid) == [
        guard.Process(42, 1, 501, 300, "Thu Oct 8 23:00:00 2026", "/x")
    ]
    assert runner.call_count == 2
    assert runner.call_args_list[0].kwargs["timeout"] == 1
    assert runner.call_args_list[1].kwargs["timeout"] == 0.85
    sleep.assert_called_once_with(0.1)
    assert capsys.readouterr().err == "Invalid ps elapsed time: '00:60'\n"


@pytest.mark.parametrize("recheck", [False, True])
def test_elapsed_exhaustion_never_signals(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    recheck: bool,
) -> None:
    valid = "\n".join(
        f"{pid} 1 {os.getuid()} 05:00 Thu Oct 8 23:00:00 2026 {HELPER}" for pid in range(42, 342)
    )
    invalid = subprocess.CompletedProcess(
        [], 0, "42 1 501 60:00 Thu Oct 8 23:00:00 2026 /private-command", ""
    )
    runner = Mock(
        side_effect=([subprocess.CompletedProcess([], 0, valid, "")] if recheck else [])
        + [invalid, invalid, invalid]
    )
    sleep = Mock()
    kill = Mock()
    monkeypatch.setattr(subprocess, "run", runner)
    monkeypatch.setattr(time, "monotonic", lambda: 0)
    monkeypatch.setattr(time, "sleep", sleep)
    monkeypatch.setattr(os, "kill", kill)
    with (
        guard.private_file(tmp_path / "events") as stream,
        pytest.raises(guard.ElapsedParseError, match="60:00"),
    ):
        guard.reclaim(tmp_path, HELPER, stream)
    assert runner.call_count == (4 if recheck else 3)
    assert [call.args for call in sleep.call_args_list] == [(0.1,), (0.2,)]
    kill.assert_not_called()
    assert capsys.readouterr().err == (
        "Invalid ps elapsed time: '60:00'\n"
        "Invalid ps elapsed time: '60:00'\n"
        "Invalid ps elapsed time: '60:00'\n"
    )


@pytest.mark.parametrize("clock", [[0, 0.95], [0, 0, 1]])
def test_elapsed_retry_uses_original_deadline(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, clock: list[float]
) -> None:
    error = guard.ElapsedParseError("00:60")
    runner = Mock(return_value=subprocess.CompletedProcess([], 0, "synthetic", ""))
    monkeypatch.setattr(subprocess, "run", runner)
    monkeypatch.setattr(guard, "parse_processes", Mock(side_effect=error))
    monkeypatch.setattr(time, "monotonic", Mock(side_effect=clock))
    monkeypatch.setattr(time, "sleep", Mock())
    with pytest.raises(guard.ElapsedParseError) as caught:
        guard.processes(tmp_path, deadline=1)
    assert caught.value is error
    runner.assert_called_once()


def test_elapsed_exhaustion_preserves_first_error(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    first = guard.ElapsedParseError("00:60")
    parse = Mock(
        side_effect=[first, guard.ElapsedParseError("x"), guard.ElapsedParseError("60:00")]
    )
    monkeypatch.setattr(guard, "parse_processes", parse)
    monkeypatch.setattr(
        subprocess, "run", Mock(return_value=subprocess.CompletedProcess([], 0, "synthetic", ""))
    )
    monkeypatch.setattr(time, "monotonic", lambda: 0)
    monkeypatch.setattr(time, "sleep", Mock())
    with pytest.raises(guard.ElapsedParseError) as caught:
        guard.processes(tmp_path, deadline=1)
    assert caught.value is first
    assert parse.call_count == 3


@pytest.mark.parametrize(
    "failure",
    [
        ReplayError("Malformed ps metadata"),
        ValueError("bad date"),
        OSError("unavailable"),
        subprocess.TimeoutExpired("ps", 1),
    ],
)
def test_unrelated_observation_error_not_retried(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, failure: Exception
) -> None:
    runner = Mock(return_value=subprocess.CompletedProcess([], 0, "synthetic", ""))
    parse = Mock(side_effect=[guard.ElapsedParseError("x"), failure])
    sleep = Mock()
    monkeypatch.setattr(subprocess, "run", runner)
    monkeypatch.setattr(guard, "parse_processes", parse)
    monkeypatch.setattr(time, "monotonic", lambda: 0)
    monkeypatch.setattr(time, "sleep", sleep)
    with pytest.raises(type(failure)) as caught:
        guard.processes(tmp_path, deadline=1)
    assert caught.value is failure
    assert runner.call_count == 2
    sleep.assert_called_once_with(0.1)


@pytest.mark.parametrize(
    "failure",
    [
        OSError("unavailable"),
        subprocess.TimeoutExpired("ps", 1),
        subprocess.CompletedProcess([], 2, "", "private diagnostic"),
    ],
)
def test_ps_failure_not_retried(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    failure: OSError | subprocess.TimeoutExpired | subprocess.CompletedProcess[str],
) -> None:
    runner = Mock(side_effect=[failure])
    sleep = Mock()
    monkeypatch.setattr(subprocess, "run", runner)
    monkeypatch.setattr(time, "monotonic", lambda: 0)
    monkeypatch.setattr(time, "sleep", sleep)
    with pytest.raises((OSError, subprocess.TimeoutExpired, ReplayError)):
        guard.processes(tmp_path, deadline=1)
    runner.assert_called_once()
    sleep.assert_not_called()


def test_elapsed_diagnostic_is_bounded_and_sanitized() -> None:
    with pytest.raises(guard.ElapsedParseError) as caught:
        guard.elapsed_seconds("\x1b\n/private-command" + "a" * 1000)
    assert str(caught.value) == "Invalid ps elapsed time: '" + "?" * 10 + "-" + "?" * 53 + "'"


def test_selected_exact_helper(tmp_path: Path) -> None:
    executable = tmp_path / "Selected.app/Contents/MacOS/Compressor"
    helper = tmp_path / "Selected.app/Contents" / guard.HELPER
    executable.parent.mkdir(parents=True)
    helper.parent.mkdir(parents=True)
    executable.touch()
    helper.touch()
    assert guard.selected_helper(executable) == str(helper)


def test_selected_rejects_wrong_executable(tmp_path: Path) -> None:
    executable = tmp_path / "Compressor"
    executable.touch()
    with pytest.raises(ReplayError):
        guard.selected_helper(executable)


def test_selected_rejects_linked_helper(tmp_path: Path) -> None:
    executable = tmp_path / "Selected.app/Contents/MacOS/Compressor"
    helper = tmp_path / "Selected.app/Contents" / guard.HELPER
    executable.parent.mkdir(parents=True)
    helper.parent.mkdir(parents=True)
    executable.touch()
    helper.symlink_to(executable)
    with pytest.raises(ReplayError):
        guard.selected_helper(executable)


def test_directory_containment(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    root = tmp_path / "root"
    root.mkdir()
    monkeypatch.chdir(root)
    assert guard.guard_directory(root / "guard") == root / "guard"
    (root / "escape").symlink_to(tmp_path, target_is_directory=True)
    with pytest.raises(ReplayError):
        guard.guard_directory(root / "escape/new")
    with pytest.raises(ReplayError):
        guard.guard_directory(root)
    (root / "public").mkdir(mode=0o755)
    with pytest.raises(ReplayError):
        guard.guard_directory(root / "public")


@pytest.mark.parametrize("kind", ["public", "link", "hardlink"])
def test_private_file_rejects_unsafe(tmp_path: Path, kind: str) -> None:
    path = tmp_path / "events"
    if kind == "public":
        path.touch(mode=0o644)
    else:
        other = tmp_path / "other"
        other.touch(mode=0o600)
        if kind == "link":
            path.symlink_to(other)
        else:
            path.hardlink_to(other)
    with pytest.raises((ReplayError, OSError)), guard.private_file(path):
        pytest.fail("unsafe file accepted")


def test_log_size_fails_closed(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(guard, "MAX_LOG_BYTES", 1)
    with (
        guard.private_file(tmp_path / "events") as stream,
        pytest.raises(ReplayError, match="limit"),
    ):
        guard.evidence(stream, "test")
    assert (tmp_path / "events").read_bytes() == b""


@pytest.mark.parametrize("blocked", [False, True])
def test_check_serializes_and_records_errors(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, blocked: bool
) -> None:
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(guard, "selected_helper", lambda _path: HELPER)
    reclaim = Mock(side_effect=ReplayError("unsafe") if blocked else None)
    monkeypatch.setattr(guard, "reclaim", reclaim)
    flock = Mock(side_effect=[BlockingIOError(), None])
    monkeypatch.setattr(fcntl, "flock", flock)
    sleep = Mock()
    monkeypatch.setattr(time, "sleep", sleep)
    if blocked:
        with pytest.raises(ReplayError, match="unsafe"):
            guard.check(tmp_path / "guard", Path("Compressor"))
        assert '"event": "blocked"' in (tmp_path / "guard/guard.jsonl").read_text()
    else:
        guard.check(tmp_path / "guard", Path("Compressor"))
    assert flock.call_count == 2
    sleep.assert_called_once_with(0.25)
    reclaim.assert_called_once()


def test_lock_timeout_never_observes(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(guard, "selected_helper", lambda _path: HELPER)
    monkeypatch.setattr(fcntl, "flock", Mock(side_effect=BlockingIOError()))
    monkeypatch.setattr(time, "monotonic", Mock(side_effect=[0, 35]))
    with pytest.raises(ReplayError, match="lock wait"):
        guard.check(tmp_path / "guard", Path("Compressor"))
    assert not (tmp_path / "guard/guard.jsonl").exists()


def test_real_flock_serializes_cooperating_handles(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.chdir(tmp_path)
    directory = guard.guard_directory(tmp_path / "guard")
    monkeypatch.setattr(guard, "selected_helper", lambda _path: HELPER)
    events: list[str] = []

    def observe(path: Path, helper: str, stream: object) -> None:
        events.append("second_entered")

    monkeypatch.setattr(guard, "reclaim", observe)
    with guard.private_file(directory / "guard.lock") as first:
        fcntl.flock(first.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)

        def release(seconds: float) -> None:
            events.append("first_released")
            fcntl.flock(first.fileno(), fcntl.LOCK_UN)

        monkeypatch.setattr(time, "sleep", release)
        guard.check(directory, Path("Compressor"))
    assert events == ["first_released", "second_entered"]
