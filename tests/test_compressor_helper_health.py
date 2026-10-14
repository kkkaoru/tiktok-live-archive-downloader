from __future__ import annotations

import json
import subprocess
from pathlib import Path
from unittest.mock import Mock

import pytest

from replay import compressor_helper_health as health

CLI = "/Applications/Selected.app/Contents/MacOS/Compressor"
HELPER = "/Applications/Selected.app/Contents/Helpers/RegisterProExtension"
SCOPE = health.Scope(CLI, (HELPER,))
CLI_ROW = f"10 1 501 00:03 0.2 Fri Oct 9 12:00:00 2026 {CLI}\n"
HELPER_ROW = f"20 1 501 00:01 - Fri Oct 9 12:00:02 2026 {HELPER}\n"


def test_parse_scope_and_arguments(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    fake = Mock(
        side_effect=[
            CLI_ROW
            + HELPER_ROW
            + HELPER_ROW.replace("20 1", "21 1").replace("Selected.app", "Other.app")
            + HELPER_ROW.replace("20 1", "22 1").replace(
                "RegisterProExtension", "RegisterProExtension-copy"
            )
            + CLI_ROW.replace("10 1", "11 1").replace("Selected.app", "Selected.app.copy"),
            CLI_ROW,
            CLI + " -monitor",
            CLI_ROW,
        ]
    )
    monkeypatch.setattr(health, "ps", fake)
    rows, calls = health.sample(SCOPE, deadline=100, scratch=tmp_path)
    assert len(rows) == 2
    assert rows[1].cpu is None
    assert rows[0].start == "Fri Oct 9 12:00:00 2026"
    assert calls[0]["kind"] == "monitor"
    assert fake.call_args_list[0].args == (["-axo", health.FIELDS],)
    assert fake.call_args_list[2].args == (["-p", "10", "-o", "args="],)
    assert fake.call_count == 4


@pytest.mark.parametrize(
    "arguments, expected",
    [
        ("Compressor -monitor", "monitor"),
        ("Compressor -jobpath x", "submission"),
        ("Compressor -monitor -jobpath x", "ambiguous"),
        ("Compressor -help", "other"),
        ("Compressor /x/-monitor", "other"),
    ],
)
def test_categories(arguments: str, expected: str) -> None:
    assert health.call_kind(arguments) == expected


@pytest.mark.parametrize("response", ["", CLI_ROW.replace("12:00:00", "12:00:01")])
def test_disappeared_before_arguments(
    response: str, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    fake = Mock(side_effect=[CLI_ROW, response])
    monkeypatch.setattr(health, "ps", fake)
    _, calls = health.sample(SCOPE, deadline=100, scratch=tmp_path)
    assert calls[0]["kind"] == "disappeared_or_changed"
    assert fake.call_count == 2


def test_disappeared_after_arguments(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.setattr(health, "ps", Mock(side_effect=[CLI_ROW, CLI_ROW, "-monitor", ""]))
    _, calls = health.sample(SCOPE, deadline=100, scratch=tmp_path)
    assert calls[0]["kind"] == "disappeared_or_changed"


def test_simultaneous_calls_are_not_ownership() -> None:
    rows = health.parse_processes(CLI_ROW + CLI_ROW.replace("10 1", "11 1") + HELPER_ROW)
    events = health.helper_events(rows, scope=SCOPE, seen=set(), baseline=False)
    assert events[0]["observed_parent"] is None
    assert events[0]["confidence"] == "temporal_only_or_unknown"
    assert events[0]["causal_attribution"] is False
    identities = events[0]["concurrent_cli_identities"]
    assert isinstance(identities, list)
    assert len(identities) == 2


def test_parent_link_baseline_dedup_and_reused_pid() -> None:
    seen: set[tuple[int, int, str, str]] = set()
    rows = health.parse_processes(CLI_ROW + HELPER_ROW.replace("20 1", "20 10"))
    events = health.helper_events(rows, scope=SCOPE, seen=seen, baseline=True)
    assert events[0]["status"] == "baseline"
    assert events[0]["confidence"] == "snapshot_parent_link"
    assert events[0]["causal_attribution"] is False
    assert health.helper_events(rows, scope=SCOPE, seen=seen, baseline=False) == []
    reused = health.parse_processes(HELPER_ROW.replace("12:00:02", "12:00:03"))
    assert (
        health.helper_events(reused, scope=SCOPE, seen=seen, baseline=False)[0]["status"]
        == "first_seen"
    )


@pytest.mark.parametrize(
    "text", ["bad", CLI_ROW.replace("0.2", "nan"), CLI_ROW.replace("10 1", "0 1")]
)
def test_malformed_rows(text: str) -> None:
    with pytest.raises(ValueError):
        health.parse_processes(text)
    assert health.parse_processes("\n") == []


def test_signed_system_uid_does_not_break_selected_observation(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    # Darwin ps represents nobody's UID as -2; it is unrelated, not malformed.
    system = "646 1 -2 20:16 0.0 Fri Oct 9 14:37:05 2026 /usr/libexec/dhcp6d\n"
    assert health.parse_processes(system)[0].uid == -2
    fake = Mock(return_value=system + HELPER_ROW)
    monkeypatch.setattr(health, "ps", fake)
    rows, calls = health.sample(SCOPE, deadline=100, scratch=tmp_path)
    assert len(rows) == 1
    assert calls == []
    fake.assert_called_once()


def test_scope_derivation(tmp_path: Path) -> None:
    executable = tmp_path / "Selected.app/Contents/MacOS/Compressor"
    executable.parent.mkdir(parents=True)
    executable.touch()
    helper = executable.parents[1] / "Helpers/RegisterProExtension"
    helper.parent.mkdir()
    helper.touch()
    outside = tmp_path / "RegisterProExtension"
    outside.touch()
    (executable.parent / "RegisterProExtension").symlink_to(outside)
    scope = health.selected_scope(executable)
    assert scope.executable == str(executable)
    assert scope.helpers == (str(helper),)
    with pytest.raises(ValueError, match="Select an app"):
        health.selected_scope(outside)


def test_missing_helper(tmp_path: Path) -> None:
    executable = tmp_path / "Selected.app/Contents/MacOS/Compressor"
    executable.parent.mkdir(parents=True)
    executable.touch()
    with pytest.raises(ValueError, match="no RegisterProExtension"):
        health.selected_scope(executable)


@pytest.mark.parametrize(
    "code, stdout, stderr, selector, expected",
    [
        (0, CLI_ROW, "", ["-axo", health.FIELDS], CLI_ROW),
        (1, "", "", ["-p", "10", "-o", "args="], ""),
    ],
)
def test_ps_command(
    code: int,
    stdout: str,
    stderr: str,
    selector: list[str],
    expected: str,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    runner = Mock(return_value=subprocess.CompletedProcess([], code, stdout, stderr))
    monkeypatch.setattr("replay.compressor_helper_health.subprocess.run", runner)
    monkeypatch.setattr("replay.compressor_helper_health.time.monotonic", lambda: 5.0)
    assert health.ps(selector, deadline=6.0, scratch=tmp_path) == expected
    assert runner.call_args.kwargs["timeout"] == 1.0
    assert runner.call_args.kwargs["env"]["TMPDIR"] == str(tmp_path)
    assert runner.call_args.args[0][:2] == ["/bin/ps", "-ww"]


@pytest.mark.parametrize("code, stderr", [(1, ""), (0, "permission denied"), (2, "bad option")])
def test_ps_errors(code: int, stderr: str, monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.setattr(
        "replay.compressor_helper_health.subprocess.run",
        Mock(return_value=subprocess.CompletedProcess([], code, "", stderr)),
    )
    monkeypatch.setattr("replay.compressor_helper_health.time.monotonic", lambda: 5.0)
    with pytest.raises(RuntimeError, match="ps failed"):
        health.ps(["-axo", health.FIELDS], deadline=6, scratch=tmp_path)


def test_timeout(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    runner = Mock(side_effect=subprocess.TimeoutExpired("ps", 2))
    monkeypatch.setattr("replay.compressor_helper_health.subprocess.run", runner)
    monkeypatch.setattr("replay.compressor_helper_health.time.monotonic", lambda: 5.0)
    with pytest.raises(subprocess.TimeoutExpired):
        health.ps(["-axo", health.FIELDS], deadline=10, scratch=tmp_path)
    with pytest.raises(TimeoutError):
        health.ps(["-axo", health.FIELDS], deadline=4, scratch=tmp_path)
    assert runner.call_count == 1


def test_observe_exclusive_and_cli(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    clock = Mock(side_effect=[0.0, 0.0, 0.0, 1.0, 1.0])
    monkeypatch.setattr("replay.compressor_helper_health.time.monotonic", clock)
    monkeypatch.setattr("replay.compressor_helper_health.time.sleep", Mock())
    monkeypatch.setattr(health, "selected_scope", lambda path: SCOPE)
    monkeypatch.setattr(
        health, "sample", Mock(return_value=(health.parse_processes(CLI_ROW + HELPER_ROW), []))
    )
    output = tmp_path / "evidence.json"
    monkeypatch.setattr(
        "sys.argv", ["health", "--executable", CLI, "--output", str(output), "--duration", "1"]
    )
    assert health.main() == 0
    data = json.loads(output.read_text())
    assert data["complete"] is True
    assert len(data["samples"]) == 1
    assert output.stat().st_mode & 0o777 == 0o600
    with pytest.raises(FileExistsError):
        health.observe(Path(CLI), output, duration=1, interval=1)


def test_incomplete_evidence(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.setattr("replay.compressor_helper_health.time.monotonic", lambda: 0.0)
    monkeypatch.setattr(health, "selected_scope", lambda path: SCOPE)
    monkeypatch.setattr(health, "sample", Mock(side_effect=subprocess.TimeoutExpired("ps", 2)))
    output = tmp_path / "failed.json"
    assert health.observe(Path(CLI), output, duration=1, interval=1) == 1
    data = json.loads(output.read_text())
    assert data["complete"] is False
    assert data["error"] == "TimeoutExpired: Command 'ps' timed out after 2 seconds"


@pytest.mark.parametrize(
    "duration, interval", [(0, 1), (121, 1), (1, 0.1), (1, 11), (float("nan"), 1)]
)
def test_invalid_bounds(duration: float, interval: float, tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="Require duration"):
        health.observe(Path(CLI), tmp_path / "no.json", duration=duration, interval=interval)


def test_output_symlink_refused(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.setattr(health, "selected_scope", lambda path: SCOPE)
    target = tmp_path / "existing.json"
    target.write_text("protected", encoding="utf-8")
    output = tmp_path / "linked.json"
    output.symlink_to(target)
    with pytest.raises(FileExistsError):
        health.observe(Path(CLI), output, duration=1, interval=1)
    assert target.read_text(encoding="utf-8") == "protected"


def test_parent_escape_refused(tmp_path: Path) -> None:
    parent = tmp_path / "escape"
    parent.symlink_to(Path.cwd(), target_is_directory=True)
    with pytest.raises(ValueError, match="Evidence must"):
        health.observe(Path(CLI), parent / "no.json", duration=1, interval=1)


def test_outside_work_rejected() -> None:
    with pytest.raises(ValueError, match="Evidence must"):
        health.observe(Path(CLI), Path.cwd() / "no.json", duration=1, interval=1)
