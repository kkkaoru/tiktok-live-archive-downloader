"""A failing workflow leaves a private, redacted record and never logs plan contents."""

import json
import re
import subprocess
import sys
from pathlib import Path

import pytest

from replay import workflow, workflow_diagnostics
from replay.core import ReplayError
from replay.stages import StageError


def _plan(tmp_path: Path, command: list[str]) -> Path:
    source = tmp_path / "input"
    source.write_bytes(b"source")
    plan = tmp_path / "plan.json"
    stage = {
        "name": "create",
        "command": command,
        "inputs": [str(source)],
        "outputs": [str(tmp_path / "output")],
        "timeout_seconds": 10,
    }
    plan.write_text(json.dumps({"version": 1, "stages": [stage]}), encoding="utf-8")
    return plan


def test_success_leaves_no_failure_record(tmp_path: Path) -> None:
    plan = _plan(
        tmp_path,
        [
            sys.executable,
            "-c",
            "import sys; open(sys.argv[1], 'w').write('x')",
            str(tmp_path / "output"),
        ],
    )
    assert workflow_diagnostics.main([str(plan), "--state", str(tmp_path / "state")]) == 0
    assert not (tmp_path / "state" / "last-failure.json").exists()


def test_a_failed_stage_writes_class_message_and_location(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    plan = _plan(tmp_path, [sys.executable, "-c", "raise SystemExit(3)"])
    state = tmp_path / "state"
    assert workflow_diagnostics.main([str(plan), "--state", str(state), "--target", "create"]) == 1
    record = json.loads((state / "last-failure.json").read_text(encoding="utf-8"))
    assert record["plan"] == "plan.json"
    assert record["targets"] == ["create"]
    assert record["chain"][0]["type"] == "StageError"
    assert re.fullmatch(r"stages\.py:\w+:\d+", record["chain"][0]["where"])
    assert record["chain"][0]["message"].startswith("Stage command failed; inspect ")
    assert "Workflow stopped (StageError" in capsys.readouterr().err
    assert oct((state / "last-failure.json").stat().st_mode & 0o777) == "0o600"


def test_messages_are_redacted_and_truncated(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    def fail(_plan: Path, *, state_directory: Path, targets: tuple[str, ...]) -> None:
        raise ReplayError("login failed sessionid=abc123SECRET " + "x" * 900)

    monkeypatch.setattr(workflow, "run_plan", fail)
    with pytest.raises(ReplayError):
        fail(tmp_path, state_directory=tmp_path, targets=())
    state = tmp_path / "state"
    path = workflow_diagnostics.record(
        state,
        ReplayError("login failed sessionid=abc123SECRET " + "x" * 900),
        plan=tmp_path / "p.json",
        targets=(),
    )
    message = json.loads(path.read_text(encoding="utf-8"))["chain"][0]["message"]
    assert "abc123SECRET" not in message
    assert "sessionid=[redacted]" in message
    assert len(message) == 500


def test_other_exceptions_record_class_and_errno_but_no_text(tmp_path: Path) -> None:
    path = workflow_diagnostics.record(
        tmp_path / "state", ValueError("never-log-this"), plan=tmp_path / "p.json", targets=()
    )
    link = json.loads(path.read_text(encoding="utf-8"))["chain"][0]
    assert link == {"type": "ValueError", "message": "", "where": ""}
    error = OSError(2, "No such file or directory", "/secret/path")
    assert workflow_diagnostics.chain(error)[0]["message"] == "errno 2: No such file or directory"


def test_causes_are_followed_up_to_the_limit(tmp_path: Path) -> None:
    try:
        try:
            raise OSError(13, "Permission denied")
        except OSError as inner:
            raise StageError("Cannot reuse artifacts") from inner
    except StageError as outer:
        links = workflow_diagnostics.chain(outer)
    assert [link["type"] for link in links] == ["StageError", "PermissionError"]
    assert links[1]["message"] == "errno 13: Permission denied"


def test_a_timeout_never_records_its_arguments(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    def fail(_plan: Path, *, state_directory: Path, targets: tuple[str, ...]) -> None:
        raise subprocess.TimeoutExpired(("command", "never-log-this"), 1)

    monkeypatch.setattr(workflow, "run_plan", fail)
    state = tmp_path / "state"
    assert workflow_diagnostics.main([str(tmp_path / "plan.json"), "--state", str(state)]) == 1
    text = (state / "last-failure.json").read_text(encoding="utf-8")
    assert "never-log-this" not in text
    assert "deadline exceeded" in capsys.readouterr().err


def test_check_delegates_to_the_standard_workflow_cli(tmp_path: Path) -> None:
    plan = _plan(tmp_path, [sys.executable, "-c", "pass"])
    assert (
        workflow_diagnostics.main([str(plan), "--state", str(tmp_path / "state"), "--check"]) == 0
    )
