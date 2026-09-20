"""Trusted manifest parsing and headless CLI behavior with synthetic artifacts."""

import json
import subprocess
import sys
from pathlib import Path

import pytest

from replay import workflow
from replay.stages import StageError


@pytest.fixture
def entry(tmp_path: Path) -> dict[str, object]:
    source = tmp_path / "input"
    source.write_bytes(b"source")
    return {
        "name": "create",
        "command": [
            sys.executable,
            "-c",
            "import sys; from pathlib import Path; Path(sys.argv[1]).write_text('result')",
            str(tmp_path / "output"),
        ],
        "inputs": [str(source)],
        "outputs": [str(tmp_path / "output")],
        "timeout_seconds": 10,
    }


def test_run_whole_plan_and_idempotent_resume(entry: dict[str, object], tmp_path: Path) -> None:
    plan = tmp_path / "plan.json"
    plan.write_text(json.dumps({"version": 1, "stages": [entry]}), encoding="utf-8")
    workflow.run_plan(plan, state_directory=tmp_path / "state")
    assert (tmp_path / "output").read_text(encoding="utf-8") == "result"
    workflow.run_plan(plan, state_directory=tmp_path / "state")
    assert (tmp_path / "output").read_text(encoding="utf-8") == "result"


def test_cli_success(entry: dict[str, object], tmp_path: Path) -> None:
    plan = tmp_path / "plan.json"
    plan.write_text(json.dumps({"version": 1, "stages": [entry]}), encoding="utf-8")
    assert workflow.main([str(plan), "--state", str(tmp_path / "state")]) == 0


def test_cli_failure_does_not_print_configuration(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    plan = tmp_path / "plan.json"
    plan.write_text('{"private":"never-log-this"}', encoding="utf-8")
    assert workflow.main([str(plan), "--state", str(tmp_path / "state")]) == 1
    captured = capsys.readouterr()
    assert "StageError" in captured.err
    assert "never-log-this" not in captured.err


def test_timeout_arguments_are_not_logged(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    def fail(_plan: Path, *, state_directory: Path) -> None:
        raise subprocess.TimeoutExpired(("command", "never-log-this"), 1)

    monkeypatch.setattr(workflow, "run_plan", fail)
    assert workflow.main([str(tmp_path / "plan.json"), "--state", str(tmp_path / "state")]) == 1
    captured = capsys.readouterr()
    assert "deadline exceeded" in captured.err
    assert "never-log-this" not in captured.err


def test_busy_stage_stops_without_duplicate(
    entry: dict[str, object], tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    plan = tmp_path / "plan.json"
    plan.write_text(json.dumps({"version": 1, "stages": [entry]}), encoding="utf-8")
    monkeypatch.setattr(workflow, "run_stage", lambda *_args, **_kwargs: "busy")
    with pytest.raises(StageError, match="Another process"):
        workflow.run_plan(plan, state_directory=tmp_path / "state")


def test_plan_symlink_is_refused(tmp_path: Path) -> None:
    original = tmp_path / "original"
    original.write_text("{}", encoding="utf-8")
    alias = tmp_path / "alias"
    alias.symlink_to(original)
    with pytest.raises(StageError, match="manifest file"):
        workflow.run_plan(alias, state_directory=tmp_path / "state")


def test_oversized_plan_is_refused(tmp_path: Path) -> None:
    plan = tmp_path / "plan.json"
    plan.write_bytes(b"x" * 1_048_577)
    with pytest.raises(StageError, match="manifest file"):
        workflow.run_plan(plan, state_directory=tmp_path / "state")


@pytest.mark.parametrize("value", [None, [], {}, {"version": 1, "stages": [], "extra": 1}])
def test_invalid_manifest_fields(value: object) -> None:
    with pytest.raises(StageError, match="manifest fields"):
        workflow.parse_stages(value)


@pytest.mark.parametrize("version", [True, 0, 2, "1"])
def test_invalid_version(version: object) -> None:
    with pytest.raises(StageError, match="version"):
        workflow.parse_stages({"version": version, "stages": []})


@pytest.mark.parametrize("stages", [None, [], [{}] * 513])
def test_invalid_stage_count(stages: object) -> None:
    with pytest.raises(StageError, match="stage count"):
        workflow.parse_stages({"version": 1, "stages": stages})


def test_duplicate_names_are_refused(entry: dict[str, object]) -> None:
    with pytest.raises(StageError, match="unique"):
        workflow.parse_stages({"version": 1, "stages": [entry, entry]})


@pytest.mark.parametrize("item", [None, {"unknown": 1}])
def test_invalid_stage_fields(item: object) -> None:
    with pytest.raises(StageError, match="stage fields"):
        workflow.parse_stages({"version": 1, "stages": [item]})


@pytest.mark.parametrize(
    "changes",
    [{"name": 1}, {"timeout_seconds": True}, {"adopt_existing": 1}],
)
def test_invalid_scalar_fields(entry: dict[str, object], changes: dict[str, object]) -> None:
    with pytest.raises(StageError, match="name, timeout or adoption"):
        workflow.parse_stages({"version": 1, "stages": [entry | changes]})


@pytest.mark.parametrize("command", [None, "command", ["x"] * 257])
def test_invalid_argument_array(entry: dict[str, object], command: object) -> None:
    with pytest.raises(StageError, match="bounded"):
        workflow.parse_stages({"version": 1, "stages": [entry | {"command": command}]})


@pytest.mark.parametrize("command", [[1], [""], ["bad\0value"]])
def test_invalid_argument_text(entry: dict[str, object], command: object) -> None:
    with pytest.raises(StageError, match="nonempty text"):
        workflow.parse_stages({"version": 1, "stages": [entry | {"command": command}]})


def test_optional_verification_policy_is_retained(entry: dict[str, object]) -> None:
    stages = workflow.parse_stages(
        {"version": 1, "stages": [entry | {"verify_command": ["check"], "adopt_existing": True}]}
    )
    assert stages[0].verify_command == ("check",)
    assert stages[0].adopt_existing is True
