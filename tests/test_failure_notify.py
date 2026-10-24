"""Failure notices: one record line and one local notification, never an exception."""

import json
import subprocess
from pathlib import Path

from replay import failure_notify


def test_a_notice_records_the_failure_and_shows_a_notification(tmp_path: Path) -> None:
    calls: list[list[str]] = []
    record = tmp_path / "failures.jsonl"
    failure_notify.notify(
        "7694226728939571989", "cleanup", cls="deterministic", cause='Audit "failed"',
        action="needs a person", record=record, run=lambda command, **_k: calls.append(command),
    )  # fmt: skip
    row = json.loads(record.read_text("utf-8"))
    assert {key: row[key] for key in ("recording", "step", "class", "cause", "action")} == {
        "recording": "7694226728939571989",
        "step": "cleanup",
        "class": "deterministic",
        "cause": 'Audit "failed"',
        "action": "needs a person",
    }
    assert calls == [[
        "/usr/bin/osascript", "-e",
        'display notification "cleanup: Audit \\"failed\\"" with title "Recording pipeline" '
        'subtitle "7694226728939571989 — deterministic, needs a person"',
    ]]  # fmt: skip


def test_a_failed_notification_is_reported_and_ignored(tmp_path: Path, capsys: object) -> None:
    def broken(*_args: object, **_kwargs: object) -> object:
        raise subprocess.TimeoutExpired("osascript", 10)

    failure_notify.notify(
        "1", "asr", cls="unknown", cause="x", action="needs a person",
        record=tmp_path / "failures.jsonl", run=broken,
    )  # fmt: skip
    assert (tmp_path / "failures.jsonl").exists()


def test_the_record_lives_beside_the_recording_pipelines() -> None:
    assert failure_notify.record_path().parts[-3:] == (
        "work",
        "recording-pipeline",
        "failures.jsonl",
    )
