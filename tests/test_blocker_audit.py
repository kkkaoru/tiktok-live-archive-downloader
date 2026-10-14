"""Read-only blocker triage cannot turn old task notes into execution permission."""

import io
import json
import os
from pathlib import Path

import pytest

from replay import blocker_audit as module


@pytest.fixture
def blocker() -> dict[str, str]:
    return {
        "kind": "restriction",
        "authority": "agent",
        "scope": "other-task",
        "source": "synthetic original delegation entry: prepare conditions and return to main",
        "source_origin": "agent-note",
        "recheck_when": "Helper returns its bounded observation and resource plan to main",
        "resolution": "external",
        "next_action": "Review the original instruction and scope",
        "attempts": "Retained task/handoff inspected; original authority not established",
    }


@pytest.fixture
def stopped(blocker: dict[str, str]) -> dict[str, object]:
    return {
        "name": "encode",
        "needs": [],
        "state": "blocked",
        "evidence": "handoff",
        "blocker": blocker,
    }


def test_old_helper_restriction_cannot_stop_parent_goal(stopped: dict[str, object]) -> None:
    result = module.audit(
        {"version": 1, "task_reference": "current user request", "tasks": [stopped]}
    )
    assert result == {
        "decision": "continue",
        "candidate_tasks": [],
        "review_restriction_sources": ["encode"],
        "investigate_locally": [],
        "observe_existing": [],
        "deferred": [],
        "may_report_blocked": False,
        "execution_authorized": False,
    }


def test_agent_note_with_current_scope_still_needs_authority(
    stopped: dict[str, object], blocker: dict[str, str]
) -> None:
    blocker["scope"] = "current-task"
    result = module.audit({"version": 1, "task_reference": "current", "tasks": [stopped]})
    assert result["review_restriction_sources"] == ["encode"]
    assert result["may_report_blocked"] is False


def test_independent_work_survives_genuine_branch_blocker(
    stopped: dict[str, object], blocker: dict[str, str]
) -> None:
    blocker.update(authority="developer", scope="current-task", source_origin="original")
    result = module.audit(
        {
            "version": 1,
            "task_reference": "current",
            "tasks": [
                stopped,
                {
                    "name": "audio",
                    "needs": [],
                    "state": "pending",
                    "evidence": "authorized independent work",
                },
                {
                    "name": "delivery",
                    "needs": ["encode", "audio"],
                    "state": "pending",
                    "evidence": "not delivered",
                },
            ],
        }
    )
    assert result["candidate_tasks"] == ["audio"]
    assert result["deferred"] == ["encode", "delivery"]
    assert result["may_report_blocked"] is False


def test_unknown_setting_is_local_investigation_not_human_wait(
    stopped: dict[str, object], blocker: dict[str, str]
) -> None:
    blocker.update(
        kind="prerequisite", authority="observation", scope="current-task", resolution="local"
    )
    result = module.audit({"version": 1, "task_reference": "current", "tasks": [stopped]})
    assert result["investigate_locally"] == ["encode"]
    assert result["may_report_blocked"] is False
    assert result["execution_authorized"] is False


def test_genuine_external_blocker_can_be_reported_without_bypass(
    stopped: dict[str, object], blocker: dict[str, str]
) -> None:
    blocker.update(authority="user", scope="current-task", source_origin="original")
    result = module.audit({"version": 1, "task_reference": "current", "tasks": [stopped]})
    assert result["decision"] == "blocked"
    assert result["may_report_blocked"] is True
    assert result["execution_authorized"] is False


@pytest.mark.parametrize(
    "state,decision", [("running", "observe-existing"), ("complete", "verify-completion")]
)
def test_existing_job_or_completion_is_not_blocked(state: str, decision: str) -> None:
    result = module.audit(
        {
            "version": 1,
            "task_reference": "current",
            "tasks": [
                {
                    "name": "encode",
                    "needs": [],
                    "state": state,
                    "evidence": "job identity/receipt reference",
                },
            ],
        }
    )
    assert result["decision"] == decision
    assert result["may_report_blocked"] is False
    assert result["execution_authorized"] is False


def test_complete_dependency_allows_candidate_not_automatic_execution() -> None:
    result = module.audit(
        {
            "version": 1,
            "task_reference": "current",
            "tasks": [
                {
                    "name": "prepare",
                    "needs": [],
                    "state": "complete",
                    "evidence": "verified receipt",
                },
                {
                    "name": "encode",
                    "needs": ["prepare"],
                    "state": "pending",
                    "evidence": "not started",
                },
            ],
        }
    )
    assert result["candidate_tasks"] == ["encode"]
    assert result["execution_authorized"] is False


@pytest.mark.parametrize(
    "value",
    [
        None,
        {},
        {"version": True, "task_reference": "current", "tasks": []},
        {"version": 1, "task_reference": "current", "tasks": []},
        {"version": 1, "task_reference": "current", "tasks": [{}] * 513},
        {"version": 1, "task_reference": "", "tasks": []},
    ],
)
def test_invalid_snapshot(value: object) -> None:
    with pytest.raises(module.AuditError):
        module.audit(value)


@pytest.mark.parametrize(
    "change",
    [
        {"name": "../escape"},
        {"evidence": ""},
        {"evidence": "x" * 4097},
        {"evidence": "bad\0text"},
        {"evidence": 1},
        {"state": "guessed"},
        {"needs": "encode"},
        {"needs": ["missing"]},
        {"needs": ["encode", "encode"]},
        {"needs": ["encode"]},
        {"needs": ["encode"] * 513},
        {"command": ["never execute"]},
        {"state": "pending"},
        {"blocker": {}},
        {"blocker": None},
    ],
)
def test_invalid_task(stopped: dict[str, object], change: dict[str, object]) -> None:
    with pytest.raises(module.AuditError):
        module.audit({"version": 1, "task_reference": "current", "tasks": [stopped | change]})


def test_incomplete_task_and_duplicate_names(stopped: dict[str, object]) -> None:
    with pytest.raises(module.AuditError):
        module.audit({"version": 1, "task_reference": "current", "tasks": [{"name": "x"}]})
    with pytest.raises(module.AuditError):
        module.audit({"version": 1, "task_reference": "current", "tasks": [stopped, stopped]})


@pytest.mark.parametrize("state", ["complete", "running"])
def test_inconsistent_dependency_evidence_is_rejected(state: str) -> None:
    with pytest.raises(module.AuditError, match="incomplete prerequisites"):
        module.audit(
            {
                "version": 1,
                "task_reference": "current",
                "tasks": [
                    {"name": "prepare", "needs": [], "state": "pending", "evidence": "not started"},
                    {
                        "name": "encode",
                        "needs": ["prepare"],
                        "state": state,
                        "evidence": "contradictory",
                    },
                ],
            }
        )


def test_cli_blocked_gate_is_read_only_and_hides_evidence(
    tmp_path: Path,
    stopped: dict[str, object],
    capsys: pytest.CaptureFixture[str],
    blocker: dict[str, str],
) -> None:
    path = tmp_path / "audit.json"
    blocker["source"] = "private-never-print-this"
    path.write_text(
        json.dumps({"version": 1, "task_reference": "current", "tasks": [stopped]}),
        encoding="utf-8",
    )
    before = path.read_bytes()
    assert module.main([str(path), "--require-blocked"]) == 1
    assert "private-never-print-this" not in capsys.readouterr().out
    assert path.read_bytes() == before
    assert sorted(p.name for p in tmp_path.iterdir()) == ["audit.json"]
    assert module.main([str(path)]) == 0


@pytest.mark.parametrize(
    "kind", ["missing", "directory", "fifo", "symlink", "oversize", "malformed"]
)
def test_bad_input_cannot_hang_or_establish_blocked(tmp_path: Path, kind: str) -> None:
    path = tmp_path / "audit.json"
    if kind == "directory":
        path.mkdir()
    elif kind == "fifo":
        os.mkfifo(path)
    elif kind == "symlink":
        path.symlink_to(tmp_path / "missing")
    elif kind == "oversize":
        path.write_bytes(b"x" * 1_048_577)
    elif kind == "malformed":
        path.write_bytes(b"invalid JSON")
    assert module.main([str(path), "--require-blocked"]) == 2


def test_actual_read_is_bounded(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    path = tmp_path / "audit.json"
    path.write_text("{}", encoding="utf-8")
    monkeypatch.setattr(module, "MAX_BYTES", 4)
    monkeypatch.setattr(os, "fdopen", lambda *_args, **_kwargs: io.BytesIO(b"123456"))
    assert module.main([str(path)]) == 2


@pytest.mark.parametrize("origin", ["agent-note", "unverified"])
def test_self_authored_wakeup_is_not_original_user_prohibition(
    stopped: dict[str, object], blocker: dict[str, str], origin: str
) -> None:
    blocker.update(authority="user", scope="current-task", source_origin=origin)
    result = module.audit(
        {"version": 1, "task_reference": "original delivery request", "tasks": [stopped]}
    )
    assert result["review_restriction_sources"] == ["encode"]
    assert result["candidate_tasks"] == []
    assert result["decision"] == "continue"
    assert result["execution_authorized"] is False


@pytest.mark.parametrize(
    "field", ["source", "source_origin", "next_action", "attempts", "recheck_when"]
)
def test_missing_provenance_or_next_check_cannot_establish_blocked(
    stopped: dict[str, object], blocker: dict[str, str], field: str
) -> None:
    del blocker[field]
    with pytest.raises(module.AuditError):
        module.audit({"version": 1, "task_reference": "current", "tasks": [stopped]})


@pytest.mark.parametrize("field", ["authority", "source_origin", "scope", "resolution", "kind"])
def test_unknown_classification_is_rejected(
    stopped: dict[str, object], blocker: dict[str, str], field: str
) -> None:
    blocker[field] = "invented"
    with pytest.raises(module.AuditError):
        module.audit({"version": 1, "task_reference": "current", "tasks": [stopped]})


@pytest.mark.parametrize("payload", [b'{"version":1,"version":2}', b"[" * 2000, b"\xff"])
def test_ambiguous_deep_or_invalid_json_is_not_blocked(tmp_path: Path, payload: bytes) -> None:
    path = tmp_path / "audit.json"
    path.write_bytes(payload)
    assert module.main([str(path), "--require-blocked"]) == 2


def test_cli_genuine_blocker_passes_gate(
    tmp_path: Path, stopped: dict[str, object], blocker: dict[str, str]
) -> None:
    blocker.update(authority="user", scope="current-task", source_origin="original")
    path = tmp_path / "audit.json"
    path.write_text(
        json.dumps({"version": 1, "task_reference": "current", "tasks": [stopped]}),
        encoding="utf-8",
    )
    assert module.main([str(path), "--require-blocked"]) == 0


def test_unready_local_work_defers_without_bypassing_dependency(
    stopped: dict[str, object], blocker: dict[str, str]
) -> None:
    blocker.update(
        kind="failure", authority="observation", scope="current-task", resolution="local"
    )
    stopped["needs"] = ["prepare"]
    result = module.audit(
        {
            "version": 1,
            "task_reference": "current",
            "tasks": [
                stopped,
                {
                    "name": "prepare",
                    "needs": [],
                    "state": "running",
                    "evidence": "existing owned job",
                },
            ],
        }
    )
    assert result["deferred"] == ["encode"]
    assert result["investigate_locally"] == []
    assert result["decision"] == "observe-existing"


def test_descriptor_closed_on_validation_failure(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    path = tmp_path / "audit.json"
    path.write_text("{}", encoding="utf-8")
    descriptor = os.open(path, os.O_RDONLY)
    monkeypatch.setattr(os, "open", lambda *_args, **_kwargs: descriptor)
    monkeypatch.setattr(module, "MAX_BYTES", 1)
    assert module.main([str(path)]) == 2
    with pytest.raises(OSError):
        os.fstat(descriptor)
