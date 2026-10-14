"""Latest user intent stays visible without granting new execution authority."""

import json
from pathlib import Path

import pytest

from replay import blocker_audit as module


@pytest.fixture
def addition() -> dict[str, object]:
    return {
        "name": "remove-template",
        "source": "message-latest",
        "origin": "direct-user",
        "instruction": "Delete the old template",
        "target": "the template already identified in context",
        "acceptance": "The exact approved target is absent after verified removal",
        "tasks": [],
        "supersedes": [],
    }


@pytest.fixture
def ledger(addition: dict[str, object]) -> dict[str, object]:
    return {"version": 2, "task_reference": "original request", "tasks": [], "requests": [addition]}


def test_unassigned_addition_cannot_disappear_into_a_note(ledger: dict[str, object]) -> None:
    assert module.audit(ledger, latest_request="message-latest") == {
        "decision": "continue",
        "candidate_tasks": [],
        "review_restriction_sources": [],
        "investigate_locally": [],
        "observe_existing": [],
        "deferred": [],
        "may_report_blocked": False,
        "execution_authorized": False,
        "unassigned_requests": ["remove-template"],
        "unfulfilled_requests": [],
        "requests_to_verify": [],
        "superseded_requests": [],
        "review_superseded_tasks": [],
    }


def test_disuse_is_not_completion_of_separate_deletion_request(ledger: dict[str, object]) -> None:
    ledger["tasks"] = [
        {
            "name": "stop-using",
            "needs": [],
            "state": "complete",
            "evidence": "old template no longer selected",
        }
    ]
    result = module.audit(ledger)
    assert result["unassigned_requests"] == ["remove-template"]
    assert result["requests_to_verify"] == []
    assert result["decision"] == "continue"


def test_bound_action_is_still_unfulfilled(
    ledger: dict[str, object], addition: dict[str, object]
) -> None:
    addition["tasks"] = ["remove"]
    ledger["tasks"] = [
        {
            "name": "remove",
            "needs": [],
            "state": "pending",
            "evidence": "exact target and references checked",
        }
    ]
    result = module.audit(ledger)
    assert result["candidate_tasks"] == ["remove"]
    assert result["unfulfilled_requests"] == ["remove-template"]
    assert result["execution_authorized"] is False


def test_true_higher_priority_conflict_does_not_erase_user_request(
    ledger: dict[str, object], addition: dict[str, object]
) -> None:
    addition["tasks"] = ["remove"]
    ledger["tasks"] = [
        {
            "name": "remove",
            "needs": [],
            "state": "blocked",
            "evidence": "exact target is outside authorized scope",
            "blocker": {
                "kind": "restriction",
                "authority": "developer",
                "scope": "current-task",
                "source": "original governing storage rule",
                "source_origin": "original",
                "resolution": "external",
                "attempts": "Compared exact canonical target with governing scope",
                "next_action": "Retain unmet request and identify the governing scope conflict",
                "recheck_when": "Governing scope changes, not merely repeated user consent",
            },
        }
    ]
    result = module.audit(ledger, latest_request="message-latest")
    assert result["decision"] == "blocked"
    assert result["unfulfilled_requests"] == ["remove-template"]
    assert result["requests_to_verify"] == []
    assert result["execution_authorized"] is False


def test_completed_action_needs_outcome_verification(
    ledger: dict[str, object], addition: dict[str, object]
) -> None:
    addition["tasks"] = ["remove"]
    ledger["tasks"] = [
        {
            "name": "remove",
            "needs": [],
            "state": "complete",
            "evidence": "synthetic exact-target absence receipt",
        }
    ]
    result = module.audit(ledger)
    assert result["decision"] == "verify-completion"
    assert result["unfulfilled_requests"] == []
    assert result["requests_to_verify"] == ["remove-template"]
    assert result["execution_authorized"] is False


def test_new_user_request_supersedes_old_plan_not_unrelated_requirement(
    ledger: dict[str, object], addition: dict[str, object]
) -> None:
    old = addition | {
        "name": "retain-template",
        "source": "message-old",
        "instruction": "Keep the template",
        "tasks": ["retain"],
    }
    style = addition | {
        "name": "preserve-style",
        "source": "message-style",
        "instruction": "Preserve the selected style",
        "tasks": ["style"],
    }
    addition["supersedes"] = ["retain-template"]
    addition["tasks"] = ["remove"]
    ledger["requests"] = [old, style, addition]
    ledger["tasks"] = [
        {"name": "retain", "needs": [], "state": "pending", "evidence": "old plan"},
        {"name": "style", "needs": [], "state": "pending", "evidence": "unchanged requirement"},
        {
            "name": "remove",
            "needs": [],
            "state": "pending",
            "evidence": "new action after safe reconciliation",
        },
    ]
    result = module.audit(ledger, latest_request="message-latest")
    assert result["superseded_requests"] == ["retain-template"]
    assert result["review_superseded_tasks"] == ["retain"]
    assert result["candidate_tasks"] == ["style", "remove"]
    assert result["unfulfilled_requests"] == ["preserve-style", "remove-template"]
    assert result["execution_authorized"] is False


def test_repeated_instruction_does_not_duplicate_shared_action(
    ledger: dict[str, object], addition: dict[str, object]
) -> None:
    old = addition | {"name": "first-request", "source": "message-old", "tasks": ["remove"]}
    addition["supersedes"] = ["first-request"]
    addition["tasks"] = ["remove"]
    ledger["requests"] = [old, addition]
    ledger["tasks"] = [
        {
            "name": "remove",
            "needs": [],
            "state": "running",
            "evidence": "existing job, not resubmitted",
        }
    ]
    result = module.audit(ledger)
    assert result["superseded_requests"] == ["first-request"]
    assert result["review_superseded_tasks"] == []
    assert result["observe_existing"] == ["remove"]
    assert result["decision"] == "observe-existing"


@pytest.mark.parametrize("source", ["message-newer", ""])
def test_stale_or_invalid_latest_source_fails(ledger: dict[str, object], source: str) -> None:
    with pytest.raises(module.AuditError):
        module.audit(ledger, latest_request=source)


def test_legacy_snapshot_cannot_claim_latest_request_checked() -> None:
    snapshot = {
        "version": 1,
        "task_reference": "old",
        "tasks": [
            {"name": "old", "needs": [], "state": "complete", "evidence": "old completion"},
        ],
    }
    with pytest.raises(module.AuditError, match="version-2"):
        module.audit(snapshot, latest_request="new-message")


@pytest.mark.parametrize("entries", [None, [], [{}] * 513, [None], [{}]])
def test_invalid_request_list(ledger: dict[str, object], entries: object) -> None:
    ledger["requests"] = entries
    with pytest.raises(module.AuditError):
        module.audit(ledger)


@pytest.mark.parametrize(
    "change",
    [
        {"origin": "agent-note"},
        {"origin": "developer"},
        {"instruction": ""},
        {"target": ""},
        {"acceptance": ""},
        {"source": ""},
        {"name": "../escape"},
        {"tasks": "remove"},
        {"tasks": ["remove"] * 513},
        {"tasks": ["remove", "remove"]},
        {"tasks": ["unknown"]},
        {"supersedes": ["future"]},
        {"supersedes": ["remove-template"]},
        {"extra": "ignored intent"},
    ],
)
def test_invalid_request_fields(
    ledger: dict[str, object], addition: dict[str, object], change: dict[str, object]
) -> None:
    addition.update(change)
    with pytest.raises(module.AuditError):
        module.audit(ledger)


def test_duplicate_requests_rejected(
    ledger: dict[str, object], addition: dict[str, object]
) -> None:
    ledger["requests"] = [addition, addition]
    with pytest.raises(module.AuditError):
        module.audit(ledger)


def test_superseded_only_task_requires_review_not_automatic_resume(
    ledger: dict[str, object], addition: dict[str, object]
) -> None:
    old = addition | {"name": "old-request", "source": "old", "tasks": ["old"]}
    addition["supersedes"] = ["old-request"]
    addition["tasks"] = ["new"]
    ledger["requests"] = [old, addition]
    ledger["tasks"] = [
        {"name": "old", "needs": [], "state": "running", "evidence": "old job outcome uncertain"},
        {
            "name": "new",
            "needs": [],
            "state": "complete",
            "evidence": "new result for verification",
        },
    ]
    result = module.audit(ledger)
    assert result["decision"] == "continue"
    assert result["review_superseded_tasks"] == ["old"]
    assert result["observe_existing"] == ["old"]
    assert result["requests_to_verify"] == ["remove-template"]
    assert result["execution_authorized"] is False


def test_reconciled_old_plan_can_finish_without_executing_superseded_action(
    ledger: dict[str, object], addition: dict[str, object]
) -> None:
    old = addition | {
        "name": "old-request",
        "source": "old-message",
        "tasks": ["obsolete-action"],
    }
    addition["supersedes"] = ["old-request"]
    addition["tasks"] = ["new-action"]
    ledger["requests"] = [old, addition]
    ledger["tasks"] = [
        {
            "name": "new-action",
            "needs": [],
            "state": "complete",
            "evidence": "new result verified; old unstarted plan retired; prior snapshot kept",
        }
    ]
    result = module.audit(ledger, latest_request="message-latest")
    assert result["decision"] == "verify-completion"
    assert result["review_superseded_tasks"] == []
    assert result["superseded_requests"] == ["old-request"]
    assert result["requests_to_verify"] == ["remove-template"]
    assert result["execution_authorized"] is False


@pytest.fixture
def reversed_restriction(
    ledger: dict[str, object], addition: dict[str, object]
) -> dict[str, object]:
    old = addition | {
        "name": "retain-template",
        "source": "message-old",
        "tasks": ["remove"],
        "instruction": "Keep the old template",
        "acceptance": "The template remains",
    }
    addition["tasks"] = ["remove"]
    addition["supersedes"] = ["retain-template"]
    ledger["requests"] = [old, addition]
    ledger["tasks"] = [
        {
            "name": "remove",
            "needs": [],
            "state": "blocked",
            "evidence": "old plan not yet updated",
            "blocker": {
                "kind": "restriction",
                "authority": "user",
                "scope": "current-task",
                "source": "message-old",
                "source_origin": "original",
                "resolution": "external",
                "attempts": "Compared old retention request and new deletion instruction",
                "next_action": "Reconcile the outdated restriction with the latest request",
                "recheck_when": "Latest instruction reconciled before the affected action",
            },
        }
    ]
    return ledger


def test_latest_reversal_routes_old_user_restriction_to_review(
    reversed_restriction: dict[str, object],
) -> None:
    result = module.audit(reversed_restriction, latest_request="message-latest")
    assert result["decision"] == "continue"
    assert result["review_restriction_sources"] == ["remove"]
    assert result["deferred"] == []
    assert result["unfulfilled_requests"] == ["remove-template"]
    assert result["execution_authorized"] is False


@pytest.mark.parametrize("authority", ["system", "developer"])
def test_user_reversal_never_supersedes_higher_priority_rule(
    reversed_restriction: dict[str, object], authority: str
) -> None:
    tasks = reversed_restriction["tasks"]
    assert isinstance(tasks, list)
    task = tasks[0]
    assert isinstance(task, dict)
    blocker = task["blocker"]
    assert isinstance(blocker, dict)
    blocker["authority"] = authority
    result = module.audit(reversed_restriction, latest_request="message-latest")
    assert result["decision"] == "blocked"
    assert result["review_restriction_sources"] == []
    assert result["unfulfilled_requests"] == ["remove-template"]
    assert result["execution_authorized"] is False


def test_cli_latest_pin_is_read_only_and_hides_instruction(
    tmp_path: Path, ledger: dict[str, object], capsys: pytest.CaptureFixture[str]
) -> None:
    path = tmp_path / "ledger.json"
    path.write_text(json.dumps(ledger), encoding="utf-8")
    before = path.read_bytes()
    assert module.main([str(path), "--latest-request", "message-latest", "--require-blocked"]) == 1
    output = capsys.readouterr().out
    assert "remove-template" in output
    assert "Delete the old template" not in output
    assert path.read_bytes() == before
    assert sorted(item.name for item in tmp_path.iterdir()) == ["ledger.json"]
    assert module.main([str(path), "--latest-request", "message-newer"]) == 2
