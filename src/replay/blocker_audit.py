"""Audit a private task snapshot before reporting a whole goal blocked; never execute."""

import argparse
import json
import os
import re
import stat
import sys
from collections.abc import Sequence
from contextlib import ExitStack
from dataclasses import dataclass
from graphlib import CycleError, TopologicalSorter
from pathlib import Path

MAX_BYTES = 1_048_576
MAX_TASKS = 512
BLOCKER_FIELDS = {
    "kind",
    "authority",
    "scope",
    "source",
    "source_origin",
    "resolution",
    "next_action",
    "attempts",
    "recheck_when",
}


class AuditError(ValueError):
    """The snapshot cannot justify a blocker decision."""


@dataclass(frozen=True)
class Task:
    name: str
    needs: tuple[str, ...]
    state: str
    blocker: dict[str, str] | None


@dataclass(frozen=True)
class Request:
    name: str
    source: str
    tasks: tuple[str, ...]
    supersedes: tuple[str, ...]


class Arguments(argparse.Namespace):
    snapshot: Path
    require_blocked: bool
    latest_request: str | None


def _text(value: object) -> str:
    if not isinstance(value, str) or not value.strip() or len(value) > 4096 or "\0" in value:
        raise AuditError("Expected bounded nonempty evidence text")
    return value


def _choice(value: object, choices: set[str]) -> str:
    text = _text(value)
    if text not in choices:
        raise AuditError("Unknown snapshot classification")
    return text


def _name(value: object) -> str:
    name = _text(value)
    if re.fullmatch(r"[a-z0-9][a-z0-9_.-]{0,63}", name) is None:
        raise AuditError("Invalid task name")
    return name


def _blocker(value: object) -> dict[str, str]:
    if not isinstance(value, dict) or set(value) != BLOCKER_FIELDS:
        raise AuditError("Blocker needs source, scope, resolution, attempts and next action")
    result = {key: _text(value[key]) for key in BLOCKER_FIELDS}
    _choice(result["kind"], {"restriction", "prerequisite", "failure"})
    _choice(result["authority"], {"system", "developer", "user", "agent", "observation"})
    _choice(result["scope"], {"current-task", "other-task", "unverified"})
    _choice(result["source_origin"], {"original", "agent-note", "unverified"})
    _choice(result["resolution"], {"local", "external"})
    return result


def _task(value: object) -> Task:
    if not isinstance(value, dict) or not {"name", "needs", "state", "evidence"} <= set(value):
        raise AuditError("Incomplete task snapshot")
    if set(value) - {"name", "needs", "state", "evidence", "blocker"}:
        raise AuditError("Unknown task fields")
    _text(value["evidence"])
    state = _choice(value["state"], {"pending", "running", "complete", "blocked"})
    needs: object = value["needs"]
    if not isinstance(needs, list) or len(needs) > MAX_TASKS:
        raise AuditError("Invalid dependency list")
    parents = tuple(_name(item) for item in needs)
    if len(set(parents)) != len(parents):
        raise AuditError("Duplicate dependency")
    if (state == "blocked") != ("blocker" in value):
        raise AuditError("Only blocked tasks must carry a blocker")
    blocker = _blocker(value["blocker"]) if state == "blocked" else None
    return Task(_name(value["name"]), parents, state, blocker)


def _tasks(value: object) -> tuple[Task, ...]:
    if not isinstance(value, dict):
        raise AuditError("Invalid audit fields")
    version: object = value.get("version")
    if type(version) is not int or version not in {1, 2}:
        raise AuditError("Unsupported audit version")
    fields = {"version", "task_reference", "tasks"}
    if version == 2:
        fields.add("requests")
    if set(value) != fields:
        raise AuditError("Invalid audit fields")
    _text(value["task_reference"])
    entries: object = value["tasks"]
    minimum = 1 if version == 1 else 0
    if not isinstance(entries, list) or not minimum <= len(entries) <= MAX_TASKS:
        raise AuditError("Invalid task count")
    tasks = tuple(_task(entry) for entry in entries)
    names = {task.name for task in tasks}
    if len(names) != len(tasks) or any(set(task.needs) - names for task in tasks):
        raise AuditError("Duplicate task or unknown dependency")
    try:
        TopologicalSorter({task.name: task.needs for task in tasks}).prepare()
    except CycleError as error:
        raise AuditError("Task dependency cycle") from error
    complete = {task.name for task in tasks if task.state == "complete"}
    if any(task.state in {"complete", "running"} and set(task.needs) - complete for task in tasks):
        raise AuditError("Running/completed task has incomplete prerequisites")
    return tasks


def _references(value: object) -> tuple[str, ...]:
    if not isinstance(value, list) or len(value) > MAX_TASKS:
        raise AuditError("Invalid request references")
    names = tuple(_name(item) for item in value)
    if len(set(names)) != len(names):
        raise AuditError("Duplicate request reference")
    return names


def _request(value: object) -> Request:
    fields = {
        "name",
        "source",
        "origin",
        "instruction",
        "target",
        "acceptance",
        "tasks",
        "supersedes",
    }
    if not isinstance(value, dict) or set(value) != fields:
        raise AuditError("Incomplete request provenance or task binding")
    _choice(value["origin"], {"direct-user"})
    for field in ("instruction", "target", "acceptance"):
        _text(value[field])
    return Request(
        _name(value["name"]),
        _text(value["source"]),
        _references(value["tasks"]),
        _references(value["supersedes"]),
    )


def _request_routes(
    value: object, tasks: tuple[Task, ...], *, latest_request: str | None
) -> dict[str, list[str]]:
    if not isinstance(value, dict) or value["version"] != 2:
        if latest_request is not None:
            raise AuditError("Latest request requires a version-2 ledger")
        return {}
    entries: object = value["requests"]
    if not isinstance(entries, list) or not 1 <= len(entries) <= MAX_TASKS:
        raise AuditError("Invalid request count")
    requests = tuple(_request(entry) for entry in entries)
    if latest_request is not None and requests[-1].source != _text(latest_request):
        raise AuditError("Latest user request is missing from the ledger")
    known_tasks = {task.name for task in tasks}
    seen: set[str] = set()
    superseded: set[str] = set()
    for request in requests:
        if request.name in seen or set(request.supersedes) - seen:
            raise AuditError("Duplicate request or supersession of a non-earlier request")
        seen.add(request.name)
        superseded.update(request.supersedes)
    active = tuple(request for request in requests if request.name not in superseded)
    if any(set(request.tasks) - known_tasks for request in active):
        raise AuditError("Active request references an unknown task")
    linked: set[str] = set()
    retired: set[str] = set()
    for request in requests:
        if request.name in superseded:
            retired.update(request.tasks)
        else:
            linked.update(request.tasks)
    retired -= linked
    stale_sources = {request.source for request in requests if request.name in superseded}
    complete = {task.name for task in tasks if task.state == "complete"}
    return {
        "review_restriction_sources": [
            task.name
            for task in tasks
            if task.blocker is not None
            and task.blocker["authority"] == "user"
            and task.blocker["source"] in stale_sources
        ],
        "unassigned_requests": [request.name for request in active if not request.tasks],
        "unfulfilled_requests": [
            request.name for request in active if set(request.tasks) - complete
        ],
        "requests_to_verify": [
            request.name
            for request in active
            if request.tasks and not set(request.tasks) - complete
        ],
        "superseded_requests": [request.name for request in requests if request.name in superseded],
        "review_superseded_tasks": [task.name for task in tasks if task.name in retired],
    }


def audit(value: object, *, latest_request: str | None = None) -> dict[str, object]:
    """Classify declared work, not the truth of citations or permission to execute it."""
    tasks = _tasks(value)
    requests = _request_routes(value, tasks, latest_request=latest_request)
    stale_restrictions = set(requests.pop("review_restriction_sources", []))
    complete = {task.name for task in tasks if task.state == "complete"}
    candidates: list[str] = []
    review: list[str] = []
    investigate: list[str] = []
    observe: list[str] = []
    deferred: list[str] = []
    for task in tasks:
        ready = not set(task.needs) - complete
        blocker = task.blocker
        if blocker is not None:
            if (
                task.name in stale_restrictions
                or blocker["scope"] != "current-task"
                or (
                    blocker["kind"] == "restriction"
                    and (
                        blocker["authority"] not in {"system", "developer", "user"}
                        or blocker["source_origin"] != "original"
                    )
                )
            ):
                review.append(task.name)
            elif blocker["resolution"] == "local" and ready:
                investigate.append(task.name)
            else:
                deferred.append(task.name)
        elif task.state == "running":
            observe.append(task.name)
        elif task.state == "pending" and ready:
            candidates.append(task.name)
        elif task.state == "pending":
            deferred.append(task.name)
    superseded_tasks = set(requests.get("review_superseded_tasks", []))
    candidates = [name for name in candidates if name not in superseded_tasks]
    if (
        candidates
        or review
        or investigate
        or requests.get("unassigned_requests")
        or superseded_tasks
    ):
        decision = "continue"
    elif observe:
        decision = "observe-existing"
    elif len(complete) == len(tasks):
        decision = "verify-completion"
    else:
        decision = "blocked"
    return {
        "decision": decision,
        "candidate_tasks": candidates,
        "review_restriction_sources": review,
        "investigate_locally": investigate,
        "observe_existing": observe,
        "deferred": deferred,
        "may_report_blocked": decision == "blocked",
        "execution_authorized": False,
        **requests,
    }


def _unique_object(pairs: list[tuple[str, object]]) -> dict[str, object]:
    result = dict(pairs)
    if len(result) != len(pairs):
        raise AuditError("Duplicate JSON fields")
    return result


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("snapshot", type=Path)
    parser.add_argument(
        "--require-blocked", action="store_true", help="Fail if actionable work remains"
    )
    parser.add_argument(
        "--latest-request",
        help="Require the version-2 ledger to end with this user source reference",
    )
    arguments = Arguments()
    parser.parse_args(argv, namespace=arguments)
    try:
        with ExitStack() as stack:
            descriptor = os.open(arguments.snapshot, os.O_RDONLY | os.O_NONBLOCK | os.O_NOFOLLOW)
            stack.callback(os.close, descriptor)
            metadata = os.fstat(descriptor)
            if not stat.S_ISREG(metadata.st_mode) or metadata.st_size > MAX_BYTES:
                raise AuditError("Invalid snapshot file")
            stream = stack.enter_context(os.fdopen(descriptor, "rb", closefd=False))
            data = stream.read(MAX_BYTES + 1)
        if len(data) > MAX_BYTES:
            raise AuditError("Snapshot byte limit exceeded")
        value: object = json.loads(data.decode("utf-8"), object_pairs_hook=_unique_object)
        result = audit(value, latest_request=arguments.latest_request)
    except (OSError, ValueError, RecursionError):
        print("Invalid blocker snapshot; no blocked decision established.", file=sys.stderr)
        return 2
    print(json.dumps(result, sort_keys=True))
    return 1 if arguments.require_blocked and not result["may_report_blocked"] else 0


if __name__ == "__main__":
    raise SystemExit(main())
