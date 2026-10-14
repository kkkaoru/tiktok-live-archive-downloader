"""Opt-in pytest ownership evidence; never retires files or changes bootstrap."""

import hashlib
import json
import os
from collections.abc import Callable, Generator, Mapping
from dataclasses import dataclass, field
from pathlib import Path
from types import MappingProxyType

import pytest

from replay.fcp_native_queue import SystemHost
from replay.native_sequence import ContractError
from replay.pytest_scratch import STATE, Scratch, verify_bootstrap
from replay.recordings import record
from replay.sequence_journal import (
    bounded_read,
    decode,
    encode,
    invalid_constant,
    publish,
    unique_pairs,
)

MAX_RECORDS = 100_000
FINISH_FRAGMENT_BYTES = 1024 * 1024
MAX_FINISH_FRAGMENTS = 128
MAX_FINISH_BYTES = FINISH_FRAGMENT_BYTES * MAX_FINISH_FRAGMENTS


@dataclass
class Lifecycle:
    scratch: Scratch
    evidence: Path
    start_sha256: str
    _fixtures: dict[str, str] = field(default_factory=dict, init=False, repr=False)
    _fixture_ancestors: set[str] = field(default_factory=set, init=False, repr=False)
    outcomes: dict[str, dict[str, str]] = field(default_factory=dict)

    @property
    def fixtures(self) -> Mapping[str, str]:
        """Read-only registrations; only register may update the overlap index."""
        return MappingProxyType(self._fixtures)


LIFECYCLE = pytest.StashKey[Lifecycle]()


def canonical(path: Path, work: Path) -> Path:
    if (
        not path.is_absolute()
        or path != path.resolve()
        or not path.is_relative_to(work)
        or path == work
        or any(part.is_symlink() for part in (path, *path.parents))
    ):
        raise ContractError("Lifecycle paths must be canonical repository work descendants")
    return path


def start(config: pytest.Config, request: Path) -> Lifecycle:
    verify_bootstrap(config)
    scratch = config.stash[STATE]
    work = scratch.root / "edits/work"
    raw = decode(bounded_read(canonical(request, work)))
    if (
        set(raw) != {"version", "kind", "evidenceDirectory"}
        or raw.get("version") != 1
        or raw.get("kind") != "pytest-lifecycle-request"
        or not isinstance(raw.get("evidenceDirectory"), str)
    ):
        raise ContractError("Invalid lifecycle request")
    evidence = canonical(Path(str(raw["evidenceDirectory"])), work)
    if (
        not evidence.is_dir()
        or config.known_args_namespace.basetemp is None
        or scratch.base.exists()
    ):
        raise ContractError(
            "Lifecycle requires a fresh explicit basetemp and existing evidence directory"
        )
    if any(
        evidence.is_relative_to(root) or root.is_relative_to(evidence)
        for root in (scratch.base, scratch.python)
    ):
        raise ContractError("Evidence must be disjoint from both actual scratch trees")
    original = bounded_read(request)
    lock = work / "pytest-default/locks" / hashlib.sha256(os.fsencode(scratch.base)).hexdigest()
    canonical(lock, work)
    lock_info = lock.stat()
    value = {
        "version": 1,
        "kind": "pytest-lifecycle-start",
        "request": {"path": str(request), "sha256": hashlib.sha256(original).hexdigest()},
        "owner": {
            "pid": os.getpid(),
            "birth": SystemHost().birth(os.getpid()),
            "pgid": os.getpgrp(),
        },
        "base": str(scratch.base),
        "pythonTmp": str(scratch.python),
        "evidenceDirectory": str(evidence),
        "baseLock": {"path": str(lock), "dev": lock_info.st_dev, "inode": lock_info.st_ino},
    }
    data = encode(value)
    publish(evidence / "run-start.json", data)
    state = Lifecycle(scratch, evidence, hashlib.sha256(data).hexdigest())
    config.stash[LIFECYCLE] = state
    return state


def register(state: Lifecycle, node: str, path: Path) -> None:
    canonical(path, state.scratch.root / "edits/work")
    if (
        not node
        or not path.is_dir()
        or not any(
            path.is_relative_to(root) and path != root
            for root in (state.scratch.base, state.scratch.python)
        )
    ):
        raise ContractError("Disposable fixture must be an exact owned synthetic child directory")
    key = str(path)
    ancestors = tuple(str(parent) for parent in path.parents)
    if (
        len(state._fixtures) >= MAX_RECORDS
        or key in state._fixtures
        or key in state._fixture_ancestors
        or any(parent in state._fixtures for parent in ancestors)
    ):
        raise ContractError("Fixture ownership overlaps or exceeds its bound")
    state._fixtures[key] = node
    state._fixture_ancestors.update(ancestors)


def outcome(state: Lifecycle, *, node: str, phase: str, result: str) -> None:
    if (
        phase not in {"setup", "call", "teardown"}
        or result not in {"passed", "failed", "skipped"}
        or not node
        or len(state.outcomes) >= MAX_RECORDS
    ):
        raise ContractError("Invalid or unbounded lifecycle outcome")
    phases = state.outcomes.setdefault(node, {})
    if phase in phases:
        raise ContractError("Repeated test phase requires a separate run identity")
    phases[phase] = result


def _write_finish(path: Path, payload: dict[str, object]) -> None:
    fragments: list[dict[str, object]] = []
    pending = bytearray()
    total = 0
    digest = hashlib.sha256()

    def save(data: bytes) -> None:
        index = len(fragments)
        if index >= MAX_FINISH_FRAGMENTS:
            raise ContractError("Lifecycle finish fragment count exceeds its bound")
        child = path.with_name(f"{path.stem}-{index:04d}.fragment")
        publish(child, data)
        fragments.append(
            {"name": child.name, "bytes": len(data), "sha256": hashlib.sha256(data).hexdigest()}
        )

    encoder = json.JSONEncoder(sort_keys=True, separators=(",", ":"), allow_nan=False)
    for token in encoder.iterencode(payload):
        data = token.encode()
        total += len(data)
        if total > MAX_FINISH_BYTES:
            raise ContractError("Lifecycle finish exceeds aggregate byte budget")
        digest.update(data)
        view = memoryview(data)
        offset = 0
        while offset < len(view):
            count = min(FINISH_FRAGMENT_BYTES - len(pending), len(view) - offset)
            pending.extend(view[offset : offset + count])
            offset += count
            if len(pending) == FINISH_FRAGMENT_BYTES:
                save(bytes(pending))
                pending.clear()
    if pending:
        save(bytes(pending))
    publish(
        path,
        encode(
            {
                "version": 2,
                "kind": "pytest-lifecycle-finish-manifest",
                "startSHA256": payload["startSHA256"],
                "exitStatus": payload["exitStatus"],
                "fixtureCount": len(record(payload["fixtures"])),
                "outcomeCount": len(record(payload["outcomes"])),
                "bytes": total,
                "sha256": digest.hexdigest(),
                "fragments": fragments,
            }
        ),
    )


def load_finish(path: Path) -> dict[str, object]:
    """Read ordered bounded fragments, preserving every original node ID byte."""
    original = bounded_read(path)
    manifest = decode(original)
    if manifest.get("kind") == "pytest-lifecycle-finish":
        return manifest
    rows = manifest.get("fragments")
    count = manifest.get("bytes")
    if (
        set(manifest)
        != {
            "version",
            "kind",
            "startSHA256",
            "exitStatus",
            "fixtureCount",
            "outcomeCount",
            "bytes",
            "sha256",
            "fragments",
        }
        or manifest.get("kind") != "pytest-lifecycle-finish-manifest"
        or manifest.get("version") != 2
        or not isinstance(rows, list)
        or not 1 <= len(rows) <= MAX_FINISH_FRAGMENTS
        or type(count) is not int
        or not isinstance(count, int)
        or not 0 < count <= MAX_FINISH_BYTES
        or any(
            type(manifest.get(key)) is not int
            for key in ("exitStatus", "fixtureCount", "outcomeCount")
        )
    ):
        raise ContractError("Invalid bounded lifecycle manifest")
    data = bytearray()
    for index, value in enumerate(rows):
        row = record(value)
        name = f"{path.stem}-{index:04d}.fragment"
        if set(row) != {"name", "bytes", "sha256"} or row.get("name") != name:
            raise ContractError("Lifecycle fragments must be exact and ordered")
        fragment = bounded_read(path.with_name(name))
        if (
            not 0 < len(fragment) <= FINISH_FRAGMENT_BYTES
            or type(row.get("bytes")) is not int
            or row.get("bytes") != len(fragment)
            or row.get("sha256") != hashlib.sha256(fragment).hexdigest()
            or len(data) + len(fragment) > count
        ):
            raise ContractError("Lifecycle fragment identity or byte budget differs")
        data.extend(fragment)
    if len(data) != count or hashlib.sha256(data).hexdigest() != manifest.get("sha256"):
        raise ContractError("Lifecycle aggregate differs from manifest")
    payload = record(
        json.loads(data, object_pairs_hook=unique_pairs, parse_constant=invalid_constant)
    )
    fixtures, outcomes = record(payload.get("fixtures")), record(payload.get("outcomes"))
    if (
        payload.get("kind") != "pytest-lifecycle-finish"
        or payload.get("startSHA256") != manifest.get("startSHA256")
        or payload.get("exitStatus") != manifest.get("exitStatus")
        or len(fixtures) != manifest.get("fixtureCount")
        or len(fixtures) > MAX_RECORDS
        or len(outcomes) != manifest.get("outcomeCount")
        or len(outcomes) > MAX_RECORDS
        or bounded_read(path) != original
    ):
        raise ContractError("Lifecycle manifest and terminal payload differ")
    return payload


def finish(state: Lifecycle, exit_status: int) -> Path:
    start_path = state.evidence / "run-start.json"
    if hashlib.sha256(bounded_read(start_path)).hexdigest() != state.start_sha256:
        raise ContractError("Lifecycle start evidence changed")
    path = state.evidence / "session-finish.json"
    _write_finish(
        path,
        {
            "version": 1,
            "kind": "pytest-lifecycle-finish",
            "startSHA256": state.start_sha256,
            "exitStatus": exit_status,
            "fixtures": dict(state.fixtures),
            "outcomes": state.outcomes,
            "processTerminationVerified": False,
            "deletionPerformed": False,
        },
    )
    return path


def pytest_addoption(parser: pytest.Parser) -> None:
    parser.addoption("--lifecycle-request", default=None, help="Pinned owner lifecycle request")


@pytest.hookimpl(trylast=True)
def pytest_configure(config: pytest.Config) -> None:
    value = config.getoption("--lifecycle-request")
    if not isinstance(value, str) or not value:
        raise pytest.UsageError("Lifecycle plugin requires --lifecycle-request")
    start(config, Path(value))


def disposable_callback(request: pytest.FixtureRequest) -> Callable[[Path], None]:
    state = request.config.stash[LIFECYCLE]
    return lambda path: register(state, request.node.nodeid, path)


lifecycle_disposable = pytest.fixture(disposable_callback, name="lifecycle_disposable")


@pytest.hookimpl(wrapper=True)
def pytest_runtest_makereport(
    item: pytest.Item, call: pytest.CallInfo[None]
) -> Generator[None, pytest.TestReport, pytest.TestReport]:
    report = yield
    outcome(
        item.config.stash[LIFECYCLE], node=report.nodeid, phase=report.when, result=report.outcome
    )
    return report


def pytest_sessionfinish(session: pytest.Session, exitstatus: int) -> None:
    finish(session.config.stash[LIFECYCLE], exitstatus)
