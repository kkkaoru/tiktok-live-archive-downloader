"""Finite, once-only admission for a trusted, separately authorized workflow command.

Checks are read-only. A run retains its exclusive lease and immutable start/end
records, even after an uncertain launch. RSS and credited inactive pages are
planning estimates, never physical footprints or guaranteed reclaim. Unlisted
jobs are not discovered or reserved: the caller owns the complete family plan.
"""

from __future__ import annotations

import argparse
import ctypes
import fcntl
import hashlib
import json
import os
import re
import stat
import subprocess
import sys
import time
from collections.abc import Iterator, Sequence
from contextlib import contextmanager
from dataclasses import dataclass
from decimal import Decimal
from fractions import Fraction
from pathlib import Path
from typing import Literal, Protocol

from replay import disk_budget
from replay.core import ReplayError
from replay.fcp_native_admission import SystemHost as NativeHost
from replay.foreground_evidence import _bound_bytes
from replay.recordings import record
from replay.workflow import _stage

ROOT = Path(__file__).resolve().parents[2]
MAX_REQUEST_BYTES = 1024 * 1024
MAX_TOOL_BYTES = 64 * 1024 * 1024
MAX_PROBE_BYTES = 2 * 1024 * 1024
PROCESS_PATH_BYTES = 4096
MAX_PROCESSES = 8192
MAX_FAMILIES = 32
MAX_POLLS = 1024
MAX_OBSERVATION_BYTES = 16 * 1024
SAFE_NAME = re.compile(r"[a-z0-9][a-z0-9._-]{0,63}")
HASH = re.compile(r"[a-f0-9]{64}")


@dataclass(frozen=True)
class FileIdentity:
    dev: int
    inode: int
    size: int
    modified_ns: int
    changed_ns: int


@dataclass(frozen=True)
class Executable:
    path: Path
    sha256: str
    invocation: Path
    link_target: str | None
    identity: FileIdentity


@dataclass(frozen=True)
class Parent:
    pid: int
    birth: str
    executable: Path
    command_sha256: str


@dataclass(frozen=True)
class Completion:
    capture: Path
    sha256: str
    workflow: Path
    workflow_sha256: str
    checkpoint: Path
    status: Path
    fingerprint: str
    outputs: tuple[str, ...]
    process_groups: tuple[int, ...]


@dataclass(frozen=True)
class Family:
    parent: Parent
    peak_bytes: int
    disk_remaining_bytes: int
    completion: Completion | None = None


@dataclass(frozen=True)
class Policy:
    new_memory_bytes: int
    new_disk_bytes: int
    os_margin_bytes: int
    ocr_margin_bytes: int
    minimum_free_bytes: int
    inactive_credit: Fraction
    maximum_pressure: int
    maximum_swap_bytes: int
    maximum_swap_growth_bytes: int
    maximum_age_seconds: int
    wait_seconds: int
    poll_seconds: int


@dataclass(frozen=True)
class Request:
    path: Path
    sha256: str
    name: str
    work: Path
    scratch: Path
    cache: Path
    disk_paths: tuple[Path, ...]
    executable: Executable
    command: tuple[str, ...]
    families: tuple[Family, ...]
    policy: Policy


@dataclass(frozen=True)
class Process:
    pid: int
    ppid: int
    rss_bytes: int


@dataclass(frozen=True)
class FamilyUsage:
    parent: Parent
    pids: tuple[int, ...]
    rss_bytes: int
    state: Literal["active", "terminal-pending", "released"] = "active"
    terminal: tuple[tuple[str, str], ...] = ()


@dataclass(frozen=True)
class Snapshot:
    observed_ns: int
    page_size: int
    free_pages: int
    inactive_pages: int
    swap_bytes: int
    pressure: int
    families: tuple[FamilyUsage, ...]
    disks: tuple[disk_budget.Headroom, ...]


@dataclass(frozen=True)
class Assessment:
    admitted: bool
    reasons: tuple[str, ...]
    evidence: dict[str, object]


@dataclass(frozen=True)
class Lease:
    parent: int
    parent_identity: os.stat_result
    descriptor: int
    identity: os.stat_result


class Host(Protocol):
    def now_ns(self) -> int: ...
    def monotonic(self) -> float: ...
    def sleep(self, seconds: float) -> None: ...
    def observe(self, request: Request) -> Snapshot: ...
    def execute(self, request: Request) -> int: ...


class SystemHost:
    def now_ns(self) -> int:
        return time.time_ns()

    def monotonic(self) -> float:
        return time.monotonic()

    def sleep(self, seconds: float) -> None:
        time.sleep(seconds)

    def parent(self, expected: Parent) -> Parent:
        return self.identity(expected.pid)

    def identity(self, pid: int) -> Parent:
        native = NativeHost()
        before = native.birth(pid)
        executable = process_executable(pid)
        command = probe(["/bin/ps", "-ww", "-p", str(pid), "-o", "command="])
        if not command.strip() or native.birth(pid) != before:
            raise ReplayError("Parent disappeared or changed during identity measurement")
        return Parent(
            pid,
            before,
            executable,
            hashlib.sha256(command.encode("utf-8")).hexdigest(),
        )

    def observe(self, request: Request) -> Snapshot:
        observed = self.now_ns()
        groups = (
            parse_groups(probe(["/bin/ps", "-A", "-o", "pid=,pgid="]))
            if any(family.completion is not None for family in request.families)
            else ()
        )
        released: dict[int, FamilyUsage] = {}
        active: list[Family] = []
        for family in request.families:
            completion = family.completion
            if completion is None or any(group in completion.process_groups for _, group in groups):
                active.append(family)
            else:
                terminal = terminal_evidence(completion)
                released[family.parent.pid] = FamilyUsage(
                    family.parent, (), 0, "released" if terminal else "terminal-pending", terminal
                )
        before = tuple(self.parent(family.parent) for family in active)
        processes = parse_processes(probe(["/bin/ps", "-A", "-o", "pid=,ppid=,rss="]))
        page_size, free, inactive = parse_vm_stat(probe(["/usr/bin/vm_stat"]))
        swap = parse_swap(probe(["/usr/sbin/sysctl", "-n", "vm.swapusage"]))
        pressure = NativeHost().pressure()
        disks = tuple(disk_budget.measure(path) for path in request.disk_paths)
        after = tuple(self.parent(family.parent) for family in active)
        if before != after or before != tuple(family.parent for family in active):
            raise ReplayError("Owned parent identity changed; a new plan is required")
        if groups != ():
            final_groups = parse_groups(probe(["/bin/ps", "-A", "-o", "pid=,pgid="]))
            for family in request.families:
                if family.parent.pid in released:
                    completion = family.completion
                    if completion is None or any(
                        group in completion.process_groups for _, group in final_groups
                    ):
                        raise ReplayError("A released process group reappeared during observation")
                    if released[family.parent.pid].terminal != terminal_evidence(completion):
                        raise ReplayError("Terminal actor evidence changed during observation")
        usages = {item.parent.pid: item for item in family_usage(tuple(active), processes)}
        usages.update(released)
        families = tuple(usages[family.parent.pid] for family in request.families)
        return Snapshot(observed, page_size, free, inactive, swap, pressure, families, disks)

    def execute(self, request: Request) -> int:
        # Keep the current configured environment, including its venv invocation.
        # Only explicit scratch/cache paths are changed, and no values are logged.
        environment = dict(os.environ)
        environment.update({name: str(request.scratch) for name in ("TMPDIR", "TMP", "TEMP")})
        environment["XDG_CACHE_HOME"] = str(request.cache)
        command = [
            sys.executable,
            "-B",
            "-m",
            "replay.phase_timer",
            str(request.work / f"{request.name}.phases.jsonl"),
            request.name,
            "--",
            *request.command,
        ]
        descriptor = os.open(request.work, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW)
        try:
            _parent_guard(descriptor, request, os.fstat(descriptor))
            with os.fdopen(
                os.open(
                    f"{request.name}.child.log",
                    os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW,
                    0o600,
                    dir_fd=descriptor,
                ),
                "wb",
            ) as log:
                result = subprocess.run(
                    command,
                    cwd=ROOT,
                    env=environment,
                    stdin=subprocess.DEVNULL,
                    stdout=log,
                    stderr=subprocess.STDOUT,
                    check=False,
                )
                log.flush()
                os.fsync(log.fileno())
            os.fsync(descriptor)
            return result.returncode
        finally:
            os.close(descriptor)


def _integer(value: object, *, minimum: int = 0, maximum: int = 2**63 - 1) -> int:
    if type(value) is not int or not minimum <= value <= maximum:
        raise ReplayError("Expected a bounded admission integer")
    return value


def _text(value: object, *, maximum: int = 4096) -> str:
    if not isinstance(value, str) or not 0 < len(value) <= maximum:
        raise ReplayError("Expected bounded admission text")
    if any(character in value for character in "\0\r\n"):
        raise ReplayError("Admission text contains a control character")
    return value


def _hash(value: object) -> str:
    result = _text(value, maximum=64)
    if HASH.fullmatch(result) is None:
        raise ReplayError("Expected a SHA256 identity")
    return result


def _unique(items: list[tuple[str, object]]) -> dict[str, object]:
    result: dict[str, object] = {}
    for key, value in items:
        if key in result:
            raise ReplayError("Duplicate request key")
        result[key] = value
    return result


def _schema(raw: dict[str, object], keys: set[str]) -> None:
    if set(raw) != keys:
        raise ReplayError("Incomplete or unknown admission fields")


def _directory_path(value: object, *, directory: bool = True, work: bool = False) -> Path:
    path = Path(_text(value))
    if (
        not path.is_absolute()
        or ".." in path.parts
        or path != path.resolve(strict=True)
        or any(parent.is_symlink() for parent in (path, *path.parents))
        or any(part in {"private", "downloads", ".git"} for part in path.parts)
        or directory
        and not path.is_dir()
        or work
        and not path.is_relative_to(ROOT / "edits" / "work")
    ):
        raise ReplayError("Expected a canonical unlinked admission path")
    return path


def _identity(path: Path) -> FileIdentity:
    value = path.lstat()
    if not stat.S_ISREG(value.st_mode) or value.st_nlink != 1:
        raise ReplayError("Admission input must be a single-link regular file")
    return _stat_identity(value)


def _stat_identity(value: os.stat_result) -> FileIdentity:
    return FileIdentity(
        value.st_dev, value.st_ino, value.st_size, value.st_mtime_ns, value.st_ctime_ns
    )


def _executable(value: object, *, fresh: bool) -> Executable:
    raw = record(value)
    _schema(raw, {"path", "sha256", "invocationPath", "linkTarget"})
    path = _directory_path(raw["path"], directory=False)
    digest = _hash(raw["sha256"])
    invocation = Path(_text(raw["invocationPath"]))
    target = raw["linkTarget"]
    if target is not None:
        target = _text(target)
        if (
            not invocation.is_absolute()
            or not invocation.is_relative_to(ROOT)
            or invocation.parent != invocation.parent.resolve(strict=True)
            or not invocation.is_symlink()
            or os.readlink(invocation) != target
            or invocation.resolve(strict=True) != path
        ):
            raise ReplayError("Configured executable alias changed")
    elif invocation != path:
        raise ReplayError("Unbound executable invocation alias")
    identity = _identity(path)
    if not os.access(invocation, os.X_OK):
        raise ReplayError("Configured executable is not executable")
    if fresh:
        _bound_bytes(path, digest, MAX_TOOL_BYTES)
    return Executable(path, digest, invocation, target, identity)


def _parent(value: object) -> Parent:
    raw = record(value)
    _schema(raw, {"pid", "birth", "executable", "commandSHA256"})
    birth = _text(raw["birth"], maximum=64)
    if re.fullmatch(r"[1-9][0-9]*:[0-9]{1,6}", birth) is None:
        raise ReplayError("Expected exact libproc process birth")
    return Parent(
        _integer(raw["pid"], minimum=1),
        birth,
        _directory_path(raw["executable"], directory=False),
        _hash(raw["commandSHA256"]),
    )


def _family(value: object) -> Family:
    raw = record(value)
    keys = {"parent", "peakMemoryBytes", "remainingDiskBytes"}
    if set(raw) not in (keys, keys | {"completion"}):
        raise ReplayError("Incomplete or unknown family fields")
    parent = _parent(raw["parent"])
    return Family(
        parent,
        _integer(raw["peakMemoryBytes"], minimum=1),
        _integer(raw["remainingDiskBytes"]),
        _completion(raw["completion"], parent) if "completion" in raw else None,
    )


def _future_file(value: object) -> Path:
    path = Path(_text(value))
    parent = _directory_path(str(path.parent), work=True)
    if path.name in {".", ".."} or path != parent / path.name or path.is_symlink():
        raise ReplayError("Unsafe future terminal evidence path")
    if path.exists():
        _identity(path)
    return path


def _current_bytes(path: Path, limit: int = MAX_REQUEST_BYTES) -> tuple[bytes, str]:
    path = _directory_path(str(path), directory=False, work=True)
    before = _identity(path)
    if not 0 < before.size <= limit:
        raise ReplayError("Terminal metadata exceeds its byte budget or is empty")
    with os.fdopen(os.open(path, os.O_RDONLY | os.O_NOFOLLOW), "rb") as stream:
        opened = os.fstat(stream.fileno())
        data = stream.read(limit + 1)
        final = os.fstat(stream.fileno())
    if (
        (opened.st_dev, opened.st_ino) != (before.dev, before.inode)
        or _stat_identity(opened) != _stat_identity(final)
        or _identity(path) != before
        or len(data) != before.size
    ):
        raise ReplayError("Terminal metadata changed during its bounded read")
    return data, hashlib.sha256(data).hexdigest()


def _workflow_outputs(path: Path, digest: str, name: str) -> tuple[str, ...]:
    raw = record(
        json.loads(_bound_bytes(path, digest, MAX_REQUEST_BYTES), object_pairs_hook=_unique)
    )
    entries = raw.get("stages")
    if raw.get("version") != 1 or not isinstance(entries, list) or not 1 <= len(entries) <= 512:
        raise ReplayError("Invalid completion workflow")
    stages = tuple(_stage(item) for item in entries)
    matches = tuple(stage for stage in stages if stage.name == name)
    if len(matches) != 1:
        raise ReplayError("Completion stage is missing or ambiguous")
    return tuple(str(path) for path in matches[0].outputs)


def _completion(value: object, parent: Parent) -> Completion:
    raw = record(value)
    _schema(raw, {"capturePath", "sha256"})
    path = _directory_path(raw["capturePath"], directory=False, work=True)
    digest = _hash(raw["sha256"])
    captured = record(json.loads(_bound_bytes(path, digest, MAX_OBSERVATION_BYTES)))
    _schema(
        captured,
        {
            "version",
            "kind",
            "observedAtNs",
            "parent",
            "processGroups",
            "workflow",
            "stage",
            "checkpoint",
            "statusPath",
            "initialCheckpointSHA256",
            "initialCheckpoint",
        },
    )
    if (
        captured["version"] != 1
        or captured["kind"] != "owned-family-completion-capture"
        or _parent(captured["parent"]) != parent
    ):
        raise ReplayError("Completion capture does not bind the declared actor")
    _integer(captured["observedAtNs"], minimum=1)
    workflow = record(captured["workflow"])
    _schema(workflow, {"path", "sha256"})
    workflow_path = _directory_path(workflow["path"], directory=False, work=True)
    workflow_digest = _hash(workflow["sha256"])
    stage = _text(captured["stage"], maximum=64)
    if SAFE_NAME.fullmatch(stage) is None:
        raise ReplayError("Invalid completion stage name")
    checkpoint = _future_file(captured["checkpoint"])
    if checkpoint.name != f"{stage}.json":
        raise ReplayError("Completion checkpoint does not bind its stage name")
    initial = record(captured["initialCheckpoint"])
    _schema(initial, {"status", "fingerprint"})
    if initial["status"] != "running":
        raise ReplayError("Completion capture must observe a genuinely running stage")
    _hash(captured["initialCheckpointSHA256"])
    groups = captured["processGroups"]
    if not isinstance(groups, list) or not 1 <= len(groups) <= MAX_PROCESSES:
        raise ReplayError("Completion capture lacks bounded descendant process groups")
    process_groups = tuple(_integer(item, minimum=1) for item in groups)
    if tuple(sorted(set(process_groups))) != process_groups:
        raise ReplayError("Captured process groups must be unique and ordered")
    return Completion(
        path,
        digest,
        workflow_path,
        workflow_digest,
        checkpoint,
        _future_file(captured["statusPath"]),
        _hash(initial["fingerprint"]),
        _workflow_outputs(workflow_path, workflow_digest, stage),
        process_groups,
    )


def terminal_evidence(completion: Completion) -> tuple[tuple[str, str], ...]:
    """Read existing terminal metadata only; do not verify or regenerate media.

    Status zero requires the original fingerprint-bound complete stage and its
    exact output hash table. Failed/stopped actors need a separate current writer
    and orphan reconciliation; initial captured groups alone cannot release them.
    """
    if not completion.status.exists() and not completion.status.is_symlink():
        return ()
    status_data, status_sha = _current_bytes(completion.status, 32)
    if re.fullmatch(rb"(?:0|[1-9][0-9]{0,2})\n?", status_data) is None:
        raise ReplayError("Malformed retained terminal exit status")
    code = int(status_data)
    if code != 0:
        raise ReplayError("Failed actor release requires independent current writer reconciliation")
    data, receipt_sha = _current_bytes(completion.checkpoint)
    raw = record(json.loads(data, object_pairs_hook=_unique))
    if raw.get("fingerprint") != completion.fingerprint:
        raise ReplayError("Terminal stage fingerprint changed")
    _schema(raw, {"status", "fingerprint", "outputs"})
    outputs = record(raw["outputs"])
    if raw["status"] != "complete" or set(outputs) != set(completion.outputs):
        raise ReplayError("Zero status lacks the complete original stage output table")
    tuple(_hash(value) for value in outputs.values())
    return ((str(completion.status), status_sha), (str(completion.checkpoint), receipt_sha))


def load_completion_capture(path: Path, sha256: str) -> tuple[Parent, Completion]:
    """Read a pinned live capture without probing resources or executing a job."""
    path = _directory_path(str(path), directory=False, work=True)
    raw = record(json.loads(_bound_bytes(path, _hash(sha256), MAX_OBSERVATION_BYTES)))
    parent = _parent(raw.get("parent"))
    return parent, _completion({"capturePath": str(path), "sha256": sha256}, parent)


def verify_completion_capture(path: Path, sha256: str) -> dict[str, object]:
    """Observe captured groups absent and terminal metadata stable, without dispatch.

    This is an ownership/resource-release proof. Existing domain verifiers must
    separately approve media or partial-window adoption under their own locks.
    """
    parent, completion = load_completion_capture(path, sha256)
    before = parse_groups(probe(["/bin/ps", "-A", "-o", "pid=,pgid="]))
    if any(group in completion.process_groups for _, group in before):
        raise ReplayError("Captured owned process group is still present")
    terminal = terminal_evidence(completion)
    if not terminal:
        raise ReplayError("Captured actor lacks genuine terminal status evidence")
    after = parse_groups(probe(["/bin/ps", "-A", "-o", "pid=,pgid="]))
    if (
        any(group in completion.process_groups for _, group in after)
        or load_completion_capture(path, sha256) != (parent, completion)
        or terminal_evidence(completion) != terminal
    ):
        raise ReplayError("Owned terminal evidence changed during release verification")
    return {
        "version": 1,
        "kind": "owned-family-resource-release",
        "parent": _parent_record(parent),
        "capture": {"path": str(path), "sha256": sha256},
        "terminalEvidence": dict(terminal),
        "capturedGroupsAbsent": list(completion.process_groups),
        "resourceReleased": True,
        "stageQualityApproved": False,
        "limitations": "Only recorded ownership and terminal facts; no media/domain adoption",
    }


def parse_groups(text: str) -> tuple[tuple[int, int], ...]:
    rows = text.splitlines()
    if not 1 <= len(rows) <= MAX_PROCESSES:
        raise ReplayError("Process group observation exceeds its row budget")
    result: list[tuple[int, int]] = []
    for row in rows:
        values = row.split()
        if len(values) != 2 or any(not value.isdecimal() for value in values):
            raise ReplayError("Malformed process group observation")
        result.append((_integer(int(values[0]), minimum=1), _integer(int(values[1]), minimum=1)))
    if len({pid for pid, _ in result}) != len(result):
        raise ReplayError("Duplicate process group observation")
    return tuple(result)


def _parent_record(parent: Parent) -> dict[str, object]:
    return {
        "pid": parent.pid,
        "birth": parent.birth,
        "executable": str(parent.executable),
        "commandSHA256": parent.command_sha256,
    }


def capture(
    pid: int,
    workflow: Path,
    workflow_sha256: str,
    state: Path,
    stage: str,
    status: Path,
    output: Path,
    *,
    host: SystemHost | None = None,
) -> Path:
    """Capture an actually live actor and its existing running workflow checkpoint.

    The capture authorizes neither a future dispatch nor a completed-stage claim.
    Raw command arguments are hashed in memory and are never retained here.
    """
    current = SystemHost() if host is None else host
    _integer(pid, minimum=1)
    workflow = _directory_path(str(workflow), directory=False, work=True)
    digest = _hash(workflow_sha256)
    _workflow_outputs(workflow, digest, stage)
    state = _directory_path(str(state), work=True)
    checkpoint = _future_file(str(state / f"{stage}.json"))
    status = _future_file(str(status))
    output = _future_file(str(output))
    expected_directory = output.parent.stat()
    initial, initial_digest = _current_bytes(checkpoint)
    raw = record(json.loads(initial, object_pairs_hook=_unique))
    _schema(raw, {"status", "fingerprint"})
    if raw["status"] != "running" or status.exists():
        raise ReplayError("Capture requires a running checkpoint without a terminal status")
    _hash(raw["fingerprint"])
    before = current.identity(pid)
    processes = parse_processes(probe(["/bin/ps", "-A", "-o", "pid=,ppid=,rss="]))
    groups = dict(parse_groups(probe(["/bin/ps", "-A", "-o", "pid=,pgid="])))
    descendants = _descendants(pid, processes)
    if any(child not in groups for child in descendants):
        raise ReplayError("Descendant process group identity is incomplete")
    captured_groups = sorted({groups[child] for child in descendants})
    if current.identity(pid) != before or _current_bytes(checkpoint)[1] != initial_digest:
        raise ReplayError("Live actor or checkpoint changed during capture")
    descriptor = os.open(output.parent, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW)
    try:
        before_directory = os.fstat(descriptor)
        if (before_directory.st_dev, before_directory.st_ino) != (
            expected_directory.st_dev,
            expected_directory.st_ino,
        ):
            raise ReplayError("Capture output ancestor changed before publication")
        _save(
            descriptor,
            output.name,
            {
                "version": 1,
                "kind": "owned-family-completion-capture",
                "observedAtNs": current.now_ns(),
                "parent": _parent_record(before),
                "processGroups": captured_groups,
                "workflow": {"path": str(workflow), "sha256": digest},
                "stage": stage,
                "checkpoint": str(checkpoint),
                "statusPath": str(status),
                "initialCheckpointSHA256": initial_digest,
                "initialCheckpoint": raw,
            },
        )
        final_directory = output.parent.stat()
        if (before_directory.st_dev, before_directory.st_ino) != (
            final_directory.st_dev,
            final_directory.st_ino,
        ):
            raise ReplayError("Capture output ancestor changed")
    finally:
        os.close(descriptor)
    return output


def _policy(value: object) -> Policy:
    raw = record(value)
    required = {
        "newMemoryBytes",
        "newDiskBytes",
        "osBurstMarginBytes",
        "ocrBurstMarginBytes",
        "minimumFreeBytes",
        "maximumPressure",
        "maximumSwapBytes",
        "maximumSwapGrowthBytes",
        "maximumSnapshotAgeSeconds",
        "waitSeconds",
        "pollSeconds",
    }
    if set(raw) not in (required, required | {"inactiveCreditFraction"}):
        raise ReplayError("Incomplete or unknown admission policy")
    credit = raw.get("inactiveCreditFraction", 0)
    if isinstance(credit, bool) or not isinstance(credit, (str, int, float)):
        raise ReplayError("Invalid inactive-page credit")
    try:
        fraction = Fraction(str(credit))
    except (ValueError, ZeroDivisionError) as error:
        raise ReplayError("Invalid inactive-page credit") from error
    if not 0 <= fraction <= 1 or fraction.denominator > 1_000_000_000:
        raise ReplayError("Inactive-page credit is outside its explicit policy")
    pressure = _integer(raw["maximumPressure"], minimum=1, maximum=4)
    if pressure not in (1, 2, 4):
        raise ReplayError("Unknown pressure policy")
    wait = _integer(raw["waitSeconds"], maximum=86400)
    poll = _integer(raw["pollSeconds"], minimum=1, maximum=60)
    if wait // poll + 2 > MAX_POLLS:
        raise ReplayError("Polling exceeds its retained observation budget")
    return Policy(
        _integer(raw["newMemoryBytes"], minimum=1),
        _integer(raw["newDiskBytes"]),
        _integer(raw["osBurstMarginBytes"]),
        _integer(raw["ocrBurstMarginBytes"]),
        _integer(raw["minimumFreeBytes"]),
        fraction,
        pressure,
        _integer(raw["maximumSwapBytes"]),
        _integer(raw["maximumSwapGrowthBytes"]),
        _integer(raw["maximumSnapshotAgeSeconds"], minimum=1, maximum=30),
        wait,
        poll,
    )


def load(path: Path, sha256: str, *, fresh_tool: bool = True) -> Request:
    """Load exact pinned request bytes; never infer authorization from resource fit."""
    path = _directory_path(str(path), directory=False, work=True)
    _identity(path)
    raw = record(
        json.loads(_bound_bytes(path, _hash(sha256), MAX_REQUEST_BYTES), object_pairs_hook=_unique)
    )
    _schema(
        raw,
        {
            "version",
            "kind",
            "name",
            "workDirectory",
            "scratchDirectory",
            "cacheDirectory",
            "diskPaths",
            "executable",
            "command",
            "families",
            "policy",
        },
    )
    if raw["version"] != 1 or raw["kind"] != "measured-job-admission":
        raise ReplayError("Unsupported admission request")
    name = _text(raw["name"], maximum=64)
    if SAFE_NAME.fullmatch(name) is None:
        raise ReplayError("Invalid admission job name")
    disk_paths, command, families = raw["diskPaths"], raw["command"], raw["families"]
    if not isinstance(disk_paths, list) or not 1 <= len(disk_paths) <= 8:
        raise ReplayError("Expected bounded disk measurements")
    if not isinstance(command, list) or not 1 <= len(command) <= 128:
        raise ReplayError("Expected bounded command argument vector")
    if not isinstance(families, list) or len(families) > MAX_FAMILIES:
        raise ReplayError("Expected bounded active family plan")
    executable = _executable(raw["executable"], fresh=fresh_tool)
    argv = tuple(_text(item) for item in command)
    if argv[0] != str(executable.invocation):
        raise ReplayError("Command does not use its pinned executable")
    owners = tuple(_family(item) for item in families)
    if len({family.parent.pid for family in owners}) != len(owners):
        raise ReplayError("Ambiguous duplicate parent declaration")
    work = _directory_path(raw["workDirectory"], work=True)
    scratch = _directory_path(raw["scratchDirectory"], work=True)
    cache = _directory_path(raw["cacheDirectory"], work=True)
    paths = tuple(_directory_path(item) for item in disk_paths)
    if (
        scratch == work
        or cache == work
        or not scratch.is_relative_to(work)
        or not cache.is_relative_to(work)
    ):
        raise ReplayError("Scratch and cache must be explicit work descendants")
    if work not in paths or len(set(paths)) != len(paths):
        raise ReplayError("Disk plan must include work exactly once")
    return Request(
        path,
        sha256,
        name,
        work,
        scratch,
        cache,
        paths,
        executable,
        argv,
        owners,
        _policy(raw["policy"]),
    )


def probe(command: list[str]) -> str:
    """Read one fixed OS query, never logging command output or parent arguments."""
    result = subprocess.run(command, capture_output=True, check=False, timeout=5)
    if result.returncode or len(result.stdout) + len(result.stderr) > MAX_PROBE_BYTES:
        raise ReplayError("Bounded OS resource query failed")
    return result.stdout.decode("utf-8", errors="strict")


def process_executable(pid: int) -> Path:
    """Use the kernel executable path, not a possibly rewritten argv[0]/ps comm.

    Apple's libproc proc_pidpath returns the path length; PROC_PIDPATHINFO_MAXSIZE
    is 4*MAXPATHLEN (4096). Unknown or changed observations remain errors.
    """
    if sys.platform != "darwin":
        raise ReplayError("Process executable observation requires macOS libproc")
    library = ctypes.CDLL("/usr/lib/libproc.dylib", use_errno=True)
    function = library.proc_pidpath
    function.argtypes = [ctypes.c_int, ctypes.c_void_p, ctypes.c_uint32]
    function.restype = ctypes.c_int
    buffer = ctypes.create_string_buffer(PROCESS_PATH_BYTES)
    count = function(pid, buffer, len(buffer))
    if not 0 < count < len(buffer) or len(buffer.value) != count:
        raise ReplayError("Exact process executable path is unavailable")
    return _directory_path(os.fsdecode(buffer.value), directory=False)


def parse_vm_stat(text: str) -> tuple[int, int, int]:
    match = re.search(r"page size of ([0-9]+) bytes", text)
    free = re.findall(r"^Pages free:\s+([0-9]+)\.$", text, re.MULTILINE)
    inactive = re.findall(r"^Pages inactive:\s+([0-9]+)\.$", text, re.MULTILINE)
    if match is None or len(free) != 1 or len(inactive) != 1:
        raise ReplayError("Incomplete vm_stat observation")
    page = _integer(int(match.group(1)), minimum=1, maximum=1024 * 1024)
    if page & (page - 1):
        raise ReplayError("Invalid vm_stat page size")
    return page, _integer(int(free[0])), _integer(int(inactive[0]))


def parse_swap(text: str) -> int:
    number = r"([0-9]+(?:\.[0-9]+)?)"
    match = re.fullmatch(
        rf"total = {number}M\s+used = {number}M\s+free = {number}M(?:\s+\(encrypted\))?\s*", text
    )
    if match is None:
        raise ReplayError("Incomplete swap observation")
    total, used, free = (Decimal(value) for value in match.groups())
    if min(total, used, free) < 0 or used > total:
        raise ReplayError("Invalid swap observation")
    return int(used * 1024 * 1024)


def parse_processes(text: str) -> tuple[Process, ...]:
    rows = text.splitlines()
    if not 1 <= len(rows) <= MAX_PROCESSES:
        raise ReplayError("Process tree exceeds its row budget or is empty")
    result: list[Process] = []
    seen: set[int] = set()
    for row in rows:
        values = row.split()
        if len(values) != 3 or any(not value.isdecimal() for value in values):
            raise ReplayError("Malformed bounded process tree")
        pid, ppid, rss = (int(value) for value in values)
        if pid in seen or pid == ppid:
            raise ReplayError("Ambiguous process tree")
        seen.add(pid)
        result.append(Process(_integer(pid, minimum=1), _integer(ppid), _integer(rss) * 1024))
    return tuple(result)


def _descendants(parent: int, processes: tuple[Process, ...]) -> tuple[int, ...]:
    table = {row.pid: row for row in processes}
    if parent not in table:
        raise ReplayError("Declared active parent is missing")
    children: dict[int, list[int]] = {}
    for row in processes:
        children.setdefault(row.ppid, []).append(row.pid)
    found = {parent}
    pending = [parent]
    while pending:
        child = pending.pop()
        new = children.get(child, [])
        if found.intersection(new):
            raise ReplayError("Cycle in owned process tree")
        found.update(new)
        pending.extend(new)
    return tuple(sorted(found))


def family_usage(
    families: tuple[Family, ...], processes: tuple[Process, ...]
) -> tuple[FamilyUsage, ...]:
    rss = {row.pid: row.rss_bytes for row in processes}
    used: set[int] = set()
    result: list[FamilyUsage] = []
    for family in families:
        descendants = _descendants(family.parent.pid, processes)
        if used.intersection(descendants):
            raise ReplayError("Overlapping active families would double-count RSS")
        used.update(descendants)
        result.append(FamilyUsage(family.parent, descendants, sum(rss[pid] for pid in descendants)))
    return tuple(result)


def assess(
    request: Request, snapshot: Snapshot, *, now_ns: int, baseline_swap: int | None = None
) -> Assessment:
    """Keep pressure, swap, family identity and disk gates independent of memory credit."""
    policy = request.policy
    age = now_ns - snapshot.observed_ns
    if not 0 <= age <= policy.maximum_age_seconds * 1_000_000_000:
        raise ReplayError("Resource snapshot is stale or from the future")
    numbers = (
        snapshot.page_size,
        snapshot.free_pages,
        snapshot.inactive_pages,
        snapshot.swap_bytes,
    )
    if any(type(value) is not int or value < 0 for value in numbers) or snapshot.page_size == 0:
        raise ReplayError("Invalid measured resource numbers")
    if snapshot.pressure not in (1, 2, 4):
        raise ReplayError("Unknown measured pressure")
    if tuple(item.parent for item in snapshot.families) != tuple(
        item.parent for item in request.families
    ):
        raise ReplayError("Snapshot parent identity or order does not match the request")
    if tuple(item.path for item in snapshot.disks) != request.disk_paths:
        raise ReplayError("Snapshot disk scope does not match the request")
    usage = snapshot.families
    for family, item in zip(request.families, usage, strict=True):
        if item.state == "active":
            if (
                not item.pids
                or item.parent.pid not in item.pids
                or item.rss_bytes < 0
                or item.terminal
            ):
                raise ReplayError("Incomplete descendant RSS observation")
        elif (
            family.completion is None
            or item.pids
            or item.rss_bytes != 0
            or item.state == "released"
            and len(item.terminal) != 2
            or item.state == "terminal-pending"
            and item.terminal
        ):
            raise ReplayError("Actor release lacks exact terminal evidence")
        elif item.state not in {"released", "terminal-pending"}:
            raise ReplayError("Unknown actor state")
    pids = tuple(pid for item in usage for pid in item.pids)
    if len(set(pids)) != len(pids):
        raise ReplayError("Overlapping descendant observations")
    remaining = sum(
        max(0, family.peak_bytes - item.rss_bytes)
        for family, item in zip(request.families, usage, strict=True)
        if item.state != "released"
    )
    credit = int(snapshot.page_size * snapshot.inactive_pages * policy.inactive_credit)
    available = snapshot.page_size * snapshot.free_pages + credit
    required = (
        remaining
        + policy.new_memory_bytes
        + policy.os_margin_bytes
        + policy.ocr_margin_bytes
        + policy.minimum_free_bytes
    )
    disk_need = policy.new_disk_bytes + sum(
        family.disk_remaining_bytes
        for family, item in zip(request.families, usage, strict=True)
        if item.state != "released"
    )
    reasons: list[str] = []
    if any(item.state == "terminal-pending" for item in usage):
        reasons.append("family-terminal-pending")
    if available < required:
        reasons.append("memory-deficit")
    if snapshot.pressure > policy.maximum_pressure:
        reasons.append("pressure")
    if snapshot.swap_bytes > policy.maximum_swap_bytes:
        reasons.append("swap-limit")
    if (
        baseline_swap is not None
        and snapshot.swap_bytes - baseline_swap > policy.maximum_swap_growth_bytes
    ):
        reasons.append("swap-growth")
    if any(
        item.floor != disk_budget.floor_bytes(item.total) or item.free < 0 or item.free > item.total
        for item in snapshot.disks
    ):
        raise ReplayError("Invalid measured disk budget")
    if any(item.free - item.floor < disk_need for item in snapshot.disks):
        reasons.append("disk-deficit")
    return Assessment(
        not reasons,
        tuple(reasons),
        {
            "observedAtNs": snapshot.observed_ns,
            "pageSize": snapshot.page_size,
            "freePages": snapshot.free_pages,
            "inactivePages": snapshot.inactive_pages,
            "inactiveCreditFraction": str(policy.inactive_credit),
            "inactiveCreditedBytes": credit,
            "planningAvailableBytes": available,
            "requiredBytes": required,
            "familyRemainingBytes": remaining,
            "swapBytes": snapshot.swap_bytes,
            "baselineSwapBytes": baseline_swap,
            "pressure": snapshot.pressure,
            "diskRequiredBytes": disk_need,
            "families": [
                {
                    "pid": item.parent.pid,
                    "birth": item.parent.birth,
                    "commandSHA256": item.parent.command_sha256,
                    "descendantCount": len(item.pids),
                    "rssBytes": item.rss_bytes,
                    "state": item.state,
                    "terminalEvidence": dict(item.terminal),
                }
                for item in usage
            ],
            "disks": disk_budget.volume_table(snapshot.disks),
            "admitted": not reasons,
            "reasons": reasons,
            "limitations": (
                "RSS is a proxy, inactive credit is a planning estimate; neither is guaranteed "
                "footprint/reclaim. Unlisted jobs and other admission requests are not reserved."
            ),
        },
    )


def _guard(request: Request, *, fresh_tool: bool = False) -> None:
    if load(request.path, request.sha256, fresh_tool=fresh_tool) != request:
        raise ReplayError("Request, executable or in-memory policy changed")


def check(request: Request, *, host: Host | None = None) -> Assessment:
    """Read-only observation; do not reserve names, write records or dispatch."""
    current = SystemHost() if host is None else host
    _guard(request, fresh_tool=True)
    result = assess(request, current.observe(request), now_ns=current.now_ns())
    _guard(request)
    return result


def _parent_guard(descriptor: int, request: Request, expected: os.stat_result) -> None:
    named = _directory_path(str(request.work), work=True).stat()
    opened = os.fstat(descriptor)
    if (opened.st_dev, opened.st_ino) != (expected.st_dev, expected.st_ino) or (
        named.st_dev,
        named.st_ino,
    ) != (expected.st_dev, expected.st_ino):
        raise ReplayError("Admission work directory changed")


def _save(descriptor: int, name: str, value: dict[str, object]) -> None:
    data = (json.dumps(value, allow_nan=False, ensure_ascii=False) + "\n").encode("utf-8")
    if len(data) > MAX_OBSERVATION_BYTES:
        raise ReplayError("Admission record exceeds its byte budget")
    with os.fdopen(
        os.open(
            name, os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW, 0o600, dir_fd=descriptor
        ),
        "wb",
    ) as stream:
        stream.write(data)
        stream.flush()
        os.fsync(stream.fileno())
    os.fsync(descriptor)


def _lease_guard(lease: Lease, request: Request) -> None:
    _parent_guard(lease.parent, request, lease.parent_identity)
    opened = os.fstat(lease.descriptor)
    named = os.stat(f"{request.name}.lease", dir_fd=lease.parent, follow_symlinks=False)
    if (
        not stat.S_ISREG(named.st_mode)
        or named.st_nlink != 1
        or stat.S_IMODE(named.st_mode) != 0o600
        or named.st_uid != os.getuid()
        or (opened.st_dev, opened.st_ino) != (lease.identity.st_dev, lease.identity.st_ino)
        or (named.st_dev, named.st_ino) != (lease.identity.st_dev, lease.identity.st_ino)
    ):
        raise ReplayError("Exclusive admission lease changed")


@contextmanager
def _lease(request: Request) -> Iterator[Lease]:
    parent = os.open(request.work, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW)
    try:
        expected = os.fstat(parent)
        _parent_guard(parent, request, expected)
        descriptor = os.open(
            f"{request.name}.lease",
            os.O_RDWR | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW,
            0o600,
            dir_fd=parent,
        )
        try:
            fcntl.flock(descriptor, fcntl.LOCK_EX | fcntl.LOCK_NB)
            os.fsync(parent)
            lease = Lease(parent, expected, descriptor, os.fstat(descriptor))
            _lease_guard(lease, request)
            yield lease
        finally:
            os.close(descriptor)
    finally:
        os.close(parent)


def _wait(request: Request, host: Host, lease: Lease) -> bool:
    deadline = host.monotonic() + request.policy.wait_seconds
    baseline: int | None = None
    for index in range(MAX_POLLS):
        _guard(request)
        _lease_guard(lease, request)
        snapshot = host.observe(request)
        if baseline is None:
            baseline = snapshot.swap_bytes
        result = assess(request, snapshot, now_ns=host.now_ns(), baseline_swap=baseline)
        _save(lease.parent, f"{request.name}.observation-{index:04d}.json", result.evidence)
        if result.admitted and host.monotonic() <= deadline:
            _guard(request, fresh_tool=True)
            assess(request, snapshot, now_ns=host.now_ns(), baseline_swap=baseline)
            return host.monotonic() <= deadline
        remaining = deadline - host.monotonic()
        if remaining <= 0:
            return False
        host.sleep(min(request.policy.poll_seconds, remaining))
    raise ReplayError("Finite admission observation budget exhausted")


def run(request: Request, *, host: Host | None = None) -> int:
    """Execute once only when separately authorized; retain uncertainty without retry.

    This is a gate around an existing trusted command, not a workflow scheduler.
    Its lease is never deleted; retry/adoption requires independent reconciliation.
    Runtime bounds and original domain/source/output checks remain the workflow's.
    """
    current = SystemHost() if host is None else host
    _guard(request, fresh_tool=True)
    with _lease(request) as lease:
        descriptor = lease.parent
        _save(
            descriptor,
            f"{request.name}.start.json",
            {
                "version": 1,
                "kind": "measured-admission-start",
                "requestSHA256": request.sha256,
                "startedAtNs": current.now_ns(),
                "dispatchAttempted": False,
            },
        )
        attempted = False
        code: int | None = None
        error_kind: str | None = None
        terminal: dict[str, object] | None = None
        try:
            if not _wait(request, current, lease):
                code = 75
            else:
                _lease_guard(lease, request)
                _save(
                    descriptor,
                    f"{request.name}.launch-intent.json",
                    {
                        "requestSHA256": request.sha256,
                        "startedAtNs": current.now_ns(),
                        "uncertainUntilExit": True,
                    },
                )
                attempted = True
                code = current.execute(request)
                _save(descriptor, f"{request.name}.exit.json", {"exitCode": code})
                _guard(request, fresh_tool=True)
                terminal = assess(
                    request, current.observe(request), now_ns=current.now_ns()
                ).evidence
        except (OSError, ReplayError, ValueError, subprocess.SubprocessError) as error:
            error_kind = type(error).__name__
        _lease_guard(lease, request)
        _save(
            descriptor,
            f"{request.name}.end.json",
            {
                "version": 1,
                "kind": "measured-admission-end",
                "requestSHA256": request.sha256,
                "endedAtNs": current.now_ns(),
                "dispatchAttempted": attempted,
                "exitCode": code,
                "errorKind": error_kind,
                "reservationReleased": not attempted or code is not None,
                "uncertainLaunch": attempted and code is None,
                "completed": error_kind is None and attempted and code == 0,
                "terminalResources": terminal,
            },
        )
        if error_kind is not None:
            raise ReplayError(
                "Admission or child outcome failed; retained evidence forbids automatic retry"
            )
        return 1 if code is None else code


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="replay.measured_job_admission")
    parser.add_argument("mode", choices=("check", "run", "capture"))
    parser.add_argument("request", type=Path)
    parser.add_argument("--sha256", required=True)
    parser.add_argument("--state", type=Path)
    parser.add_argument("--stage")
    parser.add_argument("--status", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--pid", type=int)
    args = parser.parse_args(argv)
    try:
        if args.mode == "capture":
            if any(
                item is None
                for item in (args.state, args.stage, args.status, args.output, args.pid)
            ):
                raise ReplayError("Capture requires exact workflow/stage/status/actor/output facts")
            capture(
                args.pid,
                args.request,
                args.sha256,
                args.state,
                args.stage,
                args.status,
                args.output,
            )
            print("OWNED_FAMILY_CAPTURED no resource admission or dispatch")
            return 0
        request = load(args.request, args.sha256)
        if args.mode == "run":
            return run(request)
        result = check(request)
        print(json.dumps(result.evidence, allow_nan=False))
        return 0 if result.admitted else 75
    except (OSError, ReplayError, ValueError, subprocess.SubprocessError):
        print("MEASURED_ADMISSION_ERROR retained evidence must be reconciled", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
