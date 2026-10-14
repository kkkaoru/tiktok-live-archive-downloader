"""Seal completed child workflows for historical admission owners, without reruns."""

import argparse
import json
import re
import stat
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

from replay import native_retired_hardlink_verification as refs
from replay.core import ReplayError
from replay.fcp_native_queue import SystemHost as ProcessHost
from replay.fcp_part_assembly import _save
from replay.foreground_evidence import _bound_bytes
from replay.input_hash_cache import InputHashCache, input_hash_context
from replay.recordings import record
from replay.stages import _fingerprint, file_digest
from replay.workflow import load_stages

MAX_PINS = 4096
SEAL_KEYS = frozenset(
    {
        "version",
        "kind",
        "owner",
        "process",
        "originalRegistry",
        "capture",
        "producerSHA256",
        "stateDirectory",
        "request",
        "outerStatus",
        "capturedGroups",
        "endpoints",
        "pins",
        "allCapturedChildStagesComplete",
        "qualityApproved",
    }
)


class Host(Protocol):
    def groups(self) -> tuple[tuple[int, int], ...]: ...
    def birth(self, pid: int) -> str: ...


class SystemHost:
    def birth(self, pid: int) -> str:
        return ProcessHost().birth(pid)

    def groups(self) -> tuple[tuple[int, int], ...]:
        result = subprocess.run(
            ["/bin/ps", "-A", "-o", "pid=,pgid="],
            check=True,
            capture_output=True,
            text=True,
            timeout=10,
        )
        rows = tuple(
            tuple(int(value) for value in line.split()) for line in result.stdout.splitlines()
        )
        if any(len(row) != 2 or any(value <= 0 for value in row) for row in rows):
            raise ReplayError("Invalid bounded process group observation")
        return tuple((row[0], row[1]) for row in rows)


@dataclass(frozen=True)
class Parent:
    pid: int
    birth: str


@dataclass(frozen=True)
class Completion:
    workflow: Path
    workflow_sha256: str
    status: Path
    process_groups: tuple[int, ...]
    observed_ns: int


@dataclass(frozen=True)
class CompletedOwner:
    owner: str
    process: dict[str, object]
    original_registry: refs.Reference
    endpoints: dict[str, object]
    pins: tuple[refs.Reference, ...]
    predecessor: refs.Reference | None = None


class Evidence:
    def __init__(self) -> None:
        self.pins: dict[Path, str] = {}

    def read(self, value: object) -> dict[str, object]:
        ref = refs.reference(value)
        if len(self.pins) >= MAX_PINS or (
            ref.path in self.pins and self.pins[ref.path] != ref.sha256
        ):
            raise ReplayError("Excessive completed-owner evidence")
        self.pins[ref.path] = ref.sha256
        try:
            return record(json.loads(_bound_bytes(ref.path, ref.sha256, refs.MAX_BYTES)))
        except ValueError as error:
            raise ReplayError("Invalid completed-owner metadata") from error

    def verify(self) -> None:
        if any(file_digest(path) != digest for path, digest in self.pins.items()):
            raise ReplayError("Completed-owner evidence changed")


def load_capture(path: Path, digest: str) -> tuple[Parent, Completion]:
    raw = Evidence().read({"path": str(path), "sha256": digest})
    parent = record(raw.get("parent"))
    workflow = refs.reference(raw.get("workflow"))
    groups = raw.get("processGroups")
    initial = record(raw.get("initialCheckpoint"))
    pid = parent.get("pid")
    if (
        raw.get("version") != 1
        or raw.get("kind") != "owned-family-completion-capture"
        or type(pid) is not int
        or pid <= 0
        or initial.get("status") != "running"
        or not isinstance(groups, list)
        or not 1 <= len(groups) <= 512
        or any(type(group) is not int or group <= 0 for group in groups)
    ):
        raise ReplayError("Exact live owner capture is required")
    parsed = tuple(int(str(group)) for group in groups)
    if tuple(sorted(set(parsed))) != parsed:
        raise ReplayError("Captured process groups must be unique and ordered")
    observed = raw.get("observedAtNs")
    if type(observed) is not int or observed <= 0:
        raise ReplayError("Captured observation time is required")
    birth = refs._text(parent.get("birth"))
    if _birth_ns(birth) > observed:
        raise ReplayError("Captured parent birth is later than its live observation")
    return Parent(pid, birth), Completion(
        workflow.path, workflow.sha256, refs._local(raw.get("statusPath")), parsed, observed
    )


def _endpoint(path: Path) -> dict[str, object]:
    try:
        info = path.lstat()
    except FileNotFoundError:
        return {"exists": False}
    if not stat.S_ISREG(info.st_mode) or path != path.resolve():
        raise ReplayError("Completed-owner endpoint must be a canonical regular file")
    return {
        "exists": True,
        "device": info.st_dev,
        "inode": info.st_ino,
        "bytes": info.st_size,
        "modifiedAtNs": info.st_mtime_ns,
        "changedAtNs": info.st_ctime_ns,
        "nlink": info.st_nlink,
    }


def _birth_ns(value: str) -> int:
    if re.fullmatch(r"[1-9][0-9]*:[0-9]{1,6}", value) is None:
        raise ReplayError("Exact process birth is unknown")
    seconds, micros = (int(item) for item in value.split(":"))
    return seconds * 1_000_000_000 + micros * 1_000


def _family_snapshot(
    host: Host, *, parent: Parent, completion: Completion
) -> dict[tuple[int, int], str]:
    matching = tuple(
        sorted(
            (pid, group)
            for pid, group in host.groups()
            if pid == parent.pid or group in completion.process_groups
        )
    )
    if len({pid for pid, _ in matching}) != len(matching):
        raise ReplayError("Duplicate family process observation")
    return {item: host.birth(item[0]) for item in matching}


def _different_family(
    rows: dict[tuple[int, int], str], *, parent: Parent, completion: Completion
) -> None:
    for (pid, _), birth in rows.items():
        if pid == parent.pid and (
            birth == parent.birth or _birth_ns(birth) <= completion.observed_ns
        ):
            raise ReplayError("Completed owner's original parent is still present")
    for group in completion.process_groups:
        members = {pid: birth for (pid, pgid), birth in rows.items() if pgid == group}
        if not members:
            continue
        leader = members.get(group)
        if leader is None or _birth_ns(leader) <= completion.observed_ns:
            raise ReplayError("Captured group lacks a proven newly born leader")
        if any(_birth_ns(birth) < _birth_ns(leader) for birth in members.values()):
            raise ReplayError("Captured group has an older member")
    # XNU forkproc excludes live PID, process-group and session identifiers.
    # A new process whose PID equals the old PGID therefore proves reuse only
    # after that old group ended; young orphan children alone are insufficient.


def _absent(host: Host, *, parent: Parent, completion: Completion) -> None:
    before = _family_snapshot(host, parent=parent, completion=completion)
    _different_family(before, parent=parent, completion=completion)
    after = _family_snapshot(host, parent=parent, completion=completion)
    _different_family(after, parent=parent, completion=completion)
    if before != after:
        raise ReplayError("Family membership or birth changed during observation")


def _owner(registry: dict[str, object], name: object) -> dict[str, object]:
    rows = registry.get("owners")
    if not isinstance(rows, list) or registry.get("complete") is not True:
        raise ReplayError("Complete original owner inventory is required")
    matches = [record(row) for row in rows if record(row).get("id") == name]
    if len(matches) != 1 or matches[0].get("role") != "reader":
        raise ReplayError("Exact original reader owner is required")
    return matches[0]


def _paths(registry: dict[str, object], owner: str) -> tuple[Path, ...]:
    rows = registry.get("outputs")
    if not isinstance(rows, list) or not 1 <= len(rows) <= 512:
        raise ReplayError("Bounded original output inventory is required")
    owned = [record(row) for row in rows if record(row).get("owner") == owner]
    if not owned:
        raise ReplayError("Completed owner must retain its original protected rows")
    paths = tuple(refs._local(row.get("path")) for row in owned)
    if len(set(paths)) != len(paths):
        raise ReplayError("Duplicate completed-owner endpoint")
    if any(
        not path.exists() and row.get("state") != "planned"
        for path, row in zip(paths, owned, strict=True)
    ):
        raise ReplayError("Completed-owner protected output is missing")
    return paths


def _pin_map(value: object) -> dict[Path, str]:
    if not isinstance(value, list) or not 1 <= len(value) <= MAX_PINS:
        raise ReplayError("Bounded completed-owner metadata pins are required")
    pins = tuple(refs.reference(item) for item in value)
    if len({pin.path for pin in pins}) != len(pins):
        raise ReplayError("Duplicate completed-owner historical pin")
    return {pin.path: pin.sha256 for pin in pins}


def _request_body(raw: dict[str, object]) -> dict[str, object]:
    body = {
        "version": 1,
        "kind": "native-completed-owner-request",
        "originalRegistry": raw["originalRegistry"],
        "owner": raw["owner"],
        "capture": raw["capture"],
        "stateDirectory": raw["stateDirectory"],
    }
    if raw.get("predecessor") is not None:
        body["predecessor"] = raw["predecessor"]
    return body


def _history(
    evidence: Evidence,
    current: dict[str, object],
    core: dict[Path, str],
    *,
    seen: tuple[Path, ...] = (),
) -> dict[Path, str]:
    value = current.get("predecessor")
    if value is None:
        return {}
    predecessor = refs.reference(value)
    if predecessor.path in seen or len(seen) >= 32:
        raise ReplayError("Cyclic or excessive completed-owner succession")
    old = evidence.read(predecessor.json())
    shared = (
        "owner",
        "process",
        "originalRegistry",
        "capture",
        "stateDirectory",
        "outerStatus",
        "capturedGroups",
        "endpoints",
        "allCapturedChildStagesComplete",
        "qualityApproved",
    )
    producer = old.get("producerSHA256")
    if (
        set(old) not in (SEAL_KEYS, SEAL_KEYS | {"predecessor"})
        or old.get("version") != 1
        or old.get("kind") != "native-completed-owner-seal"
        or not isinstance(producer, str)
        or re.fullmatch(r"[a-f0-9]{64}", producer) is None
        or any(old.get(key) != current.get(key) for key in shared)
    ):
        raise ReplayError("Completed-owner predecessor generation differs")
    request = refs.reference(old.get("request"))
    if evidence.read(request.json()) != _request_body(old):
        raise ReplayError("Historical completed-owner request differs")
    ancestors = _history(evidence, old, core, seen=(*seen, predecessor.path))
    expected = {**core, request.path: request.sha256, **ancestors}
    if _pin_map(old.get("pins")) != expected:
        raise ReplayError("Historical completion workflow or declared output pins differ")
    return {predecessor.path: predecessor.sha256, request.path: request.sha256, **ancestors}


def seal(request: Path, output: Path, *, host: Host) -> CompletedOwner:
    """Validate actual complete child stages and seal immutable endpoint evidence."""
    evidence = Evidence()
    raw = evidence.read(refs._ref(request))
    if (
        set(raw)
        not in (
            {"version", "kind", "originalRegistry", "owner", "capture", "stateDirectory"},
            {
                "version",
                "kind",
                "originalRegistry",
                "owner",
                "capture",
                "stateDirectory",
                "predecessor",
            },
        )
        or raw.get("version") != 1
        or raw.get("kind") != "native-completed-owner-request"
    ):
        raise ReplayError("Expected exact completed-owner request")
    original = refs.reference(raw["originalRegistry"])
    registry = evidence.read(original.json())
    name = refs._text(raw["owner"])
    owner = _owner(registry, name)
    capture = refs.reference(raw["capture"])
    evidence.read(capture.json())
    parent, completion = load_capture(capture.path, capture.sha256)
    process = record(owner.get("process"))
    if process != {"pid": parent.pid, "birth": parent.birth}:
        raise ReplayError("Captured child identity differs from original owner")
    _absent(host, parent=parent, completion=completion)
    status = completion.status.read_text(encoding="utf-8").strip()
    if status not in ("0", "1"):
        raise ReplayError("Actual terminal outer status is required")
    workflow = refs.Reference(completion.workflow, completion.workflow_sha256)
    evidence.read(workflow.json())
    stages = load_stages(workflow.path)
    state = refs._local(raw["stateDirectory"])
    if not state.is_dir() or not 1 <= len(stages) <= 1536:
        raise ReplayError("Bounded actual child checkpoint directory is required")
    paths = _paths(registry, name)
    before = {str(path): _endpoint(path) for path in paths}
    cache_paths = tuple(
        dict.fromkeys(path for stage in stages for path in (*stage.inputs, *stage.outputs))
    )
    cache = InputHashCache(cache_paths, allow_uncached_overflow=True)
    cache.prime()
    with input_hash_context(cache):
        for stage in stages:
            checkpoint = evidence.read(refs._ref(state / f"{stage.name}.json"))
            if checkpoint != {
                "status": "complete",
                "fingerprint": _fingerprint(stage),
                "outputs": {str(path): file_digest(path) for path in stage.outputs},
            }:
                raise ReplayError("Captured child stage is incomplete or its pins changed")
        cache.verify()
    endpoints = {str(path): _endpoint(path) for path in paths}
    if endpoints != before or completion.status.read_text(encoding="utf-8").strip() != status:
        raise ReplayError("Completed-owner endpoints or status changed")
    _absent(host, parent=parent, completion=completion)
    evidence.verify()
    document: dict[str, object] = {
        "version": 1,
        "kind": "native-completed-owner-seal",
        "owner": name,
        "process": process,
        "originalRegistry": original.json(),
        "capture": capture.json(),
        "producerSHA256": file_digest(Path(__file__).resolve()),
        "stateDirectory": str(state),
        "request": refs._ref(request),
        "outerStatus": {
            "path": str(completion.status),
            "sha256": file_digest(completion.status),
            "exitCode": int(status),
        },
        "capturedGroups": list(completion.process_groups),
        "endpoints": endpoints,
        "pins": [refs.Reference(path, digest).json() for path, digest in evidence.pins.items()],
        "allCapturedChildStagesComplete": True,
        "qualityApproved": False,
    }
    if raw.get("predecessor") is not None:
        document["predecessor"] = raw["predecessor"]
    core = _pin_map(document["pins"])
    core.pop(request)
    _history(evidence, document, core)
    evidence.verify()
    document["pins"] = [
        refs.Reference(path, digest).json() for path, digest in evidence.pins.items()
    ]
    _save(output, document)
    return verify(output, owner=name, process=process, host=host)


def verify(path: Path, *, owner: str, process: dict[str, object], host: Host) -> CompletedOwner:
    """Check sealed metadata, absence and exact endpoints without repeating media reads."""
    evidence = Evidence()
    raw = evidence.read(refs._ref(path))
    if (
        set(raw) not in (SEAL_KEYS, SEAL_KEYS | {"predecessor"})
        or raw.get("version") != 1
        or raw.get("kind") != "native-completed-owner-seal"
        or raw.get("owner") != owner
        or raw.get("process") != process
        or raw.get("allCapturedChildStagesComplete") is not True
        or raw.get("qualityApproved") is not False
        or raw.get("producerSHA256") != file_digest(Path(__file__).resolve())
    ):
        raise ReplayError("Completed-owner seal does not bind the exact historical owner")
    capture = refs.reference(raw.get("capture"))
    parent, completion = load_capture(capture.path, capture.sha256)
    if process != {"pid": parent.pid, "birth": parent.birth} or raw.get("capturedGroups") != list(
        completion.process_groups
    ):
        raise ReplayError("Completed-owner capture identity changed")
    status = record(raw.get("outerStatus"))
    if (
        set(status) != {"path", "sha256", "exitCode"}
        or status.get("exitCode") not in (0, 1)
        or status.get("path") != str(completion.status)
        or file_digest(completion.status) != status.get("sha256")
        or completion.status.read_text(encoding="utf-8").strip() != str(status.get("exitCode"))
    ):
        raise ReplayError("Completed-owner terminal status changed")
    original = refs.reference(raw.get("originalRegistry"))
    registry = evidence.read(original.json())
    if record(_owner(registry, owner).get("process")) != process:
        raise ReplayError("Original completed owner identity changed")
    endpoints = record(raw.get("endpoints"))
    if set(endpoints) != {str(path) for path in _paths(registry, owner)}:
        raise ReplayError("Completed-owner endpoint inventory changed")
    pins = raw.get("pins")
    if not isinstance(pins, list) or not 1 <= len(pins) <= MAX_PINS:
        raise ReplayError("Bounded completed-owner metadata pins are required")
    pinned = tuple(refs.reference(value) for value in pins)
    if len({pin.path for pin in pinned}) != len(pinned):
        raise ReplayError("Duplicate completed-owner historical pin")
    declared = {pin.path: pin.sha256 for pin in pinned}
    workflow = refs.Reference(completion.workflow, completion.workflow_sha256)
    request = refs.reference(raw.get("request"))
    request_body = evidence.read(request.json())
    state = refs._local(raw.get("stateDirectory"))
    if request_body != _request_body(raw):
        raise ReplayError("Completed-owner request relationships changed")
    expected_paths = {request.path, original.path, capture.path, workflow.path}
    for stage in load_stages(workflow.path):
        checkpoint_path = state / f"{stage.name}.json"
        expected_paths.add(checkpoint_path)
        digest = declared.get(checkpoint_path)
        if digest is None:
            raise ReplayError("Completed-owner checkpoint pin is missing")
        checkpoint = evidence.read({"path": str(checkpoint_path), "sha256": digest})
        if checkpoint.get("status") != "complete" or set(record(checkpoint.get("outputs"))) != {
            str(output) for output in stage.outputs
        }:
            raise ReplayError("Historical completed child output inventory changed")
    core = {
        path: declared[path] for path in expected_paths if path in declared and path != request.path
    }
    history = _history(evidence, raw, core)
    expected_paths.update(history)
    if (
        set(declared) != expected_paths
        or any(declared.get(path) != digest for path, digest in history.items())
        or any(
            declared.get(ref.path) != ref.sha256 for ref in (request, original, capture, workflow)
        )
    ):
        raise ReplayError("Completed-owner historical pin inventory changed")
    _absent(host, parent=parent, completion=completion)
    if any(_endpoint(refs._local(key)) != value for key, value in endpoints.items()):
        raise ReplayError("Completed-owner retained endpoint changed")
    if any(file_digest(pin.path) != pin.sha256 for pin in pinned):
        raise ReplayError("Completed-owner historical metadata changed")
    evidence.verify()
    _absent(host, parent=parent, completion=completion)
    if any(_endpoint(refs._local(key)) != value for key, value in endpoints.items()):
        raise ReplayError("Completed-owner endpoint changed during verification")
    return CompletedOwner(
        owner,
        process,
        original,
        endpoints,
        pinned,
        None if raw.get("predecessor") is None else refs.reference(raw["predecessor"]),
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("request", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args(argv)
    try:
        seal(args.request.absolute(), args.output.absolute(), host=SystemHost())
    except (ReplayError, OSError, ValueError):
        print("NATIVE_COMPLETED_OWNER_ERROR")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
