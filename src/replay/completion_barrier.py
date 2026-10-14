"""Bounded read-only completion barrier for an already dispatched workflow.

The barrier never starts or retries the producer or a downstream job. Its proof
binds recorded domain checkpoints; original consumers still verify their media.
"""

from __future__ import annotations

import argparse
import json
import os
import time
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

from replay import measured_job_admission as admission
from replay.core import ReplayError
from replay.foreground_evidence import _bound_bytes
from replay.recordings import record
from replay.stages import Stage, StageError
from replay.workflow import _stage

MAX_BYTES = 1024 * 1024
MAX_STAGES = 512
MAX_CODE_BYTES = 64 * 1024 * 1024
MAX_CODE_TOTAL = 256 * 1024 * 1024
MAX_CODE_INPUTS = 1024
MAX_NESTED_BYTES = 64 * 1024 * 1024
MAX_POLLS = 1024
CODE_SUFFIXES = frozenset({".py", ".sh", ".swift", ".so", ".dylib", ".toml", ".lock"})


@dataclass(frozen=True)
class Pin:
    path: Path
    sha256: str


@dataclass(frozen=True)
class CodePin:
    pin: Pin
    identity: admission.FileIdentity


@dataclass(frozen=True)
class Request:
    path: Path
    sha256: str
    workflow: Pin
    capture: Pin
    state: Path
    status: Path
    last_stage: str
    stages: tuple[Stage, ...]
    code: tuple[CodePin, ...]
    nested: tuple[CodePin, ...]
    wait_seconds: int
    poll_seconds: int
    output: Path
    output_directory: tuple[int, int]


class Host(Protocol):
    def now_ns(self) -> int: ...
    def monotonic(self) -> float: ...
    def sleep(self, seconds: float) -> None: ...
    def groups(self) -> tuple[tuple[int, int], ...]: ...
    def release(self, pin: Pin) -> dict[str, object]: ...


class SystemHost:
    def now_ns(self) -> int:
        return time.time_ns()

    def monotonic(self) -> float:
        return time.monotonic()

    def sleep(self, seconds: float) -> None:
        time.sleep(seconds)

    def groups(self) -> tuple[tuple[int, int], ...]:
        return admission.parse_groups(admission.probe(["/bin/ps", "-A", "-o", "pid=,pgid="]))

    def release(self, pin: Pin) -> dict[str, object]:
        return admission.verify_completion_capture(pin.path, pin.sha256)


def _pin(value: object, *, code: bool = False) -> Pin:
    raw = record(value)
    admission._schema(raw, {"path", "sha256"})
    path = admission._directory_path(raw["path"], directory=False, work=not code)
    admission._identity(path)
    if code and (
        not path.is_relative_to(admission.ROOT)
        or path.suffix not in CODE_SUFFIXES
        and path.name != "METADATA"
    ):
        raise ReplayError("Barrier code pins must be declared checkout code/provider files")
    return Pin(path, admission._hash(raw["sha256"]))


def _json(pin: Pin, maximum: int = MAX_BYTES) -> dict[str, object]:
    return record(
        json.loads(_bound_bytes(pin.path, pin.sha256, maximum), object_pairs_hook=admission._unique)
    )


def _stages(workflow: Pin) -> tuple[Stage, ...]:
    raw = _json(workflow)
    entries = raw.get("stages")
    if (
        raw.get("version") != 1
        or not isinstance(entries, list)
        or not 1 <= len(entries) <= MAX_STAGES
    ):
        raise ReplayError("Barrier requires the whole bounded producer workflow")
    stages = tuple(_stage(item) for item in entries)
    if len({stage.name for stage in stages}) != len(stages) or any(
        not stage.verify_command for stage in stages
    ):
        raise ReplayError("Whole producer stages need unique names and original domain verifiers")
    return stages


def _workflow_paths(stages: tuple[Stage, ...]) -> set[Path]:
    paths: set[Path] = set()
    for stage in stages:
        matches = [
            index
            for index, value in enumerate(stage.command[:-1])
            if value == "-m" and stage.command[index + 1] == "replay.workflow"
        ]
        if not matches:
            continue
        if len(matches) != 1 or matches[0] + 2 >= len(stage.command):
            raise ReplayError("Nested workflow command is ambiguous or lacks its plan")
        path = Path(stage.command[matches[0] + 2])
        if not path.is_absolute() or path.suffix != ".json" or path not in stage.inputs:
            raise ReplayError("Nested workflow must be the exact command and input JSON")
        paths.add(path)
    return paths


def _nested_workflows(
    value: object, stages: tuple[Stage, ...]
) -> tuple[tuple[CodePin, ...], set[Path]]:
    if not isinstance(value, list) or len(value) > MAX_STAGES:
        raise ReplayError("Nested workflow pins must be a bounded list")
    pins = tuple(_pin(entry) for entry in value)
    if len({pin.path for pin in pins}) != len(pins) or {pin.path for pin in pins} != (
        _workflow_paths(stages)
    ):
        raise ReplayError("Nested workflow closure is incomplete, repeated or unlisted")
    nested: list[CodePin] = []
    declared: set[Path] = set()
    total = 0
    for pin in pins:
        identity = admission._identity(pin.path)
        total += identity.size
        if total > MAX_NESTED_BYTES:
            raise ReplayError("Nested workflow closure exceeds its byte budget")
        children = _stages(pin)
        if _workflow_paths(children):
            raise ReplayError("Nested workflow closure is limited to one child level")
        declared.update(path for stage in children for path in stage.inputs)
        nested.append(CodePin(pin, identity))
    return tuple(nested), declared


def load(path: Path, sha256: str, *, fresh_code: bool = True) -> Request:
    """Load exact request/capture/workflow and the declared producer code closure."""
    request_pin = _pin({"path": str(path), "sha256": sha256})
    raw = _json(request_pin)
    admission._schema(
        raw,
        {
            "version",
            "kind",
            "workflow",
            "capture",
            "stateDirectory",
            "terminalStatus",
            "lastStage",
            "codeInputs",
            "waitSeconds",
            "pollSeconds",
            "output",
        }
        | ({"nestedWorkflows"} if "nestedWorkflows" in raw else set()),
    )
    if raw["version"] != 1 or raw["kind"] != "completion-barrier":
        raise ReplayError("Unsupported completion barrier request")
    workflow, capture = _pin(raw["workflow"]), _pin(raw["capture"])
    state = admission._directory_path(raw["stateDirectory"], work=True)
    status = admission._future_file(raw["terminalStatus"])
    stages = _stages(workflow)
    last = admission._text(raw["lastStage"], maximum=64)
    if last not in {stage.name for stage in stages}:
        raise ReplayError("Requested terminal stage is outside the whole workflow")
    _parent, captured = admission.load_completion_capture(capture.path, capture.sha256)
    if (
        captured.workflow != workflow.path
        or captured.workflow_sha256 != workflow.sha256
        or captured.status != status
        or captured.checkpoint.parent != state
    ):
        raise ReplayError("Capture does not bind this producer workflow/state/status")
    entries = raw["codeInputs"]
    if not isinstance(entries, list) or not 1 <= len(entries) <= MAX_CODE_INPUTS:
        raise ReplayError("Barrier requires bounded explicit producer code pins")
    nested, children = _nested_workflows(raw.get("nestedWorkflows", []), stages)
    declared = {path for stage in stages for path in stage.inputs} | children
    expected = {
        path for path in declared if path.suffix in CODE_SUFFIXES or path.name == "METADATA"
    }
    code: list[CodePin] = []
    for entry in entries:
        pin = _pin(entry, code=True)
        identity = admission._identity(pin.path)
        if identity.size > MAX_CODE_BYTES or pin.path not in declared:
            raise ReplayError("Code pin is oversized or outside the producer input closure")
        if fresh_code:
            _bound_bytes(pin.path, pin.sha256, MAX_CODE_BYTES)
        code.append(CodePin(pin, identity))
    if (
        len({item.pin.path for item in code}) != len(code)
        or {item.pin.path for item in code} != expected
    ):
        raise ReplayError("Producer declared code closure is incomplete or repeated")
    if sum(item.identity.size for item in code) > MAX_CODE_TOTAL:
        raise ReplayError("Producer code closure exceeds its byte budget")
    wait = admission._integer(raw["waitSeconds"], maximum=86400)
    poll = admission._integer(raw["pollSeconds"], minimum=1, maximum=60)
    if wait // poll + 2 > MAX_POLLS:
        raise ReplayError("Barrier polling exceeds its finite budget")
    output = admission._future_file(raw["output"])
    if output in {path, workflow.path, capture.path, status} or output.is_relative_to(state):
        raise ReplayError("Barrier proof must be outside the original producer state")
    directory = output.parent.stat()
    return Request(
        request_pin.path,
        request_pin.sha256,
        workflow,
        capture,
        state,
        status,
        last,
        stages,
        tuple(code),
        nested,
        wait,
        poll,
        output,
        (directory.st_dev, directory.st_ino),
    )


def _guard(request: Request, *, fresh_code: bool = False) -> None:
    if load(request.path, request.sha256, fresh_code=fresh_code) != request:
        raise ReplayError("Barrier request or producer code identity changed")


def _checkpoints(request: Request) -> list[dict[str, object]]:
    results: list[dict[str, object]] = []
    for stage in request.stages:
        path = admission._future_file(str(request.state / f"{stage.name}.json"))
        data, digest = admission._current_bytes(path)
        raw = record(json.loads(data, object_pairs_hook=admission._unique))
        admission._schema(raw, {"status", "fingerprint", "outputs"})
        outputs = record(raw["outputs"])
        if raw["status"] != "complete" or set(outputs) != {str(path) for path in stage.outputs}:
            raise ReplayError("Terminal status lacks whole-graph completed domain checkpoints")
        admission._hash(raw["fingerprint"])
        tuple(admission._hash(value) for value in outputs.values())
        results.append(
            {
                "stage": stage.name,
                "path": str(path),
                "sha256": digest,
                "fingerprint": raw["fingerprint"],
                "outputs": outputs,
            }
        )
    return results


def observe(request: Request, host: Host) -> dict[str, object] | None:
    """Return only a stable complete-graph observation; missing status waits."""
    _guard(request)
    if not request.status.exists() and not request.status.is_symlink():
        return None
    data, status_sha = admission._current_bytes(request.status, 32)
    if data not in (b"0", b"0\n"):
        raise ReplayError("Producer failed or has malformed terminal status; no automatic retry")
    checkpoints = _checkpoints(request)
    _parent, captured = admission.load_completion_capture(
        request.capture.path, request.capture.sha256
    )
    if any(group in captured.process_groups for _, group in host.groups()):
        return None
    release = host.release(request.capture)
    if (
        release.get("resourceReleased") is not True
        or release.get("stageQualityApproved") is not False
    ):
        raise ReplayError("Actor release observation is incomplete or claims media approval")
    _guard(request, fresh_code=True)
    if (
        _checkpoints(request) != checkpoints
        or admission._current_bytes(request.status, 32)[1] != status_sha
    ):
        raise ReplayError("Producer completion changed during the terminal observation")
    result: dict[str, object] = {
        "workflow": {"path": str(request.workflow.path), "sha256": request.workflow.sha256},
        "status": {"path": str(request.status), "sha256": status_sha},
        "lastStage": request.last_stage,
        "checkpoints": checkpoints,
        "actorRelease": release,
    }
    if request.nested:
        result["nestedWorkflows"] = [
            {"path": str(item.pin.path), "sha256": item.pin.sha256} for item in request.nested
        ]
    return result


def _publish(request: Request, value: dict[str, object]) -> None:
    data = (json.dumps(value, allow_nan=False, sort_keys=True) + "\n").encode()
    if len(data) > MAX_BYTES:
        raise ReplayError("Completion proof exceeds its byte budget")
    parent = os.open(request.output.parent, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW)
    try:
        before = os.fstat(parent)
        if (before.st_dev, before.st_ino) != request.output_directory:
            raise ReplayError("Completion proof ancestor changed before publication")
        with os.fdopen(
            os.open(
                request.output.name,
                os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW,
                0o600,
                dir_fd=parent,
            ),
            "wb",
        ) as stream:
            stream.write(data)
            stream.flush()
            os.fsync(stream.fileno())
        named = request.output.parent.stat()
        if (before.st_dev, before.st_ino) != (named.st_dev, named.st_ino):
            raise ReplayError("Completion proof ancestor changed")
        os.fsync(parent)
    finally:
        os.close(parent)


def wait(request: Request, *, host: Host | None = None) -> Path:
    """Finite read-only polling, then exclusive proof; never dispatch any job."""
    current = SystemHost() if host is None else host
    _guard(request, fresh_code=True)
    if request.output.exists() or request.output.is_symlink():
        verify(request, host=current)
        return request.output
    deadline = current.monotonic() + request.wait_seconds
    for _index in range(MAX_POLLS):
        result = observe(request, current)
        if result is not None and current.monotonic() <= deadline:
            _publish(
                request,
                {
                    "version": 1,
                    "kind": "completion-barrier-proof",
                    "requestSHA256": request.sha256,
                    "observedAtNs": current.now_ns(),
                    "producer": result,
                    "producerCompleted": True,
                    "downstreamDispatched": False,
                    "qualityApproved": False,
                },
            )
            return request.output
        remaining = deadline - current.monotonic()
        if remaining <= 0:
            raise ReplayError("Completion barrier deadline reached without producer completion")
        current.sleep(min(request.poll_seconds, remaining))
    raise ReplayError("Completion barrier exhausted its finite polling budget")


def verify(request: Request, *, host: Host | None = None) -> None:
    """Read-only proof reuse under unchanged original terminal evidence."""
    current = SystemHost() if host is None else host
    data, digest = admission._current_bytes(request.output, MAX_BYTES)
    proof = record(json.loads(data, object_pairs_hook=admission._unique))
    admission._schema(
        proof,
        {
            "version",
            "kind",
            "requestSHA256",
            "observedAtNs",
            "producer",
            "producerCompleted",
            "downstreamDispatched",
            "qualityApproved",
        },
    )
    if (
        proof["version"] != 1
        or proof["kind"] != "completion-barrier-proof"
        or proof["requestSHA256"] != request.sha256
        or proof["producerCompleted"] is not True
        or proof["downstreamDispatched"] is not False
        or proof["qualityApproved"] is not False
    ):
        raise ReplayError("Invalid completion barrier proof")
    admission._integer(proof["observedAtNs"], minimum=1)
    if proof["producer"] != observe(request, current):
        raise ReplayError("Completed barrier evidence changed or is incomplete")
    if admission._current_bytes(request.output, MAX_BYTES)[1] != digest:
        raise ReplayError("Completion proof changed during read-only verification")


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="replay.completion_barrier")
    parser.add_argument("mode", choices=("wait", "verify"))
    parser.add_argument("request", type=Path)
    parser.add_argument("--sha256", required=True)
    args = parser.parse_args(argv)
    try:
        request = load(args.request, args.sha256)
        if args.mode == "wait":
            wait(request)
        else:
            verify(request)
        print("COMPLETION_BARRIER_VERIFIED no downstream dispatch or quality approval")
        return 0
    except (OSError, ValueError, ReplayError, StageError):
        print("COMPLETION_BARRIER_ERROR inspect retained producer/status evidence")
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
