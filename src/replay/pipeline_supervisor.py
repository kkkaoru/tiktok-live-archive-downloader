"""Bounded supervision; stdout never authorizes skipping checkpoint verification.

Legacy delivery mode is one-shot: its exception paths cannot prove descendant
death. Workflow mode replays original checkpoints after ordinary failures, at
most three attempts. Signals, unfinished launches and timeouts remain uncertain.
Automatic recovery from SIGKILL is intentionally unsupported, not verified.
"""

from __future__ import annotations

import argparse
import ast
import hashlib
import json
import os
import re
import signal
import subprocess
import sys
import time
from collections.abc import Callable, Sequence
from contextlib import ExitStack
from dataclasses import dataclass
from pathlib import Path
from typing import Literal, Protocol

from replay import delivery_retention, workflow
from replay import whole_technical_batch as technical_batch
from replay.core import ReplayError, write_private_text
from replay.delivery_pipeline import STEP_NAMES
from replay.source_caption_binding import validate_binding
from replay.stages import StageError, execute_command, execution_lock, file_digest

MAX_STALLED = 3
BACKOFF_SECONDS = 30
ROOT = Path(__file__).resolve().parents[2]
WORK = ROOT / "edits/work"
MAX_METADATA = 1024 * 1024


@dataclass(frozen=True)
class Outcome:
    code: int
    completed: tuple[str, ...] = ()
    uncertain: bool = False


@dataclass(frozen=True)
class Technical:
    mode: Literal["run", "verify"]
    sha256: str
    receipt: Path
    logs: Path


class Runner(Protocol):
    def __call__(
        self, command: Sequence[str], *, log: Path, timeout: int, started: Callable[[int], None]
    ) -> Outcome: ...


def _canonical(path: Path, *, scratch: bool = False) -> Path:
    if not path.is_absolute() or path.resolve() != path or ".." in path.parts:
        raise ValueError("Supervisor paths must be canonical")
    if any(part.is_symlink() for part in (path, *path.parents)):
        raise ValueError("Supervisor paths must not contain aliases")
    if not path.is_relative_to(ROOT) or (scratch and not path.is_relative_to(WORK)):
        raise ValueError("Supervisor path is outside repository scope")
    return path


def _metadata(path: Path) -> dict[str, object]:
    _canonical(path, scratch=True)
    if path.stat().st_size > MAX_METADATA:
        raise ValueError("Supervisor metadata exceeds limit")
    value: object = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict) or any(not isinstance(key, str) for key in value):
        raise ValueError("Invalid supervisor metadata")
    return dict(value)


def _steps(log: Path) -> tuple[str, ...]:
    if log.stat().st_size > MAX_METADATA:
        raise ValueError("Supervisor diagnostic log exceeds limit")
    completed: list[str] = []
    for line in log.read_text(encoding="utf-8").splitlines():
        if not line.startswith("STEP "):
            continue
        expected = (
            f"STEP {STEP_NAMES[len(completed)]}: completed"
            if len(completed) < len(STEP_NAMES)
            else ""
        )
        if line != expected:
            raise ValueError("Unknown, repeated or nonmonotonic step output")
        completed.append(STEP_NAMES[len(completed)])
    return tuple(completed)


def _group_alive(pid: int) -> bool:
    try:
        os.killpg(pid, 0)
    except ProcessLookupError:
        return False
    return True


def _run(
    command: Sequence[str], *, log: Path, timeout: int, started: Callable[[int], None]
) -> Outcome:
    with log.open("x", encoding="utf-8") as stream:
        process = subprocess.Popen(
            command,
            stdin=subprocess.DEVNULL,
            stdout=stream,
            stderr=subprocess.STDOUT,
            start_new_session=True,
        )
        started(process.pid)
        try:
            code = process.wait(timeout=timeout)
        except subprocess.TimeoutExpired:
            os.killpg(process.pid, signal.SIGTERM)
            try:
                process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                os.killpg(process.pid, signal.SIGKILL)
                process.wait(timeout=5)
            return Outcome(1, uncertain=True)
    return Outcome(code, _steps(log), code < 0 or _group_alive(process.pid))


def _pins(config: Path, declared: tuple[tuple[Path, str], ...]) -> dict[str, str]:
    paths = (config, *_controller_code_paths())
    result = {str(path): file_digest(_canonical(path)) for path in paths}
    if len(declared) > 256 or len({path for path, _ in declared}) != len(declared):
        raise ValueError("Duplicate or excessive declared inputs")
    for path, expected in declared:
        actual = file_digest(_canonical(path))
        if actual != expected:
            raise ValueError("Declared supervisor input changed")
        result[str(path)] = actual
    return result


def _controller_code_paths() -> tuple[Path, ...]:
    """Bind only current controller import dependencies, including lazy imports."""
    pending = {"pipeline_supervisor", "__init__"}
    found: set[str] = set()
    while pending:
        name = pending.pop()
        if name in found:
            continue
        if re.fullmatch(r"(?:[a-z][a-z0-9_]*|__init__)", name) is None or len(found) >= 512:
            raise ValueError("Controller import closure exceeds scope")
        found.add(name)
        path = _canonical(ROOT / "src/replay" / f"{name}.py")
        if not 0 < path.stat().st_size <= MAX_METADATA:
            raise ValueError("Controller source must be bounded and nonempty")
        for node in ast.walk(ast.parse(path.read_text(encoding="utf-8"))):
            if isinstance(node, ast.ImportFrom) and node.level:
                if node.level != 1:
                    raise ValueError("Controller import escapes the replay package")
                pending.update(
                    (node.module,)
                    if node.module is not None
                    else (alias.name for alias in node.names)
                )
            elif isinstance(node, ast.ImportFrom) and node.module == "replay":
                pending.update(alias.name for alias in node.names)
            elif isinstance(node, ast.ImportFrom) and node.module is not None:
                if node.module.startswith("replay."):
                    pending.add(node.module.removeprefix("replay."))
            elif isinstance(node, ast.Import):
                pending.update(
                    alias.name.removeprefix("replay.")
                    for alias in node.names
                    if alias.name.startswith("replay.")
                )
    return tuple(ROOT / "src/replay" / f"{name}.py" for name in sorted(found))


def _started(record: dict[str, object], attempt: Path) -> Callable[[int], None]:
    def save(pid: int) -> None:
        record.update(status="running", pid=pid, pgid=pid)
        write_private_text(attempt, json.dumps(record))

    return save


def supervise(
    config: Path,
    *,
    state: Path | None = None,
    workflow_state: Path | None = None,
    targets: tuple[str, ...] = (),
    declared: tuple[tuple[Path, str], ...] = (),
    timeout: int = 21600,
    technical: Technical | None = None,
    run: Runner = _run,
    sleep: Callable[[float], None] = time.sleep,
) -> int:
    config = _canonical(config)
    if type(timeout) is not int or not 1 <= timeout <= 86400:
        raise ValueError("Invalid supervisor deadline")
    if technical is None and workflow_state is None and delivery_retention.is_retired(config):
        return 0
    key = hashlib.sha256(str(config).encode("utf-8")).hexdigest()
    state = _canonical(state or WORK / "pipeline-supervisor" / key, scratch=True)
    command = [sys.executable, "-B", "-m", "replay.delivery_pipeline", str(config)]
    if technical is not None:
        if workflow_state is not None or targets or technical.mode not in {"run", "verify"}:
            raise ValueError("Technical and workflow modes must be separate")
        technical_batch.load(config, technical.sha256)
        _canonical(technical.receipt, scratch=True)
        _canonical(technical.logs, scratch=True)
        if state.is_relative_to(technical.logs) or technical.logs.is_relative_to(state):
            raise ValueError("Technical logs and supervisor state must be separate")
        if not technical.receipt.parent.is_dir():
            raise ValueError("Technical receipt parent must exist")
        command = [
            sys.executable,
            "-B",
            "-m",
            "replay.whole_technical_batch",
            technical.mode,
            str(config),
            "--sha256",
            technical.sha256,
            "--receipt",
            str(technical.receipt),
            "--logs",
            str(technical.logs),
        ]
    if targets and workflow_state is None:
        raise ValueError("Targets require workflow mode")
    if workflow_state is not None:
        _canonical(workflow_state, scratch=True)
        if state.is_relative_to(workflow_state) or workflow_state.is_relative_to(state):
            raise ValueError("Supervisor and workflow states must be separate")
        args = [str(config), "--state", str(workflow_state)]
        for target in targets:
            args.extend(("--target", target))
        if workflow.main([*args, "--check"]):
            raise ValueError("Workflow preflight failed")
        command = [sys.executable, "-B", "-m", "replay.workflow", *args]
    leases = _canonical(WORK / "pipeline-supervisor-leases", scratch=True)
    with execution_lock(leases, key) as acquired:
        if not acquired:
            raise StageError("Another supervisor owns this configuration")
        with execution_lock(state, "supervisor") as state_acquired:
            if not state_acquired:
                raise StageError("Another supervisor owns this state")
            return _supervise_locked(
                config,
                state=state,
                command=command,
                declared=declared,
                timeout=timeout,
                retry=workflow_state is not None,
                technical=technical,
                workflow_state=workflow_state,
                targets=targets,
                run=run,
                sleep=sleep,
            )


def _supervise_locked(
    config: Path,
    *,
    state: Path,
    command: list[str],
    declared: tuple[tuple[Path, str], ...],
    timeout: int,
    retry: bool,
    technical: Technical | None,
    workflow_state: Path | None,
    targets: tuple[str, ...],
    run: Runner,
    sleep: Callable[[float], None],
) -> int:
    pins = _pins(config, declared)
    binding = {"version": 1, "inputs": pins, "command": command, "timeout": timeout}
    binding_path = state / "binding.json"
    if binding_path.exists():
        if _metadata(binding_path) != binding:
            raise ValueError("Supervisor configuration or implementation changed")
    else:
        write_private_text(binding_path, json.dumps(binding, sort_keys=True))
    attempts = sorted(state.glob("attempt-*.json"))
    if len(attempts) > MAX_STALLED:
        raise ValueError("Attempt history exceeds retry budget")
    for number, attempt in enumerate(attempts, start=1):
        saved = _metadata(attempt)
        if (
            attempt.name != f"attempt-{number:04d}.json"
            or saved.get("number") != number
            or saved.get("command") != command
        ):
            raise ValueError("Supervisor attempt history is not contiguous and bound")
        if saved.get("status") not in {"failed", "complete"}:
            raise ValueError("Unfinished or uncertain child requires investigated recovery")
        if (
            technical is not None
            and saved.get("status") == "complete"
            and file_digest(technical.receipt) != saved.get("receiptSHA256")
        ):
            raise ValueError("Completed technical receipt changed")
    if attempts and _metadata(attempts[-1]).get("status") == "complete":
        return _reenter(
            config,
            state=state,
            command=command,
            workflow_state=workflow_state,
            targets=targets,
            technical=technical,
            pins=pins,
            declared=declared,
            timeout=timeout,
            run=run,
        )
    while len(attempts) < MAX_STALLED:
        if _pins(config, declared) != pins:
            raise ValueError("Supervisor inputs changed before launch")
        number = len(attempts) + 1
        attempt = state / f"attempt-{number:04d}.json"
        log = state / f"attempt-{number:04d}.log"
        if attempt.exists() or log.exists():
            raise ValueError("Unregistered attempt evidence exists")
        actual = list(command)
        verifying = False
        if technical is not None:
            technical_batch.load(config, technical.sha256)
            verifying = technical.mode == "verify" or technical.receipt.exists()
            if verifying:
                # Bounded metadata only; the child owns full fresh domain checks.
                _metadata(technical.receipt)
            operation = "verify" if verifying else "run"
            technical.logs.mkdir(mode=0o700, exist_ok=True)
            child_logs = technical.logs / f"{operation}-{number:04d}"
            if child_logs.exists() or child_logs.is_symlink():
                raise ValueError("Technical logs already occupied")
            actual[4] = operation
            actual[-1] = str(child_logs)
        record: dict[str, object] = {"status": "launching", "number": number, "command": command}
        record["executedCommand"] = actual
        write_private_text(attempt, json.dumps(record))

        outcome = run(actual, log=log, timeout=timeout, started=_started(record, attempt))
        uncertain = (
            outcome.uncertain
            or outcome.code < 0
            or _pins(config, declared) != pins
            or (outcome.code != 0 and not retry and technical is None)
        )
        if technical is not None and outcome.code == 0:
            _metadata(technical.receipt)
            record["receiptSHA256"] = file_digest(technical.receipt)
        status = "complete" if outcome.code == 0 else "failed"
        if uncertain:
            status = "uncertain"
        record.update(
            status=status,
            code=outcome.code,
            diagnosticSteps=list(outcome.completed),
            verifiedNextStep=None,
        )
        write_private_text(attempt, json.dumps(record))
        attempts.append(attempt)
        if uncertain:
            return 1
        if outcome.code == 0:
            return 0
        if len(attempts) < MAX_STALLED:
            sleep(BACKOFF_SECONDS)
    return 1


def _verify_workflow(config: Path, state: Path, targets: tuple[str, ...], logs: Path) -> None:
    original = workflow._load_workflow(config)
    selected = workflow._selected_graph(original.dependencies, targets)
    stages = tuple(stage for stage in original.stages if stage.name in selected)
    with ExitStack() as stack:
        if not stack.enter_context(execution_lock(state, ".workflow")):
            raise StageError("Workflow is still owned")
        for stage in stages:
            if not stack.enter_context(execution_lock(state, stage.name)):
                raise StageError("Workflow stage is still owned")
            technical_batch._checkpoint(stage, state)
        for stage in stages:
            if stage.name in original.source_caption_bindings:
                validate_binding(
                    original.source_caption_bindings[stage.name],
                    consumer_inputs=stage.inputs,
                    consumer_command=stage.command,
                )
            if stage.verify_command:
                execute_command(
                    stage.verify_command, logs / f"{stage.name}.log", stage.timeout_seconds
                )
        for stage in stages:
            technical_batch._checkpoint(stage, state)


def _reenter(
    config: Path,
    *,
    state: Path,
    command: list[str],
    workflow_state: Path | None,
    targets: tuple[str, ...],
    technical: Technical | None,
    pins: dict[str, str],
    declared: tuple[tuple[Path, str], ...],
    timeout: int,
    run: Runner,
) -> int:
    if technical is None and workflow_state is None:
        raise ValueError("Legacy completion requires an explicit readonly domain adapter")
    history = sorted(state.glob("reentry-*.json"))
    if len(history) >= 256 or any(_metadata(path).get("status") != "complete" for path in history):
        raise ValueError("Previous readonly reentry is uncertain or exceeds its bound")
    number = len(history) + 1
    receipt = state / f"reentry-{number:04d}.json"
    logs = state / f"reentry-{number:04d}"
    if logs.exists() or receipt.exists():
        raise ValueError("Readonly reentry evidence already exists")
    logs.mkdir(mode=0o700)
    record: dict[str, object] = {"status": "running", "command": command, "number": number}
    write_private_text(receipt, json.dumps(record))
    if technical is not None:
        technical_batch.load(config, technical.sha256)
        child_logs = technical.logs / f"reentry-verify-{number:04d}"
        if child_logs.exists() or child_logs.is_symlink():
            raise ValueError("Technical readonly logs are occupied")
        actual = list(command)
        actual[4] = "verify"
        actual[-1] = str(child_logs)
        record["executedCommand"] = actual
        write_private_text(receipt, json.dumps(record))
        outcome = run(
            actual, log=logs / "verify.log", timeout=timeout, started=_started(record, receipt)
        )
        if outcome.code != 0 or outcome.uncertain:
            record.update(status="uncertain", code=outcome.code)
            write_private_text(receipt, json.dumps(record))
            return 1
    elif workflow_state is not None:
        _verify_workflow(config, workflow_state, targets, logs)
    if _pins(config, declared) != pins:
        raise ValueError("Controller inputs changed during readonly verification")
    record.update(status="complete", readonly=True)
    write_private_text(receipt, json.dumps(record))
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("config", type=Path)
    parser.add_argument("--state", type=Path)
    parser.add_argument("--workflow-state", type=Path)
    parser.add_argument("--target", action="append", default=[])
    parser.add_argument("--input", nargs=2, action="append", default=[])
    parser.add_argument("--timeout", type=int, default=21600)
    parser.add_argument("--technical-mode", choices=("run", "verify"))
    parser.add_argument("--manifest-sha256")
    parser.add_argument("--receipt", type=Path)
    parser.add_argument("--logs", type=Path)
    try:
        args = parser.parse_args(argv)
        technical: Technical | None = None
        if args.technical_mode is not None:
            if args.manifest_sha256 is None or args.receipt is None or args.logs is None:
                raise ValueError("Technical mode requires manifest digest, receipt and logs")
            technical = Technical(
                args.technical_mode, args.manifest_sha256, args.receipt, args.logs
            )
        elif any(value is not None for value in (args.manifest_sha256, args.receipt, args.logs)):
            raise ValueError("Technical arguments require explicit mode")
        return supervise(
            args.config.absolute(),
            state=args.state,
            workflow_state=args.workflow_state,
            targets=tuple(args.target),
            declared=tuple((Path(path), sha) for path, sha in args.input),
            timeout=args.timeout,
            technical=technical,
        )
    except SystemExit as error:
        return int(error.code) if isinstance(error.code, int) else 2
    except (ReplayError, StageError, OSError, ValueError) as error:
        print(f"Supervisor stopped: {type(error).__name__}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
