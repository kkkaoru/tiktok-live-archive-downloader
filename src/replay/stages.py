"""Private, fingerprint-bound checkpoints for noninteractive local edit stages."""

import fcntl
import hashlib
import json
import os
import re
import signal
import stat
import subprocess
import tempfile
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

from replay.core import write_private_text

MAX_RECEIPT_BYTES = 1_048_576


class StageError(RuntimeError):
    """A stage cannot safely run or reuse its existing artifacts."""


@dataclass(frozen=True, kw_only=True)
class Stage:
    name: str
    command: tuple[str, ...]
    inputs: tuple[Path, ...]
    outputs: tuple[Path, ...]
    timeout_seconds: int
    verify_command: tuple[str, ...] = ()
    adopt_existing: bool = False

    def __post_init__(self) -> None:
        if re.fullmatch(r"[a-z0-9][a-z0-9_.-]{0,63}", self.name) is None:
            raise StageError("Invalid stage name")
        if not self.command or any(not value or "\0" in value for value in self.command):
            raise StageError("Stage requires a valid argument vector")
        if any(not value or "\0" in value for value in self.verify_command):
            raise StageError("Invalid verification argument vector")
        if type(self.timeout_seconds) is not int or not 1 <= self.timeout_seconds <= 86400:
            raise StageError("Stage timeout must be between 1 and 86400 seconds")
        if not self.inputs or not self.outputs:
            raise StageError("Explicit inputs and outputs are required")
        paths = self.inputs + self.outputs
        if any(not path.is_absolute() for path in paths) or len(
            {path.resolve() for path in paths}
        ) != len(paths):
            raise StageError("Stage paths must be absolute, distinct and non-overlapping")
        if self.adopt_existing and not self.verify_command:
            raise StageError("Adoption requires an explicit read-only provenance verifier")


def file_digest(path: Path) -> str:
    """Reject links and files changed during hashing; never read directory trees."""
    before = path.lstat()
    if not stat.S_ISREG(before.st_mode) or before.st_size == 0:
        raise StageError("Expected a nonempty regular stage artifact")
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        opened = os.fstat(stream.fileno())
        if (opened.st_dev, opened.st_ino) != (before.st_dev, before.st_ino):
            raise StageError("Stage artifact changed before hashing")
        while block := stream.read(1_048_576):
            digest.update(block)
    after = path.lstat()
    identity = (before.st_dev, before.st_ino, before.st_size, before.st_mtime_ns)
    if identity != (after.st_dev, after.st_ino, after.st_size, after.st_mtime_ns):
        raise StageError("Stage artifact changed during hashing")
    return digest.hexdigest()


def _fingerprint(stage: Stage) -> str:
    definition = {
        "command": stage.command,
        "verifyCommand": stage.verify_command,
        "inputs": {str(path): file_digest(path) for path in stage.inputs},
        "outputs": [str(path) for path in stage.outputs],
        "timeoutSeconds": stage.timeout_seconds,
        "workingDirectory": str(Path.cwd().resolve()),
    }
    return hashlib.sha256(json.dumps(definition, sort_keys=True).encode("utf-8")).hexdigest()


def execute_command(command: tuple[str, ...], log: Path, timeout: int) -> None:
    """No shell or interactive stdin; a deadline stops only this child group."""
    with log.open("xb") as stream:
        os.fchmod(stream.fileno(), 0o600)
        with subprocess.Popen(
            command,
            stdin=subprocess.DEVNULL,
            stdout=stream,
            stderr=subprocess.STDOUT,
            start_new_session=True,
        ) as child:
            try:
                code = child.wait(timeout=timeout)
            finally:
                if child.poll() is None:
                    try:
                        os.killpg(child.pid, signal.SIGKILL)
                    except ProcessLookupError:
                        # The group may exit between poll() and killpg().
                        child.poll()
                    child.wait()
            if code != 0:
                raise StageError(f"Stage command failed; inspect {log}")


def _read_receipt(path: Path) -> object:
    if path.is_symlink() or path.stat().st_size > MAX_RECEIPT_BYTES:
        raise StageError("Invalid stage receipt")
    data: object = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict) or data.get("status") not in ("running", "complete"):
        raise StageError("Invalid stage receipt schema")
    fingerprint: object = data.get("fingerprint")
    if not isinstance(fingerprint, str) or re.fullmatch(r"[a-f0-9]{64}", fingerprint) is None:
        raise StageError("Invalid stage receipt fingerprint")
    return data


def run_stage(
    stage: Stage,
    *,
    state_directory: Path,
    execute: Callable[[tuple[str, ...], Path, int], None] = execute_command,
) -> Literal["completed", "reused", "adopted", "busy"]:
    """Reuse verified outputs, or recover only through a read-only verifier.

    Include helper scripts/model files in inputs, not merely their argument paths.
    A changed/missing completed output stops; it never authorizes a rerun. Failed
    or interrupted mutations are not retried. Their outputs may be adopted only
    after the supplied domain verifier proves their input-bound provenance.
    """
    if state_directory.is_symlink():
        raise StageError("State directory must not be a symlink")
    state_directory.mkdir(parents=True, exist_ok=True, mode=0o700)
    state_directory.chmod(0o700)
    lock_path = state_directory / f"{stage.name}.lock"
    descriptor = os.open(lock_path, os.O_RDWR | os.O_CREAT | os.O_NOFOLLOW, 0o600)
    with os.fdopen(descriptor, "a+b") as lock:
        os.fchmod(lock.fileno(), 0o600)
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            return "busy"
        return _run_locked(stage, state_directory=state_directory, execute=execute)


def _run_locked(
    stage: Stage,
    *,
    state_directory: Path,
    execute: Callable[[tuple[str, ...], Path, int], None],
) -> Literal["completed", "reused", "adopted"]:
    receipt = state_directory / f"{stage.name}.json"
    fingerprint = _fingerprint(stage)
    running: dict[str, object] = {"status": "running", "fingerprint": fingerprint}
    saved = _read_receipt(receipt) if receipt.exists() or receipt.is_symlink() else None
    if saved is not None and saved != running:
        expected = {
            "status": "complete",
            "fingerprint": fingerprint,
            "outputs": {str(path): file_digest(path) for path in stage.outputs},
        }
        if saved != expected:
            raise StageError("Completed stage or its inputs changed; refusing to overwrite")
        if stage.verify_command:
            # A manifest hash alone cannot validate the artifacts it references.
            attempt = Path(tempfile.mkdtemp(prefix=f"{stage.name}-reuse-", dir=state_directory))
            execute(stage.verify_command, attempt / "verify.log", stage.timeout_seconds)
        return "reused"
    recovering = saved is not None or any(
        path.exists() or path.is_symlink() for path in stage.outputs
    )
    if recovering and not stage.verify_command:
        raise StageError("Partial outputs require a read-only provenance verifier")
    if saved is None and recovering and not stage.adopt_existing:
        raise StageError("Unregistered outputs require explicit verified adoption")
    write_private_text(receipt, json.dumps(running))
    attempt = Path(tempfile.mkdtemp(prefix=f"{stage.name}-", dir=state_directory))
    if not recovering:
        execute(stage.command, attempt / "run.log", stage.timeout_seconds)
    if stage.verify_command:
        execute(stage.verify_command, attempt / "verify.log", stage.timeout_seconds)
    if _fingerprint(stage) != fingerprint:
        raise StageError("Stage changed its immutable inputs")
    complete = {
        "status": "complete",
        "fingerprint": fingerprint,
        "outputs": {str(path): file_digest(path) for path in stage.outputs},
    }
    write_private_text(receipt, json.dumps(complete, sort_keys=True))
    return "adopted" if recovering else "completed"
