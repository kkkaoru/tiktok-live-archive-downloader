"""Optional finite recovery callback; native actions exist only behind typed adapters."""

import hashlib
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

from replay.native_sequence import Checkpoint, ContractError
from replay.sequence_adapters import reference
from replay.sequence_journal import Pin, bounded_read, decode, encode, publish
from replay.stages import execution_lock


@dataclass(frozen=True)
class Boundary:
    project: str
    pid: int
    birth: str
    # Stable observed-effect epoch, not timestamp or fluctuating memory bytes.
    progress: Pin
    reason: str
    saved_project: Pin
    evidence: tuple[Pin, ...]
    outputs: tuple[Pin, ...]
    pressure: int
    physical_bytes: int
    boot_free: int
    all_nine_idle: bool
    no_modal: bool
    no_export: bool
    responsive: bool
    owner_verified: bool


@dataclass(frozen=True)
class Quit:
    evidence: Pin
    normal_failed: bool
    unresponsive: bool
    gone: bool


class Runtime(Protocol):
    """Binding must use exact deployed Executor tools and original effect readbacks."""

    def observe(self, checkpoint: Checkpoint) -> Boundary: ...

    def verify_current(self, boundary: Boundary) -> None: ...

    def normal_quit(self, boundary: Boundary) -> Quit: ...

    def force_quit(self, boundary: Boundary) -> Quit: ...

    def launch_once(self, previous: Boundary) -> Boundary: ...

    def future_data(self, checkpoint: Checkpoint, before: Boundary, after: Boundary) -> Pin: ...


def _pins(boundary: Boundary) -> None:
    if (
        not boundary.project
        or boundary.pid <= 0
        or not boundary.birth
        or not boundary.evidence
        or boundary.pressure not in (0, 1, 2, 4)
        or boundary.physical_bytes <= 0
        or boundary.boot_free < 0
        or not boundary.owner_verified
        or not boundary.all_nine_idle
        or not boundary.no_modal
        or not boundary.no_export
    ):
        raise ContractError("Recovery requires exact owned saved/idle/non-export boundary")
    for ref in (boundary.progress, boundary.saved_project, *boundary.evidence, *boundary.outputs):
        ref.verify()


class BoundedRecovery:
    """One durable attempt per project/progress/reason; no pump/import/export method."""

    def __init__(self, root: Path, runtime: Runtime) -> None:
        if root != root.resolve() or not root.is_dir():
            raise ContractError("Recovery journal requires an existing canonical directory")
        self.root, self.runtime = root, runtime

    def refused(self, checkpoint: Checkpoint) -> None:
        if checkpoint.phase != "prepared":
            raise ContractError("Recovery cannot reconcile or replay a started effect")
        before = self.runtime.observe(checkpoint)
        _pins(before)
        if before.reason not in ("memory-pressure", "resource-budget"):
            raise ContractError("Recovery needs an observed resource refusal")
        key = hashlib.sha256(
            encode(
                {
                    "project": before.project,
                    "progressSHA256": before.progress.sha256,
                    "reason": before.reason,
                }
            )
        ).hexdigest()
        with execution_lock(self.root, key) as acquired:
            if not acquired:
                raise ContractError("Recovery owner is already active")
            intent = self.root / f"{key}.intent.json"
            if intent.exists() or intent.is_symlink():
                return  # completed, ineffective and uncertain attempts all prohibit repeats
            self.runtime.verify_current(before)
            publish(
                intent,
                encode(
                    {
                        "version": 1,
                        "kind": "finite-native-recovery-intent",
                        "config": {
                            "identity": checkpoint.config.identity,
                            "sha256": checkpoint.config.digest,
                        },
                        "pid": before.pid,
                        "birth": before.birth,
                        "savedProject": reference(before.saved_project.path),
                        "evidence": [reference(ref.path) for ref in before.evidence],
                        "outputs": [reference(ref.path) for ref in before.outputs],
                        "progress": reference(before.progress.path),
                        "reason": before.reason,
                    }
                ),
            )
            quit_result = self.runtime.normal_quit(before)
            quit_result.evidence.verify()
            if not quit_result.gone:
                if not quit_result.normal_failed or not quit_result.unresponsive:
                    raise ContractError(
                        "Normal quit did not prove termination; force is not allowed"
                    )
                quit_result = self.runtime.force_quit(before)
                quit_result.evidence.verify()
                if not quit_result.gone:
                    raise ContractError("Exact previous process is still present")
            after = self.runtime.launch_once(before)
            _pins(after)
            self.runtime.verify_current(after)
            if (
                (before.pid, before.birth) == (after.pid, after.birth)
                or before.project != after.project
                or before.saved_project != after.saved_project
                or before.outputs != after.outputs
                or before.progress != after.progress
                or not after.responsive
            ):
                raise ContractError("New responsive owner must reopen the exact saved project")
            # Physical footprint and pressure remain distinct measured facts.
            improved = (
                after.pressure < before.pressure and after.physical_bytes < before.physical_bytes
            )
            future = self.runtime.future_data(checkpoint, before, after) if improved else None
            if future is not None:
                future.verify()
                candidate = decode(bounded_read(future.path))
                if (
                    candidate.get("canDispatch") is not False
                    or candidate.get("candidateOnly") is not True
                ):
                    raise ContractError(
                        "Recovery may create future DATA only, never a dispatch grant"
                    )
            publish(
                self.root / f"{key}.result.json",
                encode(
                    {
                        "version": 1,
                        "kind": "finite-native-recovery-result",
                        "intent": reference(intent),
                        "quit": reference(quit_result.evidence.path),
                        "afterEvidence": [reference(ref.path) for ref in after.evidence],
                        "pid": after.pid,
                        "birth": after.birth,
                        "improved": improved,
                        "futureData": None if future is None else reference(future.path),
                        "canDispatch": False,
                    }
                ),
            )
