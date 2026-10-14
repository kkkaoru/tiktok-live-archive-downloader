"""Immutable prepared-generation continuation; no native effects or media processing."""

import argparse
import fcntl
import json
import os
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

from replay import fcp_native_queue as native_queue
from replay import sequence_journal as journal
from replay.native_sequence import Checkpoint, Config, ContractError
from replay.sequence_adapters import pin_value, reference, save_identical
from replay.sequence_journal import Pin, bounded_read, decode, encode
from replay.stages import execution_lock

ROOT = Path(__file__).resolve().parents[2]
WORK = ROOT / "edits/work"
MAX_PARTS = 1024
MAX_AGE_SECONDS = 30


class Host(Protocol):
    def birth(self, pid: int) -> str: ...

    def now_ns(self) -> int: ...


@dataclass(frozen=True)
class Request:
    pin: Pin
    ordered_parts: tuple[int, ...]
    old_root: Path
    generation: Pin
    prepared: Pin
    predecessors: tuple[Pin, ...]
    recovery: Pin
    ledger: Path
    destination: Path


@dataclass(frozen=True)
class Continuation:
    config: Config
    inputs: tuple[Pin, ...]
    predecessors: tuple[Pin, ...]


def _row(value: object) -> dict[str, object]:
    return journal._record(value)


def _pins(value: object, *, nonempty: bool = True) -> tuple[Pin, ...]:
    if not isinstance(value, list) or len(value) > MAX_PARTS or (nonempty and not value):
        raise ContractError("Expected bounded pinned evidence")
    pins = tuple(pin_value(item) for item in value)
    if len({pin.path for pin in pins}) != len(pins):
        raise ContractError("Duplicate pinned evidence")
    return pins


def _directory(value: object) -> Path:
    if not isinstance(value, str):
        raise ContractError("Expected existing repository-local directory")
    path = Path(value)
    if path != path.resolve() or not path.is_relative_to(WORK) or not path.is_dir():
        raise ContractError("Expected existing canonical work directory")
    return path


def _parts(value: object) -> tuple[int, ...]:
    if not isinstance(value, list) or not 1 <= len(value) <= MAX_PARTS:
        raise ContractError("Expected finite ordered parts")
    result: list[int] = []
    for part in value:
        if type(part) is not int or not isinstance(part, int) or part <= 0:
            raise ContractError("Expected positive part identity")
        result.append(part)
    if len(set(result)) != len(result):
        raise ContractError("Duplicate part identity")
    return tuple(result)


def load(pin: Pin) -> Request:
    pin.verify()
    raw = decode(bounded_read(pin.path))
    if (
        set(raw)
        != {
            "version",
            "kind",
            "orderedParts",
            "oldJournal",
            "recovery",
            "ledgerRoot",
            "journalRoot",
        }
        or raw.get("version") != 1
        or raw.get("kind") != "native-sequence-generation-request"
    ):
        raise ContractError("Unsupported continuation request")
    old = _row(raw["oldJournal"])
    if set(old) != {"root", "generation", "prepared", "predecessors"}:
        raise ContractError("Incomplete original journal binding")
    roots = tuple(
        _directory(value) for value in (old["root"], raw["ledgerRoot"], raw["journalRoot"])
    )
    if roots[1] != roots[0].with_name(roots[0].name + "-generation-ledger"):
        raise ContractError("Original journal must use its unique sibling adoption ledger")
    if len(set(roots)) != 3 or any(
        left.is_relative_to(right) for left in roots for right in roots if left != right
    ):
        raise ContractError("Journal and adoption roots must be disjoint")
    return Request(
        pin,
        _parts(raw["orderedParts"]),
        roots[0],
        pin_value(old["generation"]),
        pin_value(old["prepared"]),
        _pins(old["predecessors"], nonempty=False),
        pin_value(raw["recovery"]),
        roots[1],
        roots[2],
    )


def _current(path: Path) -> Pin:
    return Pin(path, reference(path)["sha256"])


def _history(root: Path, part: int, generation: str) -> tuple[Checkpoint | None, tuple[Pin, ...]]:
    paths = tuple(sorted(root.glob(f"part-{part}-*.json")))
    if len(paths) > len(journal.PHASES):
        raise ContractError("Too many original transitions")
    previous, state = "", None
    refs: list[Pin] = []
    for ordinal, path in enumerate(paths):
        if path.name != f"part-{part}-{ordinal:02d}.json":
            raise ContractError("Original checkpoint gap")
        pin = _current(path)
        raw = decode(bounded_read(path))
        if (
            set(raw) != {"previous", "generation", "checkpoint"}
            or raw.get("previous") != previous
            or raw.get("generation") != generation
        ):
            raise ContractError("Original checkpoint chain changed")
        current = journal._checkpoint(raw["checkpoint"])
        journal.DurableJournal._transition(part, state, current)
        refs.append(pin)
        state, previous = current, pin.sha256
    return state, tuple(refs)


@contextmanager
def _old(request: Request) -> Iterator[tuple[Checkpoint, tuple[Pin, ...]]]:
    """Use the existing lock read-only; never create or repair original records."""
    request.pin.verify()
    descriptor = os.open(request.old_root / ".lock", os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK)
    try:
        fcntl.flock(descriptor, fcntl.LOCK_SH | fcntl.LOCK_NB)
        before = journal.identity(os.fstat(descriptor))
        root_before = request.old_root.stat()
        request.generation.verify()
        if request.generation.path != request.old_root / "generation.json":
            raise ContractError("Original generation is outside its journal")
        raw = json.loads(
            bounded_read(request.generation.path),
            object_pairs_hook=journal.unique_pairs,
            parse_constant=journal.invalid_constant,
        )
        original_pins = _pins(raw)
        refs: list[Pin] = [request.pin, request.generation, *original_pins]
        states: list[Checkpoint] = []
        heads: list[Pin] = []
        expected = journal._checkpoint(
            decode(bounded_read(request.prepared.path)).get("checkpoint")
        )
        if expected.config.tail[0] not in request.ordered_parts:
            raise ContractError("Prepared part is not in the original sequence")
        target_index = request.ordered_parts.index(expected.config.tail[0])
        for index, part in enumerate(request.ordered_parts):
            state, history = _history(request.old_root, part, request.generation.sha256)
            refs.extend(history)
            if state is None:
                if index <= target_index:
                    raise ContractError("Original predecessor or prepared state is missing")
                continue
            if state.config.tail != request.ordered_parts[index:]:
                raise ContractError("Original ordered tail changed")
            states.append(state)
            heads.append(history[-1])
        positions = [index for index, pin in enumerate(heads) if pin == request.prepared]
        if len(positions) != 1:
            raise ContractError("Prepared checkpoint is missing or no longer current")
        position = positions[0]
        prepared = states[position]
        if (
            prepared.phase != "prepared"
            or any(state.phase != "released" for state in states[:position])
            or any(state.phase != "prepared" for state in states[position:])
        ):
            raise ContractError("Only an unstarted prepared suffix can continue")
        if tuple(heads[:position]) != request.predecessors:
            raise ContractError("Released predecessor evidence differs")
        original_config = Pin(Path(prepared.config.identity), prepared.config.digest)
        original_config.verify()
        refs.append(original_config)
        for state in states[:position]:
            refs.append(pin_value(decode(state.proof)))
        refs.append(request.prepared)
        yield prepared, tuple(refs)
        if journal.identity((request.old_root / ".lock").lstat()) != before or (
            request.old_root.stat().st_dev,
            request.old_root.stat().st_ino,
        ) != (root_before.st_dev, root_before.st_ino):
            raise ContractError("Original journal ownership changed")
        for pin in refs:
            pin.verify()
    finally:
        os.close(descriptor)


def _target(value: object) -> tuple[int, str]:
    row = _row(value)
    pid, birth = row.get("pid"), row.get("birth")
    if (
        type(pid) is not int
        or not isinstance(pid, int)
        or pid <= 0
        or not isinstance(birth, str)
        or not birth
    ):
        raise ContractError("Missing process identity")
    return pid, birth


def _bindings(
    config: Config, *, target: tuple[int, str], provider_pins: tuple[Pin, ...], original_queue: Pin
) -> tuple[Pin, ...]:
    pump = Pin(Path(config.identity), config.digest)
    pump.verify()
    data = decode(bounded_read(pump.path))
    if data.get("version") != 1 or data.get("kind") != "native-delivery-pump":
        raise ContractError("New configuration is not an original pump")
    queue_pin, admission_pin, runtime_pin = (
        pin_value(data.get(key)) for key in ("queue", "admissionTemplate", "runtimeEvidence")
    )
    queue = decode(bounded_read(queue_pin.path))
    previous_queue = decode(bounded_read(original_queue.path))
    parts = queue.get("parts")
    if (
        not isinstance(parts, list)
        or not 1 <= len(parts) <= MAX_PARTS
        or parts != previous_queue.get("parts")
        or tuple(_row(part).get("assemblyPart") for part in parts) != config.tail
    ):
        raise ContractError("Recovered queue changes original part order or operation bindings")
    admission = decode(bounded_read(admission_pin.path))
    runtime = decode(bounded_read(runtime_pin.path))
    if (
        queue.get("kind") != "executor-native-part-queue"
        or runtime.get("kind") != "native-delivery-runtime-evidence"
        or admission.get("kind") != "native-admission-preparation-request"
        or any(_target(row.get("target")) != target for row in (queue, admission, runtime))
    ):
        raise ContractError("New queue/admission/runtime must bind the recovered process")
    if (
        not isinstance(queue.get("provider"), str)
        or not queue["provider"]
        or runtime.get("provider") != queue["provider"]
        or runtime.get("executor") != queue.get("executor")
    ):
        raise ContractError("Current provider or executable provenance differs")
    capability = pin_value(runtime.get("capabilities"))
    if capability not in provider_pins:
        raise ContractError("Actual provider capability evidence is not observed")
    grant = pin_value(queue.get("guiGrant"))
    grant_data = decode(bounded_read(grant.path))
    if (
        grant_data.get("version") != 1
        or grant_data.get("kind") != "native-gui-idle-handoff"
        or grant_data.get("released") is not True
        or _target(grant_data.get("target")) != target
    ):
        raise ContractError("Supplied GUI grant belongs to the old target")
    return (
        pump,
        queue_pin,
        admission_pin,
        runtime_pin,
        capability,
        grant,
        pin_value(queue.get("executor")),
    )


def _recovery(request: Request, prepared: Checkpoint, host: Host, *, fresh: bool) -> Continuation:
    request.recovery.verify()
    result = decode(bounded_read(request.recovery.path))
    if (
        set(result)
        != {
            "version",
            "kind",
            "intent",
            "quit",
            "afterEvidence",
            "pid",
            "birth",
            "improved",
            "futureData",
            "canDispatch",
        }
        or result.get("version") != 1
        or result.get("kind") != "finite-native-recovery-result"
        or result.get("improved") is not True
        or result.get("canDispatch") is not False
    ):
        raise ContractError("Successful finite recovery result required")
    intent_pin, quit_pin, future_pin = (
        pin_value(result.get(key)) for key in ("intent", "quit", "futureData")
    )
    intent = decode(bounded_read(intent_pin.path))
    expected_config = {"identity": prepared.config.identity, "sha256": prepared.config.digest}
    if (
        intent.get("kind") != "finite-native-recovery-intent"
        or intent.get("config") != expected_config
        or intent.get("reason") not in {"memory-pressure", "resource-budget"}
    ):
        raise ContractError("Recovery does not bind original prepared configuration")
    old_target, target = _target(intent), _target(result)
    original_pump = decode(bounded_read(Path(prepared.config.identity)))
    original_queue = pin_value(original_pump.get("queue"))
    if _target(decode(bounded_read(original_queue.path)).get("target")) != old_target:
        raise ContractError("Recovery original target differs from the original queue")
    try:
        recovered_birth = host.birth(target[0])
    except ProcessLookupError as error:
        raise ContractError("Recovered process is no longer current") from error
    if old_target == target or recovered_birth != target[1]:
        raise ContractError("Recovered process is unchanged or no longer current")
    try:
        original_birth = host.birth(old_target[0])
    except ProcessLookupError:
        original_birth = None
    if original_birth == old_target[1]:
        raise ContractError("Original process still exists")
    quit_payload = native_queue._payload(decode(bounded_read(quit_pin.path)))
    if (
        quit_payload.get("pid") != old_target[0]
        or quit_payload.get("dispatched") is not True
        or type(quit_payload.get("terminated")) is not bool
    ):
        raise ContractError("Original quit effect evidence is unknown or foreign")
    after = _pins(result.get("afterEvidence"))
    future = decode(bounded_read(future_pin.path))
    if (
        set(future) != {"version", "kind", "candidateOnly", "canDispatch", "continuation"}
        or future.get("version") != 1
        or future.get("kind") != "native-sequence-continuation-data"
        or future.get("candidateOnly") is not True
        or future.get("canDispatch") is not False
    ):
        raise ContractError("Typed non-dispatchable future DATA required")
    continuation = _row(future["continuation"])
    if (
        set(continuation) != {"originalConfig", "newConfig", "observation"}
        or continuation["originalConfig"] != expected_config
    ):
        raise ContractError("Future DATA original provenance differs")
    row = _row(continuation["newConfig"])
    if (
        set(row) != {"identity", "tail", "sha256"}
        or not isinstance(row["identity"], str)
        or not isinstance(row["sha256"], str)
    ):
        raise ContractError("New immutable Config is incomplete")
    config = Config(row["identity"], _parts(row["tail"]), row["sha256"])
    if config == prepared.config or config.tail != prepared.config.tail:
        raise ContractError("New configuration must change only the same unstarted suffix")
    observed_pin = pin_value(continuation["observation"])
    if observed_pin not in after:
        raise ContractError("Observation is not pinned by the successful recovery")
    observation = decode(bounded_read(observed_pin.path))
    if (
        observation.get("version") != 1
        or observation.get("kind") != "native-recovery-runtime-observation"
        or _target(observation.get("target")) != target
        or not isinstance(observation.get("project"), str)
        or not observation["project"]
    ):
        raise ContractError("Actual recovered project observation required")
    if any(
        observation.get(key) != intent.get(key) for key in ("savedProject", "progress", "outputs")
    ):
        raise ContractError("Saved project, progress or output evidence changed")
    before_evidence = _pins(intent.get("evidence"))
    before_observations = [decode(bounded_read(pin.path)) for pin in before_evidence]
    prior = [
        row
        for row in before_observations
        if row.get("kind") == "native-recovery-runtime-observation"
    ]
    if (
        len(prior) != 1
        or _target(prior[0].get("target")) != old_target
        or prior[0].get("project") != observation["project"]
    ):
        raise ContractError("Original saved project observation differs")
    instant, age = observation.get("observedAtNs"), observation.get("maximumAgeSeconds")
    if (
        type(instant) is not int
        or not isinstance(instant, int)
        or instant <= 0
        or type(age) is not int
        or not isinstance(age, int)
        or not 1 <= age <= MAX_AGE_SECONDS
        or (fresh and not 0 <= host.now_ns() - instant <= age * 1_000_000_000)
    ):
        raise ContractError("Recovery observation is stale or invalid")
    preserved = (
        pin_value(intent.get("savedProject")),
        pin_value(intent.get("progress")),
        *_pins(intent.get("outputs"), nonempty=False),
        *before_evidence,
    )
    provider = _pins(observation.get("providerEvidence"))
    bindings = _bindings(
        config, target=target, provider_pins=provider, original_queue=original_queue
    )
    return Continuation(
        config,
        (
            request.recovery,
            original_queue,
            intent_pin,
            quit_pin,
            future_pin,
            *after,
            *preserved,
            *provider,
            *bindings,
        ),
        request.predecessors,
    )


def _unique(pins: tuple[Pin, ...]) -> tuple[Pin, ...]:
    by_path: dict[Path, Pin] = {}
    for pin in pins:
        if pin.path in by_path and by_path[pin.path] != pin:
            raise ContractError("Conflicting immutable input pins")
        by_path[pin.path] = pin
    return tuple(by_path.values())


def _code() -> tuple[Pin, ...]:
    return tuple(
        _current(path)
        for path in (
            Path(__file__).resolve(),
            Path(journal.__file__).resolve(),
            ROOT / "src/replay/native_sequence.py",
            ROOT / "src/replay/sequence_adapters.py",
            ROOT / "src/replay/sequence_recovery.py",
            ROOT / "src/replay/fcp_recovery_runtime.py",
            Path(native_queue.__file__).resolve(),
        )
    )


def _claim(request: Request, config: Config, pins: tuple[Pin, ...]) -> dict[str, object]:
    return {
        "version": 1,
        "kind": "native-sequence-generation-intent",
        "request": reference(request.pin.path),
        "prepared": reference(request.prepared.path),
        "recovery": reference(request.recovery.path),
        "journalRoot": str(request.destination),
        "config": {"identity": config.identity, "tail": list(config.tail), "sha256": config.digest},
        "canDispatch": False,
        "inputs": [{"path": str(pin.path), "sha256": pin.sha256} for pin in pins],
    }


def _publish(
    request: Request, prepared: Checkpoint, inputs: tuple[Pin, ...], continuation: Continuation
) -> Path:
    allowed = {
        ".lock",
        "generation.json",
        f"part-{continuation.config.tail[0]}-00.json",
        "continuation.json",
    }
    if any(path.name not in allowed for path in request.destination.iterdir()):
        raise ContractError("New journal has unrelated or started state")
    pins = _unique((*inputs, *continuation.inputs, *_code()))
    claim = request.ledger / f"{request.prepared.sha256}.json"
    if decode(bounded_read(claim)) != _claim(request, continuation.config, pins):
        raise ContractError("Adoption input/code closure changed before publication")
    created = journal.DurableJournal(request.destination, pins)
    checkpoint = Checkpoint(continuation.config, "prepared")
    current = created.get(continuation.config.tail[0])
    if current is None:
        created.put(continuation.config.tail[0], None, checkpoint)
    elif current != checkpoint:
        raise ContractError("New generation is not unstarted prepared")
    output = request.destination / "continuation.json"
    save_identical(
        output,
        {
            "version": 1,
            "kind": "native-sequence-generation",
            "request": reference(request.pin.path),
            "intent": reference(request.ledger / f"{request.prepared.sha256}.json"),
            "originalPrepared": reference(request.prepared.path),
            "releasedPredecessors": [reference(pin.path) for pin in continuation.predecessors],
            "remainingParts": list(prepared.config.tail),
            "generation": reference(request.destination / "generation.json"),
            "inputs": [{"path": str(pin.path), "sha256": pin.sha256} for pin in pins],
            "canDispatch": False,
        },
    )
    for pin in pins:
        pin.verify()
    return output


@contextmanager
def _locations(request: Request) -> Iterator[None]:
    roots = (request.ledger, request.destination)
    identities = tuple((path.stat().st_dev, path.stat().st_ino) for path in roots)
    yield
    if identities != tuple((path.stat().st_dev, path.stat().st_ino) for path in roots) or any(
        path != path.resolve() for path in roots
    ):
        raise ContractError("Adoption directory identity changed")


def adopt(request: Request, *, host: Host, resume: bool = False) -> Path:
    """Reserve one generation, or explicitly finish its exact interrupted DATA adoption."""
    with _old(request) as (prepared, inputs), _locations(request):
        continuation = _recovery(request, prepared, host, fresh=not resume)
        with execution_lock(request.ledger, request.prepared.sha256) as acquired:
            if not acquired:
                raise ContractError("Another adopter owns this prepared checkpoint")
            claim = request.ledger / f"{request.prepared.sha256}.json"
            expected = _claim(
                request, continuation.config, _unique((*inputs, *continuation.inputs, *_code()))
            )
            if resume:
                if decode(bounded_read(claim)) != expected:
                    raise ContractError("Resume must use the exact original adoption intent")
                if (request.destination / "continuation.json").exists():
                    raise ContractError("Adoption already complete; use open_generation")
            else:
                if claim.exists() or claim.is_symlink() or any(request.destination.iterdir()):
                    raise ContractError(
                        "Prepared checkpoint already adopted or destination occupied"
                    )
                journal.publish(claim, encode(expected))
            return _publish(request, prepared, inputs, continuation)


def open_generation(request: Request, *, host: Host) -> journal.DurableJournal:
    """Open the exact completed generation; original live admission still precedes effects."""
    with _old(request) as (prepared, inputs):
        continuation = _recovery(request, prepared, host, fresh=False)
        claim = request.ledger / f"{request.prepared.sha256}.json"
        expected_pins = _unique((*inputs, *continuation.inputs, *_code()))
        if decode(bounded_read(claim)) != _claim(request, continuation.config, expected_pins):
            raise ContractError("Adoption intent differs")
        receipt = decode(bounded_read(request.destination / "continuation.json"))
        if (
            receipt.get("kind") != "native-sequence-generation"
            or receipt.get("request") != reference(request.pin.path)
            or receipt.get("originalPrepared") != reference(request.prepared.path)
            or receipt.get("remainingParts") != list(prepared.config.tail)
            or receipt.get("releasedPredecessors")
            != [reference(pin.path) for pin in continuation.predecessors]
            or receipt.get("intent") != reference(claim)
            or receipt.get("canDispatch") is not False
        ):
            raise ContractError("Completed generation provenance differs")
        pins = _pins(receipt.get("inputs"))
        if pins != expected_pins:
            raise ContractError("Completed generation omits its input closure")
        generation = pin_value(receipt.get("generation"))
        if generation.path != request.destination / "generation.json":
            raise ContractError("Completed generation root differs")
        lock = request.destination / ".lock"
        if not lock.is_file() or lock.is_symlink():
            raise ContractError("Completed generation lock is missing or linked")
        opened = journal.DurableJournal(request.destination, pins)
        state = opened.get(continuation.config.tail[0])
        if state is None or state.config != continuation.config:
            raise ContractError("Completed generation has no matching checkpoint")
        return opened


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("operation", choices=("adopt", "resume", "verify"))
    parser.add_argument("request", type=Path)
    parser.add_argument("--sha256", required=True)
    args = parser.parse_args(argv)
    request = load(Pin(args.request, args.sha256))
    if args.operation == "verify":
        open_generation(request, host=native_queue.SystemHost())
        print(request.destination / "continuation.json")
    else:
        print(adopt(request, host=native_queue.SystemHost(), resume=args.operation == "resume"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
