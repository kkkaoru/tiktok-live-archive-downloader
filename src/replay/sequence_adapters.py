"""Private adapters to original APIs; importing this module causes no effects."""

import argparse
import hashlib
import os
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

from replay import fcp_native_admission as admission
from replay import fcp_native_delivery_pump as pump
from replay import fcp_native_queue as queue
from replay import fcp_native_queue_followup as followup
from replay import workflow
from replay.core import ReplayError
from replay.native_export_receipt import save_new
from replay.native_sequence import Checkpoint, Config, ContractError
from replay.part_master_publication import verify_retirement
from replay.sequence_journal import Pin, _record, bounded_read, decode, encode, publish


def reference(path: Path) -> dict[str, str]:
    pin = Pin(path, hashlib.sha256(bounded_read(path)).hexdigest())
    pin.verify()
    return {"path": str(path), "sha256": pin.sha256}


def save_identical(path: Path, value: object) -> None:
    data = encode(value)
    if path.exists() or path.is_symlink():
        if bounded_read(path) != data:
            raise ContractError("Existing prepared configuration differs")
        return
    publish(path, data)


@dataclass(frozen=True)
class Materialization:
    preparation_request: Pin
    candidate_root: Path
    queue_path: Path
    queue_fields: dict[str, object]
    pump_path: Path
    pump_fields: dict[str, object]
    tail: tuple[int, ...]


class PreparationAPI(Protocol):
    def candidate(self, request: Path, output: Path) -> Path: ...

    def reuse_candidate(self, request: Path, output: Path) -> Path: ...

    def check_queue(self, path: Path) -> None: ...

    def check_pump(self, path: Path) -> None: ...


class OriginalPreparation:
    def candidate(self, request: Path, output: Path) -> Path:
        return queue.prepare_data(request, output)

    def reuse_candidate(self, request: Path, output: Path) -> Path:
        candidate = output / "candidate.json"
        marker = decode(bounded_read(output / "request.json"))
        if marker != {"request": reference(request)}:
            raise ContractError("Prepared candidate belongs to another request")
        document = decode(bounded_read(candidate))
        inputs = document.get("inputs")
        if not isinstance(inputs, list):
            raise ContractError("Candidate input pins missing")
        for item in inputs:
            ref = _record(item)
            path, digest = ref.get("path"), ref.get("sha256")
            if not isinstance(path, str) or not isinstance(digest, str):
                raise ContractError("Candidate pin malformed")
            Pin(Path(path), digest).verify()
        queue.validate_candidate(candidate)
        return candidate

    def check_queue(self, path: Path) -> None:
        queue.load(path)

    def check_pump(self, path: Path) -> None:
        pump.load(path)

    def predecessor(self, config: Path, status: Path) -> None:
        if bounded_read(status).strip() != b"0":
            raise ContractError("Predecessor post-delete consumer is incomplete")
        verify_retirement(config)


class DeferredAPI(PreparationAPI, Protocol):
    def predecessor(self, config: Path, status: Path) -> None: ...


def path_value(value: object) -> Path:
    if not isinstance(value, str):
        raise ContractError("Expected absolute canonical artifact path")
    path = Path(value)
    if not path.is_absolute() or path != path.resolve():
        raise ContractError("Noncanonical artifact path")
    return path


def pin_value(value: object) -> Pin:
    row = _record(value)
    digest = row.get("sha256")
    if set(row) != {"path", "sha256"} or not isinstance(digest, str):
        raise ContractError("Expected exact pinned reference")
    result = Pin(path_value(row["path"]), digest)
    result.verify()
    return result


def substitute(value: object, dependencies: dict[str, Pin], *, depth: int = 0) -> object:
    if depth > 32:
        raise ContractError("Template exceeds bounded depth")
    if isinstance(value, dict):
        row = _record(value)
        if "$ref" in row:
            name = row["$ref"]
            if set(row) != {"$ref"} or not isinstance(name, str) or name not in dependencies:
                raise ContractError("Unknown or ambiguous deferred reference")
            dependency = dependencies[name]
            dependency.verify()
            return {"path": str(dependency.path), "sha256": dependency.sha256}
        return {key: substitute(item, dependencies, depth=depth + 1) for key, item in row.items()}
    if isinstance(value, list):
        return [substitute(item, dependencies, depth=depth + 1) for item in value]
    return value


def prepare_deferred(request: Pin, part: int, api: DeferredAPI) -> Config:
    """Bind future refs only after original predecessor retirement verification.

    No media, GUI, grant or capture is generated. Missing future evidence is a
    bounded failure; rerunning after real producer completion preserves outputs.
    """
    request.verify()
    raw = decode(bounded_read(request.path))
    if (
        set(raw) != {"version", "kind", "members"}
        or raw.get("version") != 1
        or raw.get("kind") != "native-sequence-data"
    ):
        raise ContractError("Unsupported finite sequence DATA schema")
    rows = raw.get("members")
    if not isinstance(rows, list) or not 1 <= len(rows) <= 1024:
        raise ContractError("Expected bounded member list")
    members = [_record(row) for row in rows]
    identities = [row.get("part") for row in members]
    if any(type(value) is not int or value <= 0 for value in identities) or len(
        set(identities)
    ) != len(identities):
        raise ContractError("Member identities must be positive and unique")
    if part not in identities:
        raise ContractError("Requested member is not in sequence")
    index = identities.index(part)
    row = members[index]
    required = {
        "part",
        "preparationRequest",
        "candidateRoot",
        "queuePath",
        "queueTemplate",
        "pumpPath",
        "pumpTemplate",
        "dependencies",
        "predecessor",
    }
    if set(row) != required:
        raise ContractError("Member DATA shape differs")
    predecessor = row["predecessor"]
    if index == 0:
        if predecessor is not None:
            raise ContractError("Initial member must use actual initial owner inventory")
    else:
        prior = _record(predecessor)
        if (
            set(prior) != {"part", "retirementConfig", "postDeleteStatus"}
            or prior["part"] != identities[index - 1]
        ):
            raise ContractError("Wrong predecessor identity")
        api.predecessor(
            path_value(prior["retirementConfig"]), path_value(prior["postDeleteStatus"])
        )
    paths = _record(row["dependencies"])
    if len(paths) > 1024:
        raise ContractError("Too many deferred references")
    dependencies = {
        name: Pin(path_value(value), reference(path_value(value))["sha256"])
        for name, value in paths.items()
    }
    queue_template, pump_template = pin_value(row["queueTemplate"]), pin_value(row["pumpTemplate"])
    tail: list[int] = []
    for value in identities[index:]:
        if not isinstance(value, int):
            raise ContractError("Invalid tail")
        tail.append(value)
    spec = Materialization(
        pin_value(row["preparationRequest"]),
        path_value(row["candidateRoot"]),
        path_value(row["queuePath"]),
        _record(substitute(decode(bounded_read(queue_template.path)), dependencies)),
        path_value(row["pumpPath"]),
        _record(substitute(decode(bounded_read(pump_template.path)), dependencies)),
        tuple(tail),
    )
    result = materialize(spec, api)
    for dependency in (*dependencies.values(), queue_template, pump_template, request):
        dependency.verify()
    return result


def materialize(spec: Materialization, api: PreparationAPI) -> Config:
    """Compose exact existing schemas; supplied grants/runtime are never invented."""
    spec.preparation_request.verify()
    occupied = spec.candidate_root / "candidate.json"
    candidate = (
        api.reuse_candidate(spec.preparation_request.path, spec.candidate_root)
        if occupied.exists() or occupied.is_symlink()
        else api.candidate(spec.preparation_request.path, spec.candidate_root)
    )
    raw = decode(bounded_read(candidate))
    if raw.get("canDispatch") is not False or raw.get("candidateOnly") is not True:
        raise ContractError("Expected original data-only candidate")
    rows = raw.get("parts")
    if not isinstance(rows, list):
        raise ContractError("Candidate parts missing")
    parts = [_record(row) for row in rows]
    if tuple(row.get("assemblyPart") for row in parts) != spec.tail:
        raise ContractError("Original manifest order differs from immutable tail")
    if {"parts", "version", "kind", "corePins"} & spec.queue_fields.keys():
        raise ContractError("Runtime fields cannot override candidate parts/pins/schema")
    inputs = raw.get("inputs")
    if not isinstance(inputs, list):
        raise ContractError("Candidate code/input pins missing")
    save_identical(
        spec.queue_path,
        {
            **spec.queue_fields,
            "version": 1,
            "kind": "executor-native-part-queue",
            "parts": parts,
            "corePins": inputs,
        },
    )
    api.check_queue(spec.queue_path)
    if {"queue", "version", "kind"} & spec.pump_fields.keys():
        raise ContractError("Pump fields cannot override queue/schema")
    save_identical(
        spec.pump_path,
        {
            **spec.pump_fields,
            "version": 1,
            "kind": "native-delivery-pump",
            "queue": reference(spec.queue_path),
        },
    )
    api.check_pump(spec.pump_path)
    spec.preparation_request.verify()
    return Config(str(spec.pump_path), spec.tail, reference(spec.pump_path)["sha256"])


class ProductionAPI:
    """Explicit live API wrapper; synthetic tests inject a separate implementation."""

    def __init__(
        self,
        workflow_runner: Callable[[list[str]], int] = workflow.main,
        loader: Callable[[Path], pump.Pump] = pump.load,
    ) -> None:
        self.workflow_runner = workflow_runner
        self.load = loader

    def run_pump(self, config: Pin, transport: queue.Transport, host: admission.Host) -> Path:
        config.verify()
        result = pump.run(self.load(config.path), transport=transport, host=host)
        if result.state != "published-candidate" or result.evidence is None:
            raise ContractError("Original pump has no canonical publication terminal")
        config.verify()
        return result.evidence

    def run_retirement(self, graph: Pin, state: Path, terminal: Path) -> Pin:
        graph.verify()
        arguments = [str(graph.path), "--state", str(state)]
        if self.workflow_runner([*arguments, "--check"]) != 0:
            raise ContractError("Original retirement graph preflight failed")
        if self.workflow_runner(arguments) != 0:
            raise ContractError("Resume original retirement state; no new deletion graph")
        graph.verify()
        return Pin(terminal, reference(terminal)["sha256"])

    def verify_release(self, request: Pin, host: queue.Host) -> followup.Release:
        request.verify()
        result = followup.verify_release(request.path, host=host)
        request.verify()
        return result

    def refresh(
        self, config: Pin, transport: queue.Transport, host: admission.Host, name: str
    ) -> Path:
        """Requires an already genuinely owned lease; never creates a GUI grant."""
        config.verify()
        plan = self.load(config.path)
        if not name or Path(name).name != name or name in {".", ".."}:
            raise ContractError("Fresh observation child name must be one component")
        pump._refresh(plan, transport, host, name)
        config.verify()
        return plan.root / name / "snapshot.json"

    def observe_owned_capacity(
        self, config: Pin, transport: queue.Transport, host: admission.Host, name: str
    ) -> Path:
        """Acquire one real granted observation lease; close only that exact lease."""
        config.verify()
        plan = self.load(config.path)
        lease = queue.GUI_LEASE_ROOT / ".fcp-native-queue-owner.json"
        save_new(
            lease,
            {
                "pid": os.getpid(),
                "birth": host.birth(os.getpid()),
                "queueSHA256": plan.native.digest,
            },
        )
        owned_path, owned_digest = pump._lease(plan, host)
        # No finally-unlink: uncertain GUI observation retains its owned lease.
        try:
            snapshot = self.refresh(config, transport, host, name)
        except ReplayError:
            # A retained explicit admission denial is not an unknown GUI effect.
            # _refresh creates this only after successful final background-close.
            snapshot = plan.root / name / "snapshot.json"
            if not snapshot.is_file() or snapshot.is_symlink():
                raise
            denial = decode(bounded_read(snapshot))
            if (
                denial.get("admitted") is not False
                or denial.get("target") != plan.template.get("target")
                or denial.get("registry") != plan.template.get("registry")
            ):
                raise
            config.verify()
        if pump._lease(plan, host) != (owned_path, owned_digest):
            raise ContractError("Observation lease changed")
        owned_path.unlink()
        return snapshot


@dataclass(frozen=True)
class Member:
    materialization: Materialization
    retirement_graph: Pin
    retirement_state: Path
    retirement_terminal: Path


class ReleaseFactory(Protocol):
    def request(
        self, current: Config, successor: Config, retirement: Pin, capacity: Pin
    ) -> Pin: ...


@dataclass(frozen=True)
class ReleaseInputs:
    output: Path
    artifacts: dict[str, Path]
    history: tuple[Path, ...]
    successor_queue: Path


class OriginalReleaseFactory:
    """Bind actual completed artifacts to the existing release request schema."""

    def __init__(self, members: dict[tuple[int, ...], ReleaseInputs]) -> None:
        self.members = members

    def request(self, current: Config, successor: Config, retirement: Pin, capacity: Pin) -> Pin:
        if current.tail[1:] != successor.tail:
            raise ContractError("Release changes original remaining tail")
        spec = self.members[current.tail]
        expected = {
            "preparation",
            "terminal",
            "publication",
            "retirement",
            "postDeleteStatus",
            "retirementAudit",
            "consumer",
        }
        if set(spec.artifacts) != expected or not spec.history:
            raise ContractError("Complete original retirement/publication closure required")
        retirement.verify()
        capacity.verify()
        if spec.artifacts["retirement"] != retirement.path:
            raise ContractError("Release retirement differs from completed workflow artifact")
        value = {
            "version": 1,
            "kind": "native-queue-nextpart-release-request",
            **{name: reference(path) for name, path in spec.artifacts.items()},
            "history": [reference(path) for path in spec.history],
            "capacity": {"path": str(capacity.path), "sha256": capacity.sha256},
            "successorQueue": reference(spec.successor_queue),
            "successorOrdinal": 1,
        }
        # Capacity is a fresh observation, not immutable completion evidence.
        # Preserve a previous attempt while binding the next exact observation.
        output = spec.output
        if output.exists() and decode(bounded_read(output)) != value:
            previous = decode(bounded_read(output))
            if {k: v for k, v in previous.items() if k != "capacity"} != {
                k: v for k, v in value.items() if k != "capacity"
            }:
                raise ContractError("Release closure changed beyond fresh capacity")
            output = output.with_name(f"{output.stem}-{capacity.sha256}{output.suffix}")
        save_identical(output, value)
        return Pin(output, reference(output)["sha256"])


class BoundSteps:
    """Executable controller adapter. Release DATA factory remains explicitly supplied."""

    def __init__(
        self,
        members: tuple[Member, ...],
        transport: queue.Transport,
        host: admission.Host,
        release_factory: ReleaseFactory,
        api: ProductionAPI | None = None,
        preparation: PreparationAPI | None = None,
    ) -> None:
        self.members = {member.materialization.tail: member for member in members}
        if len(self.members) != len(members):
            raise ContractError("Duplicate immutable tails")
        self.transport, self.host = transport, host
        self.release_factory = release_factory
        self.api = ProductionAPI() if api is None else api
        self.preparation = OriginalPreparation() if preparation is None else preparation
        self.snapshots: dict[tuple[int, ...], Pin] = {}

    def _pin(self, config: Config) -> Pin:
        path = self.members[config.tail].materialization.pump_path
        if str(path) != config.identity:
            raise ContractError("Pump identity changed")
        result = Pin(path, config.digest)
        result.verify()
        return result

    def materialize(self, tail: tuple[int, ...]) -> Config:
        return materialize(self.members[tail].materialization, self.preparation)

    def validate(self, checkpoint: Checkpoint) -> None:
        self._pin(checkpoint.config)
        if checkpoint.phase in {"published", "retired", "released"}:
            row = decode(checkpoint.proof)
            path, digest = row.get("path"), row.get("sha256")
            if not isinstance(path, str) or not isinstance(digest, str):
                raise ContractError("Checkpoint proof must be a pinned original artifact")
            Pin(Path(path), digest).verify()

    def capacity(self, config: Config) -> bool:
        plan = self.api.load(self._pin(config).path)
        prefix = "sequence-capacity-"
        # Existing observation records are never freshened or overwritten.
        ordinal = sum(1 for _ in plan.root.glob(prefix + "*"))
        if ordinal >= 1024:
            raise ContractError("Finite observation budget exhausted")
        snapshot = self.api.observe_owned_capacity(
            self._pin(config), self.transport, self.host, f"{prefix}{ordinal:04d}"
        )
        observed = decode(bounded_read(snapshot))
        if observed.get("admitted") is False:
            return False
        if observed.get("admitted") is not True:
            raise ContractError("Missing explicit admission outcome")
        # Original release API requires the exact successor inventory path.
        # Snapshot is retained by _refresh; the owned index has identical bytes.
        if bounded_read(snapshot) != bounded_read(plan.native.inventory):
            raise ContractError("Admission index differs from the original snapshot")
        self.snapshots[config.tail] = Pin(
            plan.native.inventory, reference(plan.native.inventory)["sha256"]
        )
        return True

    def pump(self, config: Config) -> str:
        proof = self.api.run_pump(self._pin(config), self.transport, self.host)
        return encode(reference(proof)).decode()

    def retire(self, config: Config, publication: str) -> str:
        self.validate(Checkpoint(config, "published", publication))
        member = self.members[config.tail]
        result = self.api.run_retirement(
            member.retirement_graph, member.retirement_state, member.retirement_terminal
        )
        return encode({"path": str(result.path), "sha256": result.sha256}).decode()

    def close_observation_lease(self) -> None:
        if self.leased():
            raise ContractError("Unexpected active lease; preserve it for its owner")

    def leased(self) -> bool:
        path = queue.GUI_LEASE_ROOT / ".fcp-native-queue-owner.json"
        return path.exists() or path.is_symlink()

    def release(self, current: Config, successor: Config, retirement: str) -> str:
        row = decode(retirement)
        path, digest = row.get("path"), row.get("sha256")
        if not isinstance(path, str) or not isinstance(digest, str):
            raise ContractError("Retirement reference missing")
        request = self.release_factory.request(
            current, successor, Pin(Path(path), digest), self.snapshots[successor.tail]
        )
        self.api.verify_release(request, self.host)
        return encode({"path": str(request.path), "sha256": request.sha256}).decode()


def main(argv: list[str] | None = None, *, api: DeferredAPI | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Prepare one deferred native sequence member using original validators"
    )
    parser.add_argument("operation", choices=("prepare-member",))
    parser.add_argument("request", type=Path)
    parser.add_argument("--sha256", required=True)
    parser.add_argument("--part", required=True, type=int)
    args = parser.parse_args(argv)
    result = prepare_deferred(
        Pin(args.request, args.sha256), args.part, OriginalPreparation() if api is None else api
    )
    print(
        encode(
            {
                "config": result.identity,
                "sha256": result.digest,
                "tail": result.tail,
                "canDispatch": False,
            }
        ).decode(),
        end="",
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
