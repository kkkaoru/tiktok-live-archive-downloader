"""Capture explicit native capacity evidence without app calls or directory scans.

No exit status or retirement authorization is evidence that a file is absent.
The initial complete owner registry is an explicit operational input; this
module measures its paths and never claims to discover unregistered jobs.
"""

import argparse
import ctypes
import json
import os
import re
import stat
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

from replay import fcp_native_queue as queue
from replay import fcp_native_queue_followup as followup
from replay import native_completed_owner_verification as completed
from replay import native_quiescent_owner as quiescent
from replay import native_retired_hardlink_verification as hardlinks
from replay.artifact_json import _json, _path
from replay.core import ReplayError
from replay.foreground_evidence import _bound_bytes
from replay.native_export_receipt import native_payload
from replay.part_master_publication import verify_published
from replay.recordings import record
from replay.stages import file_digest

ROOT = Path(__file__).resolve().parents[2]
WORK = ROOT / "edits" / "work"
MAX_ITEMS = 512
MAX_BYTES = 8 * 1024 * 1024
MAX_AGE_SECONDS = 30
STATES = frozenset({"planned", "active", "uncertain", "retained", "retired-hardlink"})


@dataclass(frozen=True)
class Identity:
    pid: int
    birth: str


@dataclass(frozen=True)
class Owner:
    name: str
    identity: Identity
    role: str
    completion: hardlinks.Reference | None = None
    quiescence: hardlinks.Reference | None = None


@dataclass(frozen=True)
class Output:
    path: Path
    owner: str
    state: str
    remaining_bytes: int
    retirement: hardlinks.Reference | None = None


@dataclass(frozen=True)
class Registry:
    generation: str
    owners: tuple[Owner, ...]
    outputs: tuple[Output, ...]
    roots: tuple[Path, ...]
    raw: dict[str, object]


@dataclass(frozen=True)
class FileState:
    exists: bool
    regular: bool
    symlink: bool
    device: int | None = None
    inode: int | None = None
    size: int = 0
    modified_ns: int | None = None

    def json(self) -> dict[str, object]:
        return {
            "exists": self.exists,
            "regular": self.regular,
            "symlink": self.symlink,
            "device": self.device,
            "inode": self.inode,
            "bytes": self.size,
            "modifiedAtNs": self.modified_ns,
        }


@dataclass(frozen=True)
class Capture:
    path: Path
    admitted: bool
    reasons: tuple[str, ...]


class Host(queue.Host, Protocol):
    def observe(self, path: Path) -> FileState: ...
    def pressure(self) -> int: ...


class SystemHost(queue.SystemHost):
    def observe(self, path: Path) -> FileState:
        try:
            value = path.lstat()
        except FileNotFoundError:
            return FileState(False, False, False)
        return FileState(
            True,
            stat.S_ISREG(value.st_mode),
            stat.S_ISLNK(value.st_mode),
            value.st_dev,
            value.st_ino,
            value.st_size,
            value.st_mtime_ns,
        )

    def pressure(self) -> int:
        if sys.platform != "darwin":
            raise ReplayError("Native pressure measurement requires macOS")
        libc = ctypes.CDLL(None, use_errno=True)
        measure = libc.sysctlbyname
        measure.argtypes = [
            ctypes.c_char_p,
            ctypes.c_void_p,
            ctypes.POINTER(ctypes.c_size_t),
            ctypes.c_void_p,
            ctypes.c_size_t,
        ]
        measure.restype = ctypes.c_int
        value, size = ctypes.c_uint32(), ctypes.c_size_t(4)
        if (
            measure(
                b"kern.memorystatus_vm_pressure_level",
                ctypes.byref(value),
                ctypes.byref(size),
                None,
                0,
            )
            != 0
            or size.value != 4
            or value.value not in (1, 2, 4)
        ):
            raise ReplayError("Native memory pressure observation is unavailable")
        return value.value


class Evidence:
    def __init__(self) -> None:
        self.pins: dict[Path, str] = {}

    def add(self, value: object) -> Path:
        raw = record(value)
        path = _path(raw.get("path"))
        digest = raw.get("sha256")
        if (
            set(raw) != {"path", "sha256"}
            or not isinstance(digest, str)
            or re.fullmatch(r"[a-f0-9]{64}", digest) is None
            or not path.is_relative_to(ROOT / "edits")
            or len(self.pins) >= MAX_ITEMS
            or path in self.pins
            and self.pins[path] != digest
        ):
            raise ReplayError("Invalid bounded immutable admission evidence")
        self.pins[path] = digest
        return path

    def read(self, value: object) -> dict[str, object]:
        path = self.add(value)
        return record(
            json.loads(_bound_bytes(path, self.pins[path], MAX_BYTES), object_pairs_hook=_unique)
        )

    def verify(self) -> None:
        if any(file_digest(path) != digest for path, digest in self.pins.items()):
            raise ReplayError("Admission registry or evidence changed during measurement")


def _unique(items: list[tuple[str, object]]) -> dict[str, object]:
    result: dict[str, object] = {}
    for key, value in items:
        if key in result:
            raise ReplayError("Duplicate admission JSON key")
        result[key] = value
    return result


def _integer(value: object, *, positive: bool = False, maximum: int = 2**63 - 1) -> int:
    if type(value) is not int or not int(positive) <= value <= maximum:
        raise ReplayError("Expected bounded admission integer")
    return value


def _text(value: object) -> str:
    if not isinstance(value, str) or not 0 < len(value) <= 512 or "\x00" in value:
        raise ReplayError("Expected bounded admission text")
    return value


def _rows(value: object, *, empty: bool = False) -> list[object]:
    if not isinstance(value, list) or not int(not empty) <= len(value) <= MAX_ITEMS:
        raise ReplayError("Expected bounded admission inventory")
    return value


def _schema(raw: dict[str, object], kind: str, keys: set[str]) -> None:
    if raw.get("version") != 1 or raw.get("kind") != kind or set(raw) != keys | {"version", "kind"}:
        raise ReplayError("Unsupported or incomplete admission schema")


def _identity(value: object) -> Identity:
    raw = record(value)
    if set(raw) != {"pid", "birth"}:
        raise ReplayError("Expected exact admission process identity")
    return Identity(_integer(raw["pid"], positive=True), _text(raw["birth"]))


def _output_path(value: object, roots: tuple[Path, ...]) -> Path:
    text = _text(value)
    path = Path(text)
    if (
        not path.is_absolute()
        or path.name == ".."
        or path.parent != path.parent.resolve()
        or not path.parent.is_dir()
        or not any(path != root and path.is_relative_to(root) for root in roots)
    ):
        raise ReplayError("Output must have an explicit canonical approved parent")
    # The final component may be a symlink. Preserve that actual observation;
    # never follow it and reclassify a dangling link as an absent output.
    return path


def _registry(raw: dict[str, object]) -> Registry:
    _schema(
        raw,
        "native-admission-registry",
        {"generation", "previous", "complete", "approvedOutputRoots", "owners", "outputs"}
        | ({"rootExtension"} if "rootExtension" in raw else set()),
    )
    if raw["complete"] is not True:
        raise ReplayError("Explicit complete owner registry is required")
    roots = tuple(_path(value) for value in _rows(raw["approvedOutputRoots"]))
    if len(set(roots)) != len(roots) or any(not path.is_dir() for path in roots):
        raise ReplayError("Approved output roots must be unique existing directories")
    owners = tuple(
        Owner(
            _text(item.get("id")),
            _identity(item.get("process")),
            _text(item.get("role")),
            None
            if item.get("completedOwner") is None
            else hardlinks.reference(item["completedOwner"]),
            None
            if item.get("quiescentOwner") is None
            else hardlinks.reference(item["quiescentOwner"]),
        )
        for item in (record(value) for value in _rows(raw["owners"]))
    )
    if (
        len({owner.name for owner in owners}) != len(owners)
        or len({owner.identity for owner in owners}) != len(owners)
        or any(owner.role not in ("native", "followup", "reader") for owner in owners)
        or any(
            owner.quiescence is not None
            and (owner.role != "reader" or owner.completion is not None)
            for owner in owners
        )
    ):
        raise ReplayError("Duplicate or unsupported registry owner")
    outputs = tuple(_output(record(value), roots) for value in _rows(raw["outputs"]))
    if (
        len({item.path for item in outputs}) != len(outputs)
        or any(item.owner not in {owner.name for owner in owners} for item in outputs)
        or any(item.state not in STATES for item in outputs)
    ):
        raise ReplayError("Duplicate output or unknown owner/state in registry")
    return Registry(_text(raw["generation"]), owners, outputs, roots, raw)


def _output(raw: dict[str, object], roots: tuple[Path, ...]) -> Output:
    state = _text(raw.get("state"))
    proof = raw.get("verifiedRetirement")
    if (state == "retired-hardlink") != (proof is not None):
        raise ReplayError("Retired alias requires its exact sealed verification")
    result = Output(
        _output_path(raw.get("path"), roots),
        _text(raw.get("owner")),
        state,
        _integer(raw.get("reservedAdditionalBytes")),
        None if proof is None else hardlinks.reference(proof),
    )
    if result.retirement is not None and result.remaining_bytes != 0:
        raise ReplayError("Retired alias cannot reserve future allocation")
    return result


def _quiescent_owners(evidence: Evidence, registry: Registry) -> None:
    for owner in registry.owners:
        if owner.quiescence is None:
            continue
        evidence.add(owner.quiescence.json())
        proof = quiescent.verify(
            owner.quiescence.path,
            owner=owner.name,
            process={"pid": owner.identity.pid, "birth": owner.identity.birth},
            host=quiescent.SystemHost(),
        )
        rows = [item for item in registry.outputs if item.owner == owner.name]
        if {str(item.path) for item in rows} != set(proof.endpoints) or any(
            item.state != "planned"
            or record(proof.endpoints[str(item.path)]).get("exists") is not False
            for item in rows
        ):
            raise ReplayError("Quiescent reader must preserve all absent planned endpoints")
        for pin in proof.pins:
            evidence.add(pin.json())


def _quiescent_transition(previous: Registry | None, registry: Registry) -> tuple[Path, ...]:
    old = {} if previous is None else {owner.name: owner for owner in previous.owners}
    current = {owner.name: owner for owner in registry.owners}
    for owner in registry.owners:
        if owner.quiescence is None:
            continue
        former = old.get(owner.name)
        if (
            former is None
            or former.identity != owner.identity
            or former.role != owner.role
            or former.completion is not None
            or former.quiescence is not None
            and former.quiescence != owner.quiescence
        ):
            raise ReplayError("Failed-reader quiescence must retain the original reader identity")
        before = (
            {}
            if previous is None
            else {item.path: item for item in previous.outputs if item.owner == owner.name}
        )
        after = {item.path: item for item in registry.outputs if item.owner == owner.name}
        if before != after:
            raise ReplayError(
                "Failed-reader quiescence cannot release or change protected reservations"
            )
        if former.quiescence is None:
            proof = quiescent.verify(
                owner.quiescence.path,
                owner=owner.name,
                process={"pid": owner.identity.pid, "birth": owner.identity.birth},
                host=quiescent.SystemHost(),
            )
            if proof.original_registry != hardlinks.reference(registry.raw["previous"]):
                raise ReplayError("Failed-reader proof must bind exact previous registry")
    if any(
        owner.quiescence is not None
        and (owner.name not in current or current[owner.name].quiescence != owner.quiescence)
        for owner in old.values()
    ):
        raise ReplayError("Historical failed-reader evidence cannot be removed or reset")
    evidence = Evidence()
    _quiescent_owners(evidence, registry)
    evidence.verify()
    return tuple(evidence.pins)


def _completed_owners(evidence: Evidence, registry: Registry) -> None:
    for owner in registry.owners:
        if owner.completion is None:
            continue
        if owner.role != "reader":
            raise ReplayError("Only historical readers can have completed-owner proofs")
        evidence.add(owner.completion.json())
        proof = completed.verify(
            owner.completion.path,
            owner=owner.name,
            process={"pid": owner.identity.pid, "birth": owner.identity.birth},
            host=completed.SystemHost(),
        )
        rows = [item for item in registry.outputs if item.owner == owner.name]
        if (
            {str(item.path) for item in rows} != set(proof.endpoints)
            or any(
                item.remaining_bytes != 0 or item.state not in ("retained", "planned")
                for item in rows
            )
            or any(
                record(proof.endpoints[str(item.path)]).get("exists") != (item.state == "retained")
                for item in rows
            )
        ):
            raise ReplayError(
                "Completed owner must preserve all endpoints with no future allocation"
            )


def _completed_transition(previous: Registry | None, registry: Registry) -> None:
    old = {} if previous is None else {owner.name: owner for owner in previous.owners}
    current = {owner.name: owner for owner in registry.owners}
    for owner in registry.owners:
        if owner.completion is None:
            continue
        former = old.get(owner.name)
        if former is None or former.identity != owner.identity or former.role != owner.role:
            raise ReplayError("Completed owner must preserve its original identity and role")
        if former.completion is not None and former.completion != owner.completion:
            successor = completed.verify(
                owner.completion.path,
                owner=owner.name,
                process={"pid": owner.identity.pid, "birth": owner.identity.birth},
                host=completed.SystemHost(),
            )
            before_rows = (
                {row.path: row for row in previous.outputs if row.owner == owner.name}
                if previous is not None
                else {}
            )
            after_rows = {row.path: row for row in registry.outputs if row.owner == owner.name}
            if successor.predecessor != former.completion or before_rows != after_rows:
                raise ReplayError(
                    "Completed-owner successor must preserve the exact predecessor and rows"
                )
        if former.completion is None:
            proof = completed.verify(
                owner.completion.path,
                owner=owner.name,
                process={"pid": owner.identity.pid, "birth": owner.identity.birth},
                host=completed.SystemHost(),
            )
            if proof.original_registry != hardlinks.reference(registry.raw["previous"]):
                raise ReplayError("Completed-owner proof must bind the exact previous registry")
    if any(
        owner.completion is not None
        and (owner.name not in current or current[owner.name].completion is None)
        for owner in old.values()
    ):
        raise ReplayError("Completed historical owner cannot be removed or reset")


def _retired(evidence: Evidence, registry: Registry) -> None:
    for item in registry.outputs:
        if item.retirement is None:
            continue
        evidence.add(item.retirement.json())
        verified = hardlinks.verify(item.retirement.path, alias=item.path, owner=item.owner)
        if not any(
            keeper.path == verified.keeper
            and keeper.owner == item.owner
            and keeper.state == "retained"
            for keeper in registry.outputs
        ):
            raise ReplayError("Retired alias requires its registered retained keeper")
        for pin in verified.pins:
            evidence.add(pin.json())


def _retired_transition(previous: Registry | None, registry: Registry) -> None:
    old = {} if previous is None else {item.path: item for item in previous.outputs}
    current = {item.path: item for item in registry.outputs}
    for item in registry.outputs:
        if item.retirement is None:
            continue
        former = old.get(item.path)
        if (
            former is None
            or former.owner != item.owner
            or former.state not in ("retained", "retired-hardlink")
        ):
            raise ReplayError("Retired alias must preserve its original retained owner row")
        if former.state == "retired-hardlink" and former.retirement != item.retirement:
            raise ReplayError("Retired alias verification cannot be replaced")
        if former.state == "retained":
            raw = hardlinks.verify(item.retirement.path, alias=item.path, owner=item.owner)
            if previous is None or raw.original_registry != hardlinks.reference(
                registry.raw["previous"]
            ):
                raise ReplayError("Retirement seal must bind the exact previous retained registry")
    if any(
        item.state == "retired-hardlink"
        and (item.path not in current or current[item.path].state != "retired-hardlink")
        for item in old.values()
    ):
        raise ReplayError("Retired alias row cannot be removed or reset")


def _released(
    evidence: Evidence, raw: dict[str, object], output: Output, *, roots: tuple[Path, ...]
) -> None:
    """Reuse existing retirement and consumer verifiers; never delete anything."""
    if SystemHost().observe(output.path).exists:
        raise ReplayError("Retirement authorization is not actual output absence")
    if set(raw) != {
        "path",
        "assemblyPart",
        "publication",
        "retirement",
        "terminal",
        "postDeleteStatus",
        "retirementAudit",
        "history",
        "consumer",
        "publishedAssembly",
    }:
        raise ReplayError("Complete retirement and post-delete consumer evidence is required")
    inputs = followup.Inputs()
    prep: dict[str, object] = {"assemblyPart": raw.get("assemblyPart"), "movie": str(output.path)}
    retirement = followup._retirement(inputs, raw, prep)
    followup._consumer(
        inputs,
        raw.get("consumer"),
        config={"publishedAssembly": raw.get("publishedAssembly")},
        retirement=retirement,
    )
    published_path = inputs.add(raw.get("publication")).path
    published = inputs.read(inputs.add(raw.get("publication")))
    survivor = _output_path(record(published.get("movie")).get("path"), roots)
    survivor_state = SystemHost().observe(survivor)
    if not survivor_state.regular or survivor_state.symlink:
        raise ReplayError("Retirement survivor must be an approved regular output")
    verify_published(
        published_path,
        inputs.add(published.get("frameProof")).path,
        part=_integer(raw.get("assemblyPart"), positive=True),
        retirement=retirement.path,
    )
    inputs.verify()
    for path, digest in inputs.refs.items():
        evidence.add({"path": str(path), "sha256": digest})


def _root_transition(previous: Registry | None, registry: Registry) -> tuple[Path, ...]:
    """Bind append-only caller authorization; do not attest who granted it."""
    extension = registry.raw.get("rootExtension")
    if previous is None:
        if extension is not None:
            raise ReplayError("Root extension cannot create a new initial registry")
        return ()
    if previous.generation == registry.generation:
        raise ReplayError("Registry transition must advance generation")
    if extension is None:
        if previous.roots != registry.roots:
            raise ReplayError("Registry transition must preserve roots without authorization")
        return ()
    path = queue._small_pin(extension)
    raw = _json(path)
    _schema(
        raw,
        "native-output-root-extension",
        {
            "previous",
            "acceptedPrevious",
            "oldRoots",
            "addedRoots",
            "authorization",
            "evidence",
        },
    )
    added = tuple(_path(value) for value in _rows(raw["addedRoots"]))
    if (
        raw["previous"] != registry.raw["previous"]
        or raw["oldRoots"] != [str(item) for item in previous.roots]
        or registry.roots != previous.roots + added
        or any(not item.is_dir() for item in added)
        or len(set(registry.roots)) != len(registry.roots)
        or any(
            a != b and (a.is_relative_to(b) or b.is_relative_to(a))
            for a in added
            for b in registry.roots
        )
    ):
        raise ReplayError("Root extension must append nonconflicting canonical roots")
    accepted_path = queue._small_pin(raw["acceptedPrevious"])
    accepted = _json(accepted_path)
    if (
        accepted.get("kind") != "native-admission-snapshot"
        or accepted.get("admitted") is not True
        or accepted.get("registry") != registry.raw["previous"]
    ):
        raise ReplayError("Root extension requires the exact accepted previous registry")
    authorization = record(raw["authorization"])
    if authorization.get("scope") != "append-native-output-roots" or authorization.get("roots") != [
        str(item) for item in added
    ]:
        raise ReplayError("Root authorization must identify its exact scope and added roots")
    pins = [path, accepted_path, queue._small_pin(authorization.get("source"))]
    pins.extend(queue._small_pin(value) for value in _rows(raw["evidence"]))
    old_outputs = {item.path: item for item in previous.outputs}
    current_outputs = {item.path: item for item in registry.outputs}
    current_owners = {item.name: item for item in registry.owners}
    if (
        any(current_owners.get(item.name) != item for item in previous.owners)
        or any(current_outputs.get(path) != item for path, item in old_outputs.items())
        or any(
            item.state != "planned"
            or item.remaining_bytes != 0
            or not any(item.path.is_relative_to(root) for root in added)
            for item in registry.outputs
            if item.path not in old_outputs
        )
        or len(current_outputs) <= len(old_outputs)
    ):
        raise ReplayError("Root extension must preserve owners/rows and add only planned outputs")
    return tuple(pins)


def _transition(evidence: Evidence, registry: Registry, releases: object) -> tuple[Path, ...]:
    release_rows = [record(value) for value in _rows(releases, empty=True)]
    previous_ref = registry.raw["previous"]
    if previous_ref is None:
        _quiescent_transition(None, registry)
        _root_transition(None, registry)
        _retired_transition(None, registry)
        _completed_transition(None, registry)
        if release_rows:
            raise ReplayError("Initial registry cannot claim omitted output releases")
        return ()
    previous = _registry(evidence.read(previous_ref))
    for path in _quiescent_transition(previous, registry):
        evidence.add({"path": str(path), "sha256": file_digest(path)})
    for path in _root_transition(previous, registry):
        evidence.add({"path": str(path), "sha256": file_digest(path)})
    _retired_transition(previous, registry)
    _completed_transition(previous, registry)
    _retired(evidence, registry)
    current = {item.path: item for item in registry.outputs}
    removed = {item.path: item for item in previous.outputs if item.path not in current}
    release_paths = [_output_path(row.get("path"), registry.roots) for row in release_rows]
    if len(set(release_paths)) != len(release_paths) or set(release_paths) != set(removed):
        raise ReplayError("Every removed protected output requires its exact release")
    if any(
        item.path in current
        and (
            item.owner != current[item.path].owner
            or item.state != "planned"
            and current[item.path].state == "planned"
        )
        for item in previous.outputs
    ):
        raise ReplayError("Protected output owner/state cannot be silently reset")
    for row, path in zip(release_rows, release_paths, strict=True):
        _released(evidence, row, removed[path], roots=registry.roots)
    return tuple(release_paths)


def _native(raw: dict[str, object], target: Identity) -> bool:
    payload = native_payload(raw)
    if payload.get("pid") != target.pid:
        raise ReplayError("Native capacity observation belongs to another app process")
    return queue._background_idle(payload)


def _publish(path: Path, raw: dict[str, object]) -> None:
    if (
        path.parent != path.parent.resolve()
        or not path.parent.is_dir()
        or not path.is_relative_to(WORK)
        or path == WORK
    ):
        raise ReplayError("Admission snapshots require existing canonical repository work storage")
    with tempfile.NamedTemporaryFile(mode="w", encoding="utf-8", dir=path.parent) as stream:
        json.dump(raw, stream, ensure_ascii=False, indent=2)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
        os.chmod(stream.name, 0o600)
        os.link(stream.name, path, follow_symlinks=False)


def _changed(before: FileState, after: FileState) -> bool:
    return (before.exists, before.regular, before.symlink, before.device, before.inode) != (
        after.exists,
        after.regular,
        after.symlink,
        after.device,
        after.inode,
    ) or after.size < before.size


def capture(request: Path, output: Path, *, host: Host) -> Capture:
    """Measure registered paths and retain admission/denial; never allocate media."""
    evidence = Evidence()
    raw = evidence.read({"path": str(request), "sha256": file_digest(request)})
    _schema(
        raw,
        "native-admission-request",
        {
            "registry",
            "nativeObservation",
            "target",
            "nativeOwner",
            "policy",
            "requested",
            "releases",
        },
    )
    producer = file_digest(Path(__file__).resolve())
    started = host.now_ns()
    registry = _registry(evidence.read(raw["registry"]))
    target = _identity(raw["target"])
    native_owner = _text(raw["nativeOwner"])
    if native_owner not in {item.name for item in registry.owners if item.role == "native"}:
        raise ReplayError("Native admission owner is not registered")
    _completed_owners(evidence, registry)
    _quiescent_owners(evidence, registry)
    identities = {
        owner.identity
        for owner in registry.owners
        if owner.completion is None and owner.quiescence is None
    } | {target}
    if any(host.birth(item.pid) != item.birth for item in identities):
        raise ReplayError("Registered owner or app process birth changed")
    released_paths = _transition(evidence, registry, raw["releases"])
    native_path = evidence.add(raw["nativeObservation"])
    native_time = native_path.stat().st_mtime_ns
    idle = _native(evidence.read(raw["nativeObservation"]), target)
    policy = record(raw["policy"])
    age = _integer(policy.get("maximumAgeSeconds"), positive=True, maximum=MAX_AGE_SECONDS)
    max_count = _integer(policy.get("maximumStagingCount"), positive=True, maximum=MAX_ITEMS)
    max_bytes = _integer(policy.get("maximumStagingBytes"), positive=True)
    max_active = _integer(policy.get("maximumActiveRenders"), positive=True, maximum=MAX_ITEMS)
    pressure_limit = _integer(policy.get("maximumPressure"), positive=True, maximum=4)
    if pressure_limit not in (1, 2, 4):
        raise ReplayError("Expected an explicit OS pressure level")
    floor = _integer(policy.get("minimumFreeBytes"))
    boot = _path(policy.get("bootVolume"))
    if not boot.is_dir() or boot.stat().st_dev != ROOT.stat().st_dev:
        raise ReplayError("Boot capacity must observe the actual checkout volume")
    requested = record(raw["requested"])
    movie = _output_path(requested.get("path"), registry.roots)
    peak = _integer(requested.get("peakAdditionalBytes"), positive=True)
    allocation, allocation_budgets, allocation_pins = queue.disk_allocation(
        requested.get("diskAllocation"), boot, movie, peak
    )
    for allocation_pin in allocation_pins:
        evidence.add({"path": str(allocation_pin), "sha256": file_digest(allocation_pin)})
    selected = [item for item in registry.outputs if item.path == movie]
    if (
        len(selected) != 1
        or selected[0].owner != native_owner
        or selected[0].state != "planned"
        or selected[0].remaining_bytes != 0
    ):
        raise ReplayError("Requested allocation must be one unreserved planned owner output")
    before = {item.path: host.observe(item.path) for item in registry.outputs}
    pressure = host.pressure()
    if pressure not in (1, 2, 4):
        raise ReplayError("Unsupported observed pressure value")
    volumes = tuple(
        dict.fromkeys(
            (
                boot,
                *(item.path.parent for item in registry.outputs),
                *(path for path, _ in allocation_budgets),
            )
        )
    )
    spaces = {path: (host.free(path), max(floor, host.floor(path))) for path in volumes}
    after = {item.path: host.observe(item.path) for item in registry.outputs}
    _retired(evidence, registry)
    _completed_owners(evidence, registry)
    _quiescent_owners(evidence, registry)
    reasons: list[str] = []
    if not idle:
        reasons.append("native-background-busy")
    if pressure > pressure_limit:
        reasons.append("memory-pressure")
    if any(_changed(before[path], value) for path, value in after.items()):
        reasons.append("output-identity-changed")
    if any(value.exists and not value.regular or value.symlink for value in after.values()):
        reasons.append("output-not-regular")
    file_ids = [(value.device, value.inode) for value in after.values() if value.regular]
    if len(set(file_ids)) != len(file_ids):
        reasons.append("duplicate-output-inode")
    if after[movie].exists or after[movie].symlink:
        reasons.append("requested-output-already-present")
    if any(
        not after[item.path].exists and item.state not in ("planned", "retired-hardlink")
        for item in registry.outputs
    ):
        reasons.append("protected-output-missing-without-release")
    if any(host.observe(path).exists for path in released_paths):
        reasons.append("released-output-reappeared")
    active = sum(
        item.state in ("active", "uncertain")
        and item.owner in {owner.name for owner in registry.owners if owner.role == "native"}
        for item in registry.outputs
    )
    count = sum(
        after[item.path].exists or item.state != "planned" or item.remaining_bytes > 0
        for item in registry.outputs
    )
    used = sum(value.size for value in after.values() if value.exists)
    reserved = sum(item.remaining_bytes for item in registry.outputs)
    if active + 1 > max_active:
        reasons.append("native-active-capacity")
    if count + 1 > max_count or used + reserved + peak > max_bytes:
        reasons.append("staging-capacity")
    if not queue.disk_capacity(
        host,
        boot,
        movie,
        peak,
        floor,
        allocation,
        tuple((item.path.parent, item.remaining_bytes) for item in registry.outputs),
    ):
        reasons.append("disk-capacity")
    evidence.verify()
    if (
        any(host.birth(item.pid) != item.birth for item in identities)
        or file_digest(Path(__file__).resolve()) != producer
    ):
        raise ReplayError("Owner/app identity or snapshot producer changed during measurement")
    finished = host.now_ns()
    if not 0 <= finished - native_time <= age * 1_000_000_000:
        reasons.append("native-observation-stale-or-future")
    if not 0 <= finished - started <= age * 1_000_000_000:
        reasons.append("measurement-exceeds-freshness")
    admitted = not reasons
    _publish(
        output,
        {
            "version": 1,
            "kind": "native-admission-snapshot",
            "admitted": admitted,
            "qualityApproved": False,
            "reasons": reasons,
            "registry": raw["registry"],
            "registryGeneration": registry.generation,
            "registryCompletenessSource": "explicit-owner-input-not-directory-discovery",
            "nativeObservation": raw["nativeObservation"],
            "nativeObservedAtNs": native_time,
            "measurementStartedAtNs": started,
            "measurementFinishedAtNs": finished,
            "capturedAtNs": min(started, native_time),
            "maximumAgeSeconds": age,
            "target": raw["target"],
            "owners": registry.raw["owners"],
            "outputs": [
                {
                    "path": str(item.path),
                    "owner": item.owner,
                    "state": item.state,
                    "reservedAdditionalBytes": item.remaining_bytes,
                    "before": before[item.path].json(),
                    "after": after[item.path].json(),
                    **(
                        {"verifiedRetirement": item.retirement.json()}
                        if item.retirement is not None
                        else {}
                    ),
                }
                for item in registry.outputs
            ],
            "requested": requested,
            "pressure": pressure,
            "maximumPressure": pressure_limit,
            "observedActiveRenders": active,
            # Legacy inventory readers require an integer here but do not yet
            # understand pressure/staging denial. A rejected snapshot must not
            # accidentally become an accepted legacy inventory.
            "activeRenders": active if admitted else None,
            "stagingCount": count,
            "stagingBytes": used,
            "reservedAdditionalBytes": reserved,
            "maximumStagingCount": max_count,
            "maximumStagingBytes": max_bytes,
            "volumes": [
                {"path": str(path), "freeBytes": free, "floorBytes": minimum}
                for path, (free, minimum) in spaces.items()
            ],
            "inputs": {str(path): digest for path, digest in evidence.pins.items()},
            "producer": {"path": str(Path(__file__).resolve()), "sha256": producer},
        },
    )
    return Capture(output, admitted, tuple(reasons))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("request", type=Path)
    parser.add_argument("output", type=Path, nargs="?")
    parser.add_argument("--prepare-data", action="store_true")
    parser.add_argument("--validate-candidate", action="store_true")
    args = parser.parse_args()
    if args.prepare_data and args.validate_candidate:
        parser.error("Choose one preparation operation")
    if args.validate_candidate:
        validate_candidate(args.request.resolve())
        print(json.dumps({"canDispatch": False, "hostObserved": False}))
        return
    if args.output is None:
        parser.error("Preparation and capture require an output path")
    if args.prepare_data:
        print(prepare_data(args.request.resolve(), args.output.absolute()))
        return
    result = capture(args.request.resolve(), args.output.absolute(), host=SystemHost())
    print(
        json.dumps(
            {"path": str(result.path), "admitted": result.admitted, "reasons": result.reasons}
        )
    )
    if not result.admitted:
        raise SystemExit(1)


def _preparation(raw: dict[str, object]) -> tuple[Path, ...]:
    """Validate static metadata without observing an app or hashing media."""
    if (
        raw.get("version") != 1
        or raw.get("kind") != "native-admission-preparation-request"
        or raw.get("nativeObservation") is not None
    ):
        raise ReplayError("Preparation must not assert a fresh native observation")
    registry_path = queue._small_pin(raw.get("registry"))
    registry = _registry(_json(registry_path))
    _identity(raw.get("target"))
    owner = _text(raw.get("nativeOwner"))
    if owner not in {item.name for item in registry.owners if item.role == "native"}:
        raise ReplayError("Native preparation owner is not registered")
    requested = record(raw.get("requested"))
    movie = _output_path(requested.get("path"), registry.roots)
    _integer(requested.get("peakAdditionalBytes"), positive=True)
    if not any(
        item.path == movie
        and item.owner == owner
        and item.state == "planned"
        and item.remaining_bytes == 0
        for item in registry.outputs
    ):
        raise ReplayError("Preparation needs one unreserved planned output")
    policy = record(raw.get("policy"))
    _integer(policy.get("maximumAgeSeconds"), positive=True, maximum=MAX_AGE_SECONDS)
    _integer(policy.get("maximumStagingCount"), positive=True, maximum=MAX_ITEMS)
    _integer(policy.get("maximumStagingBytes"), positive=True)
    _integer(policy.get("maximumActiveRenders"), positive=True, maximum=MAX_ITEMS)
    if (
        policy.get("maximumPressure") not in (1, 2, 4)
        or type(policy.get("maximumPressure")) is not int
    ):
        raise ReplayError("Expected explicit supported pressure limit")
    _integer(policy.get("minimumFreeBytes"))
    if not _path(policy.get("bootVolume")).is_dir():
        raise ReplayError("Expected existing capacity observation directory")
    refs = [registry_path]
    _, _, allocation_pins = queue.disk_allocation(
        requested.get("diskAllocation"),
        _path(policy.get("bootVolume")),
        movie,
        _integer(requested.get("peakAdditionalBytes"), positive=True),
    )
    refs.extend(allocation_pins)
    previous = registry.raw.get("previous")
    removed: set[Path] = set()
    if previous is not None:
        previous_path = queue._small_pin(previous)
        refs.append(previous_path)
        old = _registry(_json(previous_path))
        refs.extend(_quiescent_transition(old, registry))
        refs.extend(_root_transition(old, registry))
        _retired_transition(old, registry)
        _completed_transition(old, registry)
        current = {item.path: item for item in registry.outputs}
        if any(
            item.path in current
            and (
                item.owner != current[item.path].owner
                or item.state != "planned"
                and current[item.path].state == "planned"
            )
            for item in old.outputs
        ):
            raise ReplayError("Preparation cannot reset protected ownership or state")
        removed = {item.path for item in old.outputs} - {item.path for item in registry.outputs}
    if previous is None:
        _quiescent_transition(None, registry)
        _root_transition(None, registry)
        _retired_transition(None, registry)
        _completed_transition(None, registry)
    refs.extend(
        queue._small_pin(item.retirement.json())
        for item in registry.outputs
        if item.retirement is not None
    )
    releases = [record(value) for value in _rows(raw.get("releases"), empty=True)]
    paths = [_output_path(row.get("path"), registry.roots) for row in releases]
    if set(paths) != removed or len(set(paths)) != len(paths):
        raise ReplayError("Preparation must retain every retirement closure")
    for release in releases:
        _integer(release.get("assemblyPart"), positive=True)
        refs.extend(
            queue._small_pin(release.get(key))
            for key in (
                "publication",
                "retirement",
                "terminal",
                "postDeleteStatus",
                "retirementAudit",
                "consumer",
                "publishedAssembly",
            )
        )
        refs.extend(queue._small_pin(value) for value in _rows(release.get("history")))
    return tuple(dict.fromkeys(refs))


def prepare_data(request: Path, output: Path) -> Path:
    """Prepare a non-executable candidate; live identity and release remain unverified."""
    request = _path(str(request))
    refs = _preparation(_json(request))
    output = _path(str(output))
    if not output.is_relative_to(WORK) or not output.parent.is_dir():
        raise ReplayError("Candidate output must be repository-local")
    value: dict[str, object] = {
        "version": 1,
        "kind": "native-admission-candidate",
        "candidateOnly": True,
        "canDispatch": False,
        "guiGrant": None,
        "readiness": None,
        "nativeObservation": None,
        "releaseVerified": False,
        "hostObserved": False,
        "request": queue._ref(request),
        "inputs": [
            queue._ref(path)
            for path in (request, *refs, Path(__file__).resolve(), Path(queue.__file__).resolve())
        ],
    }
    queue._prepared_save(output, value)
    validate_candidate(output)
    return output


def validate_candidate(path: Path) -> None:
    raw = _json(_path(str(path)))
    if (
        raw.get("version") != 1
        or raw.get("kind") != "native-admission-candidate"
        or raw.get("candidateOnly") is not True
        or raw.get("canDispatch") is not False
        or raw.get("guiGrant") is not None
        or raw.get("readiness") is not None
        or raw.get("nativeObservation") is not None
        or raw.get("releaseVerified") is not False
        or raw.get("hostObserved") is not False
    ):
        raise ReplayError("Expected explicitly unobserved admission candidate")
    refs = {queue._small_pin(value) for value in _rows(raw.get("inputs"))}
    request = queue._small_pin(raw.get("request"))
    required = {
        request,
        Path(__file__).resolve(),
        Path(queue.__file__).resolve(),
        *_preparation(_json(request)),
    }
    if not required.issubset(refs):
        raise ReplayError("Candidate preparation evidence is incomplete")


if __name__ == "__main__":
    main()
