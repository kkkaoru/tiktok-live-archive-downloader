"""Pure version-two capacity arithmetic; observations and dispatch stay with owners.

Existing scalar admission contracts are deliberately not imported or changed.
Facts must come from a bounded host capture, including shared APFS pool identity.
An accepted result is disk/resource fit, never a GUI grant or a reservation.
"""

from dataclasses import dataclass
from pathlib import PurePosixPath

from replay.core import ReplayError

MAX_BYTES = (1 << 63) - 1
MAX_ITEMS = 512
MINIMUM_FREE_BYTES = 30 * 1024**3
ROLES = frozenset({"native-media", "publication-copy", "scratch", "cache", "metadata", "mux"})


@dataclass(frozen=True)
class ParentIdentity:
    path: str
    device: int
    inode: int
    pool: str
    exists: bool
    directory: bool
    symlink: bool
    ancestors_verified: bool


@dataclass(frozen=True)
class VolumeObservation:
    before: ParentIdentity
    after: ParentIdentity
    free_before: int
    free_after: int
    total_bytes: int
    floor_bytes: int
    observed_at_ns: int
    topology_verified: bool


@dataclass(frozen=True)
class Allocation:
    """Additional bytes only; existing bytes are already reflected in free space."""

    identifier: str
    owner: str
    parent: ParentIdentity
    role: str
    additional_bytes: int
    reservation: bool
    path: str
    peak_bytes: int
    materialized_bytes: int


@dataclass(frozen=True)
class Reserve:
    """Unspendable pool headroom, separate from every allocation's byte bound."""

    identifier: str
    parent: ParentIdentity
    bytes: int


@dataclass(frozen=True)
class StagingLimit:
    """Independent owner-observed limits, including retained and uncertain media."""

    name: str
    retained_count: int
    retained_bytes: int
    new_count: int
    allocation_ids: tuple[str, ...]
    maximum_count: int
    maximum_bytes: int


@dataclass(frozen=True)
class WorkerBudget:
    native_writers: int
    new_native_writers: int
    maximum_native_writers: int
    production_readers: int
    new_production_readers: int
    maximum_production_readers: int
    external_workers: int
    maximum_external_workers: int
    pressure: int
    maximum_pressure: int
    complete: bool


@dataclass(frozen=True)
class BudgetPlan:
    version: int
    allocations: tuple[Allocation, ...]
    reserves: tuple[Reserve, ...]
    staging: tuple[StagingLimit, ...]
    workers: WorkerBudget
    declared_additional_bytes: int
    owners: tuple[str, ...]
    registry_complete: bool
    captured_at_ns: int
    evaluated_at_ns: int
    maximum_age_ns: int
    expected_paths: tuple[str, ...]


@dataclass(frozen=True)
class PoolBudget:
    pool: str
    free_bytes: int
    floor_bytes: int
    reserve_bytes: int
    additional_bytes: int
    remaining_bytes: int


@dataclass(frozen=True)
class BudgetDecision:
    fits: bool
    reasons: tuple[str, ...]
    pools: tuple[PoolBudget, ...]
    can_dispatch: bool = False


def _number(value: int, *, positive: bool = False) -> int:
    if type(value) is not int or value < int(positive) or value > MAX_BYTES:
        raise ReplayError("Capacity facts require bounded nonnegative integers")
    return value


def _sum(values: tuple[int, ...]) -> int:
    return _number(sum(_number(value) for value in values))


def _unique(values: tuple[str, ...]) -> None:
    if not values or len(values) > MAX_ITEMS or any(not value.strip() for value in values):
        raise ReplayError("Capacity facts require bounded nonempty identifiers")
    if len(values) != len(set(values)):
        raise ReplayError("Capacity identifiers collide")


def _parent(value: ParentIdentity) -> None:
    path = PurePosixPath(value.path)
    if not path.is_absolute() or str(path) != value.path or ".." in path.parts:
        raise ReplayError("Capacity parent must be canonical and absolute")
    if not value.pool.strip():
        raise ReplayError("Storage pool identity is missing")
    _number(value.device, positive=True)
    _number(value.inode, positive=True)
    if not value.exists or not value.directory or value.symlink or not value.ancestors_verified:
        raise ReplayError("Capacity parent or ancestor identity is unverified")


def _observations(
    plan: BudgetPlan, observations: tuple[VolumeObservation, ...]
) -> dict[str, VolumeObservation]:
    _unique(tuple(item.before.path for item in observations))
    result: dict[str, VolumeObservation] = {}
    devices: dict[int, str] = {}
    for item in observations:
        _parent(item.before)
        _parent(item.after)
        if item.before != item.after or not item.topology_verified:
            raise ReplayError("Capacity identity changed or pool topology is unknown")
        device, pool = item.before.device, item.before.pool
        if device in devices and devices[device] != pool:
            raise ReplayError("One device cannot belong to two capacity pools")
        devices[device] = pool
        total = _number(item.total_bytes, positive=True)
        if max(_number(item.free_before), _number(item.free_after)) > total:
            raise ReplayError("Free space exceeds observed total")
        if _number(item.floor_bytes) < max(MINIMUM_FREE_BYTES, (total + 19) // 20):
            raise ReplayError("Capacity floor is below the normal volume minimum")
        _number(item.observed_at_ns)
        if not plan.captured_at_ns <= item.observed_at_ns <= plan.evaluated_at_ns:
            raise ReplayError("Volume observation falls outside the capture interval")
        result[item.before.path] = item
    return result


def _bound(parent: ParentIdentity, observations: dict[str, VolumeObservation]) -> None:
    if parent.path not in observations or observations[parent.path].after != parent:
        raise ReplayError("Allocation lacks matching observed parent identity")


def _workers(value: WorkerBudget) -> tuple[str, ...]:
    limits = (
        (value.native_writers, value.new_native_writers, value.maximum_native_writers),
        (value.production_readers, value.new_production_readers, value.maximum_production_readers),
        (value.external_workers, 0, value.maximum_external_workers),
    )
    reasons = tuple(
        f"worker-capacity:{index}"
        for index, (current, new, limit) in enumerate(limits)
        if _sum((current, new)) > _number(limit)
    )
    if value.pressure not in (1, 2, 4) or value.maximum_pressure not in (1, 2, 4):
        raise ReplayError("Unknown OS memory pressure fact")
    if not value.complete:
        raise ReplayError("Unknown concurrent workers prevent capacity admission")
    return reasons + (("memory-pressure",) if value.pressure > value.maximum_pressure else ())


def _staging(plan: BudgetPlan) -> tuple[str, ...]:
    _unique(tuple(item.name for item in plan.staging))
    allocations = {item.identifier: item for item in plan.allocations}
    reasons: list[str] = []
    for scope in plan.staging:
        if len(scope.allocation_ids) != len(set(scope.allocation_ids)) or any(
            identifier not in allocations for identifier in scope.allocation_ids
        ):
            raise ReplayError("Staging scope contains duplicate or unknown allocations")
        count = _sum((scope.retained_count, scope.new_count))
        size = _sum(
            (scope.retained_bytes,)
            + tuple(allocations[identifier].additional_bytes for identifier in scope.allocation_ids)
        )
        if count > _number(scope.maximum_count) or size > _number(scope.maximum_bytes):
            reasons.append(f"staging-capacity:{scope.name}")
    return tuple(reasons)


def evaluate(plan: BudgetPlan, observations: tuple[VolumeObservation, ...]) -> BudgetDecision:
    """Evaluate trusted typed facts; reject incomplete facts, never operate the host."""
    if plan.version != 2 or not plan.registry_complete:
        raise ReplayError("Expected a complete version-two capacity plan")
    _number(plan.captured_at_ns)
    _number(plan.evaluated_at_ns)
    age = _number(plan.maximum_age_ns, positive=True)
    if not 0 <= plan.evaluated_at_ns - plan.captured_at_ns <= age:
        raise ReplayError("Capacity capture is stale or future dated")
    _unique(plan.owners)
    _unique(tuple(item.identifier for item in plan.allocations + plan.reserves))
    _unique(plan.expected_paths)
    _unique(tuple(item.path for item in plan.allocations))
    if set(plan.expected_paths) != {item.path for item in plan.allocations}:
        raise ReplayError("Every planned or reserved path requires an allocation")
    observed = _observations(plan, observations)
    allocated: dict[str, int] = {}
    reserved: dict[str, int] = {}
    for item in plan.allocations:
        _bound(item.parent, observed)
        path = PurePosixPath(item.path)
        if str(path) != item.path or str(path.parent) != item.parent.path or ".." in path.parts:
            raise ReplayError("Allocation path is not a direct canonical child of its parent")
        if item.owner not in plan.owners or item.role not in ROLES:
            raise ReplayError("Allocation owner or role is unknown")
        materialized = _number(item.materialized_bytes)
        if materialized and not item.reservation:
            raise ReplayError("A new allocation cannot already be materialized")
        if _sum((materialized, item.additional_bytes)) != _number(item.peak_bytes):
            raise ReplayError("Remaining reservation must exclude already materialized bytes")
        allocated[item.parent.pool] = _sum(
            (allocated.get(item.parent.pool, 0), item.additional_bytes)
        )
    if _sum(tuple(allocated.values())) != _number(plan.declared_additional_bytes):
        raise ReplayError("Unassigned or duplicated additional bytes")
    for reserve in plan.reserves:
        _bound(reserve.parent, observed)
        reserved[reserve.parent.pool] = _sum((reserved.get(reserve.parent.pool, 0), reserve.bytes))
    reasons = list(_workers(plan.workers) + _staging(plan))
    pools: list[PoolBudget] = []
    for pool in sorted({item.after.pool for item in observations}):
        rows = tuple(item for item in observations if item.after.pool == pool)
        free = min(min(item.free_before, item.free_after) for item in rows)
        floor = max(item.floor_bytes for item in rows)
        reserve_bytes = reserved.get(pool, 0)
        additional = allocated.get(pool, 0)
        needed = _sum((floor, reserve_bytes, additional))
        remaining = free - needed
        pools.append(PoolBudget(pool, free, floor, reserve_bytes, additional, remaining))
        if remaining < 0:
            reasons.append(f"disk-capacity:{pool}")
    return BudgetDecision(not reasons, tuple(reasons), tuple(pools))
