"""Synthetic volume facts; no filesystem, app, or production-media access."""

from dataclasses import replace

import pytest

from replay.core import ReplayError
from replay.native_volume_budget import (
    Allocation,
    BudgetPlan,
    ParentIdentity,
    Reserve,
    StagingLimit,
    VolumeObservation,
    WorkerBudget,
    evaluate,
)


@pytest.fixture
def boot() -> ParentIdentity:
    return ParentIdentity("/work/cache", 1, 10, "boot", True, True, False, True)


@pytest.fixture
def ssd() -> ParentIdentity:
    return ParentIdentity("/media/staging", 2, 20, "external", True, True, False, True)


@pytest.fixture
def observations(boot: ParentIdentity, ssd: ParentIdentity) -> tuple[VolumeObservation, ...]:
    return (
        VolumeObservation(
            boot, boot, 70529000000, 70529000000, 994610155520, 49730507776, 100, True
        ),
        VolumeObservation(
            ssd, ssd, 337000000000, 337000000000, 2000000000000, 100000000000, 101, True
        ),
    )


@pytest.fixture
def plan(boot: ParentIdentity, ssd: ParentIdentity) -> BudgetPlan:
    return BudgetPlan(
        2,
        (
            Allocation(
                "movie",
                "native",
                ssd,
                "native-media",
                4500000000,
                False,
                "/media/staging/movie.mov",
                4500000000,
                0,
            ),
            Allocation(
                "cache",
                "native",
                boot,
                "cache",
                2147483648,
                False,
                "/work/cache/library",
                2147483648,
                0,
            ),
        ),
        (Reserve("boot-operational", boot, 15032385536),),
        (
            StagingLimit("boot", 5, 17315000000, 0, (), 6, 27000000000),
            StagingLimit("global", 5, 17315000000, 1, ("movie",), 8, 36000000000),
        ),
        WorkerBudget(0, 1, 1, 1, 1, 2, 1, 1, 1, 1, True),
        6647483648,
        ("native", "followup", "experiment"),
        True,
        100,
        102,
        30,
        ("/media/staging/movie.mov", "/work/cache/library"),
    )


def test_ssd_media_does_not_consume_boot_media_peak(
    plan: BudgetPlan, observations: tuple[VolumeObservation, ...]
) -> None:
    result = evaluate(plan, observations)
    assert result.fits is True
    assert result.can_dispatch is False
    assert result.reasons == ()
    assert result.pools[0].remaining_bytes == 3618623040
    assert result.pools[0].additional_bytes == 2147483648
    assert result.pools[1].additional_bytes == 4500000000


def test_same_native_output_on_boot_still_denied(
    plan: BudgetPlan, boot: ParentIdentity, observations: tuple[VolumeObservation, ...]
) -> None:
    movie = replace(plan.allocations[0], parent=boot, path="/work/cache/movie.mov")
    result = evaluate(
        replace(
            plan,
            allocations=(movie, plan.allocations[1]),
            expected_paths=("/work/cache/movie.mov", "/work/cache/library"),
        ),
        observations,
    )
    assert result.fits is False
    assert result.reasons == ("disk-capacity:boot",)
    assert result.pools[0].remaining_bytes == -881376960


def test_different_paths_same_device_charge_once_per_allocation(
    plan: BudgetPlan, observations: tuple[VolumeObservation, ...], boot: ParentIdentity
) -> None:
    other = replace(boot, path="/work/movie", inode=11)
    movie = replace(plan.allocations[0], parent=other, path="/work/movie/export.mov")
    row = replace(observations[0], before=other, after=other)
    result = evaluate(
        replace(
            plan,
            allocations=(movie, plan.allocations[1]),
            expected_paths=("/work/movie/export.mov", "/work/cache/library"),
        ),
        (observations[0], row),
    )
    assert len(result.pools) == 1
    assert result.pools[0].additional_bytes == 6647483648
    assert result.reasons == ("disk-capacity:boot",)


def test_apfs_distinct_devices_share_capacity_pool(
    plan: BudgetPlan, observations: tuple[VolumeObservation, ...]
) -> None:
    parent = replace(plan.allocations[0].parent, pool="boot")
    movie = replace(plan.allocations[0], parent=parent)
    row = replace(
        observations[1],
        before=parent,
        after=parent,
        total_bytes=994610155520,
        floor_bytes=49730507776,
    )
    result = evaluate(
        replace(plan, allocations=(movie, plan.allocations[1])), (observations[0], row)
    )
    assert len(result.pools) == 1
    assert result.pools[0].free_bytes == 70529000000
    assert result.reasons == ("disk-capacity:boot",)


def test_full_copy_and_growing_reservation_sum_in_one_pool(
    plan: BudgetPlan, observations: tuple[VolumeObservation, ...]
) -> None:
    old = replace(
        plan.allocations[0],
        identifier="old-growth",
        additional_bytes=2000000000,
        reservation=True,
        path="/media/staging/growing.mov",
        materialized_bytes=2500000000,
    )
    copy = replace(
        plan.allocations[0],
        identifier="copy",
        owner="followup",
        role="publication-copy",
        path="/media/staging/published.mov",
    )
    changed = replace(
        plan,
        allocations=plan.allocations + (old, copy),
        declared_additional_bytes=13147483648,
        expected_paths=plan.expected_paths
        + ("/media/staging/growing.mov", "/media/staging/published.mov"),
    )
    result = evaluate(changed, observations)
    assert result.pools[1].additional_bytes == 11000000000
    assert result.fits is True


def test_free_after_decline_is_used(
    plan: BudgetPlan, observations: tuple[VolumeObservation, ...]
) -> None:
    changed = replace(observations[0], free_after=60000000000)
    result = evaluate(plan, (changed, observations[1]))
    assert result.reasons == ("disk-capacity:boot",)
    assert result.pools[0].free_bytes == 60000000000


@pytest.mark.parametrize("field", ["device", "inode"])
def test_parent_change_rejected(
    plan: BudgetPlan, observations: tuple[VolumeObservation, ...], field: str
) -> None:
    identity = (
        replace(observations[0].after, device=999)
        if field == "device"
        else replace(observations[0].after, inode=999)
    )
    row = replace(observations[0], after=identity)
    with pytest.raises(ReplayError, match="identity changed"):
        evaluate(plan, (row, observations[1]))


@pytest.mark.parametrize(
    "parent",
    [
        ParentIdentity("relative", 1, 1, "pool", True, True, False, True),
        ParentIdentity("/a/../b", 1, 1, "pool", True, True, False, True),
        ParentIdentity("/a//b", 1, 1, "pool", True, True, False, True),
        ParentIdentity("/a", 0, 1, "pool", True, True, False, True),
        ParentIdentity("/a", 1, 0, "pool", True, True, False, True),
        ParentIdentity("/a", 1, 1, "", True, True, False, True),
        ParentIdentity("/a", 1, 1, "pool", False, True, False, True),
        ParentIdentity("/a", 1, 1, "pool", True, False, False, True),
        ParentIdentity("/a", 1, 1, "pool", True, True, True, True),
        ParentIdentity("/a", 1, 1, "pool", True, True, False, False),
    ],
)
def test_invalid_parent_fails_closed(
    plan: BudgetPlan, observations: tuple[VolumeObservation, ...], parent: ParentIdentity
) -> None:
    row = replace(observations[0], before=parent, after=parent)
    with pytest.raises(ReplayError):
        evaluate(plan, (row, observations[1]))


@pytest.mark.parametrize("value", [-1, True, 1 << 63])
def test_invalid_or_overflow_bytes(
    plan: BudgetPlan, observations: tuple[VolumeObservation, ...], value: int
) -> None:
    movie = replace(plan.allocations[0], additional_bytes=value)
    with pytest.raises(ReplayError, match="bounded nonnegative"):
        evaluate(replace(plan, allocations=(movie, plan.allocations[1])), observations)


def test_sum_overflow(plan: BudgetPlan, observations: tuple[VolumeObservation, ...]) -> None:
    huge = replace(
        plan.allocations[0],
        additional_bytes=(1 << 63) - 1,
        peak_bytes=(1 << 63) - 1,
        materialized_bytes=1,
        reservation=True,
    )
    with pytest.raises(ReplayError, match="bounded nonnegative"):
        evaluate(replace(plan, allocations=(huge, plan.allocations[1])), observations)


@pytest.mark.parametrize("declared", [0, 6647483647, 6647483649])
def test_unassigned_bytes_rejected(
    plan: BudgetPlan, observations: tuple[VolumeObservation, ...], declared: int
) -> None:
    with pytest.raises(ReplayError, match="Unassigned"):
        evaluate(replace(plan, declared_additional_bytes=declared), observations)


def test_identifier_collision(
    plan: BudgetPlan, observations: tuple[VolumeObservation, ...]
) -> None:
    reserve = replace(plan.reserves[0], identifier="movie")
    with pytest.raises(ReplayError, match="collide"):
        evaluate(replace(plan, reserves=(reserve,)), observations)


def test_missing_parent_observation(
    plan: BudgetPlan, observations: tuple[VolumeObservation, ...]
) -> None:
    with pytest.raises(ReplayError, match="matching observed"):
        evaluate(plan, (observations[0],))


def test_one_device_two_pools_rejected(
    plan: BudgetPlan, observations: tuple[VolumeObservation, ...]
) -> None:
    other = replace(observations[1].before, device=1)
    row = replace(observations[1], before=other, after=other)
    with pytest.raises(ReplayError, match="two capacity pools"):
        evaluate(plan, (observations[0], row))


@pytest.mark.parametrize("time", [99, 103])
def test_volume_time_outside_capture(
    plan: BudgetPlan, observations: tuple[VolumeObservation, ...], time: int
) -> None:
    row = replace(observations[0], observed_at_ns=time)
    with pytest.raises(ReplayError, match="capture interval"):
        evaluate(plan, (row, observations[1]))


def test_unknown_topology(plan: BudgetPlan, observations: tuple[VolumeObservation, ...]) -> None:
    with pytest.raises(ReplayError, match="topology"):
        evaluate(plan, (replace(observations[0], topology_verified=False), observations[1]))


def test_impossible_free(plan: BudgetPlan, observations: tuple[VolumeObservation, ...]) -> None:
    with pytest.raises(ReplayError, match="exceeds"):
        evaluate(plan, (replace(observations[0], total_bytes=1), observations[1]))


@pytest.mark.parametrize("role,owner", [("unknown", "native"), ("native-media", "foreign")])
def test_unknown_owner_or_role(
    plan: BudgetPlan, observations: tuple[VolumeObservation, ...], role: str, owner: str
) -> None:
    movie = replace(plan.allocations[0], owner=owner, role=role)
    with pytest.raises(ReplayError, match="owner or role"):
        evaluate(replace(plan, allocations=(movie, plan.allocations[1])), observations)


@pytest.mark.parametrize("version,complete", [(1, True), (3, True), (2, False)])
def test_no_legacy_reinterpretation_or_incomplete_registry(
    plan: BudgetPlan, observations: tuple[VolumeObservation, ...], version: int, complete: bool
) -> None:
    with pytest.raises(ReplayError, match="version-two"):
        evaluate(replace(plan, version=version, registry_complete=complete), observations)


@pytest.mark.parametrize("time", [99, 131])
def test_stale_or_future_capture(
    plan: BudgetPlan, observations: tuple[VolumeObservation, ...], time: int
) -> None:
    with pytest.raises(ReplayError, match="stale or future"):
        evaluate(replace(plan, evaluated_at_ns=time), observations)


def test_workers_and_pressure_remain_independent(
    plan: BudgetPlan, observations: tuple[VolumeObservation, ...]
) -> None:
    workers = replace(
        plan.workers, native_writers=1, production_readers=2, external_workers=2, pressure=2
    )
    result = evaluate(replace(plan, workers=workers), observations)
    assert result.reasons == (
        "worker-capacity:0",
        "worker-capacity:1",
        "worker-capacity:2",
        "memory-pressure",
    )


def test_unknown_workers_rejected(
    plan: BudgetPlan, observations: tuple[VolumeObservation, ...]
) -> None:
    with pytest.raises(ReplayError, match="Unknown concurrent"):
        evaluate(replace(plan, workers=replace(plan.workers, complete=False)), observations)


def test_unknown_pressure_rejected(
    plan: BudgetPlan, observations: tuple[VolumeObservation, ...]
) -> None:
    with pytest.raises(ReplayError, match="Unknown OS"):
        evaluate(replace(plan, workers=replace(plan.workers, pressure=3)), observations)


def test_scope_limits_unchanged(
    plan: BudgetPlan, observations: tuple[VolumeObservation, ...]
) -> None:
    scope = replace(
        plan.staging[0], new_count=2, allocation_ids=("movie",), retained_bytes=26000000000
    )
    result = evaluate(replace(plan, staging=(scope, plan.staging[1])), observations)
    assert result.reasons == ("staging-capacity:boot",)


@pytest.mark.parametrize("ids", [("movie", "movie"), ("missing",)])
def test_bad_scope_bindings(
    plan: BudgetPlan, observations: tuple[VolumeObservation, ...], ids: tuple[str, ...]
) -> None:
    scope = replace(plan.staging[0], allocation_ids=ids)
    with pytest.raises(ReplayError, match="duplicate or unknown"):
        evaluate(replace(plan, staging=(scope,)), observations)


@pytest.mark.parametrize("owners", [(), ("",), ("native", "native"), ("owner",) * 513])
def test_missing_duplicate_or_unbounded_facts(
    plan: BudgetPlan, observations: tuple[VolumeObservation, ...], owners: tuple[str, ...]
) -> None:
    with pytest.raises(ReplayError):
        evaluate(replace(plan, owners=owners), observations)


def test_missing_planned_path(
    plan: BudgetPlan, observations: tuple[VolumeObservation, ...]
) -> None:
    with pytest.raises(ReplayError, match="Every planned"):
        evaluate(
            replace(plan, expected_paths=plan.expected_paths + ("/media/staging/unknown",)),
            observations,
        )


def test_materialized_bytes_never_reserved_twice(
    plan: BudgetPlan, observations: tuple[VolumeObservation, ...]
) -> None:
    movie = replace(plan.allocations[0], reservation=True, materialized_bytes=100)
    with pytest.raises(ReplayError, match="exclude already"):
        evaluate(replace(plan, allocations=(movie, plan.allocations[1])), observations)


def test_new_output_must_not_be_materialized(
    plan: BudgetPlan, observations: tuple[VolumeObservation, ...]
) -> None:
    movie = replace(plan.allocations[0], materialized_bytes=100)
    with pytest.raises(ReplayError, match="already be materialized"):
        evaluate(replace(plan, allocations=(movie, plan.allocations[1])), observations)


def test_child_parent_binding(
    plan: BudgetPlan, observations: tuple[VolumeObservation, ...]
) -> None:
    movie = replace(plan.allocations[0], path="/other/movie.mov")
    with pytest.raises(ReplayError, match="direct canonical"):
        evaluate(
            replace(
                plan,
                allocations=(movie, plan.allocations[1]),
                expected_paths=("/other/movie.mov", "/work/cache/library"),
            ),
            observations,
        )


def test_normal_floor_not_lowered(
    plan: BudgetPlan, observations: tuple[VolumeObservation, ...]
) -> None:
    with pytest.raises(ReplayError, match="normal volume minimum"):
        evaluate(plan, (replace(observations[0], floor_bytes=0), observations[1]))
