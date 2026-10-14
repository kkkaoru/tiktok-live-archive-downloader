from __future__ import annotations

from collections.abc import Callable

import pytest

from replay import worker_budget
from replay.worker_budget import Resources


def read(
    *, cpus: int = 15, load: float = 2.0, total: float = 12288.0, used: float = 2048.0
) -> Callable[[], Resources]:
    return lambda: Resources(cpus, load, total, used)


def test_headroom_doubles_the_base_up_to_the_ceiling_and_half_the_cpus() -> None:
    assert worker_budget.workers(3, read=read()) == 6
    assert worker_budget.workers(3, maximum=4, read=read()) == 4
    assert worker_budget.workers(4, read=read(cpus=6)) == 4  # half of 6 cpus is below the base


def test_full_swap_or_a_busy_machine_keeps_the_base() -> None:
    assert worker_budget.workers(3, read=read(used=11700.0)) == 3
    assert worker_budget.workers(3, read=read(load=9.0)) == 3
    assert worker_budget.workers(3, read=read(total=0.0, used=0.0)) == 6  # no swap configured


def test_invalid_base_is_rejected() -> None:
    with pytest.raises(ValueError, match="positive"):
        worker_budget.workers(0, read=read())


def test_live_reading_has_sane_values() -> None:
    now = worker_budget.read_resources()
    assert now.cpus >= 1 and now.swap_total_mb >= 0.0
    assert worker_budget.workers(3) >= 3
