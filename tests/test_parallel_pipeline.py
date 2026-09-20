"""Bounded completion concurrency, ordering, and resource settlement."""

from threading import Event
from unittest.mock import Mock

import pytest

from replay.parallel_pipeline import run_parallel_pipeline


def test_empty_does_not_call_workers() -> None:
    prepare, finish = Mock(), Mock()
    assert list(run_parallel_pipeline([], prepare=prepare, finish=finish)) == []
    prepare.assert_not_called()
    finish.assert_not_called()


@pytest.mark.parametrize("workers", [0, 3, True])
def test_invalid_worker_limit(workers: int) -> None:
    with pytest.raises(ValueError, match="one or two"):
        list(run_parallel_pipeline([], prepare=Mock(), finish=Mock(), finish_workers=workers))


def test_two_finishers_overlap_and_results_remain_ordered() -> None:
    second_started = Event()
    third_prepared = Event()

    def prepare(item: int) -> int:
        if item == 3:
            third_prepared.set()
        return item * 10

    def finish(item: int, prepared: int) -> int:
        if item == 1 and not second_started.wait(timeout=5):
            raise RuntimeError("Finishers did not overlap")
        if item == 2:
            second_started.set()
            if not third_prepared.wait(timeout=5):
                raise RuntimeError("Preparation did not overlap two finishers")
        return item + prepared

    assert list(run_parallel_pipeline((1, 2, 3, 4), prepare=prepare, finish=finish)) == [
        11,
        22,
        33,
        44,
    ]


def test_close_settles_only_bounded_submitted_work() -> None:
    prepare = Mock(side_effect=lambda item: item)
    finish = Mock(side_effect=lambda item, prepared: prepared)
    iterator = run_parallel_pipeline(range(10), prepare=prepare, finish=finish)
    assert next(iterator) == 0
    iterator.close()
    assert prepare.call_count == 3
    assert finish.call_count == 3


def test_prepare_failure_settles_already_submitted_finish() -> None:
    settled = Event()
    prepare = Mock(side_effect=[1, ValueError("prepare failed")])

    def finish(item: int, prepared: int) -> int:
        settled.set()
        return item + prepared

    with pytest.raises(ValueError, match="prepare failed"):
        list(run_parallel_pipeline((1, 2, 3), prepare=prepare, finish=finish))
    assert settled.is_set()
    assert prepare.call_count == 2


def test_finish_failure_does_not_retry_or_prepare_unbounded_work() -> None:
    prepare = Mock(side_effect=lambda item: item)
    finish = Mock(side_effect=ValueError("finish failed"))
    with pytest.raises(ValueError, match="finish failed"):
        list(run_parallel_pipeline(range(20), prepare=prepare, finish=finish))
    assert prepare.call_count == 3
    assert finish.call_count == 3


def test_one_finisher_is_supported() -> None:
    assert list(
        run_parallel_pipeline(
            (1, 2, 3),
            prepare=lambda item: item * 10,
            finish=lambda item, prepared: item + prepared,
            finish_workers=1,
        )
    ) == [11, 22, 33]
