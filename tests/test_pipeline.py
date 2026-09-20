"""Bounded pipeline ordering, overlap and failure behavior."""

from threading import Event
from unittest.mock import Mock

import pytest

from replay.pipeline import run_pipeline


def test_empty_pipeline_does_not_launch_callbacks() -> None:
    prepare, finish = Mock(), Mock()
    assert list(run_pipeline([], prepare=prepare, finish=finish)) == []
    prepare.assert_not_called()
    finish.assert_not_called()


def test_ordered_results_and_one_item_lookahead() -> None:
    next_prepared = Event()
    first_finished = Event()

    def prepare(item: int) -> int:
        if item == 2:
            next_prepared.set()
        if item == 3 and not first_finished.is_set():
            raise RuntimeError("More than one item was prefetched")
        return item * 10

    def finish(item: int, prepared: int) -> int:
        if item == 1:
            if not next_prepared.wait(timeout=5):
                raise RuntimeError("Preparation did not overlap finish")
            first_finished.set()
        return item + prepared

    assert list(run_pipeline((1, 2, 3), prepare=prepare, finish=finish)) == [11, 22, 33]


def test_prepare_error_is_not_retried() -> None:
    prepare = Mock(side_effect=ValueError("prepare failed"))
    finish = Mock()
    with pytest.raises(ValueError, match="prepare failed"):
        list(run_pipeline((1, 2), prepare=prepare, finish=finish))
    assert prepare.call_count == 1
    finish.assert_not_called()


def test_finish_failure_settles_existing_preparation() -> None:
    prepared_second = Event()

    def prepare(item: int) -> int:
        if item == 2:
            prepared_second.set()
        return item

    with pytest.raises(ValueError, match="finish failed"):
        list(
            run_pipeline(
                (1, 2, 3), prepare=prepare, finish=Mock(side_effect=ValueError("finish failed"))
            )
        )
    assert prepared_second.is_set()


def test_generator_close_does_not_start_third_item() -> None:
    prepare = Mock(side_effect=lambda item: item)
    iterator = run_pipeline((1, 2, 3), prepare=prepare, finish=lambda item, prepared: prepared)
    assert next(iterator) == 1
    iterator.close()
    assert prepare.call_count == 2
