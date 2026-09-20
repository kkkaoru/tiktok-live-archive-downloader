"""Sequential preparation with bounded, ordered parallel completion."""

from collections import deque
from collections.abc import Callable, Generator, Iterable
from concurrent.futures import Future, ThreadPoolExecutor


def run_parallel_pipeline[T, P, R](
    items: Iterable[T],
    *,
    prepare: Callable[[T], P],
    finish: Callable[[T, P], R],
    finish_workers: int = 2,
) -> Generator[R, None, None]:
    """Run one producer and at most two finishers, preserving result order.

    At most workers+1 completions are outstanding, including one queued item.
    Callbacks must own disjoint output artifacts, have their own timeouts, and
    persist independent completion receipts. Failure/close settles submitted
    work without retries; it does not abandon subprocesses owned by callbacks.
    Preparation is serial and may overlap both finishers. No active callback
    configuration is mutated by this scheduler.
    """
    if type(finish_workers) is not int or not 1 <= finish_workers <= 2:
        raise ValueError("Expected one or two finish workers")
    pending: deque[Future[R]] = deque()
    with ThreadPoolExecutor(
        max_workers=finish_workers, thread_name_prefix="media-finish"
    ) as executor:
        for item in items:
            if len(pending) == finish_workers + 1:
                yield pending.popleft().result()
            prepared = prepare(item)
            pending.append(executor.submit(finish, item, prepared))
        while pending:
            yield pending.popleft().result()
