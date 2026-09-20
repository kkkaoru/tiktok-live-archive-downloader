"""Bounded one-item lookahead for independent local preparation and completion."""

from collections.abc import Callable, Generator, Iterable, Iterator
from concurrent.futures import Future, ThreadPoolExecutor


def _next_job[T, P](
    items: Iterator[T], executor: ThreadPoolExecutor, prepare: Callable[[T], P]
) -> tuple[T, Future[P]] | None:
    try:
        item = next(items)
    except StopIteration:
        return None
    return item, executor.submit(prepare, item)


def run_pipeline[T, P, R](
    items: Iterable[T], *, prepare: Callable[[T], P], finish: Callable[[T, P], R]
) -> Generator[R, None, None]:
    """Overlap next preparation with current finish, retaining ordered results.

    Exactly one preparation and one finish may run concurrently, with at most
    one prepared result queued. Callbacks must own distinct immutable artifacts
    and persist receipts for independently completed work. On failure an already
    running preparation is allowed to settle; it is never silently resubmitted.
    Closing this generator waits for that preparation rather than abandoning it.
    """
    iterator = iter(items)
    with ThreadPoolExecutor(max_workers=1, thread_name_prefix="media-prepare") as executor:
        pending = _next_job(iterator, executor, prepare)
        while pending is not None:
            item, future = pending
            prepared = future.result()
            pending = _next_job(iterator, executor, prepare)
            yield finish(item, prepared)
