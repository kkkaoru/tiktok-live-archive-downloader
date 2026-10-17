"""Release PyTorch's MPSGraph cache so macOS removes each executable's scratch directory.

Every `MPSGraphExecutable` creates an empty `mpsgraph-<pid>-<timestamp>-<n>` directory under the
per-user temporary directory and removes it again in `dealloc`. PyTorch's MPS backend keeps every
graph it compiles (one per new input shape) until `torch.mps.empty_cache()` releases them, which
it does since pytorch/pytorch#181485 (verified with torch 2.14.1). A process that exits without
releasing them leaves every directory behind, and the boot-time temporary-directory cleanup of
millions of them delayed a boot by 35 minutes. Ultralytics training already releases the cache at
each epoch end; the MPS entry points here also release it when their work ends, including when
the job is stopped with SIGTERM; libraries that start MPS work register the release at exit.
"""

import atexit
import functools
import signal
import sys
import threading
from collections.abc import Iterator
from contextlib import contextmanager
from types import FrameType

TERMINATED_EXIT_CODE = 128 + signal.SIGTERM


def release_mps_graphs() -> None:
    """Release cached MPS graphs; nothing to do when this process never loaded torch."""
    if "torch" not in sys.modules:
        return
    import torch  # already loaded; imported here so callers never pay for it

    if torch.backends.mps.is_available():
        torch.mps.empty_cache()


def exit_on_sigterm(signum: int, frame: FrameType | None) -> None:
    """Turn SIGTERM into SystemExit so cleanup blocks run before the process ends."""
    raise SystemExit(TERMINATED_EXIT_CODE)


@contextmanager
def released_mps_graphs() -> Iterator[None]:
    """Release cached MPS graphs when the block ends, raises or is stopped with SIGTERM."""
    on_main_thread = threading.current_thread() is threading.main_thread()
    previous = signal.signal(signal.SIGTERM, exit_on_sigterm) if on_main_thread else None
    try:
        yield
    finally:
        if on_main_thread:
            signal.signal(signal.SIGTERM, previous)
        release_mps_graphs()


@functools.cache
def release_mps_graphs_at_exit() -> None:
    """For library code that starts MPS work outside a `released_mps_graphs` block.

    Registers the release once per process for interpreter exit. A default (untouched) SIGTERM
    handler on the main thread becomes SystemExit so a stopped job still reaches the release;
    a handler the application installed itself is left alone.
    """
    atexit.register(release_mps_graphs)
    if threading.current_thread() is not threading.main_thread():
        return
    if signal.getsignal(signal.SIGTERM) is signal.SIG_DFL:
        signal.signal(signal.SIGTERM, exit_on_sigterm)
