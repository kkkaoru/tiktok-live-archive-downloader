"""Run ``delivery_source_assembly.assemble`` with the per-chunk input probes in parallel.

The serial per-chunk ffprobe phase dominates the wait for a many-chunk assembly. This wrapper
starts every chunk probe through the unchanged ``probe`` function in worker threads when the
assembly asks for the first one, then hands the results back in assembly order. Evidence files
and verification are those of the original function, so outputs are byte-identical.

ffprobe decodes on one core unless told otherwise; the decoded-clock probe of the 158k-frame
joined 10-07 output took 535 s single-threaded and 60 s with ``-threads 0``, with a
byte-identical ``output-frames.json``. Decoded-clock probes therefore get decoder threads: all
cores for the joined output, an equal share per worker for the concurrent chunk probes. Only the
recorded command gains ``-threads``; every verification rule stays serial and untouched.
"""

import os
import threading
from collections.abc import Callable
from concurrent.futures import Future, ThreadPoolExecutor
from pathlib import Path

from replay import delivery_source_assembly as assembly

INPUT_PREFIX = "input-"
DEFAULT_WORKERS = 10
MAX_WORKERS = 32

FRAME_ENTRIES = "frame=pts,duration"
OUTPUT_PREFIX = "output-"

Probe = Callable[..., assembly.Probe]
Run = Callable[..., None]


def decoder_threads(command: list[str], stdout: Path, workers: int) -> list[str]:
    """The command with decoder threads when it is a decoded-clock probe, else unchanged."""
    if FRAME_ENTRIES not in command or "-threads" in command:
        return command
    share = 0 if stdout.name.startswith(OUTPUT_PREFIX) else max(1, (os.cpu_count() or 1) // workers)
    at = command.index("-show_entries")
    return [*command[:at], "-threads", str(share), *command[at:]]


def assemble_parallel(
    manifest: Path,
    *,
    sha256: str,
    output: Path,
    evidence: Path,
    ffmpeg: Path,
    ffprobe: Path,
    workers: int = DEFAULT_WORKERS,
) -> Path:
    """Same contract as ``assemble``; chunk probes run concurrently."""
    if not 1 <= workers <= MAX_WORKERS:
        raise ValueError("Invalid probe worker count")
    chunks = assembly.load_chunks(manifest, sha256=sha256, bindings={})
    names = {f"{INPUT_PREFIX}{index:04d}": chunk.media for index, chunk in enumerate(chunks)}
    original: Probe = assembly.probe
    original_run: Run = assembly._run
    futures: dict[str, Future[assembly.Probe]] = {}
    lock = threading.Lock()
    with ThreadPoolExecutor(max_workers=workers) as pool:

        def prefetching(media: Path, **keywords: object) -> assembly.Probe:
            name = keywords["name"]
            with lock:
                if not futures:
                    futures.update(
                        {
                            key: pool.submit(original, path, **{**keywords, "name": key})
                            for key, path in names.items()
                        }
                    )
            if isinstance(name, str) and name in futures:
                return futures[name].result()
            return original(media, **keywords)

        def threaded(command: list[str], stdout: Path, **keywords: object) -> None:
            original_run(decoder_threads(command, stdout, workers), stdout, **keywords)

        assembly.probe = prefetching
        assembly._run = threaded
        try:
            return assembly.assemble(
                manifest,
                sha256=sha256,
                output=output,
                evidence=evidence,
                ffmpeg=ffmpeg,
                ffprobe=ffprobe,
            )
        finally:
            assembly.probe = original
            assembly._run = original_run
