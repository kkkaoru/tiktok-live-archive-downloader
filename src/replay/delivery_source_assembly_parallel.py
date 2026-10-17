"""Run ``delivery_source_assembly.assemble`` with the per-chunk input probes in parallel.

The serial per-chunk ffprobe phase dominates the wait for a many-chunk assembly. This wrapper
starts every chunk probe through the unchanged ``probe`` function in worker threads when the
assembly asks for the first one, then hands the results back in assembly order. Evidence files,
commands and verification are those of the original function, so outputs are byte-identical.
The joined-output probe and every verification rule stay serial and untouched.
"""

import threading
from collections.abc import Callable
from concurrent.futures import Future, ThreadPoolExecutor
from pathlib import Path

from replay import delivery_source_assembly as assembly

INPUT_PREFIX = "input-"
DEFAULT_WORKERS = 10
MAX_WORKERS = 32

Probe = Callable[..., assembly.Probe]


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

        assembly.probe = prefetching
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
