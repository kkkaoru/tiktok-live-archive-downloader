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

``prefetch`` probes a chunk as soon as its media is final (the cleanup queue calls it when a chunk
is accepted) with the unchanged ``probe`` into a cache. The assembly then copies a cached probe's
output into its evidence instead of running ffprobe again, but only when the cached command is
exactly the one it would record and the media hashed before that probe has the manifest's pin;
``probe`` still parses and checks the copied output, and the joined output is always probed.
"""

import hashlib
import json
import os
import shutil
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
CACHE_NAME = "probe"
CACHE_MEDIA = "media.json"
PROBE_KINDS = ("streams", "frames", "packets")
OUTPUT_SUFFIXES = (".json", ".command.json", ".stderr")

Probe = Callable[..., assembly.Probe]
Run = Callable[..., None]


def decoder_threads(command: list[str], stdout: Path, workers: int) -> list[str]:
    """The command with decoder threads when it is a decoded-clock probe, else unchanged."""
    if FRAME_ENTRIES not in command or "-threads" in command:
        return command
    share = 0 if stdout.name.startswith(OUTPUT_PREFIX) else max(1, (os.cpu_count() or 1) // workers)
    at = command.index("-show_entries")
    return [*command[:at], "-threads", str(share), *command[at:]]


def cache_directory(probes: Path, media: Path) -> Path:
    """Where ``prefetch`` keeps the probe of one chunk's media."""
    return probes / hashlib.sha256(str(media).encode()).hexdigest()[:32]


def prefetch(media: Path, *, probes: Path, ffprobe: Path) -> Path:
    """Probe final chunk media ahead of its assembly; an existing cache is kept."""
    target = cache_directory(probes, media)
    if target.exists():
        return target
    partial = target.with_name(target.name + ".partial")
    shutil.rmtree(partial, ignore_errors=True)
    partial.mkdir(mode=0o700, parents=True)
    # Hashed before the probe: media changed later can only fail the pin check, never pass it.
    sha256 = assembly._uncached_digest(media)
    assembly.probe(media, directory=partial, name=CACHE_NAME, ffprobe=ffprobe)
    (partial / CACHE_MEDIA).write_text(
        json.dumps({"path": str(media), "sha256": sha256}) + "\n", encoding="utf-8"
    )
    partial.rename(target)
    return target


def _copy_new(source: Path, target: Path) -> None:
    with source.open("rb") as reader, target.open("xb") as writer:
        os.fchmod(writer.fileno(), 0o600)
        shutil.copyfileobj(reader, writer)


def _cached(command: list[str], stdout: Path, probes: Path, pins: dict[str, str]) -> Path | None:
    """The cached output of exactly this chunk probe, or None when it must run."""
    kind = stdout.name.removesuffix(".json").rpartition("-")[2]
    if not stdout.name.startswith(INPUT_PREFIX) or kind not in PROBE_KINDS:
        return None
    media = command[-1]
    cache = cache_directory(probes, Path(media))
    output = cache / f"{CACHE_NAME}-{kind}.json"
    if not output.is_file() or not (cache / CACHE_MEDIA).is_file():
        return None
    recorded = json.loads((cache / CACHE_MEDIA).read_text(encoding="utf-8"))
    same = json.loads(output.with_suffix(".command.json").read_text(encoding="utf-8")) == command
    pinned = recorded == {"path": media, "sha256": pins.get(media)}
    return output if same and pinned else None


def assemble_parallel(
    manifest: Path,
    *,
    sha256: str,
    output: Path,
    evidence: Path,
    ffmpeg: Path,
    ffprobe: Path,
    workers: int = DEFAULT_WORKERS,
    probes: Path | None = None,
) -> Path:
    """Same contract as ``assemble``; chunk probes run concurrently or come from ``probes``."""
    if not 1 <= workers <= MAX_WORKERS:
        raise ValueError("Invalid probe worker count")
    pins: dict[str, str] = {}
    chunks = assembly.load_chunks(manifest, sha256=sha256, bindings=pins)
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
            cached = None if probes is None else _cached(command, stdout, probes, pins)
            if cached is None:
                original_run(decoder_threads(command, stdout, workers), stdout, **keywords)
                return
            for suffix in OUTPUT_SUFFIXES:
                _copy_new(cached.with_suffix(suffix), stdout.with_suffix(suffix))

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
