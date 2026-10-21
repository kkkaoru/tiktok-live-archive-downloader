"""Recording-independent runner for the sampled source-caption cleanup of a whole recording.

One JSON description names the recording's measured inputs (dated source and hash, physical
presentation clock, caption profile observations, native tool, pinned detector, roots and
budgets). Three steps reuse the shared stages: ``configs`` derives one chunk job per clock
chunk, ``run`` executes them with bounded concurrent streams and a free-space guard, and
``assemble`` adapts the validated chunks, prepares the assembly manifest, joins the media with
parallel probes and writes the pinned cleanup configuration. Nothing here is specific to a
recording; every outcome stays a candidate (quality and delivery flags are false).
"""

import argparse
import hashlib
import json
import math
import os
import re
import shutil
import subprocess
import sys
import threading
import time
from collections.abc import Callable, Sequence
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path

from replay import delivery_source_assembly_parallel as assembly_runner
from replay import gpu_lease
from replay import subtitle_cleanup_assembly as preparation
from replay import subtitle_sampled_consumer as consumer
from replay import subtitle_yolo_regions as yolo
from replay.recordings import record
from replay.subtitle_cleanup_plan import clock_chunks
from replay.workflow_parallel import MEMORY_FREE_MIN, Load, headroom, read_load

KEYS = frozenset(
    {
        "version",
        "recording",
        "source",
        "sourceSHA256",
        "clock",
        "width",
        "height",
        "profileDirectory",
        "native",
        "detectorProgram",
        "detectorProgramSHA256",
        "settings",
        "work",
        "mediaRoot",
        "ffmpeg",
        "ffprobe",
        "chunkFrames",
        "streams",
        "freeFloorGiB",
        "stride",
        "auditStride",
        "workers",
        "repairWorkers",
        "radius",
    }
)
OPTIONAL_KEYS = frozenset({"regionDetector", "auditTile", "sttn"})
# Run-only budgets: they shape the queue, not any chunk's result, so chunk jobs never see them.
RUN_KEYS = frozenset({"maxStreams"})
GIB = 2**30
MAX_STREAMS = 8
POLL_SECONDS = 15.0
# Polls to wait after a chunk starts before the next admission: a job reaches its STTN buffers
# (its peak) about 25 s in (YOLO ~6 s, repair pass ~19 s, measured 2026-10-20); longer waits
# outlast a whole job and serialise the queue (45 s did on 720x1280 chunks of ~48 s, 2026-10-21).
# The waiter wakes exactly when the settle ends instead of at the next poll (a fixed 15 s poll
# stretched 30 s settles to ~36 s between starts).
SETTLE_SECONDS = 20.0
# A chunk job's peak footprint (measured 9.5-10 GB on 2026-10-20); a further job is admitted only
# while the free memory still covers it above the floor.
JOB_PEAK_BYTES = 10 * 1024**3
JOB_PERCENT = math.ceil(
    100 * JOB_PEAK_BYTES / (os.sysconf("SC_PAGE_SIZE") * os.sysconf("SC_PHYS_PAGES"))
)

Runner = Callable[[Sequence[str], Path], int]
FreeBytes = Callable[[], int]
Measure = Callable[[], Load]


@dataclass(frozen=True, slots=True)
class Recording:
    """Validated description of one recording's cleanup inputs and budgets."""

    root: Path
    data: dict[str, object]

    def text(self, key: str) -> str:
        value = self.data[key]
        if not isinstance(value, str) or not value:
            raise ValueError(f"Expected text for {key}")
        return value

    def number(self, key: str) -> int:
        value = self.data[key]
        if type(value) is not int or value < 1:
            raise ValueError(f"Expected a positive integer for {key}")
        return value

    @property
    def max_streams(self) -> int:
        """Ceiling of concurrent chunks; ``streams`` always run, more only with headroom."""
        return self.number("maxStreams") if "maxStreams" in self.data else self.number("streams")

    @property
    def work(self) -> Path:
        return Path(self.text("work"))

    @property
    def media(self) -> Path:
        return Path(self.text("mediaRoot")) / self.text("recording")


def load(path: Path, *, root: Path) -> Recording:
    """Read a bounded description; absolute paths only, work directory inside the checkout."""
    return validate(record(json.loads(path.read_text(encoding="utf-8"))), root=root)


def validate(data: dict[str, object], *, root: Path) -> Recording:
    if not KEYS <= set(data) <= KEYS | OPTIONAL_KEYS | RUN_KEYS or data["version"] != 1:
        raise ValueError("Unsupported source cleanup description")
    if "regionDetector" in data:
        yolo.settings_from(data["regionDetector"])  # the YOLO search regions / repair masks
    tile = data.get("auditTile", 1)
    if type(tile) is not int or not 1 <= tile <= 4:
        raise ValueError("auditTile must be an integer from 1 to 4")
    result = Recording(root.resolve(), data)
    for key in (
        "source",
        "clock",
        "profileDirectory",
        "native",
        "detectorProgram",
        "settings",
        "work",
    ):
        if not Path(result.text(key)).is_absolute():
            raise ValueError(f"{key} must be an absolute path")
    if not result.work.resolve().is_relative_to(result.root / "edits/work"):
        raise ValueError("The work directory must live under the checkout's edits/work")
    if not 1 <= result.number("streams") <= MAX_STREAMS:
        raise ValueError("Invalid stream count")
    if not result.number("streams") <= result.max_streams <= MAX_STREAMS:
        raise ValueError("maxStreams must lie between streams and 8")
    return result


def _write_new(path: Path, value: object) -> None:
    text = json.dumps(value, indent=2, allow_nan=False) + "\n"
    if path.exists():
        if path.read_text(encoding="utf-8") != text:
            raise ValueError(f"{path.name} exists with different content; use a new work directory")
        return
    with path.open("x", encoding="utf-8") as stream:
        stream.write(text)


def configs(recording: Recording) -> int:
    """Write one chunk job description per clock chunk; returns the chunk count."""
    clock = record(json.loads(Path(recording.text("clock")).read_text(encoding="utf-8")))
    chunks = clock_chunks(clock, maximum_frames=recording.number("chunkFrames"))
    destination = recording.work / "configs"
    destination.mkdir(mode=0o700, parents=True, exist_ok=True)
    for index, chunk in enumerate(chunks):
        name = f"chunk-{index:05d}"
        job = {
            "source": recording.text("source"),
            "sourceSHA256": recording.text("sourceSHA256"),
            "width": recording.number("width"),
            "height": recording.number("height"),
            "clock": {
                "pts": list(chunk.pts),
                "timeBase": str(clock["timeBase"]),
                "lastDuration": chunk.last_duration,
            },
            "native": recording.text("native"),
            "checkoutRoot": str(recording.root),
            "profileDirectory": recording.text("profileDirectory"),
            "detectorProgram": recording.text("detectorProgram"),
            "detectorProgramSHA256": recording.text("detectorProgramSHA256"),
            "output": str(recording.work / "state" / name),
            "mediaOutput": str(recording.media / "sampled" / name),
            "stride": recording.number("stride"),
            "auditStride": recording.number("auditStride"),
            "workers": recording.number("workers"),
            "repairWorkers": recording.number("repairWorkers"),
            "radius": recording.number("radius"),
            "colourKey": True,
            "colourGate": True,
            "ffmpeg": recording.text("ffmpeg"),
        }
        for key in sorted(OPTIONAL_KEYS & set(recording.data)):
            job[key] = recording.data[key]
        _write_new(destination / f"{name}.json", job)
    return len(chunks)


def subprocess_runner(command: Sequence[str], log: Path) -> int:
    with log.open("xb") as stream:
        return subprocess.run(list(command), stdout=stream, stderr=stream, check=False).returncode


def free_bytes() -> int:
    return shutil.disk_usage("/").free


def run(
    recording: Recording,
    *,
    first: int,
    last: int,
    runner: Runner = subprocess_runner,
    free: FreeBytes = free_bytes,
    measure: Measure = read_load,
    poll_seconds: float = POLL_SECONDS,
    settle_seconds: float = SETTLE_SECONDS,
    leases: Path | None = None,
) -> dict[str, object]:
    """Run chunk jobs ``first..last``; skip finished, stop below the floor.

    The first chunk always runs; every further one waits until free memory covers one more job
    above ``MEMORY_FREE_MIN`` (chunk jobs reach ~10 GB, and two recordings' streams together
    exhausted a 48 GB machine). Up to ``streams`` run on memory alone; up to ``maxStreams`` while
    measured load and free memory leave headroom (16 chunks of 10-07: 278 s at 3, 245 s at 5,
    221 s at 8 streams), so the queue fills an idle machine and backs off while other stages share
    it.
    """
    floor = recording.number("freeFloorGiB") * GIB
    leases = gpu_lease.root() if leases is None else leases
    # "replay-<id>-p1" or "youtube-<id>-p1" (a YouTube ID may itself hold hyphens): the
    # pipeline leases under "<id>" or "youtube-<id>".
    replay_id = re.sub(r"-p[0-9]+$", "", recording.text("recording")).removeprefix("replay-")
    base = recording.number("streams")
    admission = threading.Condition()
    active = [0]
    last_start = [-float("inf")]

    def admitted(load: Load) -> bool:
        return load.memory_free_percent >= MEMORY_FREE_MIN + JOB_PERCENT and (
            active[0] < base or headroom(load)
        )

    (recording.work / "logs").mkdir(mode=0o700, parents=True, exist_ok=True)
    stop = threading.Event()
    results: dict[int, dict[str, object]] = {}
    started = time.time()

    def one(index: int) -> None:
        name = f"chunk-{index:05d}"
        config = recording.work / "configs" / f"{name}.json"
        if (recording.work / "state" / name / "chunk-receipt.json").exists():
            results[index] = {"skipped": True}
            return
        if stop.is_set() or free() < floor:
            stop.set()
            results[index] = {"notStarted": True}
            return
        held: int | None = None
        waiter = gpu_lease.Waiter(leases, replay_id)
        with admission:
            # Memory admits a job first; then it also needs one of the machine-wide GPU leases.
            try:
                while held is None:
                    remaining = settle_seconds - (time.monotonic() - last_start[0])
                    load = measure() if active[0] >= 1 and remaining <= 0 else None
                    if active[0] < 1 or (load is not None and admitted(load)):
                        held = waiter.try_acquire()
                    decisions.write(
                        json.dumps(
                            {"t": round(time.time(), 1), "chunk": index, "active": active[0],
                             "settleLeft": round(max(remaining, 0.0), 1),
                             "memoryFree": None if load is None else load.memory_free_percent,
                             "lease": held}
                        ) + "\n"
                    )  # fmt: skip
                    decisions.flush()
                    if held is None:
                        admission.wait(
                            poll_seconds if remaining <= 0 else min(poll_seconds, remaining)
                        )
            finally:
                waiter.close()
            active[0] += 1
            last_start[0] = time.monotonic()
        state = recording.work / "state" / name
        if state.exists():
            # A stopped job left no receipt and its attempt cannot resume: keep its state as
            # evidence, drop its partial media and run the chunk again.
            retired = recording.work / "interrupted" / f"{name}-{time.time_ns()}"
            retired.parent.mkdir(mode=0o700, exist_ok=True)
            state.rename(retired)
            shutil.rmtree(recording.media / "sampled" / name, ignore_errors=True)
        began = time.time()
        command = [sys.executable, "-B", "-m", "replay.subtitle_sampled_job", str(config)]
        try:
            code = runner(command, recording.work / "logs" / f"{name}.log")
        finally:
            gpu_lease.release(held)
            with admission:
                active[0] -= 1
                admission.notify_all()
        results[index] = {
            "exit": code,
            "startedAt": round(began, 1),
            "wallSeconds": round(time.time() - began, 1),
        }

    # One line per admission decision: the evidence for why streams did or did not overlap.
    with (
        (recording.work / "logs" / "admission.jsonl").open("a", encoding="utf-8") as decisions,
        ThreadPoolExecutor(max_workers=recording.max_streams) as pool,
    ):
        list(pool.map(one, range(first, last + 1)))
    summary: dict[str, object] = {
        "chunks": [first, last],
        "wallSeconds": round(time.time() - started, 1),
        "results": {str(key): value for key, value in sorted(results.items())},
    }
    # Each run (a retry after failed chunks too) keeps its own summary; earlier ones stay.
    name = f"queue-summary-{first}-{last}"
    attempt = 1
    while (recording.work / f"{name}.json").exists():
        attempt += 1
        name = f"queue-summary-{first}-{last}-attempt-{attempt}"
    _write_new(recording.work / f"{name}.json", summary)
    return summary


def _pin(path: Path) -> dict[str, str]:
    return {"path": str(path), "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}


def assemble(recording: Recording) -> Path:
    """Adapt validated chunks, join the media, and write the pinned cleanup configuration."""
    output = recording.work / "consumer"
    plan = consumer.adapt(
        consumer.Adaptation(
            configs=recording.work / "configs",
            states=recording.work / "state",
            settings=Path(recording.text("settings")),
            output=output,
            media_root=Path(recording.text("mediaRoot")),
            chunk_frames=recording.number("chunkFrames"),
            source_clock=Path(recording.text("clock")),
        )
    )
    preparation_directory = output / "assembly-preparation"
    preparation_directory.mkdir(mode=0o700)
    manifest = preparation.prepare(plan, preparation_directory)
    destination = recording.media / "assembly"
    destination.mkdir(mode=0o700, parents=True)
    assembly_runner.assemble_parallel(
        manifest,
        sha256=hashlib.sha256(manifest.read_bytes()).hexdigest(),
        output=destination / "source.mov",
        evidence=output / "assembly-evidence",
        ffmpeg=Path(recording.text("ffmpeg")),
        ffprobe=Path(recording.text("ffprobe")),
    )
    cleanup = output / "source-cleanup.json"
    _write_new(
        cleanup,
        {
            "assemblyManifest": _pin(manifest),
            "assemblyReceipt": _pin(output / "assembly-evidence/receipt.json"),
            "cleanIndex": _pin(output / "assembly-evidence/single-source-index.json"),
        },
    )
    return cleanup


DEFAULT_KEYS = frozenset(
    {
        "native",
        "detectorProgram",
        "detectorProgramSHA256",
        "ffmpeg",
        "ffprobe",
        "mediaRoot",
        "chunkFrames",
        "streams",
        "freeFloorGiB",
        "stride",
        "auditStride",
        "workers",
        "repairWorkers",
        "radius",
    }
)


def canvas(ffprobe: str, source: Path) -> tuple[int, int]:
    """Decoded canvas of the first video stream."""
    shown = subprocess.run(
        [ffprobe, "-v", "error", "-select_streams", "v:0", "-show_entries"]
        + ["stream=width,height", "-of", "csv=p=0", str(source)],
        capture_output=True,
        text=True,
        check=True,
    ).stdout.strip()
    width, height = shown.split(",")[:2]
    return int(width), int(height)


def describe(bootstrap: Path, defaults: Path, *, root: Path, revision: str = "v1") -> Path:
    """Write a recording description from its bootstrap directory and the machine defaults.

    The bootstrap directory holds ``source.sha256`` ("<sha>  <path>") and
    ``cleanup-bootstrap-v1/{physical-clock.json,profile,settings.json}``; the defaults name the
    shared tools and budgets once for every recording.
    """
    bootstrap = bootstrap.resolve()
    machine = record(json.loads(defaults.read_text(encoding="utf-8")))
    if not DEFAULT_KEYS <= set(machine) <= DEFAULT_KEYS | OPTIONAL_KEYS | RUN_KEYS:
        raise ValueError("Unsupported cleanup defaults")
    digest, _, name = (
        (bootstrap / "source.sha256").read_text(encoding="utf-8").strip().partition("  ")
    )
    source = Path(name)
    if len(digest) != 64 or not source.is_absolute() or not source.is_file():
        raise ValueError("Invalid recorded source hash or missing source media")
    width, height = canvas(str(machine["ffprobe"]), source)
    prepared = bootstrap / "cleanup-bootstrap-v1"
    data: dict[str, object] = {
        **machine,
        "version": 1,
        # The revision keeps each cleanup's chunk and assembly media apart under the media root.
        "recording": f"{bootstrap.name}-{revision}",
        "source": str(source),
        "sourceSHA256": digest,
        "clock": str(prepared / "physical-clock.json"),
        "width": width,
        "height": height,
        "profileDirectory": str(prepared / "profile"),
        "settings": str(prepared / "settings.json"),
        "work": str(root.resolve() / "edits/work" / f"clean-{bootstrap.name}-{revision}"),
    }
    destination = (
        root.resolve() / "edits/work" / f"clean-{bootstrap.name}-{revision}.description.json"
    )
    validate(data, root=root)
    destination.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
    with destination.open("x", encoding="utf-8") as stream:
        json.dump(data, stream, indent=2, allow_nan=False)
        stream.write("\n")
    return destination


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("description", type=Path)
    parser.add_argument("step", choices=("describe", "configs", "run", "assemble"))
    parser.add_argument("--first", type=int, default=0)
    parser.add_argument("--last", type=int)
    parser.add_argument("--defaults", type=Path)
    parser.add_argument(
        "--revision", default="v1", help="describe: suffix of the new work directory (v1, v2, ...)"
    )
    args = parser.parse_args(argv)
    if args.step == "describe":
        if args.defaults is None:
            raise ValueError("describe needs --defaults")
        if not args.revision.isalnum():
            raise ValueError("The revision must be alphanumeric")
        print(describe(args.description, args.defaults, root=Path.cwd(), revision=args.revision))
        return 0
    recording = load(args.description, root=Path.cwd())
    if args.step == "configs":
        print(configs(recording))
    elif args.step == "run":
        count = len(list((recording.work / "configs").glob("chunk-*.json")))
        last = count - 1 if args.last is None else args.last
        summary = run(recording, first=args.first, last=last)
        failed = [k for k, v in record(summary["results"]).items() if record(v).get("exit", 0) != 0]
        print(json.dumps({"failed": failed, "wallSeconds": summary["wallSeconds"]}))
        return 1 if failed else 0
    else:
        print(assemble(recording))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
