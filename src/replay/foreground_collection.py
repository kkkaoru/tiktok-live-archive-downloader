"""Collect bounded exact-frame foreground evidence, never caption approval."""

import argparse
import hashlib
import json
import os
import re
import subprocess
import sys
from dataclasses import dataclass
from fractions import Fraction
from itertools import batched
from pathlib import Path

from replay import tracking_observations as storage
from replay.foreground_observations import MAX_SAMPLES, foreground_frames
from replay.head_observations import validate_tracking_window
from replay.recordings import record
from replay.stages import Stage, file_digest, run_stage
from replay.subtitle_clock import VideoFrameClock

MAX_BATCHES = 16
MAX_FRAMES = MAX_BATCHES * MAX_SAMPLES
RESPONSE_BYTES = 1_048_576
BATCH_TIMEOUT_SECONDS = 60
STAGE_TIMEOUT_SECONDS = 1200
CODE_FILES = (*storage.CODE_FILES, "foreground_observations.py", "foreground_collection.py")


@dataclass(frozen=True, slots=True)
class ForegroundJob:
    config: Path
    config_hash: str
    source: Path
    native: Path
    source_hash: str
    native_hash: str
    clock: VideoFrameClock
    width: int
    height: int

    def request(self, ticks: tuple[int, ...]) -> dict[str, object]:
        return {
            "path": str(self.source),
            "samples": [
                {
                    "value": tick * self.clock.time_base.numerator,
                    "timescale": self.clock.time_base.denominator,
                }
                for tick in ticks
            ],
        }


def load_job(config: Path) -> ForegroundJob:
    data = storage._read(config, storage.CONFIG_BYTES)
    raw = record(json.loads(data))
    if (
        set(raw)
        != {
            "version",
            "source",
            "sourceSHA256",
            "nativeExecutable",
            "nativeSHA256",
            "clock",
            "width",
            "height",
        }
        or type(raw["version"]) is not int
        or raw["version"] != 1
    ):
        raise ValueError("Invalid foreground collection configuration")
    clock = record(raw["clock"])
    pts, base = clock.get("pts"), clock.get("timeBase")
    if (
        set(clock) != {"pts", "timeBase", "lastDuration"}
        or not isinstance(pts, list)
        or not 1 <= len(pts) <= MAX_FRAMES
        or not isinstance(base, str)
        or re.fullmatch(r"[1-9][0-9]{0,9}/[1-9][0-9]{0,9}", base) is None
    ):
        raise ValueError("Invalid bounded foreground source clock")
    source_clock = VideoFrameClock(
        tuple(storage._integer(tick) for tick in pts),
        Fraction(base),
        storage._integer(clock["lastDuration"]),
    )
    width, height = storage._integer(raw["width"]), storage._integer(raw["height"])
    validate_tracking_window(source_clock, width=width, height=height)
    return ForegroundJob(
        config.absolute(),
        hashlib.sha256(data).hexdigest(),
        storage._path(raw["source"]),
        storage._path(raw["nativeExecutable"]),
        storage._digest(raw["sourceSHA256"]),
        storage._digest(raw["nativeSHA256"]),
        source_clock,
        width,
        height,
    )


def _bindings(job: ForegroundJob) -> dict[str, str]:
    paths = (job.config, job.source, job.native, *(Path(__file__).with_name(n) for n in CODE_FILES))
    if len(set(paths)) != len(paths):
        raise ValueError("Foreground data must not alias other inputs or code")
    hashes = {str(path): file_digest(path) for path in paths}
    if (
        hashes[str(job.config)] != job.config_hash
        or hashes[str(job.source)] != job.source_hash
        or hashes[str(job.native)] != job.native_hash
    ):
        raise ValueError("Foreground source, executable or configuration changed")
    if not os.access(job.native, os.X_OK):
        raise ValueError("Foreground executable is not executable")
    return hashes


def _batch_receipt(
    job: ForegroundJob, directory: Path, ticks: tuple[int, ...]
) -> dict[str, object]:
    directory = storage._checked_output(directory)
    request, response = directory / "request.json", directory / "response.json"
    recorded = json.loads(storage._read(request, storage.CONFIG_BYTES))
    if json.dumps(recorded, sort_keys=True) != json.dumps(job.request(ticks), sort_keys=True):
        raise ValueError("Foreground request differs from its exact source clock")
    frames = foreground_frames(
        json.loads(storage._read(response, RESPONSE_BYTES)),
        times=tuple(tick * job.clock.time_base for tick in ticks),
        width=job.width,
        height=job.height,
    )
    storage._check_diagnostics(directory)
    return {
        "requestSHA256": file_digest(request),
        "responseSHA256": file_digest(response),
        "frames": len(frames),
    }


def _collect_batch(job: ForegroundJob, directory: Path, ticks: tuple[int, ...]) -> None:
    directory.mkdir(mode=0o700)
    request = directory / "request.json"
    storage._write(request, job.request(ticks))
    request_hash = file_digest(request)
    with (
        (directory / "response.json").open("xb") as stdout,
        (directory / "stderr.log").open("xb") as stderr,
    ):
        os.fchmod(stdout.fileno(), 0o600)
        os.fchmod(stderr.fileno(), 0o600)
        subprocess.run(
            (str(job.native), "measure-media", "video_foreground_measure", str(request)),
            stdin=subprocess.DEVNULL,
            stdout=stdout,
            stderr=stderr,
            timeout=BATCH_TIMEOUT_SECONDS,
            check=True,
        )
    if file_digest(request) != request_hash:
        raise ValueError("Foreground request changed during execution")


def _receipt(job: ForegroundJob, output: Path, inputs: dict[str, str]) -> dict[str, object]:
    batches = [
        _batch_receipt(job, output / f"batch-{index:04d}", ticks)
        for index, ticks in enumerate(batched(job.clock.pts, MAX_SAMPLES))
    ]
    return {
        "inputs": inputs,
        "batches": batches,
        "frames": len(job.clock.pts),
        "nativeExitCode": 0,
        "captionPlacementApproved": False,
    }


def observe(config: Path, output: Path) -> None:
    """New evidence only; preserve partial batches and never infer subject absence.

    At most sixteen native calls per stage amortize source hashing. Source clocks
    and their provenance must be independently verified by the caller. Unchanged
    completed stages are reusable through the existing stage runner, not adopted.
    """
    job = load_job(config)
    inputs = _bindings(job)
    output = storage._checked_output(output)
    output.mkdir(mode=0o700)
    storage._write(output / "inputs.json", inputs)
    for index, ticks in enumerate(batched(job.clock.pts, MAX_SAMPLES)):
        directory = output / f"batch-{index:04d}"
        _collect_batch(job, directory, ticks)
        _batch_receipt(job, directory, ticks)
    receipt = _receipt(job, output, inputs)
    if _bindings(job) != inputs:
        raise ValueError("Foreground inputs changed during collection; attempt retained")
    storage._write(output / "provenance.json", receipt)


def verify(config: Path, output: Path) -> None:
    """Check all referenced batches again; missing completed output is a stop."""
    job = load_job(config)
    inputs = _bindings(job)
    output = storage._checked_output(output)
    stored_inputs = json.loads(storage._read(output / "inputs.json", storage.CONFIG_BYTES))
    recorded = json.loads(storage._read(output / "provenance.json", storage.CONFIG_BYTES))
    expected = _receipt(job, output, inputs)
    if stored_inputs != inputs or json.dumps(recorded, sort_keys=True) != json.dumps(
        expected, sort_keys=True
    ):
        raise ValueError("Foreground provenance changed")
    if _bindings(job) != inputs:
        raise ValueError("Foreground inputs changed during verification")


def foreground_stage(config: Path, output: Path, *, name: str) -> Stage:
    job = load_job(config)
    inputs = _bindings(job)
    output = storage._checked_output(output)
    command = (sys.executable, "-m", "replay.foreground_collection")
    return Stage(
        name=name,
        command=(*command, "observe", str(job.config), str(output)),
        verify_command=(*command, "verify", str(job.config), str(output)),
        inputs=tuple(Path(path) for path in inputs),
        outputs=(output / "inputs.json", output / "provenance.json"),
        timeout_seconds=STAGE_TIMEOUT_SECONDS,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("collect", "observe", "verify"))
    parser.add_argument("config", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--state-directory", type=Path)
    parser.add_argument("--name", default="foreground-measurement")
    args = parser.parse_args()
    if args.action == "collect":
        if args.state_directory is None:
            parser.error("collect requires --state-directory")
        result = run_stage(
            foreground_stage(args.config, args.output, name=args.name),
            state_directory=storage._checked_output(args.state_directory),
        )
        print(f"Foreground window: {result}; caption placement is not approved")
    elif args.action == "observe":
        observe(args.config, args.output)
    else:
        verify(args.config, args.output)


if __name__ == "__main__":
    main()
