"""Observe every original terminal-frame PTS with a measured native selection end.

Packet presentation duration and AVFoundation asset extent remain separate facts.
This receipt is deliberately not a legacy tracking-observations receipt.
"""

import argparse
import json
import math
import os
import subprocess
from pathlib import Path

from replay.head_observations import tracking_frames
from replay.recordings import record
from replay.stages import file_digest
from replay.tracking_observations import (
    CONFIG_BYTES,
    RESPONSE_BYTES,
    TIMEOUT_SECONDS,
    TrackingJob,
    _bindings,
    _checked_output,
    _read,
    _write,
    load_job,
)

KIND = "native-terminal-selection-observations-v1"


def selection_request(job: TrackingJob, inspection: object) -> dict[str, object]:
    """Require a native extent strictly inside the final original display cell."""
    raw = record(inspection)
    duration = raw.get("durationSeconds")
    if (
        isinstance(duration, bool)
        or not isinstance(duration, (int, float))
        or not math.isfinite(duration)
        or raw.get("width") != job.width
        or raw.get("height") != job.height
        or raw.get("firstFrameDecoded") is not True
    ):
        raise ValueError("Invalid native terminal extent observation")
    base = job.clock.time_base
    end = round(duration / float(base))
    last = job.clock.pts[-1]
    if float(end * base) != duration or not last < end < last + job.clock.last_duration:
        raise ValueError("Native extent is not inside the original terminal frame")
    request = job.request()
    plan = record(request["plan"])
    plan["durationSeconds"] = float((end - job.clock.pts[0]) * base)
    if float(job.clock.pts[0] * base) + float((end - job.clock.pts[0]) * base) > duration:
        raise ValueError("Native selection cannot represent the observed extent")
    request["plan"] = plan
    return request


def _inputs(job: TrackingJob) -> dict[str, str]:
    return {**_bindings(job), str(Path(__file__).resolve()): file_digest(Path(__file__))}


def _execute(command: tuple[str, ...], output: Path, error: Path) -> None:
    with output.open("xb") as stdout, error.open("xb") as stderr:
        os.fchmod(stdout.fileno(), 0o600)
        os.fchmod(stderr.fileno(), 0o600)
        subprocess.run(
            command,
            stdin=subprocess.DEVNULL,
            stdout=stdout,
            stderr=stderr,
            timeout=TIMEOUT_SECONDS,
            check=True,
        )


def _receipt(job: TrackingJob, output: Path, inputs: dict[str, str]) -> dict[str, object]:
    inspection = json.loads(_read(output / "inspection.json", CONFIG_BYTES))
    expected = selection_request(job, inspection)
    if json.loads(_read(output / "request.json", CONFIG_BYTES)) != expected:
        raise ValueError("Terminal selection differs from the measured native extent")
    frames = tracking_frames(
        json.loads(_read(output / "response.json", RESPONSE_BYTES)),
        job.clock,
        width=job.width,
        height=job.height,
    )
    for name in ("inspection.stderr", "stderr.log"):
        path = output / name
        if path.is_symlink() or not path.is_file() or path.stat().st_size:
            raise ValueError("Native terminal selection has diagnostics")
    artifacts = ("inspection.json", "request.json", "response.json")
    return {
        "version": 1,
        "kind": KIND,
        "inputs": inputs,
        "artifacts": {name: file_digest(output / name) for name in artifacts},
        "originalClock": {
            "pts": list(job.clock.pts),
            "timeBase": str(job.clock.time_base),
            "lastDuration": job.clock.last_duration,
        },
        "frames": len(frames),
        "allOriginalPTSVerified": True,
        "nativeExitCode": 0,
        "captionPlacementApproved": False,
    }


def observe(config: Path, output: Path) -> None:
    """Execute an existing native inspector and tracker once, without changing clocks."""
    job = load_job(config)
    inputs = _inputs(job)
    output = _checked_output(output)
    output.mkdir(mode=0o700)
    _execute(
        (str(job.native), "inspect-media", str(job.source)),
        output / "inspection.json",
        output / "inspection.stderr",
    )
    request = selection_request(job, json.loads(_read(output / "inspection.json", CONFIG_BYTES)))
    _write(output / "request.json", request)
    _execute(
        (str(job.native), "measure-media", "video_face_track", str(output / "request.json")),
        output / "response.json",
        output / "stderr.log",
    )
    receipt = _receipt(job, output, inputs)
    if _inputs(job) != inputs:
        raise ValueError("Terminal selection inputs changed")
    _write(output / "receipt.json", receipt)


def verify(config: Path, output: Path) -> None:
    """Verify exact original PTS coverage and the distinct native selection contract."""
    job = load_job(config)
    inputs = _inputs(job)
    output = _checked_output(output)
    expected = _receipt(job, output, inputs)
    actual = json.loads(_read(output / "receipt.json", RESPONSE_BYTES))
    if json.dumps(actual, sort_keys=True) != json.dumps(expected, sort_keys=True):
        raise ValueError("Terminal selection receipt changed")
    if _inputs(job) != inputs:
        raise ValueError("Terminal selection inputs changed during verification")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("observe", "verify"))
    parser.add_argument("config", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    {"observe": observe, "verify": verify}[args.action](args.config, args.output)


if __name__ == "__main__":
    main()
