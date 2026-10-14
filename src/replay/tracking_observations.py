"""One source-clock-bound native tracking window, using the existing stage runner."""

import argparse
import hashlib
import json
import os
import re
import stat
import subprocess
import sys
from dataclasses import dataclass
from fractions import Fraction
from pathlib import Path

from replay.head_observations import tracking_frames, validate_tracking_window
from replay.recordings import record
from replay.stages import Stage, file_digest, run_stage
from replay.subtitle_clock import VideoFrameClock

CONFIG_BYTES = 1_048_576
RESPONSE_BYTES = 134_217_728
TIMEOUT_SECONDS = 1200
CODE_FILES = (
    "tracking_observations.py",
    "head_observations.py",
    "subtitle_clock.py",
    "vertical_captions.py",
    "caption_io.py",
    "recordings.py",
    "core.py",
    "stages.py",
)


@dataclass(frozen=True)
class TrackingJob:
    config: Path
    config_hash: str
    source: Path
    native: Path
    source_hash: str
    native_hash: str
    clock: VideoFrameClock
    width: int
    height: int
    interval: float

    def request(self) -> dict[str, object]:
        return {
            "path": str(self.source),
            "plan": {
                "startSeconds": float(self.clock.pts[0] * self.clock.time_base),
                "durationSeconds": float(
                    (self.clock.pts[-1] - self.clock.pts[0] + self.clock.last_duration)
                    * self.clock.time_base
                ),
                "sourceTimescale": self.clock.time_base.denominator,
                "maximumFrames": len(self.clock.pts),
                "redetectIntervalSeconds": self.interval,
            },
        }


def _read(path: Path, limit: int) -> bytes:
    descriptor = os.open(path, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK)
    try:
        if not stat.S_ISREG(os.fstat(descriptor).st_mode):
            raise ValueError("Tracking artifact must be a regular file")
        with os.fdopen(descriptor, "rb", closefd=False) as stream:
            data = stream.read(limit + 1)
    finally:
        os.close(descriptor)
    if not data or len(data) > limit:
        raise ValueError("Tracking artifact exceeds its byte budget or is empty")
    return data


def _integer(value: object) -> int:
    if type(value) is not int:
        raise ValueError("Tracking configuration requires integer clocks and dimensions")
    return value


def _path(value: object) -> Path:
    if not isinstance(value, str) or "\0" in value or not Path(value).is_absolute():
        raise ValueError("Tracking inputs must be explicit absolute local paths")
    return Path(value).resolve(strict=True)


def _digest(value: object) -> str:
    if not isinstance(value, str) or re.fullmatch(r"[a-f0-9]{64}", value) is None:
        raise ValueError("Tracking inputs require explicit SHA256 fingerprints")
    return value


def load_job(config: Path) -> TrackingJob:
    data = _read(config, CONFIG_BYTES)
    raw = record(json.loads(data))
    if (
        type(raw.get("version")) is not int
        or raw.get("version") != 1
        or set(raw)
        - {
            "version",
            "source",
            "nativeExecutable",
            "sourceSHA256",
            "nativeSHA256",
            "clock",
            "width",
            "height",
            "redetectIntervalSeconds",
        }
    ):
        raise ValueError("Unsupported tracking configuration version")
    clock = record(raw.get("clock"))
    pts, base = clock.get("pts"), clock.get("timeBase")
    if (
        set(clock) != {"pts", "timeBase", "lastDuration"}
        or not isinstance(pts, list)
        or not isinstance(base, str)
        or re.fullmatch(r"[1-9][0-9]{0,9}/[1-9][0-9]{0,9}", base) is None
    ):
        raise ValueError("Invalid tracking source clock")
    source_clock = VideoFrameClock(
        tuple(_integer(p) for p in pts), Fraction(base), _integer(clock.get("lastDuration"))
    )
    width, height = _integer(raw.get("width")), _integer(raw.get("height"))
    validate_tracking_window(source_clock, width=width, height=height)
    interval = raw.get("redetectIntervalSeconds", 0.5)
    if (
        isinstance(interval, bool)
        or not isinstance(interval, (int, float))
        or not 0.1 <= interval <= 10
    ):
        raise ValueError("Invalid tracking redetection interval")
    return TrackingJob(
        config.absolute(),
        hashlib.sha256(data).hexdigest(),
        _path(raw.get("source")),
        _path(raw.get("nativeExecutable")),
        _digest(raw.get("sourceSHA256")),
        _digest(raw.get("nativeSHA256")),
        source_clock,
        width,
        height,
        float(interval),
    )


def _bindings(job: TrackingJob) -> dict[str, str]:
    paths = (
        job.config,
        job.source,
        job.native,
        *(Path(__file__).with_name(name) for name in CODE_FILES),
    )
    hashes = {str(path): file_digest(path) for path in paths}
    if (
        hashes[str(job.config)] != job.config_hash
        or hashes[str(job.source)] != job.source_hash
        or hashes[str(job.native)] != job.native_hash
    ):
        raise ValueError("Tracking configuration, source or executable changed")
    if not os.access(job.native, os.X_OK):
        raise ValueError("Tracking executable is not executable")
    return hashes


def _write(path: Path, value: object) -> None:
    with path.open("x", encoding="utf-8") as stream:
        os.fchmod(stream.fileno(), 0o600)
        json.dump(value, stream, sort_keys=True)


def _checked_output(output: Path) -> Path:
    output = output.absolute()
    if output.resolve() != output:
        raise ValueError("Linked tracking output storage is not accepted")
    return output


def _check_diagnostics(output: Path) -> None:
    info = (output / "stderr.log").lstat()
    if not stat.S_ISREG(info.st_mode) or info.st_size:
        raise ValueError("Tracking diagnostics are not empty regular evidence")


def observe(config: Path, output: Path) -> None:
    """Produce new raw evidence only; a failed attempt is never overwritten."""
    job = load_job(config)
    inputs = _bindings(job)
    output = _checked_output(output)
    output.mkdir(mode=0o700)
    request = output / "request.json"
    _write(request, job.request())
    with (
        (output / "response.json").open("xb") as stdout,
        (output / "stderr.log").open("xb") as stderr,
    ):
        os.fchmod(stdout.fileno(), 0o600)
        os.fchmod(stderr.fileno(), 0o600)
        subprocess.run(
            (str(job.native), "measure-media", "video_face_track", str(request)),
            stdin=subprocess.DEVNULL,
            stdout=stdout,
            stderr=stderr,
            timeout=TIMEOUT_SECONDS,
            check=True,
        )
    frames = tracking_frames(
        json.loads(_read(output / "response.json", RESPONSE_BYTES)),
        job.clock,
        width=job.width,
        height=job.height,
    )
    _check_diagnostics(output)
    if _bindings(job) != inputs:
        raise ValueError("Tracking diagnostics or changed inputs; attempt retained")
    _write(
        output / "provenance.json",
        {
            "inputs": inputs,
            "requestSHA256": file_digest(request),
            "responseSHA256": file_digest(output / "response.json"),
            "frames": len(frames),
            "nativeExitCode": 0,
            "captionPlacementApproved": False,
        },
    )


def verify(config: Path, output: Path) -> None:
    """Read-only domain verification, including artifacts referenced by the receipt."""
    job = load_job(config)
    output = _checked_output(output)
    inputs = _bindings(job)
    request = output / "request.json"
    if json.loads(_read(request, CONFIG_BYTES)) != job.request():
        raise ValueError("Tracking request differs from the verified plan")
    frames = tracking_frames(
        json.loads(_read(output / "response.json", RESPONSE_BYTES)),
        job.clock,
        width=job.width,
        height=job.height,
    )
    expected = {
        "inputs": inputs,
        "requestSHA256": file_digest(request),
        "responseSHA256": file_digest(output / "response.json"),
        "frames": len(frames),
        "nativeExitCode": 0,
        "captionPlacementApproved": False,
    }
    recorded = json.loads(_read(output / "provenance.json", CONFIG_BYTES))
    # Canonical JSON distinguishes boolean receipt fields from equal Python numbers.
    if json.dumps(recorded, sort_keys=True) != json.dumps(expected, sort_keys=True):
        raise ValueError("Tracking provenance changed")
    _check_diagnostics(output)
    if _bindings(job) != inputs:
        raise ValueError("Tracking inputs changed during verification")


def tracking_stage(config: Path, output: Path, *, name: str) -> Stage:
    """Create a normal stage usable by the existing trusted workflow runner."""
    job = load_job(config)
    inputs = _bindings(job)
    output = _checked_output(output)
    command = (sys.executable, "-m", "replay.tracking_observations")
    return Stage(
        name=name,
        command=(*command, "observe", str(job.config), str(output)),
        verify_command=(*command, "verify", str(job.config), str(output)),
        inputs=tuple(Path(path) for path in inputs),
        outputs=(output / "provenance.json", output / "response.json", output / "request.json"),
        timeout_seconds=TIMEOUT_SECONDS,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("collect", "observe", "verify"))
    parser.add_argument("config", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--state-directory", type=Path)
    parser.add_argument("--name", default="face-tracking")
    args = parser.parse_args()
    if args.action == "collect":
        if args.state_directory is None:
            parser.error("collect requires --state-directory")
        result = run_stage(
            tracking_stage(args.config, args.output, name=args.name),
            state_directory=_checked_output(args.state_directory),
        )
        print(f"Tracking window: {result}; caption placement is not approved")
    elif args.action == "observe":
        observe(args.config, args.output)
    else:
        verify(args.config, args.output)


if __name__ == "__main__":
    main()
