"""Bootstrap measured source-caption profiles into the existing cleanup workflow.

Preparation reads pinned metadata only. OCR and cleanup remain explicit admitted
workflow stages; a sampled profile never proves exhaustive detection or removal.
"""

import argparse
import hashlib
import json
import re
import stat
import sys
from dataclasses import dataclass
from pathlib import Path

from replay import subtitle_cleanup_workflow as cleanup
from replay.code_dependency_inventory import capture
from replay.delivery_media import MAX_CLOCK_BYTES, FrameClock, parse_clock
from replay.foreground_evidence import _bound_bytes
from replay.recordings import record
from replay.sequence_journal import bounded_read, decode, encode, publish, unique_pairs
from replay.subtitle_cleanup_plan import clock_chunks, search_profile

MAX_PROFILE_SAMPLES = 512


@dataclass(frozen=True)
class Reference:
    path: Path
    sha256: str

    def document(self) -> dict[str, str]:
        return {"path": str(self.path), "sha256": self.sha256}


@dataclass(frozen=True)
class Request:
    path: Path
    source: Reference
    clock: FrameClock
    width: int
    height: int
    data: dict[str, object]
    times: tuple[float, ...]
    workers: int
    chunk_frames: int
    native_timescale: int


def _path(value: object) -> Path:
    if not isinstance(value, str) or not value or "\x00" in value:
        raise ValueError("Expected canonical path")
    path = Path(value)
    if not path.is_absolute() or path != path.resolve():
        raise ValueError("Expected canonical path")
    return path


def _local(path: Path) -> Path:
    _path(str(path))
    if not path.is_relative_to(Path.cwd().resolve()):
        raise ValueError("Cleanup metadata must remain inside checkout")
    return path


def reference(value: object) -> Reference:
    row = record(value)
    digest = row.get("sha256")
    if not isinstance(digest, str) or re.fullmatch(r"[0-9a-f]{64}", digest) is None:
        raise ValueError("Expected SHA256 reference")
    return Reference(_path(row.get("path")), digest)


def _existing_role(value: object, role: str) -> Reference:
    """Check namespace/type only; original runtime consumers still verify bytes."""
    ref = reference(value)
    try:
        info = ref.path.lstat()
    except FileNotFoundError as error:
        raise ValueError(f"Missing cleanup {role} file") from error
    if not stat.S_ISREG(info.st_mode) or info.st_size <= 0:
        raise ValueError(f"Cleanup {role} must be a nonempty regular file")
    _path(str(ref.path))
    return ref


def _metadata(value: object, *, clock: bool = False) -> dict[str, object]:
    ref = reference(value)
    _local(ref.path)
    raw = _bound_bytes(ref.path, ref.sha256, MAX_CLOCK_BYTES if clock else 8 * 1024**2)
    return record(json.loads(raw, object_pairs_hook=unique_pairs))


def _number(data: dict[str, object], name: str, maximum: int) -> int:
    value = data.get(name)
    if type(value) is not int or not 1 <= value <= maximum:
        raise ValueError(f"Invalid {name} budget")
    return value


def _physical(clock: FrameClock) -> dict[str, object]:
    return {
        "pts": list(clock.pts),
        "timeBase": str(clock.time_base),
        "lastDuration": clock.last_duration,
    }


def load(path: Path) -> Request:
    """Validate actual metadata/clock bindings without hashing model or media."""
    data = decode(bounded_read(_local(path)))
    if data.get("version") != 1:
        raise ValueError("Unsupported cleanup bootstrap version")
    source = _existing_role(data.get("source"), "source")
    for role in ("native", "decoder", "model"):
        _existing_role(data.get(role), role)
    clock = parse_clock(_metadata(data.get("frameClock"), clock=True))
    if clock.pts[0] != 0:
        raise ValueError("Cleanup assembly requires zero-origin physical PTS; no rebasing")
    chunk_frames = _number(data, "chunkFrames", 10000)
    clock_chunks(_physical(clock), maximum_frames=chunk_frames)
    measured = _metadata(data.get("sourceMetadata"))
    if measured.get("sourceSHA256") != source.sha256 or measured.get("frameClock") != data.get(
        "frameClock"
    ):
        raise ValueError("Measured canvas/source/clock binding differs")
    streams = measured.get("streams")
    if not isinstance(streams, list):
        raise ValueError("Missing measured video streams")
    videos = [record(row) for row in streams if record(row).get("codec_type") == "video"]
    if len(videos) != 1:
        raise ValueError("Expected one measured video canvas")
    width, height = (_number(videos[0], key, 16384) for key in ("width", "height"))
    samples = min(_number(data, "profileSamples", MAX_PROFILE_SAMPLES), len(clock.pts))
    indices = (
        (0,)
        if samples == 1
        else tuple(index * (len(clock.pts) - 1) // (samples - 1) for index in range(samples))
    )
    times = tuple(float(clock.pts[index] * clock.time_base) for index in indices)
    native_timescale = _number(data, "nativeTimescale", 1_000_000_000)
    if any(
        (clock.pts[index] * clock.time_base * native_timescale).denominator != 1
        for index in indices
    ):
        raise ValueError("Profile PTS are not exactly representable by the pinned native timescale")
    return Request(
        path,
        source,
        clock,
        width,
        height,
        data,
        times,
        _number(data, "workers", 8),
        chunk_frames,
        native_timescale,
    )


def _pin(path: Path) -> dict[str, str]:
    return Reference(path, hashlib.sha256(bounded_read(path)).hexdigest()).document()


def _write(path: Path, value: object) -> None:
    _local(path)
    raw = encode(value)
    if path.exists() or path.is_symlink():
        if bounded_read(path) != raw:
            raise ValueError("Prepared checkpoint changed; preserve the previous generation")
        return
    publish(path, raw)


def _command(operation: str, request: Path, output: Path) -> list[str]:
    return [
        sys.executable,
        "-B",
        "-m",
        "replay.replay_cleanup_plan",
        operation,
        str(request),
        str(output),
    ]


def _code_paths() -> tuple[Path, ...]:
    return tuple(
        item.path
        for item in capture(
            ("replay_cleanup_plan", "subtitle_observations", "subtitle_candidate", "workflow"),
            package=Path(__file__).parent,
        )
    )


def prepare(path: Path, output: Path) -> Path:
    """Create an immutable resumable profile/cleanup-planning dependency graph."""
    job = load(path)
    _local(output).mkdir(mode=0o700, parents=True, exist_ok=True)
    _write(output / "request-pin.json", _pin(path))
    codes = _code_paths()
    _write(output / "code-pins.json", [_pin(item) for item in codes])
    _write(output / "physical-clock.json", _physical(job.clock))
    _write(output / "profile-times.json", list(job.times))
    native = reference(job.data["native"])
    profile = output / "profile"
    common = [
        str(path),
        str(output / "request-pin.json"),
        str(reference(job.data["frameClock"]).path),
        str(reference(job.data["sourceMetadata"]).path),
        str(output / "code-pins.json"),
        *[str(item) for item in codes],
    ]
    stages = [
        {
            "name": "profile-ocr",
            "command": [
                sys.executable,
                "-B",
                "-m",
                "replay.subtitle_observations",
                str(job.source.path),
                str(output / "profile-times.json"),
                str(profile),
                "--native-executable",
                str(native.path),
            ],
            "inputs": [
                *common,
                str(job.source.path),
                str(native.path),
                str(output / "profile-times.json"),
            ],
            "outputs": [str(profile / "inputs.json"), str(profile / "complete.json")],
            "timeout_seconds": 7200,
            "verify_command": _command("verify-profile", path, output),
        },
        {
            "name": "profile-settings",
            "command": _command("profile", path, output),
            "inputs": [
                *common,
                str(profile / "inputs.json"),
                str(profile / "complete.json"),
                str(output / "physical-clock.json"),
            ],
            "outputs": [str(output / "classification.json"), str(output / "settings.json")],
            "timeout_seconds": 120,
            "verify_command": _command("profile", path, output),
        },
        {
            "name": "cleanup-plan",
            "command": _command("handoff", path, output),
            "inputs": [
                *common,
                str(output / "settings.json"),
                str(output / "classification.json"),
                str(reference(job.data["decoder"]).path),
                str(reference(job.data["model"]).path),
            ],
            "outputs": [
                str(output / "cleanup" / "plan.json"),
                *[
                    str(output / "cleanup" / f"worker-{index:02d}.json")
                    for index in range(
                        min(
                            job.workers,
                            (len(job.clock.pts) + job.chunk_frames - 1) // job.chunk_frames,
                        )
                    )
                ],
            ],
            "timeout_seconds": 3600,
            "verify_command": _command("handoff", path, output),
        },
    ]
    _write(output / "workflow.json", {"version": 1, "stages": stages})
    return output / "workflow.json"


def verify_profile(path: Path, output: Path) -> dict[str, object]:
    """Bind every original sampled result to actual PTS, canvas and producer."""
    job = load(path)
    pins = json.loads(bounded_read(output / "code-pins.json"), object_pairs_hook=unique_pairs)
    if not isinstance(pins, list) or not 1 <= len(pins) <= 512:
        raise ValueError("Invalid code checkpoint")
    if tuple(reference(raw).path for raw in pins) != _code_paths():
        raise ValueError("Cleanup producer code closure changed")
    for raw in pins:
        pin = reference(raw)
        if _pin(pin.path) != pin.document():
            raise ValueError("Cleanup producer code changed")
    if decode(bounded_read(output / "request-pin.json")) != _pin(path):
        raise ValueError("Bootstrap request changed")
    if decode(bounded_read(output / "physical-clock.json")) != _physical(job.clock):
        raise ValueError("Physical clock checkpoint changed")
    profile = output / "profile"
    producer = decode(bounded_read(profile / "inputs.json"))
    expected = {
        "source": str(job.source.path),
        "backend": "native-cli",
        "collectorSHA256": _pin(Path(cleanup.__file__).with_name("subtitle_observations.py"))[
            "sha256"
        ],
        "parserSHA256": _pin(Path(cleanup.__file__).with_name("caption_io.py"))["sha256"],
        "planSHA256": _pin(output / "profile-times.json")["sha256"],
    }
    if any(producer.get(key) != value for key, value in expected.items()):
        raise ValueError("Sampled collector provenance differs")
    frames, _ = cleanup.collector_frames(
        profile / "inputs.json", profile / "complete.json", job.data
    )
    if (
        tuple(frame.requested_time for frame in frames) != job.times
        or tuple(frame.actual_time for frame in frames) != job.times
        or any((frame.width, frame.height) != (job.width, job.height) for frame in frames)
    ):
        raise ValueError("Sampled OCR clock/canvas differs from actual source")
    learned = search_profile(frames)
    return {
        "version": 1,
        "kind": "measured-source-caption-profile",
        "sourceSHA256": job.source.sha256,
        "sourceMetadata": job.data["sourceMetadata"],
        "frameClock": job.data["frameClock"],
        "profileInputs": _pin(profile / "inputs.json"),
        "profileComplete": _pin(profile / "complete.json"),
        "canvas": [job.width, job.height],
        "measuredEnvelope": list(learned.measured_envelope),
        "observations": len(frames),
        "classification": "heuristic-candidate",
        "wholeDetectionVerified": False,
        "removalQualityApproved": False,
    }


def profile(path: Path, output: Path) -> None:
    classification = verify_profile(path, output)
    job = load(path)
    _write(output / "classification.json", classification)
    settings = {
        key: job.data[key] for key in ("source", "native", "decoder", "model", "chunkFrames")
    }
    settings.update(
        version=1,
        sourceClock=_pin(output / "physical-clock.json"),
        classificationEvidence=_pin(output / "classification.json"),
        profileInputs=classification["profileInputs"],
        profileComplete=classification["profileComplete"],
    )
    _write(output / "settings.json", settings)


def handoff(path: Path, output: Path) -> None:
    """Execute original pinned-source/profile validation and resumable plan factory."""
    profile(path, output)
    job = load(path)
    if any(
        (pts * job.clock.time_base * job.native_timescale).denominator != 1 for pts in job.clock.pts
    ):
        raise ValueError(
            "Full-source OCR PTS are not exactly representable; sampled profile retained"
        )
    cleanup.prepare(output / "settings.json", output / "cleanup", workers=job.workers)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("operation", choices=("prepare", "verify-profile", "profile", "handoff"))
    parser.add_argument("request", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args(argv)
    operations = {
        "prepare": prepare,
        "verify-profile": verify_profile,
        "profile": profile,
        "handoff": handoff,
    }
    operations[args.operation](args.request, args.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
