"""Finalize original face observations without deriving missing head geometry."""

import argparse
import json
import os
from collections import Counter
from contextlib import ExitStack
from fractions import Fraction
from pathlib import Path
from typing import Literal

from replay.code_dependency_inventory import capture
from replay.delivery_media import FrameClock
from replay.fcp_tracking_caption import merge_face_observations, read_verified_face_observations
from replay.head_observations import TrackingFrame, tracking_frames
from replay.input_hash_cache import InputHashCache, input_hash_context
from replay.recordings import record
from replay.stages import Stage, _fingerprint, execution_lock, file_digest
from replay.subtitle_clock import MAX_CLOCK_SECONDS, MAX_CLOCK_TIMESCALE
from replay.tracking_observations import RESPONSE_BYTES, _read, load_job
from replay.workflow import parse_stages

MAX_METADATA = 16 * 1024 * 1024
MAX_WINDOWS = 512


def _local(value: object) -> Path:
    if not isinstance(value, str):
        raise ValueError("Expected a local canonical metadata path")
    path = Path(value)
    if (
        not path.is_absolute()
        or str(path) != value
        or path.resolve() != path
        or not path.is_relative_to(Path.cwd() / "edits/work")
    ):
        raise ValueError("Finalize metadata escapes repository scratch")
    return path


def _json(path: Path) -> object:
    _local(str(path))
    return json.loads(_read(path, MAX_METADATA))


def _ref(value: object, pins: dict[str, str]) -> Path:
    row = record(value)
    if set(row) != {"path", "sha256"}:
        raise ValueError("Expected an exact pinned metadata reference")
    path = _local(row["path"])
    digest = file_digest(path)
    if row["sha256"] != digest or (str(path) in pins and pins[str(path)] != digest):
        raise ValueError("Finalize metadata pin differs")
    pins[str(path)] = digest
    return path


def _clock(value: object) -> FrameClock:
    row = record(value)
    pts = row.get("pts")
    duration, base = row.get("lastDuration"), row.get("timeBase")
    if (
        set(row) != {"pts", "lastDuration", "timeBase"}
        or not isinstance(pts, list)
        or not isinstance(base, str)
        or type(duration) is not int
    ):
        raise ValueError("Invalid full original source clock")
    ticks: list[int] = []
    for point in pts:
        if type(point) is not int:
            raise ValueError("Source PTS must be integers")
        ticks.append(point)
    if not 1 <= len(ticks) <= 1000000:
        raise ValueError("Full source clock exceeds budget")
    time_base = Fraction(base)
    if (
        any(point < 0 for point in ticks)
        or any(right <= left for left, right in zip(ticks, ticks[1:], strict=False))
        or duration <= 0
        or time_base <= 0
        or time_base.denominator > MAX_CLOCK_TIMESCALE
        or (ticks[-1] + duration) * time_base > MAX_CLOCK_SECONDS
    ):
        raise ValueError("Invalid full source clock order, time base or terminal extent")
    return FrameClock(tuple(ticks), time_base, duration)


def _windows(index: object, stages: tuple[Stage, ...]) -> list[tuple[Stage, Path, Path, Path]]:
    rows = record(index).get("windows")
    if not isinstance(rows, list) or not 1 <= len(rows) <= MAX_WINDOWS:
        raise ValueError("Expected bounded ordered original windows")
    by_name = {stage.name: stage for stage in stages}
    result: list[tuple[Stage, Path, Path, Path]] = []
    for position, item in enumerate(rows):
        row = record(item)
        name = row.get("name")
        if (
            type(row.get("index")) is not int
            or row["index"] != position
            or not isinstance(name, str)
            or name not in by_name
        ):
            raise ValueError("Face index order or stage name differs")
        stage = by_name[name]
        config, output, state = (
            _local(row.get(key)) for key in ("config", "evidence", "originalState")
        )
        checkpoint = _local(row.get("originalCheckpoint"))
        expected = {output / name for name in ("request.json", "response.json", "provenance.json")}
        if (
            config not in stage.inputs
            or set(stage.outputs) != expected
            or checkpoint != state / f"{stage.name}.json"
            or not state.is_dir()
        ):
            raise ValueError("Index does not bind the actual original Stage/namespace")
        result.append((stage, config, output, state))
    if len({stage.name for stage, _, _, _ in result}) != len(result) or len(result) != len(stages):
        raise ValueError("Duplicate or omitted original windows")
    return result


def _checkpoint(stage: Stage, state: Path) -> None:
    raw = record(_json(state / f"{stage.name}.json"))
    if raw != {
        "status": "complete",
        "fingerprint": _fingerprint(stage),
        "outputs": {str(path): file_digest(path) for path in stage.outputs},
    }:
        raise ValueError("Original complete Stage is not current")


def finalize(config: Path, output: Path, *, verify: bool = False) -> dict[str, object]:
    """Verify every original lane and full PTS with one invocation-local hash cache."""
    config, output = _local(str(config)), _local(str(output))
    raw = record(_json(config))
    if (
        set(raw) != {"version", "workflow", "clock", "index", "seedOverlap"}
        or type(raw["version"]) is not int
        or raw["version"] != 1
    ):
        raise ValueError("Invalid face finalization config")
    overlap: Literal["reject", "next"]
    if raw["seedOverlap"] == "reject":
        overlap = "reject"
    elif raw["seedOverlap"] == "next":
        overlap = "next"
    else:
        raise ValueError("Explicit seed overlap policy is required")
    pins = {str(config): file_digest(config)}
    workflow, clock_path, index = (_ref(raw[key], pins) for key in ("workflow", "clock", "index"))
    stages = parse_stages(_json(workflow))
    windows = _windows(_json(index), stages)
    clock = _clock(_json(clock_path))
    sources = capture(("tracking_face_finalize",))
    pins.update({str(item.path): item.sha256 for item in sources})
    paths = {Path(name) for name in pins}
    for stage, _, _, state in windows:
        paths.update((*stage.inputs, *stage.outputs, state / f"{stage.name}.json"))
    if output in paths or not output.parent.is_dir() or (not verify and output.exists()):
        raise ValueError("Finalize output is occupied or aliases protected inputs")
    with ExitStack() as stack:
        for state in sorted({state for _, _, _, state in windows}):
            if not stack.enter_context(execution_lock(state, ".workflow")):
                raise ValueError("An original workflow is still owned")
        for stage, _, _, state in windows:
            if not stack.enter_context(execution_lock(state, stage.name)):
                raise ValueError("An original Stage is still owned")
        cache = InputHashCache(sorted(paths))
        cache.prime()
        with input_hash_context(cache):
            if any(file_digest(Path(name)) != sha for name, sha in pins.items()):
                raise ValueError("Finalizer metadata or reader changed")
            series = []
            all_frames: list[TrackingFrame] = []
            identity: tuple[Path, str, Path, str] | None = None
            for stage, path, evidence, state in windows:
                _checkpoint(stage, state)
                job = load_job(path)
                current = (job.source, job.source_hash, job.native, job.native_hash)
                if identity is not None and current != identity:
                    raise ValueError("Windows mix source or native providers")
                identity = current
                observed = read_verified_face_observations(path, evidence)
                frames = tracking_frames(
                    json.loads(_read(evidence / "response.json", RESPONSE_BYTES)),
                    job.clock,
                    width=job.width,
                    height=job.height,
                )
                if all_frames and frames[0].time == all_frames[-1].time and overlap == "next":
                    all_frames.pop()
                all_frames.extend(frames)
                series.append(observed)
            merged = merge_face_observations(tuple(series), seed_overlap=overlap)
            expected = tuple(point * clock.time_base for point in clock.pts)
            if (
                tuple(sample.time for sample in merged.samples) != expected
                or tuple(frame.time for frame in all_frames) != expected
                or merged.coverage_end != (clock.pts[-1] + clock.last_duration) * clock.time_base
            ):
                raise ValueError("Actual observation union differs from full source PTS")
            confidences = [face.confidence for frame in all_frames for face in frame.faces]
            result: dict[str, object] = {
                "version": 1,
                "kind": "verified-raw-face-union",
                "config": str(config),
                "inputs": {str(path): file_digest(path) for path in sorted(paths)},
                "windows": record(_json(index))["windows"],
                "frames": len(all_frames),
                "clock": record(_json(clock_path)),
                "seedOverlap": overlap,
                "missing": sum(sample.reason == "missing" for sample in merged.samples),
                "ambiguous": sum(sample.reason == "ambiguous" for sample in merged.samples),
                "methods": dict(sorted(Counter(str(frame.method) for frame in all_frames).items())),
                "confidence": {
                    "count": len(confidences),
                    "minimum": min(confidences, default=None),
                    "maximum": max(confidences, default=None),
                },
                "canvas": [merged.width, merged.height],
                "geometryDerived": False,
                "headNearVerified": False,
                "deliveryApproved": False,
            }
            if verify and _json(output) != result:
                raise ValueError("Face union receipt differs from current original evidence")
            cache.verify()
        if not verify:
            with output.open("x", encoding="utf-8") as stream:
                os.fchmod(stream.fileno(), 0o600)
                json.dump(result, stream, sort_keys=True, allow_nan=False)
                stream.flush()
                os.fsync(stream.fileno())
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("operation", choices=("finalize", "verify"))
    parser.add_argument("config", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    finalize(args.config, args.output, verify=args.operation == "verify")


if __name__ == "__main__":
    main()
