"""Prepare hash-bound seeded or explicitly unseeded tracking workflows."""

import argparse
import hashlib
import json
import re
import sys
from dataclasses import asdict, replace
from fractions import Fraction
from pathlib import Path

from replay import tracking_observations as observations
from replay.head_observations import MAX_TRACKING_FRAMES, TrackingMethod, tracking_frames
from replay.recordings import record
from replay.stages import Stage, file_digest
from replay.tracking_plan import tracking_windows
from replay.workflow import MAX_PLAN_BYTES, MAX_STAGES, parse_stages

MAX_INPUT_BYTES = 64 * 1024 * 1024
MAX_SOURCE_FRAMES = MAX_STAGES * MAX_TRACKING_FRAMES


def _encoded(value: object) -> bytes:
    return json.dumps(value, sort_keys=True, allow_nan=False).encode("utf-8")


def _check(hashes: dict[str, str]) -> None:
    for path, expected in hashes.items():
        if file_digest(Path(path)) != expected:
            raise ValueError("Prepared tracking input changed")


def _reference(value: object, hashes: dict[str, str]) -> dict[str, object]:
    ref = record(value)
    if set(ref) != {"path", "sha256"}:
        raise ValueError("Evidence requires path and SHA256")
    path = observations._path(ref["path"])
    digest = observations._digest(ref["sha256"])
    if str(path) in hashes:
        raise ValueError("Evidence paths must be distinct")
    data = observations._read(path, MAX_INPUT_BYTES)
    if hashlib.sha256(data).hexdigest() != digest:
        raise ValueError("Evidence hash mismatch")
    hashes[str(path)] = digest
    return record(json.loads(data))


def _configs(
    manifest: Path, *, unseeded: bool = False
) -> tuple[list[dict[str, object]], dict[str, str]]:
    data = observations._read(manifest, MAX_INPUT_BYTES)
    raw = record(json.loads(data))
    if (
        set(raw)
        != {
            "version",
            "source",
            "sourceSHA256",
            "nativeExecutable",
            "nativeSHA256",
            "width",
            "height",
            "redetectIntervalSeconds",
            "clock",
            "seeds",
            "eof",
            "maximumSeconds",
        }
        or type(raw["version"]) is not int
        or raw["version"] != 1
    ):
        raise ValueError("Invalid tracking batch manifest")
    hashes = {str(manifest): hashlib.sha256(data).hexdigest()}
    for field in ("source", "nativeExecutable"):
        path = observations._path(raw[field])
        key = "sourceSHA256" if field == "source" else "nativeSHA256"
        if str(path) in hashes:
            raise ValueError("Tracking inputs must be distinct")
        hashes[str(path)] = observations._digest(raw[key])
    _check(hashes)
    clock = _reference(raw["clock"], hashes)
    candidates = _reference(raw["seeds"], hashes)
    eof = _reference(raw["eof"], hashes)
    pts, seeds = clock.get("pts"), candidates.get("candidateSeedPTS")
    if (
        set(clock) != {"pts", "timeBase", "lastDuration"}
        or not isinstance(pts, list)
        or not 1 <= len(pts) <= MAX_SOURCE_FRAMES
        or not isinstance(seeds, list)
        or not (0 if unseeded else 1) <= len(seeds) <= len(pts)
        or not isinstance(clock["timeBase"], str)
        or re.fullmatch(r"[1-9][0-9]{0,9}/[1-9][0-9]{0,9}", clock["timeBase"]) is None
        or not isinstance(raw["maximumSeconds"], str)
        or re.fullmatch(r"[1-9][0-9]{0,9}/[1-9][0-9]{0,9}", raw["maximumSeconds"]) is None
        or candidates.get("timeBase") != clock["timeBase"]
    ):
        raise ValueError("Invalid bounded source clock or candidate seeds")
    ticks = tuple(observations._integer(point) for point in pts)
    last_duration = observations._integer(clock["lastDuration"])
    packets = eof.get("packets")
    if not isinstance(packets, list) or not packets:
        raise ValueError("Terminal packet evidence is required")
    terminal = max((record(packet) for packet in packets), key=_packet_pts)
    if (
        _packet_pts(terminal) != ticks[-1]
        or observations._integer(terminal.get("duration")) != last_duration
    ):
        raise ValueError("Terminal decoded clock differs from packet support")
    windows = tracking_windows(
        ticks,
        time_base=Fraction(clock["timeBase"]),
        last_duration=last_duration,
        seed_pts=None if unseeded else tuple(observations._integer(seed) for seed in seeds),
        maximum_seconds=Fraction(raw["maximumSeconds"]),
    )
    if len(windows) > MAX_STAGES:
        raise ValueError("Too many tracking windows")
    job = {
        key: value
        for key, value in raw.items()
        if key not in {"seeds", "eof", "maximumSeconds", "clock"}
    }
    configs = [
        {
            **job,
            "clock": {
                "pts": list(window.pts),
                "timeBase": f"{window.time_base.numerator}/{window.time_base.denominator}",
                "lastDuration": window.last_duration,
            },
        }
        for window in windows
    ]
    return configs, hashes


def _packet_pts(packet: dict[str, object]) -> int:
    return observations._integer(packet.get("pts"))


def _stage_json(stage: Stage) -> dict[str, object]:
    return {
        **asdict(stage),
        "inputs": [str(path) for path in stage.inputs],
        "outputs": [str(path) for path in stage.outputs],
    }


def _workflow(binding: Path, seal: str, definition: dict[str, object]) -> dict[str, object]:
    stages = parse_stages({"version": 1, "stages": definition["stages"]})
    configs = {binding.parent / f"window-{index:04d}.json" for index in range(len(stages))}
    # All configs remain sealed and checked by each wrapper. Avoid repeating
    # other windows' paths in every stage of the bounded workflow manifest.
    common = tuple(
        Path(path) for path in sorted(record(definition["hashes"])) if Path(path) not in configs
    )
    command = (sys.executable, "-m", "replay.tracking_batch")
    entries = []
    for index, stage in enumerate(stages):
        arguments = (str(binding), seal, str(index))
        inputs = tuple(
            dict.fromkeys((*stage.inputs, *common, binding, binding.with_name("workflow.json")))
        )
        entries.append(
            _stage_json(
                replace(
                    stage,
                    command=(*command, "observe", *arguments),
                    verify_command=(*command, "verify", *arguments),
                    inputs=inputs,
                )
            )
        )
    result: dict[str, object] = {"version": 1, "stages": entries}
    # Round-trip tuples to the existing JSON boundary; no alternate scheduler.
    parse_stages(json.loads(_encoded(result)))
    if len(_encoded(result)) > MAX_PLAN_BYTES:
        raise ValueError("Prepared workflow exceeds its byte budget")
    return result


def prepare(manifest: Path, directory: Path, *, unseeded: bool = False) -> Path:
    """Prepare new configs only; evidence provenance must be independently verified."""
    manifest = manifest.absolute()
    directory = observations._checked_output(directory)
    configs, hashes = _configs(manifest, unseeded=unseeded)
    for name in (*observations.CODE_FILES, "tracking_plan.py", "tracking_batch.py", "workflow.py"):
        path = Path(__file__).with_name(name)
        if str(path) in hashes:
            raise ValueError("Data must not alias code")
        hashes[str(path)] = file_digest(path)
    directory.mkdir(mode=0o700)
    stages = []
    for index, config in enumerate(configs):
        path = directory / f"window-{index:04d}.json"
        observations._write(path, config)
        hashes[str(path)] = file_digest(path)
        stage = observations.tracking_stage(
            path, directory / f"window-{index:04d}", name=f"tracking-{index:04d}"
        )
        stages.append(_stage_json(stage))
    definition: dict[str, object] = {"version": 1, "hashes": hashes, "stages": stages}
    if unseeded:
        definition.update(version=2, seedPolicy="unseeded-observe")
    # Match the existing private writer's canonical bytes exactly.
    seal = hashlib.sha256(_encoded(definition)).hexdigest()
    binding = directory / "binding.json"
    workflow = _workflow(binding, seal, json.loads(_encoded(definition)))
    _check(hashes)
    observations._write(binding, definition)
    output = directory / "workflow.json"
    observations._write(output, workflow)
    return output


def _checked(binding: Path, seal: str) -> tuple[Stage, ...]:
    binding = observations._checked_output(binding)
    data = observations._read(binding, MAX_INPUT_BYTES)
    if hashlib.sha256(data).hexdigest() != observations._digest(seal):
        raise ValueError("Tracking binding changed")
    definition = record(json.loads(data))
    if (
        set(definition)
        != (
            {"version", "hashes", "stages", "seedPolicy"}
            if definition.get("version") == 2
            else {"version", "hashes", "stages"}
        )
        or type(definition["version"]) is not int
        or definition["version"] not in {1, 2}
        or (definition["version"] == 2 and definition.get("seedPolicy") != "unseeded-observe")
    ):
        raise ValueError("Invalid tracking binding")
    hashes = {
        path: observations._digest(digest) for path, digest in record(definition["hashes"]).items()
    }
    _check(hashes)
    expected = _workflow(binding, seal, definition)
    if (
        file_digest(binding.with_name("workflow.json"))
        != hashlib.sha256(_encoded(expected)).hexdigest()
    ):
        raise ValueError("Tracking workflow changed")
    return parse_stages({"version": 1, "stages": definition["stages"]})


def run_window(action: str, binding: Path, seal: str, index: int) -> None:
    """Bind first execution and reuse; preserve missing geometry without approval."""
    if action not in {"observe", "verify"}:
        raise ValueError("Invalid tracking action")
    stages = _checked(binding, seal)
    if type(index) is not int or not 0 <= index < len(stages):
        raise ValueError("Invalid window index")
    config = binding.parent / f"window-{index:04d}.json"
    output = binding.parent / f"window-{index:04d}"
    if action == "observe":
        observations.observe(config, output)
    else:
        observations.verify(config, output)
    job = observations.load_job(config)
    frames = tracking_frames(
        json.loads(observations._read(output / "response.json", observations.RESPONSE_BYTES)),
        job.clock,
        width=job.width,
        height=job.height,
    )
    definition = record(json.loads(observations._read(binding, MAX_INPUT_BYTES)))
    if definition.get("seedPolicy") != "unseeded-observe" and (
        frames[0].method != TrackingMethod.DETECTION or not frames[0].faces
    ):
        raise ValueError("Actual native detection did not establish the seed frame")
    _checked(binding, seal)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="action", required=True)
    prep = commands.add_parser("prepare")
    prep.add_argument("manifest", type=Path)
    prep.add_argument("output", type=Path)
    prep.add_argument(
        "--unseeded",
        action="store_true",
        help="Observe contiguous windows without requiring initial detection; retain missing data",
    )
    for action in ("observe", "verify"):
        command = commands.add_parser(action)
        command.add_argument("binding", type=Path)
        command.add_argument("seal")
        command.add_argument("index", type=int)
    args = parser.parse_args()
    if args.action == "prepare":
        print(prepare(args.manifest, args.output, unseeded=args.unseeded))
    else:
        run_window(args.action, args.binding, args.seal, args.index)


if __name__ == "__main__":
    main()
