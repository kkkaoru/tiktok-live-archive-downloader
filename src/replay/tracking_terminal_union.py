"""Explicit union of original tracking stages and one terminal-selection stage."""

import argparse
import json
from collections import Counter
from contextlib import ExitStack
from pathlib import Path

from replay import tracking_face_finalize as original
from replay import tracking_observations as storage
from replay import tracking_terminal_selection as terminal
from replay.code_dependency_inventory import capture
from replay.head_observations import TrackingFrame, tracking_frames
from replay.input_hash_cache import InputHashCache, input_hash_context
from replay.recordings import record
from replay.stages import execution_lock, file_digest
from replay.tracking_retained import RetainedTracking, RetainedWindow
from replay.workflow import parse_stages

KIND = "verified-terminal-selection-face-union-v1"


def _collect(config: Path) -> tuple[dict[str, object], RetainedTracking]:
    config = original._local(str(config))
    raw = record(original._json(config))
    if (
        set(raw) != {"version", "original", "terminalWorkflow", "terminalState"}
        or type(raw.get("version")) is not int
        or raw["version"] != 1
    ):
        raise ValueError("Invalid terminal face-union configuration")
    pins = {str(config): file_digest(config)}
    parent_path = original._ref(raw["original"], pins)
    parent = record(original._json(parent_path))
    if parent.get("version") != 1 or parent.get("seedOverlap") != "reject":
        raise ValueError("Terminal union requires original nonoverlapping windows")
    workflow, clock_path, index = (
        original._ref(parent[key], pins) for key in ("workflow", "clock", "index")
    )
    windows = original._windows(original._json(index), parse_stages(original._json(workflow)))
    clock = original._clock(original._json(clock_path))
    terminal_workflow = original._ref(raw["terminalWorkflow"], pins)
    stages = parse_stages(original._json(terminal_workflow))
    state = original._local(raw["terminalState"])
    if len(stages) != 1:
        raise ValueError("Expected exactly one terminal-selection stage")
    stage = stages[0]
    last_config = windows[-1][1]
    receipts = tuple(path for path in stage.outputs if path.name == "receipt.json")
    if len(receipts) != 1 or last_config not in stage.inputs:
        raise ValueError("Terminal stage does not bind the original final window")
    evidence = receipts[0].parent
    if set(stage.outputs) != {
        evidence / name
        for name in ("inspection.json", "request.json", "response.json", "receipt.json")
    }:
        raise ValueError("Terminal stage output roles differ")
    selected = [*windows[:-1], (stage, last_config, evidence, state)]
    paths = {Path(name) for name in pins}
    for current, _, _, current_state in selected:
        paths.update((*current.inputs, *current.outputs, current_state / f"{current.name}.json"))
    paths.update(item.path for item in capture(("tracking_terminal_union",)))
    with ExitStack() as locks:
        for directory in sorted({row[3] for row in windows} | {state}):
            if not locks.enter_context(execution_lock(directory, ".workflow")):
                raise ValueError("Tracking workflow is still owned")
        for current, _, _, directory in selected:
            if not locks.enter_context(execution_lock(directory, current.name)):
                raise ValueError("Tracking stage is still owned")
        cache = InputHashCache(sorted(paths))
        cache.prime()
        with input_hash_context(cache):
            if any(file_digest(Path(path)) != sha for path, sha in pins.items()):
                raise ValueError("Terminal union configuration changed")
            frames: list[TrackingFrame] = []
            retained: list[RetainedWindow] = []
            identity: tuple[Path, str, Path, str] | None = None
            rows: list[dict[str, object]] = []
            for number, (current, path, directory, current_state) in enumerate(selected):
                original._checkpoint(current, current_state)
                job = storage.load_job(path)
                provider = (job.source, job.source_hash, job.native, job.native_hash)
                if identity is not None and provider != identity:
                    raise ValueError("Face windows have different source/native identities")
                identity = provider
                final = number == len(selected) - 1
                (terminal.verify if final else storage.verify)(path, directory)
                receipt = directory / ("receipt.json" if final else "provenance.json")
                response = directory / "response.json"
                current_frames = tracking_frames(
                    json.loads(storage._read(response, storage.RESPONSE_BYTES)),
                    job.clock,
                    width=job.width,
                    height=job.height,
                )
                frames.extend(current_frames)
                retained.append(
                    RetainedWindow(
                        job,
                        directory,
                        file_digest(receipt),
                        file_digest(response),
                        len(current_frames),
                        {},
                    )
                )
                rows.append(
                    {
                        "index": number,
                        "config": str(path),
                        "evidence": str(directory),
                        "protocol": terminal.KIND if final else "tracking-observations-v1",
                        "receipt": str(receipt),
                    }
                )
            last = retained[-1].job.clock
            if (
                tuple(frame.time for frame in frames)
                != tuple(p * clock.time_base for p in clock.pts)
                or (last.pts[-1] + last.last_duration) * last.time_base
                != (clock.pts[-1] + clock.last_duration) * clock.time_base
            ):
                raise ValueError("Face union does not cover the unchanged full original clock")
            inputs = {str(path): file_digest(path) for path in sorted(paths)}
            result: dict[str, object] = {
                "version": 1,
                "kind": KIND,
                "config": str(config),
                "inputs": inputs,
                "windows": rows,
                "clock": original._json(clock_path),
                "frames": len(frames),
                "methods": dict(sorted(Counter(frame.method for frame in frames).items())),
                "allOriginalPTSVerified": True,
                "originalTerminalDurationPreserved": True,
                "legacyFailedWindowAdopted": False,
                "geometryDerived": False,
                "headNearVerified": False,
                "deliveryApproved": False,
            }
            cache.verify()
            return result, RetainedTracking(tuple(retained), inputs)


def finalize(config: Path, output: Path, *, verify: bool = False) -> RetainedTracking:
    """Return verified raw transport; never write a fabricated legacy provenance."""
    output = original._local(str(output))
    result, catalog = _collect(config)
    if verify:
        if original._json(output) != result:
            raise ValueError("Terminal union receipt changed")
    else:
        storage._write(output, result)
    return catalog


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("finalize", "verify"))
    parser.add_argument("config", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    finalize(args.config, args.output, verify=args.action == "verify")


if __name__ == "__main__":
    main()
