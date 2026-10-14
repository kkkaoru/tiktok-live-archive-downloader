"""Prepare ordinary OCR workflows and compare genuine ordered native evidence."""

import argparse
import hashlib
import json
import math
import sys
import time
from dataclasses import dataclass
from fractions import Fraction
from pathlib import Path

from replay import caption_io, code_dependency_inventory
from replay import subtitle_cleanup_workflow as cleanup
from replay import subtitle_observations as legacy
from replay import subtitle_session_observations as session
from replay.recordings import record
from replay.sequence_journal import (
    bounded_read,
    decode,
    encode,
    invalid_constant,
    publish,
    unique_pairs,
)
from replay.stages import _fingerprint, execution_lock, file_digest
from replay.subtitle_cleanup_plan import clock_chunks
from replay.workflow import load_stages, parse_stages, stage_graph


@dataclass(frozen=True, slots=True)
class Prepared:
    plan: Path
    workflow: Path
    state: Path


def _path(value: object, *, work: bool = False) -> Path:
    path = session._path(value)
    if work and not path.is_relative_to(Path.cwd() / "edits/work"):
        raise ValueError("Benchmark operations require a repository work path")
    return path


def _read(path: Path) -> object:
    _path(str(path))
    return json.loads(
        bounded_read(path), object_pairs_hook=unique_pairs, parse_constant=invalid_constant
    )


def _write(path: Path, value: object) -> None:
    _path(str(path), work=True)
    publish(path, encode(record(value)))


def _pin(path: Path, pins: dict[str, str]) -> dict[str, str]:
    _path(str(path))
    data = bounded_read(path)
    digest = hashlib.sha256(data).hexdigest()
    if not data or (str(path) in pins and pins[str(path)] != digest):
        raise ValueError("Empty or changed benchmark metadata")
    pins[str(path)] = digest
    return {"path": str(path), "sha256": digest}


def _ref(value: object, pins: dict[str, str]) -> Path:
    row = record(value)
    if set(row) != {"path", "sha256"}:
        raise ValueError("Expected an exact metadata reference")
    path = _path(row["path"])
    if _pin(path, pins)["sha256"] != session._sha(row["sha256"]):
        raise ValueError("Benchmark reference digest differs")
    return path


def _samples(clock: Path) -> tuple[Fraction, ...]:
    raw = record(_read(clock))
    clocks = clock_chunks(raw.get("clock", raw), maximum_frames=10000)
    return tuple(pts * chunk.time_base for chunk in clocks for pts in chunk.pts)


def _baseline(value: object, pins: dict[str, str], plan: session.SessionPlan) -> dict[str, object]:
    row = record(value)
    if set(row) != {"chunk", "times", "workflow", "receipt", "stageName"}:
        raise ValueError("Invalid retained baseline contract")
    chunk = _ref(row["chunk"], pins)
    times = _ref(row["times"], pins)
    workflow = _ref(row["workflow"], pins)
    receipt = _ref(row["receipt"], pins)
    if not isinstance(row["stageName"], str):
        raise ValueError("Expected original OCR Stage name")
    data = record(_read(chunk))
    settings = _ref(data.get("settings"), pins)
    settings_data = record(_read(settings))
    if record(settings_data.get("source")) != {"path": str(plan.source), "sha256": plan.source_sha}:
        raise ValueError("Retained baseline source differs")
    if times != chunk.parent / "times.json":
        raise ValueError("Retained baseline times path differs")
    actual = legacy.observation_times(_read(times))
    if actual != tuple(float(point) for point in _samples(chunk)):
        raise ValueError("Retained original clock differs")
    stages = [item for item in load_stages(workflow) if item.name == row["stageName"]]
    if len(stages) != 1 or receipt.name != row["stageName"] + ".json":
        raise ValueError("Missing original OCR Stage/receipt binding")
    directory = _path(data.get("ocrDirectory"))
    inputs, complete = directory / "inputs.json", directory / "complete.json"
    _pin(inputs, pins)
    _pin(complete, pins)
    # These are metadata pins, not a source/model hash operation at preparation.
    for field in (
        "native",
        "decoder",
        "model",
        "sourceClock",
        "classificationEvidence",
        "profileInputs",
        "profileComplete",
    ):
        reference = record(settings_data.get(field))
        name = _path(reference.get("path"))
        digest = session._sha(reference.get("sha256"))
        if str(name) in pins and pins[str(name)] != digest:
            raise ValueError("Conflicting retained input pins")
        pins[str(name)] = digest
    results = record(record(_read(complete)).get("results"))
    if not 1 <= len(results) <= 6250:
        raise ValueError("Invalid retained result inventory")
    for index, (result_name, result_digest) in enumerate(sorted(results.items())):
        if result_name != f"{index:04d}.result.json":
            raise ValueError("Retained result inventory has a gap")
        path = directory / result_name
        if _pin(path, pins)["sha256"] != session._sha(result_digest):
            raise ValueError("Retained raw result changed")
    return {**row, "directory": str(directory)}


def _command(operation: str, plan: Path) -> list[str]:
    return [sys.executable, "-B", "-m", "replay.subtitle_session_benchmark", operation, str(plan)]


def prepare(config: Path, output: Path) -> Prepared:
    """Only bounded metadata is read; source/native hashes are runtime guards."""
    _path(str(config), work=True)
    _path(str(output), work=True)
    if not output.is_dir() or any(output.iterdir()):
        raise ValueError("An existing empty benchmark directory is required")
    pins: dict[str, str] = {}
    config_ref = _pin(config, pins)
    raw = record(_read(config))
    if set(raw) != {"version", "sessionPlan", "clock", "oldNative", "retainedBaseline"} or (
        type(raw["version"]) is not int or raw["version"] != 1
    ):
        raise ValueError("Invalid benchmark configuration")
    session_path = _ref(raw["sessionPlan"], pins)
    plan = session.load_plan(session_path)
    if len(plan.samples) > 64:
        raise ValueError("A comparison must fit one native session")
    clock = _ref(raw["clock"], pins)
    all_points = _samples(clock)
    try:
        start = all_points.index(plan.samples[0])
    except ValueError as error:
        raise ValueError("Session PTS are absent from the physical clock") from error
    if all_points[start : start + len(plan.samples)] != plan.samples:
        raise ValueError("Session must select contiguous original physical PTS")
    native = record(raw["oldNative"])
    if set(native) != {"path", "sha256"}:
        raise ValueError("Expected an explicit original native executable pin")
    old_native = _path(native["path"])
    pins[str(old_native)] = session._sha(native["sha256"])
    for path, digest in ((plan.source, plan.source_sha), (plan.native, plan.native_sha)):
        if str(path) in pins and pins[str(path)] != digest:
            raise ValueError("Conflicting source/native input identity")
        pins[str(path)] = digest
    baseline = (
        None if raw["retainedBaseline"] is None else _baseline(raw["retainedBaseline"], pins, plan)
    )
    if baseline is not None:
        source_points = _samples(_path(record(baseline["chunk"])["path"]))
        try:
            start = source_points.index(plan.samples[0])
        except ValueError as error:
            raise ValueError("Retained baseline lacks selected PTS") from error
        if source_points[start : start + len(plan.samples)] != plan.samples:
            raise ValueError("Retained baseline selected clocks differ")
    times = output / "times.json"
    # Genuine legacy serialization: the original collector receives these floats.
    session._write(times, [float(point) for point in plan.samples])
    _pin(times, pins)
    for item in code_dependency_inventory.capture(("subtitle_session_benchmark",)):
        pins[str(item.path)] = item.sha256
    prepared = Prepared(output / "plan.json", output / "workflow.json", output / "state")
    data = {
        "version": 1,
        "configuration": config_ref,
        "sessionPlan": raw["sessionPlan"],
        "clock": raw["clock"],
        "oldNative": native,
        "baseline": baseline,
        "start": start,
        "times": str(times),
        "inputs": pins,
        "oldOutput": str(output / "old"),
        "newOutput": str(output / "new"),
        "oldEvidence": str(output / "old-evidence.json"),
        "newEvidence": str(output / "new-evidence.json"),
        "comparison": str(output / "comparison.json"),
    }
    _write(prepared.plan, data)
    common = [str(prepared.plan), *pins]
    if len(common) > 240:
        raise ValueError("Benchmark declared input budget exceeded")
    old_outputs = [data["oldEvidence"]]
    if baseline is None:
        old_outputs += [str(output / "old/inputs.json"), str(output / "old/complete.json")]
    rows: list[dict[str, object]] = []
    for name, outputs, additional in (
        ("old", old_outputs, []),
        (
            "new",
            [
                data["newEvidence"],
                str(output / "new/inputs.json"),
                str(output / "new/complete.json"),
            ],
            old_outputs,
        ),
        (
            "compare",
            [data["comparison"]],
            [
                data["oldEvidence"],
                data["newEvidence"],
                str(output / "new/inputs.json"),
                str(output / "new/complete.json"),
            ]
            + ([] if baseline is not None else old_outputs[1:]),
        ),
    ):
        rows.append(
            {
                "name": name,
                "command": _command(name, prepared.plan),
                "verify_command": _command("verify-" + name, prepared.plan),
                "inputs": common + additional,
                "outputs": outputs,
                "timeout_seconds": 3600,
            }
        )
    graph = {"version": 1, "stages": rows}
    stage_graph(parse_stages(graph))
    _write(prepared.workflow, graph)
    return prepared


def _load(path: Path) -> tuple[dict[str, object], session.SessionPlan, dict[str, str]]:
    data = decode(bounded_read(_path(str(path), work=True)))
    if (
        set(data)
        != {
            "version",
            "configuration",
            "sessionPlan",
            "clock",
            "oldNative",
            "baseline",
            "start",
            "times",
            "inputs",
            "oldOutput",
            "newOutput",
            "oldEvidence",
            "newEvidence",
            "comparison",
        }
        or type(data["version"]) is not int
        or data["version"] != 1
    ):
        raise ValueError("Invalid prepared benchmark")
    pins = {name: session._sha(value) for name, value in record(data["inputs"]).items()}
    for name in pins:
        _path(name)
    expected_pins: dict[str, str] = {}
    config = _ref(data["configuration"], expected_pins)
    original = record(_read(config))
    if (
        original.get("sessionPlan") != data["sessionPlan"]
        or original.get("clock") != data["clock"]
        or (original.get("oldNative") != data["oldNative"])
        or original.get("retainedBaseline")
        != (
            None
            if data["baseline"] is None
            else {
                key: value for key, value in record(data["baseline"]).items() if key != "directory"
            }
        )
    ):
        raise ValueError("Prepared plan differs from its pinned configuration")
    session_path = _ref(data["sessionPlan"], expected_pins)
    plan = session.load_plan(session_path)
    if not 1 <= len(plan.samples) <= 64:
        raise ValueError("Invalid comparison sample budget")
    clock = _ref(data["clock"], expected_pins)
    physical = _samples(clock)
    try:
        start = physical.index(plan.samples[0])
    except ValueError as error:
        raise ValueError("Session lacks original physical PTS") from error
    if physical[start : start + len(plan.samples)] != plan.samples:
        raise ValueError("Session skips physical PTS")
    old_native = record(data["oldNative"])
    for input_path, digest in (
        (_path(old_native["path"]), session._sha(old_native["sha256"])),
        (plan.source, plan.source_sha),
        (plan.native, plan.native_sha),
    ):
        if str(input_path) in expected_pins and expected_pins[str(input_path)] != digest:
            raise ValueError("Conflicting native/source pins")
        expected_pins[str(input_path)] = digest
    if data["baseline"] is not None:
        baseline = _baseline(original["retainedBaseline"], expected_pins, plan)
        points = _samples(_path(record(baseline["chunk"])["path"]))
        start = points.index(plan.samples[0])
        if (
            baseline != data["baseline"]
            or points[start : start + len(plan.samples)] != plan.samples
        ):
            raise ValueError("Prepared retained baseline differs")
    if type(data["start"]) is not int or data["start"] != start:
        raise ValueError("Prepared original selection differs")
    expected_paths = {
        "times": "times.json",
        "oldOutput": "old",
        "newOutput": "new",
        "oldEvidence": "old-evidence.json",
        "newEvidence": "new-evidence.json",
        "comparison": "comparison.json",
    }
    if any(
        data[field] != str(path.parent / filename) for field, filename in expected_paths.items()
    ):
        raise ValueError("Prepared benchmark destinations differ")
    selected = session._times(record(_read(session_path))["samples"])
    if legacy.observation_times(_read(_path(data["times"]))) != tuple(map(float, selected)):
        raise ValueError("Prepared legacy times differ")
    _pin(_path(data["times"]), expected_pins)
    for item in code_dependency_inventory.capture(("subtitle_session_benchmark",)):
        expected_pins[str(item.path)] = item.sha256
    if expected_pins != pins:
        raise ValueError("Benchmark consumed input/code inventory differs")
    return data, plan, pins


def _guard(pins: dict[str, str]) -> dict[str, dict[str, int]]:
    identities: dict[str, dict[str, int]] = {}
    for name, digest in pins.items():
        path = _path(name)
        # Legacy published raw results are hardlinks to their attempt response.
        before = path.lstat()
        if file_digest(path) != digest or path.lstat() != before:
            raise ValueError("Benchmark immutable input changed")
        identities[name] = {
            "device": before.st_dev,
            "inode": before.st_ino,
            "bytes": before.st_size,
            "mtimeNs": before.st_mtime_ns,
            "ctimeNs": before.st_ctime_ns,
            "links": before.st_nlink,
        }
    return identities


def _legacy_frames(data: dict[str, object], plan: session.SessionPlan) -> list[dict[str, object]]:
    native = record(data["oldNative"])
    baseline = data["baseline"]
    if baseline is None:
        directory = _path(data["oldOutput"], work=True)
        times = _path(data["times"])
        selected_start = 0
    else:
        old = record(baseline)
        chunk = _path(record(old["chunk"])["path"])
        cleanup.check_ocr(chunk)
        directory = _path(old["directory"])
        times = _path(record(old["times"])["path"])
        stages = [
            item
            for item in load_stages(_path(record(old["workflow"])["path"]))
            if item.name == old["stageName"]
        ]
        stage = stages[0]
        settings = record(_read(_path(record(record(_read(chunk))["settings"])["path"])))
        expected = parse_stages(
            {"version": 1, "stages": [cleanup._stages(chunk, settings, None)[0]]}
        )[0]
        if (
            stage.command != expected.command
            or stage.verify_command != expected.verify_command
            or (
                stage.outputs != expected.outputs
                or stage.timeout_seconds != expected.timeout_seconds
                or stage.adopt_existing
                or not set(expected.inputs).issubset(stage.inputs)
            )
        ):
            raise ValueError("Original Stage is not the genuine OCR producer")
        receipt = record(_read(_path(record(old["receipt"])["path"])))
        if receipt != {
            "status": "complete",
            "fingerprint": _fingerprint(stage),
            "outputs": {str(item): file_digest(item) for item in stage.outputs},
        }:
            raise ValueError("Original OCR Stage is not complete and unchanged")
        offset = data["start"]
        if type(offset) is not int or offset < 0:
            raise ValueError("Invalid original selection offset")
        selected_start = offset
    inputs = record(_read(directory / "inputs.json"))
    expected_inputs = {
        "source": str(plan.source),
        "sourceSHA256": plan.source_sha,
        "planSHA256": file_digest(times),
        "collectorSHA256": file_digest(Path(legacy.__file__).resolve()),
        "parserSHA256": file_digest(Path(caption_io.__file__).resolve()),
        "headParserSHA256": None,
        "measurementTool": "video_text_recognize",
        "fullFrame": True,
        "backend": "native-cli",
        "nativeExecutable": native["path"],
        "nativeSHA256": native["sha256"],
    }
    if inputs != expected_inputs:
        raise ValueError("Original collector provenance differs")
    frames, _ = cleanup.collector_frames(
        directory / "inputs.json",
        directory / "complete.json",
        {"source": {"path": str(plan.source), "sha256": plan.source_sha}, "native": native},
    )
    complete = record(_read(directory / "complete.json"))
    if set(complete) != {"observations", "samplingExhaustive", "results"} or (
        complete["samplingExhaustive"] is not False or type(complete["observations"]) is not int
    ):
        raise ValueError("Legacy completion schema or sampling claim differs")
    times_value = legacy.observation_times(_read(times))
    if (
        len(frames) != len(times_value)
        or tuple(frame.requested_time for frame in frames) != times_value
        or (tuple(frame.actual_time for frame in frames) != times_value)
    ):
        raise ValueError("Legacy requested/actual clock differs")
    raw: list[dict[str, object]] = []
    for name in sorted(record(record(_read(directory / "complete.json"))["results"])):
        values = _read(directory / name)
        if not isinstance(values, list):
            raise ValueError("Expected genuine raw frames")
        raw.extend(record(item) for item in values)
    selected = raw[selected_start : selected_start + len(plan.samples)]
    return _semantic(selected, plan)


def _semantic(
    values: list[dict[str, object]], plan: session.SessionPlan
) -> list[dict[str, object]]:
    if len(values) != len(plan.samples):
        raise ValueError("Selected raw frame count differs")
    for row, point in zip(values, plan.samples, strict=True):
        if set(row) != {
            "requestedTimeSeconds",
            "actualTimeSeconds",
            "width",
            "height",
            "lines",
        } or (
            row["requestedTimeSeconds"] != float(point)
            or row["actualTimeSeconds"] != float(point)
            or type(row["width"]) is not int
            or type(row["height"]) is not int
            or row["width"] != plan.width
            or row["height"] != plan.height
        ):
            raise ValueError("Raw frame schema/actual PTS/canvas differs")
        caption_io.native_ocr_frames([row])
        lines = row["lines"]
        if not isinstance(lines, list):
            raise ValueError("Expected ordered native lines")
        for item in lines:
            line = record(item)
            if set(line) != {"text", "confidence", "boundsClipped", "region"} or (
                type(line["boundsClipped"]) is not bool
                or set(record(line["region"])) != {"x", "y", "width", "height"}
            ):
                raise ValueError("Raw native clipping/rectangle schema differs")
    return values


def _new_frames(data: dict[str, object], plan: session.SessionPlan) -> list[dict[str, object]]:
    directory = _path(data["newOutput"], work=True)
    session.collect(_path(record(data["sessionPlan"])["path"]), directory, verify=True)
    complete = record(_read(directory / "complete.json"))
    raw: list[dict[str, object]] = []
    for name, sha in sorted(record(complete["boundaries"]).items()):
        boundary_path = _path(name)
        if not boundary_path.is_relative_to(directory) or file_digest(boundary_path) != sha:
            raise ValueError("Session boundary differs")
        boundary = record(_read(boundary_path))
        count = boundary["count"]
        if type(count) is not int or count <= 0:
            raise ValueError("Incomplete session boundary")
        for index in range((count + 7) // 8):
            values = _read(boundary_path.parent / f"native/block-{index}.json")
            if not isinstance(values, list):
                raise ValueError("Expected session raw frames")
            raw.extend(record(item) for item in values)
    return _semantic(raw, plan)


def _evidence(
    data: dict[str, object],
    plan: session.SessionPlan,
    mode: str,
    frames: list[dict[str, object]],
    identities: dict[str, dict[str, int]],
) -> dict[str, object]:
    return {
        "version": 1,
        "mode": mode,
        "inputs": data["inputs"],
        "identities": identities,
        "samples": session._clocks(plan.samples),
        "framesSHA256": hashlib.sha256(
            json.dumps(frames, sort_keys=True, allow_nan=False).encode()
        ).hexdigest(),
        "nativeBinaryIdentical": record(data["oldNative"])["sha256"] == plan.native_sha,
        "retainedBaseline": mode == "old" and data["baseline"] is not None,
        "qualityApproved": False,
        "deliveryApproved": False,
    }


def _elapsed(receipt: dict[str, object]) -> float:
    value = receipt.get("elapsedSeconds")
    if (
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or (not math.isfinite(value) or value < 0)
    ):
        raise ValueError("Invalid measured benchmark elapsed time")
    return float(value)


def phase(path: Path, mode: str, *, verify: bool = False) -> None:
    """Delegate collectors; verification never collects missing results or writes."""
    data, plan, pins = _load(path)
    with execution_lock(path.parent, "benchmark"):
        before = _guard(pins)
        if mode == "compare":
            phase_data = []
            for selected in ("old", "new"):
                receipt = record(_read(_path(data[selected + "Evidence"], work=True)))
                _elapsed(receipt)
                frames = (
                    _legacy_frames(data, plan) if selected == "old" else _new_frames(data, plan)
                )
                if {
                    key: value for key, value in receipt.items() if key != "elapsedSeconds"
                } != _evidence(data, plan, selected, frames, before):
                    raise ValueError("Benchmark producer evidence changed")
                phase_data.append(frames)
            differences = [
                index
                for index, (old, new) in enumerate(zip(*phase_data, strict=True))
                if old != new
            ]
            result = {
                "version": 1,
                "samples": session._clocks(plan.samples),
                "oldEvidence": _pin(_path(data["oldEvidence"]), {}),
                "newEvidence": _pin(_path(data["newEvidence"]), {}),
                "equalMeasurements": not differences,
                "differentFrameIndices": differences,
                "tolerance": None,
                "qualityApproved": False,
                "deliveryApproved": False,
                "nativeBinaryIdentical": record(data["oldNative"])["sha256"] == plan.native_sha,
            }
            destination = _path(data["comparison"], work=True)
        elif mode in {"old", "new"}:
            destination = _path(data[mode + "Evidence"], work=True)
            started = time.monotonic()
            if not verify and not destination.exists():
                if mode == "old" and data["baseline"] is None:
                    legacy.collect_observations(
                        plan.source,
                        _path(data["times"]),
                        _path(data["oldOutput"], work=True),
                        native_executable=_path(record(data["oldNative"])["path"]),
                    )
                elif mode == "new":
                    session.collect(
                        _path(record(data["sessionPlan"])["path"]),
                        _path(data["newOutput"], work=True),
                    )
            frames = _legacy_frames(data, plan) if mode == "old" else _new_frames(data, plan)
            result = _evidence(data, plan, mode, frames, before)
            if destination.exists():
                receipt = record(_read(destination))
                result["elapsedSeconds"] = _elapsed(receipt)
            else:
                result["elapsedSeconds"] = time.monotonic() - started
        else:
            raise ValueError("Unknown benchmark phase")
        if _guard(pins) != before:
            raise ValueError("Benchmark input identities changed during phase")
        if destination.exists():
            if _read(destination) != result:
                raise ValueError("Completed benchmark evidence differs; preserve it")
        elif verify:
            raise ValueError("Benchmark evidence is incomplete")
        else:
            _write(destination, result)
        if mode == "compare" and result["equalMeasurements"] is not True:
            raise ValueError("OCR measurements differ; diagnostic evidence retained")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "operation",
        choices=("prepare", "old", "new", "compare", "verify-old", "verify-new", "verify-compare"),
    )
    parser.add_argument("config", type=Path)
    parser.add_argument("output", type=Path, nargs="?")
    args = parser.parse_args(argv)
    if args.operation == "prepare":
        if args.output is None:
            parser.error("prepare requires OUTPUT")
        prepare(args.config, args.output)
    else:
        if args.output is not None:
            parser.error("Only prepare accepts OUTPUT")
        phase(
            args.config,
            args.operation.removeprefix("verify-"),
            verify=args.operation.startswith("verify-"),
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
