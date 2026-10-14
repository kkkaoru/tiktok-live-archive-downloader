"""Compare bounded complete OCR intervals through the existing session collector."""

import argparse
import hashlib
import math
import os
import resource
import subprocess
import sys
import threading
import time
from dataclasses import replace
from fractions import Fraction
from pathlib import Path

from replay import code_dependency_inventory
from replay import subtitle_observations as legacy
from replay import subtitle_session_benchmark as benchmark
from replay import subtitle_session_observations as session
from replay.core import ReplayError
from replay.process_family_observation import SystemHost, sample_family
from replay.recordings import record
from replay.sequence_journal import encode
from replay.stages import execution_lock, file_digest
from replay.workflow import load_stages, parse_stages, stage_graph

MAXIMUM_FRAMES = 10000  # Existing physical-clock decoder's validated bound.
MAXIMUM_INTERVALS = 16


def _baseline(
    value: object,
    pins: dict[str, str],
    plan: session.SessionPlan,
    decoder_root: session.SourceRoot | None,
) -> dict[str, object]:
    if decoder_root is None:
        return benchmark._baseline(value, pins, plan)
    # Only metadata binding differs: the original installed decoder remains an
    # explicitly pinned external input. Genuine raw/Stage verification is shared.
    row = record(value)
    if set(row) != {"chunk", "times", "workflow", "receipt", "stageName"}:
        raise ValueError("Invalid retained baseline contract")
    chunk, times, workflow, receipt = (
        benchmark._ref(row[key], pins) for key in ("chunk", "times", "workflow", "receipt")
    )
    data = record(benchmark._read(chunk))
    settings = record(benchmark._read(benchmark._ref(data.get("settings"), pins)))
    if (
        record(settings.get("source")) != {"path": str(plan.source), "sha256": plan.source_sha}
        or times != chunk.parent / "times.json"
    ):
        raise ValueError("Retained source/times differs")
    if legacy.observation_times(benchmark._read(times)) != tuple(
        map(float, benchmark._samples(chunk))
    ):
        raise ValueError("Retained original clock differs")
    if (
        not isinstance(row["stageName"], str)
        or receipt.name != row["stageName"] + ".json"
        or len([stage for stage in load_stages(workflow) if stage.name == row["stageName"]]) != 1
    ):
        raise ValueError("Missing original OCR Stage/receipt binding")
    directory = benchmark._path(data.get("ocrDirectory"))
    benchmark._pin(directory / "inputs.json", pins)
    benchmark._pin(directory / "complete.json", pins)
    for field in (
        "native",
        "decoder",
        "model",
        "sourceClock",
        "classificationEvidence",
        "profileInputs",
        "profileComplete",
    ):
        reference = record(settings.get(field))
        if set(reference) != {"path", "sha256"}:
            raise ValueError("Invalid original input reference")
        path = (
            session._source_path(reference["path"], decoder_root)
            if field == "decoder"
            else benchmark._path(reference["path"])
        )
        digest = session._sha(reference["sha256"])
        if str(path) in pins and pins[str(path)] != digest:
            raise ValueError("Conflicting original input pin")
        pins[str(path)] = digest
    results = record(record(benchmark._read(directory / "complete.json"))["results"])
    if not 1 <= len(results) <= 6250:
        raise ValueError("Invalid retained result inventory")
    for index, (name, result_digest) in enumerate(sorted(results.items())):
        if name != f"{index:04d}.result.json" or benchmark._pin(directory / name, pins)[
            "sha256"
        ] != session._sha(result_digest):
            raise ValueError("Retained raw result changed or has a gap")
    return {**row, "directory": str(directory)}


def _derive(config: Path, output: Path) -> tuple[dict[str, object], session.SessionPlan]:
    pins: dict[str, str] = {}
    configuration = benchmark._pin(config, pins)
    raw = record(benchmark._read(config))
    if set(raw) not in (
        {"version", "sessionPlan", "oldNative", "intervals"},
        {"version", "sessionPlan", "oldNative", "intervals", "allowedDecoderRoot"},
    ) or (type(raw["version"]) is not int or raw["version"] != 1):
        raise ValueError("Invalid interval benchmark configuration")
    decoder_root = (
        session._source_root(raw["allowedDecoderRoot"]) if "allowedDecoderRoot" in raw else None
    )
    plan_path = benchmark._ref(raw["sessionPlan"], pins)
    plan = session.load_plan(plan_path)
    if not 1 <= len(plan.samples) <= MAXIMUM_FRAMES or plan.workers != 2:
        raise ValueError("Expected bounded samples and existing two-worker protocol")
    original_native = record(raw["oldNative"])
    if set(original_native) != {"path", "sha256"}:
        raise ValueError("Expected original native executable pin")
    old_native = benchmark._path(original_native["path"])
    pins[str(old_native)] = session._sha(original_native["sha256"])
    intervals = raw["intervals"]
    if not isinstance(intervals, list) or not 1 <= len(intervals) <= MAXIMUM_INTERVALS:
        raise ValueError("Invalid complete interval inventory")
    derived: list[dict[str, object]] = []
    raw_inputs: set[str] = set()
    points: list[Fraction] = []
    for item in intervals:
        interval = record(item)
        if set(interval) not in (
            {"clock", "retainedBaseline"},
            {"clock", "retainedBaseline", "baselineFilesystemEvents"},
        ):
            raise ValueError("Expected clock and retained complete baseline")
        clock = benchmark._ref(interval["clock"], pins)
        selected = benchmark._samples(clock)
        if not selected or (points and selected[0] <= points[-1]):
            raise ValueError("Intervals overlap or have unordered physical PTS")
        points.extend(selected)
        if len(points) > MAXIMUM_FRAMES:
            raise ValueError("Aggregate physical-clock bound exceeded")
        partial = replace(plan, samples=selected)
        baseline = _baseline(interval["retainedBaseline"], pins, partial, decoder_root)
        directory = benchmark._path(baseline["directory"])
        results = record(record(benchmark._read(directory / "complete.json"))["results"])
        raw_inputs.update(str(directory / name) for name in results)
        if record(baseline["chunk"]) != interval["clock"]:
            raise ValueError("Complete interval must use its original chunk clock")
        if benchmark._samples(benchmark._path(record(baseline["chunk"])["path"])) != selected:
            raise ValueError("Retained interval clock differs")
        events: object = None
        if "baselineFilesystemEvents" in interval:
            event_refs = record(interval["baselineFilesystemEvents"])
            if set(event_refs) != {"begin", "end"}:
                raise ValueError("Expected two retained filesystem events")
            event_paths = [benchmark._ref(event_refs[key], pins) for key in ("begin", "end")]
            births = [item.stat().st_birthtime for item in event_paths]
            if not all(math.isfinite(value) for value in births) or births[1] < births[0]:
                raise ValueError("Invalid retained filesystem event order")
            events = {
                "references": event_refs,
                "birthTimesSeconds": births,
                "elapsedSeconds": births[1] - births[0],
                "processTimer": False,
            }
        derived.append(
            {"baseline": baseline, "count": len(selected), "start": 0, "filesystemEvents": events}
        )
    if tuple(points) != plan.samples:
        raise ValueError("Session must cover every physical PTS of the complete interval union")
    for path, digest in ((plan.source, plan.source_sha), (plan.native, plan.native_sha)):
        if str(path) in pins and pins[str(path)] != digest:
            raise ValueError("Conflicting source/native pins")
        pins[str(path)] = digest
    for source in code_dependency_inventory.capture(("subtitle_session_interval_benchmark",)):
        pins[str(source.path)] = source.sha256
    if len(pins) > 512 or len(set(pins) - raw_inputs) > 240:
        raise ValueError("Interval benchmark declared input budget exceeded")
    return {
        "version": 1,
        "configuration": configuration,
        "sessionPlan": raw["sessionPlan"],
        "oldNative": original_native,
        "intervals": derived,
        "inputs": pins,
        "rawInputs": sorted(raw_inputs),
        "decoderRoot": None
        if decoder_root is None
        else {
            "path": str(decoder_root.path),
            "device": decoder_root.device,
            "inode": decoder_root.inode,
        },
        "newOutput": str(output / "new"),
        "oldEvidence": str(output / "old-evidence.json"),
        "newEvidence": str(output / "new-evidence.json"),
        "comparison": str(output / "comparison.json"),
    }, plan


def _command(mode: str, plan: Path) -> list[str]:
    return [
        sys.executable,
        "-B",
        "-m",
        "replay.subtitle_session_interval_benchmark",
        mode,
        str(plan),
    ]


def prepare(config: Path, output: Path) -> benchmark.Prepared:
    """Read bounded metadata, without running OCR or hashing the original media."""
    benchmark._path(str(config), work=True)
    benchmark._path(str(output), work=True)
    if not output.is_dir() or any(output.iterdir()):
        raise ValueError("Expected existing empty benchmark destination")
    data, _ = _derive(config, output)
    prepared = benchmark.Prepared(output / "plan.json", output / "workflow.json", output / "state")
    raw_inputs = data["rawInputs"]
    if not isinstance(raw_inputs, list):
        raise ValueError("Invalid retained raw pin inventory")
    # complete.json commits every raw result; _guard additionally hashes all raw
    # pins at every phase. Keep transitive raw evidence out of the 256-input graph.
    common = [
        str(prepared.plan),
        *(name for name in record(data["inputs"]) if name not in raw_inputs),
    ]
    old = [data["oldEvidence"]]
    new = [data["newEvidence"], str(output / "new/inputs.json"), str(output / "new/complete.json")]
    rows = []
    for mode, outputs, extra in (
        ("old", old, []),
        ("new", new, old),
        ("compare", [data["comparison"]], old + new),
    ):
        rows.append(
            {
                "name": mode,
                "command": _command(mode, prepared.plan),
                "verify_command": _command("verify-" + mode, prepared.plan),
                "inputs": common + extra,
                "outputs": outputs,
                "timeout_seconds": 3600,
            }
        )
    graph = {"version": 1, "stages": rows}
    stage_graph(parse_stages(graph))
    benchmark._write(prepared.plan, data)
    benchmark._write(prepared.workflow, graph)
    return prepared


def _load(path: Path) -> tuple[dict[str, object], session.SessionPlan, dict[str, str]]:
    benchmark._path(str(path), work=True)
    data = record(benchmark._read(path))
    config = benchmark._ref(data.get("configuration"), {})
    expected, plan = _derive(config, path.parent)
    if data != expected:
        raise ValueError("Prepared interval benchmark differs from immutable configuration")
    return data, plan, {name: session._sha(value) for name, value in record(data["inputs"]).items()}


def _guard(
    pins: dict[str, str], plan: session.SessionPlan, decoder_root: session.SourceRoot | None = None
) -> dict[str, dict[str, int]]:
    session._check({str(plan.source): pins[str(plan.source)]}, plan)
    internal = {
        name: digest
        for name, digest in pins.items()
        if name != str(plan.source) and Path(name).is_relative_to(Path.cwd())
    }
    identities = benchmark._guard(internal)
    for name, digest in pins.items():
        if name == str(plan.source) or name in internal:
            continue
        path = session._source_path(name, decoder_root)
        before_decoder = session._identity(path, decoder_root)
        if file_digest(path) != digest or session._identity(path, decoder_root) != before_decoder:
            raise ValueError("Original external decoder changed")
        identities[name] = before_decoder
    before = plan.source.stat()
    identities[str(plan.source)] = {
        "device": before.st_dev,
        "inode": before.st_ino,
        "bytes": before.st_size,
        "mtimeNs": before.st_mtime_ns,
        "ctimeNs": before.st_ctime_ns,
        "links": before.st_nlink,
    }
    return identities


def _old_frames(data: dict[str, object], plan: session.SessionPlan) -> list[dict[str, object]]:
    values = data["intervals"]
    if not isinstance(values, list):
        raise ValueError("Expected prepared intervals")
    frames: list[dict[str, object]] = []
    offset = 0
    for value in values:
        interval = record(value)
        count = interval["count"]
        if type(count) is not int or count <= 0:
            raise ValueError("Invalid interval count")
        partial = replace(plan, samples=plan.samples[offset : offset + count])
        frames.extend(benchmark._legacy_frames({**data, **interval}, partial))
        offset += count
    return benchmark._semantic(frames, plan)


class PhysicalObservation:
    """Observe actual family snapshots; the maximum is not a kernel peak guarantee."""

    def __init__(self) -> None:
        self.stop = threading.Event()
        self.host = SystemHost()
        self.pid = os.getpid()
        self.birth = self.host.birth(self.pid)
        self.count = 0
        self.maximum = 0
        self.rss_maximum = 0
        self.failure: OSError | ReplayError | ValueError | subprocess.SubprocessError | None = None
        self.unavailable = False
        self.thread = threading.Thread(target=self._observe, name="ocr-physical-observer")

    def _observe(self) -> None:
        try:
            while not self.stop.is_set():
                row = sample_family(self.pid, self.birth, self.host)
                physical, rss = row["physicalBytes"], row["rssBytes"]
                if type(physical) is not int or type(rss) is not int:
                    raise ValueError("Invalid physical/RSS observation")
                self.maximum = max(self.maximum, physical)
                self.rss_maximum = max(self.rss_maximum, rss)
                self.count += 1
                self.stop.wait(0.1)
        except PermissionError:
            # The OCR result remains independently verified.  A confined
            # execution may prohibit ps, so record that this optional physical
            # diagnostic is unavailable instead of discarding the exact OCR
            # comparison or pretending that a peak was observed.
            self.unavailable = True
        except (OSError, ReplayError, ValueError, subprocess.SubprocessError) as error:
            self.failure = error

    def finish(self) -> dict[str, object]:
        self.stop.set()
        self.thread.join(timeout=5)
        if self.thread.is_alive():
            raise ValueError("Physical observer did not terminate")
        if self.failure is not None:
            raise self.failure
        if self.unavailable:
            return {
                "available": False,
                "reason": "process-family observation denied by confined execution",
                "peakGuarantee": False,
            }
        if self.count == 0:
            raise ValueError("No actual physical family observation")
        return {
            "available": True,
            "rootPID": self.pid,
            "rootBirth": self.birth,
            "samples": self.count,
            "intervalSeconds": 0.1,
            "maximumObservedPhysicalBytes": self.maximum,
            "maximumObservedRSSBytes": self.rss_maximum,
            "peakGuarantee": False,
            "includesCollectorAndObserver": True,
        }


def _cpu() -> float:
    return sum(
        usage.ru_utime + usage.ru_stime
        for usage in (
            resource.getrusage(resource.RUSAGE_SELF),
            resource.getrusage(resource.RUSAGE_CHILDREN),
        )
    )


def _collect(data: dict[str, object]) -> dict[str, object]:
    output = benchmark._path(data["newOutput"], work=True)
    resumed = output.exists() and any(output.iterdir())
    observer = PhysicalObservation()
    observer.thread.start()
    wall, cpu = time.monotonic(), _cpu()
    try:
        session.collect(
            benchmark._path(record(data["sessionPlan"])["path"]),
            output,
        )
    finally:
        physical = observer.finish()
    return {
        "wallSeconds": time.monotonic() - wall,
        "cpuSeconds": _cpu() - cpu,
        "cpuScope": "collector and reaped children including physical observer ps",
        "physicalObservation": physical,
        "gpuObservation": {"supported": False, "seconds": None},
        "resumedCollection": resumed,
        "fullIntervalTiming": not resumed,
    }


def _digest(frames: list[dict[str, object]]) -> str:
    return hashlib.sha256(encode(record({"frames": frames}))).hexdigest()


def _metrics(value: object) -> dict[str, object]:
    row = record(value)
    if set(row) != {
        "wallSeconds",
        "cpuSeconds",
        "cpuScope",
        "physicalObservation",
        "gpuObservation",
        "resumedCollection",
        "fullIntervalTiming",
    }:
        raise ValueError("Invalid actual benchmark metrics")
    for field in ("wallSeconds", "cpuSeconds"):
        number = row[field]
        if (
            isinstance(number, bool)
            or not isinstance(number, (int, float))
            or not math.isfinite(number)
            or number < 0
        ):
            raise ValueError("Invalid actual process timer")
    if row["cpuScope"] != "collector and reaped children including physical observer ps" or row[
        "gpuObservation"
    ] != {"supported": False, "seconds": None}:
        raise ValueError("Unsupported metric provenance")
    if type(row["resumedCollection"]) is not bool or row["fullIntervalTiming"] is not (
        not row["resumedCollection"]
    ):
        raise ValueError("Invalid resumed timing scope")
    physical = record(row["physicalObservation"])
    available = physical.get("available")
    if available is False:
        if physical != {
            "available": False,
            "reason": "process-family observation denied by confined execution",
            "peakGuarantee": False,
        }:
            raise ValueError("Invalid unavailable physical observation contract")
        return row
    if available is not True or set(physical) != {
        "available",
        "rootPID",
        "rootBirth",
        "samples",
        "intervalSeconds",
        "maximumObservedPhysicalBytes",
        "maximumObservedRSSBytes",
        "peakGuarantee",
        "includesCollectorAndObserver",
    }:
        raise ValueError("Invalid physical observation contract")
    for field in ("rootPID", "samples", "maximumObservedPhysicalBytes", "maximumObservedRSSBytes"):
        number = physical[field]
        if type(number) is not int or number <= 0:
            raise ValueError("Missing actual physical observation")
    if (
        not isinstance(physical["rootBirth"], str)
        or not physical["rootBirth"]
        or physical["intervalSeconds"] != 0.1
        or physical["peakGuarantee"] is not False
        or physical["includesCollectorAndObserver"] is not True
    ):
        raise ValueError("Invalid physical observation identity/scope")
    return row


def phase(path: Path, mode: str, *, verify: bool = False) -> None:
    data, plan, pins = _load(path)
    decoder = data["decoderRoot"]
    decoder_root = None if decoder is None else session._source_root(record(decoder)["path"])
    if mode not in {"old", "new", "compare"}:
        raise ValueError("Unknown benchmark phase")
    if mode == "compare":
        phase(path, "old", verify=True)
        phase(path, "new", verify=True)
    with execution_lock(path.parent, "interval-benchmark"):
        before = _guard(pins, plan, decoder_root)
        destination = benchmark._path(
            data["comparison" if mode == "compare" else mode + "Evidence"], work=True
        )
        existing = record(benchmark._read(destination)) if destination.exists() else None
        if verify and existing is None:
            raise ValueError("Missing benchmark evidence")
        metrics: object = None
        if mode == "new":
            metrics = _metrics(existing.get("metrics") if existing is not None else _collect(data))
        if mode == "compare":
            old, new = _old_frames(data, plan), benchmark._new_frames(data, plan)
            differences = [i for i, (a, b) in enumerate(zip(old, new, strict=True)) if a != b]
            result: dict[str, object] = {
                "version": 1,
                "equalMeasurements": not differences,
                "differentFrameIndices": differences,
                "oldEvidence": benchmark._pin(benchmark._path(data["oldEvidence"]), {}),
                "newEvidence": benchmark._pin(benchmark._path(data["newEvidence"]), {}),
            }
        else:
            frames = _old_frames(data, plan) if mode == "old" else benchmark._new_frames(data, plan)
            result = {
                "version": 1,
                "mode": mode,
                "framesSHA256": _digest(frames),
                "retainedBaseline": mode == "old",
                "metrics": metrics,
                "baselineProcessTimersAvailable": False if mode == "old" else None,
                "baselineFilesystemEvents": [
                    record(item)["filesystemEvents"] for item in data["intervals"]
                ]
                if mode == "old" and isinstance(data["intervals"], list)
                else None,
            }
        result.update(
            inputs=pins,
            identities=before,
            samples=session._clocks(plan.samples),
            qualityApproved=False,
            deliveryApproved=False,
            tolerance=None,
        )
        if _guard(pins, plan, decoder_root) != before:
            raise ValueError("Input changed during benchmark")
        if existing is not None:
            if existing != result:
                raise ValueError("Completed evidence differs; preserve it")
        else:
            benchmark._write(destination, result)
        if mode == "compare" and result["equalMeasurements"] is not True:
            raise ValueError("Raw OCR differs; diagnostic evidence retained")


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
