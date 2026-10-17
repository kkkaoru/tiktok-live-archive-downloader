"""Bounded full-interval comparison using genuine collectors and synthetic frames."""

from dataclasses import replace
from fractions import Fraction
from pathlib import Path
from unittest.mock import Mock

import pytest
from test_subtitle_session_benchmark import Case, case, frames, pin, write

from replay import subtitle_cleanup_workflow as cleanup
from replay import subtitle_observations as legacy
from replay import subtitle_session_benchmark as baseline
from replay import subtitle_session_interval_benchmark as benchmark
from replay import subtitle_session_observations as session
from replay.core import ReplayError
from replay.recordings import record
from replay.stages import _fingerprint, file_digest, run_stage
from replay.workflow import load_stages, stage_graph


def fixture_case(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, *, count: int = 16, interval: int = 8
) -> Case:
    fixture = case(tmp_path, monkeypatch, count=count)
    points = tuple(Fraction(index, 90) for index in range(count))
    write(fixture.clock, {"pts": list(range(count)), "timeBase": "1/90", "lastDuration": 1})
    plan = record(session._read(fixture.session_plan))
    plan.update(samples=session._clocks(points), workers=2)
    write(fixture.session_plan, plan)
    root = tmp_path / "retained"
    profile = root / "profile"
    raw = write(profile / "0000.result.json", frames(points[:8]))
    second = write(profile / "0001.result.json", frames(points[8:16]))
    inputs = write(
        profile / "inputs.json",
        {
            "sourceSHA256": file_digest(fixture.source),
            "nativeSHA256": file_digest(fixture.old_native),
            "nativeExecutable": str(fixture.old_native),
            "measurementTool": "video_text_recognize",
            "fullFrame": True,
        },
    )
    complete = write(
        profile / "complete.json",
        {
            "observations": 16,
            "results": {raw.name: file_digest(raw), second.name: file_digest(second)},
        },
    )
    model = root / "model"
    model.write_bytes(b"synthetic model")
    classification = write(root / "classification.json", {"classification": "synthetic"})
    settings = write(
        root / "settings.json",
        {
            "version": 1,
            "source": pin(fixture.source),
            "native": pin(fixture.old_native),
            "decoder": pin(fixture.old_native),
            "model": pin(model),
            "sourceClock": pin(fixture.clock),
            "classificationEvidence": pin(classification),
            "profileInputs": pin(inputs),
            "profileComplete": pin(complete),
            "chunkFrames": interval,
        },
    )
    cleanup.prepare(settings, root / "cleanup", workers=1)
    workflow = root / "cleanup/worker-00.json"
    intervals = []
    for stage in (item for item in load_stages(workflow) if item.name.endswith("-ocr")):
        chunk = stage.inputs[0]
        # Match by declared collector times, not assumptions about input ordering.
        times = Path(stage.command[5])
        directory = times.parent
        chunk = directory / "chunk.json"
        legacy.collect_observations(
            fixture.source, times, directory / "ocr", native_executable=fixture.old_native
        )
        # The original next OCR Stage declares the preceding repair dependency.
        # This synthetic predecessor is fingerprinted; no repair is claimed here.
        for dependency in stage.inputs:
            if not dependency.exists():
                write(dependency, {"syntheticPredecessor": True})
        receipt = write(
            root / "state" / (stage.name + ".json"),
            {
                "status": "complete",
                "fingerprint": _fingerprint(stage),
                "outputs": {str(path): file_digest(path) for path in stage.outputs},
            },
        )
        intervals.append(
            {
                "clock": pin(chunk),
                "retainedBaseline": {
                    "chunk": pin(chunk),
                    "times": pin(times),
                    "workflow": pin(workflow),
                    "receipt": pin(receipt),
                    "stageName": stage.name,
                },
            }
        )
    write(
        fixture.config,
        {
            "version": 1,
            "sessionPlan": pin(fixture.session_plan),
            "oldNative": pin(fixture.old_native),
            "intervals": intervals,
        },
    )
    fixture.old_calls.clear()
    monkeypatch.setattr(benchmark, "SystemHost", lambda: Mock(birth=lambda pid: "123:4"))
    monkeypatch.setattr(
        benchmark,
        "sample_family",
        lambda *args: {
            "physicalBytes": 4000,
            "rssBytes": 2000,
        },
    )
    return fixture


@pytest.mark.parametrize("count,interval", [(16, 8), (900, 450), (900, 900)])
def test_normal_workflow_exact_complete_union_and_reuse(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, count: int, interval: int
) -> None:
    fixture = fixture_case(tmp_path, monkeypatch, count=count, interval=interval)
    prepared = benchmark.prepare(fixture.config, fixture.output)
    stages = load_stages(prepared.workflow)
    assert stage_graph(stages) == {"old": (), "new": ("old",), "compare": ("old", "new")}

    def execute(command: tuple[str, ...], log: Path, timeout: int) -> None:
        assert timeout == 3600
        benchmark.main([command[4], command[5]])
        log.write_text("synthetic normal stage\n")

    for stage in stages:
        assert run_stage(stage, state_directory=prepared.state, execute=execute) == "completed"
        assert run_stage(stage, state_directory=prepared.state, execute=execute) == "reused"
    assert not fixture.old_calls
    assert fixture.run.call_count == (count + 63) // 64
    result = record(baseline._read(fixture.output / "comparison.json"))
    assert result["equalMeasurements"] is True
    samples = result["samples"]
    assert isinstance(samples, list) and len(samples) == count
    evidence = record(baseline._read(fixture.output / "new-evidence.json"))
    metrics = record(evidence["metrics"])
    physical = record(metrics["physicalObservation"])
    assert physical["available"] is True
    assert physical["maximumObservedPhysicalBytes"] == 4000
    assert physical["maximumObservedRSSBytes"] == 2000
    assert physical["peakGuarantee"] is False
    assert metrics["gpuObservation"] == {"supported": False, "seconds": None}
    assert result["qualityApproved"] is False and result["deliveryApproved"] is False


@pytest.mark.parametrize(
    "problem",
    [
        "unknown",
        "version",
        "intervals",
        "missing",
        "overlap",
        "clock",
        "workers",
        "samples",
        "oldnative",
        "conflict",
    ],
)
def test_invalid_configuration_before_inference(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, problem: str
) -> None:
    fixture = fixture_case(tmp_path, monkeypatch)
    config = record(session._read(fixture.config))
    intervals = config["intervals"]
    assert isinstance(intervals, list)
    if problem == "unknown":
        config["extra"] = 1
    elif problem == "version":
        config["version"] = True
    elif problem == "intervals":
        config["intervals"] = []
    elif problem == "missing":
        config["intervals"] = intervals[:1]
    elif problem == "overlap":
        config["intervals"] = [intervals[0], intervals[0]]
    elif problem == "clock":
        record(intervals[0])["clock"] = pin(fixture.clock)
        intervals[0] = {**record(intervals[0]), "clock": pin(fixture.clock)}
    elif problem in {"workers", "samples"}:
        plan = record(session._read(fixture.session_plan))
        plan[problem] = 1 if problem == "workers" else session._clocks((Fraction(100, 90),))
        write(fixture.session_plan, plan)
        config["sessionPlan"] = pin(fixture.session_plan)
    elif problem == "oldnative":
        config["oldNative"] = {**pin(fixture.old_native), "extra": 1}
    else:
        config["oldNative"] = {"path": str(fixture.new_native), "sha256": "0" * 64}
    write(fixture.config, config)
    with pytest.raises(ValueError):
        benchmark.prepare(fixture.config, fixture.output)
    assert fixture.run.call_count == 0


@pytest.mark.parametrize("problem", ["source", "binary", "prepared", "config", "occupied"])
def test_changed_inputs_and_occupied_output(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, problem: str
) -> None:
    fixture = fixture_case(tmp_path, monkeypatch)
    if problem == "occupied":
        (fixture.output / "keep").write_bytes(b"evidence")
        with pytest.raises(ValueError):
            benchmark.prepare(fixture.config, fixture.output)
        return
    prepared = benchmark.prepare(fixture.config, fixture.output)
    if problem == "source":
        fixture.source.write_bytes(b"changed source")
    elif problem == "binary":
        fixture.new_native.write_bytes(b"changed binary")
    elif problem == "config":
        fixture.config.write_bytes(b"{}")
    else:
        data = record(session._read(prepared.plan))
        data["newOutput"] = str(fixture.root / "elsewhere")
        write(prepared.plan, data)
    with pytest.raises(ValueError):
        benchmark.phase(prepared.plan, "new")
    assert fixture.run.call_count == 0


def test_missing_partial_and_failed_worker_are_preserved(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    fixture = fixture_case(tmp_path, monkeypatch, count=80, interval=80)
    prepared = benchmark.prepare(fixture.config, fixture.output)
    with pytest.raises(ValueError, match="Missing"):
        benchmark.phase(prepared.plan, "new", verify=True)
    original = session._attempt

    def fail_second(
        plan: session.SessionPlan, directory: Path, start: int, pins: dict[str, str]
    ) -> int:
        if start == 64:
            raise ValueError("Native worker failed")
        return original(plan, directory, start, pins)

    monkeypatch.setattr(session, "_attempt", fail_second)
    with pytest.raises(ValueError):
        benchmark.phase(prepared.plan, "new")
    assert not (fixture.output / "new-evidence.json").exists()
    assert tuple((fixture.output / "new").glob("attempt-*/boundary.json"))
    before = fixture.run.call_count
    monkeypatch.setattr(session, "_attempt", original)
    benchmark.phase(prepared.plan, "new")
    assert fixture.run.call_count == before + 1
    receipt = record(session._read(fixture.output / "new-evidence.json"))
    metrics = record(receipt["metrics"])
    assert metrics["resumedCollection"] is True and metrics["fullIntervalTiming"] is False
    benchmark.phase(prepared.plan, "new", verify=True)
    assert fixture.run.call_count == before + 1


def test_changed_text_at_interval_seam_retains_diagnostic(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    fixture = fixture_case(tmp_path, monkeypatch)
    prepared = benchmark.prepare(fixture.config, fixture.output)
    benchmark.phase(prepared.plan, "old")
    benchmark.phase(prepared.plan, "new")
    original = baseline._new_frames

    def changed(data: dict[str, object], plan: session.SessionPlan) -> list[dict[str, object]]:
        values = original(data, plan)
        values[8]["lines"] = []
        return values

    monkeypatch.setattr(baseline, "_new_frames", changed)
    # A matching forged producer receipt still cannot pass all-frame comparison.
    receipt_path = fixture.output / "new-evidence.json"
    receipt = record(session._read(receipt_path))
    data, plan, _ = benchmark._load(prepared.plan)
    receipt["framesSHA256"] = benchmark._digest(changed(data, plan))
    write(receipt_path, receipt)
    with pytest.raises(ValueError, match="Raw OCR differs"):
        benchmark.phase(prepared.plan, "compare")
    comparison = record(session._read(fixture.output / "comparison.json"))
    assert comparison["differentFrameIndices"] == [8]
    assert comparison["qualityApproved"] is False


@pytest.mark.parametrize(
    "metric",
    ["wallSeconds", "cpuSeconds", "cpuScope", "gpuObservation", "physicalObservation", "extra"],
)
def test_invalid_metrics_rejected(metric: str) -> None:
    row: dict[str, object] = {
        "wallSeconds": 1.0,
        "cpuSeconds": 2.0,
        "cpuScope": "collector and reaped children including physical observer ps",
        "gpuObservation": {"supported": False, "seconds": None},
        "resumedCollection": False,
        "fullIntervalTiming": True,
        "physicalObservation": {
            "available": True,
            "rootPID": 1,
            "rootBirth": "1:2",
            "samples": 2,
            "intervalSeconds": 0.1,
            "maximumObservedPhysicalBytes": 4,
            "maximumObservedRSSBytes": 2,
            "peakGuarantee": False,
            "includesCollectorAndObserver": True,
        },
    }
    row[metric] = -1 if metric in {"wallSeconds", "cpuSeconds"} else "bad"
    with pytest.raises((ValueError, ReplayError)):
        benchmark._metrics(row)


@pytest.mark.parametrize("failure", ["invalid", "missing", "alive"])
def test_physical_observation_fails_closed(monkeypatch: pytest.MonkeyPatch, failure: str) -> None:
    monkeypatch.setattr(benchmark, "SystemHost", lambda: Mock(birth=lambda pid: "1:2"))
    observer = benchmark.PhysicalObservation()
    if failure == "invalid":
        monkeypatch.setattr(
            benchmark, "sample_family", lambda *args: {"physicalBytes": None, "rssBytes": 1}
        )
        observer._observe()
    observer.thread = Mock(is_alive=lambda: failure == "alive")
    with pytest.raises((ValueError, OSError)):
        observer.finish()


def test_physical_observation_records_confined_permission_denial(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(benchmark, "SystemHost", lambda: Mock(birth=lambda pid: "1:2"))
    monkeypatch.setattr(benchmark, "sample_family", Mock(side_effect=PermissionError("ps denied")))
    observer = benchmark.PhysicalObservation()
    observer.thread.start()
    assert observer.finish() == {
        "available": False,
        "reason": "process-family observation denied by confined execution",
        "peakGuarantee": False,
    }


def test_cli_and_unknown_phase(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    fixture = fixture_case(tmp_path, monkeypatch)
    with pytest.raises(SystemExit):
        benchmark.main(["prepare", str(fixture.config)])
    assert benchmark.main(["prepare", str(fixture.config), str(fixture.output)]) == 0
    plan = fixture.output / "plan.json"
    with pytest.raises(SystemExit):
        benchmark.main(["old", str(plan), str(fixture.output)])
    with pytest.raises(ValueError):
        benchmark.phase(plan, "unknown")


@pytest.mark.parametrize("problem", ["good", "schema", "order"])
def test_retained_filesystem_events_stay_distinct_from_process_timers(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, problem: str
) -> None:
    fixture = fixture_case(tmp_path, monkeypatch)
    begin = write(tmp_path / "retained-begin.log", {"event": "begin"})
    end = write(tmp_path / "retained-end.log", {"event": "end"})
    events: dict[str, object] = {"begin": pin(begin), "end": pin(end)}
    if problem == "schema":
        events["extra"] = 1
    elif problem == "order":
        events = {"begin": pin(end), "end": pin(begin)}
    config = record(session._read(fixture.config))
    intervals = config["intervals"]
    assert isinstance(intervals, list)
    intervals[0] = {**record(intervals[0]), "baselineFilesystemEvents": events}
    write(fixture.config, config)
    if problem != "good":
        with pytest.raises(ValueError):
            benchmark.prepare(fixture.config, fixture.output)
        return
    prepared = benchmark.prepare(fixture.config, fixture.output)
    benchmark.phase(prepared.plan, "old")
    receipt = record(session._read(fixture.output / "old-evidence.json"))
    assert receipt["metrics"] is None and receipt["baselineProcessTimersAvailable"] is False
    timings = receipt["baselineFilesystemEvents"]
    assert isinstance(timings, list)
    assert record(timings[0])["processTimer"] is False


def test_changed_sealed_prefix_is_rejected_without_collecting_missing_samples(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    fixture = fixture_case(tmp_path, monkeypatch, count=80, interval=80)
    prepared = benchmark.prepare(fixture.config, fixture.output)
    original = session._attempt

    def fail(plan: session.SessionPlan, directory: Path, start: int, pins: dict[str, str]) -> int:
        if start:
            raise ValueError("worker failed")
        return original(plan, directory, start, pins)

    monkeypatch.setattr(session, "_attempt", fail)
    with pytest.raises(ValueError):
        benchmark.phase(prepared.plan, "new")
    raw = fixture.output / "new/attempt-000000/native/block-0.json"
    raw.write_bytes(b"[]")
    monkeypatch.setattr(session, "_attempt", original)
    before = fixture.run.call_count
    with pytest.raises(ValueError):
        benchmark.phase(prepared.plan, "new")
    assert fixture.run.call_count == before
    assert not (fixture.output / "new-evidence.json").exists()


def test_source_change_after_actual_collection_cannot_publish_receipt(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    fixture = fixture_case(tmp_path, monkeypatch)
    prepared = benchmark.prepare(fixture.config, fixture.output)
    original = session._attempt

    def change(plan: session.SessionPlan, directory: Path, start: int, pins: dict[str, str]) -> int:
        count = original(plan, directory, start, pins)
        fixture.source.write_bytes(b"source changed after completed native frames")
        return count

    monkeypatch.setattr(session, "_attempt", change)
    with pytest.raises(ValueError):
        benchmark.phase(prepared.plan, "new")
    assert not (fixture.output / "new-evidence.json").exists()


def test_input_identity_change_and_changed_evidence_are_preserved(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    fixture = fixture_case(tmp_path, monkeypatch)
    prepared = benchmark.prepare(fixture.config, fixture.output)
    original = benchmark._guard
    calls = 0

    def changed(
        pins: dict[str, str],
        plan: session.SessionPlan,
        decoder_root: session.SourceRoot | None = None,
    ) -> dict[str, dict[str, int]]:
        nonlocal calls
        calls += 1
        identities = original(pins, plan, decoder_root)
        if calls > 1:
            identities[str(plan.source)]["inode"] += 1
        return identities

    monkeypatch.setattr(benchmark, "_guard", changed)
    with pytest.raises(ValueError, match="Input changed"):
        benchmark.phase(prepared.plan, "old")
    monkeypatch.setattr(benchmark, "_guard", original)
    benchmark.phase(prepared.plan, "old")
    path = fixture.output / "old-evidence.json"
    row = record(session._read(path))
    row["framesSHA256"] = "0" * 64
    write(path, row)
    before = path.read_bytes()
    with pytest.raises(ValueError, match="Completed evidence differs"):
        benchmark.phase(prepared.plan, "old", verify=True)
    assert path.read_bytes() == before


def test_explicit_decoder_root_metadata_binding_reuses_genuine_baseline(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    fixture = fixture_case(tmp_path, monkeypatch)
    config = record(session._read(fixture.config))
    config["allowedDecoderRoot"] = str(tmp_path)
    write(fixture.config, config)
    prepared = benchmark.prepare(fixture.config, fixture.output)
    benchmark.phase(prepared.plan, "old")
    benchmark.phase(prepared.plan, "old", verify=True)
    assert not fixture.old_calls


@pytest.mark.parametrize("problem", ["none", "changed", "escape"])
def test_external_decoder_identity_digest_and_root_scope(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, problem: str
) -> None:
    fixture = fixture_case(tmp_path, monkeypatch)
    root = session._source_root(str(tmp_path))
    plan = replace(session.load_plan(fixture.session_plan), source_root=root)
    pins = {
        str(plan.source): file_digest(plan.source),
        str(fixture.old_native): file_digest(fixture.old_native),
    }
    # All fixtures remain in the actual checkout. A nested logical checkout
    # exercises external-root validation without creating OS scratch.
    nested = tmp_path / "nested"
    nested.mkdir()
    monkeypatch.setattr(Path, "cwd", classmethod(lambda cls: nested))
    if problem == "changed":
        fixture.old_native.write_bytes(b"changed decoder")
    if problem == "escape":
        with pytest.raises(ValueError):
            benchmark._guard(pins, plan, session._source_root(str(nested)))
    elif problem == "changed":
        with pytest.raises(ValueError):
            benchmark._guard(pins, plan, root)
    else:
        assert str(fixture.old_native) in benchmark._guard(pins, plan, root)
