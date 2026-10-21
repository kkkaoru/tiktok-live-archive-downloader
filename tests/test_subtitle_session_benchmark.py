"""Actual shared collector APIs with synthetic native JSON; never OCR a recording."""

import json
import subprocess
from dataclasses import dataclass
from fractions import Fraction
from io import BufferedWriter
from pathlib import Path
from unittest.mock import Mock

import pytest

from replay import code_dependency_inventory
from replay import subtitle_cleanup_workflow as cleanup
from replay import subtitle_observations as legacy
from replay import subtitle_session_benchmark as benchmark
from replay import subtitle_session_observations as session
from replay.code_dependency_inventory import Source
from replay.recordings import record
from replay.stages import _fingerprint, file_digest, run_stage
from replay.workflow import load_stages, stage_graph


def write(path: Path, value: object) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, sort_keys=True), encoding="utf-8")
    return path


def pin(path: Path) -> dict[str, str]:
    return {"path": str(path), "sha256": file_digest(path)}


def frames(points: tuple[Fraction, ...]) -> list[dict[str, object]]:
    return [
        {
            "requestedTimeSeconds": float(point),
            "actualTimeSeconds": float(point),
            "width": 64,
            "height": 96,
            "lines": [
                {
                    # Captions change wording: a fixed label's misreads would be refused as a band.
                    "text": "".join(
                        "春夏秋冬雨雲花風月空"[(round(float(point) * 90) + shift) % 10]
                        for shift in (0, 3, 7)
                    ),
                    "confidence": 0.9,
                    "boundsClipped": False,
                    "region": {"x": 1, "y": 60, "width": 50, "height": 16},
                },
                {
                    "text": "次",
                    "confidence": 0.8,
                    "boundsClipped": True,
                    "region": {"x": 2, "y": 80, "width": 40, "height": 16},
                },
            ],
        }
        for point in points
    ]


def native(
    command: list[str],
    *,
    stdin: int,
    stdout: BufferedWriter,
    stderr: BufferedWriter,
    timeout: int,
    check: bool,
    env: dict[str, str],
) -> subprocess.CompletedProcess[bytes]:
    request = record(session._read(Path(command[-1])))
    directory = Path(str(request["checkpointDirectory"]))
    points = session._times(request["samples"])
    assert timeout == 90 and stdin == subprocess.DEVNULL and check is False
    assert Path(env["TMPDIR"]).is_relative_to(Path.cwd() / "edits/work")
    blocks = []
    for start in range(0, len(points), 8):
        selected = points[start : start + 8]
        raw = write(directory / f"block-{start // 8}.json", frames(selected))
        block = {
            "firstSample": start,
            "requested": session._clocks(selected),
            "actual": session._clocks(selected),
            "measurementsPath": str(raw),
        }
        write(directory / f"block-{start // 8}-clocks.json", block)
        blocks.append(block)
    report = {
        "protocolVersion": session.PROTOCOL,
        "sourceModified": False,
        "qualityApproved": False,
        "sourceHashVerified": False,
        "sourceIdentity": session._identity(Path(str(request["path"]))),
        "frames": len(points),
        "blocks": blocks,
        "timing": {"elapsedSeconds": 0.05},
    }
    write(directory / "complete.json", report)
    stdout.write(json.dumps(report).encode())
    return subprocess.CompletedProcess(command, 0)


@dataclass(frozen=True)
class Case:
    root: Path
    config: Path
    session_plan: Path
    clock: Path
    output: Path
    source: Path
    old_native: Path
    new_native: Path
    run: Mock
    old_calls: list[tuple[float, ...]]

    def prepare(self) -> benchmark.Prepared:
        return benchmark.prepare(self.config, self.output)

    def update(self, field: str, value: object) -> None:
        data = record(benchmark._read(self.config))
        data[field] = value
        write(self.config, data)


def case(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, *, count: int = 64) -> Case:
    source, old, new = tmp_path / "source.mp4", tmp_path / "old-native", tmp_path / "new-native"
    source.write_bytes(b"tiny synthetic source")
    old.write_bytes(b"old explicit native")
    new.write_bytes(b"new explicit native")
    old.chmod(0o700)
    new.chmod(0o700)
    points = tuple(Fraction(index * 3 + index % 2, 90) for index in range(count))
    clock = write(
        tmp_path / "clock.json",
        {
            "pts": [point.numerator * (90 // point.denominator) for point in points],
            "timeBase": "1/90",
            "lastDuration": 3,
        },
    )
    plan = write(
        tmp_path / "session.json",
        {
            "version": 1,
            "protocol": session.PROTOCOL,
            "source": str(source),
            "sourceSHA256": file_digest(source),
            "nativeExecutable": str(new),
            "nativeSHA256": file_digest(new),
            "samples": session._clocks(points),
            "width": 64,
            "height": 96,
        },
    )
    config = write(
        tmp_path / "config.json",
        {
            "version": 1,
            "sessionPlan": pin(plan),
            "clock": pin(clock),
            "oldNative": pin(old),
            "retainedBaseline": None,
        },
    )
    output = tmp_path / "benchmark"
    output.mkdir()
    calls: list[tuple[float, ...]] = []

    def batch(
        source: Path,
        times: tuple[float, ...],
        output: Path,
        result: Path,
        *,
        native_executable: Path | None,
        head_tracking: bool = False,
    ) -> None:
        assert native_executable == old and not head_tracking
        calls.append(times)
        write(result, frames(tuple(Fraction(value) for value in times)))

    monkeypatch.setattr(legacy, "_collect_batch", batch)
    run = Mock(side_effect=native)
    monkeypatch.setattr(subprocess, "run", run)
    code = Path(benchmark.__file__).resolve()
    original_capture = code_dependency_inventory.capture

    def capture(modules: tuple[str, ...]) -> tuple[Source, ...]:
        if modules == ("subtitle_session_benchmark",):
            return (Source(code, file_digest(code), (0, 0, 0, 0, 0)),)
        return original_capture(modules)

    monkeypatch.setattr(code_dependency_inventory, "capture", capture)
    return Case(tmp_path, config, plan, clock, output, source, old, new, run, calls)


@pytest.mark.parametrize("count", [64, 17])
def test_normal_serial_workflow_actual_shared_collectors_and_readonly_reentry(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, count: int
) -> None:
    fixture = case(tmp_path, monkeypatch, count=count)
    prepared = fixture.prepare()
    stages = load_stages(prepared.workflow)
    assert stage_graph(stages) == {"old": (), "new": ("old",), "compare": ("old", "new")}
    before = fixture.config.read_bytes()

    def execute(command: tuple[str, ...], log: Path, timeout: int) -> None:
        assert timeout == 3600
        benchmark.main([command[4], command[5]])
        log.write_bytes(b"ordinary synthetic Stage\n")

    for stage in stages:
        assert run_stage(stage, state_directory=prepared.state, execute=execute) == "completed"
        assert run_stage(stage, state_directory=prepared.state, execute=execute) == "reused"
    assert len(fixture.old_calls) == (count + 7) // 8 and fixture.run.call_count == 1
    result = record(benchmark._read(fixture.output / "comparison.json"))
    assert result["equalMeasurements"] is True and result["nativeBinaryIdentical"] is False
    assert result["tolerance"] is None and result["qualityApproved"] is False
    assert fixture.config.read_bytes() == before


def test_explicit_parallel_plan_reaches_original_collector_and_verifier(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    fixture = case(tmp_path, monkeypatch, count=64)
    plan = record(session._read(fixture.session_plan))
    plan["workers"] = 2
    write(fixture.session_plan, plan)
    fixture.update("sessionPlan", pin(fixture.session_plan))
    prepared = fixture.prepare()
    benchmark.phase(prepared.plan, "old")
    benchmark.phase(prepared.plan, "new")
    benchmark.phase(prepared.plan, "compare")
    request = record(session._read(Path(fixture.run.call_args.args[0][-1])))
    assert request["workers"] == 2
    benchmark.phase(prepared.plan, "new", verify=True)
    assert fixture.run.call_count == 1
    comparison = record(benchmark._read(fixture.output / "comparison.json"))
    assert comparison["equalMeasurements"] is True
    assert comparison["qualityApproved"] is False


def retained(fixture: Case) -> None:
    root = fixture.root / "original"
    profile = root / "profile"
    original_clock = write(
        root / "clock.json", {"pts": list(range(80)), "timeBase": "1/90", "lastDuration": 1}
    )
    # Selected new samples are a contiguous subset of this separate original clock.
    plan = record(benchmark._read(fixture.session_plan))
    plan["samples"] = session._clocks(tuple(Fraction(i, 90) for i in range(64)))
    write(fixture.session_plan, plan)
    fixture.update("sessionPlan", pin(fixture.session_plan))
    fixture.update("clock", pin(original_clock))
    batch = write(profile / "0000.result.json", frames(tuple(Fraction(i, 90) for i in range(8))))
    second = write(
        profile / "0001.result.json", frames(tuple(Fraction(i, 90) for i in range(8, 16)))
    )
    producer = write(
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
            "results": {batch.name: file_digest(batch), second.name: file_digest(second)},
        },
    )
    model = root / "model"
    model.write_bytes(b"synthetic pinned model")
    classification = write(root / "classification.json", {"classification": "synthetic"})
    settings = write(
        root / "settings.json",
        {
            "version": 1,
            "source": pin(fixture.source),
            "native": pin(fixture.old_native),
            "decoder": pin(fixture.old_native),
            "model": pin(model),
            "sourceClock": pin(original_clock),
            "classificationEvidence": pin(classification),
            "profileInputs": pin(producer),
            "profileComplete": pin(complete),
            "chunkFrames": 80,
        },
    )
    cleanup.prepare(settings, root / "cleanup", workers=1)
    directory = root / "cleanup/chunk-00000"
    legacy.collect_observations(
        fixture.source,
        directory / "times.json",
        directory / "ocr",
        native_executable=fixture.old_native,
    )
    stage = load_stages(root / "cleanup/worker-00.json")[0]
    state = root / "cleanup/state-00"
    receipt = write(
        state / (stage.name + ".json"),
        {
            "status": "complete",
            "fingerprint": _fingerprint(stage),
            "outputs": {str(path): file_digest(path) for path in stage.outputs},
        },
    )
    fixture.update(
        "retainedBaseline",
        {
            "chunk": pin(directory / "chunk.json"),
            "times": pin(directory / "times.json"),
            "workflow": pin(root / "cleanup/worker-00.json"),
            "receipt": pin(receipt),
            "stageName": stage.name,
        },
    )
    fixture.old_calls.clear()


def test_genuine_retained_original_stage_first64_has_no_old_inference(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    fixture = case(tmp_path, monkeypatch)
    retained(fixture)
    prepared = fixture.prepare()
    old = load_stages(prepared.workflow)[0]
    assert old.outputs == (fixture.output / "old-evidence.json",)
    for mode in ("old", "new", "compare"):
        benchmark.phase(prepared.plan, mode)
        benchmark.phase(prepared.plan, mode, verify=True)
    assert fixture.old_calls == [] and fixture.run.call_count == 1
    assert record(benchmark._read(fixture.output / "old-evidence.json"))["retainedBaseline"] is True


@pytest.mark.parametrize(
    "field,value",
    [
        ("version", True),
        ("unknown", 1),
        ("oldNative", {}),
        ("retainedBaseline", {}),
        ("sessionPlan", {"path": "relative", "sha256": "a" * 64}),
        ("clock", {"path": "relative", "sha256": "a" * 64}),
    ],
)
def test_invalid_config_before_native(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, field: str, value: object
) -> None:
    fixture = case(tmp_path, monkeypatch)
    fixture.update(field, value)
    with pytest.raises((ValueError, KeyError)):
        fixture.prepare()
    assert fixture.run.call_count == 0 and fixture.old_calls == []


@pytest.mark.parametrize(
    "problem", ["oversize", "skipped", "absent", "bad-hash", "occupied", "tool-alias"]
)
def test_input_scope_and_physical_clock(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, problem: str
) -> None:
    fixture = case(tmp_path, monkeypatch, count=65 if problem == "oversize" else 17)
    if problem in {"skipped", "absent"}:
        data = record(benchmark._read(fixture.session_plan))
        points = session._times(data["samples"])
        selected = (points[0], *points[2:]) if problem == "skipped" else (Fraction(1000),)
        data["samples"] = session._clocks(selected)
        write(fixture.session_plan, data)
        fixture.update("sessionPlan", pin(fixture.session_plan))
    elif problem == "bad-hash":
        fixture.update("clock", {"path": str(fixture.clock), "sha256": "a" * 64})
    elif problem == "occupied":
        write(fixture.output / "existing.json", {})
    elif problem == "tool-alias":
        fixture.update("oldNative", {"path": str(fixture.new_native), "sha256": "a" * 64})
    with pytest.raises(ValueError):
        fixture.prepare()


@pytest.mark.parametrize("field", ["text", "confidence", "region", "boundsClipped", "order"])
def test_semantic_difference_is_retained_and_never_approved(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, field: str
) -> None:
    fixture = case(tmp_path, monkeypatch, count=17)
    prepared = fixture.prepare()
    original_native = native

    def different(
        command: list[str],
        *,
        stdin: int,
        stdout: BufferedWriter,
        stderr: BufferedWriter,
        timeout: int,
        check: bool,
        env: dict[str, str],
    ) -> subprocess.CompletedProcess[bytes]:
        result = original_native(
            command,
            stdin=stdin,
            stdout=stdout,
            stderr=stderr,
            timeout=timeout,
            check=check,
            env=env,
        )
        request = record(session._read(Path(command[-1])))
        raw = Path(str(request["checkpointDirectory"])) / "block-0.json"
        rows = benchmark._read(raw)
        assert isinstance(rows, list)
        first = record(rows[0])
        lines = first["lines"]
        assert isinstance(lines, list)
        if field == "order":
            lines.reverse()
        else:
            line = record(lines[0])
            line[field] = {
                "text": "別",
                "confidence": 0.89,
                "boundsClipped": True,
                "region": {"x": 2, "y": 60, "width": 50, "height": 12},
            }[field]
        write(raw, rows)
        return result

    fixture.run.side_effect = different
    benchmark.phase(prepared.plan, "old")
    benchmark.phase(prepared.plan, "new")
    with pytest.raises(ValueError, match="measurements differ"):
        benchmark.phase(prepared.plan, "compare")
    result = record(benchmark._read(fixture.output / "comparison.json"))
    assert result["differentFrameIndices"] == [0] and result["equalMeasurements"] is False
    with pytest.raises(ValueError, match="measurements differ"):
        benchmark.phase(prepared.plan, "compare", verify=True)


@pytest.mark.parametrize(
    "problem",
    ["source", "native", "clock", "missing-input", "extra-input", "destination", "offset", "times"],
)
def test_current_pins_and_prepared_plan_tamper_fail_closed(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, problem: str
) -> None:
    fixture = case(tmp_path, monkeypatch)
    prepared = fixture.prepare()
    data = record(benchmark._read(prepared.plan))
    if problem in {"source", "native", "clock"}:
        target = {"source": fixture.source, "native": fixture.new_native, "clock": fixture.clock}[
            problem
        ]
        target.write_bytes(b"changed")
    elif problem in {"missing-input", "extra-input"}:
        inputs = record(data["inputs"])
        if problem == "missing-input":
            del inputs[str(fixture.source)]
        else:
            inputs[str(fixture.output / "invented")] = "a" * 64
        write(prepared.plan, data)
    elif problem == "destination":
        data["newOutput"] = str(fixture.root / "other")
        write(prepared.plan, data)
    elif problem == "offset":
        data["start"] = 2
        write(prepared.plan, data)
    else:
        write(fixture.output / "times.json", [0, 1])
    with pytest.raises(ValueError):
        benchmark.phase(prepared.plan, "old")
    assert fixture.old_calls == [] and fixture.run.call_count == 0


@pytest.mark.parametrize("problem", ["unknown", "clipping", "canvas", "actual", "count"])
def test_raw_schema_clock_clipping_canvas_are_not_normalized_away(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, problem: str
) -> None:
    fixture = case(tmp_path, monkeypatch, count=1)
    plan = session.load_plan(fixture.session_plan)
    values = frames(plan.samples)
    if problem == "unknown":
        values[0]["extra"] = 1
    elif problem == "clipping":
        lines = values[0]["lines"]
        assert isinstance(lines, list)
        del record(lines[0])["boundsClipped"]
    elif problem == "count":
        values = []
    else:
        values[0]["width" if problem == "canvas" else "actualTimeSeconds"] = 2
    with pytest.raises(ValueError):
        benchmark._semantic(values, plan)


def test_missing_completion_verify_and_changed_evidence_preserved(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    fixture = case(tmp_path, monkeypatch)
    prepared = fixture.prepare()
    with pytest.raises(FileNotFoundError):
        benchmark.phase(prepared.plan, "old", verify=True)
    assert fixture.old_calls == []
    benchmark.phase(prepared.plan, "old")
    evidence = fixture.output / "old-evidence.json"
    data = record(benchmark._read(evidence))
    data["elapsedSeconds"] = -1
    write(evidence, data)
    with pytest.raises(ValueError, match="elapsed"):
        benchmark.phase(prepared.plan, "old", verify=True)
    assert record(benchmark._read(evidence))["elapsedSeconds"] == -1


def test_actual_source_mutation_during_producer_blocks_evidence(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    fixture = case(tmp_path, monkeypatch)
    prepared = fixture.prepare()
    original = legacy._collect_batch

    def batch(
        source: Path,
        times: tuple[float, ...],
        output: Path,
        result: Path,
        *,
        native_executable: Path | None,
        head_tracking: bool = False,
    ) -> None:
        original(
            source,
            times,
            output,
            result,
            native_executable=native_executable,
            head_tracking=head_tracking,
        )
        source.write_bytes(b"mutation during actual synthetic producer")

    monkeypatch.setattr(legacy, "_collect_batch", batch)
    with pytest.raises(ValueError, match="inputs changed"):
        benchmark.phase(prepared.plan, "old")
    assert not (fixture.output / "old-evidence.json").exists()
    assert (fixture.output / "old/0000.result.json").exists()


def test_cli_and_path_scope(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    fixture = case(tmp_path, monkeypatch, count=1)
    assert benchmark.main(["prepare", str(fixture.config), str(fixture.output)]) == 0
    for argv in (
        ["prepare", str(fixture.config)],
        ["old", str(fixture.config), str(fixture.output)],
    ):
        with pytest.raises(SystemExit):
            benchmark.main(argv)
    with pytest.raises(ValueError, match="Unknown"):
        benchmark.phase(fixture.output / "plan.json", "invalid")
    for path in ("relative", "/tmp/no", str(Path.cwd() / "private/no")):
        with pytest.raises(ValueError):
            benchmark._path(path)
    external = tmp_path / "link"
    external.symlink_to(fixture.config)
    with pytest.raises(ValueError):
        benchmark._path(str(external))


@pytest.mark.parametrize(
    "problem",
    ["extra", "stage-type", "wrong-stage", "source", "times", "gap", "raw-pin", "native-conflict"],
)
def test_retained_metadata_contract_rejects_unproved_baseline(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, problem: str
) -> None:
    fixture = case(tmp_path, monkeypatch)
    retained(fixture)
    data = record(benchmark._read(fixture.config))
    baseline = record(data["retainedBaseline"])
    chunk = Path(str(record(baseline["chunk"])["path"]))
    if problem == "extra":
        baseline["unknown"] = 1
    elif problem == "stage-type":
        baseline["stageName"] = 1
    elif problem == "wrong-stage":
        baseline["stageName"] = "missing-original"
    elif problem == "source":
        raw = record(benchmark._read(chunk))
        settings = Path(str(record(raw["settings"])["path"]))
        value = record(benchmark._read(settings))
        value["source"] = pin(fixture.new_native)
        write(settings, value)
        raw["settings"] = pin(settings)
        write(chunk, raw)
        baseline["chunk"] = pin(chunk)
    elif problem == "times":
        baseline["times"] = pin(fixture.clock)
    elif problem == "native-conflict":
        data["oldNative"] = {"path": str(fixture.old_native), "sha256": "a" * 64}
    else:
        complete_path = chunk.parent / "ocr/complete.json"
        complete = record(benchmark._read(complete_path))
        results = record(complete["results"])
        if problem == "gap":
            del results["0000.result.json"]
        else:
            results["0000.result.json"] = "a" * 64
        write(complete_path, complete)
    write(fixture.config, data)
    with pytest.raises((ValueError, TypeError)):
        fixture.prepare()
    assert fixture.old_calls == [] and fixture.run.call_count == 0


@pytest.mark.parametrize("problem", ["fingerprint", "command", "provenance", "actual"])
def test_retained_runtime_requires_genuine_original_domain_and_complete_stage(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, problem: str
) -> None:
    fixture = case(tmp_path, monkeypatch)
    retained(fixture)
    data = record(benchmark._read(fixture.config))
    baseline = record(data["retainedBaseline"])
    receipt_path = Path(str(record(baseline["receipt"])["path"]))
    chunk = Path(str(record(baseline["chunk"])["path"]))
    if problem == "fingerprint":
        receipt = record(benchmark._read(receipt_path))
        receipt["fingerprint"] = "a" * 64
        write(receipt_path, receipt)
        baseline["receipt"] = pin(receipt_path)
    elif problem == "command":
        workflow = Path(str(record(baseline["workflow"])["path"]))
        graph = record(benchmark._read(workflow))
        stages = graph["stages"]
        assert isinstance(stages, list)
        record(stages[0])["command"] = ["false"]
        write(workflow, graph)
        baseline["workflow"] = pin(workflow)
    elif problem == "provenance":
        producer = chunk.parent / "ocr/inputs.json"
        value = record(benchmark._read(producer))
        value["collectorSHA256"] = "a" * 64
        write(producer, value)
        stage = load_stages(Path(str(record(baseline["workflow"])["path"])))[0]
        write(
            receipt_path,
            {
                "status": "complete",
                "fingerprint": _fingerprint(stage),
                "outputs": {str(path): file_digest(path) for path in stage.outputs},
            },
        )
        baseline["receipt"] = pin(receipt_path)
    else:
        path = chunk.parent / "ocr/0000.result.json"
        rows = benchmark._read(path)
        assert isinstance(rows, list)
        record(rows[0])["actualTimeSeconds"] = 1
        write(path, rows)
        complete_path = chunk.parent / "ocr/complete.json"
        value = record(benchmark._read(complete_path))
        record(value["results"])[path.name] = file_digest(path)
        write(complete_path, value)
    write(fixture.config, data)
    prepared = fixture.prepare()
    with pytest.raises(ValueError):
        benchmark.phase(prepared.plan, "old")
    assert fixture.old_calls == [] and fixture.run.call_count == 0
    assert not (fixture.output / "old-evidence.json").exists()


def test_readonly_verifier_does_not_create_missing_wrapper_receipt_and_preserves_changed_evidence(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    fixture = case(tmp_path, monkeypatch)
    prepared = fixture.prepare()
    legacy.collect_observations(
        fixture.source,
        fixture.output / "times.json",
        fixture.output / "old",
        native_executable=fixture.old_native,
    )
    count = len(fixture.old_calls)
    with pytest.raises(ValueError, match="incomplete"):
        benchmark.phase(prepared.plan, "old", verify=True)
    assert len(fixture.old_calls) == count and not (fixture.output / "old-evidence.json").exists()
    benchmark.phase(prepared.plan, "old")
    path = fixture.output / "old-evidence.json"
    data = record(benchmark._read(path))
    data["qualityApproved"] = True
    write(path, data)
    with pytest.raises(ValueError, match="evidence differs"):
        benchmark.phase(prepared.plan, "old", verify=True)
    assert record(benchmark._read(path))["qualityApproved"] is True


def test_standard_workflow_preflight_has_actual_dependency_contract(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from replay.workflow import main

    fixture = case(tmp_path, monkeypatch)
    prepared = fixture.prepare()
    assert main([str(prepared.workflow), "--state", str(prepared.state), "--check"]) == 0
    assert fixture.run.call_count == 0 and fixture.old_calls == []


def test_comparer_rejects_false_timing_and_legacy_sampling_approval(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    fixture = case(tmp_path, monkeypatch, count=1)
    prepared = fixture.prepare()
    benchmark.phase(prepared.plan, "old")
    benchmark.phase(prepared.plan, "new")
    receipt = fixture.output / "old-evidence.json"
    value = record(benchmark._read(receipt))
    value["elapsedSeconds"] = -1
    write(receipt, value)
    with pytest.raises(ValueError, match="elapsed"):
        benchmark.phase(prepared.plan, "compare")
    assert not (fixture.output / "comparison.json").exists()
    complete = fixture.output / "old/complete.json"
    value = record(benchmark._read(complete))
    value["samplingExhaustive"] = True
    write(complete, value)
    data, plan, _ = benchmark._load(prepared.plan)
    with pytest.raises(ValueError, match="sampling claim"):
        benchmark._legacy_frames(data, plan)
