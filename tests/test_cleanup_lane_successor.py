"""Offline successor planning and normal Stage integration with tiny synthetic inputs."""

from __future__ import annotations

import fcntl
import json
import os
import shutil
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from threading import Barrier
from typing import NamedTuple

import pytest

from replay import cleanup_lane_successor as successor
from replay import code_generation_compat as compat
from replay import subtitle_candidate
from replay import subtitle_cleanup_workflow as cleanup
from replay import subtitle_observations as observations
from replay.core import ReplayError
from replay.input_hash_cache import InputHashCache, input_hash_context
from replay.sequence_journal import bounded_read, decode, encode
from replay.stages import Stage, StageError, _fingerprint, file_digest, run_stage
from replay.workflow import load_stages, stage_graph


def write(path: Path, value: object) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(encode(value))
    return path


def read(path: Path) -> dict[str, object]:
    return decode(bounded_read(path))


def pin(path: Path) -> dict[str, str]:
    return {"path": str(path), "sha256": file_digest(path)}


def frames(times: tuple[float, ...]) -> list[dict[str, object]]:
    return [
        {
            "requestedTimeSeconds": value,
            "actualTimeSeconds": value,
            "width": 100,
            "height": 100,
            "lines": [
                {
                    "text": "字幕" + "春夏秋冬雨雲花風月空"[int(value) % 10],
                    "confidence": 0.9,
                    "region": {"x": 0, "y": 40, "width": 100, "height": 20},
                }
            ],
        }
        for value in times
    ]


class Usage(NamedTuple):
    total: int
    used: int
    free: int


@dataclass(frozen=True)
class Case:
    root: Path
    settings: Path
    original: Path
    request: Path
    handoff: Path
    output: Path
    states: tuple[Path, ...]

    def directory(self, index: int) -> Path:
        return self.original.parent / f"chunk-{index:05d}"

    def collect(self, index: int) -> None:
        settings = read(self.settings)
        observations.collect_observations(
            Path(str(successor._object(settings["source"])["path"])),
            self.directory(index) / "times.json",
            self.directory(index) / "ocr",
            native_executable=Path(str(successor._object(settings["native"])["path"])),
        )

    def complete(self, index: int) -> None:
        self.collect(index)
        chunk = self.directory(index) / "chunk.json"
        cleanup.build_candidate(chunk)
        rendered(self.directory(index) / "candidate-config.json")
        for lane in range(len(self.states)):
            for stage in load_stages(self.original.parent / f"worker-{lane:02d}.json"):
                if stage.name.startswith(self.directory(index).name + "-"):
                    complete_receipt(stage, self.states[lane])

    def update_request(self, key: str, value: object) -> None:
        data = read(self.request)
        data[key] = value
        write(self.request, data)


def complete_receipt(stage: Stage, state: Path) -> None:
    write(
        state / (stage.name + ".json"),
        {
            "status": "complete",
            "fingerprint": _fingerprint(stage),
            "outputs": {str(path): file_digest(path) for path in stage.outputs},
        },
    )


def rendered(config: Path) -> Path:
    data = read(config)
    directory = Path(str(data["outputDirectory"]))
    directory.mkdir()
    media = directory / "candidate.mp4"
    media.write_bytes(b"synthetic repaired candidate, never delivery approved")
    clock = successor._object(data["clock"])
    inputs = {
        str(config): file_digest(config),
        str(data["source"]): str(data["sourceSHA256"]),
        str(data["nativeExecutable"]): str(data["nativeSHA256"]),
    }
    write(
        directory / "candidate.json",
        {
            "inputs": inputs,
            "videoSHA256": file_digest(media),
            "render": {"frames": len(successor._list(clock["pts"]))},
            "sourceStart": str(successor._list(clock["pts"])[0]),
        },
    )
    return media


def case(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, *, chunks: int = 5, budget: int = 10
) -> Case:
    root = tmp_path / "edits/work/synthetic"
    root.mkdir(parents=True)
    source, native, model = root / "source.mp4", root / "native", root / "model"
    for path in (source, native, model):
        path.write_bytes(path.name.encode())
    native.chmod(0o700)
    clock = write(
        root / "clock.json",
        {"pts": list(range(chunks * budget)), "timeBase": "1", "lastDuration": 1},
    )
    profile = root / "profile"
    inputs = write(
        profile / "inputs.json",
        {
            "sourceSHA256": file_digest(source),
            "nativeSHA256": file_digest(native),
            "nativeExecutable": str(native),
            "measurementTool": "video_text_recognize",
            "fullFrame": True,
        },
    )
    batch = write(profile / "0000.result.json", frames(tuple(float(i) for i in range(8))))
    second = write(profile / "0001.result.json", frames((8.0, 9.0)))
    complete = write(
        profile / "complete.json",
        {
            "observations": 10,
            "results": {batch.name: file_digest(batch), second.name: file_digest(second)},
        },
    )
    review = write(root / "classification.json", {"classification": "synthetic"})
    settings = write(
        root / "settings.json",
        {
            "version": 1,
            "source": pin(source),
            "native": pin(native),
            "decoder": pin(native),
            "model": pin(model),
            "sourceClock": pin(clock),
            "profileInputs": pin(inputs),
            "profileComplete": pin(complete),
            "classificationEvidence": pin(review),
            "chunkFrames": budget,
        },
    )
    original_root = root / "original"
    cleanup.prepare(settings, original_root, workers=2)
    states = tuple(original_root / f"state-{lane:02d}" for lane in range(2))
    for state in states:
        state.mkdir()
    evidence = write(
        root / "owner-terminal.json", {"terminal": True, "scope": "synthetic exact actors"}
    )
    handoff = write(
        root / "handoff.json",
        {
            "version": 1,
            "observedAtNs": time.time_ns(),
            "maximumAgeSeconds": 3600,
            "quiesced": True,
            "oldWorkersDisabled": True,
            "actors": [{"pid": 2147483647, "birth": "retained-original-birth"}],
            "oldStates": [str(path) for path in states],
            "evidence": [pin(evidence)],
        },
    )
    request = write(
        root / "request.json",
        {
            "version": 1,
            "originalPlan": pin(original_root / "plan.json"),
            "oldWorkers": [
                {"workflow": pin(original_root / f"worker-{lane:02d}.json"), "state": str(state)}
                for lane, state in enumerate(states)
            ],
            "handoff": pin(handoff),
            "workers": 8,
        },
    )
    output = root / "successor"
    output.mkdir()

    def fake_batch(
        source: Path,
        times: tuple[float, ...],
        output: Path,
        result: Path,
        *,
        native_executable: Path | None,
        head_tracking: bool = False,
    ) -> None:
        assert native_executable is not None and not head_tracking
        write(result, frames(times))

    monkeypatch.setattr(observations, "_collect_batch", fake_batch)
    monkeypatch.setattr(subtitle_candidate, "render_candidate", rendered)
    monkeypatch.setattr(
        shutil,
        "disk_usage",
        lambda path: Usage(200 * 1024**3, 20 * 1024**3, 180 * 1024**3),
    )
    code = {str(Path(successor.__file__).resolve()): file_digest(Path(successor.__file__))}
    monkeypatch.setattr(successor, "_code", lambda: code)
    return Case(root, settings, original_root / "plan.json", request, handoff, output, states)


@pytest.mark.parametrize("chunks,budget", [(5, 10), (9, 3)])
def test_pending_partition_and_normal_stages(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, chunks: int, budget: int
) -> None:
    fixture = case(tmp_path, monkeypatch, chunks=chunks, budget=budget)
    original_bytes = {
        path: path.read_bytes()
        for path in (fixture.original, fixture.settings, fixture.directory(0) / "chunk.json")
    }
    prepared = successor.prepare(fixture.request, fixture.output)
    assert len(prepared.workflows) == min(8, chunks)
    observed: set[int] = set()
    for workflow in prepared.workflows:
        stages = load_stages(workflow)
        graph = stage_graph(stages)
        for stage in stages:
            assert not stage.adopt_existing
            assert stage.verify_command and stage.name in graph
            observed.add(int(stage.name.split("-")[1]))
    assert observed == set(range(chunks))
    assert all(path.read_bytes() == data for path, data in original_bytes.items())
    assert read(fixture.directory(0) / "chunk-successor-v1.json")["maximumActiveChunks"] == min(
        8, chunks
    )


def test_verified_partial_resumes_only_missing_native_batches(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    fixture = case(tmp_path, monkeypatch, chunks=2, budget=10)
    fixture.collect(0)
    ocr = fixture.directory(0) / "ocr"
    (ocr / "complete.json").unlink()
    (ocr / "0001.result.json").unlink()
    attempt = ocr / "attempt-interrupted"
    attempt.mkdir()
    write(attempt / "request.json", {"path": str(fixture.root / "source.mp4"), "samples": []})
    (attempt / "response.json").write_bytes(b"[interrupted")
    (attempt / "stderr.log").write_bytes(b"")
    kept = {path: path.read_bytes() for path in ocr.rglob("*") if path.is_file()}
    prepared = successor.prepare(fixture.request, fixture.output)
    stage = load_stages(prepared.workflows[0])[0]
    assert stage.outputs == (ocr / "complete.json",)
    assert ocr / "inputs.json" in stage.inputs and ocr / "0000.result.json" in stage.inputs
    assert read(prepared.plan)["chunks"] != []
    calls: list[Path] = []

    def batch(
        source: Path,
        times: tuple[float, ...],
        output: Path,
        result: Path,
        *,
        native_executable: Path | None,
        head_tracking: bool = False,
    ) -> None:
        calls.append(result)
        write(result, frames(times))

    monkeypatch.setattr(observations, "_collect_batch", batch)
    successor.phase(prepared.plan, 0, "ocr")
    successor.phase(prepared.plan, 0, "verify-ocr")
    assert calls == [ocr / "0001.result.json"]
    assert all(path.read_bytes() == data for path, data in kept.items())
    assert (ocr / "complete.json").exists()


def test_complete_old_chunk_has_no_producer_replay_and_full_merge(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    fixture = case(tmp_path, monkeypatch, chunks=2, budget=3)
    fixture.complete(0)
    preserved = {
        path: path.read_bytes() for path in fixture.directory(0).rglob("*") if path.is_file()
    }
    prepared = successor.prepare(fixture.request, fixture.output)

    def execute(command: tuple[str, ...], log: Path, timeout: int) -> None:
        assert command[3] == "replay.cleanup_lane_successor"
        successor.phase(Path(command[5]), int(command[7]), command[4])
        log.write_bytes(b"synthetic normal Stage invocation\n")

    for workflow, state in zip(prepared.workflows, prepared.states, strict=True):
        for stage in load_stages(workflow):
            assert run_stage(stage, state_directory=state, execute=execute) == "completed"
            assert run_stage(stage, state_directory=state, execute=execute) == "reused"
    assert all(path.read_bytes() == data for path, data in preserved.items())
    successor.phase(prepared.plan, 0, "verify-reuse")
    output = fixture.root / "assembly"
    output.mkdir()
    manifest = successor.assemble(prepared.plan, output)
    assert read(manifest)["version"] == 1
    ranges = successor._list(read(output / "index.json")["ranges"])
    assert sum(int(str(successor._object(row)["frames"])) for row in ranges) == 6
    generation = read(output / "successor-generation.json")
    assert generation["maximumActiveChunks"] == 2
    assert generation["qualityApproved"] is False and generation["mediaBytesVerified"] is False


@pytest.mark.parametrize(
    "key,value",
    [
        ("workers", 0),
        ("workers", 9),
        ("workers", True),
        ("version", 2),
        ("unknown", 1),
        ("oldWorkers", []),
        ("oldWorkers", {}),
        ("handoff", {"path": "relative", "sha256": "a" * 64}),
    ],
)
def test_request_invalid(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, key: str, value: object
) -> None:
    fixture = case(tmp_path, monkeypatch)
    fixture.update_request(key, value)
    with pytest.raises(ValueError):
        successor.prepare(fixture.request, fixture.output)


@pytest.mark.parametrize(
    "key,value",
    [
        ("quiesced", False),
        ("oldWorkersDisabled", False),
        ("oldStates", []),
        ("maximumAgeSeconds", 0),
        ("observedAtNs", 0),
        ("actors", []),
        ("actors", [{"pid": True, "birth": "x"}]),
        ("actors", [{"pid": 1, "birth": ""}]),
        ("evidence", []),
    ],
)
def test_handoff_invalid(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, key: str, value: object
) -> None:
    fixture = case(tmp_path, monkeypatch)
    data = read(fixture.handoff)
    data[key] = value
    write(fixture.handoff, data)
    fixture.update_request("handoff", pin(fixture.handoff))
    with pytest.raises(ValueError):
        successor.prepare(fixture.request, fixture.output)


def test_live_actor_and_canonical_locks(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    fixture = case(tmp_path, monkeypatch)
    data = read(fixture.handoff)
    data["actors"] = [{"pid": os.getpid(), "birth": "real-test-process"}]
    write(fixture.handoff, data)
    fixture.update_request("handoff", pin(fixture.handoff))
    with pytest.raises(ValueError, match="still present"):
        successor.prepare(fixture.request, fixture.output)
    data["actors"] = [{"pid": 2147483647, "birth": "original"}]
    write(fixture.handoff, data)
    fixture.update_request("handoff", pin(fixture.handoff))
    for path in (fixture.states[0] / ".workflow.lock", fixture.states[0] / "chunk-00000-ocr.lock"):
        with path.open("w+b") as lock:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            with pytest.raises(ValueError, match="active"):
                successor.prepare(fixture.request, fixture.output)


@pytest.mark.parametrize(
    "problem",
    [
        "gap",
        "corrupt",
        "wrong-clock",
        "source",
        "code",
        "native",
        "unknown",
        "linked",
        "orphan",
        "repair",
        "ocr-complete",
    ],
)
def test_partial_prefix_fail_closed(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, problem: str
) -> None:
    fixture = case(tmp_path, monkeypatch, chunks=2, budget=10)
    fixture.collect(0)
    ocr = fixture.directory(0) / "ocr"
    (ocr / "complete.json").unlink()
    if problem == "gap":
        (ocr / "0000.result.json").unlink()
    elif problem == "corrupt":
        (ocr / "0000.result.json").write_bytes(b"bad json")
    elif problem == "wrong-clock":
        write(ocr / "0000.result.json", frames(tuple(float(i + 1) for i in range(8))))
    elif problem in {"source", "code", "native"}:
        data = read(ocr / "inputs.json")
        data[
            {"source": "sourceSHA256", "code": "collectorSHA256", "native": "nativeSHA256"}[problem]
        ] = "0" * 64
        write(ocr / "inputs.json", data)
    elif problem == "unknown":
        write(ocr / "hidden-writer.json", {})
    elif problem == "linked":
        (ocr / "0000.result.json").unlink()
        (ocr / "0000.result.json").symlink_to(fixture.settings)
    elif problem == "orphan":
        (ocr / "inputs.json").unlink()
    elif problem == "repair":
        (fixture.directory(0) / "repair").mkdir()
    else:
        fixture.collect(0)
    with pytest.raises((ValueError, OSError)):
        successor.prepare(fixture.request, fixture.output)
    assert not (fixture.output / "successor.json").exists()


@pytest.mark.parametrize(
    "field,value",
    [("wholeSourcePlanned", False), ("chunks", 99), ("plannedFrames", 1), ("totalSourceFrames", 1)],
)
def test_omitted_chunk_clock_accounting(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, field: str, value: object
) -> None:
    fixture = case(tmp_path, monkeypatch)
    data = read(fixture.original)
    data[field] = value
    write(fixture.original, data)
    fixture.update_request("originalPlan", pin(fixture.original))
    with pytest.raises(ValueError, match="every physical"):
        successor.prepare(fixture.request, fixture.output)


def test_reacquisition_and_source_model_mutation(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    fixture = case(tmp_path, monkeypatch)
    prepared = successor.prepare(fixture.request, fixture.output)
    lock_path = fixture.states[0] / "chunk-00000-repair.lock"
    with lock_path.open("w+b") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        with pytest.raises(ValueError, match="reacquired"):
            successor.phase(prepared.plan, 0, "ocr")
    source = fixture.root / "source.mp4"
    original = source.read_bytes()
    source.write_bytes(b"changed source")
    with pytest.raises(ValueError, match="Source changed"):
        successor.phase(prepared.plan, 0, "ocr")
    source.write_bytes(original)
    (fixture.root / "model").write_bytes(b"changed model")
    with pytest.raises(ValueError, match="fingerprint changed"):
        successor.phase(prepared.plan, 0, "ocr")


def test_capacity_missing_complete_and_invalid_phase(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    fixture = case(tmp_path, monkeypatch)
    prepared = successor.prepare(fixture.request, fixture.output)
    monkeypatch.setattr(
        shutil,
        "disk_usage",
        lambda path: Usage(200 * 1024**3, 199 * 1024**3, 1 * 1024**3),
    )
    with pytest.raises(ValueError, match="capacity"):
        successor.phase(prepared.plan, 0, "ocr")
    with pytest.raises(ValueError, match="Unknown successor phase"):
        successor.phase(prepared.plan, 0, "unknown")
    with pytest.raises(ValueError, match="Unknown successor chunk"):
        successor.phase(prepared.plan, -1, "ocr")
    with pytest.raises(ValueError, match="completion is absent"):
        successor.phase(prepared.plan, 0, "reuse")
    assembly_output = fixture.root / "assembled"
    assembly_output.mkdir()
    with pytest.raises(FileNotFoundError):
        successor.assemble(prepared.plan, assembly_output)


def test_changed_metadata_and_code_no_reprepare(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    fixture = case(tmp_path, monkeypatch)
    prepared = successor.prepare(fixture.request, fixture.output)
    with pytest.raises(ValueError, match="Occupied"):
        successor.prepare(fixture.request, fixture.output)
    write(fixture.directory(0) / "times.json", [0, 1])
    with pytest.raises(ValueError, match="input changed"):
        successor.phase(prepared.plan, 0, "ocr")
    monkeypatch.setattr(successor, "_code", lambda: {})
    with pytest.raises(ValueError, match="reader code"):
        successor.phase(prepared.plan, 0, "ocr")


def test_public_code_closure_and_cli_errors() -> None:
    sources = successor._code()
    names = {Path(path).name for path in sources}
    assert {
        "cleanup_lane_successor.py",
        "subtitle_cleanup_assembly.py",
        "subtitle_observations.py",
        "subtitle_candidate.py",
        "stages.py",
        "workflow.py",
    } <= names
    for argv in (["prepare", "/nonexistent"], ["ocr", "/nonexistent"]):
        with pytest.raises(SystemExit):
            successor.main(argv)


def test_concurrent_shared_quiescence_probes_and_exclusive_writer(tmp_path: Path) -> None:
    path = tmp_path / "worker.lock"
    path.touch()
    barrier = Barrier(8)

    def probe() -> None:
        with successor._unlocked(path), successor._unlocked(path):
            barrier.wait(timeout=5)

    with ThreadPoolExecutor(max_workers=8) as pool:
        futures = [pool.submit(probe) for _ in range(8)]
        for future in futures:
            future.result(timeout=10)
    with path.open("rb") as writer:
        fcntl.flock(writer, fcntl.LOCK_EX | fcntl.LOCK_NB)
        with pytest.raises(ValueError, match="active"), successor._unlocked(path):
            pytest.fail("An active original writer must remain exclusionary")


def next_generation(fixture: Case, prepared: successor.Prepared) -> tuple[Path, Path]:
    for state in prepared.states:
        state.mkdir()
    handoff_data = read(fixture.handoff)
    handoff_data["oldStates"] = [str(state) for state in (*fixture.states, *prepared.states)]
    handoff = write(fixture.root / "handoff-next.json", handoff_data)
    before = write(
        fixture.root / "before-fix.json", {"previousReader": read(prepared.plan)["readerCode"]}
    )
    request_data = read(fixture.request)
    request_data["handoff"] = pin(handoff)
    request_data["previousGeneration"] = {
        "plan": pin(prepared.plan),
        "workers": [
            {"workflow": pin(workflow), "state": str(state)}
            for workflow, state in zip(prepared.workflows, prepared.states, strict=True)
        ],
        "evidence": [pin(before)],
    }
    request = write(fixture.root / "request-next.json", request_data)
    output = fixture.root / "next-generation"
    output.mkdir()
    return request, output


def test_explicit_previous_generation_preserves_configs_and_resumes_prefix(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    fixture = case(tmp_path, monkeypatch, chunks=2, budget=10)
    fixture.complete(0)
    first = successor.prepare(fixture.request, fixture.output)
    fixture.collect(1)
    ocr = fixture.directory(1) / "ocr"
    (ocr / "complete.json").unlink()
    (ocr / "0001.result.json").unlink()
    kept = {path: path.read_bytes() for path in fixture.output.iterdir() if path.is_file()}
    config = fixture.directory(1) / "chunk-successor-v1.json"
    config_bytes = config.read_bytes()
    request, output = next_generation(fixture, first)
    second = successor.prepare(request, output)
    assert config.read_bytes() == config_bytes
    assert all(path.read_bytes() == data for path, data in kept.items())
    assert (
        successor._object(successor._list(read(second.plan)["chunks"])[1])["mode"] == "resume-ocr"
    )
    successor.phase(second.plan, 1, "ocr")
    successor.phase(second.plan, 1, "verify-ocr")


@pytest.mark.parametrize(
    "problem",
    [
        "config",
        "missing",
        "clock",
        "workers",
        "original",
        "evidence",
        "gap",
        "duplicate",
        "request",
    ],
)
def test_previous_generation_refuses_unverified_occupied_or_changed_inputs(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, problem: str
) -> None:
    fixture = case(tmp_path, monkeypatch, chunks=2, budget=10)
    first = successor.prepare(fixture.request, fixture.output)
    request, output = next_generation(fixture, first)
    data = read(request)
    prior = successor._object(data["previousGeneration"])
    prior_plan = read(first.plan)
    if problem == "config":
        path = fixture.directory(0) / "chunk-successor-v1.json"
        row = read(path)
        row["maximumActiveChunks"] = 1
        write(path, row)
    elif problem == "missing":
        (fixture.directory(0) / "chunk-successor-v1.json").unlink()
    elif problem == "clock":
        data["workers"] = 1
    elif problem == "workers":
        prior["workers"] = []
    elif problem == "original":
        prior_plan["originalPlan"] = str(fixture.settings)
    elif problem == "evidence":
        prior["evidence"] = []
    elif problem == "request":
        prior_plan["request"] = str(request)
    else:
        rows = successor._list(prior_plan["chunks"])
        prior_plan["chunks"] = rows[:-1] if problem == "gap" else rows + [rows[0]]
    if problem in {"original", "gap", "duplicate", "request"}:
        copy = write(fixture.root / "modified-prior.json", prior_plan)
        prior["plan"] = pin(copy)
    write(request, data)
    with pytest.raises((ValueError, FileNotFoundError)):
        successor.prepare(request, output)
    assert not (output / "successor.json").exists()


def test_bounded_owned_cache_is_fresh_per_phase_and_no_nested_context(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    fixture = case(tmp_path, monkeypatch, chunks=2)
    prepared = successor.prepare(fixture.request, fixture.output)
    actual = InputHashCache
    caches: list[InputHashCache] = []

    def create(paths: tuple[Path, ...], *, allow_uncached_overflow: bool) -> InputHashCache:
        cache = actual(paths, allow_uncached_overflow=allow_uncached_overflow)
        caches.append(cache)
        return cache

    monkeypatch.setattr(successor, "InputHashCache", create)
    successor.phase(prepared.plan, 0, "ocr")
    successor.phase(prepared.plan, 0, "verify-ocr")
    assert len(caches) == 2 and caches[0] is not caches[1]
    assert all(cache.cache_hits > 0 and cache.hash_reads > 0 for cache in caches)
    with (
        input_hash_context(actual([fixture.settings])),
        pytest.raises(ReplayError, match="Nested"),
    ):
        successor.phase(prepared.plan, 0, "verify-ocr")


def test_selected_phase_does_not_read_unconsumed_chunk_but_full_load_rejects_tamper(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    fixture = case(tmp_path, monkeypatch, chunks=3)
    prepared = successor.prepare(fixture.request, fixture.output)
    unused = fixture.directory(2) / "times.json"
    used = fixture.directory(0) / "times.json"
    original = unused.read_bytes()
    unused.write_bytes(b"[999]")
    data = read(prepared.plan)
    selected = successor._selected_inputs(data, 0)
    assert str(unused) not in selected and str(used) in selected
    successor.phase(prepared.plan, 0, "ocr")
    with pytest.raises(ValueError, match="input changed"):
        successor._load(prepared.plan)
    unused.write_bytes(original)
    used.write_bytes(b"[999]")
    with pytest.raises(ValueError, match="input changed"):
        successor.phase(prepared.plan, 0, "verify-ocr")


def test_cli_prepare_and_phase(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    fixture = case(tmp_path, monkeypatch, chunks=2)
    assert successor.main(["prepare", str(fixture.request), "--output", str(fixture.output)]) == 0
    plan = fixture.output / "successor.json"
    assert successor.main(["ocr", str(plan), "--chunk", "0"]) == 0
    successor.phase(plan, 0, "admission")
    successor.phase(plan, 0, "verify-admission")
    successor.phase(plan, 0, "repair")
    successor.phase(plan, 0, "verify-repair")


@pytest.mark.parametrize("valid", [True, False])
def test_partial_original_running_fingerprint_is_required(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, valid: bool
) -> None:
    fixture = case(tmp_path, monkeypatch, chunks=2)
    fixture.collect(0)
    (fixture.directory(0) / "ocr/complete.json").unlink()
    old_stage = load_stages(fixture.original.parent / "worker-00.json")[0]
    write(
        fixture.states[0] / (old_stage.name + ".json"),
        {
            "status": "running",
            "fingerprint": _fingerprint(old_stage) if valid else "0" * 64,
        },
    )
    prepared = successor.prepare(fixture.request, fixture.output)
    if valid:
        successor.phase(prepared.plan, 0, "ocr")
        cleanup.check_ocr(fixture.directory(0) / "chunk.json")
    else:
        with pytest.raises(ValueError, match="binding changed"):
            successor.phase(prepared.plan, 0, "ocr")
        assert not (fixture.directory(0) / "ocr/complete.json").exists()


def test_changed_model_at_terminal_retains_real_outputs(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    fixture = case(tmp_path, monkeypatch, chunks=2)
    prepared = successor.prepare(fixture.request, fixture.output)

    def batch(
        source: Path,
        times: tuple[float, ...],
        output: Path,
        result: Path,
        *,
        native_executable: Path | None,
        head_tracking: bool = False,
    ) -> None:
        write(result, frames(times))
        (fixture.root / "model").write_bytes(b"different model at terminal")

    monkeypatch.setattr(observations, "_collect_batch", batch)
    with pytest.raises(ValueError, match="fingerprint changed"):
        successor.phase(prepared.plan, 0, "ocr")
    assert (fixture.directory(0) / "ocr/complete.json").exists()
    assert not (prepared.states[0] / "chunk-00000-ocr.json").exists()


def test_actual_clock_substitution_not_reused(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    fixture = case(tmp_path, monkeypatch, chunks=2)
    fixture.collect(0)
    (fixture.directory(0) / "ocr/complete.json").unlink()
    raw = fixture.directory(0) / "ocr/0000.result.json"
    rows = frames(tuple(float(i) for i in range(8)))
    rows[0]["actualTimeSeconds"] = 0.5
    write(raw, rows)
    with pytest.raises(ValueError, match="actual PTS"):
        successor.prepare(fixture.request, fixture.output)


@pytest.mark.parametrize(
    "problem",
    [
        "unknown",
        "partial-admission",
        "forged-complete",
        "stage-command",
        "duplicate-state",
        "extra-stage",
    ],
)
def test_original_evidence_and_producer_identity(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, problem: str
) -> None:
    fixture = case(tmp_path, monkeypatch, chunks=2)
    if problem == "forged-complete":
        fixture.complete(0)
        receipt = fixture.states[0] / "chunk-00000-ocr.json"
        data = read(receipt)
        data["fingerprint"] = "0" * 64
        write(receipt, data)
        prepared = successor.prepare(fixture.request, fixture.output)
        with pytest.raises(ValueError, match="Genuine unchanged"):
            successor.phase(prepared.plan, 0, "reuse")
        assert not (fixture.output / "reused-00000.json").exists()
        return
    if problem in {"unknown", "partial-admission"}:
        operation = "ocr" if problem == "unknown" else "admission"
        write(
            fixture.states[0] / ("chunk-00000-" + operation + ".json"),
            {
                "status": "unknown" if problem == "unknown" else "running",
                "fingerprint": "0" * 64,
            },
        )
    else:
        data = read(fixture.request)
        workers = successor._list(data["oldWorkers"])
        if problem == "duplicate-state":
            successor._object(workers[1])["state"] = str(fixture.states[0])
        else:
            workflow = fixture.original.parent / "worker-00.json"
            graph = read(workflow)
            definitions = successor._list(graph["stages"])
            first = successor._object(definitions[0])
            if problem == "stage-command":
                first["command"] = ["unexpected producer"]
            else:
                definitions.append({**first, "name": "extra"})
                graph["stages"] = definitions
            write(workflow, graph)
            successor._object(workers[0])["workflow"] = pin(workflow)
        data["oldWorkers"] = workers
        write(fixture.request, data)
    with pytest.raises((ValueError, StageError)):
        successor.prepare(fixture.request, fixture.output)


def test_linked_and_outside_metadata_no_side_effects(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    fixture = case(tmp_path, monkeypatch)
    linked = fixture.root / "linked-request.json"
    linked.symlink_to(fixture.request)
    with pytest.raises(ValueError):
        successor.prepare(linked, fixture.output)
    with pytest.raises(ValueError, match="repository work"):
        successor.prepare(Path.cwd() / "README.md", fixture.output)
    assert not (fixture.output / "successor.json").exists()


def test_historical_fingerprint_matches_shared_runner(tmp_path: Path) -> None:
    code = tmp_path / "worker.py"
    code.write_bytes(b"code")
    data = tmp_path / "data.txt"
    data.write_bytes(b"data")
    output = tmp_path / "out.txt"
    output.write_bytes(b"result")
    stage = Stage(
        name="stage",
        command=("true",),
        inputs=(code, data),
        outputs=(output,),
        timeout_seconds=10,
        verify_command=("true",),
    )
    assert successor._historical_fingerprint(
        stage, {str(code): file_digest(code), str(data): file_digest(data)}
    ) == _fingerprint(stage)


def test_complete_accepts_pre_recovery_generation(tmp_path: Path) -> None:
    code = tmp_path / "subtitle_session_cleanup.py"
    code.write_bytes(b"new recovery code")
    data = tmp_path / "data.txt"
    data.write_bytes(b"data")
    output = tmp_path / "out.txt"
    output.write_bytes(b"result")
    state = tmp_path / "state"
    state.mkdir()
    stage = Stage(
        name="stage",
        command=("true",),
        inputs=(code, data),
        outputs=(output,),
        timeout_seconds=10,
        verify_command=("true",),
    )
    old_digests = {
        str(code): compat.PRE_RECOVERY_DIGESTS["subtitle_session_cleanup.py"],
        str(data): file_digest(data),
    }
    receipt = {
        "status": "complete",
        "fingerprint": successor._historical_fingerprint(stage, old_digests),
        "outputs": {str(output): file_digest(output)},
    }
    (state / "stage.json").write_text(json.dumps(receipt), encoding="utf-8")
    historical = compat.historical_stage_input_sets(
        {str(code): file_digest(code), str(data): file_digest(data)}
    )
    assert len(historical) == 2
    successor._complete(stage, state, historical_input_sets=historical)
    with pytest.raises(ValueError, match="Genuine unchanged"):
        successor._complete(stage, state)
    receipt["fingerprint"] = "0" * 64
    (state / "stage.json").write_text(json.dumps(receipt), encoding="utf-8")
    with pytest.raises(ValueError, match="Genuine unchanged"):
        successor._complete(stage, state, historical_input_sets=historical)
