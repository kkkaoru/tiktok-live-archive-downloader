"""Synthetic clocks/provider replies only; no real model/native/inference jobs."""

import json
import os
import shutil
import stat
import subprocess
import tempfile
import time
from dataclasses import dataclass
from fractions import Fraction
from io import BufferedWriter
from pathlib import Path
from typing import NamedTuple
from unittest.mock import Mock

import numpy as np
import pytest

from replay import code_generation_compat as compat
from replay import delivery_source_assembly, subtitle_detector, workflow
from replay import subtitle_cleanup_workflow as legacy
from replay import subtitle_session_cleanup as flow
from replay import subtitle_session_observations as session
from replay.recordings import record
from replay.stages import execution_lock, file_digest
from replay.subtitle_inpaint import Image
from replay.subtitle_stream import BatchDetector, EvidenceSink
from replay.subtitle_video import VideoRepairJob, VideoRepairReceipt
from replay.workflow import load_stages, stage_graph


@dataclass(frozen=True)
class Case:
    config: Path
    output: Path
    chunk: Path
    state: Path
    source: Path
    provider: Path
    native: Mock


class Usage(NamedTuple):
    total: int
    used: int
    free: int


class Detector:
    def __init__(self, model: Path, *, sha256: str) -> None:
        if file_digest(model) != sha256:
            raise ValueError("Synthetic model changed")

    def detect(self, images: tuple[Image, ...]) -> tuple[Image, ...]:
        return tuple(np.ones(image.shape[:2], dtype=np.uint8) * 255 for image in images)


class Capture:
    def __init__(self, directory: Path, frames: list[int], clock: object) -> None:
        self.frames = frames

    def manifest(self) -> dict[str, object]:
        return {"kind": "synthetic writer test double", "qualityApproval": False}


def document(path: Path, value: object) -> dict[str, str]:
    path.write_text(json.dumps(value), encoding="utf-8")
    return {"path": str(path), "sha256": file_digest(path)}


def observation(point: Fraction, *, text: str = "字幕春", width: int = 100) -> dict[str, object]:
    return {
        "requestedTimeSeconds": float(point),
        "actualTimeSeconds": float(point),
        "width": width,
        "height": 120,
        "lines": [
            {
                "text": text,
                "confidence": 0.9,
                "region": {"x": 0, "y": 74, "width": 80, "height": 20},
            }
        ],
    }


def native_block(directory: Path, samples: tuple[Fraction, ...], start: int) -> dict[str, object]:
    selected = samples[start : start + 8]
    raw = directory / f"block-{start // 8}.json"
    document(raw, [observation(point, text="字幕夏") for point in selected])
    block = {
        "firstSample": start,
        "requested": session._clocks(selected),
        "actual": session._clocks(selected),
        "measurementsPath": str(raw),
    }
    document(directory / f"block-{start // 8}-clocks.json", block)
    return block


def native_reply(
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
    blocks = [native_block(directory, points, start) for start in range(0, len(points), 8)]
    report = {
        "protocolVersion": session.PROTOCOL,
        "sourceModified": False,
        "qualityApproved": False,
        "sourceHashVerified": False,
        "sourceIdentity": session._identity(Path(str(request["path"]))),
        "frames": len(points),
        "blocks": blocks,
        "timing": {"elapsedSeconds": 0.1},
    }
    document(directory / "complete.json", report)
    stdout.write(json.dumps(report).encode())
    return subprocess.CompletedProcess(command, 0)


def render_reply(
    job: VideoRepairJob,
    *,
    detect_batch: BatchDetector | None = None,
    evidence: EvidenceSink | None = None,
) -> VideoRepairReceipt:
    if detect_batch is not None:
        active = sum(span.end_frame - span.start_frame for span in job.spans)
        if active:
            detect_batch(tuple(np.zeros((20, 80, 3), dtype=np.uint8) for index in range(active)))
    job.output.write_bytes(b"synthetic encoded local repair")
    return VideoRepairReceipt(job.frames, 0.1, 0, 1)


@pytest.fixture
def case(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Case:
    work = tmp_path / "edits/work"
    work.mkdir(parents=True)
    source, old_native, new_native, model = (
        work / name for name in ("source.mov", "old-native", "new-native", "model")
    )
    source.write_bytes(b"synthetic source")
    scratch = work / "tmp"
    scratch.mkdir()
    monkeypatch.setenv("TMPDIR", str(scratch))
    monkeypatch.setenv("TMP", str(scratch))
    monkeypatch.setenv("TEMP", str(scratch))
    monkeypatch.setattr(tempfile, "tempdir", str(scratch))
    old_native.write_bytes(b"original repair executable")
    new_native.write_bytes(b"new session executable")
    model.write_bytes(b"synthetic detector model")
    old_native.chmod(0o700)
    new_native.chmod(0o700)
    monkeypatch.setattr(shutil, "which", lambda name: str(old_native))
    clock = document(
        work / "clock.json",
        {
            "pts": [0, 34, 68, 102, 135, 169, 203, 237, 271, 305],
            "timeBase": "1/1000",
            "lastDuration": 34,
        },
    )
    classification = document(work / "classification.json", {"kind": "synthetic source review"})
    profile = work / "profile"
    profile.mkdir()
    inputs = document(
        profile / "inputs.json",
        {
            "sourceSHA256": file_digest(source),
            "source": str(source),
            "nativeSHA256": file_digest(old_native),
            "nativeExecutable": str(old_native),
            "measurementTool": "video_text_recognize",
            "fullFrame": True,
        },
    )
    raw = document(
        profile / "0000.result.json",
        [
            observation(Fraction(point, 1000), text=text)
            for point, text in zip(
                [0, 34, 68, 102, 135, 169, 203, 237],
                ["字幕春", "字幕夏", "字幕秋", "字幕冬", "字幕雨", "字幕雲", "字幕花", "字幕風"],
                strict=True,
            )
        ],
    )
    second = document(
        profile / "0001.result.json",
        [
            observation(Fraction(271, 1000), text="字幕月"),
            observation(Fraction(305, 1000), text="字幕空"),
        ],
    )
    complete = document(
        profile / "complete.json",
        {
            "observations": 10,
            "results": {"0000.result.json": raw["sha256"], "0001.result.json": second["sha256"]},
        },
    )
    settings = work / "settings.json"
    document(
        settings,
        {
            "version": 1,
            "source": {"path": str(source), "sha256": file_digest(source)},
            "native": {"path": str(old_native), "sha256": file_digest(old_native)},
            "decoder": {"path": str(old_native), "sha256": file_digest(old_native)},
            "model": {"path": str(model), "sha256": file_digest(model)},
            "sourceClock": clock,
            "classificationEvidence": classification,
            "profileInputs": inputs,
            "profileComplete": complete,
            "chunkFrames": 4,
        },
    )
    legacy_root = work / "original-cleanup"
    legacy.prepare(settings, legacy_root, workers=1)
    state = legacy_root / "state"
    state.mkdir()
    note = document(work / "owner-note.json", {"kind": "synthetic terminal owner observation"})
    handoff = document(
        work / "handoff.json",
        {
            "version": 1,
            "observedAtNs": time.time_ns(),
            "maximumAgeSeconds": 3600,
            "quiesced": True,
            "oldWorkersDisabled": True,
            "actors": [{"pid": 2147483647, "birth": "synthetic absent process"}],
            "oldStates": [str(state)],
            "evidence": [note],
        },
    )
    provider = work / "session-plan.json"
    provider_ref = document(
        provider,
        {
            "version": 1,
            "protocol": session.PROTOCOL,
            "source": str(source),
            "sourceSHA256": file_digest(source),
            "nativeExecutable": str(new_native),
            "nativeSHA256": file_digest(new_native),
            "samples": [{"value": point, "timescale": 1000} for point in [135, 169, 203, 237]],
            "width": 100,
            "height": 120,
            "workers": 2,
        },
    )
    config = work / "request.json"
    document(
        config,
        {
            "version": 1,
            "originalPlan": flow._pin(legacy_root / "plan.json"),
            "chunkIndex": 1,
            "originalWorker": {
                "workflow": flow._pin(legacy_root / "worker-00.json"),
                "state": str(state),
            },
            "handoff": handoff,
            "ocrProvider": provider_ref,
        },
    )
    output = work / "session-cleanup"
    output.mkdir()
    run = Mock(side_effect=native_reply)
    monkeypatch.setattr(subprocess, "run", run)
    monkeypatch.setattr(
        flow,
        "_code",
        lambda: {str(Path(flow.__file__).resolve()): file_digest(Path(flow.__file__))},
    )
    monkeypatch.setattr(subtitle_detector, "LearnedTextDetector", Detector)
    monkeypatch.setattr(flow, "RepairCapture", Capture)
    monkeypatch.setattr(flow, "render_repaired_video", render_reply)
    return Case(
        config, output, legacy_root / "chunk-00001/chunk.json", state, source, provider, run
    )


def prepared(case: Case) -> flow.Prepared:
    return flow.prepare(case.config, case.output)


def collect(case: Case) -> flow.Prepared:
    result = prepared(case)
    flow.run(result.operation, mode="collect")
    return result


def admitted(case: Case) -> flow.Prepared:
    result = collect(case)
    flow.run(result.operation, mode="admission")
    return result


def test_normal_phase_graph_preserves_originals_and_honest_new_provenance(case: Case) -> None:
    before = case.chunk.read_bytes()
    result = admitted(case)
    flow.run(result.operation, mode="verify-ocr")
    flow.run(result.operation, mode="verify-admission")
    flow.run(result.operation, mode="repair")
    flow.run(result.operation, mode="verify-repair")
    assert case.chunk.read_bytes() == before
    assert stage_graph(load_stages(result.workflow)) == {
        "chunk-00001-session-collect": (),
        "chunk-00001-session-admission": ("chunk-00001-session-collect",),
        "chunk-00001-session-repair": ("chunk-00001-session-admission",),
    }
    assert case.native.call_count == 1
    producer = record(session._read(case.output / "session/inputs.json"))
    assert producer["measurementTool"] == "video_text_recognize_session"
    assert producer["backend"] == "native-session-cli"
    candidate = record(flow._read(case.output / "candidate-config.json", {}))
    assert candidate["fixedBlur"] is False
    assert candidate["localRepairOnly"] is True
    assert str(candidate["nativeExecutable"]).endswith("/old-native")
    result_doc = flow._read(case.output / "repair/candidate.json", {})
    assert result_doc["detectionBackend"] == "ppocrv3-db"
    assert result_doc["activeSourceFrames"] == [0, 1, 2, 3]
    assert result_doc["sourceStart"] == "27/200"
    assert result_doc["fullDecodeVerified"] is False
    assert result_doc["consumerConnected"] is False
    assert not (case.chunk.parent / "ocr").exists()
    assert not (case.chunk.parent / "repair").exists()


def test_collect_readonly_reuse_never_reissues_native(case: Case) -> None:
    result = collect(case)
    flow.run(result.operation, mode="verify-ocr")
    flow.run(result.operation, mode="verify-ocr")
    assert case.native.call_count == 1
    assert flow._read(case.output / "session/complete.json", {})["consumerConnected"] is False


def test_model_and_source_are_not_hashed_during_plan(
    case: Case, monkeypatch: pytest.MonkeyPatch
) -> None:
    real = file_digest
    calls = Mock(wraps=real)
    monkeypatch.setattr(flow, "file_digest", calls)
    prepared(case)
    assert str(case.source) not in [str(call.args[0]) for call in calls.call_args_list]


@pytest.mark.parametrize(
    "name",
    ["ocr", "repair", "scratch", "admission.json", "candidate-config.json", "eligibility.json"],
)
def test_existing_or_partial_original_is_never_recollected(case: Case, name: str) -> None:
    (case.chunk.parent / name).write_bytes(b"uncertain original output")
    with pytest.raises(ValueError, match="already started"):
        prepared(case)
    case.native.assert_not_called()


@pytest.mark.parametrize("status", ["running", "complete", "failed"])
def test_any_original_stage_receipt_prevents_provider_takeover(case: Case, status: str) -> None:
    document(case.state / "chunk-00001-ocr.json", {"status": status})
    with pytest.raises(ValueError, match="Stage evidence"):
        prepared(case)


def test_original_inflight_stage_lease_blocks_admission(case: Case) -> None:
    with execution_lock(case.state, "chunk-00001-ocr"), pytest.raises(ValueError, match="owned"):
        prepared(case)


def test_old_owner_active_is_not_silently_ignored(case: Case) -> None:
    config = flow._read(case.config, {})
    handoff_path = Path(str(record(config["handoff"])["path"]))
    handoff = flow._read(handoff_path, {})
    handoff["actors"] = [{"pid": os.getpid(), "birth": "synthetic current owner"}]
    config["handoff"] = document(handoff_path, handoff)
    document(case.config, config)
    with pytest.raises(ValueError, match="still present"):
        prepared(case)


def test_verify_mode_accepts_historical_generation_after_rebind(case: Case) -> None:
    result = collect(case)
    operation = result.operation
    data = json.loads(operation.read_text(encoding="utf-8"))
    reader = dict(data["readerCode"])
    for path in list(reader):
        name = Path(path).name
        if name in compat.RECOVERY_FILES:
            reader[path] = compat.PRE_RECOVERY_DIGESTS[name]
    data["readerCode"] = reader
    operation.write_text(json.dumps(data), encoding="utf-8")
    marker = case.chunk.parent / "session-consumer-binding.json"
    marker.write_text(
        json.dumps(
            {
                "version": 1,
                "operation": {"path": str(operation), "sha256": file_digest(operation)},
                "workRoot": str(case.output),
            }
        ),
        encoding="utf-8",
    )
    flow.run(operation, mode="verify-ocr")
    with pytest.raises(ValueError, match="operation/code changed"):
        flow.run(operation, mode="collect")


def test_incomplete_reservation_superseded_with_lineage(case: Case) -> None:
    first = flow.prepare(case.config, case.output)
    marker = case.chunk.parent / "session-consumer-binding.json"
    second_output = case.output.with_name("second-consumer")
    second_output.mkdir()
    second = flow.prepare(case.config, second_output)
    rebound = record(json.loads(marker.read_text(encoding="utf-8")))
    assert rebound == {
        "version": 1,
        "operation": {"path": str(second.operation), "sha256": file_digest(second.operation)},
        "workRoot": str(second_output),
    }
    archived = record(json.loads((case.output / flow.SUPERSEDED_BINDING_NAME).read_text("utf-8")))
    assert archived == {
        "version": 1,
        "operation": {"path": str(first.operation), "sha256": file_digest(first.operation)},
        "workRoot": str(case.output),
    }


def test_completed_session_reservation_is_never_superseded(case: Case) -> None:
    flow.prepare(case.config, case.output)
    marker = case.chunk.parent / "session-consumer-binding.json"
    reservation = record(json.loads(marker.read_text(encoding="utf-8")))
    (Path(str(reservation["workRoot"])) / "completion.json").write_text("{}", encoding="utf-8")
    second_output = case.output.with_name("second-consumer")
    second_output.mkdir()
    with pytest.raises(ValueError, match="supersede"):
        flow.prepare(case.config, second_output)


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("sourceSHA256", "f" * 64),
        ("workers", 1),
        ("width", 99),
        ("protocol", "video_text_recognize"),
    ],
)
def test_wrong_provider_requires_fresh_correct_authority(
    case: Case, field: str, value: object
) -> None:
    provider = record(session._read(case.provider))
    provider[field] = value
    config = flow._read(case.config, {})
    config["ocrProvider"] = document(case.provider, provider)
    document(case.config, config)
    with pytest.raises(ValueError):
        prepared(case)


@pytest.mark.parametrize("points", [[135, 169, 204, 237], [135, 169, 169, 237], [0, 34, 68, 102]])
def test_provider_cannot_substitute_round_or_reorder_pts(case: Case, points: list[int]) -> None:
    provider = record(session._read(case.provider))
    provider["samples"] = [{"value": point, "timescale": 1000} for point in points]
    config = flow._read(case.config, {})
    config["ocrProvider"] = document(case.provider, provider)
    document(case.config, config)
    with pytest.raises(ValueError):
        prepared(case)


def test_changed_metadata_pin_is_not_repinned(case: Case) -> None:
    case.provider.write_text("{}", encoding="utf-8")
    with pytest.raises(ValueError, match="pin changed"):
        prepared(case)


@pytest.mark.parametrize(
    ("field", "value"),
    [("version", 2), ("chunkIndex", -1), ("chunkIndex", True), ("extra", "unsupported")],
)
def test_invalid_request_schema(case: Case, field: str, value: object) -> None:
    data = flow._read(case.config, {})
    data[field] = value
    document(case.config, data)
    with pytest.raises(ValueError):
        prepared(case)


def test_changed_source_rejected_at_runtime_before_native(case: Case) -> None:
    result = prepared(case)
    case.source.write_bytes(b"new source")
    with pytest.raises(ValueError, match="fingerprint"):
        flow.run(result.operation, mode="collect")
    case.native.assert_not_called()


def test_changed_code_rejects_runtime(case: Case, monkeypatch: pytest.MonkeyPatch) -> None:
    result = prepared(case)
    monkeypatch.setattr(flow, "_code", lambda: {})
    with pytest.raises(ValueError, match="code changed"):
        flow.run(result.operation, mode="collect")


def test_wrong_namespace_marker_blocks_runtime(case: Case) -> None:
    result = prepared(case)
    document(case.chunk.parent / "session-consumer-binding.json", {"version": 1})
    with pytest.raises(ValueError, match="reservation"):
        flow.run(result.operation, mode="collect")


def test_genuine_raw_session_pts_mutation_rejected(case: Case) -> None:
    result = collect(case)
    raw = case.output / "session/attempt-000000/native/block-0.json"
    rows = session._read(raw)
    assert isinstance(rows, list)
    record(rows[2])["actualTimeSeconds"] = 0.204
    document(raw, rows)
    with pytest.raises(ValueError):
        flow.run(result.operation, mode="verify-ocr")


def test_measured_admission_cannot_add_fixed_rectangle(case: Case) -> None:
    result = admitted(case)
    path = case.output / "admission.json"
    data = flow._read(path, {})
    data["masks"] = [
        {
            "startSeconds": 0,
            "endSeconds": 9,
            "region": {"x": 0, "y": 0, "width": 100, "height": 120},
        }
    ]
    document(path, data)
    with pytest.raises(ValueError, match="contract changed"):
        flow.run(result.operation, mode="repair")


def test_capacity_failure_keeps_ocr_and_no_repair(
    case: Case, monkeypatch: pytest.MonkeyPatch
) -> None:
    result = admitted(case)
    monkeypatch.setattr(
        shutil, "disk_usage", lambda path: Usage(100 * 1024**3, 99 * 1024**3, 1024**3)
    )
    with pytest.raises(ValueError, match="capacity"):
        flow.run(result.operation, mode="repair")
    assert (case.output / "ocr-verification.json").is_file()
    assert not (case.output / "repair").exists()


def test_normal_workflow_preflight_accepts_the_real_producer_consumer_graph(
    case: Case, capsys: pytest.CaptureFixture[str]
) -> None:
    plan = prepared(case)
    assert workflow.main([str(plan.workflow), "--state", str(plan.state), "--check"]) == 0
    assert json.loads(capsys.readouterr().out) == {
        "chunk-00001-session-collect": [],
        "chunk-00001-session-admission": ["chunk-00001-session-collect"],
        "chunk-00001-session-repair": ["chunk-00001-session-admission"],
    }
    assert not plan.state.exists()
    assert case.native.call_count == 0


@pytest.mark.parametrize("mutation", ["media", "receipt", "input", "row"])
def test_repair_readonly_verifier_rejects_changed_results(case: Case, mutation: str) -> None:
    result = admitted(case)
    flow.run(result.operation, mode="repair")
    path = {
        "media": "repair/candidate.mp4",
        "receipt": "repair/candidate.json",
        "input": "candidate-config.json",
        "row": "assembly-row.json",
    }[mutation]
    (case.output / path).write_bytes(b"changed")
    with pytest.raises((ValueError, json.JSONDecodeError)):
        flow.run(result.operation, mode="verify-repair")


def test_output_work_root_cannot_be_occupied(case: Case) -> None:
    (case.output / "retained").write_bytes(b"preserve")
    with pytest.raises(ValueError, match="empty"):
        prepared(case)


@pytest.mark.parametrize("value", ["relative", "/tmp/new", "", "bad\npath"])
def test_path_guards(value: str) -> None:
    with pytest.raises(ValueError):
        flow._path(value)


@pytest.mark.parametrize("value", [None, "notsha", "A" * 64])
def test_digest_guards(value: object) -> None:
    with pytest.raises(ValueError):
        flow._sha(value)


@pytest.mark.parametrize("value", [0, True, "4"])
def test_frame_budget_guards(value: object) -> None:
    with pytest.raises(ValueError):
        flow._positive(value)


def test_cli_and_invalid_modes(case: Case, capsys: pytest.CaptureFixture[str]) -> None:
    assert flow.main(["prepare", str(case.config), "--output", str(case.output)]) == 0
    assert "workflow.json" in capsys.readouterr().out
    assert flow.main(["collect", str(case.output / "operation.json")]) == 0
    with pytest.raises(ValueError, match="Unknown"):
        flow.run(case.output / "operation.json", mode="blur")
    with pytest.raises(ValueError, match="distinct"):
        flow.main(["prepare", str(case.config)])
    with pytest.raises(ValueError, match="pinned"):
        flow.main(["collect", str(case.config), "--output", str(case.output)])


def test_invalid_reference_shape(case: Case) -> None:
    with pytest.raises(ValueError, match="reference"):
        flow._reference(
            {"path": str(case.config), "sha256": file_digest(case.config), "extra": 1}, {}
        )
    with pytest.raises(ValueError, match="reference"):
        flow._asset({"path": str(case.source)}, {})
    with pytest.raises(ValueError, match="list"):
        flow.list_rows({})


@pytest.fixture
def mixed(case: Case) -> tuple[Path, Path]:
    result = admitted(case)
    flow.run(result.operation, mode="repair")
    job = flow.load_config(case.config)
    source, native = record(job.settings["source"]), record(job.settings["native"])
    for index, points, terminal in ((0, [0, 34, 68, 102], 33), (2, [271, 305], 34)):
        chunk = job.original_plan.parent / f"chunk-{index:05d}"
        repair = chunk / "repair"
        repair.mkdir()
        media = repair / "candidate.mp4"
        media.write_bytes(b"synthetic original-provider encoded repair")
        producer = document(
            chunk / "candidate-config.json",
            {
                "source": source["path"],
                "sourceSHA256": source["sha256"],
                "nativeExecutable": native["path"],
                "nativeSHA256": native["sha256"],
                "outputDirectory": str(repair),
                "clock": {"pts": points, "timeBase": "1/1000", "lastDuration": terminal},
                "width": 100,
                "height": 120,
            },
        )
        document(
            repair / "candidate.json",
            {
                "inputs": {
                    producer["path"]: producer["sha256"],
                    str(source["path"]): source["sha256"],
                    str(native["path"]): native["sha256"],
                },
                "render": {"frames": len(points)},
                "sourceStart": str(Fraction(points[0], 1000)),
                "videoSHA256": file_digest(media),
            },
        )
    request = case.output.parent / "mixed-config.json"
    document(
        request,
        {
            "version": 1,
            "originalPlan": flow._pin(job.original_plan),
            "sessions": [{"chunkIndex": 1, "operation": flow._pin(result.operation)}],
        },
    )
    output = case.output.parent / "mixed-assembly"
    output.mkdir()
    return request, output


def test_existing_assembly_consumer_accepts_mixed_exact_clock_and_original_bindings(
    case: Case, mixed: tuple[Path, Path]
) -> None:
    request, output = mixed
    manifest = flow.prepare_assembly(request, output)
    chunks = delivery_source_assembly.load_chunks(
        manifest, sha256=file_digest(manifest), bindings={}
    )
    assert tuple(chunk.clock.pts for chunk in chunks) == (
        (0, 34, 68, 102),
        (135, 169, 203, 237),
        (271, 305),
    )
    assert tuple(chunk.clock.last_duration for chunk in chunks) == (33, 34, 34)
    assert tuple(chunk.source[0] for chunk in chunks) == (
        str(case.source),
        str(case.source),
        str(case.source),
    )
    proof = flow._read(output / "preparation.json", {})
    assert proof["physicalSourceFrames"] == 10
    assert proof["physicalSourceTerminalPTS"] == 339
    assert proof["sessionChunkIndices"] == [1]
    assert proof["mediaBytesVerified"] is False
    assert proof["consumerConnected"] is False
    assert proof["stageFingerprintRecomputed"] is False
    assert proof["deliveryComplete"] is False


def test_mixed_preparation_does_not_hash_media_or_models(
    case: Case, mixed: tuple[Path, Path], monkeypatch: pytest.MonkeyPatch
) -> None:
    request, output = mixed
    actual = file_digest
    digest = Mock(wraps=actual)
    monkeypatch.setattr(flow, "file_digest", digest)
    flow.prepare_assembly(request, output)
    assert all(call.args[0].suffix != ".mp4" for call in digest.call_args_list)
    assert all(call.args[0] != case.source for call in digest.call_args_list)


@pytest.mark.parametrize(
    "selection",
    [
        [{"chunkIndex": 1, "operation": None}, {"chunkIndex": 1, "operation": None}],
        [{"chunkIndex": 5, "operation": None}],
        [{"chunkIndex": True, "operation": None}],
        [{"chunkIndex": 1, "operation": None, "extra": True}],
        [None, None, None, None],
    ],
)
def test_mixed_selection_is_exact_unique_and_bounded(
    mixed: tuple[Path, Path], selection: list[object]
) -> None:
    request, output = mixed
    data = flow._read(request, {})
    operation = record(flow.list_rows(data["sessions"])[0])["operation"]
    data["sessions"] = [
        dict(record(row), operation=operation) if isinstance(row, dict) else row
        for row in selection
    ]
    document(request, data)
    with pytest.raises(ValueError, match="selections|membership"):
        flow.prepare_assembly(request, output)


@pytest.mark.parametrize(
    "field,value",
    [
        ("nativeSHA256", "0" * 64),
        ("sourceSHA256", "0" * 64),
        ("clock", {"pts": [135, 169, 203, 238], "timeBase": "1/1000", "lastDuration": 34}),
        ("localRepairOnly", False),
        ("fixedBlur", True),
    ],
)
def test_mixed_session_cannot_weaken_local_repair_or_original_clock(
    case: Case, mixed: tuple[Path, Path], field: str, value: object
) -> None:
    request, output = mixed
    producer = case.output / "candidate-config.json"
    data = flow._read(producer, {})
    data[field] = value
    document(producer, data)
    with pytest.raises(ValueError, match="binding"):
        flow.prepare_assembly(request, output)


@pytest.mark.parametrize(
    "mutation", ["geometry", "missing", "source", "terminal", "row", "operation", "marker"]
)
def test_mixed_missing_or_wrong_producer_is_not_called_complete(
    case: Case, mixed: tuple[Path, Path], mutation: str
) -> None:
    request, output = mixed
    chunk = case.chunk.parent.parent / "chunk-00000"
    if mutation in {"geometry", "source", "terminal"}:
        producer = chunk / "candidate-config.json"
        data = flow._read(producer, {})
        if mutation == "geometry":
            data["width"] = 99
        elif mutation == "source":
            data["sourceSHA256"] = "0" * 64
        else:
            data["clock"] = {"pts": [0, 34, 68, 102], "timeBase": "1/1000", "lastDuration": 34}
        ref = document(producer, data)
        receipt_path = chunk / "repair/candidate.json"
        receipt = flow._read(receipt_path, {})
        record(receipt["inputs"])[ref["path"]] = ref["sha256"]
        document(receipt_path, receipt)
    elif mutation == "missing":
        (chunk / "repair/candidate.json").unlink()
    elif mutation == "row":
        document(case.output / "assembly-row.json", {"consumerConnected": True})
    elif mutation == "operation":
        document(case.output / "operation.json", {"protocol": "invented"})
    else:
        document(case.chunk.parent / "session-consumer-binding.json", {})
    with pytest.raises(
        (ValueError, FileNotFoundError), match="geometry|identity|No such|row|pin|reservation"
    ):
        flow.prepare_assembly(request, output)


def test_mixed_cannot_publish_over_existing_or_incomplete_plan(mixed: tuple[Path, Path]) -> None:
    request, output = mixed
    flow.prepare_assembly(request, output)
    with pytest.raises(ValueError, match="empty"):
        flow.prepare_assembly(request, output)


def test_mixed_cli_and_invalid_schema(
    mixed: tuple[Path, Path], capsys: pytest.CaptureFixture[str]
) -> None:
    request, output = mixed
    assert flow.main(["prepare-assembly", str(request), "--output", str(output)]) == 0
    assert capsys.readouterr().out.endswith("/manifest.json\n")
    data = flow._read(request, {})
    data["version"] = True
    document(request, data)
    fresh = output.parent / "schema-output"
    fresh.mkdir()
    with pytest.raises(ValueError, match="config"):
        flow.prepare_assembly(request, fresh)


@pytest.mark.parametrize("mutation", ["decoder", "environment", "actual"])
def test_actual_repair_decoder_and_scratch_are_checked_before_render(
    case: Case, monkeypatch: pytest.MonkeyPatch, mutation: str
) -> None:
    result = admitted(case)
    if mutation == "decoder":
        monkeypatch.setattr(shutil, "which", lambda name: None)
    elif mutation == "environment":
        monkeypatch.delenv("TEMP")
    else:
        monkeypatch.setattr(tempfile, "tempdir", str(case.output.parent))
    with pytest.raises(ValueError, match="FFmpeg|scratch"):
        flow.run(result.operation, mode="repair")
    assert not (case.output / "repair").exists()


@pytest.fixture
def terminal(case: Case, monkeypatch: pytest.MonkeyPatch) -> Path:
    request = flow._read(case.config, {})
    worker = record(request["originalWorker"])
    workflow_path = Path(str(record(worker["workflow"])["path"]))
    stages = tuple(
        stage for stage in load_stages(workflow_path) if stage.name.startswith("chunk-00000")
    )
    log = case.output.parent / "old-owned-job.log"
    log.write_text(
        "chunk-00000-ocr: completed\n"
        "chunk-00000-admission: completed\n"
        "chunk-00000-repair: completed\n",
        encoding="utf-8",
    )
    status = case.output.parent / "old-owned-job.status"
    status.write_bytes(b"0\n")
    completed: list[dict[str, object]] = []
    for stage in stages:
        receipt = document(
            case.state / f"{stage.name}.json",
            {
                "status": "complete",
                "fingerprint": "c" * 64,
                "outputs": {str(path): "d" * 64 for path in stage.outputs},
            },
        )
        completed.append({"name": stage.name, "receipt": receipt, "fingerprint": "c" * 64})
    draft = case.output.parent / "terminal-draft.json"
    ref = document(
        draft,
        {
            "version": 1,
            "kind": flow.TERMINAL_KIND,
            "observedAtNs": None,
            "maximumAgeSeconds": 3600,
            "oldWorkersDisabled": True,
            "workflow": worker["workflow"],
            "state": str(case.state),
            "terminalJob": {
                "root": str(case.output.parent),
                "name": "old-owned-job",
                "status": flow._pin(status),
                "log": flow._pin(log),
                "targets": ["chunk-00000-repair"],
            },
            "completedStages": completed,
            "currentObservation": None,
            "historicalActorIdentityRecorded": False,
        },
    )
    request["handoff"] = ref
    document(case.config, request)
    monkeypatch.setattr(flow, "_scan_command", lambda command, **keywords: b"")
    owner = case.output.parent / "owner-observation"
    owner.mkdir()
    return owner


def test_terminal_only_owner_adapter_preserves_unknown_historical_actor_and_normal_graph(
    case: Case, terminal: Path
) -> None:
    ready = flow.observe_terminal(case.config, terminal)
    observation = flow._read(terminal / "owner-observation.json", {})
    assert observation["matchedOwnerProcesses"] == []
    assert observation["historicalActorIdentityRecorded"] is False
    assert "actors" not in flow._read(terminal / "handoff.json", {})
    plan = flow.prepare(ready, case.output)
    assert workflow.main([str(plan.workflow), "--state", str(plan.state), "--check"]) == 0
    assert str(case.chunk.parent / "session-consumer-binding.json") in tuple(
        str(path) for path in load_stages(plan.workflow)[0].inputs
    )
    flow.run(plan.operation, mode="collect")
    flow.run(plan.operation, mode="verify-ocr")
    assert case.native.call_count == 1


@pytest.mark.parametrize(
    "mutation",
    [
        "status",
        "name",
        "member",
        "duplicate",
        "coverage",
        "fingerprint",
        "running",
        "log",
        "workflow",
        "state",
        "history",
        "future",
        "schema",
    ],
)
def test_terminal_handoff_rejects_missing_or_different_original_correspondence(
    case: Case, terminal: Path, mutation: str
) -> None:
    config = flow._read(case.config, {})
    draft = Path(str(record(config["handoff"])["path"]))
    data = flow._read(draft, {})
    if mutation == "status":
        status = case.output.parent / "old-owned-job.status"
        status.write_bytes(b"1\n")
        record(data["terminalJob"])["status"] = flow._pin(status)
    elif mutation == "name":
        record(data["terminalJob"])["name"] = "different-job"
    elif mutation in {"member", "future"}:
        record(data["terminalJob"])["targets"] = [
            "absent" if mutation == "member" else "chunk-00001-ocr"
        ]
    elif mutation == "duplicate":
        rows = flow.list_rows(data["completedStages"])
        data["completedStages"] = [*rows, rows[0]]
    elif mutation == "coverage":
        data["completedStages"] = flow.list_rows(data["completedStages"])[:1]
    elif mutation in {"fingerprint", "running"}:
        entry = record(flow.list_rows(data["completedStages"])[0])
        receipt_path = Path(str(record(entry["receipt"])["path"]))
        receipt = flow._read(receipt_path, {})
        receipt["status" if mutation == "running" else "fingerprint"] = (
            "running" if mutation == "running" else "a" * 64
        )
        entry["receipt"] = document(receipt_path, receipt)
    elif mutation == "log":
        log = case.output.parent / "old-owned-job.log"
        log.write_text("old job did not complete the declared targets\n", encoding="utf-8")
        record(data["terminalJob"])["log"] = flow._pin(log)
    elif mutation == "workflow":
        data["workflow"] = flow._pin(case.config)
    elif mutation == "state":
        data["state"] = str(case.output)
    elif mutation == "history":
        data["historicalActorIdentityRecorded"] = True
    else:
        data["unknown"] = True
    config["handoff"] = document(draft, data)
    document(case.config, config)
    with pytest.raises(ValueError, match="Terminal|terminal|Invalid|reference pin"):
        flow.observe_terminal(case.config, terminal)


def test_terminal_current_process_presence_never_becomes_owner_absence(
    case: Case, terminal: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(
        flow,
        "_scan_command",
        lambda command, **keywords: (str(case.chunk.parent.parent) + " worker\n").encode(),
    )
    with pytest.raises(ValueError, match="currently present"):
        flow.observe_terminal(case.config, terminal)
    assert not (terminal / "handoff.json").exists()


def test_terminal_current_workflow_lease_is_preserved(case: Case, terminal: Path) -> None:
    with execution_lock(case.state, ".workflow") as held:
        assert held
        with pytest.raises(ValueError, match="active"):
            flow.observe_terminal(case.config, terminal)


@pytest.mark.parametrize("mutation", ["stale", "observation", "locked"])
def test_terminal_ready_config_is_rechecked_at_preparation(
    case: Case, terminal: Path, mutation: str
) -> None:
    ready = flow.observe_terminal(case.config, terminal)
    config = flow._read(ready, {})
    handoff_path = terminal / "handoff.json"
    data = flow._read(handoff_path, {})
    if mutation == "stale":
        data["observedAtNs"] = 1
    elif mutation == "observation":
        data["currentObservation"] = flow._pin(case.config)
    else:
        with execution_lock(case.state, "chunk-00001-ocr") as held:
            assert held
            with pytest.raises(ValueError, match="owned"):
                flow.prepare(ready, case.output)
        return
    config["handoff"] = document(handoff_path, data)
    document(ready, config)
    with pytest.raises(ValueError, match="expired|observation"):
        flow.prepare(ready, case.output)


@pytest.mark.parametrize(
    "code,stdout,stderr,allowed",
    [
        (0, b"", b"", False),
        (1, b"", b"no server running on socket", True),
        (1, b"", b"permission denied", True),
        (0, b"a" * 1048577, b"", False),
    ],
)
def test_owner_command_has_finite_failure_and_output_contract(
    monkeypatch: pytest.MonkeyPatch, code: int, stdout: bytes, stderr: bytes, allowed: bool
) -> None:
    run = Mock(return_value=subprocess.CompletedProcess(["synthetic"], code, stdout, stderr))
    monkeypatch.setattr(subprocess, "run", run)
    if code == 0 and not stdout or code == 1 and stderr.startswith(b"no server"):
        assert flow._scan_command(("synthetic",), allow_absent_server=allowed) == b""
        assert run.call_args.kwargs["timeout"] == 10
    else:
        with pytest.raises(ValueError, match="failed|bounded"):
            flow._scan_command(("synthetic",), allow_absent_server=allowed)


def test_terminal_observer_cli_rejects_occupied_output(case: Case, terminal: Path) -> None:
    assert flow.main(["observe-terminal", str(case.config), "--output", str(terminal)]) == 0
    with pytest.raises(ValueError, match="empty"):
        flow.observe_terminal(case.config, terminal)


@pytest.mark.parametrize(
    "mutation", ["not-socket", "no-tmux", "active-pane", "good-pane", "linked"]
)
def test_terminal_pane_identity_and_command_are_not_assumed(
    case: Case, terminal: Path, monkeypatch: pytest.MonkeyPatch, mutation: str
) -> None:
    socket = case.output.parent / "tmux.sock"
    if mutation == "linked":
        socket.symlink_to(case.config)
    else:
        socket.write_bytes(b"synthetic socket metadata")
    if mutation != "not-socket":
        monkeypatch.setattr(stat, "S_ISSOCK", lambda mode: True)
    if mutation == "no-tmux":
        monkeypatch.setattr(shutil, "which", lambda name: None)
    elif mutation == "active-pane":
        monkeypatch.setattr(
            flow,
            "_scan_command",
            lambda command, **keywords: (
                b""
                if command == flow.PS_COMMAND
                else (str(case.chunk.parent.parent) + " worker\n").encode()
            ),
        )
    if mutation == "good-pane":
        ready = flow.observe_terminal(case.config, terminal)
        assert ready.name == "config-ready.json"
        assert flow._read(terminal / "owner-observation.json", {})["matchedOwnerPanes"] == []
    else:
        with pytest.raises(ValueError, match="socket|unavailable|pane"):
            flow.observe_terminal(case.config, terminal)
