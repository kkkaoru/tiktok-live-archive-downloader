"""Synthetic retained/fresh provenance without model imports or process launches."""

import fcntl
import json
import os
import subprocess
import wave
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

from replay import asr_assembly
from replay import isolated_metal_asr as lane
from replay import metal_asr_resume as resume
from replay.audio_windows import AudioWindow
from replay.core import ReplayError
from replay.native_invocation import durable, pin


class Runtime:
    def __init__(self, *, fail: bool = False) -> None:
        self.calls = 0
        self.fail = fail

    def execute(
        self, request: lane.Request, window: AudioWindow, work: Path, identities: dict[str, object]
    ) -> None:
        self.calls += 1
        durable(work / "launch.json", {"pid": 99999999, "birth": "synthetic:1"})
        durable(
            work / "child.json",
            {
                "request": str(request.path),
                "requestSHA256": request.sha256,
                "startSample": window.start_sample,
                "sampleCount": window.sample_count,
                "identities": identities,
            },
        )
        durable(work / "raw.json", {"text": "", "segments": [], "language": "ja"})
        if self.fail:
            raise ReplayError("interrupted synthetic worker")
        (work / "worker.log").write_bytes(b"")
        durable(
            work / "resources.json",
            {
                "exitCode": 0,
                "familyGone": True,
                "samples": [{"members": [{"pid": 99999999, "birth": "synthetic:1"}]}],
            },
        )
        durable(
            work / "mlx-memory.json",
            {"device": "metal", "activeBytes": 1, "peakBytes": 2, "cacheBytes": 0},
        )
        for phase in lane.PHASES:
            durable(work / f"{phase}-descriptors.json", {"externalWritableRegular": []})


@pytest.fixture
def setup(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> tuple[Path, Path, Path]:
    monkeypatch.chdir(tmp_path)
    model = tmp_path / "model"
    model.mkdir()
    config = model / "config.json"
    weights = model / "weights.npz"
    config.write_text("{}", encoding="utf-8")
    weights.write_bytes(b"synthetic")
    pcm = tmp_path / "input.wav"
    with wave.open(str(pcm), "wb") as stream:
        stream.setparams((1, 2, lane.RATE, 0, "NONE", "not compressed"))
        stream.writeframes(b"\0\0" * (61 * lane.RATE))
    code = tmp_path / "worker.py"
    code.write_text("# synthetic", encoding="utf-8")
    monkeypatch.setattr(lane, "required_code", lambda: (code,))
    monkeypatch.setattr(lane, "DEFAULT_CONFIG_SHA256", pin(config)["sha256"])
    monkeypatch.setattr(lane, "DEFAULT_WEIGHTS_SHA256", pin(weights)["sha256"])
    request = tmp_path / "request.json"
    durable(
        request,
        {
            "version": 1,
            "device": "metal",
            "fp16": True,
            "samples": 61 * lane.RATE,
            "timeoutSeconds": 10,
            "inputs": {"pcm": pin(pcm), "weights": pin(weights), "config": pin(config)},
            "codeInputs": {str(code): pin(code)["sha256"]},
        },
    )
    source = tmp_path / "edits/work/source"
    lane.collect(request, source, runtime=Runtime(), maximum_new_windows=1)
    host = MagicMock()
    host.absent.return_value = True
    monkeypatch.setattr(resume, "SystemHost", lambda: host)
    monkeypatch.setattr(resume, "_group_gone", MagicMock())
    return request, source, tmp_path / "edits/work/resumed"


def replace(path: Path, value: dict[str, object]) -> None:
    path.write_text(json.dumps(value), encoding="utf-8")


def test_retained_and_missing_full_assembly(setup: tuple[Path, Path, Path]) -> None:
    request, source, output = setup
    original = pin(source / "000000000000.metal-runtime/complete.json")
    resume.prepare(request, source, output)
    runtime = Runtime()
    assert resume.run(request, output, runtime=runtime) == output / "complete.json"
    resume.run(request, output, runtime=runtime)
    resume.run(request, output, verify_only=True)
    assert runtime.calls == 1
    assert pin(source / "000000000000.metal-runtime/complete.json") == original
    assert lane._read(output / "000000000000.receipt.json")["kind"] == "retained"
    assert lane._read(output / "000000480000.receipt.json")["kind"] == "fresh"
    assert not (output / "000000000000.metal-runtime").exists()
    assert asr_assembly.assemble(output, output.parent / "labels.json") == 0


def test_raw_only_old_window_runs_fresh_and_preserves_source(
    setup: tuple[Path, Path, Path],
) -> None:
    request, source, output = setup
    uncertain = source / "000000480000.metal-runtime"
    uncertain.mkdir()
    durable(uncertain / "raw.json", {"segments": [], "text": "untrusted"})
    durable(uncertain / "launch.json", {"pid": 99999999, "birth": "synthetic:1"})
    before = pin(uncertain / "raw.json")
    resume.prepare(request, source, output)
    runtime = Runtime()
    resume.run(request, output, runtime=runtime)
    assert runtime.calls == 1
    assert pin(uncertain / "raw.json") == before
    assert not (uncertain / "complete.json").exists()


@pytest.mark.parametrize(
    "change", ["raw", "resources", "missing-resources", "exit", "family", "intent", "child"]
)
def test_corrupt_completed_original_never_adopted(
    setup: tuple[Path, Path, Path], change: str
) -> None:
    request, source, output = setup
    work = source / "000000000000.metal-runtime"
    if change == "missing-resources":
        (work / "resources.json").unlink()
    elif change in ("exit", "family"):
        resources = lane._read(work / "resources.json")
        resources["exitCode" if change == "exit" else "familyGone"] = (
            1 if change == "exit" else False
        )
        replace(work / "resources.json", resources)
    else:
        replace(work / f"{change}.json", {"changed": True})
    with pytest.raises((ReplayError, OSError)):
        resume.prepare(request, source, output)
    assert not output.exists()


@pytest.mark.parametrize(
    "name", ["request.json", "input.wav", "worker.py", "model/weights.npz", "model/config.json"]
)
def test_changed_input_binding(setup: tuple[Path, Path, Path], name: str) -> None:
    request, source, output = setup
    resume.prepare(request, source, output)
    path = request.parent / name
    path.write_bytes(path.read_bytes() + b" ")
    with pytest.raises(ReplayError):
        resume.run(request, output, runtime=Runtime())
    assert not (output / "complete.json").exists()


def test_interrupted_successor_requires_explicit_new_namespace(
    setup: tuple[Path, Path, Path],
) -> None:
    request, source, output = setup
    resume.prepare(request, source, output)
    runtime = Runtime(fail=True)
    with pytest.raises(ReplayError, match="interrupted"):
        resume.run(request, output, runtime=runtime)
    with pytest.raises(ReplayError, match="no retry"):
        resume.run(request, output, runtime=runtime)
    assert runtime.calls == 1
    assert (
        lane._read(output / "000000480000.metal-runtime/failed.json")["automaticRetryAllowed"]
        is False
    )
    successor = output.parent / "successor"
    resume.prepare(request, output, successor)
    fresh = Runtime()
    resume.run(request, successor, runtime=fresh)
    assert fresh.calls == 1
    assert not (successor / "000000000000.metal-runtime").exists()
    assert asr_assembly.assemble(successor, successor.parent / "successor-labels.json") == 0


def test_completed_fresh_receipt_reused_in_successor(setup: tuple[Path, Path, Path]) -> None:
    request, source, output = setup
    resume.prepare(request, source, output)
    resume.run(request, output, runtime=Runtime())
    successor = output.parent / "successor"
    resume.prepare(request, output, successor)
    runtime = Runtime()
    resume.run(request, successor, runtime=runtime)
    assert runtime.calls == 0
    assert lane._read(successor / "000000480000.receipt.json")["kind"] == "retained"


@pytest.mark.parametrize(
    "change",
    [
        "inputs.json",
        "000000000000.raw.json",
        "000000000000.clock.json",
        "000000000000.receipt.json",
        "complete.json",
    ],
)
def test_changed_new_receipts_rejected(setup: tuple[Path, Path, Path], change: str) -> None:
    request, source, output = setup
    resume.prepare(request, source, output)
    resume.run(request, output, runtime=Runtime())
    replace(output / change, {"changed": True})
    with pytest.raises(ReplayError):
        resume.run(request, output, verify_only=True)


def test_source_change_after_preparation(setup: tuple[Path, Path, Path]) -> None:
    request, source, output = setup
    resume.prepare(request, source, output)
    durable(source / "000000000000.metal-runtime/failed.json", {"new": True})
    with pytest.raises(ReplayError, match="snapshot"):
        resume.run(request, output, runtime=Runtime())


def test_collector_lease_live_child_and_unknown_launch_rejected(
    setup: tuple[Path, Path, Path],
) -> None:
    request, source, output = setup
    descriptor = os.open(source / ".run.lock", os.O_RDWR)
    try:
        fcntl.flock(descriptor, fcntl.LOCK_EX | fcntl.LOCK_NB)
        with pytest.raises(BlockingIOError):
            resume.prepare(request, source, output)
    finally:
        os.close(descriptor)
    host = MagicMock()
    host.absent.return_value = False
    with pytest.raises(ReplayError, match="still present"):
        resume.prepare(request, source, output, host=host)
    uncertain = source / "000000480000.metal-runtime"
    uncertain.mkdir()
    with pytest.raises(ReplayError, match="lacks child"):
        resume.prepare(request, source, output)


def test_prepare_disjoint_local_and_verification_missing(setup: tuple[Path, Path, Path]) -> None:
    request, source, output = setup
    with pytest.raises(ReplayError, match="disjoint"):
        resume.prepare(request, source, source / "nested")
    with pytest.raises(ReplayError, match="Work-local"):
        resume.prepare(request, source, request.parent / "outside")
    resume.prepare(request, source, output)
    with pytest.raises(ReplayError, match="Missing completed"):
        resume.run(request, output, verify_only=True)


def test_limit_reuses_windows_and_invalid_limits(setup: tuple[Path, Path, Path]) -> None:
    request, source, output = setup
    resume.prepare(request, source, output)
    with pytest.raises(ReplayError, match="limit"):
        resume.run(request, output, maximum_new_windows=0)
    with pytest.raises(ReplayError, match="limit"):
        resume.run(request, output, maximum_new_windows=1, verify_only=True)
    runtime = Runtime()
    resume.run(request, output, runtime=runtime, maximum_new_windows=1)
    assert runtime.calls == 1


def test_child_semantic_binding_after_resealed_evidence(setup: tuple[Path, Path, Path]) -> None:
    request, source, output = setup
    work = source / "000000000000.metal-runtime"
    child = lane._read(work / "child.json")
    child["requestSHA256"] = "a" * 64
    replace(work / "child.json", child)
    receipt = lane._read(work / "complete.json")
    receipt["evidence"] = lane._evidence(work)
    replace(work / "complete.json", receipt)
    with pytest.raises(ReplayError, match="child request"):
        resume.prepare(request, source, output)


def test_cli_modes_and_required_options(monkeypatch: pytest.MonkeyPatch) -> None:
    prepare = MagicMock()
    run = MagicMock()
    monkeypatch.setattr(resume, "prepare", prepare)
    monkeypatch.setattr(resume, "run", run)
    assert resume.main(["prepare", "request", "output", "--source", "source"]) == 0
    prepare.assert_called_once_with(Path("request"), Path("source"), Path("output"))
    assert resume.main(["run", "request", "output", "--maximum-new-windows", "1"]) == 0
    run.assert_called_once_with(
        Path("request"), Path("output"), verify_only=False, maximum_new_windows=1
    )
    assert resume.main(["verify", "request", "output"]) == 0
    with pytest.raises(ReplayError, match="requires"):
        resume.main(["prepare", "request", "output"])
    with pytest.raises(ReplayError, match="bound"):
        resume.main(["run", "request", "output", "--source", "source"])


def test_process_group_quiescence(monkeypatch: pytest.MonkeyPatch) -> None:
    process = MagicMock(return_value=SimpleNamespace(stdout="20 20\n21 20\n"))
    monkeypatch.setattr(subprocess, "run", process)
    resume._group_gone(19)
    with pytest.raises(ReplayError, match="group still"):
        resume._group_gone(20)
    process.return_value.stdout = "unparseable"
    with pytest.raises(ReplayError, match="Invalid process"):
        resume._group_gone(19)
