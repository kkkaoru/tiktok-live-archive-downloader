"""Original CPU producer/assembler with synthetic PCM and no MLX/process actions."""

import json
import wave
from pathlib import Path
from unittest.mock import MagicMock

import pytest

from replay import asr_assembly, isolated_cpu_asr
from replay import isolated_metal_asr as metal
from replay import measured_job_admission as admission
from replay import mixed_isolated_asr as mixed
from replay.audio_windows import AudioWindow
from replay.core import ReplayError
from replay.native_invocation import durable, pin
from replay.stages import execution_lock


def write(path: Path, data: object) -> None:
    path.write_text(json.dumps(data), encoding="utf-8")


def child(command: tuple[str, ...], log: Path, timeout: int) -> None:
    work = log.parent
    request = metal._read(work / "request.json")
    raw = {
        "segments": [
            {
                "start": 1,
                "end": 2,
                "text": "声",
                "words": [{"word": "声", "start": 1, "end": 2, "probability": 0.9}],
            }
        ],
        "text": "声",
        "language": "ja",
    }
    durable(work / "raw.json", raw)
    durable(
        work / "result.json",
        {
            "rawSHA256": pin(work / "raw.json")["sha256"],
            "samplesSHA256": request["samplesSHA256"],
            "producerSHA256": request["producerSHA256"],
            "device": "cpu",
            "deliveryComplete": False,
        },
    )
    log.write_text("synthetic CPU child\n")
    (work / "runtime").mkdir()
    for phase in metal.PHASES:
        durable(work / "runtime" / f"{phase}-descriptors.json", {"externalWritableRegular": []})


@pytest.fixture
def fixture(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> tuple[Path, Path, Path]:
    source = tmp_path / "source.wav"
    with wave.open(str(source), "wb") as audio:
        audio.setparams((1, 2, 16000, 0, "NONE", "not compressed"))
        audio.writeframes(b"\x01\x00" * 91 * 16000)
    model = tmp_path / "model"
    model.mkdir()
    weights = model / "weights.npz"
    weights.write_bytes(b"synthetic")
    config = model / "config.json"
    config.write_text("{}")
    cpu, policy = tmp_path / "cpu", tmp_path / "policy"
    isolated_cpu_asr.run(
        source,
        cpu,
        policy,
        limit=1,
        model=model,
        weights_sha256=str(pin(weights)["sha256"]),
        config_sha256=str(pin(config)["sha256"]),
        execute=child,
    )
    monkeypatch.setattr(metal, "DEFAULT_CONFIG_SHA256", pin(config)["sha256"])
    monkeypatch.setattr(metal, "DEFAULT_WEIGHTS_SHA256", pin(weights)["sha256"])
    code = Path(metal.__file__).resolve()
    monkeypatch.setattr(metal, "required_code", lambda: (code,))
    metal_request = tmp_path / "metal.json"
    durable(
        metal_request,
        {
            "version": 1,
            "device": "metal",
            "fp16": True,
            "samples": 91 * 16000,
            "timeoutSeconds": 30,
            "inputs": {
                name: pin(path)
                for name, path in (("pcm", source), ("weights", weights), ("config", config))
            },
            "codeInputs": {str(code): pin(code)["sha256"]},
        },
    )
    capture = tmp_path / "capture.json"
    workflow = tmp_path / "workflow.json"
    durable(
        workflow,
        {
            "version": 1,
            "stages": [
                {
                    "name": "asr",
                    "outputs": [
                        str(cpu / "inputs.json"),
                        str(cpu / "complete.json"),
                        str(policy / "started.json"),
                    ],
                }
            ],
        },
    )
    durable(capture, {"syntheticOwnerCapture": True, "workflow": pin(workflow), "stage": "asr"})
    request = tmp_path / "mixed.json"
    own = Path(mixed.__file__).resolve()
    code_paths = (
        own,
        Path(admission.__file__).resolve(),
        Path(isolated_cpu_asr.__file__).resolve(),
        own.with_name("leaf_retirement.py"),
        own.with_name("phase_timer.py"),
    )
    durable(
        request,
        {
            "version": 1,
            "kind": "mixed-isolated-asr",
            "metalRequest": pin(metal_request),
            "cpuCollection": str(cpu),
            "cpuPolicyStarted": pin(policy / "started.json"),
            "ownerCaptures": [pin(capture)],
            "codeInputs": {str(path): pin(path)["sha256"] for path in code_paths},
        },
    )
    return request, tmp_path / "mixed", cpu


class Quiet:
    def verify(
        self,
        captures: tuple[dict[str, object], ...],
        files: tuple[Path, ...],
        *,
        maximum_stop_age: int | None = None,
    ) -> dict[str, object]:
        return {"synthetic": True, "groupsAbsent": True, "openFiles": []}


class GPU:
    def __init__(self, fail: bool = False) -> None:
        self.calls = 0
        self.fail = fail

    def execute(
        self, request: metal.Request, window: AudioWindow, work: Path, identities: dict[str, object]
    ) -> None:
        self.calls += 1
        if self.fail:
            raise ReplayError("synthetic interrupted Metal")
        durable(work / "raw.json", {"segments": [], "text": "", "language": "ja"})
        (work / "worker.log").write_bytes(b"")
        durable(
            work / "resources.json",
            {"exitCode": 0, "familyGone": True, "samples": [{"rssBytes": 1}]},
        )
        durable(
            work / "mlx-memory.json",
            {"device": "metal", "activeBytes": 1, "peakBytes": 2, "cacheBytes": 1},
        )
        for phase in metal.PHASES:
            durable(work / f"{phase}-descriptors.json", {"externalWritableRegular": []})
        durable(work / "child.json", {"synthetic": True})
        durable(work / "launch.json", {"synthetic": True})


def test_genuine_cpu_adoption_then_missing_metal_and_original_assembly(
    fixture: tuple[Path, Path, Path],
) -> None:
    request, output, cpu = fixture
    before = (cpu / "000000000000.raw.json").read_bytes()
    adoption = mixed.adopt(request, output, host=Quiet())
    assert (output / "000000000000.raw.json").read_bytes() == before
    assert metal._read(adoption)["skippedStartSamples"] == [480000, 960000]
    gpu = GPU()
    partial = mixed.run(request, output, runtime=gpu, maximum_new_windows=1)
    assert metal._read(partial)["collectionComplete"] is False
    assert not (output / "complete.json").exists()
    complete = mixed.run(request, output, runtime=gpu)
    assert gpu.calls == 2
    mixed.run(request, output, verify_only=True)
    mixed.run(request, output, runtime=gpu)
    mixed.adopt(request, output, host=Quiet())
    assert gpu.calls == 2
    assert metal._read(complete)["device"] == "mixed-cpu-metal"
    labels = output.parent / "assembled.json"
    assert asr_assembly.assemble(output, labels) == 1
    assert metal._read(labels)["labels"] == [
        {"text": "声", "start": 1.0, "end": 2.0, "provenance": "000000000000:0"}
    ]
    assert metal._read(labels)["speechAccuracyVerified"] is False
    assert (cpu / "000000000000.raw.json").read_bytes() == before
    assert not (cpu / "complete.json").exists()


def test_live_lease_refuses_even_terminal_claim(fixture: tuple[Path, Path, Path]) -> None:
    request, output, cpu = fixture
    with execution_lock(cpu, ".run") as acquired:
        assert acquired
        with pytest.raises(ReplayError, match="owned"):
            mixed.adopt(request, output, host=Quiet())
    assert not output.exists()


def test_host_failure_never_creates_output(fixture: tuple[Path, Path, Path]) -> None:
    request, output, _ = fixture
    host = MagicMock()
    host.verify.side_effect = ReplayError("live descendants")
    with pytest.raises(ReplayError, match="descendants"):
        mixed.adopt(request, output, host=host)
    assert not output.exists()


@pytest.mark.parametrize(
    "target",
    [
        "000000000000.raw.json",
        "000000000000.clock.json",
        "000000000000.cpu-runtime/request.json",
        "000000000000.cpu-runtime/result.json",
        "000000000000.cpu-runtime/runtime/after-inference-eval-descriptors.json",
        "inputs.json",
    ],
)
def test_changed_completed_cpu_rejected(fixture: tuple[Path, Path, Path], target: str) -> None:
    request, output, cpu = fixture
    write(cpu / target, {"changed": True})
    with pytest.raises(ReplayError):
        mixed.adopt(request, output, host=Quiet())
    assert not output.exists()


def test_partial_window_preserved_not_imported(fixture: tuple[Path, Path, Path]) -> None:
    request, output, cpu = fixture
    pending = cpu / "000000480000.cpu-runtime"
    pending.mkdir()
    (pending / "worker.log").write_text("interrupted")
    durable(cpu / "000000480000.raw.json", {"segments": []})
    mixed.adopt(request, output, host=Quiet())
    assert (pending / "worker.log").read_text() == "interrupted"
    assert not (output / "000000480000.raw.json").exists()


def test_payload_not_original_pcm_rejected(fixture: tuple[Path, Path, Path]) -> None:
    request, output, cpu = fixture
    payload = cpu / "000000000000.cpu-runtime/samples.npy"
    blob = bytearray(payload.read_bytes())
    blob[-4:] = b"\0\0\0\0"
    payload.write_bytes(blob)
    request_path = payload.parent / "request.json"
    data = metal._read(request_path)
    data["samplesSHA256"] = pin(payload)["sha256"]
    write(request_path, data)
    result = metal._read(payload.parent / "result.json")
    result["samplesSHA256"] = data["samplesSHA256"]
    write(payload.parent / "result.json", result)
    with pytest.raises(ReplayError, match="original PCM"):
        mixed.adopt(request, output, host=Quiet())


def test_interrupted_metal_never_repeated(fixture: tuple[Path, Path, Path]) -> None:
    request, output, _ = fixture
    mixed.adopt(request, output, host=Quiet())
    gpu = GPU(fail=True)
    with pytest.raises(ReplayError, match="interrupted"):
        mixed.run(request, output, runtime=gpu)
    with pytest.raises(ReplayError, match="no retry"):
        mixed.run(request, output, runtime=gpu)
    assert gpu.calls == 1


@pytest.mark.parametrize(
    "target",
    [
        "000000000000.raw.json",
        "000000000000.clock.json",
        "000000000000.cpu-provenance.json",
        "inputs.json",
        "complete.json",
    ],
)
def test_mixed_tamper_refused(fixture: tuple[Path, Path, Path], target: str) -> None:
    request, output, _ = fixture
    mixed.adopt(request, output, host=Quiet())
    mixed.run(request, output, runtime=GPU())
    write(output / target, {"changed": True})
    with pytest.raises(ReplayError):
        mixed.run(request, output, verify_only=True)


@pytest.mark.parametrize(
    "field,value",
    [("kind", "cpu"), ("ownerCaptures", []), ("codeInputs", {}), ("cpuCollection", "/outside")],
)
def test_invalid_mixed_request(fixture: tuple[Path, Path, Path], field: str, value: object) -> None:
    request, output, _ = fixture
    data = metal._read(request)
    data[field] = value
    write(request, data)
    with pytest.raises(ReplayError):
        mixed.adopt(request, output, host=Quiet())


@pytest.mark.parametrize("mode", ["success", "open", "unknown", "active", "changed"])
def test_system_quiescence_shared_release_and_exact_files(
    fixture: tuple[Path, Path, Path], monkeypatch: pytest.MonkeyPatch, mode: str
) -> None:
    request, _, cpu = fixture
    plan = mixed.load(request)
    release = {"resourceReleased": mode != "active", "stageQualityApproved": False}
    verifier = MagicMock(return_value=release)
    if mode == "changed":
        verifier.side_effect = [release, release | {"changed": True}]
    monkeypatch.setattr(mixed, "_release", verifier)
    monkeypatch.setattr(
        mixed,
        "lsof_open_among",
        MagicMock(
            return_value={str(cpu / "inputs.json")}
            if mode == "open"
            else None
            if mode == "unknown"
            else set()
        ),
    )
    if mode == "success":
        result = mixed.SystemQuiescence().verify(plan.captures, (cpu / "inputs.json",))
        assert result["groupsAbsent"] is True
        assert verifier.call_count == 2
    else:
        with pytest.raises(ReplayError):
            mixed.SystemQuiescence().verify(plan.captures, (cpu / "inputs.json",))


def test_cli_typed_modes(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    adopt = MagicMock()
    run = MagicMock()
    monkeypatch.setattr(mixed, "adopt", adopt)
    monkeypatch.setattr(mixed, "run", run)
    assert mixed.main(["adopt", str(tmp_path / "r"), str(tmp_path / "o")]) == 0
    adopt.assert_called_once()
    assert mixed.main(["verify", str(tmp_path / "r"), str(tmp_path / "o")]) == 0
    assert run.call_args.kwargs == {"verify_only": True, "maximum_new_windows": None}
    with pytest.raises(ReplayError):
        mixed.main(
            ["adopt", str(tmp_path / "r"), str(tmp_path / "o"), "--maximum-new-windows", "1"]
        )


@pytest.fixture
def stopped(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> tuple[Path, Path, admission.Parent, admission.Completion]:
    import hashlib
    import sys
    import time

    capture = tmp_path / "stopped-capture.json"
    status = tmp_path / "job.status"
    checkpoint = tmp_path / "asr.json"
    workflow = tmp_path / "workflow.json"
    command = f"{Path(sys.executable).resolve()} -B -m replay.isolated_cpu_asr input output\n"
    parent = admission.Parent(
        500, "1:2", Path(sys.executable).resolve(), hashlib.sha256(command.encode()).hexdigest()
    )
    durable(
        workflow,
        {
            "version": 1,
            "stages": [
                {
                    "name": "asr",
                    "command": [
                        "/usr/bin/env",
                        "TMPDIR=synthetic",
                        str(Path(sys.executable).resolve()),
                        "-B",
                        "-m",
                        "replay.phase_timer",
                        "phases.jsonl",
                        "recognize",
                        "--",
                        str(Path(sys.executable).resolve()),
                        "-B",
                        "-m",
                        "replay.isolated_cpu_asr",
                        "input",
                        "output",
                    ],
                }
            ],
        },
    )
    durable(capture, {"observedAtNs": time.time_ns(), "workflow": pin(workflow), "stage": "asr"})
    durable(checkpoint, {"status": "running", "fingerprint": "a" * 64})
    completion = admission.Completion(
        capture,
        str(pin(capture)["sha256"]),
        workflow,
        str(pin(workflow)["sha256"]),
        checkpoint,
        status,
        "a" * 64,
        (),
        (500, 501),
    )
    monkeypatch.setattr(
        admission, "load_completion_capture", MagicMock(return_value=(parent, completion))
    )
    host = MagicMock()
    host.identity.return_value = parent
    monkeypatch.setattr(admission, "SystemHost", MagicMock(return_value=host))
    monkeypatch.setattr(admission, "probe", MagicMock(side_effect=["T\n", command, "T\n"]))
    return capture, tmp_path / "barrier.json", parent, completion


def test_actual_shaped_stopped_capture_and_interrupted_release(
    stopped: tuple[Path, Path, admission.Parent, admission.Completion],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    capture, barrier, _, completion = stopped
    mixed.capture_barrier(capture, barrier)
    completion.status.write_text("143\n")
    monkeypatch.setattr(admission, "probe", MagicMock(return_value="1 1\n20 20\n"))
    result = mixed._release(capture, str(pin(capture)["sha256"]), 30, pin(barrier))
    assert result["termination"] == "interrupted" and result["stageQualityApproved"] is False
    with pytest.raises(ReplayError, match="requires fresh"):
        mixed._release(capture, str(pin(capture)["sha256"]), None)
    completion.status.write_text("0\n")
    verifier = MagicMock(return_value={"resourceReleased": True})
    monkeypatch.setattr(admission, "verify_completion_capture", verifier)
    assert mixed._release(capture, str(pin(capture)["sha256"]), None) == {"resourceReleased": True}


@pytest.mark.parametrize(
    "failure", ["live", "old", "wrong-checkpoint", "bad-status", "missing-barrier", "barrier-state"]
)
def test_interrupted_release_refuses_uncertain_boundary(
    stopped: tuple[Path, Path, admission.Parent, admission.Completion],
    monkeypatch: pytest.MonkeyPatch,
    failure: str,
) -> None:
    import os

    capture, barrier, _, completion = stopped
    mixed.capture_barrier(capture, barrier)
    completion.status.write_text("143\n")
    monkeypatch.setattr(
        admission, "probe", MagicMock(return_value="500 500\n" if failure == "live" else "1 1\n")
    )
    if failure == "old":
        now = completion.status.stat().st_mtime_ns
        os.utime(completion.status, ns=(now + 60_000_000_000, now + 60_000_000_000))
    elif failure == "wrong-checkpoint":
        write(completion.checkpoint, {"status": "running", "fingerprint": "b" * 64})
    elif failure == "bad-status":
        completion.status.write_text("unknown")
    elif failure == "barrier-state":
        value = metal._read(barrier)
        value["state"] = "R"
        write(barrier, value)
    with pytest.raises(ReplayError):
        mixed._release(
            capture,
            str(pin(capture)["sha256"]),
            30,
            None if failure == "missing-barrier" else pin(barrier),
        )


@pytest.mark.parametrize("failure", ["running", "wrong-command", "changed-state"])
def test_barrier_never_stops_or_accepts_wrong_actor(
    stopped: tuple[Path, Path, admission.Parent, admission.Completion],
    monkeypatch: pytest.MonkeyPatch,
    failure: str,
) -> None:
    capture, barrier, parent, _ = stopped
    if failure == "running":
        responses = ["R\n"]
    elif failure == "wrong-command":
        responses = ["T\n", "different command\n"]
    else:
        responses = [
            "T\n",
            f"{parent.executable} -B -m replay.isolated_cpu_asr input output\n",
            "R\n",
        ]
    monkeypatch.setattr(admission, "probe", MagicMock(side_effect=responses))
    with pytest.raises(ReplayError):
        mixed.capture_barrier(capture, barrier)
    assert not barrier.exists()


def test_wrong_owner_collection_refused(fixture: tuple[Path, Path, Path]) -> None:
    request, output, _ = fixture
    value = metal._read(request)
    refs = value["ownerCaptures"]
    assert isinstance(refs, list)
    capture = Path(str(refs[0]["path"]))
    captured = metal._read(capture)
    captured["stage"] = "unrelated"
    write(capture, captured)
    value["ownerCaptures"] = [pin(capture)]
    write(request, value)
    with pytest.raises(ReplayError, match="missing/ambiguous"):
        mixed.adopt(request, output, host=Quiet())


def test_adoption_resume_preserves_original_barrier_and_bytes(
    fixture: tuple[Path, Path, Path], monkeypatch: pytest.MonkeyPatch
) -> None:
    request, output, cpu = fixture
    original = durable

    def interrupt(path: Path, value: dict[str, object]) -> None:
        if path.name == "adoption.json":
            raise ReplayError("synthetic crash before adoption commit")
        original(path, value)

    monkeypatch.setattr(mixed, "durable", interrupt)
    with pytest.raises(ReplayError, match="crash"):
        mixed.adopt(request, output, host=Quiet())
    before = pin(output / "quiescence.json")
    raw_before = pin(output / "000000000000.raw.json")
    monkeypatch.setattr(mixed, "durable", original)
    mixed.adopt(request, output, host=Quiet())
    assert pin(output / "quiescence.json") == before
    assert pin(output / "000000000000.raw.json") == raw_before
    assert (cpu / "000000000000.raw.json").read_bytes() == (
        output / "000000000000.raw.json"
    ).read_bytes()


def test_barrier_cli_does_not_dispatch(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    capture = MagicMock()
    monkeypatch.setattr(mixed, "capture_barrier", capture)
    assert mixed.main(["capture-barrier", str(tmp_path / "capture"), str(tmp_path / "out")]) == 0
    capture.assert_called_once()


def test_quiescence_evidence_tamper_fails_before_metal(fixture: tuple[Path, Path, Path]) -> None:
    request, output, _ = fixture
    mixed.adopt(request, output, host=Quiet())
    write(output / "quiescence.json", {"changed": True})
    gpu = GPU()
    with pytest.raises(ReplayError, match="commit"):
        mixed.run(request, output, runtime=gpu)
    assert gpu.calls == 0
