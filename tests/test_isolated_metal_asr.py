"""Synthetic process and real assembly contracts; never imports MLX or launches it."""

import importlib
import json
import os
import subprocess
import time
import wave
from contextlib import nullcontext
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

from replay import asr_assembly
from replay import isolated_metal_asr as lane
from replay.audio_windows import AudioWindow
from replay.core import ReplayError
from replay.fcp_recovery_runtime import SystemHost
from replay.isolated_metal_asr_probe import planned_environment
from replay.native_invocation import durable, pin
from replay.native_sequence import ContractError
from replay.recordings import record


def write(path: Path, value: object) -> None:
    path.write_text(json.dumps(value), encoding="utf-8")


@pytest.fixture
def setup(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> tuple[Path, Path]:
    monkeypatch.chdir(tmp_path)
    model = tmp_path / "model"
    model.mkdir()
    config = model / "config.json"
    config.write_text("{}")
    weights = model / "weights.npz"
    weights.write_bytes(b"synthetic weights")
    pcm = tmp_path / "input.wav"
    with wave.open(str(pcm), "wb") as audio:
        audio.setnchannels(1)
        audio.setsampwidth(2)
        audio.setframerate(lane.RATE)
        audio.writeframes(b"\x00\x00" * lane.RATE)
    code = tmp_path / "producer.py"
    code.write_text("# synthetic producer")
    monkeypatch.setattr(lane, "required_code", lambda: (code,))
    monkeypatch.setattr(lane, "DEFAULT_CONFIG_SHA256", pin(config)["sha256"])
    monkeypatch.setattr(lane, "DEFAULT_WEIGHTS_SHA256", pin(weights)["sha256"])
    request = tmp_path / "request.json"
    write(
        request,
        {
            "version": 1,
            "device": "metal",
            "fp16": True,
            "inputs": {
                name: pin(path)
                for name, path in (("pcm", pcm), ("weights", weights), ("config", config))
            },
            "samples": lane.RATE,
            "timeoutSeconds": 10,
            "codeInputs": {str(code): pin(code)["sha256"]},
        },
    )
    output = tmp_path / "edits/work/collection"
    return request, output


class FakeRuntime:
    def __init__(self, *, fail: bool = False) -> None:
        self.calls = 0
        self.fail = fail

    def execute(
        self, request: lane.Request, window: AudioWindow, work: Path, identities: dict[str, object]
    ) -> None:
        self.calls += 1
        if self.fail:
            raise ReplayError("synthetic uncertain effect")
        durable(work / "raw.json", {"text": "", "segments": [], "language": "ja"})
        (work / "worker.log").write_bytes(b"")
        durable(
            work / "resources.json",
            {"exitCode": 0, "familyGone": True, "samples": [{"physicalBytes": 42}]},
        )
        durable(
            work / "mlx-memory.json",
            {"device": "metal", "activeBytes": 1, "peakBytes": 2, "cacheBytes": 3},
        )
        for phase in lane.PHASES:
            durable(work / f"{phase}-descriptors.json", {"externalWritableRegular": []})
        for name in ("child.json", "launch.json"):
            durable(work / name, {"synthetic": True})


def test_real_assembly_and_idempotent_resume(setup: tuple[Path, Path]) -> None:
    request, output = setup
    runtime = FakeRuntime()
    complete = lane.collect(request, output, runtime=runtime)
    assert runtime.calls == 1
    before = pin(complete)
    lane.collect(request, output, runtime=runtime)
    lane.collect(request, output, verify_only=True)
    assert runtime.calls == 1 and pin(complete) == before
    assert asr_assembly.assemble(output, output.parent / "labels.json") == 0
    manifest = lane._read(output / "inputs.json")
    assert manifest["device"] == "metal" and manifest["clock"] == "input-pcm-not-projected"
    assert not list(output.glob("*.cpu-runtime"))


def test_partial_failure_never_resubmits(setup: tuple[Path, Path]) -> None:
    request, output = setup
    runtime = FakeRuntime(fail=True)
    with pytest.raises(ReplayError, match="uncertain"):
        lane.collect(request, output, runtime=runtime)
    with pytest.raises(ReplayError, match="no retry"):
        lane.collect(request, output, runtime=runtime)
    assert runtime.calls == 1 and not (output / "complete.json").exists()


@pytest.mark.parametrize(
    "field,value",
    [
        ("device", "cpu"),
        ("fp16", False),
        ("samples", True),
        ("samples", 0),
        ("timeoutSeconds", 3601),
        ("cpuAdoption", {}),
        ("codeInputs", {}),
        ("inputs", {}),
        ("version", 2),
    ],
)
def test_invalid_contract(setup: tuple[Path, Path], field: str, value: object) -> None:
    request, _ = setup
    data = lane._read(request)
    data[field] = value
    write(request, data)
    with pytest.raises((ReplayError, FileNotFoundError)):
        lane.load_request(request)


@pytest.mark.parametrize(
    "mutation", ["hash", "source", "model", "code", "clock", "link", "duplicate"]
)
def test_input_integrity(setup: tuple[Path, Path], mutation: str) -> None:
    request, _ = setup
    data = lane._read(request)
    parsed = lane.load_request(request)
    if mutation == "hash":
        data["codeInputs"] = {str(next(reversed(parsed.pins))): "z" * 64}
    elif mutation == "source":
        parsed.pcm.write_bytes(b"changed")
    elif mutation == "model":
        inputs = record(data["inputs"])
        record(inputs["weights"])["sha256"] = "a" * 64
        # record() returns a checked copy; replace the original reference explicitly.
        inputs["weights"] = {"path": str(parsed.weights), "sha256": "a" * 64}
        data["inputs"] = inputs
    elif mutation == "code":
        next(reversed(parsed.pins)).write_text("changed")
    elif mutation == "clock":
        data["samples"] = 1
    elif mutation == "link":
        linked = parsed.pcm.with_name("linked.wav")
        linked.symlink_to(parsed.pcm)
        inputs = record(data["inputs"])
        inputs["pcm"] = {"path": str(linked), "sha256": parsed.pins[parsed.pcm]}
        data["inputs"] = inputs
    else:
        request.write_text('{"version":1,"version":1}')
        with pytest.raises(ContractError, match="Duplicate"):
            lane.load_request(request)
        return
    write(request, data)
    with pytest.raises(ReplayError):
        lane.load_request(request)


def test_request_and_identity_changed(setup: tuple[Path, Path]) -> None:
    request, _ = setup
    parsed = lane.load_request(request)
    observed = lane.verify_inputs(parsed)
    request.write_text(request.read_text() + " ")
    with pytest.raises(ReplayError, match="Request changed"):
        lane.verify_inputs(parsed)
    parsed.config.write_text("{ }")
    with pytest.raises(ReplayError, match="identity"):
        lane._identities(observed)


@pytest.mark.parametrize(
    "name",
    [
        "raw.json",
        "mlx-memory.json",
        "resources.json",
        "intent.json",
        "child.json",
        "launch.json",
        "before-core-import-descriptors.json",
        "worker.log",
    ],
)
def test_tampered_child_evidence_rejected(setup: tuple[Path, Path], name: str) -> None:
    request, output = setup
    lane.collect(request, output, runtime=FakeRuntime())
    path = output / "000000000000.metal-runtime" / name
    if name == "worker.log":
        path.write_text("sandbox: deny file-write")
    else:
        write(path, {"changed": True})
    with pytest.raises(ReplayError):
        lane.collect(request, output, verify_only=True)


@pytest.mark.parametrize(
    "name", ["inputs.json", "000000000000.raw.json", "000000000000.clock.json", "complete.json"]
)
def test_tampered_collection(setup: tuple[Path, Path], name: str) -> None:
    request, output = setup
    lane.collect(request, output, runtime=FakeRuntime())
    write(output / name, {"changed": True})
    with pytest.raises(ReplayError):
        lane.collect(request, output, verify_only=True)


def test_missing_collection_verify_writes_nothing(setup: tuple[Path, Path]) -> None:
    request, output = setup
    with pytest.raises(ReplayError):
        lane.collect(request, output, verify_only=True)
    assert not output.exists()


def test_completed_child_before_publication_reuses(
    setup: tuple[Path, Path], monkeypatch: pytest.MonkeyPatch
) -> None:
    request, output = setup
    original = lane._same

    def stop(path: Path, data: dict[str, object]) -> None:
        if path.name.endswith("raw.json"):
            raise ReplayError("synthetic crash before raw publication")
        original(path, data)

    monkeypatch.setattr(lane, "_same", stop)
    runtime = FakeRuntime()
    with pytest.raises(ReplayError, match="crash"):
        lane.collect(request, output, runtime=runtime)
    monkeypatch.setattr(lane, "_same", original)
    lane.collect(request, output, runtime=runtime)
    assert runtime.calls == 1


def child_setup(request: Path, work: Path) -> Path:
    work.mkdir(parents=True)
    parsed = lane.load_request(request)
    child = work / "child.json"
    durable(
        child,
        {
            "request": str(request),
            "requestSHA256": parsed.sha256,
            "startSample": 0,
            "sampleCount": parsed.samples,
            "identities": lane.verify_inputs(parsed),
        },
    )
    for name in ("scratch", "cache", "numba", "huggingface"):
        (work / name).mkdir()
    return child


def test_worker_real_clock_and_mlx_memory(
    setup: tuple[Path, Path], monkeypatch: pytest.MonkeyPatch
) -> None:
    request, work = setup
    child = child_setup(request, work)
    for key, value in planned_environment(work, Path.cwd()).items():
        monkeypatch.setenv(key, value)
    mx = MagicMock()
    mx.stream.return_value = nullcontext()
    mx.get_active_memory.return_value = 11
    mx.get_peak_memory.return_value = 17
    mx.get_cache_memory.return_value = 5
    provider = MagicMock()
    provider.transcribe.return_value = {"segments": [], "text": "", "language": "ja"}
    importer = MagicMock(side_effect=[mx, provider])
    audit = MagicMock()
    monkeypatch.setattr(importlib, "import_module", importer)
    monkeypatch.setattr(lane, "audit_descriptors", audit)
    lane.worker(child, work)
    assert audit.call_count == 4
    options = provider.transcribe.call_args.kwargs
    assert options["fp16"] is True and options["word_timestamps"] is True
    assert options["condition_on_previous_text"] is False
    assert len(provider.transcribe.call_args.args[0]) == lane.RATE
    assert lane._read(work / "mlx-memory.json")["peakBytes"] == 17
    mx.reset_peak_memory.assert_called_once()
    mx.synchronize.assert_called_once_with(mx.gpu)


@pytest.mark.parametrize("change", ["request", "identities", "window", "environment", "stream"])
def test_worker_rejects_unbound_boundary(
    setup: tuple[Path, Path], monkeypatch: pytest.MonkeyPatch, change: str
) -> None:
    request, work = setup
    child = child_setup(request, work)
    data = lane._read(child)
    for key, value in planned_environment(work, Path.cwd()).items():
        monkeypatch.setenv(key, value)
    if change == "request":
        data["requestSHA256"] = "b" * 64
    elif change == "identities":
        data["identities"] = {}
    elif change == "window":
        data["sampleCount"] = 1
    elif change == "environment":
        monkeypatch.setenv("HF_HUB_OFFLINE", "0")
    else:
        monkeypatch.setattr(lane, "audit_descriptors", MagicMock())
        mx = MagicMock()
        mx.stream.return_value = object()
        monkeypatch.setattr(importlib, "import_module", MagicMock(return_value=mx))
    write(child, data)
    with pytest.raises(ReplayError):
        lane.worker(child, work)


def test_family_observation_separates_physical_and_rss(monkeypatch: pytest.MonkeyPatch) -> None:
    host = MagicMock(spec=SystemHost)
    host.birth.side_effect = lambda pid: f"{pid}:100"
    host.footprint.side_effect = lambda pid: pid * 100
    monkeypatch.setattr(
        subprocess,
        "run",
        MagicMock(return_value=SimpleNamespace(stdout="10 1 8\n11 10 9\n20 1 99\n")),
    )
    result = lane.sample_family(10, "10:100", host)
    assert result["physicalBytes"] == 2100 and result["rssBytes"] == 17 * 1024
    with pytest.raises(ReplayError, match="reused"):
        lane.sample_family(10, "wrong", host)
    host.birth.side_effect = ["10:100", "10:100", "11:100", "changed", "changed"]
    with pytest.raises(ReplayError, match="Family changed"):
        lane.sample_family(10, "10:100", host)


@pytest.mark.parametrize("mode", ["success", "timeout", "failed", "unobserved", "remaining", "log"])
def test_owned_process_lifetime(
    setup: tuple[Path, Path], monkeypatch: pytest.MonkeyPatch, mode: str
) -> None:
    request, output = setup
    parsed = lane.load_request(request)
    output.mkdir(parents=True)
    child = MagicMock()
    child.pid = 321
    child.returncode = 1 if mode == "failed" else 0
    child.poll.side_effect = [0, 0] if mode == "unobserved" else [None, 0, 0]
    child.wait.return_value = child.returncode
    host = MagicMock()
    host.birth.return_value = "321:10"
    host.absent.return_value = mode != "remaining"
    monkeypatch.setattr(lane, "SystemHost", MagicMock(return_value=host))
    popen = MagicMock(return_value=child)
    monkeypatch.setattr(subprocess, "Popen", popen)
    monkeypatch.setattr(
        lane,
        "sample_family",
        MagicMock(
            return_value={
                "members": [{"pid": 321, "birth": "321:10"}],
                "rssBytes": 100,
                "physicalBytes": 90,
            }
        ),
    )
    monkeypatch.setattr(os, "killpg", MagicMock())
    if mode == "timeout":
        monkeypatch.setattr(time, "monotonic", MagicMock(side_effect=[0, 11, 12]))
        child.poll.side_effect = [None, None]
    if mode == "log":

        def growing(*args: object, **kwargs: object) -> MagicMock:
            (output / "worker.log").write_bytes(b"x" * (8 * 1024**2 + 1))
            popen(*args, **kwargs)
            return child

        monkeypatch.setattr(subprocess, "Popen", growing)
    if mode == "success":
        child.wait.side_effect = [subprocess.TimeoutExpired("fake", 1), 0]
        lane.MetalRuntime().execute(
            parsed, AudioWindow(0, parsed.samples), output, lane.verify_inputs(parsed)
        )
    else:
        with pytest.raises(ReplayError):
            lane.MetalRuntime().execute(
                parsed, AudioWindow(0, parsed.samples), output, lane.verify_inputs(parsed)
            )
    resources = lane._read(output / "resources.json")
    assert resources["absoluteFamilyPeakVerified"] is False
    assert resources["familyGone"] is (mode == "success")
    assert popen.call_args.kwargs["start_new_session"] is True
    assert str(output / "sandbox.sb") in popen.call_args.args[0]


def test_cli_modes_and_required_code(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    assert any(path.name == "transcribe.py" for path in lane.required_code())
    collect = MagicMock()
    worker = MagicMock()
    monkeypatch.setattr(lane, "collect", collect)
    monkeypatch.setattr(lane, "worker", worker)
    assert lane.main([str(tmp_path / "r"), str(tmp_path / "o"), "--verify"]) == 0
    assert collect.call_args.kwargs == {"verify_only": True, "maximum_new_windows": None}
    assert lane.main([str(tmp_path / "r"), str(tmp_path / "o"), "--worker"]) == 0
    worker.assert_called_once()


def test_overlapping_windows_keep_native_points_and_clock(setup: tuple[Path, Path]) -> None:
    request, output = setup
    data = lane._read(request)
    pcm = request.parent / "input.wav"
    with wave.open(str(pcm), "wb") as audio:
        audio.setparams((1, 2, lane.RATE, 0, "NONE", "not compressed"))
        audio.writeframes(b"\0\0" * (61 * lane.RATE))
    inputs = record(data["inputs"])
    inputs["pcm"] = pin(pcm)
    data["inputs"] = inputs
    data["samples"] = 61 * lane.RATE
    write(request, data)

    class SpeakingRuntime(FakeRuntime):
        def execute(
            self,
            request: lane.Request,
            window: AudioWindow,
            work: Path,
            identities: dict[str, object],
        ) -> None:
            super().execute(request, window, work, identities)
            start = 1 if window.start_sample == 0 else 15
            write(
                work / "raw.json",
                {
                    "language": "ja",
                    "text": "声点",
                    "segments": [
                        {
                            "start": start,
                            "end": start + 3,
                            "text": "声点",
                            "words": [
                                {
                                    "word": "声",
                                    "start": start,
                                    "end": start + 1,
                                    "probability": 0.9,
                                },
                                {
                                    "word": "点",
                                    "start": start + 3,
                                    "end": start + 3,
                                    "probability": 0.8,
                                },
                            ],
                        }
                    ],
                },
            )

    runtime = SpeakingRuntime()
    partial = lane.collect(request, output, runtime=runtime, maximum_new_windows=1)
    assert lane._read(partial)["collectionComplete"] is False
    assert runtime.calls == 1 and not (output / "complete.json").exists()
    lane.collect(request, output, runtime=runtime, maximum_new_windows=1)
    assert runtime.calls == 2
    labels = output.parent / "labels.json"
    assert asr_assembly.assemble(output, labels) == 4
    saved = lane._read(labels)
    assert saved["labels"] == [
        {"text": "声", "start": 1.0, "end": 2.0, "provenance": "000000000000:0"},
        {"text": "点", "start": 4.0, "end": 4.0, "provenance": "000000000000:1"},
        {"text": "声", "start": 45.0, "end": 46.0, "provenance": "000000480000:0"},
        {"text": "点", "start": 48.0, "end": 48.0, "provenance": "000000480000:1"},
    ]
    assert saved["speechAccuracyVerified"] is False


def test_two_callers_cannot_share_lease(setup: tuple[Path, Path]) -> None:
    request, output = setup

    class ReentrantRuntime(FakeRuntime):
        def execute(
            self,
            request: lane.Request,
            window: AudioWindow,
            work: Path,
            identities: dict[str, object],
        ) -> None:
            with pytest.raises(BlockingIOError):
                lane.collect(request.path, output, runtime=FakeRuntime())
            super().execute(request, window, work, identities)

    lane.collect(request, output, runtime=ReentrantRuntime())


@pytest.mark.parametrize("absent", [True, False])
def test_descendant_exit_during_measurement(monkeypatch: pytest.MonkeyPatch, absent: bool) -> None:
    host = MagicMock(spec=SystemHost)

    def birth(pid: int) -> str:
        if pid == 11:
            raise ReplayError("Exact process birth is unavailable")
        return "10:100"

    host.birth.side_effect = birth
    host.footprint.return_value = 200
    host.absent.return_value = absent
    monkeypatch.setattr(
        subprocess, "run", MagicMock(return_value=SimpleNamespace(stdout="10 1 8\n11 10 9\n"))
    )
    if absent:
        result = lane.sample_family(10, "10:100", host)
        assert result["exitedDuringObservation"] == [11]
        assert result["physicalBytes"] == 200 and result["rssBytes"] == 8192
    else:
        with pytest.raises(ReplayError, match="unavailable"):
            lane.sample_family(10, "10:100", host)


@pytest.mark.parametrize("mode", ["gone", "kill", "reused", "group", "denied", "wait", "unknown"])
def test_exact_cleanup_never_masks_or_targets_other_owner(
    monkeypatch: pytest.MonkeyPatch, mode: str
) -> None:
    child = MagicMock()
    child.pid = 321
    child.poll.return_value = 0 if mode == "gone" else None
    host = MagicMock(spec=SystemHost)
    host.birth.return_value = "new" if mode == "reused" else "old"
    monkeypatch.setattr(os, "getpgid", MagicMock(return_value=999 if mode == "group" else 321))
    monkeypatch.setattr(os, "getsid", MagicMock(return_value=321))
    kill = MagicMock(side_effect=PermissionError("denied") if mode == "denied" else None)
    monkeypatch.setattr(os, "killpg", kill)
    if mode == "wait":
        child.wait.side_effect = subprocess.TimeoutExpired("owned", 5)
    result = lane._cleanup_child(child, None if mode == "unknown" else "old", host)
    assert kill.call_count == (1 if mode in {"kill", "denied", "wait"} else 0)
    assert result["terminalObserved"] is (mode in {"gone", "kill"})
    if mode not in {"gone", "kill"}:
        assert "errorType" in result


@pytest.mark.parametrize("terminal", [0, 1, None])
def test_sampling_root_exit_and_cleanup_failure_preserve_original(
    setup: tuple[Path, Path], monkeypatch: pytest.MonkeyPatch, terminal: int | None
) -> None:
    request, output = setup
    parsed = lane.load_request(request)
    output.mkdir(parents=True)
    child = MagicMock()
    child.pid = 321
    child.returncode = terminal
    child.poll.side_effect = [None, None, terminal, terminal]
    child.wait.side_effect = [
        subprocess.TimeoutExpired("owned", 1),
        subprocess.TimeoutExpired("owned", 1) if terminal is None else 0,
    ]
    host = MagicMock(spec=SystemHost)
    host.birth.return_value = "321:100"
    host.absent.return_value = True
    monkeypatch.setattr(lane, "SystemHost", MagicMock(return_value=host))
    monkeypatch.setattr(subprocess, "Popen", MagicMock(return_value=child))
    monkeypatch.setattr(
        lane,
        "sample_family",
        MagicMock(
            side_effect=[
                {"members": [{"pid": 321, "birth": "321:100"}], "rssBytes": 5, "physicalBytes": 4},
                ReplayError("Exact process birth is unavailable"),
            ]
        ),
    )
    monkeypatch.setattr(os, "getpgid", MagicMock(return_value=321))
    monkeypatch.setattr(os, "getsid", MagicMock(return_value=321))
    kill = MagicMock(side_effect=PermissionError("cleanup EPERM"))
    monkeypatch.setattr(os, "killpg", kill)
    if terminal == 0:
        lane.MetalRuntime().execute(
            parsed, AudioWindow(0, parsed.samples), output, lane.verify_inputs(parsed)
        )
    else:
        with pytest.raises(ReplayError, match="birth is unavailable"):
            lane.MetalRuntime().execute(
                parsed, AudioWindow(0, parsed.samples), output, lane.verify_inputs(parsed)
            )
    resources = lane._read(output / "resources.json")
    assert resources["familyGone"] is (terminal == 0)
    assert resources["samplingErrors"]
    assert kill.call_count == (1 if terminal is None else 0)


@pytest.mark.parametrize("terminal", [0, 3, None])
def test_kernel_identity_disappears_before_waitable_exit(
    setup: tuple[Path, Path], monkeypatch: pytest.MonkeyPatch, terminal: int | None
) -> None:
    request, output = setup
    parsed = lane.load_request(request)
    output.mkdir(parents=True)
    child = MagicMock()
    child.pid = 321
    child.returncode = None
    child.poll.side_effect = [None, None, None, terminal]
    waits = 0

    def wait(*, timeout: int) -> int:
        nonlocal waits
        waits += 1
        if waits == 1 or terminal is None:
            raise subprocess.TimeoutExpired("owned", timeout)
        child.returncode = terminal
        return terminal

    child.wait.side_effect = wait
    host = MagicMock(spec=SystemHost)
    host.birth.return_value = "321:100"
    host.absent.return_value = True
    monkeypatch.setattr(lane, "SystemHost", MagicMock(return_value=host))
    monkeypatch.setattr(subprocess, "Popen", MagicMock(return_value=child))
    monkeypatch.setattr(
        lane,
        "sample_family",
        MagicMock(
            side_effect=[
                {"members": [{"pid": 321, "birth": "321:100"}], "rssBytes": 5, "physicalBytes": 4},
                ReplayError("kernel identity disappeared"),
            ]
        ),
    )
    monkeypatch.setattr(os, "getpgid", MagicMock(return_value=321))
    monkeypatch.setattr(os, "getsid", MagicMock(return_value=321))
    kill = MagicMock(side_effect=PermissionError("owned cleanup denied"))
    monkeypatch.setattr(os, "killpg", kill)
    if terminal == 0:
        lane.MetalRuntime().execute(
            parsed, AudioWindow(0, parsed.samples), output, lane.verify_inputs(parsed)
        )
    else:
        with pytest.raises(ReplayError, match="kernel identity disappeared"):
            lane.MetalRuntime().execute(
                parsed, AudioWindow(0, parsed.samples), output, lane.verify_inputs(parsed)
            )
    resources = lane._read(output / "resources.json")
    assert resources["exitCode"] == terminal
    assert resources["familyGone"] is (terminal == 0)
    errors = resources["samplingErrors"]
    assert isinstance(errors, list)
    assert record(errors[0])["terminalReconciliationExitCode"] == terminal
    assert child.wait.call_args_list[1].kwargs["timeout"] == 1
    assert kill.call_count == (1 if terminal is None else 0)
