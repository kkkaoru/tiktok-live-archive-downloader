"""Synthetic backend/process contracts, no model or accelerator execution."""

import json
import subprocess
import wave
from pathlib import Path
from unittest.mock import MagicMock

import numpy as np
import pytest
from numpy.typing import NDArray

from replay import separation_device_benchmark as bench
from replay.core import ReplayError
from replay.native_invocation import durable, identity, pin, read
from replay.separation_forward_probe import read_interval


class Backend:
    sources = ("vocals", "instrumental")

    def extract_vocals(self, chunk: NDArray[np.float32]) -> NDArray[np.float32]:
        return chunk * np.float32(0.5)

    def extract_instrumental(self, chunk: NDArray[np.float32]) -> NDArray[np.float32]:
        return chunk - self.extract_vocals(chunk)


def auditor(path: Path, phase: str) -> None:
    durable(path / f"{phase}-descriptors.json", {"externalWritableRegular": []})


@pytest.fixture
def case(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> tuple[Path, Path]:
    source = tmp_path / "input.wav"
    with wave.open(str(source), "wb") as audio:
        audio.setparams((2, 2, 44100, 0, "NONE", "not compressed"))
        audio.writeframes(b"\x01\x00\x02\x00" * 64)
    interval = read_interval(source, start_sample=0, samples=64)
    code = Path(bench.__file__).resolve()
    refs = [
        pin(code.with_name(name))
        for name in (
            code.name,
            "bgm_separation.py",
            "separation_forward_probe.py",
            "process_family_observation.py",
        )
    ]
    request = tmp_path / "request.json"
    durable(
        request,
        {
            "version": 1,
            "kind": "separation-device-benchmark",
            "samples": 64,
            "startSample": 0,
            "timeoutSeconds": 20,
            "maxFamilyBytes": 1000,
            "intervalSHA256": interval.sha256,
            "source": {"path": str(source), "identity": identity(source)},
            "inputs": refs,
        },
    )
    monkeypatch.setattr(bench, "_local_environment", lambda: {})
    return request, tmp_path


def completed(request: Path, path: Path, device: str) -> None:
    path.mkdir()
    bench.worker(request, device, path, factory=lambda _: Backend(), auditor=auditor)
    durable(
        path / "resources.json",
        {
            "exitCode": 0,
            "familyGone": True,
            "cleanup": {"terminalObserved": True},
            "samples": [{}],
            "wallSeconds": 4 if device == "cpu" else 1,
        },
    )
    (path / "worker.log").write_text("")


def test_full_comparison_and_tamper(case: tuple[Path, Path]) -> None:
    request, root = case
    for device in ("cpu", "mps"):
        completed(request, root / device, device)
    result = bench.compare(request, root / "cpu", root / "mps", root / "comparison.json")
    assert read(result)["performanceAndEquivalencePassed"] is True
    assert read(result)["sdrInfinite"] is True
    (root / "mps/worker.log").write_text("sandbox: deny file-write")
    with pytest.raises(ReplayError, match="Sandbox"):
        bench.verify(request, "mps", root / "mps")


@pytest.mark.parametrize(
    "field,value",
    [
        ("samples", 0),
        ("startSample", -1),
        ("timeoutSeconds", 0),
        ("maxFamilyBytes", 0),
        ("kind", "other"),
        ("inputs", []),
    ],
)
def test_invalid_request(case: tuple[Path, Path], field: str, value: object) -> None:
    request, _ = case
    data = read(request)
    data[field] = value
    request.write_text(json.dumps(data))
    with pytest.raises(ReplayError):
        bench.load(request)


@pytest.mark.parametrize(
    "mode", ["ok", "sampling-exit", "sampling-live", "budget", "exit", "descendant"]
)
def test_actual_parent_contract(
    case: tuple[Path, Path], monkeypatch: pytest.MonkeyPatch, mode: str
) -> None:
    request, root = case
    child = MagicMock()
    child.pid = 111
    child.returncode = 2 if mode == "exit" else 0
    child.poll.side_effect = [None, 0]
    child.wait.return_value = child.returncode
    host = MagicMock()
    host.birth.return_value = "birth"
    host.absent.return_value = mode != "descendant"
    monkeypatch.setattr(bench, "SystemHost", MagicMock(return_value=host))
    monkeypatch.setattr(subprocess, "Popen", MagicMock(return_value=child))
    sample = {
        "members": [{"pid": 111}],
        "rssBytes": 2000 if mode == "budget" else 5,
        "physicalBytes": 4,
    }
    sampler = MagicMock(return_value=sample)
    if mode.startswith("sampling"):
        child.poll.side_effect = [None, None]
        sampler.side_effect = [sample, ReplayError("birth unavailable")]
        child.wait.side_effect = [
            subprocess.TimeoutExpired("child", 0.2),
            0 if mode == "sampling-exit" else subprocess.TimeoutExpired("child", 1),
        ]
    monkeypatch.setattr(bench, "sample_family", sampler)
    monkeypatch.setattr(bench, "cleanup_child", lambda *_: {"terminalObserved": True})
    verifier = MagicMock(return_value=root / "out/worker.json")
    monkeypatch.setattr(bench, "verify", verifier)
    if mode in {"ok", "sampling-exit"}:
        bench.run(request, "mps", root / "out")
        assert verifier.call_count == 1
    else:
        with pytest.raises(ReplayError):
            bench.run(request, "mps", root / "out")
        assert verifier.call_count == 0
    assert (root / "out/resources.json").is_file()
    with pytest.raises(ReplayError, match="Existing"):
        bench.run(request, "mps", root / "out")


def test_prepare_same_interval_no_model_hash(
    case: tuple[Path, Path], monkeypatch: pytest.MonkeyPatch
) -> None:
    request, root = case
    data = read(request)
    core = root / "env/lib/python3.12/site-packages/mlx/core.fake.so"
    core.parent.mkdir(parents=True)
    core.write_text("provider")
    interpreter = root / "env/bin/python"
    interpreter.parent.mkdir()
    interpreter.write_text("fake interpreter, never run")
    weights = root / "model/hub/repo/snapshots/version/model.safetensors"
    weights.parent.mkdir(parents=True)
    weights.write_text("model, never hashed")
    prior = root / "prior.json"
    durable(
        prior,
        {
            "source": data["source"],
            "sampleCount": 64,
            "providerInputs": [{"path": str(core), "sha256": "a" * 64}],
            "modelInputs": [{"path": str(weights), "sha256": "b" * 64}],
        },
    )
    observed = root / "observed.json"
    durable(
        observed,
        {
            "sampleCount": 64,
            "intervalSHA256": data["intervalSHA256"],
            "context": {"plan": pin(prior)},
        },
    )
    monkeypatch.setattr(bench, "capture", lambda _: ())
    made = bench.prepare(prior, observed, root / "prepared", family_bytes=1000)
    assert read(made)["intervalSHA256"] == data["intervalSHA256"]
    from replay.workflow import parse_stages

    assert len(parse_stages(read(made.parent / "workflow.json"))) == 3


@pytest.mark.parametrize("action", ["worker", "run", "verify", "compare", "prepare"])
def test_cli(case: tuple[Path, Path], monkeypatch: pytest.MonkeyPatch, action: str) -> None:
    request, root = case
    called = MagicMock()
    monkeypatch.setattr(bench, action, called)
    assert (
        bench.main(
            [
                action,
                str(request),
                str(root),
                "--device",
                "cpu",
                "--cpu",
                str(root),
                "--mps",
                str(root),
                "--prior-input",
                str(request),
                "--family-bytes",
                "1000",
            ]
        )
        == 0
    )
    assert called.call_count == 1


def test_original_model_interpreter_import(tmp_path: Path) -> None:
    import os
    import sys

    interpreter = Path(os.environ.get("REPLAY_BENCHMARK_IMPORT_PYTHON", sys.executable))
    environment = os.environ | {
        "PYTHONPATH": str(Path.cwd() / "src"),
        "PYTHONDONTWRITEBYTECODE": "1",
        "TMPDIR": str(tmp_path),
        "TMP": str(tmp_path),
        "TEMP": str(tmp_path),
        "XDG_CACHE_HOME": str(tmp_path),
        "HF_HUB_OFFLINE": "1",
        "TRANSFORMERS_OFFLINE": "1",
    }
    result = subprocess.run(
        [str(interpreter), "-B", "-m", "replay.separation_device_benchmark", "--help"],
        cwd=Path.cwd(),
        env=environment,
        capture_output=True,
        text=True,
        timeout=20,
        check=True,
    )
    assert "separation_device_benchmark" in result.stdout
    assert not result.stderr
