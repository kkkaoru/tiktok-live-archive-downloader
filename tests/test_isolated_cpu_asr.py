"""Fresh-window isolation preserves original clocks and immutable resume data."""

import fcntl
import json
import subprocess
import sys
import wave
from pathlib import Path

import numpy as np
import pytest
from numpy.typing import NDArray

from replay import isolated_cpu_asr as isolated
from replay.asr_runtime import ASRDevice
from replay.core import ReplayError
from replay.local_asr import fingerprint


class FakeRuntime:
    def __init__(self, device: ASRDevice) -> None:
        assert device == ASRDevice.CPU

    def predict(self, samples: NDArray[np.float32], model: Path, *, evidence: Path) -> object:
        assert samples.dtype == np.float32 and model.is_dir()
        evidence.mkdir()
        return {"segments": [], "text": ""}


@pytest.fixture
def fixture(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> tuple[Path, Path, Path]:
    monkeypatch.chdir(tmp_path)
    root = tmp_path / "edits/work"
    root.mkdir(parents=True)
    model = tmp_path / "model"
    model.mkdir()
    (model / "weights.npz").write_bytes(b"weights")
    (model / "config.json").write_bytes(b"config")
    source = root / "source.wav"
    with wave.open(str(source), "wb") as stream:
        stream.setparams((1, 2, 16000, 0, "NONE", "not compressed"))
        stream.writeframes(bytes(16000 * 2 * 90))
    monkeypatch.setattr(isolated, "WhisperRuntime", FakeRuntime)
    return root, source, model


def execute(command: tuple[str, ...], log: Path, timeout: int) -> None:
    assert command[-2] == "--worker" and timeout == 300
    log.write_text("synthetic worker")
    isolated.worker(Path(command[-1]))


def run(
    fixture: tuple[Path, Path, Path],
    policy: str,
    *,
    limit: int | None = None,
    executor: isolated.Executor = execute,
) -> Path:
    root, source, model = fixture
    return isolated.run(
        source,
        root / "output",
        root / policy,
        model=model,
        weights_sha256=fingerprint(model / "weights.npz"),
        config_sha256=fingerprint(model / "config.json"),
        execute=executor,
        limit=limit,
    )


def test_two_fresh_children_and_verified_resume(fixture: tuple[Path, Path, Path]) -> None:
    root, _, _ = fixture
    first = run(fixture, "one", limit=1)
    assert json.loads(first.read_text())["recognitionComplete"] is False
    raw = root / "output/000000000000.raw.json"
    digest = fingerprint(raw)
    assert not (root / "output/complete.json").exists()
    second = run(fixture, "two")
    assert json.loads(second.read_text())["newWindows"] == 1
    assert fingerprint(raw) == digest
    clock = json.loads((root / "output/000000480000.clock.json").read_text())
    assert clock["startSample"] == 480000 and clock["endSample"] == 1440000
    third = run(fixture, "three")
    assert json.loads(third.read_text())["newWindows"] == 0
    assert (root / "output/000000000000.cpu-runtime/runtime").is_dir()
    assert (root / "output/000000480000.cpu-runtime/runtime").is_dir()


def test_failure_retains_partial_without_completion(fixture: tuple[Path, Path, Path]) -> None:
    root, _, _ = fixture

    def timeout(command: tuple[str, ...], log: Path, seconds: int) -> None:
        raise subprocess.TimeoutExpired(command, seconds)

    with pytest.raises(subprocess.TimeoutExpired):
        run(fixture, "failure", executor=timeout)
    assert (root / "output/000000000000.cpu-runtime/request.json").is_file()
    assert not (root / "output/complete.json").exists()
    with pytest.raises(FileExistsError):
        run(fixture, "retry")


def test_existing_mutations_rejected(fixture: tuple[Path, Path, Path]) -> None:
    root, _, _ = fixture
    run(fixture, "one")
    (root / "output/000000000000.raw.json").write_text("{}")
    with pytest.raises(ReplayError, match="changed"):
        run(fixture, "two")


def test_lease_does_not_start_duplicate(fixture: tuple[Path, Path, Path]) -> None:
    root, _, _ = fixture
    run(fixture, "one", limit=1)
    with (root / "output/.run.lock").open("rb") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        with pytest.raises(ReplayError, match="owner"):
            run(fixture, "two")


@pytest.mark.parametrize("limit", [0, -1, True, 257])
def test_limit_bounds(fixture: tuple[Path, Path, Path], limit: int) -> None:
    with pytest.raises(ReplayError, match="limit"):
        run(fixture, "bad", limit=limit)


def test_manifest_and_model_changes_rejected(fixture: tuple[Path, Path, Path]) -> None:
    root, source, model = fixture
    run(fixture, "one", limit=1)
    manifest = root / "output/inputs.json"
    manifest.write_text("{}")
    with pytest.raises(ReplayError, match="inputs changed"):
        run(fixture, "two")
    with pytest.raises(ReplayError, match="model fingerprint"):
        isolated.run(source, root / "new", root / "policy", model=model)


def test_worker_payload_rejects_tampering(fixture: tuple[Path, Path, Path]) -> None:
    root, _, _ = fixture
    run(fixture, "one", limit=1)
    evidence = root / "output/000000000000.cpu-runtime"
    (evidence / "samples.npy").write_bytes(b"bad")
    with pytest.raises(ReplayError, match="inputs changed"):
        isolated.worker(evidence)


def test_cli_routes(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    observed: list[Path] = []
    monkeypatch.setattr(isolated, "worker", lambda path: observed.append(path))
    monkeypatch.setattr(sys, "argv", ["isolated", "--worker", str(tmp_path)])
    isolated.main()

    assert observed == [tmp_path]
    monkeypatch.setattr(sys, "argv", ["isolated"])
    with pytest.raises(ReplayError, match="requires"):
        isolated.main()
    monkeypatch.setattr(isolated, "run", lambda *args, **kwargs: tmp_path)
    monkeypatch.setattr(sys, "argv", ["isolated", "a", "b", "--policy", "c"])
    isolated.main()


def test_child_timeout_and_failure(tmp_path: Path) -> None:
    with pytest.raises(subprocess.TimeoutExpired):
        isolated.execute_child(("/bin/sleep", "5"), tmp_path / "timeout.log", 1)
    with pytest.raises(ReplayError, match="return code 1"):
        isolated.execute_child(("/usr/bin/false",), tmp_path / "failure.log", 1)
    isolated.execute_child(("/usr/bin/true",), tmp_path / "success.log", 1)
