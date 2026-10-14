"""Real producer pins with only the Compressor CLI replaced by a fake runner."""

import json
from collections.abc import Sequence
from pathlib import Path

import pytest
from test_compressor_batch import SUBMITTED, SUCCESS, FakeCompressor, job_file

from replay import compressor_batch as batch
from replay.core import ReplayError


def test_runtime_and_dependencies_saved_before_call_and_reused(tmp_path: Path) -> None:
    jobs, output = job_file(tmp_path)
    executable = tmp_path / "compressor"
    executable.write_bytes(b"synthetic CLI")
    dependency = tmp_path / "prepare-input"
    dependency.write_bytes(b"prepared style")
    original = batch.read_jobs(jobs)[0]
    job = batch.Job(original.name, original.source, original.preset, output, (dependency,))
    state = tmp_path / "state"

    def runner(arguments: Sequence[str]) -> str:
        receipt = json.loads((state / "alpha-0000.json").read_text())
        assert receipt["submissionVersion"] == 2
        assert receipt["runtimeInputs"][0] == batch._pin(executable)
        assert receipt["dependencies"] == [batch._pin(dependency)]
        if "-jobpath" in arguments:
            assert receipt["status"] == "submitting"
            assert "terminalRuntimeVerified" not in receipt
            return SUBMITTED
        output.write_bytes(b"mock rendered alpha")
        return SUCCESS

    worker = batch.Batch(state, executable, "test", runner, lambda _: None)
    assert batch.run([job], worker, in_flight=1) == 1
    saved = json.loads((state / "alpha-0000.json").read_text())
    assert saved["terminalRuntimeVerified"] is True
    assert batch.verify_runtime(saved, rehash=True) is True
    assert batch.run([job], worker, in_flight=1) == 0
    dependency.write_bytes(b"changed")
    with pytest.raises(ReplayError, match="dependency changed"):
        batch.run([job], worker, in_flight=1)


@pytest.mark.parametrize("fault", ["tool", "pin", "missing", "terminal", "switch", "dependencies"])
def test_changed_or_missing_runtime_refuses_completed_reuse(tmp_path: Path, fault: str) -> None:
    jobs, output = job_file(tmp_path)
    executable = tmp_path / "compressor"
    executable.write_bytes(b"synthetic CLI")
    worker = batch.Batch(
        tmp_path / "state",
        executable,
        "test",
        FakeCompressor(output, ["Successful"]),
        lambda _: None,
    )
    job = batch.read_jobs(jobs)[0]
    batch.run([job], worker, in_flight=1)
    receipt = worker.state / "alpha-0000.json"
    saved = json.loads(receipt.read_text())
    if fault == "tool":
        executable.write_bytes(b"changed CLI")
    elif fault == "pin":
        saved["runtimeInputs"][0]["sha256"] = "f" * 64
    elif fault == "missing":
        saved.pop("runtimeInputs")
    elif fault == "terminal":
        saved.pop("terminalRuntimeVerified")
    elif fault == "dependencies":
        saved.pop("dependencies")
    else:
        worker.executable = tmp_path / "different-cli"
    receipt.write_text(json.dumps(saved))
    with pytest.raises(ReplayError):
        batch.run([job], worker, in_flight=1)


@pytest.mark.parametrize("value", [["relative"], ["/a", "/a"], [1], "invalid"])
def test_job_dependencies_parser_rejects_invalid(tmp_path: Path, value: object) -> None:
    jobs, _output = job_file(tmp_path)
    data = json.loads(jobs.read_text())
    data["jobs"][0]["dependencies"] = value
    jobs.write_text(json.dumps(data))
    with pytest.raises(ReplayError, match="dependencies"):
        batch.read_jobs(jobs)


def test_legacy_does_not_acquire_historical_runtime_pins() -> None:
    assert batch.verify_runtime({"submissionVersion": 1}, rehash=True) is False
    assert batch.verify_runtime({}, rehash=True) is False
