import hashlib
import io
import json
import os
import signal
import subprocess
import sys
import time
from pathlib import Path

import pytest

from replay import stateful_audio, workflow
from replay.core import ReplayError
from replay.native_invocation import read
from replay.stages import StageError


@pytest.fixture
def observation(tmp_path: Path) -> tuple[Path, Path]:
    request = tmp_path / "request.json"
    directory = tmp_path / "output"
    directory.mkdir()
    selection = {
        "checkpoints": [{"startValue": 1, "timescale": 48000, "sampleCount": 17}],
        "budget": {
            "maxLedgerBytes": 10000,
            "maxRecords": 10,
            "maxDecodedSamples": 4800,
            "maxWallSeconds": 10,
        },
    }
    request.write_text(
        json.dumps(
            {
                "source": str(tmp_path / "source.wav"),
                "outputDirectory": str(directory),
                "startValue": 0,
                "durationValue": 1,
                "timescale": 10,
                "statefulSelection": selection,
            }
        ),
        encoding="utf-8",
    )
    line = (
        json.dumps(
            {
                "bufferOrdinal": 0,
                "sampleOrdinal": 0,
                "sourcePTS": {"value": 0, "timescale": 48000},
                "duration": {"value": 4800, "timescale": 48000},
                "samples": 4800,
                "discontinuity": {"value": 0, "timescale": 48000},
            }
        ).encode()
        + b"\n"
    )
    (directory / "ledger.jsonl").write_bytes(line)
    (directory / "stateful.json").write_text(
        json.dumps(
            {
                "version": 1,
                "origin": {"value": 0, "timescale": 1},
                "selection": selection,
                "records": 1,
                "decodedSamples": 4800,
                "ledgerBytes": len(line),
                "ledgerSHA256": hashlib.sha256(line).hexdigest(),
            }
        ),
        encoding="utf-8",
    )
    return request, directory


@pytest.mark.parametrize("count", [960000, 2880000])
def test_representative_checkpoint_support(count: int) -> None:
    request: dict[str, object] = {
        "startValue": 0,
        "durationValue": 60,
        "timescale": 1,
        "statefulSelection": {
            "checkpoints": [{"startValue": 0, "timescale": 1, "sampleCount": count}]
        },
    }
    assert stateful_audio.checkpoints(request)[0][1] == count


def test_checkpoint_above_sixty_seconds_rejected() -> None:
    request: dict[str, object] = {
        "startValue": 0,
        "durationValue": 61,
        "timescale": 1,
        "statefulSelection": {
            "checkpoints": [{"startValue": 0, "timescale": 1, "sampleCount": 2880001}]
        },
    }
    with pytest.raises(ReplayError, match="Invalid source checkpoint"):
        stateful_audio.checkpoints(request)


def test_actual_support_ledger(observation: tuple[Path, Path]) -> None:
    request, directory = observation
    result = stateful_audio.verify_ledger(read(request), directory)
    assert result["decodedSamples"] == 4800


@pytest.mark.parametrize(
    "field,value",
    [
        ("bufferOrdinal", 1),
        ("sampleOrdinal", 1),
        ("samples", 0),
        ("duration", {"value": 1, "timescale": 1}),
        ("sourcePTS", {"value": 1, "timescale": 1}),
        ("discontinuity", {"value": 1, "timescale": 1}),
    ],
)
def test_rejects_changed_ledger(observation: tuple[Path, Path], field: str, value: object) -> None:
    request, directory = observation
    ledger = directory / "ledger.jsonl"
    row = json.loads(ledger.read_bytes())
    row[field] = value
    line = json.dumps(row).encode() + b"\n"
    ledger.write_bytes(line)
    result_path = directory / "stateful.json"
    result = json.loads(result_path.read_bytes())
    result.update(ledgerBytes=len(line), ledgerSHA256=hashlib.sha256(line).hexdigest())
    result_path.write_text(json.dumps(result), encoding="utf-8")
    with pytest.raises(ReplayError):
        stateful_audio.verify_ledger(read(request), directory)


def test_rejects_incomplete_and_hash_mismatch(observation: tuple[Path, Path]) -> None:
    request, directory = observation
    (directory / "ledger.jsonl").write_bytes(b"changed\n")
    with pytest.raises(ReplayError):
        stateful_audio.verify_ledger(read(request), directory)
    (directory / "stateful.json").unlink()
    with pytest.raises(FileNotFoundError):
        stateful_audio.verify_ledger(read(request), directory)


@pytest.mark.parametrize("material", [False, True])
def test_independent_readback_cli(
    observation: tuple[Path, Path], monkeypatch: pytest.MonkeyPatch, material: bool
) -> None:
    request, directory = observation
    if material:
        data = read(request)
        selection = {
            "productionMaterial": True,
            "checkpoints": [{"startValue": 0, "timescale": 1, "sampleCount": 4800}],
            "budget": {
                "maxLedgerBytes": 10000,
                "maxRecords": 10,
                "maxDecodedSamples": 4800,
                "maxWallSeconds": 10,
            },
        }
        data["statefulSelection"] = selection
        request.write_text(json.dumps(data), encoding="utf-8")
        result = read(directory / "stateful.json")
        result.update(
            selection=selection,
            supportedEnd={"value": 4800, "timescale": 48000},
            assetDuration={"value": 1, "timescale": 10},
        )
        (directory / "stateful.json").write_text(json.dumps(result), encoding="utf-8")
    calls: list[object] = []

    def run(command: list[str], **kwargs: object) -> subprocess.CompletedProcess[bytes]:
        calls.append(command)
        if command[0] == "/probe":
            return subprocess.CompletedProcess(
                command,
                0,
                json.dumps(
                    {
                        "streams": [
                            {
                                "sample_rate": "48000",
                                "channels": 2,
                                "codec_name": "pcm_s16le",
                                "time_base": "1/48000",
                                "duration_ts": 4800 if material else 17,
                            }
                        ]
                    }
                ).encode(),
                b"",
            )
        output = kwargs["stdout"]
        assert isinstance(output, io.BufferedIOBase)
        output.write(bytes(19200 if material else 68))
        return subprocess.CompletedProcess(command, 0, b"", b"")

    monkeypatch.setattr(subprocess, "run", run)
    assert (
        stateful_audio.main(
            ["verify", str(request), "/native", "/decode", "/probe", str(directory)]
        )
        == 0
    )
    assert len(calls) == 2


def test_prepare_executable_workflow(
    observation: tuple[Path, Path], monkeypatch: pytest.MonkeyPatch
) -> None:
    request, directory = observation
    root = directory.parent
    monkeypatch.chdir(root)
    (root / "edits/work").mkdir(parents=True)
    raw = read(request)
    raw["outputDirectory"] = str(root / "edits/work/new-output")
    request.write_text(json.dumps(raw), encoding="utf-8")
    prepared = root / "edits/work/prepared"
    assert (
        stateful_audio.main(
            ["prepare", str(request), "/native", "/decode", "/probe", str(prepared)]
        )
        == 0
    )
    stages = workflow.load_stages(prepared / "workflow.json")
    assert len(stages) == 1
    assert stages[0].timeout_seconds == 10
    assert stages[0].adopt_existing is False
    assert (
        workflow.main([str(prepared / "workflow.json"), "--state", str(root / "state"), "--check"])
        == 0
    )
    with pytest.raises(FileExistsError):
        stateful_audio.prepare(request, Path("/native"), Path("/decode"), Path("/probe"), prepared)


@pytest.mark.parametrize("value", [None, True, "1", -1])
def test_integer_boundary(value: object) -> None:
    with pytest.raises(ReplayError):
        stateful_audio.integer(value)


@pytest.mark.parametrize("cancel", [False, True])
def test_workflow_interrupt_preserves_partial_and_kills_child(tmp_path: Path, cancel: bool) -> None:
    source = tmp_path / "input"
    source.write_text(json.dumps({"outputDirectory": str(tmp_path / "native")}), encoding="utf-8")
    output = tmp_path / "native/stateful.json"
    pid_file = tmp_path / "native/pid"
    plan = tmp_path / "workflow.json"
    fixture = Path(__file__).with_name("stateful_guard_fixture.py").resolve()
    plan.write_text(
        json.dumps(
            {
                "version": 1,
                "stages": [
                    {
                        "name": "interrupted",
                        "command": [sys.executable, str(fixture), str(source)],
                        "inputs": [str(source), str(fixture)],
                        "outputs": [str(output)],
                        "timeout_seconds": 1,
                        "verify_command": ["/usr/bin/false"],
                    }
                ],
            }
        ),
        encoding="utf-8",
    )
    command = [
        sys.executable,
        "-m",
        "replay.workflow",
        str(plan),
        "--state",
        str(tmp_path / "state"),
    ]
    assert subprocess.run([*command, "--check"], capture_output=True, timeout=5).returncode == 0
    with subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE) as process:
        deadline = time.monotonic() + 5
        while not output.exists() and time.monotonic() < deadline:
            time.sleep(0.01)
        assert output.exists()
        if cancel:
            process.send_signal(signal.SIGTERM)
        process.communicate(timeout=5)
        assert process.returncode == 1
    child_pid = int(pid_file.read_text(encoding="utf-8"))
    with pytest.raises(ProcessLookupError):
        os.kill(child_pid, 0)
    assert output.read_text(encoding="utf-8") == "synthetic completion marker"
    assert json.loads((tmp_path / "state/interrupted.json").read_bytes())["status"] == "running"
    assert subprocess.run(command, capture_output=True, timeout=5).returncode == 1
    assert json.loads((tmp_path / "state/interrupted.json").read_bytes())["status"] == "running"


@pytest.mark.parametrize("endpoint", [4800, 4799])
def test_production_eof_accounting(observation: tuple[Path, Path], endpoint: int) -> None:
    request, directory = observation
    data = read(request)
    data["durationValue"] = 120
    data["timescale"] = 1
    selection = {
        "productionMaterial": True,
        "checkpoints": [{"startValue": 0, "timescale": 1, "sampleCount": 5760000}],
        "budget": {"maxLedgerBytes": 10000, "maxRecords": 10, "maxDecodedSamples": 5760000},
    }
    data["statefulSelection"] = selection
    result = read(directory / "stateful.json")
    result.update(
        selection=selection,
        supportedEnd={"value": endpoint, "timescale": 48000},
        assetDuration={"value": 1, "timescale": 10},
    )
    (directory / "stateful.json").write_text(json.dumps(result), encoding="utf-8")
    if endpoint == 4800:
        assert stateful_audio.verify_ledger(data, directory)["decodedSamples"] == 4800
    else:
        with pytest.raises(ReplayError, match="EOF"):
            stateful_audio.verify_ledger(data, directory)
    data["durationValue"] = 121
    with pytest.raises(ReplayError, match="material extent"):
        stateful_audio.checkpoints(data)


def test_workflow_cancellation_unwinds_runner() -> None:
    with pytest.raises(StageError, match="cancelled"):
        workflow._cancel(signal.SIGTERM, None)
