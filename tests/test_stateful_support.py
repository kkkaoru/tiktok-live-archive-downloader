import hashlib
import io
import json
import subprocess
import wave
from pathlib import Path

import pytest

from replay import stateful_audio, stateful_guard, stateful_support
from replay.core import ReplayError
from replay.native_invocation import read
from replay.recordings import record
from replay.stages import _fingerprint, file_digest
from replay.workflow import load_stages


@pytest.fixture
def completed(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> tuple[Path, Path, Path]:
    output = tmp_path / "observed"
    output.mkdir()
    native = tmp_path / "native"
    native.write_bytes(b"synthetic native identity")
    decode = tmp_path / "decode"
    decode.write_bytes(b"synthetic decoder identity")
    probe = tmp_path / "probe"
    probe.write_bytes(b"synthetic probe identity")
    source = tmp_path / "source"
    source.write_bytes(b"synthetic source identity")
    request = tmp_path / "request.json"
    selection = {
        "checkpoints": [{"startValue": 1, "timescale": 48000, "sampleCount": 17}],
        "budget": {
            "maxLedgerBytes": 10000,
            "maxRecords": 10,
            "maxDecodedSamples": 100,
            "maxWallSeconds": 10,
        },
    }
    request.write_text(
        json.dumps(
            {
                "source": str(source),
                "outputDirectory": str(output),
                "startValue": 0,
                "durationValue": 1,
                "timescale": 1,
                "statefulSelection": selection,
            }
        ),
        encoding="utf-8",
    )
    ledger = (
        json.dumps(
            {
                "bufferOrdinal": 0,
                "sampleOrdinal": 0,
                "sourcePTS": {"value": 0, "timescale": 48000},
                "duration": {"value": 20, "timescale": 48000},
                "samples": 20,
                "discontinuity": {"value": 0, "timescale": 1},
            }
        ).encode()
        + b"\n"
    )
    (output / "ledger.jsonl").write_bytes(ledger)
    (output / "stateful.json").write_text(
        json.dumps(
            {
                "version": 1,
                "origin": {"value": 0, "timescale": 1},
                "selection": selection,
                "records": 1,
                "decodedSamples": 20,
                "ledgerBytes": len(ledger),
                "ledgerSHA256": hashlib.sha256(ledger).hexdigest(),
            }
        ),
        encoding="utf-8",
    )
    (output / "checkpoint-0.mov").write_bytes(b"mock MOV bytes")
    # Preparation refuses existing output; prepare the declaration before restoring it.
    data = read(request)
    data["outputDirectory"] = str(tmp_path / "not-created")
    request.write_text(json.dumps(data), encoding="utf-8")
    plan = stateful_audio.prepare(request, native, decode, probe, tmp_path / "prepared")
    plan_data = read(plan)
    stage_data = plan_data["stages"]
    assert isinstance(stage_data, list)
    stage_data[0]["outputs"] = [
        str(output / "stateful.json"),
        str(output / "ledger.jsonl"),
        str(output / "checkpoint-0.mov"),
    ]
    plan.write_text(json.dumps(plan_data), encoding="utf-8")
    data["outputDirectory"] = str(output)
    request.write_text(json.dumps(data), encoding="utf-8")
    stage = load_stages(plan)[0]
    state = tmp_path / "state"
    state.mkdir()
    (state / "source-pts-observation.json").write_text(
        json.dumps(
            {
                "status": "complete",
                "fingerprint": _fingerprint(stage),
                "outputs": {str(path): file_digest(path) for path in stage.outputs},
            }
        ),
        encoding="utf-8",
    )

    def run(command: list[str], **kwargs: object) -> subprocess.CompletedProcess[bytes]:
        if command[0] == str(probe):
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
                                "duration_ts": 17,
                            }
                        ],
                        "frames": [{"pts": 0, "duration": 17, "nb_samples": 17}],
                    }
                ).encode(),
                b"",
            )
        output_stream = kwargs.get("stdout")
        if isinstance(output_stream, io.BufferedIOBase):
            output_stream.write(bytes(range(68)))
        return subprocess.CompletedProcess(command, 0, bytes(range(68)), b"")

    monkeypatch.setattr(subprocess, "run", run)
    return plan, state, tmp_path / "consumer"


def test_cli_exact_copy(
    completed: tuple[Path, Path, Path], capsys: pytest.CaptureFixture[str]
) -> None:
    plan, state, output = completed
    assert (
        stateful_support.main(
            [
                str(plan),
                str(output),
                "--sha256",
                file_digest(plan),
                "--state",
                str(state),
                "--index",
                "0",
            ]
        )
        == 0
    )
    assert capsys.readouterr().out.strip() == str(output / "binding.json")
    with wave.open(str(output / "supports/support-000000.wav")) as reader:
        assert reader.readframes(17) == bytes(range(68))


def test_cli_rejects_incomplete(completed: tuple[Path, Path, Path]) -> None:
    plan, state, output = completed
    (state / "source-pts-observation.json").write_text('{"status":"running"}', encoding="utf-8")
    with pytest.raises(ReplayError, match="incomplete or changed"):
        stateful_support.main(
            [
                str(plan),
                str(output),
                "--sha256",
                file_digest(plan),
                "--state",
                str(state),
                "--index",
                "0",
            ]
        )
    assert not output.exists()


@pytest.mark.parametrize("guarded,failed", [(False, False), (True, False), (True, True)])
def test_exact_copy(completed: tuple[Path, Path, Path], guarded: bool, failed: bool) -> None:
    plan, state, output = completed
    if guarded:
        request = plan.parent.parent / "request.json"
        data = read(request)
        selection = data["statefulSelection"]
        assert isinstance(selection, dict)
        budget = selection["budget"]
        assert isinstance(budget, dict)
        budget.update(maxRSSBytes=268435456, maxSavedBytes=67108864)
        request.write_text(json.dumps(data), encoding="utf-8")
        summary_path = plan.parent.parent / "observed/stateful.json"
        summary = read(summary_path)
        summary["selection"] = {
            **selection,
            "budget": {key: value for key, value in budget.items() if key != "maxRSSBytes"},
        }
        summary_path.write_text(json.dumps(summary), encoding="utf-8")
        definition = read(plan)
        stages = definition["stages"]
        assert isinstance(stages, list)
        stage = stages[0]
        report = plan.parent / "guard.json"
        stage["command"] = [
            stage["verify_command"][0],
            "-m",
            "replay.stateful_guard",
            str(request),
            str(plan.parent.parent / "native"),
            str(report),
        ]
        stage["inputs"].extend(
            [str(Path.cwd() / "src/replay/stateful_guard.py"), str(Path("/bin/ps").resolve())]
        )
        stage["outputs"].append(str(report))
        report.write_text(
            json.dumps(
                {
                    "status": "failed" if failed else "complete",
                    "hardCap": False,
                    "sampledGroupPeakRSSBytes": 1024,
                    "kernelChildPeakRSSBytes": 1024,
                    "sampledPeakSavedBytes": stateful_guard.saved_bytes(
                        plan.parent.parent / "observed"
                    ),
                    "finalSavedBytes": stateful_guard.saved_bytes(plan.parent.parent / "observed"),
                }
            ),
            encoding="utf-8",
        )
        plan.write_text(json.dumps(definition), encoding="utf-8")
        parsed = load_stages(plan)[0]
        (state / "source-pts-observation.json").write_text(
            json.dumps(
                {
                    "status": "complete",
                    "fingerprint": _fingerprint(parsed),
                    "outputs": {str(path): file_digest(path) for path in parsed.outputs},
                }
            ),
            encoding="utf-8",
        )
    if failed:
        # Even pinned native outputs and a complete receipt cannot override a failed guard.
        with pytest.raises(ReplayError, match="Watchdog did not complete"):
            stateful_support.copy_support(
                plan=plan, plan_sha256=file_digest(plan), state=state, index=0, output=output
            )
        assert not output.exists()
        return
    result = stateful_support.copy_support(
        plan=plan, plan_sha256=file_digest(plan), state=state, index=0, output=output
    )
    binding = read(result)
    assert binding["ledgerSupport"] == [
        {"bufferOrdinal": 0, "sampleOrdinal": 1, "sourcePTS": "1/48000", "samples": 17}
    ]
    assert binding["historicalModelApproved"] is False
    with wave.open(str(output / "supports/support-000000.wav")) as reader:
        assert reader.readframes(17) == bytes(range(68))
    assert read(output / "supports/placement.json")["missingSupport"] == []


@pytest.mark.parametrize(
    "change", ["request", "ledger", "movie", "native", "incomplete", "plan", "index"]
)
def test_changed_or_partial_refused(completed: tuple[Path, Path, Path], change: str) -> None:
    plan, state, output = completed
    digest = file_digest(plan)
    paths = {
        "request": plan.parent.parent / "request.json",
        "ledger": plan.parent.parent / "observed/ledger.jsonl",
        "movie": plan.parent.parent / "observed/checkpoint-0.mov",
        "native": plan.parent.parent / "native",
    }
    if change in paths:
        with paths[change].open("ab") as stream:
            stream.write(b" ")
    if change == "incomplete":
        receipt = state / "source-pts-observation.json"
        data = read(receipt)
        data["status"] = "running"
        receipt.write_text(json.dumps(data), encoding="utf-8")
    if change == "plan":
        digest = "0" * 64
    with pytest.raises(ReplayError):
        stateful_support.copy_support(
            plan=plan,
            plan_sha256=digest,
            state=state,
            index=1 if change == "index" else 0,
            output=output,
        )
    assert not output.exists()


@pytest.mark.parametrize("pts,count", [(2, 20), (0, 10), (1, 20)])
def test_pinned_but_unsupported_ledger_refused(
    completed: tuple[Path, Path, Path], pts: int, count: int
) -> None:
    plan, state, output = completed
    directory = plan.parent.parent / "observed"
    ledger = directory / "ledger.jsonl"
    row = record(json.loads(ledger.read_bytes()))
    scale = 96000 if pts == 1 else 48000
    row.update(
        sourcePTS={"value": pts, "timescale": scale},
        discontinuity={"value": pts, "timescale": scale},
        samples=count,
        duration={"value": count, "timescale": 48000},
    )
    content = json.dumps(row).encode() + b"\n"
    ledger.write_bytes(content)
    summary = read(directory / "stateful.json")
    summary.update(
        decodedSamples=count,
        ledgerBytes=len(content),
        ledgerSHA256=hashlib.sha256(content).hexdigest(),
    )
    (directory / "stateful.json").write_text(json.dumps(summary), encoding="utf-8")
    stage = load_stages(plan)[0]
    (state / "source-pts-observation.json").write_text(
        json.dumps(
            {
                "status": "complete",
                "fingerprint": _fingerprint(stage),
                "outputs": {str(path): file_digest(path) for path in stage.outputs},
            }
        ),
        encoding="utf-8",
    )
    with pytest.raises(ReplayError):
        stateful_support.copy_support(
            plan=plan, plan_sha256=file_digest(plan), state=state, index=0, output=output
        )
    assert not output.exists()
