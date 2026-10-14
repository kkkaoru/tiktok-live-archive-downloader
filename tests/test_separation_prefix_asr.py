"""Synthetic prefix clocks and local FFmpeg only; no model or private media."""

import hashlib
import json
import shutil
import subprocess
import wave
from dataclasses import dataclass, replace
from pathlib import Path

import numpy as np
import pytest
from numpy.typing import NDArray

from replay import replay_transcription_plan as transcription
from replay import separation_prefix_asr as prefix
from replay import separation_window_pipeline as pipeline
from replay.core import ReplayError
from replay.local_asr import DEFAULT_CONFIG_SHA256, DEFAULT_WEIGHTS_SHA256
from replay.native_invocation import pin
from replay.recordings import record
from replay.separation_window_budget import WindowBudget
from replay.stages import execution_lock


@dataclass
class Case:
    request: Path
    raw: dict[str, object]
    plan: pipeline.PipelinePlan
    output: Path
    calls: list[int]


def reference(path: Path) -> dict[str, str]:
    return {"path": str(path), "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}


@pytest.fixture
def case(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Case:
    monkeypatch.setattr(prefix, "ROOT", tmp_path)
    work = tmp_path / "edits/work"
    work.mkdir(parents=True)
    source = work / "source.wav"
    source.write_bytes(b"synthetic source; verifier is independently mocked")
    plan_path = work / "pipeline.json"
    plan_path.write_text('{"synthetic":true}', encoding="utf-8")
    directory = work / "pipeline"
    chunk = directory / "chunks/chunk-000000"
    chunk.mkdir(parents=True)
    context = directory / "input.json"
    context.write_text(json.dumps({"plan": pin(plan_path)}), encoding="utf-8")
    (chunk / "input.json").write_text('{"chunk":0}', encoding="utf-8")
    (chunk / "complete.json").write_text('{"complete":true}', encoding="utf-8")
    budget = pipeline.PipelineBudget(
        WindowBudget(
            workers=2,
            max_samples=pipeline.CHUNK,
            timeout_seconds=10,
            max_worker_rss_bytes=1024**3,
            max_saved_bytes=1024**3,
        ),
        125 * 44100,
        1024**3,
        1024**3,
        60,
    )
    plan = pipeline.PipelinePlan(
        plan_path,
        source,
        directory,
        125 * 44100,
        ((0, 65 * 44100), (60 * 44100, 125 * 44100)),
        budget,
        {},
    )
    monkeypatch.setattr(pipeline, "load_plan", lambda path: plan)
    monkeypatch.setattr(prefix, "runtime_inputs", lambda: {"synthetic.py": "a" * 64})
    calls: list[int] = []

    def verified(
        plan: pipeline.PipelinePlan, *, index: int, bound: dict[str, object]
    ) -> NDArray[np.float32]:
        assert bound == {"plan": pin(plan.path)}
        calls.append(index)
        assert index == 0  # The pending next chunk must never be requested.
        values = np.full((2, 65 * 44100), 0.125, dtype=np.float32)
        values[:, 60 * 44100 :] = 0.875
        return values

    monkeypatch.setattr(pipeline, "verify_chunk", verified)
    executable = shutil.which("ffmpeg")
    assert executable is not None
    ffmpeg = Path(executable).resolve()
    output = work / "prefix"
    raw: dict[str, object] = {
        "version": 1,
        "pipeline": reference(plan_path),
        "context": reference(context),
        "chunks": [
            {
                "input": reference(chunk / "input.json"),
                "complete": reference(chunk / "complete.json"),
            }
        ],
        "ffmpeg": reference(ffmpeg),
        "outputDirectory": str(output),
        "runtime": {"synthetic.py": "a" * 64},
        "model": {
            "id": prefix.MODEL,
            "configSHA256": DEFAULT_CONFIG_SHA256,
            "weightsSHA256": DEFAULT_WEIGHTS_SHA256,
        },
        "maxSavedBytes": 128 * 1024**2,
        "timeoutSeconds": 10,
    }
    request = work / "request.json"
    request.write_text(json.dumps(raw), encoding="utf-8")
    return Case(request, raw, plan, output, calls)


def test_real_ffmpeg_prepares_full_window_without_tail_or_inference(case: Case) -> None:
    output = prefix.prepare(case.request)
    result = record(json.loads(output.read_text()))
    assert result["stableStereoSamples"] == 2646000
    assert result["recognitionExecuted"] is False
    assert result["finalMonoWindowEqual"] is False
    assert result["adopted"] is False
    assert case.calls == [0]
    windows = result["windows"]
    assert isinstance(windows, list) and len(windows) == 1
    window = record(json.loads((case.output / "000000000000.window.json").read_text()))
    assert window["startSample"] == 0
    assert window["endSample"] == 960000
    assert window["clock"] == "input-pcm-not-projected"
    assert (case.output / "000000000000.pcm").stat().st_size == 1920000
    with wave.open(str(case.output / "stereo.wav")) as audio:
        assert audio.getnframes() == 2646000
        audio.setpos(audio.getnframes() - 1)
        assert audio.readframes(1) == b"\xff\x0f\xff\x0f"
    before = output.read_bytes()
    assert prefix.verify(case.request) == output
    assert output.read_bytes() == before
    assert case.calls == [0, 0]
    with pytest.raises(ReplayError, match="Occupied output"):
        prefix.prepare(case.request)


@pytest.mark.parametrize(
    "fault",
    [
        "version",
        "model",
        "runtime",
        "chunks",
        "order",
        "budget",
        "timeout",
        "overlap",
        "context",
        "sha",
    ],
)
def test_request_boundary_rejects_mismatch(case: Case, fault: str) -> None:
    if fault == "version":
        case.raw["version"] = True
    elif fault == "model":
        record(case.raw["model"])["id"] = "whisper-turbo"
    elif fault == "runtime":
        case.raw["runtime"] = {}
    elif fault == "chunks":
        case.raw["chunks"] = []
    elif fault == "order":
        case.raw["chunks"] = [
            {"input": reference(case.plan.path), "complete": reference(case.plan.path)}
        ]
    elif fault == "budget":
        case.raw["maxSavedBytes"] = 1
    elif fault == "timeout":
        case.raw["timeoutSeconds"] = False
    elif fault == "overlap":
        case.raw["outputDirectory"] = str(case.plan.directory / "new-prefix")
    elif fault == "context":
        case.raw["context"] = reference(case.plan.path)
    else:
        record(case.raw["pipeline"])["sha256"] = "0" * 64
    case.request.write_text(json.dumps(case.raw))
    with pytest.raises(ReplayError):
        prefix.prepare(case.request)
    assert not (case.output / "receipt.json").exists()


@pytest.mark.parametrize("fault", ["pcm", "clock", "receipt", "stereo", "mono"])
def test_verify_never_converts_and_rejects_changed_outputs(
    case: Case, monkeypatch: pytest.MonkeyPatch, fault: str
) -> None:
    prefix.prepare(case.request)
    monkeypatch.setattr(
        subprocess, "run", lambda *args, **kwargs: pytest.fail("verifier conversion")
    )
    if fault == "pcm":
        (case.output / "000000000000.pcm").write_bytes(b"changed")
    elif fault == "clock":
        path = case.output / "000000000000.window.json"
        raw = record(json.loads(path.read_text()))
        raw["startSample"] = 1
        path.write_text(json.dumps(raw))
    elif fault == "receipt":
        path = case.output / "receipt.json"
        raw = record(json.loads(path.read_text()))
        raw["adopted"] = True
        path.write_text(json.dumps(raw))
    else:
        path = case.output / f"{fault}.wav"
        with path.open("r+b") as stream:
            stream.seek(100)
            stream.write(b"changed")
    with pytest.raises(ReplayError):
        prefix.verify(case.request)


def test_chunk_complete_alone_does_not_bypass_original_verifier(
    case: Case, monkeypatch: pytest.MonkeyPatch
) -> None:
    def failed(
        plan: pipeline.PipelinePlan, *, index: int, bound: dict[str, object]
    ) -> NDArray[np.float32]:
        raise ReplayError("Original reduction mismatch")

    monkeypatch.setattr(pipeline, "verify_chunk", failed)
    with pytest.raises(ReplayError, match="Original reduction mismatch"):
        prefix.prepare(case.request)
    assert not (case.output / "receipt.json").exists()
    with pytest.raises(ReplayError, match="Occupied"):
        prefix.prepare(case.request)


def test_request_mutation_during_conversion_fails(
    case: Case, monkeypatch: pytest.MonkeyPatch
) -> None:
    original = prefix._windows

    def changed(request: prefix.Request, *, create: bool) -> list[dict[str, object]]:
        result = original(request, create=create)
        case.request.write_text(json.dumps(case.raw, indent=2))
        return result

    monkeypatch.setattr(prefix, "_windows", changed)
    with pytest.raises(ReplayError, match="inputs changed"):
        prefix.prepare(case.request)
    assert not (case.output / "receipt.json").exists()


def test_active_collection_is_not_read_or_written(case: Case) -> None:
    prefix.prepare(case.request)
    with execution_lock(case.output, ".prefix") as acquired:
        assert acquired
        with pytest.raises(ReplayError, match="still owns"):
            prefix.verify(case.request)


def test_first595_seconds_provides18_full_windows_not_a_partial_tail(case: Case) -> None:
    request = prefix.load_request(case.request)
    plan = replace(
        case.plan, bounds=((0, 600 * 44100), (595 * 44100, 1195 * 44100)), samples=1195 * 44100
    )
    assert prefix._stable_samples(replace(request, plan=plan)) == 26239500
    case.output.mkdir()
    with wave.open(str(case.output / "mono.wav"), "wb") as audio:
        audio.setnchannels(1)
        audio.setsampwidth(2)
        audio.setframerate(16000)
        audio.writeframes(b"\0" * (595 * 16000 * 2))
    rows = prefix._windows(replace(request, plan=plan), create=True)
    assert len(rows) == 18
    final = record(json.loads((case.output / "000008160000.window.json").read_text()))
    assert final["startSample"] == 8160000
    assert final["endSample"] == 9120000
    assert not (case.output / "000008640000.pcm").exists()


@pytest.fixture
def final_plan(case: Case, monkeypatch: pytest.MonkeyPatch) -> Path:
    prefix.prepare(case.request)
    directory = case.request.parent / "final"
    directory.mkdir()
    mono = directory / "mono.wav"
    shutil.copyfile(case.output / "mono.wav", mono)
    path = directory / "transcription.json"
    path.write_text('{"synthetic":true}')
    ready = transcription.Ready(
        case.plan.source, path, path, case.plan.source, case.plan.directory / "vocals.wav", path, {}
    )
    settings = transcription.Settings(None, None, directory, 30, {}, (), {}, {}, {})
    plan = transcription.Plan(
        path, directory, ready, settings, (), directory, directory / "asr", None
    )
    monkeypatch.setattr(transcription, "_load", lambda path: plan)

    def verified(path: Path, kind: transcription.Kind) -> None:
        assert kind == "mono"

    monkeypatch.setattr(transcription, "verify", verified)
    return path


def test_final_equality_is_separate_and_does_not_adopt_or_fake_asr(
    case: Case, final_plan: Path
) -> None:
    output = final_plan.parent / "comparison.json"
    assert prefix.compare_final(case.request, final_plan, output) == output
    row = record(json.loads(output.read_text()))
    assert row["finalMonoWindowEqual"] is True
    assert row["adopted"] is False
    assert row["recognitionExecuted"] is False
    assert not (final_plan.parent / "asr").exists()
    with pytest.raises(FileExistsError):
        prefix.compare_final(case.request, final_plan, output)


def test_final_single_sample_difference_refuses_adoption(case: Case, final_plan: Path) -> None:
    with (final_plan.parent / "mono.wav").open("r+b") as stream:
        stream.seek(100)
        stream.write(b"\x01\x00")
    with pytest.raises(ReplayError, match="window bytes differ"):
        prefix.compare_final(case.request, final_plan, final_plan.parent / "bad.json")


def test_missing_final_producer_is_not_replaced_by_a_boolean(
    case: Case, final_plan: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    def failed(path: Path, kind: transcription.Kind) -> None:
        raise ReplayError("Missing producer proof")

    monkeypatch.setattr(transcription, "verify", failed)
    with pytest.raises(ReplayError, match="Missing producer"):
        prefix.compare_final(case.request, final_plan, final_plan.parent / "bad.json")


def test_cli_and_strict_metadata(case: Case, capsys: pytest.CaptureFixture[str]) -> None:
    assert prefix.main(["check", str(case.request)]) == 0
    assert case.calls == []
    assert not case.output.exists()
    assert prefix.main(["prepare", str(case.request)]) == 0
    assert prefix.main(["verify", str(case.request)]) == 0
    assert '"recognitionExecuted": false' in capsys.readouterr().out
    with pytest.raises(SystemExit) as missing:
        prefix.main(["compare-final", str(case.request)])
    assert missing.value.code == 1
    case.request.write_text('{"version":1,"version":1}')
    with pytest.raises(ReplayError, match="Duplicate"):
        prefix.load_request(case.request)


def test_linked_request_and_outside_output_refused(case: Case) -> None:
    alias = case.request.with_name("alias.json")
    alias.symlink_to(case.request)
    with pytest.raises(ReplayError, match="Noncanonical"):
        prefix.load_request(alias)
    case.raw["outputDirectory"] = str(prefix.ROOT.parent / "escape")
    case.request.write_text(json.dumps(case.raw))
    with pytest.raises(ReplayError, match="stay in checkout"):
        prefix.load_request(case.request)


def test_second_chunk_resolves_original_crossfade_without_a_provisional_tail(
    case: Case, monkeypatch: pytest.MonkeyPatch
) -> None:
    folder = case.plan.directory / "chunks/chunk-000001"
    folder.mkdir()
    (folder / "input.json").write_text('{"chunk":1}')
    (folder / "complete.json").write_text('{"complete":true}')
    chunks = case.raw["chunks"]
    assert isinstance(chunks, list)
    chunks.append(
        {"input": reference(folder / "input.json"), "complete": reference(folder / "complete.json")}
    )
    case.request.write_text(json.dumps(case.raw))

    def verified(
        plan: pipeline.PipelinePlan, *, index: int, bound: dict[str, object]
    ) -> NDArray[np.float32]:
        return np.full((2, 65 * 44100), 0.125 if index == 0 else 0.25, dtype=np.float32)

    monkeypatch.setattr(pipeline, "verify_chunk", verified)
    result = prefix.prepare(case.request)
    row = record(json.loads(result.read_text()))
    assert row["stableStereoSamples"] == 5512500
    windows = row["windows"]
    assert isinstance(windows, list) and len(windows) == 3
    with wave.open(str(case.output / "stereo.wav")) as audio:
        audio.setpos(60 * 44100)
        assert audio.readframes(1) == b"\xff\x0f\xff\x0f"
        audio.setpos(65 * 44100)
        assert audio.readframes(1) == b"\xff\x1f\xff\x1f"
    assert not (case.output / "000001440000.pcm").exists()
    prefix.verify(case.request)


def test_real_converter_gets_checkout_local_scratch_and_timeout(
    case: Case, monkeypatch: pytest.MonkeyPatch
) -> None:
    def run(command: tuple[str, ...], **kwargs: object) -> subprocess.CompletedProcess[bytes]:
        assert kwargs["timeout"] == 10
        assert kwargs["cwd"] == prefix.ROOT
        env = kwargs["env"]
        assert isinstance(env, dict)
        assert env["TMPDIR"] == str(case.output / "tmp")
        # This regression stops before a child; the real local conversion is exercised elsewhere.
        raise subprocess.TimeoutExpired(command, 10)

    monkeypatch.setattr(subprocess, "run", run)
    with pytest.raises(subprocess.TimeoutExpired):
        prefix.prepare(case.request)
    assert not (case.output / "receipt.json").exists()


def test_changed_prefix_during_final_producer_verification_refuses(
    case: Case, final_plan: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    def changed(path: Path, kind: transcription.Kind) -> None:
        target = case.output / "receipt.json"
        target.write_text(json.dumps(json.loads(target.read_text()), indent=2))

    monkeypatch.setattr(transcription, "verify", changed)
    with pytest.raises(ReplayError, match="Prefix evidence changed"):
        prefix.compare_final(case.request, final_plan, final_plan.parent / "bad.json")
    assert not (final_plan.parent / "bad.json").exists()


def test_different_final_source_is_not_adopted(
    case: Case, final_plan: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    original = transcription._load(final_plan)
    wrong = replace(original, prepared=replace(original.prepared, stereo=final_plan))
    monkeypatch.setattr(transcription, "_load", lambda path: wrong)
    with pytest.raises(ReplayError, match="original separation"):
        prefix.compare_final(case.request, final_plan, final_plan.parent / "bad.json")


def test_runtime_change_between_load_and_boundary_refuses(
    case: Case, monkeypatch: pytest.MonkeyPatch
) -> None:
    request = prefix.load_request(case.request)
    monkeypatch.setattr(prefix, "runtime_inputs", lambda: {})
    with pytest.raises(ReplayError, match="runtime changed"):
        prefix._bindings(request)


def test_bounded_json_and_missing_pin_rejected(case: Case) -> None:
    with case.request.open("wb") as stream:
        stream.truncate(prefix.MAX_JSON + 1)
    with pytest.raises(ReplayError, match="byte budget"):
        prefix.load_request(case.request)
    with pytest.raises(ReplayError, match="Exact reference"):
        prefix._ref({"path": str(case.request)})
    with pytest.raises(ReplayError, match="Invalid SHA"):
        prefix._ref({"path": str(case.request), "sha256": "unknown"})


def test_non_ffmpeg_tool_cannot_be_substituted(case: Case) -> None:
    case.raw["ffmpeg"] = reference(case.plan.source)
    case.request.write_text(json.dumps(case.raw))
    with pytest.raises(ReplayError, match="FFmpeg executable"):
        prefix.load_request(case.request)
