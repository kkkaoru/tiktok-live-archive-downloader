"""Synthetic native float clocks and real collector receipts; no MLX inference."""

import json
import os
import shutil
import struct
import subprocess
import wave
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pytest

from replay import isolated_metal_asr as metal
from replay import separation_vocal_asr_compare as compare
from replay import workflow
from replay.audio_windows import AudioWindow
from replay.core import ReplayError
from replay.native_invocation import durable, pin
from replay.replay_transcription_plan import mono_command


def write(path: Path, value: object) -> None:
    path.write_text(json.dumps(value), encoding="utf-8")


def ref(path: Path) -> dict[str, object]:
    return {"path": str(path), "sha256": pin(path)["sha256"]}


@dataclass(frozen=True)
class Fixture:
    request: Path
    root: Path
    benchmark: Path
    sources: Path
    graph: Path


@pytest.fixture
def fixture(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Fixture:
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(compare, "ROOT", tmp_path)
    work = tmp_path / "edits/work"
    work.mkdir(parents=True)
    code = tmp_path / "src/code.py"
    code.parent.mkdir()
    code.write_bytes(b"# synthetic code\n")
    monkeypatch.setattr(compare, "_code", lambda: {str(code): pin(code)["sha256"]})
    monkeypatch.setattr(metal, "required_code", lambda: (code,))
    model = tmp_path / "edits/models/large-v3"
    model.mkdir(parents=True)
    config, weights = model / "config.json", model / "weights.npz"
    config.write_bytes(b"config")
    weights.write_bytes(b"weights")
    monkeypatch.setattr(compare, "DEFAULT_CONFIG_SHA256", pin(config)["sha256"])
    monkeypatch.setattr(compare, "DEFAULT_WEIGHTS_SHA256", pin(weights)["sha256"])
    monkeypatch.setattr(metal, "DEFAULT_CONFIG_SHA256", pin(config)["sha256"])
    monkeypatch.setattr(metal, "DEFAULT_WEIGHTS_SHA256", pin(weights)["sha256"])
    decoder = work / "ffmpeg"
    decoder.write_bytes(b"decoder")
    decoder.chmod(0o700)
    sources = work / "sources"
    sources.mkdir()
    benchmark = sources / "benchmark.json"
    write(
        benchmark,
        {
            "version": 1,
            "kind": "separation-device-benchmark",
            "samples": 3 * 44100,
            "startSample": 44100,
            "intervalSHA256": "1" * 64,
        },
    )
    raw: dict[str, object] = {
        "version": 1,
        "kind": compare.KIND,
        "benchmark": ref(benchmark),
        "model": {"config": ref(config), "weights": ref(weights)},
        "metalCodeInputs": {str(code): pin(code)["sha256"]},
        "ffmpeg": ref(decoder),
        "outputDirectory": str(work / "output"),
        "timeoutSeconds": 30,
        "monoTimeoutSeconds": 30,
    }
    for side in ("cpu", "mps"):
        directory = sources / side
        directory.mkdir()
        values = np.full((2, 3 * 44100), 0.125, dtype=np.float32)
        values[0, 0] = 1.125  # float conversion must not clip to [-1,1]
        with (directory / "vocals.npy").open("xb") as stream:
            np.save(stream, values, allow_pickle=False)
        write(
            directory / "worker.json",
            {
                "version": 1,
                "device": side,
                "workers": 1,
                "samples": 3 * 44100,
                "intervalSHA256": "1" * 64,
                "request": pin(benchmark),
                "output": pin(directory / "vocals.npy"),
            },
        )
        write(
            directory / "resources.json",
            {
                "exitCode": 0,
                "familyGone": True,
                "cleanup": {"terminalObserved": True},
                "samples": [{"physicalBytes": 42}],
            },
        )
        (directory / "worker.log").write_bytes(b"")
        for phase in ("before-load", "after-load", "after-inference"):
            write(directory / f"{phase}-descriptors.json", {"externalWritableRegular": []})
        raw[side + "SeparationWorker"] = ref(directory / "worker.json")
        raw[side + "Vocals"] = ref(directory / "vocals.npy")
    request = work / "request.json"
    write(request, raw)
    root = work / "output"
    root.mkdir()
    graph = compare.prepare(request)
    return Fixture(request, root, benchmark, sources, graph)


def execute(command: tuple[str, ...], log: Path, timeout: int, environment: dict[str, str]) -> None:
    assert timeout == 30
    assert environment["TMPDIR"].endswith("output/tmp")
    assert command[1:6] == ("-nostdin", "-hide_banner", "-v", "error", "-n")
    body = Path(command[7]).read_bytes()
    assert struct.unpack("<H", body[20:22])[0] == 3
    count = struct.unpack("<I", body[44:48])[0]
    log.write_bytes(b"synthetic decoder\n")
    with wave.open(command[-1], "wb") as output:
        output.setparams((1, 2, 16000, 0, "NONE", "not compressed"))
        output.writeframes(bytes(round(count * 16000 / 44100) * 2))


class Runtime:
    def __init__(self, *, difference: str = "", fail: bool = False) -> None:
        self.calls = 0
        self.difference = difference
        self.fail = fail

    def execute(
        self, request: metal.Request, window: AudioWindow, work: Path, identities: dict[str, object]
    ) -> None:
        self.calls += 1
        if self.fail:
            raise ReplayError("uncertain synthetic native child")
        assert identities[str(request.pcm)]
        assert window.start_sample == 0
        mps = request.pcm.parent.name == "mps"
        text = "フォロワーさん" if not (mps and self.difference == "text") else "別の語"
        start = 1.02 if mps and self.difference == "time" else 1.0
        words = (
            []
            if mps and self.difference == "count"
            else [{"word": text, "start": start, "end": 1.5}]
        )
        tokens = [2] if mps and self.difference == "tokens" else [1]
        durable(
            work / "raw.json",
            {
                "text": text,
                "language": "ja",
                "segments": [
                    {"text": text, "start": 1.0, "end": 1.5, "tokens": tokens, "words": words}
                ],
            },
        )
        (work / "worker.log").write_bytes(b"")
        durable(
            work / "resources.json",
            {"exitCode": 0, "familyGone": True, "samples": [{"physicalBytes": 42}]},
        )
        durable(
            work / "mlx-memory.json",
            {"device": "metal", "activeBytes": 1, "peakBytes": 2, "cacheBytes": 3},
        )
        for phase in (
            "before-core-import",
            "after-core-import",
            "after-whisper-import",
            "after-inference-eval",
        ):
            durable(work / f"{phase}-descriptors.json", {"externalWritableRegular": []})
        durable(work / "child.json", {"synthetic": True})
        durable(work / "launch.json", {"synthetic": True})


def complete(fixture: Fixture, runtime: Runtime) -> Path:
    for side in ("cpu", "mps"):
        compare.run(fixture.request, "normalize-" + side, execute=execute)
        compare.run(fixture.request, "verify-normalize-" + side)
        compare.run(fixture.request, "recognize-" + side, runtime=runtime)
        compare.run(fixture.request, "verify-recognize-" + side)
    return compare.run(fixture.request, "compare")


def test_same_model_exact_text_tokens_and_native_timing(fixture: Fixture) -> None:
    runtime = Runtime()
    receipt = complete(fixture, runtime)
    result = compare.verify(fixture.request, receipt)
    assert runtime.calls == 2
    for name in (
        "rawTextEqual",
        "orderedSegmentTextEqual",
        "orderedWordTextEqual",
        "tokensEqual",
        "wordTimingExact",
        "allNativeCollectionsVerified",
        "sameModelProviderContext",
    ):
        assert result[name] is True
    assert result["cpuSeparationWorker"] == pin(fixture.sources / "cpu/worker.json")
    assert result["mpsSeparationWorker"] == pin(fixture.sources / "mps/worker.json")
    assert result["qualityApproved"] is False
    assert result["captionFrameEffects"] is None
    assert result["captionFrameEffectsVerified"] is False
    assert (
        json.loads((fixture.root / "cpu/normalization.json").read_text())["floatSamplesExact"]
        is True
    )
    assert np.frombuffer((fixture.root / "cpu/float.wav").read_bytes()[56:], dtype="<f4")[
        0
    ] == np.float32(1.125)
    assert not (fixture.root / "policy").exists()


@pytest.mark.parametrize(
    "difference,field",
    [
        ("text", "rawTextEqual"),
        ("tokens", "tokensEqual"),
        ("time", "wordTimingExact"),
        ("count", "orderedWordTextEqual"),
    ],
)
def test_native_discrepancies_are_reported_without_tolerance(
    fixture: Fixture, difference: str, field: str
) -> None:
    result = compare.verify(fixture.request, complete(fixture, Runtime(difference=difference)))
    assert result[field] is False
    if difference == "time":
        assert json.dumps(result["wordTimingDeltas"]).__contains__("0.020000000000000018")
    assert result["qualityApproved"] is False


def test_normal_graph_sequential_context_and_readonly_reuse(fixture: Fixture) -> None:
    graph = workflow.load_stages(fixture.graph)
    mapping = workflow.stage_graph(graph)
    assert "recognize-cpu" in mapping["normalize-mps"]
    assert "recognize-cpu" in mapping["recognize-mps"]
    runtime = Runtime()
    compare.run(fixture.request, "normalize-cpu", execute=execute)
    compare.run(fixture.request, "recognize-cpu", runtime=runtime)
    # Collection resume is the original producer; wrapper outputs remain immutable.
    metal.collect(
        fixture.root / "cpu/metal-request.json", fixture.root / "cpu/asr", runtime=runtime
    )
    assert runtime.calls == 1
    compare.run(fixture.request, "verify-recognize-cpu")


@pytest.mark.parametrize(
    "target",
    [
        "vocals.npy",
        "worker.json",
        "resources.json",
        "after-inference-descriptors.json",
        "worker.log",
    ],
)
def test_changed_original_separation_evidence_rejected(fixture: Fixture, target: str) -> None:
    path = fixture.sources / "cpu" / target
    path.write_bytes(b"changed")
    with pytest.raises((ReplayError, ValueError)):
        compare.run(fixture.request, "normalize-cpu", execute=execute)
    assert not (fixture.root / "cpu/normalization.json").exists()


def test_failed_native_attempt_is_never_blindly_resubmitted(fixture: Fixture) -> None:
    compare.run(fixture.request, "normalize-cpu", execute=execute)
    runtime = Runtime(fail=True)
    with pytest.raises(ReplayError, match="uncertain synthetic"):
        compare.run(fixture.request, "recognize-cpu", runtime=runtime)
    with pytest.raises(ReplayError):
        compare.run(fixture.request, "recognize-cpu", runtime=runtime)
    assert runtime.calls == 1


@pytest.mark.parametrize(
    "target",
    [
        "cpu/float.wav",
        "cpu/mono.wav",
        "cpu/metal-request.json",
        "cpu/asr/000000000000.raw.json",
        "cpu/asr/000000000000.clock.json",
        "cpu/asr/000000000000.metal-runtime/resources.json",
        "comparison.json",
    ],
)
def test_tampered_normalized_or_native_artifacts_rejected(fixture: Fixture, target: str) -> None:
    receipt = complete(fixture, Runtime())
    (fixture.root / target).write_bytes(b"{}")
    with pytest.raises((ReplayError, ValueError, wave.Error, EOFError)):
        compare.verify(fixture.request, receipt)


def test_metadata_prepare_no_conversion_and_cli_errors(
    fixture: Fixture, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    assert not (fixture.root / "cpu/float.wav").exists()
    with pytest.raises(ReplayError, match="Empty purpose"):
        compare.prepare(fixture.request)
    with pytest.raises(ReplayError, match="Unknown phase"):
        compare.run(fixture.request, "bad")
    with pytest.raises(ReplayError, match="Unknown comparison"):
        compare.run(fixture.request, "bad-cpu")
    monkeypatch.setattr(compare, "run", lambda request, phase: fixture.root / "result.json")
    assert compare.main(["compare", str(fixture.request)]) == 0
    assert "result.json" in capsys.readouterr().out


def test_invalid_arrays_native_values_and_clock(fixture: Fixture) -> None:
    path = fixture.root / "bad.npy"
    with path.open("xb") as stream:
        np.save(stream, np.ones((2, 2), dtype=np.float64), allow_pickle=False)
    with pytest.raises(ReplayError, match="float array"):
        compare._array(path, 2)
    with pytest.raises(ReplayError, match="native timestamp"):
        compare._native({"text": "x", "segments": [{"text": "x", "start": -1, "end": 1}]})
    with pytest.raises(ReplayError, match="native tokens"):
        compare._native(
            {"text": "x", "segments": [{"text": "x", "start": 0, "end": 1, "tokens": [True]}]}
        )


def test_executor_owned_log_and_exact_deadline(
    fixture: Fixture, monkeypatch: pytest.MonkeyPatch
) -> None:
    observed: list[object] = []

    def run(command: tuple[str, ...], **kwargs: object) -> None:
        observed.append(kwargs["timeout"])

    monkeypatch.setattr(subprocess, "run", run)
    log = fixture.root / "execute.log"
    compare._execute(("synthetic",), log, 5, {})
    assert observed == [5]
    assert log.stat().st_mode & 0o777 == 0o600


def test_real_float_wav_decodes_without_stretch_or_padding(fixture: Fixture) -> None:
    executable = shutil.which("ffmpeg")
    assert executable is not None
    source = fixture.root / "synthetic-float.wav"
    source.write_bytes(compare._wav_bytes(np.full((2, 4410), 0.25, dtype=np.float32)))
    mono = fixture.root / "synthetic-mono.wav"
    environment = os.environ.copy()
    environment.update(
        {
            "TMPDIR": str(fixture.root / "tmp"),
            "TMP": str(fixture.root / "tmp"),
            "TEMP": str(fixture.root / "tmp"),
            "XDG_CACHE_HOME": str(fixture.root / "cache"),
        }
    )
    compare._execute(
        mono_command(source, mono, Path(executable).resolve()),
        fixture.root / "synthetic-decode.log",
        10,
        environment,
    )
    assert compare._mono(mono) == 1600
    with wave.open(str(mono), "rb") as audio:
        assert audio.getnchannels() == 1
        assert audio.getframerate() == 16000
        assert audio.getsampwidth() == 2
        assert struct.unpack("<h", audio.readframes(1)) == (8192,)
