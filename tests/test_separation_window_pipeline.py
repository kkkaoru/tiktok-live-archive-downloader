"""Original PCM equality, complete-only resume and bounded candidate guards."""

from __future__ import annotations

from contextlib import AbstractContextManager, nullcontext
from dataclasses import asdict
from pathlib import Path

import numpy as np
import pytest
from numpy.typing import NDArray

from replay import separation_window_budget as window
from replay import separation_window_pipeline as pipeline
from replay.audio_cpu_probe import sha256
from replay.bgm_separation import (
    MODEL_NAME,
    MODEL_REVISION,
    MODEL_SAMPLE_RATE,
    SeparationConfig,
    _chunk_bounds,
    _MlxBackend,
    _write_wav,
    separate_vocals,
)
from replay.core import ReplayError
from replay.delivery_audio import separate
from replay.native_invocation import durable, pin, read
from replay.recordings import record


class FakeMX:
    def stream(self, device: str) -> AbstractContextManager[None]:
        assert device == "cpu"
        return nullcontext()

    def array(self, value: object) -> NDArray[np.float32]:
        return np.asarray(value, dtype=np.float32)

    def eval(self, value: object) -> None:
        assert isinstance(value, np.ndarray)


class Synthetic(_MlxBackend):
    def __init__(self) -> None:
        self._mx = FakeMX()
        self._device = "cpu"
        self._model = self.forward

    def forward(self, value: object) -> NDArray[np.float32]:
        audio = np.asarray(value, dtype=np.float32)
        return audio * np.float32(0.25) + audio.mean(axis=-1, keepdims=True, dtype=np.float32)


@pytest.fixture
def setup(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> tuple[Path, window.Execute]:
    monkeypatch.chdir(tmp_path)
    work = tmp_path / "edits/work"
    work.mkdir(parents=True)
    cache = work / "cache"
    cache.mkdir()
    tuple(
        map(
            lambda name: monkeypatch.setenv(name, str(cache)),
            ("TMPDIR", "TMP", "TEMP", "XDG_CACHE_HOME", "HF_HOME", "HF_HUB_CACHE"),
        )
    )
    monkeypatch.setenv("HF_HUB_OFFLINE", "1")
    monkeypatch.setenv("TRANSFORMERS_OFFLINE", "1")
    monkeypatch.setattr(window, "provider_pins", lambda: {"fake": "synthetic"})

    def execute(directories: tuple[Path, ...], budget: window.WindowBudget) -> dict[str, object]:
        def one(directory: Path) -> None:
            environment = record(read(directory / "request.json")["environment"])
            durable(directory / "attempt.json", {"status": "reserved", "freshProcess": True})
            (directory / "worker.log").write_bytes(b"synthetic worker\n")

            def audit(path: Path, phase: str) -> None:
                durable(
                    path / f"{phase}-descriptors.json",
                    {"pid": 123, "snapshot": "p123\n", "externalWritableRegular": []},
                )

            with monkeypatch.context() as patch:
                tuple(
                    map(
                        lambda name: patch.setenv(name, str(environment[name])),
                        ("TMPDIR", "TMP", "TEMP", "XDG_CACHE_HOME"),
                    )
                )
                window.worker(directory, backend_factory=Synthetic, auditor=audit)

        tuple(map(one, reversed(directories)))
        return {"synthetic": True}

    return work, execute


@pytest.fixture
def plan(setup: tuple[Path, window.Execute], monkeypatch: pytest.MonkeyPatch) -> Path:
    work, _ = setup
    source = work / "source.wav"
    _write_wav(
        source,
        np.linspace(-0.2, 0.4, 62 * MODEL_SAMPLE_RATE, dtype=np.float32).reshape(
            2, 31 * MODEL_SAMPLE_RATE
        ),
    )
    code, provider = work / "code.py", work / "provider.so"
    code.write_text("# synthetic\n", encoding="utf-8")
    provider.write_bytes(b"provider")
    snapshot = (
        work / "cache" / ("models--" + MODEL_NAME.replace("/", "--")) / "snapshots" / MODEL_REVISION
    )
    snapshot.mkdir(parents=True)
    weights, config = snapshot / "model.safetensors", snapshot / "config.json"
    weights.write_bytes(b"weights")
    config.write_bytes(b"{}")
    monkeypatch.setattr(pipeline, "MODEL_SHA256", sha256(weights))
    monkeypatch.setattr(pipeline, "MODEL_CONFIG_SHA256", sha256(config))
    monkeypatch.setattr(pipeline, "code_inputs", lambda: {str(code): sha256(code)})
    monkeypatch.setattr(pipeline, "_required_provider_paths", lambda: {str(provider)})
    monkeypatch.setattr(
        pipeline,
        "_chunk_bounds",
        lambda total, chunk, overlap: _chunk_bounds(
            total, 30 * MODEL_SAMPLE_RATE, 5 * MODEL_SAMPLE_RATE
        ),
    )
    selected = window.WindowBudget(
        workers=2,
        max_samples=pipeline.CHUNK,
        timeout_seconds=3600,
        max_worker_rss_bytes=12884901888,
        max_saved_bytes=1073741824,
    )
    budget = pipeline.PipelineBudget(selected, pipeline.MAX_SAMPLES, 2147483648, 4294967296, 86400)
    path = work / "plan.json"
    durable(
        path,
        {
            "version": 1,
            "kind": "separation-window-pipeline",
            "source": {"path": str(source), "sha256": sha256(source)},
            "outputDirectory": str(work / "pipeline"),
            "budget": {**asdict(budget), "window": asdict(selected)},
            "codeInputs": [{"path": str(code), "sha256": sha256(code)}],
            "providerInputs": [{"path": str(provider), "sha256": sha256(provider)}],
            "modelInputs": [
                {"path": str(weights), "sha256": sha256(weights)},
                {"path": str(config), "sha256": sha256(config)},
            ],
            "chunkSamples": pipeline.CHUNK,
            "overlapSamples": pipeline.OVERLAP,
            "sampleRate": MODEL_SAMPLE_RATE,
            "qualityApproved": False,
            "productionBackendAdopted": False,
        },
    )
    return path


def replace_json(path: Path, data: dict[str, object]) -> None:
    path.unlink()
    durable(path, data)


@pytest.mark.parametrize("seconds", [29.0, 30.0, 31.0, 54.9, 55.1, 81.7])
def test_original_outer_order_and_runt_tail_pcm(
    setup: tuple[Path, window.Execute], seconds: float
) -> None:
    work, _ = setup
    samples = int(seconds * MODEL_SAMPLE_RATE)
    source = work / "source.wav"
    _write_wav(source, np.linspace(-0.3, 0.5, samples * 2, dtype=np.float32).reshape(2, samples))
    expected = work / "original.wav"
    separate_vocals(
        source,
        expected,
        SeparationConfig(chunk_seconds=30, overlap_seconds=5, device="cpu"),
        Synthetic(),
    )
    bounds = tuple(_chunk_bounds(samples, 30 * MODEL_SAMPLE_RATE, 5 * MODEL_SAMPLE_RATE))

    def prediction(index: int) -> NDArray[np.float32]:
        start, end = bounds[index]
        return Synthetic().extract_vocals(
            pipeline.read_chunk(source, start=start, count=end - start)
        )

    actual = work / "streamed.wav"
    pipeline.write_pcm(
        actual,
        samples=samples,
        blocks=pipeline.pcm_blocks(
            bounds, samples=samples, overlap=5 * MODEL_SAMPLE_RATE, prediction=prediction
        ),
    )
    assert actual.read_bytes() == expected.read_bytes()


def test_full_plan_original_receipt_and_readonly(
    plan: Path, setup: tuple[Path, window.Execute], monkeypatch: pytest.MonkeyPatch
) -> None:
    work, execute = setup
    receipt = pipeline.run(plan, execute=execute)
    original = work / "original.wav"
    separate_vocals(
        work / "source.wav",
        original,
        SeparationConfig(chunk_seconds=30, overlap_seconds=5, device="cpu"),
        Synthetic(),
    )
    assert (work / "pipeline/vocals.wav").read_bytes() == original.read_bytes()
    result = read(receipt)
    assert result["inputSampleCount"] == result["outputSampleCount"] == 31 * MODEL_SAMPLE_RATE
    before = {
        str(path): pin(path)
        for path in (work / "pipeline").rglob("*")
        if path.is_file() and path.stat().st_size
    }
    monkeypatch.setattr(pipeline, "durable", lambda *args: pytest.fail("readonly wrote"))
    pipeline.verify(plan)
    assert before == {
        str(path): pin(path)
        for path in (work / "pipeline").rglob("*")
        if path.is_file() and path.stat().st_size
    }
    assert (
        pipeline.run(plan, resume=True, execute=lambda *args: pytest.fail("repeated inference"))
        == receipt
    )


def test_resume_after_complete_first_chunk_never_repeats(
    plan: Path, setup: tuple[Path, window.Execute]
) -> None:
    work, execute = setup
    calls = 0

    def stop_second(
        directories: tuple[Path, ...], budget: window.WindowBudget
    ) -> dict[str, object]:
        nonlocal calls
        calls += 1
        if calls == 2:
            raise ReplayError("synthetic interruption before new child")
        return execute(directories, budget)

    with pytest.raises(ReplayError, match="interruption"):
        pipeline.run(plan, execute=stop_second)
    first = (work / "pipeline/chunks/chunk-000000/complete.json").read_bytes()
    pipeline.run(plan, resume=True, execute=execute)
    assert (work / "pipeline/chunks/chunk-000000/complete.json").read_bytes() == first


@pytest.mark.parametrize(
    "leaf",
    [
        "chunks/chunk-000000/complete.json",
        "chunks/chunk-000001/vocals.npy",
        "chunks/chunk-000000/parallel/worker-00/result.json",
    ],
)
def test_missing_completed_evidence_never_repaired(
    plan: Path, setup: tuple[Path, window.Execute], leaf: str
) -> None:
    work, execute = setup
    pipeline.run(plan, execute=execute)
    path = work / "pipeline" / leaf
    path.unlink()
    with pytest.raises((ReplayError, OSError)):
        pipeline.verify(plan)
    assert not path.exists()
    with pytest.raises((ReplayError, OSError)):
        pipeline.run(plan, resume=True, execute=lambda *args: pytest.fail("repeated inference"))


def test_changed_source_refuses_resume(plan: Path, setup: tuple[Path, window.Execute]) -> None:
    work, execute = setup
    pipeline.run(plan, execute=execute)
    with (work / "source.wav").open("ab") as stream:
        stream.write(b"changed")
    with pytest.raises(ReplayError, match="input changed"):
        pipeline.run(plan, resume=True, execute=execute)


def test_unknown_final_write_preserved(
    plan: Path, setup: tuple[Path, window.Execute], monkeypatch: pytest.MonkeyPatch
) -> None:
    work, execute = setup
    original = pipeline.write_pcm

    def partial(path: Path, *, samples: int, blocks: object) -> None:
        path.write_bytes(b"interrupted final")
        raise ReplayError("synthetic final interruption")

    monkeypatch.setattr(pipeline, "write_pcm", partial)
    with pytest.raises(ReplayError, match="final interruption"):
        pipeline.run(plan, execute=execute)
    monkeypatch.setattr(pipeline, "write_pcm", original)
    with pytest.raises(ReplayError, match="outcome uncertain"):
        pipeline.run(plan, resume=True, execute=lambda *args: pytest.fail("repeated model"))
    assert (work / "pipeline/vocals.assembling.wav").read_bytes() == b"interrupted final"


@pytest.mark.parametrize(
    "field,value",
    [
        ("version", True),
        ("chunkSamples", 30),
        ("sampleRate", 44100.0),
        ("qualityApproved", True),
        ("other", 1),
    ],
)
def test_invalid_plan(plan: Path, field: str, value: object) -> None:
    data = read(plan)
    data[field] = value
    replace_json(plan, data)
    with pytest.raises(ReplayError):
        pipeline.load_plan(plan)


@pytest.mark.parametrize(
    "field,value",
    [
        ("max_samples", 1),
        ("max_saved_bytes", 1),
        ("max_parent_rss_bytes", 0),
        ("timeout_seconds", 86401),
    ],
)
def test_invalid_pipeline_budget(plan: Path, field: str, value: int) -> None:
    data = read(plan)
    budget = record(data["budget"])
    budget[field] = value
    replace_json(plan, data)
    with pytest.raises(ReplayError):
        pipeline.load_plan(plan)


def test_declared_ast_provider_model_pins(plan: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    selected = pipeline.load_plan(plan)
    monkeypatch.setattr(pipeline, "code_inputs", lambda: {"missing": "a" * 64})
    with pytest.raises(ReplayError, match="AST"):
        pipeline.context(selected)


def test_owned_guard_failures(
    plan: Path, setup: tuple[Path, window.Execute], monkeypatch: pytest.MonkeyPatch
) -> None:
    _, execute = setup
    data = read(plan)
    budget = record(data["budget"])
    budget["max_parent_rss_bytes"] = 1
    replace_json(plan, data)
    with pytest.raises(ReplayError, match="resource budget"):
        pipeline.run(plan, execute=execute)


def test_pcm_shape_clock_and_truncation(setup: tuple[Path, window.Execute]) -> None:
    work, _ = setup
    path = work / "mono.wav"
    _write_wav(path, np.zeros((1, MODEL_SAMPLE_RATE), dtype=np.float32))
    with pytest.raises(ReplayError, match="stereo"):
        pipeline.pcm_info(path)
    with pytest.raises(ReplayError, match="extent"):
        pipeline.read_chunk(path, start=0, count=pipeline.CHUNK + 1)
    with pytest.raises(ReplayError, match="exceeds"):
        pipeline.read_chunk(path, start=0, count=2 * MODEL_SAMPLE_RATE)
    with pytest.raises(ReplayError, match="Partial stereo"):
        pipeline.write_pcm(work / "bad.wav", samples=1, blocks=iter([b"x"]))
    with pytest.raises(ReplayError, match="Incomplete"):
        pipeline.write_pcm(work / "short.wav", samples=2, blocks=iter([b"1234"]))


def test_outer_wrong_clock_rejected() -> None:
    with pytest.raises(ReplayError, match="outer clock"):
        tuple(
            pipeline.pcm_blocks(
                (),
                samples=1,
                overlap=1,
                prediction=lambda index: np.zeros((2, 1), dtype=np.float32),
            )
        )
    with pytest.raises(ReplayError, match="prediction"):
        tuple(
            pipeline.pcm_blocks(
                ((0, 1),),
                samples=1,
                overlap=1,
                prediction=lambda index: np.zeros((1, 1), dtype=np.float32),
            )
        )


@pytest.mark.parametrize(
    "field,value",
    [
        ("elapsedSeconds", float("nan")),
        ("kernelParentPeakRSSBytes", 0),
        ("savedBytesBeforeReport", True),
        ("resourceCapsAreHardLimits", True),
    ],
)
def test_readonly_resource_receipt_rejects_tampering(
    plan: Path, setup: tuple[Path, window.Execute], field: str, value: object
) -> None:
    _, execute = setup
    path = pipeline.run(plan, execute=execute)
    data = read(path)
    data[field] = value
    replace_json(path, data)
    with pytest.raises(ReplayError, match="receipt/clock"):
        pipeline.verify(plan)


def test_cli_routes(plan: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    calls: list[str] = []
    monkeypatch.setattr(pipeline, "run", lambda path, resume: calls.append("run"))
    monkeypatch.setattr(pipeline, "verify", lambda path: calls.append("verify"))
    assert pipeline.main([str(plan), "--resume"]) == 0
    assert pipeline.main([str(plan), "--verify"]) == 0
    assert pipeline.main([str(plan), "--resume", "--verify"]) == 1
    assert calls == ["run", "verify"]


def test_real_recipe_pcm_comparison_and_pure_verifier(
    plan: Path, setup: tuple[Path, window.Execute], monkeypatch: pytest.MonkeyPatch
) -> None:
    work, execute = setup
    baseline = work / "baseline"
    separate(
        work / "source.wav",
        baseline,
        device="cpu",
        prepared_input=True,
        separator=lambda source, output: separate_vocals(
            source,
            output,
            SeparationConfig(chunk_seconds=30, overlap_seconds=5, device="cpu"),
            Synthetic(),
        ),
    )
    pipeline.run(plan, execute=execute)
    output = work / "comparison.json"
    pipeline.compare_pcm(plan, baseline=baseline, output=output)
    result = read(output)
    assert result["pcm16MaxAbsoluteDifference"] == 0.0
    assert result["pcm16RMSDifference"] == 0.0
    assert result["pcm16DifferingChannelSamples"] == 0
    assert result["floatPredictionComparisonAvailable"] is False
    assert result["qualityApproved"] is False
    before = pin(output)
    monkeypatch.setattr(pipeline, "durable", lambda *args: pytest.fail("readonly wrote"))
    pipeline.compare_pcm(plan, baseline=baseline, output=output, verify_only=True)
    assert pin(output) == before
    changed = read(output)
    changed["pcm16DifferingChannelSamples"] = 1
    replace_json(output, changed)
    with pytest.raises(ReplayError, match="Comparison result"):
        pipeline.compare_pcm(plan, baseline=baseline, output=output, verify_only=True)


def test_comparison_cli_routes(plan: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    calls: list[bool] = []
    monkeypatch.setattr(
        pipeline,
        "compare_pcm",
        lambda path, baseline, output, verify_only: calls.append(verify_only),
    )
    assert pipeline.main([str(plan), "--compare-baseline", "missing"]) == 1
    arguments = [str(plan), "--compare-baseline", "baseline", "--comparison-output", "output"]
    assert pipeline.main(arguments) == 0
    assert pipeline.main([*arguments, "--verify"]) == 0
    assert pipeline.main([*arguments, "--resume"]) == 1
    assert calls == [False, True]
