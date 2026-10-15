"""Checkpointed CPU windows and the original ordered 600s/5s PCM assembly.

This candidate does not modify delivery_audio or resume its historical jobs.
Only this plan's complete, input-bound chunks can be reused. Unknown worker or
partial final-write outcomes remain evidence and never authorize blind inference.
"""

from __future__ import annotations

import argparse
import hashlib
import math
import os
import resource
import subprocess
import sys
import time
import wave
from collections.abc import Callable, Iterator
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
from numpy.typing import NDArray

from replay.bgm_separation import (
    MODEL_CONFIG_SHA256,
    MODEL_NAME,
    MODEL_REVISION,
    MODEL_SAMPLE_RATE,
    MODEL_SHA256,
    SeparationResult,
    _chunk_bounds,
    _crossfade_weights,
)
from replay.code_dependency_inventory import capture
from replay.core import ReplayError
from replay.delivery_audio import verify_material
from replay.native_invocation import (
    canonical,
    durable,
    identity,
    pin,
    read,
    require,
    sync_directory,
)
from replay.recordings import record
from replay.separation_forward_probe import _local_environment
from replay.separation_window_budget import (
    ARRAY_HEADER_ALLOWANCE_BYTES,
    DEVICES,
    METADATA_ALLOWANCE_BYTES,
    Execute,
    ProbePlan,
    WindowBudget,
    _integer,
    _probe_outputs,
    _references,
    _required_provider_paths,
    _verify_pool,
    estimated_saved_bytes,
    execute_children,
    load_array,
    local_directory,
    run_windows,
    save_array,
)
from replay.stages import execution_lock
from replay.stateful_guard import saved_bytes

CHUNK = 600 * MODEL_SAMPLE_RATE
OVERLAP = 5 * MODEL_SAMPLE_RATE
MAX_SAMPLES = 21600 * MODEL_SAMPLE_RATE
MAX_COMPARISON_SAMPLES = 90 * MODEL_SAMPLE_RATE
Prediction = Callable[[int], NDArray[np.float32]]


@dataclass(frozen=True)
class PipelineBudget:
    window: WindowBudget
    max_samples: int
    max_saved_bytes: int
    max_parent_rss_bytes: int
    timeout_seconds: int

    def __post_init__(self) -> None:
        require(
            all(
                type(value) is int and value > 0
                for value in (
                    self.max_samples,
                    self.max_saved_bytes,
                    self.max_parent_rss_bytes,
                    self.timeout_seconds,
                )
            )
            and self.max_samples <= MAX_SAMPLES
            and self.timeout_seconds <= 86400
            and self.window.workers == 2
            and self.window.max_samples == CHUNK,
            "Invalid pipeline budget",
        )


@dataclass(frozen=True)
class PipelinePlan:
    path: Path
    source: Path
    directory: Path
    samples: int
    bounds: tuple[tuple[int, int], ...]
    budget: PipelineBudget
    definition: dict[str, object]


def plan_device(plan: PipelinePlan) -> str:
    """The separation device the plan declares; plans without the field are CPU plans."""
    return str(plan.definition.get("device", "cpu"))


def pcm_info(path: Path) -> int:
    before = identity(path)
    with path.open("rb") as stream, wave.open(stream, "rb") as audio:
        require(
            audio.getnchannels() == 2
            and audio.getsampwidth() == 2
            and audio.getframerate() == MODEL_SAMPLE_RATE
            and audio.getcomptype() == "NONE",
            "Expected original stereo PCM16/44100",
        )
        samples = audio.getnframes()
    require(0 < samples <= MAX_SAMPLES and identity(path) == before, "Invalid/changing PCM clock")
    return samples


def read_chunk(path: Path, *, start: int, count: int) -> NDArray[np.float32]:
    require(
        type(start) is int and start >= 0 and type(count) is int and 0 < count <= CHUNK,
        "Invalid chunk extent",
    )
    before = identity(path)
    with path.open("rb") as stream, wave.open(stream, "rb") as audio:
        require(start + count <= audio.getnframes(), "Chunk exceeds source")
        audio.setpos(start)
        raw = audio.readframes(count)
    require(len(raw) == count * 4 and identity(path) == before, "Truncated/changing source chunk")
    decoded = np.frombuffer(raw, dtype="<i2").reshape(count, 2).T.astype(np.float32)
    decoded /= 32768.0
    return decoded


def load_plan(path: Path) -> PipelinePlan:
    path = canonical(str(path))
    data = read(path)
    require(
        set(data) - {"device"}
        == {
            "version",
            "kind",
            "source",
            "outputDirectory",
            "budget",
            "codeInputs",
            "providerInputs",
            "modelInputs",
            "chunkSamples",
            "overlapSamples",
            "sampleRate",
            "qualityApproved",
            "productionBackendAdopted",
        }
        and type(data.get("version")) is int
        and data.get("version") == 1
        and data.get("kind") == "separation-window-pipeline"
        and _integer(data.get("chunkSamples")) == CHUNK
        and _integer(data.get("overlapSamples")) == OVERLAP
        and _integer(data.get("sampleRate")) == MODEL_SAMPLE_RATE
        and data.get("qualityApproved") is False
        and data.get("productionBackendAdopted") is False
        and data.get("device", "cpu") in DEVICES,
        "Unsupported pipeline plan",
    )
    source_ref = _references([data.get("source")])
    source = Path(next(iter(source_ref)))
    directory = local_directory(canonical(data.get("outputDirectory")))
    row = record(data.get("budget"))
    require(
        set(row)
        == {"window", "max_samples", "max_saved_bytes", "max_parent_rss_bytes", "timeout_seconds"},
        "Invalid pipeline budget fields",
    )
    window = record(row.get("window"))
    require(
        set(window)
        == {"workers", "max_samples", "timeout_seconds", "max_worker_rss_bytes", "max_saved_bytes"},
        "Invalid window budget fields",
    )
    selected = WindowBudget(
        workers=_integer(window.get("workers")),
        max_samples=_integer(window.get("max_samples")),
        timeout_seconds=_integer(window.get("timeout_seconds")),
        max_worker_rss_bytes=_integer(window.get("max_worker_rss_bytes")),
        max_saved_bytes=_integer(window.get("max_saved_bytes")),
    )
    budget = PipelineBudget(
        selected,
        _integer(row.get("max_samples")),
        _integer(row.get("max_saved_bytes")),
        _integer(row.get("max_parent_rss_bytes")),
        _integer(row.get("timeout_seconds")),
    )
    samples = pcm_info(source)
    require(samples <= budget.max_samples, "Pipeline sample budget exceeded")
    model = _references(data.get("modelInputs"))
    snapshot = (
        Path(_local_environment()["HF_HUB_CACHE"])
        / ("models--" + MODEL_NAME.replace("/", "--"))
        / "snapshots"
        / MODEL_REVISION
    )
    require(
        model
        == {
            str(snapshot / "model.safetensors"): MODEL_SHA256,
            str(snapshot / "config.json"): MODEL_CONFIG_SHA256,
        },
        "Pipeline model/cache changed",
    )
    _references(data.get("codeInputs"))
    _references(data.get("providerInputs"))
    bounds = tuple(_chunk_bounds(samples, CHUNK, OVERLAP))
    estimate = estimated_pipeline_bytes(bounds, samples)
    require(
        estimate <= budget.max_saved_bytes
        and max(estimated_saved_bytes(end - start) for start, end in bounds)
        <= selected.max_saved_bytes,
        "Pipeline retained-byte budget insufficient",
    )
    return PipelinePlan(path, source, directory, samples, bounds, budget, data)


def estimated_pipeline_bytes(bounds: tuple[tuple[int, int], ...], samples: int) -> int:
    return (
        samples * 4
        + 4096
        + METADATA_ALLOWANCE_BYTES
        + sum(
            estimated_saved_bytes(end - start) + (end - start) * 8 + ARRAY_HEADER_ALLOWANCE_BYTES
            for start, end in bounds
        )
    )


def code_inputs() -> dict[str, str]:
    return {
        str(item.path): item.sha256
        for item in capture(("separation_window_pipeline", "code_dependency_inventory"))
    }


def context(plan: PipelinePlan) -> dict[str, object]:
    code, providers, model = (
        _references(plan.definition[name])
        for name in ("codeInputs", "providerInputs", "modelInputs")
    )
    require(code == code_inputs(), "Pipeline AST closure changed")
    require(_required_provider_paths() <= set(providers), "Missing installed provider library/code")
    source = _references([plan.definition["source"]])
    combined = {**code, **providers, **model, **source}
    require(
        len(combined) == len(code) + len(providers) + len(model) + 1,
        "Overlapping pipeline pin roles",
    )
    pinned: list[dict[str, object]] = []
    for name, digest in sorted(combined.items()):
        actual = pin(Path(name))
        require(actual["sha256"] == digest, "Pipeline declared input changed")
        pinned.append(actual)
    require(read(plan.path) == plan.definition, "Pipeline plan changed")
    return {"plan": pin(plan.path), "inputs": pinned, "environment": _local_environment()}


def chunk_proxy(plan: PipelinePlan, index: int) -> ProbePlan:
    start, end = plan.bounds[index]
    return ProbePlan(
        plan.path,
        plan.source,
        plan.directory / "chunks" / f"chunk-{index:06d}",
        end - start,
        plan.budget.window,
        plan.definition,
    )


def chunk_files(proxy: ProbePlan) -> tuple[Path, ...]:
    pool = proxy.directory / "parallel"
    return tuple(path for path in _probe_outputs(proxy) if path.is_relative_to(pool))


def chunk_definition(
    plan: PipelinePlan, *, index: int, audio: NDArray[np.float32], bound: dict[str, object]
) -> dict[str, object]:
    start, end = plan.bounds[index]
    return {
        "index": index,
        "startSample": start,
        "endSample": end,
        "audioSHA256": hashlib.sha256(audio.tobytes(order="C")).hexdigest(),
        "context": bound,
        "windowBudget": asdict(plan.budget.window),
    }


def verify_chunk(
    plan: PipelinePlan, *, index: int, bound: dict[str, object]
) -> NDArray[np.float32]:
    proxy = chunk_proxy(plan, index)
    start, end = plan.bounds[index]
    audio = read_chunk(plan.source, start=start, count=end - start)
    require(
        read(proxy.directory / "input.json")
        == chunk_definition(plan, index=index, audio=audio, bound=bound),
        "Chunk source binding changed",
    )
    outputs = (proxy.directory / "input.json", proxy.directory / "vocals.npy", *chunk_files(proxy))
    before = [pin(path) for path in outputs]
    checkpoint = read(proxy.directory / "complete.json")
    require(checkpoint == {"outputs": before, "qualityApproved": False}, "Chunk checkpoint changed")
    reduced = _verify_pool(proxy, "parallel", audio, plan_device(plan))
    saved = load_array(proxy.directory / "vocals.npy", samples=end - start)
    require(
        np.array_equal(reduced, saved) and [pin(path) for path in outputs] == before,
        "Chunk reduction/output changed",
    )
    return saved


def run_chunk(
    plan: PipelinePlan, *, index: int, bound: dict[str, object], execute: Execute, resume: bool
) -> None:
    proxy = chunk_proxy(plan, index)
    if (proxy.directory / "complete.json").exists():
        verify_chunk(plan, index=index, bound=bound)
        return
    start, end = plan.bounds[index]
    audio = read_chunk(plan.source, start=start, count=end - start)
    definition = chunk_definition(plan, index=index, audio=audio, bound=bound)
    if proxy.directory.exists():
        require(
            resume and read(proxy.directory / "input.json") == definition,
            "Unknown partial chunk; explicit matching resume required",
        )
    else:
        proxy.directory.mkdir(mode=0o700, parents=True)
        durable(proxy.directory / "input.json", definition)
    reduced = run_windows(
        audio,
        proxy.directory / "parallel",
        budget=plan.budget.window,
        execute=execute,
        device=plan_device(plan),
    )
    destination = proxy.directory / "vocals.npy"
    if destination.exists():
        require(
            resume and np.array_equal(load_array(destination, samples=end - start), reduced),
            "Partial reduced output changed",
        )
    else:
        save_array(destination, reduced)
    outputs = (proxy.directory / "input.json", destination, *chunk_files(proxy))
    durable(
        proxy.directory / "complete.json",
        {"outputs": [pin(path) for path in outputs], "qualityApproved": False},
    )
    # Verification reconstructs the reduction. Do not retain a second full pair.
    del audio, reduced
    verify_chunk(plan, index=index, bound=bound)


def pcm_blocks(
    bounds: tuple[tuple[int, int], ...], *, samples: int, overlap: int, prediction: Prediction
) -> Iterator[bytes]:
    """Keep original float32 head blend; defer only the next-overlapped tail."""
    require(
        bool(bounds) and bounds[0][0] == 0 and bounds[-1][1] == samples and overlap > 0,
        "Invalid outer clock",
    )
    weights = _crossfade_weights(overlap)
    tail: NDArray[np.float32] | None = None
    retained = 0
    for index, (start, end) in enumerate(bounds):
        current = prediction(index)
        require(
            current.dtype == np.dtype(np.float32)
            and current.shape == (2, end - start)
            and bool(np.isfinite(current).all()),
            "Invalid outer prediction",
        )
        require(
            index == 0 or start == bounds[index - 1][1] - overlap,
            "Noncontiguous original chunk clock",
        )
        if index:
            if tail is None:
                raise ReplayError("Missing outer overlap tail")
            require(current.shape[1] > overlap, "Missing outer overlap tail")
            head = tail * weights[0] + current[:, :overlap] * weights[1]
            yield (np.clip(head.T, -1.0, 1.0) * 32767.0).astype("<i2").tobytes()
            retained += overlap
        low = overlap if index else 0
        high = current.shape[1] - overlap if index + 1 < len(bounds) else current.shape[1]
        require(high >= low, "Invalid deferred outer tail")
        yield (np.clip(current[:, low:high].T, -1.0, 1.0) * 32767.0).astype("<i2").tobytes()
        retained += high - low
        tail = current[:, -overlap:].copy() if index + 1 < len(bounds) else None
    require(retained == samples, "Outer assembly changed sample count")


def write_pcm(path: Path, *, samples: int, blocks: Iterator[bytes]) -> None:
    with path.open("xb") as stream:
        os.fchmod(stream.fileno(), 0o600)
        with wave.open(stream, "wb") as audio:
            audio.setnchannels(2)
            audio.setsampwidth(2)
            audio.setframerate(MODEL_SAMPLE_RATE)
            audio.setnframes(samples)
            written = 0
            for block in blocks:
                require(len(block) % 4 == 0, "Partial stereo sample")
                audio.writeframesraw(block)
                written += len(block) // 4
            require(written == samples, "Incomplete PCM assembly")
        stream.flush()
        os.fsync(stream.fileno())


def parent_peak_rss_bytes() -> int:
    peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return peak if sys.platform == "darwin" else peak * 1024


def check_budget(plan: PipelinePlan, started: float) -> None:
    require(
        saved_bytes(plan.directory) <= plan.budget.max_saved_bytes
        and time.monotonic() - started <= plan.budget.timeout_seconds
        and parent_peak_rss_bytes() <= plan.budget.max_parent_rss_bytes,
        "Pipeline sampled/terminal resource budget exceeded",
    )


def pipeline_outputs(plan: PipelinePlan) -> tuple[Path, ...]:
    paths = [plan.directory / "input.json", plan.directory / "vocals.wav"]
    for index in range(len(plan.bounds)):
        proxy = chunk_proxy(plan, index)
        paths.extend(
            (
                proxy.directory / "input.json",
                proxy.directory / "complete.json",
                proxy.directory / "vocals.npy",
                *chunk_files(proxy),
            )
        )
    return tuple(paths)


def verify(plan_path: Path) -> None:
    """Verify every complete chunk and final PCM without inference or any writes."""
    plan = load_plan(plan_path)
    bound = context(plan)
    require(read(plan.directory / "input.json") == bound, "Pipeline input binding changed")
    before = [pin(path) for path in pipeline_outputs(plan)]
    receipt_pin = pin(plan.directory / "result.json")
    result = read(plan.directory / "result.json")
    require(
        result.get("outputs") == before
        and result.get("inputSampleCount") == result.get("outputSampleCount") == plan.samples
        and result.get("sampleRate") == MODEL_SAMPLE_RATE
        and result.get("qualityApproved") is False
        and result.get("productionBackendAdopted") is False
        and type(result.get("kernelParentPeakRSSBytes")) is int
        and 0 < int(str(result["kernelParentPeakRSSBytes"])) <= plan.budget.max_parent_rss_bytes
        and type(result.get("savedBytesBeforeReport")) is int
        and 0 < int(str(result["savedBytesBeforeReport"])) <= plan.budget.max_saved_bytes
        and type(result.get("elapsedSeconds")) in (int, float)
        and math.isfinite(float(str(result["elapsedSeconds"])))
        and 0 < float(str(result["elapsedSeconds"])) <= plan.budget.timeout_seconds
        and result.get("resourceCapsAreHardLimits") is False,
        "Pipeline receipt/clock changed",
    )
    with (plan.directory / "vocals.wav").open("rb") as stream, wave.open(stream, "rb") as audio:
        for expected in pcm_blocks(
            plan.bounds,
            samples=plan.samples,
            overlap=OVERLAP,
            prediction=lambda index: verify_chunk(plan, index=index, bound=bound),
        ):
            require(
                audio.readframes(len(expected) // 4) == expected,
                "Final PCM differs from original ordered blend",
            )
        require(not audio.readframes(1), "Final PCM has extra samples")
    verify_material(plan.source, plan.directory)
    require(
        context(plan) == bound
        and [pin(path) for path in pipeline_outputs(plan)] == before
        and pin(plan.directory / "result.json") == receipt_pin,
        "Pipeline changed during readonly verification",
    )


def run(plan_path: Path, *, resume: bool = False, execute: Execute = execute_children) -> Path:
    plan = load_plan(plan_path)
    bound = context(plan)
    if plan.directory.exists():
        require(resume, "Pipeline exists; explicit matching resume required")
    else:
        plan.directory.mkdir(mode=0o700)
    started = time.monotonic()
    with execution_lock(plan.directory, ".pipeline") as acquired:
        require(acquired, "Pipeline owner already active")
        if (plan.directory / "input.json").exists():
            require(read(plan.directory / "input.json") == bound, "Pipeline resume input changed")
        else:
            durable(plan.directory / "input.json", bound)
        if (plan.directory / "result.json").exists():
            verify(plan.path)
            return plan.directory / "result.json"
        for index in range(len(plan.bounds)):
            check_budget(plan, started)
            run_chunk(plan, index=index, bound=bound, execute=execute, resume=resume)
            check_budget(plan, started)
        destination = plan.directory / "vocals.wav"
        partial = plan.directory / "vocals.assembling.wav"
        require(
            not destination.exists() and not partial.exists(),
            "Final write outcome uncertain; preserve and reconcile",
        )
        blocks = pcm_blocks(
            plan.bounds,
            samples=plan.samples,
            overlap=OVERLAP,
            prediction=lambda index: load_array(
                chunk_proxy(plan, index).directory / "vocals.npy",
                samples=plan.bounds[index][1] - plan.bounds[index][0],
            ),
        )
        write_pcm(partial, samples=plan.samples, blocks=blocks)
        os.link(partial, destination)
        partial.unlink()
        sync_directory(plan.directory)
        check_budget(plan, started)
        require(context(plan) == bound, "Pipeline declared inputs changed")
        wall = time.monotonic() - started
        result = SeparationResult(
            plan.samples / MODEL_SAMPLE_RATE, wall, (plan.samples / MODEL_SAMPLE_RATE) / wall
        )
        durable(
            plan.directory / "result.json",
            {
                "source": str(plan.source),
                "gainDB": 0.0,
                "preparedInput": True,
                "device": plan_device(plan),
                "modelInput": str(plan.source),
                "vocals": str(destination),
                "separation": repr(result),
                "timestampCorrection": False,
                "loudnessNormalization": False,
                "deliveryComplete": False,
                "inputSampleCount": plan.samples,
                "outputSampleCount": plan.samples,
                "sampleRate": MODEL_SAMPLE_RATE,
                "outputs": [pin(path) for path in pipeline_outputs(plan)],
                "qualityApproved": False,
                "productionBackendAdopted": False,
                "elapsedSeconds": wall,
                "kernelParentPeakRSSBytes": parent_peak_rss_bytes(),
                "savedBytesBeforeReport": saved_bytes(plan.directory),
                "resourceCapsAreHardLimits": False,
            },
        )
    return plan.directory / "result.json"


def compare_pcm(
    plan_path: Path, *, baseline: Path, output: Path, verify_only: bool = False
) -> Path:
    """Measure bounded saved PCM; baseline production provenance remains a Stage input.

    This comparison has no original floating prediction, perceptual music-removal
    approval or production adoption. Its verifier never creates or repairs files.
    """
    verify(plan_path)
    plan = load_plan(plan_path)
    require(plan.samples <= MAX_COMPARISON_SAMPLES, "PCM comparison is limited to 90 seconds")
    baseline, output = local_directory(baseline), canonical(str(output))
    require(
        output.is_relative_to(Path.cwd() / "edits/work") and baseline != plan.directory,
        "Comparison namespace must be separate and local",
    )
    bound = context(plan)
    verify_material(plan.source, baseline)
    files = (
        baseline / "result.json",
        baseline / "vocals.wav",
        plan.directory / "result.json",
        plan.directory / "vocals.wav",
    )
    before = [pin(path) for path in files]
    left, right = (
        read_chunk(path, start=0, count=plan.samples).astype(np.float64) * 32768.0
        for path in (baseline / "vocals.wav", plan.directory / "vocals.wav")
    )
    difference = right - left
    result = {
        "version": 1,
        "kind": "separation-window-pipeline-pcm-comparison",
        "context": bound,
        "outputs": before,
        "sampleCount": plan.samples,
        "sampleRate": MODEL_SAMPLE_RATE,
        "originSample": 0,
        "pcm16MaxAbsoluteDifference": float(np.max(np.abs(difference))),
        "pcm16RMSDifference": float(np.sqrt(np.mean(difference * difference))),
        "pcm16DifferingChannelSamples": int(np.count_nonzero(difference)),
        "floatPredictionComparisonAvailable": False,
        "baselineWorkflowProvenanceRequired": True,
        "qualityApproved": False,
        "productionBackendAdopted": False,
    }
    require(
        context(plan) == bound and [pin(path) for path in files] == before,
        "Comparison input changed",
    )
    if verify_only:
        before_output = pin(output)
        require(
            read(output) == result and pin(output) == before_output, "Comparison result changed"
        )
    else:
        durable(output, result)
    return output


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("plan", type=Path)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--verify", action="store_true")
    parser.add_argument("--compare-baseline", type=Path)
    parser.add_argument("--comparison-output", type=Path)
    args = parser.parse_args(argv)
    try:
        require(not (args.resume and args.verify), "Readonly verification cannot resume")
        require(
            (args.compare_baseline is None) == (args.comparison_output is None),
            "Comparison requires both baseline and output",
        )
        if args.compare_baseline is not None and args.comparison_output is not None:
            require(not args.resume, "Comparison cannot resume processing")
            compare_pcm(
                args.plan,
                baseline=args.compare_baseline,
                output=args.comparison_output,
                verify_only=args.verify,
            )
        elif args.verify:
            verify(args.plan)
        else:
            run(args.plan, resume=args.resume)
    except (ReplayError, OSError, ValueError, wave.Error, subprocess.SubprocessError) as error:
        print(f"WINDOW_PIPELINE_ERROR {error}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
