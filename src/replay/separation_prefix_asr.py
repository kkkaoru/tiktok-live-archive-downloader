"""Immutable full ASR windows from verified separation chunks, never a full ASR receipt.

Prefix EOF resampling is explicitly provisional until identical windows are
observed in the genuine final mono producer output. No model is loaded here.
"""

from __future__ import annotations

import argparse
import hashlib
import itertools
import json
import os
import re
import subprocess
import wave
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path

from replay import replay_transcription_plan as transcription
from replay import separation_window_pipeline as pipeline
from replay.code_dependency_inventory import capture
from replay.core import ReplayError
from replay.disk_budget import require_headroom
from replay.local_asr import DEFAULT_CONFIG_SHA256, DEFAULT_WEIGHTS_SHA256
from replay.native_invocation import canonical, durable, identity, pin, require
from replay.recordings import record
from replay.stages import execution_lock

ROOT = Path(__file__).resolve().parents[2]
RATE = 16000
WINDOW = 60 * RATE
STRIDE = 30 * RATE
MAX_JSON = 8 * 1024**2
MAX_SAVED = 8 * 1024**3
MODEL = "mlx-community/whisper-large-v3-mlx"


@dataclass(frozen=True)
class Ref:
    path: Path
    sha256: str

    def data(self) -> dict[str, str]:
        return {"path": str(self.path), "sha256": self.sha256}


@dataclass(frozen=True)
class Request:
    path: Path
    definition: dict[str, object]
    plan: pipeline.PipelinePlan
    context: Ref
    chunks: tuple[tuple[Ref, Ref], ...]
    ffmpeg: Ref
    output: Path
    saved_limit: int
    timeout: int


def _pairs(items: list[tuple[str, object]]) -> dict[str, object]:
    result: dict[str, object] = {}
    for key, value in items:
        require(key not in result, "Duplicate metadata key")
        result[key] = value
    return result


def _read(path: Path) -> dict[str, object]:
    require(0 < identity(path)["bytes"] <= MAX_JSON, "Metadata byte budget exceeded")
    return record(json.loads(path.read_bytes(), object_pairs_hook=_pairs))


def _ref(value: object) -> Ref:
    row = record(value)
    sha = row.get("sha256")
    require(set(row) == {"path", "sha256"}, "Exact reference fields required")
    require(isinstance(sha, str) and re.fullmatch("[0-9a-f]{64}", sha) is not None, "Invalid SHA")
    return Ref(canonical(row["path"]), str(sha))


def _check(ref: Ref) -> dict[str, object]:
    actual = pin(ref.path)
    require(actual["sha256"] == ref.sha256, "Pinned input changed")
    return actual


def _local(path: Path) -> Path:
    path = canonical(str(path))
    require(path.is_relative_to(ROOT / "edits/work"), "Output must stay in checkout work")
    return path


def runtime_inputs() -> dict[str, str]:
    """Capture bounded static source dependencies, without executing their code."""
    return {str(item.path): item.sha256 for item in capture(("separation_prefix_asr",))}


def load_request(path: Path) -> Request:
    path = _local(path)
    raw = _read(path)
    require(
        set(raw)
        == {
            "version",
            "pipeline",
            "context",
            "chunks",
            "ffmpeg",
            "outputDirectory",
            "runtime",
            "model",
            "maxSavedBytes",
            "timeoutSeconds",
        }
        and type(raw["version"]) is int
        and raw["version"] == 1,
        "Unsupported prefix request",
    )
    plan_ref, context_ref, ffmpeg = (_ref(raw[key]) for key in ("pipeline", "context", "ffmpeg"))
    require(
        ffmpeg.path.name == "ffmpeg" and os.access(ffmpeg.path, os.X_OK),
        "Expected explicitly pinned FFmpeg executable",
    )
    _check(plan_ref)
    _check(context_ref)
    plan = pipeline.load_plan(plan_ref.path)
    require(context_ref.path == plan.directory / "input.json", "Original pipeline context required")
    bound = _read(context_ref.path)
    require(bound.get("plan") == pin(plan.path), "Pipeline context belongs to another plan")
    require(raw["runtime"] == runtime_inputs(), "Prefix runtime changed")
    require(
        raw["model"]
        == {
            "id": MODEL,
            "configSHA256": DEFAULT_CONFIG_SHA256,
            "weightsSHA256": DEFAULT_WEIGHTS_SHA256,
        },
        "ASR model pins differ",
    )
    chunks = raw["chunks"]
    require(isinstance(chunks, list), "Expected chunk references")
    if not isinstance(chunks, list):
        raise ReplayError("Expected chunk references")
    require(0 < len(chunks) <= len(plan.bounds), "Invalid completed prefix length")
    refs: list[tuple[Ref, Ref]] = []
    for index, item in enumerate(chunks):
        row = record(item)
        require(set(row) == {"input", "complete"}, "Exact chunk references required")
        pair = _ref(row["input"]), _ref(row["complete"])
        folder = pipeline.chunk_proxy(plan, index).directory
        require(
            pair[0].path == folder / "input.json" and pair[1].path == folder / "complete.json",
            "Only a contiguous original chunk prefix is supported",
        )
        refs.append(pair)
    maximum, timeout = raw["maxSavedBytes"], raw["timeoutSeconds"]
    require(type(maximum) is int and 0 < maximum <= MAX_SAVED, "Invalid saved-byte cap")
    require(type(timeout) is int and 0 < timeout <= 1800, "Invalid FFmpeg timeout")
    if not isinstance(maximum, int) or not isinstance(timeout, int):
        raise ReplayError("Expected integer budgets")
    output = _local(canonical(raw["outputDirectory"]))
    protected = (path, plan.path, plan.directory, plan.source, context_ref.path, ffmpeg.path)
    require(
        not any(
            p == output or p.is_relative_to(output) or output.is_relative_to(p) for p in protected
        ),
        "Output overlaps original pipeline or inputs",
    )
    return Request(path, raw, plan, context_ref, tuple(refs), ffmpeg, output, maximum, timeout)


def _stable_samples(request: Request) -> int:
    end = request.plan.bounds[len(request.chunks) - 1][1]
    return end if len(request.chunks) == len(request.plan.bounds) else end - pipeline.OVERLAP


def _bindings(request: Request) -> dict[str, object]:
    refs = (request.context, request.ffmpeg, *itertools.chain.from_iterable(request.chunks))
    runtime = runtime_inputs()
    require(
        _read(request.path) == request.definition and runtime == request.definition["runtime"],
        "Request or runtime changed at boundary",
    )
    return {
        "request": pin(request.path),
        "plan": _check(_ref(request.definition["pipeline"])),
        "references": [_check(ref) for ref in refs],
        "sourceIdentity": identity(request.plan.source),
        "runtime": runtime,
    }


def _blocks(request: Request) -> Iterator[bytes]:
    """Stop before the first uncommitted prediction without flushing its deferred tail."""
    bound = _read(request.context.path)
    blocks = pipeline.pcm_blocks(
        request.plan.bounds,
        samples=request.plan.samples,
        overlap=pipeline.OVERLAP,
        prediction=lambda index: pipeline.verify_chunk(request.plan, index=index, bound=bound),
    )
    yield from itertools.islice(blocks, 2 * len(request.chunks) - 1)


def _mono_samples(path: Path) -> int:
    before = identity(path)
    with wave.open(str(path), "rb") as audio:
        require(
            (audio.getnchannels(), audio.getsampwidth(), audio.getframerate(), audio.getcomptype())
            == (1, 2, RATE, "NONE"),
            "Expected mono PCM16/16k",
        )
        samples = audio.getnframes()
        require(
            samples > 0 and samples <= pipeline.MAX_SAMPLES * RATE // 44100 + 1,
            "Invalid mono sample count",
        )
    require(identity(path) == before, "Mono changed during inspection")
    return samples


def _windows(request: Request, *, create: bool) -> list[dict[str, object]]:
    mono = request.output / "mono.wav"
    count = _mono_samples(mono)
    require(
        abs(count * 44100 - _stable_samples(request) * RATE) <= 44100,
        "Resampling changed prefix duration",
    )
    result: list[dict[str, object]] = []
    with wave.open(str(mono), "rb") as audio:
        for start in range(0, count - WINDOW + 1, STRIDE):
            audio.setpos(start)
            payload = audio.readframes(WINDOW)
            require(len(payload) == WINDOW * 2, "Truncated complete window")
            destination = request.output / f"{start:012d}.pcm"
            receipt = destination.with_suffix(".window.json")
            sha = hashlib.sha256(payload).hexdigest()
            window: dict[str, object] = {
                "version": 1,
                "kind": "verified-separation-prefix-window",
                "request": pin(request.path),
                "startSample": start,
                "endSample": start + WINDOW,
                "sampleRate": RATE,
                "clock": "input-pcm-not-projected",
                "channels": 1,
                "sampleWidth": 2,
                "pcm": {"path": str(destination), "sha256": sha},
                "model": request.definition["model"],
                "modelBytesVerified": False,
                "recognitionExecuted": False,
                "finalMonoWindowEqual": False,
                "adopted": False,
                "qualityApproved": False,
            }
            if create:
                with destination.open("xb") as stream:
                    stream.write(payload)
                    stream.flush()
                    os.fsync(stream.fileno())
                durable(receipt, window)
            require(
                _read(receipt) == window and pin(destination)["sha256"] == sha,
                "Immutable window evidence differs",
            )
            result.append({"pcm": pin(destination), "receipt": pin(receipt)})
    require(bool(result), "No complete 60-second window is ready")
    return result


def _result(
    request: Request, bindings: dict[str, object], windows: list[dict[str, object]]
) -> dict[str, object]:
    stereo, mono = request.output / "stereo.wav", request.output / "mono.wav"
    return {
        "version": 1,
        "kind": "separation-prefix-asr-windows",
        "bindings": bindings,
        "stableStereoSamples": _stable_samples(request),
        "stereoRate": 44100,
        "stereo": pin(stereo),
        "mono": pin(mono),
        "windows": windows,
        "conversion": list(transcription.mono_command(stereo, mono, request.ffmpeg.path)),
        "conversionBoundary": "prefix-eof-candidate-requires-final-mono-equality",
        "chunkDomainVerified": True,
        "fullSourceCryptoVerified": False,
        "modelBytesVerified": False,
        "fullMonoProducerVerified": False,
        "recognitionExecuted": False,
        "finalMonoWindowEqual": False,
        "adopted": False,
        "qualityApproved": False,
        "deliveryComplete": False,
    }


def _estimate(request: Request) -> int:
    stable = _stable_samples(request)
    return stable * 4 + (stable * RATE // 44100 + 1) * 6 + 8 * 1024**2


def preflight(path: Path) -> dict[str, object]:
    """Check declared bounds and small pins; do not verify/reduce chunks or load models."""
    request = load_request(path)
    stable = _stable_samples(request)
    estimate = _estimate(request)
    require(stable * RATE // 44100 >= WINDOW, "No complete 60-second window is ready")
    require(estimate <= request.saved_limit, "Declared disk cap cannot hold prefix evidence")
    _bindings(request)
    return {
        "version": 1,
        "stableStereoSamples": stable,
        "plannedFullWindows": 1 + (stable * RATE // 44100 - WINDOW) // STRIDE,
        "estimatedAdditionalSavedBytes": estimate,
        "chunkDomainVerified": False,
        "modelBytesVerified": False,
        "recognitionExecuted": False,
        "adopted": False,
    }


def prepare(path: Path) -> Path:
    """Verify completed chunks and publish exclusive full-window evidence; no inference."""
    request = load_request(path)
    stable = _stable_samples(request)
    estimate = _estimate(request)
    require(stable * RATE // 44100 >= WINDOW, "No complete 60-second window is ready")
    require(estimate <= request.saved_limit, "Declared disk cap cannot hold prefix evidence")
    require_headroom((request.output.parent,), need=estimate)
    require(not request.output.exists(), "Occupied output: reconcile, do not repeat preparation")
    before = _bindings(request)
    request.output.mkdir(mode=0o700)
    with execution_lock(request.output, ".prefix") as acquired:
        require(acquired, "Prefix already owned")
        pipeline.write_pcm(request.output / "stereo.wav", samples=stable, blocks=_blocks(request))
        scratch = request.output / "tmp"
        scratch.mkdir(mode=0o700)
        environment = os.environ | {key: str(scratch) for key in ("TMPDIR", "TMP", "TEMP")}
        with (request.output / "conversion.log").open("xb") as log:
            subprocess.run(
                transcription.mono_command(
                    request.output / "stereo.wav", request.output / "mono.wav", request.ffmpeg.path
                ),
                stdin=subprocess.DEVNULL,
                stdout=log,
                stderr=log,
                check=True,
                timeout=request.timeout,
                cwd=ROOT,
                env=environment,
            )
        windows = _windows(request, create=True)
        require(_bindings(request) == before, "Prefix inputs changed during preparation")
        require(
            sum(p.stat().st_size for p in request.output.iterdir() if p.is_file())
            <= request.saved_limit,
            "Prefix saved-byte budget exceeded",
        )
        output = request.output / "receipt.json"
        durable(output, _result(request, before, windows))
    return output


def verify(path: Path) -> Path:
    """Read-only chunk/PCM/window verification; never rerun conversion or recognition."""
    request = load_request(path)
    with execution_lock(request.output, ".prefix") as acquired:
        require(acquired, "Prefix producer still owns its outputs")
        before = _bindings(request)
        with wave.open(str(request.output / "stereo.wav"), "rb") as audio:
            require(
                (
                    audio.getnchannels(),
                    audio.getsampwidth(),
                    audio.getframerate(),
                    audio.getnframes(),
                )
                == (2, 2, 44100, _stable_samples(request)),
                "Stereo prefix clock differs",
            )
            for block in _blocks(request):
                require(audio.readframes(len(block) // 4) == block, "Original prefix PCM differs")
            require(not audio.readframes(1), "Extra prefix PCM")
        output = request.output / "receipt.json"
        require(
            _read(output) == _result(request, before, _windows(request, create=False)),
            "Prefix receipt differs",
        )
        require(_bindings(request) == before, "Prefix inputs changed during verification")
    return output


def compare_final(path: Path, final_plan: Path, output: Path) -> Path:
    """Require a genuine final mono producer and byte-equal intervals; never adopt ASR."""
    prefix = verify(path)
    request = load_request(path)
    prefix_before = pin(prefix)
    prefix_inputs = _bindings(request)
    windows_before = _windows(request, create=False)
    final_plan, output = _local(final_plan), _local(output)
    before = pin(final_plan)
    transcription.verify(final_plan, "mono")
    final = transcription._load(final_plan)
    require(
        final.prepared.stereo == request.plan.source
        and final.prepared.vocals == request.plan.directory / "vocals.wav",
        "Final mono does not belong to the original separation",
    )
    mono = final.mono_root / "mono.wav"
    mono_before = pin(mono)
    comparisons: list[dict[str, object]] = []
    with wave.open(str(mono), "rb") as audio:
        for item in windows_before:
            window = _read(canonical(record(item["receipt"])["path"]))
            start = window["startSample"]
            require(type(start) is int, "Invalid window clock")
            if not isinstance(start, int):
                raise ReplayError("Invalid window clock")
            audio.setpos(start)
            payload = audio.readframes(WINDOW)
            require(
                len(payload) == WINDOW * 2
                and hashlib.sha256(payload).hexdigest() == record(window["pcm"])["sha256"],
                "Final mono window bytes differ",
            )
            comparisons.append({"window": item["receipt"], "finalMonoWindowEqual": True})
    require(pin(final_plan) == before and pin(mono) == mono_before, "Final mono inputs changed")
    require(
        pin(prefix) == prefix_before
        and _bindings(request) == prefix_inputs
        and _windows(request, create=False) == windows_before,
        "Prefix evidence changed during final comparison",
    )
    durable(
        output,
        {
            "version": 1,
            "kind": "final-mono-prefix-window-comparison",
            "prefix": prefix_before,
            "finalPlan": before,
            "finalMono": mono_before,
            "windows": comparisons,
            "finalMonoWindowEqual": True,
            "fullMonoProducerVerified": True,
            "recognitionExecuted": False,
            "adopted": False,
            "qualityApproved": False,
            "deliveryComplete": False,
        },
    )
    return output


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("phase", choices=("check", "prepare", "verify", "compare-final"))
    parser.add_argument("request", type=Path)
    parser.add_argument("--final-plan", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args(argv)
    try:
        if args.phase == "check":
            require(args.final_plan is None and args.output is None, "Unexpected final arguments")
            print(json.dumps(preflight(args.request)))
            return 0
        if args.phase == "compare-final":
            require(
                args.final_plan is not None and args.output is not None,
                "Final plan/output required",
            )
            result = compare_final(args.request, args.final_plan, args.output)
        else:
            require(args.final_plan is None and args.output is None, "Unexpected final arguments")
            result = {"prepare": prepare, "verify": verify}[args.phase](args.request)
    except (ReplayError, OSError, ValueError, subprocess.SubprocessError, wave.Error) as error:
        parser.exit(1, f"Prefix ASR evidence failed: {error}\n")
    print(json.dumps({"evidence": str(result), "recognitionExecuted": False, "adopted": False}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
