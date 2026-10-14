"""Compare bound CPU/MPS float vocals through the same genuine Metal ASR.

Float conversion preserves every stereo float32 sample. Continuous mono decoding
uses the ordinary command; each recognition is a sequential original collection.
Exact native text/token/timing differences are evidence, never quality approval.
"""

from __future__ import annotations

import argparse
import hashlib
import math
import os
import struct
import subprocess
import sys
import wave
from collections.abc import Callable
from pathlib import Path

import numpy as np
from numpy.typing import NDArray

from replay import isolated_metal_asr as metal
from replay import workflow
from replay.code_dependency_inventory import capture
from replay.core import ReplayError
from replay.local_asr import DEFAULT_CONFIG_SHA256, DEFAULT_WEIGHTS_SHA256
from replay.native_invocation import canonical, identity, pin, require
from replay.recordings import record
from replay.replay_transcription_plan import mono_command
from replay.sequence_journal import bounded_read, decode, encode, publish
from replay.stages import execution_lock

ROOT = Path(__file__).resolve().parents[2]
KIND = "separation-vocal-asr-compare"
RATE = 44100
MONO_RATE = 16000
MAX_SAMPLES = 60 * RATE
Executor = Callable[[tuple[str, ...], Path, int, dict[str, str]], None]


def _read(path: Path) -> dict[str, object]:
    return decode(bounded_read(path))


def _sha(value: object) -> str:
    require(
        isinstance(value, str) and len(value) == 64 and all(c in "0123456789abcdef" for c in value),
        "Invalid SHA256",
    )
    return str(value)


def _local(value: object) -> Path:
    path = canonical(value)
    require(path.is_relative_to(ROOT), "Input/output escapes checkout")
    return path


def _ref(value: object, *, tool: bool = False) -> dict[str, str]:
    raw = record(value)
    require(set(raw) == {"path", "sha256"}, "Exact reference required")
    path = canonical(raw["path"]) if tool else _local(raw["path"])
    identity(path)
    require(not tool or os.access(path, os.X_OK), "Executable required")
    return {"path": str(path), "sha256": _sha(raw["sha256"])}


def _check(ref: dict[str, str]) -> Path:
    path = canonical(ref["path"])
    require(pin(path)["sha256"] == ref["sha256"], "Declared input changed")
    return path


def _metadata(path: Path) -> dict[str, str]:
    return {"path": str(path), "sha256": hashlib.sha256(bounded_read(path)).hexdigest()}


def _code() -> dict[str, str]:
    return {str(item.path): item.sha256 for item in capture(("separation_vocal_asr_compare",))}


def _integer(value: object, high: int) -> int:
    require(type(value) is int and 1 <= value <= high, "Finite positive bound required")
    if not isinstance(value, int):
        raise ReplayError("Missing integer")
    return value


def _load(request: Path) -> tuple[dict[str, object], Path]:
    request = _local(str(request))
    raw = _read(request)
    require(
        set(raw)
        == {
            "version",
            "kind",
            "benchmark",
            "cpuSeparationWorker",
            "mpsSeparationWorker",
            "cpuVocals",
            "mpsVocals",
            "model",
            "metalCodeInputs",
            "ffmpeg",
            "outputDirectory",
            "timeoutSeconds",
            "monoTimeoutSeconds",
        }
        and type(raw["version"]) is int
        and raw["version"] == 1
        and raw["kind"] == KIND,
        "Invalid ASR comparison request",
    )
    for key in (
        "benchmark",
        "cpuSeparationWorker",
        "mpsSeparationWorker",
        "cpuVocals",
        "mpsVocals",
    ):
        _ref(raw[key])
    _ref(raw["ffmpeg"], tool=True)
    model = record(raw["model"])
    require(set(model) == {"config", "weights"}, "Exact original model required")
    config, weights = _ref(model["config"]), _ref(model["weights"])
    require(
        config["sha256"] == DEFAULT_CONFIG_SHA256
        and weights["sha256"] == DEFAULT_WEIGHTS_SHA256
        and Path(config["path"]).parent == Path(weights["path"]).parent
        and Path(config["path"]).is_relative_to(ROOT / "edits/models"),
        "Original nonTurbo large-v3 model required",
    )
    code = record(raw["metalCodeInputs"])
    require(
        1 <= len(code) <= 1024 and {str(path) for path in metal.required_code()} <= code.keys(),
        "Missing Metal producer/provider pins",
    )
    for name, sha in code.items():
        _local(name)
        _sha(sha)
    _integer(raw["timeoutSeconds"], 3600)
    _integer(raw["monoTimeoutSeconds"], 3600)
    root = _local(raw["outputDirectory"])
    require(root.is_relative_to(ROOT / "edits/work"), "Purpose scratch required")
    return raw, root


def _operation(request: Path) -> tuple[dict[str, object], Path]:
    raw, root = _load(request)
    require(
        _read(root / "operation.json")
        == {
            "version": 1,
            "kind": KIND,
            "request": _metadata(request),
            "codeInputs": _code(),
            "sourceEvidence": {side: _source(raw, side, array=False)[2] for side in ("cpu", "mps")},
        },
        "Request or reader code changed",
    )
    return raw, root


def _source(
    raw: dict[str, object], side: str, *, array: bool
) -> tuple[Path, int, dict[str, object]]:
    benchmark_ref = _ref(raw["benchmark"])
    benchmark = _read(_check(benchmark_ref))
    require(benchmark.get("kind") == "separation-device-benchmark", "Original benchmark required")
    count = _integer(benchmark.get("samples"), MAX_SAMPLES)
    start = benchmark.get("startSample")
    require(type(start) is int and start >= 0, "Invalid original interval start")
    digest = _sha(benchmark.get("intervalSHA256"))
    worker_ref = _ref(raw[side + "SeparationWorker"])
    worker_path = _check(worker_ref)
    worker = _read(worker_path)
    vocals_ref = _ref(raw[side + "Vocals"])
    vocals = canonical(vocals_ref["path"])
    require(
        worker.get("device") == side
        and worker.get("workers") == 1
        and worker.get("samples") == count
        and worker.get("intervalSHA256") == digest
        and worker.get("request") == pin(canonical(benchmark_ref["path"]))
        and record(worker.get("output")).get("path") == str(vocals)
        and record(worker["output"]).get("sha256") == vocals_ref["sha256"]
        and vocals.parent == worker_path.parent,
        "Different separation worker/interval/output",
    )
    resources = _read(worker_path.parent / "resources.json")
    require(
        resources.get("exitCode") == 0
        and resources.get("familyGone") is True
        and record(resources.get("cleanup")).get("terminalObserved") is True
        and isinstance(resources.get("samples"), list)
        and bool(resources["samples"]),
        "Uncertain separation outcome",
    )
    evidence: dict[str, object] = {
        "worker": pin(worker_path),
        "resources": pin(worker_path.parent / "resources.json"),
    }
    for phase in ("before-load", "after-load", "after-inference"):
        path = worker_path.parent / f"{phase}-descriptors.json"
        require(
            _read(path).get("externalWritableRegular") == [],
            "Observed external writable descriptor",
        )
        evidence[phase] = pin(path)
    log = worker_path.parent / "worker.log"
    require(
        not any(
            word in bounded_read(log).decode("utf-8", errors="replace").lower()
            for word in ("deny file-write", "sandbox: deny", "operation not permitted")
        ),
        "Separation sandbox failure",
    )
    evidence["workerLog"] = _metadata(log)
    if array:
        require(pin(vocals) == worker["output"], "Separation float output changed")
    return (
        vocals,
        count,
        {
            "benchmark": benchmark_ref,
            "intervalSHA256": digest,
            "startSample": start,
            "samples": count,
            "evidence": evidence,
        },
    )


def _array(path: Path, count: int) -> NDArray[np.float32]:
    before = identity(path)
    require(0 < before["bytes"] <= count * 8 + 4096, "Bounded float array required")
    with path.open("rb") as stream:
        value: object = np.load(stream, allow_pickle=False, max_header_size=4096)
    require(
        isinstance(value, np.ndarray)
        and value.dtype == np.dtype("float32")
        and value.shape == (2, count)
        and bool(np.isfinite(value).all())
        and identity(path) == before,
        "Invalid/changing stereo float array",
    )
    return np.asarray(value, dtype=np.float32)


def _wav_bytes(audio: NDArray[np.float32]) -> bytes:
    body = np.asarray(audio.T, dtype="<f4", order="C").tobytes()
    fmt = struct.pack("<HHIIHH", 3, 2, RATE, RATE * 8, 8, 32)
    return (
        b"RIFF"
        + struct.pack("<I", 48 + len(body))
        + b"WAVEfmt "
        + struct.pack("<I", 16)
        + fmt
        + b"fact"
        + struct.pack("<II", 4, audio.shape[1])
        + b"data"
        + struct.pack("<I", len(body))
        + body
    )


def _float_read(path: Path) -> bytes:
    before = identity(path)
    require(0 < before["bytes"] <= MAX_SAMPLES * 8 + 4096, "Float WAV byte budget exceeded")
    with path.open("rb") as stream:
        require(
            os.fstat(stream.fileno()).st_ino == before["inode"], "Float WAV replaced before read"
        )
        value = stream.read(MAX_SAMPLES * 8 + 4097)
    require(
        len(value) == before["bytes"] and identity(path) == before, "Float WAV changed during read"
    )
    return value


def _mono(path: Path) -> int:
    before = identity(path)
    with wave.open(str(path), "rb") as audio:
        count = audio.getnframes()
        require(
            (audio.getframerate(), audio.getnchannels(), audio.getsampwidth(), audio.getcomptype())
            == (MONO_RATE, 1, 2, "NONE")
            and 0 < count <= 60 * MONO_RATE
            and len(audio.readframes(count)) == count * 2,
            "Invalid continuous mono PCM",
        )
    require(identity(path) == before, "Mono changed during clock check")
    return count


def _normalization(request: Path, side: str) -> dict[str, object]:
    raw, root = _operation(request)
    array, count, provenance = _source(raw, side, array=True)
    float_wav, mono = root / side / "float.wav", root / side / "mono.wav"
    require(
        _float_read(float_wav) == _wav_bytes(_array(array, count)),
        "Float WAV is not exact array samples",
    )
    samples = _mono(mono)
    require(
        abs(samples * RATE - count * MONO_RATE) <= RATE, "Continuous mono sample duration differs"
    )
    ffmpeg = _ref(raw["ffmpeg"], tool=True)
    _check(ffmpeg)
    return {
        "version": 1,
        "kind": KIND,
        "side": side,
        "request": _metadata(request),
        "separation": provenance,
        "array": pin(array),
        "floatWav": pin(float_wav),
        "floatSamples": count,
        "floatSampleRate": RATE,
        "mono": pin(mono),
        "monoSamples": samples,
        "monoSampleRate": MONO_RATE,
        "command": list(mono_command(float_wav, mono, canonical(ffmpeg["path"]))),
        "ffmpeg": ffmpeg,
        "floatSamplesExact": True,
        "trimPadStretch": False,
        "qualityApproved": False,
    }


def _verify_normalization(request: Path, side: str) -> None:
    _, root = _operation(request)
    require(
        _read(root / side / "normalization.json") == _normalization(request, side),
        "Normalization receipt changed",
    )


def _request(request: Path, side: str) -> dict[str, object]:
    raw, root = _operation(request)
    _verify_normalization(request, side)
    model = record(raw["model"])
    return {
        "version": 1,
        "device": "metal",
        "fp16": True,
        "cpuAdoption": None,
        "inputs": {
            "pcm": pin(root / side / "mono.wav"),
            "config": _ref(model["config"]),
            "weights": _ref(model["weights"]),
        },
        "samples": _mono(root / side / "mono.wav"),
        "timeoutSeconds": raw["timeoutSeconds"],
        "codeInputs": raw["metalCodeInputs"],
    }


def _verify_request(request: Path, side: str) -> None:
    _, root = _operation(request)
    require(
        _read(root / side / "metal-request.json") == _request(request, side), "ASR request changed"
    )
    metal.load_request(root / side / "metal-request.json", verify=False)


def _recognition(request: Path, side: str) -> dict[str, object]:
    _, root = _operation(request)
    _verify_request(request, side)
    complete = metal.collect(
        root / side / "metal-request.json", root / side / "asr", verify_only=True
    )
    return {
        "version": 1,
        "kind": KIND,
        "side": side,
        "request": _metadata(root / side / "metal-request.json"),
        "complete": pin(complete),
        "raw": pin(root / side / "asr/000000000000.raw.json"),
        "fullNativeCollectionVerified": True,
        "qualityApproved": False,
    }


def _verify_recognition(request: Path, side: str) -> None:
    _, root = _operation(request)
    require(
        _read(root / side / "recognition.json") == _recognition(request, side),
        "Native recognition receipt changed",
    )


def _number(value: object) -> float:
    require(
        type(value) in (int, float) and math.isfinite(float(str(value))) and float(str(value)) >= 0,
        "Invalid native timestamp",
    )
    return float(str(value))


def _native(
    raw: dict[str, object],
) -> tuple[
    str, list[str], list[list[int]], list[str], list[tuple[float, float]], list[tuple[float, float]]
]:
    text, segments = raw.get("text"), raw.get("segments")
    require(
        isinstance(text, str) and isinstance(segments, list) and len(segments) <= 10000,
        "Missing bounded native text/segments",
    )
    if not isinstance(text, str) or not isinstance(segments, list):
        raise ReplayError("Missing native transcript")
    segment_texts: list[str] = []
    tokens: list[list[int]] = []
    words: list[str] = []
    segment_times: list[tuple[float, float]] = []
    word_times: list[tuple[float, float]] = []
    for item in segments:
        segment = record(item)
        require(isinstance(segment.get("text"), str), "Missing segment text")
        segment_texts.append(str(segment["text"]))
        start, end = _number(segment.get("start")), _number(segment.get("end"))
        require(start <= end, "Reversed native segment")
        segment_times.append((start, end))
        token_values = segment.get("tokens")
        require(
            isinstance(token_values, list)
            and len(token_values) <= 50000
            and all(type(token) is int and token >= 0 for token in token_values),
            "Missing native tokens",
        )
        if not isinstance(token_values, list):
            raise ReplayError("Missing token list")
        tokens.append([int(str(token)) for token in token_values])
        values = segment.get("words")
        require(isinstance(values, list) and len(values) <= 50000, "Missing native words")
        if not isinstance(values, list):
            raise ReplayError("Missing word list")
        for value in values:
            word = record(value)
            require(isinstance(word.get("word"), str), "Missing native word text")
            words.append(str(word["word"]))
            left, right = _number(word.get("start")), _number(word.get("end"))
            require(left <= right, "Reversed native word")
            word_times.append((left, right))
    require(len(words) <= 50000, "Native word budget exceeded")
    return text, segment_texts, tokens, words, segment_times, word_times


def _deltas(cpu: list[tuple[float, float]], mps: list[tuple[float, float]]) -> dict[str, object]:
    starts = [right[0] - left[0] for left, right in zip(cpu, mps, strict=False)]
    ends = [right[1] - left[1] for left, right in zip(cpu, mps, strict=False)]
    return {
        "cpuCount": len(cpu),
        "mpsCount": len(mps),
        "comparableCount": len(starts),
        "startSeconds": starts,
        "endSeconds": ends,
        "maxAbsoluteSeconds": max((abs(value) for value in starts + ends), default=0.0),
        "exact": cpu == mps,
    }


def _comparison(request: Path) -> dict[str, object]:
    raw, root = _operation(request)
    for side in ("cpu", "mps"):
        _verify_recognition(request, side)
    requests = [_read(root / side / "metal-request.json") for side in ("cpu", "mps")]
    require(
        requests[0] | {"inputs": record(requests[1]["inputs"])} == requests[1]
        and record(requests[0]["inputs"])["config"] == record(requests[1]["inputs"])["config"]
        and record(requests[0]["inputs"])["weights"] == record(requests[1]["inputs"])["weights"],
        "Different recognition model/provider/clock context",
    )
    transcripts = [
        _native(_read(root / side / "asr/000000000000.raw.json")) for side in ("cpu", "mps")
    ]
    cpu, mps = transcripts
    words = _deltas(cpu[5], mps[5])
    return {
        "version": 1,
        "kind": KIND,
        "request": _metadata(request),
        "benchmark": _ref(raw["benchmark"]),
        "cpuSeparationWorker": pin(canonical(_ref(raw["cpuSeparationWorker"])["path"])),
        "mpsSeparationWorker": pin(canonical(_ref(raw["mpsSeparationWorker"])["path"])),
        "cpuVocals": pin(canonical(_ref(raw["cpuVocals"])["path"])),
        "mpsVocals": pin(canonical(_ref(raw["mpsVocals"])["path"])),
        "normalizations": {
            side: pin(root / side / "normalization.json") for side in ("cpu", "mps")
        },
        "recognitions": {side: pin(root / side / "recognition.json") for side in ("cpu", "mps")},
        "model": raw["model"],
        "metalCodeInputs": raw["metalCodeInputs"],
        "rawTextEqual": cpu[0] == mps[0],
        "orderedSegmentTextEqual": cpu[1] == mps[1],
        "orderedWordTextEqual": cpu[3] == mps[3],
        "tokensEqual": cpu[2] == mps[2],
        "segmentTimingExact": cpu[4] == mps[4],
        "wordTimingExact": cpu[5] == mps[5],
        "segmentTimingDeltas": _deltas(cpu[4], mps[4]),
        "wordTimingDeltas": words,
        "recognizedWordCounts": {"cpu": len(cpu[3]), "mps": len(mps[3])},
        "allNativeCollectionsVerified": True,
        "sameModelProviderContext": True,
        "captionFrameEffects": None,
        "captionFrameEffectsVerified": False,
        "qualityApproved": False,
        "deliveryComplete": False,
    }


def verify(request: Path, receipt: Path) -> dict[str, object]:
    """Freshly verify source normalization, original Metal domains and comparison."""
    request = _local(str(request))
    _, root = _operation(request)
    require(
        _local(str(receipt)) == root / "comparison.json", "Receipt belongs to another comparison"
    )
    result = _comparison(request)
    saved = _read(receipt)
    require(saved == result, "Comparison receipt changed")
    return saved


def _execute(
    command: tuple[str, ...], log: Path, timeout: int, environment: dict[str, str]
) -> None:
    with log.open("xb") as stream:
        os.fchmod(stream.fileno(), 0o600)
        subprocess.run(
            command,
            stdin=subprocess.DEVNULL,
            stdout=stream,
            stderr=subprocess.STDOUT,
            timeout=timeout,
            check=True,
            cwd=ROOT,
            env=environment,
        )


def run(
    request: Path, phase: str, *, execute: Executor = _execute, runtime: metal.Runtime | None = None
) -> Path:
    """Run an exclusive normalization, genuine collection or read-only phase."""
    request = _local(str(request))
    raw, root = _operation(request)
    if phase in {"compare", "verify-compare"}:
        path = root / "comparison.json"
        if phase == "verify-compare":
            verify(request, path)
        else:
            publish(path, encode(_comparison(request)))
        return path
    operation, separator, side = phase.rpartition("-")
    require(separator == "-" and side in {"cpu", "mps"}, "Unknown phase side")
    directory = root / side
    if operation == "verify-normalize":
        _verify_request(request, side)
        return directory / "normalization.json"
    if operation == "verify-recognize":
        _verify_recognition(request, side)
        return directory / "recognition.json"
    if operation == "normalize":
        with execution_lock(directory, ".normalize") as acquired:
            require(acquired, "Normalization already owned")
            intent = directory / "normalization-intent.json"
            require(not intent.exists(), "Existing uncertain normalization: inspect; no retry")
            array, count, _ = _source(raw, side, array=True)
            publish(
                intent, encode({"request": _metadata(request), "array": pin(array), "side": side})
            )
            publish(directory / "float.wav", _wav_bytes(_array(array, count)))
            environment = os.environ.copy()
            environment.update(
                {
                    "TMPDIR": str(root / "tmp"),
                    "TMP": str(root / "tmp"),
                    "TEMP": str(root / "tmp"),
                    "XDG_CACHE_HOME": str(root / "cache"),
                    "PYTHONDONTWRITEBYTECODE": "1",
                }
            )
            decoder = canonical(_ref(raw["ffmpeg"], tool=True)["path"])
            _check(_ref(raw["ffmpeg"], tool=True))
            execute(
                mono_command(directory / "float.wav", directory / "mono.wav", decoder),
                directory / "mono.log",
                _integer(raw["monoTimeoutSeconds"], 3600),
                environment,
            )
            publish(directory / "normalization.json", encode(_normalization(request, side)))
            publish(directory / "metal-request.json", encode(_request(request, side)))
            return directory / "normalization.json"
    require(operation == "recognize", "Unknown comparison phase")
    _verify_request(request, side)
    metal.collect(directory / "metal-request.json", directory / "asr", runtime=runtime)
    path = directory / "recognition.json"
    publish(path, encode(_recognition(request, side)))
    return path


def prepare(request: Path) -> Path:
    """Metadata-only ordinary graph with two sequential recognition producers."""
    request = _local(str(request))
    raw, root = _load(request)
    require(root.is_dir() and not tuple(root.iterdir()), "Empty purpose root required")
    source_evidence = {side: _source(raw, side, array=False)[2] for side in ("cpu", "mps")}
    code = _code()
    publish(
        root / "operation.json",
        encode(
            {
                "version": 1,
                "kind": KIND,
                "request": _metadata(request),
                "codeInputs": code,
                "sourceEvidence": source_evidence,
            }
        ),
    )
    for name in ("cpu", "mps", "tmp", "cache"):
        (root / name).mkdir(mode=0o700)
    common = list(
        dict.fromkeys(
            [
                str(request),
                str(root / "operation.json"),
                *code,
                *record(raw["metalCodeInputs"]),
                *(
                    _ref(raw[key])["path"]
                    for key in (
                        "benchmark",
                        "cpuSeparationWorker",
                        "mpsSeparationWorker",
                        "cpuVocals",
                        "mpsVocals",
                    )
                ),
                _ref(raw["ffmpeg"], tool=True)["path"],
            ]
        )
    )
    for side in ("cpu", "mps"):
        worker = Path(_ref(raw[side + "SeparationWorker"])["path"])
        common += [
            str(worker.parent / name)
            for name in (
                "resources.json",
                "worker.log",
                "before-load-descriptors.json",
                "after-load-descriptors.json",
                "after-inference-descriptors.json",
            )
        ]
    # Empty worker logs are valid evidence, but the stage fingerprint requires
    # nonempty inputs. Their fresh digest/content is bound by normalization.
    common = [name for name in dict.fromkeys(common) if Path(name).stat().st_size]
    stages: list[dict[str, object]] = []
    for side in ("cpu", "mps"):
        directory = root / side
        normalization, recognition, metal_request = (
            str(directory / name)
            for name in ("normalization.json", "recognition.json", "metal-request.json")
        )
        schedule = [str(root / "cpu/recognition.json")] if side == "mps" else []
        for action, inputs, outputs in (
            (
                "normalize",
                common + schedule,
                [
                    str(directory / "float.wav"),
                    str(directory / "mono.wav"),
                    normalization,
                    metal_request,
                ],
            ),
            (
                "recognize",
                common
                + schedule
                + [
                    normalization,
                    metal_request,
                    str(directory / "mono.wav"),
                    *(_ref(value)["path"] for value in record(raw["model"]).values()),
                ],
                [str(directory / "asr/complete.json"), recognition],
            ),
        ):
            phase = action + "-" + side
            stages.append(
                {
                    "name": phase,
                    "command": [
                        sys.executable,
                        "-B",
                        "-m",
                        "replay.separation_vocal_asr_compare",
                        phase,
                        str(request),
                    ],
                    "inputs": inputs,
                    "outputs": outputs,
                    "timeout_seconds": 3600,
                    "verify_command": [
                        sys.executable,
                        "-B",
                        "-m",
                        "replay.separation_vocal_asr_compare",
                        "verify-" + phase,
                        str(request),
                    ],
                }
            )
    stages.append(
        {
            "name": "compare",
            "command": [
                sys.executable,
                "-B",
                "-m",
                "replay.separation_vocal_asr_compare",
                "compare",
                str(request),
            ],
            "inputs": common
            + [
                str(root / side / name)
                for side in ("cpu", "mps")
                for name in ("normalization.json", "recognition.json", "asr/complete.json")
            ],
            "outputs": [str(root / "comparison.json")],
            "timeout_seconds": 3600,
            "verify_command": [
                sys.executable,
                "-B",
                "-m",
                "replay.separation_vocal_asr_compare",
                "verify-compare",
                str(request),
            ],
        }
    )
    graph = {"version": 1, "stages": stages}
    workflow.stage_graph(workflow.parse_stages(graph))
    path = root / "workflow.json"
    publish(path, encode(graph))
    return path


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "phase",
        choices=(
            "prepare",
            "normalize-cpu",
            "normalize-mps",
            "recognize-cpu",
            "recognize-mps",
            "compare",
            "verify-normalize-cpu",
            "verify-normalize-mps",
            "verify-recognize-cpu",
            "verify-recognize-mps",
            "verify-compare",
        ),
    )
    parser.add_argument("request", type=Path)
    args = parser.parse_args(argv)
    print(prepare(args.request) if args.phase == "prepare" else run(args.request, args.phase))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
