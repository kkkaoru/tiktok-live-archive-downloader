"""Import equal prefix windows into a genuinely bootstrapped CPU ASR collection.

The public isolated producer must first create inputs and policy with limit=1.
This importer never creates or rewrites those files, never overwrites a raw
window, and never creates full completion. Resume with a new normal policy and
limit=None; the original producer then verifies or recognizes every window.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import wave
from dataclasses import dataclass
from pathlib import Path

from replay import isolated_cpu_asr, prefix_window_recognition
from replay.audio_windows import plan_audio_windows
from replay.code_dependency_inventory import capture
from replay.core import ReplayError
from replay.local_asr import DEFAULT_CONFIG_SHA256, DEFAULT_WEIGHTS_SHA256
from replay.native_invocation import canonical, durable, identity, pin, read, require
from replay.recordings import record
from replay.stages import execution_lock

ROOT = Path(__file__).resolve().parents[2]
RATE = 16000
WINDOW = RATE * 60
STRIDE = RATE * 30
MAX_JSON = 8 * 1024**2
MAX_CANDIDATES = 64
MAX_SECONDS = 21600


@dataclass(frozen=True)
class Ref:
    path: Path
    sha256: str

    def data(self) -> dict[str, str]:
        return {"path": str(self.path), "sha256": self.sha256}


@dataclass(frozen=True)
class Candidate:
    request: Ref
    complete: Ref
    status: Ref


@dataclass(frozen=True)
class Request:
    path: Path
    definition: dict[str, object]
    candidates: tuple[Candidate, ...]
    comparison: Ref
    manifest: Ref
    started: Ref
    finished: Ref
    bootstrap_status: Ref
    output: Path


def _local(value: object) -> Path:
    path = canonical(value)
    require(path.is_relative_to(ROOT / "edits/work"), "Expected checkout work path")
    return path


def _ref(value: object) -> Ref:
    raw = record(value)
    sha = raw.get("sha256")
    require(set(raw) == {"path", "sha256"}, "Exact reference fields required")
    require(isinstance(sha, str) and re.fullmatch("[0-9a-f]{64}", sha) is not None, "Invalid SHA")
    return Ref(_local(raw["path"]), str(sha))


def _check(ref: Ref) -> dict[str, object]:
    actual = pin(ref.path)
    require(actual["sha256"] == ref.sha256, "Pinned adoption input changed")
    return actual


def _metadata(path: Path) -> dict[str, object]:
    require(identity(path)["bytes"] <= MAX_JSON, "Adoption metadata exceeds byte budget")
    return read(path)


def runtime_inputs() -> dict[str, str]:
    """Pin current shared readers without replacing historical candidate authority."""
    return {str(item.path): item.sha256 for item in capture(("prefix_window_adoption",))}


def load_request(path: Path) -> Request:
    """Require explicit completed jobs and a genuine existing bootstrap namespace."""
    path = _local(str(path))
    raw = _metadata(path)
    require(
        set(raw) == {"version", "candidates", "comparison", "bootstrap", "output", "runtime"}
        and type(raw["version"]) is int
        and raw["version"] == 1,
        "Unsupported prefix adoption request",
    )
    rows = raw["candidates"]
    if not isinstance(rows, list) or not 1 <= len(rows) <= MAX_CANDIDATES:
        raise ReplayError("Expected bounded complete candidates")
    candidates: list[Candidate] = []
    for value in rows:
        item = record(value)
        require(
            set(item) == {"request", "complete", "jobStatus"}, "Exact candidate fields required"
        )
        candidates.append(
            Candidate(_ref(item["request"]), _ref(item["complete"]), _ref(item["jobStatus"]))
        )
    bootstrap = record(raw["bootstrap"])
    require(
        set(bootstrap) == {"manifest", "started", "finished", "jobStatus"},
        "Exact bootstrap fields required",
    )
    manifest, started, finished, status = (
        _ref(bootstrap[key]) for key in ("manifest", "started", "finished", "jobStatus")
    )
    require(
        manifest.path.name == "inputs.json"
        and started.path.name == "started.json"
        and finished.path == started.path.parent / "result.json"
        and not started.path.is_relative_to(manifest.path.parent),
        "Independent genuine bootstrap policy required",
    )
    output = _local(raw["output"])
    protected = (
        path,
        manifest.path.parent,
        started.path.parent,
        *(item.complete.path.parent for item in candidates),
    )
    require(
        output.parent.is_dir()
        and not any(
            output == p or output.is_relative_to(p) or p.is_relative_to(output) for p in protected
        ),
        "Independent adoption receipt required",
    )
    require(raw["runtime"] == runtime_inputs(), "Adoption reader code changed")
    return Request(
        path,
        raw,
        tuple(candidates),
        _ref(raw["comparison"]),
        manifest,
        started,
        finished,
        status,
        output,
    )


def _successful(ref: Ref) -> dict[str, object]:
    require(identity(ref.path)["bytes"] <= 8, "Oversized job status")
    actual = _check(ref)
    require(ref.path.read_bytes() == b"0\n", "Failed or uncertain job cannot supply adoption")
    require(pin(ref.path) == actual, "Job status changed")
    return actual


def _bootstrap(request: Request, comparison: dict[str, object]) -> dict[str, object]:
    manifest, started, finished = (
        _metadata(ref.path) for ref in (request.manifest, request.started, request.finished)
    )
    source = _local(record(comparison.get("finalMono")).get("path"))
    require(pin(source) == comparison.get("finalMono"), "Genuine final mono changed")
    with wave.open(str(source), "rb") as audio:
        require(
            (audio.getnchannels(), audio.getsampwidth(), audio.getframerate(), audio.getcomptype())
            == (1, 2, RATE, "NONE"),
            "Expected genuine final mono PCM16/16k",
        )
        samples = audio.getnframes()
    require(0 < samples <= RATE * MAX_SECONDS, "Final mono exceeds normal window budget")
    windows = plan_audio_windows(samples, window_samples=WINDOW, stride_samples=STRIDE)
    inputs = record(manifest.get("inputs"))
    require(
        len(inputs) == 5
        and str(Path(isolated_cpu_asr.__file__).with_name("batch_asr.py").resolve()) in inputs
        and str(Path(isolated_cpu_asr.__file__).with_name("asr_runtime.py").resolve()) in inputs,
        "Genuine normal producer input inventory required",
    )
    producer = pin(Path(isolated_cpu_asr.__file__).resolve())["sha256"]
    require(
        manifest
        == {
            "version": 1,
            "inputs": inputs,
            "samples": samples,
            "sampleRate": RATE,
            "windowSamples": WINDOW,
            "strideSamples": STRIDE,
            "clock": "input-pcm-not-projected",
            "device": "cpu",
            "dtypeContract": "provider-default-fp16-unchanged",
        }
        and inputs.get(str(source)) == record(comparison["finalMono"])["sha256"],
        "Bootstrap input/clock definition is not the genuine final mono",
    )
    require(
        set(started)
        == {
            "version",
            "producerSHA256",
            "inputs",
            "manifestSHA256",
            "processPolicy",
            "cachePolicy",
            "limitNewWindows",
            "childTimeoutSeconds",
        }
        and started.get("version") == 1
        and started.get("producerSHA256") == producer
        and started.get("inputs") == inputs
        and started.get("manifestSHA256") == request.manifest.sha256
        and started.get("processPolicy") == "fresh-cpu-child-per-missing-window"
        and started.get("cachePolicy") == "unchanged-provider-default"
        and type(started.get("limitNewWindows")) is int
        and started.get("limitNewWindows") == 1,
        "Normal limit=1 bootstrap policy required",
    )
    timeout = started.get("childTimeoutSeconds")
    require(type(timeout) is int and 1 <= int(str(timeout)) <= 3600, "Invalid normal child timeout")
    require(
        finished
        == {
            "version": 1,
            "producerSHA256": producer,
            "newWindows": 1,
            "verifiedWindows": 1,
            "totalWindows": len(windows),
            "recognitionComplete": len(windows) == 1,
            "deliveryComplete": False,
        },
        "Bootstrap did not successfully recognize exactly its first window",
    )
    for path, sha in inputs.items():
        actual = canonical(path)
        require(
            actual.is_relative_to(ROOT) and pin(actual)["sha256"] == sha,
            "Bootstrap source/model/code changed",
        )
    return {
        "manifest": _check(request.manifest),
        "started": _check(request.started),
        "finished": _check(request.finished),
        "status": _successful(request.bootstrap_status),
        "source": source,
        "inputs": inputs,
        "samples": samples,
    }


def _candidate(
    candidate: Candidate, comparison: dict[str, object], normal: dict[str, object]
) -> dict[str, object]:
    _successful(candidate.status)
    _check(candidate.request)
    _check(candidate.complete)
    completed = prefix_window_recognition.verify(candidate.request.path)
    require(completed == candidate.complete.path, "Candidate completion belongs to another request")
    context = prefix_window_recognition.load_request(candidate.request.path)
    evidence = _metadata(completed)
    windows = comparison.get("windows")
    if not isinstance(windows, list):
        raise ReplayError("Missing final window comparisons")
    require(
        comparison.get("prefix") == pin(context.prefix_receipt.path)
        and any(
            record(row).get("window") == pin(context.window.path)
            and record(row).get("finalMonoWindowEqual") is True
            for row in windows
        ),
        "Candidate interval has no genuine equal final comparison",
    )
    inputs = record(normal["inputs"])
    require(
        inputs.get(str(context.config.path)) == DEFAULT_CONFIG_SHA256
        and inputs.get(str(context.weights.path)) == DEFAULT_WEIGHTS_SHA256
        and evidence.get("device") == "cpu"
        and evidence.get("dtypeContract") == "provider-default-fp16-unchanged",
        "Candidate model/CPU contract differs from normal producer",
    )
    source = normal["source"]
    if not isinstance(source, Path):
        raise ReplayError("Missing genuine mono source")
    with wave.open(str(source), "rb") as audio:
        audio.setpos(context.start)
        payload = audio.readframes(WINDOW)
    require(
        len(payload) == WINDOW * 2 and hashlib.sha256(payload).hexdigest() == context.pcm.sha256,
        "Candidate final PCM interval differs",
    )
    return {
        "candidate": pin(completed),
        "request": pin(candidate.request.path),
        "jobStatus": _check(candidate.status),
        "startSample": context.start,
        "raw": context.output / "raw.json",
        "clock": context.output / "clock.json",
        "historicalRuntime": record(evidence.get("bindings")).get("runtime"),
        "model": {"config": context.config.data(), "weights": context.weights.data()},
    }


def _copy(source: Path, target: Path) -> dict[str, object]:
    before = pin(source)
    require(identity(source)["bytes"] <= MAX_JSON, "Candidate window exceeds byte budget")
    with source.open("rb") as origin, target.open("xb") as destination:
        os.fchmod(destination.fileno(), 0o600)
        destination.write(origin.read(MAX_JSON + 1))
        destination.flush()
        os.fsync(destination.fileno())
    after = pin(target)
    require(pin(source) == before and after["sha256"] == before["sha256"], "Candidate copy changed")
    return after


def _comparison(request: Request) -> tuple[dict[str, object], dict[str, object]]:
    before = _check(request.comparison)
    value = _metadata(request.comparison.path)
    require(
        value.get("version") == 1
        and value.get("kind") == "final-mono-prefix-window-comparison"
        and value.get("finalMonoWindowEqual") is True
        and value.get("fullMonoProducerVerified") is True,
        "Completed compare_final evidence required",
    )
    plan = _local(record(value.get("finalPlan")).get("path"))
    require(pin(plan) == value.get("finalPlan"), "Final mono plan changed")
    return before, value


def adopt(path: Path) -> Path:
    """Import into a stopped successful bootstrap; never create normal policy/completion."""
    request = load_request(path)
    require(not request.output.exists(), "Occupied adoption receipt: verify, do not repeat")
    collection = request.manifest.path.parent
    with execution_lock(collection, ".run") as acquired:
        require(acquired, "Normal ASR collection still owned")
        comparison_pin, comparison = _comparison(request)
        normal = _bootstrap(request, comparison)
        candidates = [_candidate(item, comparison, normal) for item in request.candidates]
        starts = [item["startSample"] for item in candidates]
        require(len(set(starts)) == len(starts), "Duplicate candidate window")
        require(
            not (collection / "complete.json").exists() or starts == [0],
            "Completed collection is immutable",
        )
        pending = [start for start in starts if start != 0]
        require(
            not any(
                target.exists() or target.is_symlink()
                for start in pending
                for target in (
                    collection / f"{start:012d}.raw.json",
                    collection / f"{start:012d}.clock.json",
                    collection / f"{start:012d}.cpu-runtime",
                )
            ),
            "Candidate target has an existing/uncertain attempt",
        )
        adopted: list[dict[str, object]] = []
        for item in candidates:
            start, raw, clock = item["startSample"], item["raw"], item["clock"]
            if (
                not isinstance(start, int)
                or not isinstance(raw, Path)
                or not isinstance(clock, Path)
            ):
                raise ReplayError("Invalid candidate window binding")
            target_raw = collection / f"{start:012d}.raw.json"
            target_clock = collection / f"{start:012d}.clock.json"
            if start == 0:
                expected = {
                    "startSample": 0,
                    "endSample": min(WINDOW, int(str(normal["samples"]))),
                    "sampleRate": RATE,
                    "humanReviewed": False,
                    "emptyRecognitionIsNotSilence": True,
                    "rawSHA256": pin(target_raw)["sha256"],
                }
                require(
                    _metadata(target_clock) == expected, "Genuine first bootstrap window changed"
                )
                adopted.append(
                    {
                        "candidate": item["candidate"],
                        "startSample": 0,
                        "adopted": False,
                        "duplicateFirstWindow": True,
                        "candidateRaw": pin(raw),
                        "bootstrapRaw": pin(target_raw),
                    }
                )
                continue
            adopted.append(
                {
                    "candidate": item["candidate"],
                    "startSample": start,
                    "adopted": True,
                    "raw": _copy(raw, target_raw),
                    "clock": _copy(clock, target_clock),
                    "historicalRuntime": item["historicalRuntime"],
                    "model": item["model"],
                }
            )
        require(_check(request.comparison) == comparison_pin, "Comparison changed during import")
        require(_bootstrap(request, comparison) == normal, "Normal bootstrap inputs changed")
        durable(
            request.output,
            {
                "version": 1,
                "kind": "imported-prefix-asr-windows",
                "request": pin(request.path),
                "comparison": comparison_pin,
                "normal": {key: value for key, value in normal.items() if key != "source"},
                "windows": adopted,
                "requiredResume": "replay.isolated_cpu_asr.run",
                "requiredLimitNewWindows": None,
                "normalFactoryConnected": False,
                "fullASRCompletionCreated": False,
                "qualityApproved": False,
                "deliveryComplete": False,
            },
        )
        return request.output


def verify(path: Path) -> Path:
    """Check immutable imported windows without reimporting or creating completion."""
    request = load_request(path)
    with execution_lock(request.manifest.path.parent, ".run") as acquired:
        require(acquired, "Normal ASR collection still owned")
        comparison_pin, comparison = _comparison(request)
        normal = _bootstrap(request, comparison)
        receipt = _metadata(request.output)
        require(
            receipt.get("request") == pin(request.path)
            and receipt.get("comparison") == comparison_pin
            and receipt.get("normal") == {k: v for k, v in normal.items() if k != "source"}
            and receipt.get("kind") == "imported-prefix-asr-windows"
            and receipt.get("version") == 1
            and receipt.get("requiredResume") == "replay.isolated_cpu_asr.run"
            and receipt.get("requiredLimitNewWindows") is None
            and receipt.get("normalFactoryConnected") is False
            and receipt.get("fullASRCompletionCreated") is False
            and receipt.get("qualityApproved") is False
            and receipt.get("deliveryComplete") is False,
            "Adoption receipt or input bindings changed",
        )
        rows = receipt.get("windows")
        require(
            isinstance(rows, list) and len(rows) == len(request.candidates),
            "Missing imported windows",
        )
        if not isinstance(rows, list):
            raise ReplayError("Missing imported windows")
        for candidate, value in zip(request.candidates, rows, strict=True):
            item, row = _candidate(candidate, comparison, normal), record(value)
            start = item["startSample"]
            target_raw = request.manifest.path.parent / f"{start:012d}.raw.json"
            target_clock = request.manifest.path.parent / f"{start:012d}.clock.json"
            require(
                row.get("candidate") == item["candidate"] and row.get("startSample") == start,
                "Imported candidate binding changed",
            )
            if start == 0:
                require(
                    row.get("adopted") is False
                    and row.get("duplicateFirstWindow") is True
                    and row.get("candidateRaw") == pin(Path(str(item["raw"])))
                    and row.get("bootstrapRaw") == pin(target_raw),
                    "Bootstrap first window changed",
                )
            else:
                require(
                    row.get("adopted") is True
                    and row.get("raw") == pin(target_raw)
                    and row.get("clock") == pin(target_clock)
                    and pin(target_raw)["sha256"] == pin(Path(str(item["raw"])))["sha256"]
                    and _metadata(target_clock) == _metadata(Path(str(item["clock"])))
                    and row.get("historicalRuntime") == item["historicalRuntime"]
                    and row.get("model") == item["model"],
                    "Imported window changed",
                )
            require(
                _metadata(target_clock)
                == {
                    "startSample": start,
                    "endSample": min(int(str(start)) + WINDOW, int(str(normal["samples"]))),
                    "sampleRate": RATE,
                    "humanReviewed": False,
                    "emptyRecognitionIsNotSilence": True,
                    "rawSHA256": pin(target_raw)["sha256"],
                },
                "Imported clock changed",
            )
        require(
            _comparison(request)[0] == comparison_pin and _bootstrap(request, comparison) == normal,
            "Inputs changed during verification",
        )
    return request.output


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("phase", choices=("import", "verify"))
    parser.add_argument("request", type=Path)
    args = parser.parse_args(argv)
    try:
        result = {"import": adopt, "verify": verify}[args.phase](args.request)
    except (ReplayError, OSError, ValueError, wave.Error) as error:
        parser.exit(1, f"Prefix adoption failed: {error}\n")
    print(json.dumps({"evidence": str(result), "fullASRCompletionCreated": False}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
