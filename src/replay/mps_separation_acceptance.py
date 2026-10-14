"""Separate user-defined MPS separation gate; never replaces CPU exact equivalence."""

from __future__ import annotations

import argparse
import importlib
import math
from collections.abc import Sequence
from pathlib import Path

import numpy as np

from replay import separation_device_benchmark as benchmark
from replay.native_invocation import canonical, durable, identity, pin, read, require, rows
from replay.recordings import record
from replay.separation_forward_probe import compare_audio, read_interval
from replay.separation_window_budget import load_array

VERSION = "mps-separation-sdr40-corr99999-v1"


def _characteristic(path: Path, start: int, count: int) -> dict[str, object]:
    interval = read_interval(path, start_sample=start, samples=count)
    audio = interval.audio.astype(np.float64)
    energy = float(np.mean(audio * audio))
    stereo = float(np.mean((audio[0] - audio[1]) ** 2))
    blocks = [float(np.mean(audio[:, left : left + 44100] ** 2)) for left in range(0, count, 44100)]
    return {
        "startSample": start,
        "sampleCount": count,
        "intervalSHA256": interval.sha256,
        "rms": math.sqrt(energy),
        "stereoDifferenceRMS": math.sqrt(stereo),
        "oneSecondEnergyRange": max(blocks) - min(blocks),
    }


def reference(value: object) -> Path:
    item = record(value)
    path = canonical(item.get("path"))
    require(pin(path)["sha256"] == item.get("sha256"), "Evidence reference changed")
    return path


def characteristics(request: Path, starts: tuple[int, ...], output: Path) -> Path:
    """Bounded PCM measurements, not speech/voicing classification or inference."""
    plan = read(request)
    source = record(plan.get("source"))
    path = canonical(source.get("path"))
    count = plan.get("samples")
    require(type(count) is int and 0 < count <= 60 * 44100, "Bounded interval required")
    count = int(str(count))
    require(
        3 <= len(starts) <= 16 and len(set(starts)) == len(starts) and all(s >= 0 for s in starts),
        "Distinct bounded starts required",
    )
    require(identity(path) == source.get("identity"), "Source changed")
    result: list[dict[str, object]] = []
    for start in starts:
        result.append(_characteristic(path, start, count))
    anchor = next((row for row in result if row["startSample"] == plan["startSample"]), None)
    require(
        anchor is not None and anchor["intervalSHA256"] == plan["intervalSHA256"],
        "Baseline interval changed/missing",
    )
    others = [row for row in result if row is not anchor]
    ordered = sorted(others, key=lambda row: float(str(row["rms"])))
    selected = [record(anchor), ordered[0], ordered[-1]]
    vectors = {
        (row["rms"], row["stereoDifferenceRMS"], row["oneSecondEnergyRange"]) for row in selected
    }
    require(len(vectors) == 3, "Selected clips do not have three distinct measured characteristics")
    require(identity(path) == source["identity"], "Source changed during analysis")
    durable(
        output,
        {
            "version": 1,
            "kind": "source-interval-characteristics",
            "request": pin(request),
            "source": source,
            "measurements": result,
            "selected": selected,
            "selection": "original-baseline-plus-lowest-and-highest-RMS-among-other-candidates",
            "semanticVoicingClassified": False,
        },
    )
    return output


def prepare(request: Path, measured: Path, output: Path) -> Path:
    """Two new immutable benchmark requests; original baseline remains original."""
    plan = read(request)
    evidence = read(measured)
    require(
        reference(evidence.get("request")) == request
        and evidence.get("source") == plan.get("source"),
        "Measurement binding mismatch",
    )
    selected = rows(evidence.get("selected"))
    require(len(selected) == 3, "Three selected intervals required")
    output = canonical(str(output))
    require(
        output.is_relative_to(Path.cwd() / "edits/work") and not output.exists(),
        "Occupied/escaping output",
    )
    original_graph = read(request.parent / "workflow.json")
    original_cpu = record(rows(original_graph["stages"])[0])
    command = rows(original_cpu["command"])
    require(
        command[-5:] == ["run", str(request), str(request.parent / "cpu"), "--device", "cpu"],
        "Unknown original command contract",
    )
    prefix = command[:-5]
    output.mkdir(mode=0o700)
    generated: list[dict[str, object]] = []
    for number, item in enumerate(selected[1:], 1):
        clip = record(item)
        directory = output / f"clip-{number:02d}"
        directory.mkdir(mode=0o700)
        target = directory / "request.json"
        candidate = dict(plan)
        candidate.update(
            startSample=clip["startSample"],
            intervalSHA256=clip["intervalSHA256"],
            sourceCharacterEvidence=pin(measured),
        )
        durable(target, candidate)
        stages: list[dict[str, object]] = []
        for device in ("cpu", "mps"):
            dest = directory / device
            dependencies = [str(target)] + (
                [str(directory / "cpu/worker.json"), str(directory / "cpu/resources.json")]
                if device == "mps"
                else []
            )
            stages.append(
                {
                    "name": device,
                    "command": prefix + ["run", str(target), str(dest), "--device", device],
                    "verify_command": prefix
                    + ["verify", str(target), str(dest), "--device", device],
                    "inputs": dependencies,
                    "outputs": [str(dest / "worker.json"), str(dest / "resources.json")],
                    "timeout_seconds": 660,
                }
            )
        stages.append(
            {
                "name": "compare",
                "command": prefix
                + [
                    "compare",
                    str(target),
                    str(directory / "comparison.json"),
                    "--cpu",
                    str(directory / "cpu"),
                    "--mps",
                    str(directory / "mps"),
                ],
                "inputs": [
                    str(target),
                    *(
                        str(directory / device / name)
                        for device in ("cpu", "mps")
                        for name in ("worker.json", "resources.json")
                    ),
                ],
                "outputs": [str(directory / "comparison.json")],
                "timeout_seconds": 120,
            }
        )
        durable(directory / "workflow.json", {"version": 1, "stages": stages})
        generated.append(
            {
                "request": pin(target),
                "workflow": pin(directory / "workflow.json"),
                "characteristics": clip,
            }
        )
    result = output / "prepared.json"
    durable(
        result,
        {
            "version": 1,
            "baseline": pin(request),
            "characteristics": pin(measured),
            "clips": generated,
            "canDispatchWithoutAdmission": False,
        },
    )
    return result


def _verify_asr(request: Path, receipt: Path) -> dict[str, object]:
    # Independent original-producer adapter; absent capability fails, never passes.
    operation: object = importlib.import_module("replay.separation_vocal_asr_compare").verify
    require(callable(operation), "Missing original ASR comparison verifier")
    if not callable(operation):
        raise TypeError("ASR verifier is not callable")
    return record(operation(request, receipt))


def _assessment(request: Path) -> dict[str, object]:
    data = read(request)
    require(data.get("version") == VERSION, "Separate MPS policy version required")
    clips = rows(data.get("clips"))
    require(3 <= len(clips) <= 16, "At least three bounded clips required")
    seen: set[tuple[str, int, int]] = set()
    vectors: set[tuple[object, object, object]] = set()
    accepted = True
    asr_count = 0
    binding: object = None
    source_binding: object = None
    for value in clips:
        clip = record(value)
        comparison_path = reference(clip.get("comparison"))
        comparison = read(comparison_path)
        plan_path = reference(comparison.get("request"))
        plan = benchmark.load(plan_path)
        if binding is None:
            binding, source_binding = plan["inputs"], plan["source"]
        require(
            plan["inputs"] == binding and plan["source"] == source_binding,
            "Source/model/provider/producer configuration differs between clips",
        )
        for device in ("cpu", "mps"):
            worker_path = reference(comparison.get(device))
            benchmark.verify(plan_path, device, worker_path.parent)
        dirs = [reference(comparison[device]).parent for device in ("cpu", "mps")]
        left, right = (
            load_array(directory / "vocals.npy", samples=int(str(plan["samples"])))
            for directory in dirs
        )
        difference = compare_audio(left, right)
        error = float(np.sum((left.astype(np.float64) - right.astype(np.float64)) ** 2))
        energy = float(np.sum(left.astype(np.float64) ** 2))
        sdr = 10 * math.log10(energy / error) if energy > 0 and error > 0 else None
        wall = [float(str(read(directory / "resources.json")["wallSeconds"])) for directory in dirs]
        require(
            all(math.isfinite(value) and value > 0 for value in wall), "Invalid actual wall times"
        )
        require(
            comparison.get("comparison") == difference
            and comparison.get("sdrDB") == sdr
            and comparison.get("sdrInfinite") == (error == 0 and energy > 0)
            and comparison.get("speedup") == wall[0] / wall[1],
            "Audio comparison differs from actual outputs",
        )
        key = (
            str(record(plan["source"])["path"]),
            int(str(plan["startSample"])),
            int(str(plan["samples"])),
        )
        require(key not in seen, "Duplicate interval")
        seen.add(key)
        characteristic = record(clip.get("characteristics"))
        require(
            characteristic
            == _characteristic(
                canonical(record(plan["source"])["path"]),
                int(str(plan["startSample"])),
                int(str(plan["samples"])),
            ),
            "Signal metrics differ from actual source",
        )
        require(
            characteristic.get("startSample") == plan["startSample"]
            and characteristic.get("sampleCount") == plan["samples"]
            and characteristic.get("intervalSHA256") == plan["intervalSHA256"],
            "Characteristic clock mismatch",
        )
        metrics = read(reference(clip.get("characterEvidence")))
        require(
            characteristic in rows(metrics.get("measurements"))
            and metrics.get("source") == plan["source"],
            "Characteristic evidence mismatch",
        )
        vectors.add(
            (
                characteristic.get("rms"),
                characteristic.get("stereoDifferenceRMS"),
                characteristic.get("oneSecondEnergyRange"),
            )
        )
        if clip.get("asrComparison") is not None:
            asr_path = reference(clip["asrComparison"])
            asr = read(asr_path)
            asr_request_path = reference(asr.get("request"))
            require(asr == _verify_asr(asr_request_path, asr_path), "Unverified ASR comparison")
            require(
                all(
                    asr.get(f"{role}SeparationWorker") == comparison.get(role)
                    for role in ("cpu", "mps")
                ),
                "ASR belongs to another clip",
            )
            asr_count += 1
            accepted = accepted and all(
                asr.get(field) is True
                for field in (
                    "rawTextEqual",
                    "orderedSegmentTextEqual",
                    "orderedWordTextEqual",
                    "tokensEqual",
                    "wordTimingExact",
                    "allNativeCollectionsVerified",
                    "sameModelProviderContext",
                )
            )
        score = comparison.get("sdrDB")
        sdr_ok = comparison.get("sdrInfinite") is True or (
            isinstance(score, (int, float)) and math.isfinite(score) and score >= 40
        )
        correlation = record(comparison.get("comparison")).get("floatCorrelation")
        corr_ok = (
            isinstance(correlation, (int, float))
            and math.isfinite(correlation)
            and correlation >= 0.99999
        )
        accepted = accepted and sdr_ok and corr_ok and float(str(comparison.get("speedup"))) >= 3
    require(len(vectors) >= 3, "Three measured signal characteristics required")
    require(asr_count >= 1, "At least one actual same-clip downstream ASR comparison required")
    return {
        "version": VERSION,
        "request": pin(request),
        "backend": "mps",
        "clips": len(clips),
        "asrClips": asr_count,
        "mpsGatePassed": accepted,
        "cpuExactGateChanged": False,
        "productionBackendAdopted": False,
        "benchmarkInputs": binding,
        "source": source_binding,
        "producer": pin(Path(__file__).resolve()),
    }


def assess(request: Path, output: Path) -> Path:
    durable(output, _assessment(request))
    return output


def verify_acceptance(request: Path, receipt: Path) -> dict[str, object]:
    expected = _assessment(request)
    require(
        read(receipt) == expected and expected["mpsGatePassed"] is True,
        "MPS acceptance missing, stale, changed or failed",
    )
    return expected


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("measure", "prepare", "assess", "verify"))
    parser.add_argument("request", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--metrics", type=Path)
    parser.add_argument("--starts")
    args = parser.parse_args(argv)
    if args.action == "measure":
        require(args.starts is not None, "Explicit candidate starts required")
        characteristics(args.request, tuple(int(s) for s in args.starts.split(",")), args.output)
    elif args.action == "prepare":
        require(args.metrics is not None, "Actual characteristics required")
        prepare(args.request, args.metrics, args.output)
    elif args.action == "verify":
        verify_acceptance(args.request, args.output)
    else:
        assess(args.request, args.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
