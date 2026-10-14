"""Verified quiescent CPU-window adoption and fresh missing-window Metal work.

Original collections, backend labels and failed attempts are never rewritten.
Adoption is a separate terminal-owner boundary, not a request to stop a process.
"""

from __future__ import annotations

import argparse
import hashlib
import os
import shlex
import time
import wave
from collections.abc import Sequence
from contextlib import ExitStack
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

import numpy as np

from replay import isolated_metal_asr as metal
from replay import measured_job_admission as admission
from replay.audio_windows import AudioWindow, plan_audio_windows
from replay.core import ReplayError
from replay.leaf_retirement import lsof_open_among
from replay.native_invocation import durable, identity, pin, require
from replay.recordings import record
from replay.sequence_journal import bounded_read
from replay.stages import execution_lock


@dataclass(frozen=True)
class Plan:
    path: Path
    sha256: str
    metal: metal.Request
    cpu: Path
    policy: Path
    captures: tuple[dict[str, object], ...]
    code: dict[Path, str]
    interrupted_stop_seconds: int | None


class Quiescence(Protocol):
    def verify(
        self,
        captures: tuple[dict[str, object], ...],
        files: tuple[Path, ...],
        *,
        maximum_stop_age: int | None = None,
    ) -> dict[str, object]: ...


class SystemQuiescence:
    """Original terminal verifier + vanished captured groups + exact file probes."""

    def verify(
        self,
        captures: tuple[dict[str, object], ...],
        files: tuple[Path, ...],
        *,
        maximum_stop_age: int | None = None,
    ) -> dict[str, object]:
        proofs = [
            _release(_reference(ref), str(ref["sha256"]), maximum_stop_age, ref.get("stopBarrier"))
            for ref in captures
        ]
        require(
            all(proof.get("resourceReleased") is True for proof in proofs),
            "CPU family not released",
        )
        for offset in range(0, len(files), 64):
            require(
                lsof_open_among([str(path) for path in files[offset : offset + 64]]) == set(),
                "CPU files are open or writer observation is unknown",
            )
        final = [
            _release(_reference(ref), str(ref["sha256"]), maximum_stop_age, ref.get("stopBarrier"))
            for ref in captures
        ]
        require(proofs == final, "CPU terminal evidence changed during writer check")
        return {
            "observedAtNs": time.time_ns(),
            "owners": proofs,
            "openFiles": [],
            "groupsAbsent": True,
        }


def _release(
    path: Path, digest: str, maximum_stop_age: int | None, barrier_ref: object = None
) -> dict[str, object]:
    parent, completion = admission.load_completion_capture(path, digest)
    status = bounded_read(completion.status)
    require(len(status) <= 32 and status.strip().isdigit(), "Malformed CPU terminal status")
    code = int(status.strip())
    require(0 <= code <= 255, "Invalid CPU exit status")
    if code == 0:
        return admission.verify_completion_capture(path, digest)
    require(maximum_stop_age is not None, "Interrupted CPU requires fresh controlled-stop capture")
    barrier = metal._read(_reference(barrier_ref))
    require(
        barrier.get("version") == 1
        and barrier.get("kind") == "stopped-asr-controller-barrier"
        and barrier.get("capture") == pin(path)
        and barrier.get("parent") == admission._parent_record(parent)
        and barrier.get("state") == "T"
        and barrier.get("processGroups") == list(completion.process_groups)
        and barrier.get("producer") == pin(Path(__file__).resolve()),
        "Missing genuine stopped-controller barrier",
    )
    captured = metal._read(path)
    barrier_time = metal._integer(barrier.get("observedAtNs"), 1, 2**63 - 1)
    require(
        int(str(captured["observedAtNs"])) <= barrier_time <= completion.status.stat().st_mtime_ns,
        "Stop barrier chronology mismatch",
    )
    observed = metal._integer(captured.get("observedAtNs"), 1, 2**63 - 1)
    age = completion.status.stat().st_mtime_ns - observed
    require(
        maximum_stop_age is not None and 0 <= age <= maximum_stop_age * 1_000_000_000,
        "Interrupted CPU capture was not taken immediately before termination",
    )
    checkpoint = metal._read(completion.checkpoint)
    require(
        checkpoint.get("fingerprint") == completion.fingerprint
        and checkpoint.get("status") in {"running", "complete"},
        "Interrupted stage fingerprint changed",
    )
    evidence = {
        "version": 1,
        "resourceReleased": True,
        "stageQualityApproved": False,
        "termination": "interrupted",
        "stopBarrier": pin(_reference(barrier_ref)),
        "capture": pin(path),
        "status": pin(completion.status),
        "checkpoint": pin(completion.checkpoint),
        "groups": list(completion.process_groups),
    }
    groups = admission.parse_groups(admission.probe(["/bin/ps", "-A", "-o", "pid=,pgid="]))
    require(
        not any(pid == parent.pid or group in completion.process_groups for pid, group in groups),
        "Stopped CPU family still exists",
    )
    require(
        admission.load_completion_capture(path, digest) == (parent, completion)
        and pin(completion.status) == evidence["status"]
        and pin(completion.checkpoint) == evidence["checkpoint"],
        "Stop evidence changed",
    )
    return evidence


def capture_barrier(capture: Path, output: Path) -> Path:
    """Read an already-stopped exact actor; never send signals or stop an app."""
    digest = str(pin(capture)["sha256"])
    parent, completion = admission.load_completion_capture(capture, digest)
    current = admission.SystemHost()
    require(current.identity(parent.pid) == parent, "Controller identity changed")
    state = admission.probe(["/bin/ps", "-p", str(parent.pid), "-o", "stat="]).strip()
    require(bool(state) and state[0] == "T", "Controller must already be stopped by its owner")
    captured = metal._read(capture)
    workflow = metal._read(_reference(captured.get("workflow")))
    stages = [
        record(row)
        for row in _rows(workflow.get("stages"))
        if record(row).get("name") == captured.get("stage")
    ]
    require(len(stages) == 1, "Missing original stage command")
    command = _rows(stages[0].get("command"))
    require(all(isinstance(item, str) for item in command), "Invalid original command")
    expected = [str(item) for item in command]
    if expected and expected[0] == "/usr/bin/env":
        expected.pop(0)
        while expected and "=" in expected[0] and not expected[0].startswith("/"):
            expected.pop(0)
    if "-m" in expected and expected[expected.index("-m") + 1 : expected.index("-m") + 2] == [
        "replay.phase_timer"
    ]:
        separator = expected.index("-m") + 4
        require(
            len(expected) > separator + 1 and expected[separator] == "--",
            "Malformed original timer wrapper",
        )
        expected = expected[separator + 1 :]
    actual_command = admission.probe(["/bin/ps", "-ww", "-p", str(parent.pid), "-o", "command="])
    actual = shlex.split(actual_command)
    require(
        hashlib.sha256(actual_command.encode()).hexdigest() == parent.command_sha256
        and bool(actual)
        and bool(expected)
        and Path(actual[0]).resolve() == parent.executable
        and Path(expected[0]).resolve() == parent.executable
        and actual[1:] == expected[1:],
        "Stopped actor is not the original stage controller",
    )
    observed = time.time_ns()
    require(
        current.identity(parent.pid) == parent
        and admission.probe(["/bin/ps", "-p", str(parent.pid), "-o", "stat="])
        .strip()
        .startswith("T"),
        "Stopped controller changed during capture",
    )
    require(
        admission.load_completion_capture(capture, digest) == (parent, completion),
        "Capture changed",
    )
    require(
        metal._local(str(output), Path.cwd()).is_relative_to(Path.cwd() / "edits/work"),
        "External barrier output",
    )
    durable(
        output,
        {
            "version": 1,
            "kind": "stopped-asr-controller-barrier",
            "capture": pin(capture),
            "parent": admission._parent_record(parent),
            "observedAtNs": observed,
            "state": "T",
            "processGroups": list(completion.process_groups),
            "producer": pin(Path(__file__).resolve()),
        },
    )
    return output


def _rows(value: object) -> list[object]:
    if not isinstance(value, list) or len(value) > 10000:
        raise ReplayError("Expected bounded rows")
    return value


def _reference(value: object) -> Path:
    ref = record(value)
    path = metal._local(ref.get("path"), Path.cwd().resolve())
    require(
        pin(path)["sha256"] == metal._digest(ref.get("sha256")), "Pinned adoption evidence changed"
    )
    return path


def load(path: Path) -> Plan:
    data = metal._read(path)
    require(
        data.get("version") == 1 and data.get("kind") == "mixed-isolated-asr",
        "Unsupported mixed request",
    )
    request = metal.load_request(_reference(data.get("metalRequest")), verify=False)
    cpu = metal._local(data.get("cpuCollection"), Path.cwd().resolve())
    require(
        cpu.is_dir() and cpu.is_relative_to(Path.cwd() / "edits/work"),
        "Missing local CPU collection",
    )
    policy = _reference(data.get("cpuPolicyStarted"))
    captures = tuple(record(item) for item in _rows(data.get("ownerCaptures")))
    require(
        1 <= len(captures) <= 8
        and len({str(row.get("path")) for row in captures}) == len(captures),
        "Explicit distinct owner captures required",
    )
    for ref in captures:
        captured = metal._read(_reference(ref))
        workflow = metal._read(_reference(captured.get("workflow")))
        matching = [
            record(row)
            for row in _rows(workflow.get("stages"))
            if record(row).get("name") == captured.get("stage")
        ]
        require(len(matching) == 1, "Captured stage is missing/ambiguous")
        outputs = _rows(matching[0].get("outputs"))
        require(
            {str(cpu / "inputs.json"), str(cpu / "complete.json"), str(policy)}
            <= set(str(item) for item in outputs),
            "Captured owner stage does not produce this CPU collection",
        )
    code = {
        metal._local(name, Path.cwd().resolve()): metal._digest(digest)
        for name, digest in record(data.get("codeInputs")).items()
    }
    required = {
        Path(__file__).resolve(),
        Path(admission.__file__).resolve(),
        Path(__file__).with_name("isolated_cpu_asr.py").resolve(),
        Path(__file__).with_name("leaf_retirement.py").resolve(),
        Path(__file__).with_name("phase_timer.py").resolve(),
    }
    require(required <= code.keys() and len(code) <= 1024, "Missing adoption/CPU code pins")
    for source, digest in code.items():
        require(pin(source)["sha256"] == digest, "Adoption code changed")
    stop_age = data.get("interruptedStopMaximumSeconds")
    seconds = None if stop_age is None else metal._integer(stop_age, 1, 30)
    return Plan(path, str(pin(path)["sha256"]), request, cpu, policy, captures, code, seconds)


def _windows(plan: Plan) -> tuple[AudioWindow, ...]:
    return plan_audio_windows(
        plan.metal.samples, window_samples=60 * metal.RATE, stride_samples=30 * metal.RATE
    )


def _cpu_manifest(plan: Plan) -> dict[str, object]:
    manifest = metal._read(plan.cpu / "inputs.json")
    inputs = record(manifest.get("inputs"))
    require(
        manifest
        == {
            "version": 1,
            "inputs": inputs,
            "samples": plan.metal.samples,
            "sampleRate": metal.RATE,
            "windowSamples": 60 * metal.RATE,
            "strideSamples": 30 * metal.RATE,
            "clock": "input-pcm-not-projected",
            "device": "cpu",
            "dtypeContract": "provider-default-fp16-unchanged",
        },
        "CPU clock/backend differs",
    )
    require(1 <= len(inputs) <= 16, "Invalid CPU inputs")
    for source in (plan.metal.pcm, plan.metal.config, plan.metal.weights):
        require(
            inputs.get(str(source)) == plan.metal.pins[source], "CPU/Metal source or model mismatch"
        )
    for name, digest in inputs.items():
        path = metal._local(name, Path.cwd().resolve())
        if path not in plan.metal.pins:
            require(pin(path)["sha256"] == digest, "CPU input/code changed")
        else:
            require(plan.metal.pins[path] == digest, "CPU input binding changed")
    policy = metal._read(plan.policy)
    require(
        policy.get("version") == 1
        and policy.get("inputs") == inputs
        and policy.get("manifestSHA256") == pin(plan.cpu / "inputs.json")["sha256"]
        and policy.get("producerSHA256")
        == plan.code[Path(__file__).with_name("isolated_cpu_asr.py").resolve()]
        and policy.get("processPolicy") == "fresh-cpu-child-per-missing-window"
        and policy.get("cachePolicy") == "unchanged-provider-default",
        "CPU policy/provenance mismatch",
    )
    return manifest


def _files(plan: Plan) -> tuple[Path, ...]:
    """Bounded exact known producer output names, including partial worker logs."""
    paths = [plan.cpu / "inputs.json", plan.policy]
    for window in _windows(plan):
        stem = f"{window.start_sample:012d}"
        paths.extend(plan.cpu / f"{stem}.{suffix}.json" for suffix in ("raw", "clock"))
        runtime = plan.cpu / f"{stem}.cpu-runtime"
        paths.extend(
            runtime / name
            for name in ("samples.npy", "request.json", "result.json", "raw.json", "worker.log")
        )
        paths.extend(runtime / "runtime" / f"{phase}-descriptors.json" for phase in metal.PHASES)
    existing = tuple(path for path in paths if path.exists() or path.is_symlink())
    for path in existing:
        identity(path)
    return existing


def _cpu_window(plan: Plan, window: AudioWindow) -> dict[str, object] | None:
    stem = f"{window.start_sample:012d}"
    raw, clock = (plan.cpu / f"{stem}.{suffix}.json" for suffix in ("raw", "clock"))
    if not raw.exists() or not clock.exists():
        return None
    runtime = plan.cpu / f"{stem}.cpu-runtime"
    request = metal._read(runtime / "request.json")
    payload = runtime / "samples.npy"
    require(
        0 < identity(payload)["bytes"] <= 60 * metal.RATE * 4 + 4096,
        "Invalid isolated sample payload",
    )
    require(
        request
        == {
            "samplesSHA256": pin(payload)["sha256"],
            "model": str(plan.metal.weights.parent),
            "producerSHA256": plan.code[Path(__file__).with_name("isolated_cpu_asr.py").resolve()],
        },
        "CPU child request differs",
    )
    child_raw = pin(runtime / "raw.json")
    require(
        metal._read(runtime / "result.json")
        == {
            "rawSHA256": child_raw["sha256"],
            "samplesSHA256": request["samplesSHA256"],
            "producerSHA256": request["producerSHA256"],
            "device": "cpu",
            "deliveryComplete": False,
        },
        "CPU child result differs",
    )
    require(metal._read(raw) == metal._read(runtime / "raw.json"), "CPU published raw differs")
    digest = str(pin(raw)["sha256"])
    require(metal._read(clock) == metal._clock(window, digest), "CPU raw clock differs")
    with payload.open("rb") as stream:
        samples: object = np.load(stream, allow_pickle=False)
    with wave.open(str(plan.metal.pcm), "rb") as audio:
        audio.setpos(window.start_sample)
        pcm = audio.readframes(window.sample_count)
    expected = np.frombuffer(pcm, dtype="<i2").astype(np.float32) / np.float32(32768)
    require(
        isinstance(samples, np.ndarray)
        and samples.dtype == np.float32
        and samples.ndim == 1
        and len(pcm) == 2 * window.sample_count
        and np.array_equal(samples, expected),
        "CPU samples are not original PCM window",
    )
    paths = [
        raw,
        clock,
        payload,
        runtime / "request.json",
        runtime / "result.json",
        runtime / "raw.json",
    ]
    for phase in metal.PHASES:
        path = runtime / "runtime" / f"{phase}-descriptors.json"
        require(
            metal._read(path).get("externalWritableRegular") == [],
            "CPU descriptor evidence rejected",
        )
        paths.append(path)
    return {
        "version": 1,
        "device": "cpu",
        "startSample": window.start_sample,
        "sampleCount": window.sample_count,
        "raw": pin(raw),
        "clock": pin(clock),
        "originalEvidence": [pin(path) for path in paths],
        "deliveryComplete": False,
    }


def _copy(source: Path, target: Path) -> None:
    data = bounded_read(source)
    if target.exists() or target.is_symlink():
        require(bounded_read(target) == data, "Occupied adopted window differs")
    else:
        with target.open("xb") as stream:
            os.fchmod(stream.fileno(), 0o600)
            stream.write(data)
            stream.flush()
            os.fsync(stream.fileno())


def adopt(request: Path, output: Path, *, host: Quiescence | None = None) -> Path:
    plan = load(request)
    require(
        metal._local(str(output), Path.cwd()).is_relative_to(Path.cwd() / "edits/work")
        and not output.is_relative_to(plan.cpu)
        and not plan.cpu.is_relative_to(output),
        "Independent mixed output required",
    )
    require((plan.cpu / ".run.lock").is_file(), "Original CPU lease is missing")
    with ExitStack() as stack:
        acquired = stack.enter_context(execution_lock(plan.cpu, ".run"))
        require(acquired, "CPU collection is still owned")
        metal.verify_inputs(plan.metal)
        _cpu_manifest(plan)
        files = _files(plan)
        observed = (host or SystemQuiescence()).verify(
            plan.captures, files, maximum_stop_age=plan.interrupted_stop_seconds
        )
        # Resolve every completed pair before creating any adopted output.
        imported = [
            proof for window in _windows(plan) if (proof := _cpu_window(plan, window)) is not None
        ]
        require(bool(imported), "No genuine completed CPU windows")
        (host or SystemQuiescence()).verify(
            plan.captures, files, maximum_stop_age=plan.interrupted_stop_seconds
        )
        output.mkdir(parents=True, exist_ok=True, mode=0o700)
        require(stack.enter_context(execution_lock(output, ".run")), "Mixed collection is owned")
        metadata = {
            "version": 1,
            "kind": "verified-cpu-window-adoption",
            "request": pin(request),
            "cpuManifest": pin(plan.cpu / "inputs.json"),
            "cpuPolicy": pin(plan.policy),
            "imported": imported,
            "skippedStartSamples": [
                window.start_sample
                for window in _windows(plan)
                if window.start_sample not in {row["startSample"] for row in imported}
            ],
            "owners": list(plan.captures),
            "deliveryComplete": False,
        }
        metal.verify_inputs(plan.metal)
        destination = output / "adoption.json"
        if destination.exists():
            require(metal._read(destination) == metadata, "Adoption input history changed")
        else:
            metal._same(output / "adoption-intent.json", metadata)
            for proof in imported:
                start = int(str(proof["startSample"]))
                _copy(_reference(proof["raw"]), output / f"{start:012d}.raw.json")
                _copy(_reference(proof["clock"]), output / f"{start:012d}.clock.json")
                metal._same(output / f"{start:012d}.cpu-provenance.json", proof)
            observation_path = output / "quiescence.json"
            if observation_path.exists():
                previous = metal._read(observation_path)
                require(
                    {k: v for k, v in previous.items() if k != "observedAtNs"}
                    == {k: v for k, v in observed.items() if k != "observedAtNs"},
                    "Quiescence history differs",
                )
            else:
                durable(observation_path, observed)
            durable(destination, metadata)
        metal._same(
            output / "adoption-complete.json",
            {
                "version": 1,
                "adoption": pin(destination),
                "intent": pin(output / "adoption-intent.json"),
                "quiescence": pin(output / "quiescence.json"),
            },
        )
        return destination


def _adoption(plan: Plan, output: Path) -> dict[int, dict[str, object]]:
    commit = metal._read(output / "adoption-complete.json")
    require(
        commit
        == {
            "version": 1,
            "adoption": pin(output / "adoption.json"),
            "intent": pin(output / "adoption-intent.json"),
            "quiescence": pin(output / "quiescence.json"),
        },
        "Adoption commit evidence changed",
    )
    data = metal._read(output / "adoption.json")
    require(metal._read(output / "adoption-intent.json") == data, "Adoption intent differs")
    require(data.get("request") == pin(plan.path), "Mixed request changed")
    _reference(data.get("cpuManifest"))
    _reference(data.get("cpuPolicy"))
    proofs: dict[int, dict[str, object]] = {}
    for value in _rows(data.get("imported")):
        proof = record(value)
        start = metal._integer(proof.get("startSample"), 0, plan.metal.samples - 1)
        require(
            start not in proofs and proof.get("device") == "cpu", "Duplicate/backend adoption error"
        )
        for ref in _rows(proof.get("originalEvidence")):
            _reference(ref)
        require(
            metal._read(output / f"{start:012d}.cpu-provenance.json") == proof,
            "CPU provenance changed",
        )
        proofs[start] = proof
    return proofs


def run(
    request: Path,
    output: Path,
    *,
    runtime: metal.Runtime | None = None,
    verify_only: bool = False,
    maximum_new_windows: int | None = None,
) -> Path:
    plan = load(request)
    require(
        output == output.resolve() and output.is_relative_to(Path.cwd() / "edits/work"),
        "Nonlocal mixed output",
    )
    if maximum_new_windows is not None:
        metal._integer(maximum_new_windows, 1, 720)
        require(not verify_only, "Verify must cover all windows")
    require(
        output.is_dir() and (output / ".run.lock").is_file(), "Missing adopted collection/lease"
    )
    with execution_lock(output, ".run") as acquired:
        require(acquired, "Mixed collection is owned")
        adopted = _adoption(plan, output)
        manifest = {
            "version": 1,
            "inputs": {
                str(plan.path): plan.sha256,
                str(plan.metal.path): plan.metal.sha256,
                **{
                    str(path): plan.metal.pins[path]
                    for path in (plan.metal.pcm, plan.metal.weights, plan.metal.config)
                },
            },
            "samples": plan.metal.samples,
            "sampleRate": metal.RATE,
            "windowSamples": 60 * metal.RATE,
            "strideSamples": 30 * metal.RATE,
            "clock": "input-pcm-not-projected",
            "device": "mixed-cpu-metal",
            "dtypeContract": "provider-default-fp16-unchanged",
        }
        _record(output / "inputs.json", manifest, verify_only)
        identities = metal.verify_inputs(plan.metal)
        hashes: list[str] = []
        receipts: list[dict[str, object]] = []
        created = 0
        for window in _windows(plan):
            start = window.start_sample
            raw = output / f"{start:012d}.raw.json"
            clock = output / f"{start:012d}.clock.json"
            if start in adopted:
                proof = adopted[start]
                require(
                    pin(raw)["sha256"] == record(proof["raw"])["sha256"]
                    and pin(clock)["sha256"] == record(proof["clock"])["sha256"],
                    "Adopted bytes changed",
                )
                receipt = output / f"{start:012d}.cpu-provenance.json"
            else:
                work = output / f"{start:012d}.metal-runtime"
                receipt = work / "complete.json"
                if not receipt.exists():
                    if maximum_new_windows is not None and created >= maximum_new_windows:
                        metal.verify_inputs(plan.metal)
                        last_start = _windows(plan)[len(hashes) - 1].start_sample
                        partial = output / f"partial-through-{last_start:012d}.json"
                        metal._same(
                            partial,
                            {
                                "version": 1,
                                "collectionComplete": False,
                                "requestSHA256": plan.sha256,
                                "rawSHA256": hashes,
                                "windowReceipts": receipts,
                            },
                        )
                        return partial
                    require(
                        not verify_only and not work.exists(), "Uncertain Metal window; no retry"
                    )
                    work.mkdir(mode=0o700)
                    durable(
                        work / "intent.json",
                        {
                            "requestSHA256": plan.metal.sha256,
                            "startSample": start,
                            "sampleCount": window.sample_count,
                            "device": "metal",
                            "inputIdentities": identities,
                        },
                    )
                    (runtime or metal.MetalRuntime()).execute(plan.metal, window, work, identities)
                    metal._identities(identities)
                    durable(
                        receipt,
                        {
                            "version": 1,
                            "device": "metal",
                            "requestSHA256": plan.metal.sha256,
                            "startSample": start,
                            "sampleCount": window.sample_count,
                            "evidence": metal._evidence(work),
                            "deliveryComplete": False,
                        },
                    )
                    created += 1
                require(
                    metal._read(receipt)
                    == {
                        "version": 1,
                        "device": "metal",
                        "requestSHA256": plan.metal.sha256,
                        "startSample": start,
                        "sampleCount": window.sample_count,
                        "evidence": metal._evidence(work),
                        "deliveryComplete": False,
                    },
                    "Metal receipt changed",
                )
                _record(raw, metal._read(work / "raw.json"), verify_only)
                _record(clock, metal._clock(window, str(pin(raw)["sha256"])), verify_only)
            hashes.append(str(pin(raw)["sha256"]))
            receipts.append(pin(receipt))
        metal.verify_inputs(plan.metal)
        complete = {
            "rawSHA256": hashes,
            "windows": len(hashes),
            "deliveryComplete": False,
            "device": "mixed-cpu-metal",
            "windowReceipts": receipts,
            "adoption": pin(output / "adoption.json"),
        }
        _record(output / "complete.json", complete, verify_only)
        return output / "complete.json"


def _record(path: Path, data: dict[str, object], verify: bool) -> None:
    if verify:
        require(metal._read(path) == data, "Mixed collection evidence changed")
    else:
        metal._same(path, data)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("adopt", "run", "verify", "capture-barrier"))
    parser.add_argument("request", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--maximum-new-windows", type=int)
    args = parser.parse_args(argv)
    if args.action == "capture-barrier":
        require(args.maximum_new_windows is None, "Barrier capture never dispatches")
        capture_barrier(args.request, args.output)
    elif args.action == "adopt":
        require(args.maximum_new_windows is None, "Adoption never dispatches")
        adopt(args.request, args.output)
    else:
        run(
            args.request,
            args.output,
            verify_only=args.action == "verify",
            maximum_new_windows=args.maximum_new_windows,
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
