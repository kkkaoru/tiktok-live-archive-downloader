"""Independent, confined one-worker CPU/MPS separation trials on identical PCM."""

from __future__ import annotations

import argparse
import math
import subprocess
import sys
import time
from collections.abc import Callable, Sequence
from contextlib import suppress
from pathlib import Path

import numpy as np

from replay.audio_cpu_probe import audit_descriptors
from replay.bgm_separation import MODEL_SAMPLE_RATE, SeparationBackend, _MlxBackend
from replay.code_dependency_inventory import capture
from replay.core import ReplayError
from replay.native_invocation import canonical, durable, identity, pin, read, require, rows
from replay.process_family_observation import (
    SystemHost,
    cleanup_child,
    sample_family,
    sandbox_profile,
)
from replay.recordings import record
from replay.separation_forward_probe import (
    _kernel_peak_rss,
    _local_environment,
    compare_audio,
    read_interval,
)
from replay.separation_window_budget import load_array, save_array
from replay.sequence_journal import bounded_read
from replay.stages import execution_lock


def load(path: Path, *, verify: bool = True) -> dict[str, object]:
    value = read(path)
    require(
        value.get("version") == 1 and value.get("kind") == "separation-device-benchmark",
        "Invalid benchmark",
    )
    samples = value.get("samples")
    require(
        type(samples) is int and 0 < samples <= 60 * MODEL_SAMPLE_RATE, "Invalid bounded interval"
    )
    start = value.get("startSample")
    require(type(start) is int and start >= 0, "Invalid sample clock")
    timeout = value.get("timeoutSeconds")
    require(type(timeout) is int and 0 < timeout <= 3600, "Invalid deadline")
    limit = value.get("maxFamilyBytes")
    require(type(limit) is int and 0 < limit <= 64 * 1024**3, "Invalid family budget")
    source = record(value.get("source"))
    actual = canonical(source.get("path"))
    require(
        actual.is_relative_to(Path.cwd()) and identity(actual) == source.get("identity"),
        "Source identity changed",
    )
    pins = rows(value.get("inputs"))
    require(bool(pins) and len(pins) <= 1024, "Missing bounded code/provider/model pins")
    required = {
        Path(__file__).resolve(),
        Path(__file__).with_name("bgm_separation.py"),
        Path(__file__).with_name("separation_forward_probe.py"),
        Path(__file__).with_name("process_family_observation.py"),
    }
    paths = {canonical(record(ref).get("path")) for ref in pins}
    require(required <= paths, "Missing benchmark producer pins")
    if verify:
        for ref in pins:
            item = record(ref)
            require(
                pin(canonical(item.get("path")))["sha256"] == item.get("sha256"),
                "Benchmark input changed",
            )
    require(
        isinstance(value.get("intervalSHA256"), str) and len(str(value["intervalSHA256"])) == 64,
        "Missing interval pin",
    )
    _local_environment()
    return value


def worker(
    request: Path,
    device: str,
    output: Path,
    *,
    factory: Callable[[str], SeparationBackend] = _MlxBackend,
    auditor: Callable[[Path, str], None] = audit_descriptors,
) -> Path:
    require(device in {"cpu", "mps"}, "Unsupported device")
    require(
        canonical(str(output)).is_relative_to(Path.cwd() / "edits/work") and output.is_dir(),
        "Worker output must be owned local directory",
    )
    plan = load(request)
    source = canonical(record(plan["source"])["path"])
    interval = read_interval(
        source, start_sample=int(str(plan["startSample"])), samples=int(str(plan["samples"]))
    )
    require(interval.sha256 == plan["intervalSHA256"], "Different original PCM interval")
    auditor(output, "before-load")
    started = time.monotonic()
    backend = factory(device)
    load_seconds = time.monotonic() - started
    auditor(output, "after-load")
    started = time.monotonic()
    audio = backend.extract_vocals(interval.audio)
    seconds = time.monotonic() - started
    auditor(output, "after-inference")
    compare_audio(interval.audio, audio)  # original shared shape/finite guard
    save_array(output / "vocals.npy", audio)
    after = read_interval(
        source, start_sample=int(str(plan["startSample"])), samples=int(str(plan["samples"]))
    )
    require(
        after.sha256 == interval.sha256 and after.source_identity == interval.source_identity,
        "PCM changed",
    )
    load(request)
    result = output / "worker.json"
    durable(
        result,
        {
            "version": 1,
            "request": pin(request),
            "device": device,
            "workers": 1,
            "intervalSHA256": interval.sha256,
            "samples": int(str(plan["samples"])),
            "loadSeconds": load_seconds,
            "inferenceSeconds": seconds,
            "kernelPeakRSSBytes": _kernel_peak_rss(),
            "output": pin(output / "vocals.npy"),
            "qualityApproved": False,
            "compilerServiceWritesVerified": False,
        },
    )
    return result


def run(request: Path, device: str, output: Path) -> Path:
    plan = load(request)
    require(device in {"cpu", "mps"}, "Unsupported device")
    output = canonical(str(output))
    require(output.is_relative_to(Path.cwd() / "edits/work"), "Nonlocal output")
    with execution_lock(output, ".benchmark"):
        require(not (output / "intent.json").exists(), "Existing attempt: inspect, never resubmit")
        durable(output / "intent.json", {"request": pin(request), "device": device})
        profile = output / "sandbox.sb"
        profile.write_text(sandbox_profile(Path.cwd().resolve()))
        command = [
            "/usr/bin/sandbox-exec",
            "-f",
            str(profile),
            sys.executable,
            "-B",
            "-m",
            "replay.separation_device_benchmark",
            "worker",
            str(request),
            str(output),
            "--device",
            device,
        ]
        host = SystemHost()
        birth: str | None = None
        samples: list[dict[str, object]] = []
        errors: list[str] = []
        family_gone = False
        started = time.monotonic()
        with (output / "worker.log").open("xb") as log:
            child = subprocess.Popen(
                command, stdout=log, stderr=subprocess.STDOUT, start_new_session=True
            )
            try:
                birth = host.birth(child.pid)
                durable(
                    output / "launch.json", {"pid": child.pid, "birth": birth, "command": command}
                )
                while child.poll() is None:
                    require(
                        time.monotonic() - started < int(str(plan["timeoutSeconds"])),
                        "Trial timeout",
                    )
                    try:
                        samples.append(sample_family(child.pid, birth, host))
                    except ReplayError as error:
                        errors.append(str(error))
                        try:
                            terminal = child.wait(timeout=1)
                        except subprocess.TimeoutExpired:
                            terminal = None
                        if terminal != 0:
                            raise
                        break
                    require(
                        max(int(str(samples[-1][key])) for key in ("rssBytes", "physicalBytes"))
                        <= int(str(plan["maxFamilyBytes"])),
                        "Family resource budget exceeded",
                    )
                    with suppress(subprocess.TimeoutExpired):
                        child.wait(timeout=0.2)
                require(child.returncode == 0 and bool(samples), "Failed/unmeasured trial")
                require(
                    all(
                        host.absent(int(str(record(member)["pid"])))
                        for sample in samples
                        for member in rows(sample["members"])
                    ),
                    "Trial family remains",
                )
                family_gone = True
            finally:
                cleanup = cleanup_child(child, birth, host)
                durable(
                    output / "resources.json",
                    {
                        "exitCode": child.returncode,
                        "familyGone": family_gone,
                        "cleanup": cleanup,
                        "samples": samples,
                        "samplingErrors": errors,
                        "wallSeconds": time.monotonic() - started,
                        "sampledPeakPhysicalBytes": max(
                            (int(str(row["physicalBytes"])) for row in samples), default=0
                        ),
                        "sampledPeakRSSBytes": max(
                            (int(str(row["rssBytes"])) for row in samples), default=0
                        ),
                        "absolutePhysicalPeakVerified": False,
                    },
                )
        return verify(request, device, output)


def verify(request: Path, device: str, output: Path) -> Path:
    load(request)
    result = read(output / "worker.json")
    require(
        result.get("request") == pin(request)
        and result.get("device") == device
        and result.get("workers") == 1,
        "Worker binding mismatch",
    )
    ref = record(result.get("output"))
    require(ref == pin(output / "vocals.npy"), "Output changed")
    resources = read(output / "resources.json")
    require(
        resources.get("exitCode") == 0
        and resources.get("familyGone") is True
        and record(resources.get("cleanup")).get("terminalObserved") is True
        and bool(rows(resources.get("samples"))),
        "Missing terminal resource evidence",
    )
    for phase in ("before-load", "after-load", "after-inference"):
        require(
            read(output / f"{phase}-descriptors.json").get("externalWritableRegular") == [],
            "External writable FD",
        )
    log = bounded_read(output / "worker.log").decode("utf-8", errors="replace")
    require(
        not any(
            word in log.lower()
            for word in ("deny file-write", "sandbox: deny", "operation not permitted")
        ),
        "Sandbox denial",
    )
    return output / "worker.json"


def compare(request: Path, cpu: Path, mps: Path, output: Path) -> Path:
    plan = load(request)
    for device, path in (("cpu", cpu), ("mps", mps)):
        verify(request, device, path)
    count = int(str(plan["samples"]))
    left, right = (load_array(path / "vocals.npy", samples=count) for path in (cpu, mps))
    difference = compare_audio(left, right)
    error = float(np.sum((left.astype(np.float64) - right.astype(np.float64)) ** 2))
    energy = float(np.sum(left.astype(np.float64) ** 2))
    sdr = 10 * math.log10(energy / error) if energy > 0 and error > 0 else None
    cpu_time, gpu_time = (
        float(str(read(path / "resources.json")["wallSeconds"])) for path in (cpu, mps)
    )
    require(cpu_time > 0 and gpu_time > 0, "Invalid measurement duration")
    speedup = cpu_time / gpu_time
    exact = (
        difference["floatMaxAbsoluteDifference"] == 0
        and difference["pcm16DifferingChannelSamples"] == 0
    )
    durable(
        output,
        {
            "request": pin(request),
            "cpu": pin(cpu / "worker.json"),
            "mps": pin(mps / "worker.json"),
            "comparison": difference,
            "sdrDB": sdr,
            "sdrInfinite": error == 0 and energy > 0,
            "speedup": speedup,
            "atLeastThreeTimes": speedup >= 3,
            "existingExactEquivalencePassed": exact,
            "performanceAndEquivalencePassed": speedup >= 3 and exact,
            "qualityApproved": False,
            "productionBackendAdopted": False,
        },
    )
    return output


def prepare(prior_plan: Path, prior_input: Path, output: Path, *, family_bytes: int) -> Path:
    """Preserve measured interval/model references; no media/model read or dispatch."""
    require(type(family_bytes) is int and 0 < family_bytes <= 64 * 1024**3, "Invalid family budget")
    old, observed = read(prior_plan), read(prior_input)
    require(
        record(record(observed.get("context")).get("plan")).get("sha256")
        == pin(prior_plan)["sha256"],
        "Prior interval plan binding differs",
    )
    require(old.get("sampleCount") == observed.get("sampleCount"), "Prior interval count differs")
    output = canonical(str(output))
    require(
        output.is_relative_to(Path.cwd() / "edits/work") and not output.exists(),
        "Occupied/escaping output",
    )
    refs = [*rows(old.get("providerInputs")), *rows(old.get("modelInputs"))]
    refs.extend(
        {"path": str(item.path), "sha256": item.sha256}
        for item in capture(("separation_device_benchmark",))
    )
    request = {
        "version": 1,
        "kind": "separation-device-benchmark",
        "source": old["source"],
        "samples": old["sampleCount"],
        "startSample": 0,
        "intervalSHA256": observed["intervalSHA256"],
        "inputs": refs,
        "timeoutSeconds": 600,
        "maxFamilyBytes": family_bytes,
        "priorPlan": pin(prior_plan),
        "priorInput": pin(prior_input),
        "productionBackendAdopted": False,
    }
    output.mkdir(mode=0o700)
    request_path = output / "request.json"
    durable(request_path, request)
    env = {name: str(output / name.lower()) for name in ("TMPDIR", "TMP", "TEMP", "XDG_CACHE_HOME")}
    for value in env.values():
        Path(value).mkdir(mode=0o700)
    models = rows(old["modelInputs"])
    weights = canonical(record(models[0])["path"])
    hub = weights.parent.parent.parent.parent
    env.update(
        HF_HOME=str(hub.parent),
        HF_HUB_CACHE=str(hub),
        HF_HUB_OFFLINE="1",
        TRANSFORMERS_OFFLINE="1",
        PYTHONDONTWRITEBYTECODE="1",
        PYTHONPATH=str(Path.cwd() / "src"),
    )
    # The selected provider interpreter is explicit and comes from prior provider evidence.
    core = next(
        canonical(record(ref)["path"])
        for ref in refs
        if "/site-packages/mlx/core." in str(record(ref)["path"])
    )
    interpreter = core.parents[4] / "bin/python"
    require(interpreter.is_file(), "Pinned provider interpreter missing")
    prefix = [
        "/usr/bin/env",
        *(f"{key}={value}" for key, value in env.items()),
        str(interpreter),
        "-B",
        "-m",
        "replay.separation_device_benchmark",
    ]
    stages: list[dict[str, object]] = []
    for device in ("cpu", "mps"):
        target = output / device
        stages.append(
            {
                "name": device,
                "command": prefix + ["run", str(request_path), str(target), "--device", device],
                "verify_command": prefix
                + ["verify", str(request_path), str(target), "--device", device],
                "inputs": [str(request_path)]
                + (
                    [str(output / "cpu/worker.json"), str(output / "cpu/resources.json")]
                    if device == "mps"
                    else []
                ),
                "outputs": [str(target / "worker.json"), str(target / "resources.json")],
                "timeout_seconds": 660,
            }
        )
    stages.append(
        {
            "name": "compare",
            "command": prefix
            + [
                "compare",
                str(request_path),
                str(output / "comparison.json"),
                "--cpu",
                str(output / "cpu"),
                "--mps",
                str(output / "mps"),
            ],
            "inputs": [
                str(request_path),
                str(output / "cpu/worker.json"),
                str(output / "cpu/resources.json"),
                str(output / "mps/worker.json"),
                str(output / "mps/resources.json"),
            ],
            "outputs": [str(output / "comparison.json")],
            "timeout_seconds": 120,
        }
    )
    durable(output / "workflow.json", {"version": 1, "stages": stages})
    return request_path


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("run", "worker", "verify", "compare", "prepare"))
    parser.add_argument("request", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--device", choices=("cpu", "mps"))
    parser.add_argument("--cpu", type=Path)
    parser.add_argument("--mps", type=Path)
    parser.add_argument("--prior-input", type=Path)
    parser.add_argument("--family-bytes", type=int)
    args = parser.parse_args(argv)
    if args.action == "prepare":
        require(
            args.prior_input is not None and args.family_bytes is not None,
            "Explicit input/budget required",
        )
        prepare(args.request, args.prior_input, args.output, family_bytes=args.family_bytes)
    elif args.action == "compare":
        require(args.cpu is not None and args.mps is not None, "Both trials required")
        compare(args.request, args.cpu, args.mps, args.output)
    else:
        require(args.device is not None, "Device required")
        {"run": run, "worker": worker, "verify": verify}[args.action](
            args.request, str(args.device), args.output
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
