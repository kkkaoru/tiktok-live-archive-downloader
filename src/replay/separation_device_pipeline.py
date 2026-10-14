"""Explicit MPS continuation with retained CPU provenance and unchanged PCM clocks.

Preparation never launches inference. CPU receipts remain immutable; an incomplete
CPU chunk is inventoried as unadopted, never relabelled or silently discarded.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
import time
import wave
from collections.abc import Callable, Iterator, Sequence
from contextlib import ExitStack, contextmanager, suppress
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from replay import mps_separation_acceptance as acceptance
from replay import separation_window_pipeline as original
from replay import workflow
from replay.audio_cpu_probe import audit_descriptors, external_writable_files
from replay.bgm_separation import MODEL_SAMPLE_RATE, SeparationBackend, _MlxBackend
from replay.code_dependency_inventory import capture
from replay.core import ReplayError
from replay.delivery_audio import verify_material
from replay.native_invocation import canonical, durable, identity, pin, require, rows
from replay.process_family_observation import (
    SystemHost,
    cleanup_child,
    sample_family,
    sandbox_profile,
)
from replay.recordings import record
from replay.separation_forward_probe import _local_environment
from replay.separation_window_budget import (
    WINDOW,
    Window,
    load_array,
    reduce_windows,
    save_array,
    windows,
)
from replay.sequence_journal import bounded_read, decode
from replay.stages import execution_lock
from replay.stateful_guard import saved_bytes

PROTOCOL = "explicit-device-separation-continuation-v1"
OWNER_PROTOCOL = "separation-device-owned-terminal-v1"
MAX_FILES = 4096
MAX_TIMEOUT = 3600
MAX_FAMILY_BYTES = 64 * 1024**3
PHASES = ("before-load", "after-load", "after-inference")
OBSERVATION_RETRY_SECONDS = 1.0
OBSERVATION_RETRY_ATTEMPTS = 10
OBSERVATION_RETRY_WAIT = 0.05
TRANSIENT_OBSERVATION_ERRORS = frozenset(
    {
        "Exact process birth unavailable",
        "Physical footprint unavailable",
        "Owned process vanished during inventory",
    }
)
Factory = Callable[[str], SeparationBackend]
Auditor = Callable[[Path, str], None]


@dataclass(frozen=True)
class Job:
    path: Path
    old: original.PipelinePlan
    work: Path
    source: Path
    configuration: dict[str, object]
    cpu: dict[int, Path]
    timeout: int
    family_bytes: int
    disk_bytes: int


class Arguments(argparse.Namespace):
    action: str
    config: Path
    output: Path | None
    index: int | None
    pid: int | None


def _read(path: Path) -> dict[str, object]:
    return decode(bounded_read(path))


def _local(value: object) -> Path:
    path = canonical(value)
    require(path.is_relative_to(Path.cwd().resolve() / "edits/work"), "Expected local work path")
    return path


def _number(value: object, *, maximum: int, zero: bool = False) -> int:
    require(
        type(value) is int and (0 if zero else 1) <= int(str(value)) <= maximum, "Invalid bound"
    )
    return int(str(value))


def _file(path: Path) -> dict[str, str]:
    before = identity(path)
    with path.open("rb") as stream:
        result = hashlib.file_digest(stream, "sha256").hexdigest()
    require(identity(path) == before, "Input changed during hashing")
    return {"path": str(path), "sha256": result}


def _ref(value: object, *, verify: bool = True, local: bool = True) -> Path:
    item = record(value)
    path = canonical(item.get("path"))
    digest = item.get("sha256")
    require(
        set(item) == {"path", "sha256"}
        and isinstance(digest, str)
        and len(digest) == 64
        and all(char in "0123456789abcdef" for char in digest),
        "Exact path/SHA reference required",
    )
    if local:
        _local(str(path))
    identity(path)
    if verify:
        require(_file(path) == item, "Pinned input changed")
    return path


def _code() -> dict[str, str]:
    return {
        str(item.path): item.sha256
        for item in capture(("separation_device_pipeline", "separation_vocal_asr_compare"))
    }


def _no_old_writers(plan: original.PipelinePlan) -> None:
    result = subprocess.run(
        ["/bin/ps", "-axo", "args="], capture_output=True, text=True, check=True, timeout=2
    )
    require(len(result.stdout) <= 4 * 1024**2, "Process observation overflow")
    require(
        not any(
            (" -m replay.separation_window_pipeline " in line and str(plan.path) in line)
            or (" -m replay.separation_window_budget " in line and str(plan.directory) in line)
            for line in result.stdout.splitlines()
        ),
        "Original CPU writer remains",
    )


def _lease_paths(plan: original.PipelinePlan) -> tuple[tuple[Path, str], ...]:
    pools = tuple(
        original.chunk_proxy(plan, index).directory / "parallel"
        for index in range(len(plan.bounds))
    )
    return ((plan.directory, ".pipeline"),) + tuple(
        (pool, ".parent") for pool in pools if pool.exists()
    )


@contextmanager
def _leases(plan: original.PipelinePlan) -> Iterator[None]:
    with ExitStack() as stack:
        for directory, name in _lease_paths(plan):
            identity(directory / f"{name}.lock")
            acquired = stack.enter_context(execution_lock(directory, name))
            require(acquired, "Original CPU lease remains owned")
        yield


def _worker_leases(plan: original.PipelinePlan) -> None:
    """Check busy cooperative leases without process inventory inside Seatbelt."""
    for directory, name in _lease_paths(plan):
        identity(directory / f"{name}.lock")
        with execution_lock(directory, name) as acquired:
            require(not acquired, "Worker has no lease-owning parent")


def _inventory(directory: Path) -> list[dict[str, str]]:
    require(directory == directory.resolve(), "Linked original partial directory")
    paths = sorted(directory.rglob("*"))
    require(
        len(paths) <= MAX_FILES and all(path == path.resolve() for path in paths),
        "Partial inventory linked or exceeds budget",
    )
    return [_file(path) for path in paths if not path.is_dir() and not path.name.endswith(".lock")]


def _retained(
    plan: original.PipelinePlan,
) -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    complete: list[dict[str, object]] = []
    partial: list[dict[str, object]] = []
    for index in range(len(plan.bounds)):
        directory = original.chunk_proxy(plan, index).directory
        if not directory.exists():
            continue
        checkpoint = directory / "complete.json"
        if checkpoint.exists():
            complete.append({"index": index, "complete": _file(checkpoint)})
        else:
            partial.append(
                {"index": index, "disposition": "UNADOPTED", "files": _inventory(directory)}
            )
    return complete, partial


def capture_owner(plan_path: Path, pid: int, output: Path) -> Path:
    """Observe a genuinely running original owner; caller must preserve this before retirement."""
    plan = original.load_plan(plan_path)
    host = SystemHost()
    birth = host.birth(pid)
    command = subprocess.run(
        ["/bin/ps", "-p", str(pid), "-o", "args="],
        check=True,
        capture_output=True,
        text=True,
        timeout=2,
    ).stdout.strip()
    require(
        len(command) <= 32768 and f" -m replay.separation_window_pipeline {plan.path}" in command,
        "Owner command is not the exact original pipeline",
    )
    family = sample_family(pid, birth, host)
    require(host.birth(pid) == birth, "Owner changed during observation")
    durable(
        _local(str(output)),
        {
            "version": 1,
            "protocol": "separation-device-running-owner-v1",
            "originalPlan": _file(plan.path),
            "source": plan.definition["source"],
            "command": command,
            "controllerPID": pid,
            "birth": birth,
            "family": family,
            "producer": _file(Path(__file__).resolve()),
        },
    )
    return output


def _baseline_pids(baseline: dict[str, object], plan: original.PipelinePlan) -> tuple[int, ...]:
    require(
        baseline.get("protocol") == "separation-device-running-owner-v1"
        and baseline.get("originalPlan") == _file(plan.path)
        and baseline.get("source") == plan.definition["source"],
        "Original owner binding changed",
    )
    members = rows(record(baseline.get("family")).get("members"))
    require(bool(members), "Original owner family missing")
    pids = [_number(record(member).get("pid"), maximum=2**31 - 1) for member in members]
    require(
        len(pids) == len(set(pids))
        and baseline.get("controllerPID") in pids
        and all(
            isinstance(record(member).get("birth"), str) and record(member)["birth"]
            for member in members
        ),
        "Original owner family is unbound",
    )
    return tuple(pids)


def _gone(baseline: dict[str, object], plan: original.PipelinePlan) -> None:
    pids = _baseline_pids(baseline, plan)
    host = SystemHost()
    require(all(host.absent(pid) for pid in pids), "Original owner family remains")
    _no_old_writers(plan)


def observe_handoff(request: Path, output: Path) -> Path:
    """After Root's actual retirement, seal complete refs and preserved unadopted partials."""
    data = _read(request)
    require(
        set(data) == {"version", "originalPlan", "baseline", "status", "futureTargetsDisabled"}
        and data.get("version") == 1
        and data.get("futureTargetsDisabled") is True,
        "Explicit terminal handoff request required",
    )
    plan = original.load_plan(_ref(data["originalPlan"]))
    baseline = _ref(data["baseline"])
    status = _ref(data["status"])
    terminal = bounded_read(status).decode("ascii").strip()
    require(terminal.lstrip("-").isdecimal(), "Actual terminal status missing")
    _gone(_read(baseline), plan)
    with _leases(plan):
        complete, partial = _retained(plan)
        _gone(_read(baseline), plan)
        durable(
            _local(str(output)),
            {
                "version": 1,
                "protocol": OWNER_PROTOCOL,
                "request": _file(request),
                "originalPlan": data["originalPlan"],
                "baseline": data["baseline"],
                "status": data["status"],
                "terminalCode": int(terminal),
                "futureTargetsDisabled": True,
                "originalFamilyGone": True,
                "historicalProductionComplete": False,
                "completeCpuChunks": complete,
                "preservedUnadoptedPartials": partial,
                "observedAtNs": time.time_ns(),
                "producer": _file(Path(__file__).resolve()),
            },
        )
    return output


def _handoff(job: Job, *, observe_processes: bool = True) -> None:
    data = _read(_ref(job.configuration["handoff"]))
    request = _read(_ref(data.get("request")))
    require(
        data.get("protocol") == OWNER_PROTOCOL
        and data.get("originalFamilyGone") is True
        and data.get("futureTargetsDisabled") is True
        and data.get("historicalProductionComplete") is False
        and data.get("producer") == _file(Path(__file__).resolve())
        and type(data.get("observedAtNs")) is int
        and int(str(data["observedAtNs"])) > 0
        and request.get("version") == 1
        and data.get("originalPlan") == job.configuration["originalPlan"]
        and request.get("originalPlan") == data["originalPlan"]
        and request.get("baseline") == data.get("baseline")
        and request.get("status") == data.get("status")
        and request.get("futureTargetsDisabled") is True,
        "Invalid or mismatched terminal handoff",
    )
    baseline, status = _ref(data["baseline"]), _ref(data["status"])
    require(
        bounded_read(status).decode("ascii").strip() == str(data.get("terminalCode")),
        "Terminal status changed",
    )
    original_owner = _read(baseline)
    _baseline_pids(original_owner, job.old)
    if observe_processes:
        _gone(original_owner, job.old)
    complete, partial = _retained(job.old)
    require(
        data.get("completeCpuChunks") == complete
        and data.get("preservedUnadoptedPartials") == partial
        and job.configuration["retainedCpuChunks"] == complete,
        "Retained CPU/partial inventory changed",
    )


def _acceptance(job: Job) -> dict[str, object]:
    item = record(job.configuration["acceptance"])
    require(set(item) == {"request", "receipt"}, "Pinned MPS acceptance request/receipt required")
    proof = _qualified(job)
    require(
        proof.get("mpsGatePassed") is True
        and proof.get("cpuExactGateChanged") is False
        and proof.get("backend") == "mps"
        and proof.get("version") == acceptance.VERSION
        and proof.get("producer") == pin(Path(acceptance.__file__).resolve()),
        "Separate MPS gate did not pass",
    )
    source = record(proof.get("source"))
    require(
        identity(canonical(source.get("path"))) == source.get("identity"),
        "Qualified benchmark source changed",
    )
    pins = {
        str(record(row).get("path")): record(row).get("sha256")
        for row in rows(proof.get("benchmarkInputs"))
    }
    required = (
        *rows(job.old.definition["modelInputs"]),
        *rows(job.old.definition["providerInputs"]),
    )
    code = {
        str(record(row)["path"]): record(row)["sha256"]
        for row in rows(job.old.definition["codeInputs"])
    }
    backend = str(Path(original.__file__).with_name("bgm_separation.py"))
    require(
        all(pins.get(str(record(row)["path"])) == record(row)["sha256"] for row in required)
        and backend in code
        and pins.get(backend) == code[backend],
        "MPS backend/model/provider differs from CPU",
    )
    return proof


def _verification_python(job: Job) -> tuple[Path, Path, Path]:
    tool = record(job.configuration["verificationPython"])
    require(
        set(tool) == {"lexical", "executable", "environment"}, "Pinned verifier Python required"
    )
    lexical = Path(str(tool["lexical"]))
    executable = _ref(tool["executable"], local=False)
    environment = _ref(tool["environment"], local=False)
    require(
        lexical.is_absolute()
        and lexical.parent == lexical.parent.resolve()
        and lexical.parent.name == "bin"
        and lexical.name == "python"
        and lexical.parent.parent.is_relative_to(Path.cwd().resolve())
        and lexical.resolve() == executable
        and environment == lexical.parent.parent / "pyvenv.cfg"
        and os.access(lexical, os.X_OK),
        "Verifier executable/virtual environment differs",
    )
    return lexical, executable, environment


def _qualified(job: Job) -> dict[str, object]:
    """Use the pinned full verification environment, not the inference provider's imports."""
    lexical, executable, environment = _verification_python(job)
    item = record(job.configuration["acceptance"])
    request, receipt = _ref(item["request"]), _ref(item["receipt"])
    before = [_file(path) for path in (executable, environment, request, receipt)]
    result = subprocess.run(
        [
            str(lexical),
            "-B",
            "-m",
            "replay.mps_separation_acceptance",
            "verify",
            str(request),
            str(receipt),
        ],
        stdin=subprocess.DEVNULL,
        capture_output=True,
        timeout=120,
        check=False,
        env={**os.environ, "PYTHONDONTWRITEBYTECODE": "1"},
    )
    require(
        result.returncode == 0 and len(result.stdout) + len(result.stderr) <= 1024**2,
        "Readonly MPS qualification failed in pinned verifier environment",
    )
    require(
        [_file(path) for path in (executable, environment, request, receipt)] == before,
        "Qualification inputs changed during verification",
    )
    return _read(receipt)


def _load(path: Path, *, ready: bool = True, observe_processes: bool = True) -> Job:
    data = _read(_local(str(path)))
    require(
        set(data)
        == {
            "version",
            "kind",
            "originalPlan",
            "preparation",
            "originalRecording",
            "allowedSourceRoot",
            "acceptance",
            "handoff",
            "retainedCpuChunks",
            "verificationPython",
            "outputDirectory",
            "budget",
        }
        and data.get("version") == 1
        and data.get("kind") == PROTOCOL,
        "Invalid device continuation config",
    )
    plan = original.load_plan(_ref(data["originalPlan"]))
    prepared = _read(_ref(data["preparation"]))
    require(
        prepared.get("version") == 1
        and prepared.get("kind") == "verified-separation-pcm-preparation"
        and prepared.get("outputSampleCount") == plan.samples
        and prepared.get("sampleRate") == MODEL_SAMPLE_RATE
        and prepared.get("channels") == 2
        and prepared.get("sampleWidthBytes") == 2
        and prepared.get("sourceProjectionApplications") == 0
        and prepared.get("timestampCorrection") is False
        and record(prepared.get("output")).get("path") == str(plan.source)
        and record(prepared.get("output")).get("sha256")
        == record(plan.definition["source"])["sha256"],
        "Prepared source/clock differs",
    )
    source = _ref(data["originalRecording"], verify=False, local=False)
    require(
        record(prepared.get("inputs")).get(str(source))
        == record(data["originalRecording"])["sha256"],
        "Original-to-prepared source binding differs",
    )
    allowed = (
        canonical(data["allowedSourceRoot"])
        if data["allowedSourceRoot"] is not None
        else Path.cwd().resolve()
    )
    require(
        allowed.is_dir() and source.is_relative_to(allowed),
        "Original recording escapes declared root",
    )
    budget = record(data["budget"])
    require(
        set(budget) == {"timeoutSeconds", "maxFamilyBytes", "maxSavedBytes"},
        "Invalid resource budget",
    )
    cpu: dict[int, Path] = {}
    for raw in rows(data["retainedCpuChunks"]):
        row = record(raw)
        index = _number(row.get("index"), maximum=len(plan.bounds) - 1, zero=True)
        checkpoint = _ref(row.get("complete"))
        require(
            set(row) == {"index", "complete"}
            and index not in cpu
            and checkpoint == original.chunk_proxy(plan, index).directory / "complete.json",
            "CPU chunk binding differs",
        )
        cpu[index] = checkpoint
    job = Job(
        path,
        plan,
        _local(data["outputDirectory"]),
        source,
        data,
        cpu,
        _number(budget["timeoutSeconds"], maximum=MAX_TIMEOUT),
        _number(budget["maxFamilyBytes"], maximum=MAX_FAMILY_BYTES),
        _number(budget["maxSavedBytes"], maximum=256 * 1024**3),
    )
    require(
        job.work != plan.directory and not job.work.is_relative_to(plan.directory),
        "New output aliases original",
    )
    if ready:
        require(
            data["acceptance"] is not None and data["handoff"] is not None,
            "Acceptance/actual terminal handoff pending",
        )
        _acceptance(job)
        _handoff(job, observe_processes=observe_processes)
        catalog = _read(job.work / "plan.json")
        require(
            catalog.get("configuration") == _file(path)
            and catalog.get("codeInputs") == _code()
            and catalog.get("canDispatch") is True
            and catalog.get("originalBounds") == [list(bound) for bound in plan.bounds],
            "Prepared operation/source generation changed",
        )
    return job


def _original_context(job: Job) -> dict[str, object]:
    actual = original.context(job.old)
    retained = _read(job.old.directory / "input.json")
    require(
        actual.get("plan") == retained.get("plan")
        and actual.get("inputs") == retained.get("inputs")
        and all(
            record(actual.get("environment")).get(key)
            == record(retained.get("environment")).get(key)
            for key in ("HF_HOME", "HF_HUB_CACHE")
        ),
        "Original current input provenance changed",
    )
    return retained


def _environment(job: Job, index: int) -> dict[str, str]:
    directory = _chunk(job, index)
    retained = record(_original_context(job)["environment"])
    require(
        set(retained) == {"TMPDIR", "TMP", "TEMP", "XDG_CACHE_HOME", "HF_HOME", "HF_HUB_CACHE"}
        and all(isinstance(value, str) for value in retained.values()),
        "Original environment policy missing",
    )
    result = {key: str(value) for key, value in retained.items()}
    result.update(
        TMPDIR=str(directory / "tmp"),
        TMP=str(directory / "tmp"),
        TEMP=str(directory / "tmp"),
        XDG_CACHE_HOME=str(directory / "cache"),
    )
    return result


def _context(job: Job, *, observe_processes: bool = True) -> dict[str, object]:
    qualification = _acceptance(job)
    _handoff(job, observe_processes=observe_processes)
    original_context = _original_context(job)
    require(
        _file(job.source) == job.configuration["originalRecording"], "Original recording changed"
    )
    return {
        "configuration": _file(job.path),
        "originalContext": original_context,
        "readerCode": _code(),
        "acceptance": job.configuration["acceptance"],
        "benchmarkSource": qualification["source"],
        "productionSource": job.old.definition["source"],
        "handoff": job.configuration["handoff"],
    }


def _chunk(job: Job, index: int) -> Path:
    _number(index, maximum=len(job.old.bounds) - 1, zero=True)
    return job.work / "chunks" / f"chunk-{index:06d}"


def _chunk_outputs(job: Job, index: int) -> list[Path]:
    directory = _chunk(job, index)
    names = (
        ("cpu-reuse.json",)
        if index in job.cpu
        else (
            "request.json",
            "launch.json",
            "resources.json",
            "worker.log",
            "worker.json",
            "vocals.npy",
            "before-load-descriptors.json",
            "after-load-descriptors.json",
            "after-inference-descriptors.json",
        )
    )
    arrays = (
        []
        if index in job.cpu
        else [
            directory / f"window-{w.index:06d}.npy"
            for w in windows(job.old.bounds[index][1] - job.old.bounds[index][0])
        ]
    )
    return [directory / name for name in (*names, "complete.json")] + arrays


def _proof(job: Job, index: int) -> dict[str, object]:
    outputs = [path for path in _chunk_outputs(job, index) if path.name != "complete.json"]
    return {
        "protocol": PROTOCOL,
        "index": index,
        "bound": list(job.old.bounds[index]),
        "device": "cpu" if index in job.cpu else "mps",
        "outputs": [pin(path) for path in outputs],
        "context": _context(job),
        "qualityApproved": False,
    }


def _prediction(job: Job, index: int) -> np.ndarray[tuple[int, ...], np.dtype[np.float32]]:
    directory = _chunk(job, index)
    require(
        _read(directory / "complete.json") == _proof(job, index), "Device chunk completion changed"
    )
    if index in job.cpu:
        require(
            _read(directory / "cpu-reuse.json")
            == {"oldComplete": _file(job.cpu[index]), "device": "cpu"},
            "CPU reuse changed",
        )
        return original.verify_chunk(job.old, index=index, bound=_original_context(job))
    request = _read(directory / "request.json")
    parent = record(request.get("parent"))
    require(
        request.get("context") == _context(job)
        and request.get("index") == index
        and request.get("device") == "mps"
        and request.get("bound") == list(job.old.bounds[index])
        and set(parent) == {"pid", "birth"}
        and type(parent.get("pid")) is int
        and 0 < int(str(parent["pid"])) < 2**31
        and isinstance(parent.get("birth"), str)
        and bool(parent["birth"])
        and _read(directory / "launch.json").get("parent") == parent,
        "MPS request changed",
    )
    result = _read(directory / "worker.json")
    require(
        result.get("request") == pin(directory / "request.json")
        and result.get("parent") == parent
        and result.get("device") == "mps"
        and result.get("workerEnvironment") == _environment(job, index),
        "Worker completion changed",
    )
    refs = rows(result.get("windows"))
    start, end = job.old.bounds[index]
    planned = windows(end - start)
    require(len(refs) == len(planned), "Window count changed")

    def prediction(window: Window) -> np.ndarray[tuple[int, ...], np.dtype[np.float32]]:
        path = directory / f"window-{window.index:06d}.npy"
        require(
            record(refs[window.index])
            == {
                "window": {"index": window.index, "start": window.start, "count": window.count},
                "output": pin(path),
            },
            "Window clock/output changed",
        )
        return load_array(path, samples=WINDOW)

    expected = reduce_windows(end - start, prediction)
    actual = load_array(directory / "vocals.npy", samples=end - start)
    require(np.array_equal(expected, actual), "MPS ordered reduction changed")
    resources = _read(directory / "resources.json")
    require(
        resources.get("exitCode") == 0
        and resources.get("familyGone") is True
        and record(resources.get("cleanup")).get("terminalObserved") is True
        and bool(rows(resources.get("samples"))),
        "MPS terminal resources missing",
    )
    require(
        all(
            max(int(str(record(sample)[key])) for key in ("rssBytes", "physicalBytes"))
            <= job.family_bytes
            for sample in rows(resources["samples"])
        ),
        "MPS sampled resources exceeded",
    )
    for phase in PHASES:
        observed = _read(directory / f"{phase}-descriptors.json")
        require(
            type(observed.get("pid")) is int
            and int(str(observed["pid"])) > 0
            and isinstance(observed.get("snapshot"), str)
            and observed.get("externalWritableRegular") == []
            and not external_writable_files(str(observed["snapshot"]), Path.cwd().resolve()),
            "External writable FD",
        )
    log = bounded_read(directory / "worker.log").decode("utf-8", errors="replace").lower()
    require(
        not any(
            text in log for text in ("deny file-write", "sandbox: deny", "operation not permitted")
        ),
        "Sandbox denied write",
    )
    return actual


def _worker_inputs(
    config: Path, index: int, request_path: Path | None = None
) -> tuple[Job, dict[str, object]]:
    # The typed parent owns the old CPU leases and process observations. Seatbelt
    # forbids ps in the inference child; all immutable evidence remains verified.
    job = _load(config, observe_processes=False)
    require(index not in job.cpu, "Retained CPU must not be inferred again")
    request = _read(
        _chunk(job, index) / "request.json" if request_path is None else _local(str(request_path))
    )
    context = _context(job, observe_processes=False)
    parent = record(request.get("parent"))
    require(
        set(parent) == {"pid", "birth"}
        and _number(parent["pid"], maximum=2**31 - 1) == os.getppid()
        and isinstance(parent["birth"], str)
        and bool(parent["birth"]),
        "Worker is not a child of the observed lease-owning parent",
    )
    require(
        request
        == {
            "index": index,
            "device": "mps",
            "bound": list(job.old.bounds[index]),
            "context": context,
            "parent": parent,
        },
        "Worker input changed",
    )
    _worker_leases(job.old)
    return job, parent


def worker(
    config: Path,
    index: int,
    *,
    factory: Factory = _MlxBackend,
    auditor: Auditor = audit_descriptors,
) -> Path:
    job, parent = _worker_inputs(config, index)
    directory = _chunk(job, index)
    require(_local_environment() == _environment(job, index), "Worker scratch/model policy changed")
    context = _context(job, observe_processes=False)
    require(
        not (directory / "worker.json").exists() and not (directory / "vocals.npy").exists(),
        "Existing worker outcome; reconcile",
    )
    start, end = job.old.bounds[index]
    audio = original.read_chunk(job.old.source, start=start, count=end - start)
    auditor(directory, "before-load")
    backend = factory("mps")
    auditor(directory, "after-load")
    refs: list[dict[str, object]] = []
    for window in windows(end - start):
        padded = np.zeros((2, WINDOW), dtype=np.float32)
        padded[:, : window.count] = audio[:, window.start : window.start + window.count]
        predicted = backend.extract_vocals(padded)
        require(
            predicted.dtype == np.dtype(np.float32)
            and predicted.shape == (2, WINDOW)
            and bool(np.isfinite(predicted).all()),
            "Invalid MPS prediction",
        )
        path = directory / f"window-{window.index:06d}.npy"
        save_array(path, predicted)
        refs.append(
            {
                "window": {"index": window.index, "start": window.start, "count": window.count},
                "output": pin(path),
            }
        )
        require(saved_bytes(job.work) <= job.disk_bytes, "Saved byte budget exceeded")
    reduced = reduce_windows(
        end - start, lambda w: load_array(directory / f"window-{w.index:06d}.npy", samples=WINDOW)
    )
    save_array(directory / "vocals.npy", reduced)
    auditor(directory, "after-inference")
    require(
        _context(job, observe_processes=False) == context and os.getppid() == parent["pid"],
        "Worker inputs or parent changed during inference",
    )
    _worker_leases(job.old)
    durable(
        directory / "worker.json",
        {
            "device": "mps",
            "request": pin(directory / "request.json"),
            "parent": parent,
            "windows": refs,
            "workerEnvironment": _local_environment(),
            "qualityApproved": False,
        },
    )
    return directory / "worker.json"


def sandbox_check(config: Path, output: Path, index: int) -> Path:
    """Exercise the exact sandboxed loader and FD audit without loading a model or inferring."""
    job = _load(config)
    output = _local(str(output))
    require(output.is_dir() and not tuple(output.iterdir()), "New empty smoke root required")
    require(index not in job.cpu, "Smoke must select an MPS chunk")
    host = SystemHost()
    with _leases(job.old):
        context = _context(job)
        request = output / "request.json"
        durable(request, _request(job, index, host))
        profile = output / "sandbox.sb"
        profile.write_text(sandbox_profile(Path.cwd().resolve()), encoding="utf-8")
        environment = os.environ.copy()
        environment.update(_local_environment())
        for name in ("TMPDIR", "TMP", "TEMP", "XDG_CACHE_HOME"):
            path = output / ("cache" if name == "XDG_CACHE_HOME" else "tmp")
            path.mkdir(mode=0o700, exist_ok=True)
            environment[name] = str(path)
        command = [
            "/usr/bin/sandbox-exec",
            "-f",
            str(profile),
            sys.executable,
            "-B",
            "-m",
            "replay.separation_device_pipeline",
            "worker-load-check",
            str(config),
            str(request),
            "--index",
            str(index),
        ]
        start = time.monotonic()
        result = subprocess.run(
            command,
            capture_output=True,
            stdin=subprocess.DEVNULL,
            timeout=120,
            check=False,
            env=environment,
        )
        require(len(result.stdout) + len(result.stderr) <= 1024**2, "Smoke log budget exceeded")
        (output / "loader.log").write_bytes(result.stdout + result.stderr)
        require(result.returncode == 0, "Sandboxed loader/FD audit failed; inspect retained smoke")
        require(_context(job) == context, "Smoke inputs changed")
        descriptors = _read(output / "bootstrap-descriptors.json")
        require(
            isinstance(descriptors.get("snapshot"), str)
            and descriptors.get("externalWritableRegular") == []
            and not external_writable_files(str(descriptors["snapshot"]), Path.cwd().resolve()),
            "Smoke external writable descriptors",
        )
        durable(
            output / "complete.json",
            {
                "configuration": _file(config),
                "request": _file(request),
                "context": context,
                "command": command,
                "descriptors": _file(output / "bootstrap-descriptors.json"),
                "exitCode": 0,
                "wallSeconds": time.monotonic() - start,
                "modelLoaded": False,
                "inferenceExecuted": False,
            },
        )
    return output / "complete.json"


def worker_load_check(config: Path, request: Path, index: int) -> None:
    _worker_inputs(config, index, request)
    audit_descriptors(request.parent, "bootstrap")


def _request(job: Job, index: int, host: SystemHost) -> dict[str, object]:
    return {
        "index": index,
        "device": "mps",
        "bound": list(job.old.bounds[index]),
        "context": _context(job),
        "parent": {"pid": os.getpid(), "birth": host.birth(os.getpid())},
    }


def _sample_owned(
    child: subprocess.Popen[bytes],
    birth: str,
    host: SystemHost,
    *,
    deadline: float,
    diagnostics: dict[str, object],
) -> dict[str, object] | None:
    """Resample a known inventory race only while the owned root is unchanged.

    A successful terminal exit can race the inventory. A live process is never
    treated as completed because a short wait expired. Keep the original error
    separately from cleanup and retain bounded first/last diagnostic records.
    """
    retry_deadline = min(deadline, time.monotonic() + OBSERVATION_RETRY_SECONDS)
    for attempt in range(OBSERVATION_RETRY_ATTEMPTS):
        try:
            return sample_family(child.pid, birth, host)
        except ReplayError as error:
            status = child.poll()
            event: dict[str, object] = {
                "errorType": type(error).__name__,
                "error": str(error),
                "pid": child.pid,
                "birth": birth,
                "exitCodeObserved": status,
                "observedAtNs": time.time_ns(),
                "attempt": attempt,
            }
            diagnostics["count"] = int(str(diagnostics["count"])) + 1
            if diagnostics["first"] is None:
                diagnostics["first"] = event
            diagnostics["last"] = event
            if str(error) not in TRANSIENT_OBSERVATION_ERRORS:
                raise
            if status is not None:
                if status == 0:
                    return None
                raise
            # Birth failures for the root are not descendant races. Never
            # resample an unverified/reused root or hide the original error.
            try:
                require(host.birth(child.pid) == birth, "Owned PID reused")
            except ReplayError as identity_error:
                event["rootIdentityError"] = str(identity_error)
                raise identity_error from error
            remaining = retry_deadline - time.monotonic()
            if remaining <= 0 or attempt + 1 == OBSERVATION_RETRY_ATTEMPTS:
                raise
            try:
                exit_code = child.wait(timeout=min(OBSERVATION_RETRY_WAIT, remaining))
            except subprocess.TimeoutExpired:
                continue
            if exit_code == 0:
                return None
            raise
    raise ReplayError("Owned observation attempt limit exceeded")


def _launch(job: Job, index: int) -> None:
    directory = _chunk(job, index)
    profile = directory / "sandbox.sb"
    profile.write_text(sandbox_profile(Path.cwd().resolve()), encoding="utf-8")
    env = os.environ.copy()
    env.update(_local_environment())
    for name in ("TMPDIR", "TMP", "TEMP", "XDG_CACHE_HOME"):
        scratch = directory / ("cache" if name == "XDG_CACHE_HOME" else "tmp")
        scratch.mkdir(mode=0o700, exist_ok=True)
        env[name] = str(scratch)
    host = SystemHost()
    parent = {"pid": os.getpid(), "birth": host.birth(os.getpid())}
    require(_read(directory / "request.json").get("parent") == parent, "Parent identity changed")
    command = [
        "/usr/bin/sandbox-exec",
        "-f",
        str(profile),
        sys.executable,
        "-B",
        "-m",
        "replay.separation_device_pipeline",
        "worker",
        str(job.path),
        "--index",
        str(index),
    ]
    samples: list[dict[str, object]] = []
    observation_errors: dict[str, object] = {"count": 0, "first": None, "last": None}
    birth: str | None = None
    gone = False
    started = time.monotonic()
    with (directory / "worker.log").open("xb") as log:
        log.write(b"MPS_WORKER reserved\n")
        log.flush()
        child = subprocess.Popen(
            command,
            stdout=log,
            stderr=subprocess.STDOUT,
            stdin=subprocess.DEVNULL,
            env=env,
            start_new_session=True,
        )
        try:
            birth = host.birth(child.pid)
            durable(
                directory / "launch.json",
                {"pid": child.pid, "birth": birth, "command": command, "parent": parent},
            )
            while child.poll() is None:
                require(time.monotonic() - started < job.timeout, "MPS deadline exceeded")
                sampled = _sample_owned(
                    child,
                    birth,
                    host,
                    deadline=started + job.timeout,
                    diagnostics=observation_errors,
                )
                if sampled is None:
                    break
                samples.append(sampled)
                require(
                    max(int(str(samples[-1][key])) for key in ("rssBytes", "physicalBytes"))
                    <= job.family_bytes,
                    "MPS sampled resource budget exceeded",
                )
                with suppress(subprocess.TimeoutExpired):
                    child.wait(timeout=0.2)
            require(
                child.returncode == 0 and bool(samples), "Unsuccessful or unmeasured MPS worker"
            )
            require(
                all(
                    host.absent(int(str(record(member)["pid"])))
                    for sample in samples
                    for member in rows(sample["members"])
                ),
                "MPS descendants remain",
            )
            gone = True
        finally:
            cleanup = cleanup_child(child, birth, host)
            durable(
                directory / "resources.json",
                {
                    "exitCode": child.returncode,
                    "familyGone": gone,
                    "cleanup": cleanup,
                    "samples": samples,
                    "observationErrors": observation_errors,
                    "wallSeconds": time.monotonic() - started,
                    "hardResourceCaps": False,
                },
            )


def run_chunk(config: Path, index: int) -> Path:
    job = _load(config)
    directory = _chunk(job, index)
    with _leases(job.old):
        if (directory / "complete.json").exists():
            _prediction(job, index)
            return directory / "complete.json"
        require(not directory.exists(), "Partial device attempt retained; reconcile, never reissue")
        directory.mkdir(parents=True, mode=0o700)
        if index in job.cpu:
            original.verify_chunk(job.old, index=index, bound=_original_context(job))
            durable(
                directory / "cpu-reuse.json",
                {"oldComplete": _file(job.cpu[index]), "device": "cpu"},
            )
        else:
            durable(directory / "request.json", _request(job, index, SystemHost()))
            _launch(job, index)
        durable(directory / "complete.json", _proof(job, index))
        _prediction(job, index)
    return directory / "complete.json"


def verify_chunk(config: Path, index: int) -> None:
    job = _load(config)
    with _leases(job.old):
        _prediction(job, index)


def _whole(job: Job) -> dict[str, object]:
    return {
        "protocol": PROTOCOL,
        "source": str(job.old.source),
        "modelInput": str(job.old.source),
        "vocals": str(job.work / "vocals.wav"),
        "preparedInput": True,
        "device": "mixed" if job.cpu else "mps",
        "backendDevicesByChunk": [
            "cpu" if index in job.cpu else "mps" for index in range(len(job.old.bounds))
        ],
        "timestampCorrection": False,
        "loudnessNormalization": False,
        "gainDB": 0.0,
        "deliveryComplete": False,
        "inputSampleCount": job.old.samples,
        "outputSampleCount": job.old.samples,
        "sampleRate": MODEL_SAMPLE_RATE,
        "context": _context(job),
        "chunks": [
            pin(_chunk(job, index) / "complete.json") for index in range(len(job.old.bounds))
        ],
        "output": pin(job.work / "vocals.wav"),
        "qualityApproved": False,
        "cpuExactGateChanged": False,
    }


def assemble(config: Path) -> Path:
    job = _load(config)
    with _leases(job.old):
        require(
            not (job.work / "vocals.wav").exists() and not (job.work / "result.json").exists(),
            "Final outcome occupied; reconcile",
        )
        original.write_pcm(
            job.work / "vocals.wav",
            samples=job.old.samples,
            blocks=original.pcm_blocks(
                job.old.bounds,
                samples=job.old.samples,
                overlap=original.OVERLAP,
                prediction=lambda index: _prediction(job, index),
            ),
        )
        require(saved_bytes(job.work) <= job.disk_bytes, "Final saved byte budget exceeded")
        durable(job.work / "result.json", _whole(job))
    verify(config)
    return job.work / "result.json"


def verify(config: Path) -> None:
    job = _load(config)
    with _leases(job.old):
        before = _file(job.work / "result.json")
        require(_read(job.work / "result.json") == _whole(job), "Mixed result binding changed")
        with (job.work / "vocals.wav").open("rb") as stream, wave.open(stream, "rb") as audio:
            for expected in original.pcm_blocks(
                job.old.bounds,
                samples=job.old.samples,
                overlap=original.OVERLAP,
                prediction=lambda index: _prediction(job, index),
            ):
                require(
                    audio.readframes(len(expected) // 4) == expected, "Final ordered PCM differs"
                )
            require(not audio.readframes(1), "Final PCM has extra samples")
        verify_material(job.old.source, job.work)
        require(
            _file(job.work / "result.json") == before and saved_bytes(job.work) <= job.disk_bytes,
            "Whole verification changed or exceeded budget",
        )


def prepare(config: Path, output: Path) -> Path:
    job = _load(config, ready=False)
    output = _local(str(output))
    require(
        output.is_dir() and not tuple(output.iterdir()) and job.work == output,
        "New empty exact output root required",
    )
    can_dispatch = (
        job.configuration["acceptance"] is not None and job.configuration["handoff"] is not None
    )
    if can_dispatch:
        _acceptance(job)
        _handoff(job)
    verifier = _verification_python(job)
    code = _code()
    common = list(
        dict.fromkeys(
            [
                str(config),
                str(job.old.path),
                str(job.source),
                str(job.old.source),
                str(job.old.directory / "input.json"),
                str(_ref(job.configuration["preparation"])),
                *code,
                str(verifier[1]),
                str(verifier[2]),
            ]
        )
    )
    common.extend(
        str(record(ref)["path"])
        for name in ("providerInputs", "modelInputs")
        for ref in rows(job.old.definition[name])
    )
    catalog = output / "plan.json"
    durable(
        catalog,
        {
            "protocol": PROTOCOL,
            "configuration": _file(config),
            "codeInputs": code,
            "canDispatch": can_dispatch,
            "originalBounds": [list(bound) for bound in job.old.bounds],
            "cpuIndices": sorted(job.cpu),
            "workers": 1,
        },
    )
    if not can_dispatch:
        return catalog
    stage_inputs = [*common, str(catalog), str(_ref(job.configuration["handoff"]))]
    gate = record(job.configuration["acceptance"])
    stage_inputs.extend(str(_ref(gate[name])) for name in ("request", "receipt"))
    stage_inputs.extend(str(path) for path in job.cpu.values())
    require(len(set(stage_inputs)) <= 256, "Direct inputs exceed ordinary workflow budget")
    stages: list[dict[str, object]] = []
    previous: list[str] = []
    for index in range(len(job.old.bounds)):
        command = [sys.executable, "-B", "-m", "replay.separation_device_pipeline"]
        outputs = [str(path) for path in _chunk_outputs(job, index)]
        stages.append(
            {
                "name": f"device-chunk-{index:06d}",
                "command": [*command, "run-chunk", str(config), "--index", str(index)],
                "verify_command": [*command, "verify-chunk", str(config), "--index", str(index)],
                "inputs": [*stage_inputs, *previous],
                "outputs": outputs,
                "timeout_seconds": job.timeout + 180,
            }
        )
        previous = [str(_chunk(job, index) / "complete.json")]
    stages.append(
        {
            "name": "device-whole",
            "command": [*command, "assemble", str(config)],
            "verify_command": [*command, "verify", str(config)],
            "inputs": [
                *stage_inputs,
                *[str(_chunk(job, i) / "complete.json") for i in range(len(job.old.bounds))],
            ],
            "outputs": [str(output / "vocals.wav"), str(output / "result.json")],
            "timeout_seconds": 3600,
        }
    )
    definition = {"version": 1, "stages": stages}
    parsed = workflow.parse_stages(definition)
    require(
        len(json.dumps(definition).encode("utf-8")) <= workflow.MAX_PLAN_BYTES,
        "Ordinary workflow exceeds its byte budget",
    )
    workflow.stage_graph(parsed)
    durable(output / "workflow.json", definition)
    return output / "workflow.json"


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "action",
        choices=(
            "prepare",
            "capture-owner",
            "observe-handoff",
            "run-chunk",
            "verify-chunk",
            "worker",
            "assemble",
            "verify",
            "verify-acceptance",
            "sandbox-check",
            "worker-load-check",
        ),
    )
    parser.add_argument("config", type=Path)
    parser.add_argument("output", type=Path, nargs="?")
    parser.add_argument("--index", type=int)
    parser.add_argument("--pid", type=int)
    args = parser.parse_args(argv, namespace=Arguments())
    try:
        if args.action == "capture-owner":
            if args.output is None or args.pid is None:
                raise ReplayError("Owner/output required")
            result: object = capture_owner(args.config, args.pid, args.output)
        elif args.action in {"prepare", "observe-handoff"}:
            if args.output is None:
                raise ReplayError("New output required")
            result = (prepare if args.action == "prepare" else observe_handoff)(
                args.config, args.output
            )
        elif args.action in {"sandbox-check", "worker-load-check"}:
            if args.output is None or args.index is None:
                raise ReplayError("Smoke root/request and index required")
            result = (sandbox_check if args.action == "sandbox-check" else worker_load_check)(
                args.config, args.output, args.index
            )
        elif args.action in {"run-chunk", "verify-chunk", "worker"}:
            if args.index is None:
                raise ReplayError("Chunk index required")
            actions = {"run-chunk": run_chunk, "verify-chunk": verify_chunk, "worker": worker}
            result = actions[args.action](args.config, args.index)
        elif args.action == "verify-acceptance":
            _acceptance(_load(args.config, ready=False))
            result = "MPS gate verified"
        else:
            result = (assemble if args.action == "assemble" else verify)(args.config)
        print(str(result) if result is not None else "verified")
        return 0
    except (OSError, ValueError, ReplayError, subprocess.SubprocessError) as error:
        print(str(error), file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
