"""Submit and track many Apple Compressor jobs through its official CLI.

Each attempt is submitted once; its ID is recorded before waiting and monitored
on resume. Explicit controller-failure recovery permits one new-output attempt,
retaining the previous receipt and outcome. Outputs are never overwritten. A job
counts as done only when Compressor reports `Successful` and a nonempty output exists.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import subprocess
import sys
import time
from collections.abc import Callable, Iterator, Sequence
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path

from replay import compressor_helper_guard, core, recordings
from replay.core import ReplayError
from replay.recordings import record

COMPRESSOR = Path("/Applications/Compressor Creator Studio.app/Contents/MacOS/Compressor")
SUBMIT_TIMEOUT_SECONDS = 120
MONITOR_TIMEOUT_SECONDS = 60
MONITOR_RETRY_BACKOFF_SECONDS = (5.0, 15.0)
# A Compressor alpha job takes ~5 s; a longer poll left every slot idle between jobs.
POLL_SECONDS = 3.0
LOST_POLLS = 100
MAX_JOBS = 20_000
MAX_IN_FLIGHT = 64
MAX_SNAPSHOT_BYTES = 32 * 1024 * 1024
JOB_ID = re.compile(r"[A-Za-z0-9][A-Za-z0-9._:-]{0,127}")
DONE = "Successful"
PENDING = "Pending"
FAILED = frozenset({"Failed", "Cancelled", "Canceled", "Stopped"})
CONTROLLER_MESSAGE = r"(?:ジョブコントローラがダウンしています|job controller is down)"
CONTROLLER_DOWN = re.compile(
    rf"(?:{CONTROLLER_MESSAGE}[\s.。]*)+"
    rf"|(?:[1-9][0-9]{{0,3}}x )?“{CONTROLLER_MESSAGE}”"
    rf"|[1-9][0-9]{{0,3}}x {CONTROLLER_MESSAGE}",
    re.IGNORECASE,
)

Runner = Callable[[Sequence[str]], str]


@dataclass(frozen=True, slots=True)
class Job:
    name: str
    source: Path
    preset: Path
    output: Path
    dependencies: tuple[Path, ...] = ()


def _run(arguments: Sequence[str]) -> str:
    result = subprocess.run(
        arguments,
        stdin=subprocess.DEVNULL,
        capture_output=True,
        text=True,
        timeout=MONITOR_TIMEOUT_SECONDS if "-monitor" in arguments else SUBMIT_TIMEOUT_SECONDS,
        check=True,
    )
    return result.stdout


def read_jobs(path: Path) -> list[Job]:
    rows = record(json.loads(path.read_text(encoding="utf-8"))).get("jobs")
    if not isinstance(rows, list) or not 1 <= len(rows) <= MAX_JOBS:
        raise ReplayError("Expected a bounded nonempty Compressor job list")
    jobs: list[Job] = []
    for raw in rows:
        row = record(raw)
        values = [row.get(key) for key in ("name", "source", "preset", "output")]
        if not all(isinstance(value, str) and value for value in values):
            raise ReplayError("Each Compressor job needs name, source, preset and output")
        name, source, preset, output = (str(value) for value in values)
        if re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._-]{0,127}", name) is None:
            raise ReplayError("Invalid Compressor job name")
        dependencies = row.get("dependencies", [])
        if (
            not isinstance(dependencies, list)
            or len(dependencies) > 128
            or not all(
                isinstance(value, str) and Path(value).is_absolute() for value in dependencies
            )
            or len(set(dependencies)) != len(dependencies)
        ):
            raise ReplayError("Expected bounded unique absolute Compressor dependencies")
        jobs.append(
            Job(
                name, Path(source), Path(preset), Path(output), tuple(Path(p) for p in dependencies)
            )
        )
    if len({job.name for job in jobs}) != len(jobs) or len({job.output for job in jobs}) != len(
        jobs
    ):
        raise ReplayError("Compressor job names and outputs must be unique")
    return jobs


def submit_arguments(job: Job, *, executable: Path, batch: str) -> list[str]:
    return [
        str(executable),
        "-batchname",
        batch,
        "-jobpath",
        str(job.source),
        "-settingpath",
        str(job.preset),
        "-locationpath",
        str(job.output),
        "-outputformat",
        "json",
    ]


def parse_submission(text: str) -> str:
    """Return the single job ID from Compressor's submission JSON."""
    start = text.find("{")
    if start < 0:
        raise ReplayError("Compressor returned no submission record")
    batch = record(record(json.loads(text[start:])).get("batch"))
    jobs = batch.get("jobs")
    if not isinstance(jobs, list) or len(jobs) != 1:
        raise ReplayError("Expected exactly one submitted Compressor job")
    job_id = record(jobs[0]).get("jobID")
    if not isinstance(job_id, str) or JOB_ID.fullmatch(job_id) is None:
        raise ReplayError("Invalid Compressor job ID")
    return job_id


def parse_status(text: str) -> str:
    start = text.find("[")
    if start < 0:
        raise ReplayError("Compressor returned no job status")
    rows = json.loads(text[start:])
    if rows == []:
        # A just-submitted job may not be listed yet; keep waiting for it.
        return PENDING
    if not isinstance(rows, list) or len(rows) != 1:
        raise ReplayError("Expected one Compressor job status")
    status = record(record(rows[0]).get("jobStatus")).get("status")
    if not isinstance(status, str):
        raise ReplayError("Compressor job status is missing")
    return status


def _status_id(row: dict[str, object], kind: str) -> str:
    value = row.get(f"{kind}id", row.get(f"{kind}ID"))
    if not isinstance(value, str) or JOB_ID.fullmatch(value) is None:
        raise ReplayError(f"Invalid Compressor status {kind} ID")
    if f"{kind}ID" in row and row[f"{kind}ID"] != value:
        raise ReplayError(f"Conflicting Compressor status {kind} IDs")
    return value


def _snapshot_jobs(rows: list[object]) -> Iterator[dict[str, object]]:
    batches: set[str] = set()
    for raw in rows:
        entry = record(raw)
        if "batchStatus" not in entry:
            yield record(entry.get("jobStatus"))
            continue
        if "jobStatus" in entry:
            raise ReplayError("Ambiguous Compressor status entry")
        batch = record(entry["batchStatus"])
        batch_id = _status_id(batch, "batch")
        jobs = batch.get("jobs")
        if batch_id in batches or not isinstance(jobs, list):
            raise ReplayError("Duplicate batch ID or missing Compressor jobs list")
        batches.add(batch_id)
        for job in jobs:
            nested = record(job)
            if "batchStatus" in nested:
                raise ReplayError("Unexpected nested Compressor batch")
            row = record(nested.get("jobStatus"))
            if _status_id(row, "batch") != batch_id:
                raise ReplayError("Conflicting Compressor job batch ID")
            yield row


def parse_snapshot(text: str) -> dict[str, str]:
    """Index one official all-jobs snapshot by its returned native IDs."""
    if len(text.encode("utf-8")) > MAX_SNAPSHOT_BYTES:
        raise ReplayError("Compressor snapshot exceeds its byte budget")
    start = text.find("[")
    if start < 0:
        raise ReplayError("Compressor returned no job status")
    rows = json.loads(text[start:])
    if not isinstance(rows, list):
        raise ReplayError("Expected Compressor job status list")
    statuses: dict[str, str] = {}
    for row in _snapshot_jobs(rows):
        job_id = _status_id(row, "job")
        status = row.get("status")
        if job_id in statuses or not isinstance(status, str):
            raise ReplayError("Duplicate ID or missing Compressor status")
        if len(statuses) >= MAX_JOBS:
            raise ReplayError("Compressor snapshot exceeds its job budget")
        statuses[job_id] = status
    return statuses


def _controller_failed(status: str) -> bool:
    outcome, separator, detail = status.partition(":")
    return (
        outcome == "Failed"
        and bool(separator)
        and CONTROLLER_DOWN.fullmatch(detail.strip()) is not None
    )


def _write(path: Path, value: object) -> None:
    pending = path.with_name(f".{path.name}.pending")
    with pending.open("w", encoding="utf-8") as stream:
        os.fchmod(stream.fileno(), 0o600)
        json.dump(value, stream)
    os.replace(pending, path)


def _stat(path: Path) -> dict[str, int]:
    value = path.stat()
    if path.absolute() != path.resolve():
        raise ReplayError(f"Compressor input must not be linked: {path}")
    if not path.is_file() or value.st_size <= 0:
        raise ReplayError(f"Compressor input must be a nonempty file: {path}")
    return {
        "device": value.st_dev,
        "inode": value.st_ino,
        "bytes": value.st_size,
        "mtime_ns": value.st_mtime_ns,
        "ctime_ns": value.st_ctime_ns,
    }


def _pin(path: Path) -> dict[str, object]:
    before = _stat(path)
    with path.open("rb") as stream:
        digest = hashlib.file_digest(stream, "sha256").hexdigest()
    if before != _stat(path):
        raise ReplayError(f"Compressor input changed while hashing: {path}")
    return {
        "path": str(path.absolute()),
        "sha256": digest,
        "bytes": before["bytes"],
        "stat": before,
    }


def runtime_paths(executable: Path) -> tuple[Path, ...]:
    """Exact CLI/interpreter and controller code, not recursive application bundles."""
    return (
        executable.absolute(),
        Path(sys.executable).resolve(strict=True),
        Path(__file__).absolute(),
        Path(compressor_helper_guard.__file__).absolute(),
        Path(core.__file__).absolute(),
        Path(recordings.__file__).absolute(),
    )


def verify_runtime(saved: dict[str, object], *, rehash: bool) -> bool:
    """Legacy records stay unbound; new pins are checked at polling/terminal boundaries."""
    if saved.get("submissionVersion") == 1 or "submissionVersion" not in saved:
        return False
    command = saved.get("command")
    pins = saved.get("runtimeInputs")
    if (
        saved.get("submissionVersion") != 2
        or not isinstance(command, list)
        or not command
        or not isinstance(command[0], str)
        or not isinstance(pins, list)
        or len(pins) != 6
    ):
        raise ReplayError("Compressor runtime pins are missing or invalid")
    for path, raw in zip(runtime_paths(Path(command[0])), pins, strict=True):
        pin = record(raw)
        if pin.get("path") != str(path) or pin.get("stat") != _stat(path):
            raise ReplayError(f"Compressor runtime input changed: {path}")
        digest = pin.get("sha256")
        if not isinstance(digest, str) or re.fullmatch(r"[a-f0-9]{64}", digest) is None:
            raise ReplayError("Invalid Compressor runtime SHA256")
        if rehash and pin != _pin(path):
            raise ReplayError(f"Compressor runtime SHA256 changed: {path}")
    return True


def _job_config(job: Job) -> dict[str, str]:
    return {
        "name": job.name,
        "source": str(job.source.absolute()),
        "preset": str(job.preset.absolute()),
        "output": str(job.output.absolute()),
    }


def _config_digest(config: dict[str, str]) -> str:
    return hashlib.sha256(
        json.dumps(config, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def verify_submission(job: Job, saved: dict[str, object], *, rehash: bool) -> bool:
    """Check retained pre-submit pins; legacy is observable, never historical proof.

    Inputs must remain immutable/read-only to all writers from pre-submit through
    terminal verification. Stat checks are mutation guards, not content proof.
    """
    if "submissionVersion" not in saved:
        return False
    config = _job_config(job)
    if (
        saved.get("submissionVersion") not in (1, 2)
        or saved.get("fingerprintMoment") != "before-submit"
        or saved.get("inputContract") != "readonly-immutable-through-terminal"
        or saved.get("jobConfig") != config
        or saved.get("jobConfigSHA256") != _config_digest(config)
    ):
        raise ReplayError("Compressor submission configuration changed or is invalid")
    command = saved.get("command")
    if (
        not isinstance(command, list)
        or len(command) != 11
        or not all(isinstance(value, str) and value for value in command)
        or command != submit_arguments(job, executable=Path(str(command[0])), batch=str(command[2]))
    ):
        raise ReplayError("Compressor submission command changed or is invalid")
    pins = saved.get("inputs")
    if not isinstance(pins, list) or len(pins) != 2:
        raise ReplayError("Compressor submission lacks source/preset pins")
    for path, raw in zip((job.source, job.preset), pins, strict=True):
        pin = record(raw)
        if (
            pin.get("path") != str(path.absolute())
            or pin.get("stat") != _stat(path)
            or pin.get("bytes") != path.stat().st_size
            or not isinstance(pin.get("sha256"), str)
            or re.fullmatch(r"[a-f0-9]{64}", str(pin.get("sha256"))) is None
        ):
            raise ReplayError(f"Compressor submission input changed: {path}")
        if rehash and pin != _pin(path):
            raise ReplayError(f"Compressor submission input SHA256 changed: {path}")
    verify_runtime(saved, rehash=rehash)
    if saved.get("submissionVersion") == 2:
        dependencies = saved.get("dependencies")
        if not isinstance(dependencies, list) or len(dependencies) != len(job.dependencies):
            raise ReplayError("Compressor dependency pins are missing")
        for path, raw in zip(job.dependencies, dependencies, strict=True):
            pin = record(raw)
            if pin.get("path") != str(path.absolute()) or pin.get("stat") != _stat(path):
                raise ReplayError("Compressor dependency changed")
            if rehash and pin != _pin(path):
                raise ReplayError("Compressor dependency SHA256 changed")
    return True


def _state(directory: Path, job: Job) -> dict[str, object]:
    path = directory / f"{job.name}.json"
    return record(json.loads(path.read_text(encoding="utf-8"))) if path.is_file() else {}


@dataclass
class Batch:
    """Mutable progress of one run; persisted per job in the state directory."""

    state: Path
    executable: Path
    batch: str
    runner: Runner
    sleep: Callable[[float], None]
    helper_guard_directory: Path | None = None
    submit_workers: int = 1
    monitor_response: str | None = None
    monitor_pin: dict[str, object] | None = None
    monitor_statuses: dict[str, str] = field(default_factory=dict)

    def guard(self) -> None:
        if self.helper_guard_directory is not None:
            compressor_helper_guard.check(self.helper_guard_directory, self.executable)

    def submit(self, job: Job, *, attempt: dict[str, object] | None = None) -> str:
        self.guard()
        return self._submit(job, attempt=attempt)

    def submit_many(
        self, jobs: Sequence[Job]
    ) -> tuple[dict[str, tuple[Job, str]], Exception | None]:
        """Submit up to `submit_workers` jobs at once; keep every accepted ID if one fails."""
        accepted: dict[str, tuple[Job, str]] = {}
        failure: Exception | None = None
        if self.submit_workers <= 1:
            for job in jobs:
                try:
                    accepted[job.name] = (job, self.submit(job))
                except Exception as error:  # noqa: BLE001 - re-raised by the caller
                    return accepted, error
            return accepted, None
        with ThreadPoolExecutor(max_workers=max(1, self.submit_workers)) as pool:
            futures = [(job, pool.submit(self.submit, job)) for job in jobs]
            for job, future in futures:
                try:
                    accepted[job.name] = (job, future.result())
                except Exception as error:  # noqa: BLE001 - re-raised by the caller after tracking
                    failure = failure or error
        return accepted, failure

    def _submit(self, job: Job, *, attempt: dict[str, object] | None = None) -> str:
        if attempt is None and (self.state / f"{job.name}.json").exists():
            raise ReplayError(f"Existing Compressor submission must be reconciled: {job.name}")
        if job.output.exists() or any(job.output.parent.glob(f"{job.output.stem}.*")):
            raise ReplayError(f"Refusing to overwrite an untracked output: {job.output}")
        if not job.source.is_file() or not job.preset.is_file():
            raise ReplayError(f"Missing Compressor source or preset for {job.name}")
        job.output.parent.mkdir(parents=True, exist_ok=True)
        receipt = dict(attempt) if attempt is not None else {}
        command = submit_arguments(job, executable=self.executable, batch=self.batch)
        config = _job_config(job)
        receipt.update(
            status="submitting",
            submissionVersion=2,
            fingerprintMoment="before-submit",
            inputContract="readonly-immutable-through-terminal",
            jobConfig=config,
            jobConfigSHA256=_config_digest(config),
            command=command,
            inputs=[_pin(job.source), _pin(job.preset)],
            runtimeInputs=[_pin(path) for path in runtime_paths(self.executable)],
            dependencies=[_pin(path) for path in job.dependencies],
        )
        _write(self.state / f"{job.name}.json", receipt)
        verify_submission(job, receipt, rehash=False)
        text = self.runner(command)
        receipt["submitResponse"] = text
        _write(self.state / f"{job.name}.json", receipt)
        job_id = parse_submission(text)
        receipt.update(status="submitted", jobID=job_id, submitResponse=text)
        _write(self.state / f"{job.name}.json", receipt)
        verify_submission(job, receipt, rehash=False)
        return job_id

    def status(self, job_id: str) -> str:
        arguments = [str(self.executable), "-monitor", "-jobid", job_id, "-once"]
        arguments += ["-timeout", "10", "-outputformat", "json"]
        return parse_status(self.runner(arguments))

    def snapshot(self) -> dict[str, str]:
        """Retry only transient monitor timeouts; never invent an observation."""
        arguments = [
            str(self.executable),
            "-monitor",
            "-once",
            "-timeout",
            "10",
            "-outputformat",
            "json",
        ]
        retries = 0
        while True:
            try:
                text = self.runner(arguments)
            except subprocess.TimeoutExpired:
                if retries == len(MONITOR_RETRY_BACKOFF_SECONDS):
                    raise
                delay = MONITOR_RETRY_BACKOFF_SECONDS[retries]
                retries += 1
                print(
                    f"COMPRESSOR_MONITOR_RETRY attempt={retries + 1}/3 delay={delay:g}s",
                    file=sys.stderr,
                    flush=True,
                )
                self.sleep(delay)
                # subprocess.run reaps the CLI, not LaunchServices-orphaned helpers.
                self.guard()
            else:
                statuses = parse_snapshot(text)
                self.monitor_response = text
                self.monitor_pin = None
                self.monitor_statuses = statuses
                return statuses

    def retry(self, job: Job, job_id: str, status: str) -> tuple[Job, str]:
        """One lifetime retry, after run reconciles the current official snapshot."""
        saved = _state(self.state, job)
        if saved.get("jobID") != job_id or saved.get("status") == "submitting":
            raise ReplayError(f"Submission outcome of {job.name} is unknown; inspect it first")
        if saved.get("retry_count", 0) != 0 or not _controller_failed(status):
            raise ReplayError(f"Controller retry not permitted for {job.name}")
        verify_submission(job, saved, rehash=True)
        candidates = [job.output, *job.output.parent.glob(f"{job.output.stem}.*")]
        if any(path.is_file() and path.stat().st_size > 0 for path in candidates):
            raise ReplayError(f"Output mapping needs explicit reconciliation: {job.output}")
        output = job.output.with_name(f"{job.output.stem}.retry-1{job.output.suffix}")
        retry_job = Job(job.name, job.source, job.preset, output, job.dependencies)
        evidence = self.state / f"{job.name}.attempt-0.receipt.json"
        self.guard()
        with evidence.open("xb") as stream:
            os.fchmod(stream.fileno(), 0o600)
            stream.write((self.state / f"{job.name}.json").read_bytes())
        attempt: dict[str, object] = {
            "retry_count": 1,
            "requested_output": str(output),
            "history": [
                {
                    "jobID": job_id,
                    "status": status,
                    "requested_output": str(job.output),
                    "receipt": str(evidence),
                }
            ],
        }
        return retry_job, self._submit(retry_job, attempt=attempt)

    def finish(self, job: Job, job_id: str) -> None:
        output = actual_output(job.output)
        size = output.stat().st_size
        saved = _state(self.state, job)
        if saved.get("jobID") != job_id:
            raise ReplayError("Compressor terminal job identity changed")
        if verify_submission(job, saved, rehash=True):
            saved["terminalInputsVerified"] = True
            if saved.get("submissionVersion") == 2:
                saved["terminalRuntimeVerified"] = True
            if self.monitor_response is None or self.monitor_statuses.get(job_id) != DONE:
                raise ReplayError("Missing terminal Compressor snapshot")
            if self.monitor_pin is None:
                data = self.monitor_response.encode("utf-8")
                digest = hashlib.sha256(data).hexdigest()
                path = self.state / f"monitor-{digest}.json"
                if not path.exists():
                    with path.open("xb") as stream:
                        os.fchmod(stream.fileno(), 0o600)
                        stream.write(data)
                self.monitor_pin = _pin(path)
                if self.monitor_pin["sha256"] != digest:
                    raise ReplayError("Retained Compressor snapshot changed")
            saved["monitorResponse"] = self.monitor_pin
        saved.update(status=DONE, jobID=job_id, output=str(output), bytes=size)
        _write(self.state / f"{job.name}.json", saved)


def actual_output(requested: Path) -> Path:
    """Resolve the preset extension or a sole retry variant if the original is empty.

    Multiple nonempty variants require explicit output-mapping reconciliation.
    """
    if requested.is_file() and requested.stat().st_size > 0:
        return requested
    matches = [
        path
        for path in requested.parent.glob(f"{requested.stem}.*")
        if path.is_file() and path.stat().st_size > 0 and not path.name.startswith(".")
    ]
    if len(matches) != 1:
        raise ReplayError(f"Compressor reported success without output: {requested.name}")
    return matches[0]


def run(
    jobs: Sequence[Job],
    batch: Batch,
    *,
    in_flight: int,
    on_ambiguous: Callable[[str], None] | None = None,
    retry_controller_failure: bool = False,
    monitor_only: bool = False,
) -> int:
    """Return newly finished count, not workflow or quality approval.

    Supplying on_ambiguous opts into deferring submitting receipts without IDs.
    Each name is reported before any submission; its receipt remains untouched.
    The caller must account for deferred jobs separately from this return value.
    """
    if not 1 <= in_flight <= MAX_IN_FLIGHT:
        raise ReplayError("In-flight limit is out of range")
    if len(jobs) > MAX_JOBS or len({job.name for job in jobs}) != len(jobs):
        raise ReplayError("Compressor tracked job budget or uniqueness exceeded")
    if monitor_only and retry_controller_failure:
        raise ReplayError("Monitor-only cannot retry submissions")
    batch.state.mkdir(mode=0o700, parents=True, exist_ok=True)
    waiting: dict[str, tuple[Job, str]] = {}
    queue: list[Job] = []
    for job in jobs:
        saved = _state(batch.state, job)
        status, job_id = saved.get("status"), saved.get("jobID")
        if saved.get("submissionVersion") == 2:
            command = saved.get("command")
            if not isinstance(command, list) or command[:1] != [str(batch.executable)]:
                raise ReplayError("Compressor executable differs from submitted command")
        requested = saved.get("requested_output", str(job.output))
        if not isinstance(requested, str) or not requested:
            raise ReplayError("Invalid saved Compressor requested output")
        tracked = Job(job.name, job.source, job.preset, Path(requested), job.dependencies)
        retry_output = job.output.with_name(f"{job.output.stem}.retry-1{job.output.suffix}")
        if (
            "submissionVersion" in saved
            and tracked.output != job.output
            and (saved.get("retry_count") != 1 or tracked.output != retry_output)
        ):
            raise ReplayError("Compressor requested output changed")
        if status == DONE:
            if verify_submission(tracked, saved, rehash=True):
                output = actual_output(tracked.output)
                if (
                    saved.get("terminalInputsVerified") is not True
                    or (
                        saved.get("submissionVersion") == 2
                        and saved.get("terminalRuntimeVerified") is not True
                    )
                    or saved.get("output") != str(output)
                    or saved.get("bytes") != output.stat().st_size
                ):
                    raise ReplayError("Completed Compressor output changed or is unverified")
            continue
        if status == "submitting":
            if on_ambiguous is not None and job_id is None:
                on_ambiguous(job.name)
                continue
            raise ReplayError(f"Submission outcome of {job.name} is unknown; inspect it first")
        if isinstance(job_id, str):
            if JOB_ID.fullmatch(job_id) is None:
                raise ReplayError("Invalid saved Compressor job ID")
            verify_submission(tracked, saved, rehash=False)
            waiting[job.name] = (tracked, job_id)
        elif (batch.state / f"{job.name}.json").exists():
            raise ReplayError(f"Submission outcome of {job.name} is unknown; inspect it first")
        elif not monitor_only:
            queue.append(job)
    finished = 0
    retried = False
    absent: dict[str, int] = {}
    unreconciled: list[str] = []
    while waiting or queue:
        batch.guard()
        fill = [queue.pop(0) for _ in range(min(in_flight - len(waiting), len(queue)))]
        if fill:
            submitted, error = batch.submit_many(fill)
            waiting.update(submitted)
            if error is not None:
                raise error
        for job, _job_id in waiting.values():
            verify_submission(job, _state(batch.state, job), rehash=False)
        statuses = batch.snapshot()
        failures: list[tuple[Job, str, str]] = []
        for name, (job, job_id) in list(waiting.items()):
            verify_submission(job, _state(batch.state, job), rehash=False)
            status = statuses.get(job_id, PENDING)
            if job_id in statuses:
                absent.pop(job_id, None)
            else:
                absent[job_id] = absent.get(job_id, 0) + 1
                if absent[job_id] >= LOST_POLLS:
                    if not (job.output.is_file() and job.output.stat().st_size > 0):
                        raise ReplayError(
                            f"Compressor job {job.name} is absent from {LOST_POLLS} consecutive "
                            "snapshots and has no output; inspect it before any resubmission"
                        )
                    # Compressor purged a finished job's history: keep the receipt untouched,
                    # let the other jobs run, and report it for explicit reconciliation.
                    unreconciled.append(job.name)
                    del waiting[name]
                    continue
            if status == DONE:
                batch.finish(job, job_id)
                del waiting[name]
                finished += 1
            elif status.partition(":")[0] in FAILED:
                failures.append((job, job_id, status))
        # Reconcile every tracked ID before considering any resubmission.
        for job, job_id, status in failures:
            if not retry_controller_failure or retried or not _controller_failed(status):
                raise ReplayError(f"Compressor job {job.name} ended as {status}")
            waiting[job.name] = batch.retry(job, job_id, status)
            retried = True
        refill = bool(queue) and len(waiting) < in_flight and not failures
        if (waiting or queue) and not refill:
            batch.sleep(POLL_SECONDS)
    if unreconciled:
        raise ReplayError(
            "Compressor history no longer lists finished-looking jobs "
            f"{sorted(unreconciled)}; their receipts are untouched, verify the outputs"
        )
    return finished


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("jobs", type=Path, help="JSON with a `jobs` list")
    parser.add_argument("--state", type=Path, required=True)
    parser.add_argument("--batch-name", default="replay delivery")
    parser.add_argument("--in-flight", type=int, default=8)
    parser.add_argument("--submit-workers", type=int, default=4)
    parser.add_argument("--executable", type=Path, default=COMPRESSOR)
    parser.add_argument(
        "--monitor-only",
        action="store_true",
        help="Monitor recorded IDs only; never submit untracked jobs or retry",
    )
    parser.add_argument(
        "--defer-ambiguous",
        action="store_true",
        help="Leave submitting receipts without IDs untouched and process independent jobs",
    )
    parser.add_argument(
        "--retry-controller-failure",
        action="store_true",
        help="Retry one confirmed controller-down failure with a new output; once per job",
    )
    parser.add_argument(
        "--helper-guard-directory",
        type=Path,
        help=(
            "Opt in to shared old-helper SIGTERM guard (300 helpers, age >=300s); "
            "private repo-local directory"
        ),
    )
    args = parser.parse_args(argv)
    deferred: list[str] = []

    def report_ambiguous(name: str) -> None:
        deferred.append(name)
        print(json.dumps({"deferred_ambiguous": name}), file=sys.stderr, flush=True)

    try:
        batch = Batch(
            args.state.absolute(),
            args.executable,
            args.batch_name,
            _run,
            time.sleep,
            helper_guard_directory=args.helper_guard_directory,
            submit_workers=args.submit_workers,
        )
        jobs = read_jobs(args.jobs)
        finished = run(
            jobs,
            batch,
            in_flight=args.in_flight,
            on_ambiguous=report_ambiguous if args.defer_ambiguous else None,
            retry_controller_failure=args.retry_controller_failure,
            monitor_only=args.monitor_only,
        )
    except (ReplayError, OSError, ValueError, subprocess.SubprocessError) as error:
        print(f"COMPRESSOR_BATCH_ERROR {error}", file=sys.stderr)
        return 1
    if args.monitor_only:
        print(
            json.dumps({"finished": finished, "monitor_only": True, "deferred_ambiguous": deferred})
        )
        return 1 if deferred else 0
    if args.defer_ambiguous:
        print(
            json.dumps(
                {
                    "finished": finished,
                    "completed": len(jobs) - len(deferred),
                    "pending": len(deferred),
                    "deferred_ambiguous": deferred,
                }
            )
        )
        return 1 if deferred else 0
    print(json.dumps({"finished": finished}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
